#!/usr/bin/env python3
"""
HARBench Pretraining Script

Self-supervised pretraining for HAR backbones.
Supports multiple SSL methods:
- MTL (Multi-Task Learning): Signal reconstruction + transformation prediction
- SimCLR: Contrastive learning with data augmentation
- MoCo: Momentum Contrast with queue-based negative sampling
- CPC: Contrastive Predictive Coding
- Timemask: Time-axis masked reconstruction
- Timechannel: Time + channel masked reconstruction

Saves results, checkpoints, and logs in results/pretrain/ directory.

Usage:
    python pretrain.py --method mtl --datasets DSADS PAMAP2 --sensors back thigh
    python pretrain.py --method simclr --datasets DSADS --sensors back
    python pretrain.py --method moco --datasets PAMAP2 --sensors thigh
    python pretrain.py --method cpc --datasets DSADS PAMAP2 --sensors back thigh
    python pretrain.py --method timemask --datasets DSADS --sensors back
    python pretrain.py --method timechannel --datasets DSADS --sensors back
"""

import argparse
import copy
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataloader import create_pretrain_dataloaders
from src.models import Resnet
from src.models.backbones import ResNetForCPC


SEED = 42

# Available SSL methods
SSL_METHODS = ["mtl", "simclr", "moco", "cpc", "timemask", "timechannel"]

# Default pretraining datasets (Table 1 in paper: 14 unlabeled datasets)
DEFAULT_PRETRAIN_DATASETS = [
    "nhanes",
    "adlrd",
    "chad",
    "capture24",
    "dog",
    "har70plus",
    "hhar",
    "imsb",
    "kddi_kitchen_left",   # KDDIKitchen in paper (left hand)
    "kddi_kitchen_right",  # KDDIKitchen in paper (right hand)
    "motionsense",
    "opportunity",
    "sbrhapt",
    "tmd",
    "wisdm",
]

# Default data root for pretraining
DEFAULT_PRETRAIN_DATA_ROOT = "har-datasets/data/processed"


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Data Augmentation
# =============================================================================

def permute_segments(x: torch.Tensor, num_segments: int = 5) -> torch.Tensor:
    """Permute segments of the time series."""
    batch_size, channels, time_steps = x.shape
    device = x.device
    segment_length = time_steps // num_segments

    if segment_length == 0:
        return x

    # Unfold into segments
    segments = x.unfold(dimension=-1, size=segment_length, step=segment_length)

    # Shuffle segments for each batch
    permuted_segments = torch.zeros_like(segments)
    for i in range(batch_size):
        permuted_indices = torch.randperm(segments.size(2), device=device)
        permuted_segments[i] = segments[i, :, permuted_indices, :]

    return permuted_segments.contiguous().view(batch_size, channels, -1)


def time_series_reverse(x: torch.Tensor) -> torch.Tensor:
    """Reverse the time series."""
    return torch.flip(x, dims=[-1])


def time_warping(x: torch.Tensor, max_warp_factor: float = 1.5) -> torch.Tensor:
    """Apply time warping augmentation."""
    batch_size, channels, time_steps = x.shape
    device = x.device

    # Create smooth warp curve
    warp_curve = torch.randn(time_steps, device=device).cumsum(0)
    warp_curve = (warp_curve - warp_curve.min()) / (warp_curve.max() - warp_curve.min() + 1e-8)
    warp_curve = warp_curve * (time_steps - 1)

    # Apply warp factor
    center = torch.linspace(0, time_steps - 1, time_steps, device=device)
    indices = center + (warp_curve - center) * (max_warp_factor - 1)
    indices = indices.clamp(0, time_steps - 1)

    # Linear interpolation
    idx_low = indices.floor().long()
    idx_high = (idx_low + 1).clamp(max=time_steps - 1)
    weight = (indices - idx_low).view(1, 1, -1)

    return (1 - weight) * x[:, :, idx_low] + weight * x[:, :, idx_high]


def rotation_3d(x: torch.Tensor) -> torch.Tensor:
    """Apply random 3D rotation to accelerometer data."""
    batch_size, channels, time_steps = x.shape
    device = x.device

    # Generate random rotation axes and angles
    axes = np.random.uniform(low=-1, high=1, size=(batch_size, 3))
    angles = np.random.uniform(low=-np.pi, high=np.pi, size=(batch_size,))

    # Normalize axes
    axes = axes / (np.linalg.norm(axes, ord=2, axis=1, keepdims=True) + 1e-8)

    # Compute rotation matrices using axis-angle representation
    x_np = axes[:, 0]
    y_np = axes[:, 1]
    z_np = axes[:, 2]
    c = np.cos(angles)
    s = np.sin(angles)
    C = 1 - c

    xs = x_np * s
    ys = y_np * s
    zs = z_np * s
    xC = x_np * C
    yC = y_np * C
    zC = z_np * C
    xyC = x_np * yC
    yzC = y_np * zC
    zxC = z_np * xC

    # Build rotation matrices (batch_size, 3, 3)
    rot_matrices = np.array([
        [x_np * xC + c, xyC - zs, zxC + ys],
        [xyC + zs, y_np * yC + c, yzC - xs],
        [zxC - ys, yzC + xs, z_np * zC + c]
    ]).transpose(2, 0, 1)  # (batch_size, 3, 3)

    rot_matrices = torch.tensor(rot_matrices, dtype=torch.float32, device=device)

    # Apply rotation: (batch, 3, time) -> (batch, time, 3) -> rotate -> (batch, 3, time)
    x_transposed = x.permute(0, 2, 1)  # (batch, time, 3)
    rotated = torch.bmm(x_transposed, rot_matrices)  # (batch, time, 3)

    return rotated.permute(0, 2, 1)  # (batch, 3, time)


def data_aug(x: torch.Tensor, aug_type: str, device: torch.device) -> torch.Tensor:
    """
    Apply data augmentation based on type.
    """
    if aug_type.startswith("permute-"):
        num_segments = int(aug_type.split("-")[1])
        return permute_segments(x, num_segments)
    elif aug_type == "time_series_reverse":
        return time_series_reverse(x)
    elif aug_type.startswith("time_warping-"):
        max_warp = float(aug_type.split("-")[1])
        return time_warping(x, max_warp)
    elif aug_type == "rotation":
        return rotation_3d(x)
    else:
        return x


# Legacy augmentation functions for other SSL methods
def jitter(x: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
    """Add random noise."""
    return x + torch.randn_like(x) * sigma


def scaling(x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """Random scaling."""
    factor = 1.0 + (torch.rand(x.size(0), 1, 1, device=x.device) - 0.5) * sigma * 2
    return x * factor


def apply_augmentation(x: torch.Tensor, aug_type: str = "random") -> torch.Tensor:
    """Apply data augmentation for contrastive learning."""
    if aug_type == "random":
        aug_type = random.choice(["jitter", "scaling", "time_warp", "permutation"])

    if aug_type == "jitter":
        return jitter(x)
    elif aug_type == "scaling":
        return scaling(x)
    elif aug_type == "time_warp":
        return time_warping(x, 1.2)
    elif aug_type == "permutation":
        return permute_segments(x, 5)
    else:
        return x


# =============================================================================
# Self-Supervised Learning Models
# =============================================================================

# Default SSL types for MTL
DEFAULT_SSL_TYPES = [
    "binary-permute-5",
    "binary-time_series_reverse",
    "binary-time_warping-1.5",
]


def _ssl_type_to_key(ssl_type: str) -> str:
    """Convert SSL type to valid module key (replace invalid characters)."""
    return ssl_type.replace("-", "_").replace(".", "_")


def _key_to_ssl_type(key: str, ssl_types: list) -> str:
    """Convert module key back to SSL type."""
    for st in ssl_types:
        if _ssl_type_to_key(st) == key:
            return st
    return key


class MTLPretrainModel(nn.Module):
    """Multi-Task Learning (MTL) pretraining model."""

    def __init__(self, backbone: nn.Module, ssl_types: list = None, use_rotation: bool = True):
        super().__init__()
        self.backbone = backbone
        self.ssl_types = ssl_types or DEFAULT_SSL_TYPES
        self.use_rotation = use_rotation

        # Binary classification head for each SSL task
        # Each head predicts: 0 = original, 1 = transformed
        self.heads = nn.ModuleDict()
        for ssl_type in self.ssl_types:
            key = _ssl_type_to_key(ssl_type)
            self.heads[key] = nn.Sequential(
                nn.Linear(backbone.output_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 2),  # Binary classification
            )

    def forward(self, x: torch.Tensor, ssl_type: str = None):
        """
        Args:
            x: (batch_size, 3, 150)
            ssl_type: Which SSL head to use

        Returns:
            logits: (batch_size, 2) for the specified SSL type
        """
        features = self.backbone(x)
        features = features.reshape(features.size(0), -1)

        if ssl_type is not None:
            key = _ssl_type_to_key(ssl_type)
            return self.heads[key](features)
        else:
            # Return all predictions
            return {st: self.heads[_ssl_type_to_key(st)](features) for st in self.ssl_types}


def compute_mtl_loss(
    model: MTLPretrainModel,
    batch: torch.Tensor,
    device: torch.device,
    use_rotation: bool = True,
):
    """Compute MTL loss using binary SSL tasks."""
    sample = batch.to(device).float()
    batch_size = sample.shape[0]

    # Apply rotation augmentation if enabled
    if use_rotation and model.use_rotation:
        sample = rotation_3d(sample)

    total_loss = 0.0
    loss_func = nn.CrossEntropyLoss()

    for ssl_type in model.ssl_types:
        if ssl_type.startswith("binary-"):
            aug_type = ssl_type.removeprefix("binary-")

            # Generate random binary labels: 0 = keep original, 1 = apply transform
            target_labels = torch.randint(0, 2, (batch_size,), device=device).long()

            # Apply augmentation based on labels
            augmented_sample = data_aug(sample, aug_type, device)

            # Create partially augmented samples
            # Where label=1, use augmented; where label=0, use original
            mask = target_labels.view(-1, 1, 1).float()
            partially_augmented = mask * augmented_sample + (1 - mask) * sample

            # Get predictions
            output = model(partially_augmented, ssl_type)

            # Compute loss
            loss = loss_func(output, target_labels)
            total_loss += loss

    # Average loss across all SSL tasks
    avg_loss = total_loss / len(model.ssl_types)

    return avg_loss, avg_loss.item(), 0.0


# =============================================================================
# SimCLR
# =============================================================================

class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss for SimCLR."""

    def __init__(self, batch_size: int, temperature: float = 0.1, device: str = "cuda"):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        batch_size = z_i.size(0)
        representations = torch.cat([z_i, z_j], dim=0)

        # Compute similarity matrix
        sim_matrix = self.similarity(representations.unsqueeze(1), representations.unsqueeze(0))
        sim_matrix = sim_matrix / self.temperature

        # Create mask for positive pairs
        sim_ij = torch.diag(sim_matrix, batch_size)
        sim_ji = torch.diag(sim_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0).reshape(2 * batch_size, 1)

        # Create mask to exclude self-similarities
        mask = (~torch.eye(2 * batch_size, dtype=torch.bool, device=self.device)).float()
        negatives = sim_matrix * mask

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=self.device)

        loss = self.criterion(logits, labels)
        return loss / (2 * batch_size)


class SimCLRModel(nn.Module):
    """SimCLR model with projection head."""

    def __init__(self, backbone: nn.Module, output_dim: int = 1024):
        super().__init__()
        self.backbone = backbone
        self.backbone_dim = backbone.output_dim
        self.output_dim = output_dim
        self.projector = nn.Linear(self.backbone_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = features.reshape(features.size(0), -1)
        z = self.projector(features)
        return F.normalize(z, dim=1)


def compute_simclr_loss(
    model: SimCLRModel,
    batch: torch.Tensor,
    device: torch.device,
    temperature: float = 0.1,
    use_rotation: bool = True,
):
    """Compute SimCLR contrastive loss."""
    x = batch.to(device)
    batch_size = x.size(0)

    # Apply 3D rotation augmentation (if enabled)
    if use_rotation:
        x = rotation_3d(x)

    # Apply two different augmentations
    x_i = data_aug(x, "time_series_reverse", device)
    x_j = data_aug(x, "permute-5", device)

    # Get representations
    z_i = model(x_i)
    z_j = model(x_j)

    # Compute loss
    criterion = NTXentLoss(batch_size, temperature, device)
    loss = criterion(z_i, z_j)

    return loss, loss.item(), 0.0


# =============================================================================
# MoCo
# =============================================================================

class MoCoModel(nn.Module):
    """Momentum Contrast model."""

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int = 512,
        queue_size: int = 65536,
        momentum: float = 0.999,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature

        # Query encoder
        self.encoder_q = backbone
        # Key encoder (momentum updated)
        self.encoder_k = copy.deepcopy(backbone)

        # Disable gradient for key encoder
        for param in self.encoder_k.parameters():
            param.requires_grad = False

        # Create queue
        self.register_buffer("queue", F.normalize(torch.randn(feature_dim, queue_size), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """Update the queue."""
        batch_size = keys.size(0)
        ptr = int(self.queue_ptr)

        # Replace the keys at ptr
        if ptr + batch_size > self.queue_size:
            # Wrap around
            self.queue[:, ptr:] = keys[:self.queue_size - ptr].T
            self.queue[:, :batch_size - (self.queue_size - ptr)] = keys[self.queue_size - ptr:].T
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T

        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(self, x_q: torch.Tensor, x_k: torch.Tensor):
        """
        Args:
            x_q: Query samples
            x_k: Key samples (different augmentation)
        """
        # Compute query features
        q = self.encoder_q(x_q)
        q = q.reshape(q.size(0), -1)
        q = F.normalize(q, dim=1)

        # Compute key features
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(x_k)
            k = k.reshape(k.size(0), -1)
            k = F.normalize(k, dim=1)

        # Positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)

        # Negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # Logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.temperature

        # Labels: positive is always 0
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        # Dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels


def compute_moco_loss(
    model: MoCoModel,
    batch: torch.Tensor,
    device: torch.device,
    use_rotation: bool = True,
):
    """Compute MoCo loss."""
    x = batch.to(device)

    # Apply 3D rotation augmentation (if enabled)
    if use_rotation:
        x = rotation_3d(x)

    # Apply two different augmentations (same as SimCLR)
    x_q = data_aug(x, "time_series_reverse", device)
    x_k = data_aug(x, "permute-5", device)

    logits, labels = model(x_q, x_k)
    loss = F.cross_entropy(logits, labels)

    # Compute accuracy
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        acc = (pred == labels).float().mean().item()

    return loss, loss.item(), acc


# =============================================================================
# CPC (Contrastive Predictive Coding)
# =============================================================================

class CPCModel(nn.Module):
    """Contrastive Predictive Coding model."""

    def __init__(
        self,
        backbone: nn.Module,
        num_steps_prediction: int = 2,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        # For CPC, input_dim is the feature dimension (256), not flattened
        self.input_dim = 256  # ResNetForCPC outputs 256 channels
        self.num_steps_prediction = num_steps_prediction
        self.temperature = temperature

        # GRU for context
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )

        # Prediction heads for each future step
        self.Wk = nn.ModuleList([
            nn.Linear(self.input_dim, self.input_dim) for _ in range(num_steps_prediction)
        ])

        # Initialize weights
        for w in self.Wk:
            nn.init.xavier_uniform_(w.weight)
            nn.init.zeros_(w.bias)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor (batch, 3, 150)
        Returns:
            nce_loss: NCE loss
            accuracy: Prediction accuracy
        """
        batch_size = x.shape[0]

        # Get encoded representations from backbone
        # ResNetForCPC outputs (batch, 256, 4) - we need to use feature_extractor directly
        z = self.backbone.feature_extractor(x)  # (batch, 256, 4)

        # Transpose to (batch, seq_len, dim) for GRU
        z = z.permute(0, 2, 1)  # (batch, 4, 256)

        # Normalize features
        z = F.normalize(z, dim=-1)

        seq_len = z.shape[1]  # Should be 4

        # Random timestep to start future prediction
        max_start = seq_len - self.num_steps_prediction
        if max_start <= 0:
            max_start = 1
        t = random.randint(0, max_start - 1)

        # Get context vector
        c, _ = self.gru(z[:, :t + 1, :])
        c_t = c[:, -1, :]  # (batch, 256)
        c_t = F.normalize(c_t, dim=-1)

        # Compute NCE loss
        nce_loss = 0.0
        correct = 0
        valid_steps = 0

        for k in range(self.num_steps_prediction):
            future_idx = t + 1 + k
            if future_idx >= seq_len:
                break

            valid_steps += 1

            # Prediction
            pred = self.Wk[k](c_t)  # (batch, 256)
            pred = F.normalize(pred, dim=-1)

            # Target (future representation)
            target = z[:, future_idx, :]  # (batch, 256)

            # Similarity matrix
            logits = torch.mm(pred, target.t()) / self.temperature  # (batch, batch)

            # Labels (diagonal is positive)
            labels = torch.arange(batch_size, device=x.device)

            # NCE loss
            nce_loss += F.cross_entropy(logits, labels)

            # Accuracy
            with torch.no_grad():
                correct += (logits.argmax(dim=1) == labels).sum().item()

        if valid_steps > 0:
            nce_loss = nce_loss / valid_steps
            accuracy = correct / (batch_size * valid_steps)
        else:
            accuracy = 0.0

        return nce_loss, accuracy


def compute_cpc_loss(
    model: CPCModel,
    batch: torch.Tensor,
    device: torch.device,
    use_rotation: bool = True,
):
    """Compute CPC loss."""
    x = batch.to(device)

    # Apply 3D rotation augmentation (if enabled)
    if use_rotation:
        x = rotation_3d(x)

    loss, accuracy = model(x)
    return loss, loss.item(), accuracy


# =============================================================================
# Timemask (Masked Time Reconstruction)
# =============================================================================

class TimemaskModel(nn.Module):
    """Time-masked reconstruction model."""

    def __init__(self, backbone: nn.Module, mask_ratio: float = 0.15):
        super().__init__()
        self.backbone = backbone
        self.mask_ratio = mask_ratio

        # Decoder with ConvTranspose1d
        self.feature_expander = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            x: Masked input (batch, 3, 150)
            mask: Binary mask (batch, 150), True = masked
        """
        # ResNet feature extraction
        features = self.backbone(x)  # (batch, 512, 1)

        # Upsample with ConvTranspose1d
        predictions = self.feature_expander(features)  # (batch, 3, 16)

        # Interpolate to 150 timesteps
        predictions = F.interpolate(predictions, size=150, mode='linear', align_corners=False)

        return predictions


def compute_timemask_loss(
    model: TimemaskModel,
    batch: torch.Tensor,
    device: torch.device,
    mask_ratio: float = 0.15,
    use_rotation: bool = True,
):
    """Compute timemask reconstruction loss."""
    original = batch.to(device)

    # Apply 3D rotation augmentation (if enabled)
    if use_rotation:
        original = rotation_3d(original)

    batch_size, channels, seq_len = original.shape

    # Create time mask
    num_masked = int(seq_len * mask_ratio)
    mask_indices = torch.randperm(seq_len, device=device)[:num_masked]
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    mask[:, mask_indices] = True

    # Apply mask (set to zero)
    masked_input = original.clone()
    mask_expanded = mask.unsqueeze(1).expand(-1, channels, -1)
    masked_input[mask_expanded] = 0

    # Reconstruct
    reconstructed = model(masked_input, mask)

    # Loss only on masked positions
    loss = F.mse_loss(
        reconstructed[mask_expanded],
        original[mask_expanded]
    )

    return loss, loss.item(), 0.0


# =============================================================================
# Timechannel (Time + Channel Masked Reconstruction)
# =============================================================================

class TimechannelModel(nn.Module):
    """Time and channel masked reconstruction model."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

        # Decoder with ConvTranspose1d
        self.feature_expander = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor):
        # ResNet feature extraction
        features = self.backbone(x)  # (batch, 512, 1)

        # Upsample with ConvTranspose1d
        predictions = self.feature_expander(features)  # (batch, 3, 16)

        # Interpolate to 150 timesteps
        predictions = F.interpolate(predictions, size=150, mode='linear', align_corners=False)

        return predictions


def compute_timechannel_loss(
    model: TimechannelModel,
    batch: torch.Tensor,
    device: torch.device,
    time_mask_ratio: float = 0.15,
    channel_mask_ratio: float = 0.15,
    alpha: float = 0.5,
    use_rotation: bool = True,
):
    """Compute time+channel masked reconstruction loss."""
    original = batch.to(device)

    # Apply 3D rotation augmentation (if enabled)
    if use_rotation:
        original = rotation_3d(original)

    batch_size, channels, seq_len = original.shape

    # Time mask (shared across batch)
    num_time_masked = int(seq_len * time_mask_ratio)
    time_mask_indices = torch.randperm(seq_len, device=device)[:num_time_masked]
    time_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    time_mask[:, time_mask_indices] = True

    # Channel mask (shared across batch, at least 1 channel)
    num_channel_masked = max(1, int(channels * channel_mask_ratio))
    channel_mask_indices = torch.randperm(channels, device=device)[:num_channel_masked]
    channel_mask = torch.zeros(batch_size, channels, dtype=torch.bool, device=device)
    channel_mask[:, channel_mask_indices] = True

    # Apply masks to input
    masked_input = original.clone()

    # Time masking: (batch, channels, seq) -> (batch, seq, channels) -> mask -> back
    masked_input = masked_input.permute(0, 2, 1)  # (batch, seq, channels)
    masked_input[time_mask] = 0
    masked_input = masked_input.permute(0, 2, 1)  # (batch, channels, seq)

    # Channel masking
    masked_input[channel_mask] = 0

    # Reconstruct
    reconstructed = model(masked_input)

    # Time mask loss calculation
    # (batch, channels, seq) -> (batch, seq, channels)
    predictions_time = reconstructed.permute(0, 2, 1)
    original_time = original.permute(0, 2, 1)

    # Extract only time-masked positions
    time_mask_expanded = time_mask.unsqueeze(-1).expand(-1, -1, channels)
    pred_time_masked = predictions_time[time_mask_expanded]
    orig_time_masked = original_time[time_mask_expanded]
    time_loss = F.mse_loss(pred_time_masked, orig_time_masked)

    # Channel mask loss calculation
    channel_mask_expanded = channel_mask.unsqueeze(-1).expand(-1, -1, seq_len)
    pred_channel_masked = reconstructed[channel_mask_expanded]
    orig_channel_masked = original[channel_mask_expanded]
    channel_loss = F.mse_loss(pred_channel_masked, orig_channel_masked)

    # Weighted combination
    loss = alpha * time_loss + (1 - alpha) * channel_loss

    return loss, loss.item(), 0.0


# =============================================================================
# Training Functions
# =============================================================================

def get_loss_function(method: str):
    """Get the appropriate loss function for the SSL method."""
    loss_functions = {
        "mtl": compute_mtl_loss,
        "simclr": compute_simclr_loss,
        "moco": compute_moco_loss,
        "cpc": compute_cpc_loss,
        "timemask": compute_timemask_loss,
        "timechannel": compute_timechannel_loss,
    }
    return loss_functions[method]


def train_epoch(model, loader, optimizer, device, method: str, use_rotation: bool = True):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_metric1 = 0.0
    total_metric2 = 0.0
    batch_count = 0

    compute_loss = get_loss_function(method)

    for batch, _ in tqdm(loader, desc="Training", leave=False):
        # DataLoader batch_size=files_per_batch, Dataset returns (sample_size, 3, 150)
        # batch shape: (files_per_batch, sample_size, 3, 150)
        # reshape to (files_per_batch * sample_size, 3, 150)
        batch = batch.view(-1, batch.shape[2], batch.shape[3])

        optimizer.zero_grad()

        loss, metric1, metric2 = compute_loss(model, batch, device, use_rotation=use_rotation)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_metric1 += metric1
        total_metric2 += metric2
        batch_count += 1

    return {
        "loss": total_loss / batch_count,
        "metric1": total_metric1 / batch_count,
        "metric2": total_metric2 / batch_count,
    }


def validate(model, loader, device, method: str, use_rotation: bool = True):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    total_metric1 = 0.0
    total_metric2 = 0.0
    batch_count = 0

    compute_loss = get_loss_function(method)

    with torch.no_grad():
        for batch, _ in loader:
            # batch shape: (files_per_batch, sample_size, 3, 150)
            # reshape to (files_per_batch * sample_size, 3, 150)
            batch = batch.view(-1, batch.shape[2], batch.shape[3])

            loss, metric1, metric2 = compute_loss(model, batch, device, use_rotation=use_rotation)

            total_loss += loss.item()
            total_metric1 += metric1
            total_metric2 += metric2
            batch_count += 1

    return {
        "loss": total_loss / batch_count,
        "metric1": total_metric1 / batch_count,
        "metric2": total_metric2 / batch_count,
    }


def create_model(method: str, device: torch.device, use_rotation: bool = True) -> nn.Module:
    """Create model based on SSL method."""
    if method == "cpc":
        # CPC uses 4-layer ResNet (no layer5) to preserve temporal info
        backbone = ResNetForCPC(n_channels=3)
    else:
        backbone = Resnet(n_channels=3)

    if method == "mtl":
        model = MTLPretrainModel(backbone, ssl_types=DEFAULT_SSL_TYPES, use_rotation=use_rotation)
    elif method == "simclr":
        model = SimCLRModel(backbone)
    elif method == "moco":
        model = MoCoModel(backbone)
    elif method == "cpc":
        model = CPCModel(backbone)
    elif method == "timemask":
        model = TimemaskModel(backbone)
    elif method == "timechannel":
        model = TimechannelModel(backbone)
    else:
        raise ValueError(f"Unknown method: {method}")

    return model.to(device)


def get_backbone_state_dict(model: nn.Module, method: str) -> dict:
    """Extract backbone state dict from the model."""
    if method == "moco":
        return model.encoder_q.state_dict()
    elif hasattr(model, "backbone"):
        return model.backbone.state_dict()
    else:
        return model.state_dict()


def main():
    parser = argparse.ArgumentParser(
        description="HARBench Pretraining",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default: MTL pretraining on 14 datasets (Table 1 in paper)
    python pretrain.py --method mtl

    # Custom datasets
    python pretrain.py --method mtl --datasets dsads pamap2 --sensors LeftArm

    # SimCLR pretraining
    python pretrain.py --method simclr --epochs 200

    # MoCo pretraining
    python pretrain.py --method moco

    # CPC pretraining
    python pretrain.py --method cpc

    # Timemask pretraining
    python pretrain.py --method timemask

    # Timechannel pretraining
    python pretrain.py --method timechannel
        """
    )
    parser.add_argument(
        "--method",
        type=str,
        default="mtl",
        choices=SSL_METHODS,
        help=f"SSL method: {', '.join(SSL_METHODS)}"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Dataset names (default: 14 datasets from Table 1)"
    )
    parser.add_argument(
        "--sensors",
        type=str,
        nargs="+",
        default=None,
        help="Sensor names (default: all available sensors)"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help=f"Data root path (default: {DEFAULT_PRETRAIN_DATA_ROOT})"
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size (windows per file)")
    parser.add_argument("--epoch_size", type=int, default=1000, help="Number of file samples per epoch")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="results/pretrain", help="Output directory")
    parser.add_argument("--rotation", action="store_true", default=True, help="Use 3D rotation augmentation (all methods)")
    parser.add_argument("--no-rotation", action="store_false", dest="rotation", help="Disable rotation augmentation")
    args = parser.parse_args()

    # Apply defaults
    if args.datasets is None:
        args.datasets = DEFAULT_PRETRAIN_DATASETS
    if args.data_root is None:
        args.data_root = DEFAULT_PRETRAIN_DATA_ROOT

    # Set seed
    set_seed(args.seed)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{args.method}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    log_path = os.path.join(output_dir, "train.log")
    log_file = open(log_path, "w")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"HARBench Pretraining")
    log(f"Method: {args.method}")
    log(f"Datasets: {args.datasets}")
    log(f"Sensors: {args.sensors if args.sensors else 'all available'}")
    log(f"Data root: {args.data_root}")
    log(f"Output: {output_dir}")
    log("")

    # Create dataloaders (train/val split)
    log("Creating dataloaders...")
    log(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, Epoch size: {args.epoch_size}")
    train_loader, val_loader = create_pretrain_dataloaders(
        datasets=args.datasets,
        sensors=args.sensors,
        data_root=args.data_root,
        batch_size=args.batch_size,
        train_epoch_size=args.epoch_size,
        val_epoch_size=100,
        val_ratio=0.1,
        seed=args.seed,
    )
    log(f"Training batches per epoch: {len(train_loader)}")

    # Create model
    log(f"Creating {args.method.upper()} model...")
    log(f"Rotation augmentation: {args.rotation}")
    if args.method == "mtl":
        log(f"SSL types: {DEFAULT_SSL_TYPES}")
    model = create_model(args.method, device, use_rotation=args.rotation)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Total parameters: {total_params:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    log("\nStarting training...")
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []
    import time
    start_time = time.time()
    n_batches = len(train_loader)

    for epoch in range(args.epochs):
        epoch_start = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, device, args.method, use_rotation=args.rotation)
        val_metrics = validate(model, val_loader, device, args.method, use_rotation=args.rotation)
        epoch_time = time.time() - epoch_start

        is_best = val_metrics["loss"] < best_val_loss
        best_marker = " *" if is_best else ""
        log(f"  Epoch {epoch+1}/{args.epochs}: 100%|{'█'*10}| {n_batches}/{n_batches} [{epoch_time:.2f}s] train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f}{best_marker}")

        history.append({
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
        })

        # Save best model
        if is_best:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch + 1
            patience_counter = 0

            # Save backbone checkpoint
            checkpoint_path = os.path.join(output_dir, "best.pth")
            backbone_state_dict = get_backbone_state_dict(model, args.method)
            torch.save(backbone_state_dict, checkpoint_path)
        else:
            patience_counter += 1

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(output_dir, f"epoch_{epoch+1}.pth")
            backbone_state_dict = get_backbone_state_dict(model, args.method)
            torch.save(backbone_state_dict, checkpoint_path)
            log(f"  -> Saved checkpoint at epoch {epoch+1}")

        if patience_counter >= args.patience:
            log(f"\nEarly stopping at epoch {epoch+1}")
            break

        scheduler.step()

    log(f"\nTraining completed. Best epoch: {best_epoch}")

    # Save final results
    results = {
        "method": args.method,
        "datasets": args.datasets,
        "sensors": args.sensors,
        "seed": args.seed,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "patience": args.patience,
        },
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "final_epoch": len(history),
        "history": history,
        "timestamp": timestamp,
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    log(f"\nResults saved to: {results_path}")
    log(f"Checkpoint saved to: {os.path.join(output_dir, 'best.pth')}")

    log_file.close()


if __name__ == "__main__":
    main()
