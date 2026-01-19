"""
HARBench Backbone Models

Feature extraction backbones for Human Activity Recognition:
- Resnet: 1D CNN ResNet backbone (SSL-Wearables style)
- NDeviceResnet: Multi-sensor ResNet with shared weights
- LIMUBert: LIMU-BERT Transformer encoder
- IMUVideoMAE: IMU-Video-MAE encoder (spectrogram-based)
"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import interpolate
import einops


# =============================================================================
# ResNet Components
# =============================================================================

class Downsample(nn.Module):
    """
    Anti-aliasing downsampling layer.

    Uses convolution with box filter kernels for smooth downsampling.
    order=0: box filter (average pooling)
    order=1: triangle filter (linear)
    order=2: cubic filter

    Reference: https://richzhang.github.io/antialiased-cnns/
    """

    def __init__(self, channels: int, factor: int = 2, order: int = 1):
        super().__init__()
        assert factor > 1, "Downsampling factor must be > 1"

        self.stride = factor
        self.channels = channels
        self.order = order

        # Compute padding
        total_padding = order * (factor - 1)
        assert total_padding % 2 == 0, (
            "order*(factor-1) must be divisible by 2"
        )
        self.padding = int(total_padding / 2)

        # Build anti-aliasing kernel
        box_kernel = np.ones(factor)
        kernel = np.ones(factor)
        for _ in range(order):
            kernel = np.convolve(kernel, box_kernel)
        kernel /= np.sum(kernel)
        kernel = torch.tensor(kernel, dtype=torch.float32)

        # Register as buffer (not a parameter)
        self.register_buffer(
            "kernel", kernel[None, None, :].repeat((channels, 1, 1))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(
            x,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            groups=x.shape[1],
        )


class ResBlock(nn.Module):
    """
    Residual Block for 1D CNN.

    Structure:
        bn-relu-conv-bn-relu-conv
       /                         \
    x --------------------------(+)->
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int = 2,
    ):
        super().__init__()

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride, padding, bias=False, padding_mode="circular"
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride, padding, bias=False, padding_mode="circular"
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.relu(self.bn1(x))
        x = self.conv1(x)
        x = self.relu(self.bn2(x))
        x = self.conv2(x)
        return x + identity


class Resnet(nn.Module):
    """
    1D ResNet for IMU time series.

    Architecture: x->[Conv-[ResBlock]^m-BN-ReLU-Down]^n->y

    Input: (batch_size, n_channels, sequence_length)
    Output: (batch_size, 512, 1) or (batch_size, 512) after squeeze

    Reference: SSL-Wearables (OxWearables)
    """

    def __init__(self, n_channels: int = 3):
        super().__init__()

        # Layer configuration:
        # (out_channels, conv_kernel, n_resblocks, resblock_kernel, downfactor, downorder)
        cfg = [
            (64, 5, 2, 5, 2, 2),
            (128, 5, 2, 5, 2, 2),
            (256, 5, 2, 5, 3, 1),
            (256, 5, 2, 5, 3, 1),
            (512, 5, 0, 5, 3, 1),
        ]

        self.output_dim = 512
        in_channels = n_channels

        feature_extractor = nn.Sequential()
        for i, layer_params in enumerate(cfg):
            (out_channels, conv_kernel, n_resblocks,
             resblock_kernel, downfactor, downorder) = layer_params

            feature_extractor.add_module(
                f"layer{i+1}",
                self._make_layer(
                    in_channels, out_channels, conv_kernel,
                    n_resblocks, resblock_kernel, downfactor, downorder
                ),
            )
            in_channels = out_channels

        self.feature_extractor = feature_extractor

    @staticmethod
    def _make_layer(
        in_channels: int,
        out_channels: int,
        conv_kernel_size: int,
        n_resblocks: int,
        resblock_kernel_size: int,
        downfactor: int,
        downorder: int,
    ) -> nn.Sequential:
        """Build a single layer: Conv-[ResBlock]^m-BN-ReLU-Down"""

        assert conv_kernel_size % 2, "conv_kernel_size must be odd"
        assert resblock_kernel_size % 2, "resblock_kernel_size must be odd"

        conv_padding = (conv_kernel_size - 1) // 2
        resblock_padding = (resblock_kernel_size - 1) // 2

        modules = [
            nn.Conv1d(
                in_channels, out_channels, conv_kernel_size,
                1, conv_padding, bias=False, padding_mode="circular"
            )
        ]

        for _ in range(n_resblocks):
            modules.append(
                ResBlock(
                    out_channels, out_channels,
                    resblock_kernel_size, 1, resblock_padding
                )
            )

        modules.append(nn.BatchNorm1d(out_channels))
        modules.append(nn.ReLU(True))
        modules.append(Downsample(out_channels, downfactor, downorder))

        return nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_channels, sequence_length)

        Returns:
            (batch_size, 512, 1)
        """
        return self.feature_extractor(x)


class NDeviceResnet(nn.Module):
    """
    Multi-sensor ResNet backbone.

    Each sensor (3 channels: x, y, z) is processed by a separate Resnet,
    and the outputs are concatenated.

    Input: (batch_size, num_devices * 3, sequence_length)
    Output: (batch_size, 512 * num_devices, 1)
    """

    def __init__(
        self,
        state_dict_path: Optional[str] = None,
        num_devices: int = 1,
        device: str = "cpu",
    ):
        """
        Args:
            state_dict_path: Path to pretrained weights (.pth file)
            num_devices: Number of sensors (each has 3 channels)
            device: Device to load model on
        """
        super().__init__()

        self.num_devices = num_devices
        self.output_dim = 512 * num_devices

        self.feature_extractors = nn.ModuleList()

        for _ in range(num_devices):
            extractor = Resnet(n_channels=3)

            # Load pretrained weights if provided
            if state_dict_path and state_dict_path.endswith(".pth"):
                state_dict = torch.load(
                    state_dict_path, map_location=device, weights_only=True
                )
                extractor.load_state_dict(state_dict, strict=True)

            self.feature_extractors.append(extractor.to(device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_devices * 3, sequence_length)

        Returns:
            (batch_size, 512 * num_devices, 1)
        """
        # Split input by device (each device has 3 channels)
        sensor_inputs = [
            x[:, i * 3:(i + 1) * 3, :]
            for i in range(self.num_devices)
        ]

        # Process each sensor
        outputs = [
            self.feature_extractors[i](sensor_input)
            for i, sensor_input in enumerate(sensor_inputs)
        ]

        # Concatenate along channel dimension
        return torch.cat(outputs, dim=1)


# =============================================================================
# LIMU-BERT
# =============================================================================

def _gelu(x: torch.Tensor) -> torch.Tensor:
    """Gaussian Error Linear Unit."""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class _LayerNorm(nn.Module):
    """Layer Normalization for LIMU-BERT."""

    def __init__(self, hidden_size: int, eps: float = 1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class _Embeddings(nn.Module):
    """Input embeddings for LIMU-BERT."""

    def __init__(self, feature_num: int, hidden: int, seq_len: int, emb_norm: bool = True):
        super().__init__()
        self.lin = nn.Linear(feature_num, hidden)
        self.pos_embed = nn.Embedding(seq_len, hidden)
        self.norm = _LayerNorm(hidden) if emb_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(batch_size, seq_len)
        e = self.lin(x) + self.pos_embed(pos)
        return self.norm(e)


class _MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention for LIMU-BERT."""

    def __init__(self, hidden: int, n_heads: int):
        super().__init__()
        self.proj_q = nn.Linear(hidden, hidden)
        self.proj_k = nn.Linear(hidden, hidden)
        self.proj_v = nn.Linear(hidden, hidden)
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, H = x.shape
        q = self.proj_q(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.proj_k(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.proj_v(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        return context.transpose(1, 2).contiguous().view(B, S, H)


class _PositionWiseFFN(nn.Module):
    """Position-wise Feed-Forward Network."""

    def __init__(self, hidden: int, hidden_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden_ff)
        self.fc2 = nn.Linear(hidden_ff, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(_gelu(self.fc1(x)))


class LIMUBert(nn.Module):
    """
    LIMU-BERT encoder for HAR.

    Handles input resampling from 30Hz (150 frames) to 20Hz (120 frames)
    to match LIMU-BERT pretrained model expectations.

    Input: (batch_size, channels, time_steps)
    Output: (batch_size, hidden)

    Reference: https://github.com/dapowan/LIMU-BERT-Public
    """

    def __init__(
        self,
        feature_num: int = 6,
        hidden: int = 72,
        hidden_ff: int = 144,
        n_layers: int = 4,
        n_heads: int = 4,
        seq_len: int = 150,
        target_seq_len: int = 120,
        emb_norm: bool = True,
        pretrained_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.seq_len = seq_len
        self.target_seq_len = target_seq_len
        self.output_dim = hidden

        # Build transformer components (matching original LIMU-BERT structure)
        self.transformer = nn.Module()
        self.transformer.embed = _Embeddings(feature_num, hidden, target_seq_len, emb_norm)
        self.transformer.attn = _MultiHeadAttention(hidden, n_heads)
        self.transformer.proj = nn.Linear(hidden, hidden)
        self.transformer.norm1 = _LayerNorm(hidden)
        self.transformer.pwff = _PositionWiseFFN(hidden, hidden_ff)
        self.transformer.norm2 = _LayerNorm(hidden)
        self.transformer.n_layers = n_layers

        if pretrained_path and pretrained_path.endswith(".pt"):
            self._load_pretrained(pretrained_path)

    def _load_pretrained(self, path: str):
        """Load pretrained weights with channel adaptation.

        Pretrained weights are trained on 6 channels (2 devices × 3 axes).
        This method extracts the first N channels to match the current model.
        """
        print(f"Loading LIMU-BERT weights from {path}")
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            encoder_state = {
                k: v for k, v in checkpoint.items()
                if k.startswith("transformer.")
            }

            if not encoder_state:
                return

            # Get current model's feature_num from embed layer
            current_feature_num = self.transformer.embed.lin.weight.shape[1]

            # Handle embed.lin.weight size mismatch (pretrained: 6 channels)
            embed_key = "transformer.embed.lin.weight"
            if embed_key in encoder_state:
                pretrained_weight = encoder_state[embed_key]
                pretrained_channels = pretrained_weight.shape[1]

                if pretrained_channels != current_feature_num:
                    # Extract first N channels from pretrained weights
                    # Pretrained: (hidden, 6), Current: (hidden, N)
                    if current_feature_num <= pretrained_channels:
                        # Use first N channels (e.g., first device's 3 axes)
                        encoder_state[embed_key] = pretrained_weight[:, :current_feature_num]
                        print(f"  Adapted embed weights: {pretrained_channels} -> {current_feature_num} channels")
                    else:
                        # Current model has more channels, tile the pretrained weights
                        repeats = (current_feature_num + pretrained_channels - 1) // pretrained_channels
                        tiled = pretrained_weight.repeat(1, repeats)[:, :current_feature_num]
                        encoder_state[embed_key] = tiled
                        print(f"  Tiled embed weights: {pretrained_channels} -> {current_feature_num} channels")

            missing, unexpected = self.load_state_dict(encoder_state, strict=False)
            loaded_count = len(encoder_state) - len(unexpected)
            print(f"  Loaded {loaded_count} parameters")

        except Exception as e:
            print(f"  Error loading weights: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, channels, time_steps)

        Returns:
            (batch_size, hidden)
        """
        # Resample if necessary
        if x.shape[2] != self.target_seq_len:
            x = F.interpolate(x, size=self.target_seq_len, mode='linear', align_corners=False)

        # Transpose: (batch, channels, time) -> (batch, time, channels)
        x = x.transpose(1, 2)

        # Embedding
        h = self.transformer.embed(x)

        # Transformer blocks (weight sharing)
        for _ in range(self.transformer.n_layers):
            attn_out = self.transformer.attn(h)
            h = self.transformer.norm1(h + self.transformer.proj(attn_out))
            h = self.transformer.norm2(h + self.transformer.pwff(h))

        # Global average pooling
        return h.mean(dim=1)


# =============================================================================
# IMU-Video-MAE
# =============================================================================

try:
    import torchaudio
    from timm.models.layers import to_2tuple, DropPath
    from timm.models.vision_transformer import Attention, Mlp
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False


def _get_sinusoid_encoding(n_position: int, d_hid: int) -> torch.Tensor:
    """Generate sinusoidal position encoding."""
    def get_pos_angle_vec(pos):
        return [pos / pow(10000, 2 * (j // 2) / d_hid) for j in range(d_hid)]

    table = torch.zeros(n_position, d_hid)
    for pos_i in range(n_position):
        table[pos_i] = torch.tensor(get_pos_angle_vec(pos_i))
    table[:, 0::2] = torch.sin(table[:, 0::2])
    table[:, 1::2] = torch.cos(table[:, 1::2])
    return table.unsqueeze(0)


class IMUVideoMAE(nn.Module):
    """
    IMU-Video-MAE encoder for HAR.

    Converts raw acceleration to spectrogram and processes with
    Vision Transformer.

    Input: (batch_size, channels, time)
    Output: (batch_size, embed_dim * num_imus)

    Reference: https://github.com/mf-zhang/IMU-Video-MAE
    """

    def __init__(
        self,
        in_channels: int = 3,
        seq_len: int = 150,
        embed_dim: int = 768,
        encoder_depth: int = 11,
        num_heads: int = 12,
        patch_size: int = 16,
        target_length: int = 512,
        pretrained_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        if not TORCHAUDIO_AVAILABLE:
            raise ImportError(
                "IMUVideoMAE requires torchaudio and timm. "
                "Install with: pip install torchaudio timm"
            )

        self.device = device or torch.device("cpu")
        self.in_channels = in_channels
        self.num_imus = max(1, in_channels // 3)
        self.embed_dim = embed_dim
        self.target_length = target_length
        self.patch_size = patch_size
        self.output_dim = embed_dim * self.num_imus

        # Spectrogram transform
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=256, win_length=24, hop_length=1, power=2
        )

        # Patch embedding
        plot_height = 128
        self.patch_embed_a = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # Position encoding
        num_patches = (target_length // patch_size) * (plot_height // patch_size)
        self.pos_embed_a = _get_sinusoid_encoding(num_patches, embed_dim)

        # Modality embedding
        self.modality_a = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer blocks
        self.blocks_a = nn.ModuleList([
            self._make_block(embed_dim, num_heads)
            for _ in range(encoder_depth)
        ])

        self.output_norm = nn.LayerNorm(embed_dim)

        self._init_weights()

        if pretrained_path:
            self._load_pretrained(pretrained_path)

    def _make_block(self, dim: int, num_heads: int) -> nn.Module:
        """Create a transformer block."""
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm1 = nn.LayerNorm(dim)
                self.norm1_a = nn.LayerNorm(dim)
                self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True)
                self.drop_path = nn.Identity()
                self.norm2 = nn.LayerNorm(dim)
                self.norm2_a = nn.LayerNorm(dim)
                self.mlp = Mlp(in_features=dim, hidden_features=dim * 4)

            def forward(self, x, modality=None):
                if modality == 'a':
                    x = x + self.drop_path(self.attn(self.norm1_a(x)))
                    x = x + self.drop_path(self.mlp(self.norm2_a(x)))
                else:
                    x = x + self.drop_path(self.attn(self.norm1(x)))
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
                return x

        return Block()

    def _init_weights(self):
        """Initialize weights."""
        w = self.patch_embed_a.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.normal_(self.modality_a, std=.02)

    def _load_pretrained(self, path: str):
        """Load pretrained weights."""
        print(f"Loading IMU-Video-MAE weights from {path}")
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            state_dict = {}
            for key, value in checkpoint.items():
                new_key = key.replace('module.', '')
                if any(k in new_key for k in ['patch_embed_a', 'modality_a', 'blocks_a']):
                    if 'norm_a' not in new_key:
                        state_dict[new_key] = value

            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            print(f"  Loaded {len(state_dict)} parameters")
        except Exception as e:
            print(f"  Error loading weights: {e}")

    def _to_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform to spectrogram."""
        batch_size, channels, _ = waveform.shape
        waveform = waveform - waveform.mean(dim=-1, keepdim=True)

        specs = []
        for c in range(channels):
            spec = self.spec_transform(waveform[:, c, :])
            spec_db = torchaudio.transforms.AmplitudeToDB()(spec)
            spec_db = spec_db.transpose(1, 2)[:, :, :128]
            specs.append(spec_db)

        result = torch.stack(specs, dim=1)

        # Pad/truncate to target length
        if result.shape[2] < self.target_length:
            pad_len = self.target_length - result.shape[2]
            result = F.pad(result, (0, 0, 0, pad_len))
        else:
            result = result[:, :, :self.target_length, :]

        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, channels, time)

        Returns:
            (batch_size, embed_dim * num_imus)
        """
        num_imus = x.shape[1] // 3
        if num_imus == 0:
            num_imus = 1

        features = []
        for i in range(num_imus):
            x_imu = x[:, i*3:(i+1)*3, :]
            x_spec = self._to_spectrogram(x_imu)

            # (batch, 3, time, freq) -> (batch, 3, freq, time)
            a = x_spec.transpose(2, 3)

            # Patch embedding
            a = self.patch_embed_a(a).flatten(2).transpose(1, 2)

            # Add position and modality embeddings
            a = a + self.pos_embed_a.type_as(a).to(a.device)
            a = a + self.modality_a

            # Transformer blocks
            for blk in self.blocks_a:
                a = blk(a)

            a = self.output_norm(a)
            feat = a.mean(dim=1)
            features.append(feat)

        return torch.cat(features, dim=1)


# =============================================================================
# SelfPAB and related models
# =============================================================================

class SelfPAB(nn.Module):
    """
    Self-supervised Positional and Attention-based Network (SelfPAB)
    Processes 3-axis sensor data from mobile devices using Transformer encoder.

    This is a standalone version for the artifact that includes the TransformerEncoder.
    """
    def __init__(self, device, num_devices=1, checkpoint_path=None):
        super(SelfPAB, self).__init__()
        # Output dimension matches stft output: (time_frames * feature_dim) = 9 * 78 * num_devices
        self.output_dim = 702 * num_devices
        self.device = device
        self.num_devices = num_devices

        # TransformerEncoder parameters
        args = {
            'input_dim': 78,  # STFT feature dim (3 sensors * 26 freq bins)
            'd_model': 1500,
            'nhead': 6,
            'dim_feedforward': 2048,
            'dropout': 0.1,
            'n_encoder_layers': 4,
            'seq_operation': None,
            'n_prediction_head_layers': 1,
            'dim_prediction_head': None,
            'output_dim': 78,
            'positional_encoding': 'AbsolutePositionalEncoding',
        }

        # Create TransformerEncoder for each device
        self.transformerEncoders = nn.ModuleList()

        # Default checkpoint path (relative to artifact directory)
        if checkpoint_path is None:
            import os
            checkpoint_path = os.path.join(os.path.dirname(__file__), "../../pretrained/selfpab.ckpt")

        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True)

        for _ in range(num_devices):
            encoder = _TransformerEncoderNetwork(args)
            encoder.load_state_dict(checkpoint["state_dict"])
            self.transformerEncoders.append(encoder)

    def convert_30Hz_to_50Hz(self, x):
        """Resample 30Hz sensor data to 50Hz."""
        original_freq = 30
        new_freq = 50
        n_samples = 150

        time_old = np.linspace(0, 5, n_samples)
        time_new = np.linspace(0, 5, int(n_samples * new_freq / original_freq))

        reshaped_data = x.cpu().numpy().reshape(x.shape[0] * x.shape[1], n_samples)
        f = interpolate.interp1d(time_old, reshaped_data, kind='linear', axis=-1)
        resampled_data = f(time_new)

        np_data_resampled = resampled_data.reshape(x.shape[0], x.shape[1], 250)
        return np_data_resampled

    def stft(self, x):
        """Apply Short-Time Fourier Transform to input data."""
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, device=self.device)

        n_fft = 50
        hop_length = 25
        window = torch.hann_window(n_fft, device=self.device)

        stft_outputs = []
        for i in range(x.shape[0]):
            x_stft = torch.stft(
                input=x[i],
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window=window,
                center=False,
                return_complex=True
            )
            stft_outputs.append(x_stft)

        x_stft_batched = torch.stack(stft_outputs)
        x_magnitude = torch.abs(x_stft_batched)
        x_stacked = einops.rearrange(x_magnitude, 'B C F T -> B T (C F)')

        mean = 0.8352
        std = 3.1103
        x_normalized = (x_stacked - mean) / (std + 1e-6)
        return x_normalized

    def forward(self, x):
        """Forward pass - acts as backbone returning features only."""
        x = self.convert_30Hz_to_50Hz(x)
        x = self.stft(x)
        x = x.float()

        device_outputs = []
        for i in range(self.num_devices):
            start_idx = i * 78
            end_idx = (i + 1) * 78
            device_x = x[:, :, start_idx:end_idx]
            device_output = self.transformerEncoders[i](device_x)
            device_outputs.append(device_output)

        x = torch.cat(device_outputs, dim=-1)
        x = x.reshape(x.shape[0], self.output_dim)

        return x


class MultiDeviceMaskedResnet(nn.Module):
    """Multi-device Masked Resnet model."""

    def __init__(self, device, num_devices=1, state_dict_path=None):
        super(MultiDeviceMaskedResnet, self).__init__()
        self.output_dim = 512 * num_devices
        self.device = device
        self.num_devices = num_devices

        self.resnet_encoders = nn.ModuleList()
        for _ in range(num_devices):
            encoder = Resnet()
            if state_dict_path and state_dict_path.endswith(".pth"):
                self._load_state_dict(encoder, state_dict_path)
            encoder.output_dim = 1024
            self.resnet_encoders.append(encoder)

    def _load_state_dict(self, encoder, state_dict_path):
        """Load MaskedResnet weights."""
        model_keys = set(encoder.state_dict().keys())
        raw_state_dict = torch.load(state_dict_path, map_location=self.device, weights_only=True)

        filtered_state_dict = {}
        for key, value in raw_state_dict.items():
            if key in model_keys:
                filtered_state_dict[key] = value
            elif key.startswith("encoder."):
                new_key = key.replace("encoder.", "")
                if new_key in model_keys:
                    filtered_state_dict[new_key] = value
            elif key.startswith("backbone.feature_extractor."):
                new_key = key.replace("backbone.", "")
                if new_key in model_keys:
                    filtered_state_dict[new_key] = value
            elif key.startswith("backbone.") and key[9:] in model_keys:
                filtered_state_dict[key[9:]] = value

        encoder.load_state_dict(filtered_state_dict, strict=True)

    def forward(self, x):
        """Process multi-device data."""
        device_outputs = []
        channels_per_device = 3
        for i in range(self.num_devices):
            start_idx = i * channels_per_device
            end_idx = (i + 1) * channels_per_device
            device_x = x[:, start_idx:end_idx, :]
            device_output = self.resnet_encoders[i](device_x)
            device_outputs.append(device_output)

        x = torch.cat(device_outputs, dim=-1)
        return x


class MultiDeviceResnetCPC(nn.Module):
    """Multi-device ResnetCPC model."""

    def __init__(self, device, num_devices=1, state_dict_path=None):
        super(MultiDeviceResnetCPC, self).__init__()
        self.output_dim = 1024 * num_devices
        self.device = device
        self.num_devices = num_devices

        self.resnet_encoders = nn.ModuleList()
        for _ in range(num_devices):
            encoder = ResNetForCPC()
            if state_dict_path and state_dict_path.endswith(".pth"):
                self._load_state_dict(encoder, state_dict_path)
            encoder.output_dim = 1024
            self.resnet_encoders.append(encoder)

    def _load_state_dict(self, encoder, state_dict_path):
        """Load ResnetCPC weights."""
        try:
            state_dict = torch.load(state_dict_path, map_location=self.device, weights_only=True)
            encoder.load_state_dict(state_dict, strict=True)
        except Exception as e:
            print(f"Warning: Could not load state dict from {state_dict_path}: {e}")

    def forward(self, x):
        """Process multi-device data."""
        device_outputs = []
        channels_per_device = 3
        for i in range(self.num_devices):
            start_idx = i * channels_per_device
            end_idx = (i + 1) * channels_per_device
            device_x = x[:, start_idx:end_idx, :]
            device_output = self.resnet_encoders[i](device_x)
            device_outputs.append(device_output)

        x = torch.cat(device_outputs, dim=-1)
        return x


class ResNetForCPC(nn.Module):
    """
    ResNet for CPC learning (removes last layer to preserve temporal info).

    Input: (batch_size, 3, 150)
    Output: (batch_size, 256, 4) -> flattened to (batch_size, 1024)
    """

    def __init__(self, n_channels=3):
        super(ResNetForCPC, self).__init__()

        # Config without the last layer
        cgf = [
            (64, 5, 2, 5, 2, 2),
            (128, 5, 2, 5, 2, 2),
            (256, 5, 2, 5, 3, 1),
            (256, 5, 2, 5, 3, 1),
        ]

        self.output_dim = 256 * 4
        in_channels = n_channels
        feature_extractor = nn.Sequential()

        for i, layer_params in enumerate(cgf):
            (
                out_channels,
                conv_kernel_size,
                n_resblocks,
                resblock_kernel_size,
                downfactor,
                downorder,
            ) = layer_params
            feature_extractor.add_module(
                f"layer{i+1}",
                Resnet._make_layer(
                    in_channels,
                    out_channels,
                    conv_kernel_size,
                    n_resblocks,
                    resblock_kernel_size,
                    downfactor,
                    downorder,
                ),
            )
            in_channels = out_channels

        self.feature_extractor = feature_extractor

    def forward(self, x):
        """
        Args:
            x: (batch_size, 3, 150)
        Returns:
            (batch_size, 1024)
        """
        x = self.feature_extractor(x)  # (batch_size, 256, 4)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 1024)
        return x


# =============================================================================
# Transformer components for SelfPAB
# =============================================================================

class _AbsolutePositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, dropout=0.0, max_len=10000, batch_first=True):
        super().__init__()
        self.dropout_prob = dropout
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        if batch_first:
            pe = einops.rearrange(pe, 'S B D -> B S D')
        self.register_buffer('pe', pe)
        self.batch_first = batch_first

    def forward(self, x):
        if type(x) == list:
            x, _ = x
        if not self.batch_first:
            pe = self.pe[:x.size(0)]
        else:
            pe = self.pe[:, :x.size(1)]
        if self.dropout_prob != 0.0:
            return self.dropout(pe)
        else:
            return pe


class _InputEmbeddingPosEncoding(nn.Module):
    """Combined InputEmbedding and PositionalEncoding."""

    def __init__(self, args):
        super().__init__()
        self.lin_proj_layer = nn.Linear(
            in_features=args['input_dim'],
            out_features=args['d_model']
        )
        self.pos_encoder = _AbsolutePositionalEncoding(
            d_model=args['d_model'],
            dropout=0.0
        )

    def forward(self, x):
        pe_x = self.pos_encoder(x)
        x = self.lin_proj_layer(x[0] if type(x) == list else x)
        return x + pe_x


class _LinearRelu(nn.Module):
    """Linear layer followed by ReLU."""

    def __init__(self, input_dim, output_dim, dropout=0.0):
        super().__init__()
        linear = nn.Linear(in_features=input_dim, out_features=output_dim)
        relu = nn.ReLU()
        if dropout > 0.0:
            dropout_layer = nn.Dropout(dropout)
            self.linear_relu = nn.Sequential(linear, relu, dropout_layer)
        else:
            self.linear_relu = nn.Sequential(linear, relu)

    def forward(self, x):
        return self.linear_relu(x)


class _MLP(nn.Module):
    """MLP implementation."""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super().__init__()
        from collections import OrderedDict
        layers = OrderedDict()
        _lastdim = input_dim
        for i in range(num_layers - 1):
            layers[f'layer{i}'] = _LinearRelu(input_dim=_lastdim, output_dim=hidden_dim)
            _lastdim = hidden_dim
        layers['out_layer'] = nn.Linear(in_features=_lastdim, out_features=output_dim)
        self.mlp = nn.Sequential(layers)

    def forward(self, x):
        return self.mlp(x)


class _TransformerEncoderNetwork(nn.Module):
    """
    TransformerEncoder network for SelfPAB.
    Standalone version without pytorch_lightning.
    """

    def __init__(self, args):
        super().__init__()
        self.emb = _InputEmbeddingPosEncoding(args)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args['d_model'],
            nhead=args['nhead'],
            dim_feedforward=args['dim_feedforward'],
            dropout=args['dropout'],
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=args['n_encoder_layers']
        )

        self.seq_operation = args.get('seq_operation', None)

        # Prediction head
        num_layers = args['n_prediction_head_layers']
        if num_layers == 0:
            self.prediction_head = nn.Identity()
        else:
            hidden_dim = args.get('dim_prediction_head', args['d_model'])
            if hidden_dim is None:
                hidden_dim = args['d_model']
            self.prediction_head = _MLP(
                num_layers=num_layers,
                input_dim=args['d_model'],
                hidden_dim=hidden_dim,
                output_dim=args['output_dim']
            )

    def forward(self, x):
        x = self.emb(x)
        x = self.transformer_encoder(x)
        x = self._apply_seq_operation(x)
        x = self.prediction_head(x)
        return x

    def _apply_seq_operation(self, x, dim=1):
        """Apply operation on sequence dimension."""
        if self.seq_operation is None:
            return x
        elif self.seq_operation == 'mean':
            return torch.mean(x, dim=dim)
        elif self.seq_operation == 'max':
            return torch.max(x, dim=dim)[0]
        elif self.seq_operation == 'last':
            return torch.select(x, dim=dim, index=-1)
        elif self.seq_operation == 'flatten':
            return torch.flatten(x, start_dim=dim)
        else:
            raise ValueError(f'Seq operation {self.seq_operation} not implemented')
