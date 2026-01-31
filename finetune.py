#!/usr/bin/env python3
"""
HARBench Evaluation Script

Fine-tune and evaluate models on HAR datasets.
Based on scripts/finetune/finetune_main.py.

Supported models (14 total):
  ResNet-based (SSL-Wearables):
    - resnet      : Random init baseline
    - mtl         : Multi-Task Learning pretrained
    - harnet      : HARNet (OxWearables official, torch.hub)
    - simclr      : SimCLR pretrained
    - moco        : MoCo pretrained
    - timechannel : Masked Resnet (time+channel masking)
    - timemask    : Masked Resnet (time masking only)
    - cpc         : Contrastive Predictive Coding

  Transformer-based:
    - selfpab     : SelfPAB (STFT + Transformer)
    - limubert    : LIMU-BERT
    - imumae      : IMU-Video-MAE (ECCV 2024)

  Foundation Models (require additional dependencies):
    - patchtst    : PatchTST (pip install transformers)
    - moment      : MOMENT (pip install momentfm)

Usage:
    python finetune.py --model mtl --dataset dsads --sensors LeftArm LeftLeg
    python finetune.py --model harnet --dataset dsads --sensors LeftArm LeftLeg --data_ratio 0.1
    python finetune.py --model mtl --zeroshot
    python finetune.py --model resnet --zeroshot-supervised
"""

import argparse
import json
import os
import random
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

# All imports from artifact/src (standalone, no parent directory dependency)
from src.data.dataloader import load_dataset, create_dataloaders
from src.data.dataset import HARDataset
from src.models import (
    Resnet, NDeviceResnet, TwoLayerClassifier,
    LIMUBert, IMUVideoMAE,
    SelfPAB, MultiDeviceMaskedResnet, MultiDeviceResnetCPC,
)
from src.utils import macro_f1_score, accuracy

# Optional imports for foundation models
try:
    from transformers import PatchTSTConfig, PatchTSTForClassification
    HAS_PATCHTST = True
except ImportError:
    HAS_PATCHTST = False

try:
    from momentfm import MOMENTPipeline
    HAS_MOMENT = True
except ImportError:
    HAS_MOMENT = False


# =============================================================================
# Model Configuration
# =============================================================================

# Supported models and their configurations
# Based on scripts/finetune/finetune_main.py get_backbone()
MODELS = {
    # SSL-Wearables ResNet variants
    "resnet": {
        "type": "resnet",
        "description": "1D ResNet backbone (scratch, random init)",
    },
    "mtl": {
        "type": "resnet",
        "weights": "pretrained/mtl.pth",
        "description": "Multi-Task Learning pretrained ResNet",
    },
    "harnet": {
        "type": "harnet",
        "description": "HARNet (OxWearables official, torch.hub)",
    },
    "simclr": {
        "type": "resnet",
        "weights": "pretrained/simclr.pth",
        "description": "SimCLR pretrained ResNet",
    },
    "moco": {
        "type": "resnet",
        "weights": "pretrained/moco.pth",
        "description": "MoCo pretrained ResNet",
    },

    # Masked Resnet variants
    "timechannel": {
        "type": "maskedresnet",
        "weights": "pretrained/timechannel.pth",
        "description": "Masked Resnet (time+channel masking)",
    },
    "timemask": {
        "type": "maskedresnet",
        "weights": "pretrained/timemask.pth",
        "description": "Masked Resnet (time masking only)",
    },

    # CPC
    "cpc": {
        "type": "cpc",
        "weights": "pretrained/cpc.pth",
        "description": "Contrastive Predictive Coding ResNet",
    },

    # Transformer-based models
    "selfpab": {
        "type": "selfpab",
        "weights": "pretrained/selfpab.ckpt",
        "description": "SelfPAB Transformer encoder (STFT + Transformer)",
    },
    "limubert": {
        "type": "limubert",
        "weights": "pretrained/limubert.pt",
        "description": "LIMU-BERT Transformer encoder",
    },
    "imumae": {
        "type": "imumae",
        "weights": "pretrained/imumae.pth",
        "description": "IMU-Video-MAE encoder (ECCV 2024)",
    },

    # Foundation Models (require additional dependencies)
    "patchtst": {
        "type": "patchtst",
        "description": "PatchTST (Time Series Foundation Model)",
    },
    "moment": {
        "type": "moment",
        "description": "MOMENT Foundation Model",
    },
}


# =============================================================================
# Constants
# =============================================================================

SEED = 42

FOLDS = [
    {"test": [1, 2], "val": [3, 4]},
    {"test": [3, 4], "val": [5, 6]},
    {"test": [5, 6], "val": [7, 8]},
    {"test": [7, 8], "val": [1, 2]},
]

# Zero-shot: Common activity mapping (6 classes)
ZEROSHOT_DATASETS = ["dsads", "mhealth", "pamap2"]
ZEROSHOT_SUPPORT = ["forthtrace", "realdisp", "realworld", "selfback", "ward"]

# Activity mapping from original dataset labels to common 6 classes (0-indexed)
# Common classes: 0=Static, 1=Walking, 2=Running, 3=Stairs, 4=Jumping, 5=Cycling
# Based on src/data/zero_shot_mapping.py (converted from 1-indexed to 0-indexed)
ACTIVITY_MAPPING = {
    "dsads": {
        # Based on dataset_info.py labels
        0: 0, 1: 0, 2: 0, 3: 0,  # Sitting, Standing, Lying(Back/Right) -> Static
        4: 3, 5: 3,  # StairsUp, StairsDown -> Stairs
        6: 0,  # Standing(Elevator) -> Static
        8: 1, 9: 1, 10: 1,  # Walking variations -> Walking
        11: 2,  # Running -> Running
        14: 5, 15: 5,  # Cycling variations -> Cycling
        17: 4,  # Jumping -> Jumping
    },
    "mhealth": {
        0: 0, 1: 0, 2: 0,  # Standing, Sitting, Lying -> Static
        3: 1,  # Walking -> Walking
        4: 3,  # StairsUp -> Stairs
        8: 5,  # Cycling -> Cycling
        9: 2, 10: 2,  # Jogging, Running -> Running
        11: 4,  # JumpFrontBack -> Jumping
    },
    "pamap2": {
        # Based on dataset_info.py: 0=lying, 1=sitting, 2=standing, 3=walking, 4=running, 5=cycling, 7=ascending stairs, 8=descending stairs, 11=rope jumping
        0: 0, 1: 0, 2: 0,  # lying, sitting, standing -> Static
        3: 1,  # walking -> Walking
        4: 2,  # running -> Running
        5: 5,  # cycling -> Cycling
        7: 3, 8: 3,  # ascending/descending stairs -> Stairs
        11: 4,  # rope jumping -> Jumping
    },
    "forthtrace": {
        0: 0, 1: 0, 2: 0,  # Stand, Sit, Sit and Talk -> Static
        3: 1, 4: 1,  # Walk, Walk and Talk -> Walking
        5: 3, 6: 3,  # Climb stairs variations -> Stairs
    },
    "realdisp": {
        0: 1,  # Walking -> Walking
        1: 2, 2: 2,  # Jogging, Running -> Running
        3: 4, 4: 4, 5: 4, 6: 4, 7: 4,  # Jump variations -> Jumping
        32: 5,  # Cycling -> Cycling
    },
    "realworld": {
        # Based on dataset_info.py: 0=ClimbingDown, 1=ClimbingUp, 2=Jumping, 3=Lying, 4=Running, 5=Sitting, 6=Standing, 7=Walking
        0: 3, 1: 3,  # ClimbingDown, ClimbingUp -> Stairs
        2: 4,  # Jumping -> Jumping
        3: 0, 5: 0, 6: 0,  # Lying, Sitting, Standing -> Static
        4: 2,  # Running -> Running
        7: 1,  # Walking -> Walking
    },
    "selfback": {
        0: 3, 1: 3,  # upstairs, downstairs -> Stairs
        2: 1, 3: 1, 4: 1,  # walk slow/mod/fast -> Walking
        5: 2,  # jogging -> Running
        6: 0, 7: 0, 8: 0,  # standing, sitting, lying -> Static
    },
    "ward": {
        0: 0, 1: 0, 2: 0,  # RestStanding, RestSitting, RestLying -> Static
        3: 1,  # WalkFoward -> Walking
        8: 3, 9: 3,  # GoUp/DownStairs -> Stairs
        10: 2,  # Jog -> Running
        11: 4,  # Jump -> Jumping
    },
}

ZEROSHOT_SENSORS = {
    "dsads": ["LeftArm", "LeftLeg"],
    "mhealth": ["RightWrist", "LeftAnkle"],
    "pamap2": ["hand", "ankle"],
    "forthtrace": ["LeftWrist", "RightThigh"],
    "realdisp": ["LeftLowerArm", "LeftThigh"],
    "realworld": ["Forearm", "Thigh"],
    "selfback": ["Wrist", "Thigh"],
    "ward": ["LeftArm", "LeftAnkle"],
}



# =============================================================================
# Core Functions
# =============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, criterion, optimizer, device, model_type="resnet", max_iterations=None):
    model.train()
    total_loss = 0.0
    total = 0

    for i, (inputs, labels) in enumerate(tqdm(loader, desc="Training", leave=False)):
        if max_iterations is not None and i >= max_iterations:
            break

        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        # Special handling for foundation models
        if model_type == "patchtst":
            # PatchTST expects (batch, seq_len, channels)
            inputs = inputs.permute(0, 2, 1)
            outputs = model(inputs).prediction_logits
        elif model_type == "moment":
            outputs = model.forward(x_enc=inputs).logits
        else:
            outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        total += labels.size(0)

    return total_loss / total if total > 0 else 0


def evaluate(model, loader, criterion, device, model_type="resnet"):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Special handling for foundation models
            if model_type == "patchtst":
                inputs = inputs.permute(0, 2, 1)
                outputs = model(inputs).prediction_logits
            elif model_type == "moment":
                outputs = model.forward(x_enc=inputs).logits
            else:
                outputs = model(inputs)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    loss = total_loss / len(all_labels) if len(all_labels) > 0 else 0
    f1 = macro_f1_score(all_labels, all_preds)
    acc = accuracy(all_labels, all_preds)

    return loss, f1, acc


def create_backbone(model_type, weights_path, num_sensors, in_channels, device):
    """Create backbone based on model type.

    Based on scripts/finetune/finetune_main.py get_backbone() function.
    """
    if model_type == "resnet":
        # NDeviceResnet: Multi-device ResNet with shared weights (SSL-Wearables style)
        backbone = NDeviceResnet(
            state_dict_path=weights_path,
            num_devices=num_sensors,
            device=device,
        )
    elif model_type == "maskedresnet":
        # MultiDeviceMaskedResnet: Masked Reconstruction Model
        backbone = MultiDeviceMaskedResnet(
            device=device,
            num_devices=num_sensors,
            state_dict_path=weights_path,
        )
    elif model_type == "cpc":
        # MultiDeviceResnetCPC: Contrastive Predictive Coding
        backbone = MultiDeviceResnetCPC(
            device=device,
            num_devices=num_sensors,
            state_dict_path=weights_path,
        )
    elif model_type == "selfpab":
        # SelfPAB: STFT + Transformer encoder
        backbone = SelfPAB(
            device=device,
            num_devices=num_sensors,
            checkpoint_path=weights_path if weights_path and weights_path.endswith(".ckpt") else None,
        )
    elif model_type == "limubert":
        # LIMU-BERT: Transformer encoder with automatic resampling
        backbone = LIMUBert(
            feature_num=in_channels,
            hidden=72,
            hidden_ff=144,
            n_layers=4,
            n_heads=4,
            seq_len=150,  # Input: 30Hz, 150 frames
            target_seq_len=120,  # Target: 20Hz for pretrained weights
            emb_norm=True,
            pretrained_path=weights_path if weights_path and weights_path.endswith(".pt") else None,
            device=device,
        )
    elif model_type == "imumae":
        # IMU-Video-MAE: Spectrogram + ViT encoder (ECCV 2024)
        backbone = IMUVideoMAE(
            in_channels=in_channels,
            seq_len=150,
            pretrained_path=weights_path if weights_path and weights_path.endswith(".pth") else None,
            device=device,
        )
    elif model_type == "harnet":
        # HARNet: OxWearables official pretrained model via torch.hub
        # Multi-sensor support: each sensor processed by a separate HARNet
        class NDeviceHARNet(nn.Module):
            def __init__(self, num_devices: int = 1):
                super().__init__()
                self.num_devices = num_devices
                self.output_dim = 512 * num_devices
                self.feature_extractors = nn.ModuleList()
                for _ in range(num_devices):
                    harnet = torch.hub.load(
                        'OxWearables/ssl-wearables', 'harnet5',
                        class_num=5, pretrained=True
                    ).feature_extractor
                    self.feature_extractors.append(harnet)

            def forward(self, x):
                # x: (batch, num_devices * 3, seq_len)
                outputs = []
                for i in range(self.num_devices):
                    x_i = x[:, i*3:(i+1)*3, :]
                    out = self.feature_extractors[i](x_i)
                    outputs.append(out)
                return torch.cat(outputs, dim=1)

        backbone = NDeviceHARNet(num_devices=num_sensors)
    elif model_type == "patchtst":
        # PatchTST: Return None, handled separately in train_model
        if not HAS_PATCHTST:
            raise ImportError("PatchTST requires: pip install transformers")
        return None  # Special case: full model created in train_model
    elif model_type == "moment":
        # MOMENT: Return None, handled separately in train_model
        if not HAS_MOMENT:
            raise ImportError("MOMENT requires: pip install momentfm")
        return None  # Special case: full model created in train_model
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return backbone


def train_model(train_loader, val_loader, test_loader, n_classes, num_sensors,
                weights_path, device, args, model_type="resnet", log_func=None):
    """Train and evaluate a model."""
    if log_func is None:
        log_func = print
    in_channels = num_sensors * 3

    # Special handling for foundation models that don't use backbone + classifier pattern
    if model_type == "patchtst":
        config = PatchTSTConfig(
            num_input_channels=in_channels,
            num_targets=n_classes,
            context_length=150,
            patch_length=16,
            stride=16,
            use_cls_token=True,
        )
        model = PatchTSTForClassification(config=config)
        model = model.to(device)
    elif model_type == "moment":
        # MOMENT foundation model
        model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-small",
            model_kwargs={
                'task_name': 'classification',
                'n_channels': in_channels,
                'num_class': n_classes,
            }
        )
        model.init()  # Initialize classification head
        model = model.to(device)
    else:
        # Standard backbone + classifier pattern
        backbone = create_backbone(model_type, weights_path, num_sensors, in_channels, device)
        model = TwoLayerClassifier(backbone, n_classes=n_classes)
        model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    best_model_state = None
    patience_counter = 0

    import time
    start_time = time.time()

    max_iter = getattr(args, 'max_iterations', None)
    for epoch in range(args.epochs):
        epoch_start = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, model_type, max_iterations=max_iter)
        val_loss, val_f1, val_acc = evaluate(model, val_loader, criterion, device, model_type)
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time

        is_best = val_f1 > best_val_f1
        best_marker = " *" if is_best else ""
        n_batches = len(train_loader)

        if not args.quiet:
            log_func(f"  Epoch {epoch+1}/{args.epochs}: 100%|{'█'*10}| {n_batches}/{n_batches} [{epoch_time:.2f}s] train_loss={train_loss:.4f} val_loss={val_loss:.4f}{best_marker}")

        if is_best:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            if not args.quiet:
                log_func(f"  Early stopping at epoch {epoch+1} (patience={args.patience})")
            break

        scheduler.step()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    test_loss, test_f1, test_acc = evaluate(model, test_loader, criterion, device, model_type)

    return {
        "test_f1": float(test_f1),
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
        "best_val_f1": float(best_val_f1),
    }


# =============================================================================
# Mode: Finetune
# =============================================================================

def run_finetune(args):
    """Standard fine-tuning with 4-fold CV."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Get model configuration
    model_config = MODELS.get(args.model, MODELS["resnet"])
    model_type = model_config["type"]

    # Determine weights path (CLI arg > model config > None)
    weights_path = args.weights
    if weights_path is None and "weights" in model_config:
        weights_path = model_config["weights"]

    # Setup logging to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{args.model}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "log.txt")
    log_file = open(log_path, "w")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"Using device: {device}")
    log(f"Model: {args.model} ({model_config['description']})")
    log(f"Weights: {weights_path or 'scratch (random init)'}")
    log(f"Loading dataset: {args.dataset}")
    log(f"Sensors: {args.sensors}")

    # Load data
    X, Y, U = load_dataset(args.dataset, args.sensors, args.data_root)
    log(f"Data shape: X={X.shape}, Y={Y.shape}, U={U.shape}")

    n_classes = len(np.unique(Y))
    num_sensors = len(args.sensors)
    log(f"Classes: {n_classes}, Sensors: {num_sensors}")

    fold_results = []

    for fold_idx, fold in enumerate(FOLDS):
        log(f"\n{'='*60}")
        log(f"Fold {fold_idx + 1}/4: test_users={fold['test']}, val_users={fold['val']}")
        log(f"{'='*60}")

        train_loader, val_loader, test_loader = create_dataloaders(
            X, Y, U, fold["test"], fold["val"],
            batch_size=args.batch_size, data_ratio=args.data_ratio,
            max_samples_per_epoch=args.max_samples_per_epoch
        )

        result = train_model(
            train_loader, val_loader, test_loader,
            n_classes, num_sensors,
            weights_path, device, args,
            model_type=model_type,
            log_func=log
        )

        log(f"Fold {fold_idx + 1} Result: F1={result['test_f1']:.4f}, Acc={result['test_acc']:.4f}")
        fold_results.append(result)

    # Aggregate
    mean_f1 = np.mean([r["test_f1"] for r in fold_results])
    std_f1 = np.std([r["test_f1"] for r in fold_results])
    mean_acc = np.mean([r["test_acc"] for r in fold_results])
    std_acc = np.std([r["test_acc"] for r in fold_results])

    log(f"\n{'='*60}")
    log(f"Final Results (4-fold CV)")
    log(f"{'='*60}")
    log(f"Macro F1: {mean_f1:.4f} +/- {std_f1:.4f}")
    log(f"Accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")

    results = {
        "mode": "finetune",
        "model": args.model,
        "model_type": model_type,
        "dataset": args.dataset,
        "sensors": args.sensors,
        "weights": weights_path,
        "seed": args.seed,
        "n_classes": n_classes,
        "num_sensors": num_sensors,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "patience": args.patience,
            "weight_decay": 1e-5,
            "scheduler": "CosineAnnealingLR",
            "optimizer": "Adam",
            "data_ratio": args.data_ratio,
        },
        "fold_results": fold_results,
        "summary": {
            "mean_f1": float(mean_f1),
            "std_f1": float(std_f1),
            "mean_acc": float(mean_acc),
            "std_acc": float(std_acc),
        },
        "timestamp": timestamp,
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    log(f"\nResults saved to: {results_path}")
    log_file.close()
    return results


# =============================================================================
# Mode: Zero-shot
# =============================================================================

def load_and_map_dataset(dataset_name, data_root=None):
    """Load dataset and map labels to common activities."""
    sensors = ZEROSHOT_SENSORS.get(dataset_name, [])
    if not sensors:
        return None, None, None

    try:
        X, Y, U = load_dataset(dataset_name, sensors, data_root)
    except Exception as e:
        print(f"Warning: Failed to load {dataset_name}: {e}")
        return None, None, None

    mapping = ACTIVITY_MAPPING.get(dataset_name, {})
    Y_mapped = np.array([mapping.get(int(y), -1) for y in Y])

    valid_mask = Y_mapped >= 0
    X = X[valid_mask]
    Y_mapped = Y_mapped[valid_mask]
    U = U[valid_mask]

    return X, Y_mapped, U


def run_zeroshot(args):
    """Zero-shot (LODO) evaluation."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Get model configuration
    model_config = MODELS.get(args.model, MODELS["resnet"])
    model_type = model_config["type"]

    # Determine weights path (CLI arg > model config > None)
    weights_path = args.weights
    if weights_path is None and "weights" in model_config:
        weights_path = model_config["weights"]

    # Create output directory and log file (same as finetune)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_suffix = f"_{args.zeroshot}" if args.zeroshot != "all" else ""
    output_dir = os.path.join(args.output_dir, f"{timestamp}_{args.model}_zeroshot{target_suffix}")
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "log.txt")
    log_file = open(log_path, "w")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    # Multi-seed configuration
    seeds = [args.seed, args.seed + 1, args.seed + 2, args.seed + 3]
    num_seeds = len(seeds)

    log(f"Using device: {device}")
    log(f"Model: {args.model} ({model_config['description']})")
    log(f"Weights: {weights_path or 'scratch (random init)'}")
    log(f"Zero-shot LODO Evaluation ({num_seeds}-seed average: {seeds})")

    results = {}

    # Determine which targets to evaluate
    if args.zeroshot == "all":
        targets = ZEROSHOT_DATASETS
    else:
        if args.zeroshot not in ZEROSHOT_DATASETS:
            log(f"Error: Invalid target '{args.zeroshot}'. Choose from: {ZEROSHOT_DATASETS}")
            return {}
        targets = [args.zeroshot]

    for target in targets:
        log(f"\n{'='*60}")
        log(f"Target: {target} (Zero-shot)")
        log(f"{'='*60}")

        X_target, Y_target, _ = load_and_map_dataset(target, args.data_root)
        if X_target is None:
            continue

        # Load training data (excluding target)
        train_datasets = [d for d in ZEROSHOT_DATASETS + ZEROSHOT_SUPPORT if d != target]
        X_train_list, Y_train_list = [], []

        for dataset in train_datasets:
            X, Y, _ = load_and_map_dataset(dataset, args.data_root)
            if X is not None:
                X_train_list.append(X)
                Y_train_list.append(Y)
                log(f"  Loaded {dataset}: {X.shape[0]} samples")

        if not X_train_list:
            continue

        X_all = np.concatenate(X_train_list, axis=0)
        Y_all = np.concatenate(Y_train_list, axis=0)

        seed_f1s = []
        seed_accs = []

        for seed_idx, seed in enumerate(seeds):
            log(f"\n  --- Seed {seed} ({seed_idx+1}/{num_seeds}) ---")
            set_seed(seed)

            # Split into train/val (80/20)
            from sklearn.model_selection import train_test_split
            X_train, X_val, Y_train, Y_val = train_test_split(
                X_all, Y_all, test_size=0.2, random_state=seed, stratify=Y_all
            )

            log(f"  Train: {X_train.shape[0]} samples, Val: {X_val.shape[0]} samples")
            log(f"  Target test: {X_target.shape[0]} samples")

            # Create datasets
            train_dataset = HARDataset(X_train, Y_train)
            val_dataset = HARDataset(X_val, Y_val)
            test_dataset = HARDataset(X_target, Y_target)

            # WeightedRandomSampler for class-balanced training
            from collections import Counter
            class_count = Counter(Y_train)
            class_weights = {cls: 1.0 / count for cls, count in class_count.items()}
            sample_weights = np.array([class_weights[y] for y in Y_train])
            sample_weights = torch.from_numpy(sample_weights).float()

            # samples_per_epoch: min(train_size, max_samples) or train_size if None
            if args.max_samples_per_epoch is not None:
                samples_per_epoch = min(len(Y_train), args.max_samples_per_epoch)
            else:
                samples_per_epoch = len(Y_train)
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=samples_per_epoch,
                replacement=True
            )

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

            # Create model
            num_sensors = len(ZEROSHOT_SENSORS[target])
            in_channels = num_sensors * 3
            n_classes = 6  # Zero-shot uses 6 common classes

            # Special handling for foundation models (same as train_model)
            if model_type == "patchtst":
                config = PatchTSTConfig(
                    num_input_channels=in_channels,
                    num_targets=n_classes,
                    context_length=150,
                    patch_length=16,
                    stride=16,
                    use_cls_token=True,
                )
                model = PatchTSTForClassification(config=config)
                model = model.to(device)
            elif model_type == "moment":
                model = MOMENTPipeline.from_pretrained(
                    "AutonLab/MOMENT-1-small",
                    model_kwargs={
                        'task_name': 'classification',
                        'n_channels': in_channels,
                        'num_class': n_classes,
                    }
                )
                model.init()
                model = model.to(device)
            else:
                # Standard backbone + classifier pattern
                backbone = create_backbone(model_type, weights_path, num_sensors, in_channels, device)
                model = TwoLayerClassifier(backbone, n_classes=n_classes)
                model = model.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            criterion = nn.CrossEntropyLoss()

            best_f1 = 0.0
            best_state = None
            patience_counter = 0
            n_batches = len(train_loader)

            for epoch in range(args.epochs):
                import time
                start_time = time.time()

                # Train
                model.train()
                total_loss = 0.0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # Special handling for foundation models
                    if model_type == "patchtst":
                        inputs_t = inputs.permute(0, 2, 1)
                        outputs = model(inputs_t).prediction_logits
                    elif model_type == "moment":
                        outputs = model.forward(x_enc=inputs).logits
                    else:
                        outputs = model(inputs)

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                train_loss = total_loss / len(train_loader)

                # Evaluate on validation (for model selection)
                val_loss, val_f1, val_acc = evaluate(model, val_loader, criterion, device, model_type)

                epoch_time = time.time() - start_time
                is_best = val_f1 > best_f1
                best_marker = " *" if is_best else ""

                log(f"  Epoch {epoch+1}/{args.epochs}: 100%|{'█'*10}| {n_batches}/{n_batches} [{epoch_time:.2f}s] train_loss={train_loss:.4f} val_loss={val_loss:.4f}{best_marker}")

                if is_best:
                    best_f1 = val_f1
                    best_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= args.patience:
                    log(f"  Early stopping at epoch {epoch+1}")
                    break

                scheduler.step()

            if best_state:
                model.load_state_dict(best_state)
            # Final evaluation on test (target dataset)
            _, f1, acc = evaluate(model, test_loader, criterion, device, model_type)

            log(f"  Seed {seed} Result: F1={f1:.4f}, Acc={acc:.4f}")
            seed_f1s.append(f1)
            seed_accs.append(acc)

        # Average over seeds
        mean_f1 = float(np.mean(seed_f1s))
        mean_acc = float(np.mean(seed_accs))
        std_f1 = float(np.std(seed_f1s))
        log(f"Target {target} Result: F1={mean_f1:.4f} (±{std_f1:.4f}), Acc={mean_acc:.4f}")
        results[target] = {"f1": mean_f1, "acc": mean_acc, "std_f1": std_f1, "seed_f1s": [float(f) for f in seed_f1s]}

    # Summary
    if results:
        mean_f1 = np.mean([r["f1"] for r in results.values()])

        log(f"\n{'='*60}")
        log(f"Zero-shot Results Summary ({num_seeds}-seed average)")
        log(f"{'='*60}")
        for dataset, r in results.items():
            log(f"  {dataset}: F1={r['f1']:.4f} (±{r['std_f1']:.4f})")
        log(f"  Average: F1={mean_f1:.4f}")

        final_results = {
            "mode": "zeroshot",
            "model": args.model,
            "model_type": model_type,
            "weights": weights_path,
            "seeds": seeds,
            "dataset_results": results,
            "summary": {"mean_f1": float(mean_f1)},
            "timestamp": timestamp,
        }

        results_path = os.path.join(output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(final_results, f, indent=2)

        log(f"\nResults saved to: {results_path}")
        log_file.close()
        return final_results

    log_file.close()
    return {}


# =============================================================================
# Mode: Zero-shot Supervised (Upper Bound Reference)
# =============================================================================

def run_zeroshot_supervised(args):
    """Supervised evaluation on zero-shot target datasets (DSADS, MHEALTH, PAMAP2).

    This provides the upper bound reference for zero-shot evaluation by training
    and testing on the same target dataset using 4-fold CV with 6 common activity classes.
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Get model configuration
    model_config = MODELS.get(args.model, MODELS["resnet"])
    model_type = model_config["type"]

    # Determine weights path (CLI arg > model config > None)
    weights_path = args.weights
    if weights_path is None and "weights" in model_config:
        weights_path = model_config["weights"]

    # Create output directory and log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{timestamp}_{args.model}_zeroshot_supervised")
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "log.txt")
    log_file = open(log_path, "w")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"Using device: {device}")
    log(f"Model: {args.model} ({model_config['description']})")
    log(f"Weights: {weights_path or 'scratch (random init)'}")
    log(f"Zero-shot Supervised Evaluation (Upper Bound)")

    results = {}

    for target in ZEROSHOT_DATASETS:
        log(f"\n{'='*60}")
        log(f"Target: {target} (Supervised - 4-fold CV)")
        log(f"{'='*60}")

        # Load and map dataset to common 6 classes
        X, Y, U = load_and_map_dataset(target, args.data_root)
        if X is None:
            log(f"  Skipping {target}: failed to load")
            continue

        log(f"  Data shape: X={X.shape}, Y={Y.shape}, U={U.shape}")
        log(f"  Classes: {len(np.unique(Y))}, Sensors: {len(ZEROSHOT_SENSORS[target])}")

        num_sensors = len(ZEROSHOT_SENSORS[target])
        n_classes = 6  # Common activity classes

        fold_results = []

        for fold_idx, fold in enumerate(FOLDS):
            log(f"\n  Fold {fold_idx + 1}/4: test_users={fold['test']}, val_users={fold['val']}")

            train_loader, val_loader, test_loader = create_dataloaders(
                X, Y, U, fold["test"], fold["val"],
                batch_size=args.batch_size, data_ratio=args.data_ratio,
                max_samples_per_epoch=args.max_samples_per_epoch
            )

            # Check if loaders have data
            if len(train_loader) == 0 or len(test_loader) == 0:
                log(f"    Skipping fold {fold_idx + 1}: insufficient data")
                continue

            result = train_model(
                train_loader, val_loader, test_loader,
                n_classes, num_sensors,
                weights_path, device, args,
                model_type=model_type,
                log_func=lambda msg: log(f"    {msg}")
            )

            log(f"    Fold {fold_idx + 1} Result: F1={result['test_f1']:.4f}, Acc={result['test_acc']:.4f}")
            fold_results.append(result)

        if fold_results:
            mean_f1 = np.mean([r["test_f1"] for r in fold_results])
            std_f1 = np.std([r["test_f1"] for r in fold_results])
            mean_acc = np.mean([r["test_acc"] for r in fold_results])

            log(f"\n  {target} Result: F1={mean_f1:.4f} +/- {std_f1:.4f}")
            results[target] = {
                "f1": float(mean_f1),
                "f1_std": float(std_f1),
                "acc": float(mean_acc),
                "fold_results": fold_results,
            }

    # Summary
    if results:
        mean_f1 = np.mean([r["f1"] for r in results.values()])

        log(f"\n{'='*60}")
        log(f"Zero-shot Supervised Results Summary (Upper Bound)")
        log(f"{'='*60}")
        for dataset, r in results.items():
            log(f"  {dataset}: F1={r['f1']:.4f} +/- {r['f1_std']:.4f}")
        log(f"  Average: F1={mean_f1:.4f}")

        final_results = {
            "mode": "zeroshot_supervised",
            "model": args.model,
            "model_type": model_type,
            "weights": weights_path,
            "seed": args.seed,
            "dataset_results": results,
            "summary": {"mean_f1": float(mean_f1)},
            "timestamp": timestamp,
        }

        results_path = os.path.join(output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(final_results, f, indent=2)

        log(f"\nResults saved to: {results_path}")
        log_file.close()
        return final_results

    log_file.close()
    return {}


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="HARBench Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported models (13 total):

  ResNet-based (SSL-Wearables):
    resnet       - Random init baseline
    mtl          - Multi-Task Learning pretrained
    harnet       - HARNet (OxWearables official)
    simclr       - SimCLR pretrained
    moco         - MoCo pretrained
    timechannel  - Masked Resnet (time+channel)
    timemask     - Masked Resnet (time only)
    cpc          - Contrastive Predictive Coding

  Transformer-based:
    selfpab      - SelfPAB (STFT + Transformer)
    limubert     - LIMU-BERT
    imumae       - IMU-Video-MAE (ECCV 2024)

  Foundation Models:
    patchtst     - PatchTST (requires transformers)
    moment       - MOMENT (requires momentfm)

Examples:
  python finetune.py --model mtl --dataset dsads --sensors LeftArm LeftLeg
  python finetune.py --model harnet --dataset dsads --sensors LeftArm
  python finetune.py --model patchtst --dataset dsads --sensors LeftArm LeftLeg
"""
    )
    parser.add_argument("--model", type=str, default="resnet",
                        choices=list(MODELS.keys()),
                        help="Model to use (default: resnet)")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name")
    parser.add_argument("--sensors", type=str, nargs="+", default=None, help="Sensor names")
    parser.add_argument("--weights", type=str, default=None,
                        help="Override pretrained weights path (optional)")
    parser.add_argument("--data_root", type=str, default=None, help="Data root path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--max_samples_per_epoch", type=int, default=3200,
                        help="Max samples per epoch (default=3200). Capped to training data size.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--data_ratio", type=float, default=1.0, help="Training data ratio (for few-shot)")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--quiet", action="store_true", help="Suppress epoch-level output")
    parser.add_argument("--zeroshot", nargs="?", const="all", default=None,
                        help="Run zero-shot (LODO) evaluation. Optionally specify target: dsads, mhealth, pamap2, or 'all'")
    parser.add_argument("--zeroshot-supervised", action="store_true",
                        help="Run supervised evaluation on zero-shot target datasets (upper bound reference)")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.zeroshot:
        args.output_dir = os.path.join(args.output_dir, "zeroshot")
        run_zeroshot(args)
    elif getattr(args, 'zeroshot_supervised', False):
        args.output_dir = os.path.join(args.output_dir, "zeroshot_supervised")
        run_zeroshot_supervised(args)
    else:
        if args.dataset is None or args.sensors is None:
            parser.error("--dataset and --sensors are required")
        args.output_dir = os.path.join(args.output_dir, "finetune")
        run_finetune(args)


if __name__ == "__main__":
    main()
