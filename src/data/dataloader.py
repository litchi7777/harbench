"""
HARBench Data Loading

Load data from processed_strict format.
Structure: {data_root}/{dataset}/USER{id}/{sensor}/{modality}/X.npy, Y.npy
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, RandomSampler
from .dataset import HARDataset


# Default data root (processed format)
# Priority: environment variable > artifact/har-datasets
DEFAULT_DATA_ROOT = os.environ.get(
    "HARBENCH_DATA_ROOT",
    os.path.join(os.path.dirname(__file__), "../../har-datasets/data/processed")
)


# =============================================================================
# Usable class definitions (based on const.py usable_labels and processed_strict activity_map)
#
# Important: const.py labels definition and processed_strict activity_map use different ordering.
# Here, we reference each preprocessor's activity_map and map activity names from
# const.py usable_labels to processed_strict class IDs.
# =============================================================================
USABLE_CLASSES = {
    # Daily
    # DSADS: All 19 classes used
    "dsads": None,

    # FORTHTRACE: 11 classes used
    "forthtrace": None,

    # HARTH: 12 -> 10 classes
    # processed_strict: {walking:0, running:1, shuffling:2, stairs(ascending):3,
    #                   stairs(descending):4, standing:5, sitting:6, lying:7,
    #                   cycling(sit):8, cycling(stand):9, cycling(sit,inactive):10,
    #                   cycling(stand,inactive):11}
    # usable: walking, shuffling, stairs(ascending), stairs(descending), standing,
    #         sitting, lying, cycling(sit), cycling(stand), cycling(sit,inactive)
    # Excluded: running(1), cycling(stand,inactive)(11)
    "harth": [0, 2, 3, 4, 5, 6, 7, 8, 9, 10],

    # IMWSHA: All 11 classes used
    "imwsha": None,

    # PAAL: 24 -> 11 classes (matching const.py n_usable_classes=11)
    # processed_strict: mapped in order 0-23
    # usable: brush_teeth(4), brush_hair(5), take_off_a_jacket(6), put_on_a_jacket(7),
    #         put_on_a_shoe(8), writing(14), type_on_a_keyboard(16),
    #         washing_dishes(20), dusting(21), ironing(22)
    # Note: washing_dishes is duplicated (20 and 23), effectively using 10 classes
    "paal": [4, 5, 6, 7, 8, 14, 16, 20, 21, 22],

    # PAMAP2: processed_strict has only 12 classes (matching const.py n_usable_classes=12)
    # All classes used
    "pamap2": None,

    # SELFBACK: All 9 classes used
    "selfback": None,

    # UCAEHAR: 8 -> 6 classes
    # processed_strict: {WALKING:0, RUNNING:1, STANDING:2, SITTING:3, LYING:4,
    #                   DRINKING:5, WALKING_UPSTAIRS:6, WALKING_DOWNSTAIRS:7}
    # usable: STANDING(2), SITTING(3), WALKING(0), LYING(4), WALKING_UPSTAIRS(6), RUNNING(1)
    # Excluded: DRINKING(5), WALKING_DOWNSTAIRS(7)
    "ucaehar": [0, 1, 2, 3, 4, 6],

    # USCHAD: All classes used (matching const.py n_usable_classes=12)
    "uschad": None,

    # WARD: All 13 classes used
    "ward": None,

    # REALWORLD: All 8 classes used
    "realworld": None,

    # Exercise
    # MEx: All 7 classes used
    "mex": None,

    # MHEALTH: All 12 classes used
    "mhealth": None,

    # REALDISP: All 33 classes used
    "realdisp": None,

    # Industry
    # LARa: 8 -> 6 classes
    # processed_strict: {Standing:0, Walking:1, Cart:2, Handling(upwards):3,
    #                   Handling(centred):4, Handling(downwards):5, Synchronization:6, None:7}
    # usable: Standing(0), Walking(1), Cart(2), Handling(upwards)(3),
    #         Handling(centred)(4), Synchronization(6)
    # Excluded: Handling(downwards)(5), None(7)
    "lara": [0, 1, 2, 3, 4, 6],

    # OPENPACK: All 10 classes used
    "openpack": None,

    # EXOSKELETONS: All 4 classes used
    "exoskeletons": None,

    # VTT_CONIOT: All 16 classes used
    "vtt_coniot": None,
}


def load_dataset(dataset, sensors, data_root=None, modality="ACC"):
    """
    Load dataset (processed_strict format).

    Structure: {data_root}/{dataset}/USER{id}/{sensor}/{modality}/X.npy, Y.npy

    Args:
        dataset: Dataset name (e.g., "DSADS", "PAMAP2")
        sensors: List of sensor names (e.g., ["Chest", "Thigh"])
        data_root: Data root path
        modality: Modality ("ACC", "GYRO", "MAG")

    Returns:
        X: Sensor data (N, C, T)
        Y: Labels (N,)
        U: User IDs (N,)
    """
    if data_root is None:
        data_root = DEFAULT_DATA_ROOT

    dataset_path = os.path.join(data_root, dataset.lower())

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Get list of users
    users = sorted([d for d in os.listdir(dataset_path) if d.startswith("USER")])

    X_all, Y_all, U_all = [], [], []

    for user in users:
        user_id = int(user.replace("USER", "").lstrip("0") or "0")
        user_path = os.path.join(dataset_path, user)

        X_sensors = []
        Y_user = None

        for sensor in sensors:
            sensor_path = os.path.join(user_path, sensor, modality)

            if not os.path.exists(sensor_path):
                continue

            x_path = os.path.join(sensor_path, "X.npy")
            y_path = os.path.join(sensor_path, "Y.npy")

            if not os.path.exists(x_path):
                continue

            x = np.load(x_path)
            y = np.load(y_path)

            # float16 -> float32
            if x.dtype == np.float16:
                x = x.astype(np.float32)

            X_sensors.append(x)
            if Y_user is None:
                Y_user = y

        if not X_sensors:
            continue

        # If sample counts differ between sensors, align to minimum sample count
        min_samples = min(x.shape[0] for x in X_sensors)
        X_sensors = [x[:min_samples] for x in X_sensors]
        Y_user = Y_user[:min_samples]

        # Concatenate sensors (N, C1, T) + (N, C2, T) -> (N, C1+C2, T)
        X_user = np.concatenate(X_sensors, axis=1)
        n_samples = X_user.shape[0]

        X_all.append(X_user)
        Y_all.append(Y_user)
        U_all.append(np.full(n_samples, user_id))

    if not X_all:
        raise ValueError(f"No data found for dataset={dataset}, sensors={sensors}")

    X = np.concatenate(X_all, axis=0)
    Y = np.concatenate(Y_all, axis=0)
    U = np.concatenate(U_all, axis=0)

    # Exclude negative labels (unlabeled data)
    valid_mask = Y >= 0
    if not np.all(valid_mask):
        X = X[valid_mask]
        Y = Y[valid_mask]
        U = U[valid_mask]

    # If USABLE_CLASSES is defined, use only specified classes
    from collections import Counter
    dataset_lower = dataset.lower()
    usable_classes = USABLE_CLASSES.get(dataset_lower)

    if usable_classes is not None:
        # If explicit class list is defined
        class_counts = Counter(Y)
        excluded_classes = [cls for cls in class_counts.keys() if cls not in usable_classes]

        if excluded_classes:
            print(f"  Using predefined usable classes for {dataset_lower}")
            print(f"  Excluding classes: {sorted(excluded_classes)}")
            print(f"  Class counts: {dict(sorted(class_counts.items()))}")

            valid_mask = np.isin(Y, usable_classes)
            X = X[valid_mask]
            Y = Y[valid_mask]
            U = U[valid_mask]

            # Remap labels to consecutive integers starting from 0
            label_map = {old: new for new, old in enumerate(sorted(usable_classes))}
            Y = np.array([label_map[y] for y in Y])

            print(f"  Remaining classes: {len(usable_classes)}, samples: {len(Y)}")

    return X, Y, U


def create_dataloaders(X, Y, U, test_users, val_users, batch_size=64, num_workers=0, data_ratio=1.0,
                       use_weighted_sampler=True, max_samples_per_epoch=None):
    """
    Create DataLoaders for train/val/test.

    Args:
        X: Sensor data (N, C, T)
        Y: Labels (N,)
        U: User IDs (N,)
        test_users: List of test users (e.g., [1, 2])
        val_users: List of validation users (e.g., [3, 4])
        batch_size: Batch size
        num_workers: Number of DataLoader workers (default 0 to avoid conflicts in parallel execution)
        data_ratio: Ratio of training data to use (0.0-1.0)
        use_weighted_sampler: Use WeightedRandomSampler to correct class imbalance
        max_samples_per_epoch: Maximum samples per epoch (None = use training data size)

    Returns:
        train_loader, val_loader, test_loader
    """
    from collections import Counter

    # Split data by user
    test_mask = np.isin(U, test_users)
    val_mask = np.isin(U, val_users)
    train_mask = ~(test_mask | val_mask)

    X_train, Y_train = X[train_mask], Y[train_mask]
    X_val, Y_val = X[val_mask], Y[val_mask]
    X_test, Y_test = X[test_mask], Y[test_mask]

    # Save original training data count (for samples_per_epoch calculation in few-shot)
    n_train_original = len(X_train)

    # Few-shot: Stratified sampling (guarantee at least 1 sample per class)
    if data_ratio < 1.0:
        n_subset = max(1, int(n_train_original * data_ratio))

        # Select samples from each class
        unique_classes = np.unique(Y_train)
        selected_indices = []

        # First select at least 1 sample from each class
        for cls in unique_classes:
            cls_indices = np.where(Y_train == cls)[0]
            selected_indices.append(np.random.choice(cls_indices, 1)[0])

        # Select remaining samples randomly (without replacement)
        remaining = n_subset - len(selected_indices)
        if remaining > 0:
            all_indices = set(range(n_train_original))
            already_selected = set(selected_indices)
            available = list(all_indices - already_selected)
            if len(available) > 0:
                extra = np.random.choice(available, min(remaining, len(available)), replace=False)
                selected_indices.extend(extra.tolist())

        indices = np.array(selected_indices)
        X_train = X_train[indices]
        Y_train = Y_train[indices]

    # Create datasets
    train_dataset = HARDataset(X_train, Y_train)
    val_dataset = HARDataset(X_val, Y_val)
    test_dataset = HARDataset(X_test, Y_test)

    # Use WeightedRandomSampler to correct class imbalance
    if use_weighted_sampler:
        class_count = Counter(Y_train)
        class_weights = {cls: 1.0 / count for cls, count in class_count.items()}
        sample_weights = np.array([class_weights[y] for y in Y_train])
        sample_weights = torch.from_numpy(sample_weights).float()

        # Determine number of samples: use original data count as baseline (ensures same update count in few-shot)
        # If max_samples_per_epoch is specified, use the smaller of it and original data count
        if max_samples_per_epoch is not None:
            samples_per_epoch = min(n_train_original, max_samples_per_epoch)
        else:
            samples_per_epoch = n_train_original

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=samples_per_epoch,
            replacement=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# =============================================================================
# For Pretraining (Self-Supervised Learning)
# =============================================================================

class PretrainDataset(Dataset):
    """
    Dataset for pretraining.

    Randomly samples from each file and returns (batch_size, 3, 150).
    DataLoader uses batch_size=1, and squeeze(0) gets (batch_size, 3, 150).
    """

    def __init__(self, file_paths, sample_size=1000, block_size=None):
        """
        Args:
            file_paths: List of npy file paths
            sample_size: Number of windows to return per getitem (effective batch size)
            block_size: (Unused, kept for compatibility)
        """
        self.file_paths = file_paths
        self.sample_size = sample_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        X = np.load(path, mmap_mode='r')

        if X.dtype == np.float16:
            X = np.array(X, dtype=np.float32)

        num_samples = X.shape[0]

        # Random sampling
        if num_samples >= self.sample_size:
            indices = np.random.choice(num_samples, self.sample_size, replace=False)
        else:
            indices = np.random.choice(num_samples, self.sample_size, replace=True)

        data = X[indices]

        # Preprocessing
        data = np.nan_to_num(data, nan=0.0)
        data = np.clip(data, -10.0, 10.0)

        data = torch.tensor(data, dtype=torch.float32)
        return data, data  # For self-supervised learning (sample_size, 3, 150)


def collect_pretrain_files(datasets, sensors=None, data_root=None, modality="ACC"):
    """
    Collect file paths for pretraining.

    Args:
        datasets: List of dataset names
        sensors: List of sensor names (None = auto-detect all sensors)
        data_root: Data root path
        modality: Modality

    Returns:
        file_paths: List of npy file paths
    """
    if data_root is None:
        data_root = DEFAULT_DATA_ROOT

    file_paths = []

    for dataset in datasets:
        dataset_path = os.path.join(data_root, dataset.lower())
        if not os.path.exists(dataset_path):
            print(f"Warning: {dataset_path} not found, skipping")
            continue

        users = sorted([d for d in os.listdir(dataset_path) if d.startswith("USER")])

        for user in users:
            user_path = os.path.join(dataset_path, user)

            # If sensors is None, auto-detect all sensors
            if sensors is None:
                available_sensors = [
                    d for d in os.listdir(user_path)
                    if os.path.isdir(os.path.join(user_path, d))
                ]
            else:
                available_sensors = sensors

            for sensor in available_sensors:
                x_path = os.path.join(user_path, sensor, modality, "X.npy")
                if os.path.exists(x_path):
                    file_paths.append(x_path)

    return file_paths


def create_pretrain_dataloaders(datasets, sensors, data_root=None, modality="ACC",
                                 batch_size=1000, num_workers=4,
                                 train_epoch_size=2000, val_epoch_size=100,
                                 val_ratio=0.1, seed=42, files_per_batch=4):
    """
    Create train/val DataLoaders for pretraining.

    Args:
        datasets: List of dataset names
        sensors: List of sensor names
        data_root: Data root path
        modality: Modality
        batch_size: Batch size (number of windows to read from one file)
        num_workers: Number of workers
        train_epoch_size: Number of batches per training epoch
        val_epoch_size: Number of batches per validation epoch
        val_ratio: Ratio of files for validation
        seed: Seed for splitting
        files_per_batch: Number of files to read simultaneously (4, same as LS-HAR)

    Returns:
        train_loader, val_loader
    """
    import random

    file_paths = collect_pretrain_files(datasets, sensors, data_root, modality)

    if not file_paths:
        raise ValueError("No data files found for pretraining")

    # Split files into train/val
    random.seed(seed)
    random.shuffle(file_paths)
    n_val = max(1, int(len(file_paths) * val_ratio))
    val_paths = file_paths[:n_val]
    train_paths = file_paths[n_val:]

    print(f"Found {len(file_paths)} files: {len(train_paths)} train, {len(val_paths)} val")
    print(f"Files per batch: {files_per_batch}, Windows per file: {batch_size}")
    print(f"Total windows per batch: {files_per_batch * batch_size}")

    # Train loader
    # LS-HAR: num_samples=1000, batch_size=4 → 250 batches/epoch
    train_dataset = PretrainDataset(train_paths, sample_size=batch_size)
    train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=train_epoch_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=files_per_batch,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Val loader
    val_dataset = PretrainDataset(val_paths, sample_size=batch_size)
    val_sampler = RandomSampler(val_dataset, replacement=True, num_samples=val_epoch_size)
    val_loader = DataLoader(
        val_dataset,
        batch_size=files_per_batch,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader, val_loader
