"""
Common utility functions for preprocessing
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging
from scipy import signal

logger = logging.getLogger(__name__)


def resample_timeseries(
    data: np.ndarray,
    labels: np.ndarray,
    original_rate: float,
    target_rate: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample time series data to target sampling rate

    Uses polyphase filtering to resample smoothly while applying
    anti-aliasing filters.

    Args:
        data: Input data (samples, features)
        labels: Labels (samples,)
        original_rate: Original sampling rate (Hz)
        target_rate: Target sampling rate (Hz)

    Returns:
        resampled_data: Resampled data
        resampled_labels: Resampled labels
    """
    if original_rate == target_rate:
        return data, labels

    num_samples = len(data)
    num_features = data.shape[1]

    # Calculate resampling ratio (expressed as integers)
    # Example: 25Hz -> 30Hz = up=6, down=5 (30/25 = 6/5)
    from math import gcd

    # Simplify the ratio
    rate_ratio = target_rate / original_rate
    # Multiply by a sufficiently large number to express as integers
    multiplier = 1000
    up = int(target_rate * multiplier)
    down = int(original_rate * multiplier)

    # Simplify using GCD
    common_divisor = gcd(up, down)
    up = up // common_divisor
    down = down // common_divisor

    logger.info(f"Resampling with polyphase filtering: up={up}, down={down} (ratio={rate_ratio:.4f})")

    # Resample each channel individually (polyphase filtering)
    # Resample the first channel to get the exact size
    first_channel_resampled = signal.resample_poly(data[:, 0], up, down)
    new_num_samples = len(first_channel_resampled)

    resampled_data = np.zeros((new_num_samples, num_features))
    resampled_data[:, 0] = first_channel_resampled

    for i in range(1, num_features):
        resampled_data[:, i] = signal.resample_poly(data[:, i], up, down)

    # Resample labels (nearest neighbor interpolation)
    original_indices = np.arange(num_samples)
    new_indices = np.linspace(0, num_samples - 1, new_num_samples)
    resampled_labels = labels[np.round(new_indices).astype(int)]

    logger.info(f"Resampled from {original_rate}Hz to {target_rate}Hz: {data.shape} -> {resampled_data.shape}")

    return resampled_data, resampled_labels


def _split_by_label_boundaries(
    data: np.ndarray,
    labels: np.ndarray
) -> list:
    """
    Split data at label change points

    Args:
        data: Input data (samples, features)
        labels: Labels (samples,)

    Returns:
        segments: List of [(data, labels), ...] (each segment has a single label)
    """
    if len(data) == 0:
        return []

    # Detect label change points
    change_points = np.where(labels[:-1] != labels[1:])[0] + 1

    # List of split points (add start and end)
    split_points = [0] + change_points.tolist() + [len(data)]

    segments = []
    for i in range(len(split_points) - 1):
        start = split_points[i]
        end = split_points[i + 1]
        segment_data = data[start:end]
        segment_labels = labels[start:end]
        segments.append((segment_data, segment_labels))

    return segments


def create_sliding_windows(
    data: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    stride: int,
    drop_last: bool = False,
    pad_last: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply sliding windows to time series data

    Windows are created only within single-label segments.
    Data is split at label change points, then windows are created within each segment.

    Args:
        data: Input data (samples, features) or (samples, channels, features)
        labels: Labels (samples,)
        window_size: Window size
        stride: Stride (slide width)
        drop_last: Whether to drop the last incomplete window
        pad_last: Whether to pad the last incomplete window

    Returns:
        windows: Windowed data (num_windows, window_size, ...)
        window_labels: Label for each window (num_windows,)
    """
    # Split at label boundaries then create windows within each segment
    segments = _split_by_label_boundaries(data, labels)
    return _create_windows_from_segments(
        segments, window_size, stride, drop_last, pad_last
    )


def _create_windows_from_segments(
    segments: list,
    window_size: int,
    stride: int,
    drop_last: bool = False,
    pad_last: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create windows from segment list (internal function for strict mode)
    Since each segment has a single label, use the first value for window labels
    """
    all_windows = []
    all_labels = []

    for segment_data, segment_labels in segments:
        if len(segment_data) < window_size:
            # Skip if segment is shorter than window_size
            continue

        # Use the first value since this segment has a single label
        segment_label = segment_labels[0]

        # Sliding windows within the segment
        for start in range(0, len(segment_data) - window_size + 1, stride):
            end = start + window_size
            window = segment_data[start:end]
            all_windows.append(window)
            all_labels.append(segment_label)

    if len(all_windows) == 0:
        return np.array([]), np.array([])

    return np.array(all_windows), np.array(all_labels)


def create_sliding_windows_multi_session(
    sessions: list,
    window_size: int,
    stride: int,
    drop_last: bool = False,
    pad_last: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply sliding windows to multi-session data

    Each session is a unit of temporally continuous data.
    Windows do not span across sessions.
    Windows are created only within single-label segments.

    Args:
        sessions: List of [(data, labels), ...]
            data: Array of (samples, features)
            labels: Array of (samples,)
        window_size: Window size
        stride: Stride (slide width)
        drop_last: Whether to drop the last incomplete window
        pad_last: Whether to pad the last incomplete window

    Returns:
        windows: Windowed data (num_windows, window_size, ...)
        window_labels: Label for each window (num_windows,)
    """
    all_windows = []
    all_labels = []

    for session_data, session_labels in sessions:
        if len(session_data) < window_size:
            # Skip if session is shorter than window_size
            logger.debug(f"Skipping session with {len(session_data)} samples (< {window_size})")
            continue

        windows, labels = create_sliding_windows(
            session_data,
            session_labels,
            window_size=window_size,
            stride=stride,
            drop_last=drop_last,
            pad_last=pad_last
        )

        if len(windows) > 0:
            all_windows.append(windows)
            all_labels.append(labels)

    if len(all_windows) == 0:
        logger.warning("No windows created from any session")
        return np.array([]), np.array([])

    return np.concatenate(all_windows, axis=0), np.concatenate(all_labels, axis=0)


def normalize_data(
    data: np.ndarray,
    method: str = 'standardize',
    axis: Optional[int] = None,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Normalize data

    Args:
        data: Input data
        method: Normalization method ('standardize', 'minmax', 'normalize')
        axis: Axis to apply normalization (None for entire array)
        epsilon: Small value to prevent division by zero

    Returns:
        Normalized data
    """
    if method == 'standardize':
        # Standardization (mean=0, std=1)
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)
        return (data - mean) / (std + epsilon)

    elif method == 'minmax':
        # Min-Max normalization [0, 1]
        min_val = np.min(data, axis=axis, keepdims=True)
        max_val = np.max(data, axis=axis, keepdims=True)
        return (data - min_val) / (max_val - min_val + epsilon)

    elif method == 'normalize':
        # L2 normalization
        norm = np.linalg.norm(data, axis=axis, keepdims=True)
        return data / (norm + epsilon)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def filter_invalid_samples(
    data: np.ndarray,
    labels: np.ndarray,
    threshold: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove invalid samples (NaN, Inf, zero variance, etc.)

    Args:
        data: Input data
        labels: Labels
        threshold: Variance threshold

    Returns:
        Filtered data and labels
    """
    # NaN/Inf check
    valid_mask = ~(np.isnan(data).any(axis=tuple(range(1, data.ndim))) |
                   np.isinf(data).any(axis=tuple(range(1, data.ndim))))

    # Variance check (exclude samples with all zeros)
    variance = np.var(data, axis=tuple(range(1, data.ndim)))
    valid_mask &= variance > threshold

    filtered_data = data[valid_mask]
    filtered_labels = labels[valid_mask]

    removed_count = len(data) - len(filtered_data)
    if removed_count > 0:
        logger.warning(f"Removed {removed_count} invalid samples")

    return filtered_data, filtered_labels


def split_train_val_test(
    data: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    shuffle: bool = True,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/val/test sets

    Args:
        data: Input data
        labels: Labels
        train_ratio: Training data ratio
        val_ratio: Validation data ratio
        test_ratio: Test data ratio
        shuffle: Whether to shuffle
        seed: Random seed

    Returns:
        train_data, train_labels, val_data, val_labels, test_data, test_labels
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    num_samples = len(data)
    indices = np.arange(num_samples)

    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)

    train_end = int(num_samples * train_ratio)
    val_end = train_end + int(num_samples * val_ratio)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    return (
        data[train_indices], labels[train_indices],
        data[val_indices], labels[val_indices],
        data[test_indices], labels[test_indices]
    )


def save_npy_dataset(
    output_path: Path,
    train_data: np.ndarray,
    train_labels: np.ndarray,
    val_data: Optional[np.ndarray] = None,
    val_labels: Optional[np.ndarray] = None,
    test_data: Optional[np.ndarray] = None,
    test_labels: Optional[np.ndarray] = None
) -> None:
    """
    Save processed data in NumPy format

    Args:
        output_path: Output directory
        train_data, train_labels: Training data
        val_data, val_labels: Validation data (optional)
        test_data, test_labels: Test data (optional)
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Training data
    np.save(output_path / 'train_data.npy', train_data)
    np.save(output_path / 'train_labels.npy', train_labels)
    logger.info(f"Saved train data: {train_data.shape}")

    # Validation data
    if val_data is not None and val_labels is not None:
        np.save(output_path / 'val_data.npy', val_data)
        np.save(output_path / 'val_labels.npy', val_labels)
        logger.info(f"Saved val data: {val_data.shape}")

    # Test data
    if test_data is not None and test_labels is not None:
        np.save(output_path / 'test_data.npy', test_data)
        np.save(output_path / 'test_labels.npy', test_labels)
        logger.info(f"Saved test data: {test_data.shape}")


def get_class_distribution(labels: np.ndarray) -> dict:
    """
    Calculate class distribution

    Args:
        labels: Label array

    Returns:
        Number of samples per class
    """
    unique, counts = np.unique(labels, return_counts=True)
    distribution = dict(zip(unique.tolist(), counts.tolist()))
    return distribution
