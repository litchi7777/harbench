"""
HARTH Dataset Preprocessing

HARTH Dataset:
- 12 physical activities (including cycling)
- 22 subjects
- 2 sensors (lower back, right thigh)
- Sampling rate: 50Hz
- Accelerometer only (3-axis, G units)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
import logging

from .base import BasePreprocessor
from .utils import (
    create_sliding_windows,
    filter_invalid_samples,
    resample_timeseries,
    get_class_distribution
)
from .common import (
    download_file,
    extract_archive,
    check_dataset_exists
)
from . import register_preprocessor
from ..dataset_info import DATASETS

logger = logging.getLogger(__name__)

# HARTH Dataset URL (manual download required)
HARTH_URL = "https://archive.ics.uci.edu/static/public/779/harth.zip"


@register_preprocessor('harth')
class HarthPreprocessor(BasePreprocessor):
    """
    Preprocessor class for HARTH dataset
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # HARTH specific settings
        self.num_activities = 12
        self.num_subjects = 22
        self.num_sensors = 2
        self.num_channels = 6  # 2 sensors × 3 axes

        # Sensor names and channel mapping
        # Channel configuration:
        # LowerBack: ACC(3) = 3
        # RightThigh: ACC(3) = 3
        self.sensor_names = ['LowerBack', 'RightThigh']
        self.sensor_channel_ranges = {
            'LowerBack': (0, 3),   # channels 0-2
            'RightThigh': (3, 6)   # channels 3-5
        }

        # Modalities (channel division within each sensor)
        self.sensor_modalities = {
            'LowerBack': {
                'ACC': (0, 3)   # 3-axis accelerometer
            },
            'RightThigh': {
                'ACC': (0, 3)   # 3-axis accelerometer
            }
        }

        # Sampling rate
        self.original_sampling_rate = 50  # Hz (HARTH original)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz (target)

        # Preprocessing parameters
        self.window_size = config.get('window_size', 150)  # 5 seconds @ 30Hz
        self.stride = config.get('stride', 30)  # 1 second @ 30Hz

        # Scaling factor (not needed as data is already in G units)
        self.scale_factor = DATASETS.get('HARTH', {}).get('scale_factor', None)

        # Label mapping (original non-sequential labels → 0-indexed)
        # 1 -> 0 (Walking)
        # 2 -> 1 (Running)
        # 3 -> 2 (Shuffling)
        # 4 -> 3 (Stairs Up)
        # 5 -> 4 (Stairs Down)
        # 6 -> 5 (Standing)
        # 7 -> 6 (Sitting)
        # 8 -> 7 (Lying)
        # 13 -> 8 (Cycling Seated)
        # 14 -> 9 (Cycling Standing)
        # 130 -> 10 (Cycling Seated Inactive)
        # 140 -> 11 (Cycling Standing Inactive)
        self.label_mapping = {
            1: 0,    # Walking
            2: 1,    # Running
            3: 2,    # Shuffling
            4: 3,    # Stairs Up
            5: 4,    # Stairs Down
            6: 5,    # Standing
            7: 6,    # Sitting
            8: 7,    # Lying
            13: 8,   # Cycling Seated
            14: 9,   # Cycling Standing
            130: 10, # Cycling Seated Inactive
            140: 11  # Cycling Standing Inactive
        }

    def get_dataset_name(self) -> str:
        return 'harth'

    def download_dataset(self) -> None:
        """
        Download HARTH dataset
        """
        logger.info("=" * 80)
        logger.info("Downloading HARTH dataset")
        logger.info("=" * 80)

        dataset_path = self.raw_data_path / self.dataset_name
        data_dir = dataset_path / self.dataset_name
        data_exists = data_dir.exists() and any(data_dir.glob('S*.csv'))

        if data_exists:
            logger.info(f"HARTH data already exists at {dataset_path}")
            return

        dataset_path.mkdir(parents=True, exist_ok=True)
        zip_path = dataset_path / "harth.zip"

        # Download
        logger.info("Downloading HARTH archive...")
        download_file(HARTH_URL, zip_path, desc='Downloading HARTH')

        # Extract
        logger.info("Extracting HARTH archive...")
        extract_archive(zip_path, dataset_path, desc='Extracting HARTH')

        # Cleanup
        if zip_path.exists():
            zip_path.unlink()

        logger.info(f"HARTH dataset downloaded to {dataset_path}")

    def load_raw_data(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Load HARTH raw data per subject

        Expected format:
        - data/raw/harth/harth/S006.csv
        - Each file: timestamp, back_x, back_y, back_z, thigh_x, thigh_y, thigh_z, label

        Returns:
            person_data: {person_id: (data, labels)} dictionary
                data: (num_samples, 6) array [back_xyz, thigh_xyz]
                labels: (num_samples,) array (0-indexed)
        """
        raw_path = self.raw_data_path / self.dataset_name / self.dataset_name

        if not raw_path.exists():
            raise FileNotFoundError(
                f"HARTH raw data not found at {raw_path}\n"
                "Expected structure: data/raw/harth/harth/S006.csv"
            )

        # Store data per subject
        result = {}

        # Search for available CSV files
        csv_files = sorted(raw_path.glob('S*.csv'))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {raw_path}")

        # person_id is managed as 1-indexed (starting from USER00001)
        for idx, subject_file in enumerate(csv_files):
            try:
                # Load CSV data
                df = pd.read_csv(subject_file)

                # Columns: timestamp, back_x, back_y, back_z, thigh_x, thigh_y, thigh_z, label
                if len(df.columns) != 8:
                    logger.warning(
                        f"Unexpected number of columns in {subject_file}: "
                        f"{len(df.columns)} (expected 8)"
                    )
                    continue

                # Extract sensor data (back_xyz + thigh_xyz)
                sensor_data = df.iloc[:, 1:7].values.astype(np.float32)
                # sensor_data: (num_samples, 6)

                # Extract and map labels
                original_labels = df.iloc[:, 7].values.astype(int)
                labels = np.array([self.label_mapping[l] for l in original_labels], dtype=int)

                # Convert person_id to 1-indexed (idx=0 -> person_id=1 -> USER00001)
                person_id = idx + 1
                result[person_id] = (sensor_data, labels)
                logger.info(
                    f"USER{person_id:05d} ({subject_file.name}): "
                    f"{sensor_data.shape}, Labels: {labels.shape}"
                )

            except Exception as e:
                logger.error(f"Error loading {subject_file}: {e}")
                continue

        if not result:
            raise ValueError("No data loaded. Please check the raw data directory structure.")

        logger.info(f"Total users loaded: {len(result)}")
        return result

    def clean_data(self, data: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Data cleaning and resampling

        Args:
            data: {person_id: (data, labels)} dictionary

        Returns:
            Cleaned and resampled {person_id: (data, labels)}
        """
        cleaned = {}
        for person_id, (person_data, labels) in data.items():
            # Remove invalid samples
            cleaned_data, cleaned_labels = filter_invalid_samples(person_data, labels)

            # Resampling (50Hz -> 30Hz)
            if self.original_sampling_rate != self.target_sampling_rate:
                resampled_data, resampled_labels = resample_timeseries(
                    cleaned_data,
                    cleaned_labels,
                    self.original_sampling_rate,
                    self.target_sampling_rate
                )
                cleaned[person_id] = (resampled_data, resampled_labels)
                logger.info(f"USER{person_id:05d} cleaned and resampled: {resampled_data.shape}")
            else:
                cleaned[person_id] = (cleaned_data, cleaned_labels)
                logger.info(f"USER{person_id:05d} cleaned: {cleaned_data.shape}")

        return cleaned

    def extract_features(self, data: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
        """
        Feature extraction (windowing per sensor × modality)

        Args:
            data: {person_id: (data, labels)} dictionary

        Returns:
            {person_id: {sensor/modality: {'X': data, 'Y': labels}}}
            Example: {'LowerBack/ACC': {'X': (N, 3, 150), 'Y': (N,)}}
        """
        processed = {}

        for person_id, (person_data, labels) in data.items():
            logger.info(f"Processing USER{person_id:05d}")

            processed[person_id] = {}

            # Process each sensor
            for sensor_name in self.sensor_names:
                sensor_start_ch, sensor_end_ch = self.sensor_channel_ranges[sensor_name]

                # Extract sensor channels
                sensor_data = person_data[:, sensor_start_ch:sensor_end_ch]

                # Apply sliding window (pad last window if needed)
                windowed_data, windowed_labels = create_sliding_windows(
                    sensor_data,
                    labels,
                    window_size=self.window_size,
                    stride=self.stride,
                    drop_last=False,
                    pad_last=True  # HARTH: pad if less than 150 samples
                )
                # windowed_data: (num_windows, window_size, sensor_channels)

                # Split into modalities (HARTH has ACC only)
                modalities = self.sensor_modalities[sensor_name]
                for modality_name, (mod_start_ch, mod_end_ch) in modalities.items():
                    # Extract modality channels
                    modality_data = windowed_data[:, :, mod_start_ch:mod_end_ch]
                    # modality_data: (num_windows, window_size, channels)

                    # Scaling not needed (already in G units)
                    if self.scale_factor is not None:
                        modality_data = modality_data / self.scale_factor
                        logger.info(f"  Applied scale_factor={self.scale_factor} to {sensor_name}/{modality_name}")

                    # Transpose shape: (num_windows, window_size, C) -> (num_windows, C, window_size)
                    modality_data = np.transpose(modality_data, (0, 2, 1))

                    # Convert to float16
                    modality_data = modality_data.astype(np.float16)

                    # Sensor/modality hierarchical structure
                    sensor_modality_key = f"{sensor_name}/{modality_name}"

                    processed[person_id][sensor_modality_key] = {
                        'X': modality_data,
                        'Y': windowed_labels
                    }

                    logger.info(
                        f"  {sensor_modality_key}: X.shape={modality_data.shape}, "
                        f"Y.shape={windowed_labels.shape}"
                    )

        return processed

    def save_processed_data(self, data: Dict[int, Dict[str, Dict[str, np.ndarray]]]) -> None:
        """
        Save processed data

        Args:
            data: {person_id: {sensor_modality: {'X': data, 'Y': labels}}}

        Save format:
            data/processed/harth/USER00001/LowerBack/ACC/X.npy, Y.npy
            data/processed/harth/USER00001/RightThigh/ACC/X.npy, Y.npy
            ...
        """
        import json

        base_path = self.processed_data_path / self.dataset_name
        base_path.mkdir(parents=True, exist_ok=True)

        total_stats = {
            'dataset': self.dataset_name,
            'num_activities': self.num_activities,
            'num_sensors': self.num_sensors,
            'sensor_names': self.sensor_names,
            'original_sampling_rate': self.original_sampling_rate,
            'target_sampling_rate': self.target_sampling_rate,
            'window_size': self.window_size,
            'stride': self.stride,
            'normalization': 'none',  # No normalization (raw data preserved)
            'scale_factor': self.scale_factor,  # Scaling factor (None as already in G units)
            'data_dtype': 'float16',  # Data type
            'users': {}
        }

        for person_id, sensor_modality_data in data.items():
            user_name = f"USER{person_id:05d}"
            user_path = base_path / user_name
            user_path.mkdir(parents=True, exist_ok=True)

            user_stats = {'sensor_modalities': {}}

            for sensor_modality_name, arrays in sensor_modality_data.items():
                sensor_modality_path = user_path / sensor_modality_name
                sensor_modality_path.mkdir(parents=True, exist_ok=True)

                # Save X.npy, Y.npy
                X = arrays['X']  # (num_windows, C, window_size)
                Y = arrays['Y']  # (num_windows,)

                np.save(sensor_modality_path / 'X.npy', X)
                np.save(sensor_modality_path / 'Y.npy', Y)

                # Statistics
                user_stats['sensor_modalities'][sensor_modality_name] = {
                    'X_shape': X.shape,
                    'Y_shape': Y.shape,
                    'num_windows': len(Y),
                    'class_distribution': get_class_distribution(Y)
                }

                logger.info(
                    f"Saved {user_name}/{sensor_modality_name}: "
                    f"X{X.shape}, Y{Y.shape}"
                )

            total_stats['users'][user_name] = user_stats

        # Save overall metadata
        metadata_path = base_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            # Convert NumPy types to JSON compatible
            def convert_to_serializable(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, tuple):
                    return list(obj)
                return obj

            def recursive_convert(d):
                if isinstance(d, dict):
                    return {k: recursive_convert(v) for k, v in d.items()}
                elif isinstance(d, list):
                    return [recursive_convert(v) for v in d]
                else:
                    return convert_to_serializable(d)

            serializable_stats = recursive_convert(total_stats)
            json.dump(serializable_stats, f, indent=2)

        logger.info(f"Saved metadata to {metadata_path}")
        logger.info(f"Preprocessing completed: {base_path}")
