"""
LARa (Logistic Activity Recognition) Dataset Preprocessing

LARa Dataset:
- 8 types of physical activities (logistics warehouse work)
- 8 subjects (S07-S14)
- 5 sensors (left arm, left leg, neck, right arm, right leg)
- Sampling rate: 100Hz
- Accelerometer (3-axis, G units) + Gyroscope (3-axis)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
import logging
import json

from .base import BasePreprocessor
from .utils import (
    create_sliding_windows,
    create_sliding_windows_multi_session,
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

# LARA dataset URL (manual download required)
LARA_URL = "https://zenodo.org/api/records/3862782/files/IMU%20data.zip/content"


@register_preprocessor('lara')
class LaraPreprocessor(BasePreprocessor):
    """
    Preprocessing class for LARA dataset
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # LARA specific settings
        self.num_activities = 8
        self.num_subjects = 8  # S07-S14
        self.num_sensors = 5
        self.num_channels = 30  # 5 sensors × (3-axis ACC + 3-axis GYRO)

        # Sensor names and channel mapping
        # Channel configuration:
        # LeftArm: ACC(3) + GYRO(3) = 6
        # LeftLeg: ACC(3) + GYRO(3) = 6
        # Neck: ACC(3) + GYRO(3) = 6
        # RightArm: ACC(3) + GYRO(3) = 6
        # RightLeg: ACC(3) + GYRO(3) = 6
        self.sensor_names = ['LeftArm', 'LeftLeg', 'Neck', 'RightArm', 'RightLeg']
        self.sensor_channel_ranges = {
            'LeftArm': (0, 6),      # channels 0-5
            'LeftLeg': (6, 12),     # channels 6-11
            'Neck': (12, 18),       # channels 12-17
            'RightArm': (18, 24),   # channels 18-23
            'RightLeg': (24, 30)    # channels 24-29
        }

        # Modality (channel division within each sensor)
        self.sensor_modalities = {
            'LeftArm': {
                'ACC': (0, 3),   # 3-axis accelerometer
                'GYRO': (3, 6)   # 3-axis gyroscope
            },
            'LeftLeg': {
                'ACC': (0, 3),
                'GYRO': (3, 6)
            },
            'Neck': {
                'ACC': (0, 3),
                'GYRO': (3, 6)
            },
            'RightArm': {
                'ACC': (0, 3),
                'GYRO': (3, 6)
            },
            'RightLeg': {
                'ACC': (0, 3),
                'GYRO': (3, 6)
            }
        }

        # Sampling rate
        self.original_sampling_rate = 100  # Hz (LARA original)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz (target)

        # Preprocessing parameters
        self.window_size = config.get('window_size', 150)  # 5 seconds @ 30Hz
        self.stride = config.get('stride', 30)  # 1 second @ 30Hz

        # Scaling factor (not needed as already in G units)
        self.scale_factor = DATASETS.get('LARA', {}).get('scale_factor', None)

    def get_dataset_name(self) -> str:
        return 'lara'

    def download_dataset(self) -> None:
        """
        Download LARA dataset
        """
        logger.info("=" * 80)
        logger.info("Downloading LARA dataset")
        logger.info("=" * 80)

        dataset_path = self.raw_data_path / self.dataset_name
        imu_dir = dataset_path / "IMU data"
        data_exists = imu_dir.exists() and any(imu_dir.rglob('*.csv'))

        if data_exists:
            logger.info(f"LARA data already exists at {dataset_path}")
            return

        dataset_path.mkdir(parents=True, exist_ok=True)
        zip_path = dataset_path / "lara_imu.zip"

        # Download
        logger.info("Downloading LARA archive...")
        download_file(LARA_URL, zip_path, desc='Downloading LARA')

        # Extract
        logger.info("Extracting LARA archive...")
        extract_archive(zip_path, dataset_path, desc='Extracting LARA')

        # Cleanup
        if zip_path.exists():
            zip_path.unlink()

        logger.info(f"LARA dataset downloaded to {dataset_path}")

    def load_raw_data(self) -> Dict[int, list]:
        """
        Load LARA raw data

        Returns:
            Dictionary of {person_id: [(sensor_data, labels), ...]}
            Each element is one session (file)
            - sensor_data: Array of shape (T, 30) (T=time series length, 30 channels)
            - labels: Array of shape (T,)
        """
        logger.info(f"Loading LARA data from {self.raw_data_path}")

        # IMU data directory
        imu_dir = self.raw_data_path / self.dataset_name / "IMU data"
        if not imu_dir.exists():
            raise FileNotFoundError(f"IMU data directory not found: {imu_dir}")

        # Subject directories (S07-S14)
        subject_dirs = sorted([d for d in imu_dir.iterdir() if d.is_dir() and d.name.startswith('S')])
        logger.info(f"Found {len(subject_dirs)} subjects: {[d.name for d in subject_dirs]}")

        result = {}
        person_id = 1  # 1-indexed

        for subject_dir in subject_dirs:
            subject_id = subject_dir.name  # e.g., 'S07'
            logger.info(f"Processing subject {subject_id}...")

            # Get all session files for each subject
            data_files = sorted([f for f in subject_dir.glob('*.csv') if not f.name.endswith('_labels.csv')])
            logger.info(f"  Found {len(data_files)} sessions for {subject_id}")

            # Keep data per session (do not merge)
            sessions = []

            for data_file in data_files:
                # Corresponding label file
                label_file = data_file.parent / f"{data_file.stem}_labels.csv"
                if not label_file.exists():
                    logger.warning(f"  Label file not found for {data_file.name}, skipping...")
                    continue

                # Load data
                df_data = pd.read_csv(data_file)
                df_labels = pd.read_csv(label_file)

                # Extract sensor data excluding Time column
                # Unify sensor data order
                # LA (LeftArm), LL (LeftLeg), N (Neck), RA (RightArm), RL (RightLeg)
                # Each sensor: AccelerometerX,Y,Z, GyroscopeX,Y,Z
                ordered_columns = []
                for sensor_prefix in ['LA', 'LL', 'N', 'RA', 'RL']:
                    for measurement in ['Accelerometer', 'Gyroscope']:
                        for axis in ['X', 'Y', 'Z']:
                            col_name = f"{sensor_prefix}_{measurement}{axis}"
                            ordered_columns.append(col_name)

                sensor_data = df_data[ordered_columns].values  # (T, 30)

                # Extract labels (Class column)
                labels = df_labels['Class'].values  # (T,)

                # Add as session
                sessions.append((sensor_data, labels))

            if not sessions:
                logger.warning(f"  No valid data for {subject_id}, skipping...")
                continue

            result[person_id] = sessions
            total_samples = sum(len(s[0]) for s in sessions)
            logger.info(f"  Subject {subject_id}: {len(sessions)} sessions, {total_samples} samples")
            person_id += 1

        logger.info(f"Loaded data for {len(result)} subjects")
        return result

    def clean_data(self, data: Dict[int, list]) -> Dict[int, list]:
        """
        Data cleaning and resampling (processed per session)

        Args:
            data: Dictionary of {person_id: [(data, labels), ...]}

        Returns:
            Cleaned and resampled {person_id: [(data, labels), ...]}
        """
        cleaned = {}
        for person_id, sessions in data.items():
            cleaned_sessions = []

            for session_data, session_labels in sessions:
                # Resampling (100Hz -> 30Hz)
                if self.original_sampling_rate != self.target_sampling_rate:
                    resampled_data, resampled_labels = resample_timeseries(
                        session_data,
                        session_labels,
                        self.original_sampling_rate,
                        self.target_sampling_rate
                    )
                    cleaned_sessions.append((resampled_data, resampled_labels))
                else:
                    cleaned_sessions.append((session_data, session_labels))

            if cleaned_sessions:
                cleaned[person_id] = cleaned_sessions
                total_samples = sum(len(s[0]) for s in cleaned_sessions)
                logger.info(f"USER{person_id:05d} resampled: {len(cleaned_sessions)} sessions, {total_samples} samples")

        return cleaned

    def extract_features(self, data: Dict[int, list]) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
        """
        Feature extraction and window segmentation
        Windowing per session, ensuring no cross-session windows

        Args:
            data: Dictionary of {person_id: [(data, labels), ...]}

        Returns:
            {person_id: {sensor_modality: {'X': data, 'Y': labels}}}
        """
        result = {}

        for person_id, sessions in data.items():
            logger.info(f"Extracting features for USER{person_id:05d} ({len(sessions)} sessions)")

            # Sliding window per session
            X, Y = create_sliding_windows_multi_session(
                sessions,
                window_size=self.window_size,
                stride=self.stride
            )

            if len(X) == 0:
                logger.warning(f"  No windows created")
                continue

            # X shape after create_sliding_windows: (num_windows, window_size, 30)
            # Transpose to (num_windows, 30, window_size)
            X = X.transpose(0, 2, 1)  # (num_windows, 30_channels, window_size)
            # Y shape: (num_windows,)

            logger.info(f"  Generated {len(X)} windows")

            # Split by sensor and modality
            person_dict = {}

            for sensor_name in self.sensor_names:
                # Get channel range
                start_ch, end_ch = self.sensor_channel_ranges[sensor_name]

                # Extract sensor data
                X_sensor = X[:, start_ch:end_ch, :]  # (num_windows, 6, window_size)

                # Split by modality
                for modality_name, (mod_start, mod_end) in self.sensor_modalities[sensor_name].items():
                    X_modality = X_sensor[:, mod_start:mod_end, :]  # (num_windows, 3, window_size)

                    sensor_modality_key = f"{sensor_name}/{modality_name}"
                    person_dict[sensor_modality_key] = {
                        'X': X_modality.astype(np.float16),
                        'Y': Y.astype(np.int32)
                    }

                    logger.info(f"  {sensor_modality_key}: X={X_modality.shape}, Y={Y.shape}")

            result[person_id] = person_dict

        return result

    def save_processed_data(self, data: Dict[int, Dict[str, Dict[str, np.ndarray]]]) -> None:
        """
        Save processed data

        Args:
            data: {person_id: {sensor_modality: {'X': data, 'Y': labels}}}

        Save format:
            data/processed/lara/USER00001/LeftArm/ACC/X.npy, Y.npy
            data/processed/lara/USER00001/LeftArm/GYRO/X.npy, Y.npy
            ...
        """
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
            'normalization': 'none',  # No normalization (preserve raw data)
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

                # Record statistics
                user_stats['sensor_modalities'][sensor_modality_name] = {
                    'num_windows': len(X),
                    'shape': list(X.shape),
                    'class_distribution': get_class_distribution(Y)
                }

                logger.info(f"  Saved {user_name}/{sensor_modality_name}: X={X.shape}, Y={Y.shape}")

            total_stats['users'][user_name] = user_stats

        # Save metadata (same format as DSADS)
        metadata_path = base_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            # Convert NumPy types to JSON-compatible types
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

    def process(self) -> None:
        """
        Execute LARA dataset preprocessing
        """
        logger.info("="*80)
        logger.info("Starting LARA dataset preprocessing")
        logger.info("="*80)

        # Load raw data
        raw_data = self.load_raw_data()

        # Data cleaning and resampling
        cleaned_data = self.clean_data(raw_data)

        # Feature extraction and window segmentation
        features = self.extract_features(cleaned_data)

        # Save
        self.save_processed_data(features)

        logger.info("="*80)
        logger.info("LARA preprocessing completed")
        logger.info("="*80)
