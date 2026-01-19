"""
VTT-ConIot (Construction Worker HAR) Preprocessing

VTT-ConIot Dataset:
- 16 types of construction work activities
- 13 subjects
- 3 IMU sensors (hip, back, upper arm)
- Each sensor: 3-axis accelerometer, 3-axis gyroscope, 3-axis magnetometer
- Sampling rate: 100Hz
- Data units: acceleration in half-G (1/2 G), gyroscope in deg/s, magnetometer in μT
- License: CC BY 4.0
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
import logging

from .base import BasePreprocessor
from .utils import (
    create_sliding_windows,
    create_sliding_windows_multi_session,
    filter_invalid_samples,
    get_class_distribution,
    resample_timeseries
)
from .common import check_dataset_exists, download_file, extract_archive
from . import register_preprocessor
from ..dataset_info import DATASETS

logger = logging.getLogger(__name__)

VTT_CONIOT_URL = "https://zenodo.org/records/4683703/files/VTT_ConIot_Dataset.zip?download=1"


@register_preprocessor('vtt_coniot')
class VTTConIotPreprocessor(BasePreprocessor):
    """
    Preprocessing class for VTT-ConIot dataset
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # VTT-ConIot specific settings
        self.num_activities = 16
        self.num_subjects = 13
        self.num_sensors = 3
        self.channels_per_sensor = 9  # 3-axis acc, gyro, mag

        # Sampling rates
        self.original_sampling_rate = 100  # Hz
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz

        # Sensor names and CSV column mapping
        # CSV column names: trousers=Hip, back=Back, hand=UpperArm
        self.sensor_names = ['Hip', 'Back', 'UpperArm']
        self.csv_sensor_prefixes = {
            'Hip': 'trousers',
            'Back': 'back',
            'UpperArm': 'hand'
        }

        # Modalities (channels for each sensor)
        self.modalities = ['ACC', 'GYRO', 'MAG']
        self.csv_modality_suffixes = {
            'ACC': ['Ax_g', 'Ay_g', 'Az_g'],
            'GYRO': ['Gx_dps', 'Gy_dps', 'Gz_dps'],
            'MAG': ['Mx_uT', 'My_uT', 'Mz_uT']
        }
        self.channels_per_modality = 3

        # Preprocessing parameters
        self.window_size = config.get('window_size', 150)  # 5 seconds @ 30Hz
        self.stride = config.get('stride', 30)  # 1 second @ 30Hz

        # Scaling factor (convert half-G -> G)
        self.scale_factor = DATASETS.get('VTT_CONIOT', {}).get('scale_factor', 0.5)

    def get_dataset_name(self) -> str:
        return 'vtt_coniot'

    def download_dataset(self) -> None:
        """
        Download VTT-ConIot dataset from Zenodo
        https://zenodo.org/records/4683703
        """
        logger.info("=" * 80)
        logger.info("Downloading VTT-ConIot dataset")
        logger.info("=" * 80)

        vtt_raw_path = self.raw_data_path / self.dataset_name

        if check_dataset_exists(vtt_raw_path, required_files=['VTT_ConIot_Dataset/VTT-ConIot-IMU-Data/*.csv']):
            logger.info(f"VTT-ConIot data already exists at {vtt_raw_path}")
            return

        vtt_raw_path.mkdir(parents=True, exist_ok=True)
        zip_path = vtt_raw_path / "VTT_ConIot_Dataset.zip"

        # Download
        logger.info("Step 1/2: Downloading archive (363 MB)")
        download_file(VTT_CONIOT_URL, zip_path, desc='Downloading VTT-ConIot')

        # Extract
        logger.info("Step 2/2: Extracting archive")
        extract_archive(zip_path, vtt_raw_path, desc='Extracting VTT-ConIot')

        # Cleanup
        if zip_path.exists():
            zip_path.unlink()

        logger.info(f"VTT-ConIot dataset downloaded to {vtt_raw_path}")

    def load_raw_data(self) -> Dict[int, list]:
        """
        Load VTT-ConIot raw data by subject

        Expected format:
        - data/raw/vtt_coniot/VTT_ConIot_Dataset/VTT-ConIot-IMU-Data/
          activity_{1-16}_user_{1-13}_combined.csv

        Returns:
            user_data: dictionary of {user_id: [(data, labels), ...]}
                Each CSV file (activity) is kept as a separate session
                data: array of (num_samples, 27) (3 sensors × 9 channels)
                labels: array of (num_samples,)
        """
        raw_path = self.raw_data_path / self.dataset_name / 'VTT_ConIot_Dataset' / 'VTT-ConIot-IMU-Data'

        if not raw_path.exists():
            raise FileNotFoundError(
                f"VTT-ConIot raw data not found at {raw_path}\n"
                "Expected structure: data/raw/vtt_coniot/VTT_ConIot_Dataset/VTT-ConIot-IMU-Data/*.csv"
            )

        # Store session list for each subject
        from collections import defaultdict
        user_sessions = defaultdict(list)

        # For each activity and user
        for activity_id in range(1, self.num_activities + 1):
            for user_id in range(1, self.num_subjects + 1):
                csv_file = raw_path / f"activity_{activity_id}_user_{user_id}_combined.csv"

                if not csv_file.exists():
                    logger.warning(f"File not found: {csv_file}")
                    continue

                try:
                    # Load CSV
                    df = pd.read_csv(csv_file)

                    # Extract required columns and arrange by sensor order
                    sensor_data_list = []

                    for sensor_name in self.sensor_names:
                        prefix = self.csv_sensor_prefixes[sensor_name]

                        for modality_name in self.modalities:
                            suffixes = self.csv_modality_suffixes[modality_name]
                            for suffix in suffixes:
                                col_name = f"{prefix}_{suffix}"
                                if col_name in df.columns:
                                    sensor_data_list.append(df[col_name].values)
                                else:
                                    logger.warning(f"Column {col_name} not found in {csv_file}")
                                    sensor_data_list.append(np.zeros(len(df)))

                    # Convert to array of (samples, 27)
                    segment_data = np.column_stack(sensor_data_list)

                    # Generate labels (0-indexed)
                    segment_labels = np.full(len(segment_data), activity_id - 1)

                    # Add as session (do not concatenate)
                    if len(segment_data) > 0:
                        user_sessions[user_id].append((segment_data, segment_labels))

                except Exception as e:
                    logger.error(f"Error loading {csv_file}: {e}")
                    continue

        # Return session list
        result = {}
        for user_id in range(1, self.num_subjects + 1):
            if user_sessions[user_id]:
                result[user_id] = user_sessions[user_id]
                total_samples = sum(d.shape[0] for d, l in user_sessions[user_id])
                logger.info(f"USER{user_id:05d}: {len(user_sessions[user_id])} sessions, {total_samples} samples")
            else:
                logger.warning(f"No data loaded for USER{user_id:05d}")

        if not result:
            raise ValueError("No data loaded. Please check the raw data directory structure.")

        logger.info(f"Total users loaded: {len(result)}")
        return result

    def clean_data(self, data: Dict[int, list]) -> Dict[int, list]:
        """
        Clean and resample data (process by session)

        Args:
            data: dictionary of {user_id: [(data, labels), ...]}

        Returns:
            Cleaned and resampled {user_id: [(data, labels), ...]}
        """
        cleaned = {}
        for user_id, sessions in data.items():
            cleaned_sessions = []
            for session_data, session_labels in sessions:
                # Remove invalid samples
                cleaned_data, cleaned_labels = filter_invalid_samples(session_data, session_labels)

                if len(cleaned_data) == 0:
                    continue

                # Resampling (100Hz -> 30Hz)
                if self.original_sampling_rate != self.target_sampling_rate:
                    resampled_data, resampled_labels = resample_timeseries(
                        cleaned_data,
                        cleaned_labels,
                        self.original_sampling_rate,
                        self.target_sampling_rate
                    )
                    cleaned_sessions.append((resampled_data, resampled_labels))
                else:
                    cleaned_sessions.append((cleaned_data, cleaned_labels))

            if cleaned_sessions:
                cleaned[user_id] = cleaned_sessions
                total_samples = sum(d.shape[0] for d, l in cleaned_sessions)
                logger.info(f"USER{user_id:05d} cleaned: {len(cleaned_sessions)} sessions, {total_samples} samples")

        return cleaned

    def extract_features(self, data: Dict[int, list]) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
        """
        Feature extraction (windowing and normalization by sensor × modality)
        Windows do not cross session boundaries

        Args:
            data: dictionary of {user_id: [(data, labels), ...]}

        Returns:
            {user_id: {sensor_modality: {'X': data, 'Y': labels}}}
            Example: {'Hip/ACC': {'X': (N, 3, 150), 'Y': (N,)}}
        """
        processed = {}

        # Calculate channel ranges for sensor/modality
        # Data array: [Hip_ACC(3), Hip_GYRO(3), Hip_MAG(3), Back_ACC(3), ..., UpperArm_MAG(3)]
        sensor_modality_ranges = {}
        ch_idx = 0
        for sensor_name in self.sensor_names:
            for modality_name in self.modalities:
                key = f"{sensor_name}/{modality_name}"
                sensor_modality_ranges[key] = (ch_idx, ch_idx + self.channels_per_modality)
                ch_idx += self.channels_per_modality

        for user_id, sessions in data.items():
            logger.info(f"Processing USER{user_id:05d} ({len(sessions)} sessions)")

            processed[user_id] = {}

            # Process for each sensor and modality
            for sensor_modality_key, (start_ch, end_ch) in sensor_modality_ranges.items():
                sensor_name = sensor_modality_key.split('/')[0]
                modality_name = sensor_modality_key.split('/')[1]

                # Extract channels for each session
                modality_sessions = [(d[:, start_ch:end_ch], l) for d, l in sessions]

                # Apply sliding windows per session
                windowed_data, windowed_labels = create_sliding_windows_multi_session(
                    modality_sessions,
                    window_size=self.window_size,
                    stride=self.stride,
                    drop_last=False,
                    pad_last=True
                )

                if len(windowed_data) == 0:
                    logger.warning(f"  {sensor_modality_key}: no valid windows")
                    continue

                # windowed_data: (num_windows, window_size, 3)

                # Apply scaling (acceleration only: half-G -> G)
                if modality_name == 'ACC' and self.scale_factor is not None:
                    windowed_data = windowed_data * self.scale_factor
                    logger.info(f"  Applied scale_factor={self.scale_factor} to {sensor_modality_key}")

                # Transform shape: (num_windows, window_size, 3) -> (num_windows, 3, window_size)
                windowed_data = np.transpose(windowed_data, (0, 2, 1))

                # Convert to float16
                windowed_data = windowed_data.astype(np.float16)

                processed[user_id][sensor_modality_key] = {
                    'X': windowed_data,
                    'Y': windowed_labels
                }

                logger.info(
                    f"  {sensor_modality_key}: X.shape={windowed_data.shape}, "
                    f"Y.shape={windowed_labels.shape}"
                )

        return processed

    def save_processed_data(self, data: Dict[int, Dict[str, Dict[str, np.ndarray]]]) -> None:
        """
        Save processed data

        Args:
            data: {user_id: {sensor_modality: {'X': data, 'Y': labels}}}

        Save format:
            data/processed/vtt_coniot/USER00001/Hip/ACC/X.npy, Y.npy
            data/processed/vtt_coniot/USER00001/Hip/GYRO/X.npy, Y.npy
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
            'modalities': self.modalities,
            'channels_per_modality': self.channels_per_modality,
            'original_sampling_rate': self.original_sampling_rate,
            'target_sampling_rate': self.target_sampling_rate,
            'window_size': self.window_size,
            'stride': self.stride,
            'normalization': 'none',
            'scale_factor': self.scale_factor,
            'data_dtype': 'float16',
            'data_shape': f'(num_windows, {self.channels_per_modality}, {self.window_size})',
            'users': {}
        }

        for user_id, sensor_modality_data in data.items():
            user_name = f"USER{user_id:05d}"
            user_path = base_path / user_name
            user_path.mkdir(parents=True, exist_ok=True)

            user_stats = {'sensor_modalities': {}}

            for sensor_modality_name, arrays in sensor_modality_data.items():
                sensor_modality_path = user_path / sensor_modality_name
                sensor_modality_path.mkdir(parents=True, exist_ok=True)

                # Save X.npy, Y.npy
                X = arrays['X']  # (num_windows, 3, window_size)
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
            # Convert NumPy types to JSON-compatible format
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
