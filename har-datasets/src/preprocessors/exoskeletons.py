"""
Exoskeletons (IMU-based Human Activity Recognition for Low-Back Exoskeletons) Preprocessing

Exoskeletons Dataset:
- 12 subjects (6M+6F)
- 5 IMU sensors (Chest, RightLeg, LeftLeg, RightWrist, LeftWrist)
- 2 tasks: intention (4 classes), payload (4 classes: 0/5/10/15kg)
- Sampling rate: estimated 50Hz
- Source: https://zenodo.org/records/7182799
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging

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

logger = logging.getLogger(__name__)

# Exoskeletons Dataset URL
EXOSKELETONS_URL = "https://zenodo.org/api/records/7182799/files-archive"


@register_preprocessor('exoskeletons')
class ExoskeletonsPreprocessor(BasePreprocessor):
    """
    Preprocessor class for Exoskeletons dataset

    This dataset uses only the intention task (appropriate for HAR).
    The payload task is for load estimation and not suitable for HAR purposes.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Exoskeletons specific settings
        self.num_activities = 4  # intention: 4 classes
        self.num_subjects = 12
        self.num_sensors = 5
        self.channels_per_sensor = 6  # 3-axis acc + 3-axis gyro
        self.num_channels = 30  # 5 sensors × 6 channels

        # Sampling rate (estimated)
        self.original_sampling_rate = 50  # Hz (estimated)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz

        # Sensor ID to name mapping
        self.sensor_id_to_name = {
            'B6': 'Chest',
            '3B': 'RightLeg',
            'C9': 'LeftLeg',
            'BB': 'RightWrist',
            'B5': 'LeftWrist'
        }

        # Sensor names (for body parts)
        self.sensor_names = ['Chest', 'RightLeg', 'LeftLeg', 'RightWrist', 'LeftWrist']

        # Sensor channel ranges
        self.sensor_channel_ranges = {
            'Chest': (0, 6),
            'RightLeg': (6, 12),
            'LeftLeg': (12, 18),
            'RightWrist': (18, 24),
            'LeftWrist': (24, 30)
        }

        # Modalities
        self.modalities = ['ACC', 'GYRO']
        self.modality_channel_ranges = {
            'ACC': (0, 3),
            'GYRO': (3, 6)
        }
        self.channels_per_modality = 3

        # Preprocessing parameters
        self.window_size = config.get('window_size', 150)  # 5 seconds @ 30Hz
        self.stride = config.get('stride', 30)  # 1 second @ 30Hz

        # Data directory name (within raw data folder)
        self.raw_dir_name = config.get('raw_dir_name', 'imupay')

        # Scaling factor (data is already in m/s^2, convert to G)
        self.scale_factor = 1.0 / 9.80665  # m/s^2 -> G

    def get_dataset_name(self) -> str:
        return 'exoskeletons'

    def download_dataset(self) -> None:
        """
        Download Exoskeletons dataset
        """
        logger.info("=" * 80)
        logger.info("Downloading Exoskeletons dataset")
        logger.info("=" * 80)

        raw_path = self.raw_data_path / self.raw_dir_name

        # Check if data already exists
        if check_dataset_exists(raw_path, required_files=['U001_train_intention.csv']):
            logger.warning(f"Exoskeletons data already exists at {raw_path}")
            return

        try:
            # Download
            logger.info("Downloading from Zenodo...")
            raw_path.mkdir(parents=True, exist_ok=True)
            zip_path = raw_path / 'exoskeletons.zip'
            download_file(EXOSKELETONS_URL, zip_path, desc='Downloading Exoskeletons')

            # Extract
            logger.info("Extracting data...")
            extract_archive(zip_path, raw_path, desc='Extracting Exoskeletons')

            # Cleanup
            if zip_path.exists():
                zip_path.unlink()

            logger.info("=" * 80)
            logger.info(f"SUCCESS: Exoskeletons dataset downloaded to {raw_path}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Failed to download Exoskeletons dataset: {e}", exc_info=True)
            raise

    def load_raw_data(self) -> Dict[int, list]:
        """
        Load Exoskeletons raw data per subject (intention task only)

        Returns:
            person_data: {person_id: [(data, labels), ...]} dictionary
                Each element is one session (train/val/test)
                data: (num_samples, 30) array
                labels: (num_samples,) array
        """
        raw_path = self.raw_data_path / self.raw_dir_name

        if not raw_path.exists():
            raise FileNotFoundError(
                f"Exoskeletons data not found at {raw_path}\n"
                "Please download the dataset first using download_dataset()"
            )

        logger.info(f"Loading Exoskeletons data from {raw_path}")

        # Original column order: C9(LeftLeg), 3B(RightLeg), B5(LeftWrist), B6(Chest), BB(RightWrist)
        # Reorder to unified order: Chest, RightLeg, LeftLeg, RightWrist, LeftWrist
        column_order = [
            # Chest (B6)
            'Axel_X_B6', 'Axel_Y_B6', 'Axel_Z_B6', 'Gyro_X_B6', 'Gyro_Y_B6', 'Gyro_Z_B6',
            # RightLeg (3B)
            'Axel_X_3B', 'Axel_Y_3B', 'Axel_Z_3B', 'Gyro_X_3B', 'Gyro_Y_3B', 'Gyro_Z_3B',
            # LeftLeg (C9)
            'Axel_X_C9', 'Axel_Y_C9', 'Axel_Z_C9', 'Gyro_X_C9', 'Gyro_Y_C9', 'Gyro_Z_C9',
            # RightWrist (BB)
            'Axel_X_BB', 'Axel_Y_BB', 'Axel_Z_BB', 'Gyro_X_BB', 'Gyro_Y_BB', 'Gyro_Z_BB',
            # LeftWrist (B5)
            'Axel_X_B5', 'Axel_Y_B5', 'Axel_Z_B5', 'Gyro_X_B5', 'Gyro_Y_B5', 'Gyro_Z_B5',
        ]

        # Store session list per subject
        person_sessions = {person_id: []
                          for person_id in range(1, self.num_subjects + 1)}

        for subject_id in range(1, 13):  # U001-U012
            subject_str = f'U{subject_id:03d}'

            for split in ['train', 'val', 'test']:
                file_path = raw_path / f'{subject_str}_{split}_intention.csv'

                if not file_path.exists():
                    logger.warning(f"File not found: {file_path}")
                    continue

                try:
                    df = pd.read_csv(file_path)

                    # Reorder columns
                    data = df[column_order].values.astype(np.float32)
                    labels = df['target'].values.astype(np.int64)

                    # Store each session separately
                    person_sessions[subject_id].append((data, labels))

                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
                    continue

        # Build result
        result = {}
        for person_id in range(1, self.num_subjects + 1):
            if person_sessions[person_id]:
                num_sessions = len(person_sessions[person_id])
                total_samples = sum(len(d) for d, l in person_sessions[person_id])
                result[person_id] = person_sessions[person_id]
                logger.info(f"USER{person_id:05d}: {num_sessions} sessions, {total_samples} total samples")
            else:
                logger.warning(f"No data loaded for USER{person_id:05d}")

        if not result:
            raise ValueError("No data loaded. Please check the raw data directory structure.")

        logger.info(f"Total users loaded: {len(result)}")
        return result

    def clean_data(self, data: Dict[int, list]) -> Dict[int, list]:
        """
        Data cleaning and resampling (processed per session)

        Args:
            data: {person_id: [(data, labels), ...]} dictionary

        Returns:
            Cleaned and resampled {person_id: [(data, labels), ...]}
        """
        cleaned = {}
        for person_id, sessions in data.items():
            cleaned_sessions = []
            for session_data, session_labels in sessions:
                # Remove invalid samples
                clean_data, clean_labels = filter_invalid_samples(session_data, session_labels)

                if len(clean_data) == 0:
                    continue

                # Resampling (50Hz -> 30Hz)
                if self.original_sampling_rate != self.target_sampling_rate:
                    resampled_data, resampled_labels = resample_timeseries(
                        clean_data,
                        clean_labels,
                        self.original_sampling_rate,
                        self.target_sampling_rate
                    )
                    cleaned_sessions.append((resampled_data, resampled_labels))
                else:
                    cleaned_sessions.append((clean_data, clean_labels))

            if cleaned_sessions:
                cleaned[person_id] = cleaned_sessions
                total_samples = sum(len(d) for d, l in cleaned_sessions)
                logger.info(f"USER{person_id:05d} cleaned: {len(cleaned_sessions)} sessions, {total_samples} samples")

        return cleaned

    def extract_features(self, data: Dict[int, list]) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
        """
        Feature extraction (windowing and normalization per sensor x modality)

        Windows are created per session; no windows span across sessions.

        Args:
            data: {person_id: [(data, labels), ...]} dictionary

        Returns:
            {person_id: {sensor_modality: {'X': data, 'Y': labels}}}
            Example: {'Chest/ACC': {'X': (N, 3, 150), 'Y': (N,)}}
        """
        processed = {}

        for person_id, sessions in data.items():
            logger.info(f"Processing USER{person_id:05d} ({len(sessions)} sessions)")

            processed[person_id] = {}

            # Process each sensor
            for sensor_name in self.sensor_names:
                sensor_start_ch, sensor_end_ch = self.sensor_channel_ranges[sensor_name]

                # Extract sensor data per session
                sensor_sessions = []
                for session_data, session_labels in sessions:
                    sensor_data = session_data[:, sensor_start_ch:sensor_end_ch]  # (samples, 6)
                    sensor_sessions.append((sensor_data, session_labels))

                # Apply sliding window per session
                # No windows are generated across session boundaries
                windowed_data, windowed_labels = create_sliding_windows_multi_session(
                    sensor_sessions,
                    window_size=self.window_size,
                    stride=self.stride,
                    drop_last=False,
                    pad_last=False  # Skip if session is too short
                )

                if len(windowed_data) == 0:
                    logger.warning(f"  No windows for {sensor_name}")
                    continue

                # windowed_data: (num_windows, window_size, 6)

                # Split into modalities
                for modality_name in self.modalities:
                    mod_start_ch, mod_end_ch = self.modality_channel_ranges[modality_name]

                    # Extract modality channels
                    modality_data = windowed_data[:, :, mod_start_ch:mod_end_ch]
                    # modality_data: (num_windows, window_size, 3)

                    # Apply scaling (accelerometer only)
                    if modality_name == 'ACC' and self.scale_factor is not None:
                        modality_data = modality_data * self.scale_factor
                        logger.info(f"  Applied scale_factor={self.scale_factor} to {sensor_name}/{modality_name}")

                    # Transpose shape: (num_windows, window_size, 3) -> (num_windows, 3, window_size)
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
            data/processed/exoskeletons/USER00001/Chest/ACC/X.npy, Y.npy
            data/processed/exoskeletons/USER00001/Chest/GYRO/X.npy, Y.npy
            data/processed/exoskeletons/USER00001/RightLeg/ACC/X.npy, Y.npy
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
            'normalization': 'none',  # No normalization (raw data preserved)
            'scale_factor': self.scale_factor,  # Scaling factor (applied to ACC only)
            'data_dtype': 'float16',  # Data type
            'data_shape': f'(num_windows, {self.channels_per_modality}, {self.window_size})',
            'activity_names': {
                '0': 'idle',
                '1': 'walking',
                '2': 'lifting',
                '3': 'lowering'
            },
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

    def preprocess(self) -> None:
        """
        Execute preprocessing pipeline
        """
        logger.info("=" * 80)
        logger.info(f"Starting Exoskeletons preprocessing")
        logger.info("=" * 80)

        # 1. Load data
        logger.info("Step 1/4: Loading raw data")
        raw_data = self.load_raw_data()

        # 2. Cleaning
        logger.info("Step 2/4: Cleaning data")
        cleaned_data = self.clean_data(raw_data)

        # 3. Feature extraction
        logger.info("Step 3/4: Extracting features")
        processed_data = self.extract_features(cleaned_data)

        # 4. Save
        logger.info("Step 4/4: Saving processed data")
        self.save_processed_data(processed_data)

        logger.info("=" * 80)
        logger.info("Preprocessing completed!")
        logger.info("=" * 80)
