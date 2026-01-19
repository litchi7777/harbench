"""
SBRHAPT (Smartphone-Based Recognition of Human Activities and Postural Transitions) dataset preprocessing

SBRHAPT dataset:
- 30 subjects (19-48 years old)
- 12 activity types: 6 basic activities + 6 postural transitions
  - Basic activities: Standing, Sitting, Lying, Walking, Walking Downstairs, Walking Upstairs
  - Postural transitions: Stand-to-Sit, Sit-to-Stand, Sit-to-Lie, Lie-to-Sit, Stand-to-Lie, Lie-to-Stand
- Smartphone worn on waist (Samsung Galaxy S II)
- 6-axis data: ACC(3) + GYRO(3)
- Sampling rate: 50Hz
- Reference: https://archive.ics.uci.edu/dataset/341
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
    resample_timeseries,
    get_class_distribution
)
from .common import (
    download_file,
    extract_archive,
    cleanup_temp_files,
    check_dataset_exists
)
from . import register_preprocessor
from ..dataset_info import DATASETS

logger = logging.getLogger(__name__)


# SBRHAPT dataset URL
SBRHAPT_URL = "https://archive.ics.uci.edu/static/public/341/smartphone+based+recognition+of+human+activities+and+postural+transitions.zip"


@register_preprocessor('sbrhapt')
class SBRHAPTPreprocessor(BasePreprocessor):
    """
    Preprocessing class for SBRHAPT (Smartphone-Based Recognition of Human Activities and Postural Transitions) dataset
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # SBRHAPT-specific settings
        self.num_activities = 12  # 6 basic + 6 postural transitions
        self.num_subjects = 30

        # Activity label mapping (corresponds to activity_id in activity_labels.txt)
        self.activity_labels = {
            1: 0,   # WALKING
            2: 1,   # WALKING_UPSTAIRS
            3: 2,   # WALKING_DOWNSTAIRS
            4: 3,   # SITTING
            5: 4,   # STANDING
            6: 5,   # LAYING
            7: 6,   # STAND_TO_SIT
            8: 7,   # SIT_TO_STAND
            9: 8,   # SIT_TO_LIE
            10: 9,  # LIE_TO_SIT
            11: 10, # STAND_TO_LIE
            12: 11, # LIE_TO_STAND
        }

        # Sensor names
        self.sensor_names = ['Waist']

        # Modalities
        self.sensor_modalities = {
            'Waist': {
                'ACC': (0, 3),   # body_acc_x, body_acc_y, body_acc_z
                'GYRO': (3, 6),  # body_gyro_x, body_gyro_y, body_gyro_z
            }
        }

        # Sampling rate
        self.original_sampling_rate = config.get('original_sampling_rate', 50)  # Hz
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz

        # Preprocessing parameters
        self.window_size = config.get('window_size', 150)  # 5 seconds @ 30Hz
        self.stride = config.get('stride', 30)  # 1 second @ 30Hz

        # Scaling factor (m/s² → G)
        self.scale_factor = DATASETS.get('SBRHAPT', {}).get('scale_factor', 9.8)

    def get_dataset_name(self) -> str:
        return 'sbrhapt'

    def download_dataset(self) -> None:
        """
        Download and extract SBRHAPT dataset
        """
        import zipfile

        sbrhapt_dir = self.raw_data_path / 'sbrhapt'
        target_dir = sbrhapt_dir / 'RawData'

        # Check if already downloaded
        if target_dir.exists():
            acc_files = list(target_dir.glob('acc_*.txt'))
            gyro_files = list(target_dir.glob('gyro_*.txt'))
            if len(acc_files) >= 60 and len(gyro_files) >= 60:  # Minimum number of files
                logger.info(f"SBRHAPT data already exists at {target_dir}")
                return

        sbrhapt_dir.mkdir(parents=True, exist_ok=True)
        zip_path = sbrhapt_dir / 'sbrhapt.zip'

        # Download
        if not zip_path.exists():
            logger.info(f"Downloading SBRHAPT dataset from {SBRHAPT_URL}...")
            download_file(SBRHAPT_URL, zip_path)
        else:
            logger.info(f"Using existing zip file: {zip_path}")

        # Extract
        logger.info(f"Extracting SBRHAPT data to {sbrhapt_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(sbrhapt_dir)

        logger.info(f"Extraction completed: {target_dir}")

        # Verify download completion
        acc_files = list(target_dir.glob('acc_*.txt'))
        gyro_files = list(target_dir.glob('gyro_*.txt'))
        logger.info(f"Found {len(acc_files)} accelerometer files and {len(gyro_files)} gyroscope files")

    def load_labels_file(self) -> Dict[Tuple[int, int], list]:
        """
        Load labels.txt and retrieve label information for each experiment

        Returns:
            {(exp_id, user_id): [(start, end, activity_id), ...]}
        """
        labels_file = self.raw_data_path / 'sbrhapt' / 'RawData' / 'labels.txt'

        if not labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_file}")

        labels_dict = {}
        with open(labels_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    exp_id = int(parts[0])
                    user_id = int(parts[1])
                    activity_id = int(parts[2])
                    start_sample = int(parts[3])
                    end_sample = int(parts[4])

                    key = (exp_id, user_id)
                    if key not in labels_dict:
                        labels_dict[key] = []
                    labels_dict[key].append((start_sample, end_sample, activity_id))

        logger.info(f"Loaded labels for {len(labels_dict)} experiments")
        return labels_dict

    def load_raw_data(self) -> Dict[int, list]:
        """
        Load all data files and group by subject

        Returns:
            Dictionary of {subject_id: [(data, labels), ...]}
                Each element represents an experiment file (session)
                data: (num_samples, 6) - ACC(3) + GYRO(3)
                labels: (num_samples,) - activity labels (0-indexed)
        """
        raw_dir = self.raw_data_path / 'sbrhapt' / 'RawData'

        if not raw_dir.exists():
            raise FileNotFoundError(
                f"SBRHAPT raw data not found at {raw_dir}\n"
                "Please run download_dataset() first."
            )

        logger.info(f"Loading SBRHAPT data from {raw_dir}")

        # Load label information
        labels_dict = self.load_labels_file()

        # Store data per subject
        from collections import defaultdict
        subject_data_list = defaultdict(list)

        # Process all experiment files
        acc_files = sorted(raw_dir.glob('acc_exp*.txt'))

        for acc_file in acc_files:
            # Extract exp_id and user_id from filename
            # Filename format: acc_exp01_user01.txt
            stem = acc_file.stem  # 'acc_exp01_user01'
            parts = stem.split('_')
            exp_id = int(parts[1][3:])  # 'exp01' -> 1
            user_id = int(parts[2][4:])  # 'user01' -> 1

            # Corresponding gyro file
            gyro_file = raw_dir / f"gyro_exp{exp_id:02d}_user{user_id:02d}.txt"

            if not gyro_file.exists():
                logger.warning(f"Gyroscope file not found: {gyro_file}")
                continue

            # Get label information
            key = (exp_id, user_id)
            if key not in labels_dict:
                logger.warning(f"No labels found for exp={exp_id}, user={user_id}")
                continue

            try:
                # Load accelerometer data (3 columns)
                acc_data = np.loadtxt(acc_file)  # (N, 3)

                # Load gyroscope data (3 columns)
                gyro_data = np.loadtxt(gyro_file)  # (N, 3)

                # Check data length
                if acc_data.shape[0] != gyro_data.shape[0]:
                    logger.warning(f"Data length mismatch for {acc_file.name}: "
                                 f"acc={acc_data.shape[0]}, gyro={gyro_data.shape[0]}")
                    continue

                # Concatenate data: (N, 6)
                data = np.hstack([acc_data, gyro_data])

                # Create label array (initial value is -1: undefined)
                labels = np.full(len(data), -1, dtype=np.int32)

                # Apply label information
                for start_sample, end_sample, activity_id in labels_dict[key]:
                    # Sample indices are 1-indexed, so convert to 0-indexed
                    start_idx = start_sample - 1
                    end_idx = end_sample  # end is exclusive

                    # Range check
                    if start_idx < 0 or end_idx > len(labels):
                        logger.warning(f"Label range out of bounds: [{start_idx}, {end_idx})")
                        continue

                    # Convert activity ID to 0-indexed label
                    if activity_id in self.activity_labels:
                        labels[start_idx:end_idx] = self.activity_labels[activity_id]
                    else:
                        logger.warning(f"Unknown activity ID: {activity_id}")

                # Extract only samples with defined labels
                valid_mask = labels >= 0
                data = data[valid_mask]
                labels = labels[valid_mask]

                # Add to session list per subject
                subject_data_list[user_id].append((data, labels))
                logger.info(f"Loaded exp{exp_id:02d}_user{user_id:02d}: data={data.shape}, "
                           f"unique labels={np.unique(labels)}")

            except Exception as e:
                logger.error(f"Error loading {acc_file}: {e}")

        # Return session list (do not concatenate)
        all_data = {}
        for user_id, sessions in subject_data_list.items():
            all_data[user_id] = sessions
            total_samples = sum(d.shape[0] for d, l in sessions)
            logger.info(f"USER{user_id:05d}: {len(sessions)} sessions, {total_samples} samples")

        logger.info(f"Loaded {len(all_data)} subjects successfully")
        return all_data

    def clean_data(
        self,
        data: Dict[int, list]
    ) -> Dict[int, list]:
        """
        Data cleaning and resampling (process per session)
        """
        cleaned = {}

        for user_id, sessions in data.items():
            cleaned_sessions = []
            for session_data, session_labels in sessions:
                # Remove NaN
                valid_mask = ~np.isnan(session_data).any(axis=1)
                cleaned_data = session_data[valid_mask]
                cleaned_labels = session_labels[valid_mask]

                if len(cleaned_data) == 0:
                    continue

                # Resample (50Hz -> 30Hz)
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

    def extract_features(
        self,
        data: Dict[int, list]
    ) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
        """
        Feature extraction (windowing and scaling per sensor/modality)
        Does not generate windows crossing session boundaries

        Returns:
            {subject_id: {sensor/modality: {'X': data, 'Y': labels}}}
        """
        processed = {}

        for user_id, sessions in data.items():
            logger.info(f"Processing USER{user_id:05d} ({len(sessions)} sessions)")
            processed[user_id] = {}

            # Apply sliding window per session (all 6 channels at once)
            windowed_data, windowed_labels = create_sliding_windows_multi_session(
                sessions,
                window_size=self.window_size,
                stride=self.stride,
                drop_last=False,
                pad_last=True
            )

            if len(windowed_data) == 0:
                logger.warning(f"  USER{user_id:05d} has no valid windows, skipping")
                continue

            # Split into each modality
            for sensor_name, modalities in self.sensor_modalities.items():
                for modality_name, (mod_start_ch, mod_end_ch) in modalities.items():
                    modality_data = windowed_data[:, :, mod_start_ch:mod_end_ch]  # (num_windows, window_size, 3)

                    # Apply scaling (ACC only, m/s² → G)
                    if modality_name == 'ACC' and self.scale_factor is not None:
                        modality_data = modality_data / self.scale_factor
                        logger.info(f"  Applied scale_factor={self.scale_factor} to {sensor_name}/{modality_name}")

                    # Shape transformation: (N, T, C) -> (N, C, T)
                    modality_data = np.transpose(modality_data, (0, 2, 1))

                    # Convert to float16 (memory efficiency)
                    modality_data = modality_data.astype(np.float16)

                    sensor_modality_key = f"{sensor_name}/{modality_name}"
                    processed[user_id][sensor_modality_key] = {
                        'X': modality_data,
                        'Y': windowed_labels
                    }

                    logger.info(f"  {sensor_modality_key}: X.shape={modality_data.shape}, "
                               f"Y.shape={windowed_labels.shape}")

        return processed

    def save_processed_data(
        self,
        data: Dict[int, Dict[str, Dict[str, np.ndarray]]]
    ) -> None:
        """
        Save processed data

        Save format:
            data/processed/sbrhapt/USER00001/Waist/ACC/X.npy, Y.npy
            data/processed/sbrhapt/USER00001/Waist/GYRO/X.npy, Y.npy
        """
        import json

        base_path = self.processed_data_path / self.dataset_name
        base_path.mkdir(parents=True, exist_ok=True)

        total_stats = {
            'dataset': self.dataset_name,
            'num_activities': self.num_activities,
            'num_sensors': 1,
            'sensor_names': self.sensor_names,
            'original_sampling_rate': self.original_sampling_rate,
            'target_sampling_rate': self.target_sampling_rate,
            'window_size': self.window_size,
            'stride': self.stride,
            'normalization': 'none',
            'scale_factor': self.scale_factor,
            'data_dtype': 'float16',
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

                X = arrays['X']
                Y = arrays['Y']

                np.save(sensor_modality_path / 'X.npy', X)
                np.save(sensor_modality_path / 'Y.npy', Y)

                user_stats['sensor_modalities'][sensor_modality_name] = {
                    'X_shape': X.shape,
                    'Y_shape': Y.shape,
                    'num_windows': len(Y),
                    'class_distribution': get_class_distribution(Y)
                }

                logger.info(f"Saved {user_name}/{sensor_modality_name}: X{X.shape}, Y{Y.shape}")

            total_stats['users'][user_name] = user_stats

        # Save metadata (convert NumPy types to JSON-compatible types)
        metadata_path = base_path / 'metadata.json'

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

        with open(metadata_path, 'w') as f:
            json.dump(serializable_stats, f, indent=2)

        logger.info(f"Saved metadata to {metadata_path}")
        logger.info(f"Preprocessing completed: {base_path}")
