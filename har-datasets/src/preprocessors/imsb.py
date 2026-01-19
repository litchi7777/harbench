"""
IMSB (IM-SportingBehaviors) dataset preprocessing

IMSB dataset:
- 6 types of sporting behaviors (Badminton, Basketball, Cycling, Football, Skipping, Table Tennis)
- 20 subjects (professional + amateur athletes, aged 20-30)
- 2 sensors (Wrist, Neck) * Thigh excluded due to many missing values
- Sampling rate: 20Hz (estimated)
- Reference: portals.au.edu.pk/imc
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


# IMSB dataset URL
IMSB_URL = "https://portals.au.edu.pk/imc/Content/dataset/IMSB%20dataset.zip"


@register_preprocessor('imsb')
class IMSBPreprocessor(BasePreprocessor):
    """
    Preprocessing class for IMSB (IM-SportingBehaviors) dataset
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # IMSB-specific configuration
        self.num_activities = 6
        self.num_subjects = 20  # Actually 85 files, but approximately 20 subjects
        self.num_sensors = 2  # Only Wrist and Neck (Thigh excluded)

        # Activity to directory mapping
        self.activity_map = {
            'badminton': 0,
            'basketball': 1,
            'cycling': 2,
            'football': 3,
            'skipping': 4,
            'tabletennis': 5
        }

        # Sensor names and CSV column mapping
        self.sensor_names = ['Wrist', 'Neck']
        self.sensor_columns = {
            'Wrist': ['wx', 'wy', 'wz'],  # 3-axis acceleration
            'Neck': ['nx', 'ny', 'nz']     # 3-axis acceleration
        }

        # Modalities (each sensor has ACC only)
        self.sensor_modalities = {
            'Wrist': {
                'ACC': (0, 3),  # 3-axis acceleration
            },
            'Neck': {
                'ACC': (0, 3),  # 3-axis acceleration
            }
        }

        # Sampling rate
        self.original_sampling_rate = 20  # Hz (estimated: 1000samples/50s)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz

        # Preprocessing parameters
        self.window_size = config.get('window_size', 150)  # 5 seconds @ 30Hz
        self.stride = config.get('stride', 30)  # 1 second @ 30Hz

        # Scaling factor (G unit, no conversion needed)
        self.scale_factor = DATASETS.get('IMSB', {}).get('scale_factor', None)

    def get_dataset_name(self) -> str:
        return 'imsb'

    def download_dataset(self) -> None:
        """
        Download and extract IMSB dataset
        """
        zip_path = self.raw_data_path / 'imsb.zip'
        extract_dir = self.raw_data_path / 'imsb'

        # Check if already downloaded
        if extract_dir.exists():
            csv_files = list(extract_dir.rglob('*.csv'))
            if len(csv_files) >= 85:
                logger.info(f"IMSB data already exists at {extract_dir}")
                return

        # Download
        if not zip_path.exists():
            logger.info(f"Downloading IMSB dataset from {IMSB_URL}...")
            download_file(IMSB_URL, zip_path)
        else:
            logger.info(f"Using existing zip file: {zip_path}")

        # Extract
        logger.info(f"Extracting IMSB data to {extract_dir}...")
        extract_archive(zip_path, self.raw_data_path / 'imsb')

        logger.info(f"Extraction completed: {extract_dir}")

    def parse_filename(self, filename: str) -> Tuple[int, str, str]:
        """
        Extract information from filename

        Filename format: 001-M-badminton.csv
        - 001: subject number
        - M: gender (M=male, F=female)
        - badminton: activity name

        Returns:
            (subject_id, gender, activity)
        """
        stem = Path(filename).stem
        parts = stem.split('-')

        if len(parts) >= 3:
            subject_id = int(parts[0])
            gender = parts[1]
            activity = parts[2]
            return subject_id, gender, activity
        else:
            raise ValueError(f"Invalid filename format: {filename}")

    def load_csv_file(self, csv_file: Path) -> Tuple[np.ndarray, int]:
        """
        Load CSV file

        Returns:
            data: (num_samples, 6) - Wrist(3) + Neck(3)
            label: activity label ID
        """
        # Extract activity from filename
        _, _, activity = self.parse_filename(csv_file.name)
        label = self.activity_map.get(activity.lower())

        if label is None:
            raise ValueError(f"Unknown activity: {activity}")

        # Load CSV
        df = pd.read_csv(csv_file)

        # Extract required columns (Wrist + Neck, Thigh excluded)
        wrist_cols = self.sensor_columns['Wrist']
        neck_cols = self.sensor_columns['Neck']

        # Combine data
        wrist_data = df[wrist_cols].values  # (N, 3)
        neck_data = df[neck_cols].values    # (N, 3)

        # Remove missing values
        valid_mask = ~(np.isnan(wrist_data).any(axis=1) | np.isnan(neck_data).any(axis=1))
        wrist_data = wrist_data[valid_mask]
        neck_data = neck_data[valid_mask]

        # Concatenate: (N, 6)
        data = np.hstack([wrist_data, neck_data])

        return data, label

    def load_raw_data(self) -> Dict[int, list]:
        """
        Load all data files and group by subject

        Returns:
            Dictionary of {subject_id: [(data, labels), ...]}
                Each element is a CSV file (session) unit
                data: (num_samples, 6) - Wrist(3) + Neck(3)
                labels: (num_samples,) - activity labels
        """
        imsb_dir = self.raw_data_path / 'imsb'

        # Get all CSV files (under activity directories)
        csv_files = []
        for activity_dir in imsb_dir.iterdir():
            if activity_dir.is_dir() and not activity_dir.name.startswith('.'):
                csv_files.extend(activity_dir.glob('*.csv'))

        if len(csv_files) == 0:
            raise FileNotFoundError(
                f"No CSV files found in {imsb_dir}\n"
                "Please run download_dataset() first."
            )

        logger.info(f"Loading {len(csv_files)} CSV files...")

        # Group files by subject (maintain as session list)
        from collections import defaultdict
        subject_data_list = defaultdict(list)

        for csv_file in csv_files:
            try:
                data, label = self.load_csv_file(csv_file)

                # Get subject ID from filename
                subject_id, _, _ = self.parse_filename(csv_file.name)

                # Assign same label to all samples
                labels = np.full(len(data), label, dtype=np.int32)

                # Add to subject's session list
                subject_data_list[subject_id].append((data, labels))
                logger.info(f"Loaded {csv_file.name}: data={data.shape}, label={label}")

            except Exception as e:
                logger.error(f"Error loading {csv_file}: {e}")

        # Return session list (do not concatenate)
        all_data = {}
        for subject_id, sessions in subject_data_list.items():
            all_data[subject_id] = sessions
            total_samples = sum(d.shape[0] for d, l in sessions)
            logger.info(f"USER{subject_id:05d}: {len(sessions)} sessions, {total_samples} samples")

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

        for subject_id, sessions in data.items():
            cleaned_sessions = []
            for session_data, session_labels in sessions:
                # Remove NaN (already done at load time, but just in case)
                valid_mask = ~np.isnan(session_data).any(axis=1)
                cleaned_data = session_data[valid_mask]
                cleaned_labels = session_labels[valid_mask]

                if len(cleaned_data) == 0:
                    continue

                # Resampling (20Hz -> 30Hz)
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
                cleaned[subject_id] = cleaned_sessions
                total_samples = sum(d.shape[0] for d, l in cleaned_sessions)
                logger.info(f"USER{subject_id:05d} cleaned: {len(cleaned_sessions)} sessions, {total_samples} samples")

        return cleaned

    def extract_features(
        self,
        data: Dict[int, list]
    ) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
        """
        Feature extraction (windowing per sensor/modality)
        Windows do not cross session boundaries

        Returns:
            {subject_id: {sensor/modality: {'X': data, 'Y': labels}}}
        """
        processed = {}

        for subject_id, sessions in data.items():
            logger.info(f"Processing USER{subject_id:05d} ({len(sessions)} sessions)")
            processed[subject_id] = {}

            # Split into Wrist and Neck session lists
            wrist_sessions = [(d[:, 0:3], l) for d, l in sessions]
            neck_sessions = [(d[:, 3:6], l) for d, l in sessions]

            for sensor_name, sensor_sessions in [('Wrist', wrist_sessions), ('Neck', neck_sessions)]:
                # Apply sliding window per session
                windowed_data, windowed_labels = create_sliding_windows_multi_session(
                    sensor_sessions,
                    window_size=self.window_size,
                    stride=self.stride,
                    drop_last=False,
                    pad_last=True
                )

                if len(windowed_data) == 0:
                    logger.warning(f"  {sensor_name}: no valid windows")
                    continue

                # ACC modality
                modality_name = 'ACC'
                modality_data = windowed_data  # (num_windows, window_size, 3)

                # Apply scaling (not needed for G unit, but check anyway)
                if modality_name == 'ACC' and self.scale_factor is not None:
                    modality_data = modality_data / self.scale_factor
                    logger.info(f"  Applied scale_factor={self.scale_factor} to {sensor_name}/{modality_name}")

                # Reshape: (N, T, C) -> (N, C, T)
                modality_data = np.transpose(modality_data, (0, 2, 1))

                # Convert to float16 (memory efficiency)
                modality_data = modality_data.astype(np.float16)

                sensor_modality_key = f"{sensor_name}/{modality_name}"
                processed[subject_id][sensor_modality_key] = {
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
            data/processed/imsb/USER00001/Wrist/ACC/X.npy, Y.npy
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
            'normalization': 'none',
            'scale_factor': self.scale_factor,
            'data_dtype': 'float16',
            'users': {}
        }

        for subject_id, sensor_modality_data in data.items():
            user_name = f"USER{subject_id:05d}"
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

        # Save metadata (convert NumPy types to JSON-compatible)
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
