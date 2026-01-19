"""
IM-WSHA (IM-Wearable Smart Home Activities) Dataset Preprocessing

IM-WSHA Dataset:
- 11 types of smart home activities
- 10 subjects (5 females, 5 males, 19-60 years old, 55-85kg)
- 3 IMU sensors (MPU-9250): Wrist, Chest, Thigh
- 9-axis data: ACC(3) + GYRO(3) + MAG(3)
- 220 sequences (variable length 45-60 seconds)
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


# IM-WSHA dataset URL
IMWSHA_URL = "https://portals.au.edu.pk/imc/Content/dataset/IM-WSHA_Dataset.zip"


@register_preprocessor('imwsha')
class IMWSHAPreprocessor(BasePreprocessor):
    """
    Preprocessing class for IM-WSHA (IM-Wearable Smart Home Activities) dataset
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # IM-WSHA specific configuration
        self.num_activities = 11
        self.num_subjects = 10
        self.num_sensors = 3  # Wrist, Chest, Thigh

        # Activity labels (1-indexed → 0-indexed)
        self.activity_labels = {
            1: 'Using Computer',
            2: 'Phone Conversation',
            3: 'Vacuum Cleaning',
            4: 'Reading Book',
            5: 'Watching TV',
            6: 'Ironing',
            7: 'Walking',
            8: 'Exercise',
            9: 'Cooking',
            10: 'Drinking',
            11: 'Brushing Hair'
        }

        # Sensor names
        self.sensor_names = ['Wrist', 'Chest', 'Thigh']

        # CSV column names (27 columns: activity + 9 axes × 3 sensors)
        self.csv_columns = ['activity_label'] + \
            [f'{m}{i}' for i in [1, 2, 3] for m in ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz']]

        # Modalities (channel split within each sensor)
        self.sensor_modalities = {
            'Wrist': {
                'ACC': (0, 3),   # ax1, ay1, az1
                'GYRO': (3, 6),  # gx1, gy1, gz1
                'MAG': (6, 9),   # mx1, my1, mz1
            },
            'Chest': {
                'ACC': (9, 12),   # ax2, ay2, az2
                'GYRO': (12, 15), # gx2, gy2, gz2
                'MAG': (15, 18),  # mx2, my2, mz2
            },
            'Thigh': {
                'ACC': (18, 21),  # ax3, ay3, az3
                'GYRO': (21, 24), # gx3, gy3, gz3
                'MAG': (24, 27),  # mx3, my3, mz3
            }
        }

        # Sampling rate (variable, using estimated value)
        self.original_sampling_rate = config.get('original_sampling_rate', 50)  # Hz (estimated)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz

        # Preprocessing parameters
        self.window_size = config.get('window_size', 150)  # 5 seconds @ 30Hz
        self.stride = config.get('stride', 30)  # 1 second @ 30Hz

        # Scaling factor (retrieved from dataset_info.py)
        self.scale_factor = DATASETS.get('IMWSHA', {}).get('scale_factor', None)

    def get_dataset_name(self) -> str:
        return 'imwsha'

    def download_dataset(self) -> None:
        """
        Download and extract IM-WSHA dataset
        """
        import zipfile

        imwsha_dir = self.raw_data_path / 'imwsha'
        target_dir = imwsha_dir / 'IMSHA_Dataset'

        # Check if already downloaded
        if target_dir.exists():
            subject_dirs = list(target_dir.glob('Subject *'))
            if len(subject_dirs) >= 10:
                logger.info(f"IM-WSHA data already exists at {target_dir}")
                return

        imwsha_dir.mkdir(parents=True, exist_ok=True)
        zip_path = imwsha_dir / 'imwsha.zip'

        # Download
        if not zip_path.exists():
            logger.info(f"Downloading IM-WSHA dataset from {IMWSHA_URL}...")
            download_file(IMWSHA_URL, zip_path)
        else:
            logger.info(f"Using existing zip file: {zip_path}")

        # Extract
        logger.info(f"Extracting IM-WSHA data to {imwsha_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(imwsha_dir)

        logger.info(f"Extraction completed: {target_dir}")

        # Verify download completion
        subject_dirs = list(target_dir.glob('Subject *'))
        logger.info(f"Found {len(subject_dirs)} subject directories")

    def load_raw_data(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Load all data files and group by subject

        Returns:
            Dictionary of {subject_id: (data, labels)}
                data: (num_samples, 27) - 3 sensors × 9 axes
                labels: (num_samples,) - activity labels (0-indexed)
        """
        imwsha_dir = self.raw_data_path / 'imwsha' / 'IM-WSHA_Dataset' / 'IMSHA_Dataset'

        if not imwsha_dir.exists():
            raise FileNotFoundError(
                f"IM-WSHA data not found at {imwsha_dir}\n"
                "Please run download_dataset() first."
            )

        logger.info(f"Loading IM-WSHA data from {imwsha_dir}")

        all_data = {}

        # Get subject directories ("Subject 1", "Subject 2", ...)
        subject_dirs = sorted(imwsha_dir.glob('Subject *'))

        for subject_dir in subject_dirs:
            # Extract subject ID ("Subject 1" → 1)
            try:
                subject_id = int(subject_dir.name.split()[-1])
            except (IndexError, ValueError):
                logger.warning(f"Invalid subject directory name: {subject_dir.name}")
                continue

            # Search for CSV files (filename varies by subject)
            csv_files = list(subject_dir.glob('3-imu*.csv'))

            if len(csv_files) == 0:
                logger.warning(f"No CSV file found in {subject_dir}")
                continue

            csv_file = csv_files[0]  # Use the first CSV file

            try:
                # Load CSV
                df = pd.read_csv(csv_file)

                # Convert activity_label to 0-indexed (1-11 → 0-10)
                labels = df['activity_label'].values - 1

                # Extract 27-channel sensor data (all columns except activity label)
                data = df.iloc[:, 1:].values  # (N, 27)

                all_data[subject_id] = (data, labels)
                logger.info(f"USER{subject_id:05d}: data={data.shape}, labels={labels.shape}, "
                           f"unique labels={np.unique(labels)}")

            except Exception as e:
                logger.error(f"Error loading {csv_file}: {e}")

        logger.info(f"Loaded {len(all_data)} subjects successfully")
        return all_data

    def clean_data(
        self,
        data: Dict[int, Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Data cleaning and resampling
        """
        cleaned = {}

        for subject_id, (subject_data, labels) in data.items():
            # Remove NaN values
            valid_mask = ~np.isnan(subject_data).any(axis=1)
            cleaned_data = subject_data[valid_mask]
            cleaned_labels = labels[valid_mask]

            # Estimate sampling rate (from sequence length)
            # Average 45-60 seconds, assuming estimated 50Hz
            estimated_duration = len(cleaned_data) / self.original_sampling_rate
            logger.info(f"USER{subject_id:05d}: estimated duration={estimated_duration:.1f}s "
                       f"({len(cleaned_data)} samples @ {self.original_sampling_rate}Hz)")

            # Resampling
            if self.original_sampling_rate != self.target_sampling_rate:
                resampled_data, resampled_labels = resample_timeseries(
                    cleaned_data,
                    cleaned_labels,
                    self.original_sampling_rate,
                    self.target_sampling_rate
                )
                cleaned[subject_id] = (resampled_data, resampled_labels)
                logger.info(f"USER{subject_id:05d} resampled: {resampled_data.shape}")
            else:
                cleaned[subject_id] = (cleaned_data, cleaned_labels)
                logger.info(f"USER{subject_id:05d} cleaned: {cleaned_data.shape}")

        return cleaned

    def extract_features(
        self,
        data: Dict[int, Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
        """
        Feature extraction (windowing and scaling for each sensor × modality)

        Returns:
            {subject_id: {sensor/modality: {'X': data, 'Y': labels}}}
        """
        processed = {}

        for subject_id, (subject_data, labels) in data.items():
            logger.info(f"Processing USER{subject_id:05d}")
            processed[subject_id] = {}

            # Apply sliding window (all 27 channels at once)
            windowed_data, windowed_labels = create_sliding_windows(
                subject_data,
                labels,
                window_size=self.window_size,
                stride=self.stride,
                drop_last=False,
                pad_last=True
            )

            # Split by each sensor × modality
            for sensor_name, modalities in self.sensor_modalities.items():
                for modality_name, (mod_start_ch, mod_end_ch) in modalities.items():
                    modality_data = windowed_data[:, :, mod_start_ch:mod_end_ch]  # (num_windows, window_size, 3)

                    # Apply scaling (ACC only, if scale_factor is defined)
                    if modality_name == 'ACC' and self.scale_factor is not None:
                        modality_data = modality_data / self.scale_factor
                        logger.info(f"  Applied scale_factor={self.scale_factor} to {sensor_name}/{modality_name}")

                    # Shape transformation: (N, T, C) -> (N, C, T)
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
            data/processed/imwsha/USER00001/Wrist/ACC/X.npy, Y.npy
            data/processed/imwsha/USER00001/Wrist/GYRO/X.npy, Y.npy
            data/processed/imwsha/USER00001/Wrist/MAG/X.npy, Y.npy
            ...(same for Chest, Thigh)
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

        # Save metadata (convert NumPy types to JSON-compatible format)
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
