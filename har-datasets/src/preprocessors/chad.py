"""
CHAD (Complex Human Activities Dataset) Preprocessing

CHAD Dataset (formerly UT-Complex):
- 13 complex human activities
- 2 sensor positions (smartphone: Pocket, Wrist)
- Sensor types: ACC(3-axis) + Linear ACC(3-axis) + GYRO(3-axis) + MAG(3-axis)
- Sampling rate: estimated ~50Hz
- Single CSV file per position
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


# CHAD Dataset URL
CHAD_URL = "https://www.utwente.nl/en/eemcs/ps/dataset-folder/ut-data-complex.rar"


@register_preprocessor('chad')
class CHADPreprocessor(BasePreprocessor):
    """
    Preprocessor class for CHAD dataset
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # CHAD specific settings
        self.num_activities = 13

        # Activity label mapping (5-digit code -> 0-indexed)
        self.activity_map = {
            11111: 0,  # walk
            11112: 1,  # stand
            11113: 2,  # jog
            11114: 3,  # sit
            11115: 4,  # bike
            11116: 5,  # upstairs
            11117: 6,  # downstairs
            11118: 7,  # type
            11119: 8,  # write
            11120: 9,  # coffee
            11121: 10,  # talk
            11122: 11,  # smoke
            11123: 12   # eat
        }

        # Sensor positions
        self.sensor_names = ['Pocket', 'Wrist']
        self.csv_files = {
            'Pocket': 'smartphoneatpocket.csv',
            'Wrist': 'smartphoneatwrist.csv'
        }

        # Modalities and channel configuration
        # CSV columns: timestamp(0) + acc(1-3) + linear_acc(4-6) + gyro(7-9) + mag(10-12) + label(13) = 14
        # person_data = iloc[:, 1:13] → 0-indexed with 12 channels
        self.modalities = ['ACC', 'LINACC', 'GYRO', 'MAG']
        self.channel_ranges = {
            'ACC': (0, 3),      # person_data indices 0-2 (CSV columns 1-3)
            'LINACC': (3, 6),   # person_data indices 3-5 (CSV columns 4-6)
            'GYRO': (6, 9),     # person_data indices 6-8 (CSV columns 7-9)
            'MAG': (9, 12)      # person_data indices 9-11 (CSV columns 10-12)
        }

        # Sampling rate
        self.original_sampling_rate = config.get('original_sampling_rate', 50)  # Hz (estimated)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz

        # Preprocessing parameters
        self.window_size = config.get('window_size', 150)  # 5 seconds @ 30Hz
        self.stride = config.get('stride', 30)  # 1 second @ 30Hz

        # Scaling factor (needs data verification)
        self.scale_factor = DATASETS.get('CHAD', {}).get('scale_factor', None)

    def get_dataset_name(self) -> str:
        return 'chad'

    def download_dataset(self) -> None:
        """
        Download and extract CHAD dataset
        """
        logger.info("=" * 80)
        logger.info("Downloading CHAD dataset")
        logger.info("=" * 80)

        chad_raw_path = self.raw_data_path / self.dataset_name

        # Check if data already exists
        if check_dataset_exists(chad_raw_path, required_files=['*.csv']):
            logger.warning(f"CHAD data already exists at {chad_raw_path}")
            response = input("Do you want to re-download? (y/N): ")
            if response.lower() != 'y':
                logger.info("Skipping download")
                return

        try:
            # 1. Download
            logger.info("Step 1/2: Downloading archive")
            chad_raw_path.parent.mkdir(parents=True, exist_ok=True)
            rar_path = chad_raw_path.parent / 'chad.rar'
            download_file(CHAD_URL, rar_path, desc='Downloading CHAD')

            # 2. Extract
            logger.info("Step 2/2: Extracting data")
            extract_to = chad_raw_path.parent / 'chad_temp'
            extract_archive(rar_path, extract_to, desc='Extracting CHAD')

            # Organize data
            self._organize_chad_data(extract_to, chad_raw_path)

            # Cleanup
            cleanup_temp_files(extract_to)
            if rar_path.exists():
                rar_path.unlink()

            logger.info("=" * 80)
            logger.info(f"SUCCESS: CHAD dataset downloaded to {chad_raw_path}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Failed to download CHAD dataset: {e}", exc_info=True)
            raise

    def _organize_chad_data(self, extracted_path: Path, target_path: Path) -> None:
        """
        Organize CHAD data into appropriate directory structure

        Args:
            extracted_path: Path to extracted data
            target_path: Target path for organized data (data/raw/chad)
        """
        logger.info(f"Organizing CHAD data from {extracted_path} to {target_path}")

        # Look for "UT_Data_Complex" folder
        data_root = extracted_path / "UT_Data_Complex"
        if not data_root.exists():
            # Folder is directly under extracted path
            data_root = extracted_path

        target_path.mkdir(parents=True, exist_ok=True)

        # Copy CSV files
        import shutil
        for csv_file in ['smartphoneatpocket.csv', 'smartphoneatwrist.csv']:
            src = data_root / csv_file
            dst = target_path / csv_file
            if src.exists():
                shutil.copy2(src, dst)
                logger.info(f"Copied {csv_file}")
            else:
                logger.warning(f"File not found: {csv_file}")

        logger.info(f"Data organized at: {target_path}")

    def load_raw_data(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Load CHAD raw data

        Load all data from CSV for each sensor position, assign pseudo user IDs
        (dataset is assumed to be long-term recording from single subject)

        Returns:
            person_data: {person_id: (data, labels)} dictionary
                data: (num_samples, 12) array (ACC:3 + LINACC:3 + GYRO:3 + MAG:3)
                labels: (num_samples,) array
        """
        raw_path = self.raw_data_path / self.dataset_name

        if not raw_path.exists():
            raise FileNotFoundError(
                f"CHAD raw data not found at {raw_path}\n"
                "Expected structure: data/raw/chad/smartphoneatpocket.csv"
            )

        # Load data for each sensor position
        # person_id = 1: Pocket, person_id = 2: Wrist
        person_data = {}

        for person_id, sensor_name in enumerate(self.sensor_names, start=1):
            csv_file = raw_path / self.csv_files[sensor_name]

            if not csv_file.exists():
                logger.warning(f"CSV file not found: {csv_file}")
                continue

            logger.info(f"Loading {sensor_name} data from {csv_file.name}")

            # Load CSV without header
            df = pd.read_csv(csv_file, header=None)

            # Extract data and labels
            # columns: timestamp(0) + acc(1-3) + linear_acc(4-6) + gyro(7-9) + mag(10-12) + label(13)
            data = df.iloc[:, 1:13].values.astype(np.float32)  # 12 channels
            labels_raw = df.iloc[:, 13].values.astype(np.int32)

            # Label conversion (5-digit code -> 0-indexed)
            labels = np.array([self.activity_map.get(label, -1) for label in labels_raw])

            # Check for invalid labels
            invalid_count = np.sum(labels == -1)
            if invalid_count > 0:
                logger.warning(f"{sensor_name}: {invalid_count} samples with invalid labels")

            person_data[person_id] = (data, labels)
            logger.info(f"USER{person_id:05d} ({sensor_name}): {data.shape}, Labels: {labels.shape}, "
                       f"Unique labels: {np.unique(labels)}")

        if not person_data:
            raise ValueError("No data loaded. Please check the raw data directory structure.")

        logger.info(f"Total sensor positions loaded: {len(person_data)}")
        return person_data

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

            # Resampling (estimated 50Hz -> 30Hz)
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
        Feature extraction (windowing and scaling per sensor x modality)

        Args:
            data: {person_id: (data, labels)} dictionary

        Returns:
            {person_id: {sensor_modality: {'X': data, 'Y': labels}}}
        """
        processed = {}

        for person_id, (person_data, labels) in data.items():
            sensor_name = self.sensor_names[person_id - 1]  # person_id 1=Pocket, 2=Wrist
            logger.info(f"Processing USER{person_id:05d} ({sensor_name})")

            processed[person_id] = {}

            # Process each modality
            for modality_name in self.modalities:
                start_ch, end_ch = self.channel_ranges[modality_name]

                # Extract modality channels
                modality_data = person_data[:, start_ch:end_ch]

                # Apply sliding window
                windowed_data, windowed_labels = create_sliding_windows(
                    modality_data,
                    labels,
                    window_size=self.window_size,
                    stride=self.stride,
                    drop_last=False,
                    pad_last=True
                )

                # Apply scaling (ACC and LINACC only)
                if modality_name in ['ACC', 'LINACC'] and self.scale_factor is not None:
                    windowed_data = windowed_data / self.scale_factor
                    logger.info(f"  Applied scale_factor={self.scale_factor} to {sensor_name}/{modality_name}")

                # Shape transformation: (num_windows, window_size, 3) -> (num_windows, 3, window_size)
                windowed_data = np.transpose(windowed_data, (0, 2, 1))

                # Convert to float16
                windowed_data = windowed_data.astype(np.float16)

                # Sensor/modality hierarchy
                sensor_modality_key = f"{sensor_name}/{modality_name}"

                processed[person_id][sensor_modality_key] = {
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
            data: {person_id: {sensor_modality: {'X': data, 'Y': labels}}}

        Save format:
            data/processed/utcomplex/USER00001/Pocket/ACC/X.npy, Y.npy
            data/processed/utcomplex/USER00002/Wrist/ACC/X.npy, Y.npy
        """
        import json

        base_path = self.processed_data_path / self.dataset_name
        base_path.mkdir(parents=True, exist_ok=True)

        total_stats = {
            'dataset': self.dataset_name,
            'num_activities': self.num_activities,
            'sensor_names': self.sensor_names,
            'modalities': self.modalities,
            'channels_per_modality': 3,
            'original_sampling_rate': self.original_sampling_rate,
            'target_sampling_rate': self.target_sampling_rate,
            'window_size': self.window_size,
            'stride': self.stride,
            'normalization': 'none',
            'scale_factor': self.scale_factor,
            'data_dtype': 'float16',
            'data_shape': f'(num_windows, 3, {self.window_size})',
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

                logger.info(
                    f"Saved {user_name}/{sensor_modality_name}: "
                    f"X{X.shape}, Y{Y.shape}"
                )

            total_stats['users'][user_name] = user_stats

        # Save metadata
        metadata_path = base_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
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
