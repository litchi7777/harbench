"""
Kitchen Smartwatch Dataset (KDDI) Preprocessing

Dataset Overview:
- Cooking activity recognition (kitchen environment)
- 10 subjects
- Sony SmartWatch 3 worn on both wrists
- Sampling rate: 64Hz
- Left arm: 24 activity classes + Undefined (class 0)
- Right arm: 24 activity classes + Undefined (class 0)

Citation:
Y. Mohammad, K. Matsumoto and K. Hoashi, "A dataset for activity recognition
in an unmodified kitchen using smart-watch accelerometers",
MUM 2017, Stuttgart, Germany
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
import logging
import json

import pandas as pd

from .base import BasePreprocessor
from .utils import (
    create_sliding_windows,
    resample_timeseries,
    get_class_distribution,
)
from .common import extract_archive
from . import register_preprocessor
from ..dataset_info import DATASETS

logger = logging.getLogger(__name__)

KDDI_KITCHEN_URL = "https://github.com/yasserfarouk/kitchen_smartwatch_dataset/archive/refs/heads/master.zip"


class KDDIKitchenBasePreprocessor(BasePreprocessor):
    """
    KDDI Kitchen Smartwatch dataset base class
    Provides common processing for left/right arm
    """

    # Override in subclass
    CSV_FILENAME: str = ""
    DATASET_KEY: str = ""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Dataset-specific configuration
        self.num_subjects = 10
        self.num_channels = 3  # 3-axis acceleration

        # Sampling rate
        self.original_sampling_rate = 64  # Hz
        self.target_sampling_rate = config.get('target_sampling_rate', 30)

        # Preprocessing parameters
        self.window_size = config.get('window_size', 150)  # 5 seconds @ 30Hz
        self.stride = config.get('stride', 30)  # 1 second @ 30Hz

        # Scaling factor (m/s² -> G)
        self.scale_factor = DATASETS.get(self.DATASET_KEY, {}).get('scale_factor', 9.8)

        # Class mapping (set in subclass)
        self._class_mapping: Dict[int, int] = {}

    def download_dataset(self) -> None:
        """
        Download KDDI Kitchen Smartwatch dataset from GitHub
        """
        import requests
        import zipfile
        import shutil

        logger.info("=" * 80)
        logger.info("Downloading KDDI Kitchen Smartwatch dataset")
        logger.info("=" * 80)

        raw_path = self.raw_data_path / 'kitchen_smartwatch'
        csv_path = raw_path / self.CSV_FILENAME

        if csv_path.exists():
            logger.info(f"Dataset already exists at {raw_path}")
            return

        raw_path.mkdir(parents=True, exist_ok=True)
        zip_path = raw_path / "kitchen_smartwatch.zip"

        # Download from GitHub
        logger.info("Downloading from GitHub...")
        response = requests.get(KDDI_KITCHEN_URL, stream=True)
        response.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Extract
        logger.info("Extracting archive...")
        extract_archive(zip_path, raw_path, desc='Extracting KDDI Kitchen')

        # Move files from extracted folder
        extracted_dir = raw_path / "kitchen_smartwatch_dataset-master"
        if extracted_dir.exists():
            for item in extracted_dir.iterdir():
                dest = raw_path / item.name
                if not dest.exists():
                    shutil.move(str(item), str(dest))
            shutil.rmtree(str(extracted_dir))

        # Extract inner ZIP files (left_data.csv.zip, right_data.csv.zip)
        for inner_zip in raw_path.glob("*.csv.zip"):
            logger.info(f"Extracting {inner_zip.name}...")
            with zipfile.ZipFile(inner_zip, 'r') as zf:
                zf.extractall(raw_path)
            inner_zip.unlink()

        # Cleanup
        if zip_path.exists():
            zip_path.unlink()

        logger.info(f"KDDI Kitchen dataset downloaded to {raw_path}")

    def load_raw_data(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Load data by subject from CSV file

        Returns:
            {subject_id: (data, labels)}
        """
        raw_path = self.raw_data_path / 'kitchen_smartwatch'
        csv_path = raw_path / self.CSV_FILENAME

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        logger.info(f"Loading {csv_path}")

        # Load CSV
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df):,} rows")

        # Build class mapping (class 0 -> -1, others are 0-indexed)
        unique_classes = sorted([c for c in df['class'].unique() if c != 0])
        self._class_mapping = {0: -1}
        for new_id, orig_id in enumerate(unique_classes):
            self._class_mapping[orig_id] = new_id

        logger.info(f"Class mapping: {len(unique_classes)} activity classes + Undefined")

        # Split data by subject
        result = {}
        for subject_id in sorted(df['subject'].unique()):
            subject_df = df[df['subject'] == subject_id].copy()

            # Sort by time
            subject_df = subject_df.sort_values('time')

            # Extract data and labels
            data = subject_df[['x', 'y', 'z']].values.astype(np.float32)
            original_labels = subject_df['class'].values

            # Map labels
            labels = np.array([self._class_mapping[c] for c in original_labels], dtype=np.int32)

            result[subject_id] = (data, labels)
            logger.info(
                f"Subject {subject_id}: {data.shape[0]:,} samples, "
                f"classes: {np.unique(labels)}"
            )

        return result

    def clean_data(
        self, data: Dict[int, Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Resampling (64Hz -> 30Hz)
        """
        cleaned = {}

        for subject_id, (subject_data, labels) in data.items():
            # Resampling
            resampled_data, resampled_labels = resample_timeseries(
                subject_data,
                labels,
                self.original_sampling_rate,
                self.target_sampling_rate
            )

            cleaned[subject_id] = (resampled_data, resampled_labels)
            logger.info(
                f"Subject {subject_id} resampled: {subject_data.shape[0]:,} -> {resampled_data.shape[0]:,}"
            )

        return cleaned

    def extract_features(
        self, data: Dict[int, Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
        """
        Apply sliding window and scaling
        """
        processed = {}

        for subject_id, (subject_data, labels) in data.items():
            logger.info(f"Processing Subject {subject_id}")

            # Sliding window
            windowed_data, windowed_labels = create_sliding_windows(
                subject_data,
                labels,
                window_size=self.window_size,
                stride=self.stride,
                drop_last=False,
                pad_last=True
            )

            # Apply scaling (m/s² -> G)
            if self.scale_factor is not None:
                windowed_data = windowed_data / self.scale_factor
                logger.info(f"  Applied scale_factor={self.scale_factor}")

            # Reshape: (N, T, C) -> (N, C, T)
            windowed_data = np.transpose(windowed_data, (0, 2, 1))

            # Convert to float16
            windowed_data = windowed_data.astype(np.float16)

            # Save as Wrist/ACC
            processed[subject_id] = {
                'Wrist/ACC': {
                    'X': windowed_data,
                    'Y': windowed_labels
                }
            }

            logger.info(
                f"  Wrist/ACC: X.shape={windowed_data.shape}, Y.shape={windowed_labels.shape}"
            )

        return processed

    def save_processed_data(
        self, data: Dict[int, Dict[str, Dict[str, np.ndarray]]]
    ) -> None:
        """
        Save processed data
        """
        base_path = self.processed_data_path / self.dataset_name
        base_path.mkdir(parents=True, exist_ok=True)

        total_stats = {
            'dataset': self.dataset_name,
            'num_subjects': self.num_subjects,
            'original_sampling_rate': self.original_sampling_rate,
            'target_sampling_rate': self.target_sampling_rate,
            'window_size': self.window_size,
            'stride': self.stride,
            'normalization': 'none',
            'scale_factor': self.scale_factor,
            'data_dtype': 'float16',
            'class_mapping': {str(k): v for k, v in self._class_mapping.items()},
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
                    'X_shape': list(X.shape),
                    'Y_shape': list(Y.shape),
                    'num_windows': len(Y),
                    'class_distribution': get_class_distribution(Y)
                }

                logger.info(f"Saved {user_name}/{sensor_modality_name}: X{X.shape}, Y{Y.shape}")

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


@register_preprocessor('kddi_kitchen_left')
class KDDIKitchenLeftPreprocessor(KDDIKitchenBasePreprocessor):
    """
    KDDI Kitchen Smartwatch - Left arm data
    """

    CSV_FILENAME = "cooking-clean-interpolated-left.csv"
    DATASET_KEY = "KDDI_KITCHEN_LEFT"

    def get_dataset_name(self) -> str:
        return 'kddi_kitchen_left'


@register_preprocessor('kddi_kitchen_right')
class KDDIKitchenRightPreprocessor(KDDIKitchenBasePreprocessor):
    """
    KDDI Kitchen Smartwatch - Right arm data
    """

    CSV_FILENAME = "cooking-clean-interpolated-right.csv"
    DATASET_KEY = "KDDI_KITCHEN_RIGHT"

    def get_dataset_name(self) -> str:
        return 'kddi_kitchen_right'
