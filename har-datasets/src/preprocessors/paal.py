"""
PAAL Dataset Preprocessor

Dataset overview:
- Name: PAAL ADL Accelerometry Dataset v2.0
- Source: Zenodo (https://zenodo.org/records/5785955)
- Subjects: 52 (26 male, 26 female, age 18-77)
- Activities: 24 types of ADL (Activities of Daily Living)
- Sensor: Empatica E4 accelerometer (worn on dominant hand)
- Sampling rate: 32Hz
- Data range: ±2g (8-bit resolution: 0.015g)
- File format: CSV (each file is one recording of one activity)
- File naming: {activity}_{user_id}_{repetition}.csv
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from .base import BasePreprocessor
from . import register_preprocessor
from .common import download_file, extract_archive, cleanup_temp_files, check_dataset_exists
from .utils import create_sliding_windows

logger = logging.getLogger(__name__)

# PAAL dataset download URL
PAAL_URL = "https://zenodo.org/api/records/5785955/files-archive"


@register_preprocessor('paal')
class PAALPreprocessor(BasePreprocessor):
    """PAAL Dataset Preprocessor"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialization

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        # Window parameters (from config)
        self.window_size = config.get('window_size', 150)  # 5 seconds @ 30Hz
        self.stride = config.get('stride', 90)  # v1.0 compatible: stride=90

        # Dataset-specific parameters
        self.num_activities = 24
        self.num_subjects = 52
        self.num_sensors = 1  # Phone (dominant hand)
        self.original_sampling_rate = 32  # Hz
        self.target_sampling_rate = 30  # Hz (unified sampling rate)
        self.scale_factor = 0.015  # Conversion factor from integer to G units

        # Activity name mapping (file name → label ID)
        # Aligned with ADLs.csv order (0-indexed, v1.0 compatible)
        self.activity_names = {
            'drink_water': 0,       # ADL 1
            'eat_meal': 1,          # ADL 2
            'open_a_bottle': 2,     # ADL 3 (open bottle)
            'open_a_box': 3,        # ADL 4
            'brush_teeth': 4,       # ADL 5
            'brush_hair': 5,        # ADL 6
            'take_off_a_jacket': 6, # ADL 7 (take off jacket)
            'put_on_a_jacket': 7,   # ADL 8 (put on jacket)
            'put_on_a_shoe': 8,     # ADL 9
            'take_off_a_shoe': 9,   # ADL 10
            'put_on_glasses': 10,   # ADL 11
            'take_off_glasses': 11, # ADL 12
            'sit_down': 12,         # ADL 13
            'stand_up': 13,         # ADL 14
            'writing': 14,          # ADL 15
            'phone_call': 15,       # ADL 16
            'type_on_a_keyboard': 16, # ADL 17
            'salute': 17,           # ADL 18
            'sneeze_cough': 18,     # ADL 19 (sneeze/cough)
            'blow_nose': 19,        # ADL 20
            'washing_hands': 20,    # ADL 21
            'dusting': 21,          # ADL 22
            'ironing': 22,          # ADL 23
            'washing_dishes': 23,   # ADL 24
        }

    def get_dataset_name(self) -> str:
        """Return dataset name"""
        return 'paal'

    def download_dataset(self) -> None:
        """
        Download and extract PAAL dataset

        Downloads the following files from Zenodo:
        - data.zip (accelerometer data)
        - users.csv (user information)
        - ADLs.csv (activity list)
        """
        logger.info("=" * 80)
        logger.info("Downloading PAAL dataset")
        logger.info("=" * 80)

        paal_raw_path = self.raw_data_path / self.dataset_name

        # Check if data already exists
        if check_dataset_exists(paal_raw_path, required_files=['*.csv']):
            logger.warning(f"PAAL data already exists at {paal_raw_path}")
            response = input("Do you want to re-download? (y/N): ")
            if response.lower() != 'y':
                logger.info("Skipping download")
                return

        try:
            # 1. Download
            logger.info("Step 1/2: Downloading archive from Zenodo")
            paal_raw_path.parent.mkdir(parents=True, exist_ok=True)
            zip_path = paal_raw_path.parent / 'paal_data.zip'
            download_file(PAAL_URL, zip_path, desc='Downloading PAAL')

            # 2. Extract and organize data
            logger.info("Step 2/2: Extracting and organizing data")
            extract_to = paal_raw_path.parent / 'paal_temp'
            extract_archive(zip_path, extract_to, desc='Extracting PAAL')
            self._organize_paal_data(extract_to, paal_raw_path)

            # Cleanup
            cleanup_temp_files(extract_to)
            if zip_path.exists():
                zip_path.unlink()

            logger.info("=" * 80)
            logger.info(f"SUCCESS: PAAL dataset downloaded to {paal_raw_path}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Failed to download PAAL dataset: {e}", exc_info=True)
            raise

    def _organize_paal_data(self, extracted_path: Path, target_path: Path) -> None:
        """
        Organize PAAL data into appropriate directory structure

        Extracted files:
        - data.zip (contains dataset/*.csv internally)
        - users.csv
        - ADLs.csv

        Args:
            extracted_path: Path to extracted data
            target_path: Path to save organized data (data/raw/paal)
        """
        import shutil
        import zipfile

        logger.info(f"Organizing PAAL data from {extracted_path} to {target_path}")

        # Create target directory
        target_path.mkdir(parents=True, exist_ok=True)

        # Extract data.zip
        data_zip = extracted_path / "data.zip"
        if data_zip.exists():
            logger.info("Extracting data.zip (accelerometer data)")
            with zipfile.ZipFile(data_zip, 'r') as zip_ref:
                zip_ref.extractall(extracted_path / "data_extracted")

            # Move dataset/*.csv to target
            dataset_dir = extracted_path / "data_extracted" / "dataset"
            if dataset_dir.exists():
                for csv_file in dataset_dir.glob("*.csv"):
                    # Skip hidden files starting with ._ (macOS)
                    if not csv_file.name.startswith('._'):
                        shutil.copy(csv_file, target_path / csv_file.name)
                logger.info(f"Copied {len(list(target_path.glob('*.csv')))} CSV files")
            else:
                raise FileNotFoundError(f"dataset directory not found in data.zip")

        # Copy users.csv and ADLs.csv (optional: for reference)
        for meta_file in ['users.csv', 'ADLs.csv']:
            src = extracted_path / meta_file
            if src.exists():
                shutil.copy(src, target_path / meta_file)
                logger.debug(f"Copied {meta_file}")

        logger.info(f"Data organized at: {target_path}")

    def clean_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Data cleaning

        For PAAL, load_raw_data() already converts data to appropriate format,
        so no additional cleaning is required. Just return as is.

        Args:
            data: Output from load_raw_data()

        Returns:
            Cleaned data
        """
        # Filter out invalid samples (empty or extremely short data)
        valid_data = []
        for sample in data:
            if len(sample['data']) >= self.window_size:
                valid_data.append(sample)
            else:
                logger.debug(
                    f"Skipping short sample: subject={sample['subject_id']}, "
                    f"activity={sample['activity']}, length={len(sample['data'])}"
                )

        logger.info(f"Cleaned data: {len(valid_data)}/{len(data)} samples valid")
        return valid_data

    def load_raw_data(self) -> List[Dict[str, Any]]:
        """
        Load raw data

        Returns:
            Data list. Each element is a dictionary with the following keys:
                - subject_id: Subject ID (1-52)
                - activity: Activity ID (0-23)
                - data: Sensor data (N, 3) - X, Y, Z axis acceleration
        """
        raw_data_dir = Path(self.raw_data_path) / self.get_dataset_name()

        if not raw_data_dir.exists():
            raise FileNotFoundError(
                f"PAAL raw data not found at {raw_data_dir}. "
                f"Please download from: https://zenodo.org/records/5785955"
            )

        logger.info(f"Loading PAAL data from {raw_data_dir}")

        # Load all data files
        data_files = sorted(raw_data_dir.glob("*.csv"))

        if len(data_files) == 0:
            raise FileNotFoundError(
                f"No CSV files found in {raw_data_dir}. "
                f"Expected format: {{activity}}_{{user_id}}_{{repetition}}.csv"
            )

        logger.info(f"Found {len(data_files)} data files")

        all_data = []

        for file_path in data_files:
            # Extract information from filename: {activity}_{user_id}_{repetition}.csv
            parts = file_path.stem.rsplit('_', 2)

            if len(parts) != 3:
                logger.warning(f"Skipping file with unexpected name format: {file_path.name}")
                continue

            activity_name = parts[0]
            user_id = int(parts[1])
            repetition = int(parts[2])

            # Convert activity name to label ID
            if activity_name not in self.activity_names:
                logger.warning(f"Unknown activity '{activity_name}' in file: {file_path.name}")
                continue

            activity_id = self.activity_names[activity_name]

            # Read CSV file (no header, 3 columns: x, y, z)
            try:
                sensor_data = pd.read_csv(
                    file_path,
                    header=None,
                    names=['x', 'y', 'z']
                ).values  # (N, 3)

                # Convert integer values to G units
                sensor_data = sensor_data * self.scale_factor

            except Exception as e:
                logger.error(f"Error reading file {file_path.name}: {e}")
                continue

            # Add data
            all_data.append({
                'subject_id': user_id,
                'activity': activity_id,
                'data': sensor_data,
                'repetition': repetition,
            })

        logger.info(
            f"Loaded {len(all_data)} samples from {len(set(d['subject_id'] for d in all_data))} subjects"
        )

        return all_data

    def extract_features(
        self,
        raw_data: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract features

        Steps:
        1. Resample each sample from 32Hz → 30Hz
        2. Split each sample into sliding windows
        3. Save per window

        Args:
            raw_data: Output from load_raw_data()

        Returns:
            Tuple of (features, labels, subjects)
                - features: (N, C, T) - N windows, C=3 channels, T=150 samples
                - labels: (N,) - Activity labels
                - subjects: (N,) - Subject IDs
        """
        window_size = self.window_size  # 150 samples @ 30Hz = 5 seconds
        stride = self.stride  # v1.0 compatible: stride=90

        all_features = []
        all_labels = []
        all_subjects = []

        logger.info(
            f"Extracting features with window_size={window_size}, stride={stride}"
        )

        for sample in raw_data:
            subject_id = sample['subject_id']
            activity = sample['activity']
            data = sample['data']  # (N, 3)

            # Resampling: 32Hz → 30Hz
            # Exact ratio: 30/32 = 15/16
            data_resampled = resample_poly(
                data,
                up=15,
                down=16,
                axis=0,
                padtype='line'
            )  # (N', 3)

            # Create label array since each sample has a single label
            num_samples = len(data_resampled)
            sample_labels = np.full(num_samples, activity, dtype=np.int64)

            # Skip if minimum window size is not met
            if num_samples < window_size:
                continue

            # Use create_sliding_windows (same mechanism as other datasets)
            windows, window_labels = create_sliding_windows(
                data_resampled,  # (N', 3)
                sample_labels,
                window_size=window_size,
                stride=stride
            )

            if len(windows) > 0:
                # Transpose (N, T, 3) → (N, 3, T)
                windows = windows.transpose(0, 2, 1)  # (N, 3, T)
                all_features.append(windows)
                all_labels.append(window_labels)
                all_subjects.append(np.full(len(windows), subject_id, dtype=np.int64))

        # Concatenate all data
        if len(all_features) == 0:
            features = np.array([], dtype=np.float32).reshape(0, 3, window_size)
            labels = np.array([], dtype=np.int64)
            subjects = np.array([], dtype=np.int64)
        else:
            features = np.concatenate(all_features, axis=0).astype(np.float32)
            labels = np.concatenate(all_labels, axis=0).astype(np.int64)
            subjects = np.concatenate(all_subjects, axis=0).astype(np.int64)

        logger.info(
            f"Extracted {len(features)} windows from {len(raw_data)} samples"
        )
        logger.info(f"Features shape: {features.shape}")
        logger.info(f"Labels shape: {labels.shape}")
        logger.info(f"Unique activities: {np.unique(labels)}")
        logger.info(f"Unique subjects: {np.unique(subjects)}")

        return features, labels, subjects

    def save_processed_data(self, data: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        """
        Save processed data

        Directory structure:
            processed_data_path/
                USER{subject_id:05d}/
                    Phone/
                        ACC/
                            X.npy  # (N, T, 3) - Sensor data (X, Y, Z axes)
                            Y.npy  # (N,) - Labels

        Args:
            data: Tuple of (features, labels, subjects)
                - features: (N, C, T) - C=3 (X, Y, Z)
                - labels: (N,)
                - subjects: (N,)
        """
        features, labels, subjects = data
        save_dir = Path(self.processed_data_path) / self.get_dataset_name()
        save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving processed data to {save_dir}")

        # Split and save data per subject
        unique_subjects = np.unique(subjects)
        users_dict = {}

        for subject_id in unique_subjects:
            # Get indices for this subject
            subject_mask = subjects == subject_id

            subject_features = features[subject_mask]  # (n, 3, T)
            subject_labels = labels[subject_mask]  # (n,)

            # Target directory
            subject_dir = save_dir / f"USER{subject_id:05d}" / "Phone" / "ACC"
            subject_dir.mkdir(parents=True, exist_ok=True)

            # X.npy: All 3 channels of sensor data (N, 3, T) → float16
            # Save in (N, 3, T) format to match other datasets
            X_data = subject_features.astype(np.float16)  # (N, 3, T) - no transpose
            np.save(subject_dir / "X.npy", X_data)

            # Y.npy: Labels (N,)
            np.save(subject_dir / "Y.npy", subject_labels)

            logger.debug(
                f"Saved {len(subject_features)} windows for USER{subject_id:05d}"
            )

            # Add to users dictionary (for visualization tools)
            user_id_str = f"USER{subject_id:05d}"
            users_dict[user_id_str] = {
                "sensor_modalities": {
                    "Phone/ACC": {
                        "X_shape": [len(subject_labels), 3, self.window_size],  # Fixed to (N, 3, T)
                        "Y_shape": [len(subject_labels)],
                        "num_windows": len(subject_labels),
                        "unique_labels": sorted([int(l) for l in np.unique(subject_labels)])
                    }
                }
            }

        # Save metadata
        import json
        metadata = {
            'dataset': self.get_dataset_name(),
            'num_subjects': len(unique_subjects),
            'num_windows': len(features),
            'num_activities': self.num_activities,
            'window_size': self.window_size,
            'stride': self.stride,
            'sampling_rate': self.target_sampling_rate,
            'original_sampling_rate': self.original_sampling_rate,
            'scale_factor': self.scale_factor,
            'sensor_names': ['Phone'],
            'sensor_positions': ['Phone'],
            'modalities': ['ACC'],
            'channels_per_modality': 3,
            'users': users_dict,
        }

        metadata_path = save_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata to {metadata_path}")
        logger.info(
            f"Successfully saved data for {len(unique_subjects)} subjects "
            f"({len(features)} windows total)"
        )
