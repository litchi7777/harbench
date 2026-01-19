"""
MHEALTH (Mobile Health) Dataset Preprocessing

MHEALTH Dataset:
- 12 types of physical activities
- 10 subjects
- 3 sensors (chest, left ankle, right wrist)
- Sampling rate: 50Hz
"""

import numpy as np
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


# MHEALTH dataset URL
MHEALTH_URL = "https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip"


@register_preprocessor('mhealth')
class MHEALTHPreprocessor(BasePreprocessor):
    """
    Preprocessing class for MHEALTH dataset
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # MHEALTH-specific settings
        self.num_activities = 12
        self.num_subjects = 10
        self.num_sensors = 3
        self.num_channels = 23  # Chest 5 + Ankle 9 + Wrist 9

        # Sensor names and channel mapping
        # Channel configuration:
        # Chest: ACC(3) + ECG(2) = 5
        # LeftAnkle: ACC(3) + GYRO(3) + MAG(3) = 9
        # RightWrist: ACC(3) + GYRO(3) + MAG(3) = 9
        self.sensor_names = ['Chest', 'LeftAnkle', 'RightWrist']
        self.sensor_channel_ranges = {
            'Chest': (0, 5),         # channels 0-4
            'LeftAnkle': (5, 14),    # channels 5-13
            'RightWrist': (14, 23)   # channels 14-22
        }

        # Modalities (channel split within each sensor)
        self.sensor_modalities = {
            'Chest': {
                'ACC': (0, 3),   # 3-axis acceleration
                'ECG': (3, 5)    # 2-channel ECG
            },
            'LeftAnkle': {
                'ACC': (0, 3),   # 3-axis acceleration
                'GYRO': (3, 6),  # 3-axis gyroscope
                'MAG': (6, 9)    # 3-axis magnetometer
            },
            'RightWrist': {
                'ACC': (0, 3),   # 3-axis acceleration
                'GYRO': (3, 6),  # 3-axis gyroscope
                'MAG': (6, 9)    # 3-axis magnetometer
            }
        }

        # Sampling rate
        self.original_sampling_rate = 50  # Hz (MHEALTH original)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz (target)

        # Preprocessing parameters
        self.window_size = config.get('window_size', 150)  # 5 seconds @ 30Hz
        self.stride = config.get('stride', 30)  # 1 second @ 30Hz

        # Scaling factor (convert m/s^2 -> G)
        self.scale_factor = DATASETS.get('MHEALTH', {}).get('scale_factor', None)

    def get_dataset_name(self) -> str:
        return 'mhealth'

    def download_dataset(self) -> None:
        """
        Download and extract MHEALTH dataset
        """
        logger.info("=" * 80)
        logger.info("Downloading MHEALTH dataset")
        logger.info("=" * 80)

        mhealth_raw_path = self.raw_data_path / self.dataset_name

        # Check if data already exists
        if check_dataset_exists(mhealth_raw_path, required_files=['mHealth_subject*.log']):
            logger.warning(f"MHEALTH data already exists at {mhealth_raw_path}")
            response = input("Do you want to re-download? (y/N): ")
            if response.lower() != 'y':
                logger.info("Skipping download")
                return

        try:
            # 1. Download
            logger.info("Step 1/2: Downloading archive")
            mhealth_raw_path.parent.mkdir(parents=True, exist_ok=True)
            zip_path = mhealth_raw_path.parent / 'mhealth.zip'
            download_file(MHEALTH_URL, zip_path, desc='Downloading MHEALTH')

            # 2. Extract and organize data
            logger.info("Step 2/2: Extracting and organizing data")
            extract_to = mhealth_raw_path.parent / 'mhealth_temp'
            extract_archive(zip_path, extract_to, desc='Extracting MHEALTH')
            self._organize_mhealth_data(extract_to, mhealth_raw_path)

            # Cleanup
            cleanup_temp_files(extract_to)
            if zip_path.exists():
                zip_path.unlink()

            logger.info("=" * 80)
            logger.info(f"SUCCESS: MHEALTH dataset downloaded to {mhealth_raw_path}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Failed to download MHEALTH dataset: {e}", exc_info=True)
            raise

    def _organize_mhealth_data(self, extracted_path: Path, target_path: Path) -> None:
        """
        Organize MHEALTH data into proper directory structure

        Args:
            extracted_path: Path to extracted data
            target_path: Target path for organized data (data/raw/mhealth)
        """
        logger.info(f"Organizing MHEALTH data from {extracted_path} to {target_path}")

        # Find the root of extracted data
        data_root = extracted_path

        # Look for "MHEALTH" folder
        possible_roots = [
            extracted_path / "MHEALTH",
            extracted_path / "mHealth",
            extracted_path / "mhealth_dataset",
            extracted_path
        ]

        for root in possible_roots:
            if root.exists():
                # Look for mHealth_subject*.log files
                log_files = list(root.glob("mHealth_subject*.log"))
                if log_files:
                    data_root = root
                    break

        # Create target directory
        target_path.mkdir(parents=True, exist_ok=True)

        # Find and copy .log files
        log_files = list(data_root.rglob("mHealth_subject*.log"))

        if not log_files:
            raise FileNotFoundError(f"Could not find mHealth_subject*.log files in {extracted_path}")

        from tqdm import tqdm
        import shutil

        for log_file in tqdm(log_files, desc='Organizing files'):
            target_file = target_path / log_file.name
            shutil.copy2(log_file, target_file)

        logger.info(f"Data organized at: {target_path}")
        logger.info(f"Found {len(log_files)} subject files")

    def load_raw_data(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Load MHEALTH raw data per subject

        Expected format:
        - data/raw/mhealth/mHealth_subject1.log
        - Each file: (samples, 24) text file
          - 23 channels + 1 label

        Returns:
            person_data: {person_id: (data, labels)} dictionary
                data: (num_samples, 23) array
                labels: (num_samples,) array
        """
        raw_path = self.raw_data_path / self.dataset_name

        if not raw_path.exists():
            raise FileNotFoundError(
                f"MHEALTH raw data not found at {raw_path}\n"
                "Expected structure: data/raw/mhealth/mHealth_subject1.log"
            )

        # Store data per subject
        result = {}

        # For each subject
        for subject_id in range(1, self.num_subjects + 1):
            subject_file = raw_path / f"mHealth_subject{subject_id}.log"

            if not subject_file.exists():
                logger.warning(f"Subject file not found: {subject_file}")
                continue

            try:
                # Load data (tab-delimited)
                data = np.loadtxt(subject_file, delimiter='\t')

                # Check data shape
                if data.shape[1] != self.num_channels + 1:  # 23 + 1 label
                    logger.warning(
                        f"Unexpected number of columns in {subject_file}: "
                        f"{data.shape[1]} (expected {self.num_channels + 1})"
                    )
                    continue

                # Separate sensor data and labels
                sensor_data = data[:, :-1]  # First 23 columns
                labels = data[:, -1].astype(int)  # Last column

                # Convert label 0 (no activity) to -1, others to 0-indexed
                # 0 -> -1 (undefined class)
                # 1 -> 0, 2 -> 1, ... (valid classes)
                labels = np.where(labels == 0, -1, labels - 1)

                result[subject_id] = (sensor_data, labels)
                logger.info(f"USER{subject_id:05d}: {sensor_data.shape}, Labels: {labels.shape}")

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
        Feature extraction (windowing and normalization per sensor x modality)

        Args:
            data: {person_id: (data, labels)} dictionary

        Returns:
            {person_id: {sensor/modality: {'X': data, 'Y': labels}}}
            Example: {'Chest/ACC': {'X': (N, 3, 150), 'Y': (N,)}}
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

                # Apply sliding windows (pad last window)
                # Keep raw sensor data without normalization
                windowed_data, windowed_labels = create_sliding_windows(
                    sensor_data,
                    labels,
                    window_size=self.window_size,
                    stride=self.stride,
                    drop_last=False,
                    pad_last=True  # MHEALTH: pad if window size < 150
                )
                # windowed_data: (num_windows, window_size, sensor_channels)

                # Split into modalities
                modalities = self.sensor_modalities[sensor_name]
                for modality_name, (mod_start_ch, mod_end_ch) in modalities.items():
                    # Extract modality channels
                    modality_data = windowed_data[:, :, mod_start_ch:mod_end_ch]
                    # modality_data: (num_windows, window_size, channels)

                    # Apply scaling (acceleration only)
                    if modality_name == 'ACC' and self.scale_factor is not None:
                        modality_data = modality_data / self.scale_factor
                        logger.info(f"  Applied scale_factor={self.scale_factor} to {sensor_name}/{modality_name}")

                    # Reshape: (num_windows, window_size, C) -> (num_windows, C, window_size)
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
            data/processed/mhealth/USER00001/Chest/ACC/X.npy, Y.npy
            data/processed/mhealth/USER00001/Chest/ECG/X.npy, Y.npy
            data/processed/mhealth/USER00001/LeftAnkle/ACC/X.npy, Y.npy
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
            'normalization': 'none',  # No normalization (raw data retained)
            'scale_factor': self.scale_factor,  # Scaling factor (applied to ACC only)
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
