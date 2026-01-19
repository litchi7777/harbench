"""
PAMAP2 (Physical Activity Monitoring) Dataset Preprocessing

PAMAP2 Dataset:
- 12 types of main physical activities (protocol sessions)
- 9 subjects
- 3 IMU sensors (hand, chest, ankle) + heart rate monitor
- Sampling rate: IMU 100Hz, HR ~9Hz
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


# PAMAP2 dataset URL
PAMAP2_URL = "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"


@register_preprocessor('pamap2')
class PAMAP2Preprocessor(BasePreprocessor):
    """
    Preprocessing class for PAMAP2 dataset
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # PAMAP2-specific settings
        self.num_activities = 12
        self.num_subjects = 9
        self.num_sensors = 3

        # Sensor names and channel mapping
        # Data file structure (54 columns):
        # 0: timestamp
        # 1: activity_id
        # 2: heart_rate
        # 3-19: IMU hand (17 channels)
        # 20-36: IMU chest (17 channels)
        # 37-53: IMU ankle (17 channels)

        # Each IMU has 17 channels:
        # 0: temperature
        # 1-3: ACC 16g (3-axis)
        # 4-6: ACC 6g (3-axis)
        # 7-9: gyroscope (3-axis)
        # 10-12: magnetometer (3-axis)
        # 13-16: orientation (invalid, 4 values)

        self.sensor_names = ['hand', 'chest', 'ankle']
        self.sensor_column_ranges = {
            'hand': (3, 20),      # columns 3-19
            'chest': (20, 37),    # columns 20-36
            'ankle': (37, 54)     # columns 37-53
        }

        # Modalities (channel division within each sensor)
        # Using ACC 6g (higher precision)
        self.sensor_modalities = {
            'hand': {
                'ACC': (4, 7),    # ACC 6g: 3-axis acceleration
                'GYRO': (7, 10),  # 3-axis gyroscope
                'MAG': (10, 13)   # 3-axis magnetometer
            },
            'chest': {
                'ACC': (4, 7),
                'GYRO': (7, 10),
                'MAG': (10, 13)
            },
            'ankle': {
                'ACC': (4, 7),
                'GYRO': (7, 10),
                'MAG': (10, 13)
            }
        }

        # Sampling rate
        self.original_sampling_rate = 100  # Hz (PAMAP2 original)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz (target)

        # Preprocessing parameters
        self.window_size = config.get('window_size', 150)  # 5 seconds @ 30Hz
        self.stride = config.get('stride', 30)  # 1 second @ 30Hz

        # Scale factor (m/s^2 -> G conversion)
        self.scale_factor = DATASETS.get('PAMAP2', {}).get('scale_factor', None)

        # Activity ID mapping
        # Main 12 activities included in PAMAP2 protocol sessions
        # Original ID -> 0-indexed label (0 is converted to -1)
        self.activity_id_mapping = {
            0: -1,   # other (transient activities)
            1: 0,    # lying
            2: 1,    # sitting
            3: 2,    # standing
            4: 3,    # walking
            5: 4,    # running
            6: 5,    # cycling
            7: 6,    # Nordic walking
            12: 7,   # ascending stairs
            13: 8,   # descending stairs
            16: 9,   # vacuum cleaning
            17: 10,  # ironing
            24: 11   # rope jumping
        }

    def get_dataset_name(self) -> str:
        return 'pamap2'

    def download_dataset(self) -> None:
        """
        Download and extract PAMAP2 dataset
        """
        logger.info("=" * 80)
        logger.info("Downloading PAMAP2 dataset")
        logger.info("=" * 80)

        pamap2_raw_path = self.raw_data_path / self.dataset_name

        # Check if data already exists
        if check_dataset_exists(pamap2_raw_path, required_files=['subject10*.dat']):
            logger.warning(f"PAMAP2 data already exists at {pamap2_raw_path}")
            response = input("Do you want to re-download? (y/N): ")
            if response.lower() != 'y':
                logger.info("Skipping download")
                return

        try:
            # 1. Download
            logger.info("Step 1/2: Downloading archive")
            pamap2_raw_path.parent.mkdir(parents=True, exist_ok=True)
            zip_path = pamap2_raw_path.parent / 'pamap2.zip'
            download_file(PAMAP2_URL, zip_path, desc='Downloading PAMAP2')

            # 2. Extract and organize data
            logger.info("Step 2/2: Extracting and organizing data")
            extract_to = pamap2_raw_path.parent / 'pamap2_temp'
            extract_archive(zip_path, extract_to, desc='Extracting PAMAP2')
            self._organize_pamap2_data(extract_to, pamap2_raw_path)

            # Cleanup
            cleanup_temp_files(extract_to)
            if zip_path.exists():
                zip_path.unlink()

            logger.info("=" * 80)
            logger.info(f"SUCCESS: PAMAP2 dataset downloaded to {pamap2_raw_path}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Failed to download PAMAP2 dataset: {e}", exc_info=True)
            raise

    def _organize_pamap2_data(self, extracted_path: Path, target_path: Path) -> None:
        """
        Organize PAMAP2 data into proper directory structure

        Args:
            extracted_path: Path to extracted data
            target_path: Path to save organized data (data/raw/pamap2)
        """
        logger.info(f"Organizing PAMAP2 data from {extracted_path} to {target_path}")

        # Check and extract nested ZIP file
        nested_zip = extracted_path / "PAMAP2_Dataset.zip"
        if nested_zip.exists():
            logger.info(f"Found nested ZIP file: {nested_zip}")
            logger.info("Extracting nested archive...")
            extract_archive(nested_zip, extracted_path, desc='Extracting nested PAMAP2')
            # Delete nested ZIP file
            nested_zip.unlink()

        # Find root of extracted data
        # PAMAP2 data is split into "PAMAP2_Dataset/Protocol" and "PAMAP2_Dataset/Optional"
        possible_roots = [
            extracted_path / "PAMAP2_Dataset",
            extracted_path / "Protocol",
            extracted_path
        ]

        data_root = None
        for root in possible_roots:
            if root.exists():
                # Find Protocol folder
                protocol_dir = root / "Protocol" if (root / "Protocol").exists() else root
                # Find subject*.dat files
                dat_files = list(protocol_dir.glob("subject10*.dat"))
                if dat_files:
                    data_root = protocol_dir
                    break

        if data_root is None:
            # Search more extensively
            dat_files = list(extracted_path.rglob("subject10*.dat"))
            if dat_files:
                data_root = dat_files[0].parent
            else:
                raise FileNotFoundError(f"Could not find subject*.dat files in {extracted_path}")

        # Create target directory
        target_path.mkdir(parents=True, exist_ok=True)

        # Find and copy .dat files (Protocol sessions only)
        dat_files = list(data_root.glob("subject10*.dat"))

        if not dat_files:
            raise FileNotFoundError(f"Could not find subject10*.dat files in {data_root}")

        from tqdm import tqdm
        import shutil

        for dat_file in tqdm(dat_files, desc='Organizing files'):
            target_file = target_path / dat_file.name
            shutil.copy2(dat_file, target_file)

        logger.info(f"Data organized at: {target_path}")
        logger.info(f"Found {len(dat_files)} subject files")

    def load_raw_data(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Load PAMAP2 raw data for each subject

        Expected format:
        - data/raw/pamap2/subject101.dat
        - Each file: (samples, 54) space-separated text file
          - 1 column: timestamp
          - 1 column: activity_id
          - 1 column: heart_rate
          - 51 columns: IMU data (3 sensors × 17 channels)

        Returns:
            person_data: dict of {person_id: (data, labels)}
                data: array of (num_samples, 51) (3 IMU sensors)
                labels: array of (num_samples,)
        """
        raw_path = self.raw_data_path / self.dataset_name

        if not raw_path.exists():
            raise FileNotFoundError(
                f"PAMAP2 raw data not found at {raw_path}\n"
                "Expected structure: data/raw/pamap2/subject101.dat\n"
                "Run with --download flag to download the dataset."
            )

        # Store data for each subject
        result = {}

        # Subject IDs are 101-109
        subject_ids = [101, 102, 103, 104, 105, 106, 107, 108, 109]

        for idx, subject_id in enumerate(subject_ids, start=1):
            subject_file = raw_path / f"subject{subject_id}.dat"

            if not subject_file.exists():
                logger.warning(f"Subject file not found: {subject_file}")
                continue

            try:
                # Load data (space-separated)
                data = np.loadtxt(subject_file)

                # Check data shape
                if data.shape[1] != 54:
                    logger.warning(
                        f"Unexpected number of columns in {subject_file}: "
                        f"{data.shape[1]} (expected 54)"
                    )
                    continue

                # Get IMU data only, excluding timestamp, activity_id, heart_rate
                # columns 3-53 (51 channels)
                sensor_data = data[:, 3:54]  # 51 columns (3 sensors × 17 channels)
                labels = data[:, 1].astype(int)  # activity_id

                # Map labels
                mapped_labels = np.array([
                    self.activity_id_mapping.get(label, -1) for label in labels
                ])

                result[idx] = (sensor_data, mapped_labels)
                logger.info(f"USER{idx:05d} (subject{subject_id}): {sensor_data.shape}, Labels: {mapped_labels.shape}")

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
            data: dict of {person_id: (data, labels)}

        Returns:
            Cleaned and resampled {person_id: (data, labels)}
        """
        cleaned = {}
        for person_id, (person_data, labels) in data.items():
            # Remove invalid samples (rows containing NaN)
            cleaned_data, cleaned_labels = filter_invalid_samples(person_data, labels)

            # Resampling (100Hz -> 30Hz)
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
        Feature extraction (windowing and normalization per sensor × modality)

        Args:
            data: dict of {person_id: (data, labels)}

        Returns:
            {person_id: {sensor/modality: {'X': data, 'Y': labels}}}
            Example: {'hand/ACC': {'X': (N, 3, 150), 'Y': (N,)}}
        """
        processed = {}

        for person_id, (person_data, labels) in data.items():
            logger.info(f"Processing USER{person_id:05d}")

            processed[person_id] = {}

            # Process each sensor
            for sensor_name in self.sensor_names:
                sensor_start_col, sensor_end_col = self.sensor_column_ranges[sensor_name]

                # Extract columns for this sensor from entire IMU data
                # person_data is (samples, 51), starting from column 3, so adjust
                sensor_start_idx = sensor_start_col - 3
                sensor_end_idx = sensor_end_col - 3

                # Extract 17 channels of this sensor
                sensor_data = person_data[:, sensor_start_idx:sensor_end_idx]

                # Apply sliding windows (pad last window if needed)
                # No normalization, keep raw sensor data
                windowed_data, windowed_labels = create_sliding_windows(
                    sensor_data,
                    labels,
                    window_size=self.window_size,
                    stride=self.stride,
                    drop_last=False,
                    pad_last=True  # PAMAP2: pad if less than 150
                )
                # windowed_data: (num_windows, window_size, sensor_channels)

                # Split into each modality
                modalities = self.sensor_modalities[sensor_name]
                for modality_name, (mod_start_ch, mod_end_ch) in modalities.items():
                    # Extract channels for this modality
                    modality_data = windowed_data[:, :, mod_start_ch:mod_end_ch]
                    # modality_data: (num_windows, window_size, channels)

                    # Apply scaling (accelerometer only)
                    if modality_name == 'ACC' and self.scale_factor is not None:
                        modality_data = modality_data / self.scale_factor
                        logger.info(f"  Applied scale_factor={self.scale_factor} to {sensor_name}/{modality_name}")

                    # Transform shape: (num_windows, window_size, C) -> (num_windows, C, window_size)
                    modality_data = np.transpose(modality_data, (0, 2, 1))

                    # Convert to float16
                    modality_data = modality_data.astype(np.float16)

                    # Sensor/modality hierarchy
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
            data/processed/pamap2/USER00001/hand/ACC/X.npy, Y.npy
            data/processed/pamap2/USER00001/hand/GYRO/X.npy, Y.npy
            data/processed/pamap2/USER00001/hand/MAG/X.npy, Y.npy
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
            'normalization': 'none',  # No normalization (keep raw data)
            'scale_factor': self.scale_factor,  # Scale factor (applied to ACC only)
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
