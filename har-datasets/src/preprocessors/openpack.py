"""
OpenPack Dataset Preprocessing

OpenPack Dataset:
- **Activities**: 9 classes (0-8 + undefined/-1) of logistics operation tasks
  - 0: Assemble, 1: Insert, 2: Put, 3: Walk, 4: Pick,
  - 5: Scan, 6: Press, 7: Open, 8: Close
  - -1: Undefined (no operation, unlabeled, other)
  - Note: Raw data operation=10 (other/unknown, ~0.7%) is excluded as undefined class
- **Subjects**: Variable depending on dataset (typically U0101-U0110, etc.)
- **Sensors**: 4 ATR TSND151 IMU sensors (atr01, atr02, atr03, atr04)
  - Each sensor: ACC (3-axis), GYRO (3-axis), QUAT (4 values) = 10 channels
  - Total: 40 channels (4 sensors × 10 channels)
- **Sampling Rate**: 30Hz
- **Units**:
  - ACC: G (gravitational acceleration) - already normalized
  - GYRO: dps (degrees per second)
  - QUAT: dimensionless (quaternion)

Reference: https://open-pack.github.io/
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
    get_class_distribution
)
from .common import (
    download_file,
    extract_archive,
    cleanup_temp_files,
    check_dataset_exists
)
from . import register_preprocessor

logger = logging.getLogger(__name__)


# OpenPack dataset URL
OPENPACK_URL = "https://zenodo.org/records/8145223/files/preprocessed-IMU-with-operation-labels.zip?download=1"


@register_preprocessor('openpack')
class OpenPackPreprocessor(BasePreprocessor):
    """
    Preprocessor class for OpenPack dataset
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # OpenPack-specific settings
        self.num_activities = 9  # Operation classes (0-8) + undefined (-1) = effectively 10 classes
        self.num_subjects = None  # Variable depending on dataset (detected dynamically)
        self.num_sensors = 4  # ATR TSND151 IMU × 4
        self.num_channels = 40  # 4 sensors × 10 channels (acc:3 + gyro:3 + quat:4)

        # Sensor names and channel mapping
        # atr01=RightWrist, atr02=LeftWrist, atr03=RightUpperArm, atr04=LeftUpperArm
        self.sensor_names = ['RightWrist', 'LeftWrist', 'RightUpperArm', 'LeftUpperArm']
        # Mapping from original ATR sensor names
        self._atr_to_sensor = {
            'atr01': 'RightWrist',
            'atr02': 'LeftWrist',
            'atr03': 'RightUpperArm',
            'atr04': 'LeftUpperArm'
        }

        # Number of channels per sensor (ACC:3 + GYRO:3 + QUAT:4)
        self.channels_per_sensor = 10

        # Sensor channel ranges (position within overall channels)
        # Will be set dynamically during data loading
        self.sensor_channel_ranges = {}

        # Modalities (channel divisions within each sensor)
        self.modalities = ['ACC', 'GYRO', 'QUAT']
        self.modality_channel_ranges = {
            'ACC': (0, 3),    # 3-axis acceleration
            'GYRO': (3, 6),   # 3-axis gyroscope
            'QUAT': (6, 10)   # 4-value quaternion
        }

        # Sampling rate
        self.original_sampling_rate = 30  # Hz (OpenPack original)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz (target)

        # Preprocessing parameters
        self.window_size = config.get('window_size', 150)  # 5 seconds @ 30Hz
        self.stride = config.get('stride', 30)  # 1 second @ 30Hz

    def get_dataset_name(self) -> str:
        return 'openpack'

    def download_dataset(self) -> None:
        """
        Download and extract OpenPack dataset
        """
        logger.info("=" * 80)
        logger.info("Downloading OpenPack dataset")
        logger.info("=" * 80)

        openpack_raw_path = self.raw_data_path / self.dataset_name

        # Check if data already exists
        if check_dataset_exists(openpack_raw_path, required_files=['*.csv']):
            logger.warning(f"OpenPack data already exists at {openpack_raw_path}")
            response = input("Do you want to re-download? (y/N): ")
            if response.lower() != 'y':
                logger.info("Skipping download")
                return

        try:
            # 1. Download
            logger.info("Step 1/2: Downloading archive (491 MB)")
            openpack_raw_path.parent.mkdir(parents=True, exist_ok=True)
            zip_path = openpack_raw_path.parent / 'openpack.zip'
            download_file(OPENPACK_URL, zip_path, desc='Downloading OpenPack')

            # 2. Extract and organize data
            logger.info("Step 2/2: Extracting and organizing data")
            extract_to = openpack_raw_path.parent / 'openpack_temp'
            extract_archive(zip_path, extract_to, desc='Extracting OpenPack')
            self._organize_openpack_data(extract_to, openpack_raw_path)

            # Cleanup
            cleanup_temp_files(extract_to)
            if zip_path.exists():
                zip_path.unlink()

            logger.info("=" * 80)
            logger.info(f"SUCCESS: OpenPack dataset downloaded to {openpack_raw_path}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Failed to download OpenPack dataset: {e}", exc_info=True)
            raise

    def _organize_openpack_data(self, extracted_path: Path, target_path: Path) -> None:
        """
        Organize OpenPack data into appropriate directory structure

        Args:
            extracted_path: Path to extracted data
            target_path: Target path for organized data (data/raw/openpack)
        """
        import shutil
        from tqdm import tqdm

        logger.info(f"Organizing OpenPack data from {extracted_path} to {target_path}")

        # Find the root of extracted data
        data_root = extracted_path / "imuWithOperationLabel"

        if not data_root.exists():
            # Try alternative structure
            possible_roots = list(extracted_path.rglob("imuWithOperationLabel"))
            if possible_roots:
                data_root = possible_roots[0]
            else:
                raise FileNotFoundError(f"Could not find imuWithOperationLabel directory in {extracted_path}")

        # Create target directory
        target_path.mkdir(parents=True, exist_ok=True)

        # Copy CSV files
        csv_files = list(data_root.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"Could not find CSV files in {data_root}")

        for csv_file in tqdm(csv_files, desc='Organizing files'):
            target_file = target_path / csv_file.name
            shutil.copy2(csv_file, target_file)

        logger.info(f"Data organized at: {target_path}")
        logger.info(f"Found {len(csv_files)} CSV files")

    def load_raw_data(self) -> Dict[str, list]:
        """
        Load OpenPack raw data per subject

        Expected format:
        - data/raw/openpack/U0101-S0100.csv
        - Each file: unixtim, operation, atr01/acc_x, ..., atr04/quat_z

        Returns:
            person_data: Dictionary of {person_id: [(data, labels), ...]}
                Each session (CSV file) is stored separately
                data: array of shape (num_samples, 40)
                labels: array of shape (num_samples,)
        """
        raw_path = self.raw_data_path / self.dataset_name

        if not raw_path.exists():
            raise FileNotFoundError(
                f"OpenPack raw data not found at {raw_path}\n"
                "Expected structure: data/raw/openpack/U0101-S0100.csv"
            )

        # Get CSV files
        csv_files = sorted(raw_path.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {raw_path}")

        # Store session list per subject
        from collections import defaultdict
        person_sessions = defaultdict(list)

        for csv_file in csv_files:
            # Extract subject ID from filename (U0101-S0100.csv -> U0101)
            user_id_str = csv_file.stem.split('-')[0]  # "U0101"

            try:
                # U0101 -> 101 -> USER00001
                # OpenPack user IDs are in the range 101-111, 201-210
                # 101-111 -> USER00001-USER00011 (11 people)
                # 201-210 -> USER00012-USER00021 (10 people)
                original_user_id = int(user_id_str[1:])

                if 101 <= original_user_id <= 111:
                    user_id = f"USER{(original_user_id - 100):05d}"  # 101 -> USER00001, 111 -> USER00011
                elif 201 <= original_user_id <= 210:
                    user_id = f"USER{(original_user_id - 189):05d}"  # 201 -> USER00012, 210 -> USER00021
                else:
                    logger.warning(f"Unexpected user ID: {user_id_str}, skipping")
                    continue

                # Load data
                df = pd.read_csv(csv_file)

                # Get sensor data columns (excluding unixtime, operation)
                sensor_columns = [col for col in df.columns if '/' in col]

                # Separate sensor data and labels
                sensor_data = df[sensor_columns].values
                labels = df['operation'].values.astype(int)

                # Label conversion:
                # - operation=0 (no operation) -> label=-1 (undefined class)
                # - operation=1-9 -> label=0-8 (valid 9 classes)
                # - operation=10 (other/unknown) -> label=-1 (excluded as undefined class)
                # Note: Data contains operation=0-10, but operation=10 is rare (~0.7%) and
                #       definition is unclear, so treated as undefined class
                labels = np.where((labels == 0) | (labels == 10), -1, labels - 1)

                # Add as session (don't concatenate)
                person_sessions[user_id].append((sensor_data, labels))

                logger.debug(f"Loaded {csv_file.name}: {sensor_data.shape}")

            except Exception as e:
                logger.error(f"Error loading {csv_file}: {e}")
                continue

        # Return session list
        result = {}
        for user_id, sessions in person_sessions.items():
            if sessions:
                result[user_id] = sessions
                total_samples = sum(d.shape[0] for d, l in sessions)
                logger.info(f"{user_id}: {len(sessions)} sessions, {total_samples} samples")

        if not result:
            raise ValueError("No data loaded. Please check the raw data directory structure.")

        logger.info(f"Total users loaded: {len(result)}")
        return result

    def clean_data(self, data: Dict[str, list]) -> Dict[str, list]:
        """
        Data cleaning (processed per session)

        Args:
            data: Dictionary of {person_id: [(data, labels), ...]}

        Returns:
            Cleaned {person_id: [(data, labels), ...]}
        """
        cleaned = {}
        for person_id, sessions in data.items():
            cleaned_sessions = []
            for session_data, session_labels in sessions:
                # Remove invalid samples
                cleaned_data, cleaned_labels = filter_invalid_samples(session_data, session_labels)
                if len(cleaned_data) > 0:
                    cleaned_sessions.append((cleaned_data, cleaned_labels))

            if cleaned_sessions:
                cleaned[person_id] = cleaned_sessions
                total_samples = sum(d.shape[0] for d, l in cleaned_sessions)
                logger.info(f"{person_id} cleaned: {len(cleaned_sessions)} sessions, {total_samples} samples")

        return cleaned

    def extract_features(self, data: Dict[str, list]) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        """
        Feature extraction (windowing per sensor × modality)
        Windows do not span across session boundaries

        Args:
            data: Dictionary of {person_id: [(data, labels), ...]}

        Returns:
            {person_id: {sensor/modality: {'X': data, 'Y': labels}}}
            Example: {'RightWrist/ACC': {'X': (N, 3, 150), 'Y': (N,)}}
        """
        processed = {}

        for person_id, sessions in data.items():
            logger.info(f"Processing {person_id} ({len(sessions)} sessions)")

            processed[person_id] = {}

            # Process each sensor
            for sensor_idx, sensor_name in enumerate(self.sensor_names):
                # Calculate sensor channel range
                sensor_start_ch = sensor_idx * self.channels_per_sensor
                sensor_end_ch = sensor_start_ch + self.channels_per_sensor

                # Extract sensor channels per session
                sensor_sessions = [
                    (d[:, sensor_start_ch:sensor_end_ch], l)
                    for d, l in sessions
                ]

                # Apply sliding windows per session
                windowed_data, windowed_labels = create_sliding_windows_multi_session(
                    sensor_sessions,
                    window_size=self.window_size,
                    stride=self.stride,
                    drop_last=False,
                    pad_last=True  # OpenPack: pad if less than 150 samples
                )

                if len(windowed_data) == 0:
                    logger.warning(f"  {sensor_name}: no valid windows")
                    continue

                # windowed_data: (num_windows, window_size, 10)

                # Split into each modality
                for modality_name in self.modalities:
                    mod_start_ch, mod_end_ch = self.modality_channel_ranges[modality_name]

                    # Extract modality channels
                    modality_data = windowed_data[:, :, mod_start_ch:mod_end_ch]
                    # modality_data: (num_windows, window_size, C)

                    # Transform shape: (num_windows, window_size, C) -> (num_windows, C, window_size)
                    modality_data = np.transpose(modality_data, (0, 2, 1))

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
            data/processed/openpack/USER00101/atr01/ACC/X.npy, Y.npy
            data/processed/openpack/USER00101/atr01/GYRO/X.npy, Y.npy
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
            'normalization': 'none',  # No normalization (preserve raw data)
            'users': {}
        }

        for person_id, sensor_modality_data in data.items():
            # person_id is already in "USER00001" format
            user_path = base_path / person_id
            user_path.mkdir(parents=True, exist_ok=True)

            user_stats = {'sensor_modalities': {}}

            for sensor_modality_name, arrays in sensor_modality_data.items():
                sensor_modality_path = user_path / sensor_modality_name
                sensor_modality_path.mkdir(parents=True, exist_ok=True)

                # Save X.npy, Y.npy (use float16 for efficiency)
                X = arrays['X'].astype(np.float16)  # (num_windows, C, window_size)
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
                    f"Saved {person_id}/{sensor_modality_name}: "
                    f"X{X.shape}, Y{Y.shape}"
                )

            total_stats['users'][person_id] = user_stats

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
