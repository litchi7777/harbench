"""
WARD (Wearable Action Recognition Database) Preprocessing

WARD Dataset:
- 13 types of daily activities
- 20 subjects
- 5 sensors (left arm, right arm, waist, left ankle, right ankle)
- Sampling rate: 20Hz
- Each sensor: 3-axis accelerometer + 2-axis gyroscope
"""

import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import Dict, Any, Tuple
import logging

from .base import BasePreprocessor
from .utils import (
    create_sliding_windows,
    create_sliding_windows_multi_session,
    filter_invalid_samples,
    get_class_distribution,
    resample_timeseries
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


# WARD dataset URL
WARD_URL = "https://people.eecs.berkeley.edu/~yang/software/WAR/WARD1.zip"


@register_preprocessor('ward')
class WARDPreprocessor(BasePreprocessor):
    """
    Preprocessor class for WARD dataset
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # WARD-specific settings
        self.num_activities = 13
        self.num_subjects = 20
        self.num_sensors = 5

        # Sampling rate
        self.original_sampling_rate = 20  # Hz (WARD original)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz (target)

        # Sensor name and index mapping
        # From PDF p.3: Sensor1=LeftArm, Sensor2=RightArm, Sensor3=Waist, Sensor4=LeftAnkle, Sensor5=RightAnkle
        self.sensor_names = ['LeftArm', 'RightArm', 'Waist', 'LeftAnkle', 'RightAnkle']
        self.sensor_indices = {
            'LeftArm': 0,      # Sensor 1
            'RightArm': 1,     # Sensor 2
            'Waist': 2,        # Sensor 3
            'LeftAnkle': 3,    # Sensor 4
            'RightAnkle': 4    # Sensor 5
        }

        # Modalities (channel configuration for each sensor)
        # Each sensor: 5 channels [ACC_X, ACC_Y, ACC_Z, GYRO_X, GYRO_Y]
        self.modalities = ['ACC', 'GYRO']
        self.modality_channel_ranges = {
            'ACC': (0, 3),   # 3-axis accelerometer
            'GYRO': (3, 5),  # 2-axis gyroscope
        }

        # Preprocessing parameters
        self.window_size = config.get('window_size', 150)  # 5 seconds @ 30Hz (after resampling)
        self.stride = config.get('stride', 30)  # 1 second @ 30Hz

        # Scale factor (12-bit digital values (±2g) -> G conversion)
        # 12-bit signed: -2048 to 2047 for ±2g → 1024 = 1g
        self.scale_factor = DATASETS.get('WARD', {}).get('scale_factor', None)

        # Activity name mapping (from Table 1 in the paper)
        self.activity_names = [
            'Stand',              # a1
            'Sit',                # a2
            'Lie',                # a3
            'Walk Forward',       # a4
            'Walk Left-Circle',   # a5
            'Walk Right-Circle',  # a6
            'Turn Left',          # a7
            'Turn Right',         # a8
            'Go Upstairs',        # a9
            'Go Downstairs',      # a10
            'Jog',                # a11
            'Jump',               # a12
            'Push Wheelchair'     # a13
        ]

    def get_dataset_name(self) -> str:
        return 'ward'

    def download_dataset(self) -> None:
        """
        Download and extract WARD dataset
        """
        logger.info("=" * 80)
        logger.info("Downloading WARD dataset")
        logger.info("=" * 80)

        ward_raw_path = self.raw_data_path / self.dataset_name

        # Check if data already exists
        if check_dataset_exists(ward_raw_path, required_files=['WARD1.0']):
            logger.warning(f"WARD data already exists at {ward_raw_path}")
            response = input("Do you want to re-download? (y/N): ")
            if response.lower() != 'y':
                logger.info("Skipping download")
                return

        try:
            # 1. Download
            logger.info("Step 1/2: Downloading archive")
            ward_raw_path.parent.mkdir(parents=True, exist_ok=True)
            zip_path = ward_raw_path.parent / 'WARD1.zip'
            download_file(WARD_URL, zip_path, desc='Downloading WARD')

            # 2. Extract
            logger.info("Step 2/2: Extracting archive")
            extract_archive(zip_path, ward_raw_path.parent, desc='Extracting WARD')

            # Rename WARD1.0 to ward
            extracted_dir = ward_raw_path.parent / 'WARD1.0'
            if extracted_dir.exists() and not ward_raw_path.exists():
                extracted_dir.rename(ward_raw_path)

            # Cleanup
            if zip_path.exists():
                zip_path.unlink()

            logger.info("=" * 80)
            logger.info(f"SUCCESS: WARD dataset downloaded to {ward_raw_path}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Failed to download WARD dataset: {e}", exc_info=True)
            raise

    def load_raw_data(self) -> Dict[int, list]:
        """
        Load WARD raw data per subject

        Expected format:
        - data/raw/ward/Subject1/a1t1.mat (activity 1, trial 1)
        - Inside MAT file: WearableData structure
          - Class: activity class (1-13)
          - Subject: subject number (1-20)
          - Reading: 5 sensor data (each sensor: (samples, 5))

        Returns:
            person_data: dictionary of {person_id: [(data, labels), ...]}
                Each element is a MAT file (trial/session) unit
                data: array of (num_samples, 5, 5) [5 sensors × 5 channels]
                labels: array of (num_samples,)
        """
        raw_path = self.raw_data_path / self.dataset_name

        if not raw_path.exists():
            raise FileNotFoundError(
                f"WARD raw data not found at {raw_path}\n"
                "Expected structure: data/raw/ward/Subject1/a1t1.mat"
            )

        # Store data per subject
        person_data = {person_id: {'sessions': []}
                       for person_id in range(1, self.num_subjects + 1)}

        # Process each subject's directory
        for person_id in range(1, self.num_subjects + 1):
            subject_dir = raw_path / f"Subject{person_id}"

            if not subject_dir.exists():
                logger.warning(f"Subject directory not found: {subject_dir}")
                continue

            # Load all MAT files
            mat_files = sorted(subject_dir.glob("*.mat"))

            for mat_file in mat_files:
                try:
                    # Load MAT file
                    mat = sio.loadmat(mat_file)
                    wd = mat['WearableData']

                    # Get class ID (1-13 -> 0-12)
                    class_id = int(wd['Class'][0, 0][0]) - 1

                    # Get Reading (sensor data)
                    reading = wd['Reading'][0, 0]

                    # Combine 5 sensor data
                    # Each sensor: (samples, 5) -> All sensors: (5, samples, 5)
                    sensor_arrays = []
                    sample_lengths = []

                    for sensor_idx in range(5):
                        sensor_data = reading[0][sensor_idx]  # (samples, 5)

                        # Check for Inf (missing data)
                        if np.isinf(sensor_data).any():
                            logger.warning(
                                f"Inf values detected in {mat_file.name}, "
                                f"Sensor {sensor_idx+1} - skipping this trial"
                            )
                            break

                        sensor_arrays.append(sensor_data)
                        sample_lengths.append(sensor_data.shape[0])
                    else:
                        # All sensor data is valid
                        # Check if sample lengths match
                        if len(set(sample_lengths)) != 1:
                            logger.warning(
                                f"Sample length mismatch in {mat_file.name}: {sample_lengths} - skipping"
                            )
                            continue

                        num_samples = sample_lengths[0]

                        # Integrate sensor data: (samples, 5_sensors, 5_channels)
                        # sensor_arrays: list of 5 × (samples, 5)
                        combined_data = np.stack(sensor_arrays, axis=1)  # (samples, 5_sensors, 5_channels)

                        # Generate labels
                        labels = np.full(num_samples, class_id, dtype=np.int32)

                        person_data[person_id]['sessions'].append((combined_data, labels))

                except Exception as e:
                    logger.error(f"Error loading {mat_file}: {e}")
                    continue

        # Return session list for each subject (do not concatenate)
        result = {}
        for person_id in range(1, self.num_subjects + 1):
            sessions = person_data[person_id]['sessions']
            if sessions:
                result[person_id] = sessions
                total_samples = sum(d.shape[0] for d, l in sessions)
                logger.info(
                    f"USER{person_id:05d}: {len(sessions)} sessions, {total_samples} samples"
                )
            else:
                logger.warning(f"No valid data loaded for USER{person_id:05d}")

        if not result:
            raise ValueError("No data loaded. Please check the raw data directory structure.")

        logger.info(f"Total users loaded: {len(result)}")
        return result

    def clean_data(self, data: Dict[int, list]) -> Dict[int, list]:
        """
        Data cleaning and resampling (processed per session)

        Args:
            data: dictionary of {person_id: [(data, labels), ...]}
                  data: (samples, 5_sensors, 5_channels)

        Returns:
            Cleaned and resampled {person_id: [(data, labels), ...]}
        """
        cleaned = {}
        for person_id, sessions in data.items():
            cleaned_sessions = []
            for session_data, session_labels in sessions:
                # Temporarily reshape: (samples, 5, 5) -> (samples, 25)
                original_shape = session_data.shape
                flat_data = session_data.reshape(original_shape[0], -1)

                # Remove invalid samples
                cleaned_data, cleaned_labels = filter_invalid_samples(flat_data, session_labels)

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
                    # Restore shape: (samples, 25) -> (samples, 5, 5)
                    resampled_data = resampled_data.reshape(-1, 5, 5)
                    cleaned_sessions.append((resampled_data, resampled_labels))
                else:
                    # Restore shape
                    cleaned_data = cleaned_data.reshape(-1, 5, 5)
                    cleaned_sessions.append((cleaned_data, cleaned_labels))

            if cleaned_sessions:
                cleaned[person_id] = cleaned_sessions
                total_samples = sum(d.shape[0] for d, l in cleaned_sessions)
                logger.info(f"USER{person_id:05d} cleaned: {len(cleaned_sessions)} sessions, {total_samples} samples")

        return cleaned

    def extract_features(self, data: Dict[int, list]) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
        """
        Feature extraction (windowing and scaling per sensor × modality)
        Windows do not cross session boundaries

        Args:
            data: dictionary of {person_id: [(data, labels), ...]}
                  data: (samples, 5_sensors, 5_channels)

        Returns:
            {person_id: {sensor/modality: {'X': data, 'Y': labels}}}
            Example: {'LeftArm/ACC': {'X': (N, 3, 150), 'Y': (N,)}}
        """
        processed = {}

        for person_id, sessions in data.items():
            logger.info(f"Processing USER{person_id:05d} ({len(sessions)} sessions)")

            processed[person_id] = {}

            # Process each sensor
            for sensor_name in self.sensor_names:
                sensor_idx = self.sensor_indices[sensor_name]

                # Extract sensor data per session
                sensor_sessions = [
                    (session_data[:, sensor_idx, :], session_labels)
                    for session_data, session_labels in sessions
                ]

                # Apply sliding windows per session
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

                # Split into each modality
                for modality_name in self.modalities:
                    mod_start_ch, mod_end_ch = self.modality_channel_ranges[modality_name]

                    # Extract modality channels
                    modality_data = windowed_data[:, :, mod_start_ch:mod_end_ch]
                    # modality_data: (num_windows, window_size, channels)

                    # Apply scaling (accelerometer only)
                    if modality_name == 'ACC' and self.scale_factor is not None:
                        modality_data = modality_data / self.scale_factor
                        logger.info(f"  Applied scale_factor={self.scale_factor} to {sensor_name}/{modality_name}")

                    # Reshape: (num_windows, window_size, channels) -> (num_windows, channels, window_size)
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
            data/processed/ward/USER00001/LeftArm/ACC/X.npy, Y.npy
            data/processed/ward/USER00001/LeftArm/GYRO/X.npy, Y.npy
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
            'original_sampling_rate': self.original_sampling_rate,
            'target_sampling_rate': self.target_sampling_rate,
            'window_size': self.window_size,
            'stride': self.stride,
            'normalization': 'none',  # No normalization (raw data retained)
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
                X = arrays['X']  # (num_windows, channels, window_size)
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
