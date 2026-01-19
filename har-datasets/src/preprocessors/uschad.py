"""
USC-HAD (USC Human Activity Dataset) Preprocessor

USC-HAD Dataset:
- 12 daily activities
- 14 subjects (7 male, 7 female)
- 1 IMU sensor (6 channels: 3-axis ACC + 3-axis GYRO)
- Sampling rate: 100Hz
- Sensor position: Front right hip (Phone)
- Acceleration unit: G (already in G units)
"""

import numpy as np
import scipy.io
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
    check_dataset_exists
)
from . import register_preprocessor
from ..dataset_info import DATASETS

logger = logging.getLogger(__name__)

USCHAD_URL = "https://sipi.usc.edu/had/USC-HAD.zip"


@register_preprocessor('uschad')
class USCHADPreprocessor(BasePreprocessor):
    """
    Preprocessor class for USC-HAD dataset
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # USC-HAD specific settings
        self.num_activities = 12
        self.num_subjects = 14
        self.num_sensors = 1  # Phone (front right hip) only
        self.channels_per_sensor = 6  # 3-axis acc + 3-axis gyro
        self.num_channels = 6

        # Sampling rate
        self.original_sampling_rate = 100  # Hz (USC-HAD original)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz (target)

        # Sensor names and channel mapping
        self.sensor_names = ['Phone']
        self.sensor_channel_ranges = {
            'Phone': (0, 6),  # channels 0-5: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
        }

        # Modalities (channel division within each sensor)
        self.modalities = ['ACC', 'GYRO']
        self.modality_channel_ranges = {
            'ACC': (0, 3),   # 3-axis acceleration (G units)
            'GYRO': (3, 6),  # 3-axis gyroscope (dps)
        }
        self.channels_per_modality = 3

        # Preprocessing parameters
        self.window_size = config.get('window_size', 150)  # 5 seconds @ 30Hz
        self.stride = config.get('stride', 30)  # 1 second @ 30Hz

        # Scaling factor (USC-HAD is already in G units, so None)
        self.scale_factor = DATASETS.get('USCHAD', {}).get('scale_factor', None)

        # Activity name mapping
        self.activity_names = {
            1: 'walking-forward',
            2: 'walking-left',
            3: 'walking-right',
            4: 'walking-upstairs',
            5: 'walking-downstairs',
            6: 'running-forward',
            7: 'jumping-up',
            8: 'sitting',
            9: 'standing',
            10: 'sleeping',
            11: 'elevator-up',
            12: 'elevator-down'
        }

    def get_dataset_name(self) -> str:
        return 'uschad'

    def download_dataset(self) -> None:
        """
        Download USC-HAD dataset
        """
        logger.info("=" * 80)
        logger.info("Downloading USC-HAD dataset")
        logger.info("=" * 80)

        dataset_path = self.raw_data_path / self.dataset_name

        if check_dataset_exists(dataset_path, required_files=['Subject*/a*.mat']):
            logger.info(f"USC-HAD data already exists at {dataset_path}")
            return

        dataset_path.mkdir(parents=True, exist_ok=True)
        zip_path = dataset_path / "USC-HAD.zip"

        # Download
        logger.info("Downloading USC-HAD archive...")
        download_file(USCHAD_URL, zip_path, desc='Downloading USC-HAD')

        # Extract
        logger.info("Extracting USC-HAD archive...")
        extract_archive(zip_path, dataset_path, desc='Extracting USC-HAD')

        # Move files from USC-HAD subfolder to dataset_path if needed
        uschad_subdir = dataset_path / "USC-HAD"
        if uschad_subdir.exists():
            import shutil
            for item in uschad_subdir.iterdir():
                shutil.move(str(item), str(dataset_path / item.name))
            uschad_subdir.rmdir()

        # Cleanup
        if zip_path.exists():
            zip_path.unlink()

        logger.info(f"USC-HAD dataset downloaded to {dataset_path}")

    def _load_mat_file(self, mat_path: Path) -> Dict:
        """
        Load .mat file

        Args:
            mat_path: Path to the .mat file

        Returns:
            Dictionary of loaded data
        """
        data = scipy.io.loadmat(mat_path)

        # Extract metadata (extract values from 1-element arrays)
        result = {
            "subject": data["subject"][0] if "subject" in data else "unknown",
            "activity": data["activity"][0] if "activity" in data else "unknown",
            "activity_number": int(data["activity_number"][0]) if "activity_number" in data else -1,
            "trial": int(data["trial"][0]) if "trial" in data else -1,
            "sensor_readings": data["sensor_readings"],  # shape: (N, 6)
        }

        return result

    def load_raw_data(self) -> Dict[int, list]:
        """
        Load USC-HAD raw data for each subject

        Expected format:
        - data/raw/uschad/Subject1/a1t1.mat (activity 1, trial 1)
        - Each file: sensor_readings (N, 6) - [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]

        Returns:
            person_data: Dictionary of {person_id: [(data, labels), ...]}
                Each element is a file (session/trial) unit
                data: Array of shape (num_samples, 6)
                labels: Array of shape (num_samples,)
        """
        raw_path = self.raw_data_path / self.dataset_name

        if not raw_path.exists():
            raise FileNotFoundError(
                f"USC-HAD raw data not found at {raw_path}\n"
                "Expected structure: data/raw/uschad/Subject1/a1t1.mat"
            )

        # Store data for each subject
        person_data = {}

        # For each subject
        for subject_id in range(1, self.num_subjects + 1):
            subject_dir = raw_path / f"Subject{subject_id}"

            if not subject_dir.exists():
                logger.warning(f"Subject directory not found: {subject_dir}")
                continue

            person_data[subject_id] = {'sessions': []}

            # Load all .mat files
            mat_files = sorted(subject_dir.glob("*.mat"))

            if len(mat_files) == 0:
                logger.warning(f"No .mat files found in {subject_dir}")
                continue

            for mat_file in mat_files:
                try:
                    # Load data
                    mat_data = self._load_mat_file(mat_file)
                    sensor_readings = mat_data["sensor_readings"]  # (N, 6)
                    activity_num = mat_data["activity_number"]

                    # Fix invalid activity_number (e.g., a11t4.mat has activity_number=0)
                    # Extract correct activity number from filename: a{activity}t{trial}.mat
                    if activity_num < 1 or activity_num > self.num_activities:
                        import re
                        match = re.match(r'a(\d+)t\d+\.mat', mat_file.name)
                        if match:
                            correct_activity = int(match.group(1))
                            logger.warning(
                                f"Invalid activity_number={activity_num} in {mat_file.name}, "
                                f"using filename-derived value: {correct_activity}"
                            )
                            activity_num = correct_activity
                        else:
                            logger.warning(f"Skipping {mat_file.name}: invalid activity_number={activity_num}")
                            continue

                    # Check data shape
                    if sensor_readings.shape[1] != self.num_channels:
                        logger.warning(
                            f"Unexpected number of channels in {mat_file}: "
                            f"{sensor_readings.shape[1]} (expected {self.num_channels})"
                        )
                        continue

                    # Generate labels (0-indexed)
                    segment_labels = np.full(len(sensor_readings), activity_num - 1)

                    person_data[subject_id]['sessions'].append((sensor_readings, segment_labels))

                except Exception as e:
                    logger.error(f"Error loading {mat_file}: {e}")
                    continue

        # Return session list for each subject (not concatenated)
        result = {}
        for subject_id in range(1, self.num_subjects + 1):
            if subject_id in person_data and person_data[subject_id]['sessions']:
                sessions = person_data[subject_id]['sessions']
                result[subject_id] = sessions
                total_samples = sum(d.shape[0] for d, l in sessions)
                logger.info(f"USER{subject_id:05d}: {len(sessions)} sessions, {total_samples} samples")
            else:
                logger.warning(f"No data loaded for USER{subject_id:05d}")

        if not result:
            raise ValueError("No data loaded. Please check the raw data directory structure.")

        logger.info(f"Total users loaded: {len(result)}")
        return result

    def clean_data(self, data: Dict[int, list]) -> Dict[int, list]:
        """
        Data cleaning and resampling (processed per session)

        Args:
            data: Dictionary of {person_id: [(data, labels), ...]}

        Returns:
            Cleaned and resampled {person_id: [(data, labels), ...]}
        """
        cleaned = {}
        for person_id, sessions in data.items():
            cleaned_sessions = []
            for session_data, session_labels in sessions:
                # Remove invalid samples
                cleaned_data, cleaned_labels = filter_invalid_samples(session_data, session_labels)

                if len(cleaned_data) == 0:
                    continue

                # Resampling (100Hz -> 30Hz)
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
                cleaned[person_id] = cleaned_sessions
                total_samples = sum(d.shape[0] for d, l in cleaned_sessions)
                logger.info(f"USER{person_id:05d} cleaned: {len(cleaned_sessions)} sessions, {total_samples} samples")

        return cleaned

    def extract_features(self, data: Dict[int, list]) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
        """
        Feature extraction (windowing per sensor × modality)
        Does not generate windows across session boundaries

        Args:
            data: Dictionary of {person_id: [(data, labels), ...]}

        Returns:
            {person_id: {sensor_modality: {'X': data, 'Y': labels}}}
            Example: {'Phone/ACC': {'X': (N, 3, 150), 'Y': (N,)}}
        """
        processed = {}

        for person_id, sessions in data.items():
            logger.info(f"Processing USER{person_id:05d} ({len(sessions)} sessions)")

            processed[person_id] = {}

            # Process each sensor (USC-HAD has only Phone)
            for sensor_name in self.sensor_names:
                sensor_start_ch, sensor_end_ch = self.sensor_channel_ranges[sensor_name]

                # Extract sensor channels for each session
                sensor_sessions = [
                    (session_data[:, sensor_start_ch:sensor_end_ch], session_labels)
                    for session_data, session_labels in sessions
                ]

                # Apply sliding window per session
                windowed_data, windowed_labels = create_sliding_windows_multi_session(
                    sensor_sessions,
                    window_size=self.window_size,
                    stride=self.stride,
                    drop_last=False,
                    pad_last=True
                )

                if len(windowed_data) == 0:
                    logger.warning(f"  USER{person_id:05d} has no valid windows, skipping")
                    continue

                # Split into each modality
                for modality_name in self.modalities:
                    mod_start_ch, mod_end_ch = self.modality_channel_ranges[modality_name]

                    # Extract modality channels
                    modality_data = windowed_data[:, :, mod_start_ch:mod_end_ch]
                    # modality_data: (num_windows, window_size, 3)

                    # Apply scaling (USC-HAD is already in G units, so scale_factor=None)
                    if modality_name == 'ACC' and self.scale_factor is not None:
                        modality_data = modality_data / self.scale_factor
                        logger.info(f"  Applied scale_factor={self.scale_factor} to {sensor_name}/{modality_name}")

                    # Transform shape: (num_windows, window_size, 3) -> (num_windows, 3, window_size)
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
            data/processed/uschad/USER00001/Phone/ACC/X.npy, Y.npy
            data/processed/uschad/USER00001/Phone/GYRO/X.npy, Y.npy
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
            'channels_per_modality': self.channels_per_modality,
            'original_sampling_rate': self.original_sampling_rate,
            'target_sampling_rate': self.target_sampling_rate,
            'window_size': self.window_size,
            'stride': self.stride,
            'normalization': 'none',  # No normalization (raw data retained)
            'scale_factor': self.scale_factor,  # USC-HAD is already in G units, so None
            'data_dtype': 'float16',  # Data type
            'data_shape': f'(num_windows, {self.channels_per_modality}, {self.window_size})',
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
                X = arrays['X']  # (num_windows, 3, window_size)
                Y = arrays['Y']  # (num_windows,)

                np.save(sensor_modality_path / 'X.npy', X)
                np.save(sensor_modality_path / 'Y.npy', Y)

                # Statistical information
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
