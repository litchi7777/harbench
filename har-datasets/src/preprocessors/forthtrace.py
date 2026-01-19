"""
FORTHTRACE (FORTH-TRACE) Dataset Preprocessing

FORTHTRACE Dataset:
- 16 activities (7 basic + 9 postural transitions)
- 15 subjects
- 5 Shimmer sensors (9 channels × 5)
- Sampling rate: 51.2Hz
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
import logging
import shutil

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


# FORTHTRACE Dataset URL
FORTHTRACE_URL = "https://zenodo.org/records/841301/files/FORTH_TRACE_DATASET.zip?download=1"


@register_preprocessor('forthtrace')
class ForthtracePreprocessor(BasePreprocessor):
    """
    Preprocessor class for FORTHTRACE dataset
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # FORTHTRACE specific settings
        self.num_activities = 16
        self.num_subjects = 15
        self.num_sensors = 5
        self.channels_per_sensor = 9  # 3-axis acc, gyro, mag
        self.num_channels = 9  # Each CSV file contains data for 1 sensor

        # Sampling rate
        self.original_sampling_rate = 51.2  # Hz (FORTHTRACE original)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz (target)

        # Sensor names (corresponding to device IDs 1-5)
        self.sensor_names = ['LeftWrist', 'RightWrist', 'Torso', 'RightThigh', 'LeftAnkle']
        self.device_id_to_sensor = {
            1: 'LeftWrist',
            2: 'RightWrist',
            3: 'Torso',
            4: 'RightThigh',
            5: 'LeftAnkle'
        }

        # Modalities (channel division within each sensor)
        self.modalities = ['ACC', 'GYRO', 'MAG']
        self.modality_channel_ranges = {
            'ACC': (0, 3),   # 3-axis accelerometer (columns 1,2,3)
            'GYRO': (3, 6),  # 3-axis gyroscope (columns 4,5,6)
            'MAG': (6, 9)    # 3-axis magnetometer (columns 7,8,9)
        }
        self.channels_per_modality = 3

        # Preprocessing parameters
        self.window_size = config.get('window_size', 150)  # 5 seconds @ 30Hz
        self.stride = config.get('stride', 30)  # 1 second @ 30Hz

        # Scaling factor (convert m/s^2 -> G)
        self.scale_factor = DATASETS.get('FORTHTRACE', {}).get('scale_factor', None)

    def get_dataset_name(self) -> str:
        return 'forthtrace'

    def download_dataset(self) -> None:
        """
        Download and extract FORTHTRACE dataset
        """
        logger.info("=" * 80)
        logger.info("Downloading FORTHTRACE dataset")
        logger.info("=" * 80)

        forthtrace_raw_path = self.raw_data_path / self.dataset_name

        # Check if data already exists
        if check_dataset_exists(forthtrace_raw_path, required_files=['part*/part*dev*.csv']):
            logger.warning(f"FORTHTRACE data already exists at {forthtrace_raw_path}")
            response = input("Do you want to re-download? (y/N): ")
            if response.lower() != 'y':
                logger.info("Skipping download")
                return

        try:
            # 1. Download
            logger.info("Step 1/2: Downloading archive")
            forthtrace_raw_path.parent.mkdir(parents=True, exist_ok=True)
            zip_path = forthtrace_raw_path.parent / 'forthtrace.zip'
            download_file(FORTHTRACE_URL, zip_path, desc='Downloading FORTHTRACE')

            # 2. Extract and organize data
            logger.info("Step 2/2: Extracting and organizing data")
            extract_to = forthtrace_raw_path.parent / 'forthtrace_temp'
            extract_archive(zip_path, extract_to, desc='Extracting FORTHTRACE')
            self._organize_forthtrace_data(extract_to, forthtrace_raw_path)

            # Cleanup
            cleanup_temp_files(extract_to)
            if zip_path.exists():
                zip_path.unlink()

            logger.info("=" * 80)
            logger.info(f"SUCCESS: FORTHTRACE dataset downloaded to {forthtrace_raw_path}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Failed to download FORTHTRACE dataset: {e}", exc_info=True)
            raise

    def _organize_forthtrace_data(self, extracted_path: Path, target_path: Path) -> None:
        """
        Organize FORTHTRACE data into appropriate directory structure

        Args:
            extracted_path: Path to extracted data
            target_path: Target path for organized data (data/raw/forthtrace)
        """
        logger.info(f"Organizing FORTHTRACE data from {extracted_path} to {target_path}")

        # Find root of extracted data
        data_root = extracted_path

        # Look for "FORTH_TRACE_DATASET-master" folder
        if (extracted_path / "FORTH_TRACE_DATASET-master").exists():
            data_root = extracted_path / "FORTH_TRACE_DATASET-master"
        elif (extracted_path / "FORTH_TRACE_DATASET").exists():
            data_root = extracted_path / "FORTH_TRACE_DATASET"

        # Create target directory
        target_path.mkdir(parents=True, exist_ok=True)

        if not data_root.exists():
            raise FileNotFoundError(f"Could not find data directory in {extracted_path}")

        # Find and copy part0, part1, ... directories
        part_dirs = sorted([d for d in data_root.glob("part*") if d.is_dir()])

        if not part_dirs:
            raise FileNotFoundError(f"Could not find part directories in {data_root}")

        from tqdm import tqdm
        for part_dir in tqdm(part_dirs, desc='Organizing participants'):
            part_name = part_dir.name
            target_part_dir = target_path / part_name

            # Copy part directory
            if target_part_dir.exists():
                shutil.rmtree(target_part_dir)
            shutil.copytree(part_dir, target_part_dir)

        logger.info(f"Data organized at: {target_path}")

    def load_raw_data(self) -> Dict[int, list]:
        """
        Load FORTHTRACE raw data per subject with timestamp-based sensor synchronization

        Expected format:
        - data/raw/forthtrace/part0/part0dev1.csv (participant 0, device 1)
        - Each CSV file: 12 columns (device_id, acc_x/y/z, gyro_x/y/z, mag_x/y/z, timestamp, label)

        Returns:
            person_data: {person_id: [(data, labels), ...]} dictionary
                Each session is a continuous label segment
                data: (num_samples, 45) array (5 sensors × 9 channels, synchronized)
                labels: (num_samples,) array
        """
        raw_path = self.raw_data_path / self.dataset_name

        if not raw_path.exists():
            raise FileNotFoundError(
                f"FORTHTRACE raw data not found at {raw_path}\n"
                "Expected structure: data/raw/forthtrace/part0/part0dev1.csv"
            )

        # Store data per subject
        person_data = {}

        # For each subject (part0 ~ part14 = 15 subjects)
        # person_id is managed as 1-indexed (starting from USER00001)
        for part_idx in range(self.num_subjects):
            person_dir = raw_path / f"part{part_idx}"

            if not person_dir.exists():
                logger.warning(f"Participant directory not found: {person_dir}")
                continue

            # Convert person_id to 1-indexed (part0 -> USER00001)
            person_id = part_idx + 1

            # Load data from all devices
            device_data = {}
            all_devices_loaded = True

            for device_id in range(1, self.num_sensors + 1):
                device_file = person_dir / f"part{part_idx}dev{device_id}.csv"

                if not device_file.exists():
                    logger.warning(f"Device file not found: {device_file}")
                    all_devices_loaded = False
                    break

                try:
                    # Load CSV data (no header)
                    df = pd.read_csv(device_file, header=None)

                    # Validate columns
                    if df.shape[1] != 12:
                        logger.warning(
                            f"Unexpected number of columns in {device_file}: "
                            f"{df.shape[1]} (expected 12)"
                        )
                        all_devices_loaded = False
                        break

                    # Extract sensor data (columns 1-9: acc, gyro, mag)
                    sensor_data = df.iloc[:, 1:10].values.astype(np.float32)
                    # Extract timestamp (column 10)
                    timestamps = df.iloc[:, 10].values.astype(np.float64)
                    # Extract label (column 11, convert from 1-indexed to 0-indexed)
                    labels = df.iloc[:, 11].values.astype(int) - 1

                    sensor_name = self.device_id_to_sensor[device_id]
                    device_data[sensor_name] = {
                        'data': sensor_data,
                        'timestamps': timestamps,
                        'labels': labels
                    }

                except Exception as e:
                    logger.error(f"Error loading {device_file}: {e}")
                    all_devices_loaded = False
                    break

            if not all_devices_loaded or len(device_data) != self.num_sensors:
                logger.warning(f"Skipping USER{person_id:05d}: incomplete device data")
                continue

            # Synchronize sensors based on timestamp
            synchronized_data = self._synchronize_sensors(device_data, person_id)

            if synchronized_data is not None:
                person_data[person_id] = synchronized_data
                total_samples = sum(len(s[0]) for s in synchronized_data)
                logger.info(
                    f"USER{person_id:05d}: {len(synchronized_data)} sessions, "
                    f"total_samples={total_samples}"
                )

        if not person_data:
            raise ValueError("No data loaded. Please check the raw data directory structure.")

        logger.info(f"Total users loaded: {len(person_data)}")
        return person_data

    def _synchronize_sensors(
        self,
        device_data: Dict[str, Dict],
        person_id: int
    ) -> list:
        """
        Synchronize sensors based on timestamp

        Args:
            device_data: {sensor_name: {'data': array, 'timestamps': array, 'labels': array}}
            person_id: Subject ID

        Returns:
            sessions: [(data, labels), ...] list
                data: (num_samples, 45) - 5 sensors × 9 channels, synchronized
                labels: (num_samples,) - synchronized labels
        """
        # Calculate common timestamp range across all sensors
        all_starts = [device_data[s]['timestamps'][0] for s in self.sensor_names]
        all_ends = [device_data[s]['timestamps'][-1] for s in self.sensor_names]
        common_start = max(all_starts)
        common_end = min(all_ends)

        if common_start >= common_end:
            logger.warning(f"USER{person_id:05d}: No common timestamp range")
            return None

        # Restrict each sensor to common range
        aligned_data = {}
        for sensor_name in self.sensor_names:
            ts = device_data[sensor_name]['timestamps']
            data = device_data[sensor_name]['data']
            labels = device_data[sensor_name]['labels']

            # Get indices within common range
            mask = (ts >= common_start) & (ts <= common_end)
            aligned_data[sensor_name] = {
                'data': data[mask],
                'timestamps': ts[mask],
                'labels': labels[mask]
            }

        # Align to minimum sample count (absorb minor differences due to timestamp offset)
        min_samples = min(len(aligned_data[s]['data']) for s in self.sensor_names)

        # Concatenate data from all sensors (fixed sensor order)
        # Order: LeftWrist, RightWrist, Torso, RightThigh, LeftAnkle
        combined_data = np.hstack([
            aligned_data[sensor_name]['data'][:min_samples]
            for sensor_name in self.sensor_names
        ])  # (min_samples, 45)

        # Get labels from first sensor (LeftWrist) - should be same across all sensors
        combined_labels = aligned_data[self.sensor_names[0]]['labels'][:min_samples]

        # Verify label consistency
        for sensor_name in self.sensor_names[1:]:
            other_labels = aligned_data[sensor_name]['labels'][:min_samples]
            mismatch_rate = np.mean(combined_labels != other_labels)
            if mismatch_rate > 0.01:  # Warn if >1% mismatch
                logger.warning(
                    f"USER{person_id:05d}: Label mismatch between LeftWrist and {sensor_name}: "
                    f"{mismatch_rate*100:.2f}%"
                )

        # Split into sessions at label change points (continuous label segments become sessions)
        sessions = self._split_by_label_changes(combined_data, combined_labels)

        return sessions

    def _split_by_label_changes(
        self,
        data: np.ndarray,
        labels: np.ndarray
    ) -> list:
        """
        Split data into sessions at label change points

        Args:
            data: (num_samples, channels) array
            labels: (num_samples,) array

        Returns:
            sessions: [(data, labels), ...] list
        """
        if len(data) == 0:
            return []

        # Detect label change points
        change_points = np.where(labels[:-1] != labels[1:])[0] + 1
        split_points = [0] + change_points.tolist() + [len(data)]

        sessions = []
        for i in range(len(split_points) - 1):
            start = split_points[i]
            end = split_points[i + 1]
            session_data = data[start:end]
            session_labels = labels[start:end]
            sessions.append((session_data, session_labels))

        return sessions

    def clean_data(self, data: Dict[int, list]) -> Dict[int, list]:
        """
        Data cleaning and resampling (processed per session)

        Args:
            data: {person_id: [(data, labels), ...]} dictionary

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

                # Resampling (51.2Hz -> 30Hz)
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
                total_samples = sum(len(s[0]) for s in cleaned_sessions)
                logger.info(f"USER{person_id:05d} cleaned: {len(cleaned_sessions)} sessions, {total_samples} samples")

        return cleaned

    def extract_features(self, data: Dict[int, list]) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
        """
        Feature extraction (windowing and scaling per sensor × modality)
        Split synchronized sensor data into sensor/modality components

        Args:
            data: {person_id: [(data, labels), ...]} dictionary
                data: (num_samples, 45) - 5 sensors × 9 channels

        Returns:
            {person_id: {sensor/modality: {'X': data, 'Y': labels}}}
            Example: {'LeftWrist/ACC': {'X': (N, 3, 150), 'Y': (N,)}}
        """
        processed = {}

        # Channel ranges per sensor (concatenation order: LeftWrist, RightWrist, Torso, RightThigh, LeftAnkle)
        sensor_channel_ranges = {
            sensor_name: (i * 9, (i + 1) * 9)
            for i, sensor_name in enumerate(self.sensor_names)
        }

        for person_id, sessions in data.items():
            logger.info(f"Processing USER{person_id:05d} ({len(sessions)} sessions)")

            processed[person_id] = {}

            # Process each sensor
            for sensor_name in self.sensor_names:
                sensor_start_ch, sensor_end_ch = sensor_channel_ranges[sensor_name]

                # Extract sensor data per session
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
                # windowed_data: (num_windows, window_size, 9)

                if len(windowed_data) == 0:
                    logger.warning(f"  {sensor_name}: no windows created")
                    continue

                # Split into modalities
                for modality_name in self.modalities:
                    mod_start_ch, mod_end_ch = self.modality_channel_ranges[modality_name]

                    # Extract modality channels
                    modality_data = windowed_data[:, :, mod_start_ch:mod_end_ch]
                    # modality_data: (num_windows, window_size, 3)

                    # Apply scaling (accelerometer only)
                    if modality_name == 'ACC' and self.scale_factor is not None:
                        modality_data = modality_data / self.scale_factor
                        logger.info(
                            f"  Applied scale_factor={self.scale_factor} to "
                            f"{sensor_name}/{modality_name}"
                        )

                    # Transpose shape: (num_windows, window_size, 3) -> (num_windows, 3, window_size)
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

    def save_processed_data(
        self, data: Dict[int, Dict[str, Dict[str, np.ndarray]]]
    ) -> None:
        """
        Save processed data

        Args:
            data: {person_id: {sensor_modality: {'X': data, 'Y': labels}}}

        Save format:
            data/processed/forthtrace/USER00000/LeftWrist/ACC/X.npy, Y.npy
            data/processed/forthtrace/USER00000/LeftWrist/GYRO/X.npy, Y.npy
            data/processed/forthtrace/USER00000/Torso/ACC/X.npy, Y.npy
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
            'channels_per_modality': self.channels_per_modality,
            'original_sampling_rate': self.original_sampling_rate,
            'target_sampling_rate': self.target_sampling_rate,
            'window_size': self.window_size,
            'stride': self.stride,
            'normalization': 'none',  # No normalization (raw data preserved)
            'scale_factor': self.scale_factor,  # Scaling factor (applied to ACC only)
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
