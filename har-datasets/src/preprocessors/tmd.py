"""
TMD (Transportation Mode Detection) Preprocessing

TMD Dataset:
- 5 types of transportation modes (Walking, Car, Still, Train, Bus)
- 16 subjects (U1-U16)
- Smartphone sensors (accelerometer, gyroscope, etc.)
- Event-driven sampling (variable rate)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, List
import logging
from collections import defaultdict

from .base import BasePreprocessor
from .utils import (
    create_sliding_windows,
    create_sliding_windows_multi_session,
    filter_invalid_samples,
    resample_timeseries,
    get_class_distribution
)
from .common import (
    download_file,
    extract_archive,
    check_dataset_exists
)
from . import register_preprocessor
from ..dataset_info import DATASETS

logger = logging.getLogger(__name__)

TMD_URL = "https://cs.unibo.it/projects/us-tm2017/static/dataset/raw_data/raw_data.tar.gz"


@register_preprocessor('tmd')
class TMDPreprocessor(BasePreprocessor):
    """
    Preprocessing class for TMD dataset
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # TMD-specific settings
        self.num_activities = 5
        self.num_subjects = 16

        # Sensor names
        self.sensor_names = ['Smartphone']

        # Modalities
        self.modalities = ['ACC', 'GYRO']
        self.channels_per_modality = {
            'ACC': 3,   # 3-axis accelerometer
            'GYRO': 3   # 3-axis gyroscope
        }

        # Activity mapping
        self.activity_map = {
            'Walking': 0,
            'Car': 1,
            'Still': 2,
            'Train': 3,
            'Bus': 4
        }

        # Sampling rate (variable -> fixed conversion)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz

        # Preprocessing parameters
        self.window_size = config.get('window_size', 150)  # 5 seconds @ 30Hz
        self.stride = config.get('stride', 30)  # 1 second @ 30Hz

        # Scaling factor (m/s^2 -> G conversion)
        self.scale_factor = DATASETS.get('TMD', {}).get('scale_factor', None)

    def get_dataset_name(self) -> str:
        return 'tmd'

    def download_dataset(self) -> None:
        """
        Download TMD dataset from University of Bologna
        https://cs.unibo.it/projects/us-tm2017/
        """
        logger.info("=" * 80)
        logger.info("Downloading TMD dataset")
        logger.info("=" * 80)

        dataset_path = self.raw_data_path / self.dataset_name
        raw_data_dir = dataset_path / "raw_data"
        data_exists = raw_data_dir.exists() and any(raw_data_dir.rglob('*.csv'))

        if data_exists:
            logger.info(f"TMD data already exists at {dataset_path}")
            return

        dataset_path.mkdir(parents=True, exist_ok=True)
        tar_path = dataset_path / "raw_data.tar.gz"

        # Download
        logger.info("Downloading TMD archive from University of Bologna...")
        download_file(TMD_URL, tar_path, desc='Downloading TMD')

        # Extract
        logger.info("Extracting TMD archive...")
        extract_archive(tar_path, dataset_path, desc='Extracting TMD')

        # Cleanup
        if tar_path.exists():
            tar_path.unlink()

        logger.info(f"TMD dataset downloaded to {dataset_path}")

    def load_raw_data(self) -> Dict[int, list]:
        """
        Load TMD data per subject

        Returns:
            Dictionary of {subject_id: [(data, labels), ...]}
                Each element represents a CSV file (session)
                data: (num_samples, 6) array (ACC 3ch + GYRO 3ch)
                labels: (num_samples,) array
        """
        logger.info("=" * 80)
        logger.info("Loading TMD raw data")
        logger.info("=" * 80)

        tmd_data_path = self.raw_data_path / self.dataset_name / 'raw_data'

        if not tmd_data_path.exists():
            raise FileNotFoundError(
                f"TMD data not found at {tmd_data_path}. "
                "Please download the dataset manually."
            )

        all_data = {}

        # Load data for each subject
        for subject_id in range(1, 17):  # U1 to U16
            subject_name = f"U{subject_id}"
            subject_path = tmd_data_path / subject_name

            if not subject_path.exists():
                logger.warning(f"Subject {subject_name} not found, skipping")
                continue

            logger.info(f"Loading {subject_name}")

            # Load all CSV files for the subject (held as session list)
            sessions = []

            csv_files = list(subject_path.glob("sensorfile_*.csv"))

            for csv_file in csv_files:
                # Extract activity label from filename
                # Example: sensorfile_U1_Walking_1480512323378.csv -> Walking
                filename_parts = csv_file.stem.split('_')
                if len(filename_parts) >= 3:
                    activity_name = filename_parts[2]
                    if activity_name in self.activity_map:
                        label = self.activity_map[activity_name]

                        # Load CSV file and extract sensor data
                        try:
                            data = self._parse_csv_file(csv_file)
                            if data is not None and len(data) > 0:
                                labels = np.full(len(data), label, dtype=np.int32)
                                sessions.append((data, labels))
                        except Exception as e:
                            logger.warning(f"Failed to parse {csv_file.name}: {e}")
                            continue

            if sessions:
                all_data[subject_id] = sessions
                total_samples = sum(d.shape[0] for d, l in sessions)
                logger.info(
                    f"  {subject_name}: {len(sessions)} sessions, {total_samples} samples"
                )

        logger.info(f"Loaded {len(all_data)} subjects")
        return all_data

    def _parse_csv_file(self, csv_file: Path) -> np.ndarray:
        """
        Parse TMD CSV file and extract sensor data

        Args:
            csv_file: Path to CSV file

        Returns:
            (num_samples, 6) array (ACC 3ch + GYRO 3ch)
        """
        # TMD CSV has different column counts per row, so parse manually
        return self._parse_csv_manual(csv_file)

    def _parse_csv_manual(self, csv_file: Path) -> np.ndarray:
        """
        Parse CSV manually (when pandas encounters errors)
        """
        acc_data = []
        gyro_data = []

        with open(csv_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    try:
                        timestamp = float(parts[0])
                        sensor_type = parts[1]
                        x = float(parts[2])
                        y = float(parts[3])
                        z = float(parts[4]) if len(parts) > 4 else 0.0

                        if sensor_type == 'android.sensor.accelerometer':
                            acc_data.append([timestamp, x, y, z])
                        elif sensor_type == 'android.sensor.gyroscope':
                            gyro_data.append([timestamp, x, y, z])
                    except ValueError:
                        continue

        if not acc_data or not gyro_data:
            return None

        acc_df = pd.DataFrame(acc_data, columns=['timestamp', 'x', 'y', 'z'])
        gyro_df = pd.DataFrame(gyro_data, columns=['timestamp', 'x', 'y', 'z'])

        # Sort by timestamp and remove duplicates
        # Spline interpolation requires strictly monotonically increasing sequence
        acc_df = acc_df.sort_values('timestamp').drop_duplicates(subset='timestamp', keep='first').reset_index(drop=True)
        gyro_df = gyro_df.sort_values('timestamp').drop_duplicates(subset='timestamp', keep='first').reset_index(drop=True)

        return self._align_sensor_data(acc_df, gyro_df)

    def _align_sensor_data(self, acc_df: pd.DataFrame, gyro_df: pd.DataFrame) -> np.ndarray:
        """
        Align timestamps and integrate accelerometer and gyroscope data

        Paper states: TMD data measured at 20Hz (actually event-driven and sparse)
        Generate smooth 30Hz signal via high-density spline interpolation (1ms) + lowpass filter + downsampling

        Args:
            acc_df: Accelerometer data (timestamp, x, y, z)
            gyro_df: Gyroscope data (timestamp, x, y, z)

        Returns:
            (num_samples, 6) array (ACC 3ch + GYRO 3ch)
        """
        from scipy import signal, interpolate

        # Get common timestamp range
        min_time = max(acc_df['timestamp'].min(), gyro_df['timestamp'].min())
        max_time = min(acc_df['timestamp'].max(), gyro_df['timestamp'].max())

        duration = (max_time - min_time) / 1000.0  # seconds

        if duration <= 0:
            return None

        # Step 1: High-density spline interpolation (1ms = 1000Hz)
        # Spline interpolation is smoother at higher density
        high_rate = 1000  # Hz (1ms interval)
        num_high_samples = int(duration * high_rate)

        if num_high_samples <= 0:
            return None

        high_res_times = np.linspace(min_time, max_time, num_high_samples)

        # High-density via cubic spline interpolation (smoother than linear)
        acc_high = np.zeros((num_high_samples, 3))
        gyro_high = np.zeros((num_high_samples, 3))

        for i, col in enumerate(['x', 'y', 'z']):
            # Accelerometer: Cubic spline interpolation (k=3 for smoothness)
            # Lower degree if few data points
            k_acc = min(3, len(acc_df) - 1)
            if k_acc >= 1:
                spline_acc = interpolate.make_interp_spline(
                    acc_df['timestamp'].values,
                    acc_df[col].values,
                    k=k_acc
                )
                acc_high[:, i] = spline_acc(high_res_times)
            else:
                # Fill with constant value if 1 or fewer data points
                acc_high[:, i] = acc_df[col].values[0] if len(acc_df) > 0 else 0.0

            # Gyroscope: Cubic spline interpolation
            k_gyro = min(3, len(gyro_df) - 1)
            if k_gyro >= 1:
                spline_gyro = interpolate.make_interp_spline(
                    gyro_df['timestamp'].values,
                    gyro_df[col].values,
                    k=k_gyro
                )
                gyro_high[:, i] = spline_gyro(high_res_times)
            else:
                gyro_high[:, i] = gyro_df[col].values[0] if len(gyro_df) > 0 else 0.0

        # Step 2: Apply lowpass filter (remove high-frequency noise, smoother)
        # Cutoff frequency: 10Hz (lower than paper-stated 20Hz)
        nyquist = high_rate / 2.0
        cutoff = 10.0  # Hz
        normalized_cutoff = cutoff / nyquist

        # Butterworth filter (4th order)
        b, a = signal.butter(4, normalized_cutoff, btype='low')

        for i in range(3):
            acc_high[:, i] = signal.filtfilt(b, a, acc_high[:, i])
            gyro_high[:, i] = signal.filtfilt(b, a, gyro_high[:, i])

        # Step 3: Downsample 1000Hz → 30Hz (polyphase filter)
        target_rate = self.target_sampling_rate  # 30 Hz

        from math import gcd
        up = target_rate
        down = high_rate
        common_divisor = gcd(up, down)
        up = up // common_divisor
        down = down // common_divisor

        # Polyphase filtering for each channel
        acc_resampled = np.zeros((signal.resample_poly(acc_high[:, 0], up, down).shape[0], 3))
        gyro_resampled = np.zeros((signal.resample_poly(gyro_high[:, 0], up, down).shape[0], 3))

        for i in range(3):
            acc_resampled[:, i] = signal.resample_poly(acc_high[:, i], up, down)
            gyro_resampled[:, i] = signal.resample_poly(gyro_high[:, i], up, down)

        # Combine ACC + GYRO
        combined = np.hstack([acc_resampled, gyro_resampled])

        return combined

    def clean_data(self, data: Dict[int, list]) -> Dict[int, list]:
        """
        Data cleaning (process per session)

        Note: TMD is already resampled to 30Hz in _align_sensor_data(),
        so no additional resampling is needed here
        """
        logger.info("=" * 80)
        logger.info("Cleaning data (already resampled to 30Hz)")
        logger.info("=" * 80)

        cleaned = {}

        for subject_id, sessions in data.items():
            cleaned_sessions = []
            for session_data, session_labels in sessions:
                # Remove invalid samples (NaN, Inf)
                valid_mask = ~(np.isnan(session_data).any(axis=1) | np.isinf(session_data).any(axis=1))
                cleaned_data = session_data[valid_mask]
                cleaned_labels = session_labels[valid_mask]

                if len(cleaned_data) == 0:
                    continue

                cleaned_sessions.append((cleaned_data, cleaned_labels))

            if cleaned_sessions:
                cleaned[subject_id] = cleaned_sessions
                total_samples = sum(d.shape[0] for d, l in cleaned_sessions)
                logger.info(
                    f"Subject {subject_id:02d}: {len(cleaned_sessions)} sessions, {total_samples} samples "
                    f"(cleaned, already at {self.target_sampling_rate}Hz)"
                )

        return cleaned

    def extract_features(self, data: Dict[int, list]) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
        """
        Feature extraction (windowing and scaling per sensor/modality)
        Does not generate windows across session boundaries

        Returns:
            {subject_id: {sensor/modality: {'X': data, 'Y': labels}}}
        """
        logger.info("=" * 80)
        logger.info("Extracting features (windowing and scaling)")
        logger.info("=" * 80)

        processed = {}

        for subject_id, sessions in data.items():
            logger.info(f"Processing Subject {subject_id:02d} ({len(sessions)} sessions)")
            processed[subject_id] = {}

            # Apply sliding windows per session
            windowed_data, windowed_labels = create_sliding_windows_multi_session(
                sessions,
                window_size=self.window_size,
                stride=self.stride,
                drop_last=False,
                pad_last=True
            )

            if len(windowed_data) == 0:
                logger.warning(f"  Subject {subject_id:02d} has no valid windows, skipping")
                continue

            # Split into ACC (0-2) and GYRO (3-5)
            for modality_name, ch_range in [('ACC', (0, 3)), ('GYRO', (3, 6))]:
                modality_data = windowed_data[:, :, ch_range[0]:ch_range[1]]

                # Apply scaling (ACC only, if scale_factor is defined)
                if modality_name == 'ACC' and self.scale_factor is not None:
                    modality_data = modality_data / self.scale_factor
                    logger.info(f"  Applied scale_factor={self.scale_factor} to Smartphone/{modality_name}")

                # Reshape: (N, T, C) -> (N, C, T)
                modality_data = np.transpose(modality_data, (0, 2, 1))

                # Convert to float16 (memory efficiency)
                modality_data = modality_data.astype(np.float16)

                sensor_modality_key = f"Smartphone/{modality_name}"
                processed[subject_id][sensor_modality_key] = {
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

        Save format:
            data/processed/tmd/USER00001/Smartphone/ACC/X.npy, Y.npy
        """
        logger.info("=" * 80)
        logger.info("Saving processed data")
        logger.info("=" * 80)

        import json

        base_path = self.processed_data_path / self.dataset_name
        base_path.mkdir(parents=True, exist_ok=True)

        total_stats = {
            'dataset': self.dataset_name,
            'num_activities': self.num_activities,
            'num_subjects': self.num_subjects,
            'num_sensors': 1,
            'sensor_names': self.sensor_names,
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

        # Save metadata (convert NumPy types to JSON-compatible)
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
        logger.info("=" * 80)
        logger.info(f"SUCCESS: Preprocessing completed -> {base_path}")
        logger.info("=" * 80)
