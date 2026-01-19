"""
REALDISP Activity Recognition Dataset Preprocessing

REALDISP (REAListic sensor DISPlacement) Dataset:
- 33 types of physical activities (warming up, cooling down, fitness exercises)
- 17 subjects
- 9 sensors (full-body worn)
- 3 scenarios (ideal-placement, self-placement, induced-displacement)
- Sampling rate: TBD (estimated from data)
- Each sensor: 3-axis accelerometer, 3-axis gyroscope, 3-axis magnetometer, 4D quaternion
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, List
import logging

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
    cleanup_temp_files,
    check_dataset_exists
)
from . import register_preprocessor
from ..dataset_info import DATASETS

logger = logging.getLogger(__name__)

# REALDISP dataset URL
REALDISP_URL = "https://archive.ics.uci.edu/static/public/305/realdisp+activity+recognition+dataset.zip"


@register_preprocessor('realdisp')
class RealDispPreprocessor(BasePreprocessor):
    """
    Preprocessing class for REALDISP dataset
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # REALDISP-specific settings
        self.num_activities = 33
        self.num_subjects = 17
        self.num_sensors = 9

        # Sensor names and body part mapping
        self.sensor_names = [
            'LeftCalf',      # S1
            'LeftThigh',     # S2
            'RightCalf',     # S3
            'RightThigh',    # S4
            'Back',          # S5
            'LeftLowerArm',  # S6
            'LeftUpperArm',  # S7
            'RightLowerArm', # S8
            'RightUpperArm'  # S9
        ]

        # Channel configuration for each sensor
        # ACC(3) + GYRO(3) + MAG(3) + QUAT(4) = 13 channels per sensor
        self.channels_per_sensor = 13

        # Modalities (channel division within each sensor)
        self.sensor_modalities = {
            'ACC': (0, 3),    # 3-axis accelerometer
            'GYRO': (3, 6),   # 3-axis gyroscope
            'MAG': (6, 9),    # 3-axis magnetometer
            'QUAT': (9, 13)   # 4D quaternion
        }

        # Sampling rate
        self.original_sampling_rate = DATASETS.get('REALDISP', {}).get('original_sampling_rate', 50)  # Hz (estimated)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz (target)

        # Preprocessing parameters
        self.window_size = config.get('window_size', 150)  # 5 seconds @ 30Hz
        self.stride = config.get('stride', 30)  # 1 second @ 30Hz

        # Scaling factor (set after confirming data units)
        self.scale_factor = DATASETS.get('REALDISP', {}).get('scale_factor', None)

        # Scenarios to process (default is ideal placement)
        self.scenarios = config.get('scenarios', ['ideal'])  # ['ideal', 'self', 'mutual']

    def get_dataset_name(self) -> str:
        return 'realdisp'

    def download_dataset(self) -> None:
        """
        Download and extract REALDISP dataset from UCI ML repository
        """
        logger.info("=" * 80)
        logger.info("Downloading REALDISP dataset")
        logger.info("=" * 80)

        realdisp_raw_path = self.raw_data_path / self.dataset_name

        # Check if data already exists
        if check_dataset_exists(realdisp_raw_path, required_files=['*.log']):
            logger.warning(f"REALDISP data already exists at {realdisp_raw_path}")
            response = input("Do you want to re-download? (y/N): ")
            if response.lower() != 'y':
                logger.info("Skipping download")
                return

        try:
            # 1. Download
            logger.info("Step 1/2: Downloading archive (2.5 GB, may take a while)")
            realdisp_raw_path.parent.mkdir(parents=True, exist_ok=True)
            zip_path = realdisp_raw_path.parent / 'realdisp.zip'
            download_file(REALDISP_URL, zip_path, desc='Downloading REALDISP')

            # 2. Extract and organize data
            logger.info("Step 2/2: Extracting and organizing data")
            extract_to = realdisp_raw_path.parent / 'realdisp_temp'
            extract_archive(zip_path, extract_to, desc='Extracting REALDISP')
            self._organize_realdisp_data(extract_to, realdisp_raw_path)

            # Cleanup
            cleanup_temp_files(extract_to)
            if zip_path.exists():
                zip_path.unlink()

            logger.info("=" * 80)
            logger.info(f"SUCCESS: REALDISP dataset downloaded to {realdisp_raw_path}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Failed to download REALDISP dataset: {e}", exc_info=True)
            raise

    def _organize_realdisp_data(self, extracted_path: Path, target_path: Path) -> None:
        """
        Organize REALDISP data into proper directory structure

        Args:
            extracted_path: Path to extracted data
            target_path: Target path for organized data (data/raw/realdisp)
        """
        import shutil

        logger.info(f"Organizing REALDISP data from {extracted_path} to {target_path}")

        # Find the root of extracted data
        # Explore ZIP file structure
        log_files = list(extracted_path.rglob('*.log'))

        if not log_files:
            raise FileNotFoundError(
                f"No .log files found in extracted archive at {extracted_path}"
            )

        logger.info(f"Found {len(log_files)} log files")

        # Create target directory
        target_path.mkdir(parents=True, exist_ok=True)

        # Copy all log files in flat structure
        for log_file in log_files:
            dest_file = target_path / log_file.name
            shutil.copy2(log_file, dest_file)
            logger.info(f"  Copied: {log_file.name}")

        logger.info(f"Organized {len(log_files)} files to {target_path}")

    def _estimate_sampling_rate(self, timestamps: np.ndarray) -> float:
        """
        Estimate sampling rate from timestamps

        Args:
            timestamps: Timestamp array (in seconds)

        Returns:
            Estimated sampling rate (Hz)
        """
        # Estimate sampling rate from median of time differences
        time_diffs = np.diff(timestamps)
        median_diff = np.median(time_diffs)

        if median_diff > 0:
            sampling_rate = 1.0 / median_diff
            return sampling_rate
        else:
            logger.warning("Could not estimate sampling rate from timestamps")
            return 50.0  # Default value

    def load_raw_data(self) -> Dict[int, list]:
        """
        Load REALDISP raw data by subject
        Treat each scenario as a separate session

        Expected format:
        - data/raw/realdisp/*.log
        - Filename: subjectXX_scenario.log

        Returns:
            person_data: Dictionary {person_id: [(data, labels), ...]}
                Each element is one scenario session
                data: Array of shape (num_samples, num_sensors * 13)
                labels: Array of shape (num_samples,) (0-indexed)
        """
        raw_path = self.raw_data_path / self.dataset_name

        if not raw_path.exists():
            raise FileNotFoundError(
                f"REALDISP raw data not found at {raw_path}\n"
                "Expected structure: data/raw/realdisp/*.log\n"
                "Please download the dataset first using --download flag or manually."
            )

        # Search for log files
        log_files = sorted(raw_path.glob('*.log'))

        if not log_files:
            raise FileNotFoundError(f"No .log files found in {raw_path}")

        logger.info(f"Found {len(log_files)} log files")

        # Store sessions by subject
        result = {}

        for log_file in log_files:
            # Extract subject ID and scenario from filename
            # Example: subject15_mutual5.log -> subject=15, scenario=mutual
            filename = log_file.stem  # Filename without extension

            try:
                # Parse filename
                parts = filename.split('_')
                if len(parts) < 2:
                    logger.warning(f"Unexpected filename format: {filename}")
                    continue

                # Extract subject number ("subject15" -> 15)
                subject_str = parts[0]
                if not subject_str.startswith('subject'):
                    continue

                person_id = int(subject_str.replace('subject', ''))

                # Extract scenario ("mutual5" -> "mutual", "ideal" -> "ideal")
                scenario_str = parts[1]
                if scenario_str.startswith('mutual'):
                    scenario = 'mutual'
                elif scenario_str.startswith('self'):
                    scenario = 'self'
                elif scenario_str.startswith('ideal'):
                    scenario = 'ideal'
                else:
                    logger.warning(f"Unknown scenario in {filename}: {scenario_str}")
                    continue

                # Process only scenarios specified in config
                if scenario not in self.scenarios:
                    logger.info(f"Skipping {filename} (scenario '{scenario}' not in config)")
                    continue

                logger.info(f"Loading {filename} (Subject={person_id}, Scenario={scenario})")

                # Load data
                df = pd.read_csv(log_file, sep=r'\s+', header=None)

                if len(df.columns) != 120:
                    logger.warning(
                        f"Unexpected number of columns in {log_file}: "
                        f"{len(df.columns)} (expected 120)"
                    )
                    continue

                # Timestamps (seconds + microseconds)
                timestamps = df.iloc[:, 0].values + df.iloc[:, 1].values / 1e6

                # Estimate sampling rate (first file only)
                if self.original_sampling_rate is None:
                    self.original_sampling_rate = self._estimate_sampling_rate(timestamps)
                    logger.info(f"Estimated sampling rate: {self.original_sampling_rate:.2f} Hz")

                # Extract sensor data (columns 2-118)
                sensor_data = df.iloc[:, 2:119].values.astype(np.float32)

                # Extract labels (column 119, 1-indexed A1-A33 -> 0-indexed 0-32)
                labels = df.iloc[:, 119].values.astype(int) - 1

                # Initialize list per person_id and add session
                if person_id not in result:
                    result[person_id] = []

                result[person_id].append((sensor_data, labels))
                logger.info(
                    f"  USER{person_id:05d} ({scenario}): "
                    f"{sensor_data.shape}, Labels: {labels.shape}"
                )

            except Exception as e:
                logger.error(f"Error loading {log_file}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if not result:
            raise ValueError("No data loaded. Please check the raw data directory structure.")

        # Output result summary
        for person_id, sessions in result.items():
            total_samples = sum(len(s[0]) for s in sessions)
            logger.info(f"USER{person_id:05d}: {len(sessions)} sessions, {total_samples} samples")

        logger.info(f"Total users loaded: {len(result)}")
        return result

    def clean_data(self, data: Dict[int, list]) -> Dict[int, list]:
        """
        Data cleaning and resampling (process by session)

        Args:
            data: Dictionary {person_id: [(data, labels), ...]}

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

                # Resampling
                if self.original_sampling_rate and self.original_sampling_rate != self.target_sampling_rate:
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
        Feature extraction (windowing by sensor × modality)
        Window by session and avoid crossing session boundaries

        Args:
            data: Dictionary {person_id: [(data, labels), ...]}

        Returns:
            {person_id: {sensor/modality: {'X': data, 'Y': labels}}}
            Example: {'LeftCalf/ACC': {'X': (N, 3, 150), 'Y': (N,)}}
        """
        processed = {}

        for person_id, sessions in data.items():
            logger.info(f"Processing USER{person_id:05d} ({len(sessions)} sessions)")

            processed[person_id] = {}

            # Process each sensor
            for sensor_idx, sensor_name in enumerate(self.sensor_names):
                # Calculate channel range for sensor
                sensor_start_ch = sensor_idx * self.channels_per_sensor
                sensor_end_ch = sensor_start_ch + self.channels_per_sensor

                # Extract sensor data per session
                sensor_sessions = [
                    (session_data[:, sensor_start_ch:sensor_end_ch], session_labels)
                    for session_data, session_labels in sessions
                ]

                # Apply sliding window by session
                windowed_data, windowed_labels = create_sliding_windows_multi_session(
                    sensor_sessions,
                    window_size=self.window_size,
                    stride=self.stride,
                    drop_last=False,
                    pad_last=True
                )

                if len(windowed_data) == 0:
                    logger.warning(f"  {sensor_name}: no windows created")
                    continue

                # Split into each modality
                for modality_name, (mod_start_ch, mod_end_ch) in self.sensor_modalities.items():
                    # Extract modality channels
                    modality_data = windowed_data[:, :, mod_start_ch:mod_end_ch]

                    # Apply scaling (accelerometer only)
                    if self.scale_factor is not None and modality_name == 'ACC':
                        modality_data = modality_data / self.scale_factor
                        logger.info(
                            f"  Applied scale_factor={self.scale_factor} to "
                            f"{sensor_name}/{modality_name}"
                        )

                    # Transform shape: (num_windows, window_size, C) -> (num_windows, C, window_size)
                    modality_data = np.transpose(modality_data, (0, 2, 1))

                    # Convert to float16
                    modality_data = modality_data.astype(np.float16)

                    # Hierarchical structure of sensor/modality
                    sensor_modality_key = f"{sensor_name}/{modality_name}"

                    processed[person_id][sensor_modality_key] = {
                        'X': modality_data,
                        'Y': windowed_labels
                    }

            # Output statistics
            for sensor_modality_key, arrays in processed[person_id].items():
                X = arrays['X']
                Y = arrays['Y']
                logger.info(
                    f"  {sensor_modality_key}: X.shape={X.shape}, Y.shape={Y.shape}"
                )

        return processed

    def save_processed_data(self, data: Dict[int, Dict[str, Dict[str, np.ndarray]]]) -> None:
        """
        Save processed data

        Args:
            data: {person_id: {sensor_modality: {'X': data, 'Y': labels}}}

        Save format:
            data/processed/realdisp/USER00001/LeftCalf/ACC/X.npy, Y.npy
            data/processed/realdisp/USER00001/LeftCalf/GYRO/X.npy, Y.npy
            data/processed/realdisp/USER00001/LeftCalf/MAG/X.npy, Y.npy
            data/processed/realdisp/USER00001/LeftCalf/QUAT/X.npy, Y.npy
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
            'normalization': 'none',  # No normalization (raw data preserved)
            'scale_factor': self.scale_factor,
            'data_dtype': 'float16',
            'scenarios': self.scenarios,
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
                    f"Saved {user_name}/{sensor_modality_name}: X{X.shape}, Y{Y.shape}"
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
