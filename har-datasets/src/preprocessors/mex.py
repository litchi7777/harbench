"""
MEx (Multi-modal Exercise) dataset preprocessing

MEx dataset:
- 7 types of physiotherapy exercises
- 30 subjects
- 2 Accelerometers (wrist, thigh)
- Sampling rate: 100Hz
- Accelerometer (3-axis, ±8g, G units)
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

# MEX dataset download URL
MEX_URL = "https://archive.ics.uci.edu/static/public/500/mex.zip"


@register_preprocessor('mex')
class MexPreprocessor(BasePreprocessor):
    """
    Preprocessing class for MEX dataset
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # MEx-specific settings
        self.num_activities = 7
        self.num_subjects = 30
        self.num_sensors = 2
        self.num_channels = 6  # 2 sensors × 3 axes

        # Sensor names and channel mapping
        # Channel configuration:
        # Wrist: ACC(3) = 3
        # Thigh: ACC(3) = 3
        self.sensor_names = ['Wrist', 'Thigh']
        self.sensor_channel_ranges = {
            'Wrist': (0, 3),   # channels 0-2
            'Thigh': (3, 6)    # channels 3-5
        }

        # Modalities (channel division within each sensor)
        self.sensor_modalities = {
            'Wrist': {
                'ACC': (0, 3)   # 3-axis acceleration
            },
            'Thigh': {
                'ACC': (0, 3)   # 3-axis acceleration
            }
        }

        # Sampling rate
        self.original_sampling_rate = 100  # Hz (MEX original)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz (target)

        # Preprocessing parameters
        self.window_size = config.get('window_size', 150)  # 5 seconds @ 30Hz
        self.stride = config.get('stride', 30)  # 1 second @ 30Hz

        # Scaling factor (not needed since already in G units)
        self.scale_factor = DATASETS.get('MEX', {}).get('scale_factor', None)

    def get_dataset_name(self) -> str:
        return 'mex'

    def download_dataset(self) -> None:
        """
        Download and extract MEX dataset
        """
        logger.info("=" * 80)
        logger.info("Downloading MEX dataset")
        logger.info("=" * 80)

        dataset_path = self.raw_data_path / self.dataset_name

        # Check if data already exists
        if check_dataset_exists(dataset_path, required_files=['act/*/01_act_1.csv']):
            logger.warning(f"MEX data already exists at {dataset_path}")
            response = input("Do you want to re-download? (y/N): ")
            if response.lower() != 'y':
                logger.info("Skipping download")
                return

        try:
            # 1. Download
            logger.info("Step 1/3: Downloading archive")
            zip_path = self.raw_data_path / "mex.zip"
            download_file(MEX_URL, zip_path, desc='Downloading MEX')

            # 2. Extract outer ZIP file (to temporary directory)
            logger.info("Step 2/3: Extracting outer archive")
            temp_dir = self.raw_data_path / "mex_temp"
            extract_archive(zip_path, temp_dir, desc='Extracting MEX outer archive')

            # 3. Check and extract nested data.zip
            logger.info("Step 3/3: Extracting nested data.zip")
            nested_zip = temp_dir / "data.zip"
            if nested_zip.exists():
                dataset_path.mkdir(parents=True, exist_ok=True)
                extract_archive(nested_zip, dataset_path, desc='Extracting MEX data')
                nested_zip.unlink()
            else:
                logger.error(f"data.zip not found in {temp_dir}")
                raise FileNotFoundError(f"Expected data.zip in {temp_dir}")

            # Cleanup
            if zip_path.exists():
                zip_path.unlink()
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)

            logger.info(f"MEX dataset downloaded and extracted to {dataset_path}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Failed to download MEX dataset: {e}")
            raise

    def load_raw_data(self) -> Dict[int, list]:
        """
        Load MEX raw data per subject

        Expected format:
        - data/raw/mex/MEx/acw/{subject_id}/{exercise_id}.txt (wrist accelerometer)
        - data/raw/mex/MEx/act/{subject_id}/{exercise_id}.txt (thigh accelerometer)
        - Each file: timestamp x y z (space-separated)
        - Exercise ID: 1-7 (exercise 4 performed twice: 4L.txt, 4R.txt)
        - Labels: 0-6 (0-indexed)

        Returns:
            person_data: Dictionary of {person_id: [(data, labels), ...]}
                Each element is one exercise session
                data: Array of shape (num_samples, 6) [wrist_xyz, thigh_xyz]
                labels: Array of shape (num_samples,) (0-indexed)
        """
        raw_path = self.raw_data_path / self.dataset_name

        if not raw_path.exists():
            raise FileNotFoundError(
                f"MEx raw data not found at {raw_path}\n"
                "Expected structure: data/raw/mex/acw/, data/raw/mex/act/"
            )

        acw_path = raw_path / "acw"  # wrist accelerometer
        act_path = raw_path / "act"  # thigh accelerometer

        if not acw_path.exists() or not act_path.exists():
            raise FileNotFoundError(
                f"MEx sensor folders not found\n"
                f"Expected: {acw_path} and {act_path}"
            )

        # Store data per subject
        result = {}

        # Subject IDs are 1-30 (folder names are 01, 02, ... 30 with zero-padding)
        for subject_id in range(1, 31):
            subject_str = f"{subject_id:02d}"  # 01, 02, ... 30
            subject_acw_dir = acw_path / subject_str
            subject_act_dir = act_path / subject_str

            if not subject_acw_dir.exists() or not subject_act_dir.exists():
                logger.warning(f"Subject {subject_id} folders not found, skipping")
                continue

            # Keep each exercise as a session (don't merge)
            sessions = []

            # Load exercise files (1-7)
            # Filename format: {subject_id}_act_{ex_id}.csv and {subject_id}_acw_{ex_id}.csv
            # Exercise 4 has 2 files (04_act_1.csv, 04_act_2.csv)
            exercise_files = []
            for ex_id in range(1, 8):
                if ex_id == 4:
                    # Exercise 4 is performed twice (_1 and _2)
                    exercise_files.append((f"{ex_id:02d}_act_1.csv", f"{ex_id:02d}_acw_1.csv", ex_id - 1))
                    exercise_files.append((f"{ex_id:02d}_act_2.csv", f"{ex_id:02d}_acw_2.csv", ex_id - 1))
                else:
                    exercise_files.append((f"{ex_id:02d}_act_1.csv", f"{ex_id:02d}_acw_1.csv", ex_id - 1))

            for thigh_filename, wrist_filename, label in exercise_files:
                wrist_file = subject_acw_dir / wrist_filename
                thigh_file = subject_act_dir / thigh_filename

                if not wrist_file.exists() or not thigh_file.exists():
                    logger.debug(f"Subject {subject_id}, files {wrist_filename}/{thigh_filename} not found, skipping")
                    continue

                try:
                    # Load wrist acceleration data (comma-separated, timestamp,x,y,z)
                    wrist_df = pd.read_csv(wrist_file, header=None, names=['timestamp', 'x', 'y', 'z'])
                    wrist_data = wrist_df[['x', 'y', 'z']].values.astype(np.float32)

                    # Load thigh acceleration data (comma-separated, timestamp,x,y,z)
                    thigh_df = pd.read_csv(thigh_file, header=None, names=['timestamp', 'x', 'y', 'z'])
                    thigh_data = thigh_df[['x', 'y', 'z']].values.astype(np.float32)

                    # Align sample counts (truncate to shorter length)
                    min_samples = min(len(wrist_data), len(thigh_data))
                    wrist_data = wrist_data[:min_samples]
                    thigh_data = thigh_data[:min_samples]

                    # Merge wrist and thigh: (num_samples, 6)
                    sensor_data = np.hstack([wrist_data, thigh_data])

                    # Generate labels
                    exercise_labels = np.full(min_samples, label, dtype=int)

                    # Add as session
                    sessions.append((sensor_data, exercise_labels))

                except Exception as e:
                    logger.error(f"Error loading subject {subject_id}, file {wrist_filename}: {e}")
                    continue

            if not sessions:
                logger.warning(f"No data loaded for subject {subject_id}")
                continue

            # person_id is 1-indexed (starting from USER00001)
            person_id = subject_id
            result[person_id] = sessions
            total_samples = sum(len(s[0]) for s in sessions)
            logger.info(
                f"USER{person_id:05d}: {len(sessions)} sessions, total_samples={total_samples}"
            )

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
                total_samples = sum(len(s[0]) for s in cleaned_sessions)
                logger.info(f"USER{person_id:05d} cleaned: {len(cleaned_sessions)} sessions, {total_samples} samples")

        return cleaned

    def extract_features(self, data: Dict[int, list]) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
        """
        Feature extraction (windowing per sensor × modality)
        Windowing is done per session to avoid crossing session boundaries

        Args:
            data: Dictionary of {person_id: [(data, labels), ...]}

        Returns:
            {person_id: {sensor/modality: {'X': data, 'Y': labels}}}
            Example: {'Wrist/ACC': {'X': (N, 3, 150), 'Y': (N,)}}
        """
        processed = {}

        for person_id, sessions in data.items():
            logger.info(f"Processing USER{person_id:05d} ({len(sessions)} sessions)")

            processed[person_id] = {}

            # Process each sensor
            for sensor_name in self.sensor_names:
                sensor_start_ch, sensor_end_ch = self.sensor_channel_ranges[sensor_name]

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

                if len(windowed_data) == 0:
                    logger.warning(f"  {sensor_name}: no windows created")
                    continue

                # Split into each modality (MEx has ACC only)
                modalities = self.sensor_modalities[sensor_name]
                for modality_name, (mod_start_ch, mod_end_ch) in modalities.items():
                    # Extract modality channels
                    modality_data = windowed_data[:, :, mod_start_ch:mod_end_ch]

                    # Scaling is not needed (already in G units)
                    if self.scale_factor is not None:
                        modality_data = modality_data / self.scale_factor
                        logger.info(f"  Applied scale_factor={self.scale_factor} to {sensor_name}/{modality_name}")

                    # Transform shape: (num_windows, window_size, C) -> (num_windows, C, window_size)
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
            data/processed/mex/USER00001/Wrist/ACC/X.npy, Y.npy
            data/processed/mex/USER00001/Thigh/ACC/X.npy, Y.npy
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
            'scale_factor': self.scale_factor,  # Scaling factor (None since already in G units)
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
