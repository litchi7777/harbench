"""
SelfBack (Self-management support for low back pain) preprocessing

SelfBack dataset:
- 9 types of daily activities (3 types of walking, 2 types of stairs, jogging, sitting, standing, lying)
- 33 subjects
- 2 acceleration sensors (3-axis each): Wrist + Thigh
- Sampling rate: 100Hz
- Acceleration unit: G (±8g range, already in G units)
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


# SelfBack dataset URL
SELFBACK_URL = "https://archive.ics.uci.edu/static/public/521/selfback.zip"


@register_preprocessor('selfback')
class SelfBackPreprocessor(BasePreprocessor):
    """
    Preprocessing class for SelfBack dataset
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # SelfBack-specific configuration
        self.num_activities = 9
        self.num_subjects = 33
        self.num_sensors = 2  # Wrist + Thigh
        self.channels_per_sensor = 3  # 3-axis acc
        self.num_channels = 6  # 2 sensors × 3 channels

        # Sampling rate
        self.original_sampling_rate = 100  # Hz (SelfBack original)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz (target)

        # Sensor names and channel mapping
        self.sensor_names = ['Wrist', 'Thigh']
        self.sensor_channel_ranges = {
            'Wrist': (0, 3),  # channels 0-2
            'Thigh': (3, 6),  # channels 3-5
        }

        # Modality (each sensor has ACC only)
        self.modalities = ['ACC']
        self.modality_channel_ranges = {
            'ACC': (0, 3),  # 3-axis acceleration
        }
        self.channels_per_modality = 3

        # Preprocessing parameters
        self.window_size = config.get('window_size', 150)  # 5 seconds @ 30Hz
        self.stride = config.get('stride', 30)  # 1 second @ 30Hz

        # Scale factor (SelfBack is already in G units, so None)
        self.scale_factor = DATASETS.get('SELFBACK', {}).get('scale_factor', None)

        # Activity name mapping
        self.activity_names = {
            'downstairs': 0,
            'upstairs': 1,
            'walk_slow': 2,
            'walk_mod': 3,
            'walk_fast': 4,
            'jogging': 5,
            'sitting': 6,
            'standing': 7,
            'lying': 8,
        }

        # User ID remapping (initialized in load_raw_data)
        self.user_id_mapping = {}

    def get_dataset_name(self) -> str:
        return 'selfback'

    def download_dataset(self) -> None:
        """
        Download and extract SelfBack dataset
        """
        logger.info("=" * 80)
        logger.info("Downloading SelfBack dataset")
        logger.info("=" * 80)

        selfback_raw_path = self.raw_data_path / self.dataset_name

        # Check if data already exists
        if check_dataset_exists(selfback_raw_path, required_files=['w/*/*.csv', 't/*/*.csv']):
            logger.warning(f"SelfBack data already exists at {selfback_raw_path}")
            response = input("Do you want to re-download? (y/N): ")
            if response.lower() != 'y':
                logger.info("Skipping download")
                return

        try:
            # 1. Download
            logger.info("Step 1/2: Downloading archive")
            selfback_raw_path.parent.mkdir(parents=True, exist_ok=True)
            zip_path = selfback_raw_path.parent / 'selfback.zip'
            download_file(SELFBACK_URL, zip_path, desc='Downloading SelfBack')

            # 2. Extract and organize data
            logger.info("Step 2/2: Extracting and organizing data")
            extract_to = selfback_raw_path.parent / 'selfback_temp'
            extract_archive(zip_path, extract_to, desc='Extracting SelfBack')
            self._organize_selfback_data(extract_to, selfback_raw_path)

            # Cleanup
            cleanup_temp_files(extract_to)
            if zip_path.exists():
                zip_path.unlink()

            logger.info("=" * 80)
            logger.info(f"SUCCESS: SelfBack dataset downloaded to {selfback_raw_path}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Failed to download SelfBack dataset: {e}", exc_info=True)
            raise

    def _organize_selfback_data(self, extracted_path: Path, target_path: Path) -> None:
        """
        Organize SelfBack data into appropriate directory structure

        Args:
            extracted_path: Path to extracted data
            target_path: Path to save organized data (data/raw/selfback)
        """
        import shutil
        from tqdm import tqdm

        logger.info(f"Organizing SelfBack data from {extracted_path} to {target_path}")

        # Find selfBACK folder
        selfback_root = extracted_path / "selfBACK"
        if not selfback_root.exists():
            # If extracted directly
            selfback_root = extracted_path

        if not selfback_root.exists():
            raise FileNotFoundError(f"Could not find selfBACK directory in {extracted_path}")

        # Create target directory
        target_path.mkdir(parents=True, exist_ok=True)

        # Copy w, t directories
        for sensor_dir in ['w', 't']:
            source_dir = selfback_root / sensor_dir
            if source_dir.exists():
                target_sensor_dir = target_path / sensor_dir
                if target_sensor_dir.exists():
                    shutil.rmtree(target_sensor_dir)
                shutil.copytree(source_dir, target_sensor_dir)
                logger.info(f"Copied {sensor_dir}/ directory")

        logger.info(f"Data organized at: {target_path}")

    def load_raw_data(self) -> Dict[int, list]:
        """
        Load SelfBack raw data per subject

        Expected format:
        - data/raw/selfback/w/{activity}/{subject_id}.csv (wrist sensor)
        - data/raw/selfback/t/{activity}/{subject_id}.csv (thigh sensor)
        - Each file: time,x,y,z (with CSV header)

        Returns:
            person_data: Dictionary of {person_id: [(data, labels), ...]}
                Each element is one activity session
                data: (num_samples, 6) array [wrist_x, wrist_y, wrist_z, thigh_x, thigh_y, thigh_z]
                labels: (num_samples,) array
        """
        raw_path = self.raw_data_path / self.dataset_name

        if not raw_path.exists():
            raise FileNotFoundError(
                f"SelfBack raw data not found at {raw_path}\n"
                "Expected structure: data/raw/selfback/w/{activity}/{subject_id}.csv"
            )

        # Extract subject IDs (from w directory)
        w_dir = raw_path / 'w'
        if not w_dir.exists():
            raise FileNotFoundError(f"Wrist sensor directory not found: {w_dir}")

        # Get subject ID list from first activity directory (skip hidden/metadata files)
        first_activity = [d for d in w_dir.iterdir() if d.is_dir() and not d.name.startswith('.')][0]
        subject_files = [f for f in first_activity.glob("*.csv") if not f.name.startswith('._')]
        original_subject_ids = sorted([int(f.stem) for f in subject_files])

        logger.info(f"Found {len(original_subject_ids)} subjects: {min(original_subject_ids)}-{max(original_subject_ids)}")

        # Create user ID mapping (original ID → sequential ID)
        # Example: 26 -> 1, 27 -> 2, ..., 63 -> 33
        self.user_id_mapping = {
            original_id: new_id
            for new_id, original_id in enumerate(original_subject_ids, start=1)
        }
        logger.info(f"User ID mapping created: {min(original_subject_ids)} -> 1, {max(original_subject_ids)} -> {len(original_subject_ids)}")

        # Store sessions per subject (using mapped IDs)
        person_data = {}

        for original_id in original_subject_ids:
            mapped_id = self.user_id_mapping[original_id]
            sessions = []

            # For each activity
            for activity_name, activity_id in self.activity_names.items():
                w_file = w_dir / activity_name / f"{original_id:03d}.csv"
                t_file = (raw_path / 't') / activity_name / f"{original_id:03d}.csv"

                if not w_file.exists() or not t_file.exists():
                    logger.warning(f"Missing files for USER{mapped_id:05d} (original ID: {original_id}), activity {activity_name}")
                    continue

                try:
                    # Load wrist sensor data
                    w_data = pd.read_csv(w_file)
                    # Load thigh sensor data
                    t_data = pd.read_csv(t_file)

                    # Extract only x,y,z columns excluding timestamp
                    w_values = w_data[['x', 'y', 'z']].values
                    t_values = t_data[['x', 'y', 'z']].values

                    # Align lengths (use shorter length)
                    min_len = min(len(w_values), len(t_values))
                    w_values = w_values[:min_len]
                    t_values = t_values[:min_len]

                    # Merge: [wrist_x, wrist_y, wrist_z, thigh_x, thigh_y, thigh_z]
                    merged_data = np.hstack([w_values, t_values])  # (N, 6)

                    # Generate labels
                    labels = np.full(len(merged_data), activity_id)

                    # Add as session (do not concatenate)
                    sessions.append((merged_data, labels))

                except Exception as e:
                    logger.error(f"Error loading USER{mapped_id:05d} (original ID: {original_id}), activity {activity_name}: {e}")
                    continue

            if sessions:
                person_data[mapped_id] = sessions

        # Organize results
        result = {}
        for original_id in original_subject_ids:
            mapped_id = self.user_id_mapping[original_id]
            if mapped_id in person_data and person_data[mapped_id]:
                sessions = person_data[mapped_id]
                result[mapped_id] = sessions
                total_samples = sum(len(s[0]) for s in sessions)
                logger.info(f"USER{mapped_id:05d} (original ID: {original_id}): {len(sessions)} sessions, {total_samples} samples")
            else:
                logger.warning(f"No data loaded for USER{mapped_id:05d} (original ID: {original_id})")

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
        Window per session without crossing session boundaries

        Args:
            data: Dictionary of {person_id: [(data, labels), ...]}

        Returns:
            {person_id: {sensor_modality: {'X': data, 'Y': labels}}}
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

                # Split by modality (SelfBack has ACC only)
                for modality_name in self.modalities:
                    mod_start_ch, mod_end_ch = self.modality_channel_ranges[modality_name]

                    # Extract modality channels
                    modality_data = windowed_data[:, :, mod_start_ch:mod_end_ch]

                    # Apply scaling (SelfBack is already in G units, so scale_factor=None)
                    if modality_name == 'ACC' and self.scale_factor is not None:
                        modality_data = modality_data / self.scale_factor
                        logger.info(f"  Applied scale_factor={self.scale_factor} to {sensor_name}/{modality_name}")

                    # Transform shape: (num_windows, window_size, 3) -> (num_windows, 3, window_size)
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
            data/processed/selfback/USER00001/Wrist/ACC/X.npy, Y.npy
            data/processed/selfback/USER00001/Thigh/ACC/X.npy, Y.npy
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
            'normalization': 'none',  # No normalization (retain raw data)
            'scale_factor': self.scale_factor,  # SelfBack is already in G units, so None
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
