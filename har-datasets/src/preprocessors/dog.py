"""
Daphnet Freezing of Gait Dataset Preprocessing

Daphnet Dataset:
- Freezing of Gait (FoG) detection in Parkinson's disease patients
- 10 subjects
- 3 sensors (ankle, thigh, trunk)
- Sampling rate: 64Hz
- Accelerometer only (3-axis, mg unit)

Data format:
- Each file: 11 columns (Time, Ankle_X/Y/Z, Thigh_X/Y/Z, Trunk_X/Y/Z, Annotation)
- Acceleration unit: mg (milli-G)
- Labels: 0=non-experiment, 1=No Freeze, 2=Freeze

Reference: https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait
"""

import numpy as np
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
    check_dataset_exists,
    cleanup_temp_files
)
from . import register_preprocessor
from ..dataset_info import DATASETS

logger = logging.getLogger(__name__)

# Daphnet Dataset URL
DAPHNET_URL = "https://archive.ics.uci.edu/static/public/245/daphnet+freezing+of+gait.zip"


@register_preprocessor('dog')
class DogPreprocessor(BasePreprocessor):
    """
    Preprocessor class for Daphnet Freezing of Gait dataset
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Daphnet specific settings
        self.num_activities = 2  # No Freeze, Freeze
        self.num_subjects = 10
        self.num_sensors = 3
        self.num_channels = 9  # 3 sensors × 3 axes

        # Sensor names and channel mapping
        # Data file column structure:
        # 0: Time (ms)
        # 1-3: Ankle (horizontal forward, vertical, lateral)
        # 4-6: Thigh (horizontal forward, vertical, lateral)
        # 7-9: Trunk (horizontal forward, vertical, lateral)
        # 10: Annotation
        self.sensor_names = ['Ankle', 'Thigh', 'Trunk']
        self.sensor_channel_ranges = {
            'Ankle': (0, 3),   # channels 0-2
            'Thigh': (3, 6),   # channels 3-5
            'Trunk': (6, 9)    # channels 6-8
        }

        # Modalities (channel division within each sensor)
        self.sensor_modalities = {
            'Ankle': {
                'ACC': (0, 3)   # 3-axis acceleration
            },
            'Thigh': {
                'ACC': (0, 3)   # 3-axis acceleration
            },
            'Trunk': {
                'ACC': (0, 3)   # 3-axis acceleration
            }
        }

        # Sampling rate
        self.original_sampling_rate = 64  # Hz (Daphnet original)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz (target)

        # Preprocessing parameters
        self.window_size = config.get('window_size', 150)  # 5 seconds @ 30Hz
        self.stride = config.get('stride', 30)  # 1 second @ 30Hz

        # Scaling factor (mg -> G conversion)
        self.scale_factor = DATASETS.get('DOG', {}).get('scale_factor', 1000.0)

        # Label mapping (original label -> 0-indexed)
        # 0 -> -1 (NonExperiment: setup/debriefing)
        # 1 -> 0 (No Freeze: during experiment, no freeze)
        # 2 -> 1 (Freeze: Freezing of Gait event)
        self.label_mapping = {
            0: -1,  # NonExperiment
            1: 0,   # NoFreeze
            2: 1    # Freeze
        }

    def get_dataset_name(self) -> str:
        return 'dog'

    def download_dataset(self) -> None:
        """
        Download and extract Daphnet dataset
        """
        logger.info("=" * 80)
        logger.info("Downloading Daphnet Freezing of Gait dataset")
        logger.info("=" * 80)

        daphnet_raw_path = self.raw_data_path / self.dataset_name

        # Check if data already exists
        if check_dataset_exists(daphnet_raw_path, required_files=['S*.txt']):
            logger.warning(f"Daphnet data already exists at {daphnet_raw_path}")
            try:
                response = input("Do you want to re-download? (y/N): ")
                if response.lower() != 'y':
                    logger.info("Skipping download")
                    return
            except EOFError:
                logger.info("Non-interactive mode: Skipping download")
                return

        try:
            # 1. Download
            logger.info("Step 1/2: Downloading archive")
            daphnet_raw_path.parent.mkdir(parents=True, exist_ok=True)
            zip_path = daphnet_raw_path.parent / 'daphnet.zip'
            download_file(DAPHNET_URL, zip_path, desc='Downloading Daphnet')

            # 2. Extract and organize data
            logger.info("Step 2/2: Extracting and organizing data")
            extract_to = daphnet_raw_path.parent / 'daphnet_temp'
            extract_archive(zip_path, extract_to, desc='Extracting Daphnet')
            self._organize_daphnet_data(extract_to, daphnet_raw_path)

            # Cleanup
            cleanup_temp_files(extract_to)
            if zip_path.exists():
                zip_path.unlink()

            logger.info("=" * 80)
            logger.info(f"SUCCESS: Daphnet dataset downloaded to {daphnet_raw_path}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Failed to download Daphnet dataset: {e}", exc_info=True)
            raise

    def _organize_daphnet_data(self, extracted_path: Path, target_path: Path) -> None:
        """
        Organize Daphnet data into appropriate directory structure

        Args:
            extracted_path: Path to extracted data
            target_path: Target path for organized data (data/raw/daphnet)
        """
        logger.info(f"Organizing Daphnet data from {extracted_path} to {target_path}")

        # Find the root of extracted data
        data_root = extracted_path

        # Look for "dataset_fog_release/dataset" folder
        if (extracted_path / "dataset_fog_release" / "dataset").exists():
            data_root = extracted_path / "dataset_fog_release" / "dataset"
        elif (extracted_path / "dataset").exists():
            data_root = extracted_path / "dataset"
        elif (extracted_path / "daphnet+freezing+of+gait").exists():
            data_root = extracted_path / "daphnet+freezing+of+gait"
            if (data_root / "dataset").exists():
                data_root = data_root / "dataset"

        # Create target directory
        target_path.mkdir(parents=True, exist_ok=True)

        if not data_root.exists():
            raise FileNotFoundError(f"Could not find data directory in {extracted_path}")

        # Find and copy S*.txt files
        import shutil
        txt_files = list(data_root.glob("S*.txt"))

        if not txt_files:
            # Try alternative structure
            for subdir in data_root.iterdir():
                if subdir.is_dir():
                    txt_files.extend(list(subdir.glob("S*.txt")))

        if not txt_files:
            raise FileNotFoundError(f"No S*.txt files found in {data_root}")

        from tqdm import tqdm
        for txt_file in tqdm(txt_files, desc='Organizing data files'):
            target_file = target_path / txt_file.name
            shutil.copy2(txt_file, target_file)

        logger.info(f"Data organized at: {target_path}")
        logger.info(f"Found {len(txt_files)} data files")

    def load_raw_data(self) -> Dict[int, list]:
        """
        Load Daphnet raw data per subject

        Expected format:
        - data/raw/dog/S01R01.txt, S01R02.txt, ...
        - Each file: 11 columns (Time, Ankle_xyz, Thigh_xyz, Trunk_xyz, Annotation)
        - S01 = Subject 1, R01 = Recording 1

        Returns:
            person_data: {person_id: [(data, labels), ...]} dictionary
                Each element is one recording session (file)
                data: (num_samples, 9) array [ankle_xyz, thigh_xyz, trunk_xyz]
                labels: (num_samples,) array (0-indexed)
        """
        raw_path = self.raw_data_path / self.dataset_name

        if not raw_path.exists():
            raise FileNotFoundError(
                f"Daphnet raw data not found at {raw_path}\n"
                "Expected structure: data/raw/dog/S01R01.txt"
            )

        # Store data per subject
        person_data = {}

        # Find files in format S01R01.txt, S01R02.txt, ...
        txt_files = sorted(raw_path.glob("S*.txt"))

        if not txt_files:
            raise FileNotFoundError(
                f"No data files found in {raw_path}\n"
                "Expected files like: S01R01.txt, S01R02.txt, ..."
            )

        # Group files by subject ID
        subject_files = {}
        for txt_file in txt_files:
            # Extract Subject ID from filename (e.g., S01R01.txt -> 1)
            filename = txt_file.stem
            if filename.startswith('S') and len(filename) >= 3:
                try:
                    subject_id = int(filename[1:3])
                    if subject_id not in subject_files:
                        subject_files[subject_id] = []
                    subject_files[subject_id].append(txt_file)
                except ValueError:
                    logger.warning(f"Could not parse subject ID from {filename}")
                    continue

        logger.info(f"Found {len(subject_files)} subjects")

        # Load data for each subject (keep sessions separate)
        for subject_id in sorted(subject_files.keys()):
            files = subject_files[subject_id]
            sessions = []  # List of sessions

            for txt_file in sorted(files):
                try:
                    # Load data (space-separated)
                    raw_data = np.loadtxt(txt_file)

                    if raw_data.shape[1] != 11:
                        logger.warning(
                            f"Unexpected number of columns in {txt_file}: "
                            f"{raw_data.shape[1]} (expected 11)"
                        )
                        continue

                    # Extract sensor data (columns 1-9: Ankle, Thigh, Trunk)
                    sensor_data = raw_data[:, 1:10].astype(np.float32)

                    # Extract and map labels (column 10: Annotation)
                    original_labels = raw_data[:, 10].astype(int)
                    labels = np.array([self.label_mapping.get(l, -1) for l in original_labels], dtype=int)

                    sessions.append((sensor_data, labels))

                    logger.debug(f"Loaded {txt_file.name}: {sensor_data.shape}")

                except Exception as e:
                    logger.error(f"Error loading {txt_file}: {e}")
                    continue

            if sessions:
                person_data[subject_id] = sessions
                total_samples = sum(len(s[0]) for s in sessions)
                total_freeze = sum(np.sum(s[1] == 1) for s in sessions)
                logger.info(
                    f"USER{subject_id:05d}: {len(sessions)} sessions, "
                    f"total_samples={total_samples}, freeze_samples={total_freeze}"
                )

        if not person_data:
            raise ValueError("No data loaded. Please check the raw data directory structure.")

        logger.info(f"Total users loaded: {len(person_data)}")
        return person_data

    def clean_data(self, data: Dict[int, list]) -> Dict[int, list]:
        """
        Data cleaning and resampling (process per session)

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
                    logger.warning(f"USER{person_id:05d}: session skipped (no valid samples)")
                    continue

                # Resampling (64Hz -> 30Hz)
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
        Apply windowing per session to avoid crossing session boundaries

        Args:
            data: {person_id: [(data, labels), ...]} dictionary

        Returns:
            {person_id: {sensor/modality: {'X': data, 'Y': labels}}}
            e.g., {'Ankle/ACC': {'X': (N, 3, 150), 'Y': (N,)}}
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

                # Split into modalities (Daphnet has ACC only)
                modalities = self.sensor_modalities[sensor_name]
                for modality_name, (mod_start_ch, mod_end_ch) in modalities.items():
                    # Extract modality channels
                    modality_data = windowed_data[:, :, mod_start_ch:mod_end_ch]
                    # modality_data: (num_windows, window_size, channels)

                    # Apply scaling (mg -> G)
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
            data/processed/daphnet/USER00001/Ankle/ACC/X.npy, Y.npy
            data/processed/daphnet/USER00001/Thigh/ACC/X.npy, Y.npy
            data/processed/daphnet/USER00001/Trunk/ACC/X.npy, Y.npy
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
            'scale_factor': self.scale_factor,  # Scaling factor (mg -> G)
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
            # Convert NumPy types to JSON-compatible
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
