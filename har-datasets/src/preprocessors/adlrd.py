"""
ADLRD (Dataset for ADL Recognition with Wrist-worn Accelerometer) Preprocessor

ADLRD Dataset:
- 14 types of ADL (Activities of Daily Living)
- 16 subjects
- One 3-axis accelerometer (right wrist)
- Sampling rate: 32Hz
- Measurement range: ±1.5g (6-bit resolution: 0-63)
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
import logging
import shutil

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


# ADLRD dataset URL
ADLRD_URL = "https://archive.ics.uci.edu/static/public/283/dataset+for+adl+recognition+with+wrist+worn+accelerometer.zip"


@register_preprocessor('adlrd')
class ADLRDPreprocessor(BasePreprocessor):
    """
    Preprocessor class for ADLRD dataset
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # ADLRD-specific configuration
        self.num_activities = 14
        self.num_channels = 3  # 3-axis accelerometer

        # Activity mapping (directory name -> label ID)
        self.activity_map = {
            'Brush_teeth': 0,
            'Climb_stairs': 1,
            'Comb_hair': 2,
            'Descend_stairs': 3,
            'Drink_glass': 4,
            'Eat_meat': 5,
            'Eat_soup': 6,
            'Getup_bed': 7,
            'Liedown_bed': 8,
            'Pour_water': 9,
            'Sitdown_chair': 10,
            'Standup_chair': 11,
            'Use_telephone': 12,
            'Walk': 13
        }

        # Sampling rate
        self.original_sampling_rate = 32  # Hz (ADLRD original)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz (target)

        # Sensor name
        self.sensor_name = 'RightWrist'

        # Modality
        self.modality = 'ACC'

        # Preprocessing parameters
        self.window_size = config.get('window_size', 150)  # 5 seconds @ 30Hz
        self.stride = config.get('stride', 30)  # 1 second @ 30Hz

        # Scaling factor (6-bit coded values: 0-63 = -1.5g to +1.5g -> G units)
        # Conversion formula: real_val = -1.5g + (coded_val/63) * 3g
        # First convert coded_val to the range -1.5 ~ +1.5, then to G units
        self.scale_factor = DATASETS.get('ADLRD', {}).get('scale_factor', None)

    def get_dataset_name(self) -> str:
        return 'adlrd'

    def download_dataset(self) -> None:
        """
        Download and extract ADLRD dataset
        """
        logger.info("=" * 80)
        logger.info("Downloading ADLRD dataset")
        logger.info("=" * 80)

        adlrd_raw_path = self.raw_data_path / self.dataset_name

        # Check if data already exists
        if check_dataset_exists(adlrd_raw_path, required_files=['*/Accelerometer-*.txt']):
            logger.warning(f"ADLRD data already exists at {adlrd_raw_path}")
            response = input("Do you want to re-download? (y/N): ")
            if response.lower() != 'y':
                logger.info("Skipping download")
                return

        try:
            # 1. Download
            logger.info("Step 1/2: Downloading archive")
            adlrd_raw_path.parent.mkdir(parents=True, exist_ok=True)
            zip_path = adlrd_raw_path.parent / 'adlrd.zip'
            download_file(ADLRD_URL, zip_path, desc='Downloading ADLRD')

            # 2. Extract and organize data
            logger.info("Step 2/2: Extracting and organizing data")
            extract_to = adlrd_raw_path.parent / 'adlrd_temp'
            extract_archive(zip_path, extract_to, desc='Extracting ADLRD')
            self._organize_adlrd_data(extract_to, adlrd_raw_path)

            # Cleanup
            cleanup_temp_files(extract_to)
            if zip_path.exists():
                zip_path.unlink()

            logger.info("=" * 80)
            logger.info(f"SUCCESS: ADLRD dataset downloaded to {adlrd_raw_path}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Failed to download ADLRD dataset: {e}", exc_info=True)
            raise

    def _organize_adlrd_data(self, extracted_path: Path, target_path: Path) -> None:
        """
        Organize ADLRD data into proper directory structure

        Args:
            extracted_path: Path to extracted data
            target_path: Path to save organized data (data/raw/adlrd)
        """
        logger.info(f"Organizing ADLRD data from {extracted_path} to {target_path}")

        # Find the root of extracted data
        data_root = extracted_path

        # Look for "HMP_Dataset" folder
        if (extracted_path / "HMP_Dataset").exists():
            data_root = extracted_path / "HMP_Dataset"

        # Create target directory
        target_path.mkdir(parents=True, exist_ok=True)

        if not data_root.exists():
            raise FileNotFoundError(f"Could not find HMP_Dataset directory in {extracted_path}")

        # Find and copy activity directories (except _MODEL)
        from tqdm import tqdm
        activity_dirs = [d for d in data_root.iterdir()
                        if d.is_dir() and not d.name.endswith('_MODEL')]

        if not activity_dirs:
            raise FileNotFoundError(f"No activity directories found in {data_root}")

        for activity_dir in tqdm(activity_dirs, desc='Organizing activities'):
            activity_name = activity_dir.name
            target_activity_dir = target_path / activity_name

            # Copy activity directory
            if target_activity_dir.exists():
                shutil.rmtree(target_activity_dir)
            shutil.copytree(activity_dir, target_activity_dir)

        logger.info(f"Data organized at: {target_path}")

    def load_raw_data(self) -> Dict[int, list]:
        """
        Load ADLRD raw data for each subject

        Expected format:
        - data/raw/adlrd/Brush_teeth/Accelerometer-2011-03-24-10-24-39-brush_teeth-f1.txt
        - Each file: (samples, 3) space-delimited text file
        - Values are 0-63 integers (6-bit)

        Returns:
            person_data: Dictionary of {person_id: [(data, labels), ...]}
                Each element is one session (file)
                data: Array of shape (num_samples, 3)
                labels: Array of shape (num_samples,)
        """
        raw_path = self.raw_data_path / self.dataset_name

        if not raw_path.exists():
            raise FileNotFoundError(
                f"ADLRD raw data not found at {raw_path}\n"
                "Expected structure: data/raw/adlrd/Brush_teeth/Accelerometer-*.txt"
            )

        # Collect volunteer IDs (extract from filenames)
        volunteer_ids = set()
        for activity_dir in raw_path.iterdir():
            if activity_dir.is_dir():
                for file_path in activity_dir.glob('Accelerometer-*.txt'):
                    # Extract volunteer ID from filename
                    # Example: Accelerometer-2011-03-24-10-24-39-brush_teeth-f1.txt -> f1
                    filename_parts = file_path.stem.split('-')
                    if len(filename_parts) >= 9:
                        volunteer_id = filename_parts[-1]
                        volunteer_ids.add(volunteer_id)

        volunteer_ids = sorted(volunteer_ids)
        logger.info(f"Found {len(volunteer_ids)} volunteers: {volunteer_ids}")

        # Map volunteer IDs to numeric person IDs (dictionary order)
        volunteer_to_person_id = {vid: i+1 for i, vid in enumerate(volunteer_ids)}

        # Store session list for each subject
        person_data = {person_id: []
                       for person_id in volunteer_to_person_id.values()}

        # For each activity
        for activity_name, activity_id in self.activity_map.items():
            activity_dir = raw_path / activity_name

            if not activity_dir.exists():
                logger.warning(f"Activity directory not found: {activity_dir}")
                continue

            # For each file
            txt_files = list(activity_dir.glob('Accelerometer-*.txt'))

            for file_path in txt_files:
                try:
                    # Extract volunteer ID from filename
                    filename_parts = file_path.stem.split('-')
                    if len(filename_parts) < 9:
                        logger.warning(f"Unexpected filename format: {file_path.name}")
                        continue

                    volunteer_id = filename_parts[-1]
                    if volunteer_id not in volunteer_to_person_id:
                        logger.warning(f"Unknown volunteer ID: {volunteer_id}")
                        continue

                    person_id = volunteer_to_person_id[volunteer_id]

                    # Load data (space-delimited)
                    data = np.loadtxt(file_path)

                    # Check data shape
                    if data.ndim == 1:
                        # If 1D, reshape to (samples, 3)
                        if len(data) % 3 == 0:
                            data = data.reshape(-1, 3)
                        else:
                            logger.warning(f"Cannot reshape {file_path.name}: {data.shape}")
                            continue

                    if data.shape[1] != self.num_channels:
                        logger.warning(
                            f"Unexpected number of channels in {file_path.name}: "
                            f"{data.shape[1]} (expected {self.num_channels})"
                        )
                        continue

                    # Generate labels (same label for all samples)
                    labels = np.full(len(data), activity_id, dtype=np.int32)

                    # Add as session to list (do not concatenate)
                    person_data[person_id].append((data, labels))

                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
                    continue

        # Organize results
        result = {}
        for person_id in sorted(person_data.keys()):
            if person_data[person_id]:
                sessions = person_data[person_id]
                total_samples = sum(len(s[0]) for s in sessions)
                unique_labels = set()
                for _, labels in sessions:
                    unique_labels.update(np.unique(labels).tolist())
                result[person_id] = sessions
                logger.info(f"USER{person_id:05d}: {len(sessions)} sessions, "
                           f"total_samples={total_samples}, unique_labels={sorted(unique_labels)}")
            else:
                logger.warning(f"No data loaded for USER{person_id:05d}")

        if not result:
            raise ValueError("No data loaded. Please check the raw data directory structure.")

        logger.info(f"Total users loaded: {len(result)}")
        return result

    def clean_data(self, data: Dict[int, list]) -> Dict[int, list]:
        """
        Data cleaning and resampling (process by session)

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

                # Resampling (32Hz -> 30Hz)
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
        Feature extraction (windowing and scaling)
        Apply windowing by session to avoid crossing session boundaries

        Args:
            data: Dictionary of {person_id: [(data, labels), ...]}

        Returns:
            {person_id: {sensor_modality: {'X': data, 'Y': labels}}}
            Example: {'RightWrist/ACC': {'X': (N, 3, 150), 'Y': (N,)}}
        """
        processed = {}

        for person_id, sessions in data.items():
            logger.info(f"Processing USER{person_id:05d} ({len(sessions)} sessions)")

            # Apply sliding windows by session
            windowed_data, windowed_labels = create_sliding_windows_multi_session(
                sessions,
                window_size=self.window_size,
                stride=self.stride,
                drop_last=False,
                pad_last=True
            )

            if len(windowed_data) == 0:
                logger.warning(f"  No windows created")
                continue

            # Apply scaling (6-bit coded values -> G units)
            # Conversion formula: real_val = -1.5 + (coded_val/63) * 3.0
            # First convert [0, 63] -> [-1.5, +1.5]
            windowed_data = -1.5 + (windowed_data / 63.0) * 3.0
            logger.info(f"  Applied ADLRD-specific scaling: coded_val -> G")

            # Transform shape: (num_windows, window_size, 3) -> (num_windows, 3, window_size)
            windowed_data = np.transpose(windowed_data, (0, 2, 1))

            # Convert to float16
            windowed_data = windowed_data.astype(np.float16)

            # Sensor/modality hierarchy
            sensor_modality_key = f"{self.sensor_name}/{self.modality}"

            processed[person_id] = {
                sensor_modality_key: {
                    'X': windowed_data,
                    'Y': windowed_labels
                }
            }

            logger.info(
                f"  {sensor_modality_key}: X.shape={windowed_data.shape}, "
                f"Y.shape={windowed_labels.shape}"
            )

        return processed

    def save_processed_data(self, data: Dict[int, Dict[str, Dict[str, np.ndarray]]]) -> None:
        """
        Save processed data

        Args:
            data: {person_id: {sensor_modality: {'X': data, 'Y': labels}}}

        Save format:
            data/processed/adlrd/USER00001/RightWrist/ACC/X.npy, Y.npy
        """
        import json

        base_path = self.processed_data_path / self.dataset_name
        base_path.mkdir(parents=True, exist_ok=True)

        total_stats = {
            'dataset': self.dataset_name,
            'num_activities': self.num_activities,
            'sensor_names': [self.sensor_name],
            'modalities': [self.modality],
            'channels_per_modality': self.num_channels,
            'original_sampling_rate': self.original_sampling_rate,
            'target_sampling_rate': self.target_sampling_rate,
            'window_size': self.window_size,
            'stride': self.stride,
            'normalization': 'none',  # No normalization (retain raw data)
            'scale_factor': 'ADLRD-specific: coded_val (0-63) -> G (-1.5 to +1.5)',
            'data_dtype': 'float16',
            'data_shape': f'(num_windows, {self.num_channels}, {self.window_size})',
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
