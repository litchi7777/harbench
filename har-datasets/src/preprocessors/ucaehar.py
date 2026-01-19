"""
UCA-EHAR (Université Côte d'Azur - Embedded Human Activity Recognition) Preprocessor

UCA-EHAR Dataset:
- 8 activities (walking, running, standing, sitting, lying, drinking, stair up/down)
  - Original data has 4 additional posture transitions, but these are excluded (labeled as -1)
- 20 subjects (T1-T21, T11 missing)
- Smart glasses sensors (accelerometer, gyroscope, barometer)
- Sampling rate: ~25Hz
- Data format: CSV (semicolon-separated)
- Reference: https://doi.org/10.3390/app12083849
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
import logging
import pandas as pd

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


# UCA-EHAR dataset URL
UCAEHAR_URL = "https://zenodo.org/records/5659336/files/UCA-EHAR-1.0.0.zip?download=1"


@register_preprocessor('ucaehar')
class UCAEHARPreprocessor(BasePreprocessor):
    """
    Preprocessor for UCA-EHAR dataset
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # UCA-EHAR specific settings
        # Note: Original dataset has 12 labels (8 activities + 4 posture transitions)
        # But the paper only considers 8 main activities, transitions are excluded
        self.num_activities = 8
        self.num_subjects = 20  # T1-T21, T11 missing

        # Sensor names (smart glasses only)
        self.sensor_names = ['SmartGlasses']

        # Modality settings
        self.modalities = ['ACC', 'GYRO', 'BAR']
        self.modality_column_ranges = {
            'ACC': ['Ax', 'Ay', 'Az'],   # 3-axis accelerometer
            'GYRO': ['Gx', 'Gy', 'Gz'],  # 3-axis gyroscope
            'BAR': ['P']                  # Barometer (1 channel)
        }

        # Label mapping (CSV CLASS column -> numeric)
        # Only 8 main activities are used (posture transitions are excluded as -1)
        self.activity_map = {
            'WALKING': 0,
            'RUNNING': 1,
            'STANDING': 2,
            'SITTING': 3,
            'LYING': 4,
            'DRINKING': 5,
            'WALKING_UPSTAIRS': 6,
            'WALKING_DOWNSTAIRS': 7,
            # Posture transitions are excluded (labeled as -1, filtered out)
            'STAND_TO_SIT': -1,
            'SIT_TO_STAND': -1,
            'SIT_TO_LIE': -1,
            'LIE_TO_SIT': -1
        }

        # Sampling rates
        self.original_sampling_rate = 25  # Hz (estimated from timestamps)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz (target)

        # Preprocessing parameters
        self.window_size = config.get('window_size', 150)  # 5 seconds @ 30Hz
        self.stride = config.get('stride', 30)  # 1 second @ 30Hz

        # Scaling factor (convert m/s^2 -> G)
        self.scale_factor = DATASETS.get('UCAEHAR', {}).get('scale_factor', None)

    def get_dataset_name(self) -> str:
        return 'ucaehar'

    def download_dataset(self) -> None:
        """
        Download and extract UCA-EHAR dataset
        """
        logger.info("=" * 80)
        logger.info("Downloading UCA-EHAR dataset")
        logger.info("=" * 80)

        ucaehar_raw_path = self.raw_data_path / self.dataset_name

        # Check if data already exists
        if check_dataset_exists(ucaehar_raw_path, required_files=['*_T*.csv']):
            logger.warning(f"UCA-EHAR data already exists at {ucaehar_raw_path}")
            response = input("Do you want to re-download? (y/N): ")
            if response.lower() != 'y':
                logger.info("Skipping download")
                return

        try:
            # 1. Download
            logger.info("Step 1/2: Downloading archive")
            ucaehar_raw_path.parent.mkdir(parents=True, exist_ok=True)
            zip_path = ucaehar_raw_path.parent / 'ucaehar.zip'
            download_file(UCAEHAR_URL, zip_path, desc='Downloading UCA-EHAR')

            # 2. Extract and organize data
            logger.info("Step 2/2: Extracting and organizing data")
            extract_to = ucaehar_raw_path.parent / 'ucaehar_temp'
            extract_archive(zip_path, extract_to, desc='Extracting UCA-EHAR')
            self._organize_ucaehar_data(extract_to, ucaehar_raw_path)

            # Cleanup
            cleanup_temp_files(extract_to)
            if zip_path.exists():
                zip_path.unlink()

            logger.info("=" * 80)
            logger.info(f"SUCCESS: UCA-EHAR dataset downloaded to {ucaehar_raw_path}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Failed to download UCA-EHAR dataset: {e}", exc_info=True)
            raise

    def _organize_ucaehar_data(self, extracted_path: Path, target_path: Path) -> None:
        """
        Organize UCA-EHAR data into proper directory structure

        Args:
            extracted_path: Path to extracted data
            target_path: Path to save organized data (data/raw/ucaehar)
        """
        import shutil

        logger.info(f"Organizing UCA-EHAR data from {extracted_path} to {target_path}")

        # Find the root of extracted data
        data_root = extracted_path

        # Look for UCA-EHAR-1.0.0 folder
        if (extracted_path / "UCA-EHAR-1.0.0").exists():
            data_root = extracted_path / "UCA-EHAR-1.0.0"

        # Create target directory
        target_path.mkdir(parents=True, exist_ok=True)

        # Copy CSV files
        csv_files = list(data_root.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_root}")

        for csv_file in csv_files:
            target_file = target_path / csv_file.name
            shutil.copy2(csv_file, target_file)

        logger.info(f"Copied {len(csv_files)} CSV files to {target_path}")

    def load_raw_data(self) -> Dict[int, list]:
        """
        Load UCA-EHAR raw data by subject

        Expected format:
        - data/raw/ucaehar/{ACTIVITY}_T{SUBJECT}.csv or {ACTIVITY}_T{SUBJECT}_{TRIAL}.csv
        - CSV columns: T;Ax;Ay;Az;Gx;Gy;Gz;P;CLASS

        Returns:
            person_data: {person_id: [(data, labels), ...]} dictionary
                Each CSV file (session) is kept separate
                data: (num_samples, 7) array [Ax, Ay, Az, Gx, Gy, Gz, P]
                labels: (num_samples,) array
        """
        raw_path = self.raw_data_path / self.dataset_name

        if not raw_path.exists():
            raise FileNotFoundError(
                f"UCA-EHAR raw data not found at {raw_path}\n"
                "Expected structure: data/raw/ucaehar/*.csv"
            )

        # Subject ID list (T11 is missing)
        subject_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

        # Store session list per subject
        from collections import defaultdict
        person_sessions = defaultdict(list)

        # Process all CSV files
        csv_files = sorted(raw_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {raw_path}")

        for csv_file in csv_files:
            # Extract subject ID from filename
            # Pattern: {ACTIVITY}_T{SUBJECT}.csv or {ACTIVITY}_T{SUBJECT}_{TRIAL}.csv
            filename = csv_file.stem  # without extension
            parts = filename.split('_')

            if len(parts) < 2:
                logger.warning(f"Unexpected filename format: {csv_file.name}")
                continue

            # Extract subject ID (T1, T2, ... -> 1, 2, ...)
            subject_part = parts[1]  # T1, T1_1, T1_2, etc.
            try:
                subject_id = int(subject_part.replace('T', '').split('_')[0])
            except ValueError:
                logger.warning(f"Could not parse subject ID from: {csv_file.name}")
                continue

            if subject_id not in subject_ids:
                logger.warning(f"Unknown subject ID {subject_id} in {csv_file.name}")
                continue

            try:
                # Load CSV (semicolon-separated)
                df = pd.read_csv(csv_file, sep=';')

                # Check required columns
                required_cols = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'P', 'CLASS']
                if not all(col in df.columns for col in required_cols):
                    logger.warning(f"Missing columns in {csv_file.name}: {set(required_cols) - set(df.columns)}")
                    continue

                # Extract sensor data [Ax, Ay, Az, Gx, Gy, Gz, P]
                sensor_data = df[['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'P']].values

                # Convert labels to numeric
                labels = df['CLASS'].map(self.activity_map).values

                # Check for unmapped labels
                if np.any(np.isnan(labels)):
                    unknown_labels = df['CLASS'][pd.isna(df['CLASS'].map(self.activity_map))].unique()
                    logger.warning(f"Unknown labels in {csv_file.name}: {unknown_labels}")
                    # Remove NaN
                    valid_mask = ~np.isnan(labels)
                    sensor_data = sensor_data[valid_mask]
                    labels = labels[valid_mask]

                labels = labels.astype(np.int32)

                # Add as session (do not concatenate)
                if len(sensor_data) > 0:
                    person_sessions[subject_id].append((sensor_data, labels))

            except Exception as e:
                logger.error(f"Error loading {csv_file}: {e}")
                continue

        # Return session list
        result = {}
        for person_id in subject_ids:
            if person_sessions[person_id]:
                result[person_id] = person_sessions[person_id]
                total_samples = sum(d.shape[0] for d, l in person_sessions[person_id])
                logger.info(f"USER{person_id:05d}: {len(person_sessions[person_id])} sessions, {total_samples} samples")
            else:
                logger.warning(f"No data loaded for USER{person_id:05d}")

        if not result:
            raise ValueError("No data loaded. Please check the raw data directory structure.")

        logger.info(f"Total users loaded: {len(result)}")
        return result

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
                    continue

                # Resampling (25Hz -> 30Hz)
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
        Feature extraction (windowing and normalization per sensor x modality)
        Do not create windows that cross session boundaries

        Args:
            data: {person_id: [(data, labels), ...]} dictionary
                data: (num_samples, 7) [Ax, Ay, Az, Gx, Gy, Gz, P]

        Returns:
            {person_id: {sensor_modality: {'X': data, 'Y': labels}}}
            Example: {'SmartGlasses/ACC': {'X': (N, 3, 150), 'Y': (N,)}}
        """
        processed = {}

        # Column indices
        col_indices = {
            'ACC': [0, 1, 2],   # Ax, Ay, Az
            'GYRO': [3, 4, 5],  # Gx, Gy, Gz
            'BAR': [6]          # P
        }

        for person_id, sessions in data.items():
            logger.info(f"Processing USER{person_id:05d} ({len(sessions)} sessions)")

            processed[person_id] = {}

            # Sensor name (SmartGlasses)
            sensor_name = self.sensor_names[0]

            # Process each modality
            for modality_name in self.modalities:
                # Extract modality columns (from each session)
                modality_cols = col_indices[modality_name]
                modality_sessions = [(d[:, modality_cols], l) for d, l in sessions]

                # Apply sliding window per session
                windowed_data, windowed_labels = create_sliding_windows_multi_session(
                    modality_sessions,
                    window_size=self.window_size,
                    stride=self.stride,
                    drop_last=False,
                    pad_last=True
                )

                if len(windowed_data) == 0:
                    logger.warning(f"  {sensor_name}/{modality_name}: no valid windows")
                    continue

                # windowed_data: (num_windows, window_size, channels)

                # Apply scaling (accelerometer only)
                if modality_name == 'ACC' and self.scale_factor is not None:
                    windowed_data = windowed_data / self.scale_factor
                    logger.info(f"  Applied scale_factor={self.scale_factor} to {sensor_name}/{modality_name}")

                # Transform shape: (num_windows, window_size, channels) -> (num_windows, channels, window_size)
                windowed_data = np.transpose(windowed_data, (0, 2, 1))

                # Convert to float16
                windowed_data = windowed_data.astype(np.float16)

                # Sensor/modality hierarchy
                sensor_modality_key = f"{sensor_name}/{modality_name}"

                processed[person_id][sensor_modality_key] = {
                    'X': windowed_data,
                    'Y': windowed_labels
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
            data/processed/ucaehar/USER00001/SmartGlasses/ACC/X.npy, Y.npy
            data/processed/ucaehar/USER00001/SmartGlasses/GYRO/X.npy, Y.npy
            data/processed/ucaehar/USER00001/SmartGlasses/BAR/X.npy, Y.npy
            ...
        """
        import json

        base_path = self.processed_data_path / self.dataset_name
        base_path.mkdir(parents=True, exist_ok=True)

        # Number of channels (per modality)
        channels_per_modality = {
            'ACC': 3,
            'GYRO': 3,
            'BAR': 1
        }

        total_stats = {
            'dataset': self.dataset_name,
            'num_activities': self.num_activities,
            'num_sensors': len(self.sensor_names),
            'sensor_names': self.sensor_names,
            'modalities': self.modalities,
            'channels_per_modality': channels_per_modality,
            'original_sampling_rate': self.original_sampling_rate,
            'target_sampling_rate': self.target_sampling_rate,
            'window_size': self.window_size,
            'stride': self.stride,
            'normalization': 'none',  # No normalization (raw data retained)
            'scale_factor': self.scale_factor,  # Scaling factor (applied to ACC only)
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
                X = arrays['X']
                Y = arrays['Y']

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
