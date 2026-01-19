"""
CAPTURE-24 (Large-scale daily living HAR) Dataset Preprocessing

CAPTURE-24 Dataset:
- 151 participants wearing device for ~24 hours
- Axivity AX3 wrist-worn accelerometer
- Sampling rate: 100Hz (3-axis acceleration)
- 200+ daily living activities (Compendium of Physical Activity)
- Label schema: Walmsley2020 or WillettsSpecific2018
- Reference: https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
import logging
import zipfile
import urllib.request as urllib
from tqdm.auto import tqdm

from .base import BasePreprocessor
from .utils import (
    create_sliding_windows,
    filter_invalid_samples,
    resample_timeseries,
    get_class_distribution
)
from .common import (
    cleanup_temp_files,
    check_dataset_exists
)
from . import register_preprocessor
from ..dataset_info import DATASETS

logger = logging.getLogger(__name__)


# CAPTURE-24 Dataset URL
CAPTURE24_URL = "https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001" + \
                "/download_file?file_format=&safe_filename=capture24.zip&type_of_work=Dataset"

# Data file and annotation file patterns
DATAFILES_PATTERN = 'P[0-9][0-9][0-9].csv.gz'  # P001.csv.gz ... P151.csv.gz
ANNOFILE = 'annotation-label-dictionary.csv'


@register_preprocessor('capture24')
class Capture24Preprocessor(BasePreprocessor):
    """
    Preprocessor class for CAPTURE-24 dataset

    Features:
    - 151 participants, ~4000 hours total
    - Wrist-worn accelerometer (100Hz, 3-axis)
    - 200+ activities from Compendium of Physical Activity
    - Labels: Walmsley2020 or WillettsSpecific2018 schema
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # CAPTURE-24 specific settings
        self.num_participants = 151
        self.sensor_names = ['Wrist']  # Wrist-worn only

        # Label schema selection
        self.label_schema = config.get('label_schema', 'WillettsSpecific2018')  # or 'Walmsley2020'
        logger.info(f"Using label schema: {self.label_schema}")

        # Subsetting (optional: for memory/time savings)
        self.max_participants = config.get('max_participants', None)  # None = all 151
        if self.max_participants:
            logger.info(f"Limiting to first {self.max_participants} participants")

        # Modalities
        self.sensor_modalities = {
            'Wrist': {
                'ACC': (0, 3),  # 3-axis acceleration (x, y, z)
            }
        }

        # Sampling rate
        self.original_sampling_rate = 100  # Hz (Axivity AX3)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz

        # Preprocessing parameters
        self.window_size = config.get('window_size', 150)  # 5 seconds @ 30Hz
        self.stride = config.get('stride', 30)  # 1 second @ 30Hz (80% overlap)

        # Scaling factor (set after dataset verification)
        self.scale_factor = DATASETS.get('CAPTURE24', {}).get('scale_factor', None)

        # Label mapping (loaded dynamically)
        self.label_to_id = {}
        self.id_to_label = {}
        self.anno_df = None  # annotation-label-dictionary.csv

    def get_dataset_name(self) -> str:
        return 'capture24'

    def download_dataset(self) -> None:
        """
        Download and extract CAPTURE-24 dataset

        Warning: Large file over 6.5GB
        """
        zip_path = self.raw_data_path / 'capture24.zip'
        extract_dir = self.raw_data_path / 'capture24'

        # Check if already downloaded
        if extract_dir.exists():
            csv_files = list(extract_dir.glob(DATAFILES_PATTERN))
            anno_file = extract_dir / ANNOFILE

            if len(csv_files) >= 151 and anno_file.exists():
                logger.info(f"CAPTURE-24 data already exists at {extract_dir}")
                return

        # Download
        if not zip_path.exists():
            logger.info(f"Downloading CAPTURE-24 dataset (6.5GB+)...")
            logger.info(f"URL: {CAPTURE24_URL}")
            logger.info("This may take 10-30 minutes depending on your connection.")

            with tqdm(total=6.9e9, unit="B", unit_scale=True, unit_divisor=1024,
                      miniters=1, ascii=True, desc="Downloading capture24.zip") as pbar:
                urllib.urlretrieve(
                    CAPTURE24_URL,
                    filename=str(zip_path),
                    reporthook=lambda b, bsize, tsize: pbar.update(bsize)
                )

            logger.info(f"Download completed: {zip_path}")
        else:
            logger.info(f"Using existing zip file: {zip_path}")

        # Extract
        extract_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Extracting CAPTURE-24 data...")
        with zipfile.ZipFile(zip_path, "r") as f:
            for member in tqdm(f.namelist(), desc="Unzipping"):
                try:
                    f.extract(member, self.raw_data_path)
                except zipfile.error as e:
                    logger.warning(f"Error extracting {member}: {e}")

        logger.info(f"Extraction completed: {extract_dir}")

    def load_annotation_dictionary(self) -> pd.DataFrame:
        """
        Load annotation-label-dictionary.csv

        Returns:
            DataFrame with columns: annotation, label:Walmsley2020, label:WillettsSpecific2018, etc.
        """
        anno_file = self.raw_data_path / 'capture24' / ANNOFILE

        if not anno_file.exists():
            raise FileNotFoundError(
                f"Annotation dictionary not found: {anno_file}\n"
                "Please run download_dataset() first."
            )

        # Load CSV (with annotation as index)
        anno_df = pd.read_csv(anno_file, index_col='annotation', dtype=str)
        logger.info(f"Loaded annotation dictionary: {anno_file}")
        logger.info(f"  Available schemas: {[c for c in anno_df.columns if c.startswith('label:')]}")

        return anno_df

    def build_label_mapping(self, anno_df: pd.DataFrame) -> None:
        """
        Build label_to_id and id_to_label from selected label schema

        Args:
            anno_df: annotation-label-dictionary DataFrame
        """
        schema_col = f'label:{self.label_schema}'

        if schema_col not in anno_df.columns:
            raise ValueError(
                f"Schema '{self.label_schema}' not found in annotation dictionary.\n"
                f"Available: {anno_df.columns.tolist()}"
            )

        # Get unique labels (excluding NA)
        unique_labels = anno_df[schema_col].dropna().unique()
        unique_labels = sorted(unique_labels)

        # label -> id mapping
        self.label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        self.label_to_id['NA'] = -1  # Undefined class

        # id -> label mapping
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}

        logger.info(f"Built label mapping for schema '{self.label_schema}':")
        logger.info(f"  Number of classes: {len(unique_labels)}")
        logger.info(f"  Labels: {unique_labels[:10]}{'...' if len(unique_labels) > 10 else ''}")

    def load_participant_data(self, participant_file: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data for one participant

        Args:
            participant_file: P###.csv.gz file

        Returns:
            data: (num_samples, 3) - x, y, z acceleration
            labels: (num_samples,) - label IDs
        """
        # Load CSV
        df = pd.read_csv(
            participant_file,
            index_col='time',
            parse_dates=['time'],
            dtype={'x': 'f4', 'y': 'f4', 'z': 'f4', 'annotation': 'string'}
        )

        # Acceleration data
        acc_data = df[['x', 'y', 'z']].to_numpy()

        # Convert annotations to label IDs
        annotations = df['annotation'].to_numpy()
        schema_col = f'label:{self.label_schema}'

        labels = np.full(len(annotations), -1, dtype=np.int32)
        for i, anno in enumerate(annotations):
            if pd.isna(anno):
                labels[i] = -1
            elif anno in self.anno_df.index:
                label_str = self.anno_df.loc[anno, schema_col]
                if pd.isna(label_str):
                    labels[i] = -1
                else:
                    labels[i] = self.label_to_id.get(label_str, -1)
            else:
                labels[i] = -1

        return acc_data, labels

    def load_raw_data(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Load data for all participants

        Returns:
            {participant_id: (data, labels)} dictionary
                data: (num_samples, 3) - acceleration
                labels: (num_samples,) - label IDs
        """
        capture24_dir = self.raw_data_path / 'capture24'

        # Load annotation dictionary
        self.anno_df = self.load_annotation_dictionary()
        self.build_label_mapping(self.anno_df)

        # Get participant files
        participant_files = sorted(capture24_dir.glob(DATAFILES_PATTERN))

        if len(participant_files) == 0:
            raise FileNotFoundError(
                f"No participant files found in {capture24_dir}\n"
                "Please run download_dataset() first."
            )

        # Subsetting
        if self.max_participants:
            participant_files = participant_files[:self.max_participants]

        logger.info(f"Loading {len(participant_files)} participants...")

        all_data = {}
        for pfile in tqdm(participant_files, desc="Loading participants"):
            # Participant ID (P001 -> USER00001)
            pid_num = int(pfile.stem.split('.')[0][1:])  # P001.csv.gz -> 1
            pid = f"USER{pid_num:05d}"  # -> USER00001

            try:
                data, labels = self.load_participant_data(pfile)
                all_data[pid] = (data, labels)
                logger.info(f"{pid}: data={data.shape}, labels={labels.shape}, "
                           f"unique_labels={np.unique(labels)}")
            except Exception as e:
                logger.error(f"Error loading {pid}: {e}")

        logger.info(f"Loaded {len(all_data)} participants successfully")
        return all_data

    def clean_data(
        self,
        data: Dict[str, Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Data cleaning and resampling
        """
        cleaned = {}

        for pid, (person_data, labels) in data.items():
            # Remove NaN
            valid_mask = ~np.isnan(person_data).any(axis=1)
            cleaned_data = person_data[valid_mask]
            cleaned_labels = labels[valid_mask]

            # Resampling (100Hz -> 30Hz)
            if self.original_sampling_rate != self.target_sampling_rate:
                resampled_data, resampled_labels = resample_timeseries(
                    cleaned_data,
                    cleaned_labels,
                    self.original_sampling_rate,
                    self.target_sampling_rate
                )
                cleaned[pid] = (resampled_data, resampled_labels)
                logger.info(f"{pid} cleaned and resampled: {resampled_data.shape}")
            else:
                cleaned[pid] = (cleaned_data, cleaned_labels)
                logger.info(f"{pid} cleaned: {cleaned_data.shape}")

        return cleaned

    def extract_features(
        self,
        data: Dict[str, Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        """
        Feature extraction (windowing and scaling per sensor × modality)

        Returns:
            {participant_id: {sensor/modality: {'X': data, 'Y': labels}}}
        """
        processed = {}

        for pid, (person_data, labels) in data.items():
            logger.info(f"Processing {pid}")
            processed[pid] = {}

            # Wrist sensor only
            sensor_name = 'Wrist'
            sensor_data = person_data  # Already (N, 3)

            # Apply sliding window
            windowed_data, windowed_labels = create_sliding_windows(
                sensor_data,
                labels,
                window_size=self.window_size,
                stride=self.stride,
                drop_last=False,
                pad_last=True
            )

            # ACC modality
            modality_name = 'ACC'
            modality_data = windowed_data  # (num_windows, window_size, 3)

            # Apply scaling (ACC only, if scale_factor is defined)
            if self.scale_factor is not None:
                modality_data = modality_data / self.scale_factor
                logger.info(f"  Applied scale_factor={self.scale_factor} to {sensor_name}/{modality_name}")

            # Transpose shape: (N, T, C) -> (N, C, T)
            modality_data = np.transpose(modality_data, (0, 2, 1))

            # Convert to float16 (memory efficiency)
            modality_data = modality_data.astype(np.float16)

            sensor_modality_key = f"{sensor_name}/{modality_name}"
            processed[pid][sensor_modality_key] = {
                'X': modality_data,
                'Y': windowed_labels
            }

            logger.info(f"  {sensor_modality_key}: X.shape={modality_data.shape}, "
                       f"Y.shape={windowed_labels.shape}")

        return processed

    def save_processed_data(
        self,
        data: Dict[str, Dict[str, Dict[str, np.ndarray]]]
    ) -> None:
        """
        Save processed data

        Save format:
            data/processed/capture24/P001/Wrist/ACC/X.npy, Y.npy
        """
        import json

        base_path = self.processed_data_path / self.dataset_name
        base_path.mkdir(parents=True, exist_ok=True)

        total_stats = {
            'dataset': self.dataset_name,
            'num_participants': len(data),
            'label_schema': self.label_schema,
            'num_classes': len(self.label_to_id) - 1,  # -1 to exclude NA
            'sensor_names': self.sensor_names,
            'original_sampling_rate': self.original_sampling_rate,
            'target_sampling_rate': self.target_sampling_rate,
            'window_size': self.window_size,
            'stride': self.stride,
            'normalization': 'none',
            'scale_factor': self.scale_factor,
            'data_dtype': 'float16',
            'labels': self.id_to_label,
            'participants': {}
        }

        for pid, sensor_modality_data in data.items():
            participant_name = pid  # Already in USER00001 format
            participant_path = base_path / participant_name
            participant_path.mkdir(parents=True, exist_ok=True)

            participant_stats = {'sensor_modalities': {}}

            for sensor_modality_name, arrays in sensor_modality_data.items():
                sensor_modality_path = participant_path / sensor_modality_name
                sensor_modality_path.mkdir(parents=True, exist_ok=True)

                X = arrays['X']
                Y = arrays['Y']

                np.save(sensor_modality_path / 'X.npy', X)
                np.save(sensor_modality_path / 'Y.npy', Y)

                participant_stats['sensor_modalities'][sensor_modality_name] = {
                    'X_shape': X.shape,
                    'Y_shape': Y.shape,
                    'num_windows': len(Y),
                    'class_distribution': get_class_distribution(Y)
                }

                logger.info(f"Saved {participant_name}/{sensor_modality_name}: "
                           f"X{X.shape}, Y{Y.shape}")

            total_stats['participants'][participant_name] = participant_stats

        # Save metadata (convert NumPy types to JSON compatible)
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
        logger.info(f"Preprocessing completed: {base_path}")
