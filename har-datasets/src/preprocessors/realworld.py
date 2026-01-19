"""
RealWorld (HAR) Dataset Preprocessing

RealWorld Dataset:
- 8 activities (climbing down/up, jumping, lying, running, sitting, standing, walking)
- 15 subjects (8 male, 7 female)
- 7 body positions (chest, forearm, head, shin, thigh, upper arm, waist)
- Sensors: accelerometer, gyroscope, magnetometer, GPS, light, sound
- Sampling rate: 50Hz

Reference:
    Sztyler, T. and Stuckenschmidt, H. (2016).
    "On-body localization of wearable devices: An investigation of position-aware activity recognition."
    In 2016 IEEE International Conference on Pervasive Computing and Communications (PerCom), pp. 1-9.
    https://doi.org/10.1109/PERCOM.2016.7456521

URL: https://www.uni-mannheim.de/dws/research/projects/activity-recognition/dataset/dataset-realworld/
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import logging
import zipfile
import requests
from tqdm import tqdm

from .base import BasePreprocessor
from .utils import (
    create_sliding_windows,
    filter_invalid_samples,
    resample_timeseries,
    get_class_distribution
)
from . import register_preprocessor
from ..dataset_info import DATASETS

logger = logging.getLogger(__name__)


@register_preprocessor('realworld')
class RealWorldPreprocessor(BasePreprocessor):
    """
    Preprocessor class for RealWorld dataset
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # RealWorld specific settings
        self.num_activities = 8
        self.num_subjects = 15
        self.num_sensors = 7

        # Sensor names (following RealWorld naming conventions)
        # Actual data uses lowercase (head, chest, ...), so we use lower() here
        self.sensor_names = ['Chest', 'Forearm', 'Head', 'Shin', 'Thigh', 'UpperArm', 'Waist']

        # Modalities (actual data has acc/gyr/mag/gps, but we only handle 3 here)
        self.modality_names = ['ACC', 'GYR', 'MAG']

        # Each modality has 3 axes (x, y, z)
        self.channels_per_modality = 3

        # Sampling rate
        self.original_sampling_rate = 50  # Hz (RealWorld original)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz (target)

        # Preprocessing parameters
        self.window_size = config.get('window_size', 150)  # 5 seconds @ 30Hz
        self.stride = config.get('stride', 30)  # 1 second @ 30Hz

        # Scaling factor (determined after data inspection)
        self.scale_factor = DATASETS.get('REALWORLD', {}).get('scale_factor', None)

        # Activity name mapping (directory name → label ID)
        # RealWorld dataset activity names are lowercase
        self.activity_mapping = {
            'climbingdown': 0,
            'climbingup': 1,
            'jumping': 2,
            'lying': 3,
            'running': 4,
            'sitting': 5,
            'standing': 6,
            'walking': 7
        }

        # Download URL
        self.download_url = "http://wifo5-14.informatik.uni-mannheim.de/sensor/dataset/realworld2016/realworld2016_dataset.zip"

    def get_dataset_name(self) -> str:
        return 'realworld'

    def download_dataset(self) -> None:
        """
        Download RealWorld dataset
        """
        download_path = self.raw_data_path / f"{self.dataset_name}.zip"
        extract_path = self.raw_data_path / self.dataset_name

        # Check if already downloaded
        if extract_path.exists() and any(extract_path.iterdir()):
            logger.info(f"Dataset already exists at {extract_path}")
            return

        # Download
        logger.info(f"Downloading RealWorld dataset from {self.download_url}")
        try:
            response = requests.get(self.download_url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(download_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading') as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            logger.info(f"Downloaded to {download_path}")

            # Extract
            logger.info(f"Extracting {download_path} to {extract_path}")
            extract_path.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(download_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)

            logger.info("Download and extraction completed")

        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            raise

    def _load_sensor_data(self, proband_path: Path, sensor: str, modality: str) -> Optional[Dict[str, List[pd.DataFrame]]]:
        """
        Loader adapted to the actual RealWorld2016 distribution format
        (CSV files per sensor inside activity-specific zip files)

        Example structure:
            proband1/data/acc_climbingdown_csv.zip
                ├── acc_climbingdown_head.csv
                ├── acc_climbingdown_waist.csv
                └── ...

        Args:
            proband_path: Subject directory path (…/proband1)
            sensor: Sensor name (e.g., 'Chest')
            modality: Modality name (e.g., 'ACC')

        Returns:
            Dictionary with activity names as keys and DataFrame lists as values
        """
        sensor_lower = sensor.lower()     # chest, head, ...
        modality_lower = modality.lower() # acc, gyr, mag

        data_dir = proband_path / 'data'
        if not data_dir.exists():
            logger.warning(f"Data dir not found: {data_dir}")
            return None

        # Look for all "acc_***_csv.zip" files in proband1/data
        pattern = f"{modality_lower}_*_csv.zip"
        zip_files = sorted(data_dir.glob(pattern))

        if not zip_files:
            logger.warning(f"No {modality_lower} zip files found in {data_dir}")
            return None

        activity_chunks: Dict[str, List[pd.DataFrame]] = {}

        for zip_path in zip_files:
            # Extract activity from zip filename: acc_climbingdown_csv.zip → climbingdown
            # ["acc", "climbingdown", "csv"]
            parts = zip_path.stem.split('_')
            activity = None
            if len(parts) >= 2:
                activity = parts[1]  # climbingdown, walking, ...

            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    # Read only CSVs for this sensor from the zip
                    # Example: acc_climbingdown_head.csv
                    target_suffix = f"_{sensor_lower}.csv"  # "_head.csv"
                    for member in zf.namelist():
                        name_lower = member.lower()
                        if not member.lower().endswith(".csv"):
                            continue
                        if not member.lower().endswith(target_suffix):
                            continue

                        if modality_lower == 'gyr':
                            if not ('gyro' in name_lower or 'gyroscope' in name_lower):
                                continue
                        elif modality_lower == 'acc':
                            if not ('acc' in name_lower or 'accelerometer' in name_lower):
                                continue

                        with zf.open(member) as fp:
                            df = pd.read_csv(fp)

                        # Handle column name variations
                        # RealWorld mostly uses attr_time, attr_x, attr_y, attr_z
                        # but also has timestamp, x, y, z
                        # Return df as-is and let caller extract x, y, z columns
                        if activity:
                            df["activity"] = activity
                        activity_chunks.setdefault(activity or "unknown", []).append(df)

            except Exception as e:
                logger.warning(f"Error reading zip {zip_path}: {e}")
                continue

        if not activity_chunks:
            logger.warning(
                f"No CSV found for sensor={sensor_lower}, modality={modality_lower} in {data_dir}"
            )
            return None

        combined_chunks: Dict[str, List[pd.DataFrame]] = {}
        for activity_name, dfs in activity_chunks.items():
            combined_chunks[activity_name] = [pd.concat(dfs, ignore_index=True)]

        return combined_chunks

    def load_raw_data(self) -> Dict[int, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]:
        """
        Load RealWorld raw data for each subject

        Expected format:
        - data/raw/realworld/realworld2016_dataset/proband1/
        - data/raw/realworld/realworld2016_dataset/proband2/

        Returns:
            {person_id: {sensor: {modality: (data, labels)}}}
        """
        # Dataset root path
        dataset_root = self.raw_data_path / self.dataset_name / 'realworld2016_dataset'

        if not dataset_root.exists():
            # Case where proband folders are directly under realworld2016
            dataset_root = self.raw_data_path / self.dataset_name

        if not dataset_root.exists():
            raise FileNotFoundError(
                f"RealWorld raw data not found at {dataset_root}\n"
                "Expected structure: data/raw/realworld/realworld2016_dataset/proband1/"
            )

        fuse_sources = ('ACC', 'GYR')
        result = {}

        # Load proband1 to proband15
        for person_id in range(1, self.num_subjects + 1):
            proband_path = dataset_root / f'proband{person_id}'

            if not proband_path.exists():
                logger.warning(f"Proband {person_id} not found at {proband_path}")
                continue

            logger.info(f"Loading USER{person_id:05d} from {proband_path.name}")

            person_entry: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}

            # Process each sensor
            for sensor in self.sensor_names:
                sensor_data_store: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
                modality_chunks: Dict[str, Dict[str, List[pd.DataFrame]]] = {}

                for modality in self.modality_names:
                    activity_chunks = self._load_sensor_data(proband_path, sensor, modality)

                    if not activity_chunks:
                        logger.warning(
                            f"USER{person_id:05d}/{sensor}/{modality}: No data loaded"
                        )
                        continue

                    modality_chunks[modality] = activity_chunks

                    # Save standalone modalities (e.g., magnetometer) as before
                    if modality not in fuse_sources:
                        combined_df = pd.concat(
                            [fragment for fragments in activity_chunks.values() for fragment in fragments],
                            ignore_index=True
                        )
                        extracted = self._extract_xyz_data_with_labels(combined_df)
                        if extracted is None:
                            continue
                        sensor_data, labels = extracted
                        sensor_data_store[modality] = (sensor_data, labels)
                        logger.info(
                            f"  {sensor}/{modality}: {sensor_data.shape}, "
                            f"Labels: {np.unique(labels)}"
                        )

                # Synchronized ACC/GYR data
                if all(src in modality_chunks for src in fuse_sources):
                    fused = self._build_synchronized_activity_chunks(
                        modality_chunks[fuse_sources[0]],
                        modality_chunks[fuse_sources[1]],
                        sensor
                    )
                    if fused is not None:
                        fused_data, fused_labels = fused
                        acc_data = fused_data[:, :3]
                        gyro_data = fused_data[:, 3:]
                        sensor_data_store['ACC'] = (acc_data, fused_labels)
                        sensor_data_store['GYR'] = (gyro_data, fused_labels)
                        logger.info(
                            f"  {sensor}/ACC(sync): {acc_data.shape}, Labels: {np.unique(fused_labels)}"
                        )
                        logger.info(
                            f"  {sensor}/GYR(sync): {gyro_data.shape}, Labels: {np.unique(fused_labels)}"
                        )
                else:
                    missing = [src for src in fuse_sources if src not in modality_chunks]
                    if missing:
                        logger.warning(
                            f"USER{person_id:05d}/{sensor}: missing modalities for fusion: {missing}"
                        )

                if sensor_data_store:
                    person_entry[sensor] = sensor_data_store

            if person_entry:
                result[person_id] = person_entry

        if not result:
            raise ValueError("No data loaded. Please check the raw data directory structure.")

        logger.info(f"Total users loaded: {len(result)}")
        return result

    def _extract_xyz_data_with_labels(
        self,
        df: pd.DataFrame
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        xyz_columns = self._get_xyz_columns(df)

        if len(xyz_columns) < 3:
            logger.warning("Could not find three axis columns in DataFrame")
            return None

        sensor_data = df[xyz_columns[:3]].values.astype(np.float32)

        if 'activity' in df.columns:
            activity_series = df['activity'].astype(str).str.lower()
            labels = activity_series.map(self.activity_mapping).fillna(-1).astype(int).values
        else:
            labels = np.full(len(sensor_data), -1, dtype=int)

        return sensor_data, labels

    def _get_xyz_columns(self, df: pd.DataFrame) -> List[str]:
        axis_candidates = {
            'x': ['attr_x', 'x'],
            'y': ['attr_y', 'y'],
            'z': ['attr_z', 'z']
        }

        xyz_columns: List[str] = []
        for axis in ('x', 'y', 'z'):
            candidates = axis_candidates[axis]
            column = next(
                (col for col in df.columns if col.lower() in candidates),
                None
            )
            if column:
                xyz_columns.append(column)

        return xyz_columns

    def _get_time_column(self, df: pd.DataFrame) -> Optional[str]:
        time_candidates = ['attr_time', 'timestamp', 'time', 'ts', 't']
        return next((col for col in df.columns if col.lower() in time_candidates), None)

    def _prepare_chunk_arrays(
        self,
        df: pd.DataFrame,
        activity_name: Optional[str]
    ) -> Optional[Tuple[Optional[np.ndarray], np.ndarray, int]]:
        xyz_columns = self._get_xyz_columns(df)
        if len(xyz_columns) < 3:
            logger.warning("Chunk is missing xyz columns")
            return None

        values = df[xyz_columns[:3]].values.astype(np.float32)
        time_col = self._get_time_column(df)
        timestamps = df[time_col].to_numpy(np.float64) if time_col else None

        candidate_activity = activity_name
        if not candidate_activity and 'activity' in df.columns and len(df['activity']):
            candidate_activity = str(df['activity'].iloc[0])

        label_id = self.activity_mapping.get(str(candidate_activity).lower(), -1) if candidate_activity else -1

        return timestamps, values, label_id

    def _align_chunk_arrays(
        self,
        acc_time: Optional[np.ndarray],
        acc_values: np.ndarray,
        gyro_time: Optional[np.ndarray],
        gyro_values: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if acc_values.size == 0 or gyro_values.size == 0:
            return None

        if acc_time is not None and gyro_time is not None:
            start = max(acc_time[0], gyro_time[0])
            end = min(acc_time[-1], gyro_time[-1])
            if end <= start:
                return None

            acc_mask = (acc_time >= start) & (acc_time <= end)
            gyro_mask = (gyro_time >= start) & (gyro_time <= end)

            acc_values = acc_values[acc_mask]
            gyro_values = gyro_values[gyro_mask]

        min_len = min(len(acc_values), len(gyro_values))
        if min_len == 0:
            return None

        return acc_values[:min_len], gyro_values[:min_len]

    def _build_synchronized_activity_chunks(
        self,
        acc_chunks: Dict[str, List[pd.DataFrame]],
        gyro_chunks: Dict[str, List[pd.DataFrame]],
        sensor_name: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        shared_activities = sorted(set(acc_chunks.keys()) & set(gyro_chunks.keys()))
        if not shared_activities:
            logger.warning(f"{sensor_name}: no overlapping activities for ACC/GYR fusion")
            return None

        fused_segments: List[np.ndarray] = []
        fused_labels: List[np.ndarray] = []

        for activity in shared_activities:
            acc_list = acc_chunks.get(activity, [])
            gyro_list = gyro_chunks.get(activity, [])

            if not acc_list or not gyro_list:
                continue

            if len(acc_list) != len(gyro_list):
                logger.warning(
                    f"{sensor_name}/{activity}: chunk count mismatch (ACC={len(acc_list)}, GYR={len(gyro_list)})"
                )

            pair_count = min(len(acc_list), len(gyro_list))

            for idx in range(pair_count):
                acc_chunk = acc_list[idx]
                gyro_chunk = gyro_list[idx]

                acc_prepared = self._prepare_chunk_arrays(acc_chunk, activity)
                gyro_prepared = self._prepare_chunk_arrays(gyro_chunk, activity)

                if acc_prepared is None or gyro_prepared is None:
                    continue

                acc_time, acc_values, label_id = acc_prepared
                gyro_time, gyro_values, _ = gyro_prepared

                aligned = self._align_chunk_arrays(acc_time, acc_values, gyro_time, gyro_values)
                if aligned is None:
                    logger.warning(f"{sensor_name}/{activity}: failed to align ACC/GYR chunk {idx}")
                    continue

                acc_aligned, gyro_aligned = aligned
                fused_values = np.concatenate([acc_aligned, gyro_aligned], axis=1)
                labels = np.full(len(fused_values), label_id, dtype=int)

                fused_segments.append(fused_values.astype(np.float32))
                fused_labels.append(labels)

        if not fused_segments:
            return None

        fused_data = np.concatenate(fused_segments, axis=0)
        fused_label_array = np.concatenate(fused_labels, axis=0)
        return fused_data, fused_label_array

    def clean_data(
        self,
        data: Dict[int, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]
    ) -> Dict[int, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]:
        """
        Data cleaning and resampling
        """
        cleaned = {}

        for person_id, sensor_dict in data.items():
            cleaned[person_id] = {}

            for sensor, modality_dict in sensor_dict.items():
                cleaned[person_id][sensor] = {}

                processed_modalities = set()

                # Combine and filter ACC/GYR
                if 'ACC' in modality_dict and 'GYR' in modality_dict:
                    joint_result = self._process_joint_acc_gyro(
                        person_id,
                        sensor,
                        modality_dict['ACC'],
                        modality_dict['GYR']
                    )
                    if joint_result is not None:
                        cleaned[person_id][sensor]['ACC_GYR'] = joint_result
                        processed_modalities.update({'ACC', 'GYR', 'ACC_GYR'})

                for modality, (sensor_data, labels) in modality_dict.items():
                    if modality in processed_modalities:
                        continue

                    cleaned_result = self._filter_and_resample(
                        sensor_data,
                        labels,
                        f"USER{person_id:05d}/{sensor}/{modality}"
                    )

                    if cleaned_result is None:
                        continue

                    cleaned[person_id][sensor][modality] = cleaned_result

        return cleaned

    def _process_joint_acc_gyro(
        self,
        person_id: int,
        sensor: str,
        acc_entry: Tuple[np.ndarray, np.ndarray],
        gyro_entry: Tuple[np.ndarray, np.ndarray]
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        acc_data, acc_labels = acc_entry
        gyro_data, gyro_labels = gyro_entry

        if len(acc_data) == 0 or len(gyro_data) == 0:
            logger.warning(
                f"USER{person_id:05d}/{sensor}/ACC+GYR: One of the modalities is empty"
            )
            return None

        if len(acc_data) != len(gyro_data):
            min_len = min(len(acc_data), len(gyro_data))
            logger.warning(
                f"USER{person_id:05d}/{sensor}/ACC+GYR: "
                f"length mismatch (ACC={len(acc_data)}, GYR={len(gyro_data)}), "
                f"truncating to {min_len}"
            )
            acc_data = acc_data[:min_len]
            gyro_data = gyro_data[:min_len]
            acc_labels = acc_labels[:min_len]
            gyro_labels = gyro_labels[:min_len]

        if not np.array_equal(acc_labels, gyro_labels):
            logger.warning(
                f"USER{person_id:05d}/{sensor}/ACC+GYR: label mismatch detected, "
                "using ACC labels as reference"
            )

        combined_data = np.concatenate([acc_data, gyro_data], axis=1)
        cleaned_result = self._filter_and_resample(
            combined_data,
            acc_labels,
            f"USER{person_id:05d}/{sensor}/ACC+GYR"
        )

        if cleaned_result is None:
            return None

        logger.info(
            f"USER{person_id:05d}/{sensor}/ACC+GYR joint cleaned: "
            f"{cleaned_result[0].shape}"
        )

        return cleaned_result

    def _filter_and_resample(
        self,
        sensor_data: np.ndarray,
        labels: np.ndarray,
        log_prefix: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        cleaned_data, cleaned_labels = filter_invalid_samples(sensor_data, labels)

        valid_mask = cleaned_labels >= 0
        cleaned_data = cleaned_data[valid_mask]
        cleaned_labels = cleaned_labels[valid_mask]

        if len(cleaned_data) == 0:
            logger.warning(
                f"{log_prefix}: No valid data after filtering"
            )
            return None

        if self.original_sampling_rate != self.target_sampling_rate:
            resampled_data, resampled_labels = resample_timeseries(
                cleaned_data,
                cleaned_labels,
                self.original_sampling_rate,
                self.target_sampling_rate
            )
            logger.info(
                f"{log_prefix} resampled: {resampled_data.shape}"
            )
            return resampled_data, resampled_labels
        else:
            logger.info(
                f"{log_prefix} cleaned: {cleaned_data.shape}"
            )
            return cleaned_data, cleaned_labels

    def extract_features(
        self,
        data: Dict[int, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]
    ) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
        """
        Feature extraction (windowing per sensor × modality)
        """
        processed = {}

        for person_id, sensor_dict in data.items():
            logger.info(f"Processing USER{person_id:05d}")
            processed[person_id] = {}

            for sensor, modality_dict in sensor_dict.items():
                processed_modalities = set()

                # Window ACC/GYR as 6-axis
                if 'ACC_GYR' in modality_dict:
                    joint_entries = self._windowize_joint_acc_gyro_features(
                        sensor,
                        modality_dict['ACC_GYR']
                    )
                    if joint_entries is not None:
                        processed[person_id].update(joint_entries)
                        processed_modalities.update({'ACC', 'GYR', 'ACC_GYR'})

                for modality, (sensor_data, labels) in modality_dict.items():
                    if modality in processed_modalities or len(sensor_data) == 0:
                        continue

                    entry = self._windowize_single_modality(sensor, modality, sensor_data, labels)
                    if entry is None:
                        continue

                    key, arrays = entry
                    processed[person_id][key] = arrays

        return processed

    def _windowize_joint_acc_gyro_features(
        self,
        sensor: str,
        combined_entry: Tuple[np.ndarray, np.ndarray]
    ) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
        combined_data, labels = combined_entry

        if combined_data.shape[1] < 6:
            logger.warning(f"{sensor}/ACC_GYR: expected 6 channels, got {combined_data.shape[1]}")
            return None

        windows = self._create_windows(
            combined_data,
            labels,
            f"{sensor}/ACC_GYR",
            log_suffix="(6-axis)"
        )

        if windows is None:
            return None

        windowed_data, windowed_labels = windows
        acc_windows = windowed_data[:, :3, :]
        gyro_windows = windowed_data[:, 3:, :]

        logger.info(
            f"  {sensor}/ACC split from ACC_GYR: X{acc_windows.shape}, Y{windowed_labels.shape}"
        )
        logger.info(
            f"  {sensor}/GYRO split from ACC_GYR: X{gyro_windows.shape}, Y{windowed_labels.shape}"
        )

        return {
            f"{sensor}/ACC": {
                'X': acc_windows,
                'Y': windowed_labels
            },
            f"{sensor}/GYRO": {
                'X': gyro_windows,
                'Y': windowed_labels
            }
        }

    def _windowize_single_modality(
        self,
        sensor: str,
        modality: str,
        sensor_data: np.ndarray,
        labels: np.ndarray
    ) -> Optional[Tuple[str, Dict[str, np.ndarray]]]:
        key_modality = 'GYRO' if modality.upper() == 'GYR' else modality
        key = f"{sensor}/{key_modality}"
        arrays = self._create_window_entry(sensor_data, labels, key)
        if arrays is None:
            return None
        return key, arrays

    def _create_window_entry(
        self,
        sensor_data: np.ndarray,
        labels: np.ndarray,
        key: str,
        log_suffix: str = ""
    ) -> Optional[Dict[str, np.ndarray]]:
        windows = self._create_windows(sensor_data, labels, key, log_suffix)
        if windows is None:
            return None
        windowed_data, windowed_labels = windows
        return {
            'X': windowed_data,
            'Y': windowed_labels
        }

    def _create_windows(
        self,
        sensor_data: np.ndarray,
        labels: np.ndarray,
        key: str,
        log_suffix: str = ""
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if len(sensor_data) == 0:
            return None

        windowed_data, windowed_labels = create_sliding_windows(
            sensor_data,
            labels,
            window_size=self.window_size,
            stride=self.stride,
            drop_last=False,
            pad_last=True
        )

        if len(windowed_data) == 0:
            logger.warning(f"{key}: No windows generated")
            return None

        if self.scale_factor is not None:
            windowed_data = windowed_data / self.scale_factor
            logger.info(f"  Applied scale_factor={self.scale_factor} to {key}")

        windowed_data = np.transpose(windowed_data, (0, 2, 1))
        windowed_data = windowed_data.astype(np.float16)

        logger.info(
            f"  {key}{(' ' + log_suffix) if log_suffix else ''}: "
            f"X.shape={windowed_data.shape}, Y.shape={windowed_labels.shape}"
        )

        return windowed_data, windowed_labels

    def save_processed_data(
        self,
        data: Dict[int, Dict[str, Dict[str, np.ndarray]]]
    ) -> None:
        """
        Save processed data
        """
        import json

        base_path = self.processed_data_path / self.dataset_name
        base_path.mkdir(parents=True, exist_ok=True)

        total_stats = {
            'dataset': self.dataset_name,
            'num_activities': self.num_activities,
            'num_sensors': self.num_sensors,
            'sensor_names': self.sensor_names,
            'modality_names': self.modality_names,
            'original_sampling_rate': self.original_sampling_rate,
            'target_sampling_rate': self.target_sampling_rate,
            'window_size': self.window_size,
            'stride': self.stride,
            'normalization': 'none',
            'scale_factor': self.scale_factor,
            'data_dtype': 'float16',
            'users': {}
        }

        for person_id, sensor_modality_data in data.items():
            user_name = f"USER{person_id:05d}"
            user_path = base_path / user_name
            user_path.mkdir(parents=True, exist_ok=True)

            user_stats = {'sensor_modalities': {}}
            acc_gyro_tracker: Dict[str, Dict[str, int]] = {}

            for sensor_modality_name, arrays in sensor_modality_data.items():
                sensor_modality_path = user_path / sensor_modality_name
                sensor_modality_path.mkdir(parents=True, exist_ok=True)

                # Save X.npy and Y.npy
                X = arrays['X']  # (num_windows, C, window_size)
                Y = arrays['Y']  # (num_windows,)

                assert isinstance(X, np.ndarray) and isinstance(Y, np.ndarray), \
                    f"{user_name}/{sensor_modality_name}: X/Y must be numpy arrays"
                assert X.dtype == np.float16, \
                    f"{user_name}/{sensor_modality_name}: X dtype {X.dtype} != float16"
                assert np.issubdtype(Y.dtype, np.integer), \
                    f"{user_name}/{sensor_modality_name}: Y dtype {Y.dtype} is not integer"

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

                if '/' in sensor_modality_name:
                    sensor_name, modality_name = sensor_modality_name.split('/', 1)
                    modality_upper = modality_name.upper()
                    if modality_upper in {'ACC', 'GYRO'}:
                        acc_gyro_tracker.setdefault(sensor_name, {})[modality_upper] = len(Y)

            for sensor_name, modal_counts in acc_gyro_tracker.items():
                if {'ACC', 'GYRO'}.issubset(modal_counts.keys()):
                    acc_count = modal_counts['ACC']
                    gyr_count = modal_counts['GYRO']
                    assert acc_count == gyr_count, (
                        f"{user_name}/{sensor_name}: ACC windows ({acc_count}) "
                        f"!= GYRO windows ({gyr_count})"
                    )
                    logger.info(
                        f"{user_name}/{sensor_name}: ACC/GYRO window counts aligned ({acc_count})"
                    )

            total_stats['users'][user_name] = user_stats

        # Save overall metadata
        metadata_path = base_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
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
