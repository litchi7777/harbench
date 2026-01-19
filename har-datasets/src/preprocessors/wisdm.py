"""
WISDM (Wireless Sensor Data Mining) Smartphone & Smartwatch
Activity and Biometrics Dataset Preprocessing

WISDM Dataset:
- 18 types of daily, sports, eating, and writing activities
- 51 subjects (ID: 1600-1650)
- 2 devices × 2 modalities (Phone/Watch × ACC/GYRO) = 12 channels
- Sampling rate: 20Hz
"""

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import register_preprocessor
from .base import BasePreprocessor
from .common import (
    check_dataset_exists,
    cleanup_temp_files,
    download_file,
    extract_archive,
)
from .utils import (
    create_sliding_windows,
    filter_invalid_samples,
    get_class_distribution,
    resample_timeseries,
)
from ..dataset_info import DATASETS

logger = logging.getLogger(__name__)

# Archive provided from UCI Machine Learning Repository
WISDM_URL = (
    "https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip"
)

# Activity codes and names (following activity_key.txt definition)
WISDM_ACTIVITY_CODES: List[Tuple[str, str]] = [
    ("A", "Walking"),
    ("B", "Jogging"),
    ("C", "Stairs"),
    ("D", "Sitting"),
    ("E", "Standing"),
    ("F", "Typing"),
    ("G", "BrushingTeeth"),
    ("H", "EatingSoup"),
    ("I", "EatingChips"),
    ("J", "EatingPasta"),
    ("K", "Drinking"),
    ("L", "EatingSandwich"),
    ("M", "Kicking"),
    ("O", "Catching"),
    ("P", "Dribbling"),
    ("Q", "Writing"),
    ("R", "Clapping"),
    ("S", "FoldingClothes"),
]

ACTIVITY_CODE_TO_ID: Dict[str, int] = {
    code: idx for idx, (code, _) in enumerate(WISDM_ACTIVITY_CODES)
}


@register_preprocessor("wisdm")
class WISDMPreprocessor(BasePreprocessor):
    """
    Preprocessing class for WISDM dataset
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.num_subjects = 51
        self.num_activities = len(WISDM_ACTIVITY_CODES)
        self.device_names = ["Phone", "Watch"]
        self.modalities = ["ACC", "GYRO"]
        self.channels_per_modality = 3  # 3-axis

        self.original_sampling_rate = 20  # Hz
        self.target_sampling_rate = config.get("target_sampling_rate", 30)  # Hz

        # Default: 5-second window & 1-second stride (based on target SR)
        default_window = int(self.target_sampling_rate * 5)
        default_stride = int(self.target_sampling_rate * 1)
        self.window_size = config.get("window_size", default_window)
        self.stride = config.get("stride", default_stride)

        self.scale_factor = DATASETS.get("WISDM", {}).get("scale_factor")

        # Path structure
        self.dataset_root_name = "wisdm-dataset"
        self.device_folder_map = {"Phone": "phone", "Watch": "watch"}
        self.modality_folder_map = {"ACC": "accel", "GYRO": "gyro"}

    def get_dataset_name(self) -> str:
        return "wisdm"

    # ------------------------------------------------------------------
    # Download & Organization
    # ------------------------------------------------------------------
    def download_dataset(self) -> None:
        """
        Download and extract WISDM dataset
        """
        logger.info("=" * 80)
        logger.info("Downloading WISDM dataset")
        logger.info("=" * 80)

        wisdm_raw_path = self.raw_data_path / self.dataset_name

        required_pattern = [
            f"{self.dataset_root_name}/raw/phone/accel/data_*_accel_phone.txt"
        ]
        if check_dataset_exists(wisdm_raw_path, required_files=required_pattern):
            logger.warning(f"WISDM data already exists at {wisdm_raw_path}")
            response = input("Do you want to re-download? (y/N): ")
            if response.lower() != "y":
                logger.info("Skipping download")
                return

        try:
            wisdm_raw_path.mkdir(parents=True, exist_ok=True)
            zip_path = wisdm_raw_path.parent / "wisdm.zip"
            download_file(WISDM_URL, zip_path, desc="Downloading WISDM archive")

            extract_to = wisdm_raw_path.parent / "wisdm_temp"
            extract_archive(zip_path, extract_to, desc="Extracting WISDM archive")
            self._organize_wisdm_data(extract_to, wisdm_raw_path)

            cleanup_temp_files(extract_to)
            if zip_path.exists():
                zip_path.unlink()

            logger.info("=" * 80)
            logger.info(f"SUCCESS: WISDM dataset available at {wisdm_raw_path}")
            logger.info("=" * 80)
        except Exception as exc:
            logger.error(f"Failed to download WISDM dataset: {exc}", exc_info=True)
            raise

    def _organize_wisdm_data(self, extracted_path: Path, target_path: Path) -> None:
        """
        Extract nested archives and organize under data/raw/wisdm
        """
        logger.info(f"Organizing WISDM data from {extracted_path} to {target_path}")
        target_path.mkdir(parents=True, exist_ok=True)

        # Extract nested ZIP archives (wisdm-dataset.zip, etc.)
        nested_archives = list(extracted_path.glob("*.zip"))
        for nested_archive in nested_archives:
            logger.info(f"Found nested archive: {nested_archive.name}")
            nested_extract = extracted_path / nested_archive.stem
            extract_archive(nested_archive, nested_extract, desc="Extracting nested WISDM")
            nested_archive.unlink()

        dataset_root = self._locate_dataset_root(extracted_path)
        if dataset_root is None:
            raise FileNotFoundError(
                f"Unable to locate WISDM raw directory under {extracted_path}"
            )

        target_dataset_dir = target_path / dataset_root.name
        if target_dataset_dir.exists():
            shutil.rmtree(target_dataset_dir)

        shutil.copytree(dataset_root, target_dataset_dir)
        logger.info(f"Copied dataset contents to {target_dataset_dir}")

    def _locate_dataset_root(self, search_root: Path) -> Optional[Path]:
        """
        Search for base folder containing the final raw directory
        """
        candidate = search_root / self.dataset_root_name
        if (candidate / "raw").exists():
            return candidate

        # Recursively search for raw directory
        for raw_dir in search_root.rglob("raw"):
            if (raw_dir / "phone").exists() and (raw_dir / "watch").exists():
                return raw_dir.parent

        return None

    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------
    def load_raw_data(self) -> Dict[int, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]]:
        """
        Load raw data for each subject × device × modality
        Returns: {subject_id: {device_name: {modality_name: (data, labels, timestamps)}}}
        """
        dataset_root = self._find_existing_dataset_root()
        raw_dir = dataset_root / "raw"

        person_data: Dict[int, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]] = {}

        for device_name in self.device_names:
            device_folder = self.device_folder_map[device_name]

            for modality_name in self.modalities:
                modality_folder = self.modality_folder_map[modality_name]
                sensor_dir = raw_dir / device_folder / modality_folder

                if not sensor_dir.exists():
                    logger.warning(f"Sensor directory missing: {sensor_dir}")
                    continue

                for sensor_file in sorted(sensor_dir.glob("data_*_*.txt")):
                    subject_id = self._parse_subject_id(sensor_file.name)
                    data, labels, timestamps = self._load_sensor_file(sensor_file)

                    if (
                        data is None
                        or labels is None
                        or timestamps is None
                        or len(data) == 0
                    ):
                        logger.warning(f"No valid samples in {sensor_file}")
                        continue

                    person_entry = person_data.setdefault(subject_id, {})
                    device_entry = person_entry.setdefault(device_name, {})
                    device_entry[modality_name] = (data, labels, timestamps)

                    logger.info(
                        f"Loaded {sensor_file.name}: subject={subject_id}, "
                        f"{device_name}/{modality_name} -> {data.shape}"
                    )

        if not person_data:
            raise ValueError("No WISDM data loaded. Please check the raw data directory.")

        logger.info(f"Total subjects loaded: {len(person_data)}")
        return person_data

    def _find_existing_dataset_root(self) -> Path:
        """
        Search for and return raw directory under data/raw/wisdm
        """
        base_path = self.raw_data_path / self.dataset_name
        candidates = [
            base_path / self.dataset_root_name,
            base_path,
        ]

        for candidate in candidates:
            raw_dir = candidate / "raw"
            if raw_dir.exists():
                return candidate

        for raw_dir in base_path.rglob("raw"):
            if (raw_dir / "phone").exists():
                return raw_dir.parent

        raise FileNotFoundError(
            f"Could not locate WISDM raw directory under {base_path}. "
            "Expected structure: data/raw/wisdm/wisdm-dataset/raw/phone/accel/*.txt"
        )

    @staticmethod
    def _parse_subject_id(filename: str) -> int:
        """
        Extract subject_id from filename in data_<subject>_*_* format
        """
        try:
            return int(filename.split("_")[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Invalid WISDM filename format: {filename}") from exc

    def _load_sensor_file(
        self, file_path: Path
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load a single sensor file and return data and labels
        """
        samples: List[List[float]] = []
        labels: List[int] = []
        timestamps: List[int] = []

        with file_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.endswith(";"):
                    line = line[:-1]

                parts = line.split(",")
                if len(parts) != 6:
                    logger.debug(f"Skipping malformed line ({file_path.name}): {line}")
                    continue

                _, activity_code, timestamp_str, x_str, y_str, z_str = parts
                activity_code = activity_code.strip().upper()
                if activity_code not in ACTIVITY_CODE_TO_ID:
                    logger.debug(
                        f"Unknown activity code '{activity_code}' in {file_path.name}"
                    )
                    continue

                try:
                    timestamp = int(timestamp_str)
                    x = float(x_str)
                    y = float(y_str)
                    z = float(z_str)
                except ValueError:
                    logger.debug(f"Invalid numeric values in {file_path.name}: {line}")
                    continue

                samples.append([x, y, z])
                labels.append(ACTIVITY_CODE_TO_ID[activity_code])
                timestamps.append(timestamp)

        if not samples:
            return None, None, None

        data = np.asarray(samples, dtype=np.float32)
        label_array = np.asarray(labels, dtype=np.int32)
        timestamp_array = np.asarray(timestamps, dtype=np.int64)

        # Sort by timestamp to ensure chronological order
        order = np.argsort(timestamp_array)
        data = data[order]
        label_array = label_array[order]
        timestamp_array = timestamp_array[order]

        return data, label_array, timestamp_array

    # ------------------------------------------------------------------
    # Cleaning & Resampling
    # ------------------------------------------------------------------
    def clean_data(
        self, data: Dict[int, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]]
    ) -> Dict[int, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]:
        cleaned: Dict[int, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]] = {}

        for person_id, device_data in data.items():
            cleaned[person_id] = {}
            for device_name, modality_data in device_data.items():
                cleaned[person_id][device_name] = {}

                processed_modalities = set()

                if "ACC" in modality_data and "GYRO" in modality_data:
                    joint_result = self._process_joint_acc_gyro(
                        person_id,
                        device_name,
                        modality_data["ACC"],
                        modality_data["GYRO"],
                    )
                    if joint_result is not None:
                        cleaned[person_id][device_name]["ACC_GYR"] = joint_result
                        processed_modalities.update({"ACC", "GYRO", "ACC_GYR"})

                for modality_name, entry in modality_data.items():
                    if modality_name in processed_modalities:
                        continue

                    sensor_data, labels = entry[:2]

                    cleaned_result = self._filter_and_resample(
                        sensor_data,
                        labels,
                        f"USER{person_id:05d} {device_name}/{modality_name}",
                    )

                    if cleaned_result is None:
                        continue

                    cleaned[person_id][device_name][modality_name] = cleaned_result

        return cleaned

    def _process_joint_acc_gyro(
        self,
        person_id: int,
        device_name: str,
        acc_entry: Tuple[np.ndarray, np.ndarray, np.ndarray],
        gyro_entry: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        acc_data, acc_labels, acc_time = acc_entry
        gyro_data, gyro_labels, gyro_time = gyro_entry

        if (
            len(acc_data) == 0
            or len(gyro_data) == 0
            or acc_time is None
            or gyro_time is None
        ):
            logger.warning(
                f"USER{person_id:05d} {device_name}/ACC+GYRO: One modality is empty or missing timestamps"
            )
            return None

        # Internal consistency check (align data/labels/time lengths within each modality)
        if not (len(acc_data) == len(acc_labels) == len(acc_time)):
            min_len = min(len(acc_data), len(acc_labels), len(acc_time))
            logger.warning(
                f"USER{person_id:05d} {device_name}/ACC: internal length mismatch; truncating to {min_len}"
            )
            acc_data = acc_data[:min_len]
            acc_labels = acc_labels[:min_len]
            acc_time = acc_time[:min_len]

        if not (len(gyro_data) == len(gyro_labels) == len(gyro_time)):
            min_len = min(len(gyro_data), len(gyro_labels), len(gyro_time))
            logger.warning(
                f"USER{person_id:05d} {device_name}/GYRO: internal length mismatch; truncating to {min_len}"
            )
            gyro_data = gyro_data[:min_len]
            gyro_labels = gyro_labels[:min_len]
            gyro_time = gyro_time[:min_len]

        # Align length mismatch between modalities to the shorter one (including timestamps)
        if len(acc_data) != len(gyro_data):
            min_len = min(len(acc_data), len(gyro_data))
            logger.warning(
                f"USER{person_id:05d} {device_name}/ACC+GYRO: length mismatch "
                f"(ACC={len(acc_data)}, GYRO={len(gyro_data)}), truncating to {min_len}"
            )
            acc_data = acc_data[:min_len]
            acc_labels = acc_labels[:min_len]
            acc_time = acc_time[:min_len]
            gyro_data = gyro_data[:min_len]
            gyro_labels = gyro_labels[:min_len]
            gyro_time = gyro_time[:min_len]

        # Extract overlapping timestamp region
        overlap_start = max(acc_time[0], gyro_time[0])
        overlap_end = min(acc_time[-1], gyro_time[-1])

        if overlap_end <= overlap_start:
            logger.warning(
                f"USER{person_id:05d} {device_name}/ACC+GYRO: no overlapping timestamps"
            )
            return None

        acc_mask = (acc_time >= overlap_start) & (acc_time <= overlap_end)
        gyro_mask = (gyro_time >= overlap_start) & (gyro_time <= overlap_end)

        if acc_mask.sum() == 0 or gyro_mask.sum() == 0:
            logger.warning(
                f"USER{person_id:05d} {device_name}/ACC+GYRO: empty after overlap masking"
            )
            return None

        acc_data = acc_data[acc_mask]
        acc_labels = acc_labels[acc_mask]
        acc_time = acc_time[acc_mask]

        gyro_data = gyro_data[gyro_mask]
        gyro_labels = gyro_labels[gyro_mask]
        gyro_time = gyro_time[gyro_mask]

        if len(acc_data) < 2 or len(gyro_data) < 2:
            logger.warning(
                f"USER{person_id:05d} {device_name}/ACC+GYRO: insufficient samples after overlap trim"
            )
            return None

        # Sort by timestamp (data may not be in chronological order)
        acc_sort_idx = np.argsort(acc_time)
        acc_data = acc_data[acc_sort_idx]
        acc_labels = acc_labels[acc_sort_idx]
        acc_time = acc_time[acc_sort_idx]

        gyro_sort_idx = np.argsort(gyro_time)
        gyro_data = gyro_data[gyro_sort_idx]
        gyro_labels = gyro_labels[gyro_sort_idx]
        gyro_time = gyro_time[gyro_sort_idx]

        # Process by splitting into sessions (split on large timestamp jumps)
        # Threshold: split on gaps of 10 seconds or more
        gap_threshold_ns = 10 * 1e9

        all_combined_data = []
        all_combined_labels = []

        # Detect session boundaries for ACC and GYRO
        acc_segments = self._split_into_segments(acc_time, gap_threshold_ns)
        gyro_segments = self._split_into_segments(gyro_time, gap_threshold_ns)

        # For each ACC segment, find overlapping GYRO segments
        for acc_start_idx, acc_end_idx in acc_segments:
            acc_seg_time = acc_time[acc_start_idx:acc_end_idx]
            acc_seg_data = acc_data[acc_start_idx:acc_end_idx]
            acc_seg_labels = acc_labels[acc_start_idx:acc_end_idx]

            for gyro_start_idx, gyro_end_idx in gyro_segments:
                gyro_seg_time = gyro_time[gyro_start_idx:gyro_end_idx]
                gyro_seg_data = gyro_data[gyro_start_idx:gyro_end_idx]
                gyro_seg_labels = gyro_labels[gyro_start_idx:gyro_end_idx]

                # Calculate overlap region
                overlap_start = max(acc_seg_time[0], gyro_seg_time[0])
                overlap_end = min(acc_seg_time[-1], gyro_seg_time[-1])

                if overlap_end <= overlap_start:
                    continue

                # Extract only overlap region
                acc_mask = (acc_seg_time >= overlap_start) & (acc_seg_time <= overlap_end)
                gyro_mask = (gyro_seg_time >= overlap_start) & (gyro_seg_time <= overlap_end)

                if acc_mask.sum() < 2 or gyro_mask.sum() < 2:
                    continue

                seg_acc_time = acc_seg_time[acc_mask]
                seg_acc_data = acc_seg_data[acc_mask]
                seg_acc_labels = acc_seg_labels[acc_mask]

                seg_gyro_time = gyro_seg_time[gyro_mask]
                seg_gyro_data = gyro_seg_data[gyro_mask]
                seg_gyro_labels = gyro_seg_labels[gyro_mask]

                # Convert to relative seconds (timestamps are in nanoseconds)
                seg_acc_seconds = (seg_acc_time - overlap_start).astype(np.float64) / 1e9
                seg_gyro_seconds = (seg_gyro_time - overlap_start).astype(np.float64) / 1e9

                # Target time grid
                target_step = 1.0 / float(self.target_sampling_rate)
                duration = (overlap_end - overlap_start) / 1e9
                num_samples = int(np.floor(duration / target_step)) + 1

                if num_samples < 2:
                    continue

                target_times = np.arange(num_samples, dtype=np.float64) * target_step

                # Resampling
                acc_resampled, acc_resampled_labels = self._resample_with_timestamps(
                    seg_acc_data,
                    seg_acc_labels,
                    seg_acc_seconds,
                    target_times,
                )
                gyro_resampled, _ = self._resample_with_timestamps(
                    seg_gyro_data,
                    seg_gyro_labels,
                    seg_gyro_seconds,
                    target_times,
                )

                if acc_resampled is None or gyro_resampled is None:
                    continue

                # Combine into 6 channels
                segment_combined = np.concatenate([acc_resampled, gyro_resampled], axis=1)
                all_combined_data.append(segment_combined)
                all_combined_labels.append(acc_resampled_labels)

        if not all_combined_data:
            logger.warning(
                f"USER{person_id:05d} {device_name}/ACC+GYRO: no valid segments after processing"
            )
            return None

        # Concatenate all segments
        combined_data = np.concatenate(all_combined_data, axis=0)
        combined_labels = np.concatenate(all_combined_labels, axis=0)

        # Remove invalid samples (NaN, etc.)
        combined_data, combined_labels = filter_invalid_samples(
            combined_data,
            combined_labels,
        )
        valid_mask = combined_labels >= 0
        combined_data = combined_data[valid_mask]
        combined_labels = combined_labels[valid_mask]

        if len(combined_data) == 0:
            logger.warning(
                f"USER{person_id:05d} {device_name}/ACC+GYRO: no valid samples after filtering"
            )
            return None

        logger.info(
            f"USER{person_id:05d} {device_name}/ACC+GYRO joint cleaned: {combined_data.shape}"
        )

        return combined_data, combined_labels

    def _split_into_segments(
        self,
        timestamps: np.ndarray,
        gap_threshold_ns: float,
    ) -> List[Tuple[int, int]]:
        """
        Split timestamp array by large gaps and return segment index ranges
        """
        if len(timestamps) < 2:
            return [(0, len(timestamps))]

        diffs = np.diff(timestamps)
        # Detect points where gap exceeds threshold
        gap_indices = np.where(diffs > gap_threshold_ns)[0] + 1

        segments = []
        start = 0
        for gap_idx in gap_indices:
            if gap_idx > start:
                segments.append((start, gap_idx))
            start = gap_idx

        # Last segment
        if start < len(timestamps):
            segments.append((start, len(timestamps)))

        return segments

    def _filter_and_resample(
        self,
        sensor_data: np.ndarray,
        labels: np.ndarray,
        log_prefix: str,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        filtered_data, filtered_labels = filter_invalid_samples(sensor_data, labels)

        valid_mask = filtered_labels >= 0
        filtered_data = filtered_data[valid_mask]
        filtered_labels = filtered_labels[valid_mask]

        if len(filtered_data) == 0:
            logger.warning(f"{log_prefix}: No valid samples after filtering")
            return None

        if self.original_sampling_rate != self.target_sampling_rate:
            resampled_data, resampled_labels = resample_timeseries(
                filtered_data,
                filtered_labels,
                self.original_sampling_rate,
                self.target_sampling_rate,
            )
            logger.info(f"{log_prefix}: resampled {filtered_data.shape} -> {resampled_data.shape}")
            return resampled_data, resampled_labels

        logger.info(f"{log_prefix}: cleaned {filtered_data.shape}")
        return filtered_data, filtered_labels

    def _resample_with_timestamps(
        self,
        values: np.ndarray,
        labels: np.ndarray,
        time_sec: np.ndarray,
        target_times: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if len(values) < 2 or len(time_sec) < 2:
            return None, None

        # Ensure strictly increasing timeline
        order = np.argsort(time_sec)
        time_sec = time_sec[order]
        values = values[order]
        labels = labels[order]

        unique_time, unique_indices = np.unique(time_sec, return_index=True)
        if len(unique_time) < 2:
            return None, None

        time_sec = unique_time
        values = values[unique_indices]
        labels = labels[unique_indices]

        resampled = np.empty((len(target_times), values.shape[1]), dtype=np.float32)
        for axis in range(values.shape[1]):
            resampled[:, axis] = np.interp(
                target_times,
                time_sec,
                values[:, axis],
            )

        indices = np.searchsorted(time_sec, target_times, side="right") - 1
        indices = np.clip(indices, 0, len(labels) - 1)
        resampled_labels = labels[indices].astype(np.int32)

        return resampled.astype(np.float32), resampled_labels

    # ------------------------------------------------------------------
    # Feature Extraction
    # ------------------------------------------------------------------
    def extract_features(
        self, data: Dict[int, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]
    ) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
        processed: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {}

        for person_id, device_data in data.items():
            processed[person_id] = {}
            logger.info(f"Processing USER{person_id:05d}")

            for device_name, modality_data in device_data.items():
                processed_modalities = set()

                if "ACC_GYR" in modality_data:
                    joint_entries = self._windowize_joint_acc_gyro_features(
                        device_name,
                        modality_data["ACC_GYR"],
                    )
                    if joint_entries is not None:
                        processed[person_id].update(joint_entries)
                        processed_modalities.update({"ACC", "GYRO", "ACC_GYR"})

                for modality_name, entry in modality_data.items():
                    if modality_name in processed_modalities:
                        continue

                    logger.warning(
                        f"USER{person_id:05d} {device_name}/{modality_name}: "
                        "skipped because joint ACC/GYRO data was required"
                    )

        return processed

    def _windowize_joint_acc_gyro_features(
        self,
        device_name: str,
        combined_entry: Tuple[np.ndarray, np.ndarray],
    ) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
        combined_data, labels = combined_entry

        if combined_data.shape[1] < 6:
            logger.warning(
                f"{device_name}/ACC_GYR: expected 6 channels, got {combined_data.shape[1]}"
            )
            return None

        windows = self._create_windows(
            combined_data,
            labels,
            f"{device_name}/ACC_GYR",
            log_suffix="(6-axis)",
        )
        if windows is None:
            return None

        windowed_data, windowed_labels = windows
        acc_windows = windowed_data[:, :3, :]
        gyro_windows = windowed_data[:, 3:, :]

        if self.scale_factor is not None:
            acc_windows = (
                acc_windows.astype(np.float32) / self.scale_factor
            ).astype(np.float16)

        logger.info(
            f"  {device_name}/ACC split from ACC_GYR: X{acc_windows.shape}, Y{windowed_labels.shape}"
        )
        logger.info(
            f"  {device_name}/GYRO split from ACC_GYR: X{gyro_windows.shape}, Y{windowed_labels.shape}"
        )

        return {
            f"{device_name}/ACC": {"X": acc_windows, "Y": windowed_labels},
            f"{device_name}/GYRO": {"X": gyro_windows, "Y": windowed_labels},
        }

    def _create_window_entry(
        self,
        sensor_data: np.ndarray,
        labels: np.ndarray,
        key: str,
    ) -> Optional[Dict[str, np.ndarray]]:
        windows = self._create_windows(sensor_data, labels, key)
        if windows is None:
            return None
        windowed_data, windowed_labels = windows
        return {"X": windowed_data, "Y": windowed_labels}

    def _create_windows(
        self,
        sensor_data: np.ndarray,
        labels: np.ndarray,
        key: str,
        log_suffix: str = "",
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if len(sensor_data) == 0:
            return None

        windows, window_labels = create_sliding_windows(
            sensor_data,
            labels,
            window_size=self.window_size,
            stride=self.stride,
            drop_last=False,
            pad_last=True
        )

        if len(windows) == 0:
            logger.warning(f"{key}: No windows generated")
            return None

        if key.endswith("/ACC") and self.scale_factor is not None and sensor_data.shape[1] == 3:
            windows = windows / self.scale_factor

        # Transpose: (N, T, C) -> (N, C, T)
        windows = np.transpose(windows, (0, 2, 1)).astype(np.float16)

        logger.info(
            f"  {key}{(' ' + log_suffix) if log_suffix else ''}: "
            f"X{windows.shape}, Y{window_labels.shape}"
        )

        return windows, window_labels

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def save_processed_data(
        self, data: Dict[int, Dict[str, Dict[str, np.ndarray]]]
    ) -> None:
        base_path = self.processed_data_path / self.dataset_name
        base_path.mkdir(parents=True, exist_ok=True)

        total_stats = {
            "dataset": self.dataset_name,
            "num_activities": self.num_activities,
            "num_devices": len(self.device_names),
            "device_names": self.device_names,
            "modalities": self.modalities,
            "channels_per_modality": self.channels_per_modality,
            "original_sampling_rate": self.original_sampling_rate,
            "target_sampling_rate": self.target_sampling_rate,
            "window_size": self.window_size,
            "stride": self.stride,
            "normalization": "none",
            "scale_factor": self.scale_factor,
            "data_dtype": "float16",
            "data_shape": f"(num_windows, {self.channels_per_modality}, {self.window_size})",
            "users": {},
        }

        for person_id, sensor_data in data.items():
            user_name = f"USER{person_id:05d}"
            user_path = base_path / user_name
            user_path.mkdir(parents=True, exist_ok=True)

            user_stats = {"sensor_modalities": {}}
            acc_gyro_tracker: Dict[str, Dict[str, int]] = {}

            for sensor_modality_name, arrays in sensor_data.items():
                sensor_modality_path = user_path / sensor_modality_name
                sensor_modality_path.mkdir(parents=True, exist_ok=True)

                X = arrays["X"]
                Y = arrays["Y"]

                assert isinstance(X, np.ndarray) and isinstance(Y, np.ndarray), \
                    f"{user_name}/{sensor_modality_name}: X/Y must be numpy arrays"
                assert X.dtype == np.float16, \
                    f"{user_name}/{sensor_modality_name}: X dtype {X.dtype} != float16"
                assert np.issubdtype(Y.dtype, np.integer), \
                    f"{user_name}/{sensor_modality_name}: Y dtype {Y.dtype} is not integer"

                np.save(sensor_modality_path / "X.npy", X)
                np.save(sensor_modality_path / "Y.npy", Y)

                user_stats["sensor_modalities"][sensor_modality_name] = {
                    "X_shape": X.shape,
                    "Y_shape": Y.shape,
                    "num_windows": len(Y),
                    "class_distribution": get_class_distribution(Y),
                }

                logger.info(
                    f"Saved {user_name}/{sensor_modality_name}: X{X.shape}, Y{Y.shape}"
                )

                if "/" in sensor_modality_name:
                    device_key, modality_key = sensor_modality_name.split("/", 1)
                    modality_upper = modality_key.upper()
                    if modality_upper in {"ACC", "GYRO"}:
                        acc_gyro_tracker.setdefault(device_key, {})[modality_upper] = len(Y)

            for device_key, modal_counts in acc_gyro_tracker.items():
                if {"ACC", "GYRO"}.issubset(modal_counts.keys()):
                    acc_count = modal_counts["ACC"]
                    gyr_count = modal_counts["GYRO"]
                    assert acc_count == gyr_count, (
                        f"{user_name}/{device_key}: ACC windows ({acc_count}) "
                        f"!= GYRO windows ({gyr_count})"
                    )
                    logger.info(
                        f"{user_name}/{device_key}: ACC/GYRO window counts aligned ({acc_count})"
                    )

            total_stats["users"][user_name] = user_stats

        metadata_path = base_path / "metadata.json"
        self._write_metadata(metadata_path, total_stats)
        logger.info(f"Saved metadata to {metadata_path}")
        logger.info(f"Preprocessing completed: {base_path}")

    def _write_metadata(self, metadata_path: Path, stats: Dict[str, Any]) -> None:
        """
        Convert dictionary containing NumPy types to JSON-serializable format and save
        """
        import json

        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, tuple):
                return list(obj)
            return obj

        def recursive(item):
            if isinstance(item, dict):
                return {k: recursive(v) for k, v in item.items()}
            if isinstance(item, list):
                return [recursive(v) for v in item]
            return convert(item)

        serializable_stats = recursive(stats)
        with open(metadata_path, "w") as f:
            json.dump(serializable_stats, f, indent=2)
