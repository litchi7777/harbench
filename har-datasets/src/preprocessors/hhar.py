"""
HHAR (Heterogeneity Human Activity Recognition) Dataset Preprocessing

Resample smartphone/smartwatch accelerometer and gyroscope data
to a constant rate and convert to sliding windows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, List
import logging
import json

import numpy as np
import pandas as pd

from .base import BasePreprocessor
from . import register_preprocessor
from ..dataset_info import DATASETS
from .common import download_file, extract_zip, check_dataset_exists
from .utils import create_sliding_windows, resample_timeseries, get_class_distribution

logger = logging.getLogger(__name__)

HHAR_ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/344/heterogeneity+activity+recognition.zip"


def _mode_label(values: pd.Series) -> int:
    """For DataFrame groupby: return the most frequent label"""
    counts = values.value_counts()
    return int(counts.index[0])


@register_preprocessor('hhar')
class HHARPreprocessor(BasePreprocessor):
    """
    HHAR Dataset Preprocessing

    Configuration example:
        - raw_data_path / processed_data_path
        - hhar_src_root: Raw data location (default: raw_data_path/hhar)
        - target_sampling_rate (Hz) default 30
        - window_size (samples)     default target_sampling_rate
        - stride (samples)          default window_size
        - max_gap_sec               default 1.0
        - min_segment_sec           default 5
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        dataset_meta = DATASETS.get('HHAR', {})

        self.target_sampling_rate = float(config.get('target_sampling_rate', dataset_meta.get('sampling_rate', 30.0)))
        self.window_size = int(config.get('window_size', int(self.target_sampling_rate * 5.0)))
        self.stride = int(config.get('stride', self.window_size))
        self.max_gap_sec = float(config.get('max_gap_sec', 1.0))
        self.min_segment_sec = float(config.get('min_segment_sec', 0.9933333))
        self.scale_factor = dataset_meta.get('scale_factor', 9.8)

        labels_meta = dataset_meta.get('labels', {})
        self.undefined_label = -1
        self.label_name_by_id = {int(k): v for k, v in labels_meta.items()}
        self.activity_to_label = {
            v.lower(): int(k)
            for k, v in labels_meta.items()
        }
        # Dataset-specific null class
        self.activity_to_label.setdefault('null', self.undefined_label)

        self.sensor_file_map = {
            'Phones': {
                'ACC': 'Phones_accelerometer.csv',
                'GYRO': 'Phones_gyroscope.csv',
            },
            'Watch': {
                'ACC': 'Watch_accelerometer.csv',
                'GYRO': 'Watch_gyroscope.csv',
            },
        }

        self.src_root = Path(config.get('hhar_src_root', self.raw_data_path / self.dataset_name))
        self.activity_dir = self.src_root / 'Activity recognition exp'

        self._user_to_person_id: Dict[str, int] = {}

        # Device-specific original sampling rates
        # Fixed rates for HHAR Watch devices. Others fall back to estimation.
        self.device_base_rates: Dict[str, float] = {
            "gear_1": 100.0,
            "gear": 100.0,      # for safety
            "gear_2": 100.0,
            "lgwatch_1": 200.0,
            "lgwatch_2": 200.0,
            "lgwatch": 200.0,   # for safety
        }

    def get_dataset_name(self) -> str:
        return 'hhar'

    def download_dataset(self) -> None:
        required_files = [
            str(Path('Activity recognition exp') / fname)
            for block in self.sensor_file_map.values()
            for fname in block.values()
        ]

        if check_dataset_exists(self.src_root, required_files=required_files):
            logger.info(f"HHAR raw files already present under {self.src_root}, skipping download")
            return

        self.src_root.mkdir(parents=True, exist_ok=True)
        archive_path = self.src_root / 'heterogeneity+activity+recognition.zip'

        if not archive_path.exists():
            download_file(HHAR_ARCHIVE_URL, archive_path, desc='HHAR dataset')
        else:
            logger.info(f"Archive already exists: {archive_path}")

        # Extract outer ZIP
        extract_zip(archive_path, self.src_root, desc='Extract HHAR outer archive')

        # Extract inner ZIPs (Activity recognition / Still)
        for nested_name in ['Activity recognition exp.zip', 'Still exp.zip']:
            nested_path = self.src_root / nested_name
            if nested_path.exists():
                extract_zip(nested_path, self.src_root, desc=f'Extract {nested_name}')

        if not check_dataset_exists(self.src_root, required_files=required_files):
            raise FileNotFoundError(
                "HHAR raw CSV files were not found after extraction. "
                "Please verify the archive contents."
            )

    def load_raw_data(self) -> Dict[int, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]:
        if not self.activity_dir.exists():
            raise FileNotFoundError(f"HHAR activity directory not found: {self.activity_dir}")

        combined_streams: Dict[
            Tuple[str, str],
            Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]
        ] = {}

        for sensor_family, modality_map in self.sensor_file_map.items():
            for modality_name, csv_name in modality_map.items():
                csv_path = self.activity_dir / csv_name
                if not csv_path.exists():
                    logger.warning(f"Missing HHAR file: {csv_path}")
                    continue

                logger.info(f"Loading {sensor_family} {modality_name} from {csv_path}")
                stream_map = self._load_sensor_stream(csv_path)

                for key, stream in stream_map.items():
                    device_stream = combined_streams.setdefault(key, {})
                    device_stream[modality_name] = stream

                logger.info(
                    f"Collected {len(stream_map)} sensor streams for {sensor_family}/{modality_name}"
                )

        data: Dict[int, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]] = {}

        for (user_token, device_name), modality_dict in combined_streams.items():
            if 'ACC' not in modality_dict or 'GYRO' not in modality_dict:
                logger.warning(
                    f"Skipping {user_token}/{device_name}: missing modality for 6-axis sync"
                )
                continue

            # Pass device name for resampling
            acc_windows, gyro_windows, labels = self._build_joint_windows(
                modality_dict['ACC'],
                modality_dict['GYRO'],
                device_name=device_name,
            )

            if acc_windows.size == 0 or gyro_windows.size == 0:
                logger.warning(
                    f"No synchronized windows produced for {user_token}/{device_name}"
                )
                continue

            person_id = self._get_or_assign_person_id(user_token)
            person_entry = data.setdefault(person_id, {})
            device_entry = person_entry.setdefault(device_name, {})
            device_entry['ACC'] = (acc_windows, labels)
            device_entry['GYRO'] = (gyro_windows, labels)
            logger.info(
                f"{user_token}/{device_name}: synchronized windows {labels.shape[0]}"
            )

        if not data:
            raise ValueError("HHAR preprocessing produced no synchronized windows. Check raw files and settings.")

        return data

    def clean_data(
        self,
        data: Dict[int, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]
    ) -> Dict[int, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]:
        # No additional cleaning needed (already windowed)
        return data

    def extract_features(
        self,
        data: Dict[int, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]
    ) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
        result: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {}

        for person_id, device_dict in data.items():
            person_result: Dict[str, Dict[str, np.ndarray]] = {}

            for device_name, modality_dict in device_dict.items():
                for modality_name, (windows, labels) in modality_dict.items():
                    if windows.size == 0:
                        continue
                    # (N, W, 3) -> (N, 3, W)
                    X = np.transpose(windows, (0, 2, 1)).astype(np.float32)
                    if self.scale_factor and modality_name == 'ACC':
                        X = X / self.scale_factor
                    X = X.astype(np.float16)
                    Y = labels.astype(np.int32)
                    key = f"{device_name}/{modality_name}"
                    person_result[key] = {
                        'X': X,
                        'Y': Y,
                    }

            if person_result:
                result[person_id] = person_result

        if not result:
            raise ValueError("No HHAR features were extracted. Please check preprocessing logs.")

        return result

    def save_processed_data(self, data) -> None:
        base_path = self.processed_data_path / self.dataset_name
        base_path.mkdir(parents=True, exist_ok=True)

        stats = {
            'dataset': self.dataset_name,
            'target_sampling_rate': self.target_sampling_rate,
            'window_size': self.window_size,
            'stride': self.stride,
            'max_gap_sec': self.max_gap_sec,
            'min_segment_sec': self.min_segment_sec,
            'scale_factor': self.scale_factor,
            'label_map': self.label_name_by_id,
            'users': {},
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

                X = arrays['X']
                Y = arrays['Y']

                np.save(sensor_modality_path / 'X.npy', X)
                np.save(sensor_modality_path / 'Y.npy', Y)

                user_stats['sensor_modalities'][sensor_modality_name] = {
                    'X_shape': list(X.shape),
                    'Y_shape': list(Y.shape),
                    'num_windows': int(len(Y)),
                    'class_distribution': get_class_distribution(Y)
                }

                logger.info(f"Saved {user_name}/{sensor_modality_name}: X={X.shape}, Y={Y.shape}")

                # Track ACC/GYRO window counts
                if '/' in sensor_modality_name:
                    device_name, modality_name = sensor_modality_name.split('/', 1)
                    modality_upper = modality_name.upper()
                    if modality_upper in {'ACC', 'GYRO'}:
                        acc_gyro_tracker.setdefault(device_name, {})[modality_upper] = len(Y)

            # Verify ACC/GYRO window count consistency
            for device_name, modal_counts in acc_gyro_tracker.items():
                if {'ACC', 'GYRO'}.issubset(modal_counts.keys()):
                    acc_count = modal_counts['ACC']
                    gyro_count = modal_counts['GYRO']
                    assert acc_count == gyro_count, (
                        f"{user_name}/{device_name}: ACC windows ({acc_count}) "
                        f"!= GYRO windows ({gyro_count})"
                    )
                    logger.info(
                        f"{user_name}/{device_name}: ACC/GYRO window counts aligned ({acc_count})"
                    )

            stats['users'][user_name] = user_stats

        # Save metadata (same format as DSADS)
        metadata_path = base_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            # Convert NumPy types to JSON compatible
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

            serializable_stats = recursive_convert(stats)
            json.dump(serializable_stats, f, indent=2)

        logger.info(f"Saved metadata to {metadata_path}")
        logger.info(f"Preprocessing completed: {base_path}")

    # --------------------------------------------------------------------- #
    # Internal Utilities
    # --------------------------------------------------------------------- #
    def _get_or_assign_person_id(self, user_token: str) -> int:
        if user_token not in self._user_to_person_id:
            self._user_to_person_id[user_token] = len(self._user_to_person_id) + 1
        return self._user_to_person_id[user_token]

    def _load_sensor_stream(
        self,
        csv_path: Path,
    ) -> Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        usecols = ['Arrival_Time', 'x', 'y', 'z', 'User', 'Device', 'gt']
        dtype = {
            'Arrival_Time': np.int64,
            'x': np.float32,
            'y': np.float32,
            'z': np.float32,
            'User': 'string',
            'Device': 'string',
            'gt': 'string',
        }

        df = pd.read_csv(
            csv_path,
            usecols=usecols,
            dtype=dtype,
            low_memory=False
        )

        df = df.dropna(subset=['User', 'Device', 'gt'])
        df['gt'] = df['gt'].str.strip().str.lower()

        df = df[df['gt'].isin(self.activity_to_label.keys())]
        df['label'] = df['gt'].map(self.activity_to_label).astype(np.int16)

        grouped: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        for (user_token, device_name), group in df.groupby(['User', 'Device']):
            group = group.sort_values('Arrival_Time')
            if group['Arrival_Time'].duplicated().any():
                group = group.groupby('Arrival_Time', as_index=False).agg({
                    'x': 'mean',
                    'y': 'mean',
                    'z': 'mean',
                    'label': _mode_label,
                })

            t_sec = (group['Arrival_Time'].to_numpy(np.float64)) / 1000.0
            values = group[['x', 'y', 'z']].to_numpy(np.float32)
            labels = group['label'].to_numpy(np.int16)

            if t_sec.size < 2:
                continue

            grouped[(user_token, device_name)] = (t_sec, values, labels)

        return grouped

    def _split_segments(self, t_sec: np.ndarray) -> List[Tuple[int, int]]:
        if t_sec.size < 2:
            return []

        dt = np.diff(t_sec)
        split_points = np.where(dt > self.max_gap_sec)[0] + 1
        indices = list(split_points) + [len(t_sec)]

        segments: List[Tuple[int, int]] = []
        start = 0
        for end in indices:
            if end - start < 2:
                start = end
                continue

            duration = t_sec[end - 1] - t_sec[start]
            if duration >= self.min_segment_sec:
                segments.append((start, end))

            start = end

        return segments

    def _build_joint_windows(
        self,
        acc_stream: Tuple[np.ndarray, np.ndarray, np.ndarray],
        gyro_stream: Tuple[np.ndarray, np.ndarray, np.ndarray],
        device_name: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Resample with device name
        acc_segments = self._resample_segments(acc_stream, device_name=device_name)
        gyro_segments = self._resample_segments(gyro_stream, device_name=device_name)

        if not acc_segments or not gyro_segments:
            return (
                np.empty((0, self.window_size, 3), dtype=np.float32),
                np.empty((0, self.window_size, 3), dtype=np.float32),
                np.empty((0,), dtype=np.int32)
            )

        return self._synchronize_segments(acc_segments, gyro_segments)

    def _resample_segments(
        self,
        stream: Tuple[np.ndarray, np.ndarray, np.ndarray],
        device_name: str,
    ) -> List[Dict[str, Any]]:
        t_sec, values, labels = stream
        segments_idx = self._split_segments(t_sec)
        if not segments_idx:
            return []

        resampled_segments: List[Dict[str, Any]] = []
        for start, end in segments_idx:
            seg_t = t_sec[start:end]
            seg_values = values[start:end]
            seg_labels = labels[start:end]

            dt_seg = np.diff(seg_t)
            dt_seg = dt_seg[dt_seg > 0]
            if dt_seg.size == 0:
                continue

            # Standard estimation
            median_dt = float(np.median(dt_seg))
            if median_dt <= 0:
                continue
            estimated_rate = 1.0 / median_dt
            if estimated_rate <= 0:
                continue
            quantized_rate = max(5.0, round(estimated_rate / 5.0) * 5.0)

            # Use fixed rate based on device name
            dev_key = device_name.strip().lower()
            if dev_key in self.device_base_rates:
                original_rate = self.device_base_rates[dev_key]
            else:
                # Unregistered devices fall back to estimation
                original_rate = quantized_rate

            uniform_values, uniform_labels = resample_timeseries(
                seg_values,
                seg_labels,
                original_rate=original_rate,
                target_rate=self.target_sampling_rate,
            )

            if uniform_values.size == 0:
                continue

            resampled_segments.append({
                'start': seg_t[0],
                'values': uniform_values.astype(np.float32),
                'labels': uniform_labels.astype(np.int16),
            })

        return resampled_segments

    def _synchronize_segments(
        self,
        acc_segments: List[Dict[str, Any]],
        gyro_segments: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        windows_list: List[np.ndarray] = []
        labels_list: List[np.ndarray] = []

        i = j = 0
        rate = self.target_sampling_rate
        min_overlap_sec = max(
            self.min_segment_sec,
            self.window_size / rate
        )

        while i < len(acc_segments) and j < len(gyro_segments):
            acc_seg = acc_segments[i]
            gyro_seg = gyro_segments[j]

            acc_start = acc_seg['start']
            gyro_start = gyro_seg['start']
            acc_end = acc_start + len(acc_seg['values']) / rate
            gyro_end = gyro_start + len(gyro_seg['values']) / rate

            overlap_start = max(acc_start, gyro_start)
            overlap_end = min(acc_end, gyro_end)

            if overlap_end - overlap_start >= min_overlap_sec:
                acc_slice, acc_labels = self._slice_segment(acc_seg, overlap_start, overlap_end)
                gyro_slice, _ = self._slice_segment(gyro_seg, overlap_start, overlap_end)

                if acc_slice.size and gyro_slice.size:
                    min_len = min(acc_slice.shape[0], gyro_slice.shape[0], acc_labels.shape[0])
                    if min_len >= self.window_size:
                        acc_slice = acc_slice[:min_len]
                        gyro_slice = gyro_slice[:min_len]
                        acc_labels = acc_labels[:min_len]

                        fused = np.concatenate([acc_slice, gyro_slice], axis=1)
                        windows, window_labels = create_sliding_windows(
                            fused,
                            acc_labels,
                            window_size=self.window_size,
                            stride=self.stride,
                            drop_last=True
                        )

                        if windows.size:
                            windows_list.append(windows.astype(np.float32))
                            labels_list.append(window_labels.astype(np.int32))

            if acc_end <= gyro_end:
                i += 1
            else:
                j += 1

        if not windows_list:
            return (
                np.empty((0, self.window_size, 3), dtype=np.float32),
                np.empty((0, self.window_size, 3), dtype=np.float32),
                np.empty((0,), dtype=np.int32)
            )

        windows = np.concatenate(windows_list, axis=0)
        window_labels = np.concatenate(labels_list, axis=0)
        acc_windows = windows[:, :, :3]
        gyro_windows = windows[:, :, 3:]
        return acc_windows, gyro_windows, window_labels

    def _slice_segment(
        self,
        segment: Dict[str, Any],
        overlap_start: float,
        overlap_end: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        rate = self.target_sampling_rate
        seg_start = segment['start']
        values = segment['values']
        labels = segment['labels']

        start_offset = max(0.0, overlap_start - seg_start)
        end_offset = max(0.0, overlap_end - seg_start)

        start_idx = int(np.round(start_offset * rate))
        end_idx = int(np.round(end_offset * rate))
        end_idx = min(end_idx, len(values))

        if end_idx - start_idx <= 0:
            return np.empty((0, values.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int32)

        return (
            values[start_idx:end_idx],
            labels[start_idx:end_idx].astype(np.int32)
        )