"""
Preprocessing for NHANES PAX-G data

Downloads and processes data from CDC FTP server one user at a time,
then deletes files after processing to save storage space.
"""

import tarfile
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import urllib.request
from urllib.error import URLError, HTTPError
import time
import logging

import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm

from .base import BasePreprocessor
from . import register_preprocessor

logger = logging.getLogger(__name__)


class NHANESPAXProcessor:
    """Preprocessing class for NHANES PAX-G data"""
    
    def __init__(
        self,
        output_base: str = "/mnt/home/processed_data/NHANES_PAX",
        window_size: int = 5,  # seconds
        sampling_rate: int = 80,  # Hz
        target_sampling_rate: Optional[int] = None,  # Hz (no resampling if None)
        std_threshold: float = 0.02,  # threshold for sum of standard deviations
        temp_dir: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: int = 5
    ):
        """
        Args:
            output_base: Directory to save processed data
            window_size: Window size (seconds)
            sampling_rate: Original sampling rate (Hz) - NHANES PAX is 80Hz
            target_sampling_rate: Target sampling rate (Hz) - no resampling if None
            std_threshold: Standard deviation threshold for activity detection
            temp_dir: Temporary directory (system default if None)
            max_retries: Maximum number of retries on download failure
            retry_delay: Delay between retries (seconds)
        """
        self.output_base = Path(output_base)
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.target_sampling_rate = target_sampling_rate if target_sampling_rate is not None else sampling_rate
        self.std_threshold = std_threshold
        self.temp_dir = temp_dir
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Window length (number of samples) - calculated with post-resampling rate
        self.window_length = window_size * self.target_sampling_rate

        # Create output directory
        self.output_base.mkdir(parents=True, exist_ok=True)

        # FTP server base URLs (two sources: PAX-G and PAX-H)
        self.ftp_base_g = "https://ftp.cdc.gov/pub/pax_g/"
        self.ftp_base_h = "https://ftp.cdc.gov/pub/pax_h/"
        self.pax_h_start_id = 73557  # Start ID for PAX-H

        # Progress tracking files
        self.progress_file = self.output_base / "processing_progress.json"
        self.failed_users_file = self.output_base / "failed_users.json"
    
    def check_user_exists(self, user_id: int) -> bool:
        """
        Check if a file for a specific user exists on the FTP server

        Args:
            user_id: User ID

        Returns:
            Whether the file exists
        """
        import urllib.request
        from urllib.error import URLError, HTTPError

        # Check if exists in either PAX-G or PAX-H
        if user_id < self.pax_h_start_id:
            url = f"{self.ftp_base_g}{user_id}.tar.bz2"
        else:
            url = f"{self.ftp_base_h}{user_id}.tar.bz2"

        try:
            # Check file existence with HEAD request (without downloading)
            req = urllib.request.Request(url)
            req.get_method = lambda: 'HEAD'
            response = urllib.request.urlopen(req, timeout=10)
            return response.status == 200
        except HTTPError as e:
            return e.code == 200
        except (URLError, Exception):
            return False
    
    def discover_available_users(self, start_id: int = 62161, end_id: int = 72161,
                               batch_size: int = 100) -> List[int]:
        """
        Automatically discover available user IDs

        Args:
            start_id: Start ID
            end_id: End ID
            batch_size: Number of IDs to check at once

        Returns:
            List of user IDs that actually exist
        """
        print(f"Discovering available users from {start_id} to {end_id}...")
        
        available_users = []
        total_range = end_id - start_id + 1

        # Check by batch
        for batch_start in tqdm(range(start_id, end_id + 1, batch_size),
                               desc="Checking user availability"):
            batch_end = min(batch_start + batch_size - 1, end_id)
            batch_ids = list(range(batch_start, batch_end + 1))

            # Check in parallel
            from multiprocessing import Pool
            with Pool(processes=4) as pool:
                results = pool.map(self.check_user_exists, batch_ids)

            # Add existing IDs
            for user_id, exists in zip(batch_ids, results):
                if exists:
                    available_users.append(user_id)
        
        print(f"Found {len(available_users)} available users out of {total_range} checked")
        if available_users:
            print(f"Available user IDs: {sorted(available_users)}")
        return available_users
    
    def get_user_ids(self, start_id: int = 62161, end_id: int = 72161,
                    discover: bool = False) -> List[int]:
        """
        Generate list of user IDs to process

        Args:
            start_id: Start ID
            end_id: End ID
            discover: Whether to get only user IDs that actually exist

        Returns:
            List of user IDs
        """
        if discover:
            return self.discover_available_users(start_id, end_id)
        else:
            return list(range(start_id, end_id + 1))
    
    def save_progress(self, processed_users: List[int], failed_users: List[int],
                     current_batch: int = 0, total_batches: int = 0) -> None:
        """
        Save processing progress

        Args:
            processed_users: List of processed user IDs
            failed_users: List of failed user IDs
            current_batch: Current batch number
            total_batches: Total number of batches
        """
        import json
        from datetime import datetime
        
        progress_data = {
            "last_updated": datetime.now().isoformat(),
            "processed_users": processed_users,
            "failed_users": failed_users,
            "current_batch": current_batch,
            "total_batches": total_batches,
            "total_processed": len(processed_users),
            "total_failed": len(failed_users)
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    def load_progress(self) -> Tuple[List[int], List[int], int, int]:
        """
        Load processing progress

        Returns:
            (processed_users, failed_users, current_batch, total_batches)
        """
        import json
        
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                return (
                    data.get("processed_users", []),
                    data.get("failed_users", []),
                    data.get("current_batch", 0),
                    data.get("total_batches", 0)
                )
            except Exception as e:
                print(f"Warning: Could not load progress file: {e}")
        
        return [], [], 0, 0
    
    def get_remaining_users(self, all_users: List[int]) -> List[int]:
        """
        Get remaining user IDs that need processing

        Args:
            all_users: List of all user IDs

        Returns:
            List of user IDs that need processing
        """
        processed_users, failed_users, _, _ = self.load_progress()
        completed_users = set(processed_users + failed_users)

        # Also check file existence
        remaining_users = []
        for user_id in all_users:
            if user_id not in completed_users:
                output_path = self.output_base / "nhanes" / f"USER{user_id}" / "PAX" / "ACC" / "X.npy"
                if not output_path.exists():
                    remaining_users.append(user_id)
                else:
                    # Record if file exists but not in progress log
                    processed_users.append(user_id)

        # Update progress
        if len(processed_users) != len(set(processed_users)):
            processed_users = list(set(processed_users))
            self.save_progress(processed_users, failed_users)
        
        return remaining_users
    
    def download_user_data(self, user_id: int, temp_path: Path) -> bool:
        """
        Download data for a specific user

        Args:
            user_id: User ID
            temp_path: Temporary path for download

        Returns:
            Whether download was successful
        """
        # Download from either PAX-G or PAX-H
        if user_id < self.pax_h_start_id:
            url = f"{self.ftp_base_g}{user_id}.tar.bz2"
        else:
            url = f"{self.ftp_base_h}{user_id}.tar.bz2"
            
        download_path = temp_path / f"{user_id}.tar.bz2"
        
        for attempt in range(self.max_retries):
            try:
                # pbar for download progress
                pbar = None

                def show_progress(block_num, block_size, total_size):
                    nonlocal pbar
                    if pbar is None:
                        pbar = tqdm(
                            total=total_size,
                            unit='B',
                            unit_scale=True,
                            desc=f"Downloading user {user_id}"
                        )
                    downloaded = block_num * block_size
                    if downloaded < total_size:
                        pbar.update(block_size)
                    else:
                        pbar.update(total_size - pbar.n)
                        pbar.close()

                # Execute download (with progress display)
                urllib.request.urlretrieve(url, download_path, reporthook=show_progress)
                
                if pbar and not pbar.disable:
                    pbar.close()
                    
                return True
                
            except HTTPError as e:
                if pbar and not pbar.disable:
                    pbar.close()

                # Fail immediately for 404 error (file does not exist)
                if e.code == 404:
                    return False

                # Retry for other HTTP errors
                if attempt < self.max_retries - 1:
                    print(f"Download failed for user {user_id}, attempt {attempt + 1}/{self.max_retries}: {e}")
                    time.sleep(self.retry_delay)
                else:
                    print(f"Failed to download user {user_id} after {self.max_retries} attempts: {e}")
                    return False
            except URLError as e:
                if pbar and not pbar.disable:
                    pbar.close()

                # Retry for network errors
                if attempt < self.max_retries - 1:
                    print(f"Download failed for user {user_id}, attempt {attempt + 1}/{self.max_retries}: {e}")
                    time.sleep(self.retry_delay)
                else:
                    print(f"Failed to download user {user_id} after {self.max_retries} attempts: {e}")
                    return False
            except Exception as e:
                if pbar and not pbar.disable:
                    pbar.close()
                    
                print(f"Unexpected error downloading user {user_id}: {e}")
                return False
        
        return False
    
    def extract_archive(self, archive_path: Path, extract_path: Path) -> bool:
        """
        Extract tar.bz2 archive

        Args:
            archive_path: Path to archive file
            extract_path: Path to extract to

        Returns:
            Whether extraction was successful
        """
        try:
            with tarfile.open(archive_path, 'r:bz2') as tar:
                tar.extractall(extract_path)
            return True
        except Exception as e:
            print(f"Failed to extract {archive_path}: {e}")
            return False
    
    def process_sensor_file(self, file_path: Path) -> Optional[np.ndarray]:
        """
        Process individual sensor CSV file (including resampling)

        Args:
            file_path: Path to CSV file

        Returns:
            Processed sensor data (3 channels × samples)
        """
        try:
            # Read CSV
            df = pd.read_csv(
                file_path,
                names=["timestamp", "X", "Y", "Z"],
                skiprows=1,
                on_bad_lines='skip',
                dtype={"timestamp": str, "X": float, "Y": float, "Z": float}
            )

            # Convert to numeric (errors become NaN)
            x = pd.to_numeric(df["X"], errors='coerce').values
            y = pd.to_numeric(df["Y"], errors='coerce').values
            z = pd.to_numeric(df["Z"], errors='coerce').values

            # Exclude NaNs
            valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
            x = x[valid_mask]
            y = y[valid_mask]
            z = z[valid_mask]

            if len(x) == 0:
                return None

            # Resampling (if necessary)
            if self.target_sampling_rate != self.sampling_rate:
                # Resample with polyphase filtering (same method as other datasets)
                from math import gcd

                # Simplify ratio (e.g., 80Hz -> 30Hz = up=3, down=8)
                multiplier = 1000
                up = int(self.target_sampling_rate * multiplier)
                down = int(self.sampling_rate * multiplier)
                common_divisor = gcd(up, down)
                up = up // common_divisor
                down = down // common_divisor

                # Resample each channel with polyphase filtering
                x = signal.resample_poly(x, up, down)
                y = signal.resample_poly(y, up, down)
                z = signal.resample_poly(z, up, down)

            # Return in (3, samples) shape, optimized with float16
            result = np.stack([x, y, z], axis=0).astype(np.float16)
            return result

        except Exception as e:
            print(f"Error processing sensor file {file_path}: {e}")
            return None
    
    def extract_valid_windows(self, data: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract windows with activity above threshold from data

        Args:
            data: Sensor data (3 channels × samples)

        Returns:
            Valid windows (num_windows × 3 channels × window_length)
        """
        total_length = data.shape[1]
        n_windows = total_length // self.window_length

        if n_windows == 0:
            return None

        # Split into windows
        segments = data[:, :n_windows * self.window_length].reshape(
            3, n_windows, self.window_length
        )

        # Calculate standard deviation for each window
        stds = np.std(segments, axis=2)  # shape: (3, n_windows)
        std_sum = np.sum(stds, axis=0)   # shape: (n_windows,)

        # Extract windows above threshold
        valid_mask = std_sum >= self.std_threshold
        valid_segments = segments[:, valid_mask, :]  # shape: (3, n_valid, window_length)

        if valid_segments.shape[1] > 0:
            # Convert to (n_valid, 3, window_length) shape
            return valid_segments.transpose(1, 0, 2)
        else:
            return None
    
    def process_user(self, user_id: int) -> Tuple[bool, str]:
        """
        Process data for one user (download → process → save → delete)

        Args:
            user_id: User ID

        Returns:
            (success flag, message)
        """
        # Check if already processed (check for X.npy)
        output_dir = self.output_base / "nhanes" / f"USER{user_id}" / "PAX" / "ACC"
        data_path = output_dir / "X.npy"
        if data_path.exists():
            return True, f"User {user_id} already processed, skipping"

        # Create temporary directory
        with tempfile.TemporaryDirectory(dir=self.temp_dir) as temp_dir:
            temp_path = Path(temp_dir)

            # 1. Download
            if not self.download_user_data(user_id, temp_path):
                return False, f"Failed to download user {user_id}"

            # 2. Extract
            archive_path = temp_path / f"{user_id}.tar.bz2"
            extract_path = temp_path / "extracted"
            extract_path.mkdir(exist_ok=True)

            if not self.extract_archive(archive_path, extract_path):
                return False, f"Failed to extract user {user_id}"

            # Delete archive as it's no longer needed
            archive_path.unlink()

            # 3. Process sensor files
            sensor_files = list(extract_path.glob("**/*.sensor.csv"))
            if not sensor_files:
                return False, f"No sensor files found for user {user_id}"
            
            all_segments = []
            for sensor_file in sensor_files:
                # Read sensor data
                data = self.process_sensor_file(sensor_file)
                if data is None:
                    continue

                # Extract valid windows
                valid_windows = self.extract_valid_windows(data)
                if valid_windows is not None:
                    all_segments.append(valid_windows)

            # 4. Save data
            if all_segments:
                # Concatenate all segments
                user_data = np.concatenate(all_segments, axis=0)

                # Save as float16 (memory efficiency)
                user_data = user_data.astype(np.float16)

                # Generate labels (all -1, float16)
                labels = np.full(user_data.shape[0], -1, dtype=np.float16)

                # Create save directory
                output_dir.mkdir(parents=True, exist_ok=True)

                # Save data
                np.save(data_path, user_data)
                np.save(output_dir / "Y.npy", labels)

                return True, f"Successfully processed user {user_id}: {user_data.shape[0]} windows saved"
            else:
                return False, f"No valid windows found for user {user_id}"
    
    def process_all_users(
        self,
        user_ids: Optional[List[int]] = None,
        start_id: int = 62161,
        end_id: int = 72161,
        parallel: bool = False,
        n_workers: int = 4
    ):
        """
        Process all users (with progress tracking)

        Args:
            user_ids: List of user IDs to process (range-based if None)
            start_id: Start ID (if user_ids is None)
            end_id: End ID (if user_ids is None)
            parallel: Whether to use parallel processing
            n_workers: Number of workers for parallel processing
        """
        # Prepare user ID list
        if user_ids is None:
            user_ids = self.get_user_ids(start_id, end_id)

        # Load progress
        processed_users, failed_users, last_batch, total_batches = self.load_progress()

        # Get remaining users
        remaining_users = self.get_remaining_users(user_ids)
        
        print(f"Total users: {len(user_ids)}")
        print(f"Already processed: {len(processed_users)}")
        print(f"Previously failed: {len(failed_users)}")
        print(f"Remaining to process: {len(remaining_users)}")
        
        if not remaining_users:
            print("All users have been processed!")
            return
        
        print(f"Resuming processing from user batch...")

        if parallel:
            # Parallel processing (multiprocessing)
            from multiprocessing import Pool

            with Pool(processes=n_workers) as pool:
                results = list(tqdm(
                    pool.imap_unordered(self.process_user, remaining_users),
                    total=len(remaining_users),
                    desc="Processing remaining users"
                ))
        else:
            # Sequential processing (with progress saving)
            results = []
            for i, user_id in enumerate(tqdm(remaining_users, desc="Processing remaining users")):
                result = self.process_user(user_id)
                results.append(result)

                success, message = result
                if success and "already processed" not in message:
                    processed_users.append(user_id)
                    print(f"✓ {message}")
                else:
                    failed_users.append(user_id)
                    print(f"⚠️  {message}")

                # Save progress periodically (every 10 users)
                if (i + 1) % 10 == 0:
                    self.save_progress(processed_users, failed_users)
                    print(f"Progress saved: {len(processed_users)} processed, {len(failed_users)} failed")

        # Save final progress
        if not parallel:
            self.save_progress(processed_users, failed_users)

        # Display statistics
        successful = sum(1 for success, _ in results if success)
        failed = len(results) - successful
        
        print(f"\n" + "="*60)
        print(f"Processing complete!")
        print(f"✓ Successful this session: {successful}/{len(results)}")
        print(f"✗ Failed this session: {failed}/{len(results)}")
        print(f"📊 Total processed: {len(processed_users)}/{len(user_ids)}")
        print(f"📊 Total failed: {len(failed_users)}/{len(user_ids)}")
        print("="*60)


@register_preprocessor('nhanes')
class NHANESPreprocessor(BasePreprocessor):
    """Preprocessing class for NHANES PAX-G data (integrated version)"""
    
    def get_dataset_name(self) -> str:
        return "nhanes"
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # NHANES-specific settings (two sources: PAX-G and PAX-H)
        self.ftp_base_g = "https://ftp.cdc.gov/pub/pax_g/"
        self.ftp_base_h = "https://ftp.cdc.gov/pub/pax_h/"
        self.pax_h_start_id = 73557  # Start ID for PAX-H
        self.window_size = config.get('window_size', 5)  # seconds
        self.sampling_rate = config.get('sampling_rate', 80)  # Hz (original sampling rate)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz (resample target, default 30Hz)
        self.std_threshold = config.get('std_threshold', 0.02)
        self.start_id = config.get('start_id', 62161)
        self.end_id = config.get('end_id', 62170)
        self.batch_size = config.get('batch_size', 100)
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 5)
        self.parallel = config.get('parallel', True)
        self.n_workers = config.get('workers', 4)

        # Final sampling rate
        final_rate = self.target_sampling_rate if self.target_sampling_rate is not None else self.sampling_rate

        # Window length (number of samples) - calculated with post-resampling rate
        self.window_length = self.window_size * final_rate

        # Initialize processor
        self.processor = NHANESPAXProcessor(
            output_base=str(self.processed_data_path),
            window_size=self.window_size,
            sampling_rate=self.sampling_rate,
            target_sampling_rate=self.target_sampling_rate,
            std_threshold=self.std_threshold,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay
        )
    
    def download_dataset(self) -> None:
        """
        Download dataset

        For NHANES, downloads and processes are done per user,
        so nothing is done here (executed in process_all_users)
        """
        logger.info("NHANES PAX-G dataset will be downloaded automatically during processing")
    
    def load_raw_data(self) -> List[int]:
        """
        Return list of user IDs to process

        Returns:
            List of user IDs
        """
        # Check if discovery mode is enabled in config
        discover_mode = self.config.get('discover_users', False)
        
        if discover_mode:
            logger.info("User discovery mode enabled. Scanning for available users...")
            user_ids = self.processor.discover_available_users(
                start_id=self.start_id,
                end_id=self.end_id,
                batch_size=self.config.get('discovery_batch_size', 100)
            )
            logger.info(f"Discovered {len(user_ids)} available users: {sorted(user_ids)}")
        else:
            user_ids = list(range(self.start_id, self.end_id + 1))
            logger.info(f"Using ID range mode: {len(user_ids)} users ({self.start_id} to {self.end_id})")
        
        return user_ids
    
    def clean_data(self, user_ids: List[int]) -> List[int]:
        """
        Clean data (exclude already processed users)

        Args:
            user_ids: List of all user IDs

        Returns:
            List of user IDs to process
        """
        # Use progress tracking functionality
        remaining_users = self.processor.get_remaining_users(user_ids)
        
        already_processed = len(user_ids) - len(remaining_users)
        if already_processed > 0:
            logger.info(f"Skipping {already_processed} already processed users")
        
        logger.info(f"Remaining users to process: {len(remaining_users)}")
        return remaining_users
    
    def extract_features(self, user_ids: List[int]) -> Dict[str, Any]:
        """
        Extract features (process user data)

        Args:
            user_ids: List of user IDs to process

        Returns:
            Statistics of processing results
        """
        if not user_ids:
            logger.info("No users to process")
            return {"processed_users": 0, "total_users": 0}

        # Process by batch
        total_users = len(user_ids)
        processed_users = 0
        failed_users = 0

        for i in range(0, len(user_ids), self.batch_size):
            batch_users = user_ids[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (len(user_ids) + self.batch_size - 1) // self.batch_size

            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_users)} users)")

            # Batch processing
            if self.parallel:
                from multiprocessing import Pool

                with Pool(processes=self.n_workers) as pool:
                    results = list(tqdm(
                        pool.imap_unordered(self.processor.process_user, batch_users),
                        total=len(batch_users),
                        desc=f"Batch {batch_num}"
                    ))
            else:
                results = []
                for user_id in tqdm(batch_users, desc=f"Batch {batch_num}"):
                    results.append(self.processor.process_user(user_id))

            # Aggregate results
            batch_success = sum(1 for success, _ in results if success)
            batch_failed = len(results) - batch_success

            processed_users += batch_success
            failed_users += batch_failed

            # Log failed users with reasons
            failed_reasons = [(user_id, msg) for (success, msg), user_id in zip(results, batch_users) if not success]
            if failed_reasons:
                for user_id, msg in failed_reasons[:5]:  # Show first 5 failures
                    logger.warning(f"Failed user {user_id}: {msg}")
                if len(failed_reasons) > 5:
                    logger.warning(f"... and {len(failed_reasons) - 5} more failures")

            logger.info(f"Batch {batch_num} complete: {batch_success}/{len(batch_users)} succeeded")

        # Return statistics
        return {
            "total_users": total_users,
            "processed_users": processed_users,
            "failed_users": failed_users,
            "success_rate": processed_users / total_users if total_users > 0 else 0
        }
    
    def save_processed_data(self, stats: Dict[str, Any]) -> None:
        """
        Save processed data (save statistics)

        Args:
            stats: Statistics of processing results
        """
        import json
        import numpy as np
        from .utils import get_class_distribution

        # Save statistics to JSON file
        stats_file = self.processed_data_path / "processing_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Processing statistics saved to {stats_file}")

        # Generate metadata.json (for consistency with other datasets)
        base_path = self.processed_data_path
        metadata = {
            'dataset': 'nhanes',
            'sensor_names': ['PAX'],
            'modalities': ['ACC'],
            'channels_per_modality': 3,
            'original_sampling_rate': 80,
            'target_sampling_rate': self.config.get('target_sampling_rate', 80),
            'window_size': self.config.get('window_size', 400),
            'stride': self.config.get('stride', 80),
            'normalization': 'none',
            'scale_factor': None,
            'data_dtype': 'float16',
            'users': {}
        }

        # Read data for each user and collect statistics
        user_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('USER')])
        logger.info(f"Collecting metadata for {len(user_dirs)} users...")

        for user_dir in user_dirs:
            user_name = user_dir.name
            acc_path = user_dir / 'PAX' / 'ACC'

            if not acc_path.exists():
                continue

            X_path = acc_path / 'X.npy'
            Y_path = acc_path / 'Y.npy'

            if not X_path.exists() or not Y_path.exists():
                continue

            try:
                # Load only header to get shape (memory efficiency)
                X_shape = np.load(X_path, mmap_mode='r').shape
                Y = np.load(Y_path)

                metadata['users'][user_name] = {
                    'sensor_modalities': {
                        'PAX/ACC': {
                            'X_shape': list(X_shape),
                            'Y_shape': list(Y.shape),
                            'num_windows': int(len(Y)),
                            'class_distribution': get_class_distribution(Y)
                        }
                    }
                }
            except Exception as e:
                logger.warning(f"Failed to read metadata for {user_name}: {e}")
                continue

        # Save metadata.json
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

            serializable_metadata = recursive_convert(metadata)
            json.dump(serializable_metadata, f, indent=2)

        logger.info(f"Metadata saved to {metadata_path}")
        logger.info(f"Total users in metadata: {len(metadata['users'])}")
    
    def preprocess(self) -> None:
        """
        Main preprocessing routine
        """
        logger.info("Starting NHANES PAX-G preprocessing...")

        # 1. Get list of user IDs to process
        user_ids = self.load_raw_data()

        # 2. Data cleaning (exclude already processed)
        remaining_users = self.clean_data(user_ids)

        # 3. Feature extraction (download → process → save)
        stats = self.extract_features(remaining_users)

        # 4. Save statistics
        self.save_processed_data(stats)

        logger.info("NHANES PAX-G preprocessing completed!")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics

        Returns:
            Dictionary of statistics
        """
        stats_file = self.processed_data_path / "processing_stats.json"

        if stats_file.exists():
            import json
            with open(stats_file, 'r') as f:
                return json.load(f)
        else:
            # Count processed files if statistics file doesn't exist
            processed_count = len(list(self.processed_data_path.glob("nhanes/USER*/PAX/ACC/X.npy")))
            return {
                "processed_users": processed_count,
                "total_users": self.end_id - self.start_id + 1
            }