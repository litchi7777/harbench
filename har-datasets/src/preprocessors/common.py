"""
Common utilities for dataset download and extraction
"""

import requests
import zipfile
import tarfile
import shutil
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def download_file(url: str, output_path: Path, desc: Optional[str] = None, verify_ssl: bool = True) -> None:
    """
    Download a file with progress bar

    Args:
        url: Download URL
        output_path: Output file path
        desc: Progress bar description
        verify_ssl: Whether to verify SSL certificate (some servers require False)
    """
    import urllib3
    if not verify_ssl:
        # Suppress warnings when SSL verification is disabled
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    if desc is None:
        desc = 'Downloading'

    logger.info(f"Downloading from {url}")
    logger.info(f"Saving to {output_path}")
    if not verify_ssl:
        logger.warning("SSL verification disabled for this download")

    # Create directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Download
    response = requests.get(url, stream=True, verify=verify_ssl)
    response.raise_for_status()

    # Get file size
    total_size = int(response.headers.get('content-length', 0))

    # Download with progress bar
    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    logger.info(f"Download complete: {output_path}")


def extract_zip(zip_path: Path, extract_to: Path, desc: Optional[str] = None) -> Path:
    """
    Extract a ZIP file

    Args:
        zip_path: Path to ZIP file
        extract_to: Extraction destination directory
        desc: Progress bar description

    Returns:
        Path to extraction destination
    """
    if desc is None:
        desc = 'Extracting'

    logger.info(f"Extracting {zip_path} to {extract_to}")

    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get file count
        file_count = len(zip_ref.namelist())

        # Extract with progress bar
        with tqdm(total=file_count, desc=desc) as pbar:
            for file in zip_ref.namelist():
                zip_ref.extract(file, extract_to)
                pbar.update(1)

    logger.info(f"Extraction complete: {extract_to}")
    return extract_to


def extract_tar(tar_path: Path, extract_to: Path, desc: Optional[str] = None) -> Path:
    """
    Extract a TAR/TGZ file

    Args:
        tar_path: Path to TAR file
        extract_to: Extraction destination directory
        desc: Progress bar description

    Returns:
        Path to extraction destination
    """
    if desc is None:
        desc = 'Extracting'

    logger.info(f"Extracting {tar_path} to {extract_to}")

    extract_to.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, 'r:*') as tar_ref:
        # Get file count
        members = tar_ref.getmembers()
        file_count = len(members)

        # Extract with progress bar
        with tqdm(total=file_count, desc=desc) as pbar:
            for member in members:
                tar_ref.extract(member, extract_to)
                pbar.update(1)

    logger.info(f"Extraction complete: {extract_to}")
    return extract_to


def extract_rar(rar_path: Path, extract_to: Path, desc: Optional[str] = None) -> Path:
    """
    Extract a RAR archive (using unrar command)

    Args:
        rar_path: Path to RAR file
        extract_to: Extraction destination directory
        desc: Progress bar description

    Returns:
        Path to extraction destination
    """
    import subprocess

    if desc is None:
        desc = 'Extracting RAR'

    logger.info(f"Extracting {rar_path} to {extract_to}")

    extract_to.mkdir(parents=True, exist_ok=True)

    # Extract using unrar command
    # x: extract with full path, -y: assume Yes for all prompts
    try:
        result = subprocess.run(
            ['unrar', 'x', '-y', str(rar_path), str(extract_to) + '/'],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Extraction complete: {extract_to}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to extract RAR: {e.stderr}")
        raise
    except FileNotFoundError:
        raise RuntimeError(
            "unrar command not found. "
            "Please install it: sudo apt-get install unrar"
        )

    return extract_to


def extract_archive(archive_path: Path, extract_to: Path, desc: Optional[str] = None) -> Path:
    """
    Auto-detect and extract an archive file

    Args:
        archive_path: Path to archive file
        extract_to: Extraction destination directory
        desc: Progress bar description

    Returns:
        Path to extraction destination
    """
    suffix = archive_path.suffix.lower()

    if suffix == '.zip':
        return extract_zip(archive_path, extract_to, desc)
    elif suffix in ['.tar', '.gz', '.tgz', '.bz2', '.xz']:
        return extract_tar(archive_path, extract_to, desc)
    elif suffix == '.rar':
        return extract_rar(archive_path, extract_to, desc)
    else:
        raise ValueError(f"Unsupported archive format: {suffix}")


def cleanup_temp_files(temp_dir: Path) -> None:
    """
    Clean up temporary files

    Args:
        temp_dir: Path to temporary directory
    """
    if temp_dir.exists():
        logger.info(f"Cleaning up temporary files: {temp_dir}")
        shutil.rmtree(temp_dir)
        logger.info("Cleanup complete")


def check_dataset_exists(dataset_path: Path, required_files: Optional[list] = None) -> bool:
    """
    Check if dataset already exists

    Args:
        dataset_path: Path to dataset
        required_files: List of required files (pattern matching supported)

    Returns:
        True if exists
    """
    if not dataset_path.exists():
        return False

    if required_files is None:
        # OK if directory exists
        return True

    # Check required files
    for pattern in required_files:
        if not list(dataset_path.glob(pattern)):
            logger.warning(f"Required file pattern not found: {pattern}")
            return False

    return True


def move_or_copy_directory(src: Path, dst: Path, move: bool = True) -> None:
    """
    Move or copy a directory

    Args:
        src: Source path
        dst: Destination path
        move: True=move, False=copy
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        shutil.rmtree(dst)

    if move:
        shutil.move(str(src), str(dst))
        logger.info(f"Moved {src} -> {dst}")
    else:
        shutil.copytree(src, dst)
        logger.info(f"Copied {src} -> {dst}")
