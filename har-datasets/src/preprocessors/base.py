"""
Base preprocessing class
All dataset-specific preprocessing classes inherit from this class
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BasePreprocessor(ABC):
    """
    Base class for dataset preprocessing

    All dataset-specific preprocessing classes must inherit from this class
    and implement the required methods.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Preprocessing configuration (dictionary loaded from YAML)
        """
        self.config = config
        self.dataset_name = self.get_dataset_name()
        self.raw_data_path = Path(config.get('raw_data_path', 'data/raw'))
        self.processed_data_path = Path(config.get('processed_data_path', 'data/processed'))

        self.setup_paths()

    @abstractmethod
    def get_dataset_name(self) -> str:
        """
        Return the dataset name

        Returns:
            Dataset name (e.g., 'dsads', 'opportunity')
        """
        pass

    def download_dataset(self) -> None:
        """
        Download the dataset (optional)

        Implement as needed for each dataset.
        Raises NotImplementedError if not implemented.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement download_dataset(). "
            "Please download the dataset manually or implement this method."
        )

    @abstractmethod
    def load_raw_data(self) -> Any:
        """
        Load raw data

        Returns:
            Loaded raw data (format varies by dataset)
        """
        pass

    @abstractmethod
    def clean_data(self, data: Any) -> Any:
        """
        Data cleaning process

        Args:
            data: Raw data

        Returns:
            Cleaned data
        """
        pass

    @abstractmethod
    def extract_features(self, data: Any) -> Any:
        """
        Feature extraction process

        Args:
            data: Cleaned data

        Returns:
            Extracted features data
        """
        pass

    @abstractmethod
    def save_processed_data(self, data: Any) -> None:
        """
        Save processed data

        Args:
            data: Processed data
        """
        pass

    def setup_paths(self) -> None:
        """
        Create necessary directories
        """
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)

        # Dataset-specific directory
        dataset_processed_path = self.processed_data_path / self.dataset_name
        dataset_processed_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Setup paths for {self.dataset_name}")
        logger.info(f"  Raw data: {self.raw_data_path}")
        logger.info(f"  Processed data: {dataset_processed_path}")

    def validate_raw_data(self) -> bool:
        """
        Validate that raw data exists

        Returns:
            True if raw data exists
        """
        if not self.raw_data_path.exists():
            logger.error(f"Raw data path does not exist: {self.raw_data_path}")
            return False
        return True

    def preprocess(self) -> None:
        """
        Execute the entire preprocessing pipeline

        This method typically does not need to be overridden.
        Customize preprocessing by overriding individual step methods.
        """
        logger.info(f"Starting preprocessing for {self.dataset_name}")

        # 1. Validate raw data
        if not self.validate_raw_data():
            raise FileNotFoundError(f"Raw data not found for {self.dataset_name}")

        # 2. Load data
        logger.info("Loading raw data...")
        raw_data = self.load_raw_data()

        # 3. Clean data
        logger.info("Cleaning data...")
        cleaned_data = self.clean_data(raw_data)

        # 4. Extract features
        logger.info("Extracting features...")
        processed_data = self.extract_features(cleaned_data)

        # 5. Save
        logger.info("Saving processed data...")
        self.save_processed_data(processed_data)

        logger.info(f"Preprocessing completed for {self.dataset_name}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics of processed data (optional)

        Returns:
            Dictionary of statistics
        """
        return {
            "dataset": self.dataset_name,
            "raw_data_path": str(self.raw_data_path),
            "processed_data_path": str(self.processed_data_path),
        }
