"""Base DataModule class for organizing data loading."""

from typing import Optional
from torch.utils.data import DataLoader, Dataset


class DataModule:
    """Base class for data modules.

    Organizes data loading logic - setup datasets, create dataloaders.
    Similar to LightningDataModule but simpler.
    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        """Initialize DataModule.

        Args:
            data_dir: Root directory for data
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            pin_memory: Whether to pin memory (useful for GPU)
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Datasets will be set in setup()
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def prepare_data(self):
        """Download data, tokenize, etc.

        This is called only once and on a single process.
        Do not set state here (e.g., self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Setup datasets.

        Called on every process. Can set state here (e.g., self.train_dataset = ...).

        Args:
            stage: Either 'fit', 'validate', 'test', or None
        """
        raise NotImplementedError("Subclass must implement setup()")

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        if self.train_dataset is None:
            raise ValueError("train_dataset is None. Did you call setup()?")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        if self.val_dataset is None:
            raise ValueError("val_dataset is None. Did you call setup()?")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        if self.test_dataset is None:
            raise ValueError("test_dataset is None. Did you call setup()?")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def get_vocab_size(self):
        """Get vocabulary size."""
        if self.vocab is None:
            raise RuntimeError("Vocabulary not built. Call setup() first.")
        return len(self.vocab)

    def get_vocab(self):
        """Get vocabulary object."""
        if self.vocab is None:
            raise RuntimeError("Vocabulary not built. Call setup() first.")
        return self.vocab
