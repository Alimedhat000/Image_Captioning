"""MNIST DataModule."""

from typing import Optional
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from src.data.datamodule import DataModule


class MNISTDataModule(DataModule):
    """DataModule for MNIST dataset."""

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_val_split: tuple = (55000, 5000),
        **kwargs,
    ):
        """Initialize MNIST DataModule.

        Args:
            data_dir: Root directory for data
            batch_size: Batch size
            num_workers: Number of dataloader workers
            pin_memory: Whether to pin memory
            train_val_split: Train/val split (must sum to 60000)
        """
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        self.train_val_split = train_val_split

        # Transforms
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
            ]
        )

    def prepare_data(self):
        """Download MNIST data."""
        # Download train and test sets
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Setup MNIST datasets."""
        if stage == "fit" or stage is None:
            # Load full training set
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)

            # Split into train and val
            self.train_dataset, self.val_dataset = random_split(
                mnist_full,
                self.train_val_split,
            )

        if stage == "test" or stage is None:
            # Load test set
            self.test_dataset = MNIST(
                self.data_dir, train=False, transform=self.transform
            )
