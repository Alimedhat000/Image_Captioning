"""Base model module"""

from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class BaseModule(nn.Module):
    """Base class for all models.

    Provides a Lightning-like interface with training_step, validation_step, etc.
    Your models should inherit from this and implement the required methods.
    """

    def __init__(self):
        super().__init__()
        self.automatic_optimization = True  # Can disable for manual optimization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - must be implemented by subclass."""
        raise NotImplementedError("Subclass must implement forward()")

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Training step - must be implemented by subclass.

        Args:
            batch: Batch of data from dataloader
            batch_idx: Index of batch

        Returns:
            Dictionary with 'loss' key and optional metrics
            Example: {'loss': loss, 'acc': accuracy}
        """
        raise NotImplementedError("Subclass must implement training_step()")

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step - must be implemented by subclass.

        Args:
            batch: Batch of data from dataloader
            batch_idx: Index of batch

        Returns:
            Dictionary with metrics
            Example: {'val/loss': loss, 'val/acc': accuracy}
        """
        raise NotImplementedError("Subclass must implement validation_step()")

    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step - optional, defaults to validation_step."""
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self) -> Optimizer | Tuple[Optimizer, _LRScheduler]:
        """Configure optimizer and optionally scheduler.

        Returns:
            Either:
            - optimizer
            - (optimizer, scheduler)
        """
        raise NotImplementedError("Subclass must implement configure_optimizers()")

    def on_train_epoch_start(self):
        """Called at the start of training epoch."""
        pass

    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        pass

    def on_validation_epoch_start(self):
        """Called at the start of validation epoch."""
        pass

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        pass
