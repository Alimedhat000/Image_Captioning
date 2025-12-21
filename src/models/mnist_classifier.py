"""MNIST Classifier model."""

from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

from src.models.base_module import BaseModule


class MNISTClassifier(BaseModule):
    """MNIST Classifier using any backbone network.

    This wraps any CNN architecture and adapts it for MNIST.
    """

    def __init__(
        self,
        net: nn.Module,
        optimizer: Optimizer,
        scheduler: Any = None,
        **kwargs,
    ):
        """Initialize MNIST Classifier.

        Args:
            net: Backbone network (ResNet, EfficientNet, etc.)
            optimizer: Optimizer (will be instantiated by Hydra)
            scheduler: Optional learning rate scheduler
        """
        super().__init__()

        self.net = net
        self.optimizer_config = optimizer
        self.scheduler_config = scheduler

        # For tracking accuracy
        self.train_acc = 0.0
        self.val_acc = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Training step.

        Args:
            batch: Tuple of (images, labels)
            batch_idx: Batch index

        Returns:
            Dictionary with loss and metrics
        """
        x, y = batch

        # Forward pass
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        return loss, {
            "train/loss": loss,
            "train/acc": acc,
        }

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validation step.

        Args:
            batch: Tuple of (images, labels)
            batch_idx: Batch index

        Returns:
            Dictionary with metrics
        """
        x, y = batch

        # Forward pass
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        return {
            "val/loss": loss,
            "val/acc": acc,
        }

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        # Instantiate optimizer
        optimizer = self.optimizer_config(params=self.parameters())

        # Optionally instantiate scheduler
        if self.scheduler_config is not None:
            scheduler = self.scheduler_config(optimizer=optimizer)
            return optimizer, scheduler

        return optimizer
