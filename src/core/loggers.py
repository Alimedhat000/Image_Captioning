"""Loggers for tracking experiments - TensorBoard, CSV."""

import csv
from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Base logger class."""

    def log_metrics(self, metrics: Dict[str, Union[float, torch.Tensor]], step: int):
        """Log metrics.

        Args:
            metrics: Dictionary of metric names and values
            step: Global step
        """
        raise NotImplementedError("Subclass must implement log_metrics()")

    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters.

        Args:
            params: Dictionary of hyperparameters
        """
        pass

    def finalize(self):
        """Finalize logger (close files, etc.)."""
        pass


class TensorBoardLogger(Logger):
    """TensorBoard logger."""

    def __init__(
        self,
        save_dir: str,
        name: str = "default",
        version: Optional[str] = None,
    ):
        """Initialize TensorBoard logger.

        Args:
            save_dir: Root directory for logs
            name: Experiment name
            version: Experiment version
        """
        self.save_dir = Path(save_dir)
        self.name = name
        self.version = version or "version_0"

        # Create log directory
        self.log_dir = self.save_dir / self.name / self.version
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        print(f"TensorBoard logging to: {self.log_dir}")

    def log_metrics(self, metrics: Dict[str, Union[float, torch.Tensor]], step: int):
        """Log metrics to TensorBoard."""
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.writer.add_scalar(key, value, step)

    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters to TensorBoard."""
        # Convert to string representation
        hparams_str = "\n".join([f"{k}: {v}" for k, v in params.items()])
        self.writer.add_text("hyperparameters", hparams_str, 0)

    def finalize(self):
        """Close writer."""
        self.writer.close()


class CSVLogger(Logger):
    """CSV logger."""

    def __init__(
        self,
        save_dir: str,
        name: str = "default",
        version: Optional[str] = None,
    ):
        """Initialize CSV logger.

        Args:
            save_dir: Root directory for logs
            name: Experiment name
            version: Experiment version
        """
        self.save_dir = Path(save_dir)
        self.name = name
        self.version = version or "version_0"

        # Create log directory
        self.log_dir = self.save_dir / self.name / self.version
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_file = self.log_dir / "metrics.csv"
        self.metrics_keys = set()

        print(f"CSV logging to: {self.metrics_file}")

    def log_metrics(self, metrics: Dict[str, Union[float, torch.Tensor]], step: int):
        """Log metrics to CSV."""
        # Convert tensors to floats
        metrics = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in metrics.items()
        }

        # Add step
        metrics["step"] = step

        # Update known keys
        self.metrics_keys.update(metrics.keys())

        # Write to file
        file_exists = self.metrics_file.exists()

        with open(self.metrics_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(self.metrics_keys))

            if not file_exists:
                writer.writeheader()

            writer.writerow(metrics)

    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters to separate file."""
        hparams_file = self.log_dir / "hparams.txt"

        with open(hparams_file, "w") as f:
            for key, value in params.items():
                f.write(f"{key}: {value}\n")

    def finalize(self):
        """Nothing to clean up for CSV."""
        pass


class MultiLogger(Logger):
    """Combine multiple loggers."""

    def __init__(self, loggers: list[Logger]):
        """Initialize MultiLogger.

        Args:
            loggers: List of logger instances
        """
        self.loggers = loggers

    def log_metrics(self, metrics: Dict[str, Union[float, torch.Tensor]], step: int):
        """Log to all loggers."""
        for logger in self.loggers:
            logger.log_metrics(metrics, step)

    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters to all loggers."""
        for logger in self.loggers:
            logger.log_hyperparameters(params)

    def finalize(self):
        """Finalize all loggers."""
        for logger in self.loggers:
            logger.finalize()
