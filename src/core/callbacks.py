"""Callbacks for training - ModelCheckpoint, EarlyStopping, RichProgressBar."""

from pathlib import Path
import torch
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from src.utils.rich_utils import print_text


class Callback:
    """Base callback class."""

    def on_train_start(self, trainer, model):
        """Called at the start of training."""
        pass

    def on_train_end(self, trainer, model):
        """Called at the end of training."""
        pass

    def on_epoch_start(self, trainer, model):
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, trainer, model):
        """Called at the end of each epoch."""
        pass

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        """Called before training batch."""
        pass

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        """Called after training batch."""
        pass

    def on_validation_start(self, trainer, model):
        """Called at the start of validation."""
        pass

    def on_validation_end(self, trainer, model):
        """Called at the end of validation."""
        pass

    def on_validation_batch_start(self, trainer, model, batch, batch_idx):
        """Called before validation batch."""
        pass

    def on_validation_batch_end(self, trainer, model, outputs, batch, batch_idx):
        """Called after validation batch."""
        pass


class ModelCheckpoint(Callback):
    """Save model checkpoints based on monitoring a metric."""

    def __init__(
        self,
        dirpath: str,
        filename: str = "epoch_{epoch:03d}",
        monitor: str = "val/loss",
        mode: str = "min",
        save_last: bool = True,
        save_top_k: int = 1,
        verbose: bool = True,
        **kwargs,
    ):
        """Initialize ModelCheckpoint.
        Args:
            dirpath: Directory to save checkpoints
            filename: Checkpoint filename format (can use {epoch}, {step})
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_last: Whether to save last checkpoint
            save_top_k: Number of best checkpoints to keep
            verbose: Print when saving
        """
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_last = save_last
        self.save_top_k = save_top_k
        self.verbose = verbose

        self.top_k_scores = []
        self.top_k_paths = []

        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.best_model_path = ""

        # Create directory
        self.dirpath.mkdir(parents=True, exist_ok=True)

    def _is_better(self, current: float, best: float) -> bool:
        """Check if current metric is better than best."""
        if self.mode == "min":
            return current < best
        return current > best

    def _get_live_console(self, trainer):
        """Find the console from the RichProgressBar if it exists."""
        for callback in trainer.callbacks:
            if isinstance(callback, RichProgressBar) and hasattr(callback, "live"):
                return callback.live.console
        return None

    def _save_file(self, trainer, model, filepath: Path) -> str:
        """Saves the checkpoint file."""
        checkpoint = {
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "state_dict": model.state_dict(),
            "optimizer_state": trainer.optimizer.state_dict(),
        }
        if trainer.scheduler is not None:
            checkpoint["scheduler_state"] = trainer.scheduler.state_dict()
        torch.save(checkpoint, filepath)
        return str(filepath)

    def _save_checkpoint(self, trainer, model, filename: str) -> str:
        """Saves a checkpoint and handles verbose printing."""
        filepath = self.dirpath / filename
        filepath_str = self._save_file(trainer, model, filepath)

        if self.verbose:
            metric_val = trainer.callback_metrics[self.monitor].item()
            console = self._get_live_console(trainer)
            msg = (
                f"\nSaved checkpoint for epoch {trainer.current_epoch}: "
                f"{self.monitor}={metric_val:.4f}"
            )
            if console:
                console.print(msg, style="bold green")
            else:
                print_text(msg, style="bold green")
        return filepath_str

    def on_validation_end(self, trainer, model):
        """Check and save checkpoints based on monitored metric."""
        metrics = trainer.callback_metrics
        if self.monitor not in metrics:
            return

        current_metric = metrics[self.monitor].item()

        if self._is_better(current_metric, self.best_metric):
            self.best_metric = current_metric

        if self.save_top_k > 0:
            is_full = len(self.top_k_scores) >= self.save_top_k
            worst_score = (
                (
                    max(self.top_k_scores)
                    if self.mode == "min"
                    else min(self.top_k_scores)
                )
                if is_full
                else None
            )

            if not is_full or self._is_better(current_metric, worst_score):
                if is_full:
                    idx_to_remove = self.top_k_scores.index(worst_score)
                    path_to_remove = self.top_k_paths.pop(idx_to_remove)
                    self.top_k_scores.pop(idx_to_remove)
                    Path(path_to_remove).unlink(missing_ok=True)
                    if self.verbose:
                        console = self._get_live_console(trainer)
                        msg = f"Removed checkpoint: {path_to_remove}"
                        if console:
                            console.print(msg, style="yellow")
                        else:
                            print_text(msg, style="yellow")

                filename = (
                    self.filename.format(
                        epoch=trainer.current_epoch, step=trainer.global_step
                    )
                    + ".ckpt"
                )
                filepath = self._save_checkpoint(trainer, model, filename)
                self.top_k_scores.append(current_metric)
                self.top_k_paths.append(filepath)
                self.best_model_path = filepath

        if self.save_last:
            self._save_file(trainer, model, self.dirpath / "last.ckpt")


class EarlyStopping(Callback):
    """Stop training when a monitored metric stops improving."""

    def __init__(
        self,
        monitor: str = "val/loss",
        patience: int = 3,
        mode: str = "min",
        min_delta: float = 0.0,
        verbose: bool = True,
        **kwargs,
    ):
        """Initialize EarlyStopping.

        Args:
            monitor: Metric to monitor
            patience: Number of epochs with no improvement after which training stops
            mode: 'min' or 'max'
            min_delta: Minimum change to qualify as improvement
            verbose: Print when stopping
        """
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose
        self.wait_count = 0
        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.stopped_epoch = 0

    def _is_better(self, current: float, best: float) -> bool:
        """Check if current metric is better than best."""
        if self.mode == "min":
            return current < (best - self.min_delta)
        return current > (best + self.min_delta)

    def _get_live_console(self, trainer):
        """Find the console from the RichProgressBar if it exists."""
        for callback in trainer.callbacks:
            if isinstance(callback, RichProgressBar) and hasattr(callback, "live"):
                return callback.live.console
        return None

    def on_validation_end(self, trainer, model):
        """Check if we should stop training."""
        metrics = trainer.callback_metrics
        if self.monitor not in metrics:
            return

        current_metric = metrics[self.monitor].item()

        if self._is_better(current_metric, self.best_metric):
            self.best_metric = current_metric
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                self.stopped_epoch = trainer.current_epoch
                trainer.should_stop = True
                if self.verbose:
                    console = self._get_live_console(trainer)
                    msg1 = f"\nEarly stopping triggered at epoch {self.stopped_epoch}"
                    msg2 = f"Best {self.monitor}: {self.best_metric:.4f}"
                    if console:
                        console.print(msg1, style="bold red")
                        console.print(msg2, style="bold red")
                    else:
                        print_text(msg1, style="bold red")
                        print_text(msg2, style="bold red")


class RichProgressBar(Callback):
    """Rich progress bar for training with a live-updating metrics table."""

    def __init__(self, refresh_rate: int = 4, metric_modes: dict = None, **kwargs):
        """Initialize RichProgressBar."""
        self.refresh_rate = refresh_rate
        self.metric_modes = metric_modes if metric_modes is not None else {}
        self.console = Console()
        self.progress = None
        self.live = None
        self.epoch_task = None
        self.train_task = None
        self.val_task = None
        self.best_metrics = {}

    def on_train_start(self, trainer, model):
        """Setup progress bars and live display."""
        self.progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        )
        metrics_table = Table(
            show_header=True, header_style="bold magenta", title="Metrics"
        )
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", justify="right", style="green")
        metrics_table.add_column("Best", justify="right", style="yellow")
        metrics_table.add_column("Change", justify="right")

        render_group = Group(self.progress, metrics_table)
        self.live = Live(
            render_group,
            console=self.console,
            refresh_per_second=self.refresh_rate,
            vertical_overflow="visible",
        )
        self.live.start()

        self.epoch_task = self.progress.add_task(
            "[cyan]Epochs", total=trainer.max_epochs
        )

    def on_train_end(self, trainer, model):
        """Clean up live display."""
        if self.live:
            self.live.stop()

    def on_epoch_start(self, trainer, model):
        """Start epoch progress."""
        self.progress.update(self.epoch_task, advance=1)
        self.train_task = self.progress.add_task(
            f"[green]Epoch {trainer.current_epoch} - Train",
            total=len(trainer.train_dataloader),
        )

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        """Update train progress."""
        if self.train_task is not None:
            self.progress.update(self.train_task, advance=1)

    def on_validation_start(self, trainer, model):
        """Start validation progress."""
        self.val_task = self.progress.add_task(
            f"[yellow]Epoch {trainer.current_epoch} - Val",
            total=len(trainer.val_dataloader),
        )

    def on_validation_batch_end(self, trainer, model, outputs, batch, batch_idx):
        """Update validation progress."""
        if self.val_task is not None:
            self.progress.update(self.val_task, advance=1)

    def on_validation_end(self, trainer, model):
        """Update metrics table."""
        if self.train_task is not None:
            self.progress.remove_task(self.train_task)
        if self.val_task is not None:
            self.progress.remove_task(self.val_task)

        metrics_table = Table(show_header=True, header_style="bold magenta")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", justify="right", style="green")
        metrics_table.add_column("Best", justify="right", style="yellow")
        metrics_table.add_column("Change", justify="right")

        metrics = trainer.callback_metrics
        if metrics:
            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()

                mode = self.metric_modes.get(
                    key, "min" if "loss" in key.lower() else "max"
                )

                best_value = self.best_metrics.get(key)

                if best_value is None:
                    self.best_metrics[key] = value
                    best_value = value
                    change_str = "[grey]N/A[/grey]"
                else:
                    change = value - best_value
                    is_improvement = (mode == "min" and change < 0) or (
                        mode == "max" and change > 0
                    )

                    if change == 0:
                        change_str = f"[grey]{change:+.4f}[/grey]"
                    elif is_improvement:
                        self.best_metrics[key] = value
                        change_str = f"[green]{change:+.4f}[/green]"
                    else:
                        change_str = f"[red]{change:+.4f}[/red]"

                metrics_table.add_row(
                    key, f"{value:.4f}", f"{self.best_metrics[key]:.4f}", change_str
                )

        render_group = Group(self.progress, metrics_table)
        self.live.update(render_group)
