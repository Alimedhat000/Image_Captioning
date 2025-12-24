"""Main Trainer class - handles training loop, validation, callbacks, logging."""

from typing import Dict, List, Optional, Union
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.profiler import ProfilerActivity, profile, record_function

from src.models.base_module import BaseModule
from src.data.datamodule import DataModule
from src.core.callbacks import Callback
from src.core.loggers import Logger, MultiLogger
from src.utils.rich_utils import print_text


class Trainer:
    """Main trainer class.

    Handles the training loop, validation, callbacks, and logging.
    Similar to PyTorch Lightning's Trainer but simpler.
    """

    def __init__(
        self,
        max_epochs: int = 10,
        accelerator: str = "auto",
        devices: int = 1,
        log_every_n_steps: int = 50,
        check_val_every_n_epoch: int = 1,
        callbacks: Optional[List[Callback]] = None,
        logger: Optional[Union[Logger, List[Logger]]] = None,
        gradient_clip_val: Optional[float] = None,
        profiler: Optional[str] = None,
        profile_dir: str = "logs/profiler",
        limit_train_batches: Optional[Union[int, float]] = None,
        limit_val_batches: Optional[Union[int, float]] = None,
    ):
        """Initialize Trainer.

        Args:
            max_epochs: Maximum number of epochs
            accelerator: 'auto', 'cpu', or 'gpu'
            devices: Number of devices (we only support 1)
            log_every_n_steps: Log every N training steps
            check_val_every_n_epoch: Run validation every N epochs
            callbacks: List of callbacks
            logger: Logger or list of loggers
            gradient_clip_val: Gradient clipping value (optional)
            profiler: Profiler to use ('pytorch' or None)
            profile_dir: Directory to save profiler results
            limit_train_batches: Limit number of training batches
            limit_val_batches: Limit number of validation batches
        """
        self.max_epochs = max_epochs
        self.log_every_n_steps = log_every_n_steps
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.gradient_clip_val = gradient_clip_val
        self.profiler = profiler
        self.profile_dir = profile_dir
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches

        # Setup device
        if accelerator == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif accelerator == "gpu":
            if not torch.cuda.is_available():
                raise ValueError("GPU not available but accelerator='gpu'")
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print_text(f"Using device: {self.device}", style="bold blue")

        # Setup callbacks
        self.callbacks = callbacks or []

        # Setup logger
        if logger is None:
            self.logger = None
        elif isinstance(logger, list):
            self.logger = MultiLogger(logger)
        else:
            self.logger = logger

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.should_stop = False

        # These will be set during fit/test
        self.model: Optional[BaseModule] = None
        self.optimizer: Optional[Optimizer] = None
        self.scheduler: Optional[_LRScheduler] = None
        self.train_dataloader: Optional[DataLoader] = None
        self.val_dataloader: Optional[DataLoader] = None
        self.test_dataloader: Optional[DataLoader] = None

        # Metrics storage
        self.callback_metrics: Dict[str, torch.Tensor] = {}

        # Checkpoint callback (for easy access to best model path)
        self.checkpoint_callback = None
        for callback in self.callbacks:
            if callback.__class__.__name__ == "ModelCheckpoint":
                self.checkpoint_callback = callback
                break

    def fit(
        self,
        model: BaseModule,
        datamodule: DataModule,
        ckpt_path: Optional[str] = None,
    ):
        """Fit the model.

        Args:
            model: Model to train
            datamodule: DataModule with train/val data
            ckpt_path: Path to checkpoint to resume from
        """
        # Setup
        self.model = model.to(self.device)

        self.train_dataloader = datamodule.train_dataloader()
        self.val_dataloader = datamodule.val_dataloader()

        # Determine effective limits for dataloaders
        self._effective_limit_train_batches = None
        if self.limit_train_batches is not None:
            if isinstance(self.limit_train_batches, int):
                self._effective_limit_train_batches = self.limit_train_batches
            else:  # float, interpreted as percentage
                self._effective_limit_train_batches = int(
                    len(self.train_dataloader) * self.limit_train_batches
                )

        self._effective_limit_val_batches = None
        if self.limit_val_batches is not None:
            if isinstance(self.limit_val_batches, int):
                self._effective_limit_val_batches = self.limit_val_batches
            else:  # float, interpreted as percentage
                self._effective_limit_val_batches = int(
                    len(self.val_dataloader) * self.limit_val_batches
                )

        # Setup optimizer and scheduler
        opt_config = self.model.configure_optimizers()
        if isinstance(opt_config, tuple):
            self.optimizer, self.scheduler = opt_config
        else:
            self.optimizer = opt_config
            self.scheduler = None

        # Load checkpoint if provided
        if ckpt_path is not None:
            self._load_checkpoint(ckpt_path)

        # Log hyperparameters
        if self.logger:
            hparams = {
                "max_epochs": self.max_epochs,
                "batch_size": datamodule.batch_size,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            }
            self.logger.log_hyperparameters(hparams)

        # Callbacks: on_train_start
        for callback in self.callbacks:
            callback.on_train_start(self, self.model)

        # Profiler context
        self.prof = None
        if self.profiler:
            print_text(f"{self.device.type}")
            print_text(f"Starting profiler ({self.profiler})...", style="bold yellow")
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            if self.profiler == "pytorch":
                # Use schedule to profile only a few steps (more efficient)
                print_text(
                    f"Starting profiler ({self.profiler})...", style="bold yellow"
                )
                self.prof = profile(
                    activities=activities,
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                    # Profile: wait 1 step, warmup 1 step, actively profile 3 steps, repeat 2 times
                    schedule=torch.profiler.schedule(
                        wait=1, warmup=1, active=3, repeat=2
                    ),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(
                        self.profile_dir
                    ),
                )
                self.prof.__enter__()  # Start the profiler context

        # Training loop
        try:
            for epoch in range(self.current_epoch, self.max_epochs):
                if self.should_stop:
                    break

                self.current_epoch = epoch

                # Callbacks: on_epoch_start
                for callback in self.callbacks:
                    callback.on_epoch_start(self, self.model)

                # Train epoch
                self._train_epoch()

                # Validation
                if (epoch + 1) % self.check_val_every_n_epoch == 0:
                    self._validation_epoch()

                # Scheduler step
                if self.scheduler is not None:
                    if isinstance(
                        self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        if (epoch + 1) % self.check_val_every_n_epoch == 0:
                            val_loss = self.callback_metrics.get("val/loss")
                            if val_loss is not None:
                                self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

                # Callbacks: on_epoch_end
                for callback in self.callbacks:
                    callback.on_epoch_end(self, self.model)

        finally:
            # Stop profiler
            if self.prof:
                self.prof.__exit__(None, None, None)
                print_text(
                    f"Profiler results saved to: {self.profile_dir}", style="bold green"
                )
            # Callbacks: on_train_end
            for callback in self.callbacks:
                callback.on_train_end(self, self.model)

            # Finalize logger
            if self.logger:
                self.logger.finalize()

    def test(
        self,
        model: Optional[BaseModule] = None,
        datamodule: Optional[DataModule] = None,
        ckpt_path: Optional[str] = None,
    ):
        """Test the model.

        Args:
            model: Model to test (if None, uses model from fit())
            datamodule: DataModule with test data
            ckpt_path: Path to checkpoint to load
        """
        if model is not None:
            self.model = model.to(self.device)

        if self.model is None:
            raise ValueError("No model provided and no model from fit()")

        # Load checkpoint if provided
        if ckpt_path is not None:
            self._load_checkpoint(ckpt_path)

        # Setup data
        if datamodule is not None:
            datamodule.prepare_data()
            datamodule.setup(stage="test")
            self.test_dataloader = datamodule.test_dataloader()

        if self.test_dataloader is None:
            raise ValueError("No test dataloader available")

        # Test loop
        self.model.eval()
        test_outputs = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_dataloader):
                # Move batch to device
                batch = self._batch_to_device(batch)

                # Test step
                outputs = self.model.test_step(batch, batch_idx)
                test_outputs.append(outputs)

        # Aggregate test metrics
        test_metrics = self._aggregate_outputs(test_outputs)
        self.callback_metrics.update(test_metrics)

        # Print test metrics with rich formatting
        print_text("\nTest Results:", style="bold yellow")
        for key, value in test_metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            print_text(f"  {key}: {value:.4f}", style="green")

        return test_metrics

    def _train_epoch(self):
        """Train for one epoch."""
        with record_function("train_epoch"):
            self.model.train()
            self.model.on_train_epoch_start()

            epoch_outputs = []

            num_batches = 0
            for batch_idx, batch in enumerate(self.train_dataloader):
                if (
                    self._effective_limit_train_batches is not None
                    and num_batches >= self._effective_limit_train_batches
                ):
                    break
                num_batches += 1

                # Callbacks: on_train_batch_start
                for callback in self.callbacks:
                    callback.on_train_batch_start(self, self.model, batch, batch_idx)

                # Move batch to device
                batch = self._batch_to_device(batch)

                # Training step
                self.optimizer.zero_grad()
                loss, outputs = self.model.training_step(batch, batch_idx)

                # Backward
                loss.backward()

                # Gradient clipping
                if self.gradient_clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_val
                    )

                # Optimizer step
                self.optimizer.step()

                # Store outputs
                epoch_outputs.append(outputs)

                # Log metrics
                if self.logger and (self.global_step % self.log_every_n_steps == 0):
                    self.logger.log_metrics(
                        {k + "_step": v for k, v in outputs.items()}, self.global_step
                    )

                # Callbacks: on_train_batch_end
                for callback in self.callbacks:
                    callback.on_train_batch_end(
                        self, self.model, outputs, batch, batch_idx
                    )

                self.global_step += 1

                if self.prof is not None:
                    self.prof.step()

            # Aggregate epoch metrics
            train_metrics = self._aggregate_outputs(epoch_outputs)
            self.callback_metrics.update(train_metrics)

            # Log epoch metrics with epoch number
            if self.logger:
                self.logger.log_metrics(train_metrics, self.current_epoch)

            self.model.on_train_epoch_end()

    def _validation_epoch(self):
        """Validate for one epoch."""
        with record_function("validation_epoch"):
            # Callbacks: on_validation_start
            for callback in self.callbacks:
                callback.on_validation_start(self, self.model)

            self.model.eval()
            self.model.on_validation_epoch_start()

            val_outputs = []

            num_batches = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(self.val_dataloader):
                    if (
                        self._effective_limit_val_batches is not None
                        and num_batches >= self._effective_limit_val_batches
                    ):
                        break
                    num_batches += 1

                    # Callbacks: on_validation_batch_start
                    for callback in self.callbacks:
                        callback.on_validation_batch_start(
                            self, self.model, batch, batch_idx
                        )

                    # Move batch to device
                    batch = self._batch_to_device(batch)

                    # Validation step
                    outputs = self.model.validation_step(batch, batch_idx)
                    val_outputs.append(outputs)

                    # Callbacks: on_validation_batch_end
                    for callback in self.callbacks:
                        callback.on_validation_batch_end(
                            self, self.model, outputs, batch, batch_idx
                        )

                    if self.prof is not None:
                        self.prof.step()

            # Aggregate validation metrics
            val_metrics = self._aggregate_outputs(val_outputs)
            self.callback_metrics.update(val_metrics)

            # Log metrics
            if self.logger:
                self.logger.log_metrics(
                    {k + "_epoch": v for k, v in val_metrics.items()},
                    self.current_epoch,
                )

            self.model.on_validation_epoch_end()

            # Callbacks: on_validation_end
            for callback in self.callbacks:
                callback.on_validation_end(self, self.model)

    def _batch_to_device(self, batch):
        """Move batch to device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, (list, tuple)):
            return [self._batch_to_device(item) for item in batch]
        elif isinstance(batch, dict):
            return {key: self._batch_to_device(value) for key, value in batch.items()}
        return batch

    def _aggregate_outputs(
        self, outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Aggregate outputs from multiple batches."""
        if not outputs:
            return {}

        # Get all keys
        keys = outputs[0].keys()

        # Average each metric
        aggregated = {}
        for key in keys:
            values = [output[key] for output in outputs if key in output]
            if values and isinstance(values[0], torch.Tensor):
                aggregated[key] = torch.stack(values).mean()

        return aggregated

    def _load_checkpoint(self, ckpt_path: str):
        """Load checkpoint."""

        print_text(f"Loading checkpoint from: {ckpt_path}", style="bold blue")

        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        # Load model state
        self.model.load_state_dict(checkpoint["state_dict"])

        # Load optimizer state
        if self.optimizer is not None and "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        # Load scheduler state
        if self.scheduler is not None and "scheduler_state" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])

        # Load training state
        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)

        print_text(
            f"Resumed from epoch {self.current_epoch}, step {self.global_step}",
            style="bold green",
        )
