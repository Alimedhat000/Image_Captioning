import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

"""Evaluation script with Hydra configuration."""

from typing import List, Optional
import hydra
from omegaconf import DictConfig
import torch

# Fix for PyTorch 2.6 checkpoint loading
_original_load = torch.load


def _patched_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)


torch.load = _patched_load

from src.core.trainer import Trainer
from src.core.callbacks import Callback
from src.core.loggers import Logger, MultiLogger
from src.models.base_module import BaseModule
from src.data.datamodule import DataModule
from src.utils.instantiate import instantiate_callbacks, instantiate_loggers
from src.utils.rich_utils import (
    print_config,
    print_model_summary,
    print_colored_separator,
    print_text,
)


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main evaluation function.

    Args:
        cfg: Hydra configuration

    Returns:
        Test metric value
    """
    # Set seed
    if cfg.get("seed"):
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.seed)

    print_text(f"Instantiating datamodule <{cfg.data._target_}>", style="bold blue")
    datamodule: DataModule = hydra.utils.instantiate(cfg.data)

    print_text(f"Instantiating model <{cfg.model._target_}>", style="bold blue")
    model: BaseModule = hydra.utils.instantiate(cfg.model)

    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"), cfg)

    loggers: List[Logger] = instantiate_loggers(cfg.get("logger"), cfg)

    # Explicitly create MultiLogger if loggers are present
    trainer_logger = MultiLogger(loggers) if loggers else None

    print_text(f"Instantiating trainer <{cfg.trainer._target_}>", style="bold blue")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=trainer_logger,
    )

    # Print config
    print_config(cfg)

    # Check checkpoint path
    if not cfg.get("ckpt_path"):
        print_text(
            "ERROR: No checkpoint path provided! Use ckpt_path=/path/to/checkpoint",
            style="bold red",
        )
        return None

    # Print model summary
    try:
        datamodule.prepare_data()
        datamodule.setup(stage="test")
        input_size = datamodule.test_dataset[0][0].shape
        input_size = (datamodule.batch_size, *input_size)

        print_model_summary(
            model.net,
            input_size=input_size,
            col_names=["input_size", "output_size", "num_params"],
            verbose=0,
        )
    except Exception as e:
        print_text(f"Could not generate model summary: {e}", style="bold yellow")

    # Evaluation
    print_colored_separator("Starting Evaluation!", style="bold green")

    test_metrics = trainer.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=cfg.ckpt_path,
    )

    # Return metric for logging
    metric_name = cfg.get("optimized_metric")
    if metric_name and metric_name in trainer.callback_metrics:
        return trainer.callback_metrics[metric_name].item()

    return None


if __name__ == "__main__":
    main()
