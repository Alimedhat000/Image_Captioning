"""Utilities for instantiating objects from Hydra configs."""

from typing import List
import hydra
from omegaconf import DictConfig

from src.core.callbacks import Callback
from src.core.loggers import Logger
from src.utils.rich_utils import print_text


def instantiate_callbacks(callbacks_cfg: DictConfig, cfg: DictConfig) -> List[Callback]:
    """Instantiate callbacks from config.

    Args:
        callbacks_cfg: Callbacks configuration
        cfg: Full configuration (for resolving references)

    Returns:
        List of instantiated callbacks
    """
    callbacks = []

    if not callbacks_cfg:
        return callbacks

    print_text("Instantiating callbacks...", style="bold blue")

    for cb_name, cb_conf in callbacks_cfg.items():
        if "_target_" in cb_conf:
            print_text(f"  - {cb_name}: {cb_conf._target_}", style="cyan")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig, cfg: DictConfig) -> List[Logger]:
    """Instantiate loggers from config.

    Args:
        logger_cfg: Logger configuration
        cfg: Full configuration (for resolving references)

    Returns:
        List of instantiated loggers
    """
    loggers = []

    if not logger_cfg:
        return loggers

    print_text("Instantiating loggers...", style="bold blue")

    if "_target_" in logger_cfg:
        # Single logger
        print_text(f"  - {logger_cfg._target_}", style="cyan")
        loggers.append(hydra.utils.instantiate(logger_cfg))
    else:
        # Multiple loggers
        for lg_name, lg_conf in logger_cfg.items():
            if "_target_" in lg_conf:
                print_text(f"  - {lg_name}: {lg_conf._target_}", style="cyan")
                loggers.append(hydra.utils.instantiate(lg_conf))

    return loggers
