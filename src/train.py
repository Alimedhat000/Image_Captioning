"""Training script for image captioning with custom Trainer."""

import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List, Optional
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from datetime import datetime

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


def get_experiment_name(cfg: DictConfig) -> str:
    """Get experiment name from config or prompt user.

    Args:
        cfg: Hydra configuration

    Returns:
        Experiment name
    """
    # If experiment_name is set in config, use it
    if cfg.get("experiment_name"):
        return cfg.experiment_name

    # In multirun mode, use timestamp
    if cfg.get("hydra", {}).get("mode") == "MULTIRUN":
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    # Interactive mode - prompt user
    try:
        print_colored_separator("Experiment Setup", style="bold cyan")
        print_text(
            "Enter experiment name (or press Enter to use timestamp):",
            style="bold yellow",
        )
        name = input("> ").strip()

        if name:
            return name
        else:
            return datetime.now().strftime("%Y%m%d_%H%M%S")
    except (EOFError, KeyboardInterrupt):
        # Non-interactive mode or user interrupted
        return datetime.now().strftime("%Y%m%d_%H%M%S")


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main training function.

    Args:
        cfg: Hydra configuration

    Returns:
        Optimized metric value (for hyperparameter optimization)
    """
    # Get experiment name (prompt if not set)
    experiment_name = get_experiment_name(cfg)

    # Update config with experiment name
    OmegaConf.set_struct(cfg, False)  # Allow adding new keys
    cfg.experiment_name = experiment_name
    OmegaConf.set_struct(cfg, True)

    print_text(f"Experiment name: {experiment_name}", style="bold green")

    # Set seed
    if cfg.get("seed"):
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.seed)

    # ========== Initialize DataModule FIRST ==========
    print_text(f"Instantiating datamodule <{cfg.data._target_}>", style="bold blue")
    datamodule: DataModule = hydra.utils.instantiate(cfg.data)

    # ========== Setup DataModule to Build Vocabulary ==========
    print_text("Building vocabulary from training data...", style="bold blue")
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    # ========== Get Vocabulary Info ==========
    # Check if datamodule has vocab methods (for image captioning)
    if hasattr(datamodule, "get_vocab") and hasattr(datamodule, "get_vocab_size"):
        vocab = datamodule.get_vocab()
        vocab_size = datamodule.get_vocab_size()
        pad_idx = vocab.pad_idx

        print_text(f"Vocabulary size: {vocab_size}", style="bold green")
        print_text(f"Padding index: {pad_idx}", style="bold green")

        # ========== Update Config with Vocab Size ==========
        OmegaConf.set_struct(cfg, False)
        cfg.model.vocab_size = vocab_size
        cfg.model.decoder.vocab_size = vocab_size
        cfg.model.pad_idx = pad_idx
        OmegaConf.set_struct(cfg, True)
    else:
        vocab = None
        print_text(
            "DataModule does not require vocabulary (non-captioning task)",
            style="yellow",
        )

    print_text(
        f"Train dataset size: {len(datamodule.train_dataset)}", style="bold green"
    )
    if hasattr(datamodule, "val_dataset") and datamodule.val_dataset is not None:
        print_text(
            f"Val dataset size: {len(datamodule.val_dataset)}", style="bold green"
        )

    # ========== Initialize Model ==========
    print_text(f"Instantiating model <{cfg.model._target_}>", style="bold blue")
    model: BaseModule = hydra.utils.instantiate(cfg.model)

    # Print model info
    if hasattr(model, "model_name"):
        print_text(f"\nModel: {model.model_name}", style="bold cyan")
    if hasattr(model, "encoder"):
        print_text(f"Encoder: {model.encoder.__class__.__name__}", style="cyan")
    if hasattr(model, "decoder"):
        print_text(f"Decoder: {model.decoder.__class__.__name__}", style="cyan")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_text(f"Total parameters: {total_params:,}", style="bold green")
    print_text(f"Trainable parameters: {trainable_params:,}", style="bold green")

    # ========== Initialize Callbacks and Loggers ==========
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"), cfg)
    loggers: List[Logger] = instantiate_loggers(cfg.get("logger"), cfg)

    # Explicitly create MultiLogger if loggers are present
    trainer_logger = MultiLogger(loggers) if loggers else None

    # ========== Initialize Trainer ==========
    print_text(f"Instantiating trainer <{cfg.trainer._target_}>", style="bold blue")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=trainer_logger,
    )

    # Print config and model summary with rich formatting
    print_config(cfg)

    # Try to print model summary
    try:
        # Get input size from datamodule
        input_size = datamodule.train_dataset[0][0].shape

        # Add batch dimension
        input_size = (datamodule.batch_size, *input_size)

        print_model_summary(
            model.net,
            input_size=input_size,
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            verbose=0,  # 0 for no individual layer info, 1 for full summary
        )
    except Exception as e:
        print_text(f"Could not generate model summary: {e}", style="bold red")

    # ========== Training ==========
    if cfg.get("train"):
        print_colored_separator("Starting Training!", style="bold green")
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.get("ckpt_path"),
        )

    # ========== Testing ==========
    if cfg.get("test"):
        print_colored_separator("Starting Testing!", style="bold green")
        ckpt_path = None

        # Use best checkpoint if available
        if trainer.checkpoint_callback is not None:
            ckpt_path = trainer.checkpoint_callback.best_model_path
            if ckpt_path == "":
                print_text(
                    "Best checkpoint not found! Using current weights...",
                    style="bold red",
                )
                ckpt_path = None

        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
        )

    # ========== Generate Sample Captions (if applicable) ==========
    if cfg.get("generate_samples", False) and vocab is not None:
        print_colored_separator("Generating Sample Captions", style="bold cyan")

        model.eval()

        # Get a batch from validation set
        val_loader = datamodule.val_dataloader()
        val_batch = next(iter(val_loader))
        images, captions, lengths = val_batch

        # Move to device
        images = images.to(trainer.device)

        # Generate captions (greedy)
        print_text("\nGreedy Search:", style="bold yellow")
        try:
            generated_captions = model.generate_caption(
                images[:5],  # First 5 images
                vocab,
                max_length=20,
                method="greedy",
            )

            for i, cap_indices in enumerate(generated_captions):
                caption_text = vocab.decode(cap_indices, skip_special_tokens=True)
                print_text(f"  Image {i + 1}: {caption_text}", style="green")
        except Exception as e:
            print_text(f"Error generating captions: {e}", style="bold red")

        # Generate with beam search
        print_text("\nBeam Search:", style="bold yellow")
        try:
            generated_captions_beam = model.generate_caption(
                images[:5], vocab, max_length=20, method="beam"
            )

            for i, cap_indices in enumerate(generated_captions_beam):
                caption_text = vocab.decode(cap_indices, skip_special_tokens=True)
                print_text(f"  Image {i + 1}: {caption_text}", style="green")
        except Exception as e:
            print_text(
                f"Error generating captions with beam search: {e}", style="bold red"
            )

        # Show reference captions for comparison
        print_text("\nReference Captions:", style="bold yellow")
        for i in range(min(5, len(captions))):
            ref_caption = vocab.decode(captions[i].tolist(), skip_special_tokens=True)
            print_text(f"  Image {i + 1}: {ref_caption}", style="blue")

    print_colored_separator("Training Complete!", style="bold green")

    # Return metric for hyperparameter optimization
    metric_name = cfg.get("optimized_metric")
    if metric_name and metric_name in trainer.callback_metrics:
        return trainer.callback_metrics[metric_name].item()

    return None


if __name__ == "__main__":
    main()
