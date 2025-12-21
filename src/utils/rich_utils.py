"""Rich utilities for beautiful console outputs."""

from typing import Optional

import torch
import torchinfo
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.syntax import Syntax
from rich.text import Text


console = Console()


def print_config(
    config: DictConfig,
    order: tuple = ("datamodule", "model", "callbacks", "logger", "trainer"),
    exclude_keys: tuple = ("extras", "paths"),
) -> None:
    """Prints the configuration in a beautiful format with rich colors and formatting.

    Args:
        config: Configuration dictionary to print
        order: Order to print config sections
        exclude_keys: Keys to exclude from printing
    """
    console.rule("[bold blue]CONFIGURATION", style="bold blue")

    # Print config sections in specified order
    for section in order:
        if section in config:
            config_section = {section: config[section]}
            yaml_str = OmegaConf.to_yaml(config_section)

            syntax = Syntax(
                yaml_str,
                "yaml",
                theme="ansi_dark",
                line_numbers=False,
                word_wrap=True,
            )
            console.print(syntax)

    # Print remaining sections not in order
    for key in config:
        if key not in order and key not in exclude_keys:
            config_section = {key: config[key]}
            yaml_str = OmegaConf.to_yaml(config_section)

            syntax = Syntax(
                yaml_str,
                "yaml",
                theme="ansi_dark",
                line_numbers=False,
                word_wrap=True,
            )
            console.print(syntax)


def print_model_summary(
    model: torch.nn.Module,
    input_size: tuple,
    col_names: Optional[list] = None,
    verbose: int = 0,
) -> None:
    """Prints the model summary in a beautiful format with rich colors.

    Args:
        model: PyTorch model to summarize
        input_size: Input size for the model (batch_size, channels, height, width)
        col_names: Column names to display in the summary
        verbose: Verbosity level (0 for concise, 1 for detailed)
    """
    if col_names is None:
        col_names = ["input_size", "output_size", "num_params", "mult_adds"]

    try:
        # Get the string representation of the summary
        summary_str = torchinfo.summary(
            model,
            input_size=input_size,
            col_names=col_names,
            verbose=verbose,
            depth=3,  # How many levels to print (for nested modules)
        )

        console.rule("[bold green]MODEL SUMMARY", style="bold green")

        # Print the summary with rich formatting
        syntax = Syntax(
            str(summary_str),
            "text",
            theme="ansi_dark",
            line_numbers=False,
            word_wrap=True,
        )
        console.print(syntax)

        # Also print summary statistics separately
        total_params = (
            summary_str.total_params if hasattr(summary_str, "total_params") else 0
        )
        trainable_params = (
            summary_str.trainable_params
            if hasattr(summary_str, "trainable_params")
            else 0
        )
        mult_adds = (
            summary_str.total_mult_adds
            if hasattr(summary_str, "total_mult_adds")
            else 0
        )

        stats_text = Text("\nModel Statistics:\n", style="bold yellow")
        stats_text.append(f"Total params: {total_params:,}\n")
        stats_text.append(f"Trainable params: {trainable_params:,}\n")
        stats_text.append(
            f"Non-trainable params: {total_params - trainable_params:,}\n"
        )
        stats_text.append(f"Mult-adds: {mult_adds:,}\n")

        console.print(stats_text)

    except Exception as e:
        console.print(f"[bold red]Could not generate model summary: {e}[/bold red]")


def print_colored_separator(
    title: str, style: str = "bold blue", char: str = "="
) -> None:
    """Prints a colored separator with a title.

    Args:
        title: Title to display in the separator
        style: Rich style for the text
        char: Character to use for the separator
    """
    console.rule(
        f"[{style}]{title}[/{style}]", style=style.split()[1] if " " in style else style
    )


def print_text(text: str, style: Optional[str] = None) -> None:
    """Prints text with optional styling.

    Args:
        text: Text to print
        style: Rich style for the text
    """
    if style:
        console.print(Text(text, style=style))
    else:
        console.print(text)
