#!/usr/bin/env python3
"""List all available checkpoints in the logs directory."""

import os
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table

console = Console()


def get_checkpoint_info(ckpt_path: Path):
    """Get information about a checkpoint."""
    stat = ckpt_path.stat()
    size_mb = stat.st_size / (1024 * 1024)
    modified = datetime.fromtimestamp(stat.st_mtime)

    return {
        "path": str(ckpt_path),
        "size_mb": size_mb,
        "modified": modified,
        "is_best": "epoch" in ckpt_path.stem and "last" not in ckpt_path.stem,
    }


def list_checkpoints(logs_dir: str = "logs"):
    """List all checkpoints in logs directory."""
    logs_path = Path(logs_dir)

    if not logs_path.exists():
        console.print("[red]Logs directory not found![/red]")
        return

    # Find all checkpoint files
    checkpoints = []
    for ckpt_path in logs_path.rglob("*.ckpt"):
        info = get_checkpoint_info(ckpt_path)
        checkpoints.append(info)

    if not checkpoints:
        console.print("[yellow]No checkpoints found![/yellow]")
        return

    # Sort by modified time (newest first)
    checkpoints.sort(key=lambda x: x["modified"], reverse=True)

    # Create table
    table = Table(
        title="Available Checkpoints", show_header=True, header_style="bold magenta"
    )
    table.add_column("Type", style="cyan", width=10)
    table.add_column("Path", style="green")
    table.add_column("Size (MB)", justify="right", style="yellow")
    table.add_column("Modified", style="blue")

    for info in checkpoints:
        ckpt_type = "✓ Best" if info["is_best"] else "  Last"
        table.add_row(
            ckpt_type,
            info["path"],
            f"{info['size_mb']:.2f}",
            info["modified"].strftime("%Y-%m-%d %H:%M:%S"),
        )

    console.print(table)
    console.print(f"\n[bold]Total checkpoints: {len(checkpoints)}[/bold]")

    # Print example eval command
    if checkpoints:
        best_ckpt = next((c for c in checkpoints if c["is_best"]), checkpoints[0])
        console.print("\n[bold green]Example eval command:[/bold green]")
        console.print(f"make eval ckpt={best_ckpt['path']}")


if __name__ == "__main__":
    list_checkpoints()
