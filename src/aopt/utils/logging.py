"""Rich logging utilities for Asset Optimizer."""

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn


# Global console instance
_console: Console | None = None


def get_console() -> Console:
    """Get or create the global Rich console."""
    global _console
    if _console is None:
        _console = Console()
    return _console


def create_progress(description: str = "Processing") -> Progress:
    """Create a styled progress bar.
    
    Args:
        description: Default task description.
        
    Returns:
        Configured Progress instance.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        console=get_console(),
    )


def log_success(message: str) -> None:
    """Log a success message."""
    get_console().print(f"[bold green]✓[/] {message}")


def log_error(message: str) -> None:
    """Log an error message."""
    get_console().print(f"[bold red]✗[/] {message}")


def log_warning(message: str) -> None:
    """Log a warning message."""
    get_console().print(f"[bold yellow]⚠[/] {message}")


def log_info(message: str) -> None:
    """Log an info message."""
    get_console().print(f"[bold blue]ℹ[/] {message}")
