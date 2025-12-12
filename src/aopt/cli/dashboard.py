"""Rich dashboard for real-time progress and statistics."""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table
from rich.layout import Layout
from rich.live import Live
from rich.text import Text


@dataclass
class ProcessingResult:
    """Result of an image processing operation."""
    
    input_path: Path
    output_path: Path
    input_size: int
    output_size: int
    success: bool
    message: str = ""
    
    @property
    def ratio(self) -> float:
        """Compression ratio (0-1, lower is better)."""
        if self.input_size == 0:
            return 1.0
        return self.output_size / self.input_size
    
    @property
    def savings_percent(self) -> float:
        """Savings percentage."""
        return (1 - self.ratio) * 100
    
    @property
    def savings_bytes(self) -> int:
        """Bytes saved."""
        return self.input_size - self.output_size


@dataclass
class ImageInfo:
    """Image metadata information."""
    
    path: Path
    format: str
    mode: str
    width: int
    height: int
    size_bytes: int
    has_exif: bool
    exif_data: dict[str, Any] | None = None


@dataclass
class ShieldResult:
    """Result of Privacy Shield processing."""
    
    input_path: Path
    output_path: Path
    detections: list[dict[str, Any]]
    redacted_count: int
    success: bool


class Dashboard:
    """Rich dashboard for displaying progress and results."""
    
    def __init__(self, console: Console) -> None:
        self.console = console
    
    @contextmanager
    def progress_context(self, description: str = "Processing") -> Generator[Progress, None, None]:
        """Create a progress context with Rich styling."""
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )
        with progress:
            yield progress
    
    def create_batch_progress(self) -> Progress:
        """Create a progress bar for batch operations."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TextColumn("•"),
            TransferSpeedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console,
        )
    
    def show_result(self, result: ProcessingResult) -> None:
        """Display processing result with Rich formatting."""
        if result.success:
            status = "[bold green]✓ Success[/]"
            
            # Build stats table
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column("Label", style="dim")
            table.add_column("Value")
            
            table.add_row("Input", f"{result.input_path.name}")
            table.add_row("Output", f"{result.output_path.name}")
            table.add_row("Original size", self._format_size(result.input_size))
            table.add_row("New size", self._format_size(result.output_size))
            table.add_row(
                "Savings",
                f"[green]{result.savings_percent:.1f}%[/] ({self._format_size(result.savings_bytes)})"
            )
            
            panel = Panel(
                table,
                title=status,
                border_style="green",
            )
        else:
            panel = Panel(
                f"[red]{result.message}[/]",
                title="[bold red]✗ Error[/]",
                border_style="red",
            )
        
        self.console.print(panel)
    
    def show_batch_summary(
        self,
        results: list[ProcessingResult],
        elapsed: float,
    ) -> None:
        """Display batch processing summary."""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        total_input = sum(r.input_size for r in successful)
        total_output = sum(r.output_size for r in successful)
        total_savings = total_input - total_output
        
        # Summary table
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Label", style="dim")
        table.add_column("Value")
        
        table.add_row("Files processed", f"[green]{len(successful)}[/]")
        if failed:
            table.add_row("Failed", f"[red]{len(failed)}[/]")
        table.add_row("Total input size", self._format_size(total_input))
        table.add_row("Total output size", self._format_size(total_output))
        table.add_row(
            "Total savings",
            f"[bold green]{self._format_size(total_savings)}[/]"
        )
        if total_input > 0:
            savings_pct = (total_savings / total_input) * 100
            table.add_row("Compression ratio", f"[green]{savings_pct:.1f}%[/]")
        table.add_row("Time elapsed", f"{elapsed:.2f}s")
        table.add_row(
            "Speed",
            f"{len(successful) / elapsed:.1f} images/sec" if elapsed > 0 else "N/A"
        )
        
        self.console.print()
        self.console.print(Panel(
            table,
            title="[bold cyan]📊 Batch Summary[/]",
            border_style="cyan",
        ))
    
    def show_image_info(self, info: ImageInfo) -> None:
        """Display image information."""
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Property", style="cyan")
        table.add_column("Value")
        
        table.add_row("File", str(info.path.name))
        table.add_row("Format", info.format)
        table.add_row("Mode", info.mode)
        table.add_row("Dimensions", f"{info.width} × {info.height} px")
        table.add_row("File size", self._format_size(info.size_bytes))
        table.add_row("Has EXIF", "[green]Yes[/]" if info.has_exif else "[dim]No[/]")
        
        if info.exif_data:
            self.console.print(Panel(table, title="[bold]📷 Image Info[/]", border_style="blue"))
            
            # EXIF table
            exif_table = Table(show_header=True, box=None, padding=(0, 2))
            exif_table.add_column("Tag", style="cyan")
            exif_table.add_column("Value")
            
            for tag, value in list(info.exif_data.items())[:15]:  # Limit to 15 entries
                exif_table.add_row(str(tag), str(value)[:50])
            
            self.console.print(Panel(exif_table, title="[bold]🏷️ EXIF Data[/]", border_style="dim"))
        else:
            self.console.print(Panel(table, title="[bold]📷 Image Info[/]", border_style="blue"))
    
    def show_presets(self, presets: list[dict[str, Any]]) -> None:
        """Display available presets."""
        table = Table(title="[bold]🎨 Available Presets[/]")
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Description")
        table.add_column("Format", style="green")
        table.add_column("Quality", justify="right")
        table.add_column("Max Size", justify="right")
        
        for preset in presets:
            table.add_row(
                preset["name"],
                preset.get("description", ""),
                preset.get("format", "auto"),
                str(preset.get("quality", "-")),
                preset.get("max_size", "-"),
            )
        
        self.console.print(table)
    
    def show_preset_details(self, preset: dict[str, Any]) -> None:
        """Show detailed preset information."""
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Property", style="cyan")
        table.add_column("Value")
        
        for key, value in preset.items():
            table.add_row(key, str(value))
        
        self.console.print(Panel(
            table,
            title=f"[bold]Preset: {preset.get('name', 'Unknown')}[/]",
            border_style="cyan",
        ))
    
    def show_shield_result(self, result: ShieldResult) -> None:
        """Display Privacy Shield results."""
        if result.success:
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column("Label", style="dim")
            table.add_column("Value")
            
            table.add_row("Input", result.input_path.name)
            table.add_row("Output", result.output_path.name)
            table.add_row("Detections", f"[yellow]{len(result.detections)}[/]")
            table.add_row("Redacted", f"[green]{result.redacted_count}[/]")
            
            panel = Panel(
                table,
                title="[bold green]🛡️ Privacy Shield Complete[/]",
                border_style="green",
            )
        else:
            panel = Panel(
                "[yellow]No sensitive information detected.[/]",
                title="[bold]🛡️ Privacy Shield[/]",
                border_style="yellow",
            )
        
        self.console.print(panel)
    
    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
