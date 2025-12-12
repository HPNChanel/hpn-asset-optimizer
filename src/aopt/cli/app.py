"""Main Typer CLI application for Asset Optimizer."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from aopt import __app_name__, __version__

# Initialize console and app
console = Console()
app = typer.Typer(
    name=__app_name__,
    help="🚀 Asset Optimizer - Ultra-high-performance CLI for image optimization",
    no_args_is_help=True,
)

# Sub-apps
preset_app = typer.Typer(help="Manage optimization presets")
app.add_typer(preset_app, name="preset")


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"[bold cyan]{__app_name__}[/] version [green]{__version__}[/]")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version", "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """Asset Optimizer - Speed, Privacy, Beautiful CLI."""
    pass


@app.command()
def compress(
    path: Path = typer.Argument(..., help="Image file or directory to compress"),
    quality: int = typer.Option(85, "--quality", "-q", help="Quality (1-100)"),
    mode: str = typer.Option("lossy", "--mode", "-m", help="Mode: lossy or lossless"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path"),
) -> None:
    """Compress images with optimized algorithms.
    
    Examples:
        aopt compress image.jpg --quality 80
        aopt compress ./photos/ -q 70 -o ./compressed/
    """
    from aopt.cli.dashboard import Dashboard
    from aopt.core.compressor import Compressor, CompressionMode
    from aopt.core.batch import BatchProcessor
    
    compressor = Compressor()
    dashboard = Dashboard(console)
    compression_mode = CompressionMode.LOSSLESS if mode == "lossless" else CompressionMode.LOSSY
    
    if path.is_dir():
        processor = BatchProcessor(console)
        processor.process_directory(
            path,
            output or path.parent / "optimized",
            callback=lambda p: compressor.compress(p, quality=quality, mode=compression_mode),
        )
    else:
        with dashboard.progress_context("Compressing") as progress:
            task = progress.add_task(f"[cyan]{path.name}", total=100)
            result = compressor.compress(path, quality=quality, mode=compression_mode, output=output)
            progress.update(task, completed=100)
            dashboard.show_result(result)


@app.command()
def convert(
    path: Path = typer.Argument(..., help="Image file to convert"),
    format: str = typer.Option("webp", "--format", "-f", help="Target format: jpeg, png, webp, avif, gif, bmp"),
    quality: int = typer.Option(85, "--quality", "-q", help="Quality (1-100)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path"),
) -> None:
    """Convert images between formats.
    
    Examples:
        aopt convert image.png --format webp
        aopt convert photo.jpg -f avif -q 80
    """
    from aopt.cli.dashboard import Dashboard
    from aopt.core.converter import Converter, ImageFormat
    
    converter = Converter()
    dashboard = Dashboard(console)
    
    try:
        img_format = ImageFormat(format.lower())
    except ValueError:
        console.print(f"[red]Error:[/] Unknown format '{format}'")
        console.print("Supported formats: jpeg, png, webp, avif, gif, bmp")
        raise typer.Exit(1)
    
    with dashboard.progress_context("Converting") as progress:
        task = progress.add_task(f"[cyan]{path.name} → {format}", total=100)
        result = converter.convert(path, format=img_format, quality=quality, output=output)
        progress.update(task, completed=100)
        dashboard.show_result(result)


@app.command()
def batch(
    directory: Path = typer.Argument(..., help="Directory to process"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    format: Optional[str] = typer.Option(None, "--format", "-f", help="Convert to format"),
    quality: int = typer.Option(85, "--quality", "-q", help="Quality (1-100)"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Process subdirs"),
    workers: Optional[int] = typer.Option(None, "--workers", "-w", help="Number of worker threads"),
) -> None:
    """Process entire directories with live dashboard.
    
    Examples:
        aopt batch ./images/ -o ./optimized/
        aopt batch ./photos/ -f webp -q 80 -r
    """
    from aopt.core.batch import BatchProcessor
    from aopt.core.converter import ImageFormat
    
    processor = BatchProcessor(console, workers=workers)
    out_dir = output or directory.parent / f"{directory.name}_optimized"
    
    img_format = None
    if format:
        try:
            img_format = ImageFormat(format.lower())
        except ValueError:
            console.print(f"[red]Error:[/] Unknown format '{format}'")
            raise typer.Exit(1)
    
    processor.process_directory(
        directory,
        out_dir,
        quality=quality,
        format=img_format,
        recursive=recursive,
    )


@app.command()
def info(
    path: Path = typer.Argument(..., help="Image file path"),
) -> None:
    """Display detailed image information.
    
    Examples:
        aopt info photo.jpg
    """
    from aopt.cli.dashboard import Dashboard
    from aopt.utils.image_io import get_image_info
    
    info_data = get_image_info(path)
    dashboard = Dashboard(console)
    dashboard.show_image_info(info_data)


@app.command()
def strip(
    path: Path = typer.Argument(..., help="Image file path"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path"),
) -> None:
    """Remove all metadata from images (EXIF, IPTC, XMP).
    
    Examples:
        aopt strip photo.jpg
        aopt strip photo.jpg -o photo_clean.jpg
    """
    from aopt.cli.dashboard import Dashboard
    from aopt.utils.metadata import strip_metadata
    
    result = strip_metadata(path, output)
    dashboard = Dashboard(console)
    dashboard.show_result(result)


@app.command()
def shield(
    path: Path = typer.Argument(..., help="Image file path"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path"),
    mode: str = typer.Option("blur", "--mode", "-m", help="Redaction mode: blur, box, pixelate"),
) -> None:
    """Detect and redact PII (Privacy Shield).
    
    Uses OCR to find sensitive text (emails, phones, SSN, credit cards)
    and redacts them automatically.
    
    Examples:
        aopt shield document.png
        aopt shield screenshot.jpg -m pixelate
    """
    try:
        from aopt.ai.privacy import PrivacyShield
        from aopt.cli.dashboard import Dashboard
        
        shield_processor = PrivacyShield()
        result = shield_processor.process(path, output=output, mode=mode)
        dashboard = Dashboard(console)
        dashboard.show_shield_result(result)
    except ImportError:
        console.print(
            "[red]Error:[/] Privacy Shield requires AI dependencies. "
            "Install with: [cyan]pip install easyocr[/]"
        )
        raise typer.Exit(1)


# Preset subcommands
@preset_app.command("list")
def preset_list() -> None:
    """List all available optimization presets."""
    from aopt.cli.dashboard import Dashboard
    from aopt.config.presets import PresetManager
    
    manager = PresetManager()
    dashboard = Dashboard(console)
    dashboard.show_presets(manager.list_presets())


@preset_app.command("apply")
def preset_apply(
    name: str = typer.Argument(..., help="Preset name"),
    path: Path = typer.Argument(..., help="Image file or directory"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path"),
) -> None:
    """Apply an optimization preset.
    
    Examples:
        aopt preset apply web image.jpg
        aopt preset apply thumbnail ./photos/ -o ./thumbs/
    """
    from aopt.cli.dashboard import Dashboard
    from aopt.config.presets import PresetManager
    from aopt.core.batch import BatchProcessor
    
    manager = PresetManager()
    preset = manager.get_preset(name)
    
    if preset is None:
        console.print(f"[red]Error:[/] Preset '{name}' not found.")
        console.print("Use [cyan]aopt preset list[/] to see available presets.")
        raise typer.Exit(1)
    
    dashboard = Dashboard(console)
    
    if path.is_dir():
        processor = BatchProcessor(console)
        processor.process_with_preset(path, output or path.parent / "optimized", preset)
    else:
        with dashboard.progress_context(f"Applying '{name}' preset") as progress:
            task = progress.add_task(f"[cyan]{path.name}", total=100)
            result = manager.apply_preset(preset, path, output)
            progress.update(task, completed=100)
            dashboard.show_result(result)


@preset_app.command("show")
def preset_show(
    name: str = typer.Argument(..., help="Preset name"),
) -> None:
    """Show details of a specific preset."""
    from aopt.cli.dashboard import Dashboard
    from aopt.config.presets import PresetManager
    
    manager = PresetManager()
    preset = manager.get_preset(name)
    
    if preset is None:
        console.print(f"[red]Error:[/] Preset '{name}' not found.")
        raise typer.Exit(1)
    
    dashboard = Dashboard(console)
    dashboard.show_preset_details(preset)


@app.command()
def variants(
    path: Path = typer.Argument(..., help="Image file path"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path"),
    dark: bool = typer.Option(False, "--dark", "-d", help="Generate dark mode variant"),
    aspect: Optional[str] = typer.Option(None, "--aspect", "-a", help="Target aspect ratio: portrait, landscape, square"),
    method: str = typer.Option("mirror", "--method", "-m", help="Fill method: mirror, blur, solid"),
) -> None:
    """Generate smart image variants.
    
    Create dark mode versions or extend to different aspect ratios.
    
    Examples:
        aopt variants icon.png --dark
        aopt variants photo.jpg --aspect portrait --method blur
    """
    try:
        from aopt.ai.variants import VariantGenerator
        from aopt.cli.dashboard import Dashboard
        
        generator = VariantGenerator()
        dashboard = Dashboard(console)
        
        if dark:
            with dashboard.progress_context("Generating dark mode") as progress:
                task = progress.add_task(f"[cyan]{path.name}", total=100)
                result = generator.generate_dark_mode(path, output)
                progress.update(task, completed=100)
                dashboard.show_result(result)
        elif aspect:
            with dashboard.progress_context(f"Extending to {aspect}") as progress:
                task = progress.add_task(f"[cyan]{path.name}", total=100)
                result = generator.extend_to_aspect(path, aspect, output, method=method)
                progress.update(task, completed=100)
                dashboard.show_result(result)
        else:
            console.print("[yellow]Specify --dark or --aspect to generate variants[/]")
    except ImportError as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)


@app.command()
def face_blur(
    path: Path = typer.Argument(..., help="Image file path"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path"),
    radius: int = typer.Option(20, "--radius", "-r", help="Blur radius"),
) -> None:
    """Detect and blur faces in an image.
    
    Uses OpenCV Haar Cascade for face detection.
    
    Examples:
        aopt face-blur photo.jpg
        aopt face-blur group.png -r 30
    """
    try:
        from aopt.ai.privacy import FaceBlur
        from aopt.cli.dashboard import Dashboard
        
        blurrer = FaceBlur()
        dashboard = Dashboard(console)
        
        with dashboard.progress_context("Detecting faces") as progress:
            task = progress.add_task(f"[cyan]{path.name}", total=100)
            result = blurrer.process(path, output, blur_radius=radius)
            progress.update(task, completed=100)
            dashboard.show_shield_result(result)
    except ImportError as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)


@app.command()
def cache(
    action: str = typer.Argument("stats", help="Action: stats, clear"),
) -> None:
    """Manage the processing cache.
    
    The cache tracks processed files to skip reprocessing.
    
    Examples:
        aopt cache stats
        aopt cache clear
    """
    from aopt.core.batch import BatchProcessor
    
    processor = BatchProcessor(console)
    
    if action == "stats":
        stats = processor.cache_stats()
        console.print(f"[cyan]Cache entries:[/] {stats.get('entries', 0)}")
        console.print(f"[dim]Cache file:[/] {stats.get('cache_file', 'N/A')}")
    elif action == "clear":
        processor.clear_cache()
    else:
        console.print(f"[red]Unknown action:[/] {action}")
        console.print("Available actions: stats, clear")


if __name__ == "__main__":
    app()

