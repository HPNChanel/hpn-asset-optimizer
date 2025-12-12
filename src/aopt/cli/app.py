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


@app.command()
def remove_bg(
    path: Path = typer.Argument(..., help="Image file path"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path (PNG)"),
    model: str = typer.Option("rmbg-1.4", "--model", "-m", help="Model: rmbg-1.4 or u2net"),
    threshold: float = typer.Option(0.5, "--threshold", "-t", help="Alpha threshold (0-1)"),
) -> None:
    """Remove background from images using AI.
    
    Uses RMBG-1.4 or U2Net models for high-quality background removal.
    Models are downloaded automatically on first use (~176MB for RMBG-1.4).
    
    Examples:
        aopt remove-bg photo.jpg
        aopt remove-bg portrait.png -o portrait_nobg.png
        aopt remove-bg product.jpg --model u2net
    """
    try:
        from aopt.ai.rembg import BackgroundRemover
        from aopt.cli.dashboard import Dashboard
        
        remover = BackgroundRemover(model_name=model, console=console)
        dashboard = Dashboard(console)
        
        with dashboard.progress_context("Removing background") as progress:
            task = progress.add_task(f"[cyan]{path.name}", total=100)
            result = remover.remove_background(path, output, threshold=threshold)
            progress.update(task, completed=100)
        
        if result.success:
            console.print(f"\n[bold green]✅ Background removed![/]")
            console.print(f"   [dim]Input:[/]  {result.input_path}")
            console.print(f"   [dim]Output:[/] {result.output_path}")
            console.print(f"   [dim]Size:[/]   {result.original_size[0]}x{result.original_size[1]}")
        else:
            console.print(f"[red]Error:[/] {result.message}")
            raise typer.Exit(1)
            
    except ImportError as e:
        console.print(
            "[red]Error:[/] Background removal requires AI dependencies. "
            "Install with: [cyan]poetry install --with ai[/]"
        )
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)


@app.command()
def upscale(
    path: Path = typer.Argument(..., help="Image file path"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path"),
    scale: int = typer.Option(4, "--scale", "-s", help="Scale factor (currently only 4x)"),
) -> None:
    """AI-powered image upscaling using Real-ESRGAN.
    
    Upscales images by 4x using Real-ESRGAN neural network.
    Large images are automatically processed in tiles to prevent memory issues.
    Model is downloaded automatically on first use (~64MB).
    
    Examples:
        aopt upscale small.jpg
        aopt upscale icon.png -o icon_4x.png
        aopt upscale thumbnail.jpg --scale 4
    """
    try:
        from aopt.ai.upscale import AIUpscaler
        from aopt.cli.dashboard import Dashboard
        
        upscaler = AIUpscaler(console=console)
        dashboard = Dashboard(console)
        
        with dashboard.progress_context(f"Upscaling {scale}x") as progress:
            task = progress.add_task(f"[cyan]{path.name}", total=100)
            result = upscaler.upscale(path, output, scale=scale)
            progress.update(task, completed=100)
        
        if result.success:
            console.print(f"\n[bold green]✅ Image upscaled {scale}x![/]")
            console.print(f"   [dim]Input:[/]  {result.input_path}")
            console.print(f"   [dim]Output:[/] {result.output_path}")
            console.print(
                f"   [dim]Size:[/]   {result.original_size[0]}x{result.original_size[1]} → "
                f"{result.upscaled_size[0]}x{result.upscaled_size[1]}"
            )
        else:
            console.print(f"[red]Error:[/] {result.message}")
            raise typer.Exit(1)
            
    except ImportError as e:
        console.print(
            "[red]Error:[/] AI upscaling requires AI dependencies. "
            "Install with: [cyan]poetry install --with ai[/]"
        )
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)


@app.command()
def models(
    action: str = typer.Argument("list", help="Action: list, download, info, delete"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Model name"),
) -> None:
    """Manage AI models.
    
    List, download, or delete ONNX models used by AI features.
    
    Examples:
        aopt models list
        aopt models download -n rmbg-1.4
        aopt models info -n realesrgan-x4
        aopt models delete -n u2net
    """
    try:
        from aopt.ai.models import ModelManager, MODEL_REGISTRY
        
        manager = ModelManager(console=console)
        
        if action == "list":
            console.print("\n[bold cyan]📦 AI Models[/]\n")
            
            # Show registry models
            for model_name, info in MODEL_REGISTRY.items():
                available = manager.is_model_available(model_name)
                status = "[green]●[/] Downloaded" if available else "[dim]○[/] Not downloaded"
                console.print(f"  [bold]{model_name}[/]")
                console.print(f"    {status}")
                console.print(f"    [dim]{info.get('description', '')}[/]")
        
        elif action == "download":
            if not name:
                console.print("[red]Error:[/] Specify model with --name")
                raise typer.Exit(1)
            manager.ensure_model(name)
        
        elif action == "info":
            if not name:
                console.print("[red]Error:[/] Specify model with --name")
                raise typer.Exit(1)
            info = manager.get_model_info(name)
            console.print(f"\n[bold cyan]Model: {name}[/]")
            console.print(f"  [dim]Path:[/] {info['path']}")
            console.print(f"  [dim]Providers:[/] {', '.join(info['providers'])}")
            console.print(f"  [dim]Inputs:[/]")
            for inp in info['inputs']:
                console.print(f"    - {inp['name']}: {inp['shape']} ({inp['type']})")
            console.print(f"  [dim]Outputs:[/]")
            for out in info['outputs']:
                console.print(f"    - {out['name']}: {out['shape']} ({out['type']})")
        
        elif action == "delete":
            if not name:
                console.print("[red]Error:[/] Specify model with --name")
                raise typer.Exit(1)
            if manager.delete_model(name):
                console.print(f"[green]✓ Model '{name}' deleted[/]")
            else:
                console.print(f"[yellow]Model '{name}' not found[/]")
        
        else:
            console.print(f"[red]Unknown action:[/] {action}")
            console.print("Available: list, download, info, delete")
            
    except ImportError:
        console.print(
            "[red]Error:[/] Model management requires AI dependencies. "
            "Install with: [cyan]poetry install --with ai[/]"
        )
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

