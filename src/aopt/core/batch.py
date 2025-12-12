"""Concurrent batch processor with Rich dashboard and hash caching."""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Any

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout

from aopt.cli.dashboard import ProcessingResult, Dashboard
from aopt.core.converter import ImageFormat
from aopt.utils.image_io import collect_images, load_image, save_image, get_file_size
from aopt.ai.variants import HashCache


class BatchProcessor:
    """High-performance concurrent batch processor with hash caching.
    
    Uses ThreadPoolExecutor for parallel processing with a real-time
    Rich dashboard showing progress and statistics. Includes content-based
    hash caching to skip already processed files.
    """
    
    def __init__(
        self,
        console: Console | None = None,
        workers: int | None = None,
        use_cache: bool = True,
        cache_file: Path | None = None,
    ) -> None:
        """Initialize batch processor.
        
        Args:
            console: Rich console for output.
            workers: Number of worker threads (defaults to CPU count).
            use_cache: Enable hash-based caching to skip processed files.
            cache_file: Path to cache file for persistence across sessions.
        """
        self.console = console or Console()
        self.workers = workers or min(os.cpu_count() or 4, 8)
        self.dashboard = Dashboard(self.console)
        self.use_cache = use_cache
        
        # Initialize hash cache
        if use_cache:
            if cache_file is None:
                cache_file = Path.home() / ".aopt" / "hash_cache.json"
            self.cache = HashCache(cache_file)
        else:
            self.cache = None
    
    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        quality: int = 85,
        format: ImageFormat | None = None,
        recursive: bool = False,
        callback: Callable[[Path], ProcessingResult] | None = None,
        force: bool = False,
    ) -> list[ProcessingResult]:
        """Process all images in a directory.
        
        Args:
            input_dir: Input directory.
            output_dir: Output directory.
            quality: Compression quality.
            format: Convert to format (None = keep original).
            recursive: Include subdirectories.
            callback: Custom processing function.
            force: Force reprocessing even if cached.
            
        Returns:
            List of ProcessingResult for each file.
        """
        # Collect images
        self.console.print(f"[dim]Scanning {input_dir}...[/]")
        images = collect_images(input_dir, recursive=recursive)
        
        if not images:
            self.console.print("[yellow]No images found.[/]")
            return []
        
        self.console.print(f"[cyan]Found {len(images)} images[/]")
        
        # Filter out already processed files using hash cache
        operation_id = self._get_operation_id(quality, format)
        images_to_process = []
        skipped = []
        
        if self.use_cache and self.cache and not force:
            for img_path in images:
                if self.cache.is_processed(img_path, operation_id):
                    skipped.append(img_path)
                else:
                    images_to_process.append(img_path)
            
            if skipped:
                self.console.print(
                    f"[dim]⏭️  Skipping {len(skipped)} already processed files[/]"
                )
        else:
            images_to_process = images
        
        if not images_to_process:
            self.console.print("[green]All files already processed![/]")
            return []
        
        self.console.print(f"[cyan]Processing {len(images_to_process)} images...[/]")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process with progress
        results: list[ProcessingResult] = []
        start_time = time.time()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(
                "[cyan]Processing images...",
                total=len(images_to_process)
            )
            
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                # Submit all tasks
                futures = {}
                for img_path in images_to_process:
                    # Calculate relative path for output
                    rel_path = img_path.relative_to(input_dir)
                    out_path = output_dir / rel_path
                    
                    if format:
                        from aopt.core.converter import FORMAT_EXTENSIONS
                        out_path = out_path.with_suffix(FORMAT_EXTENSIONS[format])
                    
                    future = executor.submit(
                        self._process_single,
                        img_path, out_path, quality, format, callback
                    )
                    futures[future] = img_path
                
                # Collect results as they complete
                for future in as_completed(futures):
                    img_path = futures[future]
                    result = future.result()
                    results.append(result)
                    
                    # Mark as processed in cache
                    if self.use_cache and self.cache and result.success:
                        self.cache.mark_processed(
                            img_path,
                            operation_id,
                            {
                                "output_path": str(result.output_path),
                                "output_size": result.output_size,
                            }
                        )
                    
                    # Update progress with file name
                    status = "✓" if result.success else "✗"
                    progress.update(
                        task,
                        advance=1,
                        description=f"[cyan]{status} {result.input_path.name}",
                    )
        
        elapsed = time.time() - start_time
        
        # Show summary
        self.dashboard.show_batch_summary(results, elapsed)
        
        return results
    
    def _get_operation_id(
        self,
        quality: int,
        format: ImageFormat | None = None,
    ) -> str:
        """Generate a unique operation ID for caching."""
        format_str = format.value if format else "original"
        return f"optimize_{format_str}_q{quality}"
    
    def _process_single(
        self,
        input_path: Path,
        output_path: Path,
        quality: int,
        format: ImageFormat | None,
        callback: Callable[[Path], ProcessingResult] | None,
    ) -> ProcessingResult:
        """Process a single image."""
        try:
            if callback:
                return callback(input_path)
            
            input_size = get_file_size(input_path)
            img = load_image(input_path)
            
            # Create output directory
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Process and save
            if format:
                from aopt.core.converter import Converter
                converter = Converter()
                return converter.convert(input_path, format, quality, output_path)
            else:
                output_size = save_image(img, output_path, quality=quality)
                return ProcessingResult(
                    input_path=input_path,
                    output_path=output_path,
                    input_size=input_size,
                    output_size=output_size,
                    success=True,
                )
        except Exception as e:
            return ProcessingResult(
                input_path=input_path,
                output_path=output_path,
                input_size=0,
                output_size=0,
                success=False,
                message=str(e),
            )
    
    def process_with_preset(
        self,
        input_dir: Path,
        output_dir: Path,
        preset: dict[str, Any],
    ) -> list[ProcessingResult]:
        """Process directory with a preset configuration.
        
        Args:
            input_dir: Input directory.
            output_dir: Output directory.
            preset: Preset configuration dict.
            
        Returns:
            List of ProcessingResult for each file.
        """
        quality = preset.get("quality", 85)
        format_str = preset.get("format")
        format = ImageFormat(format_str) if format_str else None
        max_width = preset.get("max_width")
        max_height = preset.get("max_height")
        
        def process_with_preset_config(path: Path) -> ProcessingResult:
            from aopt.core.processor import ImageProcessor
            processor = ImageProcessor()
            
            # Calculate output path
            rel_path = path.relative_to(input_dir)
            out_path = output_dir / rel_path
            if format:
                from aopt.core.converter import FORMAT_EXTENSIONS
                out_path = out_path.with_suffix(FORMAT_EXTENSIONS[format])
            
            return processor.optimize(
                path,
                output=out_path,
                quality=quality,
                format=format,
                max_width=max_width,
                max_height=max_height,
            )
        
        return self.process_directory(
            input_dir, output_dir,
            callback=process_with_preset_config,
        )
    
    def clear_cache(self) -> None:
        """Clear the processing cache."""
        if self.cache:
            self.cache.clear()
            self.console.print("[green]Cache cleared![/]")
    
    def cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        if self.cache:
            return {
                "entries": len(self.cache),
                "cache_file": str(self.cache.cache_file),
            }
        return {"enabled": False}
