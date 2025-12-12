"""Image compression engine."""

from enum import Enum
from pathlib import Path

from PIL import Image

from aopt.cli.dashboard import ProcessingResult
from aopt.utils.image_io import load_image, save_image, get_file_size


class CompressionMode(str, Enum):
    """Compression mode options."""
    
    LOSSY = "lossy"
    LOSSLESS = "lossless"


class Compressor:
    """High-performance image compressor.
    
    Supports JPEG, PNG, WebP, and AVIF with optimized settings
    for each format.
    """
    
    # Quality presets for different use cases
    QUALITY_PRESETS = {
        "web": 80,
        "high": 90,
        "maximum": 95,
        "low": 60,
    }
    
    def compress(
        self,
        input_path: Path,
        quality: int = 85,
        mode: CompressionMode = CompressionMode.LOSSY,
        output: Path | None = None,
    ) -> ProcessingResult:
        """Compress an image with optimized settings.
        
        Args:
            input_path: Path to input image.
            quality: Quality level (1-100, only for lossy mode).
            mode: Compression mode (lossy or lossless).
            output: Output path (defaults to input with _compressed suffix).
            
        Returns:
            ProcessingResult with compression details.
        """
        try:
            input_size = get_file_size(input_path)
            img = load_image(input_path)
            
            # Determine output path
            if output is None:
                stem = input_path.stem
                suffix = input_path.suffix
                output = input_path.parent / f"{stem}_compressed{suffix}"
            
            # Apply compression based on mode
            if mode == CompressionMode.LOSSLESS:
                output_size = self._compress_lossless(img, output)
            else:
                output_size = self._compress_lossy(img, output, quality)
            
            return ProcessingResult(
                input_path=input_path,
                output_path=output,
                input_size=input_size,
                output_size=output_size,
                success=True,
                message=f"Compressed with {mode.value} mode",
            )
        except Exception as e:
            return ProcessingResult(
                input_path=input_path,
                output_path=output or input_path,
                input_size=0,
                output_size=0,
                success=False,
                message=str(e),
            )
    
    def _compress_lossy(
        self,
        img: Image.Image,
        output: Path,
        quality: int,
    ) -> int:
        """Apply lossy compression."""
        ext = output.suffix.lower()
        
        # Convert to optimal color mode for format
        if ext in {".jpg", ".jpeg"}:
            if img.mode in {"RGBA", "LA", "P"}:
                # Handle transparency
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                if "A" in img.mode:
                    background.paste(img, mask=img.split()[-1])
                else:
                    background.paste(img)
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")
        
        return save_image(img, output, quality=quality)
    
    def _compress_lossless(
        self,
        img: Image.Image,
        output: Path,
    ) -> int:
        """Apply lossless compression."""
        ext = output.suffix.lower()
        
        if ext == ".png":
            # Maximum PNG compression
            return save_image(img, output, optimize=True, compress_level=9)
        elif ext == ".webp":
            # Lossless WebP
            return save_image(img, output, lossless=True, method=6)
        else:
            # For formats without lossless mode, use high quality
            return save_image(img, output, quality=100)
    
    def compress_to_target_size(
        self,
        input_path: Path,
        target_kb: int,
        output: Path | None = None,
        min_quality: int = 30,
    ) -> ProcessingResult:
        """Compress to achieve a target file size.
        
        Uses binary search to find optimal quality setting.
        
        Args:
            input_path: Input image path.
            target_kb: Target file size in KB.
            output: Output path.
            min_quality: Minimum quality to accept.
            
        Returns:
            ProcessingResult with compression details.
        """
        target_bytes = target_kb * 1024
        
        try:
            input_size = get_file_size(input_path)
            img = load_image(input_path)
            
            if output is None:
                stem = input_path.stem
                suffix = input_path.suffix
                output = input_path.parent / f"{stem}_sized{suffix}"
            
            # Binary search for optimal quality
            low, high = min_quality, 95
            best_quality = high
            best_size = float("inf")
            
            while low <= high:
                mid = (low + high) // 2
                output_size = save_image(img, output, quality=mid)
                
                if output_size <= target_bytes:
                    # We can try higher quality
                    best_quality = mid
                    best_size = output_size
                    low = mid + 1
                else:
                    # Need lower quality
                    high = mid - 1
            
            # Save with best found quality
            if best_size > target_bytes and best_quality > min_quality:
                best_quality = min_quality
                best_size = save_image(img, output, quality=best_quality)
            
            return ProcessingResult(
                input_path=input_path,
                output_path=output,
                input_size=input_size,
                output_size=int(best_size),
                success=True,
                message=f"Compressed to {target_kb}KB target (quality: {best_quality})",
            )
        except Exception as e:
            return ProcessingResult(
                input_path=input_path,
                output_path=output or input_path,
                input_size=0,
                output_size=0,
                success=False,
                message=str(e),
            )
