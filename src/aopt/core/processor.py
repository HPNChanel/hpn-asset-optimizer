"""Main image processor combining all operations."""

from pathlib import Path
from typing import Any

from PIL import Image

from aopt.cli.dashboard import ProcessingResult
from aopt.core.compressor import Compressor, CompressionMode
from aopt.core.converter import Converter, ImageFormat
from aopt.utils.image_io import load_image, save_image, get_file_size


class ImageProcessor:
    """Central image processor that combines all optimization operations.
    
    This is the main entry point for image processing, providing a unified
    interface for compression, conversion, resizing, and optimization.
    """
    
    def __init__(self) -> None:
        self.compressor = Compressor()
        self.converter = Converter()
    
    def optimize(
        self,
        input_path: Path,
        output: Path | None = None,
        quality: int = 85,
        format: ImageFormat | None = None,
        max_width: int | None = None,
        max_height: int | None = None,
        strip_metadata: bool = True,
    ) -> ProcessingResult:
        """Optimize an image with multiple operations.
        
        Args:
            input_path: Input image path.
            output: Output path.
            quality: Quality for lossy compression.
            format: Target format (auto-detect if None).
            max_width: Maximum width (resize if larger).
            max_height: Maximum height (resize if larger).
            strip_metadata: Remove EXIF/metadata.
            
        Returns:
            ProcessingResult with optimization details.
        """
        try:
            input_size = get_file_size(input_path)
            img = load_image(input_path)
            
            # Determine output path and format
            target_format = format
            if output is None:
                if target_format:
                    from aopt.core.converter import FORMAT_EXTENSIONS
                    ext = FORMAT_EXTENSIONS[target_format]
                    output = input_path.with_suffix(ext)
                else:
                    output = input_path.parent / f"{input_path.stem}_optimized{input_path.suffix}"
            
            # 1. Resize if needed
            if max_width or max_height:
                img = self._resize_to_fit(img, max_width, max_height)
            
            # 2. Strip metadata if requested
            if strip_metadata:
                img = self._strip_metadata(img)
            
            # 3. Save with compression
            if target_format:
                # Convert format
                result = self.converter.convert(
                    input_path, target_format,
                    quality=quality, output=output
                )
                return result
            else:
                # Compress in original format
                output_size = save_image(img, output, quality=quality)
            
            return ProcessingResult(
                input_path=input_path,
                output_path=output,
                input_size=input_size,
                output_size=output_size,
                success=True,
                message="Optimized successfully",
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
    
    def _resize_to_fit(
        self,
        img: Image.Image,
        max_width: int | None,
        max_height: int | None,
    ) -> Image.Image:
        """Resize image to fit within constraints while preserving aspect ratio."""
        width, height = img.size
        
        # Calculate new dimensions
        scale = 1.0
        if max_width and width > max_width:
            scale = min(scale, max_width / width)
        if max_height and height > max_height:
            scale = min(scale, max_height / height)
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return img
    
    def _strip_metadata(self, img: Image.Image) -> Image.Image:
        """Create a clean copy without metadata."""
        clean_img = Image.new(img.mode, img.size)
        clean_img.putdata(list(img.getdata()))
        return clean_img
    
    def resize(
        self,
        input_path: Path,
        width: int | None = None,
        height: int | None = None,
        output: Path | None = None,
        quality: int = 85,
    ) -> ProcessingResult:
        """Resize an image.
        
        Args:
            input_path: Input image path.
            width: Target width (or None to auto-calculate).
            height: Target height (or None to auto-calculate).
            output: Output path.
            quality: Quality for saving.
            
        Returns:
            ProcessingResult with resize details.
        """
        try:
            input_size = get_file_size(input_path)
            img = load_image(input_path)
            
            if output is None:
                output = input_path.parent / f"{input_path.stem}_resized{input_path.suffix}"
            
            # Calculate dimensions
            orig_width, orig_height = img.size
            
            if width and height:
                new_width, new_height = width, height
            elif width:
                scale = width / orig_width
                new_width, new_height = width, int(orig_height * scale)
            elif height:
                scale = height / orig_height
                new_width, new_height = int(orig_width * scale), height
            else:
                new_width, new_height = orig_width, orig_height
            
            # Resize with high-quality resampling
            resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            output_size = save_image(resized, output, quality=quality)
            
            return ProcessingResult(
                input_path=input_path,
                output_path=output,
                input_size=input_size,
                output_size=output_size,
                success=True,
                message=f"Resized to {new_width}×{new_height}",
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
