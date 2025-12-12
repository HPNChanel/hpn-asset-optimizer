"""Smart resize with content-aware algorithms."""

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from aopt.cli.dashboard import ProcessingResult
from aopt.utils.image_io import load_image, save_image, get_file_size


class SmartResize:
    """Smart resize with content-aware capabilities.
    
    Provides intelligent resizing that considers image content
    for better visual results.
    """
    
    def resize(
        self,
        input_path: Path,
        width: int | None = None,
        height: int | None = None,
        output: Path | None = None,
        mode: str = "fit",
        quality: int = 85,
    ) -> ProcessingResult:
        """Smart resize an image.
        
        Args:
            input_path: Input image path.
            width: Target width.
            height: Target height.
            output: Output path.
            mode: Resize mode (fit, fill, cover, contain).
            quality: Output quality.
            
        Returns:
            ProcessingResult with resize details.
        """
        try:
            input_size = get_file_size(input_path)
            img = load_image(input_path)
            
            if output is None:
                output = input_path.parent / f"{input_path.stem}_resized{input_path.suffix}"
            
            # Calculate target dimensions
            target_width, target_height = self._calculate_dimensions(
                img.width, img.height, width, height, mode
            )
            
            # Resize with high-quality resampling
            resized = img.resize(
                (target_width, target_height),
                Image.Resampling.LANCZOS
            )
            
            # Handle fill/cover modes that require cropping
            if mode in {"fill", "cover"} and width and height:
                resized = self._center_crop(resized, width, height)
            
            output_size = save_image(resized, output, quality=quality)
            
            return ProcessingResult(
                input_path=input_path,
                output_path=output,
                input_size=input_size,
                output_size=output_size,
                success=True,
                message=f"Resized to {resized.width}×{resized.height}",
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
    
    def _calculate_dimensions(
        self,
        orig_width: int,
        orig_height: int,
        target_width: int | None,
        target_height: int | None,
        mode: str,
    ) -> tuple[int, int]:
        """Calculate target dimensions based on mode."""
        if not target_width and not target_height:
            return orig_width, orig_height
        
        if mode == "fit" or mode == "contain":
            # Scale to fit within bounds
            if target_width and target_height:
                scale = min(target_width / orig_width, target_height / orig_height)
            elif target_width:
                scale = target_width / orig_width
            else:
                scale = target_height / orig_height  # type: ignore
            
            return int(orig_width * scale), int(orig_height * scale)
        
        elif mode == "fill" or mode == "cover":
            # Scale to cover bounds (may need cropping)
            if target_width and target_height:
                scale = max(target_width / orig_width, target_height / orig_height)
            elif target_width:
                scale = target_width / orig_width
            else:
                scale = target_height / orig_height  # type: ignore
            
            return int(orig_width * scale), int(orig_height * scale)
        
        else:
            # Direct resize
            return target_width or orig_width, target_height or orig_height
    
    def _center_crop(
        self,
        img: Image.Image,
        width: int,
        height: int,
    ) -> Image.Image:
        """Center crop an image to exact dimensions."""
        left = (img.width - width) // 2
        top = (img.height - height) // 2
        right = left + width
        bottom = top + height
        
        return img.crop((left, top, right, bottom))
    
    def thumbnail(
        self,
        input_path: Path,
        size: int = 256,
        output: Path | None = None,
        square: bool = True,
        quality: int = 80,
    ) -> ProcessingResult:
        """Create a thumbnail.
        
        Args:
            input_path: Input image path.
            size: Thumbnail size.
            output: Output path.
            square: Create square thumbnail (center crop).
            quality: Output quality.
            
        Returns:
            ProcessingResult with thumbnail details.
        """
        if square:
            return self.resize(
                input_path,
                width=size,
                height=size,
                output=output,
                mode="cover",
                quality=quality,
            )
        else:
            return self.resize(
                input_path,
                width=size,
                height=size,
                output=output,
                mode="fit",
                quality=quality,
            )
