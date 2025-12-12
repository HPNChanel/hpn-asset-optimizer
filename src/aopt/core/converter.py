"""Image format converter."""

from enum import Enum
from pathlib import Path

from PIL import Image

from aopt.cli.dashboard import ProcessingResult
from aopt.utils.image_io import load_image, save_image, get_file_size


class ImageFormat(str, Enum):
    """Supported output formats."""
    
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"
    AVIF = "avif"
    GIF = "gif"
    BMP = "bmp"


# Format extension mapping
FORMAT_EXTENSIONS = {
    ImageFormat.JPEG: ".jpg",
    ImageFormat.PNG: ".png",
    ImageFormat.WEBP: ".webp",
    ImageFormat.AVIF: ".avif",
    ImageFormat.GIF: ".gif",
    ImageFormat.BMP: ".bmp",
}


class Converter:
    """Format converter with optimal settings for each format."""
    
    def convert(
        self,
        input_path: Path,
        format: ImageFormat,
        quality: int = 85,
        output: Path | None = None,
        preserve_transparency: bool = True,
    ) -> ProcessingResult:
        """Convert image to a different format.
        
        Args:
            input_path: Input image path.
            format: Target format.
            quality: Quality for lossy formats.
            output: Output path (auto-generated if None).
            preserve_transparency: Keep alpha channel if possible.
            
        Returns:
            ProcessingResult with conversion details.
        """
        try:
            input_size = get_file_size(input_path)
            img = load_image(input_path)
            
            # Determine output path
            ext = FORMAT_EXTENSIONS[format]
            if output is None:
                output = input_path.with_suffix(ext)
            elif output.is_dir():
                output = output / f"{input_path.stem}{ext}"
            
            # Prepare image for target format
            img = self._prepare_for_format(img, format, preserve_transparency)
            
            # Save with format-specific settings
            output_size = self._save_format(img, output, format, quality)
            
            return ProcessingResult(
                input_path=input_path,
                output_path=output,
                input_size=input_size,
                output_size=output_size,
                success=True,
                message=f"Converted to {format.value.upper()}",
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
    
    def _prepare_for_format(
        self,
        img: Image.Image,
        format: ImageFormat,
        preserve_transparency: bool,
    ) -> Image.Image:
        """Prepare image color mode for target format."""
        # JPEG doesn't support transparency
        if format == ImageFormat.JPEG:
            if img.mode in {"RGBA", "LA", "P"}:
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                if "A" in img.mode:
                    background.paste(img, mask=img.split()[-1])
                else:
                    background.paste(img)
                return background
            elif img.mode != "RGB":
                return img.convert("RGB")
        
        # PNG, WebP, GIF can preserve transparency
        elif format in {ImageFormat.PNG, ImageFormat.WEBP, ImageFormat.GIF}:
            if preserve_transparency and img.mode == "RGBA":
                return img
            elif img.mode == "P" and preserve_transparency:
                return img.convert("RGBA")
        
        # BMP doesn't support transparency
        elif format == ImageFormat.BMP:
            if img.mode not in {"RGB", "L"}:
                return img.convert("RGB")
        
        return img
    
    def _save_format(
        self,
        img: Image.Image,
        output: Path,
        format: ImageFormat,
        quality: int,
    ) -> int:
        """Save with format-specific optimal settings."""
        if format == ImageFormat.JPEG:
            return save_image(
                img, output,
                quality=quality,
                optimize=True,
                progressive=True,
            )
        
        elif format == ImageFormat.PNG:
            return save_image(
                img, output,
                optimize=True,
                compress_level=9,
            )
        
        elif format == ImageFormat.WEBP:
            return save_image(
                img, output,
                quality=quality,
                method=6,
            )
        
        elif format == ImageFormat.AVIF:
            return save_image(
                img, output,
                quality=quality,
            )
        
        elif format == ImageFormat.GIF:
            # For GIF, reduce colors if needed
            if img.mode != "P":
                img = img.convert("P", palette=Image.ADAPTIVE, colors=256)
            return save_image(img, output, optimize=True)
        
        else:
            return save_image(img, output, quality=quality)
    
    def auto_convert(
        self,
        input_path: Path,
        output: Path | None = None,
    ) -> ProcessingResult:
        """Auto-select best format based on image content.
        
        - Photos → JPEG or WebP
        - Graphics/Screenshots → PNG or WebP
        - Transparent images → WebP or PNG
        """
        try:
            img = load_image(input_path)
            
            # Determine best format
            has_transparency = img.mode in {"RGBA", "LA", "PA"}
            is_photo = self._is_photo(img)
            
            if has_transparency:
                format = ImageFormat.WEBP  # Best for transparency + compression
            elif is_photo:
                format = ImageFormat.WEBP  # Great for photos
            else:
                format = ImageFormat.PNG  # Best for graphics
            
            return self.convert(input_path, format, output=output)
        except Exception as e:
            return ProcessingResult(
                input_path=input_path,
                output_path=output or input_path,
                input_size=0,
                output_size=0,
                success=False,
                message=str(e),
            )
    
    def _is_photo(self, img: Image.Image) -> bool:
        """Detect if image is a photo (many colors) vs graphic (few colors).
        
        Uses a sample of pixels to estimate color diversity.
        """
        # Sample pixels for performance
        sample_size = min(1000, img.width * img.height)
        if sample_size < 100:
            return True
        
        # Get a sample of pixels
        pixels = list(img.convert("RGB").getdata())
        sample = pixels[::len(pixels) // sample_size]
        unique_colors = len(set(sample))
        
        # Photos typically have high color diversity
        return unique_colors > sample_size * 0.3
