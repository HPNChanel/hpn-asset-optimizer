"""Image I/O utilities with Pillow optimization."""

from pathlib import Path
from typing import Any

from PIL import Image, ExifTags

from aopt.cli.dashboard import ImageInfo


# Supported image extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".avif"}


def is_supported_image(path: Path) -> bool:
    """Check if a file is a supported image format."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def load_image(path: Path) -> Image.Image:
    """Load an image with Pillow.
    
    Args:
        path: Path to the image file.
        
    Returns:
        PIL Image object.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file is not a supported image.
    """
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    
    if not is_supported_image(path):
        raise ValueError(f"Unsupported image format: {path.suffix}")
    
    img = Image.open(path)
    # Load image data into memory for faster processing
    img.load()
    return img


def save_image(
    img: Image.Image,
    path: Path,
    quality: int = 85,
    optimize: bool = True,
    **kwargs: Any,
) -> int:
    """Save an image with optimized settings.
    
    Args:
        img: PIL Image to save.
        path: Output path.
        quality: Quality for lossy formats (1-100).
        optimize: Enable format-specific optimizations.
        **kwargs: Additional format-specific options.
        
    Returns:
        File size in bytes.
    """
    # Create output directory if needed
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine format from extension
    ext = path.suffix.lower()
    save_kwargs: dict[str, Any] = {"optimize": optimize, **kwargs}
    
    # Handle transparency for JPEG
    if ext in {".jpg", ".jpeg"} and img.mode in {"RGBA", "LA", "P"}:
        # Convert to RGB, filling transparent areas with white
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
        img = background
    
    # Format-specific settings
    if ext in {".jpg", ".jpeg"}:
        save_kwargs["quality"] = quality
        save_kwargs["progressive"] = True
    elif ext == ".png":
        save_kwargs["compress_level"] = 9
    elif ext == ".webp":
        save_kwargs["quality"] = quality
        save_kwargs["method"] = 6  # Best compression
    elif ext == ".avif":
        save_kwargs["quality"] = quality
    
    img.save(path, **save_kwargs)
    return path.stat().st_size


def get_image_info(path: Path) -> ImageInfo:
    """Get detailed information about an image.
    
    Args:
        path: Path to the image file.
        
    Returns:
        ImageInfo object with metadata.
    """
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    
    img = Image.open(path)
    
    # Extract EXIF data
    exif_data: dict[str, Any] | None = None
    has_exif = False
    
    try:
        raw_exif = img.getexif()
        if raw_exif:
            has_exif = True
            exif_data = {}
            for tag_id, value in raw_exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                # Skip binary data
                if isinstance(value, bytes):
                    continue
                exif_data[tag_name] = value
    except Exception:
        pass  # Some formats don't support EXIF
    
    return ImageInfo(
        path=path,
        format=img.format or "Unknown",
        mode=img.mode,
        width=img.width,
        height=img.height,
        size_bytes=path.stat().st_size,
        has_exif=has_exif,
        exif_data=exif_data,
    )


def get_file_size(path: Path) -> int:
    """Get file size in bytes."""
    return path.stat().st_size


def collect_images(
    directory: Path,
    recursive: bool = False,
) -> list[Path]:
    """Collect all image files from a directory.
    
    Args:
        directory: Directory to scan.
        recursive: Include subdirectories.
        
    Returns:
        List of image file paths.
    """
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    
    pattern = "**/*" if recursive else "*"
    images = []
    
    for path in directory.glob(pattern):
        if path.is_file() and is_supported_image(path):
            images.append(path)
    
    return sorted(images)
