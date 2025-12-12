"""Smart Adaptive Variants Generator.

Generates multiple variants of images for different use cases:
- Dark Mode variants (color inversion based on luminance)
- Social media aspect ratio adaptations (1:1 to 9:16 with intelligent fill)
- Thumbnail and icon variants
"""

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, ImageFilter

from aopt.cli.dashboard import ProcessingResult
from aopt.utils.image_io import load_image, save_image, get_file_size


class VariantGenerator:
    """Generates adaptive image variants for different use cases.
    
    Supports dark mode conversion, social media aspect ratio adaptation,
    and intelligent background extension using mirroring/edge extension.
    """
    
    # Standard aspect ratios
    ASPECT_RATIOS = {
        "square": (1, 1),      # Instagram, Profile pics
        "portrait": (9, 16),   # Stories, Reels, TikTok
        "landscape": (16, 9),  # YouTube, Desktop
        "classic": (4, 3),     # Traditional photo
        "ultrawide": (21, 9),  # Cinematic
    }
    
    def generate_dark_mode(
        self,
        input_path: Path,
        output: Path | None = None,
        threshold: float = 0.5,
        preserve_colors: bool = False,
    ) -> ProcessingResult:
        """Generate a dark mode variant of an image.
        
        Uses luminance analysis to intelligently invert colors while
        preserving image quality and readability.
        
        Args:
            input_path: Input image path.
            output: Output path.
            threshold: Luminance threshold for inversion decision.
            preserve_colors: If True, only invert lightness in HSL.
            
        Returns:
            ProcessingResult with generation details.
        """
        try:
            input_size = get_file_size(input_path)
            img = load_image(input_path)
            
            if output is None:
                output = input_path.parent / f"{input_path.stem}_dark{input_path.suffix}"
            
            # Convert to numpy array for luminance calculation
            img_rgb = img.convert("RGB")
            img_array = np.array(img_rgb, dtype=np.float32) / 255.0
            
            # Calculate luminance using standard coefficients (ITU-R BT.709)
            luminance = self._calculate_luminance(img_array)
            avg_luminance = float(np.mean(luminance))
            
            # Decide if we should invert (for light images)
            if avg_luminance > threshold:
                if preserve_colors:
                    # Invert only the luminance channel (preserve hue)
                    dark_img = self._invert_luminance(img)
                else:
                    # Full color inversion
                    dark_img = ImageOps.invert(img_rgb)
            else:
                # Image is already dark, apply subtle adjustment
                dark_img = self._darken_midtones(img_rgb)
            
            # Convert back to original mode if needed
            if img.mode == "RGBA":
                dark_img = dark_img.convert("RGB")
                dark_img.putalpha(img.split()[-1])
            
            output_size = save_image(dark_img, output)
            
            return ProcessingResult(
                input_path=input_path,
                output_path=output,
                input_size=input_size,
                output_size=output_size,
                success=True,
                message=f"Dark mode variant (avg luminance: {avg_luminance:.2f})",
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
    
    def _calculate_luminance(self, img_array: np.ndarray) -> np.ndarray:
        """Calculate relative luminance per pixel.
        
        Uses ITU-R BT.709 coefficients for sRGB.
        L = 0.2126 * R + 0.7152 * G + 0.0722 * B
        """
        return (
            0.2126 * img_array[:, :, 0] +
            0.7152 * img_array[:, :, 1] +
            0.0722 * img_array[:, :, 2]
        )
    
    def _invert_luminance(self, img: Image.Image) -> Image.Image:
        """Invert luminance while preserving hue and saturation."""
        # Convert to HSV
        hsv = img.convert("HSV")
        h, s, v = hsv.split()
        
        # Invert the value (brightness) channel
        v_array = np.array(v, dtype=np.uint8)
        v_inverted = 255 - v_array
        v_new = Image.fromarray(v_inverted, mode="L")
        
        # Recombine
        hsv_new = Image.merge("HSV", (h, s, v_new))
        return hsv_new.convert("RGB")
    
    def _darken_midtones(self, img: Image.Image) -> Image.Image:
        """Darken midtones for already dark images."""
        # Apply a subtle darkening curve
        def darken_curve(x):
            return int(x * 0.85)
        
        lut = [darken_curve(i) for i in range(256)] * 3
        return img.point(lut)
    
    def extend_to_aspect(
        self,
        input_path: Path,
        target_aspect: str | tuple[int, int] = "portrait",
        output: Path | None = None,
        method: str = "mirror",
        quality: int = 85,
    ) -> ProcessingResult:
        """Extend an image to a different aspect ratio.
        
        Intelligently fills the background using mirror, blur, or solid fill.
        
        Args:
            input_path: Input image path.
            target_aspect: Target aspect ratio name or tuple.
            output: Output path.
            method: Fill method (mirror, blur, solid).
            quality: Output quality.
            
        Returns:
            ProcessingResult with extension details.
        """
        try:
            input_size = get_file_size(input_path)
            img = load_image(input_path)
            
            if output is None:
                aspect_name = target_aspect if isinstance(target_aspect, str) else f"{target_aspect[0]}x{target_aspect[1]}"
                output = input_path.parent / f"{input_path.stem}_{aspect_name}{input_path.suffix}"
            
            # Get target ratio
            if isinstance(target_aspect, str):
                if target_aspect not in self.ASPECT_RATIOS:
                    raise ValueError(f"Unknown aspect ratio: {target_aspect}")
                ratio = self.ASPECT_RATIOS[target_aspect]
            else:
                ratio = target_aspect
            
            # Calculate target dimensions
            target_width, target_height = self._calculate_target_size(
                img.width, img.height, ratio
            )
            
            # Create the extended image
            if method == "mirror":
                result = self._extend_mirror(img, target_width, target_height)
            elif method == "blur":
                result = self._extend_blur(img, target_width, target_height)
            elif method == "solid":
                result = self._extend_solid(img, target_width, target_height)
            else:
                raise ValueError(f"Unknown fill method: {method}")
            
            output_size = save_image(result, output, quality=quality)
            
            return ProcessingResult(
                input_path=input_path,
                output_path=output,
                input_size=input_size,
                output_size=output_size,
                success=True,
                message=f"Extended to {target_width}×{target_height} ({method})",
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
    
    def _calculate_target_size(
        self,
        orig_width: int,
        orig_height: int,
        ratio: tuple[int, int],
    ) -> tuple[int, int]:
        """Calculate target dimensions to fit the aspect ratio."""
        target_w_ratio, target_h_ratio = ratio
        target_aspect = target_w_ratio / target_h_ratio
        current_aspect = orig_width / orig_height
        
        if abs(current_aspect - target_aspect) < 0.01:
            # Already correct aspect ratio
            return orig_width, orig_height
        
        if current_aspect > target_aspect:
            # Image is wider, need to extend height
            new_height = int(orig_width * target_h_ratio / target_w_ratio)
            return orig_width, new_height
        else:
            # Image is taller, need to extend width
            new_width = int(orig_height * target_w_ratio / target_h_ratio)
            return new_width, orig_height
    
    def _extend_mirror(
        self,
        img: Image.Image,
        target_width: int,
        target_height: int,
    ) -> Image.Image:
        """Extend image using mirrored edges."""
        result = Image.new(img.mode, (target_width, target_height))
        
        # Calculate offset to center the original
        x_offset = (target_width - img.width) // 2
        y_offset = (target_height - img.height) // 2
        
        # Paste original centered
        result.paste(img, (x_offset, y_offset))
        
        # Fill edges with mirrored content
        if y_offset > 0:
            # Top edge
            top_strip = img.crop((0, 0, img.width, min(y_offset, img.height)))
            top_strip = ImageOps.flip(top_strip)
            result.paste(top_strip, (x_offset, 0))
            
            # Bottom edge
            bottom_strip = img.crop((0, max(0, img.height - y_offset), img.width, img.height))
            bottom_strip = ImageOps.flip(bottom_strip)
            result.paste(bottom_strip, (x_offset, y_offset + img.height))
        
        if x_offset > 0:
            # Left edge
            left_strip = img.crop((0, 0, min(x_offset, img.width), img.height))
            left_strip = ImageOps.mirror(left_strip)
            result.paste(left_strip, (0, y_offset))
            
            # Right edge
            right_strip = img.crop((max(0, img.width - x_offset), 0, img.width, img.height))
            right_strip = ImageOps.mirror(right_strip)
            result.paste(right_strip, (x_offset + img.width, y_offset))
        
        return result
    
    def _extend_blur(
        self,
        img: Image.Image,
        target_width: int,
        target_height: int,
    ) -> Image.Image:
        """Extend image using blurred scaled background."""
        # Create blurred background
        bg = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        bg = bg.filter(ImageFilter.GaussianBlur(radius=30))
        
        # Calculate offset to center the original
        x_offset = (target_width - img.width) // 2
        y_offset = (target_height - img.height) // 2
        
        # Paste original centered
        bg.paste(img, (x_offset, y_offset))
        
        return bg
    
    def _extend_solid(
        self,
        img: Image.Image,
        target_width: int,
        target_height: int,
        color: tuple[int, int, int] = (0, 0, 0),
    ) -> Image.Image:
        """Extend image with solid color background."""
        mode = img.mode if img.mode in ("RGB", "RGBA") else "RGB"
        result = Image.new(mode, (target_width, target_height), color)
        
        # Calculate offset to center the original
        x_offset = (target_width - img.width) // 2
        y_offset = (target_height - img.height) // 2
        
        # Paste original centered
        result.paste(img, (x_offset, y_offset))
        
        return result
    
    def generate_icon_variants(
        self,
        input_path: Path,
        output_dir: Path | None = None,
        sizes: list[int] | None = None,
        include_dark: bool = True,
    ) -> list[ProcessingResult]:
        """Generate multiple icon size variants.
        
        Args:
            input_path: Input image path.
            output_dir: Output directory.
            sizes: Icon sizes to generate (default: common iOS/Android sizes).
            include_dark: Generate dark mode variants too.
            
        Returns:
            List of ProcessingResult for each variant.
        """
        if output_dir is None:
            output_dir = input_path.parent / f"{input_path.stem}_icons"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if sizes is None:
            # Common icon sizes for iOS/Android/Web
            sizes = [16, 32, 48, 64, 128, 256, 512, 1024]
        
        results = []
        
        img = load_image(input_path)
        input_size = get_file_size(input_path)
        
        for size in sizes:
            # Light mode variant
            resized = img.resize((size, size), Image.Resampling.LANCZOS)
            light_output = output_dir / f"icon_{size}.png"
            output_size = save_image(resized, light_output)
            
            results.append(ProcessingResult(
                input_path=input_path,
                output_path=light_output,
                input_size=input_size,
                output_size=output_size,
                success=True,
                message=f"Icon {size}×{size}",
            ))
            
            # Dark mode variant
            if include_dark:
                dark_output = output_dir / f"icon_{size}_dark.png"
                dark_result = self.generate_dark_mode(light_output, dark_output)
                results.append(dark_result)
        
        return results


class HashCache:
    """Content-based hash cache for skipping already processed files.
    
    Uses SHA-256 hash of file content to detect duplicates and
    track processed files across sessions.
    """
    
    def __init__(self, cache_file: Path | None = None) -> None:
        """Initialize hash cache.
        
        Args:
            cache_file: Path to cache file for persistence.
        """
        self.cache_file = cache_file
        self._cache: dict[str, dict[str, Any]] = {}
        
        if cache_file and cache_file.exists():
            self._load_cache()
    
    def _load_cache(self) -> None:
        """Load cache from disk."""
        try:
            import json
            with open(self.cache_file, "r") as f:
                self._cache = json.load(f)
        except Exception:
            self._cache = {}
    
    def _save_cache(self) -> None:
        """Persist cache to disk."""
        if self.cache_file:
            import json
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w") as f:
                json.dump(self._cache, f)
    
    def compute_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file content."""
        hasher = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def is_processed(
        self,
        file_path: Path,
        operation: str = "default",
    ) -> bool:
        """Check if a file has already been processed.
        
        Args:
            file_path: Path to check.
            operation: Operation type (for multi-operation caching).
            
        Returns:
            True if file was already processed with same content.
        """
        file_hash = self.compute_hash(file_path)
        cache_key = f"{file_hash}:{operation}"
        
        return cache_key in self._cache
    
    def mark_processed(
        self,
        file_path: Path,
        operation: str = "default",
        result: dict[str, Any] | None = None,
    ) -> None:
        """Mark a file as processed.
        
        Args:
            file_path: Processed file path.
            operation: Operation type.
            result: Optional result data to cache.
        """
        file_hash = self.compute_hash(file_path)
        cache_key = f"{file_hash}:{operation}"
        
        self._cache[cache_key] = {
            "path": str(file_path),
            "hash": file_hash,
            "operation": operation,
            "result": result,
        }
        
        self._save_cache()
    
    def get_cached_result(
        self,
        file_path: Path,
        operation: str = "default",
    ) -> dict[str, Any] | None:
        """Get cached result for a file.
        
        Args:
            file_path: File path to lookup.
            operation: Operation type.
            
        Returns:
            Cached result data or None if not found.
        """
        file_hash = self.compute_hash(file_path)
        cache_key = f"{file_hash}:{operation}"
        
        if cache_key in self._cache:
            return self._cache[cache_key].get("result")
        
        return None
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache = {}
        self._save_cache()
    
    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)
