"""AI-powered image upscaling using Real-ESRGAN ONNX model."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from aopt.ai.models import ModelManager


@dataclass
class UpscaleResult:
    """Result from AI upscaling processing."""
    
    input_path: Path
    output_path: Path
    original_size: tuple[int, int]
    upscaled_size: tuple[int, int]
    scale_factor: int
    success: bool
    message: str = ""


class AIUpscaler:
    """AI-powered image upscaling using Real-ESRGAN.
    
    Supports 4x upscaling with automatic tiling for large images
    to prevent out-of-memory errors. Uses overlap blending for
    seamless tile stitching.
    
    Example:
        upscaler = AIUpscaler()
        result = upscaler.upscale("small.jpg", output="large.png")
    """
    
    # Tiling configuration
    TILE_SIZE = 512      # Size of each tile
    TILE_PAD = 32        # Overlap padding for seamless stitching
    SCALE = 4            # Real-ESRGAN scale factor
    
    # Memory threshold (approximate, in pixels)
    # Images larger than this will use tiling
    TILE_THRESHOLD = 1024 * 1024  # ~1MP
    
    def __init__(
        self,
        model_name: str = "realesrgan-x4",
        console: Console | None = None,
    ) -> None:
        """Initialize AI upscaler.
        
        Args:
            model_name: ONNX model name for upscaling.
            console: Rich console for output.
        """
        self.model_name = model_name
        self.console = console or Console()
        self.model_manager = ModelManager(console=self.console)
        self._session = None
    
    @property
    def session(self):
        """Lazy load the ONNX model."""
        if self._session is None:
            self._session = self.model_manager.load_model(self.model_name)
        return self._session
    
    def upscale(
        self,
        input_path: Path | str,
        output: Path | str | None = None,
        scale: int = 4,
    ) -> UpscaleResult:
        """Upscale an image using AI.
        
        Args:
            input_path: Path to input image.
            output: Path for output image. Defaults to input_4x.png.
            scale: Scale factor (currently only 4x supported).
            
        Returns:
            UpscaleResult with processing details.
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            return UpscaleResult(
                input_path=input_path,
                output_path=Path(""),
                original_size=(0, 0),
                upscaled_size=(0, 0),
                scale_factor=scale,
                success=False,
                message=f"Input file not found: {input_path}",
            )
        
        # Determine output path
        if output is None:
            output = input_path.parent / f"{input_path.stem}_{scale}x.png"
        else:
            output = Path(output)
        
        try:
            # Load original image
            original = Image.open(input_path).convert("RGB")
            original_size = original.size
            img_array = np.array(original)
            
            self.console.print(f"[dim]Input: {input_path.name} ({original_size[0]}x{original_size[1]})[/]")
            
            # Determine if tiling is needed
            num_pixels = original_size[0] * original_size[1]
            use_tiling = num_pixels > self.TILE_THRESHOLD
            
            if use_tiling:
                self.console.print(f"[dim]Using tiled upscaling (image > {self.TILE_THRESHOLD} pixels)[/]")
                upscaled = self._tile_upscale(img_array)
            else:
                self.console.print("[dim]Running direct upscaling...[/]")
                upscaled = self._upscale_tile(img_array)
            
            # Convert back to PIL and save
            result_image = Image.fromarray(upscaled)
            upscaled_size = result_image.size
            
            result_image.save(output, quality=95)
            
            self.console.print(
                f"[green]✓ Upscaled {scale}x: {output} "
                f"({upscaled_size[0]}x{upscaled_size[1]})[/]"
            )
            
            return UpscaleResult(
                input_path=input_path,
                output_path=output,
                original_size=original_size,
                upscaled_size=upscaled_size,
                scale_factor=scale,
                success=True,
                message="Image upscaled successfully",
            )
            
        except Exception as e:
            return UpscaleResult(
                input_path=input_path,
                output_path=output,
                original_size=(0, 0),
                upscaled_size=(0, 0),
                scale_factor=scale,
                success=False,
                message=f"Error: {e}",
            )
    
    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for Real-ESRGAN input.
        
        Args:
            img: Input image as HWC numpy array (0-255).
            
        Returns:
            Preprocessed NCHW tensor.
        """
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Convert BGR to RGB if needed (OpenCV loads as BGR)
        # PIL loads as RGB, so we don't need this
        
        # Convert HWC to NCHW
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        
        return img.astype(np.float32)
    
    def _postprocess(self, output: np.ndarray) -> np.ndarray:
        """Postprocess Real-ESRGAN output.
        
        Args:
            output: Model output as NCHW tensor.
            
        Returns:
            Image as HWC numpy array (0-255).
        """
        # Remove batch dimension
        if output.ndim == 4:
            output = output[0]
        
        # Convert NCHW to HWC
        output = np.transpose(output, (1, 2, 0))
        
        # Clip to [0, 1] and convert to uint8
        output = np.clip(output, 0, 1)
        output = (output * 255).astype(np.uint8)
        
        return output
    
    def _upscale_tile(self, tile: np.ndarray) -> np.ndarray:
        """Upscale a single tile through the model.
        
        Args:
            tile: Input tile as HWC numpy array.
            
        Returns:
            Upscaled tile as HWC numpy array.
        """
        # Preprocess
        input_tensor = self._preprocess(tile)
        
        # Get input name
        input_name = self.session.get_inputs()[0].name
        
        # Run inference
        output = self.session.run(None, {input_name: input_tensor})[0]
        
        # Postprocess
        return self._postprocess(output)
    
    def _tile_upscale(self, img: np.ndarray) -> np.ndarray:
        """Upscale large image using tiling with overlap blending.
        
        Args:
            img: Input image as HWC numpy array.
            
        Returns:
            Upscaled image as HWC numpy array.
        """
        h, w, c = img.shape
        tile_size = self.TILE_SIZE
        pad = self.TILE_PAD
        scale = self.SCALE
        
        # Calculate output dimensions
        out_h = h * scale
        out_w = w * scale
        
        # Create output array
        output = np.zeros((out_h, out_w, c), dtype=np.float32)
        # Weight accumulator for blending
        weight = np.zeros((out_h, out_w, 1), dtype=np.float32)
        
        # Calculate number of tiles
        tiles_h = max(1, (h + tile_size - 1) // tile_size)
        tiles_w = max(1, (w + tile_size - 1) // tile_size)
        total_tiles = tiles_h * tiles_w
        
        self.console.print(f"[dim]Processing {total_tiles} tiles ({tiles_w}x{tiles_h})[/]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console,
        ) as progress:
            task = progress.add_task("[cyan]Upscaling tiles", total=total_tiles)
            
            tile_idx = 0
            for tile_y in range(tiles_h):
                for tile_x in range(tiles_w):
                    # Calculate tile boundaries with padding
                    y1 = tile_y * tile_size
                    x1 = tile_x * tile_size
                    y2 = min(y1 + tile_size, h)
                    x2 = min(x1 + tile_size, w)
                    
                    # Add padding
                    y1_pad = max(0, y1 - pad)
                    x1_pad = max(0, x1 - pad)
                    y2_pad = min(h, y2 + pad)
                    x2_pad = min(w, x2 + pad)
                    
                    # Extract tile with padding
                    tile = img[y1_pad:y2_pad, x1_pad:x2_pad]
                    
                    # Upscale tile
                    upscaled_tile = self._upscale_tile(tile)
                    
                    # Calculate output positions
                    out_y1 = y1_pad * scale
                    out_x1 = x1_pad * scale
                    out_y2 = y2_pad * scale
                    out_x2 = x2_pad * scale
                    
                    # Create blending weight (linear falloff at edges)
                    tile_h, tile_w = upscaled_tile.shape[:2]
                    blend_weight = self._create_blend_weight(tile_h, tile_w, pad * scale)
                    
                    # Add to output with blending
                    output[out_y1:out_y2, out_x1:out_x2] += upscaled_tile.astype(np.float32) * blend_weight
                    weight[out_y1:out_y2, out_x1:out_x2] += blend_weight
                    
                    tile_idx += 1
                    progress.update(task, advance=1)
        
        # Normalize by weight
        weight = np.maximum(weight, 1e-8)
        output = output / weight
        
        return output.astype(np.uint8)
    
    def _create_blend_weight(self, h: int, w: int, pad: int) -> np.ndarray:
        """Create blending weight mask for tile overlap.
        
        Args:
            h: Tile height.
            w: Tile width.
            pad: Padding size.
            
        Returns:
            Weight mask with linear falloff at edges.
        """
        # Create 1D ramps
        ramp_y = np.ones(h, dtype=np.float32)
        ramp_x = np.ones(w, dtype=np.float32)
        
        if pad > 0:
            # Linear ramp from 0 to 1 at start
            ramp = np.linspace(0, 1, pad)
            ramp_y[:pad] = ramp
            ramp_y[-pad:] = ramp[::-1]
            ramp_x[:pad] = ramp
            ramp_x[-pad:] = ramp[::-1]
        
        # Create 2D weight
        weight = np.outer(ramp_y, ramp_x)
        weight = np.expand_dims(weight, axis=2)
        
        return weight
    
    def process(
        self,
        input_path: Path | str,
        output: Path | str | None = None,
        **kwargs,
    ) -> UpscaleResult:
        """Alias for upscale for consistency with other processors."""
        return self.upscale(input_path, output, **kwargs)


def upscale_image(
    input_path: Path | str,
    output: Path | str | None = None,
    scale: int = 4,
) -> UpscaleResult:
    """Convenience function for quick image upscaling.
    
    Args:
        input_path: Path to input image.
        output: Path for output image.
        scale: Scale factor (currently only 4x supported).
        
    Returns:
        UpscaleResult with processing details.
    """
    upscaler = AIUpscaler()
    return upscaler.upscale(input_path, output, scale=scale)
