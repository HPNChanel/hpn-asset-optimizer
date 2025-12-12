"""Background removal using ONNX models (RMBG-1.4 / U2Net)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image
from rich.console import Console

from aopt.ai.models import ModelManager


@dataclass
class RemBgResult:
    """Result from background removal processing."""
    
    input_path: Path
    output_path: Path
    original_size: tuple[int, int]
    success: bool
    message: str = ""


class BackgroundRemover:
    """AI-powered background removal using ONNX models.
    
    Supports RMBG-1.4 (recommended) and U2Net models. Images are
    automatically preprocessed to the required size and the resulting
    alpha mask is applied to create transparent PNGs.
    
    Example:
        remover = BackgroundRemover()
        result = remover.remove_background("photo.jpg", output="photo_nobg.png")
    """
    
    # Model input specifications
    MODEL_CONFIGS = {
        "rmbg-1.4": {
            "input_size": (1024, 1024),
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
        },
        "u2net": {
            "input_size": (320, 320),
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
        },
    }
    
    def __init__(
        self,
        model_name: Literal["rmbg-1.4", "u2net"] = "rmbg-1.4",
        console: Console | None = None,
    ) -> None:
        """Initialize background remover.
        
        Args:
            model_name: Model to use ("rmbg-1.4" or "u2net").
            console: Rich console for output.
        """
        self.model_name = model_name
        self.console = console or Console()
        self.model_manager = ModelManager(console=self.console)
        self._session = None
        
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Use 'rmbg-1.4' or 'u2net'.")
        
        self.config = self.MODEL_CONFIGS[model_name]
    
    @property
    def session(self):
        """Lazy load the ONNX model."""
        if self._session is None:
            self._session = self.model_manager.load_model(self.model_name)
        return self._session
    
    def remove_background(
        self,
        input_path: Path | str,
        output: Path | str | None = None,
        threshold: float = 0.5,
    ) -> RemBgResult:
        """Remove background from an image.
        
        Args:
            input_path: Path to input image.
            output: Path for output PNG. Defaults to input_nobg.png.
            threshold: Alpha threshold (0-1) for mask refinement.
            
        Returns:
            RemBgResult with processing details.
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            return RemBgResult(
                input_path=input_path,
                output_path=Path(""),
                original_size=(0, 0),
                success=False,
                message=f"Input file not found: {input_path}",
            )
        
        # Determine output path
        if output is None:
            output = input_path.parent / f"{input_path.stem}_nobg.png"
        else:
            output = Path(output)
        
        # Ensure output is PNG (transparency)
        if output.suffix.lower() != ".png":
            output = output.with_suffix(".png")
        
        try:
            # Load original image
            original = Image.open(input_path).convert("RGB")
            original_size = original.size
            
            self.console.print(f"[dim]Processing {input_path.name} ({original_size[0]}x{original_size[1]})[/]")
            
            # Preprocess
            input_tensor = self._preprocess(original)
            
            # Run inference
            self.console.print("[dim]Running background removal model...[/]")
            mask = self._run_inference(input_tensor)
            
            # Postprocess
            result_image = self._postprocess(original, mask, threshold)
            
            # Save result
            result_image.save(output, "PNG")
            
            self.console.print(f"[green]✓ Background removed: {output}[/]")
            
            return RemBgResult(
                input_path=input_path,
                output_path=output,
                original_size=original_size,
                success=True,
                message="Background removed successfully",
            )
            
        except Exception as e:
            return RemBgResult(
                input_path=input_path,
                output_path=output,
                original_size=(0, 0),
                success=False,
                message=f"Error: {e}",
            )
    
    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for model input.
        
        Args:
            image: PIL Image in RGB mode.
            
        Returns:
            Preprocessed numpy array in NCHW format.
        """
        input_size = self.config["input_size"]
        
        # Resize to model input size
        resized = image.resize(input_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy and normalize to [0, 1]
        img_array = np.array(resized).astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = np.array(self.config["normalize_mean"])
        std = np.array(self.config["normalize_std"])
        img_array = (img_array - mean) / std
        
        # Convert to NCHW format (batch, channels, height, width)
        img_array = img_array.transpose(2, 0, 1)
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
        
        return img_array
    
    def _run_inference(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run model inference to get alpha mask.
        
        Args:
            input_tensor: Preprocessed input in NCHW format.
            
        Returns:
            Alpha mask as 2D numpy array.
        """
        # Get input name
        input_name = self.session.get_inputs()[0].name
        
        # Run inference
        outputs = self.session.run(None, {input_name: input_tensor})
        
        # Extract mask from output (model-dependent)
        mask = outputs[0]
        
        # Handle different output formats
        if mask.ndim == 4:  # NCHW
            mask = mask[0, 0]  # Take first batch, first channel
        elif mask.ndim == 3:  # CHW or HWC
            if mask.shape[0] in [1, 3]:  # CHW
                mask = mask[0]
            else:  # HWC
                mask = mask[:, :, 0]
        
        # Normalize to [0, 1]
        mask = mask.astype(np.float32)
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        
        return mask
    
    def _postprocess(
        self,
        original: Image.Image,
        mask: np.ndarray,
        threshold: float,
    ) -> Image.Image:
        """Apply mask to original image.
        
        Args:
            original: Original PIL Image.
            mask: Alpha mask from model.
            threshold: Threshold for binary mask refinement.
            
        Returns:
            RGBA image with transparent background.
        """
        # Resize mask to original size
        mask_resized = cv2.resize(
            mask,
            (original.width, original.height),
            interpolation=cv2.INTER_LINEAR,
        )
        
        # Apply soft thresholding for cleaner edges
        mask_resized = np.clip(mask_resized, 0, 1)
        
        # Optional: apply threshold for harder edges
        if threshold > 0:
            mask_resized = np.where(mask_resized > threshold, 1.0, mask_resized)
        
        # Convert to 8-bit alpha channel
        alpha = (mask_resized * 255).astype(np.uint8)
        
        # Convert original to RGBA
        result = original.convert("RGBA")
        
        # Apply alpha channel
        result.putalpha(Image.fromarray(alpha))
        
        return result
    
    def process(
        self,
        input_path: Path | str,
        output: Path | str | None = None,
        **kwargs,
    ) -> RemBgResult:
        """Alias for remove_background for consistency with other processors."""
        return self.remove_background(input_path, output, **kwargs)


def remove_background(
    input_path: Path | str,
    output: Path | str | None = None,
    model: str = "rmbg-1.4",
) -> RemBgResult:
    """Convenience function for quick background removal.
    
    Args:
        input_path: Path to input image.
        output: Path for output PNG.
        model: Model name ("rmbg-1.4" or "u2net").
        
    Returns:
        RemBgResult with processing details.
    """
    remover = BackgroundRemover(model_name=model)
    return remover.remove_background(input_path, output)
