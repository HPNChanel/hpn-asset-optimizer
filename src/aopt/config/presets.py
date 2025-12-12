"""Optimization presets management."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from aopt.cli.dashboard import ProcessingResult
from aopt.utils.image_io import load_image, save_image, get_file_size


class Preset(BaseModel):
    """Optimization preset configuration."""
    
    name: str
    description: str = ""
    format: str | None = None  # Target format (jpeg, png, webp, avif)
    quality: int = Field(default=85, ge=1, le=100)
    max_width: int | None = None
    max_height: int | None = None
    strip_metadata: bool = True
    lossless: bool = False


# Built-in presets
BUILTIN_PRESETS: dict[str, Preset] = {
    "web": Preset(
        name="web",
        description="Optimized for web - WebP with good compression",
        format="webp",
        quality=80,
        max_width=1920,
        strip_metadata=True,
    ),
    "thumbnail": Preset(
        name="thumbnail",
        description="Small thumbnails for galleries",
        format="jpeg",
        quality=70,
        max_width=300,
        max_height=300,
        strip_metadata=True,
    ),
    "print": Preset(
        name="print",
        description="High quality for printing",
        format="png",
        quality=100,
        strip_metadata=False,
        lossless=True,
    ),
    "social": Preset(
        name="social",
        description="Optimized for social media sharing",
        format="jpeg",
        quality=85,
        max_width=1200,
        strip_metadata=True,
    ),
    "email": Preset(
        name="email",
        description="Lightweight for email attachments",
        format="jpeg",
        quality=60,
        max_width=800,
        max_height=800,
        strip_metadata=True,
    ),
    "avatar": Preset(
        name="avatar",
        description="Square avatars/profile pictures",
        format="webp",
        quality=85,
        max_width=256,
        max_height=256,
        strip_metadata=True,
    ),
    "hd": Preset(
        name="hd",
        description="HD quality with balanced compression",
        format="webp",
        quality=90,
        max_width=1920,
        max_height=1080,
        strip_metadata=True,
    ),
}


class PresetManager:
    """Manage optimization presets."""
    
    def __init__(self, presets_dir: Path | None = None) -> None:
        """Initialize preset manager.
        
        Args:
            presets_dir: Directory for custom preset YAML files.
        """
        self.presets_dir = presets_dir or Path("presets")
        self._custom_presets: dict[str, Preset] = {}
        self._load_custom_presets()
    
    def _load_custom_presets(self) -> None:
        """Load custom presets from YAML files."""
        if not self.presets_dir.exists():
            return
        
        for yaml_file in self.presets_dir.glob("*.yaml"):
            try:
                with open(yaml_file) as f:
                    data = yaml.safe_load(f)
                if data:
                    preset = Preset(**data)
                    self._custom_presets[preset.name] = preset
            except Exception:
                pass  # Skip invalid preset files
    
    def list_presets(self) -> list[dict[str, Any]]:
        """List all available presets."""
        presets = []
        
        # Built-in presets
        for name, preset in BUILTIN_PRESETS.items():
            presets.append({
                "name": name,
                "description": preset.description,
                "format": preset.format or "auto",
                "quality": preset.quality,
                "max_size": f"{preset.max_width}×{preset.max_height}" 
                    if preset.max_width and preset.max_height 
                    else (f"{preset.max_width}w" if preset.max_width else "-"),
                "builtin": True,
            })
        
        # Custom presets
        for name, preset in self._custom_presets.items():
            presets.append({
                "name": name,
                "description": preset.description,
                "format": preset.format or "auto",
                "quality": preset.quality,
                "max_size": f"{preset.max_width}×{preset.max_height}"
                    if preset.max_width and preset.max_height
                    else (f"{preset.max_width}w" if preset.max_width else "-"),
                "builtin": False,
            })
        
        return presets
    
    def get_preset(self, name: str) -> Preset | None:
        """Get a preset by name."""
        # Check custom first (allows overriding built-ins)
        if name in self._custom_presets:
            return self._custom_presets[name]
        return BUILTIN_PRESETS.get(name)
    
    def apply_preset(
        self,
        preset: Preset,
        input_path: Path,
        output: Path | None = None,
    ) -> ProcessingResult:
        """Apply a preset to an image.
        
        Args:
            preset: Preset configuration.
            input_path: Input image path.
            output: Output path.
            
        Returns:
            ProcessingResult with processing details.
        """
        from aopt.core.processor import ImageProcessor
        from aopt.core.converter import ImageFormat
        
        processor = ImageProcessor()
        
        # Convert format string to enum if present
        target_format = None
        if preset.format:
            try:
                target_format = ImageFormat(preset.format)
            except ValueError:
                pass
        
        return processor.optimize(
            input_path,
            output=output,
            quality=preset.quality,
            format=target_format,
            max_width=preset.max_width,
            max_height=preset.max_height,
            strip_metadata=preset.strip_metadata,
        )
    
    def save_preset(self, preset: Preset) -> None:
        """Save a preset to YAML file.
        
        Args:
            preset: Preset to save.
        """
        self.presets_dir.mkdir(parents=True, exist_ok=True)
        
        yaml_path = self.presets_dir / f"{preset.name}.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(preset.model_dump(), f, default_flow_style=False)
        
        self._custom_presets[preset.name] = preset
