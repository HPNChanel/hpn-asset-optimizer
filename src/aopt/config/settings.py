"""Application settings and configuration."""

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # General settings
    app_name: str = "aopt"
    version: str = "0.1.0"
    
    # Processing defaults
    default_quality: int = Field(default=85, ge=1, le=100)
    default_workers: int = Field(default=4, ge=1, le=32)
    
    # Paths
    presets_dir: Path = Field(default=Path("presets"))
    models_dir: Path = Field(default=Path("models"))
    
    # Output settings
    output_suffix: str = "_optimized"
    preserve_structure: bool = True
    
    # AI features
    enable_ocr: bool = True
    ocr_languages: list[str] = Field(default_factory=lambda: ["en"])
    
    # Performance
    max_memory_mb: int = Field(default=2048, ge=256)
    
    class Config:
        env_prefix = "AOPT_"
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
