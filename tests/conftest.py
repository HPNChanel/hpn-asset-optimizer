"""Pytest configuration and fixtures."""

import tempfile
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image(temp_dir: Path) -> Path:
    """Create a sample test image."""
    img = Image.new("RGB", (100, 100), color="red")
    path = temp_dir / "sample.jpg"
    img.save(path, quality=95)
    return path


@pytest.fixture
def sample_png(temp_dir: Path) -> Path:
    """Create a sample PNG with transparency."""
    img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
    path = temp_dir / "sample.png"
    img.save(path)
    return path


@pytest.fixture
def sample_directory(temp_dir: Path) -> Path:
    """Create a directory with multiple test images."""
    for i in range(5):
        img = Image.new("RGB", (100, 100), color=(i * 50, 0, 0))
        path = temp_dir / f"image_{i}.jpg"
        img.save(path, quality=95)
    return temp_dir
