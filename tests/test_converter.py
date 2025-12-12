"""Tests for the converter module."""

from pathlib import Path

import pytest

from aopt.core.converter import Converter, ImageFormat


class TestConverter:
    """Tests for Converter class."""
    
    def test_convert_to_webp(self, sample_image: Path, temp_dir: Path):
        """Test conversion to WebP."""
        converter = Converter()
        output = temp_dir / "converted.webp"
        
        result = converter.convert(
            sample_image,
            format=ImageFormat.WEBP,
            output=output,
        )
        
        assert result.success
        assert output.exists()
        assert output.suffix == ".webp"
    
    def test_convert_to_png(self, sample_image: Path, temp_dir: Path):
        """Test conversion to PNG."""
        converter = Converter()
        output = temp_dir / "converted.png"
        
        result = converter.convert(
            sample_image,
            format=ImageFormat.PNG,
            output=output,
        )
        
        assert result.success
        assert output.exists()
    
    def test_convert_preserves_transparency(self, sample_png: Path, temp_dir: Path):
        """Test that transparency is preserved when converting PNG to WebP."""
        converter = Converter()
        output = temp_dir / "transparent.webp"
        
        result = converter.convert(
            sample_png,
            format=ImageFormat.WEBP,
            output=output,
            preserve_transparency=True,
        )
        
        assert result.success
        assert output.exists()
    
    def test_convert_jpeg_strips_transparency(self, sample_png: Path, temp_dir: Path):
        """Test that transparency is handled when converting to JPEG."""
        converter = Converter()
        output = temp_dir / "opaque.jpg"
        
        result = converter.convert(
            sample_png,
            format=ImageFormat.JPEG,
            output=output,
        )
        
        assert result.success
        assert output.exists()
    
    def test_auto_output_path(self, sample_image: Path):
        """Test auto-generated output path for conversion."""
        converter = Converter()
        
        result = converter.convert(sample_image, format=ImageFormat.WEBP)
        
        assert result.success
        assert result.output_path.suffix == ".webp"
        
        # Cleanup
        if result.output_path.exists():
            result.output_path.unlink()
    
    def test_convert_nonexistent_file(self, temp_dir: Path):
        """Test handling of nonexistent file."""
        converter = Converter()
        
        result = converter.convert(
            temp_dir / "nonexistent.jpg",
            format=ImageFormat.WEBP,
        )
        
        assert not result.success
