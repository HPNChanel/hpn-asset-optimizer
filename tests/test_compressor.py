"""Tests for the compressor module."""

from pathlib import Path

import pytest

from aopt.core.compressor import Compressor, CompressionMode


class TestCompressor:
    """Tests for Compressor class."""
    
    def test_compress_lossy(self, sample_image: Path, temp_dir: Path):
        """Test lossy compression."""
        compressor = Compressor()
        output = temp_dir / "compressed.jpg"
        
        result = compressor.compress(
            sample_image,
            quality=70,
            mode=CompressionMode.LOSSY,
            output=output,
        )
        
        assert result.success
        assert result.output_path == output
        assert output.exists()
    
    def test_compress_lossless(self, sample_png: Path, temp_dir: Path):
        """Test lossless compression."""
        compressor = Compressor()
        output = temp_dir / "compressed.png"
        
        result = compressor.compress(
            sample_png,
            mode=CompressionMode.LOSSLESS,
            output=output,
        )
        
        assert result.success
        assert output.exists()
    
    def test_compress_auto_output(self, sample_image: Path):
        """Test auto-generated output path."""
        compressor = Compressor()
        
        result = compressor.compress(sample_image, quality=80)
        
        assert result.success
        assert "_compressed" in result.output_path.stem
        
        # Cleanup
        if result.output_path.exists():
            result.output_path.unlink()
    
    def test_compress_nonexistent_file(self, temp_dir: Path):
        """Test handling of nonexistent file."""
        compressor = Compressor()
        
        result = compressor.compress(temp_dir / "nonexistent.jpg")
        
        assert not result.success
    
    def test_compress_quality_range(self, sample_image: Path, temp_dir: Path):
        """Test different quality levels."""
        compressor = Compressor()
        
        for quality in [30, 50, 70, 90]:
            output = temp_dir / f"q{quality}.jpg"
            result = compressor.compress(sample_image, quality=quality, output=output)
            assert result.success
            assert output.exists()
