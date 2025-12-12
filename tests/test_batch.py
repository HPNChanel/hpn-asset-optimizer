"""Tests for the batch processor module."""

from pathlib import Path

import pytest
from rich.console import Console

from aopt.core.batch import BatchProcessor


class TestBatchProcessor:
    """Tests for BatchProcessor class."""
    
    def test_process_directory(self, sample_directory: Path, temp_dir: Path):
        """Test processing a directory of images."""
        output_dir = temp_dir / "output"
        console = Console(quiet=True)
        processor = BatchProcessor(console=console, workers=2)
        
        results = processor.process_directory(
            sample_directory,
            output_dir,
            quality=70,
        )
        
        assert len(results) == 5
        assert all(r.success for r in results)
        assert output_dir.exists()
    
    def test_process_empty_directory(self, temp_dir: Path):
        """Test processing an empty directory."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()
        output_dir = temp_dir / "output"
        console = Console(quiet=True)
        processor = BatchProcessor(console=console)
        
        results = processor.process_directory(empty_dir, output_dir)
        
        assert len(results) == 0
    
    def test_worker_count(self):
        """Test worker count initialization."""
        console = Console(quiet=True)
        
        processor_default = BatchProcessor(console=console)
        assert processor_default.workers >= 1
        
        processor_custom = BatchProcessor(console=console, workers=4)
        assert processor_custom.workers == 4
    
    def test_process_with_format_conversion(
        self,
        sample_directory: Path,
        temp_dir: Path,
    ):
        """Test batch processing with format conversion."""
        from aopt.core.converter import ImageFormat
        
        output_dir = temp_dir / "webp_output"
        console = Console(quiet=True)
        processor = BatchProcessor(console=console, workers=2)
        
        results = processor.process_directory(
            sample_directory,
            output_dir,
            format=ImageFormat.WEBP,
        )
        
        assert len(results) == 5
        # Check that output files are WebP
        for result in results:
            if result.success:
                assert result.output_path.suffix == ".webp"
