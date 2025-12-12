"""Metadata stripping utilities."""

from pathlib import Path

from PIL import Image

from aopt.cli.dashboard import ProcessingResult
from aopt.utils.image_io import load_image, save_image, get_file_size


def strip_metadata(
    input_path: Path,
    output_path: Path | None = None,
) -> ProcessingResult:
    """Remove all metadata (EXIF, IPTC, XMP) from an image.
    
    Args:
        input_path: Input image path.
        output_path: Output path (defaults to overwriting input).
        
    Returns:
        ProcessingResult with operation details.
    """
    try:
        input_size = get_file_size(input_path)
        
        # Load image data only (no metadata)
        img = load_image(input_path)
        
        # Create a clean copy without metadata
        clean_img = Image.new(img.mode, img.size)
        clean_img.putdata(list(img.getdata()))
        
        # Determine output path
        out_path = output_path or input_path
        
        # Save without metadata
        output_size = save_image(clean_img, out_path)
        
        return ProcessingResult(
            input_path=input_path,
            output_path=out_path,
            input_size=input_size,
            output_size=output_size,
            success=True,
            message="Metadata stripped successfully",
        )
    except Exception as e:
        return ProcessingResult(
            input_path=input_path,
            output_path=output_path or input_path,
            input_size=0,
            output_size=0,
            success=False,
            message=str(e),
        )
