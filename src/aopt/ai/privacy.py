"""Privacy Shield - OCR-based PII detection and face redaction.

Complete implementation with EasyOCR for text detection and OpenCV
for face detection using Haar Cascade classifier.
"""

import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from aopt.cli.dashboard import ShieldResult
from aopt.utils.image_io import load_image, save_image, get_file_size


# PII patterns for detection
PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "phone": re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"),
    "ssn": re.compile(r"\d{3}[-.\s]?\d{2}[-.\s]?\d{4}"),
    "credit_card": re.compile(r"\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}"),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


class PrivacyShield:
    """Privacy Shield for detecting and redacting PII in images.
    
    Uses EasyOCR for text detection and regex patterns for PII identification.
    Uses OpenCV Haar Cascade for face detection and blurring.
    Supports multiple redaction modes: blur, box, and pixelate.
    """
    
    def __init__(self, languages: list[str] | None = None) -> None:
        """Initialize Privacy Shield.
        
        Args:
            languages: Language codes for OCR (default: ["en"]).
        """
        self.languages = languages or ["en"]
        self._reader = None
        self._face_cascade = None
    
    @property
    def reader(self) -> Any:
        """Lazy load EasyOCR reader."""
        if self._reader is None:
            try:
                import easyocr
                self._reader = easyocr.Reader(self.languages, gpu=False)
            except ImportError:
                raise ImportError(
                    "EasyOCR is required for Privacy Shield. "
                    "Install with: poetry install --with ai"
                )
        return self._reader
    
    @property
    def face_cascade(self) -> Any:
        """Lazy load OpenCV Haar Cascade face detector."""
        if self._face_cascade is None:
            try:
                import cv2
                # Use OpenCV's pre-trained Haar Cascade for frontal face detection
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self._face_cascade = cv2.CascadeClassifier(cascade_path)
                
                if self._face_cascade.empty():
                    raise RuntimeError("Failed to load Haar Cascade classifier")
            except ImportError:
                raise ImportError(
                    "OpenCV is required for face detection. "
                    "Install with: pip install opencv-python"
                )
        return self._face_cascade
    
    def process(
        self,
        input_path: Path,
        output: Path | None = None,
        mode: str = "blur",
        patterns: list[str] | None = None,
        detect_faces: bool = True,
    ) -> ShieldResult:
        """Process an image to detect and redact PII.
        
        Args:
            input_path: Input image path.
            output: Output path.
            mode: Redaction mode (blur, box, pixelate).
            patterns: PII patterns to detect (default: all).
            detect_faces: Whether to detect and blur faces.
            
        Returns:
            ShieldResult with detection and redaction details.
        """
        try:
            img = load_image(input_path)
            
            # Determine output path
            if output is None:
                output = input_path.parent / f"{input_path.stem}_shielded{input_path.suffix}"
            
            # Run OCR for text detection
            img_np = np.array(img.convert("RGB"))
            detections = self.reader.readtext(img_np)
            
            # Filter for PII matches
            active_patterns = patterns or list(PII_PATTERNS.keys())
            pii_detections = []
            
            for detection in detections:
                bbox, text, confidence = detection
                matched_pattern = self._match_pii(text, active_patterns)
                if matched_pattern:
                    pii_detections.append({
                        "bbox": bbox,
                        "text": text,
                        "pattern": matched_pattern,
                        "confidence": confidence,
                        "type": "text_pii",
                    })
            
            # Detect faces
            face_detections = []
            if detect_faces:
                face_detections = self._detect_faces(img_np)
                for face_bbox in face_detections:
                    pii_detections.append({
                        "bbox": face_bbox,
                        "text": "[FACE]",
                        "pattern": "face",
                        "confidence": 1.0,
                        "type": "face",
                    })
            
            # Redact if any PII or faces found
            if pii_detections:
                redacted_img = self._redact(img, pii_detections, mode)
                save_image(redacted_img, output)
                
                return ShieldResult(
                    input_path=input_path,
                    output_path=output,
                    detections=pii_detections,
                    redacted_count=len(pii_detections),
                    success=True,
                )
            else:
                # No PII found, copy original
                save_image(img, output)
                
                return ShieldResult(
                    input_path=input_path,
                    output_path=output,
                    detections=[],
                    redacted_count=0,
                    success=True,
                )
        except Exception as e:
            return ShieldResult(
                input_path=input_path,
                output_path=output or input_path,
                detections=[],
                redacted_count=0,
                success=False,
            )
    
    def _detect_faces(self, img_np: np.ndarray) -> list[list[tuple[int, int]]]:
        """Detect faces in an image using Haar Cascade.
        
        Args:
            img_np: Image as numpy array (RGB).
            
        Returns:
            List of face bounding boxes as 4-point polygons.
        """
        try:
            import cv2
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            # Detect faces with tuned parameters
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Convert (x, y, w, h) to 4-point bounding box format
            face_bboxes = []
            for (x, y, w, h) in faces:
                # Return as 4-point polygon (same format as EasyOCR)
                bbox = [
                    (x, y),           # top-left
                    (x + w, y),       # top-right
                    (x + w, y + h),   # bottom-right
                    (x, y + h),       # bottom-left
                ]
                face_bboxes.append(bbox)
            
            return face_bboxes
        except Exception:
            return []
    
    def detect_faces_only(
        self,
        input_path: Path,
    ) -> list[dict[str, Any]]:
        """Detect faces in an image without redaction.
        
        Args:
            input_path: Input image path.
            
        Returns:
            List of detected face bounding boxes.
        """
        img = load_image(input_path)
        img_np = np.array(img.convert("RGB"))
        
        faces = self._detect_faces(img_np)
        
        return [
            {
                "bbox": bbox,
                "type": "face",
                "confidence": 1.0,
            }
            for bbox in faces
        ]
    
    def _match_pii(self, text: str, patterns: list[str]) -> str | None:
        """Check if text matches any PII pattern."""
        for pattern_name in patterns:
            if pattern_name in PII_PATTERNS:
                if PII_PATTERNS[pattern_name].search(text):
                    return pattern_name
        return None
    
    def _redact(
        self,
        img: Image.Image,
        detections: list[dict[str, Any]],
        mode: str,
    ) -> Image.Image:
        """Apply redaction to detected PII regions."""
        img = img.copy()
        
        for detection in detections:
            bbox = detection["bbox"]
            # Convert bbox to (x1, y1, x2, y2)
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            x1, y1 = min(x_coords), min(y_coords)
            x2, y2 = max(x_coords), max(y_coords)
            
            # Add padding
            padding = 5
            x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
            x2, y2 = min(img.width, x2 + padding), min(img.height, y2 + padding)
            
            if mode == "blur":
                img = self._apply_blur(img, (x1, y1, x2, y2))
            elif mode == "box":
                img = self._apply_box(img, (x1, y1, x2, y2))
            elif mode == "pixelate":
                img = self._apply_pixelate(img, (x1, y1, x2, y2))
        
        return img
    
    def _apply_blur(
        self,
        img: Image.Image,
        bbox: tuple[int, int, int, int],
    ) -> Image.Image:
        """Apply blur redaction."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        region = img.crop((x1, y1, x2, y2))
        blurred = region.filter(ImageFilter.GaussianBlur(radius=15))
        img.paste(blurred, (x1, y1))
        return img
    
    def _apply_box(
        self,
        img: Image.Image,
        bbox: tuple[int, int, int, int],
    ) -> Image.Image:
        """Apply black box redaction."""
        draw = ImageDraw.Draw(img)
        draw.rectangle(bbox, fill="black")
        return img
    
    def _apply_pixelate(
        self,
        img: Image.Image,
        bbox: tuple[int, int, int, int],
    ) -> Image.Image:
        """Apply pixelation redaction."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        region = img.crop((x1, y1, x2, y2))
        
        # Downscale and upscale to pixelate
        small_size = (max(1, (x2 - x1) // 10), max(1, (y2 - y1) // 10))
        pixelated = region.resize(small_size, Image.Resampling.NEAREST)
        pixelated = pixelated.resize(region.size, Image.Resampling.NEAREST)
        
        img.paste(pixelated, (x1, y1))
        return img
    
    def detect_only(
        self,
        input_path: Path,
        patterns: list[str] | None = None,
        detect_faces: bool = True,
    ) -> list[dict[str, Any]]:
        """Detect PII and faces without redacting.
        
        Args:
            input_path: Input image path.
            patterns: PII patterns to check.
            detect_faces: Whether to detect faces.
            
        Returns:
            List of detected PII and face items.
        """
        img = load_image(input_path)
        img_np = np.array(img.convert("RGB"))
        detections = self.reader.readtext(img_np)
        
        active_patterns = patterns or list(PII_PATTERNS.keys())
        pii_detections = []
        
        # Text PII
        for detection in detections:
            bbox, text, confidence = detection
            matched_pattern = self._match_pii(text, active_patterns)
            if matched_pattern:
                pii_detections.append({
                    "bbox": bbox,
                    "text": text,
                    "pattern": matched_pattern,
                    "confidence": confidence,
                    "type": "text_pii",
                })
        
        # Faces
        if detect_faces:
            for bbox in self._detect_faces(img_np):
                pii_detections.append({
                    "bbox": bbox,
                    "text": "[FACE]",
                    "pattern": "face",
                    "confidence": 1.0,
                    "type": "face",
                })
        
        return pii_detections


class FaceBlur:
    """Standalone face detection and blurring.
    
    Uses OpenCV Haar Cascade for detection and PIL for blurring.
    Provides a simpler interface for face-only operations.
    """
    
    def __init__(self) -> None:
        """Initialize FaceBlur."""
        self._cascade = None
    
    @property
    def cascade(self) -> Any:
        """Lazy load Haar Cascade."""
        if self._cascade is None:
            try:
                import cv2
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self._cascade = cv2.CascadeClassifier(cascade_path)
            except ImportError:
                raise ImportError(
                    "OpenCV is required. Install with: pip install opencv-python"
                )
        return self._cascade
    
    def process(
        self,
        input_path: Path,
        output: Path | None = None,
        blur_radius: int = 20,
    ) -> ShieldResult:
        """Detect and blur all faces in an image.
        
        Args:
            input_path: Input image path.
            output: Output path.
            blur_radius: Gaussian blur radius.
            
        Returns:
            ShieldResult with face detection details.
        """
        try:
            import cv2
            
            img = load_image(input_path)
            img_np = np.array(img.convert("RGB"))
            
            if output is None:
                output = input_path.parent / f"{input_path.stem}_faces_blurred{input_path.suffix}"
            
            # Detect faces
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            faces = self.cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Blur each face
            result = img.copy()
            detections = []
            
            for (x, y, w, h) in faces:
                # Extract face region
                region = result.crop((x, y, x + w, y + h))
                blurred = region.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                result.paste(blurred, (x, y))
                
                detections.append({
                    "bbox": [(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
                    "type": "face",
                })
            
            save_image(result, output)
            
            return ShieldResult(
                input_path=input_path,
                output_path=output,
                detections=detections,
                redacted_count=len(detections),
                success=True,
            )
        except Exception as e:
            return ShieldResult(
                input_path=input_path,
                output_path=output or input_path,
                detections=[],
                redacted_count=0,
                success=False,
            )
