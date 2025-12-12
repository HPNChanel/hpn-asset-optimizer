"""AI module for Asset Optimizer.

Provides AI-powered features:
- Privacy Shield (OCR + face detection)
- Smart Adaptive Variants (dark mode, aspect ratio conversion)
- Content-based hash caching
- Background Removal (RMBG-1.4 / U2Net)
- AI Upscaling (Real-ESRGAN)
"""

from aopt.ai.privacy import PrivacyShield, FaceBlur
from aopt.ai.variants import VariantGenerator, HashCache
from aopt.ai.smart_resize import SmartResize
from aopt.ai.models import ModelManager
from aopt.ai.rembg import BackgroundRemover
from aopt.ai.upscale import AIUpscaler

__all__ = [
    "PrivacyShield",
    "FaceBlur",
    "VariantGenerator",
    "HashCache",
    "SmartResize",
    "ModelManager",
    "BackgroundRemover",
    "AIUpscaler",
]
