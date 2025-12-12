"""AI module for Asset Optimizer.

Provides AI-powered features:
- Privacy Shield (OCR + face detection)
- Smart Adaptive Variants (dark mode, aspect ratio conversion)
- Content-based hash caching
"""

from aopt.ai.privacy import PrivacyShield, FaceBlur
from aopt.ai.variants import VariantGenerator, HashCache
from aopt.ai.smart_resize import SmartResize

__all__ = [
    "PrivacyShield",
    "FaceBlur",
    "VariantGenerator",
    "HashCache",
    "SmartResize",
]
