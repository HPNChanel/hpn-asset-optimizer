"""ONNX model management for AI features."""

from pathlib import Path
from typing import Any

import numpy as np


class ModelManager:
    """ONNX Runtime model manager with lazy loading and caching.
    
    Provides a unified interface for loading and running ONNX models
    with support for CPU and GPU acceleration.
    """
    
    def __init__(self, models_dir: Path | None = None) -> None:
        """Initialize model manager.
        
        Args:
            models_dir: Directory containing ONNX models.
        """
        self.models_dir = models_dir or Path("models")
        self._sessions: dict[str, Any] = {}
        self._ort = None
    
    @property
    def ort(self) -> Any:
        """Lazy load ONNX Runtime."""
        if self._ort is None:
            try:
                import onnxruntime as ort
                self._ort = ort
            except ImportError:
                raise ImportError(
                    "ONNX Runtime is required for AI features. "
                    "Install with: poetry install --with ai"
                )
        return self._ort
    
    def load_model(self, model_name: str) -> Any:
        """Load an ONNX model.
        
        Args:
            model_name: Name of the model file (without .onnx extension).
            
        Returns:
            ONNX Runtime inference session.
        """
        if model_name in self._sessions:
            return self._sessions[model_name]
        
        model_path = self.models_dir / f"{model_name}.onnx"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Configure session options
        sess_options = self.ort.SessionOptions()
        sess_options.graph_optimization_level = (
            self.ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        
        # Try to use GPU providers if available
        providers = self._get_available_providers()
        
        session = self.ort.InferenceSession(
            str(model_path),
            sess_options,
            providers=providers,
        )
        
        self._sessions[model_name] = session
        return session
    
    def _get_available_providers(self) -> list[str]:
        """Get list of available execution providers ordered by priority."""
        providers = []
        available = self.ort.get_available_providers()
        
        # Prefer GPU providers
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        if "DmlExecutionProvider" in available:  # DirectML for Windows
            providers.append("DmlExecutionProvider")
        
        # Always include CPU as fallback
        providers.append("CPUExecutionProvider")
        
        return providers
    
    def run(
        self,
        model_name: str,
        input_data: np.ndarray,
        input_name: str | None = None,
    ) -> np.ndarray:
        """Run inference on a model.
        
        Args:
            model_name: Name of the model.
            input_data: Input numpy array.
            input_name: Name of the input tensor (auto-detected if None).
            
        Returns:
            Output numpy array.
        """
        session = self.load_model(model_name)
        
        # Get input name if not provided
        if input_name is None:
            input_name = session.get_inputs()[0].name
        
        # Run inference
        outputs = session.run(None, {input_name: input_data})
        
        return outputs[0]
    
    def list_models(self) -> list[str]:
        """List available models in the models directory."""
        if not self.models_dir.exists():
            return []
        
        return [
            model.stem
            for model in self.models_dir.glob("*.onnx")
        ]
    
    def unload_model(self, model_name: str) -> None:
        """Unload a model from memory."""
        if model_name in self._sessions:
            del self._sessions[model_name]
    
    def unload_all(self) -> None:
        """Unload all models from memory."""
        self._sessions.clear()
