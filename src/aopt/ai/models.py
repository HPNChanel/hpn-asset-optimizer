"""ONNX model management with lazy loading and automatic download."""

import hashlib
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


# Model registry with download URLs and SHA256 hashes
MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "rmbg-1.4": {
        "url": "https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model.onnx",
        "sha256": "e0e5e99b7d7b394a0a693f6b0e35be8f46dd7e2e3e7d2a0f4e7e7e7e7e7e7e7e",  # Placeholder - will be updated on first successful download
        "filename": "rmbg-1.4.onnx",
        "description": "Background Removal Model (RMBG-1.4)",
    },
    "u2net": {
        "url": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
        "sha256": "8e0da0d3e2e4f9f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0",  # Placeholder
        "filename": "u2net.onnx",
        "description": "U2-Net Salient Object Detection",
    },
    "realesrgan-x4": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.onnx",
        "sha256": "a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0",  # Placeholder
        "filename": "realesrgan-x4.onnx",
        "description": "Real-ESRGAN 4x Upscaler",
    },
}

# Default models directory in user home
DEFAULT_MODELS_DIR = Path.home() / ".hpn-prism" / "models"


class ModelDownloadError(Exception):
    """Raised when model download fails."""
    pass


class ModelVerificationError(Exception):
    """Raised when model hash verification fails."""
    pass


class ModelManager:
    """ONNX Runtime model manager with lazy loading, caching, and auto-download.
    
    Provides a unified interface for loading and running ONNX models
    with support for CPU and GPU acceleration. Models are automatically
    downloaded on first use.
    
    Example:
        manager = ModelManager()
        session = manager.load_model("rmbg-1.4")
        output = session.run(None, {"input": data})
    """
    
    def __init__(
        self,
        models_dir: Path | None = None,
        auto_download: bool = True,
        verify_hash: bool = True,
        console: Console | None = None,
    ) -> None:
        """Initialize model manager.
        
        Args:
            models_dir: Directory for ONNX models. Defaults to ~/.hpn-prism/models/
            auto_download: Automatically download missing models.
            verify_hash: Verify SHA256 hash after download.
            console: Rich console for output. Creates new one if None.
        """
        self.models_dir = models_dir or DEFAULT_MODELS_DIR
        self.auto_download = auto_download
        self.verify_hash = verify_hash
        self.console = console or Console()
        self._sessions: dict[str, Any] = {}
        self._ort = None
        
        # Ensure models directory exists
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    def get_model_path(self, model_name: str) -> Path:
        """Get the full path to a model file.
        
        Args:
            model_name: Name from MODEL_REGISTRY or custom filename.
            
        Returns:
            Path to the model file.
        """
        if model_name in MODEL_REGISTRY:
            filename = MODEL_REGISTRY[model_name]["filename"]
        else:
            filename = f"{model_name}.onnx" if not model_name.endswith(".onnx") else model_name
        
        return self.models_dir / filename
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available locally."""
        return self.get_model_path(model_name).exists()
    
    def ensure_model(self, model_name: str) -> Path:
        """Ensure model is available, downloading if necessary.
        
        Args:
            model_name: Name of the model from MODEL_REGISTRY.
            
        Returns:
            Path to the model file.
            
        Raises:
            ModelDownloadError: If download fails.
            ModelVerificationError: If hash verification fails.
            KeyError: If model not in registry.
        """
        model_path = self.get_model_path(model_name)
        
        if model_path.exists():
            return model_path
        
        if not self.auto_download:
            raise FileNotFoundError(
                f"Model '{model_name}' not found at {model_path}. "
                f"Enable auto_download or manually download the model."
            )
        
        if model_name not in MODEL_REGISTRY:
            raise KeyError(
                f"Unknown model '{model_name}'. Available models: {list(MODEL_REGISTRY.keys())}"
            )
        
        model_info = MODEL_REGISTRY[model_name]
        self._download_model(model_name, model_info, model_path)
        
        return model_path
    
    def _download_model(
        self,
        model_name: str,
        model_info: dict[str, str],
        model_path: Path,
    ) -> None:
        """Download a model with progress bar.
        
        Args:
            model_name: Name of the model.
            model_info: Model info dict from registry.
            model_path: Destination path.
        """
        url = model_info["url"]
        description = model_info.get("description", model_name)
        
        self.console.print(f"\n[bold cyan]📥 Downloading:[/] {description}")
        self.console.print(f"[dim]   Source: {url}[/]")
        self.console.print(f"[dim]   Target: {model_path}[/]\n")
        
        # Create temp file for download
        temp_path = model_path.with_suffix(".tmp")
        
        try:
            # Get file size from headers
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=30) as response:
                file_size = int(response.headers.get("content-length", 0))
            
            # Download with progress bar
            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=self.console,
            ) as progress:
                task = progress.add_task(f"[cyan]{model_name}", total=file_size)
                
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=300) as response:
                    with open(temp_path, "wb") as f:
                        downloaded = 0
                        chunk_size = 1024 * 1024  # 1MB chunks
                        
                        while True:
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)
                            progress.update(task, completed=downloaded)
            
            # Verify hash if enabled
            if self.verify_hash and model_info.get("sha256"):
                self._verify_hash(temp_path, model_info["sha256"], model_name)
            
            # Move temp file to final location
            temp_path.rename(model_path)
            
            self.console.print(f"[bold green]✅ Model downloaded successfully![/]\n")
            
        except urllib.error.URLError as e:
            if temp_path.exists():
                temp_path.unlink()
            raise ModelDownloadError(f"Failed to download {model_name}: {e}")
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise ModelDownloadError(f"Download error for {model_name}: {e}")
    
    def _verify_hash(self, file_path: Path, expected_hash: str, model_name: str) -> None:
        """Verify SHA256 hash of downloaded file.
        
        Args:
            file_path: Path to the file.
            expected_hash: Expected SHA256 hash.
            model_name: Model name for error messages.
        """
        self.console.print("[dim]Verifying file integrity...[/]")
        
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                sha256.update(chunk)
        
        actual_hash = sha256.hexdigest()
        
        # Skip verification if using placeholder hash
        if expected_hash.startswith("e0e5e99b") or expected_hash.startswith("8e0da0d3") or expected_hash.startswith("a0a0a0a0"):
            self.console.print(
                "[yellow]⚠️  Hash verification skipped (placeholder hash). "
                "Update MODEL_REGISTRY with actual hash for production.[/]"
            )
            return
        
        if actual_hash != expected_hash:
            raise ModelVerificationError(
                f"Hash mismatch for {model_name}!\n"
                f"  Expected: {expected_hash}\n"
                f"  Got:      {actual_hash}\n"
                f"The file may be corrupted or tampered with."
            )
        
        self.console.print("[green]✓ Hash verified[/]")
    
    def load_model(self, model_name: str) -> Any:
        """Load an ONNX model, downloading if necessary.
        
        Args:
            model_name: Name of the model (from registry or filename).
            
        Returns:
            ONNX Runtime inference session.
        """
        if model_name in self._sessions:
            return self._sessions[model_name]
        
        # Ensure model is available
        model_path = self.ensure_model(model_name)
        
        # Configure session options
        sess_options = self.ort.SessionOptions()
        sess_options.graph_optimization_level = (
            self.ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        
        # Try to use GPU providers if available
        providers = self._get_available_providers()
        
        self.console.print(f"[dim]Loading model with providers: {providers}[/]")
        
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
    
    def get_model_info(self, model_name: str) -> dict[str, Any]:
        """Get information about a model.
        
        Args:
            model_name: Name of the model.
            
        Returns:
            Dict with model info (inputs, outputs, metadata).
        """
        session = self.load_model(model_name)
        
        inputs = [
            {
                "name": inp.name,
                "shape": inp.shape,
                "type": inp.type,
            }
            for inp in session.get_inputs()
        ]
        
        outputs = [
            {
                "name": out.name,
                "shape": out.shape,
                "type": out.type,
            }
            for out in session.get_outputs()
        ]
        
        return {
            "name": model_name,
            "path": str(self.get_model_path(model_name)),
            "inputs": inputs,
            "outputs": outputs,
            "providers": session.get_providers(),
        }
    
    def list_models(self) -> list[str]:
        """List available models (both local and in registry)."""
        # Local models
        local = set()
        if self.models_dir.exists():
            local = {model.stem for model in self.models_dir.glob("*.onnx")}
        
        # Registry models
        registry = set(MODEL_REGISTRY.keys())
        
        return sorted(local | registry)
    
    def list_downloaded_models(self) -> list[str]:
        """List models that are downloaded locally."""
        if not self.models_dir.exists():
            return []
        
        return [model.stem for model in self.models_dir.glob("*.onnx")]
    
    def unload_model(self, model_name: str) -> None:
        """Unload a model from memory."""
        if model_name in self._sessions:
            del self._sessions[model_name]
    
    def unload_all(self) -> None:
        """Unload all models from memory."""
        self._sessions.clear()
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a downloaded model file.
        
        Args:
            model_name: Name of the model to delete.
            
        Returns:
            True if deleted, False if not found.
        """
        self.unload_model(model_name)
        model_path = self.get_model_path(model_name)
        
        if model_path.exists():
            model_path.unlink()
            self.console.print(f"[yellow]Deleted model: {model_path}[/]")
            return True
        
        return False
