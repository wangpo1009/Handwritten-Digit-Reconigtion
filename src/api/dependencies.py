"""Lifecycle and dependency-injection primitives for the inference API."""

from __future__ import annotations

import os
from pathlib import Path

from src.api.routes.model_service import (
    ModelService,
    ModelUnavailableError,
    UnsupportedImageError,
)

# Deployment may set MNIST_MODEL_PATH without requiring a source-code change.
_project_root = Path(__file__).resolve().parents[2]
_model_path = Path(
    os.getenv("MNIST_MODEL_PATH", _project_root / "models" / "mnist_cnn.pt")
)
_model_service = ModelService(
    _model_path,
    model_version=os.getenv("MNIST_MODEL_VERSION", "1.0.0"),
)


def load_model() -> ModelService:
    """Run at application startup and return the singleton model service."""
    _model_service.load()
    return _model_service


def get_model_service() -> ModelService:
    """FastAPI injection seam; tests can override this dependency cleanly."""
    return _model_service


__all__ = [
    "ModelService",
    "ModelUnavailableError",
    "UnsupportedImageError",
    "get_model_service",
    "load_model",
]
