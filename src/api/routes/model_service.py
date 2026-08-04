"""Inference service that keeps PyTorch details out of HTTP route handlers."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from src.models.neural_network import PoModel


class ModelUnavailableError(RuntimeError):
    """Raised when a request arrives before a model is ready."""


class UnsupportedImageError(ValueError):
    """Raised when supplied bytes cannot produce a usable digit image."""


class ModelService:
    """Load once and serve the project's MNIST PyTorch classifier.

    Supports a serialized module, a plain ``state_dict``, or a mapping with a
    ``state_dict`` key, so the training checkpoint format can evolve safely.
    """

    def __init__(self, model_path: Path, model_version: str = "1.0.0") -> None:
        self.model_path = model_path
        self.model_version = model_version
        self._model: torch.nn.Module | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_error: str | None = None

    @property
    def is_ready(self) -> bool:
        """Whether a loaded model is currently able to serve predictions."""
        return self._model is not None

    def load(self) -> None:
        """Load the configured artifact without taking down liveness on failure."""
        self._model = None
        self.load_error = None
        try:
            if not self.model_path.is_file():
                raise FileNotFoundError(
                    f"Model artifact was not found: {self.model_path}"
                )
            checkpoint: Any = torch.load(
                self.model_path, map_location=self._device, weights_only=False
            )
            model = self._build_model(checkpoint)
            model.to(self._device)
            model.eval()
            self._model = model
        except Exception as exc:
            self.load_error = str(exc)

    def predict(self, image_bytes: bytes) -> tuple[int, float]:
        """Preprocess bytes into MNIST shape and return class plus confidence."""
        if self._model is None:
            raise ModelUnavailableError(self.load_error or "Model has not been loaded.")
        tensor = self._preprocess(image_bytes).to(self._device)
        with torch.inference_mode():
            probabilities = torch.softmax(self._model(tensor), dim=1)[0]
            confidence, prediction = torch.max(probabilities, dim=0)
        return int(prediction.item()), float(confidence.item())

    @staticmethod
    def _build_model(checkpoint: Any) -> torch.nn.Module:
        if isinstance(checkpoint, torch.nn.Module):
            return checkpoint
        state_dict = (
            checkpoint.get("state_dict", checkpoint)
            if isinstance(checkpoint, dict)
            else None
        )
        if not isinstance(state_dict, dict):
            raise ValueError("Unsupported model checkpoint format.")
        state_dict = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }
        model = PoModel()
        model.load_state_dict(state_dict)
        return model

    @staticmethod
    def _preprocess(image_bytes: bytes) -> torch.Tensor:
        """Convert an image to a normalized MNIST tensor ``[1, 1, 28, 28]``.

        Bright backgrounds are inverted to MNIST's bright-stroke convention;
        the glyph is aspect-ratio-preserving resized to at most 20px and
        centered on a 28px canvas.
        """
        try:
            with Image.open(BytesIO(image_bytes)) as source:
                image = ImageOps.grayscale(source)
                image.load()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise UnsupportedImageError("Uploaded file is not a valid image.") from exc
        if image.getextrema() == (0, 0):
            raise UnsupportedImageError("Image does not contain a visible digit.")
        if sum(image.getdata()) / (image.width * image.height) > 127:
            image = ImageOps.invert(image)
        bounding_box = image.getbbox()
        if bounding_box is None:
            raise UnsupportedImageError("Image does not contain a visible digit.")
        digit = image.crop(bounding_box)
        scale = min(20 / digit.width, 20 / digit.height)
        size = (max(1, round(digit.width * scale)), max(1, round(digit.height * scale)))
        digit = digit.resize(size, Image.Resampling.LANCZOS)
        canvas = Image.new("L", (28, 28), color=0)
        canvas.paste(digit, ((28 - digit.width) // 2, (28 - digit.height) // 2))
        values = torch.tensor(list(canvas.getdata()), dtype=torch.float32)
        return values.div(255.0).reshape(1, 1, 28, 28)
