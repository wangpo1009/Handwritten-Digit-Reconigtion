"""HTTP route for recognizing one handwritten digit."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from src.api.dependencies import (
    ModelService,
    ModelUnavailableError,
    UnsupportedImageError,
    get_model_service,
)
from src.api.schemas import ErrorResponse, PredictResponse

logger = logging.getLogger("mnist.api.inference")
router = APIRouter(prefix="/api/v1", tags=["Inference"])

# Refuse oversized uploads before they can consume unbounded process memory.
MAX_IMAGE_BYTES = 5 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


async def _read_upload(file: UploadFile) -> bytes:
    """Read an upload safely and close its stream after it has been consumed."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PNG, JPEG, and WEBP images are supported.",
        )
    try:
        content = await file.read(MAX_IMAGE_BYTES + 1)
    finally:
        await file.close()
    if not content:
        raise HTTPException(status_code=422, detail="The uploaded image is empty.")
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image must not exceed 5 MiB.")
    return content


@router.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        413: {"model": ErrorResponse, "description": "Image is too large."},
        415: {"model": ErrorResponse, "description": "Unsupported media type."},
        422: {"model": ErrorResponse, "description": "Invalid image content."},
        503: {"model": ErrorResponse, "description": "Model is unavailable."},
    },
    summary="Recognize a handwritten digit",
)
async def predict_digit(
    image: UploadFile = File(
        ..., description="PNG, JPEG, or WEBP image containing one digit."
    ),
    service: ModelService = Depends(get_model_service),
) -> PredictResponse:
    """Validate HTTP input, then delegate preprocessing and inference to a service.

    The handler contains no PyTorch code, so a batch worker or another backend
    can reuse/replace ``ModelService`` without changing this API contract.
    """
    image_bytes = await _read_upload(image)
    started_at = time.perf_counter()
    try:
        prediction, confidence = service.predict(image_bytes)
    except ModelUnavailableError as exc:
        logger.warning("Prediction rejected because model is unavailable: %s", exc)
        raise HTTPException(
            status_code=503, detail="Model is not ready to serve predictions."
        ) from exc
    except UnsupportedImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PredictResponse(
        prediction=prediction,
        confidence=confidence,
        time_taken=round(time.perf_counter() - started_at, 6),
    )
