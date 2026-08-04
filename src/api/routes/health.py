"""Health endpoints for users, dashboards, and container orchestration."""

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import ModelService, get_model_service
from src.api.schemas import HealthCheckResponse

router = APIRouter(tags=["Health"])


def _response(service: ModelService) -> HealthCheckResponse:
    """Map the internal model state to the stable public response contract."""
    return HealthCheckResponse(
        status="healthy" if service.is_ready else "degraded",
        model_loaded=service.is_ready,
        version=service.model_version,
    )


@router.get("/health", response_model=HealthCheckResponse, summary="Check API status")
def health_check(
    service: ModelService = Depends(get_model_service),
) -> HealthCheckResponse:
    """Return API/model state; 200 allows a UI to show a degraded service."""
    return _response(service)


@router.get(
    "/health/live", response_model=HealthCheckResponse, summary="Liveness probe"
)
def liveness_check(
    service: ModelService = Depends(get_model_service),
) -> HealthCheckResponse:
    """Confirm the HTTP process is alive; a model is not required here."""
    return _response(service)


@router.get(
    "/health/ready", response_model=HealthCheckResponse, summary="Readiness probe"
)
def readiness_check(
    service: ModelService = Depends(get_model_service),
) -> HealthCheckResponse:
    """Return 200 only after the model has been loaded successfully."""
    response = _response(service)
    if not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready to serve predictions.",
        )
    return response
