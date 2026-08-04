"""Central registration point for the public API routers."""

from fastapi import APIRouter

from src.api.routes.health import router as health_router
from src.api.routes.inference import router as inference_router

# ``main.py`` only needs this one router. Feature modules stay independent.
router = APIRouter()
router.include_router(health_router)
router.include_router(inference_router)

__all__ = ["router"]
