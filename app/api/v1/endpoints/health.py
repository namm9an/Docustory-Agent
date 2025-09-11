from fastapi import APIRouter
from datetime import datetime
from app.models.health import HealthResponse, PingResponse
from app.core.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns the current status of the application including:
    - Service status
    - Version information
    - Current timestamp
    - Additional system details
    """
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        timestamp=datetime.utcnow().isoformat(),
        details={
            "app_name": settings.APP_NAME,
            "debug_mode": settings.DEBUG,
            "max_file_size_mb": settings.MAX_FILE_SIZE_MB,
            "max_pages": settings.MAX_PAGES,
            "session_timeout_minutes": settings.SESSION_TIMEOUT_MINUTES
        }
    )


@router.get("/ping", response_model=PingResponse, tags=["Health"])
async def ping() -> PingResponse:
    """
    Simple ping endpoint for connectivity testing.
    
    Returns a pong response with timestamp.
    """
    return PingResponse(
        pong=True,
        timestamp=datetime.utcnow().isoformat()
    )