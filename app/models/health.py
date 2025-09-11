from pydantic import BaseModel
from typing import Dict, Any


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = "ok"
    version: str
    timestamp: str
    details: Dict[str, Any] = {}


class PingResponse(BaseModel):
    """Ping response model."""
    pong: bool = True
    timestamp: str