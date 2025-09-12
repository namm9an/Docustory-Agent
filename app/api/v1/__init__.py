from fastapi import APIRouter
from app.api.v1.endpoints import health, upload, ask, sessions, voice_streaming, system, history, errors, audio_summary

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(health.router)
api_router.include_router(upload.router)
api_router.include_router(ask.router)
api_router.include_router(sessions.router)
api_router.include_router(voice_streaming.router)
api_router.include_router(system.router)
api_router.include_router(history.router)
api_router.include_router(errors.router)
api_router.include_router(audio_summary.router)