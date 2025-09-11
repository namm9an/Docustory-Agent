"""
Phase 5 Error Handling and System Monitoring Endpoints.

Provides endpoints for:
- System health and status monitoring
- Error testing and validation
- Performance metrics and capacity monitoring
- Queue status and management
- Manual cleanup operations
"""

import time
from fastapi import APIRouter, HTTPException, Query, Depends
from datetime import datetime
from typing import Optional, Dict, Any

from app.services.error_service import get_error_service
from app.services.queue_manager import get_queue_manager
from app.core.session import session_manager
from app.core.logging import get_logger
from app.core.config import settings
from app.models.error import ErrorCodes, ErrorCategory, ErrorSeverity
from app.core.exceptions import (
    SystemAtCapacityError, FileTooLargeError, SessionNotFoundError,
    UnsupportedFileTypeError, AIServiceTimeoutError
)

logger = get_logger()
router = APIRouter()


@router.get("/system/status", tags=["System Monitoring"])
async def get_system_status():
    """
    Get comprehensive system status including capacity, performance, and health metrics.
    """
    try:
        start_time = time.perf_counter()
        
        # Get queue manager status
        queue_manager = get_queue_manager()
        queue_status = await queue_manager.get_system_status()
        
        # Get session manager status
        session_stats = session_manager.get_session_stats()
        
        # Get memory usage estimation
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        system_status = {
            "status": "operational",
            "timestamp": datetime.utcnow().isoformat(),
            "version": settings.APP_VERSION,
            "uptime_seconds": time.time() - start_time,
            
            # Capacity Status
            "capacity": {
                "active_sessions": queue_status["active_sessions"],
                "max_sessions": queue_status["max_sessions"],
                "utilization_percent": queue_status["capacity_utilization"],
                "available_slots": queue_status["max_sessions"] - queue_status["active_sessions"],
                "at_capacity": queue_status["at_capacity"]
            },
            
            # Queue Status
            "queue": {
                "size": queue_status["queue_size"],
                "max_size": queue_status["max_queue_size"],
                "utilization_percent": queue_status["queue_utilization"],
                "estimated_wait_minutes": queue_status["estimated_wait_minutes"],
                "queue_full": queue_status["queue_full"]
            },
            
            # Performance Metrics
            "performance": {
                "requests_processed": queue_status["stats"]["requests_processed"],
                "requests_queued": queue_status["stats"]["requests_queued"],
                "requests_timeout": queue_status["stats"]["requests_timeout"],
                "requests_rejected": queue_status["stats"]["requests_rejected"],
                "peak_queue_size": queue_status["stats"]["peak_queue_size"],
                "peak_active_sessions": queue_status["stats"]["peak_active_sessions"]
            },
            
            # Memory Status
            "memory": {
                "process_memory_mb": memory_info.rss / 1024 / 1024,
                "session_memory_mb": session_stats["total_memory_mb"],
                "max_memory_per_session_mb": settings.MAX_MEMORY_PER_SESSION_MB
            },
            
            # Configuration
            "limits": {
                "max_file_size_mb": settings.MAX_FILE_SIZE_MB,
                "max_pages": settings.MAX_PAGES,
                "session_timeout_minutes": settings.SESSION_TIMEOUT_MINUTES,
                "queue_timeout_seconds": settings.QUEUE_TIMEOUT_SECONDS,
                "ai_model_timeout_seconds": settings.AI_MODEL_TIMEOUT_SECONDS
            },
            
            # Feature Flags
            "features": {
                "conversation_memory": settings.ENABLE_CONVERSATION_MEMORY,
                "yake_search": settings.ENABLE_YAKE_SEARCH,
                "queue_system": settings.ENABLE_QUEUE_SYSTEM,
                "error_logging": settings.ENABLE_ERROR_LOGGING
            }
        }
        
        # Determine overall system health
        if queue_status["capacity_utilization"] >= 90:
            system_status["status"] = "critical"
        elif queue_status["capacity_utilization"] >= 75:
            system_status["status"] = "warning"
        elif queue_status["queue_size"] > queue_status["max_queue_size"] * 0.5:
            system_status["status"] = "degraded"
        
        processing_time = (time.perf_counter() - start_time) * 1000
        system_status["response_time_ms"] = round(processing_time, 2)
        
        return system_status
        
    except Exception as e:
        logger.log_error(e, "SYSTEM_STATUS_ERROR", "system", include_stack_trace=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "SYSTEM_STATUS_ERROR",
                "message": "Failed to retrieve system status",
                "details": str(e)
            }
        )


@router.get("/system/health", tags=["System Monitoring"])
async def health_check():
    """
    Simple health check endpoint for load balancers and monitoring systems.
    """
    try:
        # Quick health indicators
        queue_manager = get_queue_manager()
        system_status = await queue_manager.get_system_status()
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {
                "api": "ok",
                "queue_system": "ok" if not system_status["queue_full"] else "degraded",
                "capacity": "ok" if not system_status["at_capacity"] else "critical",
                "memory": "ok"  # Could add memory checks here
            }
        }
        
        # Determine overall health
        if any(check == "critical" for check in health_status["checks"].values()):
            health_status["status"] = "critical"
        elif any(check == "degraded" for check in health_status["checks"].values()):
            health_status["status"] = "degraded"
        
        status_code = 200
        if health_status["status"] == "critical":
            status_code = 503
        elif health_status["status"] == "degraded":
            status_code = 200  # Still accept traffic but indicate degraded performance
        
        return health_status
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


@router.get("/queue/status", tags=["Queue Management"])
async def get_queue_status():
    """Get detailed queue status and metrics."""
    try:
        queue_manager = get_queue_manager()
        status = await queue_manager.get_system_status()
        
        return {
            "success": True,
            "queue_status": status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.log_error(e, "QUEUE_STATUS_ERROR", "system")
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "QUEUE_STATUS_ERROR",
                "message": "Failed to retrieve queue status"
            }
        )


@router.get("/queue/position/{request_id}", tags=["Queue Management"])
async def get_queue_position(request_id: str):
    """Get position of specific request in queue."""
    try:
        queue_manager = get_queue_manager()
        
        position = await queue_manager.get_queue_position(request_id)
        queue_info = await queue_manager.get_queue_info(request_id)
        
        if position is None:
            return {
                "success": False,
                "message": "Request not found in queue",
                "request_id": request_id,
                "possible_reasons": [
                    "Request has already been processed",
                    "Request has timed out and been removed",
                    "Invalid request ID"
                ]
            }
        
        return {
            "success": True,
            "request_id": request_id,
            "queue_position": position,
            "queue_info": queue_info,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.log_error(e, "QUEUE_POSITION_ERROR", "system")
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "QUEUE_POSITION_ERROR",
                "message": f"Failed to get queue position for request {request_id}"
            }
        )


@router.post("/test/errors/{error_type}", tags=["Error Testing"])
async def test_error_handling(error_type: str):
    """
    Test endpoint for validating error handling behavior.
    Useful for development and testing purposes.
    """
    if not settings.DEBUG:
        raise HTTPException(
            status_code=403,
            detail={
                "error_code": "ERROR_TESTING_DISABLED",
                "message": "Error testing endpoints are only available in debug mode"
            }
        )
    
    try:
        if error_type == "file_too_large":
            raise FileTooLargeError(
                file_size_mb=250.0,
                max_size_mb=200,
                filename="test_large_file.pdf"
            )
        
        elif error_type == "session_not_found":
            raise SessionNotFoundError("test-session-id")
        
        elif error_type == "unsupported_file":
            raise UnsupportedFileTypeError(
                filename="test.txt",
                detected_type="text/plain",
                supported_types=[".pdf", ".docx"]
            )
        
        elif error_type == "system_capacity":
            raise SystemAtCapacityError(
                active_sessions=15,
                max_sessions=15,
                queue_size=25
            )
        
        elif error_type == "ai_timeout":
            raise AIServiceTimeoutError("qwen", 120)
        
        elif error_type == "generic_error":
            raise ValueError("This is a test generic error")
        
        elif error_type == "http_error":
            raise HTTPException(
                status_code=418,
                detail={
                    "error_code": "IM_A_TEAPOT",
                    "message": "This is a test HTTP error"
                }
            )
        
        else:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": "INVALID_ERROR_TYPE",
                    "message": f"Unknown error type: {error_type}",
                    "available_types": [
                        "file_too_large", "session_not_found", "unsupported_file",
                        "system_capacity", "ai_timeout", "generic_error", "http_error"
                    ]
                }
            )
        
    except Exception as e:
        # Let the error propagate to be handled by global error handlers
        raise


@router.post("/system/cleanup", tags=["System Management"])
async def manual_system_cleanup(
    cleanup_sessions: bool = Query(default=True, description="Clean up expired sessions"),
    cleanup_queue: bool = Query(default=True, description="Clean up expired queue requests"),
    force: bool = Query(default=False, description="Force cleanup even if not needed")
):
    """
    Manually trigger system cleanup operations.
    """
    try:
        cleanup_results = {
            "cleanup_performed": [],
            "sessions_cleaned": 0,
            "queue_requests_cleaned": 0,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Clean up expired sessions
        if cleanup_sessions:
            initial_count = len(session_manager._sessions)
            session_manager.cleanup_expired_sessions()
            sessions_cleaned = initial_count - len(session_manager._sessions)
            cleanup_results["sessions_cleaned"] = sessions_cleaned
            cleanup_results["cleanup_performed"].append("sessions")
        
        # Clean up expired queue requests
        if cleanup_queue:
            queue_manager = get_queue_manager()
            await queue_manager._cleanup_expired_requests()
            cleanup_results["cleanup_performed"].append("queue")
        
        logger.log_info(
            "Manual system cleanup completed",
            cleanup_results=cleanup_results
        )
        
        return {
            "success": True,
            "message": "System cleanup completed successfully",
            "results": cleanup_results
        }
        
    except Exception as e:
        logger.log_error(e, "MANUAL_CLEANUP_ERROR", "system", include_stack_trace=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "MANUAL_CLEANUP_ERROR",
                "message": "Failed to perform manual cleanup",
                "details": str(e)
            }
        )


@router.get("/errors/codes", tags=["Error Reference"])
async def get_error_codes():
    """
    Get comprehensive list of error codes and their meanings.
    Useful for client-side error handling and documentation.
    """
    try:
        error_codes_info = {
            "categories": {
                "validation": {
                    "description": "Request validation and format errors",
                    "typical_status_codes": [400, 422],
                    "codes": [
                        ErrorCodes.VALIDATION_ERROR,
                        ErrorCodes.INVALID_REQUEST_FORMAT,
                        ErrorCodes.MISSING_REQUIRED_FIELD,
                        ErrorCodes.INVALID_FIELD_VALUE
                    ]
                },
                "file_upload": {
                    "description": "File upload and processing errors",
                    "typical_status_codes": [400, 413, 415, 422],
                    "codes": [
                        ErrorCodes.FILE_TOO_LARGE,
                        ErrorCodes.UNSUPPORTED_FILE_TYPE,
                        ErrorCodes.CORRUPT_FILE,
                        ErrorCodes.TOO_MANY_PAGES,
                        ErrorCodes.FILE_PROCESSING_ERROR
                    ]
                },
                "session": {
                    "description": "Session management errors",
                    "typical_status_codes": [400, 404, 410],
                    "codes": [
                        ErrorCodes.SESSION_NOT_FOUND,
                        ErrorCodes.SESSION_EXPIRED,
                        ErrorCodes.INVALID_SESSION_ID,
                        ErrorCodes.SESSION_CREATION_FAILED
                    ]
                },
                "performance": {
                    "description": "System capacity and performance errors",
                    "typical_status_codes": [408, 429, 503],
                    "codes": [
                        ErrorCodes.SYSTEM_AT_CAPACITY,
                        ErrorCodes.QUEUE_FULL,
                        ErrorCodes.REQUEST_TIMEOUT,
                        ErrorCodes.RATE_LIMIT_EXCEEDED
                    ]
                },
                "ai_service": {
                    "description": "AI model service errors",
                    "typical_status_codes": [502, 504, 422],
                    "codes": [
                        ErrorCodes.AI_SERVICE_UNAVAILABLE,
                        ErrorCodes.AI_SERVICE_TIMEOUT,
                        ErrorCodes.AI_SERVICE_ERROR,
                        ErrorCodes.MODEL_INFERENCE_FAILED
                    ]
                },
                "system": {
                    "description": "Internal system errors",
                    "typical_status_codes": [500],
                    "codes": [
                        ErrorCodes.INTERNAL_SERVER_ERROR,
                        ErrorCodes.DATABASE_ERROR,
                        ErrorCodes.CONFIGURATION_ERROR,
                        ErrorCodes.MEMORY_ERROR
                    ]
                }
            },
            
            "severity_levels": {
                "low": "Minor issues that don't significantly impact functionality",
                "medium": "Moderate issues that may affect user experience", 
                "high": "Serious issues that significantly impact functionality",
                "critical": "Critical system errors requiring immediate attention"
            },
            
            "general_guidance": {
                "4xx_errors": "Client errors - check request format and parameters",
                "5xx_errors": "Server errors - retry may help, contact support if persistent",
                "timeout_errors": "Usually temporary - retry with exponential backoff",
                "capacity_errors": "System at limit - wait and retry, or try during off-peak hours"
            },
            
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return error_codes_info
        
    except Exception as e:
        logger.log_error(e, "ERROR_CODES_RETRIEVAL_ERROR", "system")
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "ERROR_CODES_RETRIEVAL_ERROR",
                "message": "Failed to retrieve error codes information"
            }
        )