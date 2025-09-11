import logging
from fastapi import APIRouter, HTTPException, Depends, Query
from datetime import datetime
from typing import Optional, Dict, Any

from app.core.error_handler import (
    error_handler, get_error_stats, reset_error_stats, 
    ErrorCategory, ErrorSeverity, error_metrics
)
from app.core.session import session_manager
from app.services.parser_service import DocumentParserService
from app.services.yake_service import YAKEService
from app.services.qwen_client import qwen_client
from app.services.stt import whisper_client
from app.services.microsoft_tts import xtts_client
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/system/health", tags=["System"])
async def comprehensive_health_check():
    """
    Comprehensive system health check with error handling capabilities.
    
    Checks all major system components and their error handling status.
    """
    try:
        health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "healthy",
            "version": "3.0.0",  # Phase 3 version
            "components": {},
            "error_handling": {},
            "performance": {}
        }
        
        # Check core components
        try:
            # Session Manager
            session_stats = session_manager.get_session_stats()
            health_status["components"]["session_manager"] = {
                "status": "operational",
                "active_sessions": session_stats["active_sessions"],
                "memory_usage_mb": session_stats["total_memory_mb"],
                "capacity_used_percent": round(
                    (session_stats["active_sessions"] / session_stats["max_sessions"]) * 100, 1
                )
            }
        except Exception as e:
            health_status["components"]["session_manager"] = {
                "status": "degraded",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        # Document Parser Service
        try:
            parser_service = DocumentParserService()
            parser_status = await parser_service.get_parser_status()
            health_status["components"]["document_parser"] = {
                "status": "operational",
                **parser_status
            }
        except Exception as e:
            health_status["components"]["document_parser"] = {
                "status": "degraded",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        # YAKE Service
        try:
            yake_service = YAKEService()
            health_status["components"]["yake_service"] = {
                "status": "operational" if settings.ENABLE_YAKE_SEARCH else "disabled",
                "enabled": settings.ENABLE_YAKE_SEARCH,
                "max_keywords": settings.YAKE_MAX_KEYWORDS,
                "ngram_size": settings.YAKE_NGRAM_SIZE
            }
        except Exception as e:
            health_status["components"]["yake_service"] = {
                "status": "degraded",
                "error": str(e)
            }
            if settings.ENABLE_YAKE_SEARCH:
                health_status["status"] = "degraded"
        
        # AI Services
        try:
            qwen_info = qwen_client.get_model_info()
            health_status["components"]["qwen_service"] = {
                "status": "operational" if qwen_info.get("available", False) else "unavailable",
                **qwen_info
            }
        except Exception as e:
            health_status["components"]["qwen_service"] = {
                "status": "unavailable",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        try:
            whisper_info = whisper_client.get_model_info()
            health_status["components"]["whisper_service"] = {
                "status": "operational" if whisper_info.get("available", False) else "unavailable",
                **whisper_info
            }
        except Exception as e:
            health_status["components"]["whisper_service"] = {
                "status": "unavailable",
                "error": str(e)
            }
        
        try:
            xtts_info = xtts_client.get_model_info()
            health_status["components"]["xtts_service"] = {
                "status": "operational" if xtts_info.get("available", False) else "unavailable",
                **xtts_info
            }
        except Exception as e:
            health_status["components"]["xtts_service"] = {
                "status": "unavailable",
                "error": str(e)
            }
        
        # Error Handling Status
        error_stats = get_error_stats()
        health_status["error_handling"] = {
            "system_status": "operational",
            "total_errors_handled": error_stats["total_errors"],
            "error_categories": list(error_stats["errors_by_category"].keys()),
            "recent_error_rate_per_hour": error_stats["error_rate_per_hour"],
            "last_error": error_stats["last_error_time"],
            "fallback_responses_available": len(error_handler.fallback_responses)
        }
        
        # Performance Metrics
        health_status["performance"] = {
            "memory_usage": {
                "sessions_mb": session_stats.get("total_memory_mb", 0),
                "max_per_session_mb": settings.MAX_MEMORY_PER_SESSION_MB
            },
            "capacity": {
                "max_sessions": settings.MAX_CONCURRENT_SESSIONS,
                "current_sessions": session_stats.get("active_sessions", 0),
                "file_size_limit_mb": settings.MAX_FILE_SIZE_MB,
                "page_limit": settings.MAX_PAGES
            }
        }
        
        # Overall status determination
        component_statuses = [comp.get("status", "unknown") for comp in health_status["components"].values()]
        if "unavailable" in component_statuses:
            health_status["status"] = "degraded"
        elif "degraded" in component_statuses:
            health_status["status"] = "degraded"
        
        # High error rate check
        if error_stats["error_rate_per_hour"] > 50:  # More than 50 errors per hour
            health_status["status"] = "degraded"
            health_status["error_handling"]["warning"] = "High error rate detected"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return error_handler.handle_exception(e, context={"operation": "health_check"})


@router.get("/system/errors/stats", tags=["System"])
async def get_error_statistics():
    """Get detailed error statistics and monitoring information."""
    
    try:
        error_stats = get_error_stats()
        
        # Enhanced statistics
        detailed_stats = {
            **error_stats,
            "error_categories_info": {
                category.value: {
                    "description": _get_category_description(category),
                    "severity_level": _get_typical_severity(category).value,
                    "count": error_stats["errors_by_category"].get(category.value, 0)
                }
                for category in ErrorCategory
            },
            "severity_levels_info": {
                severity.value: {
                    "description": _get_severity_description(severity),
                    "count": error_stats["errors_by_severity"].get(severity.value, 0)
                }
                for severity in ErrorSeverity
            },
            "recent_errors_sample": error_metrics.recent_errors[-10:] if error_metrics.recent_errors else [],
            "system_health": "good" if error_stats["error_rate_per_hour"] < 10 else "warning" if error_stats["error_rate_per_hour"] < 50 else "critical"
        }
        
        return detailed_stats
        
    except Exception as e:
        logger.error(f"Error statistics retrieval failed: {e}")
        return error_handler.handle_exception(e, context={"operation": "error_stats"})


@router.post("/system/errors/reset", tags=["System"])
async def reset_error_statistics(
    confirm: bool = Query(False, description="Confirm reset of error statistics")
):
    """Reset error statistics (admin operation)."""
    
    try:
        if not confirm:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": "CONFIRMATION_REQUIRED",
                    "message": "Error statistics reset requires confirmation",
                    "suggestion": "Add confirm=true to the request"
                }
            )
        
        # Get stats before reset for logging
        old_stats = get_error_stats()
        
        # Reset statistics
        reset_error_stats()
        
        logger.info(f"Error statistics reset. Previous total: {old_stats['total_errors']}")
        
        return {
            "success": True,
            "message": "Error statistics have been reset",
            "previous_stats": {
                "total_errors": old_stats["total_errors"],
                "reset_time": datetime.utcnow().isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error statistics reset failed: {e}")
        return error_handler.handle_exception(e, context={"operation": "error_reset"})


@router.get("/system/errors/test", tags=["System"])
async def test_error_handling(
    error_type: str = Query("validation", description="Type of error to simulate"),
    include_details: bool = Query(False, description="Include technical details in response")
):
    """
    Test error handling system by simulating different types of errors.
    
    Useful for testing error responses and fallback mechanisms.
    """
    try:
        # Map error types to test scenarios
        test_scenarios = {
            "validation": ValueError("Invalid input format provided"),
            "not_found": HTTPException(status_code=404, detail="Resource not found"),
            "memory": MemoryError("Insufficient memory to process request"),
            "timeout": TimeoutError("Request timed out after 30 seconds"),
            "external_service": ConnectionError("Failed to connect to external API"),
            "parsing": Exception("Failed to parse document: corrupted file format"),
            "internal": RuntimeError("Internal server error occurred"),
            "rate_limit": HTTPException(status_code=429, detail="Rate limit exceeded")
        }
        
        if error_type not in test_scenarios:
            available_types = list(test_scenarios.keys())
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": "INVALID_ERROR_TYPE",
                    "message": f"Invalid error type: {error_type}",
                    "available_types": available_types
                }
            )
        
        # Simulate the error
        test_error = test_scenarios[error_type]
        
        # Create error response using error handler
        error_response = error_handler.create_error_response(
            error=test_error,
            context={"operation": "error_handling_test", "simulated": True},
            include_details=include_details
        )
        
        # Return the error response structure (but don't raise it)
        return {
            "test_result": "success",
            "simulated_error_type": error_type,
            "error_response": error_response.to_dict(),
            "note": "This is a simulated error for testing purposes"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error handling test failed: {e}")
        return error_handler.handle_exception(e, context={"operation": "error_test"})


def _get_category_description(category: ErrorCategory) -> str:
    """Get human-readable description for error category."""
    descriptions = {
        ErrorCategory.VALIDATION: "Input validation and format errors",
        ErrorCategory.AUTHENTICATION: "Authentication and credential errors",
        ErrorCategory.AUTHORIZATION: "Permission and access control errors",
        ErrorCategory.NOT_FOUND: "Resource not found errors",
        ErrorCategory.RATE_LIMIT: "Rate limiting and quota exceeded errors",
        ErrorCategory.SERVICE_UNAVAILABLE: "Service temporarily unavailable",
        ErrorCategory.INTERNAL_ERROR: "Internal server and application errors",
        ErrorCategory.EXTERNAL_SERVICE: "External API and service connectivity errors",
        ErrorCategory.PARSING_ERROR: "Document parsing and format errors",
        ErrorCategory.MEMORY_ERROR: "Memory allocation and resource errors",
        ErrorCategory.TIMEOUT: "Request timeout and processing delays",
        ErrorCategory.CONFIGURATION: "Configuration and setup errors"
    }
    return descriptions.get(category, "Unknown error category")


def _get_severity_description(severity: ErrorSeverity) -> str:
    """Get human-readable description for error severity."""
    descriptions = {
        ErrorSeverity.LOW: "Minor issues that don't significantly impact functionality",
        ErrorSeverity.MEDIUM: "Moderate issues that may affect some operations",
        ErrorSeverity.HIGH: "Serious issues that significantly impact functionality",
        ErrorSeverity.CRITICAL: "Critical issues that may cause system failure"
    }
    return descriptions.get(severity, "Unknown severity level")


def _get_typical_severity(category: ErrorCategory) -> ErrorSeverity:
    """Get typical severity level for error category."""
    typical_severities = {
        ErrorCategory.VALIDATION: ErrorSeverity.LOW,
        ErrorCategory.AUTHENTICATION: ErrorSeverity.MEDIUM,
        ErrorCategory.AUTHORIZATION: ErrorSeverity.MEDIUM,
        ErrorCategory.NOT_FOUND: ErrorSeverity.LOW,
        ErrorCategory.RATE_LIMIT: ErrorSeverity.HIGH,
        ErrorCategory.SERVICE_UNAVAILABLE: ErrorSeverity.HIGH,
        ErrorCategory.INTERNAL_ERROR: ErrorSeverity.HIGH,
        ErrorCategory.EXTERNAL_SERVICE: ErrorSeverity.MEDIUM,
        ErrorCategory.PARSING_ERROR: ErrorSeverity.MEDIUM,
        ErrorCategory.MEMORY_ERROR: ErrorSeverity.HIGH,
        ErrorCategory.TIMEOUT: ErrorSeverity.MEDIUM,
        ErrorCategory.CONFIGURATION: ErrorSeverity.CRITICAL
    }
    return typical_severities.get(category, ErrorSeverity.MEDIUM)