import logging
import traceback
import time
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import asyncio

from app.core.config import settings

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for monitoring and alerting."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high" 
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification and handling."""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    NOT_FOUND = "not_found"
    RATE_LIMIT = "rate_limit"
    SERVICE_UNAVAILABLE = "service_unavailable"
    INTERNAL_ERROR = "internal_error"
    EXTERNAL_SERVICE = "external_service"
    PARSING_ERROR = "parsing_error"
    MEMORY_ERROR = "memory_error"
    TIMEOUT = "timeout"
    CONFIGURATION = "configuration"


@dataclass
class ErrorMetrics:
    """Track error metrics for monitoring."""
    total_errors: int = 0
    errors_by_category: Dict[str, int] = field(default_factory=dict)
    errors_by_severity: Dict[str, int] = field(default_factory=dict)
    recent_errors: List[Dict[str, Any]] = field(default_factory=list)
    last_error_time: Optional[datetime] = None
    
    def record_error(self, category: ErrorCategory, severity: ErrorSeverity, details: Dict[str, Any]):
        """Record an error occurrence."""
        self.total_errors += 1
        self.errors_by_category[category.value] = self.errors_by_category.get(category.value, 0) + 1
        self.errors_by_severity[severity.value] = self.errors_by_severity.get(severity.value, 0) + 1
        self.last_error_time = datetime.utcnow()
        
        # Keep only recent 100 errors
        error_record = {
            "timestamp": self.last_error_time.isoformat(),
            "category": category.value,
            "severity": severity.value,
            **details
        }
        self.recent_errors.append(error_record)
        if len(self.recent_errors) > 100:
            self.recent_errors.pop(0)


# Global error metrics instance
error_metrics = ErrorMetrics()


@dataclass
class ErrorResponse:
    """Standardized error response structure."""
    success: bool = False
    error_code: str = "UNKNOWN_ERROR"
    message: str = "An unexpected error occurred"
    details: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    request_id: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)
    retry_after: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response."""
        result = {
            "success": self.success,
            "error_code": self.error_code,
            "message": self.message,
            "timestamp": self.timestamp
        }
        
        if self.details:
            result["details"] = self.details
        if self.request_id:
            result["request_id"] = self.request_id
        if self.suggestions:
            result["suggestions"] = self.suggestions
        if self.retry_after:
            result["retry_after"] = self.retry_after
            
        return result


class ErrorHandler:
    """Comprehensive error handling system with fallbacks."""
    
    def __init__(self):
        self.fallback_responses = {
            ErrorCategory.PARSING_ERROR: "I encountered an issue processing your document. Please ensure it's a valid PDF or DOCX file and try uploading again.",
            ErrorCategory.EXTERNAL_SERVICE: "I'm experiencing connectivity issues with external services. Please try again in a moment.",
            ErrorCategory.MEMORY_ERROR: "The system is currently under high load. Please try again with a smaller document or wait a moment.",
            ErrorCategory.TIMEOUT: "The request took longer than expected. Please try again with a shorter document or simpler query.",
            ErrorCategory.SERVICE_UNAVAILABLE: "The service is temporarily unavailable. Please try again in a few moments.",
            ErrorCategory.VALIDATION: "There was an issue with the provided input. Please check the format and try again.",
            ErrorCategory.NOT_FOUND: "The requested resource was not found. Please verify the session ID or document exists.",
            ErrorCategory.INTERNAL_ERROR: "An unexpected error occurred. Our team has been notified and is working on a fix."
        }
    
    def classify_error(self, error: Exception, context: Dict[str, Any] = None) -> tuple[ErrorCategory, ErrorSeverity]:
        """Classify error by category and severity."""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Classification logic based on error type and message
        if isinstance(error, HTTPException):
            if error.status_code == 404:
                return ErrorCategory.NOT_FOUND, ErrorSeverity.LOW
            elif error.status_code == 401:
                return ErrorCategory.AUTHENTICATION, ErrorSeverity.MEDIUM
            elif error.status_code == 403:
                return ErrorCategory.AUTHORIZATION, ErrorSeverity.MEDIUM
            elif error.status_code == 413:
                return ErrorCategory.VALIDATION, ErrorSeverity.MEDIUM
            elif error.status_code == 429:
                return ErrorCategory.RATE_LIMIT, ErrorSeverity.HIGH
            elif 500 <= error.status_code < 600:
                return ErrorCategory.INTERNAL_ERROR, ErrorSeverity.HIGH
        
        # Memory and resource errors
        if "memory" in error_message or "out of memory" in error_message:
            return ErrorCategory.MEMORY_ERROR, ErrorSeverity.HIGH
        
        # Timeout errors
        if "timeout" in error_message or "timed out" in error_message:
            return ErrorCategory.TIMEOUT, ErrorSeverity.MEDIUM
        
        # External service errors
        if any(keyword in error_message for keyword in ["connection", "network", "api", "endpoint"]):
            return ErrorCategory.EXTERNAL_SERVICE, ErrorSeverity.MEDIUM
        
        # Parsing errors
        if any(keyword in error_message for keyword in ["parse", "format", "invalid", "corrupt"]):
            return ErrorCategory.PARSING_ERROR, ErrorSeverity.MEDIUM
        
        # Configuration errors
        if any(keyword in error_message for keyword in ["config", "setting", "environment"]):
            return ErrorCategory.CONFIGURATION, ErrorSeverity.HIGH
        
        # Default classification
        return ErrorCategory.INTERNAL_ERROR, ErrorSeverity.MEDIUM
    
    def get_suggestions(self, category: ErrorCategory, error: Exception, context: Dict[str, Any] = None) -> List[str]:
        """Get contextual suggestions based on error category."""
        suggestions = []
        
        if category == ErrorCategory.PARSING_ERROR:
            suggestions = [
                "Ensure your document is a valid PDF or DOCX file",
                "Try reducing the document size if it's very large",
                "Check if the document is password-protected or corrupted"
            ]
        elif category == ErrorCategory.EXTERNAL_SERVICE:
            suggestions = [
                "Try again in a few moments",
                "Check your internet connection",
                "Contact support if the issue persists"
            ]
        elif category == ErrorCategory.MEMORY_ERROR:
            suggestions = [
                "Try uploading a smaller document",
                "Wait a moment for system resources to free up",
                "Consider breaking large documents into smaller parts"
            ]
        elif category == ErrorCategory.TIMEOUT:
            suggestions = [
                "Try with a shorter document or simpler question",
                "Wait a moment and try again",
                "Check your internet connection stability"
            ]
        elif category == ErrorCategory.NOT_FOUND:
            suggestions = [
                "Verify the session ID is correct",
                "Upload your document again if the session expired",
                "Check if you're using the correct endpoint"
            ]
        elif category == ErrorCategory.VALIDATION:
            suggestions = [
                "Check the input format and requirements",
                "Ensure all required fields are provided",
                "Verify file types and size limits"
            ]
        elif category == ErrorCategory.SERVICE_UNAVAILABLE:
            suggestions = [
                "Try again in a few minutes",
                "Check system status for maintenance updates",
                "Contact support if the issue continues"
            ]
        else:
            suggestions = [
                "Try again in a moment",
                "Contact support if the issue persists",
                "Check the system status page for updates"
            ]
        
        return suggestions
    
    def create_error_response(
        self, 
        error: Exception, 
        context: Dict[str, Any] = None,
        request_id: str = None,
        include_details: bool = None
    ) -> ErrorResponse:
        """Create a standardized error response."""
        category, severity = self.classify_error(error, context)
        suggestions = self.get_suggestions(category, error, context)
        
        # Determine if we should include technical details
        if include_details is None:
            include_details = settings.DEBUG
        
        # Create error response
        error_response = ErrorResponse(
            error_code=category.value.upper(),
            message=self.fallback_responses.get(category, str(error)),
            request_id=request_id,
            suggestions=suggestions
        )
        
        # Add technical details if requested
        if include_details:
            error_response.details = {
                "error_type": type(error).__name__,
                "original_message": str(error),
                "severity": severity.value,
                "context": context or {}
            }
            
            # Add traceback in debug mode
            if settings.DEBUG:
                error_response.details["traceback"] = traceback.format_exc()
        
        # Add retry information for certain categories
        if category in [ErrorCategory.EXTERNAL_SERVICE, ErrorCategory.TIMEOUT, ErrorCategory.RATE_LIMIT]:
            error_response.retry_after = 30  # Suggest retry after 30 seconds
        elif category == ErrorCategory.MEMORY_ERROR:
            error_response.retry_after = 60  # Suggest retry after 60 seconds
        
        # Record error metrics
        error_metrics.record_error(category, severity, {
            "error_type": type(error).__name__,
            "message": str(error),
            "context": context or {}
        })
        
        # Log error
        log_level = logging.ERROR if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL] else logging.WARNING
        logger.log(
            log_level,
            f"Error handled: {category.value} ({severity.value}) - {error}",
            extra={
                "error_category": category.value,
                "error_severity": severity.value,
                "request_id": request_id,
                "context": context
            }
        )
        
        return error_response
    
    def handle_exception(
        self, 
        error: Exception, 
        status_code: int = None,
        context: Dict[str, Any] = None,
        request_id: str = None
    ) -> JSONResponse:
        """Handle exception and return appropriate JSON response."""
        error_response = self.create_error_response(error, context, request_id)
        
        # Determine status code
        if status_code is None:
            if isinstance(error, HTTPException):
                status_code = error.status_code
            else:
                category, severity = self.classify_error(error, context)
                status_code = self._get_status_code_for_category(category)
        
        return JSONResponse(
            status_code=status_code,
            content=error_response.to_dict()
        )
    
    def _get_status_code_for_category(self, category: ErrorCategory) -> int:
        """Get appropriate HTTP status code for error category."""
        status_codes = {
            ErrorCategory.VALIDATION: 400,
            ErrorCategory.AUTHENTICATION: 401,
            ErrorCategory.AUTHORIZATION: 403,
            ErrorCategory.NOT_FOUND: 404,
            ErrorCategory.RATE_LIMIT: 429,
            ErrorCategory.PARSING_ERROR: 422,
            ErrorCategory.EXTERNAL_SERVICE: 502,
            ErrorCategory.SERVICE_UNAVAILABLE: 503,
            ErrorCategory.TIMEOUT: 504,
            ErrorCategory.MEMORY_ERROR: 507,
            ErrorCategory.CONFIGURATION: 500,
            ErrorCategory.INTERNAL_ERROR: 500
        }
        return status_codes.get(category, 500)


# Global error handler instance
error_handler = ErrorHandler()


@asynccontextmanager
async def error_context(operation_name: str, context: Dict[str, Any] = None):
    """Context manager for handling errors with automatic fallback."""
    start_time = time.time()
    operation_context = {
        "operation": operation_name,
        "start_time": start_time,
        **(context or {})
    }
    
    try:
        yield operation_context
    except Exception as e:
        operation_context["duration"] = time.time() - start_time
        logger.error(f"Operation '{operation_name}' failed: {e}", extra=operation_context)
        raise
    else:
        operation_context["duration"] = time.time() - start_time
        logger.debug(f"Operation '{operation_name}' completed successfully", extra=operation_context)


async def retry_with_backoff(
    func,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Retry function with exponential backoff."""
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func()
            else:
                return func()
        except exceptions as e:
            last_exception = e
            
            if attempt < max_retries:
                delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                await asyncio.sleep(delay)
            else:
                logger.error(f"All {max_retries + 1} attempts failed: {e}")
    
    raise last_exception


def get_error_stats() -> Dict[str, Any]:
    """Get current error statistics."""
    return {
        "total_errors": error_metrics.total_errors,
        "errors_by_category": error_metrics.errors_by_category,
        "errors_by_severity": error_metrics.errors_by_severity,
        "last_error_time": error_metrics.last_error_time.isoformat() if error_metrics.last_error_time else None,
        "recent_error_count": len(error_metrics.recent_errors),
        "error_rate_per_hour": len([
            e for e in error_metrics.recent_errors 
            if datetime.fromisoformat(e["timestamp"]) > datetime.utcnow().replace(hour=datetime.utcnow().hour-1)
        ]) if error_metrics.recent_errors else 0
    }


def reset_error_stats():
    """Reset error statistics (for testing or maintenance)."""
    global error_metrics
    error_metrics = ErrorMetrics()
    logger.info("Error statistics reset")


# Custom exception classes for specific error scenarios
class DocumentProcessingError(Exception):
    """Raised when document processing fails."""
    pass


class AIServiceError(Exception):
    """Raised when AI service calls fail."""
    pass


class SessionError(Exception):
    """Raised when session operations fail."""
    pass


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass