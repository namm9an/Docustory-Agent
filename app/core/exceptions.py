"""
Custom exception classes for Phase 5 error handling.

Defines structured exceptions that map to specific error responses
and HTTP status codes for consistent error handling across the application.
"""

from typing import Dict, Any, Optional, List
from app.models.error import ErrorCategory, ErrorSeverity, ErrorCodes


class DocustoryBaseException(Exception):
    """
    Base exception class for all Docustory-specific errors.
    
    Provides structured error information including category, severity,
    error codes, and additional context for detailed error responses.
    """
    
    def __init__(
        self,
        message: str,
        error_code: str,
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None,
        status_code: Optional[int] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.suggestions = suggestions or []
        self.status_code = status_code
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API response."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "details": self.details,
            "suggestions": self.suggestions
        }


# File Upload Exceptions
class FileUploadException(DocustoryBaseException):
    """Base exception for file upload related errors."""
    
    def __init__(self, message: str, error_code: str, **kwargs):
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.FILE_UPLOAD,
            **kwargs
        )


class FileTooLargeError(FileUploadException):
    """Exception raised when uploaded file exceeds size limits."""
    
    def __init__(self, file_size_mb: float, max_size_mb: int, filename: str):
        super().__init__(
            message=f"File '{filename}' ({file_size_mb:.1f}MB) exceeds maximum size limit of {max_size_mb}MB",
            error_code=ErrorCodes.FILE_TOO_LARGE,
            severity=ErrorSeverity.MEDIUM,
            status_code=413,
            details={
                "file_size_mb": file_size_mb,
                "max_allowed_mb": max_size_mb,
                "filename": filename
            },
            suggestions=[
                "Reduce file size by compressing the document",
                "Split large documents into smaller sections",
                "Contact support for enterprise file size limits"
            ]
        )


class UnsupportedFileTypeError(FileUploadException):
    """Exception raised when file type is not supported."""
    
    def __init__(self, filename: str, detected_type: str, supported_types: List[str]):
        super().__init__(
            message=f"File type '{detected_type}' not supported. Supported types: {', '.join(supported_types)}",
            error_code=ErrorCodes.UNSUPPORTED_FILE_TYPE,
            severity=ErrorSeverity.MEDIUM,
            status_code=415,
            details={
                "filename": filename,
                "detected_type": detected_type,
                "supported_types": supported_types
            },
            suggestions=[
                f"Convert file to one of these supported formats: {', '.join(supported_types)}",
                "Ensure file has correct extension"
            ]
        )


class CorruptFileError(FileUploadException):
    """Exception raised when file is corrupted or cannot be processed."""
    
    def __init__(self, filename: str, processing_error: str):
        super().__init__(
            message=f"File '{filename}' appears to be corrupted or cannot be processed",
            error_code=ErrorCodes.CORRUPT_FILE,
            severity=ErrorSeverity.HIGH,
            status_code=400,
            details={
                "filename": filename,
                "processing_error": processing_error
            },
            suggestions=[
                "Try uploading a different copy of the file",
                "Ensure file is not password-protected",
                "Verify file is not corrupted"
            ]
        )


class TooManyPagesError(FileUploadException):
    """Exception raised when document has too many pages."""
    
    def __init__(self, filename: str, page_count: int, max_pages: int):
        super().__init__(
            message=f"Document '{filename}' has {page_count} pages, exceeds limit of {max_pages} pages",
            error_code=ErrorCodes.TOO_MANY_PAGES,
            severity=ErrorSeverity.MEDIUM,
            status_code=400,
            details={
                "filename": filename,
                "page_count": page_count,
                "max_pages": max_pages
            },
            suggestions=[
                "Split document into smaller sections",
                "Process document in multiple uploads",
                "Contact support for enterprise page limits"
            ]
        )


# Session Management Exceptions
class SessionException(DocustoryBaseException):
    """Base exception for session management errors."""
    
    def __init__(self, message: str, error_code: str, **kwargs):
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.SESSION,
            **kwargs
        )


class SessionNotFoundError(SessionException):
    """Exception raised when session cannot be found."""
    
    def __init__(self, session_id: str):
        super().__init__(
            message=f"Session '{session_id}' not found or has expired",
            error_code=ErrorCodes.SESSION_NOT_FOUND,
            severity=ErrorSeverity.MEDIUM,
            status_code=404,
            details={
                "session_id": session_id
            },
            suggestions=[
                "Upload a document to create a new session",
                "Sessions expire after 10 minutes of inactivity",
                "Ensure session ID is correct"
            ]
        )


class SessionExpiredError(SessionException):
    """Exception raised when session has expired."""
    
    def __init__(self, session_id: str, expired_minutes_ago: int):
        super().__init__(
            message=f"Session '{session_id}' expired {expired_minutes_ago} minutes ago",
            error_code=ErrorCodes.SESSION_EXPIRED,
            severity=ErrorSeverity.MEDIUM,
            status_code=410,
            details={
                "session_id": session_id,
                "expired_minutes_ago": expired_minutes_ago,
                "timeout_minutes": 10
            },
            suggestions=[
                "Upload a document to create a new session",
                "Sessions automatically expire after 10 minutes of inactivity"
            ]
        )


class InvalidSessionIdError(SessionException):
    """Exception raised when session ID format is invalid."""
    
    def __init__(self, session_id: str):
        super().__init__(
            message=f"Invalid session ID format: '{session_id}'",
            error_code=ErrorCodes.INVALID_SESSION_ID,
            severity=ErrorSeverity.LOW,
            status_code=400,
            details={
                "session_id": session_id,
                "expected_format": "UUID4 format"
            },
            suggestions=[
                "Ensure session ID is in valid UUID4 format",
                "Upload a document to get a valid session ID"
            ]
        )


# Performance and Capacity Exceptions
class PerformanceException(DocustoryBaseException):
    """Base exception for performance and capacity related errors."""
    
    def __init__(self, message: str, error_code: str, **kwargs):
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.PERFORMANCE,
            **kwargs
        )


class SystemAtCapacityError(PerformanceException):
    """Exception raised when system is at maximum capacity."""
    
    def __init__(self, active_sessions: int, max_sessions: int, queue_size: int):
        super().__init__(
            message=f"System at capacity ({active_sessions}/{max_sessions} sessions, {queue_size} queued)",
            error_code=ErrorCodes.SYSTEM_AT_CAPACITY,
            severity=ErrorSeverity.HIGH,
            status_code=503,
            details={
                "active_sessions": active_sessions,
                "max_sessions": max_sessions,
                "queue_size": queue_size,
                "estimated_wait_minutes": max(1, queue_size // 3)  # Rough estimate
            },
            suggestions=[
                "Please wait and try again in a few minutes",
                "System capacity resets as sessions expire",
                "Peak usage times may have longer wait times"
            ]
        )


class QueueFullError(PerformanceException):
    """Exception raised when request queue is full."""
    
    def __init__(self, queue_size: int, max_queue_size: int):
        super().__init__(
            message=f"Request queue is full ({queue_size}/{max_queue_size})",
            error_code=ErrorCodes.QUEUE_FULL,
            severity=ErrorSeverity.CRITICAL,
            status_code=503,
            details={
                "queue_size": queue_size,
                "max_queue_size": max_queue_size
            },
            suggestions=[
                "System is experiencing high load",
                "Please try again in 5-10 minutes",
                "Consider using the service during off-peak hours"
            ]
        )


class RequestTimeoutError(PerformanceException):
    """Exception raised when request processing times out."""
    
    def __init__(self, timeout_seconds: int, operation: str):
        super().__init__(
            message=f"Request timed out after {timeout_seconds} seconds during {operation}",
            error_code=ErrorCodes.REQUEST_TIMEOUT,
            severity=ErrorSeverity.HIGH,
            status_code=408,
            details={
                "timeout_seconds": timeout_seconds,
                "operation": operation
            },
            suggestions=[
                "Try again with a smaller document",
                "Operation may complete on retry",
                "Consider breaking large tasks into smaller parts"
            ]
        )


# AI Service Exceptions
class AIServiceException(DocustoryBaseException):
    """Base exception for AI service related errors."""
    
    def __init__(self, message: str, error_code: str, **kwargs):
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.AI_SERVICE,
            **kwargs
        )


class AIServiceUnavailableError(AIServiceException):
    """Exception raised when AI service is unavailable."""
    
    def __init__(self, service_name: str, last_successful: Optional[str] = None):
        super().__init__(
            message=f"AI service '{service_name}' is currently unavailable",
            error_code=ErrorCodes.AI_SERVICE_UNAVAILABLE,
            severity=ErrorSeverity.HIGH,
            status_code=502,
            details={
                "service": service_name,
                "last_successful": last_successful,
                "fallback_available": False
            },
            suggestions=[
                "Please try again in a few minutes",
                "Service may be temporarily down for maintenance",
                "Check system status page for updates"
            ]
        )


class AIServiceTimeoutError(AIServiceException):
    """Exception raised when AI service times out."""
    
    def __init__(self, service_name: str, timeout_seconds: int):
        super().__init__(
            message=f"AI service '{service_name}' timed out after {timeout_seconds} seconds",
            error_code=ErrorCodes.AI_SERVICE_TIMEOUT,
            severity=ErrorSeverity.HIGH,
            status_code=504,
            details={
                "service": service_name,
                "timeout_seconds": timeout_seconds,
                "fallback_available": True
            },
            suggestions=[
                "Try again - service may be temporarily slow",
                "Reduce input size if possible",
                "Fallback response may be provided"
            ]
        )


class ModelInferenceFailedError(AIServiceException):
    """Exception raised when AI model inference fails."""
    
    def __init__(self, model_name: str, error_details: str):
        super().__init__(
            message=f"Model '{model_name}' inference failed",
            error_code=ErrorCodes.MODEL_INFERENCE_FAILED,
            severity=ErrorSeverity.HIGH,
            status_code=422,
            details={
                "model": model_name,
                "error_details": error_details
            },
            suggestions=[
                "Try rephrasing your request",
                "Reduce input complexity",
                "Try again in a moment"
            ]
        )


# System and Validation Exceptions
class ValidationException(DocustoryBaseException):
    """Exception for request validation errors."""
    
    def __init__(self, message: str, field_errors: Optional[List[Dict[str, Any]]] = None):
        super().__init__(
            message=message,
            error_code=ErrorCodes.VALIDATION_ERROR,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            status_code=400,
            details={
                "field_errors": field_errors or []
            },
            suggestions=[
                "Check request format and required fields",
                "Refer to API documentation for correct format"
            ]
        )


class ConfigurationError(DocustoryBaseException):
    """Exception for system configuration errors."""
    
    def __init__(self, message: str, config_item: str):
        super().__init__(
            message=f"Configuration error: {message}",
            error_code=ErrorCodes.CONFIGURATION_ERROR,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            status_code=500,
            details={
                "config_item": config_item
            },
            suggestions=[
                "Contact system administrator",
                "Check system configuration"
            ]
        )


class MemoryError(DocustoryBaseException):
    """Exception for memory-related errors."""
    
    def __init__(self, message: str, memory_usage_mb: float):
        super().__init__(
            message=f"Memory limit exceeded: {message}",
            error_code=ErrorCodes.MEMORY_ERROR,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            status_code=500,
            details={
                "memory_usage_mb": memory_usage_mb
            },
            suggestions=[
                "Try with smaller documents",
                "System may recover automatically",
                "Contact support if issue persists"
            ]
        )