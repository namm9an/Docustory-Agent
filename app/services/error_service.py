"""
Error Service for Phase 5 - Structured Error Response Management.

Provides centralized error handling with:
- Structured error response generation
- Exception mapping to HTTP status codes
- Error categorization and severity assessment
- Contextual error information
- Integration with logging and monitoring
"""

import traceback
from typing import Dict, Any, Optional, Union, Type
from datetime import datetime
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from app.models.error import (
    BaseErrorResponse, DetailedErrorResponse, ValidationErrorResponse,
    FileUploadErrorResponse, SessionErrorResponse, PerformanceErrorResponse,
    AIServiceErrorResponse, SystemErrorResponse, ErrorCodes, ERROR_STATUS_CODES,
    ErrorCategory, ErrorSeverity
)
from app.core.exceptions import (
    DocustoryBaseException, FileUploadException, SessionException,
    PerformanceException, AIServiceException, ValidationException
)
from app.core.config import settings
from app.core.logging import get_logger


class ErrorService:
    """
    Centralized error handling service for converting exceptions
    to structured API responses with proper logging and monitoring.
    """
    
    def __init__(self):
        self.logger = get_logger()
        
        # Exception type to response model mapping
        self._exception_response_mapping: Dict[Type, Type] = {
            FileUploadException: FileUploadErrorResponse,
            SessionException: SessionErrorResponse,
            PerformanceException: PerformanceErrorResponse,
            AIServiceException: AIServiceErrorResponse,
            ValidationException: ValidationErrorResponse,
            DocustoryBaseException: DetailedErrorResponse
        }
    
    def handle_exception(self, 
                        error: Exception,
                        request_id: Optional[str] = None,
                        session_id: Optional[str] = None,
                        context: Optional[Dict[str, Any]] = None) -> JSONResponse:
        """
        Convert exception to structured JSON error response.
        
        Args:
            error: The exception to handle
            request_id: Optional request ID for tracing
            session_id: Optional session ID for context
            context: Additional context information
            
        Returns:
            JSONResponse with structured error data
        """
        context = context or {}
        
        # Handle Docustory custom exceptions
        if isinstance(error, DocustoryBaseException):
            return self._handle_custom_exception(error, request_id, session_id, context)
        
        # Handle FastAPI HTTPExceptions
        elif isinstance(error, HTTPException):
            return self._handle_http_exception(error, request_id, session_id, context)
        
        # Handle validation errors (Pydantic)
        elif hasattr(error, 'errors') and callable(getattr(error, 'errors')):
            return self._handle_validation_error(error, request_id, session_id, context)
        
        # Handle generic exceptions
        else:
            return self._handle_generic_exception(error, request_id, session_id, context)
    
    def create_error_response(self,
                             error_code: str,
                             message: str,
                             category: ErrorCategory,
                             severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                             status_code: Optional[int] = None,
                             details: Optional[Dict[str, Any]] = None,
                             suggestions: Optional[list] = None,
                             request_id: Optional[str] = None) -> JSONResponse:
        """
        Create a structured error response from components.
        
        Args:
            error_code: Machine-readable error code
            message: Human-readable error message
            category: Error category
            severity: Error severity level
            status_code: HTTP status code (auto-determined if not provided)
            details: Additional error details
            suggestions: List of suggested solutions
            request_id: Request ID for tracing
            
        Returns:
            JSONResponse with structured error data
        """
        # Determine status code if not provided
        if status_code is None:
            status_code = ERROR_STATUS_CODES.get(error_code, 500)
        
        # Create response model
        response_data = DetailedErrorResponse(
            error_code=error_code,
            message=message,
            category=category,
            severity=severity,
            details=details,
            suggestions=suggestions or [],
            request_id=request_id
        )
        
        # Log the error
        self.logger.log_error(
            Exception(message), error_code, category.value,
            request_id=request_id,
            include_stack_trace=False
        )
        
        return JSONResponse(
            status_code=status_code,
            content=response_data.model_dump()
        )
    
    def _handle_custom_exception(self,
                                error: DocustoryBaseException,
                                request_id: Optional[str],
                                session_id: Optional[str],
                                context: Dict[str, Any]) -> JSONResponse:
        """Handle Docustory custom exceptions."""
        
        # Determine appropriate response model
        response_class = self._get_response_class(error)
        
        # Get status code from exception or mapping
        status_code = error.status_code or ERROR_STATUS_CODES.get(error.error_code, 500)
        
        # Create response data
        response_data = {
            "error_code": error.error_code,
            "message": error.message,
            "category": error.category.value,
            "severity": error.severity.value,
            "request_id": request_id,
            "timestamp": datetime.utcnow()
        }
        
        # Add details if available
        if error.details:
            response_data["details"] = error.details
        
        # Add suggestions if available
        if error.suggestions:
            response_data["suggestions"] = error.suggestions
        
        # Add context information
        if context:
            response_data.setdefault("details", {}).update(context)
        
        # Add session info for session-related errors
        if isinstance(error, SessionException) and session_id:
            response_data.setdefault("details", {})["session_id"] = session_id
        
        # Log the error with full context
        self.logger.log_error(
            error, error.error_code, error.category.value,
            session_id=session_id,
            request_id=request_id,
            include_stack_trace=settings.INCLUDE_ERROR_DETAILS
        )
        
        return JSONResponse(
            status_code=status_code,
            content=response_data
        )
    
    def _handle_http_exception(self,
                              error: HTTPException,
                              request_id: Optional[str],
                              session_id: Optional[str],
                              context: Dict[str, Any]) -> JSONResponse:
        """Handle FastAPI HTTPExceptions."""
        
        # Extract error details from HTTPException
        error_detail = error.detail
        if isinstance(error_detail, dict):
            error_code = error_detail.get("error_code", "HTTP_ERROR")
            message = error_detail.get("message", str(error_detail))
            details = error_detail.get("details")
        else:
            error_code = "HTTP_ERROR"
            message = str(error_detail)
            details = None
        
        # Determine category based on status code
        category = self._categorize_http_error(error.status_code)
        severity = self._assess_severity(error.status_code)
        
        response_data = DetailedErrorResponse(
            error_code=error_code,
            message=message,
            category=category,
            severity=severity,
            details=details,
            request_id=request_id
        )
        
        # Log the error
        self.logger.log_error(
            error, error_code, category.value,
            session_id=session_id,
            request_id=request_id,
            include_stack_trace=False
        )
        
        return JSONResponse(
            status_code=error.status_code,
            content=response_data.model_dump()
        )
    
    def _handle_validation_error(self,
                                error: Exception,
                                request_id: Optional[str],
                                session_id: Optional[str],
                                context: Dict[str, Any]) -> JSONResponse:
        """Handle Pydantic validation errors."""
        
        # Extract validation errors
        validation_errors = []
        if hasattr(error, 'errors'):
            for err in error.errors():
                validation_errors.append({
                    "field": ".".join(str(loc) for loc in err.get("loc", [])),
                    "error": err.get("msg", "Validation error"),
                    "type": err.get("type", "unknown"),
                    "value": err.get("input")
                })
        
        response_data = ValidationErrorResponse(
            error_code=ErrorCodes.VALIDATION_ERROR,
            message="Request validation failed",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            field_errors=validation_errors,
            details={"validation_errors": validation_errors},
            suggestions=[
                "Check request format and required fields",
                "Refer to API documentation for correct format"
            ],
            request_id=request_id
        )
        
        # Log validation error
        self.logger.log_error(
            error, ErrorCodes.VALIDATION_ERROR, ErrorCategory.VALIDATION.value,
            session_id=session_id,
            request_id=request_id,
            include_stack_trace=False
        )
        
        return JSONResponse(
            status_code=400,
            content=response_data.model_dump()
        )
    
    def _handle_generic_exception(self,
                                 error: Exception,
                                 request_id: Optional[str],
                                 session_id: Optional[str],
                                 context: Dict[str, Any]) -> JSONResponse:
        """Handle generic Python exceptions."""
        
        error_code = ErrorCodes.INTERNAL_SERVER_ERROR
        message = "An unexpected error occurred"
        
        # Try to provide more specific error info for common exceptions
        if isinstance(error, (IOError, OSError)):
            error_code = "FILE_SYSTEM_ERROR"
            message = "File system operation failed"
        elif isinstance(error, MemoryError):
            error_code = ErrorCodes.MEMORY_ERROR
            message = "System ran out of memory"
        elif isinstance(error, TimeoutError):
            error_code = ErrorCodes.REQUEST_TIMEOUT
            message = "Operation timed out"
        
        # Include stack trace in development
        stack_trace = None
        if settings.INCLUDE_ERROR_DETAILS and settings.DEBUG:
            stack_trace = traceback.format_exc()
        
        response_data = SystemErrorResponse(
            error_code=error_code,
            message=message,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            details={
                "error_type": error.__class__.__name__,
                "error_message": str(error)
            },
            stack_trace=stack_trace,
            system_info={
                "version": settings.APP_VERSION,
                "debug_mode": settings.DEBUG
            },
            suggestions=[
                "Please try again in a moment",
                "Contact support if the issue persists"
            ],
            request_id=request_id
        )
        
        # Log the error with full details
        self.logger.log_error(
            error, error_code, ErrorCategory.SYSTEM.value,
            session_id=session_id,
            request_id=request_id,
            include_stack_trace=True
        )
        
        return JSONResponse(
            status_code=500,
            content=response_data.model_dump()
        )
    
    def _get_response_class(self, error: DocustoryBaseException) -> Type:
        """Get appropriate response class for exception type."""
        for exc_type, response_type in self._exception_response_mapping.items():
            if isinstance(error, exc_type):
                return response_type
        return DetailedErrorResponse
    
    def _categorize_http_error(self, status_code: int) -> ErrorCategory:
        """Categorize HTTP error by status code."""
        if 400 <= status_code < 500:
            if status_code == 401:
                return ErrorCategory.AUTHENTICATION
            elif status_code == 403:
                return ErrorCategory.AUTHORIZATION
            elif status_code in (413, 415):
                return ErrorCategory.FILE_UPLOAD
            elif status_code == 404:
                return ErrorCategory.SESSION
            elif status_code == 429:
                return ErrorCategory.PERFORMANCE
            else:
                return ErrorCategory.VALIDATION
        elif 500 <= status_code < 600:
            if status_code in (502, 504):
                return ErrorCategory.AI_SERVICE
            elif status_code == 503:
                return ErrorCategory.PERFORMANCE
            else:
                return ErrorCategory.SYSTEM
        else:
            return ErrorCategory.SYSTEM
    
    def _assess_severity(self, status_code: int) -> ErrorSeverity:
        """Assess error severity based on status code."""
        if status_code < 400:
            return ErrorSeverity.LOW
        elif 400 <= status_code < 500:
            return ErrorSeverity.MEDIUM
        elif status_code == 500:
            return ErrorSeverity.CRITICAL
        else:
            return ErrorSeverity.HIGH
    
    def create_file_too_large_error(self, filename: str, file_size_mb: float, 
                                   max_size_mb: int, request_id: Optional[str] = None) -> JSONResponse:
        """Convenience method for file size errors."""
        return self.create_error_response(
            error_code=ErrorCodes.FILE_TOO_LARGE,
            message=f"File '{filename}' ({file_size_mb:.1f}MB) exceeds maximum size limit of {max_size_mb}MB",
            category=ErrorCategory.FILE_UPLOAD,
            severity=ErrorSeverity.MEDIUM,
            status_code=413,
            details={
                "filename": filename,
                "file_size_mb": file_size_mb,
                "max_allowed_mb": max_size_mb
            },
            suggestions=[
                "Reduce file size by compressing the document",
                "Split large documents into smaller sections"
            ],
            request_id=request_id
        )
    
    def create_session_not_found_error(self, session_id: str, 
                                      request_id: Optional[str] = None) -> JSONResponse:
        """Convenience method for session not found errors."""
        return self.create_error_response(
            error_code=ErrorCodes.SESSION_NOT_FOUND,
            message=f"Session '{session_id}' not found or has expired",
            category=ErrorCategory.SESSION,
            severity=ErrorSeverity.MEDIUM,
            status_code=404,
            details={"session_id": session_id},
            suggestions=[
                "Upload a document to create a new session",
                "Sessions expire after 10 minutes of inactivity"
            ],
            request_id=request_id
        )
    
    def create_system_capacity_error(self, active_sessions: int, max_sessions: int,
                                   queue_size: int, request_id: Optional[str] = None) -> JSONResponse:
        """Convenience method for system capacity errors."""
        return self.create_error_response(
            error_code=ErrorCodes.SYSTEM_AT_CAPACITY,
            message=f"System at capacity ({active_sessions}/{max_sessions} sessions, {queue_size} queued)",
            category=ErrorCategory.PERFORMANCE,
            severity=ErrorSeverity.HIGH,
            status_code=503,
            details={
                "active_sessions": active_sessions,
                "max_sessions": max_sessions,
                "queue_size": queue_size,
                "estimated_wait_minutes": max(1, queue_size // 3)
            },
            suggestions=[
                "Please wait and try again in a few minutes",
                "System capacity resets as sessions expire"
            ],
            request_id=request_id
        )


# Global error service instance
_error_service_instance = None

def get_error_service() -> ErrorService:
    """Get the global error service instance."""
    global _error_service_instance
    if _error_service_instance is None:
        _error_service_instance = ErrorService()
    return _error_service_instance