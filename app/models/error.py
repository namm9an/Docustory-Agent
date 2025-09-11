"""
Pydantic models for Phase 5 error handling and structured error responses.

Defines comprehensive error response models for:
- API errors with proper HTTP status codes
- File upload errors
- Session management errors  
- Performance and capacity errors
- AI model service errors
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class ErrorSeverity(str, Enum):
    """Error severity levels for categorization and alerting."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Error categories for better classification and handling."""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    FILE_UPLOAD = "file_upload"
    SESSION = "session"
    AI_SERVICE = "ai_service"
    PERFORMANCE = "performance"
    SYSTEM = "system"
    NETWORK = "network"


class BaseErrorResponse(BaseModel):
    """Base error response model with common fields."""
    
    success: bool = Field(default=False, description="Always false for errors")
    error_code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    category: ErrorCategory = Field(..., description="Error category")
    severity: ErrorSeverity = Field(default=ErrorSeverity.MEDIUM, description="Error severity level")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error occurrence timestamp")
    request_id: Optional[str] = Field(default=None, description="Request ID for tracing")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error_code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "category": "validation",
                "severity": "medium",
                "timestamp": "2024-01-15T10:30:00Z",
                "request_id": "req_12345"
            }
        }


class DetailedErrorResponse(BaseErrorResponse):
    """Detailed error response with additional context and debugging info."""
    
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    suggestions: Optional[List[str]] = Field(default=None, description="Suggested solutions")
    documentation_url: Optional[str] = Field(default=None, description="Link to relevant documentation")
    support_info: Optional[Dict[str, str]] = Field(default=None, description="Support contact information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error_code": "FILE_TOO_LARGE",
                "message": "Uploaded file exceeds maximum size limit",
                "category": "file_upload",
                "severity": "medium",
                "timestamp": "2024-01-15T10:30:00Z",
                "request_id": "req_12345",
                "details": {
                    "file_size_mb": 250.5,
                    "max_allowed_mb": 200,
                    "filename": "large_document.pdf"
                },
                "suggestions": [
                    "Reduce file size by compressing the PDF",
                    "Split large documents into smaller sections",
                    "Contact support for enterprise limits"
                ],
                "documentation_url": "https://docs.docustory.in/file-limits",
                "support_info": {
                    "email": "support@docustory.in",
                    "chat": "Available 24/7"
                }
            }
        }


class ValidationErrorResponse(DetailedErrorResponse):
    """Specific error response for request validation failures."""
    
    field_errors: Optional[List[Dict[str, Any]]] = Field(default=None, description="Field-specific validation errors")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error_code": "REQUEST_VALIDATION_ERROR",
                "message": "Request contains invalid fields",
                "category": "validation",
                "severity": "medium",
                "field_errors": [
                    {
                        "field": "session_id",
                        "error": "Invalid UUID format",
                        "value": "invalid-id"
                    },
                    {
                        "field": "max_tokens",
                        "error": "Must be between 1 and 4096",
                        "value": 5000
                    }
                ]
            }
        }


class FileUploadErrorResponse(DetailedErrorResponse):
    """Error response for file upload related issues."""
    
    file_info: Optional[Dict[str, Any]] = Field(default=None, description="Information about the failed file")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error_code": "UNSUPPORTED_FILE_TYPE",
                "message": "File type not supported for processing",
                "category": "file_upload",
                "severity": "medium",
                "file_info": {
                    "filename": "document.txt",
                    "size_mb": 2.5,
                    "detected_type": "text/plain",
                    "supported_types": [".pdf", ".docx"]
                },
                "suggestions": [
                    "Convert file to PDF or DOCX format",
                    "Use supported file types only"
                ]
            }
        }


class SessionErrorResponse(DetailedErrorResponse):
    """Error response for session management issues."""
    
    session_info: Optional[Dict[str, Any]] = Field(default=None, description="Session-related information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error_code": "SESSION_EXPIRED",
                "message": "Session has expired or is no longer valid",
                "category": "session",
                "severity": "medium",
                "session_info": {
                    "session_id": "sess_12345",
                    "expired_at": "2024-01-15T10:20:00Z",
                    "timeout_minutes": 10
                },
                "suggestions": [
                    "Upload a document to create a new session",
                    "Sessions expire after 10 minutes of inactivity"
                ]
            }
        }


class PerformanceErrorResponse(DetailedErrorResponse):
    """Error response for performance and capacity issues."""
    
    system_status: Optional[Dict[str, Any]] = Field(default=None, description="Current system status")
    queue_info: Optional[Dict[str, Any]] = Field(default=None, description="Queue status information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error_code": "SYSTEM_AT_CAPACITY",
                "message": "System is currently at maximum capacity",
                "category": "performance",
                "severity": "high",
                "system_status": {
                    "active_sessions": 15,
                    "max_sessions": 15,
                    "queue_size": 25,
                    "estimated_wait_minutes": 5
                },
                "queue_info": {
                    "position": 26,
                    "estimated_wait_time": "5-10 minutes"
                },
                "suggestions": [
                    "Please wait and try again in a few minutes",
                    "System capacity resets as sessions expire"
                ]
            }
        }


class AIServiceErrorResponse(DetailedErrorResponse):
    """Error response for AI model service failures."""
    
    service_info: Optional[Dict[str, Any]] = Field(default=None, description="AI service status information")
    fallback_available: Optional[bool] = Field(default=None, description="Whether fallback service is available")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error_code": "AI_SERVICE_TIMEOUT",
                "message": "AI model service response timed out",
                "category": "ai_service",
                "severity": "high",
                "service_info": {
                    "service": "qwen",
                    "timeout_seconds": 120,
                    "last_successful": "2024-01-15T10:25:00Z"
                },
                "fallback_available": True,
                "suggestions": [
                    "Try again - service may be temporarily slow",
                    "Fallback response provided if available"
                ]
            }
        }


class SystemErrorResponse(DetailedErrorResponse):
    """Error response for system-level errors and exceptions."""
    
    stack_trace: Optional[str] = Field(default=None, description="Stack trace (development mode only)")
    system_info: Optional[Dict[str, Any]] = Field(default=None, description="System information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error_code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected system error occurred",
                "category": "system",
                "severity": "critical",
                "system_info": {
                    "version": "1.0.0",
                    "environment": "production",
                    "memory_usage_mb": 1250
                },
                "suggestions": [
                    "Please try again in a moment",
                    "Contact support if the issue persists"
                ]
            }
        }


# Common error code constants
class ErrorCodes:
    """Centralized error code constants for consistency."""
    
    # Validation Errors (400)
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_REQUEST_FORMAT = "INVALID_REQUEST_FORMAT"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    INVALID_FIELD_VALUE = "INVALID_FIELD_VALUE"
    
    # File Upload Errors (400, 413, 415)
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"
    CORRUPT_FILE = "CORRUPT_FILE"
    TOO_MANY_PAGES = "TOO_MANY_PAGES"
    FILE_PROCESSING_ERROR = "FILE_PROCESSING_ERROR"
    
    # Session Errors (404, 410)
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    SESSION_EXPIRED = "SESSION_EXPIRED"
    INVALID_SESSION_ID = "INVALID_SESSION_ID"
    SESSION_CREATION_FAILED = "SESSION_CREATION_FAILED"
    
    # Performance Errors (429, 503)
    SYSTEM_AT_CAPACITY = "SYSTEM_AT_CAPACITY"
    QUEUE_FULL = "QUEUE_FULL"
    REQUEST_TIMEOUT = "REQUEST_TIMEOUT"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    
    # AI Service Errors (502, 504)
    AI_SERVICE_UNAVAILABLE = "AI_SERVICE_UNAVAILABLE"
    AI_SERVICE_TIMEOUT = "AI_SERVICE_TIMEOUT"
    AI_SERVICE_ERROR = "AI_SERVICE_ERROR"
    MODEL_INFERENCE_FAILED = "MODEL_INFERENCE_FAILED"
    
    # System Errors (500)
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    MEMORY_ERROR = "MEMORY_ERROR"


# HTTP Status Code Mappings
ERROR_STATUS_CODES = {
    # Validation Errors
    ErrorCodes.VALIDATION_ERROR: 400,
    ErrorCodes.INVALID_REQUEST_FORMAT: 400,
    ErrorCodes.MISSING_REQUIRED_FIELD: 400,
    ErrorCodes.INVALID_FIELD_VALUE: 400,
    
    # File Upload Errors
    ErrorCodes.FILE_TOO_LARGE: 413,
    ErrorCodes.UNSUPPORTED_FILE_TYPE: 415,
    ErrorCodes.CORRUPT_FILE: 400,
    ErrorCodes.TOO_MANY_PAGES: 400,
    ErrorCodes.FILE_PROCESSING_ERROR: 422,
    
    # Session Errors
    ErrorCodes.SESSION_NOT_FOUND: 404,
    ErrorCodes.SESSION_EXPIRED: 410,
    ErrorCodes.INVALID_SESSION_ID: 400,
    ErrorCodes.SESSION_CREATION_FAILED: 500,
    
    # Performance Errors
    ErrorCodes.SYSTEM_AT_CAPACITY: 503,
    ErrorCodes.QUEUE_FULL: 503,
    ErrorCodes.REQUEST_TIMEOUT: 408,
    ErrorCodes.RATE_LIMIT_EXCEEDED: 429,
    
    # AI Service Errors
    ErrorCodes.AI_SERVICE_UNAVAILABLE: 502,
    ErrorCodes.AI_SERVICE_TIMEOUT: 504,
    ErrorCodes.AI_SERVICE_ERROR: 502,
    ErrorCodes.MODEL_INFERENCE_FAILED: 422,
    
    # System Errors
    ErrorCodes.INTERNAL_SERVER_ERROR: 500,
    ErrorCodes.DATABASE_ERROR: 500,
    ErrorCodes.CONFIGURATION_ERROR: 500,
    ErrorCodes.MEMORY_ERROR: 500,
}