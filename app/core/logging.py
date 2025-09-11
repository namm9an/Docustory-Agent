"""
Comprehensive logging system for Phase 5 - Error Handling & Performance.

Provides structured logging with:
- Rotating file handlers
- Request tracking and correlation
- Performance metrics
- Error categorization and alerting
- Session lifecycle tracking
"""

import logging
import logging.handlers
import sys
import os
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import contextmanager
from pathlib import Path

from app.core.config import settings


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging with JSON output."""
    
    def format(self, record: logging.LogRecord) -> str:
        # Create base structured log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add request context if available
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        
        if hasattr(record, 'session_id'):
            log_entry['session_id'] = record.session_id
            
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        
        # Add performance metrics if available
        if hasattr(record, 'duration_ms'):
            log_entry['duration_ms'] = record.duration_ms
            
        if hasattr(record, 'memory_mb'):
            log_entry['memory_mb'] = record.memory_mb
        
        # Add error context if available
        if hasattr(record, 'error_code'):
            log_entry['error_code'] = record.error_code
            
        if hasattr(record, 'error_category'):
            log_entry['error_category'] = record.error_category
            
        if hasattr(record, 'stack_trace') and record.stack_trace:
            log_entry['stack_trace'] = record.stack_trace
        
        # Add custom fields
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry, default=str)


class DocustoryLogger:
    """
    Enhanced logger for Docustory application with structured logging,
    performance tracking, and error categorization.
    """
    
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger("docustory")
        
    def setup_logging(self):
        """Set up logging configuration with file rotation and structured formatting."""
        
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # File handler with rotation
        if settings.ENABLE_ERROR_LOGGING:
            file_handler = logging.handlers.RotatingFileHandler(
                filename="logs/docustory.log",
                maxBytes=settings.LOG_FILE_MAX_SIZE_MB * 1024 * 1024,
                backupCount=settings.LOG_BACKUP_COUNT,
                encoding='utf-8'
            )
            file_handler.setFormatter(StructuredFormatter())
            root_logger.addHandler(file_handler)
        
        # Console handler for development
        if settings.DEBUG:
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
        
        # Error file handler for critical errors
        error_handler = logging.handlers.RotatingFileHandler(
            filename="logs/errors.log",
            maxBytes=settings.LOG_FILE_MAX_SIZE_MB * 1024 * 1024,
            backupCount=settings.LOG_BACKUP_COUNT,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(error_handler)
    
    def log_request_start(self, request_id: str, method: str, path: str, 
                         session_id: Optional[str] = None, **kwargs):
        """Log the start of a request."""
        self.logger.info(
            f"Request started: {method} {path}",
            extra={
                'request_id': request_id,
                'session_id': session_id,
                'http_method': method,
                'path': path,
                'event_type': 'request_start',
                'extra_fields': kwargs
            }
        )
    
    def log_request_end(self, request_id: str, method: str, path: str, 
                       status_code: int, duration_ms: float,
                       session_id: Optional[str] = None, **kwargs):
        """Log the end of a request with performance metrics."""
        level = logging.INFO if status_code < 400 else logging.WARNING
        
        self.logger.log(
            level,
            f"Request completed: {method} {path} - {status_code} ({duration_ms:.2f}ms)",
            extra={
                'request_id': request_id,
                'session_id': session_id,
                'http_method': method,
                'path': path,
                'status_code': status_code,
                'duration_ms': duration_ms,
                'event_type': 'request_end',
                'extra_fields': kwargs
            }
        )
    
    def log_session_created(self, session_id: str, request_id: Optional[str] = None,
                           **kwargs):
        """Log session creation."""
        self.logger.info(
            f"Session created: {session_id}",
            extra={
                'session_id': session_id,
                'request_id': request_id,
                'event_type': 'session_created',
                'extra_fields': kwargs
            }
        )
    
    def log_session_expired(self, session_id: str, age_minutes: float, **kwargs):
        """Log session expiration."""
        self.logger.info(
            f"Session expired: {session_id} (age: {age_minutes:.1f} minutes)",
            extra={
                'session_id': session_id,
                'age_minutes': age_minutes,
                'event_type': 'session_expired',
                'extra_fields': kwargs
            }
        )
    
    def log_session_cleanup(self, sessions_cleaned: int, **kwargs):
        """Log session cleanup operations."""
        self.logger.info(
            f"Session cleanup completed: {sessions_cleaned} sessions removed",
            extra={
                'sessions_cleaned': sessions_cleaned,
                'event_type': 'session_cleanup',
                'extra_fields': kwargs
            }
        )
    
    def log_file_upload(self, filename: str, file_size_mb: float, 
                       session_id: str, request_id: Optional[str] = None,
                       **kwargs):
        """Log file upload events."""
        self.logger.info(
            f"File uploaded: {filename} ({file_size_mb:.1f}MB)",
            extra={
                'session_id': session_id,
                'request_id': request_id,
                'filename': filename,
                'file_size_mb': file_size_mb,
                'event_type': 'file_upload',
                'extra_fields': kwargs
            }
        )
    
    def log_ai_service_call(self, service: str, operation: str, 
                           duration_ms: float, success: bool,
                           session_id: Optional[str] = None,
                           request_id: Optional[str] = None, **kwargs):
        """Log AI service interactions."""
        level = logging.INFO if success else logging.WARNING
        
        self.logger.log(
            level,
            f"AI service call: {service}.{operation} - {'success' if success else 'failed'} ({duration_ms:.2f}ms)",
            extra={
                'session_id': session_id,
                'request_id': request_id,
                'service': service,
                'operation': operation,
                'duration_ms': duration_ms,
                'success': success,
                'event_type': 'ai_service_call',
                'extra_fields': kwargs
            }
        )
    
    def log_performance_metrics(self, operation: str, duration_ms: float,
                               memory_mb: Optional[float] = None,
                               session_id: Optional[str] = None,
                               request_id: Optional[str] = None, **kwargs):
        """Log performance metrics."""
        self.logger.info(
            f"Performance: {operation} completed in {duration_ms:.2f}ms",
            extra={
                'session_id': session_id,
                'request_id': request_id,
                'operation': operation,
                'duration_ms': duration_ms,
                'memory_mb': memory_mb,
                'event_type': 'performance_metric',
                'extra_fields': kwargs
            }
        )
    
    def log_error(self, error: Exception, error_code: str, error_category: str,
                 session_id: Optional[str] = None,
                 request_id: Optional[str] = None,
                 include_stack_trace: bool = True, **kwargs):
        """Log structured error information."""
        stack_trace = None
        if include_stack_trace and settings.INCLUDE_ERROR_DETAILS:
            import traceback
            stack_trace = traceback.format_exc()
        
        self.logger.error(
            f"Error: {error_code} - {str(error)}",
            extra={
                'session_id': session_id,
                'request_id': request_id,
                'error_code': error_code,
                'error_category': error_category,
                'error_type': error.__class__.__name__,
                'stack_trace': stack_trace,
                'event_type': 'error',
                'extra_fields': kwargs
            }
        )
    
    def log_capacity_warning(self, active_sessions: int, max_sessions: int,
                            queue_size: int, **kwargs):
        """Log capacity warnings."""
        self.logger.warning(
            f"High system load: {active_sessions}/{max_sessions} sessions active, {queue_size} queued",
            extra={
                'active_sessions': active_sessions,
                'max_sessions': max_sessions,
                'queue_size': queue_size,
                'capacity_utilization': (active_sessions / max_sessions) * 100,
                'event_type': 'capacity_warning',
                'extra_fields': kwargs
            }
        )
    
    def log_memory_usage(self, operation: str, memory_usage_mb: float,
                        threshold_mb: Optional[float] = None,
                        session_id: Optional[str] = None, **kwargs):
        """Log memory usage information."""
        level = logging.WARNING if threshold_mb and memory_usage_mb > threshold_mb else logging.INFO
        
        self.logger.log(
            level,
            f"Memory usage: {operation} using {memory_usage_mb:.1f}MB",
            extra={
                'session_id': session_id,
                'operation': operation,
                'memory_mb': memory_usage_mb,
                'threshold_mb': threshold_mb,
                'event_type': 'memory_usage',
                'extra_fields': kwargs
            }
        )


@contextmanager
def log_performance(operation: str, logger: DocustoryLogger, 
                   session_id: Optional[str] = None,
                   request_id: Optional[str] = None):
    """Context manager for logging operation performance."""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.log_performance_metrics(
            operation=operation,
            duration_ms=duration_ms,
            session_id=session_id,
            request_id=request_id
        )


@contextmanager
def log_request_lifecycle(method: str, path: str, request_id: str,
                         session_id: Optional[str] = None):
    """Context manager for logging full request lifecycle."""
    logger = get_logger()
    logger.log_request_start(request_id, method, path, session_id)
    
    start_time = time.perf_counter()
    status_code = 200
    
    try:
        yield
    except Exception as e:
        status_code = getattr(e, 'status_code', 500)
        raise
    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.log_request_end(request_id, method, path, status_code, duration_ms, session_id)


# Global logger instance
_logger_instance = None

def get_logger() -> DocustoryLogger:
    """Get the global logger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = DocustoryLogger()
    return _logger_instance


def setup_logging():
    """Initialize logging system."""
    get_logger()


# Convenience functions
def log_info(message: str, **kwargs):
    """Log info message with structured context."""
    get_logger().logger.info(message, extra={'extra_fields': kwargs})


def log_warning(message: str, **kwargs):
    """Log warning message with structured context."""
    get_logger().logger.warning(message, extra={'extra_fields': kwargs})


def log_error(message: str, **kwargs):
    """Log error message with structured context."""
    get_logger().logger.error(message, extra={'extra_fields': kwargs})