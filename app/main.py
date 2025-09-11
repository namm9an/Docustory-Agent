import logging
import time
import uuid
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.core.config import settings
from app.api.v1 import api_router
from app.core.error_handler import (
    error_handler, error_context, ErrorCategory, ErrorSeverity,
    DocumentProcessingError, AIServiceError, SessionError, ValidationError as CustomValidationError
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.DESCRIPTION,
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)


@app.middleware("http")
async def enhanced_logging_middleware(request: Request, call_next):
    """
    Enhanced logging middleware with error context and request tracking.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]
    
    # Add request ID to the request state
    request.state.request_id = request_id
    
    # Log request with ID
    logger.info(f"Request [{request_id}]: {request.method} {request.url}")
    
    try:
        # Process request with error context
        async with error_context(
            operation_name=f"{request.method} {request.url.path}",
            context={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else "unknown"
            }
        ):
            response = await call_next(request)
            
        # Log successful response
        process_time = time.time() - start_time
        logger.info(
            f"Response [{request_id}]: {response.status_code} | "
            f"Duration: {process_time:.4f}s | "
            f"Path: {request.url.path}"
        )
        
        return response
        
    except Exception as e:
        # This shouldn't normally be reached due to exception handlers,
        # but provides a final safety net
        process_time = time.time() - start_time
        logger.error(
            f"Request [{request_id}] failed after {process_time:.4f}s: {e}",
            extra={"request_id": request_id, "error": str(e)}
        )
        raise


# Custom exception handlers for specific error types
@app.exception_handler(DocumentProcessingError)
async def document_processing_exception_handler(request: Request, exc: DocumentProcessingError):
    """Handle document processing errors with intelligent fallbacks."""
    request_id = getattr(request.state, 'request_id', None)
    context = {
        "request_path": request.url.path,
        "request_method": request.method,
        "error_type": "document_processing"
    }
    return error_handler.handle_exception(exc, context=context, request_id=request_id)


@app.exception_handler(AIServiceError)
async def ai_service_exception_handler(request: Request, exc: AIServiceError):
    """Handle AI service errors with fallback responses."""
    request_id = getattr(request.state, 'request_id', None)
    context = {
        "request_path": request.url.path,
        "request_method": request.method,
        "error_type": "ai_service"
    }
    return error_handler.handle_exception(exc, context=context, request_id=request_id)


@app.exception_handler(SessionError)
async def session_exception_handler(request: Request, exc: SessionError):
    """Handle session-related errors."""
    request_id = getattr(request.state, 'request_id', None)
    context = {
        "request_path": request.url.path,
        "request_method": request.method,
        "error_type": "session"
    }
    return error_handler.handle_exception(exc, context=context, request_id=request_id)


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors with enhanced details."""
    request_id = getattr(request.state, 'request_id', None)
    
    # Create a custom validation error for better handling
    validation_error = CustomValidationError(f"Request validation failed: {exc}")
    context = {
        "request_path": request.url.path,
        "request_method": request.method,
        "validation_errors": exc.errors(),
        "error_type": "pydantic_validation"
    }
    
    return error_handler.handle_exception(
        validation_error,
        status_code=422,
        context=context,
        request_id=request_id
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with enhanced error responses."""
    request_id = getattr(request.state, 'request_id', None)
    context = {
        "request_path": request.url.path,
        "request_method": request.method,
        "status_code": exc.status_code,
        "error_type": "http_exception"
    }
    
    return error_handler.handle_exception(
        exc,
        status_code=exc.status_code,
        context=context,
        request_id=request_id
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other unexpected exceptions with comprehensive error handling."""
    request_id = getattr(request.state, 'request_id', None)
    context = {
        "request_path": request.url.path,
        "request_method": request.method,
        "client_ip": request.client.host if request.client else "unknown",
        "error_type": "unexpected_exception"
    }
    
    # Log the full exception details
    logger.error(
        f"Unexpected error [{request_id}] for {request.url}: {exc}",
        exc_info=True,
        extra=context
    )
    
    return error_handler.handle_exception(
        exc,
        status_code=500,
        context=context,
        request_id=request_id
    )


# Include API routes
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with basic service information.
    """
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "operational",
        "docs": "/docs",
        "health": f"{settings.API_V1_STR}/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    logger.info(f"Server will run on {settings.HOST}:{settings.PORT}")
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level="info"
    )