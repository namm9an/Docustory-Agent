import time
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from datetime import datetime
import os
from app.models.upload import UploadResponse, UploadError, DocumentMetadata
from app.core.config import settings
from app.core.session import session_manager
from app.services.parser_service import DocumentParserService, DocumentParsingError
from app.services.yake_service import YAKEService

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize services
document_parser = DocumentParserService()
yake_service = YAKEService()


@router.post("/upload_pdf", response_model=UploadResponse, tags=["Document Upload"])
async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    """
    Upload and process a PDF or DOCX document.
    
    Features:
    - Validates file type, size, and page count
    - Extracts text and metadata from document
    - Creates session with document data
    - Optional YAKE keyword extraction
    - Structured error handling with fallbacks
    
    Args:
        file: Uploaded PDF or DOCX file (max 300 pages, 50MB)
        
    Returns:
        UploadResponse with session details and processed document metadata
        
    Raises:
        HTTPException: If upload or processing fails
    """
    start_time = time.time()
    
    try:
        logger.info(f"Starting document upload: {file.filename}")
        
        # Validate file extension
        if not file.filename:
            raise HTTPException(
                status_code=400, 
                detail={
                    "error_code": "NO_FILENAME",
                    "message": "No filename provided"
                }
            )
            
        file_extension = os.path.splitext(file.filename.lower())[1]
        if file_extension not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail={
                    "error_code": "UNSUPPORTED_FILE_TYPE",
                    "message": f"Unsupported file type: {file_extension}",
                    "allowed_types": settings.ALLOWED_EXTENSIONS
                }
            )
        
        # Read file content
        file_content = await file.read()
        file_size_bytes = len(file_content)
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        logger.info(f"File {file.filename} read: {file_size_mb:.2f}MB")
        
        # Validate file size
        if file_size_mb > settings.MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413, 
                detail={
                    "error_code": "FILE_TOO_LARGE",
                    "message": f"File too large: {file_size_mb:.2f}MB > {settings.MAX_FILE_SIZE_MB}MB",
                    "file_size_mb": file_size_mb,
                    "max_size_mb": settings.MAX_FILE_SIZE_MB
                }
            )
        
        # Parse document with comprehensive error handling
        try:
            parsed_document = await document_parser.parse_document(
                file_content=file_content,
                filename=file.filename,
                validate_limits=True
            )
            
            logger.info(f"Document parsed successfully: {parsed_document.parsing_stats.get('total_pages', 0)} pages, {len(parsed_document.content)} chars")
            
        except DocumentParsingError as e:
            logger.error(f"Document parsing failed for {file.filename}: {str(e)}")
            raise HTTPException(
                status_code=422,
                detail={
                    "error_code": "DOCUMENT_PARSING_ERROR",
                    "message": str(e),
                    "details": {"filename": file.filename}
                }
            )
        
        # Create YAKE index with fallback
        document_index = None
        try:
            if settings.ENABLE_YAKE_SEARCH:
                document_index = await yake_service.create_document_index(parsed_document)
                logger.info(f"YAKE index created with {len(document_index.keywords)} keywords")
        except Exception as e:
            logger.warning(f"YAKE indexing failed, continuing without index: {e}")
        
        # Create session with parsed document and index
        try:
            session_id = session_manager.create_session()
            
            # Update session with Phase 3 data
            session_manager.update_session(
                session_id,
                parsed_document=parsed_document,
                document_index=document_index,
                document_text=parsed_document.content,  # Keep for backward compatibility
                document_metadata=parsed_document.metadata.__dict__ if parsed_document.metadata else {}
            )
            
            # Record upload statistics
            session = session_manager.get_session(session_id)
            if session:
                session.record_upload(file.filename, time.time() - start_time)
            
            logger.info(f"Session created: {session_id}")
            
        except RuntimeError as e:
            logger.error(f"Session creation failed: {e}")
            raise HTTPException(
                status_code=507,  # Insufficient Storage
                detail={
                    "error_code": "SESSION_CREATION_FAILED",
                    "message": str(e),
                    "suggestion": "Server at capacity. Please try again later."
                }
            )
        
        # Create response metadata
        document_metadata = DocumentMetadata(
            filename=file.filename,
            size_bytes=file_size_bytes,
            pages=parsed_document.parsing_stats.get('total_pages', 0),
            title=parsed_document.metadata.title if parsed_document.metadata else file.filename,
            file_type=file_extension,
            upload_timestamp=datetime.utcnow()
        )
        
        processing_time = time.time() - start_time
        
        # Return successful response
        return UploadResponse(
            success=True,
            message=f"Document '{file.filename}' uploaded and processed successfully. {parsed_document.parsing_stats.get('total_pages', 0)} pages extracted.",
            session_id=session_id,
            document_metadata=document_metadata,
            processing_status="complete",
            processing_time=round(processing_time, 3),
            additional_info={
                "text_length": len(parsed_document.content),
                "memory_estimate_mb": round(parsed_document.get_memory_estimate_mb(), 2),
                "chunks_created": len(parsed_document.chunks) if parsed_document.chunks else 0,
                "keywords_extracted": len(document_index.keywords) if document_index else 0,
                "yake_enabled": settings.ENABLE_YAKE_SEARCH,
                "parser_status": await document_parser.get_parser_status()
            }
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Handle unexpected errors
        processing_time = time.time() - start_time
        logger.error(f"Unexpected error in upload_pdf for {getattr(file, 'filename', 'unknown')}: {e}")
        
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred during document processing",
                "processing_time": round(processing_time, 3)
            }
        )


@router.get("/upload/status", tags=["Document Upload"])
async def get_upload_status():
    """Get upload service status and capabilities."""
    
    parser_status = await document_parser.get_parser_status()
    session_stats = session_manager.get_session_stats()
    
    return {
        "service": "Document Upload Service",
        "status": "operational",
        "capabilities": {
            "supported_formats": parser_status["supported_formats"],
            "max_file_size_mb": parser_status["max_file_size_mb"],
            "max_pages": parser_status["max_pages"],
            "pdf_available": parser_status["pdf_available"],
            "docx_available": parser_status["docx_available"],
            "yake_enabled": parser_status["yake_enabled"]
        },
        "current_load": {
            "active_sessions": session_stats["active_sessions"],
            "max_sessions": session_stats["max_sessions"],
            "total_memory_mb": session_stats["total_memory_mb"],
            "capacity_used_percent": round((session_stats["active_sessions"] / session_stats["max_sessions"]) * 100, 1)
        },
        "yake_service": {
            "enabled": settings.ENABLE_YAKE_SEARCH,
            "max_keywords": settings.YAKE_MAX_KEYWORDS,
            "ngram_size": settings.YAKE_NGRAM_SIZE,
            "status": "operational" if settings.ENABLE_YAKE_SEARCH else "disabled"
        },
        "timestamp": datetime.utcnow().isoformat()
    }