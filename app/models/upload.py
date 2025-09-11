from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class UploadRequest(BaseModel):
    """Upload request validation model."""
    pass  # File handled by FastAPI UploadFile, no additional fields needed


class DocumentMetadata(BaseModel):
    """Document metadata model."""
    filename: str
    size_bytes: int
    pages: Optional[int] = None
    title: Optional[str] = None
    file_type: str
    upload_timestamp: datetime


class UploadResponse(BaseModel):
    """Upload response model with comprehensive processing information."""
    success: bool = True
    message: str = "File uploaded successfully"
    session_id: str
    document_metadata: DocumentMetadata
    processing_status: str = "pending"
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")
    additional_info: Optional[Dict[str, Any]] = Field(default=None, description="Additional processing information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Document 'research_paper.pdf' uploaded and processed successfully. 25 pages extracted.",
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "document_metadata": {
                    "filename": "research_paper.pdf",
                    "size_bytes": 2048000,
                    "pages": 25,
                    "title": "AI Research Advances",
                    "file_type": ".pdf",
                    "upload_timestamp": "2024-01-15T10:30:00Z"
                },
                "processing_status": "complete",
                "processing_time": 2.34,
                "additional_info": {
                    "text_length": 45000,
                    "memory_estimate_mb": 15.7,
                    "keywords_extracted": 20,
                    "parser_status": {
                        "pdf_available": True,
                        "docx_available": True,
                        "yake_enabled": True
                    }
                }
            }
        }
    
    
class UploadError(BaseModel):
    """Upload error response model."""
    success: bool = False
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None