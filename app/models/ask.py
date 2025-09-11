from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class AskRequest(BaseModel):
    """Request model for /ask endpoint with Phase 4 conversation memory support."""
    
    session_id: str = Field(..., description="Session ID from document upload")
    query: str = Field(..., min_length=1, max_length=2000, description="Text query or question")
    conversation: bool = Field(default=True, description="Include conversation memory in context")
    voice_enabled: bool = Field(default=False, description="Whether to return voice response")
    voice_config: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Voice configuration (voice, speed, pitch, format)"
    )
    stream_response: bool = Field(default=False, description="Whether to stream the response")
    max_tokens: int = Field(default=1024, ge=1, le=4096, description="Maximum tokens in response")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Response creativity (0-2)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "query": "What are the main topics covered in this document?",
                "voice_enabled": False,
                "voice_config": {
                    "voice": "professional",
                    "speed": 1.0,
                    "pitch": 1.0,
                    "format": "mp3"
                },
                "stream_response": False,
                "max_tokens": 1024,
                "temperature": 0.7
            }
        }


class VoiceAskRequest(BaseModel):
    """Request model for /ask endpoint with voice input."""
    
    session_id: str = Field(..., description="Session ID from document upload")
    voice_enabled: bool = Field(default=True, description="Whether to return voice response")
    voice_config: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Voice configuration for response"
    )
    transcription_language: str = Field(default="auto", description="Language for voice transcription")
    stream_response: bool = Field(default=False, description="Whether to stream the response")
    max_tokens: int = Field(default=1024, ge=1, le=4096, description="Maximum tokens in response")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Response creativity (0-2)")


class AskResponse(BaseModel):
    """Response model for /ask endpoint with Phase 4 conversation memory support."""
    
    success: bool = Field(..., description="Whether the request was successful")
    answer: str = Field(..., description="Text answer to the question")
    session_id: str = Field(..., description="Session ID used for the request")
    query: str = Field(..., description="Original query/question")
    conversation_turn_id: int = Field(..., description="ID of this conversation turn")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    memory_metadata: Dict[str, Any] = Field(default_factory=dict, description="Conversation memory metadata")
    voice_available: bool = Field(default=False, description="Whether voice response is available")
    voice_url: Optional[str] = Field(default=None, description="URL to download voice response")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "answer": "Based on the document, the main topics include artificial intelligence, machine learning, and natural language processing. The document covers these areas in detail with practical examples.",
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "query": "What are the main topics covered in this document?",
                "metadata": {
                    "model": "qwen2.5-14b",
                    "tokens_used": 150,
                    "confidence": 0.95,
                    "context_length": 5000
                },
                "voice_available": False,
                "voice_url": None,
                "processing_time": 1.23,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class StreamingAskResponse(BaseModel):
    """Response model for streaming /ask endpoint."""
    
    session_id: str = Field(..., description="Session ID used for the request")
    query: str = Field(..., description="Original query/question")
    stream_type: str = Field(default="text", description="Type of stream (text or audio)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Stream metadata")


class AskError(BaseModel):
    """Error response for /ask endpoint."""
    
    success: bool = Field(default=False, description="Always false for errors")
    error_code: str = Field(..., description="Error code identifier")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    session_id: Optional[str] = Field(default=None, description="Session ID if available")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error_code": "SESSION_NOT_FOUND",
                "message": "Session not found or expired",
                "details": {
                    "session_id": "invalid-session-id",
                    "suggestion": "Please upload a document first to create a session"
                },
                "session_id": "invalid-session-id",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class SessionStatus(BaseModel):
    """Session status information."""
    
    session_id: str = Field(..., description="Session identifier")
    active: bool = Field(..., description="Whether session is active")
    created_at: datetime = Field(..., description="Session creation time")
    last_accessed: datetime = Field(..., description="Last access time")
    expires_at: datetime = Field(..., description="Session expiration time")
    has_document: bool = Field(..., description="Whether document is loaded")
    document_info: Optional[Dict[str, Any]] = Field(default=None, description="Document metadata")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "active": True,
                "created_at": "2024-01-15T10:00:00Z",
                "last_accessed": "2024-01-15T10:25:00Z",
                "expires_at": "2024-01-15T10:35:00Z",
                "has_document": True,
                "document_info": {
                    "filename": "research_paper.pdf",
                    "pages": 25,
                    "title": "AI Research Advances"
                },
                "memory_usage_mb": 15.7
            }
        }


class ModelInfo(BaseModel):
    """AI model information."""
    
    qwen_model: Dict[str, Any] = Field(..., description="Qwen model information")
    whisper_model: Dict[str, Any] = Field(..., description="Whisper model information")
    xtts_model: Dict[str, Any] = Field(..., description="XTTS model information")
    session_stats: Dict[str, Any] = Field(..., description="Current session statistics")
    
    class Config:
        json_schema_extra = {
            "example": {
                "qwen_model": {
                    "model": "qwen2.5-14b",
                    "status": "configured",
                    "endpoint": "https://api.example.com/v1/qwen2.5"
                },
                "whisper_model": {
                    "model": "whisper-large",
                    "status": "configured",
                    "supported_formats": ["mp3", "wav", "m4a"]
                },
                "xtts_model": {
                    "model": "vibe-voice-streaming",
                    "status": "configured",
                    "available_voices": ["default", "professional", "casual"]
                },
                "session_stats": {
                    "active_sessions": 3,
                    "max_sessions": 15,
                    "total_memory_mb": 45.2
                }
            }
        }