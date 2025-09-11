"""
Pydantic models for Phase 4 conversation history and memory management.

Defines request/response models for:
- Conversation history retrieval
- Memory management operations
- Turn-by-turn conversation data
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class ConversationTurn(BaseModel):
    """Individual conversation turn data."""
    
    turn_id: int = Field(..., description="Unique turn identifier within session")
    user: str = Field(..., description="User's message/question")
    assistant: str = Field(..., description="Assistant's response")
    timestamp: str = Field(..., description="ISO timestamp when turn was created")
    tokens: int = Field(default=0, description="Estimated token count for this turn")
    processing_time: float = Field(default=0.0, description="Time taken to process this turn")
    context_method: str = Field(default="unknown", description="Method used to generate context")
    
    class Config:
        json_schema_extra = {
            "example": {
                "turn_id": 1,
                "user": "What are the main topics in this document?",
                "assistant": "Based on the document, the main topics include machine learning algorithms, data preprocessing techniques, and model evaluation methods.",
                "timestamp": "2024-01-15T10:30:00Z",
                "tokens": 45,
                "processing_time": 1.23,
                "context_method": "yake_enhanced"
            }
        }


class ConversationHistoryResponse(BaseModel):
    """Response model for conversation history retrieval."""
    
    success: bool = Field(default=True, description="Request success status")
    session_id: str = Field(..., description="Session identifier")
    history: List[ConversationTurn] = Field(..., description="List of conversation turns")
    stats: Dict[str, Any] = Field(..., description="Memory statistics")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "history": [
                    {
                        "turn_id": 1,
                        "user": "What is machine learning?",
                        "assistant": "Machine learning is a subset of AI that enables computers to learn from data.",
                        "timestamp": "2024-01-15T10:30:00Z",
                        "tokens": 35,
                        "processing_time": 0.95,
                        "context_method": "document_only"
                    }
                ],
                "stats": {
                    "total_turns_ever": 3,
                    "current_turns": 1,
                    "estimated_tokens": 150,
                    "memory_age_minutes": 15.5
                },
                "timestamp": "2024-01-15T10:45:00Z"
            }
        }


class ClearHistoryRequest(BaseModel):
    """Request model for clearing conversation history."""
    
    session_id: str = Field(..., description="Session ID to clear history for")
    confirm: bool = Field(default=False, description="Confirmation flag to prevent accidental clearing")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "confirm": True
            }
        }


class ClearHistoryResponse(BaseModel):
    """Response model for conversation history clearing."""
    
    success: bool = Field(default=True, description="Operation success status")
    message: str = Field(..., description="Result message")
    session_id: str = Field(..., description="Session identifier")
    turns_cleared: int = Field(..., description="Number of conversation turns cleared")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Operation timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Conversation history cleared successfully",
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "turns_cleared": 5,
                "timestamp": "2024-01-15T10:45:00Z"
            }
        }


class MemoryStatsResponse(BaseModel):
    """Response model for memory statistics."""
    
    success: bool = Field(default=True, description="Request success status")
    stats: Dict[str, Any] = Field(..., description="Memory statistics")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "stats": {
                    "total_sessions_with_memory": 12,
                    "total_conversation_turns": 47,
                    "total_estimated_tokens": 2340,
                    "average_turns_per_session": 3.9,
                    "memory_service_status": "operational"
                },
                "timestamp": "2024-01-15T10:45:00Z"
            }
        }


class ConversationConfig(BaseModel):
    """Configuration for conversation memory behavior."""
    
    enabled: bool = Field(default=True, description="Enable conversation memory")
    max_turns: int = Field(default=10, ge=1, le=50, description="Maximum turns to remember")
    max_tokens: int = Field(default=4000, ge=100, le=8000, description="Maximum tokens in conversation context")
    
    class Config:
        json_schema_extra = {
            "example": {
                "enabled": True,
                "max_turns": 10,
                "max_tokens": 4000
            }
        }


class EnhancedAskRequest(BaseModel):
    """Enhanced ask request model with conversation memory support."""
    
    session_id: str = Field(..., description="Session ID from document upload")
    query: str = Field(..., min_length=1, max_length=2000, description="Text query or question")
    conversation: bool = Field(default=True, description="Include conversation memory in context")
    voice_enabled: bool = Field(default=False, description="Whether to return voice response")
    voice_config: Optional[Dict[str, Any]] = Field(default=None, description="Voice configuration")
    stream_response: bool = Field(default=False, description="Whether to stream the response")
    max_tokens: int = Field(default=1024, ge=1, le=4096, description="Maximum tokens in response")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Response creativity")
    memory_config: Optional[ConversationConfig] = Field(default=None, description="Memory configuration override")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "query": "Can you elaborate on the machine learning algorithms mentioned earlier?",
                "conversation": True,
                "voice_enabled": False,
                "stream_response": False,
                "max_tokens": 1024,
                "temperature": 0.7,
                "memory_config": {
                    "enabled": True,
                    "max_turns": 5,
                    "max_tokens": 2000
                }
            }
        }


class EnhancedAskResponse(BaseModel):
    """Enhanced ask response with conversation memory metadata."""
    
    success: bool = Field(default=True, description="Request success status")
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
                "answer": "The machine learning algorithms mentioned include neural networks, decision trees, and support vector machines. Each has specific use cases: neural networks excel at pattern recognition, decision trees provide interpretable results, and SVMs work well for classification tasks.",
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "query": "Can you elaborate on the machine learning algorithms mentioned earlier?",
                "conversation_turn_id": 3,
                "metadata": {
                    "model": "qwen2.5-14b",
                    "tokens_used": 95,
                    "context_length": 850,
                    "temperature": 0.7
                },
                "memory_metadata": {
                    "has_conversation_memory": True,
                    "conversation_tokens": 245,
                    "document_tokens": 605,
                    "total_context_tokens": 850,
                    "turns_included": 2
                },
                "voice_available": False,
                "processing_time": 1.45,
                "timestamp": "2024-01-15T10:45:00Z"
            }
        }


class MemoryPruneResponse(BaseModel):
    """Response model for memory pruning operations."""
    
    success: bool = Field(default=True, description="Operation success status")
    message: str = Field(..., description="Result message")
    session_id: str = Field(..., description="Session identifier")
    turns_pruned: int = Field(..., description="Number of turns removed")
    pruning_method: str = Field(..., description="Method used for pruning (token_limit, turn_limit, auto)")
    remaining_turns: int = Field(..., description="Number of turns remaining")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Operation timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Memory automatically pruned due to token limit",
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "turns_pruned": 3,
                "pruning_method": "token_limit",
                "remaining_turns": 7,
                "timestamp": "2024-01-15T10:45:00Z"
            }
        }


class HistoryError(BaseModel):
    """Error response for history operations."""
    
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
                "timestamp": "2024-01-15T10:45:00Z"
            }
        }