import time
import logging
from fastapi import APIRouter, HTTPException, Query
from datetime import datetime, timedelta
from typing import Optional, List
from pydantic import BaseModel, Field

from app.core.session import session_manager
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


class SessionInfo(BaseModel):
    """Detailed session information model."""
    session_id: str = Field(..., description="Session identifier")
    active: bool = Field(..., description="Whether session is active")
    created_at: datetime = Field(..., description="Session creation time")
    last_accessed: datetime = Field(..., description="Last access time")
    last_query_at: Optional[datetime] = Field(default=None, description="Last query time")
    expires_at: datetime = Field(..., description="Session expiration time")
    
    # Document information
    has_document: bool = Field(..., description="Whether document is loaded")
    has_parsed_document: bool = Field(default=False, description="Whether Phase 3 parsed document exists")
    has_yake_index: bool = Field(default=False, description="Whether YAKE index exists")
    document_chunks: int = Field(default=0, description="Number of document chunks")
    
    # Statistics
    upload_count: int = Field(default=0, description="Number of uploads in session")
    query_count: int = Field(default=0, description="Number of queries processed")
    total_processing_time: float = Field(default=0.0, description="Total processing time")
    last_upload_filename: Optional[str] = Field(default=None, description="Last uploaded filename")
    
    # Memory and performance
    memory_usage_mb: float = Field(..., description="Current memory usage in MB")
    is_expired: bool = Field(..., description="Whether session is expired")


class SessionsListResponse(BaseModel):
    """Response model for session listing."""
    success: bool = Field(default=True, description="Request success")
    total_sessions: int = Field(..., description="Total number of active sessions")
    sessions: List[SessionInfo] = Field(..., description="List of session information")
    system_stats: dict = Field(..., description="System-wide session statistics")


class SessionCleanupResponse(BaseModel):
    """Response model for cleanup operations."""
    success: bool = Field(default=True, description="Operation success")
    message: str = Field(..., description="Result message")
    sessions_cleaned: int = Field(..., description="Number of sessions cleaned")
    memory_freed_mb: float = Field(..., description="Memory freed in MB")
    cleanup_type: str = Field(..., description="Type of cleanup performed")


@router.get("/sessions", response_model=SessionsListResponse, tags=["Session Management"])
async def list_all_sessions(
    include_expired: bool = Query(default=False, description="Include expired sessions"),
    sort_by: str = Query(default="created_at", description="Sort field (created_at, last_accessed, memory_usage_mb)"),
    limit: int = Query(default=50, ge=1, le=100, description="Maximum number of sessions to return")
) -> SessionsListResponse:
    """
    List all active sessions with detailed information.
    
    Provides comprehensive session monitoring and management capabilities.
    """
    try:
        logger.info(f"Listing sessions: include_expired={include_expired}, sort_by={sort_by}, limit={limit}")
        
        # Get system statistics
        system_stats = session_manager.get_session_stats()
        
        # Get all session information
        session_infos = []
        
        with session_manager._lock:
            for session_id, session_data in session_manager._sessions.items():
                # Skip expired sessions if not requested
                if not include_expired and session_data.is_expired():
                    continue
                
                # Calculate expiration time
                expires_at = session_data.last_accessed + timedelta(minutes=settings.SESSION_TIMEOUT_MINUTES)
                
                # Create session info
                session_info = SessionInfo(
                    session_id=session_id,
                    active=not session_data.is_expired(),
                    created_at=session_data.created_at,
                    last_accessed=session_data.last_accessed,
                    last_query_at=session_data.last_query_at,
                    expires_at=expires_at,
                    
                    # Document information
                    has_document=session_data.document_text is not None,
                    has_parsed_document=session_data.parsed_document is not None,
                    has_yake_index=session_data.document_index is not None,
                    document_chunks=len(session_data.document_chunks) if session_data.document_chunks else 0,
                    
                    # Statistics
                    upload_count=session_data.upload_count,
                    query_count=session_data.query_count,
                    total_processing_time=session_data.total_processing_time,
                    last_upload_filename=session_data.last_upload_filename,
                    
                    # Memory and performance
                    memory_usage_mb=round(session_data.get_memory_usage_mb(), 2),
                    is_expired=session_data.is_expired()
                )
                
                session_infos.append(session_info)
        
        # Sort sessions
        if sort_by == "created_at":
            session_infos.sort(key=lambda x: x.created_at, reverse=True)
        elif sort_by == "last_accessed":
            session_infos.sort(key=lambda x: x.last_accessed, reverse=True)
        elif sort_by == "memory_usage_mb":
            session_infos.sort(key=lambda x: x.memory_usage_mb, reverse=True)
        else:
            # Default to creation time
            session_infos.sort(key=lambda x: x.created_at, reverse=True)
        
        # Apply limit
        session_infos = session_infos[:limit]
        
        logger.info(f"Listed {len(session_infos)} sessions")
        
        return SessionsListResponse(
            total_sessions=len(session_infos),
            sessions=session_infos,
            system_stats=system_stats
        )
        
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "SESSION_LIST_ERROR",
                "message": "Failed to retrieve session list",
                "details": str(e)
            }
        )


@router.post("/sessions/cleanup", response_model=SessionCleanupResponse, tags=["Session Management"])
async def cleanup_sessions(
    cleanup_type: str = Query(default="expired", description="Cleanup type: expired, all, memory_pressure"),
    force: bool = Query(default=False, description="Force cleanup even if sessions are active")
) -> SessionCleanupResponse:
    """
    Perform session cleanup operations.
    
    Supports different cleanup strategies:
    - expired: Remove only expired sessions (default)
    - all: Remove all sessions (requires force=true)
    - memory_pressure: Remove sessions based on memory usage
    """
    start_time = time.time()
    
    try:
        logger.info(f"Starting session cleanup: type={cleanup_type}, force={force}")
        
        # Get initial memory usage
        initial_memory = session_manager.get_total_memory_usage()
        initial_sessions = session_manager.get_session_count()
        
        sessions_cleaned = 0
        cleanup_message = ""
        
        if cleanup_type == "expired":
            # Clean up expired sessions
            with session_manager._lock:
                expired_sessions = [
                    session_id for session_id, session in session_manager._sessions.items()
                    if session.is_expired()
                ]
                
                for session_id in expired_sessions:
                    session_manager._force_delete_session(session_id)
                    sessions_cleaned += 1
            
            cleanup_message = f"Cleaned up {sessions_cleaned} expired sessions"
            
        elif cleanup_type == "all":
            if not force:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error_code": "FORCE_REQUIRED",
                        "message": "cleanup_type='all' requires force=true parameter",
                        "suggestion": "Add force=true to confirm deletion of all sessions"
                    }
                )
            
            # Clean up all sessions
            sessions_cleaned = session_manager.cleanup_all_sessions()
            cleanup_message = f"Cleaned up all {sessions_cleaned} sessions"
            
        elif cleanup_type == "memory_pressure":
            # Clean up sessions based on memory usage priority
            with session_manager._lock:
                sessions_by_memory = [
                    (session_id, session.get_memory_usage_mb(), session)
                    for session_id, session in session_manager._sessions.items()
                ]
                
                # Sort by memory usage (highest first) and age (oldest first)
                sessions_by_memory.sort(key=lambda x: (-x[1], x[2].last_accessed))
                
                # Target: free up 50% of current memory usage or clean expired sessions
                target_memory_reduction = initial_memory * 0.5
                memory_freed = 0.0
                
                for session_id, memory_usage, session in sessions_by_memory:
                    if memory_freed >= target_memory_reduction and not session.is_expired():
                        break
                    
                    # Always clean expired sessions, or clean by memory pressure
                    if session.is_expired() or force or memory_freed < target_memory_reduction:
                        session_manager._force_delete_session(session_id)
                        sessions_cleaned += 1
                        memory_freed += memory_usage
            
            cleanup_message = f"Cleaned up {sessions_cleaned} sessions under memory pressure"
            
        else:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": "INVALID_CLEANUP_TYPE",
                    "message": f"Invalid cleanup_type: {cleanup_type}",
                    "allowed_types": ["expired", "all", "memory_pressure"]
                }
            )
        
        # Calculate final statistics
        final_memory = session_manager.get_total_memory_usage()
        memory_freed = initial_memory - final_memory
        processing_time = time.time() - start_time
        
        logger.info(f"Session cleanup completed: {sessions_cleaned} sessions, {memory_freed:.2f}MB freed in {processing_time:.3f}s")
        
        return SessionCleanupResponse(
            message=cleanup_message,
            sessions_cleaned=sessions_cleaned,
            memory_freed_mb=round(memory_freed, 2),
            cleanup_type=cleanup_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session cleanup error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "CLEANUP_ERROR",
                "message": "Session cleanup failed",
                "details": str(e)
            }
        )


@router.get("/sessions/{session_id}", response_model=SessionInfo, tags=["Session Management"])
async def get_session_details(session_id: str) -> SessionInfo:
    """Get detailed information about a specific session."""
    
    try:
        session_data = session_manager.get_session(session_id)
        if not session_data:
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": "SESSION_NOT_FOUND",
                    "message": "Session not found or expired",
                    "session_id": session_id
                }
            )
        
        # Calculate expiration time
        expires_at = session_data.last_accessed + timedelta(minutes=settings.SESSION_TIMEOUT_MINUTES)
        
        return SessionInfo(
            session_id=session_id,
            active=not session_data.is_expired(),
            created_at=session_data.created_at,
            last_accessed=session_data.last_accessed,
            last_query_at=session_data.last_query_at,
            expires_at=expires_at,
            
            # Document information
            has_document=session_data.document_text is not None,
            has_parsed_document=session_data.parsed_document is not None,
            has_yake_index=session_data.document_index is not None,
            document_chunks=len(session_data.document_chunks) if session_data.document_chunks else 0,
            
            # Statistics
            upload_count=session_data.upload_count,
            query_count=session_data.query_count,
            total_processing_time=session_data.total_processing_time,
            last_upload_filename=session_data.last_upload_filename,
            
            # Memory and performance
            memory_usage_mb=round(session_data.get_memory_usage_mb(), 2),
            is_expired=session_data.is_expired()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session {session_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve session details"
        )


@router.delete("/sessions/{session_id}", tags=["Session Management"])
async def delete_specific_session(
    session_id: str,
    force: bool = Query(default=False, description="Force delete even if session is active")
):
    """Delete a specific session with optional force parameter."""
    
    try:
        # Check if session exists first
        session_data = session_manager.get_session(session_id)
        if not session_data:
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": "SESSION_NOT_FOUND",
                    "message": "Session not found or already deleted",
                    "session_id": session_id
                }
            )
        
        # Check if session is expired or if force is requested
        if not session_data.is_expired() and not force:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": "SESSION_STILL_ACTIVE",
                    "message": "Session is still active. Use force=true to delete anyway.",
                    "session_id": session_id,
                    "expires_at": (session_data.last_accessed + timedelta(minutes=settings.SESSION_TIMEOUT_MINUTES)).isoformat()
                }
            )
        
        # Get memory usage before deletion
        memory_before = session_data.get_memory_usage_mb()
        
        # Delete the session
        success = session_manager.delete_session(session_id)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail={
                    "error_code": "DELETE_FAILED",
                    "message": "Failed to delete session",
                    "session_id": session_id
                }
            )
        
        logger.info(f"Session {session_id} deleted successfully, freed {memory_before:.2f}MB")
        
        return {
            "success": True,
            "message": f"Session {session_id} deleted successfully",
            "session_id": session_id,
            "memory_freed_mb": round(memory_before, 2),
            "forced": force
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to delete session"
        )


@router.get("/sessions/stats/system", tags=["Session Management"])
async def get_system_session_stats():
    """Get comprehensive system-wide session statistics."""
    
    try:
        stats = session_manager.get_session_stats()
        
        # Add additional computed statistics
        capacity_percent = (stats["active_sessions"] / stats["max_sessions"]) * 100 if stats["max_sessions"] > 0 else 0
        memory_per_session_avg = stats["total_memory_mb"] / stats["active_sessions"] if stats["active_sessions"] > 0 else 0
        
        enhanced_stats = {
            **stats,
            "capacity_used_percent": round(capacity_percent, 1),
            "average_memory_per_session_mb": round(memory_per_session_avg, 2),
            "memory_efficiency": round((stats["total_memory_mb"] / stats["max_memory_per_session_mb"]) * 100, 1) if stats["max_memory_per_session_mb"] > 0 else 0,
            "system_health": "healthy" if capacity_percent < 80 else "high_load" if capacity_percent < 95 else "critical"
        }
        
        return enhanced_stats
        
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve system statistics"
        )