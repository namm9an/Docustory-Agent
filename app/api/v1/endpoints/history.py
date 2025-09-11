"""
Phase 4 conversation history management endpoints.

Provides endpoints for:
- Retrieving conversation history
- Clearing conversation history
- Memory statistics and management
"""

import logging
import time
from fastapi import APIRouter, HTTPException, Query, Depends
from datetime import datetime
from typing import Optional

from app.models.history import (
    ConversationHistoryResponse, ClearHistoryRequest, ClearHistoryResponse,
    MemoryStatsResponse, HistoryError
)
from app.services.memory_service import memory_service
from app.core.session import session_manager
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/history/{session_id}", response_model=ConversationHistoryResponse, tags=["Conversation History"])
async def get_conversation_history(
    session_id: str,
    max_turns: Optional[int] = Query(default=None, ge=1, le=50, description="Maximum turns to return"),
    include_stats: bool = Query(default=True, description="Include memory statistics")
) -> ConversationHistoryResponse:
    """
    Retrieve conversation history for a session.
    
    Returns the conversation history with optional statistics about
    memory usage and conversation metadata.
    """
    try:
        logger.info(f"Retrieving conversation history for session {session_id}")
        
        # Validate session exists
        session_data = session_manager.get_session(session_id)
        if not session_data:
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": "SESSION_NOT_FOUND",
                    "message": "Session not found or expired",
                    "session_id": session_id,
                    "suggestion": "Please upload a document first to create a session"
                }
            )
        
        # Get conversation history
        history_data = memory_service.get_conversation_history(session_id, max_turns)
        
        # Get memory statistics if requested
        stats = {}
        if include_stats:
            stats = memory_service.get_memory_stats(session_id)
        
        response = ConversationHistoryResponse(
            session_id=session_id,
            history=history_data,
            stats=stats,
            timestamp=datetime.utcnow()
        )
        
        logger.info(f"Retrieved {len(history_data)} conversation turns for session {session_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving conversation history for session {session_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "HISTORY_RETRIEVAL_ERROR",
                "message": "Failed to retrieve conversation history",
                "session_id": session_id,
                "details": str(e)
            }
        )


@router.post("/clear_history", response_model=ClearHistoryResponse, tags=["Conversation History"])
async def clear_conversation_history(request: ClearHistoryRequest) -> ClearHistoryResponse:
    """
    Clear conversation history for a session.
    
    Removes all conversation turns while keeping the document and session intact.
    Requires confirmation to prevent accidental clearing.
    """
    try:
        logger.info(f"Clearing conversation history for session {request.session_id}")
        
        # Validation
        if not request.confirm:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": "CONFIRMATION_REQUIRED",
                    "message": "History clearing requires explicit confirmation",
                    "session_id": request.session_id,
                    "suggestion": "Set confirm=true to proceed"
                }
            )
        
        # Validate session exists
        session_data = session_manager.get_session(request.session_id)
        if not session_data:
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": "SESSION_NOT_FOUND",
                    "message": "Session not found or expired",
                    "session_id": request.session_id
                }
            )
        
        # Clear conversation history
        turns_cleared = memory_service.clear_conversation_history(request.session_id)
        
        response = ClearHistoryResponse(
            message="Conversation history cleared successfully",
            session_id=request.session_id,
            turns_cleared=turns_cleared,
            timestamp=datetime.utcnow()
        )
        
        logger.info(f"Cleared {turns_cleared} conversation turns for session {request.session_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing conversation history for session {request.session_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "HISTORY_CLEAR_ERROR",
                "message": "Failed to clear conversation history",
                "session_id": request.session_id,
                "details": str(e)
            }
        )


@router.delete("/history/{session_id}", response_model=ClearHistoryResponse, tags=["Conversation History"])
async def delete_conversation_history(
    session_id: str,
    force: bool = Query(default=False, description="Force delete without confirmation")
) -> ClearHistoryResponse:
    """
    Delete conversation history for a session (alternative endpoint).
    
    Provides a RESTful DELETE endpoint for clearing conversation history.
    """
    try:
        # Convert to request format for reuse
        request = ClearHistoryRequest(session_id=session_id, confirm=force)
        return await clear_conversation_history(request)
        
    except Exception as e:
        logger.error(f"Error in delete conversation history for session {session_id}: {e}")
        raise


@router.get("/memory/stats", response_model=MemoryStatsResponse, tags=["Memory Management"])
async def get_memory_statistics(
    session_id: Optional[str] = Query(default=None, description="Specific session ID or all sessions"),
    include_global: bool = Query(default=True, description="Include global memory statistics")
) -> MemoryStatsResponse:
    """
    Get memory statistics for conversation management.
    
    Returns detailed statistics about memory usage, conversation turns,
    and system-wide memory management metrics.
    """
    try:
        logger.info(f"Retrieving memory statistics for session: {session_id or 'all'}")
        
        if session_id:
            # Validate session exists if specific session requested
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
            
            # Get specific session stats
            stats = memory_service.get_memory_stats(session_id)
        else:
            # Get global stats
            stats = memory_service.get_memory_stats()
        
        # Add additional context
        if include_global and not session_id:
            session_stats = session_manager.get_session_stats()
            stats.update({
                "session_manager_stats": {
                    "active_sessions": session_stats["active_sessions"],
                    "total_memory_mb": session_stats["total_memory_mb"],
                    "max_sessions": session_stats["max_sessions"]
                },
                "memory_configuration": {
                    "max_conversation_turns": settings.MAX_CONVERSATION_TURNS,
                    "max_conversation_tokens": settings.MAX_CONVERSATION_TOKENS,
                    "conversation_enabled": settings.ENABLE_CONVERSATION_MEMORY,
                    "prune_threshold": settings.CONVERSATION_PRUNE_THRESHOLD
                }
            })
        
        response = MemoryStatsResponse(
            stats=stats,
            timestamp=datetime.utcnow()
        )
        
        logger.info(f"Retrieved memory statistics: {len(stats)} data points")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving memory statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "MEMORY_STATS_ERROR",
                "message": "Failed to retrieve memory statistics",
                "details": str(e)
            }
        )


@router.post("/memory/cleanup", tags=["Memory Management"])
async def cleanup_expired_memories(
    max_age_hours: int = Query(default=24, ge=1, le=168, description="Maximum age in hours"),
    dry_run: bool = Query(default=False, description="Preview cleanup without executing")
):
    """
    Clean up expired conversation memories.
    
    Removes conversation memories that haven't been accessed within
    the specified time period. Helps manage memory usage.
    """
    try:
        logger.info(f"{'Previewing' if dry_run else 'Performing'} memory cleanup for memories older than {max_age_hours}h")
        
        if dry_run:
            # Simulate cleanup to show what would be removed
            # This is a simplified preview - in production you might want more detail
            current_memories = len(memory_service._memories)
            return {
                "success": True,
                "message": f"Preview: Would clean up memories older than {max_age_hours} hours",
                "dry_run": True,
                "current_memories": current_memories,
                "max_age_hours": max_age_hours,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            # Perform actual cleanup
            cleaned_count = memory_service.cleanup_expired_memories(max_age_hours)
            
            return {
                "success": True,
                "message": f"Memory cleanup completed",
                "memories_cleaned": cleaned_count,
                "max_age_hours": max_age_hours,
                "timestamp": datetime.utcnow().isoformat()
            }
        
    except Exception as e:
        logger.error(f"Error during memory cleanup: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "MEMORY_CLEANUP_ERROR",
                "message": "Failed to perform memory cleanup",
                "details": str(e)
            }
        )


@router.get("/memory/config", tags=["Memory Management"])
async def get_memory_configuration():
    """
    Get current memory management configuration.
    
    Returns the current settings for conversation memory management
    including limits, thresholds, and feature flags.
    """
    try:
        config = {
            "conversation_memory": {
                "enabled": settings.ENABLE_CONVERSATION_MEMORY,
                "max_turns": settings.MAX_CONVERSATION_TURNS,
                "max_tokens": settings.MAX_CONVERSATION_TOKENS,
                "prune_threshold": settings.CONVERSATION_PRUNE_THRESHOLD,
                "token_estimation_ratio": settings.TOKEN_ESTIMATION_RATIO
            },
            "session_management": {
                "session_timeout_minutes": settings.SESSION_TIMEOUT_MINUTES,
                "max_concurrent_sessions": settings.MAX_CONCURRENT_SESSIONS,
                "max_memory_per_session_mb": settings.MAX_MEMORY_PER_SESSION_MB
            },
            "feature_flags": {
                "yake_search_enabled": settings.ENABLE_YAKE_SEARCH,
                "conversation_memory_enabled": settings.ENABLE_CONVERSATION_MEMORY
            },
            "system_info": {
                "version": "4.0.0",  # Phase 4 version
                "phase": "Phase 4 - Multi-turn Conversational Memory",
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        return {
            "success": True,
            "configuration": config
        }
        
    except Exception as e:
        logger.error(f"Error retrieving memory configuration: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "CONFIG_RETRIEVAL_ERROR",
                "message": "Failed to retrieve memory configuration",
                "details": str(e)
            }
        )