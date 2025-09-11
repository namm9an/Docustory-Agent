import time
import uuid
import threading
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List, Set
from dataclasses import dataclass, field
from contextlib import contextmanager
import logging
import traceback
from collections import defaultdict

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class SessionData:
    """Enhanced data structure for session storage with Phase 3 features."""
    session_id: str
    
    # Document data (Phase 3 enhanced)
    parsed_document: Optional[Any] = None  # ParsedDocument from parser_service
    document_index: Optional[Any] = None   # DocumentIndex from yake_service
    document_chunks: Optional[List[Any]] = None  # DocumentChunk objects
    
    # Legacy fields (maintained for backward compatibility)
    document_text: Optional[str] = None
    document_metadata: Optional[Dict[str, Any]] = None
    yake_index: Optional[Dict[str, Any]] = None
    file_path: Optional[str] = None
    
    # Session lifecycle
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    last_query_at: Optional[datetime] = None
    
    # Statistics and metadata
    upload_count: int = 0
    query_count: int = 0
    total_processing_time: float = 0.0
    last_upload_filename: Optional[str] = None
    session_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_access_time(self):
        """Update last accessed timestamp."""
        self.last_accessed = datetime.utcnow()
    
    def update_query_time(self):
        """Update last query timestamp and increment query count."""
        self.last_query_at = datetime.utcnow()
        self.query_count += 1
        self.update_access_time()
    
    def record_upload(self, filename: str, processing_time: float):
        """Record document upload statistics."""
        self.upload_count += 1
        self.last_upload_filename = filename
        self.total_processing_time += processing_time
        self.update_access_time()
    
    def is_expired(self) -> bool:
        """Check if session has expired based on timeout."""
        timeout_delta = timedelta(minutes=settings.SESSION_TIMEOUT_MINUTES)
        return datetime.utcnow() - self.last_accessed > timeout_delta
    
    def get_memory_usage_mb(self) -> float:
        """Enhanced memory usage calculation including Phase 3 data."""
        try:
            size = 0
            
            # Legacy fields
            if self.document_text:
                size += len(self.document_text.encode('utf-8'))
            if self.document_metadata:
                size += len(str(self.document_metadata).encode('utf-8'))
            if self.yake_index:
                size += len(str(self.yake_index).encode('utf-8'))
            
            # Phase 3 enhanced fields
            if self.parsed_document:
                # Estimate ParsedDocument size
                size += len(str(self.parsed_document).encode('utf-8'))
            
            if self.document_index:
                # Estimate DocumentIndex size
                size += len(str(self.document_index).encode('utf-8'))
            
            if self.document_chunks:
                # Estimate chunks size
                for chunk in self.document_chunks:
                    size += len(str(chunk).encode('utf-8'))
            
            # Session metadata
            if self.session_metadata:
                size += len(str(self.session_metadata).encode('utf-8'))
            
            return size / (1024 * 1024)  # Convert to MB
            
        except (AttributeError, TypeError) as e:
            logger.warning(f"Error calculating memory usage for session {self.session_id}: {e}")
            return 0.0
    
    def validate_memory_limit(self) -> bool:
        """Check if session memory usage is within limits."""
        return self.get_memory_usage_mb() <= settings.MAX_MEMORY_PER_SESSION_MB
    
    def clear_document_data(self) -> None:
        """Clear document data to free memory."""
        # Clear legacy fields
        self.document_text = None
        self.document_metadata = None
        self.yake_index = None
        self.file_path = None
        
        # Clear Phase 3 enhanced fields
        self.parsed_document = None
        self.document_index = None
        self.document_chunks = None
        
        logger.info(f"Cleared all document data for session {self.session_id}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session data to dictionary for logging."""
        return {
            "session_id": self.session_id,
            "has_document": self.document_text is not None,
            "memory_mb": round(self.get_memory_usage_mb(), 2),
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "expired": self.is_expired()
        }


class SessionManager:
    """
    Thread-safe session manager for document processing.
    
    Manages session-scoped memory storage with automatic cleanup,
    memory limits, and proper error handling.
    """
    
    def __init__(self):
        self._sessions: Dict[str, SessionData] = {}
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.time()
        self._lock = threading.RLock()  # Reentrant lock for thread safety
    
    def create_session(self, document_text: Optional[str] = None, document_metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new session and return session ID.
        
        Args:
            document_text: Extracted document text
            document_metadata: Document metadata
            
        Returns:
            Session ID string
            
        Raises:
            RuntimeError: If session creation fails
        """
        with self._lock:
            try:
                session_id = str(uuid.uuid4())
                
                # Cleanup expired sessions before creating new one
                self._cleanup_expired_sessions()
                
                # Check concurrent session limit
                if len(self._sessions) >= settings.MAX_CONCURRENT_SESSIONS:
                    # Remove oldest session
                    oldest_session_id = min(
                        self._sessions.keys(), 
                        key=lambda x: self._sessions[x].last_accessed
                    )
                    self._force_delete_session(oldest_session_id)
                    logger.warning(f"Session limit reached. Removed oldest session: {oldest_session_id}")
                
                # Create new session
                session_data = SessionData(
                    session_id=session_id,
                    document_text=document_text,
                    document_metadata=document_metadata
                )
                
                # Validate memory limits
                if not session_data.validate_memory_limit():
                    raise RuntimeError(f"Session data exceeds memory limit: {session_data.get_memory_usage_mb():.2f}MB")
                
                self._sessions[session_id] = session_data
                
                logger.info(f"Created session {session_id} with {session_data.get_memory_usage_mb():.2f}MB")
                return session_id
                
            except Exception as e:
                logger.error(f"Failed to create session: {e}")
                raise RuntimeError(f"Session creation failed: {str(e)}")
    
    def get_session(self, session_id: str) -> Optional[SessionData]:
        """
        Get session data by session ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            SessionData if found and valid, None otherwise
        """
        with self._lock:
            try:
                if session_id not in self._sessions:
                    logger.debug(f"Session not found: {session_id}")
                    return None
                
                session = self._sessions[session_id]
                
                # Check if session expired
                if session.is_expired():
                    logger.info(f"Session expired: {session_id}")
                    self._force_delete_session(session_id)
                    return None
                
                # Update access time
                session.update_access_time()
                logger.debug(f"Retrieved session {session_id}, memory: {session.get_memory_usage_mb():.2f}MB")
                return session
                
            except Exception as e:
                logger.error(f"Error retrieving session {session_id}: {e}")
                return None
    
    def update_session(self, session_id: str, **kwargs) -> bool:
        """
        Update session data with validation.
        
        Args:
            session_id: Session identifier
            **kwargs: Fields to update
            
        Returns:
            True if updated successfully, False otherwise
        """
        with self._lock:
            try:
                session = self.get_session(session_id)
                if not session:
                    return False
                
                for key, value in kwargs.items():
                    if hasattr(session, key):
                        setattr(session, key, value)
                    else:
                        logger.warning(f"Invalid session field: {key}")
                
                # Validate memory limits after update
                if not session.validate_memory_limit():
                    logger.error(f"Session {session_id} exceeds memory limit after update")
                    return False
                
                session.update_access_time()
                logger.debug(f"Updated session {session_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to update session {session_id}: {e}")
                return False
    
    def delete_session(self, session_id: str) -> bool:
        """
        Safely delete a session with proper cleanup.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted successfully, False if not found
        """
        with self._lock:
            return self._force_delete_session(session_id)
    
    def _force_delete_session(self, session_id: str) -> bool:
        """
        Internal method to delete session without acquiring lock.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted, False if not found
        """
        try:
            if session_id in self._sessions:
                session = self._sessions[session_id]
                # Clear document data first to free memory
                session.clear_document_data()
                del self._sessions[session_id]
                logger.info(f"Deleted session: {session_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return False
    
    def get_session_count(self) -> int:
        """Get current number of active sessions."""
        return len(self._sessions)
    
    def get_total_memory_usage(self) -> float:
        """Get total memory usage across all sessions in MB."""
        return sum(session.get_memory_usage_mb() for session in self._sessions.values())
    
    def _cleanup_expired_sessions(self):
        """Remove expired sessions."""
        current_time = time.time()
        
        # Only cleanup every N seconds to avoid excessive processing
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
        
        expired_sessions = [
            session_id for session_id, session in self._sessions.items()
            if session.is_expired()
        ]
        
        for session_id in expired_sessions:
            self.delete_session(session_id)
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        
        self._last_cleanup = current_time
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get comprehensive session statistics for monitoring."""
        with self._lock:
            self._cleanup_expired_sessions()
            
            return {
                "active_sessions": len(self._sessions),
                "max_sessions": settings.MAX_CONCURRENT_SESSIONS,
                "total_memory_mb": round(self.get_total_memory_usage(), 2),
                "max_memory_per_session_mb": settings.MAX_MEMORY_PER_SESSION_MB,
                "cleanup_interval_seconds": self._cleanup_interval,
                "last_cleanup": self._last_cleanup,
                "sessions_detail": [
                    session.to_dict() for session in self._sessions.values()
                ]
            }
    
    @contextmanager
    def session_context(self, session_id: str):
        """
        Context manager for safe session access with automatic cleanup.
        
        Usage:
            with session_manager.session_context(session_id) as session:
                if session:
                    # Use session safely
                    pass
        """
        session = None
        try:
            session = self.get_session(session_id)
            yield session
        except Exception as e:
            logger.error(f"Error in session context {session_id}: {e}")
            if session:
                logger.info(f"Cleaning up session {session_id} after error")
                self._force_delete_session(session_id)
            raise
        finally:
            if session:
                # Update access time on successful completion
                session.update_access_time()
    
    def cleanup_all_sessions(self) -> int:
        """
        Force cleanup of all sessions. Used for testing or shutdown.
        
        Returns:
            Number of sessions cleaned up
        """
        with self._lock:
            count = len(self._sessions)
            session_ids = list(self._sessions.keys())
            
            for session_id in session_ids:
                self._force_delete_session(session_id)
            
            logger.info(f"Cleaned up all {count} sessions")
            return count


# Global session manager instance
session_manager = SessionManager()