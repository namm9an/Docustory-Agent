"""
Queue Manager for Phase 5 - Request Queuing and Capacity Management.

Manages request queuing when system is at capacity, providing:
- Concurrency control with session limits
- Request queuing with timeout handling
- Priority-based queue management
- Capacity monitoring and metrics
- Graceful degradation under load
"""

import asyncio
import time
import uuid
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.exceptions import SystemAtCapacityError, QueueFullError, RequestTimeoutError
from app.core.logging import get_logger


class QueuePriority(str, Enum):
    """Request priority levels for queue management."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class QueuedRequest:
    """Represents a queued request waiting for processing."""
    
    request_id: str
    session_id: Optional[str]
    priority: QueuePriority
    operation: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    timeout_at: datetime = field(init=False)
    callback: Optional[Callable[[], Awaitable[Any]]] = field(default=None, repr=False)
    
    def __post_init__(self):
        self.timeout_at = self.created_at + timedelta(seconds=settings.QUEUE_TIMEOUT_SECONDS)
    
    @property
    def age_seconds(self) -> float:
        """Get age of request in seconds."""
        return (datetime.utcnow() - self.created_at).total_seconds()
    
    @property
    def is_expired(self) -> bool:
        """Check if request has expired."""
        return datetime.utcnow() > self.timeout_at
    
    @property
    def time_until_timeout(self) -> float:
        """Get seconds until request times out."""
        return max(0, (self.timeout_at - datetime.utcnow()).total_seconds())


class QueueManager:
    """
    Advanced queue manager for handling request concurrency and capacity limits.
    
    Features:
    - Session-based concurrency control
    - Priority-based request queuing
    - Automatic timeout and cleanup
    - Capacity monitoring and metrics
    - Graceful degradation strategies
    """
    
    def __init__(self):
        self.logger = get_logger()
        
        # Active session tracking
        self._active_sessions: Dict[str, dict] = {}
        self._session_lock = asyncio.Lock()
        
        # Request queue management
        self._request_queue: List[QueuedRequest] = []
        self._queue_lock = asyncio.Lock()
        
        # Processing control
        self._processing_events: Dict[str, asyncio.Event] = {}
        
        # Metrics and monitoring
        self._stats = {
            "requests_queued": 0,
            "requests_processed": 0,
            "requests_timeout": 0,
            "requests_rejected": 0,
            "peak_queue_size": 0,
            "peak_active_sessions": 0
        }
        
        # Background cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and metrics."""
        async with self._session_lock:
            active_count = len(self._active_sessions)
        
        async with self._queue_lock:
            queue_size = len(self._request_queue)
            
        return {
            "active_sessions": active_count,
            "max_sessions": settings.MAX_CONCURRENT_SESSIONS,
            "capacity_utilization": (active_count / settings.MAX_CONCURRENT_SESSIONS) * 100,
            "queue_size": queue_size,
            "max_queue_size": settings.QUEUE_MAX_SIZE,
            "queue_utilization": (queue_size / settings.QUEUE_MAX_SIZE) * 100,
            "at_capacity": active_count >= settings.MAX_CONCURRENT_SESSIONS,
            "queue_full": queue_size >= settings.QUEUE_MAX_SIZE,
            "estimated_wait_minutes": max(1, queue_size // 3),
            "stats": self._stats.copy()
        }
    
    async def can_process_immediately(self, session_id: Optional[str] = None) -> bool:
        """Check if request can be processed immediately without queuing."""
        async with self._session_lock:
            active_count = len(self._active_sessions)
            
        return active_count < settings.MAX_CONCURRENT_SESSIONS
    
    async def acquire_session_slot(self, session_id: str, operation: str,
                                 request_id: Optional[str] = None,
                                 timeout_seconds: Optional[float] = None) -> bool:
        """
        Acquire a session processing slot, queuing if necessary.
        
        Returns True if slot acquired immediately, False if queued.
        Raises exception if queue is full or timeout exceeded.
        """
        request_id = request_id or str(uuid.uuid4())[:8]
        
        # Check immediate availability
        if await self.can_process_immediately():
            await self._activate_session(session_id, operation, request_id)
            return True
        
        # System at capacity - check queue availability
        async with self._queue_lock:
            if len(self._request_queue) >= settings.QUEUE_MAX_SIZE:
                self._stats["requests_rejected"] += 1
                raise QueueFullError(len(self._request_queue), settings.QUEUE_MAX_SIZE)
        
        # Queue the request
        await self._queue_request(
            request_id=request_id,
            session_id=session_id,
            operation=operation,
            priority=QueuePriority.NORMAL,
            timeout_seconds=timeout_seconds
        )
        
        # Wait for processing slot
        await self._wait_for_processing_slot(request_id)
        return False
    
    async def release_session_slot(self, session_id: str):
        """Release a session processing slot and process next queued request."""
        async with self._session_lock:
            if session_id in self._active_sessions:
                session_info = self._active_sessions.pop(session_id)
                self.logger.log_performance_metrics(
                    operation=f"session_processing:{session_info['operation']}",
                    duration_ms=(time.time() - session_info['started_at']) * 1000,
                    session_id=session_id
                )
        
        # Process next queued request
        await self._process_next_queued_request()
    
    async def queue_request(self, operation: str, priority: QueuePriority = QueuePriority.NORMAL,
                          session_id: Optional[str] = None,
                          request_id: Optional[str] = None,
                          timeout_seconds: Optional[float] = None) -> str:
        """
        Queue a request for processing with priority.
        Returns request_id for tracking.
        """
        request_id = request_id or str(uuid.uuid4())
        
        await self._queue_request(
            request_id=request_id,
            session_id=session_id,
            operation=operation,
            priority=priority,
            timeout_seconds=timeout_seconds
        )
        
        return request_id
    
    async def get_queue_position(self, request_id: str) -> Optional[int]:
        """Get position of request in queue (1-based), None if not found."""
        async with self._queue_lock:
            for i, req in enumerate(self._request_queue):
                if req.request_id == request_id:
                    return i + 1
        return None
    
    async def get_queue_info(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about queued request."""
        async with self._queue_lock:
            for i, req in enumerate(self._request_queue):
                if req.request_id == request_id:
                    return {
                        "request_id": req.request_id,
                        "position": i + 1,
                        "priority": req.priority.value,
                        "operation": req.operation,
                        "age_seconds": req.age_seconds,
                        "timeout_seconds": req.time_until_timeout,
                        "estimated_wait_minutes": max(1, i // 3)
                    }
        return None
    
    @asynccontextmanager
    async def session_processing_context(self, session_id: str, operation: str,
                                       request_id: Optional[str] = None):
        """Context manager for session processing with automatic cleanup."""
        request_id = request_id or str(uuid.uuid4())[:8]
        
        try:
            # Acquire processing slot
            queued = not await self.acquire_session_slot(session_id, operation, request_id)
            if queued:
                self.logger.log_info(
                    f"Request queued due to capacity limits",
                    session_id=session_id,
                    request_id=request_id,
                    operation=operation
                )
            
            yield {"queued": queued, "request_id": request_id}
            
        finally:
            # Always release the slot
            await self.release_session_slot(session_id)
    
    async def _activate_session(self, session_id: str, operation: str, request_id: str):
        """Activate a session for processing."""
        async with self._session_lock:
            self._active_sessions[session_id] = {
                "session_id": session_id,
                "operation": operation,
                "request_id": request_id,
                "started_at": time.time(),
                "activated_at": datetime.utcnow()
            }
            
            # Update peak tracking
            active_count = len(self._active_sessions)
            if active_count > self._stats["peak_active_sessions"]:
                self._stats["peak_active_sessions"] = active_count
            
            self.logger.log_info(
                f"Session activated for processing: {operation}",
                session_id=session_id,
                request_id=request_id,
                active_sessions=active_count
            )
    
    async def _queue_request(self, request_id: str, session_id: Optional[str],
                           operation: str, priority: QueuePriority,
                           timeout_seconds: Optional[float]):
        """Add request to queue with priority ordering."""
        timeout_seconds = timeout_seconds or settings.QUEUE_TIMEOUT_SECONDS
        
        queued_request = QueuedRequest(
            request_id=request_id,
            session_id=session_id,
            priority=priority,
            operation=operation
        )
        
        # Override timeout if specified
        if timeout_seconds != settings.QUEUE_TIMEOUT_SECONDS:
            queued_request.timeout_at = queued_request.created_at + timedelta(seconds=timeout_seconds)
        
        async with self._queue_lock:
            # Insert based on priority (higher priority first)
            priority_order = {
                QueuePriority.CRITICAL: 0,
                QueuePriority.HIGH: 1,
                QueuePriority.NORMAL: 2,
                QueuePriority.LOW: 3
            }
            
            insert_index = len(self._request_queue)
            for i, existing_req in enumerate(self._request_queue):
                if priority_order[priority] < priority_order[existing_req.priority]:
                    insert_index = i
                    break
            
            self._request_queue.insert(insert_index, queued_request)
            self._processing_events[request_id] = asyncio.Event()
            
            # Update stats
            self._stats["requests_queued"] += 1
            queue_size = len(self._request_queue)
            if queue_size > self._stats["peak_queue_size"]:
                self._stats["peak_queue_size"] = queue_size
            
            self.logger.log_info(
                f"Request queued: {operation} (position: {insert_index + 1})",
                session_id=session_id,
                request_id=request_id,
                priority=priority.value,
                queue_size=queue_size
            )
    
    async def _wait_for_processing_slot(self, request_id: str):
        """Wait for processing slot to become available."""
        if request_id not in self._processing_events:
            raise RequestTimeoutError(0, "queue_wait")
        
        try:
            # Wait for processing event with timeout
            await asyncio.wait_for(
                self._processing_events[request_id].wait(),
                timeout=settings.QUEUE_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            # Clean up timed out request
            await self._remove_from_queue(request_id)
            self._stats["requests_timeout"] += 1
            raise RequestTimeoutError(settings.QUEUE_TIMEOUT_SECONDS, "queue_wait")
        finally:
            # Clean up event
            if request_id in self._processing_events:
                del self._processing_events[request_id]
    
    async def _process_next_queued_request(self):
        """Process the next request in queue if system has capacity."""
        if not await self.can_process_immediately():
            return
        
        async with self._queue_lock:
            if not self._request_queue:
                return
            
            # Get next request (highest priority, oldest first)
            next_request = self._request_queue.pop(0)
        
        # Activate the session
        if next_request.session_id:
            await self._activate_session(
                next_request.session_id,
                next_request.operation,
                next_request.request_id
            )
        
        # Signal the waiting request
        if next_request.request_id in self._processing_events:
            self._processing_events[next_request.request_id].set()
        
        self._stats["requests_processed"] += 1
        
        self.logger.log_info(
            f"Processing queued request: {next_request.operation}",
            session_id=next_request.session_id,
            request_id=next_request.request_id,
            queue_wait_seconds=next_request.age_seconds
        )
    
    async def _remove_from_queue(self, request_id: str) -> bool:
        """Remove request from queue by ID."""
        async with self._queue_lock:
            for i, req in enumerate(self._request_queue):
                if req.request_id == request_id:
                    self._request_queue.pop(i)
                    return True
        return False
    
    async def _cleanup_loop(self):
        """Background task for cleaning up expired requests and sessions."""
        while True:
            try:
                await asyncio.sleep(30)  # Cleanup every 30 seconds
                await self._cleanup_expired_requests()
                await self._log_capacity_warnings()
            except Exception as e:
                self.logger.log_error(
                    e, "QUEUE_CLEANUP_ERROR", "system",
                    include_stack_trace=True
                )
    
    async def _cleanup_expired_requests(self):
        """Remove expired requests from queue."""
        expired_requests = []
        
        async with self._queue_lock:
            # Identify expired requests
            for i, req in enumerate(self._request_queue):
                if req.is_expired:
                    expired_requests.append((i, req))
            
            # Remove expired requests (reverse order to maintain indices)
            for i, req in reversed(expired_requests):
                self._request_queue.pop(i)
                # Signal timeout to waiting coroutines
                if req.request_id in self._processing_events:
                    # Don't set the event - let it timeout naturally
                    pass
        
        # Log cleanup if any requests were expired
        if expired_requests:
            self._stats["requests_timeout"] += len(expired_requests)
            self.logger.log_info(
                f"Cleaned up {len(expired_requests)} expired queue requests",
                expired_count=len(expired_requests)
            )
    
    async def _log_capacity_warnings(self):
        """Log warnings when system is near capacity."""
        system_status = await self.get_system_status()
        
        # Log capacity warnings
        if system_status["capacity_utilization"] >= 80:
            self.logger.log_capacity_warning(
                active_sessions=system_status["active_sessions"],
                max_sessions=system_status["max_sessions"],
                queue_size=system_status["queue_size"]
            )
    
    async def shutdown(self):
        """Gracefully shutdown the queue manager."""
        self._cleanup_task.cancel()
        
        try:
            await self._cleanup_task
        except asyncio.CancelledError:
            pass
        
        # Signal all waiting requests
        for event in self._processing_events.values():
            event.set()
        
        self.logger.log_info("Queue manager shutdown completed")


# Global queue manager instance
_queue_manager_instance = None

def get_queue_manager() -> QueueManager:
    """Get the global queue manager instance."""
    global _queue_manager_instance
    if _queue_manager_instance is None:
        _queue_manager_instance = QueueManager()
    return _queue_manager_instance