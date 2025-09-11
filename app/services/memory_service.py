"""
Conversation Memory Service for Phase 4 - Multi-turn Conversational Memory

Manages conversation history within sessions, including:
- Turn-by-turn conversation storage
- Intelligent memory pruning based on token limits
- Context injection with conversation history
- Memory retrieval and management
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Represents a single conversation turn (user question + assistant response)."""
    
    turn_id: int
    user_message: str
    assistant_response: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    user_message_tokens: int = 0
    assistant_tokens: int = 0
    processing_time: float = 0.0
    context_method: str = "unknown"  # How context was generated for this turn
    
    def get_total_tokens(self) -> int:
        """Get total token count for this turn."""
        return self.user_message_tokens + self.assistant_tokens
    
    def estimate_tokens(self) -> int:
        """Estimate token count based on character length."""
        user_tokens = len(self.user_message) / settings.TOKEN_ESTIMATION_RATIO
        assistant_tokens = len(self.assistant_response) / settings.TOKEN_ESTIMATION_RATIO
        return int(user_tokens + assistant_tokens)
    
    def to_context_string(self) -> str:
        """Convert turn to context string for injection."""
        return f"User: {self.user_message}\nAssistant: {self.assistant_response}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert turn to dictionary for API responses."""
        return {
            "turn_id": self.turn_id,
            "user": self.user_message,
            "assistant": self.assistant_response,
            "timestamp": self.timestamp.isoformat(),
            "tokens": self.get_total_tokens() or self.estimate_tokens(),
            "processing_time": self.processing_time,
            "context_method": self.context_method
        }


@dataclass
class ConversationMemory:
    """Manages conversation memory for a session."""
    
    session_id: str
    turns: List[ConversationTurn] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    total_turns: int = 0  # Total turns ever created (not just current)
    
    def add_turn(self, user_message: str, assistant_response: str, 
                 processing_time: float = 0.0, context_method: str = "unknown") -> ConversationTurn:
        """Add a new conversation turn."""
        self.total_turns += 1
        turn = ConversationTurn(
            turn_id=self.total_turns,
            user_message=user_message,
            assistant_response=assistant_response,
            processing_time=processing_time,
            context_method=context_method
        )
        
        self.turns.append(turn)
        self.last_updated = datetime.utcnow()
        
        # Auto-prune if needed
        if len(self.turns) >= settings.CONVERSATION_PRUNE_THRESHOLD:
            self._auto_prune()
        
        logger.debug(f"Added turn {turn.turn_id} to session {self.session_id}")
        return turn
    
    def get_recent_turns(self, max_turns: int = None) -> List[ConversationTurn]:
        """Get recent conversation turns."""
        if max_turns is None:
            max_turns = settings.MAX_CONVERSATION_TURNS
        return self.turns[-max_turns:] if self.turns else []
    
    def get_conversation_context(self, max_tokens: int = None) -> str:
        """Get conversation context as formatted string within token limit."""
        if max_tokens is None:
            max_tokens = settings.MAX_CONVERSATION_TOKENS // 2  # Reserve half for document context
        
        context_parts = []
        current_tokens = 0
        
        # Start from most recent turns and work backwards
        for turn in reversed(self.turns):
            turn_tokens = turn.get_total_tokens() or turn.estimate_tokens()
            
            if current_tokens + turn_tokens > max_tokens:
                break
                
            context_parts.insert(0, turn.to_context_string())
            current_tokens += turn_tokens
        
        if context_parts:
            return "\n\n".join(context_parts)
        return ""
    
    def estimate_total_tokens(self) -> int:
        """Estimate total tokens in conversation memory."""
        return sum(turn.get_total_tokens() or turn.estimate_tokens() for turn in self.turns)
    
    def clear_history(self) -> int:
        """Clear conversation history and return number of turns cleared."""
        cleared_count = len(self.turns)
        self.turns.clear()
        self.last_updated = datetime.utcnow()
        logger.info(f"Cleared {cleared_count} turns from session {self.session_id}")
        return cleared_count
    
    def _auto_prune(self) -> int:
        """Auto-prune oldest turns to stay within limits."""
        if len(self.turns) <= settings.MAX_CONVERSATION_TURNS:
            return 0
        
        # Calculate how many turns to remove
        turns_to_remove = len(self.turns) - settings.MAX_CONVERSATION_TURNS
        
        # Remove oldest turns
        removed_turns = self.turns[:turns_to_remove]
        self.turns = self.turns[turns_to_remove:]
        
        logger.info(f"Auto-pruned {len(removed_turns)} turns from session {self.session_id}")
        return len(removed_turns)
    
    def prune_by_token_limit(self, max_tokens: int = None) -> int:
        """Prune turns to stay within token limit."""
        if max_tokens is None:
            max_tokens = settings.MAX_CONVERSATION_TOKENS
        
        current_tokens = self.estimate_total_tokens()
        if current_tokens <= max_tokens:
            return 0
        
        pruned_count = 0
        while self.turns and self.estimate_total_tokens() > max_tokens:
            removed_turn = self.turns.pop(0)  # Remove oldest
            pruned_count += 1
            logger.debug(f"Pruned turn {removed_turn.turn_id} from session {self.session_id}")
        
        if pruned_count > 0:
            logger.info(f"Token-pruned {pruned_count} turns from session {self.session_id}")
        
        return pruned_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get conversation memory statistics."""
        return {
            "session_id": self.session_id,
            "total_turns_ever": self.total_turns,
            "current_turns": len(self.turns),
            "estimated_tokens": self.estimate_total_tokens(),
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "memory_age_minutes": (datetime.utcnow() - self.created_at).total_seconds() / 60
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary for API responses."""
        return {
            "session_id": self.session_id,
            "turns": [turn.to_dict() for turn in self.turns],
            "stats": self.get_stats()
        }


class MemoryService:
    """Service for managing conversation memory across sessions."""
    
    def __init__(self):
        self._memories: Dict[str, ConversationMemory] = {}
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.time()
    
    def get_or_create_memory(self, session_id: str) -> ConversationMemory:
        """Get existing memory or create new one for session."""
        if session_id not in self._memories:
            self._memories[session_id] = ConversationMemory(session_id=session_id)
            logger.debug(f"Created new conversation memory for session {session_id}")
        
        return self._memories[session_id]
    
    def add_conversation_turn(self, session_id: str, user_message: str, 
                            assistant_response: str, processing_time: float = 0.0,
                            context_method: str = "unknown") -> ConversationTurn:
        """Add a conversation turn to session memory."""
        memory = self.get_or_create_memory(session_id)
        return memory.add_turn(user_message, assistant_response, processing_time, context_method)
    
    def get_conversation_context(self, session_id: str, max_tokens: int = None) -> str:
        """Get conversation context for a session."""
        if session_id not in self._memories:
            return ""
        
        return self._memories[session_id].get_conversation_context(max_tokens)
    
    def get_conversation_history(self, session_id: str, max_turns: int = None) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        if session_id not in self._memories:
            return []
        
        turns = self._memories[session_id].get_recent_turns(max_turns)
        return [turn.to_dict() for turn in turns]
    
    def clear_conversation_history(self, session_id: str) -> int:
        """Clear conversation history for a session."""
        if session_id not in self._memories:
            return 0
        
        return self._memories[session_id].clear_history()
    
    def delete_session_memory(self, session_id: str) -> bool:
        """Delete all memory for a session."""
        if session_id in self._memories:
            del self._memories[session_id]
            logger.info(f"Deleted conversation memory for session {session_id}")
            return True
        return False
    
    def get_memory_stats(self, session_id: str = None) -> Dict[str, Any]:
        """Get memory statistics for a session or all sessions."""
        if session_id:
            if session_id in self._memories:
                return self._memories[session_id].get_stats()
            return {}
        
        # Global stats
        total_sessions = len(self._memories)
        total_turns = sum(len(memory.turns) for memory in self._memories.values())
        total_tokens = sum(memory.estimate_total_tokens() for memory in self._memories.values())
        
        return {
            "total_sessions_with_memory": total_sessions,
            "total_conversation_turns": total_turns,
            "total_estimated_tokens": total_tokens,
            "average_turns_per_session": total_turns / total_sessions if total_sessions > 0 else 0,
            "memory_service_status": "operational",
            "last_cleanup": self._last_cleanup
        }
    
    def create_enhanced_context(self, session_id: str, document_context: str, 
                              user_question: str) -> Tuple[str, Dict[str, Any]]:
        """
        Create enhanced context combining conversation history and document context.
        
        Returns:
            Tuple of (enhanced_context, metadata)
        """
        # Get conversation context
        conversation_context = self.get_conversation_context(session_id, 
            max_tokens=settings.MAX_CONVERSATION_TOKENS // 2)
        
        # Calculate token usage
        conversation_tokens = len(conversation_context) / settings.TOKEN_ESTIMATION_RATIO
        document_tokens = len(document_context) / settings.TOKEN_ESTIMATION_RATIO
        question_tokens = len(user_question) / settings.TOKEN_ESTIMATION_RATIO
        
        # Build enhanced context
        context_parts = []
        
        if conversation_context and settings.ENABLE_CONVERSATION_MEMORY:
            context_parts.append("Previous conversation:")
            context_parts.append(conversation_context)
            context_parts.append("\n" + "="*50 + "\n")
        
        if document_context:
            context_parts.append("Document context:")
            context_parts.append(document_context)
            context_parts.append("\n" + "="*50 + "\n")
        
        context_parts.append(f"Current question: {user_question}")
        
        enhanced_context = "\n".join(context_parts)
        
        # Metadata about context creation
        metadata = {
            "has_conversation_memory": bool(conversation_context),
            "conversation_tokens": int(conversation_tokens),
            "document_tokens": int(document_tokens),
            "question_tokens": int(question_tokens),
            "total_context_tokens": int(conversation_tokens + document_tokens + question_tokens),
            "memory_enabled": settings.ENABLE_CONVERSATION_MEMORY,
            "turns_included": len(self._memories.get(session_id, ConversationMemory("")).turns)
        }
        
        return enhanced_context, metadata
    
    def cleanup_expired_memories(self, max_age_hours: int = 24) -> int:
        """Clean up old conversation memories."""
        current_time = time.time()
        
        # Only cleanup periodically
        if current_time - self._last_cleanup < self._cleanup_interval:
            return 0
        
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        expired_sessions = []
        
        for session_id, memory in self._memories.items():
            if memory.last_updated < cutoff_time:
                expired_sessions.append(session_id)
        
        # Remove expired memories
        for session_id in expired_sessions:
            del self._memories[session_id]
        
        self._last_cleanup = current_time
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired conversation memories")
        
        return len(expired_sessions)


# Global memory service instance
memory_service = MemoryService()