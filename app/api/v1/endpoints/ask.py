import time
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import StreamingResponse, Response
from typing import Optional
import json

from app.models.ask import (
    AskRequest, VoiceAskRequest, AskResponse, AskError, 
    SessionStatus, ModelInfo, StreamingAskResponse
)
from app.core.session import session_manager
from app.services.qwen_client import qwen_client
from app.services.stt import whisper_client
from app.services.tts import xtts_tts_client as xtts_client
from app.services.yake_service import YAKEService
from app.services.memory_service import memory_service
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize YAKE service for enhanced context
yake_service = YAKEService()


@router.post("/ask", response_model=AskResponse, tags=["Q&A"])
async def ask_question(request: AskRequest) -> AskResponse:
    """
    Ask a question about the uploaded document.
    
    Process text query using Qwen 2.5 with document context.
    Optionally return voice response using Vibe Voice TTS.
    
    Features:
    - Robust error handling with fallbacks
    - Session validation and memory management
    - Optional voice synthesis
    - Structured logging and monitoring
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing question for session {request.session_id}: {request.query[:100]}...")
        
        # Validate and get session with context manager for safety
        with session_manager.session_context(request.session_id) as session:
            if not session:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error_code": "SESSION_NOT_FOUND",
                        "message": "Session not found or expired. Please upload a document first.",
                        "session_id": request.session_id,
                        "suggestion": "Upload a document to create a new session"
                    }
                )
            
            # Check if document is loaded (support both legacy and Phase 3)
            document_available = (
                session.document_text is not None or 
                session.parsed_document is not None
            )
            
            if not document_available:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error_code": "NO_DOCUMENT",
                        "message": "No document found in session. Please upload a document first.",
                        "session_id": request.session_id
                    }
                )
            
            # Phase 4: Enhanced context injection with conversation memory
            if request.conversation and settings.ENABLE_CONVERSATION_MEMORY:
                # Get document context first
                document_context = await _get_document_context(session, request.query)
                
                # Create enhanced context with conversation memory
                enhanced_context, memory_metadata = memory_service.create_enhanced_context(
                    session_id=request.session_id,
                    document_context=document_context,
                    user_question=request.query
                )
            else:
                # Stateless mode - no conversation memory
                enhanced_context = await _get_document_context(session, request.query)
                memory_metadata = {
                    "has_conversation_memory": False,
                    "conversation_tokens": 0,
                    "memory_enabled": False,
                    "turns_included": 0
                }
            
            # Update query count and time
            session.update_query_time()
            
            # Get answer from Qwen with fallback handling
            try:
                qwen_response = await qwen_client.ask_question(
                    question=request.query,
                    context=enhanced_context,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature
                )
                
                logger.info(f"Qwen response generated successfully for session {request.session_id}")
                
            except Exception as e:
                logger.error(f"Qwen API failed for session {request.session_id}: {e}")
                
                # Fallback response when Qwen fails
                fallback_answer = (
                    f"I apologize, but I'm currently experiencing technical difficulties processing your question: '{request.query}'. "
                    f"The document analysis service is temporarily unavailable. Please try again in a moment. "
                    f"Your session and document are still available."
                )
                
                qwen_response = type('MockResponse', (), {
                    'answer': fallback_answer,
                    'model': 'fallback',
                    'usage': {'tokens': len(request.query.split())},
                    'metadata': {'error': str(e), 'fallback': True}
                })()
            
            # Prepare response metadata
            processing_time = time.time() - start_time
            metadata = {
                "model": qwen_response.model,
                "tokens_used": qwen_response.usage.get("tokens", 0) if qwen_response.usage else 0,
                "context_length": len(enhanced_context.split()) if isinstance(enhanced_context, str) else 0,
                "session_memory_mb": round(session.get_memory_usage_mb(), 2),
                "query_length": len(request.query),
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "conversation_enabled": request.conversation and settings.ENABLE_CONVERSATION_MEMORY,
                "session_stats": {
                    "query_count": session.query_count,
                    "upload_count": session.upload_count,
                    "last_upload": session.last_upload_filename
                }
            }
            
            # Handle voice response if requested
            voice_available = False
            voice_url = None
            
            if request.voice_enabled:
                try:
                    # Get voice configuration with defaults
                    voice_config = request.voice_config or {}
                    voice = voice_config.get("voice", "default")
                    speed = voice_config.get("speed", 1.0)
                    pitch = voice_config.get("pitch", 1.0)
                    format_type = voice_config.get("format", "mp3")
                    
                    # Validate voice parameters
                    if not xtts_client.validate_voice_parameters(voice, speed, pitch):
                        logger.warning(f"Invalid voice parameters for session {request.session_id}: {voice_config}")
                        metadata["voice_error"] = "Invalid voice parameters, using defaults"
                        voice = "default"
                        speed = 1.0
                        pitch = 1.0
                    
                    # Generate voice response with fallback
                    try:
                        tts_response = await xtts_client.synthesize_speech(
                            text=qwen_response.answer,
                            voice=voice,
                            speed=speed,
                            pitch=pitch,
                            format=format_type
                        )
                        voice_available = True
                        # In production, save audio file and return URL
                        voice_url = f"/api/v1/audio/{request.session_id}/response.{format_type}"
                        metadata["voice_duration"] = tts_response.duration
                        metadata["voice_format"] = format_type
                        
                        logger.info(f"Voice response generated successfully for session {request.session_id}")
                        
                    except Exception as voice_e:
                        logger.error(f"Voice synthesis failed for session {request.session_id}: {voice_e}")
                        metadata["voice_error"] = f"Voice synthesis failed: {str(voice_e)}"
                        # Continue without voice response - graceful degradation
                        
                except Exception as e:
                    logger.error(f"Voice processing error for session {request.session_id}: {e}")
                    metadata["voice_error"] = f"Voice processing error: {str(e)}"
            
            # Add conversation turn to memory (Phase 4)
            conversation_turn = None
            if request.conversation and settings.ENABLE_CONVERSATION_MEMORY:
                conversation_turn = memory_service.add_conversation_turn(
                    session_id=request.session_id,
                    user_message=request.query,
                    assistant_response=qwen_response.answer,
                    processing_time=processing_time,
                    context_method="conversation_memory"
                )
            
            # Create successful response
            response = AskResponse(
                success=True,
                answer=qwen_response.answer,
                session_id=request.session_id,
                query=request.query,
                conversation_turn_id=conversation_turn.turn_id if conversation_turn else 0,
                metadata=metadata,
                memory_metadata=memory_metadata,
                voice_available=voice_available,
                voice_url=voice_url,
                processing_time=round(processing_time, 3)
            )
            
            logger.info(f"Question processed successfully for session {request.session_id} in {processing_time:.3f}s (Turn: {response.conversation_turn_id})")
            return response
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Handle unexpected errors
        processing_time = time.time() - start_time
        logger.error(f"Unexpected error in ask_question for session {getattr(request, 'session_id', 'unknown')}: {e}")
        
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "PROCESSING_ERROR",
                "message": "An unexpected error occurred while processing your question",
                "session_id": getattr(request, "session_id", None),
                "processing_time": round(processing_time, 3),
                "suggestion": "Please try again. If the problem persists, upload your document again."
            }
        )


@router.post("/ask_voice", response_model=AskResponse, tags=["Q&A"])
async def ask_question_voice(
    session_id: str = Form(...),
    voice_enabled: bool = Form(default=True),
    transcription_language: str = Form(default="auto"),
    stream_response: bool = Form(default=False),
    max_tokens: int = Form(default=1024),
    temperature: float = Form(default=0.7),
    voice_file: UploadFile = File(...),
    voice_config: Optional[str] = Form(default=None)
) -> AskResponse:
    """
    Ask a question using voice input with comprehensive error handling.
    
    Transcribe voice to text using Whisper, then process with Qwen.
    Optionally return voice response using Vibe Voice TTS.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing voice question for session {session_id}: {voice_file.filename}")
        
        # Validate session exists
        session_data = session_manager.get_session(session_id)
        if not session_data:
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": "SESSION_NOT_FOUND",
                    "message": "Session not found or expired. Please upload a document first.",
                    "session_id": session_id
                }
            )
        
        # Validate audio file
        if not whisper_client.validate_audio_file(voice_file.filename):
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": "INVALID_AUDIO_FORMAT",
                    "message": f"Unsupported audio format: {voice_file.filename}",
                    "supported_formats": whisper_client.get_supported_formats(),
                    "filename": voice_file.filename
                }
            )
        
        # Read and validate audio file size
        audio_content = await voice_file.read()
        audio_size_mb = len(audio_content) / (1024 * 1024)
        
        if audio_size_mb > 25:  # 25MB limit for audio files
            raise HTTPException(
                status_code=413,
                detail={
                    "error_code": "AUDIO_FILE_TOO_LARGE",
                    "message": f"Audio file too large: {audio_size_mb:.2f}MB > 25MB",
                    "file_size_mb": audio_size_mb
                }
            )
        
        # Transcribe audio to text with fallback
        try:
            stt_response = await whisper_client.transcribe_audio(
                audio_content=audio_content,
                filename=voice_file.filename,
                language=transcription_language
            )
            
            if not stt_response.text.strip():
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error_code": "NO_SPEECH_DETECTED",
                        "message": "No speech detected in audio file. Please record a clear question.",
                        "filename": voice_file.filename
                    }
                )
            
            logger.info(f"Audio transcribed successfully for session {session_id}: '{stt_response.text[:100]}...'")
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Speech transcription failed for session {session_id}: {e}")
            raise HTTPException(
                status_code=422,
                detail={
                    "error_code": "TRANSCRIPTION_FAILED",
                    "message": f"Failed to transcribe audio: {str(e)}",
                    "suggestion": "Please ensure audio is clear and in a supported format",
                    "filename": voice_file.filename
                }
            )
        
        # Parse voice configuration
        parsed_voice_config = None
        if voice_config:
            try:
                parsed_voice_config = json.loads(voice_config)
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid voice config JSON for session {session_id}: {e}")
                # Continue with default voice config
        
        # Create text request from voice input
        text_request = AskRequest(
            session_id=session_id,
            query=stt_response.text,
            voice_enabled=voice_enabled,
            voice_config=parsed_voice_config,
            stream_response=stream_response,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Process as text question
        response = await ask_question(text_request)
        
        # Add transcription metadata to response
        if response.metadata:
            response.metadata["transcription"] = {
                "original_filename": voice_file.filename,
                "confidence": stt_response.confidence,
                "detected_language": stt_response.language,
                "audio_duration": stt_response.duration,
                "audio_size_mb": round(audio_size_mb, 2),
                "transcribed_text": stt_response.text
            }
        
        logger.info(f"Voice question processed successfully for session {session_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Unexpected error in ask_question_voice for session {session_id}: {e}")
        
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "VOICE_PROCESSING_ERROR",
                "message": "An unexpected error occurred while processing your voice question",
                "session_id": session_id,
                "processing_time": round(processing_time, 3)
            }
        )


@router.get("/ask_stream/{session_id}", tags=["Q&A"])
async def ask_question_stream(session_id: str, query: str, max_tokens: int = 1024, temperature: float = 0.7):
    """Stream response for real-time Q&A with error handling."""
    
    try:
        # Validate session
        session_data = session_manager.get_session(session_id)
        if not session_data:
            raise HTTPException(
                status_code=404,
                detail="Session not found or expired"
            )
        
        # Check document availability (Phase 3 compatible)
        document_available = (
            session_data.document_text is not None or 
            session_data.parsed_document is not None
        )
        
        if not document_available:
            raise HTTPException(
                status_code=400,
                detail="No document found in session"
            )
        
        logger.info(f"Starting streaming response for session {session_id}")
        
        # Get document context for streaming
        document_context = await _get_document_context(session_data, query)
        
        async def generate_stream():
            try:
                async for chunk in qwen_client.ask_question_stream(
                    question=query,
                    context=document_context,
                    max_tokens=max_tokens,
                    temperature=temperature
                ):
                    yield f"data: {chunk}\n\n"
                    
            except Exception as e:
                logger.error(f"Streaming error for session {session_id}: {e}")
                yield f"data: Error: {str(e)}\n\n"
            finally:
                yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stream setup error for session {session_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to setup streaming response"
        )


@router.get("/session/{session_id}/status", response_model=SessionStatus, tags=["Session"])
async def get_session_status(session_id: str) -> SessionStatus:
    """Get detailed session status and information with error handling."""
    
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
        
        from datetime import timedelta
        
        expires_at = session_data.last_accessed + timedelta(minutes=settings.SESSION_TIMEOUT_MINUTES)
        
        # Enhanced document info for Phase 3
        document_info = session_data.document_metadata or {}
        
        # Add Phase 3 specific information if available
        if session_data.parsed_document:
            document_info.update({
                "enhanced_parsing": True,
                "chunk_count": len(session_data.parsed_document.chunks) if session_data.parsed_document.chunks else 0,
                "page_count": session_data.parsed_document.parsing_stats.get('total_pages', 0),
                "content_length": len(session_data.parsed_document.content)
            })
            
        if session_data.document_index:
            document_info.update({
                "yake_index": True,
                "keywords_extracted": len(session_data.document_index.keywords),
                "indexed_chunks": len(session_data.document_index.chunks)
            })

        return SessionStatus(
            session_id=session_id,
            active=True,
            created_at=session_data.created_at,
            last_accessed=session_data.last_accessed,
            expires_at=expires_at,
            has_document=(session_data.document_text is not None or session_data.parsed_document is not None),
            document_info=document_info,
            memory_usage_mb=round(session_data.get_memory_usage_mb(), 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session status for {session_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve session status"
        )


@router.get("/models/info", response_model=ModelInfo, tags=["System"])
async def get_model_info() -> ModelInfo:
    """Get information about all AI models and system status."""
    
    try:
        session_stats = session_manager.get_session_stats()
        
        return ModelInfo(
            qwen_model=qwen_client.get_model_info(),
            whisper_model=whisper_client.get_model_info(),
            xtts_model=xtts_client.get_model_info(),
            session_stats=session_stats
        )
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve model information"
        )


@router.delete("/session/{session_id}", tags=["Session"])
async def delete_session(session_id: str):
    """Manually delete a session and free up memory."""
    
    try:
        success = session_manager.delete_session(session_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": "SESSION_NOT_FOUND",
                    "message": "Session not found",
                    "session_id": session_id
                }
            )
        
        return {
            "success": True,
            "message": f"Session {session_id} deleted successfully",
            "session_id": session_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to delete session"
        )


async def _get_document_context(session, query: str) -> str:
    """
    Get document context using intelligent chunk selection.
    
    Uses YAKE-based relevance scoring with fallbacks for optimal context injection.
    Returns only the document context string for use with memory service.
    """
    try:
        # Phase 3: Use parsed document and index if available
        if session.parsed_document and session.document_index:
            try:
                # Search for relevant chunks using YAKE index
                search_results = await yake_service.search_document(
                    document_index=session.document_index,
                    query=query,
                    max_results=5  # Get top 5 most relevant chunks
                )
                
                if search_results:
                    # Combine relevant chunks for context
                    relevant_chunks = []
                    total_tokens = 0
                    max_context_tokens = 3000  # Leave room for question and response
                    
                    for result in search_results:
                        chunk_tokens = len(result.content.split())
                        if total_tokens + chunk_tokens > max_context_tokens:
                            break
                        relevant_chunks.append(result.content)
                        total_tokens += chunk_tokens
                    
                    enhanced_context = "\n\n".join(relevant_chunks)
                    return enhanced_context
                    
            except Exception as e:
                logger.warning(f"YAKE-enhanced context failed, falling back: {e}")
        
        # Fallback 1: Use document chunks if available (Phase 3 without YAKE)
        if session.parsed_document and session.parsed_document.chunks:
            try:
                # Use simple text matching for chunk relevance
                query_words = query.lower().split()
                chunk_scores = []
                
                for i, chunk in enumerate(session.parsed_document.chunks):
                    content_lower = chunk.content.lower()
                    score = sum(1 for word in query_words if word in content_lower)
                    chunk_scores.append((score, i, chunk))
                
                # Sort by relevance and take top chunks
                chunk_scores.sort(key=lambda x: x[0], reverse=True)
                
                relevant_chunks = []
                total_tokens = 0
                max_context_tokens = 3000
                
                for score, idx, chunk in chunk_scores[:10]:  # Check top 10 scored chunks
                    chunk_tokens = len(chunk.content.split())
                    if total_tokens + chunk_tokens > max_context_tokens:
                        break
                    if score > 0:  # Only include chunks with some relevance
                        relevant_chunks.append(chunk.content)
                        total_tokens += chunk_tokens
                
                if relevant_chunks:
                    enhanced_context = "\n\n".join(relevant_chunks)
                    return enhanced_context
                    
            except Exception as e:
                logger.warning(f"Chunk-based context failed, falling back: {e}")
        
        # Fallback 2: Use legacy document text (backward compatibility)
        if session.document_text:
            # Truncate if too long for context window
            context_words = session.document_text.split()
            max_context_tokens = 3000
            
            if len(context_words) > max_context_tokens:
                # Try to find relevant sections
                query_words = query.lower().split()
                
                # Simple sliding window approach to find most relevant section
                best_score = 0
                best_start = 0
                window_size = max_context_tokens // 2
                
                for i in range(0, len(context_words) - window_size, window_size // 4):
                    window_text = " ".join(context_words[i:i + window_size]).lower()
                    score = sum(1 for word in query_words if word in window_text)
                    
                    if score > best_score:
                        best_score = score
                        best_start = i
                
                # Use best window with some padding
                start = max(0, best_start - window_size // 4)
                end = min(len(context_words), start + max_context_tokens)
                enhanced_context = " ".join(context_words[start:end])
            else:
                enhanced_context = session.document_text
            
            return enhanced_context
        
        # Final fallback: empty context with error
        logger.error(f"No document context available for session {session.session_id}")
        return ""
        
    except Exception as e:
        logger.error(f"Document context generation failed for session {session.session_id}: {e}")
        # Return basic context as final fallback
        fallback_context = getattr(session, 'document_text', '') or ""
        return fallback_context[:10000]  # Limit to prevent issues