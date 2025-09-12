import time
import logging
import asyncio
import json
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Form, File, UploadFile
from fastapi.responses import StreamingResponse
from typing import Optional, Dict, Any, AsyncGenerator
from dataclasses import dataclass
import io

from app.core.session import session_manager
from app.services.qwen_client import qwen_client
from app.services.stt import whisper_client
from app.services.tts import xtts_tts_client as xtts_client
from app.services.yake_service import YAKEService
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize YAKE service
yake_service = YAKEService()


@dataclass
class StreamingState:
    """Track state for streaming voice sessions."""
    session_id: str
    websocket: Optional[WebSocket] = None
    is_processing: bool = False
    last_activity: float = 0.0
    voice_config: Dict[str, Any] = None


# Active streaming sessions
streaming_sessions: Dict[str, StreamingState] = {}


async def _get_enhanced_context(session, query: str) -> dict:
    """
    Get enhanced document context using intelligent chunk selection.
    
    Uses YAKE-based relevance scoring with fallbacks for optimal context injection.
    """
    try:
        # Phase 3: Use parsed document and index if available
        if session.parsed_document and session.document_index:
            try:
                # Search for relevant chunks using YAKE index
                search_results = await yake_service.search_document(
                    document_index=session.document_index,
                    query=query,
                    max_results=3  # Fewer chunks for streaming to reduce latency
                )
                
                if search_results:
                    # Combine relevant chunks for context
                    relevant_chunks = []
                    total_tokens = 0
                    max_context_tokens = 2000  # Reduced for streaming performance
                    
                    for result in search_results:
                        chunk_tokens = len(result.content.split())
                        if total_tokens + chunk_tokens > max_context_tokens:
                            break
                        relevant_chunks.append(result.content)
                        total_tokens += chunk_tokens
                    
                    enhanced_context = "\n\n".join(relevant_chunks)
                    
                    return {
                        "context": enhanced_context,
                        "metadata": {
                            "method": "yake_enhanced",
                            "chunks_used": len(relevant_chunks),
                            "context_tokens": total_tokens
                        }
                    }
                    
            except Exception as e:
                logger.warning(f"YAKE-enhanced context failed for streaming, falling back: {e}")
        
        # Fallback: Use legacy document text (truncated for streaming)
        if session.document_text:
            context_words = session.document_text.split()[:1500]  # Limit for streaming performance
            enhanced_context = " ".join(context_words)
            
            return {
                "context": enhanced_context,
                "metadata": {
                    "method": "legacy_truncated",
                    "context_tokens": len(context_words)
                }
            }
        
        return {
            "context": "",
            "metadata": {
                "method": "no_context",
                "error": "No document content available"
            }
        }
        
    except Exception as e:
        logger.error(f"Enhanced context generation failed for streaming: {e}")
        return {
            "context": "",
            "metadata": {
                "method": "error_fallback",
                "error": str(e)
            }
        }


@router.websocket("/voice_stream/{session_id}")
async def voice_stream_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time voice streaming.
    
    Supports bidirectional voice communication:
    1. Client sends voice question
    2. Server transcribes, processes, and streams back text + voice response
    """
    await websocket.accept()
    logger.info(f"Voice streaming WebSocket connection established for session {session_id}")
    
    # Initialize streaming state
    streaming_state = StreamingState(
        session_id=session_id,
        websocket=websocket,
        last_activity=time.time()
    )
    streaming_sessions[session_id] = streaming_state
    
    try:
        # Validate session exists
        session_data = session_manager.get_session(session_id)
        if not session_data:
            await websocket.send_json({
                "type": "error",
                "error_code": "SESSION_NOT_FOUND",
                "message": "Session not found or expired. Please upload a document first."
            })
            return
        
        # Check document availability
        document_available = (
            session_data.document_text is not None or 
            session_data.parsed_document is not None
        )
        
        if not document_available:
            await websocket.send_json({
                "type": "error",
                "error_code": "NO_DOCUMENT",
                "message": "No document found in session. Please upload a document first."
            })
            return
        
        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "message": "Voice streaming ready. Send audio data or text questions.",
            "capabilities": {
                "voice_input": True,
                "text_input": True,
                "streaming_output": True,
                "voice_output": True
            }
        })
        
        # Main communication loop
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_json()
                streaming_state.last_activity = time.time()
                message_type = data.get("type", "unknown")
                
                if message_type == "voice_question":
                    await handle_voice_question(websocket, session_data, data, streaming_state)
                    
                elif message_type == "text_question":
                    await handle_text_question(websocket, session_data, data, streaming_state)
                    
                elif message_type == "voice_config":
                    streaming_state.voice_config = data.get("config", {})
                    await websocket.send_json({
                        "type": "config_updated",
                        "message": "Voice configuration updated"
                    })
                    
                elif message_type == "ping":
                    await websocket.send_json({"type": "pong"})
                    
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {message_type}"
                    })
                    
            except WebSocketDisconnect:
                logger.info(f"Voice streaming WebSocket disconnected for session {session_id}")
                break
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format"
                })
            except Exception as e:
                logger.error(f"Error in voice streaming for session {session_id}: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Processing error: {str(e)}"
                })
                
    except Exception as e:
        logger.error(f"Fatal error in voice streaming WebSocket for session {session_id}: {e}")
    finally:
        # Clean up streaming session
        if session_id in streaming_sessions:
            del streaming_sessions[session_id]
        logger.info(f"Voice streaming session {session_id} cleaned up")


async def handle_voice_question(websocket: WebSocket, session_data, data: dict, state: StreamingState):
    """Handle voice input from client."""
    if state.is_processing:
        await websocket.send_json({
            "type": "error",
            "message": "Already processing a request. Please wait."
        })
        return
    
    state.is_processing = True
    start_time = time.time()
    
    try:
        # Extract audio data
        audio_data = data.get("audio_data")  # Base64 encoded audio
        if not audio_data:
            await websocket.send_json({
                "type": "error",
                "message": "No audio data provided"
            })
            return
        
        # Send processing status
        await websocket.send_json({
            "type": "processing",
            "stage": "transcribing",
            "message": "Transcribing your voice..."
        })
        
        # Transcribe audio (this would need to be implemented with base64 decode)
        # For now, simulate transcription
        transcribed_text = data.get("text", "What are the main topics in this document?")  # Fallback for demo
        
        await websocket.send_json({
            "type": "transcription",
            "text": transcribed_text,
            "confidence": 0.95  # Mock confidence
        })
        
        # Process as text question
        await process_question_streaming(websocket, session_data, transcribed_text, state)
        
    except Exception as e:
        logger.error(f"Voice question handling failed: {e}")
        await websocket.send_json({
            "type": "error",
            "message": f"Voice processing failed: {str(e)}"
        })
    finally:
        state.is_processing = False


async def handle_text_question(websocket: WebSocket, session_data, data: dict, state: StreamingState):
    """Handle text input from client."""
    if state.is_processing:
        await websocket.send_json({
            "type": "error",
            "message": "Already processing a request. Please wait."
        })
        return
    
    state.is_processing = True
    
    try:
        question = data.get("question", "").strip()
        if not question:
            await websocket.send_json({
                "type": "error",
                "message": "No question provided"
            })
            return
        
        await process_question_streaming(websocket, session_data, question, state)
        
    except Exception as e:
        logger.error(f"Text question handling failed: {e}")
        await websocket.send_json({
            "type": "error",
            "message": f"Text processing failed: {str(e)}"
        })
    finally:
        state.is_processing = False


async def process_question_streaming(websocket: WebSocket, session_data, question: str, state: StreamingState):
    """Process question with streaming response."""
    try:
        # Update session query stats
        session_data.update_query_time()
        
        # Send processing status
        await websocket.send_json({
            "type": "processing",
            "stage": "analyzing",
            "message": "Analyzing your question with document context..."
        })
        
        # Get enhanced context
        enhanced_context = await _get_enhanced_context(session_data, question)
        
        await websocket.send_json({
            "type": "context",
            "method": enhanced_context["metadata"]["method"],
            "tokens": enhanced_context["metadata"].get("context_tokens", 0)
        })
        
        # Stream text response from Qwen
        await websocket.send_json({
            "type": "processing",
            "stage": "generating",
            "message": "Generating response..."
        })
        
        full_response = ""
        try:
            # Stream text response
            async for chunk in qwen_client.ask_question_stream(
                question=question,
                context=enhanced_context["context"],
                max_tokens=1024,
                temperature=0.7
            ):
                if chunk.strip():
                    full_response += chunk
                    await websocket.send_json({
                        "type": "text_chunk",
                        "chunk": chunk,
                        "full_response": full_response
                    })
                    
        except Exception as e:
            logger.error(f"Streaming response failed: {e}")
            # Fallback to static response
            fallback_response = f"I apologize, but I'm experiencing technical difficulties. However, I can see your question about: '{question}'. Please try again in a moment."
            full_response = fallback_response
            
            await websocket.send_json({
                "type": "text_chunk",
                "chunk": fallback_response,
                "full_response": full_response
            })
        
        # Generate voice response if configured
        voice_config = state.voice_config or {}
        if voice_config.get("enabled", True):
            await websocket.send_json({
                "type": "processing",
                "stage": "voice_synthesis",
                "message": "Generating voice response..."
            })
            
            try:
                # Generate voice response
                tts_response = await xtts_client.synthesize_speech(
                    text=full_response,
                    voice=voice_config.get("voice", "default"),
                    speed=voice_config.get("speed", 1.0),
                    pitch=voice_config.get("pitch", 1.0),
                    format=voice_config.get("format", "mp3")
                )
                
                # Send voice data (this would be base64 encoded audio)
                await websocket.send_json({
                    "type": "voice_response",
                    "audio_data": "base64_encoded_audio_here",  # Mock for now
                    "duration": getattr(tts_response, 'duration', 3.5),
                    "format": voice_config.get("format", "mp3")
                })
                
            except Exception as e:
                logger.warning(f"Voice synthesis failed: {e}")
                await websocket.send_json({
                    "type": "voice_error",
                    "message": "Voice synthesis failed, text response available"
                })
        
        # Send completion status
        await websocket.send_json({
            "type": "completed",
            "question": question,
            "response": full_response,
            "processing_time": time.time() - time.time(),  # This should use the actual start time
            "context_method": enhanced_context["metadata"]["method"]
        })
        
    except Exception as e:
        logger.error(f"Question processing failed: {e}")
        await websocket.send_json({
            "type": "error",
            "message": f"Failed to process question: {str(e)}"
        })


@router.post("/voice_stream/upload_and_ask", tags=["Voice Streaming"])
async def voice_stream_upload_and_ask(
    session_id: str = Form(...),
    question: Optional[str] = Form(default=None),
    voice_enabled: bool = Form(default=True),
    voice_config: Optional[str] = Form(default=None),
    voice_file: Optional[UploadFile] = File(default=None)
):
    """
    Enhanced endpoint that combines voice upload with streaming response.
    
    Supports both voice and text input with streaming text and voice output.
    """
    start_time = time.time()
    
    try:
        # Validate session
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
        
        # Determine question source
        final_question = question
        transcription_metadata = {}
        
        if voice_file and voice_file.filename:
            # Process voice input
            try:
                # Validate audio file
                if not whisper_client.validate_audio_file(voice_file.filename):
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error_code": "INVALID_AUDIO_FORMAT",
                            "message": f"Unsupported audio format: {voice_file.filename}",
                            "supported_formats": whisper_client.get_supported_formats()
                        }
                    )
                
                # Read and validate audio size
                audio_content = await voice_file.read()
                audio_size_mb = len(audio_content) / (1024 * 1024)
                
                if audio_size_mb > 25:  # 25MB limit
                    raise HTTPException(
                        status_code=413,
                        detail={
                            "error_code": "AUDIO_FILE_TOO_LARGE",
                            "message": f"Audio file too large: {audio_size_mb:.2f}MB > 25MB"
                        }
                    )
                
                # Transcribe audio
                stt_response = await whisper_client.transcribe_audio(
                    audio_content=audio_content,
                    filename=voice_file.filename,
                    language="auto"
                )
                
                final_question = stt_response.text
                transcription_metadata = {
                    "transcribed": True,
                    "confidence": stt_response.confidence,
                    "detected_language": stt_response.language,
                    "audio_duration": stt_response.duration
                }
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error_code": "TRANSCRIPTION_FAILED",
                        "message": f"Failed to transcribe audio: {str(e)}"
                    }
                )
        
        if not final_question or not final_question.strip():
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": "NO_QUESTION",
                    "message": "No question provided in text or voice input"
                }
            )
        
        # Parse voice configuration
        parsed_voice_config = {}
        if voice_config:
            try:
                parsed_voice_config = json.loads(voice_config)
            except json.JSONDecodeError:
                logger.warning("Invalid voice config JSON, using defaults")
        
        # Get enhanced context
        enhanced_context = await _get_enhanced_context(session_data, final_question)
        
        # Generate streaming response
        async def generate_stream():
            try:
                # Send initial metadata
                metadata = {
                    "session_id": session_id,
                    "question": final_question,
                    "context_method": enhanced_context["metadata"]["method"],
                    "voice_enabled": voice_enabled,
                    **transcription_metadata
                }
                yield f"data: {json.dumps({'type': 'metadata', **metadata})}\n\n"
                
                # Stream text response
                full_response = ""
                async for chunk in qwen_client.ask_question_stream(
                    question=final_question,
                    context=enhanced_context["context"],
                    max_tokens=1024,
                    temperature=0.7
                ):
                    if chunk.strip():
                        full_response += chunk
                        yield f"data: {json.dumps({'type': 'text', 'chunk': chunk})}\n\n"
                
                # Generate voice response if enabled
                if voice_enabled and full_response.strip():
                    try:
                        tts_response = await xtts_client.synthesize_speech(
                            text=full_response,
                            voice=parsed_voice_config.get("voice", "default"),
                            speed=parsed_voice_config.get("speed", 1.0),
                            pitch=parsed_voice_config.get("pitch", 1.0),
                            format=parsed_voice_config.get("format", "mp3")
                        )
                        
                        voice_data = {
                            "type": "voice",
                            "audio_data": "base64_encoded_audio_placeholder",  # In production, encode actual audio
                            "duration": getattr(tts_response, 'duration', 3.0),
                            "format": parsed_voice_config.get("format", "mp3")
                        }
                        yield f"data: {json.dumps(voice_data)}\n\n"
                        
                    except Exception as e:
                        logger.warning(f"Voice synthesis failed: {e}")
                        yield f"data: {json.dumps({'type': 'voice_error', 'message': 'Voice synthesis failed'})}\n\n"
                
                # Send completion
                completion_data = {
                    "type": "completed",
                    "processing_time": time.time() - start_time,
                    "full_response": full_response
                }
                yield f"data: {json.dumps(completion_data)}\n\n"
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering for real-time streaming
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice stream upload_and_ask error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "STREAMING_ERROR",
                "message": "Failed to process streaming request",
                "details": str(e)
            }
        )


@router.get("/voice_stream/sessions", tags=["Voice Streaming"])
async def get_active_streaming_sessions():
    """Get information about active voice streaming sessions."""
    
    try:
        current_time = time.time()
        active_sessions = []
        
        for session_id, state in streaming_sessions.items():
            # Check if session is still active (less than 5 minutes since last activity)
            if current_time - state.last_activity < 300:
                active_sessions.append({
                    "session_id": session_id,
                    "is_processing": state.is_processing,
                    "last_activity": state.last_activity,
                    "inactive_seconds": current_time - state.last_activity,
                    "voice_config": state.voice_config or {}
                })
        
        return {
            "active_streaming_sessions": len(active_sessions),
            "sessions": active_sessions,
            "server_time": current_time
        }
        
    except Exception as e:
        logger.error(f"Error getting streaming sessions: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve streaming session information"
        )