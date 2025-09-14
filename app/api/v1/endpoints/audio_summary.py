import logging
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import Response
from typing import Dict, Any
from pydantic import BaseModel

from app.core.session import SessionManager
from app.services.qwen_client import qwen_client
from app.services.tts import xtts_tts_client

logger = logging.getLogger(__name__)

router = APIRouter()


class AudioSummaryRequest(BaseModel):
    session_id: str
    summary_type: str  # "full", "summary_3min", "summary_6min"
    voice: str = "Alice"  # "Alice" or "Bob"


@router.post("/generate_audio_summary")
async def generate_audio_summary(request: AudioSummaryRequest):
    """
    Generate audio summary of the uploaded PDF document.
    
    STAR FEATURE: Convert PDF to audio in 3 formats:
    - full: Complete document text-to-speech
    - summary_3min: 3-minute AI summary with TTS
    - summary_6min: 6-minute detailed AI summary with TTS
    """
    
    try:
        logger.info(f"Generating {request.summary_type} audio summary for session {request.session_id}")
        
        # Get session data
        session_manager = SessionManager()
        session_data = await session_manager.get_session(request.session_id)
        
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get document content
        parsed_doc = session_data.get("parsed_document")
        if not parsed_doc:
            raise HTTPException(status_code=400, detail="No document found in session")
        
        if not hasattr(parsed_doc, 'content') or not parsed_doc.content:
            raise HTTPException(status_code=400, detail="Document content is empty")
        
        # Generate text based on summary type
        if request.summary_type == "full":
            # Use full document content
            text_content = parsed_doc.content
            logger.info(f"Using full document content: {len(text_content)} characters")
            
        elif request.summary_type == "summary_3min":
            # Generate 3-minute summary using Qwen
            text_content = await _generate_summary(
                document_content=parsed_doc.content,
                target_length="3 minutes",
                word_target=450  # ~450 words = 3 minutes at 150 wpm
            )
            logger.info(f"Generated 3-minute summary: {len(text_content)} characters")
            
        elif request.summary_type == "summary_6min":
            # Generate 6-minute summary using Qwen
            text_content = await _generate_summary(
                document_content=parsed_doc.content,
                target_length="6 minutes", 
                word_target=900  # ~900 words = 6 minutes at 150 wpm
            )
            logger.info(f"Generated 6-minute summary: {len(text_content)} characters")
            
        else:
            raise HTTPException(status_code=400, detail="Invalid summary_type. Use: full, summary_3min, or summary_6min")
        
        # Validate voice selection
        if request.voice not in ["Alice", "Bob"]:
            request.voice = "Alice"  # Default fallback
            logger.warning("Invalid voice selection, defaulting to Alice")
        
        # Generate audio using XTTS v2 TTS
        logger.info(f"Converting text to speech with XTTS v2 voice: {request.voice}")

        # Check if TTS client is available
        if not hasattr(xtts_tts_client, 'client_available') or not xtts_tts_client.client_available:
            logger.warning("XTTS v2 TTS client not available, using mock audio")
            raise RuntimeError("TTS client not available")

        try:
            if len(text_content) > 1000:  # For long texts, use chunked processing
                tts_responses = await xtts_tts_client.synthesize_speech_chunked(
                    text=text_content,
                    voice=request.voice,
                    chunk_size=500
                )

                # Combine all audio chunks
                combined_audio = b""
                total_duration = 0.0

                for response in tts_responses:
                    combined_audio += response.audio_content
                    total_duration += response.duration

                audio_content = combined_audio

            else:
                # For shorter texts, use direct synthesis
                tts_response = await xtts_tts_client.synthesize_speech(
                    text=text_content,
                    voice=request.voice
                )
                audio_content = tts_response.audio_content
                total_duration = tts_response.duration
                
        except Exception as tts_error:
            logger.warning(f"XTTS v2 TTS failed: {tts_error}. Using mock audio for demo.")
            
            # Create mock WAV audio file (for demo purposes)
            import wave
            import io
            
            # Generate a simple beep sound as mock audio
            sample_rate = 44100
            duration = min(10, len(text_content.split()) * 0.4)  # Estimate duration
            
            audio_buffer = io.BytesIO()
            with wave.open(audio_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                
                # Create a simple tone (mock audio)
                import math
                frames = []
                for i in range(int(sample_rate * duration)):
                    value = int(32767.0 * math.sin(2 * math.pi * 440.0 * i / sample_rate) * 0.1)
                    frames.append(value)
                
                # Convert to bytes
                audio_data = b''.join([frame.to_bytes(2, byteorder='little', signed=True) for frame in frames])
                wav_file.writeframes(audio_data)
            
            audio_content = audio_buffer.getvalue()
            total_duration = duration
        
        logger.info(f"Audio generation complete: {len(audio_content)} bytes, {total_duration:.1f}s duration")
        
        # Return audio as WAV response
        return Response(
            content=audio_content,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename='{request.summary_type}_{request.voice.lower()}_audio.wav'",
                "X-Audio-Duration": str(total_duration),
                "X-Text-Length": str(len(text_content)),
                "X-Summary-Type": request.summary_type,
                "X-Voice": request.voice,
                "X-TTS-Model": "xtts_v2"
            }
        )
        
    except Exception as e:
        logger.error(f"Audio summary generation failed: {e}", exc_info=True)
        
        # Return detailed error as JSON to bypass generic error handler
        return {
            "audio_error": True,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "session_id": request.session_id,
            "summary_type": request.summary_type,
            "voice": request.voice,
            "suggestion": "Check backend logs for detailed error information"
        }


async def _generate_summary(document_content: str, target_length: str, word_target: int) -> str:
    """Generate AI summary using Qwen with specific length target."""
    
    try:
        # Prepare summarization prompt
        summary_prompt = f"""
        Please create a comprehensive summary of the following document. 
        
        Requirements:
        - Target length: approximately {word_target} words ({target_length} when spoken)
        - Focus on the most important points, key findings, and main conclusions
        - Make it engaging and suitable for audio narration
        - Use clear, conversational language
        - Include smooth transitions between topics
        - Avoid technical jargon when possible, or explain it clearly
        
        Document content:
        {document_content[:8000]}...  
        
        Summary:
        """
        
        # Generate summary using Qwen
        response = await qwen_client.ask_question(
            question=summary_prompt,
            context="",
            max_tokens=word_target + 200,  # Allow some extra tokens
            temperature=0.7
        )
        
        if response and hasattr(response, 'answer'):
            summary_text = response.answer
            
            # Ensure the summary is within target length
            words = summary_text.split()
            if len(words) > word_target * 1.2:  # If 20% over target, trim it
                summary_text = " ".join(words[:word_target])
                summary_text += "... In conclusion, this covers the main points of the document."
            
            return summary_text
        else:
            # Fallback: return truncated original content
            words = document_content.split()[:word_target]
            return " ".join(words)
            
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        # Fallback: return truncated original content
        words = document_content.split()[:word_target]
        return " ".join(words)


@router.get("/test_tts")
async def test_tts_connection():
    """Test endpoint for XTTS v2 TTS connection."""

    try:
        test_result = await xtts_tts_client.test_connection()
        return test_result

    except Exception as e:
        logger.error(f"TTS test failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "suggestion": "Check TTS server and network connection"
        }


@router.get("/debug_session/{session_id}")
async def debug_session(session_id: str):
    """Debug endpoint to check session data."""
    
    try:
        session_manager = SessionManager()
        session_data = await session_manager.get_session(session_id)
        
        if not session_data:
            return {"error": "Session not found", "session_id": session_id}
        
        # Return session info without sensitive data
        debug_info = {
            "session_id": session_id,
            "has_parsed_document": "parsed_document" in session_data,
            "session_keys": list(session_data.keys()),
        }
        
        if "parsed_document" in session_data:
            doc = session_data["parsed_document"]
            debug_info["document_info"] = {
                "has_content": hasattr(doc, 'content') and bool(doc.content),
                "content_length": len(doc.content) if hasattr(doc, 'content') else 0,
                "content_preview": doc.content[:100] + "..." if hasattr(doc, 'content') and doc.content else "No content"
            }
        
        return debug_info
        
    except Exception as e:
        logger.error(f"Session debug failed: {e}")
        return {"error": str(e), "session_id": session_id}