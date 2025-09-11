import logging
import asyncio
import traceback
from typing import Dict, Any, Optional
from dataclasses import dataclass
import io

try:
    from openai import AsyncOpenAI
except ImportError as e:
    AsyncOpenAI = None
    logging.warning(f"OpenAI client not available: {e}")

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class STTResponse:
    """Response from Speech-to-Text service via E2E Networks."""
    text: str
    confidence: float
    language: str
    duration: float
    model: str
    metadata: Optional[Dict[str, Any]] = None


class WhisperE2EClient:
    """
    Production-ready Whisper Large-v3 client for E2E Networks.
    
    Uses OpenAI-compatible API client to communicate with E2E Networks
    openai/whisper-large-v3 model endpoint.
    """
    
    def __init__(self):
        self.endpoint = settings.WHISPER_ENDPOINT
        self.api_key = settings.WHISPER_API_KEY
        self.model = settings.WHISPER_MODEL
        self.client_available = AsyncOpenAI is not None
        
        if not self.client_available:
            logger.error("OpenAI client not available. Install with: pip install openai")
            self.client = None
        else:
            try:
                self.client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.endpoint
                )
                logger.info(f"Whisper E2E client initialized - Model: {self.model}")
            except Exception as e:
                logger.error(f"Failed to initialize Whisper E2E client: {e}")
                self.client = None
                self.client_available = False
    
    async def transcribe_audio(
        self, 
        audio_content: bytes, 
        filename: str,
        language: str = "auto",
        response_format: str = "text"
    ) -> STTResponse:
        """
        Transcribe audio to text using Whisper Large-v3 via E2E Networks.
        
        Args:
            audio_content: Raw audio bytes
            filename: Original filename for context
            language: Language code (or 'auto' for auto-detection)
            response_format: Response format ('text', 'json', 'verbose_json')
            
        Returns:
            STTResponse with transcription and metadata
        """
        
        if not self.client_available or not self.client:
            raise RuntimeError("Whisper E2E client not available. Check OpenAI installation and configuration.")
        
        try:
            logger.info(f"Transcribing audio file: {filename} ({len(audio_content)} bytes)")
            
            # Create audio file-like object for OpenAI client
            audio_file = io.BytesIO(audio_content)
            audio_file.name = filename
            
            # Make transcription request
            transcription = await self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                language=None if language == "auto" else language,
                response_format=response_format,
                temperature=0.0
            )
            
            # Extract text based on response format
            if response_format == "text":
                text = transcription
                language_detected = language if language != "auto" else "en"
                confidence = 0.95  # Default confidence for text format
            else:
                # JSON/verbose_json formats
                text = transcription.text
                language_detected = getattr(transcription, 'language', 'en')
                confidence = getattr(transcription, 'confidence', 0.95)
            
            # Prepare metadata
            metadata = {
                "filename": filename,
                "file_size": len(audio_content),
                "endpoint": self.endpoint,
                "model_used": self.model,
                "response_format": response_format,
                "language_requested": language,
                "language_detected": language_detected
            }
            
            logger.info(f"Whisper E2E transcription completed: {len(text)} characters")
            
            return STTResponse(
                text=text,
                confidence=confidence,
                language=language_detected,
                duration=len(audio_content) / 16000,  # Estimate duration
                model=self.model,
                metadata=metadata
            )
            
        except Exception as e:
            error_msg = f"Whisper E2E transcription failed: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            
            # Provide detailed error information
            if "401" in str(e) or "unauthorized" in str(e).lower():
                raise RuntimeError("Authentication failed. Please check your E2E Networks API key.")
            elif "429" in str(e) or "rate limit" in str(e).lower():
                raise RuntimeError("Rate limit exceeded. Please try again later.")
            elif "timeout" in str(e).lower():
                raise RuntimeError("Request timeout. The audio file took too long to process.")
            else:
                raise RuntimeError(f"Speech-to-text conversion failed: {str(e)}")
    
    async def transcribe_audio_with_timestamps(
        self, 
        audio_content: bytes, 
        filename: str,
        language: str = "auto"
    ) -> Dict[str, Any]:
        """
        Transcribe audio with word-level timestamps using verbose_json format.
        """
        
        try:
            # Get verbose transcription with timestamps
            transcription = await self.client.audio.transcriptions.create(
                model=self.model,
                file=io.BytesIO(audio_content),
                language=None if language == "auto" else language,
                response_format="verbose_json",
                timestamp_granularities=["word"]
            )
            
            return {
                "text": transcription.text,
                "language": transcription.language,
                "duration": transcription.duration,
                "words": transcription.words if hasattr(transcription, 'words') else [],
                "segments": transcription.segments if hasattr(transcription, 'segments') else [],
                "model": self.model
            }
            
        except Exception as e:
            logger.error(f"Timestamped transcription failed: {e}")
            raise RuntimeError(f"Timestamped transcription failed: {str(e)}")
    
    def get_supported_formats(self) -> list:
        """Get list of supported audio formats."""
        return [
            "mp3", "mp4", "mpeg", "mpga", "m4a", 
            "wav", "webm", "flac", "ogg", "3gp"
        ]
    
    def validate_audio_file(self, filename: str) -> bool:
        """Validate if audio file format is supported."""
        if not filename:
            return False
        
        extension = filename.lower().split('.')[-1]
        return extension in self.get_supported_formats()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration and status information."""
        return {
            "model": self.model,
            "endpoint": self.endpoint,
            "provider": "E2E Networks",
            "client_available": self.client_available,
            "supported_formats": self.get_supported_formats(),
            "status": "configured" if self.client_available and "your-whisper-api-key-here" not in self.api_key else "placeholder",
            "features": {
                "transcription": True,
                "language_detection": True,
                "timestamps": True,
                "multiple_formats": True
            }
        }
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test connection to E2E Networks Whisper endpoint."""
        
        if not self.client_available or not self.client:
            return {
                "success": False,
                "error": "OpenAI client not available",
                "suggestion": "Install OpenAI client: pip install openai"
            }
        
        try:
            # Create a small test audio file (silence)
            import wave
            test_audio = io.BytesIO()
            with wave.open(test_audio, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)  # 16kHz
                wav_file.writeframes(b'\x00' * 1600)  # 0.1 seconds of silence
            
            test_response = await self.transcribe_audio(
                audio_content=test_audio.getvalue(),
                filename="test.wav",
                response_format="text"
            )
            
            return {
                "success": True,
                "message": "Connection successful",
                "model": self.model,
                "response_preview": test_response.text[:50] + "..." if len(test_response.text) > 50 else test_response.text
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "suggestion": "Check your E2E Networks API key and endpoint configuration"
            }


# Global Whisper E2E client instance
whisper_client = WhisperE2EClient()