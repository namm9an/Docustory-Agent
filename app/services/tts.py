import logging
import asyncio
import traceback
from typing import Dict, Any, Optional
from dataclasses import dataclass

try:
    import aiohttp
except ImportError as e:
    aiohttp = None
    logging.warning(f"aiohttp not available: {e}")

logger = logging.getLogger(__name__)

# Import config after defining logger to avoid circular imports
try:
    from app.core.config import settings
except ImportError:
    settings = None
    logger.warning("Could not import settings from app.core.config")


@dataclass
class TTSResponse:
    """Response from Vibe Voice Text-to-Speech service."""
    audio_content: bytes
    format: str = "wav"
    duration: float = 0.0
    voice: str = "Alice"
    metadata: Optional[Dict[str, Any]] = None


class XTTSv2TTSClient:
    """
    XTTS v2 Text-to-Speech client for audio generation.
    
    Uses Coqui XTTS v2 model for high-quality multilingual speech synthesis
    with support for multiple speakers (Alice=female, Bob=male).
    """
    
    def __init__(self):
        # XTTS v2 server endpoint - use config if available, fallback to default
        if settings and hasattr(settings, 'TTS_ENDPOINT'):
            self.base_url = settings.TTS_ENDPOINT.rstrip('/')
        else:
            self.base_url = "http://192.168.2.183:8002"  # Default fallback

        self.endpoint = f"{self.base_url}/generate-speech"
        self.available_voices = {
            "Alice": "female",
            "Bob": "male"
        }

        # Check if aiohttp is available
        if aiohttp is None:
            logger.error("aiohttp is not available. Install with: pip install aiohttp")
            self.client_available = False
        else:
            self.timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes for long audio generation
            self.client_available = True
            logger.info(f"XTTS v2 TTS client initialized - Endpoint: {self.endpoint}")
            logger.info("Using Coqui XTTS v2 multilingual model via Flask API")
    
    async def synthesize_speech(
        self,
        text: str,
        voice: str = "Alice"
    ) -> TTSResponse:
        """
        Convert text to speech using Vibe Voice API.

        Args:
            text: Text to convert to speech
            voice: Speaker voice ("Alice" for female, "Bob" for male)

        Returns:
            TTSResponse with audio content
        """

        if not self.client_available:
            raise RuntimeError("XTTS v2 TTS client not available. Check aiohttp installation.")

        if voice not in self.available_voices:
            voice = "Alice"  # Default to Alice if invalid voice
            logger.warning(f"Invalid voice requested, defaulting to Alice")

        try:
            logger.info(f"Generating speech with XTTS v2 for {len(text)} characters using voice: {voice}")
            
            payload = {
                "text": text,
                "speakers": voice,
                "language": "en"  # English language for XTTS v2
            }
            
            audio_content = await self._make_api_call(payload)
            
            # Estimate duration (XTTS v2 is typically ~150 words per minute)
            word_count = len(text.split())
            estimated_duration = (word_count / 150) * 60  # seconds
            
            return TTSResponse(
                audio_content=audio_content,
                format="wav",
                duration=estimated_duration,
                voice=voice,
                metadata={
                    "text_length": len(text),
                    "word_count": word_count,
                    "voice_type": self.available_voices[voice],
                    "model": "xtts_v2",
                    "endpoint": self.endpoint
                }
            )
            
        except Exception as e:
            logger.error(f"XTTS v2 TTS synthesis failed: {e}")
            raise RuntimeError(f"XTTS v2 speech synthesis failed: {str(e)}")
    
    async def synthesize_speech_chunked(
        self, 
        text: str, 
        voice: str = "Alice",
        chunk_size: int = 500
    ) -> list[TTSResponse]:
        """
        Convert long text to speech by splitting into chunks.
        Useful for very long texts like full PDFs.
        
        Args:
            text: Long text to convert
            voice: Speaker voice
            chunk_size: Maximum characters per chunk
            
        Returns:
            List of TTSResponse objects
        """
        
        chunks = self._split_text_into_chunks(text, chunk_size)
        responses = []
        
        logger.info(f"Processing {len(chunks)} text chunks for XTTS v2 TTS")
        
        for i, chunk in enumerate(chunks):
            try:
                response = await self.synthesize_speech(chunk.strip(), voice)
                response.metadata["chunk_index"] = i
                response.metadata["total_chunks"] = len(chunks)
                responses.append(response)
                
                # Small delay to avoid overwhelming the API
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to process chunk {i}: {e}")
                continue
        
        return responses
    
    def _split_text_into_chunks(self, text: str, chunk_size: int) -> list[str]:
        """Split text into chunks at sentence boundaries when possible."""
        
        if len(text) <= chunk_size:
            return [text]
        
        # Split by sentences first
        import re
        sentences = re.split(r'[.!?]+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) + 1 > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = sentence
                else:
                    # Single sentence is too long, split it
                    chunks.extend(self._split_long_sentence(sentence, chunk_size))
            else:
                current_chunk += (" " + sentence) if current_chunk else sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_long_sentence(self, sentence: str, chunk_size: int) -> list[str]:
        """Split a very long sentence into smaller chunks."""
        
        words = sentence.split()
        chunks = []
        current_chunk = ""
        
        for word in words:
            if len(current_chunk) + len(word) + 1 > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = word
                else:
                    # Single word is too long, just add it
                    chunks.append(word)
            else:
                current_chunk += (" " + word) if current_chunk else word
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    async def _make_api_call(self, payload: Dict[str, Any]) -> bytes:
        """Make API call to XTTS v2 endpoint."""

        if not self.client_available or aiohttp is None:
            raise RuntimeError("aiohttp not available for API call")

        headers = {
            "Content-Type": "application/json"
        }

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(
                self.endpoint,
                json=payload,
                headers=headers
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"XTTS v2 API error {response.status}: {error_text}")
                
                return await response.read()  # Return raw audio bytes
    
    def get_available_voices(self) -> Dict[str, str]:
        """Get available voices with their gender."""
        return self.available_voices.copy()
    
    def get_voice_by_gender(self, gender: str) -> str:
        """Get voice name by gender preference."""
        gender = gender.lower()
        for voice, voice_gender in self.available_voices.items():
            if voice_gender.lower() == gender:
                return voice
        return "Alice"  # Default fallback
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test connection to XTTS v2 API."""

        if not self.client_available:
            return {
                "success": False,
                "error": "XTTS v2 client not available",
                "suggestion": "Install aiohttp: pip install aiohttp"
            }

        try:
            test_text = "Hello, this is a test of the XTTS v2 system."
            response = await self.synthesize_speech(test_text, "Alice")
            
            return {
                "success": True,
                "message": "XTTS v2 connection successful",
                "endpoint": self.endpoint,
                "audio_size_bytes": len(response.audio_content),
                "estimated_duration": response.duration,
                "model": "xtts_v2",
                "available_voices": self.available_voices
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "endpoint": self.endpoint,
                "suggestion": f"Check if your XTTS v2 server is running at {self.base_url}"
            }


# Global XTTS v2 TTS client instance
xtts_tts_client = XTTSv2TTSClient()