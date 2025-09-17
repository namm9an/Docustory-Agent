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
            self.base_url = "http://164.52.192.118:8000"  # Your XTTS v2 server

        self.endpoint = f"{self.base_url}/tts"  # Updated endpoint
        self.speakers_endpoint = f"{self.base_url}/speakers"
        self.languages_endpoint = f"{self.base_url}/languages"
        self.health_endpoint = f"{self.base_url}/health"

        # Initialize with default voices, will be updated from API
        self.available_voices = {}
        self.available_languages = []
        self.client_available = False

        # Check if aiohttp is available
        if aiohttp is None:
            logger.error("aiohttp is not available. Install with: pip install aiohttp")
            self.client_available = False
        else:
            self.timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes for long audio generation
            self.client_available = True
            logger.info(f"XTTS v2 TTS client initialized - Endpoint: {self.endpoint}")
            logger.info("Using your XTTS v2 server API")

            # Load available voices and languages
            asyncio.create_task(self._load_available_options())
    
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

        # Validate voice against available voices
        if self.available_voices and voice not in self.available_voices:
            # Try to find a voice with similar name or default to first available
            available_voice_names = list(self.available_voices.keys())
            if available_voice_names:
                original_voice = voice
                voice = available_voice_names[0]  # Use first available voice
                logger.warning(f"Voice '{original_voice}' not available, using '{voice}' instead")
            else:
                logger.warning(f"No voices available, proceeding with '{voice}' (may fail)")

        # If no voices are available at all, try with known XTTS speakers
        if not self.available_voices:
            logger.warning("No voices loaded from API, using known XTTS v2 speakers")
            self.available_voices = {
                "Claribel Dervla": "female",
                "Daisy Studious": "female",
                "Andrew Chipper": "male",
                "Ana Florence": "female"
            }
            # Use first available voice
            voice = list(self.available_voices.keys())[0]

        try:
            logger.info(f"Generating speech with XTTS v2 for {len(text)} characters using voice: {voice}")

            # Use the correct payload format for your API
            payload = {
                "text": text,
                "speaker": voice,  # Changed from "speakers" to "speaker"
                "language": "en"   # Language code
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

    async def _load_available_options(self):
        """Load available voices and languages from the API"""
        try:
            # Load speakers
            speakers_data = await self._make_api_call_get(self.speakers_endpoint)
            if speakers_data and 'voices' in speakers_data:
                voices_list = speakers_data['voices']
                if voices_list:
                    self.available_voices = {
                        voice['name']: voice.get('gender', 'unknown')
                        for voice in voices_list
                    }
                    logger.info(f"Loaded {len(self.available_voices)} voices from API")
                else:
                    # If no voices returned, the API might be using built-in XTTS speakers
                    logger.warning("No voices in 'voices' array, checking for 'speakers' array")
                    speakers_list = speakers_data.get('speakers', [])
                    if speakers_list:
                        # Use the built-in XTTS speakers
                        self.available_voices = {}
                        for speaker in speakers_list[:10]:  # Limit to first 10 for performance
                            # Try to infer gender from name (basic heuristic)
                            if any(female_indicator in speaker.lower() for female_indicator in ['ana', 'maria', 'sofia', 'emma', 'lily', 'rose', 'anna']):
                                self.available_voices[speaker] = 'female'
                            elif any(male_indicator in speaker.lower() for male_indicator in ['john', 'david', 'michael', 'robert', 'james']):
                                self.available_voices[speaker] = 'male'
                            else:
                                self.available_voices[speaker] = 'unknown'
                        logger.info(f"Loaded {len(self.available_voices)} built-in XTTS speakers")
                    else:
                        # Final fallback
                        self.available_voices = {
                            "Claribel Dervla": "female",  # Known working XTTS speaker
                            "Daisy Studious": "female",
                            "Andrew Chipper": "male",
                            "Ana Florence": "female"
                        }
                        logger.warning("Using known XTTS v2 speakers as fallback")
            else:
                # Fallback to known working XTTS speakers
                self.available_voices = {
                    "Claribel Dervla": "female",
                    "Daisy Studious": "female",
                    "Andrew Chipper": "male",
                    "Ana Florence": "female"
                }
                logger.warning("Failed to load voices from API, using known XTTS v2 speakers")

            # Load languages
            languages_data = await self._make_api_call_get(self.languages_endpoint)
            if languages_data and 'languages' in languages_data:
                self.available_languages = languages_data['languages']
                logger.info(f"Loaded {len(self.available_languages)} languages from API")
            else:
                # Fallback to common languages
                self.available_languages = ["en", "es", "fr", "de", "it", "pt"]
                logger.warning("Failed to load languages from API, using defaults")

        except Exception as e:
            logger.warning(f"Failed to load available options from API: {e}")
            # Set defaults to known working XTTS speakers if API call fails
            self.available_voices = {
                "Claribel Dervla": "female",
                "Daisy Studious": "female",
                "Andrew Chipper": "male",
                "Ana Florence": "female"
            }
            self.available_languages = ["en", "es", "fr", "de", "it", "pt"]

    async def _make_api_call_get(self, url: str) -> Optional[Dict[str, Any]]:
        """Make GET API call to retrieve data"""
        if not self.client_available or aiohttp is None:
            return None

        headers = {"Content-Type": "application/json"}

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.warning(f"API GET request failed: {response.status}")
                        return None
        except Exception as e:
            logger.warning(f"API GET request error: {e}")
            return None

    def get_available_voices(self) -> Dict[str, str]:
        """Get available voices with their gender."""
        return self.available_voices.copy()

    def get_available_languages(self) -> list:
        """Get available languages."""
        return self.available_languages.copy()

    def get_voice_by_gender(self, gender: str) -> str:
        """Get voice name by gender preference."""
        gender = gender.lower()
        for voice, voice_gender in self.available_voices.items():
            if voice_gender.lower() == gender:
                return voice
        # Return first available voice if no match
        if self.available_voices:
            return list(self.available_voices.keys())[0]
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
            # First test health endpoint
            health_data = await self._make_api_call_get(self.health_endpoint)
            if health_data:
                logger.info(f"Health check successful: {health_data}")

            # Test with a voice from available voices
            test_voice = list(self.available_voices.keys())[0] if self.available_voices else "Alice"
            test_text = "Hello, this is a test of the XTTS v2 system."
            response = await self.synthesize_speech(test_text, test_voice)

            return {
                "success": True,
                "message": "XTTS v2 connection successful",
                "endpoint": self.endpoint,
                "base_url": self.base_url,
                "audio_size_bytes": len(response.audio_content),
                "estimated_duration": response.duration,
                "test_voice": test_voice,
                "available_voices": self.available_voices,
                "available_languages": self.available_languages
            }

        except Exception as e:
            logger.error(f"TTS connection test failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "endpoint": self.endpoint,
                "base_url": self.base_url,
                "suggestion": f"Check if your XTTS v2 server is running at {self.base_url} and endpoints are accessible"
            }


# Global XTTS v2 TTS client instance
xtts_tts_client = XTTSv2TTSClient()