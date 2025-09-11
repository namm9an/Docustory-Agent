"""
Microsoft T5 Speech TTS Service Integration.

Provides text-to-speech functionality using Microsoft's T5 Speech model
with comprehensive error handling, streaming support, and voice customization.
"""

import logging
import asyncio
import traceback
import aiohttp
import json
import base64
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class TTSResponse:
    """Response from Microsoft T5 Speech TTS service."""
    audio_content: bytes
    format: str
    duration: float
    model: str
    voice: str
    metadata: Optional[Dict[str, Any]] = None


class MicrosoftT5TTSClient:
    """
    Production-ready Microsoft T5 Speech TTS client.
    
    Provides text-to-speech conversion using Microsoft's T5 Speech model
    with support for multiple voices, formats, and streaming.
    """
    
    def __init__(self):
        self.endpoint = settings.TTS_ENDPOINT
        self.api_key = settings.TTS_API_KEY
        self.model = settings.TTS_MODEL
        self.provider = settings.TTS_PROVIDER
        self.timeout = aiohttp.ClientTimeout(total=60)  # 1 minute for speech synthesis
        
        # Supported voices for Microsoft T5 Speech
        self.available_voices = [
            "default",
            "female",
            "male", 
            "neutral",
            "professional",
            "casual",
            "energetic",
            "calm",
            "friendly",
            "authoritative"
        ]
        
        # Supported audio formats
        self.supported_formats = ["mp3", "wav", "ogg", "flac", "aac"]
        
        logger.info(f"Initialized Microsoft T5 Speech TTS client: {self.model}")
    
    async def synthesize_speech(self,
                               text: str,
                               voice: str = "default",
                               speed: float = 1.0,
                               pitch: float = 1.0,
                               format: str = "mp3") -> TTSResponse:
        """
        Convert text to speech using Microsoft T5 Speech model.
        
        Args:
            text: Text to convert to speech
            voice: Voice type to use
            speed: Speech speed (0.5 to 2.0)
            pitch: Voice pitch (0.5 to 2.0)
            format: Audio format (mp3, wav, ogg, etc.)
            
        Returns:
            TTSResponse with audio content and metadata
            
        Raises:
            Exception: If TTS synthesis fails
        """
        try:
            # Validate inputs
            if not text or not text.strip():
                raise ValueError("Text cannot be empty")
            
            if len(text) > 5000:  # Reasonable limit for TTS
                raise ValueError(f"Text too long ({len(text)} chars). Maximum 5000 characters.")
            
            # Validate voice
            if voice not in self.available_voices:
                logger.warning(f"Unknown voice '{voice}', using 'default'")
                voice = "default"
            
            # Validate format
            if format not in self.supported_formats:
                logger.warning(f"Unknown format '{format}', using 'mp3'")
                format = "mp3"
            
            # Clamp speed and pitch values
            speed = max(0.5, min(2.0, speed))
            pitch = max(0.5, min(2.0, pitch))
            
            # Prepare request payload for Microsoft T5 Speech
            payload = {
                "model": self.model,
                "input": text,
                "voice": {
                    "name": voice,
                    "speed": speed,
                    "pitch": pitch
                },
                "response_format": format,
                "max_tokens": 4096  # For audio generation
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "Docustory-TTS-Client/1.0"
            }
            
            logger.info(f"Synthesizing speech: {len(text)} chars, voice={voice}, format={format}")
            
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    self.endpoint,
                    json=payload,
                    headers=headers
                ) as response:
                    
                    if response.status == 200:
                        # Check if response is JSON (error) or binary (audio)
                        content_type = response.headers.get('content-type', '')
                        
                        if 'application/json' in content_type:
                            # JSON response - could be base64 encoded audio or error
                            json_response = await response.json()
                            
                            if 'audio' in json_response:
                                # Base64 encoded audio
                                audio_b64 = json_response['audio']
                                audio_content = base64.b64decode(audio_b64)
                            else:
                                raise Exception(f"Unexpected JSON response: {json_response}")
                        
                        elif 'audio/' in content_type or 'application/octet-stream' in content_type:
                            # Direct binary audio response
                            audio_content = await response.read()
                        
                        else:
                            raise Exception(f"Unexpected content type: {content_type}")
                        
                        # Estimate duration (rough calculation)
                        # Assume average speaking rate of ~150 words per minute
                        word_count = len(text.split())
                        estimated_duration = (word_count / 150) * 60  # seconds
                        
                        return TTSResponse(
                            audio_content=audio_content,
                            format=format,
                            duration=estimated_duration,
                            model=self.model,
                            voice=voice,
                            metadata={
                                "text_length": len(text),
                                "word_count": word_count,
                                "speed": speed,
                                "pitch": pitch,
                                "provider": self.provider,
                                "audio_size_bytes": len(audio_content)
                            }
                        )
                    
                    else:
                        error_text = await response.text()
                        logger.error(f"TTS API error {response.status}: {error_text}")
                        raise Exception(f"TTS API returned {response.status}: {error_text}")
        
        except aiohttp.ClientTimeout:
            logger.error("TTS request timed out")
            raise Exception("Text-to-speech request timed out")
        
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            logger.error(traceback.format_exc())
            raise Exception(f"Speech synthesis failed: {str(e)}")
    
    async def synthesize_speech_stream(self,
                                     text: str,
                                     voice: str = "default",
                                     speed: float = 1.0,
                                     pitch: float = 1.0,
                                     format: str = "mp3"):
        """
        Stream speech synthesis for real-time audio generation.
        
        Yields audio chunks as they are generated for low-latency playback.
        """
        try:
            # For streaming, we can implement chunked processing
            # Split text into sentences for streaming
            sentences = self._split_into_sentences(text)
            
            for sentence in sentences:
                if sentence.strip():
                    chunk_response = await self.synthesize_speech(
                        text=sentence,
                        voice=voice,
                        speed=speed,
                        pitch=pitch,
                        format=format
                    )
                    yield chunk_response.audio_content
                    
        except Exception as e:
            logger.error(f"Streaming TTS failed: {e}")
            raise Exception(f"Streaming speech synthesis failed: {str(e)}")
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for streaming synthesis."""
        import re
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() + '.' for s in sentences if s.strip()]
    
    def validate_voice_parameters(self, voice: str, speed: float, pitch: float) -> bool:
        """Validate voice synthesis parameters."""
        try:
            if voice not in self.available_voices:
                return False
            
            if not (0.5 <= speed <= 2.0):
                return False
                
            if not (0.5 <= pitch <= 2.0):
                return False
                
            return True
            
        except Exception:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            "model": self.model,
            "provider": self.provider,
            "endpoint": self.endpoint.replace(self.api_key, "***") if self.api_key in self.endpoint else self.endpoint,
            "status": "configured",
            "available_voices": self.available_voices,
            "supported_formats": self.supported_formats,
            "features": {
                "text_to_speech": True,
                "voice_synthesis": True,
                "streaming": True,
                "voice_control": True,
                "speed_control": True,
                "pitch_control": True
            },
            "limits": {
                "max_text_length": 5000,
                "speed_range": [0.5, 2.0],
                "pitch_range": [0.5, 2.0]
            },
            "note": f"{self.model} text-to-speech model"
        }
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported audio formats."""
        return self.supported_formats.copy()
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test connection to Microsoft T5 Speech service."""
        try:
            # Test with minimal text
            test_response = await self.synthesize_speech(
                text="Hello, this is a connection test.",
                voice="default",
                format="mp3"
            )
            
            return {
                "status": "connected",
                "model": self.model,
                "test_audio_size": len(test_response.audio_content),
                "response_time_ms": 0  # Could measure actual time
            }
            
        except Exception as e:
            return {
                "status": "connection_failed",
                "error": str(e),
                "model": self.model
            }


# Create global client instance
xtts_client = MicrosoftT5TTSClient()