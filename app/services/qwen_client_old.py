import logging
import aiohttp
import asyncio
from typing import Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class QwenResponse:
    """Response from Qwen API."""
    answer: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class QwenClient:
    """Client for Qwen 2.5 14B model API."""
    
    def __init__(self):
        self.endpoint = settings.QWEN_ENDPOINT
        self.api_key = settings.QWEN_API_KEY
        self.model = settings.QWEN_MODEL
        self.timeout = aiohttp.ClientTimeout(total=60)  # 60 second timeout
    
    async def ask_question(
        self, 
        question: str, 
        context: str = "", 
        system_prompt: str = None,
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> QwenResponse:
        """Ask a question to Qwen model with optional context."""
        
        # Default system prompt for document Q&A
        if system_prompt is None:
            system_prompt = (
                "You are an intelligent document assistant. Use the provided document context "
                "to answer questions accurately and concisely. If the answer is not in the "
                "document, clearly state that the information is not available in the provided text."
            )
        
        # Construct the prompt
        if context.strip():
            full_prompt = f"""System: {system_prompt}

Document Context:
{context}

Question: {question}

Answer:"""
        else:
            full_prompt = f"""System: {system_prompt}

Question: {question}

Answer:"""
        
        # Placeholder implementation - replace with actual API call
        try:
            # TODO: Replace with actual Qwen API implementation
            answer = await self._mock_qwen_response(question, context)
            
            return QwenResponse(
                answer=answer,
                model=self.model,
                usage={"tokens": len(full_prompt.split())},
                metadata={"endpoint": self.endpoint}
            )
            
        except Exception as e:
            logger.error(f"Qwen API call failed: {e}")
            raise RuntimeError(f"Failed to get response from Qwen: {str(e)}")
    
    async def ask_question_stream(
        self, 
        question: str, 
        context: str = "", 
        system_prompt: str = None,
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> AsyncGenerator[str, None]:
        """Stream response from Qwen model."""
        
        try:
            # TODO: Replace with actual streaming implementation
            response = await self.ask_question(question, context, system_prompt, max_tokens, temperature)
            
            # Mock streaming by yielding chunks
            words = response.answer.split()
            for i in range(0, len(words), 3):  # Yield 3 words at a time
                chunk = " ".join(words[i:i+3]) + " "
                yield chunk
                await asyncio.sleep(0.05)  # Small delay to simulate streaming
                
        except Exception as e:
            logger.error(f"Qwen streaming failed: {e}")
            yield f"Error: {str(e)}"
    
    async def _mock_qwen_response(self, question: str, context: str) -> str:
        """Mock response for testing - replace with actual API call."""
        
        # Simulate processing delay
        await asyncio.sleep(0.5)
        
        if context.strip():
            return (
                f"Based on the provided document, I can help answer your question: '{question}'. "
                f"This is a placeholder response from Qwen 2.5 14B model. "
                f"The document contains {len(context.split())} words of context. "
                f"In a real implementation, this would be processed by the actual Qwen API."
            )
        else:
            return (
                f"I received your question: '{question}'. "
                f"This is a placeholder response from Qwen 2.5 14B model. "
                f"No document context was provided. "
                f"In a real implementation, this would be processed by the actual Qwen API."
            )
    
    async def _make_api_call(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make actual API call to Qwen endpoint - placeholder implementation."""
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(
                self.endpoint,
                json=payload,
                headers=headers
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Qwen API error {response.status}: {error_text}")
                
                return await response.json()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration information."""
        return {
            "model": self.model,
            "endpoint": self.endpoint,
            "status": "placeholder" if self.api_key == "your-qwen-api-key-here" else "configured"
        }


# Global Qwen client instance
qwen_client = QwenClient()