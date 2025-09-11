import logging
import asyncio
import traceback
from typing import Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass

try:
    from openai import AsyncOpenAI
except ImportError as e:
    AsyncOpenAI = None
    logging.warning(f"OpenAI client not available: {e}")

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class QwenResponse:
    """Response from Qwen API via E2E Networks."""
    answer: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class QwenE2EClient:
    """
    Production-ready Qwen 2.5 14B client for E2E Networks.
    
    Uses OpenAI-compatible API client to communicate with E2E Networks
    Qwen/Qwen2.5-14B-Instruct model endpoint.
    
    Features:
    - OpenAI-compatible API integration
    - Comprehensive error handling with fallbacks
    - Streaming response support
    - Context-aware document Q&A
    - Usage tracking and monitoring
    """
    
    def __init__(self):
        self.endpoint = settings.QWEN_ENDPOINT
        self.api_key = settings.QWEN_API_KEY
        self.model = settings.QWEN_MODEL
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
                logger.info(f"Qwen E2E client initialized - Model: {self.model}")
            except Exception as e:
                logger.error(f"Failed to initialize Qwen E2E client: {e}")
                self.client = None
                self.client_available = False
    
    async def ask_question(
        self, 
        question: str, 
        context: str = "", 
        system_prompt: str = None,
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> QwenResponse:
        """
        Ask a question to Qwen model with document context.
        
        Args:
            question: User's question
            context: Document context for answering
            system_prompt: Optional custom system prompt
            max_tokens: Maximum tokens in response
            temperature: Response creativity (0-2)
            
        Returns:
            QwenResponse with answer and metadata
            
        Raises:
            RuntimeError: If API call fails
        """
        
        if not self.client_available or not self.client:
            raise RuntimeError("Qwen E2E client not available. Check OpenAI installation and configuration.")
        
        # Default system prompt for document Q&A
        if system_prompt is None:
            system_prompt = (
                "You are an intelligent document assistant. Use the provided document context "
                "to answer questions accurately and concisely. If the answer is not in the "
                "document, clearly state that the information is not available in the provided text. "
                "Always provide helpful and informative responses based on the given context."
            )
        
        try:
            logger.info(f"Sending question to Qwen E2E: '{question[:100]}...'")
            
            # Prepare messages for the chat completion
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Add context if provided
            if context.strip():
                context_message = f"Document Context:\n{context}\n\nBased on this document, please answer the following question:"
                messages.append({"role": "user", "content": context_message})
                messages.append({"role": "user", "content": f"Question: {question}"})
            else:
                messages.append({"role": "user", "content": question})
            
            # Make API call to E2E Networks Qwen endpoint
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )
            
            # Extract response
            answer = completion.choices[0].message.content
            
            # Prepare usage information
            usage_info = {
                "prompt_tokens": getattr(completion.usage, 'prompt_tokens', 0) if completion.usage else 0,
                "completion_tokens": getattr(completion.usage, 'completion_tokens', 0) if completion.usage else 0,
                "total_tokens": getattr(completion.usage, 'total_tokens', 0) if completion.usage else 0
            }
            
            # Prepare metadata
            metadata = {
                "endpoint": self.endpoint,
                "model_used": self.model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "context_length": len(context.split()) if context else 0,
                "question_length": len(question.split())
            }
            
            logger.info(f"Qwen E2E response received: {len(answer)} chars, {usage_info.get('total_tokens', 0)} tokens")
            
            return QwenResponse(
                answer=answer,
                model=self.model,
                usage=usage_info,
                metadata=metadata
            )
            
        except Exception as e:
            error_msg = f"Qwen E2E API call failed: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            
            # Provide detailed error information
            if "401" in str(e) or "unauthorized" in str(e).lower():
                raise RuntimeError("Authentication failed. Please check your E2E Networks API key.")
            elif "429" in str(e) or "rate limit" in str(e).lower():
                raise RuntimeError("Rate limit exceeded. Please try again later.")
            elif "timeout" in str(e).lower():
                raise RuntimeError("Request timeout. The model took too long to respond.")
            else:
                raise RuntimeError(f"Failed to get response from Qwen: {str(e)}")
    
    async def ask_question_stream(
        self, 
        question: str, 
        context: str = "", 
        system_prompt: str = None,
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> AsyncGenerator[str, None]:
        """
        Stream response from Qwen model for real-time output.
        
        Args:
            question: User's question
            context: Document context
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens
            temperature: Response creativity
            
        Yields:
            String chunks of the response as they arrive
        """
        
        if not self.client_available or not self.client:
            yield "Error: Qwen E2E client not available."
            return
        
        # Default system prompt
        if system_prompt is None:
            system_prompt = (
                "You are an intelligent document assistant. Use the provided document context "
                "to answer questions accurately and concisely."
            )
        
        try:
            logger.info(f"Starting streaming response for: '{question[:100]}...'")
            
            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            if context.strip():
                context_message = f"Document Context:\n{context}\n\nBased on this document, please answer:"
                messages.append({"role": "user", "content": context_message})
                messages.append({"role": "user", "content": f"Question: {question}"})
            else:
                messages.append({"role": "user", "content": question})
            
            # Make streaming API call
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )
            
            # Stream the response
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    yield content
            
            logger.info("Streaming response completed successfully")
            
        except Exception as e:
            error_msg = f"Streaming failed: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            yield f"Error: {error_msg}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration and status information."""
        
        return {
            "model": self.model,
            "endpoint": self.endpoint,
            "provider": "E2E Networks",
            "client_available": self.client_available,
            "status": "configured" if self.client_available and self.api_key != "your-e2e-networks-jwt-token-here" else "placeholder",
            "features": {
                "chat_completion": True,
                "streaming": True,
                "context_aware": True,
                "document_qa": True
            },
            "limits": {
                "max_tokens": 4096,
                "context_window": 32768,
                "temperature_range": [0.0, 2.0]
            }
        }
    
    async def test_connection(self) -> Dict[str, Any]:
        """
        Test connection to E2E Networks Qwen endpoint.
        
        Returns:
            Dict with connection test results
        """
        
        if not self.client_available or not self.client:
            return {
                "success": False,
                "error": "OpenAI client not available",
                "suggestion": "Install OpenAI client: pip install openai"
            }
        
        try:
            # Simple test query
            test_response = await self.ask_question(
                question="Hello, are you working correctly?",
                context="",
                max_tokens=50,
                temperature=0.1
            )
            
            return {
                "success": True,
                "message": "Connection successful",
                "model": self.model,
                "response_preview": test_response.answer[:100] + "..." if len(test_response.answer) > 100 else test_response.answer,
                "tokens_used": test_response.usage.get("total_tokens", 0) if test_response.usage else 0
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "suggestion": "Check your E2E Networks API key and endpoint configuration"
            }


# Global Qwen E2E client instance
qwen_client = QwenE2EClient()