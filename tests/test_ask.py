import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from app.main import app
from app.models.ask import AskRequest
from app.core.session import session_manager


client = TestClient(app)


class TestAskEndpoint:
    """Test the /ask endpoint functionality."""
    
    def setup_method(self):
        """Setup test environment before each test."""
        # Clean up any existing sessions
        session_manager.cleanup_all_sessions()
    
    def teardown_method(self):
        """Clean up after each test."""
        session_manager.cleanup_all_sessions()
    
    @patch('app.services.qwen_client.qwen_client.ask_question')
    def test_ask_question_success(self, mock_qwen):
        """Test successful question processing."""
        # Setup mock response
        mock_response = Mock()
        mock_response.answer = "This is a test answer from Qwen."
        mock_response.model = "qwen2.5-14b"
        mock_response.usage = {"tokens": 50}
        mock_response.metadata = {}
        mock_qwen.return_value = AsyncMock(return_value=mock_response)
        
        # Create a test session with document
        session_id = session_manager.create_session(
            document_text="This is a test document with sample content.",
            document_metadata={"filename": "test.pdf", "pages": 1}
        )
        
        # Make request
        request_data = {
            "session_id": session_id,
            "query": "What is this document about?",
            "voice_enabled": False,
            "max_tokens": 1024,
            "temperature": 0.7
        }
        
        response = client.post("/api/v1/ask", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["session_id"] == session_id
        assert "answer" in data
        assert data["query"] == "What is this document about?"
        assert "processing_time" in data
    
    def test_ask_question_session_not_found(self):
        """Test handling of non-existent session."""
        request_data = {
            "session_id": "non-existent-session-id",
            "query": "What is this about?",
            "voice_enabled": False
        }
        
        response = client.post("/api/v1/ask", json=request_data)
        
        assert response.status_code == 404
        data = response.json()["detail"]
        assert data["error_code"] == "SESSION_NOT_FOUND"
        assert "session_id" in data
    
    def test_ask_question_no_document(self):
        """Test handling of session without document."""
        # Create empty session
        session_id = session_manager.create_session()
        
        request_data = {
            "session_id": session_id,
            "query": "What is this about?",
            "voice_enabled": False
        }
        
        response = client.post("/api/v1/ask", json=request_data)
        
        assert response.status_code == 400
        data = response.json()["detail"]
        assert data["error_code"] == "NO_DOCUMENT"
    
    def test_ask_question_invalid_request(self):
        """Test validation of invalid request data."""
        request_data = {
            "session_id": "",  # Empty session ID
            "query": "",  # Empty query
        }
        
        response = client.post("/api/v1/ask", json=request_data)
        
        assert response.status_code == 422  # Validation error


class TestSessionManagement:
    """Test session management functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        session_manager.cleanup_all_sessions()
    
    def teardown_method(self):
        """Clean up after tests."""
        session_manager.cleanup_all_sessions()
    
    def test_get_session_status(self):
        """Test session status endpoint."""
        # Create test session
        session_id = session_manager.create_session(
            document_text="Test document",
            document_metadata={"filename": "test.pdf"}
        )
        
        response = client.get(f"/api/v1/session/{session_id}/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id
        assert data["active"] is True
        assert data["has_document"] is True
        assert "memory_usage_mb" in data
    
    def test_delete_session(self):
        """Test session deletion."""
        # Create test session
        session_id = session_manager.create_session(
            document_text="Test document"
        )
        
        # Verify session exists
        response = client.get(f"/api/v1/session/{session_id}/status")
        assert response.status_code == 200
        
        # Delete session
        response = client.delete(f"/api/v1/session/{session_id}")
        assert response.status_code == 200
        
        # Verify session is gone
        response = client.get(f"/api/v1/session/{session_id}/status")
        assert response.status_code == 404


class TestModelInfo:
    """Test model information endpoint."""
    
    def test_get_model_info(self):
        """Test model info endpoint."""
        response = client.get("/api/v1/models/info")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "qwen_model" in data
        assert "whisper_model" in data
        assert "vibe_voice_model" in data
        assert "session_stats" in data
        
        # Check session stats structure
        session_stats = data["session_stats"]
        assert "active_sessions" in session_stats
        assert "max_sessions" in session_stats
        assert "total_memory_mb" in session_stats


class TestErrorHandling:
    """Test error handling and fallback mechanisms."""
    
    def setup_method(self):
        session_manager.cleanup_all_sessions()
    
    def teardown_method(self):
        session_manager.cleanup_all_sessions()
    
    @patch('app.services.qwen_client.qwen_client.ask_question')
    def test_qwen_api_failure_fallback(self, mock_qwen):
        """Test fallback when Qwen API fails."""
        # Setup mock to raise exception
        mock_qwen.side_effect = Exception("API connection failed")
        
        # Create test session
        session_id = session_manager.create_session(
            document_text="Test document content"
        )
        
        request_data = {
            "session_id": session_id,
            "query": "What is this about?",
            "voice_enabled": False
        }
        
        response = client.post("/api/v1/ask", json=request_data)
        
        # Should still return 200 with fallback response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "technical difficulties" in data["answer"]
        assert data["metadata"]["model"] == "fallback"


if __name__ == "__main__":
    pytest.main([__file__])