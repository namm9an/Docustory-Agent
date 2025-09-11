"""
Comprehensive Phase 3 integration tests for Docustory.in

Tests all major Phase 3 features including:
- Enhanced document parsing with PyMuPDF and python-docx
- YAKE keyword extraction with fallbacks
- Robust session lifecycle management
- Comprehensive upload endpoint with validation
- Enhanced ask endpoint with context injection
- Session management endpoints
- Streaming voice responses
- Error handling with fallbacks
"""

import pytest
import asyncio
import time
import tempfile
import os
from typing import Dict, Any
from httpx import AsyncClient
from fastapi.testclient import TestClient

from app.main import app
from app.core.session import session_manager
from app.core.error_handler import reset_error_stats
from app.services.parser_service import DocumentParserService
from app.services.yake_service import YAKEService

# Test client
client = TestClient(app)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
async def setup_and_cleanup():
    """Setup and cleanup for each test."""
    # Reset error stats before each test
    reset_error_stats()
    
    # Cleanup sessions before each test
    session_manager.cleanup_all_sessions()
    
    yield
    
    # Cleanup after each test
    session_manager.cleanup_all_sessions()


@pytest.fixture
def sample_pdf_content():
    """Create a simple PDF-like content for testing."""
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n>>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000074 00000 n \n0000000120 00000 n \ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n179\n%%EOF"


@pytest.fixture
def sample_text_document():
    """Sample text document content for testing."""
    return """
    # Test Document for Docustory.in

    This is a comprehensive test document that contains various sections and topics
    to test the YAKE keyword extraction and document parsing capabilities.

    ## Introduction
    
    Artificial intelligence and machine learning are transformative technologies
    that are reshaping industries across the globe. Natural language processing,
    computer vision, and deep learning are key components of modern AI systems.

    ## Machine Learning Fundamentals
    
    Machine learning algorithms can be categorized into supervised learning,
    unsupervised learning, and reinforcement learning. Popular algorithms include
    neural networks, decision trees, and support vector machines.

    ## Applications
    
    AI applications span across healthcare, finance, autonomous vehicles,
    and smart city infrastructure. These technologies enable predictive analytics,
    automated decision making, and intelligent automation.

    ## Conclusion
    
    The future of artificial intelligence looks promising with continued
    advancements in computational power and algorithmic sophistication.
    """


class TestDocumentParsingService:
    """Test enhanced document parsing capabilities."""

    async def test_parser_service_initialization(self):
        """Test that parser service initializes correctly."""
        parser_service = DocumentParserService()
        status = await parser_service.get_parser_status()
        
        assert isinstance(status, dict)
        assert "pdf_available" in status
        assert "docx_available" in status
        assert "supported_formats" in status
        assert ".pdf" in status["supported_formats"]

    async def test_parse_text_document(self, sample_text_document):
        """Test parsing of text document content."""
        parser_service = DocumentParserService()
        
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(sample_text_document)
            temp_path = f.name
        
        try:
            # Parse the document
            parsed_doc = await parser_service.parse_document(
                file_content=sample_text_document.encode('utf-8'),
                filename="test_document.txt"
            )
            
            assert parsed_doc is not None
            assert len(parsed_doc.content) > 0
            assert parsed_doc.page_count >= 1
            assert parsed_doc.chunks is not None
            assert len(parsed_doc.chunks) > 0
            
            # Test memory estimation
            memory_estimate = parsed_doc.get_memory_estimate_mb()
            assert memory_estimate > 0
            
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    async def test_document_chunking(self, sample_text_document):
        """Test document chunking functionality."""
        parser_service = DocumentParserService()
        
        parsed_doc = await parser_service.parse_document(
            file_content=sample_text_document.encode('utf-8'),
            filename="test_chunking.txt"
        )
        
        # Verify chunking
        assert parsed_doc.chunks is not None
        assert len(parsed_doc.chunks) > 1  # Should create multiple chunks
        
        # Verify chunk properties
        for chunk in parsed_doc.chunks:
            assert hasattr(chunk, 'content')
            assert hasattr(chunk, 'chunk_id')
            assert hasattr(chunk, 'start_index')
            assert hasattr(chunk, 'end_index')
            assert len(chunk.content.strip()) > 0


class TestYAKEService:
    """Test YAKE keyword extraction and document indexing."""

    async def test_yake_service_initialization(self):
        """Test YAKE service initialization."""
        yake_service = YAKEService()
        assert yake_service is not None

    async def test_create_document_index(self, sample_text_document):
        """Test document index creation with YAKE."""
        parser_service = DocumentParserService()
        yake_service = YAKEService()
        
        # Parse document first
        parsed_doc = await parser_service.parse_document(
            file_content=sample_text_document.encode('utf-8'),
            filename="test_yake.txt"
        )
        
        # Create YAKE index
        document_index = await yake_service.create_document_index(parsed_doc)
        
        assert document_index is not None
        assert hasattr(document_index, 'keywords')
        assert hasattr(document_index, 'chunks')
        assert len(document_index.keywords) > 0
        
        # Verify keywords contain relevant terms
        keyword_texts = [kw.keyword for kw in document_index.keywords]
        assert any("machine learning" in kw.lower() for kw in keyword_texts)

    async def test_document_search(self, sample_text_document):
        """Test document search functionality."""
        parser_service = DocumentParserService()
        yake_service = YAKEService()
        
        # Parse and index document
        parsed_doc = await parser_service.parse_document(
            file_content=sample_text_document.encode('utf-8'),
            filename="test_search.txt"
        )
        
        document_index = await yake_service.create_document_index(parsed_doc)
        
        # Test search
        search_results = await yake_service.search_document(
            document_index=document_index,
            query="machine learning algorithms",
            max_results=3
        )
        
        assert search_results is not None
        assert len(search_results) > 0
        
        # Verify search results
        for result in search_results:
            assert hasattr(result, 'content')
            assert hasattr(result, 'relevance_score')
            assert hasattr(result, 'chunk_id')
            assert result.relevance_score >= 0


class TestSessionManagement:
    """Test enhanced session lifecycle management."""

    def test_session_creation(self):
        """Test session creation with Phase 3 features."""
        # Create session
        session_id = session_manager.create_session()
        assert session_id is not None
        assert len(session_id) > 0
        
        # Verify session exists
        session = session_manager.get_session(session_id)
        assert session is not None
        assert session.session_id == session_id

    def test_session_update_with_phase3_data(self, sample_text_document):
        """Test session updates with Phase 3 document data."""
        # Create session
        session_id = session_manager.create_session()
        
        # Mock Phase 3 data
        mock_parsed_doc = type('MockParsedDoc', (), {
            'content': sample_text_document,
            'page_count': 1,
            'chunks': [],
            'metadata': None
        })()
        
        # Update session with Phase 3 data
        success = session_manager.update_session(
            session_id,
            parsed_document=mock_parsed_doc,
            document_text=sample_text_document  # Legacy field
        )
        
        assert success is True
        
        # Verify update
        session = session_manager.get_session(session_id)
        assert session.parsed_document is not None
        assert session.document_text == sample_text_document

    def test_session_statistics_tracking(self):
        """Test session statistics tracking."""
        session_id = session_manager.create_session()
        session = session_manager.get_session(session_id)
        
        # Test query tracking
        initial_query_count = session.query_count
        session.update_query_time()
        assert session.query_count == initial_query_count + 1
        assert session.last_query_at is not None
        
        # Test upload tracking
        initial_upload_count = session.upload_count
        session.record_upload("test_file.pdf", 1.5)
        assert session.upload_count == initial_upload_count + 1
        assert session.last_upload_filename == "test_file.pdf"

    def test_session_memory_calculation(self):
        """Test enhanced memory usage calculation."""
        session_id = session_manager.create_session()
        session = session_manager.get_session(session_id)
        
        # Add some data
        session_manager.update_session(
            session_id,
            document_text="Test document content for memory calculation"
        )
        
        # Test memory calculation
        memory_usage = session.get_memory_usage_mb()
        assert memory_usage >= 0
        assert isinstance(memory_usage, float)

    def test_session_cleanup(self):
        """Test session cleanup functionality."""
        # Create multiple sessions
        session_ids = []
        for i in range(3):
            session_id = session_manager.create_session()
            session_ids.append(session_id)
        
        # Verify sessions exist
        initial_count = session_manager.get_session_count()
        assert initial_count >= 3
        
        # Cleanup all sessions
        cleaned_count = session_manager.cleanup_all_sessions()
        assert cleaned_count >= 3
        
        # Verify cleanup
        final_count = session_manager.get_session_count()
        assert final_count == 0


class TestUploadEndpoint:
    """Test enhanced upload endpoint with Phase 3 features."""

    def test_upload_status_endpoint(self):
        """Test upload status endpoint."""
        response = client.get("/api/v1/upload/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "service" in data
        assert "capabilities" in data
        assert "current_load" in data
        assert "yake_service" in data

    def test_upload_endpoint_validation(self):
        """Test upload endpoint input validation."""
        # Test missing file
        response = client.post("/api/v1/upload_pdf")
        assert response.status_code == 422  # Validation error

    def test_upload_file_size_validation(self):
        """Test file size validation."""
        # Create a large file content (mock)
        large_content = b"x" * (60 * 1024 * 1024)  # 60MB - over limit
        
        response = client.post(
            "/api/v1/upload_pdf",
            files={"file": ("large_file.pdf", large_content, "application/pdf")}
        )
        
        # Should return file too large error
        assert response.status_code in [413, 422, 400]


class TestAskEndpoint:
    """Test enhanced ask endpoint with context injection."""

    def test_ask_endpoint_without_session(self):
        """Test ask endpoint with invalid session."""
        response = client.post(
            "/api/v1/ask",
            json={
                "session_id": "invalid-session-id",
                "query": "What is this document about?"
            }
        )
        
        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False
        assert "SESSION_NOT_FOUND" in data["error_code"]

    def test_ask_endpoint_validation(self):
        """Test ask endpoint input validation."""
        # Test missing required fields
        response = client.post("/api/v1/ask", json={})
        assert response.status_code == 422

    def test_session_status_endpoint(self):
        """Test session status endpoint."""
        # Create a session first
        session_id = session_manager.create_session()
        
        response = client.get(f"/api/v1/session/{session_id}/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["session_id"] == session_id
        assert "active" in data
        assert "memory_usage_mb" in data

    def test_session_deletion(self):
        """Test session deletion endpoint."""
        # Create a session
        session_id = session_manager.create_session()
        
        # Delete session
        response = client.delete(f"/api/v1/session/{session_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True


class TestSessionManagementEndpoints:
    """Test session management endpoints."""

    def test_list_sessions_endpoint(self):
        """Test session listing endpoint."""
        # Create some test sessions
        session_ids = []
        for i in range(3):
            session_id = session_manager.create_session()
            session_ids.append(session_id)
        
        response = client.get("/api/v1/sessions")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "total_sessions" in data
        assert "sessions" in data
        assert len(data["sessions"]) >= 3

    def test_session_cleanup_endpoint(self):
        """Test session cleanup endpoint."""
        # Create some sessions
        for i in range(2):
            session_manager.create_session()
        
        response = client.post("/api/v1/sessions/cleanup?cleanup_type=all&force=true")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "sessions_cleaned" in data

    def test_session_details_endpoint(self):
        """Test session details endpoint."""
        session_id = session_manager.create_session()
        
        response = client.get(f"/api/v1/sessions/{session_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["session_id"] == session_id
        assert "memory_usage_mb" in data

    def test_system_stats_endpoint(self):
        """Test system session statistics endpoint."""
        response = client.get("/api/v1/sessions/stats/system")
        assert response.status_code == 200
        
        data = response.json()
        assert "active_sessions" in data
        assert "capacity_used_percent" in data


class TestVoiceStreamingEndpoints:
    """Test streaming voice response capabilities."""

    def test_voice_streaming_sessions_endpoint(self):
        """Test voice streaming sessions endpoint."""
        response = client.get("/api/v1/voice_stream/sessions")
        assert response.status_code == 200
        
        data = response.json()
        assert "active_streaming_sessions" in data
        assert "sessions" in data

    def test_voice_stream_upload_and_ask_validation(self):
        """Test voice streaming upload endpoint validation."""
        response = client.post("/api/v1/voice_stream/upload_and_ask")
        assert response.status_code == 422  # Missing required fields


class TestSystemEndpoints:
    """Test system monitoring and error handling endpoints."""

    def test_comprehensive_health_check(self):
        """Test comprehensive system health check."""
        response = client.get("/api/v1/system/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "components" in data
        assert "error_handling" in data
        assert "performance" in data

    def test_error_statistics_endpoint(self):
        """Test error statistics endpoint."""
        response = client.get("/api/v1/system/errors/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_errors" in data
        assert "error_categories_info" in data
        assert "system_health" in data

    def test_error_handling_test_endpoint(self):
        """Test error handling test endpoint."""
        response = client.get("/api/v1/system/errors/test?error_type=validation")
        assert response.status_code == 200
        
        data = response.json()
        assert data["test_result"] == "success"
        assert "error_response" in data

    def test_error_stats_reset_endpoint(self):
        """Test error statistics reset endpoint."""
        response = client.post("/api/v1/system/errors/reset?confirm=true")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True


class TestErrorHandling:
    """Test comprehensive error handling system."""

    def test_error_classification(self):
        """Test error classification and handling."""
        from app.core.error_handler import error_handler, ErrorCategory, ErrorSeverity
        
        # Test different error types
        validation_error = ValueError("Invalid input")
        category, severity = error_handler.classify_error(validation_error)
        assert category in ErrorCategory
        assert severity in ErrorSeverity

    def test_error_response_creation(self):
        """Test error response creation."""
        from app.core.error_handler import error_handler
        
        test_error = RuntimeError("Test error")
        error_response = error_handler.create_error_response(
            test_error,
            context={"test": True}
        )
        
        assert error_response.success is False
        assert error_response.error_code is not None
        assert error_response.message is not None
        assert error_response.timestamp is not None

    def test_fallback_responses(self):
        """Test that fallback responses are available."""
        from app.core.error_handler import error_handler, ErrorCategory
        
        # Verify fallback responses exist for all categories
        for category in ErrorCategory:
            assert category in error_handler.fallback_responses
            assert len(error_handler.fallback_responses[category]) > 0


class TestIntegrationWorkflow:
    """Test complete workflow integration."""

    async def test_complete_document_processing_workflow(self, sample_text_document):
        """Test complete workflow from document upload to query."""
        # 1. Create session
        session_id = session_manager.create_session()
        assert session_id is not None
        
        # 2. Simulate document processing
        parser_service = DocumentParserService()
        parsed_doc = await parser_service.parse_document(
            file_content=sample_text_document.encode('utf-8'),
            filename="integration_test.txt"
        )
        
        # 3. Create YAKE index
        yake_service = YAKEService()
        document_index = await yake_service.create_document_index(parsed_doc)
        
        # 4. Update session with processed data
        success = session_manager.update_session(
            session_id,
            parsed_document=parsed_doc,
            document_index=document_index,
            document_text=sample_text_document
        )
        assert success is True
        
        # 5. Verify session contains all data
        session = session_manager.get_session(session_id)
        assert session.parsed_document is not None
        assert session.document_index is not None
        assert session.document_text is not None
        
        # 6. Test session statistics
        session.update_query_time()
        assert session.query_count > 0

    def test_error_recovery_workflow(self):
        """Test system recovery from various error conditions."""
        # Test session recovery after cleanup
        initial_count = session_manager.get_session_count()
        
        # Create sessions
        session_ids = []
        for i in range(3):
            session_id = session_manager.create_session()
            session_ids.append(session_id)
        
        # Force cleanup
        session_manager.cleanup_all_sessions()
        assert session_manager.get_session_count() == 0
        
        # Test system can create new sessions after cleanup
        new_session_id = session_manager.create_session()
        assert new_session_id is not None
        assert session_manager.get_session_count() == 1


# Run specific test groups
if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--durations=10"
    ])