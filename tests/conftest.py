"""
Pytest configuration and fixtures for Docustory.in tests.

Provides common fixtures and configuration for the test suite.
"""

import pytest
import asyncio
import tempfile
import os
from typing import Generator, AsyncGenerator

from app.main import app
from app.core.session import session_manager
from app.core.error_handler import reset_error_stats


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
async def cleanup_test_environment():
    """Automatic cleanup before and after each test."""
    # Pre-test cleanup
    session_manager.cleanup_all_sessions()
    reset_error_stats()
    
    yield
    
    # Post-test cleanup
    session_manager.cleanup_all_sessions()


@pytest.fixture
def test_session_id():
    """Create a test session and return its ID."""
    session_id = session_manager.create_session()
    yield session_id
    # Cleanup handled by autouse fixture


@pytest.fixture
def sample_pdf_content():
    """Minimal valid PDF content for testing."""
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\nxref\n0 1\n0000000000 65535 f \ntrailer\n<<\n/Size 1\n/Root 1 0 R\n>>\nstartxref\n32\n%%EOF"


@pytest.fixture
def sample_text_content():
    """Sample text content for document processing tests."""
    return """
    Machine Learning and Artificial Intelligence

    Introduction to Machine Learning
    Machine learning is a subset of artificial intelligence that enables computers
    to learn and improve from experience without being explicitly programmed.

    Key Concepts:
    - Supervised Learning: Learning with labeled training data
    - Unsupervised Learning: Finding patterns in data without labels  
    - Neural Networks: Computing systems inspired by biological neural networks
    - Deep Learning: Machine learning using deep neural networks

    Applications:
    - Computer Vision: Image recognition and processing
    - Natural Language Processing: Understanding human language
    - Predictive Analytics: Forecasting future trends
    - Autonomous Systems: Self-driving cars and robotics

    The future of AI looks promising with continued advances in
    computational power and algorithmic sophistication.
    """


@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def mock_file_upload():
    """Mock file upload data."""
    return {
        "filename": "test_document.pdf",
        "content_type": "application/pdf",
        "size": 1024
    }


@pytest.fixture(scope="session")
def test_config():
    """Test configuration settings."""
    return {
        "test_timeout": 30,
        "max_file_size": 50 * 1024 * 1024,  # 50MB
        "test_session_timeout": 600,  # 10 minutes
        "mock_ai_responses": True
    }