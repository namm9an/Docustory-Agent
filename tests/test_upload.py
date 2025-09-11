import pytest
from fastapi.testclient import TestClient
from io import BytesIO
from app.main import app
from app.core.config import settings

client = TestClient(app)


def test_upload_pdf_success():
    """Test successful PDF upload."""
    # Create a mock PDF file
    pdf_content = b"%PDF-1.4 mock pdf content"
    pdf_file = BytesIO(pdf_content)
    
    response = client.post(
        f"{settings.API_V1_STR}/upload_pdf",
        files={"file": ("test.pdf", pdf_file, "application/pdf")}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["success"] is True
    assert "session_id" in data
    assert data["document_metadata"]["filename"] == "test.pdf"
    assert data["document_metadata"]["file_type"] == ".pdf"
    assert data["processing_status"] == "placeholder_complete"


def test_upload_docx_success():
    """Test successful DOCX upload."""
    # Create a mock DOCX file
    docx_content = b"mock docx content"
    docx_file = BytesIO(docx_content)
    
    response = client.post(
        f"{settings.API_V1_STR}/upload_pdf",
        files={"file": ("test.docx", docx_file, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["success"] is True
    assert data["document_metadata"]["filename"] == "test.docx"
    assert data["document_metadata"]["file_type"] == ".docx"


def test_upload_invalid_file_type():
    """Test upload with invalid file type."""
    txt_content = b"This is a text file"
    txt_file = BytesIO(txt_content)
    
    response = client.post(
        f"{settings.API_V1_STR}/upload_pdf",
        files={"file": ("test.txt", txt_file, "text/plain")}
    )
    
    assert response.status_code == 400
    data = response.json()
    
    assert data["success"] is False
    assert "Unsupported file type" in data["message"]


def test_upload_no_filename():
    """Test upload without filename."""
    content = b"some content"
    
    response = client.post(
        f"{settings.API_V1_STR}/upload_pdf",
        files={"file": (None, BytesIO(content), "application/pdf")}
    )
    
    assert response.status_code == 400