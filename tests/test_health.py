import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings

client = TestClient(app)


def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get(f"{settings.API_V1_STR}/health")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "ok"
    assert data["version"] == settings.APP_VERSION
    assert "timestamp" in data
    assert "details" in data
    assert data["details"]["app_name"] == settings.APP_NAME


def test_ping_endpoint():
    """Test the ping endpoint."""
    response = client.get(f"{settings.API_V1_STR}/ping")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["pong"] is True
    assert "timestamp" in data


def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["service"] == settings.APP_NAME
    assert data["version"] == settings.APP_VERSION
    assert data["status"] == "operational"
    assert data["docs"] == "/docs"
    assert data["health"] == f"{settings.API_V1_STR}/health"