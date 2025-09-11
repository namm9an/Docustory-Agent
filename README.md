# Docustory.in - Voice-First PDF Intelligence Agent

A production-ready, session-based document processing and Q&A system with voice capabilities.

## 🎯 Overview

Docustory.in is a professional voice-first PDF/DOCX intelligence agent that provides session-based document processing and interactive Q&A capabilities. Built with modern Python frameworks and production-quality error handling.

### Key Features

- **Document Processing**: PDF and DOCX parsing with text extraction and metadata
- **Session Management**: RAM-only session storage with automatic cleanup (10min timeout)
- **AI-Powered Q&A**: Question answering with Qwen 2.5 14B integration
- **Voice Support**: Speech-to-text (Whisper) and text-to-speech (Vibe Voice) capabilities
- **Keyword Extraction**: Optional YAKE-based keyword indexing for improved search
- **Production-Ready**: Comprehensive error handling, logging, and monitoring
- **Scalability**: Concurrent session support (up to 15 active sessions)

### Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI       │    │   AI Services   │
│   (Next.js)     │───▶│   Backend       │───▶│   - Qwen 2.5    │
└─────────────────┘    │                 │    │   - Whisper     │
                       │   - Session Mgr │    │   - Vibe Voice  │
                       │   - Doc Parser  │    └─────────────────┘
                       │   - Error Handle│    
                       └─────────────────┘    
                              │
                       ┌─────────────────┐
                       │   RAM Storage   │
                       │   (Sessions)    │
                       └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)
- AI model API keys (for production use)

### Installation

1. **Clone and setup:**
   ```bash
   git clone <repository-url>
   cd Docustory.in
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

4. **Run the server:**
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

5. **Access the API:**
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/api/v1/health
   - Upload Status: http://localhost:8000/api/v1/upload/status

## 📖 API Usage

### 1. Document Upload

Upload and process a PDF or DOCX document:

```bash
curl -X POST "http://localhost:8000/api/v1/upload_pdf" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_document.pdf"
```

**Response:**
```json
{
  "success": true,
  "message": "Document 'your_document.pdf' uploaded and processed successfully. 25 pages extracted.",
  "session_id": "123e4567-e89b-12d3-a456-426614174000",
  "document_metadata": {
    "filename": "your_document.pdf",
    "pages": 25,
    "title": "Document Title"
  },
  "processing_status": "complete",
  "processing_time": 2.34
}
```

### 2. Ask Questions (Text)

Ask questions about your uploaded document:

```bash
curl -X POST "http://localhost:8000/api/v1/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "123e4567-e89b-12d3-a456-426614174000",
    "query": "What are the main topics covered in this document?",
    "voice_enabled": false,
    "max_tokens": 1024,
    "temperature": 0.7
  }'
```

### 3. Ask Questions (Voice)

Ask questions using voice input:

```bash
curl -X POST "http://localhost:8000/api/v1/ask_voice" \
  -H "Content-Type: multipart/form-data" \
  -F "session_id=123e4567-e89b-12d3-a456-426614174000" \
  -F "voice_file=@question.wav" \
  -F "voice_enabled=true"
```

### 4. Session Management

Check session status:
```bash
curl "http://localhost:8000/api/v1/session/{session_id}/status"
```

Delete a session:
```bash
curl -X DELETE "http://localhost:8000/api/v1/session/{session_id}"
```

## 🏗️ Project Structure

```
Docustory.in/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── endpoints/
│   │       │   ├── health.py      # Health checks
│   │       │   ├── upload.py      # Document upload
│   │       │   └── ask.py         # Q&A endpoints
│   │       └── __init__.py        # API router
│   ├── core/
│   │   ├── config.py              # Configuration
│   │   └── session.py             # Session management
│   ├── models/
│   │   ├── health.py              # Health models
│   │   ├── upload.py              # Upload models
│   │   └── ask.py                 # Q&A models
│   ├── services/
│   │   ├── parsing.py             # Document parsing
│   │   ├── qwen_client.py         # Qwen AI client
│   │   ├── stt.py                 # Speech-to-text
│   │   └── tts.py                 # Text-to-speech
│   └── main.py                    # FastAPI application
├── tests/
│   ├── test_health.py
│   ├── test_upload.py
│   └── test_ask.py
├── requirements.txt
├── .env.example
└── README.md
```

## ⚙️ Configuration

Key configuration options in `.env`:

```env
# Application
APP_NAME="Docustory.in - Voice PDF Agent"
DEBUG=false
HOST="127.0.0.1"
PORT=8000

# File Processing Limits
MAX_FILE_SIZE_MB=50
MAX_PAGES=300

# Session Management
SESSION_TIMEOUT_MINUTES=10
MAX_CONCURRENT_SESSIONS=15

# AI Model Endpoints
QWEN_ENDPOINT="https://api.example.com/v1/qwen2.5"
QWEN_API_KEY="your-qwen-api-key-here"
WHISPER_ENDPOINT="https://api.example.com/v1/whisper"
WHISPER_API_KEY="your-whisper-api-key-here"
VIBE_VOICE_ENDPOINT="https://api.example.com/v1/vibe-voice"
VIBE_VOICE_API_KEY="your-vibe-voice-api-key-here"

# Optional Features
ENABLE_YAKE_SEARCH=true
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_ask.py

# Run with verbose output
pytest -v
```

## 🔧 Development

### Code Quality Tools

```bash
# Format code
black app/ tests/

# Lint code
flake8 app/ tests/

# Type checking
mypy app/

# Run all quality checks
black app/ tests/ && flake8 app/ tests/ && mypy app/ && pytest
```

### Adding New Features

1. **Create feature branch:** `git checkout -b feature/new-feature`
2. **Add tests first** (TDD approach)
3. **Implement feature** with proper error handling
4. **Update documentation** and type hints
5. **Run quality checks** before committing

## 📊 Monitoring and Observability

### Health Checks

- **Application Health:** `GET /api/v1/health`
- **Upload Service Status:** `GET /api/v1/upload/status`
- **Model Information:** `GET /api/v1/models/info`

### Logging

Structured logging with different levels:
- **INFO**: Normal operations, session creation/deletion
- **WARNING**: Recoverable errors, fallback usage
- **ERROR**: Failures requiring attention

### Session Monitoring

Monitor session usage and memory consumption:
```bash
curl "http://localhost:8000/api/v1/models/info" | jq '.session_stats'
```

## 🚨 Error Handling

The API uses comprehensive error handling with specific error codes:

- **SESSION_NOT_FOUND**: Session expired or doesn't exist
- **NO_DOCUMENT**: Session exists but no document uploaded
- **FILE_TOO_LARGE**: File exceeds size limits
- **UNSUPPORTED_FORMAT**: Invalid file type
- **PROCESSING_ERROR**: Document parsing failed
- **API_ERROR**: External AI service failure

All errors include:
- Specific error codes for programmatic handling
- Human-readable messages
- Suggestions for resolution
- Relevant metadata (session_id, processing_time, etc.)

## 📈 Performance and Scalability

### Current Limits

- **Max concurrent sessions:** 15
- **Max file size:** 50MB
- **Max pages:** 300
- **Session timeout:** 10 minutes
- **Memory per session:** ~100MB

### Performance Targets

- **Text Q&A:** <2s response time
- **Voice Q&A:** <4s with streaming
- **Document processing:** <2min for 300 pages
- **Memory usage:** Monitored and limited per session

## 🔒 Security

- **File validation**: Type and size checks
- **Memory limits**: Per-session and total memory monitoring
- **Session isolation**: Each session is independent
- **Input sanitization**: All inputs validated with Pydantic
- **Error information**: Sensitive details not exposed in error messages

## 🔄 Production Deployment

### Environment Setup

```bash
# Production environment variables
DEBUG=false
HOST="0.0.0.0"
PORT=8000

# Use production AI API endpoints
QWEN_ENDPOINT="https://production-qwen-api.com"
# ... etc
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Health Monitoring

Set up monitoring for:
- Application health endpoints
- Memory usage and session counts
- API response times
- Error rates and types

## 📝 API Documentation

Full interactive API documentation is available at `/docs` when running the server.

Key endpoints:
- `POST /api/v1/upload_pdf` - Upload document
- `POST /api/v1/ask` - Text-based Q&A
- `POST /api/v1/ask_voice` - Voice-based Q&A
- `GET /api/v1/ask_stream/{session_id}` - Streaming responses
- `GET /api/v1/session/{session_id}/status` - Session status
- `DELETE /api/v1/session/{session_id}` - Delete session

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the `/docs` endpoint for API documentation
- Review the error codes and troubleshooting guide

---

**Built with production-quality standards:** Comprehensive error handling, structured logging, type safety, testing, and monitoring.