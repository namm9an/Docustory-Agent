from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings and configuration."""
    
    # Application
    APP_NAME: str = "Docustory.in - Voice PDF Agent"
    APP_VERSION: str = "1.0.0"
    DESCRIPTION: str = "Voice-first PDF intelligence agent with session-based processing"
    DEBUG: bool = False
    
    # API
    API_V1_STR: str = "/api/v1"
    
    # CORS
    CORS_ORIGINS: List[str] = ["*"]  # Allow all origins for now
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    # File Upload Limits (Phase 5 Enhanced)
    MAX_FILE_SIZE_MB: int = 200  # 200MB max file size for Phase 5
    MAX_PAGES: int = 300  # Maximum pages per document
    ALLOWED_EXTENSIONS: List[str] = [".pdf", ".docx"]
    
    # Session Management (Phase 5 Enhanced)
    SESSION_TIMEOUT_MINUTES: int = 10
    MAX_CONCURRENT_SESSIONS: int = 15
    MAX_MEMORY_PER_SESSION_MB: int = 200  # Increased for Phase 5
    SESSION_CLEANUP_INTERVAL_SECONDS: int = 60  # Check for expired sessions every minute
    ENABLE_QUEUE_SYSTEM: bool = True  # Queue requests when at capacity
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = False
    
    # AI Model Endpoints (Phase 2)
    QWEN_ENDPOINT: str = "https://api.example.com/v1/qwen2.5"
    QWEN_API_KEY: str = "your-qwen-api-key-here"
    QWEN_MODEL: str = "qwen2.5-14b"
    
    WHISPER_ENDPOINT: str = "https://api.example.com/v1/whisper"
    WHISPER_API_KEY: str = "your-whisper-api-key-here"
    WHISPER_MODEL: str = "whisper-large"
    
    # TTS Service Configuration (Microsoft T5 Speech)
    TTS_ENDPOINT: str = "https://your-microsoft-t5-endpoint.com/v1/speech"
    TTS_API_KEY: str = "your-microsoft-t5-api-key-here"
    TTS_MODEL: str = "microsoft/t5-speech"
    TTS_PROVIDER: str = "microsoft"
    
    # Optional Features
    ENABLE_YAKE_SEARCH: bool = True
    YAKE_MAX_KEYWORDS: int = 20
    YAKE_NGRAM_SIZE: int = 3
    
    # Phase 4: Conversation Memory Configuration
    MAX_CONVERSATION_TURNS: int = 10  # Maximum conversation history turns to keep
    MAX_CONVERSATION_TOKENS: int = 4000  # Token limit for conversation context
    ENABLE_CONVERSATION_MEMORY: bool = True
    CONVERSATION_PRUNE_THRESHOLD: int = 8  # Start pruning when this many turns
    TOKEN_ESTIMATION_RATIO: float = 4.0  # Approximate chars per token for estimation
    
    # Phase 5: Error Handling & Performance Configuration
    ENABLE_ERROR_LOGGING: bool = True
    LOG_FILE_MAX_SIZE_MB: int = 5  # 5MB max log file size
    LOG_BACKUP_COUNT: int = 3  # Keep 3 backup log files
    LOG_LEVEL: str = "INFO"  # INFO, DEBUG, WARNING, ERROR
    
    # Performance Safeguards
    REQUEST_TIMEOUT_SECONDS: int = 300  # 5 minutes max request time
    AI_MODEL_TIMEOUT_SECONDS: int = 120  # 2 minutes max AI model response time
    QUEUE_MAX_SIZE: int = 50  # Maximum requests in queue
    QUEUE_TIMEOUT_SECONDS: int = 600  # 10 minutes max queue wait time
    
    # Error Response Configuration
    INCLUDE_ERROR_DETAILS: bool = True  # Include stack traces in dev mode
    ENABLE_ERROR_NOTIFICATIONS: bool = False  # Future: email/slack alerts
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()