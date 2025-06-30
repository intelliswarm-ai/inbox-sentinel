"""
Application settings using Pydantic
"""

from typing import Optional, Dict, Any
from pathlib import Path
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError:
    from pydantic import BaseSettings, Field
from functools import lru_cache

from inbox_sentinel.core.constants import PROJECT_ROOT, MODELS_DIR, LOGS_DIR


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    app_name: str = "Inbox Sentinel"
    version: str = "1.0.0"
    debug: bool = Field(default=False)
    
    # Paths
    project_root: Path = PROJECT_ROOT
    models_dir: Path = MODELS_DIR
    logs_dir: Path = LOGS_DIR
    data_dir: Path = PROJECT_ROOT / "data"
    
    # Model settings
    use_pretrained_models: bool = Field(default=True)
    auto_save_models: bool = Field(default=True)
    model_cache_size: int = Field(default=5)
    
    # Training settings
    max_training_samples: int = Field(default=10000)
    test_size: float = Field(default=0.2)
    random_state: int = Field(default=42)
    
    # Feature extraction
    max_tfidf_features: int = Field(default=5000)
    min_word_length: int = Field(default=2)
    
    # MCP Server settings
    mcp_host: str = Field(default="127.0.0.1")
    mcp_base_port: int = Field(default=8000)
    
    # Logging
    log_level: str = Field(default="INFO")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Performance
    enable_caching: bool = Field(default=True)
    cache_ttl: int = Field(default=3600)  # seconds
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False
    }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()