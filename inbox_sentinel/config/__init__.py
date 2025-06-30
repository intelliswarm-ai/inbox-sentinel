"""
Configuration management for Inbox Sentinel
"""

from .settings import Settings, get_settings
from .model_config import ModelConfig, get_model_config

__all__ = [
    "Settings",
    "get_settings",
    "ModelConfig",
    "get_model_config",
]