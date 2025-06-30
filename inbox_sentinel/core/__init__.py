"""
Core functionality for Inbox Sentinel
"""

from .base_detector import BaseDetector
from .exceptions import *
from .constants import *
from .types import *

__all__ = [
    "BaseDetector",
]