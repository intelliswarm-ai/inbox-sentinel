"""
Custom exceptions for Inbox Sentinel
"""


class InboxSentinelError(Exception):
    """Base exception for Inbox Sentinel"""
    pass


class ModelNotFoundError(InboxSentinelError):
    """Raised when a model file is not found"""
    pass


class ModelNotTrainedError(InboxSentinelError):
    """Raised when attempting to use an untrained model"""
    pass


class PreprocessingError(InboxSentinelError):
    """Raised when email preprocessing fails"""
    pass


class PredictionError(InboxSentinelError):
    """Raised when model prediction fails"""
    pass


class ConfigurationError(InboxSentinelError):
    """Raised when configuration is invalid"""
    pass