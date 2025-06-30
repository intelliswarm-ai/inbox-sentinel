"""
Type definitions for Inbox Sentinel
"""

from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np


class ConsensusStrategy(Enum):
    """Strategies for combining predictions from multiple models"""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    HIGHEST_CONFIDENCE = "highest_confidence"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"


@dataclass
class Email:
    """Email data structure"""
    content: str
    subject: str
    sender: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary"""
        return {
            "email_content": self.content,
            "subject": self.subject,
            "sender": self.sender
        }


@dataclass
class PredictionResult:
    """Result from a single model prediction"""
    model_name: str
    algorithm: str
    is_spam: bool
    spam_probability: float
    ham_probability: float
    confidence: float
    features: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def is_valid(self) -> bool:
        """Check if prediction is valid (no errors)"""
        return self.error is None


@dataclass
class ConsensusResult:
    """Result from orchestrator consensus"""
    is_spam: bool
    confidence: float
    spam_probability: float
    ham_probability: float
    strategy: str
    models_used: int
    individual_predictions: List[Dict[str, Any]]
    errors: Optional[List[Dict[str, str]]] = None
    explanation: Optional[str] = None


@dataclass
class EnsembleResult:
    """Result from ensemble prediction"""
    is_spam: bool
    confidence: float
    strategy_used: ConsensusStrategy
    individual_predictions: List[PredictionResult]
    consensus_details: Dict[str, Any]
    final_indicators: List[str]
    
    @property
    def agreement_score(self) -> float:
        """Calculate agreement score among models"""
        if not self.individual_predictions:
            return 0.0
        spam_votes = sum(1 for p in self.individual_predictions if p.is_spam)
        return max(spam_votes, len(self.individual_predictions) - spam_votes) / len(self.individual_predictions)


@dataclass
class TrainingResult:
    """Result from model training"""
    success: bool
    model_name: str
    training_accuracy: float
    test_accuracy: Optional[float] = None
    training_samples: int = 0
    test_samples: int = 0
    feature_count: int = 0
    training_time: float = 0.0
    additional_metrics: Dict[str, Any] = None
    error: Optional[str] = None


@dataclass
class ModelInfo:
    """Information about a trained model"""
    model_name: str
    algorithm: str
    is_trained: bool
    accuracy: Optional[float] = None
    training_date: Optional[str] = None
    feature_count: Optional[int] = None
    parameters: Dict[str, Any] = None