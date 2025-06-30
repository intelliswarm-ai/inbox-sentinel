"""
Base class for all email detectors
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
import pickle
from pathlib import Path

from inbox_sentinel.core.types import Email, PredictionResult, ModelInfo
from inbox_sentinel.core.exceptions import ModelNotFoundError, ModelNotTrainedError
from inbox_sentinel.core.constants import MODELS_DIR


class BaseDetector(ABC):
    """Abstract base class for email spam/phishing detectors"""
    
    def __init__(self, model_name: str, algorithm: str):
        self.model_name = model_name
        self.algorithm = algorithm
        self.model = None
        self.preprocessor = None
        self.is_trained = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    @abstractmethod
    async def analyze(self, email: Email) -> PredictionResult:
        """
        Analyze an email and return prediction result
        
        Args:
            email: Email object to analyze
            
        Returns:
            PredictionResult with classification details
        """
        pass
    
    @abstractmethod
    async def train(self, training_data: list, **kwargs) -> Dict[str, Any]:
        """
        Train the model with provided data
        
        Args:
            training_data: List of training samples
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training results
        """
        pass
    
    @abstractmethod
    async def initialize(self, use_pretrained: bool = True) -> Dict[str, str]:
        """
        Initialize the model
        
        Args:
            use_pretrained: Whether to load pre-trained model
            
        Returns:
            Dictionary with initialization status
        """
        pass
    
    async def get_info(self) -> ModelInfo:
        """Get information about the model"""
        return ModelInfo(
            model_name=self.model_name,
            algorithm=self.algorithm,
            is_trained=self.is_trained,
            parameters=self._get_model_parameters()
        )
    
    def _get_model_parameters(self) -> Dict[str, Any]:
        """Get model-specific parameters"""
        return {}
    
    def save_model(self, filepath: Optional[Path] = None) -> Path:
        """
        Save the trained model to disk
        
        Args:
            filepath: Optional custom filepath, defaults to standard location
            
        Returns:
            Path where model was saved
        """
        if not self.is_trained:
            raise ModelNotTrainedError(f"Model {self.model_name} is not trained")
        
        if filepath is None:
            filepath = MODELS_DIR / f"{self.model_name}_model.pkl"
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'algorithm': self.algorithm,
            'model_name': self.model_name
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {filepath}")
        return filepath
    
    def load_model(self, filepath: Optional[Path] = None) -> bool:
        """
        Load a trained model from disk
        
        Args:
            filepath: Optional custom filepath, defaults to standard location
            
        Returns:
            True if model loaded successfully
        """
        if filepath is None:
            filepath = MODELS_DIR / f"{self.model_name}_model.pkl"
        
        if not filepath.exists():
            raise ModelNotFoundError(f"Model file not found: {filepath}")
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.preprocessor = model_data.get('preprocessor')
            self.is_trained = True
            
            self.logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise