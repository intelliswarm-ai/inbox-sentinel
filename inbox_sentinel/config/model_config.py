"""
Model-specific configurations
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from functools import lru_cache

from inbox_sentinel.core.constants import (
    MODEL_NAIVE_BAYES, MODEL_SVM, MODEL_RANDOM_FOREST,
    MODEL_LOGISTIC_REGRESSION, MODEL_NEURAL_NETWORK
)


@dataclass
class ModelParameters:
    """Parameters for a specific model"""
    name: str
    algorithm: str
    default_params: Dict[str, Any] = field(default_factory=dict)
    training_params: Dict[str, Any] = field(default_factory=dict)
    preprocessing_params: Dict[str, Any] = field(default_factory=dict)


class ModelConfig:
    """Configuration for all models"""
    
    def __init__(self):
        self._configs = {
            MODEL_NAIVE_BAYES: ModelParameters(
                name=MODEL_NAIVE_BAYES,
                algorithm="Multinomial Naive Bayes",
                default_params={"alpha": 1.0},
                training_params={"model_type": "multinomial"},
                preprocessing_params={"use_tfidf": True, "max_features": 5000}
            ),
            
            MODEL_SVM: ModelParameters(
                name=MODEL_SVM,
                algorithm="Support Vector Machine",
                default_params={
                    "kernel": "rbf",
                    "C": 1.0,
                    "gamma": "scale",
                    "probability": True
                },
                training_params={"max_iter": 10000},
                preprocessing_params={"use_scaling": True, "max_features": 2000}
            ),
            
            MODEL_RANDOM_FOREST: ModelParameters(
                name=MODEL_RANDOM_FOREST,
                algorithm="Random Forest",
                default_params={
                    "n_estimators": 100,
                    "max_depth": 20,
                    "min_samples_split": 5,
                    "random_state": 42,
                    "n_jobs": -1
                },
                training_params={},
                preprocessing_params={"use_combined_features": True, "max_features": 3000}
            ),
            
            MODEL_LOGISTIC_REGRESSION: ModelParameters(
                name=MODEL_LOGISTIC_REGRESSION,
                algorithm="Logistic Regression",
                default_params={
                    "penalty": "l2",
                    "C": 1.0,
                    "solver": "lbfgs",
                    "max_iter": 1000,
                    "random_state": 42
                },
                training_params={},
                preprocessing_params={"use_scaling": True, "max_features": 3000}
            ),
            
            MODEL_NEURAL_NETWORK: ModelParameters(
                name=MODEL_NEURAL_NETWORK,
                algorithm="Neural Network (MLP)",
                default_params={
                    "hidden_layer_sizes": (128, 64, 32),
                    "activation": "relu",
                    "solver": "adam",
                    "learning_rate_init": 0.001,
                    "max_iter": 500,
                    "early_stopping": True,
                    "validation_fraction": 0.1,
                    "random_state": 42
                },
                training_params={"n_iter_no_change": 20},
                preprocessing_params={"use_scaling": True, "max_features": 2000}
            )
        }
    
    def get_model_params(self, model_name: str) -> ModelParameters:
        """Get parameters for a specific model"""
        if model_name not in self._configs:
            raise ValueError(f"Unknown model: {model_name}")
        return self._configs[model_name]
    
    def get_all_models(self) -> Dict[str, ModelParameters]:
        """Get all model configurations"""
        return self._configs.copy()
    
    def update_model_params(self, model_name: str, 
                          default_params: Optional[Dict[str, Any]] = None,
                          training_params: Optional[Dict[str, Any]] = None,
                          preprocessing_params: Optional[Dict[str, Any]] = None):
        """Update parameters for a specific model"""
        if model_name not in self._configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = self._configs[model_name]
        
        if default_params:
            config.default_params.update(default_params)
        if training_params:
            config.training_params.update(training_params)
        if preprocessing_params:
            config.preprocessing_params.update(preprocessing_params)


@lru_cache()
def get_model_config() -> ModelConfig:
    """Get cached model configuration instance"""
    return ModelConfig()