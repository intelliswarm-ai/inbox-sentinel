"""
Orchestrator for coordinating multiple detection models
"""

import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging

from inbox_sentinel.core.types import Email, PredictionResult, ConsensusResult
from inbox_sentinel.core.constants import (
    STRATEGY_MAJORITY_VOTE,
    STRATEGY_WEIGHTED_AVERAGE,
    STRATEGY_HIGHEST_CONFIDENCE,
    STRATEGY_CONSERVATIVE,
    STRATEGY_AGGRESSIVE
)
from inbox_sentinel.ml.models import (
    NaiveBayesDetector,
    SVMDetector,
    RandomForestDetector,
    LogisticRegressionDetector,
    NeuralNetworkDetector
)


logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a model"""
    name: str
    detector: Any
    accuracy: float = 0.85  # Default accuracy, can be updated based on validation


class DetectionOrchestrator:
    """Orchestrates multiple detection models for ensemble predictions"""
    
    def __init__(self):
        self.models = self._initialize_models()
        self.logger = logger
        
    def _initialize_models(self) -> Dict[str, ModelInfo]:
        """Initialize all available models"""
        return {
            "naive_bayes": ModelInfo("Naive Bayes", NaiveBayesDetector(), accuracy=0.88),
            "svm": ModelInfo("SVM", SVMDetector(), accuracy=0.90),
            "random_forest": ModelInfo("Random Forest", RandomForestDetector(), accuracy=0.92),
            "logistic_regression": ModelInfo("Logistic Regression", LogisticRegressionDetector(), accuracy=0.89),
            "neural_network": ModelInfo("Neural Network", NeuralNetworkDetector(), accuracy=0.93)
        }
    
    async def analyze_ensemble(
        self,
        email: Email,
        strategy: str = STRATEGY_WEIGHTED_AVERAGE,
        models_to_use: Optional[List[str]] = None
    ) -> ConsensusResult:
        """
        Analyze email using ensemble of models
        
        Args:
            email: Email to analyze
            strategy: Consensus strategy to use
            models_to_use: List of model names to use (default: all)
            
        Returns:
            ConsensusResult with ensemble prediction
        """
        if models_to_use is None:
            models_to_use = list(self.models.keys())
        
        # Collect predictions from all models
        predictions = []
        errors = []
        
        # Run all model predictions concurrently
        tasks = []
        for model_name in models_to_use:
            if model_name not in self.models:
                continue
            
            model_info = self.models[model_name]
            task = self._get_model_prediction(model_info, email)
            tasks.append((model_name, task))
        
        # Gather all results
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        # Process results
        for (model_name, _), result in zip(tasks, results):
            model_info = self.models[model_name]
            
            if isinstance(result, Exception):
                errors.append({
                    'model': model_name,
                    'error': str(result)
                })
            elif result.error:
                errors.append({
                    'model': model_name,
                    'error': result.error
                })
            else:
                predictions.append({
                    'model': model_name,
                    'result': result,
                    'accuracy': model_info.accuracy
                })
        
        if not predictions:
            return ConsensusResult(
                is_spam=False,
                confidence=0.0,
                spam_probability=0.0,
                ham_probability=1.0,
                strategy=strategy,
                models_used=0,
                individual_predictions=[],
                errors=errors
            )
        
        # Apply consensus strategy
        consensus_result = self._apply_consensus_strategy(predictions, strategy)
        
        # Build individual predictions list
        individual_predictions = [
            {
                'model': p['model'],
                'is_spam': p['result'].is_spam,
                'confidence': p['result'].confidence,
                'spam_probability': p['result'].spam_probability,
                'algorithm': p['result'].algorithm
            }
            for p in predictions
        ]
        
        return ConsensusResult(
            is_spam=consensus_result['is_spam'],
            confidence=consensus_result['confidence'],
            spam_probability=consensus_result['spam_probability'],
            ham_probability=consensus_result['ham_probability'],
            strategy=strategy,
            models_used=len(predictions),
            individual_predictions=individual_predictions,
            errors=errors,
            explanation=consensus_result.get('explanation', '')
        )
    
    async def _get_model_prediction(self, model_info: ModelInfo, email: Email) -> PredictionResult:
        """Get prediction from a single model"""
        try:
            # Initialize model if needed
            if not model_info.detector.is_trained:
                await model_info.detector.initialize(use_pretrained=True)
            
            return await model_info.detector.analyze(email)
        except Exception as e:
            self.logger.error(f"Error getting prediction from {model_info.name}: {e}")
            raise
    
    def _apply_consensus_strategy(self, predictions: List[Dict], strategy: str) -> Dict[str, Any]:
        """Apply the specified consensus strategy to combine predictions"""
        
        if strategy == STRATEGY_MAJORITY_VOTE:
            spam_votes = sum(1 for p in predictions if p['result'].is_spam)
            total_votes = len(predictions)
            is_spam = spam_votes > total_votes / 2
            confidence = spam_votes / total_votes if is_spam else (total_votes - spam_votes) / total_votes
            
            return {
                'is_spam': is_spam,
                'confidence': confidence,
                'spam_probability': spam_votes / total_votes,
                'ham_probability': 1 - (spam_votes / total_votes),
                'explanation': f"{'Spam' if is_spam else 'Ham'} by majority vote ({spam_votes}/{total_votes})"
            }
        
        elif strategy == STRATEGY_WEIGHTED_AVERAGE:
            # Weight by model accuracy
            total_weight = sum(p['accuracy'] for p in predictions)
            weighted_spam_prob = sum(
                p['result'].spam_probability * p['accuracy'] 
                for p in predictions
            ) / total_weight
            
            is_spam = weighted_spam_prob > 0.5
            
            return {
                'is_spam': is_spam,
                'confidence': abs(weighted_spam_prob - 0.5) * 2,
                'spam_probability': weighted_spam_prob,
                'ham_probability': 1 - weighted_spam_prob,
                'explanation': f"Weighted average spam probability: {weighted_spam_prob:.2%}"
            }
        
        elif strategy == STRATEGY_HIGHEST_CONFIDENCE:
            # Use prediction with highest confidence
            best_pred = max(predictions, key=lambda p: p['result'].confidence)
            result = best_pred['result']
            
            return {
                'is_spam': result.is_spam,
                'confidence': result.confidence,
                'spam_probability': result.spam_probability,
                'ham_probability': result.ham_probability,
                'explanation': f"Highest confidence from {best_pred['model']} ({result.confidence:.2%})"
            }
        
        elif strategy == STRATEGY_CONSERVATIVE:
            # If any model says spam, it's spam
            any_spam = any(p['result'].is_spam for p in predictions)
            
            if any_spam:
                # Average of spam probabilities from models that detected spam
                spam_probs = [p['result'].spam_probability for p in predictions if p['result'].is_spam]
                avg_spam_prob = sum(spam_probs) / len(spam_probs)
                confidence = max(p['result'].confidence for p in predictions if p['result'].is_spam)
            else:
                avg_spam_prob = sum(p['result'].spam_probability for p in predictions) / len(predictions)
                confidence = sum(p['result'].confidence for p in predictions) / len(predictions)
            
            return {
                'is_spam': any_spam,
                'confidence': confidence,
                'spam_probability': avg_spam_prob,
                'ham_probability': 1 - avg_spam_prob,
                'explanation': f"Conservative: {'Spam' if any_spam else 'Ham'} (any model flagged)" if any_spam else "Conservative: Ham (no models flagged)"
            }
        
        elif strategy == STRATEGY_AGGRESSIVE:
            # Only spam if all models agree
            all_spam = all(p['result'].is_spam for p in predictions)
            
            # Average probabilities
            avg_spam_prob = sum(p['result'].spam_probability for p in predictions) / len(predictions)
            avg_confidence = sum(p['result'].confidence for p in predictions) / len(predictions)
            
            return {
                'is_spam': all_spam,
                'confidence': avg_confidence,
                'spam_probability': avg_spam_prob,
                'ham_probability': 1 - avg_spam_prob,
                'explanation': f"Aggressive: {'Spam' if all_spam else 'Ham'} ({'all' if all_spam else 'not all'} models agree)"
            }
        
        else:
            # Default to weighted average
            return self._apply_consensus_strategy(predictions, STRATEGY_WEIGHTED_AVERAGE)
    
    async def train_all_models(
        self,
        training_samples: List[Dict[str, Any]],
        models_to_train: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Train all specified models with the given samples"""
        if models_to_train is None:
            models_to_train = list(self.models.keys())
        
        results = {}
        
        for model_name in models_to_train:
            if model_name not in self.models:
                continue
            
            model_info = self.models[model_name]
            try:
                result = await model_info.detector.train(training_samples)
                results[model_name] = result
            except Exception as e:
                results[model_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    async def initialize_all_models(
        self,
        use_pretrained: bool = True,
        models_to_init: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Initialize all specified models"""
        if models_to_init is None:
            models_to_init = list(self.models.keys())
        
        results = {}
        
        for model_name in models_to_init:
            if model_name not in self.models:
                continue
            
            model_info = self.models[model_name]
            try:
                result = await model_info.detector.initialize(use_pretrained)
                results[model_name] = result
            except Exception as e:
                results[model_name] = {
                    'status': f'Failed to initialize: {str(e)}'
                }
        
        return results
    
    def get_models_info(self) -> Dict[str, Any]:
        """Get information about all available models"""
        info = {}
        
        for model_name, model_info in self.models.items():
            try:
                detector_info = model_info.detector.get_info()
                info[model_name] = {
                    'name': model_info.name,
                    'accuracy': model_info.accuracy,
                    'is_trained': detector_info.is_trained,
                    'algorithm': detector_info.algorithm,
                    'parameters': detector_info.parameters
                }
            except Exception as e:
                info[model_name] = {
                    'name': model_info.name,
                    'accuracy': model_info.accuracy,
                    'error': str(e)
                }
        
        return info