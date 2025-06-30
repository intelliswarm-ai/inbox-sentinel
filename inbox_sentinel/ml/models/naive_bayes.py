"""
Naive Bayes detector implementation
"""

import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.naive_bayes import MultinomialNB, GaussianNB

from inbox_sentinel.core.base_detector import BaseDetector
from inbox_sentinel.core.types import Email, PredictionResult
from inbox_sentinel.core.constants import MODEL_NAIVE_BAYES
from inbox_sentinel.core.exceptions import PredictionError, ModelNotTrainedError
from inbox_sentinel.config import get_model_config
from inbox_sentinel.ml.preprocessing import EmailPreprocessor


class NaiveBayesDetector(BaseDetector):
    """Naive Bayes spam/phishing detector"""
    
    def __init__(self, model_type: str = "multinomial"):
        super().__init__(
            model_name=MODEL_NAIVE_BAYES,
            algorithm="Naive Bayes"
        )
        self.model_type = model_type
        self.config = get_model_config().get_model_params(MODEL_NAIVE_BAYES)
        self.scaler = None
        
    async def analyze(self, email: Email) -> PredictionResult:
        """Analyze email for spam/phishing"""
        if not self.is_trained:
            raise ModelNotTrainedError(f"{self.algorithm} model not trained")
        
        try:
            # Preprocess email
            features = self.preprocessor.transform(
                email.content,
                email.subject,
                email.sender
            )
            
            # Use appropriate features for model type
            if self.model_type == "multinomial":
                X = features['tfidf'].reshape(1, -1)
            else:
                X = features['combined'].reshape(1, -1)
            
            # Get prediction and probabilities
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            
            # Extract feature importance
            feature_importance = self._get_feature_importance(X, prediction)
            
            # Calculate confidence
            confidence = abs(probabilities[1] - probabilities[0])
            
            return PredictionResult(
                model_name=self.model_name,
                algorithm=self.algorithm,
                is_spam=bool(prediction == 1),
                spam_probability=float(probabilities[1]),
                ham_probability=float(probabilities[0]),
                confidence=float(confidence),
                features=feature_importance
            )
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return PredictionResult(
                model_name=self.model_name,
                algorithm=self.algorithm,
                is_spam=False,
                spam_probability=0.0,
                ham_probability=1.0,
                confidence=0.0,
                error=str(e)
            )
    
    async def train(self, training_samples: List[Dict[str, Any]], 
                   model_type: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Train the Naive Bayes model"""
        if model_type:
            self.model_type = model_type
        
        try:
            # Initialize preprocessor
            self.preprocessor = EmailPreprocessor(
                max_features=self.config.preprocessing_params["max_features"]
            )
            
            # Prepare training data
            emails = []
            labels = []
            
            for sample in training_samples:
                emails.append((
                    sample['email_content'],
                    sample['subject'],
                    sample['sender']
                ))
                labels.append(1 if sample['is_spam'] else 0)
            
            # Fit preprocessor
            self.preprocessor.fit(emails, labels)
            
            # Transform emails
            X_list = []
            for content, subject, sender in emails:
                features = self.preprocessor.transform(content, subject, sender)
                if self.model_type == "multinomial":
                    X_list.append(features['tfidf'])
                else:
                    X_list.append(features['combined'])
            
            X = np.array(X_list)
            y = np.array(labels)
            
            # Initialize and train model
            if self.model_type == "multinomial":
                self.model = MultinomialNB(**self.config.default_params)
            else:
                self.model = GaussianNB()
            
            self.model.fit(X, y)
            self.is_trained = True
            
            # Calculate training accuracy
            train_predictions = self.model.predict(X)
            accuracy = np.mean(train_predictions == y)
            
            return {
                'success': True,
                'model_type': self.model_type,
                'samples_trained': len(training_samples),
                'training_accuracy': float(accuracy),
                'spam_samples': int(np.sum(y)),
                'ham_samples': len(y) - int(np.sum(y)),
                'feature_count': X.shape[1]
            }
            
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def initialize(self, use_pretrained: bool = True) -> Dict[str, str]:
        """Initialize the model"""
        if use_pretrained:
            try:
                self.load_model()
                return {'status': f'Loaded pre-trained {self.algorithm} model'}
            except Exception as e:
                self.logger.warning(f"Could not load pre-trained model: {e}")
        
        # Initialize with default samples if no pre-trained model
        return {'status': f'Initialized {self.algorithm} model (not trained)'}
    
    def _get_feature_importance(self, X: np.ndarray, prediction: int) -> List[Dict[str, Any]]:
        """Extract feature importance for the prediction"""
        feature_importance = []
        
        if hasattr(self.model, 'feature_log_prob_') and self.model_type == "multinomial":
            # For MultinomialNB
            log_probs = self.model.feature_log_prob_[prediction]
            feature_names = self.preprocessor.get_feature_names()['tfidf']
            
            # Get indices of features present in the email
            present_features = np.where(X[0] > 0)[0]
            
            # Sort by log probability
            important_indices = present_features[np.argsort(log_probs[present_features])[-10:]]
            
            for idx in important_indices:
                if idx < len(feature_names):
                    feature_importance.append({
                        'feature': feature_names[idx],
                        'importance': float(np.exp(log_probs[idx])),
                        'value': float(X[0, idx])
                    })
        
        return feature_importance[:5]  # Top 5 features
    
    def _get_model_parameters(self) -> Dict[str, Any]:
        """Get model-specific parameters"""
        if self.model is None:
            return {}
        
        params = {
            'model_type': self.model_type,
            'algorithm': type(self.model).__name__
        }
        
        if hasattr(self.model, 'class_count_'):
            params['class_distribution'] = {
                'ham': int(self.model.class_count_[0]),
                'spam': int(self.model.class_count_[1])
            }
        
        if hasattr(self.model, 'alpha'):
            params['alpha'] = float(self.model.alpha)
        
        return params