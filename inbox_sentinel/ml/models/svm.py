"""
SVM (Support Vector Machine) detector implementation
"""

import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from inbox_sentinel.core.base_detector import BaseDetector
from inbox_sentinel.core.types import Email, PredictionResult
from inbox_sentinel.core.constants import MODEL_SVM
from inbox_sentinel.core.exceptions import PredictionError, ModelNotTrainedError
from inbox_sentinel.config import get_model_config
from inbox_sentinel.ml.preprocessing import EmailPreprocessor


class SVMDetector(BaseDetector):
    """Support Vector Machine spam/phishing detector"""
    
    def __init__(self, kernel: str = "rbf"):
        super().__init__(
            model_name=MODEL_SVM,
            algorithm="Support Vector Machine"
        )
        self.kernel = kernel
        self.config = get_model_config().get_model_params(MODEL_SVM)
        self.scaler = StandardScaler()
        
    async def analyze(self, email: Email) -> PredictionResult:
        """Analyze email for spam/phishing using SVM"""
        if not self.is_trained:
            raise ModelNotTrainedError(f"{self.algorithm} model not trained")
        
        try:
            # Preprocess email
            features = self.preprocessor.transform(
                email.content,
                email.subject,
                email.sender
            )
            
            # Use combined features for SVM
            X = features['combined'].reshape(1, -1)
            
            # Scale features
            if self.scaler is not None:
                X = self.scaler.transform(X)
            
            # Get prediction
            prediction = self.model.predict(X)[0]
            
            # Get decision function value (distance from hyperplane)
            decision_value = self.model.decision_function(X)[0]
            
            # Convert decision value to probability-like score
            confidence = 1 / (1 + np.exp(-abs(decision_value)))
            
            # If model has probability=True, use predict_proba
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X)[0]
                spam_probability = probabilities[1]
                ham_probability = probabilities[0]
            else:
                # Estimate probabilities from decision function
                prob = 1 / (1 + np.exp(-decision_value))
                spam_probability = prob if prediction == 1 else 1 - prob
                ham_probability = 1 - spam_probability
            
            # Get feature importance
            feature_importance = self._get_feature_importance(X)
            
            return PredictionResult(
                model_name=self.model_name,
                algorithm=self.algorithm,
                is_spam=bool(prediction == 1),
                spam_probability=float(spam_probability),
                ham_probability=float(ham_probability),
                confidence=float(confidence),
                features=feature_importance,
                metadata={
                    'kernel': self.kernel,
                    'decision_value': float(decision_value),
                    'support_vectors': self._get_support_info()
                }
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
                   kernel: Optional[str] = None,
                   C: float = 1.0,
                   gamma: str = "scale",
                   **kwargs) -> Dict[str, Any]:
        """Train the SVM model"""
        if kernel:
            self.kernel = kernel
        
        try:
            # Initialize preprocessor
            self.preprocessor = EmailPreprocessor(
                max_features=self.config.preprocessing_params.get("max_features", 5000)
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
            
            # Transform all emails
            X_list = []
            for content, subject, sender in emails:
                features = self.preprocessor.transform(content, subject, sender)
                X_list.append(features['combined'])
            
            X = np.array(X_list)
            y = np.array(labels)
            
            # Scale features (important for SVM)
            X = self.scaler.fit_transform(X)
            
            # Initialize and train SVM
            svm_params = self.config.default_params.copy()
            svm_params.update({
                'kernel': self.kernel,
                'C': C,
                'gamma': gamma,
                'probability': True,  # Enable probability estimates
                'random_state': 42
            })
            
            self.model = SVC(**svm_params)
            self.model.fit(X, y)
            self.is_trained = True
            
            # Calculate training accuracy
            train_predictions = self.model.predict(X)
            accuracy = np.mean(train_predictions == y)
            
            return {
                'success': True,
                'kernel': self.kernel,
                'C': C,
                'gamma': gamma,
                'samples_trained': len(training_samples),
                'training_accuracy': float(accuracy),
                'spam_samples': int(np.sum(y)),
                'ham_samples': len(y) - int(np.sum(y)),
                'feature_count': X.shape[1],
                'n_support_vectors': int(np.sum(self.model.n_support_)),
                'support_vector_classes': {
                    'ham': int(self.model.n_support_[0]),
                    'spam': int(self.model.n_support_[1])
                }
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
    
    def _get_feature_importance(self, X: np.ndarray) -> List[Dict[str, Any]]:
        """Extract feature importance for SVM"""
        feature_importance = []
        
        # Feature importance (approximation using support vectors)
        if hasattr(self.model, 'coef_'):
            # For linear kernel
            coef = self.model.coef_[0]
            feature_names = self.preprocessor.get_feature_names()
            
            # Get top contributing features
            top_indices = np.argsort(np.abs(coef))[-10:]
            for idx in top_indices:
                if idx < len(feature_names['tfidf']):
                    feature_name = feature_names['tfidf'][idx]
                else:
                    manual_idx = idx - len(feature_names['tfidf'])
                    if manual_idx < len(feature_names['manual']):
                        feature_name = feature_names['manual'][manual_idx]
                    else:
                        feature_name = f"feature_{idx}"
                
                feature_importance.append({
                    'feature': feature_name,
                    'weight': float(coef[idx])
                })
        
        return feature_importance[-5:] if feature_importance else []
    
    def _get_support_info(self) -> Dict[str, Any]:
        """Get support vector information"""
        if not hasattr(self.model, 'support_vectors_'):
            return {}
        
        return {
            'n_support_vectors': int(np.sum(self.model.n_support_)),
            'support_vector_ratio': float(np.sum(self.model.n_support_) / len(self.model.support_vectors_))
        }
    
    def _get_model_parameters(self) -> Dict[str, Any]:
        """Get model-specific parameters"""
        if self.model is None:
            return {}
        
        params = {
            'kernel': self.kernel,
            'algorithm': type(self.model).__name__
        }
        
        if hasattr(self.model, 'C'):
            params['C'] = float(self.model.C)
        
        if hasattr(self.model, 'gamma'):
            params['gamma'] = str(self.model.gamma) if hasattr(self.model, 'gamma') else 'N/A'
        
        if hasattr(self.model, 'support_vectors_'):
            params['support_vectors'] = {
                'total': int(len(self.model.support_vectors_)),
                'per_class': {
                    'ham': int(self.model.n_support_[0]),
                    'spam': int(self.model.n_support_[1])
                }
            }
        
        if hasattr(self.model, 'classes_'):
            params['classes'] = self.model.classes_.tolist()
        
        return params