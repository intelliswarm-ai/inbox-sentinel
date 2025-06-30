"""
Logistic Regression detector implementation
"""

import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from inbox_sentinel.core.base_detector import BaseDetector
from inbox_sentinel.core.types import Email, PredictionResult
from inbox_sentinel.core.constants import MODEL_LOGISTIC_REGRESSION
from inbox_sentinel.core.exceptions import PredictionError, ModelNotTrainedError
from inbox_sentinel.config import get_model_config
from inbox_sentinel.ml.preprocessing import EmailPreprocessor


class LogisticRegressionDetector(BaseDetector):
    """Logistic Regression spam/phishing detector"""
    
    def __init__(self):
        super().__init__(
            model_name=MODEL_LOGISTIC_REGRESSION,
            algorithm="Logistic Regression"
        )
        self.config = get_model_config().get_model_params(MODEL_LOGISTIC_REGRESSION)
        self.scaler = StandardScaler()
        
    async def analyze(self, email: Email) -> PredictionResult:
        """Analyze email for spam/phishing using Logistic Regression"""
        if not self.is_trained:
            raise ModelNotTrainedError(f"{self.algorithm} model not trained")
        
        try:
            # Preprocess email
            features = self.preprocessor.transform(
                email.content,
                email.subject,
                email.sender
            )
            
            # Use combined features
            X = features['combined'].reshape(1, -1)
            
            # Scale features
            if self.scaler is not None:
                X = self.scaler.transform(X)
            
            # Get prediction and probabilities
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            
            # Get feature importance
            feature_importance = self._get_feature_importance(X)
            
            # Calculate confidence
            confidence = abs(probabilities[1] - probabilities[0])
            
            # Get decision function value
            decision_value = self.model.decision_function(X)[0]
            
            return PredictionResult(
                model_name=self.model_name,
                algorithm=self.algorithm,
                is_spam=bool(prediction == 1),
                spam_probability=float(probabilities[1]),
                ham_probability=float(probabilities[0]),
                confidence=float(confidence),
                features=feature_importance,
                metadata={
                    'decision_value': float(decision_value),
                    'regularization': self.model.C,
                    'solver': self.model.solver
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
                   C: float = 1.0,
                   solver: str = "lbfgs",
                   max_iter: int = 1000,
                   **kwargs) -> Dict[str, Any]:
        """Train the Logistic Regression model"""
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
            
            # Scale features (important for logistic regression)
            X = self.scaler.fit_transform(X)
            
            # Initialize and train Logistic Regression
            lr_params = self.config.default_params.copy()
            lr_params.update({
                'C': C,
                'solver': solver,
                'max_iter': max_iter,
                'random_state': 42,
                'n_jobs': -1  # Use all available cores
            })
            
            self.model = LogisticRegression(**lr_params)
            self.model.fit(X, y)
            self.is_trained = True
            
            # Calculate training accuracy
            train_predictions = self.model.predict(X)
            accuracy = np.mean(train_predictions == y)
            
            # Get model coefficients info
            coef_stats = {
                'n_features': len(self.model.coef_[0]),
                'n_nonzero': int(np.sum(self.model.coef_[0] != 0)),
                'coef_mean': float(np.mean(np.abs(self.model.coef_[0]))),
                'coef_std': float(np.std(self.model.coef_[0]))
            }
            
            return {
                'success': True,
                'C': C,
                'solver': solver,
                'max_iter': max_iter,
                'samples_trained': len(training_samples),
                'training_accuracy': float(accuracy),
                'spam_samples': int(np.sum(y)),
                'ham_samples': len(y) - int(np.sum(y)),
                'feature_count': X.shape[1],
                'coefficient_stats': coef_stats,
                'converged': bool(self.model.n_iter_ < max_iter)
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
        """Extract feature importance from Logistic Regression coefficients"""
        feature_importance = []
        
        if hasattr(self.model, 'coef_'):
            coef = self.model.coef_[0]
            feature_names = self.preprocessor.get_feature_names()
            
            # Get all feature names
            all_features = []
            all_features.extend([(f, 'tfidf') for f in feature_names['tfidf']])
            all_features.extend([(f, 'manual') for f in feature_names['manual']])
            
            # Get indices of top features by absolute coefficient value
            top_indices = np.argsort(np.abs(coef))[-10:]
            
            for idx in top_indices:
                if idx < len(all_features):
                    feature_name, feature_type = all_features[idx]
                    feature_importance.append({
                        'feature': feature_name,
                        'coefficient': float(coef[idx]),
                        'abs_importance': float(abs(coef[idx])),
                        'type': feature_type,
                        'contribution': 'spam' if coef[idx] > 0 else 'ham'
                    })
        
        return sorted(feature_importance, key=lambda x: x['abs_importance'], reverse=True)[:5]
    
    def _get_model_parameters(self) -> Dict[str, Any]:
        """Get model-specific parameters"""
        if self.model is None:
            return {}
        
        params = {
            'algorithm': type(self.model).__name__,
            'C': float(self.model.C),
            'solver': self.model.solver,
            'max_iter': self.model.max_iter,
            'multi_class': self.model.multi_class
        }
        
        if hasattr(self.model, 'n_iter_'):
            params['iterations_used'] = int(self.model.n_iter_[0])
            params['converged'] = bool(self.model.n_iter_ < self.model.max_iter)
        
        if hasattr(self.model, 'classes_'):
            params['classes'] = self.model.classes_.tolist()
        
        if hasattr(self.model, 'intercept_'):
            params['intercept'] = float(self.model.intercept_[0])
        
        return params