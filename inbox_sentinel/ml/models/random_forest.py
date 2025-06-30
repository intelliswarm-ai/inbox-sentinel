"""
Random Forest detector implementation
"""

import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.ensemble import RandomForestClassifier

from inbox_sentinel.core.base_detector import BaseDetector
from inbox_sentinel.core.types import Email, PredictionResult
from inbox_sentinel.core.constants import MODEL_RANDOM_FOREST
from inbox_sentinel.core.exceptions import PredictionError, ModelNotTrainedError
from inbox_sentinel.config import get_model_config
from inbox_sentinel.ml.preprocessing import EmailPreprocessor


class RandomForestDetector(BaseDetector):
    """Random Forest spam/phishing detector"""
    
    def __init__(self):
        super().__init__(
            model_name=MODEL_RANDOM_FOREST,
            algorithm="Random Forest"
        )
        self.config = get_model_config().get_model_params(MODEL_RANDOM_FOREST)
        
    async def analyze(self, email: Email) -> PredictionResult:
        """Analyze email for spam/phishing using Random Forest"""
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
            
            # Get prediction and probabilities
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            
            # Get feature importance
            feature_importance = self._get_feature_importance()
            
            # Calculate confidence
            confidence = abs(probabilities[1] - probabilities[0])
            
            return PredictionResult(
                model_name=self.model_name,
                algorithm=self.algorithm,
                is_spam=bool(prediction == 1),
                spam_probability=float(probabilities[1]),
                ham_probability=float(probabilities[0]),
                confidence=float(confidence),
                features=feature_importance,
                metadata={
                    'n_estimators': self.model.n_estimators,
                    'max_depth': self.model.max_depth,
                    'oob_score': getattr(self.model, 'oob_score_', None)
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
                   n_estimators: int = 100,
                   max_depth: Optional[int] = None,
                   min_samples_split: int = 2,
                   **kwargs) -> Dict[str, Any]:
        """Train the Random Forest model"""
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
            
            # Initialize and train Random Forest
            rf_params = self.config.default_params.copy()
            rf_params.update({
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'random_state': 42,
                'oob_score': True,  # Out-of-bag score for validation
                'n_jobs': -1  # Use all available cores
            })
            
            self.model = RandomForestClassifier(**rf_params)
            self.model.fit(X, y)
            self.is_trained = True
            
            # Calculate training accuracy
            train_predictions = self.model.predict(X)
            accuracy = np.mean(train_predictions == y)
            
            # Get out-of-bag score if available
            oob_score = getattr(self.model, 'oob_score_', None)
            
            return {
                'success': True,
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'samples_trained': len(training_samples),
                'training_accuracy': float(accuracy),
                'oob_score': float(oob_score) if oob_score else None,
                'spam_samples': int(np.sum(y)),
                'ham_samples': len(y) - int(np.sum(y)),
                'feature_count': X.shape[1],
                'n_trees': len(self.model.estimators_)
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
    
    def _get_feature_importance(self) -> List[Dict[str, Any]]:
        """Extract feature importance from Random Forest"""
        feature_importance = []
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_names = self.preprocessor.get_feature_names()
            
            # Get all feature names
            all_features = []
            all_features.extend([(f, 'tfidf') for f in feature_names['tfidf']])
            all_features.extend([(f, 'manual') for f in feature_names['manual']])
            
            # Get indices of top features
            top_indices = np.argsort(importances)[-10:]
            
            for idx in top_indices:
                if idx < len(all_features):
                    feature_name, feature_type = all_features[idx]
                    feature_importance.append({
                        'feature': feature_name,
                        'importance': float(importances[idx]),
                        'type': feature_type
                    })
        
        return sorted(feature_importance, key=lambda x: x['importance'], reverse=True)[:5]
    
    def _get_model_parameters(self) -> Dict[str, Any]:
        """Get model-specific parameters"""
        if self.model is None:
            return {}
        
        params = {
            'algorithm': type(self.model).__name__,
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'min_samples_split': self.model.min_samples_split,
            'min_samples_leaf': self.model.min_samples_leaf,
            'n_trees': len(self.model.estimators_)
        }
        
        if hasattr(self.model, 'oob_score_'):
            params['oob_score'] = float(self.model.oob_score_)
        
        if hasattr(self.model, 'classes_'):
            params['classes'] = self.model.classes_.tolist()
        
        return params