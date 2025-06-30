"""
Neural Network detector implementation
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from inbox_sentinel.core.base_detector import BaseDetector
from inbox_sentinel.core.types import Email, PredictionResult
from inbox_sentinel.core.constants import MODEL_NEURAL_NETWORK
from inbox_sentinel.core.exceptions import PredictionError, ModelNotTrainedError
from inbox_sentinel.config import get_model_config
from inbox_sentinel.ml.preprocessing import EmailPreprocessor


class NeuralNetworkDetector(BaseDetector):
    """Neural Network (Multi-Layer Perceptron) spam/phishing detector"""
    
    def __init__(self):
        super().__init__(
            model_name=MODEL_NEURAL_NETWORK,
            algorithm="Neural Network"
        )
        self.config = get_model_config().get_model_params(MODEL_NEURAL_NETWORK)
        self.scaler = StandardScaler()
        
    async def analyze(self, email: Email) -> PredictionResult:
        """Analyze email for spam/phishing using Neural Network"""
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
            
            # Scale features (crucial for neural networks)
            if self.scaler is not None:
                X = self.scaler.transform(X)
            
            # Get prediction and probabilities
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            
            # Get neuron activations for interpretability
            activations = self._get_activations(X)
            
            # Calculate confidence
            confidence = abs(probabilities[1] - probabilities[0])
            
            return PredictionResult(
                model_name=self.model_name,
                algorithm=self.algorithm,
                is_spam=bool(prediction == 1),
                spam_probability=float(probabilities[1]),
                ham_probability=float(probabilities[0]),
                confidence=float(confidence),
                features=self._get_feature_importance(X),
                metadata={
                    'hidden_layers': self.model.hidden_layer_sizes,
                    'activation': self.model.activation,
                    'n_layers': len(self.model.hidden_layer_sizes) + 2,  # input + hidden + output
                    'n_iter': self.model.n_iter_,
                    'loss': float(self.model.loss_),
                    'activations_summary': activations
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
                   hidden_layer_sizes: Tuple[int, ...] = (100, 50),
                   activation: str = "relu",
                   solver: str = "adam",
                   learning_rate_init: float = 0.001,
                   max_iter: int = 500,
                   **kwargs) -> Dict[str, Any]:
        """Train the Neural Network model"""
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
            
            # Scale features (crucial for neural networks)
            X = self.scaler.fit_transform(X)
            
            # Initialize and train Neural Network
            nn_params = self.config.default_params.copy()
            nn_params.update({
                'hidden_layer_sizes': hidden_layer_sizes,
                'activation': activation,
                'solver': solver,
                'learning_rate_init': learning_rate_init,
                'max_iter': max_iter,
                'random_state': 42,
                'early_stopping': True,
                'validation_fraction': 0.1,
                'n_iter_no_change': 10,
                'verbose': False
            })
            
            self.model = MLPClassifier(**nn_params)
            self.model.fit(X, y)
            self.is_trained = True
            
            # Calculate training accuracy
            train_predictions = self.model.predict(X)
            accuracy = np.mean(train_predictions == y)
            
            # Get network statistics
            network_stats = {
                'total_parameters': self._count_parameters(),
                'n_layers': len(hidden_layer_sizes) + 2,
                'neurons_per_layer': [X.shape[1]] + list(hidden_layer_sizes) + [2],
                'converged': self.model.n_iter_ < max_iter,
                'final_loss': float(self.model.loss_),
                'best_loss': float(self.model.best_loss_) if hasattr(self.model, 'best_loss_') else None
            }
            
            return {
                'success': True,
                'hidden_layer_sizes': hidden_layer_sizes,
                'activation': activation,
                'solver': solver,
                'learning_rate_init': learning_rate_init,
                'max_iter': max_iter,
                'samples_trained': len(training_samples),
                'training_accuracy': float(accuracy),
                'spam_samples': int(np.sum(y)),
                'ham_samples': len(y) - int(np.sum(y)),
                'feature_count': X.shape[1],
                'n_iterations': self.model.n_iter_,
                'network_stats': network_stats
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
    
    def _get_activations(self, X: np.ndarray) -> Dict[str, Any]:
        """Get summary of neuron activations in the network"""
        try:
            activations = {'layers': []}
            
            # Forward pass to get activations
            hidden_layer_sizes = list(self.model.hidden_layer_sizes)
            n_layers = len(hidden_layer_sizes)
            
            # Get activations for each hidden layer
            activation = X
            for i in range(n_layers):
                # Apply weights and biases
                activation = np.dot(activation, self.model.coefs_[i]) + self.model.intercepts_[i]
                
                # Apply activation function
                if self.model.activation == 'relu':
                    activation = np.maximum(0, activation)
                elif self.model.activation == 'tanh':
                    activation = np.tanh(activation)
                elif self.model.activation == 'logistic':
                    activation = 1 / (1 + np.exp(-activation))
                
                # Store activation statistics
                activations['layers'].append({
                    'layer': i + 1,
                    'n_neurons': hidden_layer_sizes[i],
                    'active_neurons': int(np.sum(activation > 0)),
                    'mean_activation': float(np.mean(activation)),
                    'max_activation': float(np.max(activation))
                })
            
            return activations
            
        except Exception as e:
            self.logger.warning(f"Could not get activations: {e}")
            return {}
    
    def _get_feature_importance(self, X: np.ndarray) -> List[Dict[str, Any]]:
        """
        Approximate feature importance for neural networks
        Using the gradient-based approach
        """
        feature_importance = []
        
        try:
            # Get input layer weights
            input_weights = self.model.coefs_[0]
            
            # Calculate importance as sum of absolute weights across all neurons in first hidden layer
            importance_scores = np.sum(np.abs(input_weights), axis=1)
            
            feature_names = self.preprocessor.get_feature_names()
            
            # Get all feature names
            all_features = []
            all_features.extend([(f, 'tfidf') for f in feature_names['tfidf']])
            all_features.extend([(f, 'manual') for f in feature_names['manual']])
            
            # Get indices of top features
            top_indices = np.argsort(importance_scores)[-10:]
            
            for idx in top_indices:
                if idx < len(all_features):
                    feature_name, feature_type = all_features[idx]
                    feature_importance.append({
                        'feature': feature_name,
                        'importance': float(importance_scores[idx]),
                        'type': feature_type,
                        'input_value': float(X[0, idx]) if X.shape[1] > idx else 0.0
                    })
            
        except Exception as e:
            self.logger.warning(f"Could not calculate feature importance: {e}")
        
        return sorted(feature_importance, key=lambda x: x['importance'], reverse=True)[:5]
    
    def _count_parameters(self) -> int:
        """Count total number of parameters in the network"""
        if not hasattr(self.model, 'coefs_'):
            return 0
        
        n_params = 0
        # Count weights
        for coef in self.model.coefs_:
            n_params += coef.size
        # Count biases
        for intercept in self.model.intercepts_:
            n_params += intercept.size
        
        return n_params
    
    def _get_model_parameters(self) -> Dict[str, Any]:
        """Get model-specific parameters"""
        if self.model is None:
            return {}
        
        params = {
            'algorithm': type(self.model).__name__,
            'hidden_layer_sizes': self.model.hidden_layer_sizes,
            'activation': self.model.activation,
            'solver': self.model.solver,
            'learning_rate_init': self.model.learning_rate_init,
            'max_iter': self.model.max_iter
        }
        
        if hasattr(self.model, 'n_iter_'):
            params['iterations_used'] = int(self.model.n_iter_)
            params['converged'] = bool(self.model.n_iter_ < self.model.max_iter)
        
        if hasattr(self.model, 'loss_'):
            params['final_loss'] = float(self.model.loss_)
        
        if hasattr(self.model, 'classes_'):
            params['classes'] = self.model.classes_.tolist()
        
        params['total_parameters'] = self._count_parameters()
        
        return params