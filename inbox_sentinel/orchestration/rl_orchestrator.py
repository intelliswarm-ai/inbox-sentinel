"""
Reinforcement Learning Enhanced Orchestrator
Uses Q-Learning to optimize ensemble decision making
"""

import asyncio
import json
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import random
import time
from dataclasses import dataclass

from inbox_sentinel.core.types import Email, PredictionResult
from inbox_sentinel.ml.models.naive_bayes import NaiveBayesDetector
from inbox_sentinel.ml.models.svm import SVMDetector
from inbox_sentinel.ml.models.random_forest import RandomForestDetector
from inbox_sentinel.ml.models.logistic_regression import LogisticRegressionDetector
from inbox_sentinel.ml.models.neural_network import NeuralNetworkDetector


@dataclass
class RLState:
    """State representation for RL agent"""
    # Model predictions (5 models x 3 features each)
    naive_bayes_spam_prob: float
    naive_bayes_confidence: float
    naive_bayes_is_spam: bool
    
    svm_spam_prob: float
    svm_confidence: float
    svm_is_spam: bool
    
    random_forest_spam_prob: float
    random_forest_confidence: float
    random_forest_is_spam: bool
    
    logistic_spam_prob: float
    logistic_confidence: float
    logistic_is_spam: bool
    
    neural_spam_prob: float
    neural_confidence: float
    neural_is_spam: bool
    
    # Consensus metrics
    spam_votes: int  # Number of models predicting spam
    agreement_level: float  # How much models agree (0-1)
    avg_confidence: float  # Average confidence across all models
    confidence_variance: float  # Variance in confidence scores
    
    # Email context features
    subject_length: int
    content_length: int
    has_urls: bool
    has_money_terms: bool
    has_urgent_terms: bool
    
    def to_vector(self) -> np.ndarray:
        """Convert state to feature vector for RL agent"""
        return np.array([
            # Model predictions (normalized to 0-1)
            self.naive_bayes_spam_prob,
            self.naive_bayes_confidence,
            float(self.naive_bayes_is_spam),
            
            self.svm_spam_prob,
            self.svm_confidence,
            float(self.svm_is_spam),
            
            self.random_forest_spam_prob,
            self.random_forest_confidence,
            float(self.random_forest_is_spam),
            
            self.logistic_spam_prob,
            self.logistic_confidence,
            float(self.logistic_is_spam),
            
            self.neural_spam_prob,
            self.neural_confidence,
            float(self.neural_is_spam),
            
            # Consensus features
            self.spam_votes / 5.0,  # Normalized to 0-1
            self.agreement_level,
            self.avg_confidence,
            self.confidence_variance,
            
            # Email context (normalized)
            min(self.subject_length / 100.0, 1.0),  # Cap at 100 chars
            min(self.content_length / 1000.0, 1.0),  # Cap at 1000 chars
            float(self.has_urls),
            float(self.has_money_terms),
            float(self.has_urgent_terms)
        ])


class QLearningAgent:
    """Q-Learning agent for ensemble decision making"""
    
    def __init__(self, state_dim: int = 24, action_dim: int = 2, 
                 learning_rate: float = 0.1, discount_factor: float = 0.95,
                 epsilon: float = 0.1, epsilon_decay: float = 0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim  # 0: classify as ham, 1: classify as spam
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = 0.01
        
        # Q-table using discretized states
        self.q_table = defaultdict(lambda: np.zeros(action_dim))
        self.training_stats = {
            'episodes': 0,
            'total_reward': 0,
            'accuracy_history': [],
            'epsilon_history': []
        }
    
    def discretize_state(self, state_vector: np.ndarray, bins: int = 10) -> str:
        """Discretize continuous state into discrete bins for Q-table"""
        # Discretize each feature into bins
        discretized = []
        for value in state_vector:
            # Clamp to [0, 1] range and discretize
            clamped = np.clip(value, 0, 1)
            bin_idx = min(int(clamped * bins), bins - 1)
            discretized.append(str(bin_idx))
        
        return '_'.join(discretized)
    
    def select_action(self, state_vector: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        state_key = self.discretize_state(state_vector)
        
        if training and random.random() < self.epsilon:
            # Exploration: random action
            return random.randint(0, 1)
        else:
            # Exploitation: best known action
            q_values = self.q_table[state_key]
            return int(np.argmax(q_values))
    
    def update_q_value(self, state_vector: np.ndarray, action: int, 
                      reward: float, next_state_vector: Optional[np.ndarray] = None):
        """Update Q-value using Q-learning update rule"""
        state_key = self.discretize_state(state_vector)
        
        if next_state_vector is not None:
            next_state_key = self.discretize_state(next_state_vector)
            next_q_max = np.max(self.q_table[next_state_key])
        else:
            next_q_max = 0  # Terminal state
        
        # Q-learning update
        current_q = self.q_table[state_key][action]
        target_q = reward + self.discount_factor * next_q_max
        self.q_table[state_key][action] += self.learning_rate * (target_q - current_q)
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def save_model(self, filepath: str):
        """Save Q-table and training stats"""
        model_data = {
            'q_table': dict(self.q_table),
            'training_stats': self.training_stats,
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon': self.epsilon,
                'epsilon_decay': self.epsilon_decay
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load Q-table and training stats"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
                
            self.q_table = defaultdict(lambda: np.zeros(self.action_dim))
            self.q_table.update(model_data['q_table'])
            self.training_stats = model_data['training_stats']
            
            # Load hyperparameters
            hp = model_data['hyperparameters']
            self.learning_rate = hp['learning_rate']
            self.discount_factor = hp['discount_factor']
            self.epsilon = hp['epsilon']
            self.epsilon_decay = hp['epsilon_decay']
            
            return True
        except (FileNotFoundError, KeyError, pickle.PickleError):
            return False


class RLEnhancedOrchestrator:
    """Reinforcement Learning Enhanced Orchestrator"""
    
    def __init__(self, model_path: Optional[str] = None):
        # Initialize individual models
        self.models = {
            'naive_bayes': NaiveBayesDetector(),
            'svm': SVMDetector(),
            'random_forest': RandomForestDetector(),
            'logistic_regression': LogisticRegressionDetector(),
            'neural_network': NeuralNetworkDetector(),
        }
        
        # RL Agent
        self.rl_agent = QLearningAgent()
        
        # Training data storage
        self.training_episodes = []
        self.is_initialized = False
        
        # Model performance tracking
        self.model_stats = {
            'total_decisions': 0,
            'correct_decisions': 0,
            'training_accuracy_history': [],
            'validation_accuracy_history': []
        }
        
        # Load pre-trained model if provided
        if model_path and Path(model_path).exists():
            self.rl_agent.load_model(model_path)
    
    async def initialize(self):
        """Initialize all ML models"""
        if self.is_initialized:
            return
        
        print("Initializing RL Enhanced Orchestrator...")
        for name, model in self.models.items():
            try:
                await model.initialize(use_pretrained=True)
                if model.is_trained:
                    print(f"✅ {name} initialized successfully")
                else:
                    print(f"❌ {name} not trained")
                    raise Exception(f"{name} model not available")
            except Exception as e:
                print(f"❌ {name} initialization failed: {e}")
                raise
        
        self.is_initialized = True
        print("✅ RL Enhanced Orchestrator ready!")
    
    def extract_email_features(self, email: Email) -> Dict[str, Any]:
        """Extract contextual features from email"""
        content_lower = email.content.lower()
        subject_lower = email.subject.lower()
        full_text = f"{subject_lower} {content_lower}"
        
        # Money-related terms
        money_terms = ['$', 'money', 'cash', 'prize', 'winner', 'lottery', 
                      'million', 'thousand', 'payment', 'refund', 'credit']
        
        # Urgency terms
        urgent_terms = ['urgent', 'immediate', 'act now', 'expires', 'limited time',
                       'hurry', 'quickly', 'don\'t miss', 'last chance']
        
        return {
            'subject_length': len(email.subject),
            'content_length': len(email.content),
            'has_urls': 'http' in content_lower or 'www.' in content_lower,
            'has_money_terms': any(term in full_text for term in money_terms),
            'has_urgent_terms': any(term in full_text for term in urgent_terms)
        }
    
    async def get_model_predictions(self, email: Email) -> Dict[str, PredictionResult]:
        """Get predictions from all individual models"""
        predictions = {}
        
        for name, model in self.models.items():
            if model.is_trained:
                try:
                    result = await model.analyze(email)
                    predictions[name] = result
                except Exception as e:
                    print(f"Error getting prediction from {name}: {e}")
        
        return predictions
    
    def create_rl_state(self, email: Email, predictions: Dict[str, PredictionResult]) -> RLState:
        """Create RL state from email and model predictions"""
        email_features = self.extract_email_features(email)
        
        # Default values for missing predictions
        default_pred = PredictionResult(
            model_name="missing",
            algorithm="N/A",
            is_spam=False,
            spam_probability=0.5,
            ham_probability=0.5,
            confidence=0.0,
            features=[]
        )
        
        # Extract model predictions
        nb_pred = predictions.get('naive_bayes', default_pred)
        svm_pred = predictions.get('svm', default_pred)
        rf_pred = predictions.get('random_forest', default_pred)
        lr_pred = predictions.get('logistic_regression', default_pred)
        nn_pred = predictions.get('neural_network', default_pred)
        
        # Calculate consensus metrics
        spam_votes = sum([p.is_spam for p in predictions.values()])
        confidences = [p.confidence for p in predictions.values()]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        confidence_variance = np.var(confidences) if len(confidences) > 1 else 0.0
        
        # Agreement level (how unanimous the decision is)
        agreement_level = abs(spam_votes - (len(predictions) - spam_votes)) / len(predictions) if predictions else 0.0
        
        return RLState(
            # Naive Bayes
            naive_bayes_spam_prob=nb_pred.spam_probability,
            naive_bayes_confidence=nb_pred.confidence,
            naive_bayes_is_spam=nb_pred.is_spam,
            
            # SVM
            svm_spam_prob=svm_pred.spam_probability,
            svm_confidence=svm_pred.confidence,
            svm_is_spam=svm_pred.is_spam,
            
            # Random Forest
            random_forest_spam_prob=rf_pred.spam_probability,
            random_forest_confidence=rf_pred.confidence,
            random_forest_is_spam=rf_pred.is_spam,
            
            # Logistic Regression
            logistic_spam_prob=lr_pred.spam_probability,
            logistic_confidence=lr_pred.confidence,
            logistic_is_spam=lr_pred.is_spam,
            
            # Neural Network
            neural_spam_prob=nn_pred.spam_probability,
            neural_confidence=nn_pred.confidence,
            neural_is_spam=nn_pred.is_spam,
            
            # Consensus
            spam_votes=spam_votes,
            agreement_level=agreement_level,
            avg_confidence=avg_confidence,
            confidence_variance=confidence_variance,
            
            # Email features
            **email_features
        )
    
    async def predict(self, email: Email, training: bool = False) -> Dict:
        """Make prediction using RL-enhanced orchestration"""
        if not self.is_initialized:
            await self.initialize()
        
        # Get predictions from all models
        predictions = await self.get_model_predictions(email)
        
        if not predictions:
            return {'success': False, 'error': 'No model predictions available'}
        
        # Create RL state
        rl_state = self.create_rl_state(email, predictions)
        state_vector = rl_state.to_vector()
        
        # Get RL agent's decision
        action = self.rl_agent.select_action(state_vector, training=training)
        is_spam = bool(action)  # 0 = ham, 1 = spam
        
        # Calculate ensemble confidence based on agreement
        ensemble_confidence = rl_state.avg_confidence * rl_state.agreement_level
        
        return {
            'success': True,
            'is_spam': is_spam,
            'confidence': ensemble_confidence,
            'spam_probability': np.mean([p.spam_probability for p in predictions.values()]),
            'rl_state': rl_state,
            'detailed_results': {name: {
                'is_spam': pred.is_spam,
                'spam_probability': pred.spam_probability,
                'confidence': pred.confidence,
                'algorithm': pred.algorithm
            } for name, pred in predictions.items()}
        }
    
    def calculate_reward(self, predicted_spam: bool, actual_spam: bool, confidence: float) -> float:
        """Calculate reward for RL training"""
        # Base reward for correct/incorrect prediction
        if predicted_spam == actual_spam:
            base_reward = 1.0
        else:
            # Harsh penalty for wrong predictions
            base_reward = -2.0
        
        # Confidence bonus/penalty
        if predicted_spam == actual_spam:
            # Reward higher confidence for correct predictions
            confidence_bonus = confidence * 0.5
        else:
            # Penalty for being confidently wrong
            confidence_bonus = -confidence * 0.5
        
        return base_reward + confidence_bonus
    
    async def train_episode(self, email: Email, true_label: bool) -> float:
        """Train RL agent on single episode"""
        # Get prediction in training mode
        result = await self.predict(email, training=True)
        
        if not result['success']:
            return 0.0
        
        predicted_spam = result['is_spam']
        confidence = result['confidence']
        rl_state = result['rl_state']
        
        # Calculate reward
        reward = self.calculate_reward(predicted_spam, true_label, confidence)
        
        # Update Q-values
        state_vector = rl_state.to_vector()
        action = int(predicted_spam)
        self.rl_agent.update_q_value(state_vector, action, reward)
        
        # Update stats
        self.model_stats['total_decisions'] += 1
        if predicted_spam == true_label:
            self.model_stats['correct_decisions'] += 1
        
        return reward
    
    async def train_on_dataset(self, training_data: List[Tuple[Email, bool]], 
                             episodes: int = 100, validation_split: float = 0.2):
        """Train RL agent on dataset"""
        print(f"Starting RL training for {episodes} episodes...")
        
        # Split data
        split_idx = int(len(training_data) * (1 - validation_split))
        train_data = training_data[:split_idx]
        val_data = training_data[split_idx:]
        
        print(f"Training on {len(train_data)} samples, validating on {len(val_data)} samples")
        
        for episode in range(episodes):
            # Shuffle training data
            random.shuffle(train_data)
            
            episode_reward = 0.0
            correct_predictions = 0
            
            # Training phase
            for email, true_label in train_data:
                reward = await self.train_episode(email, true_label)
                episode_reward += reward
                
                # Check accuracy
                result = await self.predict(email, training=False)
                if result['success'] and result['is_spam'] == true_label:
                    correct_predictions += 1
            
            # Calculate training accuracy
            training_accuracy = correct_predictions / len(train_data)
            self.model_stats['training_accuracy_history'].append(training_accuracy)
            
            # Validation phase
            val_correct = 0
            for email, true_label in val_data:
                result = await self.predict(email, training=False)
                if result['success'] and result['is_spam'] == true_label:
                    val_correct += 1
            
            val_accuracy = val_correct / len(val_data)
            self.model_stats['validation_accuracy_history'].append(val_accuracy)
            
            # Decay exploration
            self.rl_agent.decay_epsilon()
            
            # Update RL agent stats
            self.rl_agent.training_stats['episodes'] += 1
            self.rl_agent.training_stats['total_reward'] += episode_reward
            self.rl_agent.training_stats['accuracy_history'].append(training_accuracy)
            self.rl_agent.training_stats['epsilon_history'].append(self.rl_agent.epsilon)
            
            # Progress logging
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{episodes}: "
                      f"Train Acc: {training_accuracy:.3f}, "
                      f"Val Acc: {val_accuracy:.3f}, "
                      f"Reward: {episode_reward:.2f}, "
                      f"Epsilon: {self.rl_agent.epsilon:.3f}")
        
        print("✅ RL training complete!")
        print(f"Final training accuracy: {self.model_stats['training_accuracy_history'][-1]:.3f}")
        print(f"Final validation accuracy: {self.model_stats['validation_accuracy_history'][-1]:.3f}")
    
    def save_model(self, filepath: str = "rl_orchestrator_model.pkl"):
        """Save trained RL model"""
        model_dir = Path(filepath).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        self.rl_agent.save_model(filepath)
        print(f"✅ RL model saved to {filepath}")
    
    def get_training_stats(self) -> Dict:
        """Get comprehensive training statistics"""
        return {
            'rl_agent_stats': self.rl_agent.training_stats,
            'orchestrator_stats': self.model_stats,
            'current_epsilon': self.rl_agent.epsilon,
            'q_table_size': len(self.rl_agent.q_table)
        }

# Legacy interface compatibility
async def analyze_email(email: Email, rl_orchestrator: RLEnhancedOrchestrator) -> Dict:
    """Analyze email using RL orchestrator (compatible with existing interface)"""
    result = await rl_orchestrator.predict(email)
    
    if result['success']:
        return {
            'success': True,
            'analysis': f"""
## RL-Enhanced Ensemble Analysis

**Final Decision:** {'SPAM/PHISHING' if result['is_spam'] else 'LEGITIMATE'}
**Confidence:** {result['confidence']:.3f}
**Spam Probability:** {result['spam_probability']:.3f}

### Individual Model Results:
""" + '\n'.join([f"- **{name.replace('_', ' ').title()}**: {'SPAM' if details['is_spam'] else 'HAM'} (confidence: {details['confidence']:.3f})"
                 for name, details in result['detailed_results'].items()]) + f"""

### RL Decision Process:
- State features analyzed: {len(result['rl_state'].to_vector())}
- Model consensus: {result['rl_state'].spam_votes}/5 voted spam
- Agreement level: {result['rl_state'].agreement_level:.3f}
- Average model confidence: {result['rl_state'].avg_confidence:.3f}

*Powered by Reinforcement Learning Enhanced Orchestration*
""",
            'detailed_results': result['detailed_results']
        }
    else:
        return result