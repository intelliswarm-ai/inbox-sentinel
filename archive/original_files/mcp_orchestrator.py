#!/usr/bin/env python3
"""
Orchestrator for coordinating multiple MCP servers for phishing detection
"""

import asyncio
import subprocess
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from fastmcp import FastMCP, Context
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the orchestrator MCP server
mcp = FastMCP("Phishing Detection Orchestrator")


class ConsensusStrategy(Enum):
    """Strategies for combining predictions from multiple models"""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    HIGHEST_CONFIDENCE = "highest_confidence"
    CONSERVATIVE = "conservative"  # If any says spam, it's spam
    AGGRESSIVE = "aggressive"  # Only spam if all agree


@dataclass
class ModelServer:
    """Information about a model server"""
    name: str
    port: int
    process: Optional[subprocess.Popen] = None
    accuracy: float = 0.85  # Default accuracy


# Model servers configuration
MODEL_SERVERS = {
    "naive_bayes": ModelServer("Naive Bayes", 8001, accuracy=0.88),
    "svm": ModelServer("SVM", 8002, accuracy=0.90),
    "random_forest": ModelServer("Random Forest", 8003, accuracy=0.92),
    "logistic_regression": ModelServer("Logistic Regression", 8004, accuracy=0.89),
    "neural_network": ModelServer("Neural Network", 8005, accuracy=0.93)
}


@mcp.tool()
async def analyze_email_ensemble(
    ctx: Context,
    email_content: str,
    subject: str,
    sender: str,
    strategy: str = "weighted_average",
    models_to_use: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Analyze email using ensemble of all available models.
    
    Parameters:
    - email_content: The email body
    - subject: Email subject line
    - sender: Sender email address
    - strategy: Consensus strategy (majority_vote, weighted_average, highest_confidence, conservative, aggressive)
    - models_to_use: List of model names to use (default: all)
    """
    
    if models_to_use is None:
        models_to_use = list(MODEL_SERVERS.keys())
    
    # Collect predictions from all models
    predictions = []
    errors = []
    
    for model_name in models_to_use:
        if model_name not in MODEL_SERVERS:
            continue
            
        try:
            # Call the model's analyze function
            result = await call_model_server(
                model_name,
                "analyze",
                {
                    "email_content": email_content,
                    "subject": subject,
                    "sender": sender
                }
            )
            
            if result and 'error' not in result:
                predictions.append({
                    'model': model_name,
                    'result': result,
                    'accuracy': MODEL_SERVERS[model_name].accuracy
                })
            else:
                errors.append({
                    'model': model_name,
                    'error': result.get('error', 'Unknown error') if result else 'No response'
                })
                
        except Exception as e:
            errors.append({
                'model': model_name,
                'error': str(e)
            })
    
    if not predictions:
        return {
            'success': False,
            'error': 'No models provided valid predictions',
            'errors': errors
        }
    
    # Apply consensus strategy
    consensus_result = apply_consensus_strategy(predictions, ConsensusStrategy(strategy))
    
    # Add detailed results
    consensus_result['individual_predictions'] = [
        {
            'model': p['model'],
            'is_spam': p['result'].get('is_spam', False),
            'confidence': p['result'].get('confidence', 0),
            'spam_probability': p['result'].get('spam_probability', 0)
        }
        for p in predictions
    ]
    
    consensus_result['errors'] = errors
    consensus_result['models_used'] = len(predictions)
    consensus_result['strategy_used'] = strategy
    
    return consensus_result


def apply_consensus_strategy(predictions: List[Dict], strategy: ConsensusStrategy) -> Dict[str, Any]:
    """Apply the specified consensus strategy to combine predictions"""
    
    if strategy == ConsensusStrategy.MAJORITY_VOTE:
        spam_votes = sum(1 for p in predictions if p['result'].get('is_spam', False))
        total_votes = len(predictions)
        is_spam = spam_votes > total_votes / 2
        confidence = spam_votes / total_votes if is_spam else (total_votes - spam_votes) / total_votes
        
        return {
            'is_spam': is_spam,
            'confidence': confidence,
            'spam_votes': spam_votes,
            'total_votes': total_votes,
            'consensus_type': 'majority_vote'
        }
    
    elif strategy == ConsensusStrategy.WEIGHTED_AVERAGE:
        # Weight by model accuracy
        total_weight = sum(p['accuracy'] for p in predictions)
        weighted_spam_prob = sum(
            p['result'].get('spam_probability', 0.5) * p['accuracy'] 
            for p in predictions
        ) / total_weight
        
        is_spam = weighted_spam_prob > 0.5
        
        return {
            'is_spam': is_spam,
            'confidence': weighted_spam_prob if is_spam else 1 - weighted_spam_prob,
            'spam_probability': weighted_spam_prob,
            'consensus_type': 'weighted_average'
        }
    
    elif strategy == ConsensusStrategy.HIGHEST_CONFIDENCE:
        # Use prediction with highest confidence
        best_prediction = max(predictions, key=lambda p: p['result'].get('confidence', 0))
        
        return {
            'is_spam': best_prediction['result'].get('is_spam', False),
            'confidence': best_prediction['result'].get('confidence', 0),
            'best_model': best_prediction['model'],
            'consensus_type': 'highest_confidence'
        }
    
    elif strategy == ConsensusStrategy.CONSERVATIVE:
        # If any model says spam with decent confidence, it's spam
        spam_predictions = [p for p in predictions 
                          if p['result'].get('is_spam', False) 
                          and p['result'].get('confidence', 0) > 0.6]
        
        if spam_predictions:
            # Use the most confident spam prediction
            best_spam = max(spam_predictions, key=lambda p: p['result'].get('confidence', 0))
            return {
                'is_spam': True,
                'confidence': best_spam['result'].get('confidence', 0),
                'triggered_by': best_spam['model'],
                'consensus_type': 'conservative'
            }
        else:
            # Average confidence of non-spam predictions
            confidences = [p['result'].get('confidence', 0) 
                         for p in predictions 
                         if not p['result'].get('is_spam', False)]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
            
            return {
                'is_spam': False,
                'confidence': avg_confidence,
                'consensus_type': 'conservative'
            }
    
    elif strategy == ConsensusStrategy.AGGRESSIVE:
        # Only spam if all models agree
        all_spam = all(p['result'].get('is_spam', False) for p in predictions)
        
        if all_spam:
            avg_confidence = sum(p['result'].get('confidence', 0) for p in predictions) / len(predictions)
            return {
                'is_spam': True,
                'confidence': avg_confidence,
                'consensus_type': 'aggressive',
                'unanimous': True
            }
        else:
            # If not all agree, it's not spam
            ham_predictions = [p for p in predictions if not p['result'].get('is_spam', False)]
            avg_confidence = sum(p['result'].get('confidence', 0) for p in ham_predictions) / len(ham_predictions) if ham_predictions else 0.7
            
            return {
                'is_spam': False,
                'confidence': avg_confidence,
                'consensus_type': 'aggressive',
                'unanimous': False
            }


async def call_model_server(model_name: str, method: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Call a specific model server's method"""
    
    # For this demo, we'll simulate the calls
    # In production, you would use actual MCP client connections
    
    # Simulate calling the appropriate analyze function
    if method == "analyze":
        if model_name == "naive_bayes":
            # Simulate Naive Bayes response
            return {
                'algorithm': 'Naive Bayes',
                'is_spam': "urgent" in params['email_content'].lower() or "winner" in params['subject'].lower(),
                'spam_probability': 0.75 if "urgent" in params['email_content'].lower() else 0.25,
                'confidence': 0.85
            }
        elif model_name == "svm":
            # Simulate SVM response
            return {
                'algorithm': 'Support Vector Machine',
                'is_spam': "click here" in params['email_content'].lower() or ".tk" in params['sender'],
                'spam_probability': 0.8 if "click here" in params['email_content'].lower() else 0.2,
                'confidence': 0.90
            }
        elif model_name == "random_forest":
            # Simulate Random Forest response
            return {
                'algorithm': 'Random Forest',
                'is_spam': "congratulations" in params['email_content'].lower() or "$" in params['subject'],
                'spam_probability': 0.85 if "congratulations" in params['email_content'].lower() else 0.15,
                'confidence': 0.92
            }
        elif model_name == "logistic_regression":
            # Simulate Logistic Regression response
            return {
                'algorithm': 'Logistic Regression',
                'is_spam': len(params['email_content']) < 100 and "!" in params['subject'],
                'spam_probability': 0.7 if "!" in params['subject'] else 0.3,
                'confidence': 0.88
            }
        elif model_name == "neural_network":
            # Simulate Neural Network response
            return {
                'algorithm': 'Neural Network',
                'is_spam': params['email_content'].count('!') > 2 or "free" in params['subject'].lower(),
                'spam_probability': 0.9 if "free" in params['subject'].lower() else 0.1,
                'confidence': 0.93
            }
    
    return None


@mcp.tool()
async def compare_models(
    ctx: Context,
    email_content: str,
    subject: str,
    sender: str
) -> Dict[str, Any]:
    """
    Compare predictions from all models side by side.
    Useful for understanding model disagreements.
    """
    
    results = []
    
    for model_name in MODEL_SERVERS.keys():
        try:
            result = await call_model_server(
                model_name,
                "analyze",
                {
                    "email_content": email_content,
                    "subject": subject,
                    "sender": sender
                }
            )
            
            if result:
                results.append({
                    'model': model_name,
                    'algorithm': result.get('algorithm', model_name),
                    'is_spam': result.get('is_spam', False),
                    'spam_probability': result.get('spam_probability', 0),
                    'confidence': result.get('confidence', 0),
                    'accuracy': MODEL_SERVERS[model_name].accuracy
                })
        except Exception as e:
            results.append({
                'model': model_name,
                'error': str(e)
            })
    
    # Calculate agreement metrics
    spam_predictions = [r for r in results if r.get('is_spam', False) and 'error' not in r]
    ham_predictions = [r for r in results if not r.get('is_spam', False) and 'error' not in r]
    
    agreement_score = max(len(spam_predictions), len(ham_predictions)) / len(results)
    
    return {
        'comparison': results,
        'summary': {
            'total_models': len(results),
            'spam_votes': len(spam_predictions),
            'ham_votes': len(ham_predictions),
            'agreement_score': agreement_score,
            'unanimous': agreement_score == 1.0
        },
        'recommendation': {
            'is_spam': len(spam_predictions) > len(ham_predictions),
            'confidence': agreement_score
        }
    }


@mcp.tool()
async def get_orchestrator_info(ctx: Context) -> Dict[str, Any]:
    """Get information about the orchestrator and available models"""
    
    return {
        'orchestrator': 'Phishing Detection Ensemble',
        'available_models': [
            {
                'name': name,
                'algorithm': server.name,
                'accuracy': server.accuracy,
                'port': server.port
            }
            for name, server in MODEL_SERVERS.items()
        ],
        'consensus_strategies': [s.value for s in ConsensusStrategy],
        'status': 'ready'
    }


if __name__ == "__main__":
    mcp.run()