#!/usr/bin/env python3
"""
FastMCP server for Neural Network (MLP) email classification
"""

from fastmcp import FastMCP, Context
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import os
from typing import Dict, Any, List, Tuple
from email_preprocessor import EmailPreprocessor

# Initialize the MCP server
mcp = FastMCP("Neural Network Classifier")

# Global model and preprocessor
nn_model = None
preprocessor = None
scaler = None


@mcp.tool()
async def analyze_email_neural_network(
    ctx: Context,
    email_content: str,
    subject: str,
    sender: str
) -> Dict[str, Any]:
    """
    Analyze email for spam/phishing using Neural Network (Multi-Layer Perceptron).
    Returns classification result with probability scores and activation analysis.
    """
    global nn_model, preprocessor, scaler
    
    # Initialize if needed
    if nn_model is None:
        await initialize_neural_network(ctx, use_pretrained=True)
    
    try:
        # Preprocess email
        features = preprocessor.transform(email_content, subject, sender)
        
        # Use combined features for Neural Network
        X = features['combined'].reshape(1, -1)
        
        # Scale features (critical for neural networks)
        if scaler is not None:
            X = scaler.transform(X)
        
        # Get prediction and probabilities
        prediction = nn_model.predict(X)[0]
        probabilities = nn_model.predict_proba(X)[0]
        
        # Get activation values from hidden layers
        activations = []
        input_data = X
        
        # Forward pass through layers to get activations
        for i, (weights, biases) in enumerate(zip(nn_model.coefs_, nn_model.intercepts_)):
            linear_output = np.dot(input_data, weights) + biases
            
            if i < len(nn_model.coefs_) - 1:  # Hidden layers
                if nn_model.activation == 'relu':
                    activation = np.maximum(0, linear_output)
                elif nn_model.activation == 'tanh':
                    activation = np.tanh(linear_output)
                elif nn_model.activation == 'logistic':
                    activation = 1 / (1 + np.exp(-linear_output))
                else:
                    activation = linear_output
                
                # Calculate layer statistics
                activations.append({
                    'layer': i + 1,
                    'neurons': len(activation[0]),
                    'active_neurons': int(np.sum(activation[0] > 0.1)),
                    'mean_activation': float(np.mean(activation[0])),
                    'max_activation': float(np.max(activation[0]))
                })
                
                input_data = activation
            else:  # Output layer
                output_activation = 1 / (1 + np.exp(-linear_output))  # Sigmoid for output
        
        # Calculate confidence based on probability spread
        confidence = abs(probabilities[1] - probabilities[0])
        
        # Analyze feature importance (approximation using first layer weights)
        feature_importance = []
        if len(nn_model.coefs_) > 0:
            first_layer_weights = nn_model.coefs_[0]
            # Sum absolute weights for each input feature across all neurons
            feature_impacts = np.sum(np.abs(first_layer_weights), axis=1)
            
            feature_names = preprocessor.get_feature_names()
            all_feature_names = feature_names.get('tfidf', []) + feature_names.get('manual', [])
            
            top_indices = np.argsort(feature_impacts)[-15:]
            for idx in top_indices:
                if idx < len(all_feature_names):
                    feature_name = all_feature_names[idx]
                else:
                    feature_name = f"feature_{idx}"
                
                feature_importance.append({
                    'feature': feature_name,
                    'impact': float(feature_impacts[idx]),
                    'input_value': float(X[0, idx])
                })
        
        result = {
            'algorithm': 'Neural Network (MLP)',
            'is_spam': bool(prediction == 1),
            'spam_probability': float(probabilities[1]),
            'ham_probability': float(probabilities[0]),
            'confidence': float(confidence),
            'network_architecture': {
                'input_features': X.shape[1],
                'hidden_layers': len(nn_model.hidden_layer_sizes),
                'hidden_layer_sizes': nn_model.hidden_layer_sizes,
                'activation_function': nn_model.activation,
                'output_neurons': 2
            },
            'layer_activations': activations,
            'top_features': sorted(feature_importance, key=lambda x: x['impact'], reverse=True)[:10],
            'training_info': {
                'iterations': nn_model.n_iter_,
                'loss': float(nn_model.loss_) if hasattr(nn_model, 'loss_') else None
            },
            'preprocessing': {
                'content_length': len(email_content),
                'subject_length': len(subject),
                'total_features': X.shape[1]
            }
        }
        
        # Add interpretation
        if result['is_spam']:
            result['explanation'] = (
                f"Email classified as SPAM with {result['spam_probability']:.1%} probability. "
                f"Neural network activated {sum(a['active_neurons'] for a in activations)} neurons across hidden layers."
            )
        else:
            result['explanation'] = (
                f"Email classified as HAM (legitimate) with {result['ham_probability']:.1%} probability. "
                f"Neural network showed low activation patterns typical of legitimate emails."
            )
        
        return result
        
    except Exception as e:
        return {
            'algorithm': 'Neural Network (MLP)',
            'error': str(e),
            'is_spam': False,
            'confidence': 0.0,
            'explanation': f"Error during analysis: {str(e)}"
        }


@mcp.tool()
async def train_neural_network(
    ctx: Context,
    training_samples: List[Dict[str, Any]],
    hidden_layer_sizes: Tuple[int, ...] = (100, 50),
    activation: str = "relu",
    solver: str = "adam",
    learning_rate: float = 0.001,
    max_iter: int = 1000
) -> Dict[str, Any]:
    """
    Train the Neural Network model with provided samples.
    Each sample should have: email_content, subject, sender, is_spam (bool)
    
    Parameters:
    - hidden_layer_sizes: Tuple of hidden layer sizes, e.g., (100, 50) for two layers
    - activation: 'relu', 'tanh', 'logistic', or 'identity'
    - solver: 'adam', 'sgd', or 'lbfgs'
    - learning_rate: Learning rate for weight updates
    - max_iter: Maximum iterations for training
    """
    global nn_model, preprocessor, scaler
    
    try:
        # Initialize preprocessor and scaler
        preprocessor = EmailPreprocessor()
        scaler = StandardScaler()
        
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
        preprocessor.fit(emails, labels)
        
        # Transform all emails
        X_list = []
        for content, subject, sender in emails:
            features = preprocessor.transform(content, subject, sender)
            X_list.append(features['combined'])
        
        X = np.array(X_list)
        y = np.array(labels)
        
        # Scale features (crucial for neural networks)
        X = scaler.fit_transform(X)
        
        # Initialize and train Neural Network
        nn_model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            learning_rate_init=learning_rate,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42
        )
        
        nn_model.fit(X, y)
        
        # Calculate training accuracy
        train_predictions = nn_model.predict(X)
        accuracy = np.mean(train_predictions == y)
        
        # Calculate total parameters
        total_params = 0
        for i, (w, b) in enumerate(zip(nn_model.coefs_, nn_model.intercepts_)):
            total_params += w.size + b.size
        
        result = {
            'success': True,
            'hidden_layer_sizes': hidden_layer_sizes,
            'activation': activation,
            'solver': solver,
            'learning_rate': learning_rate,
            'samples_trained': len(training_samples),
            'training_accuracy': float(accuracy),
            'spam_samples': int(np.sum(y)),
            'ham_samples': len(y) - int(np.sum(y)),
            'feature_count': X.shape[1],
            'network_info': {
                'total_parameters': total_params,
                'n_layers': len(nn_model.coefs_),
                'iterations': nn_model.n_iter_,
                'final_loss': float(nn_model.loss_) if hasattr(nn_model, 'loss_') else None,
                'converged': nn_model.n_iter_ < max_iter
            }
        }
        
        # Add validation score if available
        if hasattr(nn_model, 'best_validation_score_'):
            result['validation_score'] = float(nn_model.best_validation_score_)
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


@mcp.tool()
async def initialize_neural_network(
    ctx: Context,
    use_pretrained: bool = True
) -> Dict[str, str]:
    """
    Initialize the Neural Network model.
    If use_pretrained is True, loads a pre-trained model if available.
    Otherwise, initializes with default training samples.
    """
    global nn_model, preprocessor, scaler
    
    if use_pretrained and os.path.exists('neural_network_model.pkl'):
        # Load pre-trained model
        with open('neural_network_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
            nn_model = model_data['model']
            preprocessor = model_data['preprocessor']
            scaler = model_data.get('scaler')
        return {'status': 'Loaded pre-trained Neural Network model'}
    
    # Initialize with comprehensive training samples
    default_samples = [
        # Sophisticated spam examples
        {
            'email_content': 'Dear Customer, We noticed unusual activity on your account. For your security, please verify your identity within 24 hours to avoid suspension.',
            'subject': 'Important: Account Security Notice',
            'sender': 'security@account-verify.com',
            'is_spam': True
        },
        {
            'email_content': 'You have been specially selected to receive a $500 gift card. Complete this short survey to claim your reward. Limited time only!',
            'subject': 'Your $500 Gift Card is Waiting',
            'sender': 'rewards@survey-gifts.net',
            'is_spam': True
        },
        {
            'email_content': 'Investment opportunity: Triple your money in 30 days! Our AI trading system guarantees profits. Join thousands of successful investors.',
            'subject': 'Exclusive Investment Opportunity',
            'sender': 'invest@crypto-profits.io',
            'is_spam': True
        },
        {
            'email_content': 'Your package delivery failed. Update your shipping address immediately or your order will be returned. Track: XYZ123456789',
            'subject': 'Delivery Failed - Action Required',
            'sender': 'shipping@delivery-update.org',
            'is_spam': True
        },
        {
            'email_content': 'Congratulations! Your email was selected in our monthly draw. You have won £850,000. Contact our claims department immediately.',
            'subject': 'Winner Notification - £850,000',
            'sender': 'claims@uk-lottery.co',
            'is_spam': True
        },
        {
            'email_content': 'IRS Tax Refund: You are eligible for a refund of $1,842. Click here to submit your refund request. This offer expires soon.',
            'subject': 'Tax Refund Available',
            'sender': 'refund@irs-gov.us',
            'is_spam': True
        },
        {
            'email_content': 'Beautiful Russian women are looking for serious relationships. View profiles and start chatting today. No fees!',
            'subject': 'Find Your Perfect Match',
            'sender': 'dating@russian-brides.love',
            'is_spam': True
        },
        # Legitimate examples
        {
            'email_content': 'Hi Team, Please review the attached quarterly report before our meeting on Friday. Let me know if you have any questions or need clarification on any points.',
            'subject': 'Q4 Report for Review',
            'sender': 'jennifer.chen@company.com',
            'is_spam': False
        },
        {
            'email_content': 'Your GitHub pull request #423 has been approved and merged. Thanks for your contribution to the project!',
            'subject': 'PR #423 Merged',
            'sender': 'notifications@github.com',
            'is_spam': False
        },
        {
            'email_content': 'This is a reminder that your subscription will renew on March 15th for $9.99/month. No action needed unless you wish to cancel.',
            'subject': 'Subscription Renewal Reminder',
            'sender': 'billing@netflix.com',
            'is_spam': False
        },
        {
            'email_content': 'Your order has been shipped! Expected delivery: March 10-12. Track your package: https://fedex.com/track/123456',
            'subject': 'Order Shipped - #ORD-789456',
            'sender': 'orders@amazon.com',
            'is_spam': False
        },
        {
            'email_content': 'Thanks for attending our webinar on cloud security. Here are the slides and recording as promised. Feel free to reach out with questions.',
            'subject': 'Webinar Follow-up: Cloud Security',
            'sender': 'events@techconference.com',
            'is_spam': False
        },
        {
            'email_content': 'Your appointment with Dr. Smith is confirmed for Tuesday, March 15 at 2:30 PM. Please arrive 15 minutes early for paperwork.',
            'subject': 'Appointment Confirmation',
            'sender': 'appointments@medicalcenter.com',
            'is_spam': False
        },
        {
            'email_content': 'New comment on your blog post: "Great article! I found the section on API security particularly helpful." View and reply to comments.',
            'subject': 'New Comment on Your Post',
            'sender': 'notifications@wordpress.com',
            'is_spam': False
        }
    ]
    
    # Train with a more complex architecture for better performance
    result = await train_neural_network(
        ctx, 
        default_samples, 
        hidden_layer_sizes=(128, 64, 32),  # 3 hidden layers
        activation="relu",
        solver="adam",
        learning_rate=0.001
    )
    
    if result['success']:
        return {'status': f'Initialized Neural Network model with {len(default_samples)} samples'}
    else:
        return {'status': f'Failed to initialize: {result.get("error", "Unknown error")}'}


@mcp.tool()
async def get_neural_network_info(ctx: Context) -> Dict[str, Any]:
    """Get information about the current Neural Network model"""
    global nn_model
    
    if nn_model is None:
        return {
            'status': 'Model not initialized',
            'initialized': False
        }
    
    info = {
        'status': 'Model ready',
        'initialized': True,
        'algorithm': 'Neural Network (Multi-Layer Perceptron)',
        'architecture': {
            'input_size': nn_model.n_features_in_ if hasattr(nn_model, 'n_features_in_') else 'Unknown',
            'hidden_layers': len(nn_model.hidden_layer_sizes),
            'hidden_layer_sizes': nn_model.hidden_layer_sizes,
            'output_size': nn_model.n_outputs_ if hasattr(nn_model, 'n_outputs_') else 2,
            'activation': nn_model.activation,
            'output_activation': 'softmax'
        },
        'training': {
            'solver': nn_model.solver,
            'learning_rate_init': nn_model.learning_rate_init,
            'max_iter': nn_model.max_iter,
            'early_stopping': nn_model.early_stopping
        }
    }
    
    if hasattr(nn_model, 'n_iter_'):
        info['training']['iterations_run'] = nn_model.n_iter_
        info['training']['converged'] = nn_model.n_iter_ < nn_model.max_iter
    
    if hasattr(nn_model, 'loss_'):
        info['training']['final_loss'] = float(nn_model.loss_)
    
    # Calculate total parameters
    if hasattr(nn_model, 'coefs_'):
        total_params = sum(w.size + b.size for w, b in zip(nn_model.coefs_, nn_model.intercepts_))
        info['total_parameters'] = total_params
    
    return info


if __name__ == "__main__":
    mcp.run()