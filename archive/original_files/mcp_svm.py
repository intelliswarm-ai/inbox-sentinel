#!/usr/bin/env python3
"""
FastMCP server for SVM (Support Vector Machine) email classification
"""

from fastmcp import FastMCP, Context
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import os
from typing import Dict, Any, List
from email_preprocessor import EmailPreprocessor

# Initialize the MCP server
mcp = FastMCP("SVM Classifier")

# Global model and preprocessor
svm_model = None
preprocessor = None
scaler = None


@mcp.tool()
async def analyze_email_svm(
    ctx: Context,
    email_content: str,
    subject: str,
    sender: str
) -> Dict[str, Any]:
    """
    Analyze email for spam/phishing using Support Vector Machine classifier.
    Returns classification result with confidence scores based on decision function.
    """
    global svm_model, preprocessor, scaler
    
    # Initialize if needed
    if svm_model is None:
        await initialize_svm(ctx, use_pretrained=True)
    
    try:
        # Preprocess email
        features = preprocessor.transform(email_content, subject, sender)
        
        # Use combined features for SVM
        X = features['combined'].reshape(1, -1)
        
        # Scale features
        if scaler is not None:
            X = scaler.transform(X)
        
        # Get prediction
        prediction = svm_model.predict(X)[0]
        
        # Get decision function value (distance from hyperplane)
        decision_value = svm_model.decision_function(X)[0]
        
        # Convert decision value to probability-like score
        # Using sigmoid function to map decision values to [0, 1]
        confidence = 1 / (1 + np.exp(-abs(decision_value)))
        
        # If model has probability=True, use predict_proba
        if hasattr(svm_model, 'predict_proba'):
            probabilities = svm_model.predict_proba(X)[0]
            spam_probability = probabilities[1]
            ham_probability = probabilities[0]
        else:
            # Estimate probabilities from decision function
            prob = 1 / (1 + np.exp(-decision_value))
            spam_probability = prob if prediction == 1 else 1 - prob
            ham_probability = 1 - spam_probability
        
        # Get support vector information
        support_info = {
            'n_support_vectors': int(np.sum(svm_model.n_support_)),
            'support_vector_ratio': float(np.sum(svm_model.n_support_) / len(svm_model.support_))
        }
        
        # Feature importance (approximation using support vectors)
        feature_importance = []
        if hasattr(svm_model, 'coef_'):
            # For linear kernel
            coef = svm_model.coef_[0]
            feature_names = preprocessor.get_feature_names()
            
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
        
        result = {
            'algorithm': 'Support Vector Machine',
            'kernel': svm_model.kernel,
            'is_spam': bool(prediction == 1),
            'spam_probability': float(spam_probability),
            'ham_probability': float(ham_probability),
            'confidence': float(confidence),
            'decision_value': float(decision_value),
            'support_vectors': support_info,
            'top_features': feature_importance[-5:] if feature_importance else [],
            'preprocessing': {
                'content_length': len(email_content),
                'subject_length': len(subject),
                'total_features': X.shape[1]
            }
        }
        
        # Add explanation
        if result['is_spam']:
            result['explanation'] = f"Email classified as SPAM with decision value {decision_value:.2f}"
        else:
            result['explanation'] = f"Email classified as HAM (legitimate) with decision value {decision_value:.2f}"
        
        return result
        
    except Exception as e:
        return {
            'algorithm': 'Support Vector Machine',
            'error': str(e),
            'is_spam': False,
            'confidence': 0.0,
            'explanation': f"Error during analysis: {str(e)}"
        }


@mcp.tool()
async def train_svm(
    ctx: Context,
    training_samples: List[Dict[str, Any]],
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str = "scale"
) -> Dict[str, Any]:
    """
    Train the SVM model with provided samples.
    Each sample should have: email_content, subject, sender, is_spam (bool)
    
    Parameters:
    - kernel: 'linear', 'rbf', 'poly', or 'sigmoid'
    - C: Regularization parameter (higher = less regularization)
    - gamma: Kernel coefficient ('scale', 'auto', or float)
    """
    global svm_model, preprocessor, scaler
    
    try:
        # Initialize preprocessor
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
        
        # Scale features (important for SVM)
        X = scaler.fit_transform(X)
        
        # Initialize and train SVM
        svm_model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=True,  # Enable probability estimates
            random_state=42
        )
        
        svm_model.fit(X, y)
        
        # Calculate training accuracy
        train_predictions = svm_model.predict(X)
        accuracy = np.mean(train_predictions == y)
        
        return {
            'success': True,
            'kernel': kernel,
            'C': C,
            'gamma': gamma,
            'samples_trained': len(training_samples),
            'training_accuracy': float(accuracy),
            'spam_samples': int(np.sum(y)),
            'ham_samples': len(y) - int(np.sum(y)),
            'feature_count': X.shape[1],
            'n_support_vectors': int(np.sum(svm_model.n_support_)),
            'support_vector_classes': {
                'ham': int(svm_model.n_support_[0]),
                'spam': int(svm_model.n_support_[1])
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


@mcp.tool()
async def initialize_svm(
    ctx: Context,
    use_pretrained: bool = True
) -> Dict[str, str]:
    """
    Initialize the SVM model.
    If use_pretrained is True, loads a pre-trained model if available.
    Otherwise, initializes with default training samples.
    """
    global svm_model, preprocessor, scaler
    
    if use_pretrained and os.path.exists('svm_model.pkl'):
        # Load pre-trained model
        with open('svm_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
            svm_model = model_data['model']
            preprocessor = model_data['preprocessor']
            scaler = model_data.get('scaler')
        return {'status': 'Loaded pre-trained SVM model'}
    
    # Initialize with default training samples
    default_samples = [
        # Spam examples
        {
            'email_content': 'Get rich quick! Make $5000 per week working from home. No experience needed!',
            'subject': 'Financial Freedom Awaits',
            'sender': 'money@quickcash.biz',
            'is_spam': True
        },
        {
            'email_content': 'Your account security is at risk. Update your password immediately by clicking this link.',
            'subject': 'Security Alert - Action Required',
            'sender': 'alert@secure-banking.net',
            'is_spam': True
        },
        {
            'email_content': 'Congratulations! You have been selected for a free iPhone. Claim it now before it expires!',
            'subject': 'Your Free iPhone is Waiting',
            'sender': 'prizes@win-now.tk',
            'is_spam': True
        },
        {
            'email_content': 'Hot singles in your area! Click here to meet them tonight. No credit card required.',
            'subject': 'Someone wants to meet you',
            'sender': 'dating@hot-meets.com',
            'is_spam': True
        },
        # Ham examples
        {
            'email_content': 'The weekly team meeting has been rescheduled to Thursday at 2 PM. Please update your calendars.',
            'subject': 'Meeting Reschedule Notice',
            'sender': 'manager@company.com',
            'is_spam': False
        },
        {
            'email_content': 'Thank you for your recent purchase. Your order will be delivered within 3-5 business days.',
            'subject': 'Order Confirmation #78234',
            'sender': 'support@shop.com',
            'is_spam': False
        },
        {
            'email_content': 'Here is the report you requested. Please let me know if you need any clarifications.',
            'subject': 'Monthly Sales Report',
            'sender': 'analyst@company.com',
            'is_spam': False
        },
        {
            'email_content': 'Reminder: Your dentist appointment is scheduled for tomorrow at 10 AM.',
            'subject': 'Appointment Reminder',
            'sender': 'noreply@dentalclinic.com',
            'is_spam': False
        }
    ]
    
    result = await train_svm(ctx, default_samples, kernel="rbf", C=1.0, gamma="scale")
    
    if result['success']:
        return {'status': f'Initialized SVM model with {len(default_samples)} samples'}
    else:
        return {'status': f'Failed to initialize: {result.get("error", "Unknown error")}'}


@mcp.tool()
async def get_svm_info(ctx: Context) -> Dict[str, Any]:
    """Get information about the current SVM model"""
    global svm_model
    
    if svm_model is None:
        return {
            'status': 'Model not initialized',
            'initialized': False
        }
    
    info = {
        'status': 'Model ready',
        'initialized': True,
        'algorithm': 'Support Vector Machine',
        'kernel': svm_model.kernel,
        'parameters': {
            'C': float(svm_model.C),
            'gamma': str(svm_model.gamma) if hasattr(svm_model, 'gamma') else 'N/A'
        }
    }
    
    if hasattr(svm_model, 'support_vectors_'):
        info['support_vectors'] = {
            'total': int(len(svm_model.support_vectors_)),
            'per_class': {
                'ham': int(svm_model.n_support_[0]),
                'spam': int(svm_model.n_support_[1])
            }
        }
    
    if hasattr(svm_model, 'classes_'):
        info['classes'] = svm_model.classes_.tolist()
    
    return info


if __name__ == "__main__":
    mcp.run()