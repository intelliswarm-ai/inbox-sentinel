#!/usr/bin/env python3
"""
FastMCP server for Random Forest email classification
"""

from fastmcp import FastMCP, Context
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
import os
from typing import Dict, Any, List
from email_preprocessor import EmailPreprocessor

# Initialize the MCP server
mcp = FastMCP("Random Forest Classifier")

# Global model and preprocessor
rf_model = None
preprocessor = None


@mcp.tool()
async def analyze_email_random_forest(
    ctx: Context,
    email_content: str,
    subject: str,
    sender: str
) -> Dict[str, Any]:
    """
    Analyze email for spam/phishing using Random Forest classifier.
    Returns classification result with probability scores and feature importance.
    """
    global rf_model, preprocessor
    
    # Initialize if needed
    if rf_model is None:
        await initialize_random_forest(ctx, use_pretrained=True)
    
    try:
        # Preprocess email
        features = preprocessor.transform(email_content, subject, sender)
        
        # Use combined features for Random Forest
        X = features['combined'].reshape(1, -1)
        
        # Get prediction and probabilities
        prediction = rf_model.predict(X)[0]
        probabilities = rf_model.predict_proba(X)[0]
        
        # Get feature importance
        feature_importance = []
        feature_names = preprocessor.get_feature_names()
        importances = rf_model.feature_importances_
        
        # Combine all feature names
        all_feature_names = feature_names.get('tfidf', []) + feature_names.get('manual', [])
        
        # Get top important features
        top_indices = np.argsort(importances)[-15:]
        for idx in top_indices:
            if idx < len(all_feature_names):
                feature_name = all_feature_names[idx]
            else:
                feature_name = f"feature_{idx}"
            
            feature_importance.append({
                'feature': feature_name,
                'importance': float(importances[idx]),
                'value': float(X[0, idx]) if X[0, idx] != 0 else 0
            })
        
        # Sort by importance
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        # Get tree voting information
        tree_predictions = []
        for tree in rf_model.estimators_[:10]:  # Sample first 10 trees
            tree_pred = tree.predict(X)[0]
            tree_predictions.append(int(tree_pred))
        
        spam_votes = sum(tree_predictions)
        total_votes = len(tree_predictions)
        
        # Calculate confidence based on probability and tree agreement
        confidence = probabilities[1] if prediction == 1 else probabilities[0]
        tree_agreement = spam_votes / total_votes if prediction == 1 else (total_votes - spam_votes) / total_votes
        
        result = {
            'algorithm': 'Random Forest',
            'is_spam': bool(prediction == 1),
            'spam_probability': float(probabilities[1]),
            'ham_probability': float(probabilities[0]),
            'confidence': float(confidence),
            'tree_voting': {
                'spam_votes': spam_votes,
                'ham_votes': total_votes - spam_votes,
                'total_trees_sampled': total_votes,
                'agreement_ratio': float(tree_agreement)
            },
            'top_features': feature_importance[:10],  # Top 10 features
            'model_info': {
                'n_estimators': rf_model.n_estimators,
                'max_depth': rf_model.max_depth if rf_model.max_depth else 'unlimited',
                'min_samples_split': rf_model.min_samples_split
            },
            'preprocessing': {
                'content_length': len(email_content),
                'subject_length': len(subject),
                'total_features': X.shape[1]
            }
        }
        
        # Add detailed explanation
        if result['is_spam']:
            result['explanation'] = (
                f"Email classified as SPAM with {result['spam_probability']:.1%} probability. "
                f"{spam_votes}/{total_votes} trees voted for spam classification."
            )
        else:
            result['explanation'] = (
                f"Email classified as HAM (legitimate) with {result['ham_probability']:.1%} probability. "
                f"{total_votes - spam_votes}/{total_votes} trees voted for legitimate classification."
            )
        
        return result
        
    except Exception as e:
        return {
            'algorithm': 'Random Forest',
            'error': str(e),
            'is_spam': False,
            'confidence': 0.0,
            'explanation': f"Error during analysis: {str(e)}"
        }


@mcp.tool()
async def train_random_forest(
    ctx: Context,
    training_samples: List[Dict[str, Any]],
    n_estimators: int = 100,
    max_depth: int = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1
) -> Dict[str, Any]:
    """
    Train the Random Forest model with provided samples.
    Each sample should have: email_content, subject, sender, is_spam (bool)
    
    Parameters:
    - n_estimators: Number of trees in the forest
    - max_depth: Maximum depth of trees (None for unlimited)
    - min_samples_split: Minimum samples required to split a node
    - min_samples_leaf: Minimum samples required at a leaf node
    """
    global rf_model, preprocessor
    
    try:
        # Initialize preprocessor
        preprocessor = EmailPreprocessor()
        
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
        
        # Initialize and train Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1  # Use all available processors
        )
        
        rf_model.fit(X, y)
        
        # Calculate training accuracy and OOB score if available
        train_predictions = rf_model.predict(X)
        accuracy = np.mean(train_predictions == y)
        
        # Get feature importance statistics
        importances = rf_model.feature_importances_
        top_features_idx = np.argsort(importances)[-5:]
        
        result = {
            'success': True,
            'n_estimators': n_estimators,
            'max_depth': max_depth if max_depth else 'unlimited',
            'samples_trained': len(training_samples),
            'training_accuracy': float(accuracy),
            'spam_samples': int(np.sum(y)),
            'ham_samples': len(y) - int(np.sum(y)),
            'feature_count': X.shape[1],
            'top_feature_importances': [
                {'index': int(idx), 'importance': float(importances[idx])}
                for idx in top_features_idx
            ]
        }
        
        # Add OOB score if available
        if hasattr(rf_model, 'oob_score_') and rf_model.oob_score_:
            result['oob_score'] = float(rf_model.oob_score_)
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


@mcp.tool()
async def initialize_random_forest(
    ctx: Context,
    use_pretrained: bool = True
) -> Dict[str, str]:
    """
    Initialize the Random Forest model.
    If use_pretrained is True, loads a pre-trained model if available.
    Otherwise, initializes with default training samples.
    """
    global rf_model, preprocessor
    
    if use_pretrained and os.path.exists('random_forest_model.pkl'):
        # Load pre-trained model
        with open('random_forest_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
            rf_model = model_data['model']
            preprocessor = model_data['preprocessor']
        return {'status': 'Loaded pre-trained Random Forest model'}
    
    # Initialize with comprehensive training samples
    default_samples = [
        # Spam examples
        {
            'email_content': 'URGENT: Your account will be closed! Click here immediately to verify your identity and keep your account active.',
            'subject': 'Account Closure Warning!!!',
            'sender': 'security@paypal-verification.tk',
            'is_spam': True
        },
        {
            'email_content': 'You have won the international lottery! Send us your bank details to claim your $1,000,000 prize.',
            'subject': 'Congratulations Winner!',
            'sender': 'lottery@winbig.ml',
            'is_spam': True
        },
        {
            'email_content': 'Lose 30 pounds in 30 days! This miracle pill will change your life. Order now with 50% discount.',
            'subject': 'Amazing Weight Loss Secret',
            'sender': 'deals@diet-pills.biz',
            'is_spam': True
        },
        {
            'email_content': 'Hot Russian girls want to meet you! Click here for instant access. No credit card needed.',
            'subject': 'Someone is waiting for you',
            'sender': 'dating@meet-hot-girls.ru',
            'is_spam': True
        },
        {
            'email_content': 'Make $5000 per week from home! No experience necessary. Start earning today!',
            'subject': 'Work From Home Opportunity',
            'sender': 'jobs@easy-money.net',
            'is_spam': True
        },
        # Ham examples
        {
            'email_content': 'Hi Sarah, Just wanted to confirm our meeting tomorrow at 3 PM. Let me know if you need to reschedule.',
            'subject': 'Meeting Tomorrow',
            'sender': 'john.smith@company.com',
            'is_spam': False
        },
        {
            'email_content': 'Your Amazon order #123-456789 has been shipped. Track your package using the link in your account.',
            'subject': 'Your order has shipped',
            'sender': 'ship-confirm@amazon.com',
            'is_spam': False
        },
        {
            'email_content': 'Please find attached the quarterly financial report as requested. Let me know if you have any questions.',
            'subject': 'Q3 Financial Report',
            'sender': 'cfo@corporation.com',
            'is_spam': False
        },
        {
            'email_content': 'Thank you for subscribing to our newsletter. You can unsubscribe at any time using the link below.',
            'subject': 'Welcome to Tech News',
            'sender': 'newsletter@technews.com',
            'is_spam': False
        },
        {
            'email_content': 'Reminder: Your dental appointment is scheduled for Friday at 2 PM. Please call if you need to cancel.',
            'subject': 'Appointment Reminder',
            'sender': 'appointments@dentalcare.com',
            'is_spam': False
        }
    ]
    
    result = await train_random_forest(ctx, default_samples, n_estimators=100, max_depth=10)
    
    if result['success']:
        return {'status': f'Initialized Random Forest model with {len(default_samples)} samples'}
    else:
        return {'status': f'Failed to initialize: {result.get("error", "Unknown error")}'}


@mcp.tool()
async def get_random_forest_info(ctx: Context) -> Dict[str, Any]:
    """Get information about the current Random Forest model"""
    global rf_model
    
    if rf_model is None:
        return {
            'status': 'Model not initialized',
            'initialized': False
        }
    
    info = {
        'status': 'Model ready',
        'initialized': True,
        'algorithm': 'Random Forest',
        'parameters': {
            'n_estimators': rf_model.n_estimators,
            'max_depth': rf_model.max_depth if rf_model.max_depth else 'unlimited',
            'min_samples_split': rf_model.min_samples_split,
            'min_samples_leaf': rf_model.min_samples_leaf,
            'max_features': rf_model.max_features
        }
    }
    
    if hasattr(rf_model, 'feature_importances_'):
        # Get top 5 most important features
        importances = rf_model.feature_importances_
        top_indices = np.argsort(importances)[-5:]
        info['top_feature_indices'] = [
            {'index': int(idx), 'importance': float(importances[idx])}
            for idx in reversed(top_indices)
        ]
    
    if hasattr(rf_model, 'estimators_'):
        info['forest_info'] = {
            'total_trees': len(rf_model.estimators_),
            'trees_fitted': sum(1 for tree in rf_model.estimators_ if hasattr(tree, 'tree_'))
        }
    
    return info


if __name__ == "__main__":
    mcp.run()