#!/usr/bin/env python3
"""
FastMCP server for Logistic Regression email classification
"""

from fastmcp import FastMCP, Context
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import os
from typing import Dict, Any, List
from email_preprocessor import EmailPreprocessor

# Initialize the MCP server
mcp = FastMCP("Logistic Regression Classifier")

# Global model and preprocessor
lr_model = None
preprocessor = None
scaler = None


@mcp.tool()
async def analyze_email_logistic_regression(
    ctx: Context,
    email_content: str,
    subject: str,
    sender: str
) -> Dict[str, Any]:
    """
    Analyze email for spam/phishing using Logistic Regression classifier.
    Returns classification result with probability scores and feature coefficients.
    """
    global lr_model, preprocessor, scaler
    
    # Initialize if needed
    if lr_model is None:
        await initialize_logistic_regression(ctx, use_pretrained=True)
    
    try:
        # Preprocess email
        features = preprocessor.transform(email_content, subject, sender)
        
        # Use combined features for Logistic Regression
        X = features['combined'].reshape(1, -1)
        
        # Scale features
        if scaler is not None:
            X = scaler.transform(X)
        
        # Get prediction and probabilities
        prediction = lr_model.predict(X)[0]
        probabilities = lr_model.predict_proba(X)[0]
        
        # Get decision function value (log odds)
        decision_value = lr_model.decision_function(X)[0]
        
        # Get feature contributions
        coefficients = lr_model.coef_[0]
        intercept = lr_model.intercept_[0]
        
        # Calculate feature contributions to the decision
        feature_contributions = []
        feature_names = preprocessor.get_feature_names()
        all_feature_names = feature_names.get('tfidf', []) + feature_names.get('manual', [])
        
        # Get features with highest absolute coefficients
        top_indices = np.argsort(np.abs(coefficients))[-20:]
        
        for idx in top_indices:
            if idx < len(all_feature_names):
                feature_name = all_feature_names[idx]
            else:
                feature_name = f"feature_{idx}"
            
            contribution = coefficients[idx] * X[0, idx]
            if abs(contribution) > 0.01:  # Only include meaningful contributions
                feature_contributions.append({
                    'feature': feature_name,
                    'coefficient': float(coefficients[idx]),
                    'value': float(X[0, idx]),
                    'contribution': float(contribution)
                })
        
        # Sort by absolute contribution
        feature_contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        # Calculate odds ratio
        odds_ratio = np.exp(decision_value)
        
        result = {
            'algorithm': 'Logistic Regression',
            'is_spam': bool(prediction == 1),
            'spam_probability': float(probabilities[1]),
            'ham_probability': float(probabilities[0]),
            'confidence': float(max(probabilities)),
            'decision_value': float(decision_value),
            'odds_ratio': float(odds_ratio),
            'intercept': float(intercept),
            'top_feature_contributions': feature_contributions[:10],  # Top 10 contributions
            'model_info': {
                'regularization': lr_model.penalty,
                'C': float(lr_model.C),
                'solver': lr_model.solver,
                'convergence': bool(lr_model.n_iter_ < lr_model.max_iter) if hasattr(lr_model, 'n_iter_') else None
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
                f"Odds ratio: {odds_ratio:.2f}:1 in favor of spam."
            )
        else:
            result['explanation'] = (
                f"Email classified as HAM (legitimate) with {result['ham_probability']:.1%} probability. "
                f"Odds ratio: {1/odds_ratio:.2f}:1 in favor of legitimate."
            )
        
        # Add feature interpretation
        if feature_contributions:
            top_spam_features = [f for f in feature_contributions[:5] if f['contribution'] > 0]
            top_ham_features = [f for f in feature_contributions[:5] if f['contribution'] < 0]
            
            if top_spam_features:
                result['spam_indicators'] = [f['feature'] for f in top_spam_features]
            if top_ham_features:
                result['ham_indicators'] = [f['feature'] for f in top_ham_features]
        
        return result
        
    except Exception as e:
        return {
            'algorithm': 'Logistic Regression',
            'error': str(e),
            'is_spam': False,
            'confidence': 0.0,
            'explanation': f"Error during analysis: {str(e)}"
        }


@mcp.tool()
async def train_logistic_regression(
    ctx: Context,
    training_samples: List[Dict[str, Any]],
    penalty: str = "l2",
    C: float = 1.0,
    solver: str = "lbfgs",
    max_iter: int = 1000
) -> Dict[str, Any]:
    """
    Train the Logistic Regression model with provided samples.
    Each sample should have: email_content, subject, sender, is_spam (bool)
    
    Parameters:
    - penalty: 'l1', 'l2', 'elasticnet', or None
    - C: Inverse regularization strength (higher = less regularization)
    - solver: 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'
    - max_iter: Maximum iterations for solver convergence
    """
    global lr_model, preprocessor, scaler
    
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
        
        # Scale features (important for logistic regression)
        X = scaler.fit_transform(X)
        
        # Initialize and train Logistic Regression
        lr_model = LogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver,
            max_iter=max_iter,
            random_state=42
        )
        
        lr_model.fit(X, y)
        
        # Calculate training accuracy
        train_predictions = lr_model.predict(X)
        accuracy = np.mean(train_predictions == y)
        
        # Get coefficient statistics
        coef_stats = {
            'n_features': len(lr_model.coef_[0]),
            'n_nonzero_coef': np.sum(lr_model.coef_[0] != 0),
            'max_coef': float(np.max(np.abs(lr_model.coef_[0]))),
            'mean_coef': float(np.mean(np.abs(lr_model.coef_[0])))
        }
        
        result = {
            'success': True,
            'penalty': penalty,
            'C': C,
            'solver': solver,
            'samples_trained': len(training_samples),
            'training_accuracy': float(accuracy),
            'spam_samples': int(np.sum(y)),
            'ham_samples': len(y) - int(np.sum(y)),
            'feature_count': X.shape[1],
            'coefficient_stats': coef_stats,
            'converged': bool(lr_model.n_iter_ < max_iter) if hasattr(lr_model, 'n_iter_') else None,
            'iterations': int(lr_model.n_iter_) if hasattr(lr_model, 'n_iter_') else None
        }
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


@mcp.tool()
async def initialize_logistic_regression(
    ctx: Context,
    use_pretrained: bool = True
) -> Dict[str, str]:
    """
    Initialize the Logistic Regression model.
    If use_pretrained is True, loads a pre-trained model if available.
    Otherwise, initializes with default training samples.
    """
    global lr_model, preprocessor, scaler
    
    if use_pretrained and os.path.exists('logistic_regression_model.pkl'):
        # Load pre-trained model
        with open('logistic_regression_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
            lr_model = model_data['model']
            preprocessor = model_data['preprocessor']
            scaler = model_data.get('scaler')
        return {'status': 'Loaded pre-trained Logistic Regression model'}
    
    # Initialize with diverse training samples
    default_samples = [
        # Spam examples with varying characteristics
        {
            'email_content': 'Act now! Limited time offer. Get 90% off all products. This deal expires in 24 hours!',
            'subject': '⚡ FLASH SALE - 90% OFF Everything!',
            'sender': 'deals@mega-discounts.shop',
            'is_spam': True
        },
        {
            'email_content': 'Dear valued customer, your account has been compromised. Reset your password immediately.',
            'subject': 'Security Alert: Immediate Action Required',
            'sender': 'no-reply@secure-bank-alert.net',
            'is_spam': True
        },
        {
            'email_content': 'Congratulations! You are our lucky winner. Claim your free vacation package today.',
            'subject': 'You Won a Free Vacation!',
            'sender': 'prizes@vacation-winners.com',
            'is_spam': True
        },
        {
            'email_content': 'Make money fast! Work from home and earn $10,000 per month. No experience needed.',
            'subject': 'Financial Freedom Opportunity',
            'sender': 'jobs@work-home-rich.biz',
            'is_spam': True
        },
        {
            'email_content': 'Singles in your area want to meet! Join now for free. No credit card required.',
            'subject': 'Someone likes you!',
            'sender': 'notifications@dating-local.net',
            'is_spam': True
        },
        {
            'email_content': 'URGENT: Tax refund waiting. Provide your bank details to receive $3,421.00',
            'subject': 'Tax Refund Notice',
            'sender': 'refunds@irs-gov.us',
            'is_spam': True
        },
        # Ham examples
        {
            'email_content': 'Hi Mike, I have reviewed the proposal and it looks good. Can we schedule a call to discuss next steps?',
            'subject': 'Re: Project Proposal',
            'sender': 'sarah.johnson@techcorp.com',
            'is_spam': False
        },
        {
            'email_content': 'Your monthly statement is now available. Log in to your account to view and download.',
            'subject': 'Your Statement is Ready',
            'sender': 'statements@creditcard.com',
            'is_spam': False
        },
        {
            'email_content': 'Thank you for your purchase. Your order #98765 will be delivered by Friday.',
            'subject': 'Order Confirmation',
            'sender': 'orders@online-store.com',
            'is_spam': False
        },
        {
            'email_content': 'Reminder: Team meeting tomorrow at 10 AM in conference room B. Please review the attached agenda.',
            'subject': 'Meeting Reminder',
            'sender': 'calendar@company.com',
            'is_spam': False
        },
        {
            'email_content': 'Your subscription renewal is coming up. No action needed - your plan will auto-renew.',
            'subject': 'Subscription Renewal Notice',
            'sender': 'billing@streaming-service.com',
            'is_spam': False
        },
        {
            'email_content': 'Here are the meeting notes from today. Let me know if I missed anything important.',
            'subject': 'Meeting Notes - Product Review',
            'sender': 'assistant@company.com',
            'is_spam': False
        }
    ]
    
    result = await train_logistic_regression(ctx, default_samples, penalty="l2", C=1.0)
    
    if result['success']:
        return {'status': f'Initialized Logistic Regression model with {len(default_samples)} samples'}
    else:
        return {'status': f'Failed to initialize: {result.get("error", "Unknown error")}'}


@mcp.tool()
async def get_logistic_regression_info(ctx: Context) -> Dict[str, Any]:
    """Get information about the current Logistic Regression model"""
    global lr_model
    
    if lr_model is None:
        return {
            'status': 'Model not initialized',
            'initialized': False
        }
    
    info = {
        'status': 'Model ready',
        'initialized': True,
        'algorithm': 'Logistic Regression',
        'parameters': {
            'penalty': lr_model.penalty,
            'C': float(lr_model.C),
            'solver': lr_model.solver,
            'max_iter': lr_model.max_iter
        }
    }
    
    if hasattr(lr_model, 'coef_'):
        coef = lr_model.coef_[0]
        info['coefficient_stats'] = {
            'n_features': len(coef),
            'n_nonzero': int(np.sum(coef != 0)),
            'sparsity': float(np.sum(coef == 0) / len(coef)),
            'max_abs_coef': float(np.max(np.abs(coef))),
            'mean_abs_coef': float(np.mean(np.abs(coef)))
        }
    
    if hasattr(lr_model, 'intercept_'):
        info['intercept'] = float(lr_model.intercept_[0])
    
    if hasattr(lr_model, 'n_iter_'):
        info['convergence'] = {
            'iterations': int(lr_model.n_iter_),
            'converged': bool(lr_model.n_iter_ < lr_model.max_iter)
        }
    
    return info


if __name__ == "__main__":
    mcp.run()