#!/usr/bin/env python3
"""
FastMCP server for Naive Bayes email classification
"""

from fastmcp import FastMCP, Context
from sklearn.naive_bayes import MultinomialNB, GaussianNB
import numpy as np
import pickle
import os
from typing import Dict, Any
from email_preprocessor import EmailPreprocessor

# Initialize the MCP server
mcp = FastMCP("Naive Bayes Classifier")

# Global model and preprocessor
naive_bayes_model = None
preprocessor = None
model_type = "multinomial"  # or "gaussian"


@mcp.tool()
async def analyze_email_naive_bayes(
    ctx: Context,
    email_content: str,
    subject: str,
    sender: str
) -> Dict[str, Any]:
    """
    Analyze email for spam/phishing using Naive Bayes classifier.
    Returns classification result with probability scores.
    """
    global naive_bayes_model, preprocessor
    
    # Initialize if needed
    if naive_bayes_model is None:
        await initialize_naive_bayes(ctx, use_pretrained=True)
    
    try:
        # Preprocess email
        features = preprocessor.transform(email_content, subject, sender)
        
        # Use appropriate features for the model type
        if model_type == "multinomial":
            # MultinomialNB works best with count/tfidf features
            X = features['tfidf'].reshape(1, -1)
        else:
            # GaussianNB can use all features
            X = features['combined'].reshape(1, -1)
        
        # Get prediction and probabilities
        prediction = naive_bayes_model.predict(X)[0]
        probabilities = naive_bayes_model.predict_proba(X)[0]
        
        # Extract top features contributing to the decision
        feature_importance = []
        if hasattr(naive_bayes_model, 'feature_log_prob_'):
            # For MultinomialNB
            if prediction == 1:  # Spam/phishing
                log_probs = naive_bayes_model.feature_log_prob_[1]
            else:
                log_probs = naive_bayes_model.feature_log_prob_[0]
            
            # Get feature names
            if model_type == "multinomial" and preprocessor.is_fitted:
                feature_names = preprocessor.get_feature_names()['tfidf']
                top_indices = np.argsort(log_probs)[-10:]
                for idx in top_indices:
                    if X[0, idx] > 0:  # Only include features present in the email
                        feature_importance.append({
                            'feature': feature_names[idx],
                            'importance': float(np.exp(log_probs[idx]))
                        })
        
        # Calculate confidence based on probability difference
        confidence = abs(probabilities[1] - probabilities[0])
        
        result = {
            'algorithm': 'Naive Bayes',
            'model_type': model_type,
            'is_spam': bool(prediction == 1),
            'spam_probability': float(probabilities[1]),
            'ham_probability': float(probabilities[0]),
            'confidence': float(confidence),
            'top_features': feature_importance[-5:],  # Top 5 features
            'preprocessing': {
                'content_length': len(email_content),
                'subject_length': len(subject),
                'cleaned_word_count': len(preprocessor.clean_text(email_content + " " + subject).split())
            }
        }
        
        # Add explanation
        if result['is_spam']:
            result['explanation'] = f"Email classified as SPAM with {result['spam_probability']:.1%} probability"
        else:
            result['explanation'] = f"Email classified as HAM (legitimate) with {result['ham_probability']:.1%} probability"
        
        return result
        
    except Exception as e:
        return {
            'algorithm': 'Naive Bayes',
            'error': str(e),
            'is_spam': False,
            'confidence': 0.0,
            'explanation': f"Error during analysis: {str(e)}"
        }


@mcp.tool()
async def train_naive_bayes(
    ctx: Context,
    training_samples: list[Dict[str, Any]],
    model_type_choice: str = "multinomial"
) -> Dict[str, Any]:
    """
    Train the Naive Bayes model with provided samples.
    Each sample should have: email_content, subject, sender, is_spam (bool)
    model_type_choice: "multinomial" or "gaussian"
    """
    global naive_bayes_model, preprocessor, model_type
    
    try:
        model_type = model_type_choice
        
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
            if model_type == "multinomial":
                X_list.append(features['tfidf'])
            else:
                X_list.append(features['combined'])
        
        X = np.array(X_list)
        y = np.array(labels)
        
        # Initialize and train model
        if model_type == "multinomial":
            naive_bayes_model = MultinomialNB(alpha=1.0)
        else:
            naive_bayes_model = GaussianNB()
        
        naive_bayes_model.fit(X, y)
        
        # Calculate training accuracy
        train_predictions = naive_bayes_model.predict(X)
        accuracy = np.mean(train_predictions == y)
        
        return {
            'success': True,
            'model_type': model_type,
            'samples_trained': len(training_samples),
            'training_accuracy': float(accuracy),
            'spam_samples': int(np.sum(y)),
            'ham_samples': len(y) - int(np.sum(y)),
            'feature_count': X.shape[1]
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


@mcp.tool()
async def initialize_naive_bayes(
    ctx: Context,
    use_pretrained: bool = True
) -> Dict[str, str]:
    """
    Initialize the Naive Bayes model.
    If use_pretrained is True, loads a pre-trained model if available.
    Otherwise, initializes with default training samples.
    """
    global naive_bayes_model, preprocessor, model_type
    
    if use_pretrained and os.path.exists('naive_bayes_model.pkl'):
        # Load pre-trained model
        with open('naive_bayes_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
            naive_bayes_model = model_data['model']
            preprocessor = model_data['preprocessor']
            model_type = model_data.get('model_type', 'multinomial')
        return {'status': 'Loaded pre-trained Naive Bayes model (trained on real spam datasets)'}
    
    # Initialize with default training samples
    default_samples = [
        # Spam examples
        {
            'email_content': 'Congratulations! You have won $1,000,000. Click here to claim your prize now!',
            'subject': 'You are a WINNER!',
            'sender': 'lottery@quickmoney.tk',
            'is_spam': True
        },
        {
            'email_content': 'URGENT: Your account will be suspended. Verify your details immediately at this link.',
            'subject': 'Account Security Alert',
            'sender': 'security@paypal-verify.com',
            'is_spam': True
        },
        {
            'email_content': 'Buy cheap medications online! No prescription needed. Best prices guaranteed.',
            'subject': 'Discount Pharmacy',
            'sender': 'sales@pharmadiscount.ru',
            'is_spam': True
        },
        # Ham examples
        {
            'email_content': 'Hi team, Please find attached the quarterly report for review. Let me know if you have questions.',
            'subject': 'Q3 Report for Review',
            'sender': 'john.doe@company.com',
            'is_spam': False
        },
        {
            'email_content': 'Your order #12345 has been shipped. Track your package at our website.',
            'subject': 'Order Shipped',
            'sender': 'orders@amazon.com',
            'is_spam': False
        },
        {
            'email_content': 'Thanks for subscribing to our newsletter. Here are this week\'s tech news highlights.',
            'subject': 'Weekly Tech Newsletter',
            'sender': 'newsletter@techblog.com',
            'is_spam': False
        }
    ]
    
    result = await train_naive_bayes(ctx, default_samples, "multinomial")
    
    if result['success']:
        return {'status': f'Initialized Naive Bayes model with {len(default_samples)} samples'}
    else:
        return {'status': f'Failed to initialize: {result.get("error", "Unknown error")}'}


@mcp.tool()
async def get_naive_bayes_info(ctx: Context) -> Dict[str, Any]:
    """Get information about the current Naive Bayes model"""
    global naive_bayes_model, model_type
    
    if naive_bayes_model is None:
        return {
            'status': 'Model not initialized',
            'initialized': False
        }
    
    info = {
        'status': 'Model ready',
        'initialized': True,
        'algorithm': 'Naive Bayes',
        'model_type': model_type,
        'implementation': type(naive_bayes_model).__name__
    }
    
    if hasattr(naive_bayes_model, 'class_count_'):
        info['class_distribution'] = {
            'ham': int(naive_bayes_model.class_count_[0]),
            'spam': int(naive_bayes_model.class_count_[1])
        }
    
    if hasattr(naive_bayes_model, 'feature_count_'):
        info['total_features'] = naive_bayes_model.feature_count_.shape[1]
    
    return info


if __name__ == "__main__":
    mcp.run()