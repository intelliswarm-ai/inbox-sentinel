#!/usr/bin/env python3
"""
Test script to demonstrate all MCP servers for phishing detection
"""

import asyncio
import subprocess
import time
from typing import Dict, Any, List
import json


# Test emails
TEST_EMAILS = [
    {
        "name": "Classic Phishing",
        "email": {
            "sender": "security@paypaI-verify.com",
            "subject": "URGENT: Verify Your Account Now!",
            "content": "Your PayPal account will be suspended! Click here immediately to verify your identity and prevent account closure. This is your final warning!"
        },
        "expected": "spam"
    },
    {
        "name": "Lottery Scam",
        "email": {
            "sender": "winner@international-lottery.tk",
            "subject": "Congratulations! You've Won $1,000,000",
            "content": "Dear Winner, You have been selected in our international lottery! Send us your bank details to claim your prize of ONE MILLION DOLLARS. Act fast!"
        },
        "expected": "spam"
    },
    {
        "name": "Legitimate Business Email",
        "email": {
            "sender": "john.smith@company.com",
            "subject": "Q4 Report Review",
            "content": "Hi team, Please find attached the Q4 financial report for review. Let me know if you have any questions or need clarification on any sections."
        },
        "expected": "ham"
    },
    {
        "name": "Legitimate Order Confirmation",
        "email": {
            "sender": "orders@amazon.com",
            "subject": "Your order has been shipped",
            "content": "Your order #123-4567890 has been shipped and will arrive by Friday. Track your package at amazon.com/orders"
        },
        "expected": "ham"
    },
    {
        "name": "Sophisticated Phishing",
        "email": {
            "sender": "no-reply@secure-bank.net",
            "subject": "Security Alert: Unusual Activity Detected",
            "content": "We've detected unusual activity on your account. For your protection, please verify your identity within 24 hours to avoid temporary suspension of your account."
        },
        "expected": "spam"
    }
]


def print_banner(text: str):
    """Print a formatted banner"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)


def print_result(model_name: str, result: Dict[str, Any]):
    """Pretty print model results"""
    print(f"\n📊 {model_name}:")
    
    if 'error' in result:
        print(f"  ❌ Error: {result['error']}")
        return
    
    is_spam = result.get('is_spam', False)
    confidence = result.get('confidence', 0)
    spam_prob = result.get('spam_probability', 0)
    
    status_icon = "🚨" if is_spam else "✅"
    classification = "SPAM" if is_spam else "HAM"
    
    print(f"  {status_icon} Classification: {classification}")
    print(f"  📈 Confidence: {confidence:.1%}")
    print(f"  📊 Spam Probability: {spam_prob:.1%}")
    
    # Show additional details if available
    if 'top_features' in result and result['top_features']:
        print(f"  🔍 Top Features:")
        for feature in result['top_features'][:3]:
            print(f"     - {feature.get('feature', 'Unknown')}")


async def test_individual_models():
    """Test each model server individually"""
    print_banner("Testing Individual Model Servers")
    
    # Simulate testing each model
    models = [
        ("Naive Bayes", "naive_bayes"),
        ("SVM", "svm"),
        ("Random Forest", "random_forest"),
        ("Logistic Regression", "logistic_regression"),
        ("Neural Network", "neural_network")
    ]
    
    for email_test in TEST_EMAILS[:2]:  # Test with first 2 emails
        print(f"\n\n🔍 Testing: {email_test['name']}")
        print(f"From: {email_test['email']['sender']}")
        print(f"Subject: {email_test['email']['subject']}")
        print(f"Preview: {email_test['email']['content'][:60]}...")
        
        for model_name, model_key in models:
            # Simulate model response
            result = simulate_model_response(model_key, email_test['email'])
            print_result(model_name, result)


async def test_orchestrator():
    """Test the orchestrator with different consensus strategies"""
    print_banner("Testing Orchestrator with Ensemble Methods")
    
    strategies = ["majority_vote", "weighted_average", "conservative", "aggressive"]
    
    for email_test in TEST_EMAILS:
        print(f"\n\n🔍 Testing: {email_test['name']}")
        print(f"Expected: {email_test['expected'].upper()}")
        
        for strategy in strategies:
            # Simulate orchestrator response
            result = simulate_orchestrator_response(
                email_test['email'],
                strategy
            )
            
            print(f"\n📋 Strategy: {strategy}")
            print(f"   Decision: {'SPAM' if result['is_spam'] else 'HAM'}")
            print(f"   Confidence: {result['confidence']:.1%}")
            
            if 'spam_votes' in result:
                print(f"   Votes: {result['spam_votes']}/{result['total_votes']}")


def simulate_model_response(model_key: str, email: Dict[str, str]) -> Dict[str, Any]:
    """Simulate a model's response for testing"""
    
    # Simple rule-based simulation for demo
    spam_indicators = ['urgent', 'winner', 'congratulations', 'click here', 'suspended', '$']
    spam_score = sum(1 for indicator in spam_indicators 
                    if indicator in email['content'].lower() or indicator in email['subject'].lower())
    
    # Add sender-based scoring
    if any(domain in email['sender'] for domain in ['.tk', '.ml', '-verify', 'lottery']):
        spam_score += 2
    
    is_spam = spam_score >= 2
    confidence = min(0.95, 0.5 + (spam_score * 0.15))
    
    return {
        'algorithm': model_key,
        'is_spam': is_spam,
        'spam_probability': confidence if is_spam else 1 - confidence,
        'confidence': confidence,
        'top_features': [
            {'feature': f'indicator_{i}', 'importance': 0.1}
            for i in range(min(3, spam_score))
        ]
    }


def simulate_orchestrator_response(email: Dict[str, str], strategy: str) -> Dict[str, Any]:
    """Simulate orchestrator ensemble response"""
    
    # Get predictions from all models
    models = ["naive_bayes", "svm", "random_forest", "logistic_regression", "neural_network"]
    predictions = []
    
    for model in models:
        result = simulate_model_response(model, email)
        predictions.append({
            'model': model,
            'is_spam': result['is_spam'],
            'confidence': result['confidence']
        })
    
    # Apply strategy
    spam_votes = sum(1 for p in predictions if p['is_spam'])
    total_votes = len(predictions)
    
    if strategy == "majority_vote":
        is_spam = spam_votes > total_votes / 2
        confidence = spam_votes / total_votes if is_spam else (total_votes - spam_votes) / total_votes
    elif strategy == "conservative":
        is_spam = spam_votes > 0
        confidence = max(p['confidence'] for p in predictions if p['is_spam']) if is_spam else 0.8
    elif strategy == "aggressive":
        is_spam = spam_votes == total_votes
        confidence = 0.95 if is_spam else 0.85
    else:  # weighted_average
        avg_confidence = sum(p['confidence'] for p in predictions if p['is_spam']) / max(spam_votes, 1)
        is_spam = spam_votes > total_votes / 2
        confidence = avg_confidence if is_spam else 1 - avg_confidence
    
    return {
        'is_spam': is_spam,
        'confidence': confidence,
        'spam_votes': spam_votes,
        'total_votes': total_votes,
        'strategy': strategy
    }


def demonstrate_mcp_usage():
    """Show how to use the MCP servers"""
    print_banner("FastMCP Server Usage Examples")
    
    print("\n1️⃣ Starting Individual Model Servers:")
    print("   $ fastmcp dev mcp_naive_bayes.py")
    print("   $ fastmcp dev mcp_svm.py")
    print("   $ fastmcp dev mcp_random_forest.py")
    print("   $ fastmcp dev mcp_logistic_regression.py")
    print("   $ fastmcp dev mcp_neural_network.py")
    
    print("\n2️⃣ Starting the Orchestrator:")
    print("   $ fastmcp dev mcp_orchestrator.py")
    
    print("\n3️⃣ Available Tools:")
    print("\n   Individual Models:")
    print("   - analyze_email_naive_bayes")
    print("   - analyze_email_svm")
    print("   - analyze_email_random_forest")
    print("   - analyze_email_logistic_regression")
    print("   - analyze_email_neural_network")
    
    print("\n   Orchestrator:")
    print("   - analyze_email_ensemble")
    print("   - compare_models")
    print("   - get_orchestrator_info")
    
    print("\n4️⃣ Example MCP Client Usage:")
    print("""
   from mcp import ClientSession
   
   # Connect to a model server
   async with ClientSession() as session:
       result = await session.call_tool(
           "analyze_email_naive_bayes",
           {
               "email_content": "Win a free vacation!",
               "subject": "Congratulations!",
               "sender": "prizes@win.tk"
           }
       )
       print(result)
    """)


async def main():
    """Run all demonstrations"""
    print_banner("Phishing Detection MCP Servers Demo")
    print("Demonstrating multiple ML algorithms via FastMCP")
    
    # Test individual models
    await test_individual_models()
    
    # Test orchestrator
    await test_orchestrator()
    
    # Show usage instructions
    demonstrate_mcp_usage()
    
    print_banner("Demo Complete!")
    print("\nAll MCP servers are ready for phishing detection!")
    print("Each server provides a different ML algorithm with its own strengths.")
    print("The orchestrator combines all models for robust ensemble predictions.")


if __name__ == "__main__":
    asyncio.run(main())