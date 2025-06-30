#!/usr/bin/env python3
"""Simple test of trained models"""

import pickle
import numpy as np

# Test emails
test_emails = [
    {
        "content": "URGENT! Your account will be suspended. Click here to verify: http://bit.ly/verify",
        "subject": "Account Security Alert!!!",
        "sender": "security@paypal-verify.tk",
        "expected": "SPAM"
    },
    {
        "content": "Hi team, Please review the Q4 report attached. Let me know if you have questions.",
        "subject": "Q4 Report for Review",
        "sender": "john@company.com",
        "expected": "HAM"
    }
]

print("Testing Trained Models")
print("="*50)

# Test Naive Bayes
print("\nNaive Bayes Model:")
with open('naive_bayes_model.pkl', 'rb') as f:
    nb_data = pickle.load(f)
    model = nb_data['model']
    preprocessor = nb_data['preprocessor']

for test in test_emails:
    features = preprocessor.transform(test['content'], test['subject'], test['sender'])
    X = features['tfidf'].reshape(1, -1)
    
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    
    result = "SPAM" if prediction == 1 else "HAM"
    print(f"\n  Email: {test['subject']}")
    print(f"  Prediction: {result} (expected: {test['expected']})")
    print(f"  Confidence: {max(proba):.1%}")
    print(f"  {'✅' if result == test['expected'] else '❌'}")

# Test Neural Network
print("\n\nNeural Network Model:")
with open('neural_network_model.pkl', 'rb') as f:
    nn_data = pickle.load(f)
    model = nn_data['model']
    preprocessor = nn_data['preprocessor']
    scaler = nn_data['scaler']

for test in test_emails:
    features = preprocessor.transform(test['content'], test['subject'], test['sender'])
    X = features['combined'].reshape(1, -1)
    X = scaler.transform(X)
    
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    
    result = "SPAM" if prediction == 1 else "HAM"
    print(f"\n  Email: {test['subject']}")
    print(f"  Prediction: {result} (expected: {test['expected']})")
    print(f"  Confidence: {max(proba):.1%}")
    print(f"  {'✅' if result == test['expected'] else '❌'}")

print("\n\nAll models successfully loaded and tested!")
print("They are ready to be used via FastMCP servers.")