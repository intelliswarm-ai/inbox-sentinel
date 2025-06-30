#!/usr/bin/env python3
"""Quick test of a trained MCP server"""

import asyncio
from mcp_naive_bayes import analyze_email_naive_bayes, initialize_naive_bayes

async def test():
    class MockContext:
        pass
    
    ctx = MockContext()
    
    # Test emails
    test_cases = [
        {
            "name": "Obvious Spam",
            "email": {
                "content": "CONGRATULATIONS! You've won $1,000,000! Click here NOW to claim your prize!",
                "subject": "YOU ARE A WINNER!!!",
                "sender": "lottery@win-big.tk"
            },
            "expected": "spam"
        },
        {
            "name": "Business Email",
            "email": {
                "content": "Hi John, Please review the attached proposal and let me know your thoughts. Thanks, Sarah",
                "subject": "Project Proposal for Review",
                "sender": "sarah@company.com"
            },
            "expected": "ham"
        }
    ]
    
    print("Testing Naive Bayes MCP Server with Trained Model")
    print("="*50)
    
    # Initialize will load pre-trained model
    init_result = await initialize_naive_bayes._tool_func(ctx, use_pretrained=True)
    print(f"Initialization: {init_result['status']}")
    
    for test in test_cases:
        print(f"\n{test['name']}:")
        result = await analyze_email_naive_bayes._tool_func(
            ctx,
            test['email']['content'],
            test['email']['subject'],
            test['email']['sender']
        )
        
        print(f"  Prediction: {'SPAM' if result['is_spam'] else 'HAM'}")
        print(f"  Confidence: {result['confidence']:.1%}")
        print(f"  Spam Probability: {result['spam_probability']:.1%}")
        
        expected_is_spam = test['expected'] == 'spam'
        correct = result['is_spam'] == expected_is_spam
        print(f"  Result: {'✅ Correct' if correct else '❌ Incorrect'}")

if __name__ == "__main__":
    asyncio.run(test())