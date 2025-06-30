#!/usr/bin/env python3
"""
Test suite for the phishing detection workflow
"""

import asyncio
import json
from phishing_workflow import PhishingDetectionWorkflow, AnalysisStrategy


test_emails = [
    {
        "name": "Classic PayPal Phishing",
        "sender": "service@paypaI-security.com",
        "subject": "Your Account Will Be Suspended",
        "content": """URGENT: Your PayPal account will be suspended!

Click here immediately to verify your identity: http://bit.ly/paypal-urgent-verify

This is your final notice. Act within 2 hours or lose access permanently.

PayPal Security Department""",
        "expected": "phishing"
    },
    {
        "name": "Lottery Scam",
        "sender": "winner@lottery-international.tk",
        "subject": "Congratulations! You've Won $1,000,000",
        "content": """Dear Winner,

CONGRATULATIONS! You have won ONE MILLION DOLLARS in our international lottery!

To claim your prize, please send us your bank details and pay the processing fee of $500.

Click here to claim: http://192.168.1.100/claim-prize

Best regards,
International Lottery Commission""",
        "expected": "phishing"
    },
    {
        "name": "Legitimate Newsletter",
        "sender": "news@techcompany.com",
        "subject": "TechCompany Weekly Update",
        "content": """Hi there,

Here's what's new at TechCompany this week:

- New feature release: Dark mode is now available
- Blog post: Best practices for API security
- Upcoming webinar on cloud architecture

Read more on our blog: https://techcompany.com/blog

Unsubscribe: https://techcompany.com/unsubscribe

Best,
TechCompany Team""",
        "expected": "legitimate"
    },
    {
        "name": "Bank Security Alert (Legitimate)",
        "sender": "alerts@wellsfargo.com",
        "subject": "New device sign-in detected",
        "content": """We noticed a new sign-in to your Wells Fargo account from:

Device: Chrome on Windows
Location: San Francisco, CA
Time: 2:34 PM PST

If this was you, no action is needed. If not, please visit wellsfargo.com/security

This is an automated message. Please do not reply.""",
        "expected": "legitimate"
    },
    {
        "name": "Sophisticated Phishing",
        "sender": "no-reply@amaz0n-delivery.com",
        "subject": "Re: Your Order #7823984729",
        "content": """Your Amazon order could not be delivered.

Order #7823984729
Status: Delivery Failed - Address verification needed

Please confirm your delivery address within 24 hours to avoid order cancellation:
https://amaz0n-delivery.com/verify?order=7823984729

If you did not place this order, please contact us immediately.

Amazon Customer Service""",
        "expected": "phishing"
    }
]


async def test_strategies():
    """Test different analysis strategies"""
    strategies = [
        AnalysisStrategy.WEIGHTED_CONSENSUS,
        AnalysisStrategy.CONSERVATIVE,
        AnalysisStrategy.MAJORITY_VOTE
    ]
    
    results = {}
    
    for strategy in strategies:
        print(f"\nTesting strategy: {strategy.value}")
        print("=" * 50)
        
        workflow = PhishingDetectionWorkflow(strategy=strategy)
        
        try:
            await workflow.connect()
            
            strategy_results = []
            
            for test_email in test_emails:
                result = await workflow.analyze_email(
                    test_email["content"],
                    test_email["subject"],
                    test_email["sender"]
                )
                
                is_correct = (
                    (result.is_phishing and test_email["expected"] == "phishing") or
                    (not result.is_phishing and test_email["expected"] == "legitimate")
                )
                
                strategy_results.append({
                    "email": test_email["name"],
                    "expected": test_email["expected"],
                    "detected": "phishing" if result.is_phishing else "legitimate",
                    "confidence": result.confidence,
                    "correct": is_correct,
                    "indicators": result.final_indicators[:3]
                })
                
                print(f"\n{test_email['name']}:")
                print(f"  Expected: {test_email['expected']}")
                print(f"  Detected: {'phishing' if result.is_phishing else 'legitimate'}")
                print(f"  Confidence: {result.confidence:.2%}")
                print(f"  Result: {'✓ Correct' if is_correct else '✗ Incorrect'}")
                
            accuracy = sum(r["correct"] for r in strategy_results) / len(strategy_results)
            results[strategy.value] = {
                "accuracy": accuracy,
                "details": strategy_results
            }
            
            print(f"\nOverall accuracy for {strategy.value}: {accuracy:.2%}")
            
        finally:
            await workflow.disconnect()
    
    return results


async def test_edge_cases():
    """Test edge cases and conflicting analyses"""
    workflow = PhishingDetectionWorkflow()
    
    edge_cases = [
        {
            "name": "Mixed Signals",
            "sender": "newsletter@paypal.com",  # Legitimate domain
            "subject": "PayPal Newsletter",  # Normal subject
            "content": """URGENT: Click here to verify your account: http://bit.ly/verify

Just kidding! This is your monthly security newsletter.
Never click suspicious links!""",  # Mixed content
        },
        {
            "name": "Empty Email",
            "sender": "test@example.com",
            "subject": "",
            "content": ""
        }
    ]
    
    try:
        await workflow.connect()
        
        for test_case in edge_cases:
            print(f"\nTesting edge case: {test_case['name']}")
            
            result = await workflow.analyze_email(
                test_case["content"],
                test_case["subject"],
                test_case["sender"]
            )
            
            print(f"Result: {'PHISHING' if result.is_phishing else 'LEGITIMATE'}")
            print(f"Confidence: {result.confidence:.2%}")
            print(f"Consensus: {result.consensus_details}")
            
    finally:
        await workflow.disconnect()


async def main():
    """Run all tests"""
    print("Phishing Detection Workflow Test Suite")
    print("=" * 60)
    
    # Test different strategies
    print("\n1. Testing Different Strategies")
    strategy_results = await test_strategies()
    
    # Test edge cases
    print("\n\n2. Testing Edge Cases")
    await test_edge_cases()
    
    # Summary
    print("\n\n3. Summary")
    print("=" * 60)
    
    for strategy, results in strategy_results.items():
        print(f"{strategy}: {results['accuracy']:.2%} accuracy")
    
    # Find best strategy
    best_strategy = max(strategy_results.items(), key=lambda x: x[1]["accuracy"])
    print(f"\nBest performing strategy: {best_strategy[0]} ({best_strategy[1]['accuracy']:.2%})")


if __name__ == "__main__":
    asyncio.run(main())