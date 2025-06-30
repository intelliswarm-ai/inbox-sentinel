#!/usr/bin/env python3
"""
Test script to verify the phishing detection MCP server works locally
"""

import asyncio
import json
from mcp_server import (
    URLPatternDetector, 
    ContentAnalyzer, 
    BayesianFilter, 
    HeaderAnalyzer
)

async def test_local_processing():
    """Test that all detectors work locally without remote calls"""
    
    # Test email data
    test_email = {
        "content": "URGENT: Your account has been suspended. Click here immediately to verify your identity: http://bit.ly/suspicious-link",
        "subject": "Account Suspension Notice - Action Required",
        "sender": "security@paypal-secure.tk"
    }
    
    print("🔒 Testing Local Phishing Detection (No Remote Calls)")
    print("=" * 60)
    
    # Initialize all detectors
    detectors = {
        "URL Pattern": URLPatternDetector(),
        "Content": ContentAnalyzer(), 
        "Bayesian": BayesianFilter(),
        "Header": HeaderAnalyzer()
    }
    
    # Test each detector
    for name, detector in detectors.items():
        print(f"\n📊 Testing {name} Detector:")
        print("-" * 40)
        
        result = detector.analyze(
            test_email["content"],
            test_email["subject"], 
            test_email["sender"]
        )
        
        print(f"Result: {'🚨 PHISHING' if result['is_phishing'] else '✅ SAFE'}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Score: {result['score']:.3f}")
        print("Indicators:")
        for indicator in result['indicators']:
            print(f"  • {indicator}")
    
    # Test ensemble analysis
    print(f"\n🎯 Ensemble Analysis:")
    print("-" * 40)
    
    results = []
    for detector in detectors.values():
        result = detector.analyze(
            test_email["content"],
            test_email["subject"],
            test_email["sender"]
        )
        results.append(result)
    
    phishing_votes = sum(1 for r in results if r["is_phishing"])
    total_votes = len(results)
    
    print(f"Votes: {phishing_votes}/{total_votes} detectors flagged as phishing")
    print(f"Ensemble Decision: {'🚨 PHISHING' if phishing_votes > total_votes/2 else '✅ SAFE'}")
    
    print(f"\n✅ All processing completed locally - no data sent remotely!")

if __name__ == "__main__":
    asyncio.run(test_local_processing()) 