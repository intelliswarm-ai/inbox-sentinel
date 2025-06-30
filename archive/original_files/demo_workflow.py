#!/usr/bin/env python3
"""
Demonstration of the phishing detection workflow concept
This simulates the MCP tool calls without requiring actual MCP connection
"""

import asyncio
import json
from typing import Dict, Any, List
from dataclasses import dataclass
# We'll use our own classes for the demo


# Simulate the detector responses
def simulate_url_pattern_detector(email_content: str, subject: str, sender: str) -> Dict[str, Any]:
    """Simulate URL Pattern Detector response"""
    indicators = []
    score = 0.0
    
    if 'bit.ly' in email_content or 'tinyurl' in email_content:
        score += 0.3
        indicators.append("URL shortener detected")
    
    if '@' in sender and any(char.isdigit() for char in sender.split('@')[1]):
        score += 0.2
        indicators.append("Sender domain contains numbers")
        
    return {
        "detector": "URL Pattern Detector",
        "is_phishing": score > 0.3,
        "confidence": 0.85,
        "score": score,
        "indicators": indicators
    }


def simulate_content_analyzer(email_content: str, subject: str, sender: str) -> Dict[str, Any]:
    """Simulate Content Analyzer response"""
    indicators = []
    score = 0.0
    
    content_lower = (email_content + " " + subject).lower()
    phishing_keywords = ["urgent", "verify", "suspended", "click here", "congratulations"]
    
    for keyword in phishing_keywords:
        if keyword in content_lower:
            score += 0.15
            indicators.append(f"Phishing keyword: {keyword}")
            
    return {
        "detector": "Content Analyzer",
        "is_phishing": score > 0.3,
        "confidence": 0.88,
        "score": score,
        "indicators": indicators
    }


def simulate_bayesian_filter(email_content: str, subject: str, sender: str) -> Dict[str, Any]:
    """Simulate Bayesian Filter response"""
    indicators = []
    score = 0.5  # Start neutral
    
    # Simple token analysis
    tokens = (email_content + " " + subject).lower().split()
    high_risk_tokens = ["verify", "suspended", "urgent", "winner", "congratulations"]
    
    for token in tokens:
        if token in high_risk_tokens:
            score += 0.1
            indicators.append(f"High-risk token: {token}")
            
    score = min(1.0, score)
    
    return {
        "detector": "Bayesian Filter",
        "is_phishing": score > 0.6,
        "confidence": 0.92,
        "score": score,
        "indicators": indicators
    }


def simulate_header_analyzer(email_content: str, subject: str, sender: str) -> Dict[str, Any]:
    """Simulate Header Analyzer response"""
    indicators = []
    score = 0.0
    
    if '@' in sender:
        domain = sender.split('@')[1]
        # Check for lookalike domains
        if 'paypal' in domain and domain != 'paypal.com':
            score += 0.4
            indicators.append(f"Lookalike domain: {domain}")
        elif 'amazon' in domain and domain != 'amazon.com':
            score += 0.4
            indicators.append(f"Lookalike domain: {domain}")
            
    return {
        "detector": "Header Analyzer",
        "is_phishing": score > 0.3,
        "confidence": 0.90,
        "score": score,
        "indicators": indicators
    }


def analyze_with_ensemble(email_content: str, subject: str, sender: str) -> Dict[str, Any]:
    """Run all detectors and provide ensemble analysis"""
    results = [
        simulate_url_pattern_detector(email_content, subject, sender),
        simulate_content_analyzer(email_content, subject, sender),
        simulate_bayesian_filter(email_content, subject, sender),
        simulate_header_analyzer(email_content, subject, sender)
    ]
    
    # Calculate ensemble decision
    phishing_votes = sum(1 for r in results if r["is_phishing"])
    total_votes = len(results)
    
    # Weighted voting based on confidence
    weighted_score = sum(
        r["score"] * r["confidence"] for r in results
    ) / sum(r["confidence"] for r in results)
    
    return {
        "ensemble_result": {
            "is_phishing": phishing_votes > total_votes / 2,
            "confidence": sum(r["confidence"] for r in results) / total_votes,
            "weighted_score": weighted_score,
            "votes": f"{phishing_votes}/{total_votes}",
            "consensus_level": "high" if phishing_votes == 0 or phishing_votes == total_votes else "low"
        },
        "individual_results": results
    }


def demonstrate_workflow():
    """Demonstrate the phishing detection workflow"""
    
    print("AI Agentic Phishing Detection Workflow Demonstration")
    print("=" * 60)
    
    # Test emails
    test_cases = [
        {
            "name": "Classic Phishing Email",
            "sender": "security@paypaI.com",  # Note the capital I
            "subject": "Urgent: Verify Your Account",
            "content": "Your account will be suspended! Click here immediately: http://bit.ly/verify-account"
        },
        {
            "name": "Legitimate Email",
            "sender": "newsletter@company.com",
            "subject": "Monthly Newsletter",
            "content": "Here's your monthly update with our latest blog posts and company news."
        },
        {
            "name": "Sophisticated Phishing",
            "sender": "no-reply@amaz0n-delivery.com",
            "subject": "Your order could not be delivered",
            "content": "Order #7823984729 delivery failed. Verify your address at http://amaz0n-delivery.com/verify"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n\nAnalyzing: {test_case['name']}")
        print("-" * 40)
        print(f"From: {test_case['sender']}")
        print(f"Subject: {test_case['subject']}")
        print(f"Preview: {test_case['content'][:60]}...")
        
        # Run ensemble analysis
        result = analyze_with_ensemble(
            test_case["content"],
            test_case["subject"],
            test_case["sender"]
        )
        
        ensemble = result["ensemble_result"]
        individual = result["individual_results"]
        
        print(f"\n📊 Analysis Results:")
        print(f"Final Decision: {'🚨 PHISHING' if ensemble['is_phishing'] else '✅ LEGITIMATE'}")
        print(f"Confidence: {ensemble['confidence']:.1%}")
        print(f"Consensus: {ensemble['consensus_level'].upper()} ({ensemble['votes']} detectors)")
        
        # Show individual detector results
        print(f"\n🔍 Individual Detector Results:")
        for detector_result in individual:
            status = "🔴" if detector_result["is_phishing"] else "🟢"
            print(f"  {status} {detector_result['detector']}: "
                  f"{'Phishing' if detector_result['is_phishing'] else 'Legitimate'} "
                  f"(confidence: {detector_result['confidence']:.0%})")
            
        # Show top indicators
        all_indicators = []
        for r in individual:
            all_indicators.extend(r.get("indicators", []))
        
        if all_indicators:
            print(f"\n⚠️  Key Indicators Found:")
            for indicator in set(all_indicators[:5]):
                print(f"  • {indicator}")
                
        # Demonstrate adaptive behavior
        if ensemble["consensus_level"] == "low":
            print(f"\n🔄 Adaptive Workflow Triggered:")
            print("  Low consensus detected - would switch to targeted re-analysis")
            print("  of conflicting indicators using specialized detectors")


def show_workflow_diagram():
    """Display the workflow architecture"""
    print("\n\n" + "="*60)
    print("Phishing Detection Workflow Architecture")
    print("="*60)
    print("""
    ┌─────────────────┐
    │  Email Input    │
    │ (sender, subject│
    │  content)       │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐     MCP Server
    │ Workflow Agent  │ ◄──────────────┐
    │  Orchestrator   │                │
    └────────┬────────┘                │
             │                         │
             ▼                         ▼
    ┌─────────────────────────────────────────┐
    │         Ensemble Analysis                │
    │  ┌─────────┐ ┌─────────┐ ┌─────────┐   │
    │  │   URL   │ │Content  │ │Bayesian │   │
    │  │Pattern  │ │Analyzer │ │ Filter  │   │
    │  │Detector │ │         │ │         │   │
    │  └─────────┘ └─────────┘ └─────────┘   │
    │  ┌─────────┐                            │
    │  │ Header  │   All detectors run       │
    │  │Analyzer │   in parallel             │
    │  └─────────┘                            │
    └──────────────┬──────────────────────────┘
                   │
                   ▼
         ┌─────────────────┐
         │ Consensus Check │
         └────────┬────────┘
                  │
        High ◄────┴────► Low
        consensus      consensus
           │              │
           ▼              ▼
    ┌──────────┐   ┌─────────────┐
    │  Apply   │   │  Adaptive   │
    │ Strategy │   │  Analysis   │
    └─────┬────┘   └──────┬──────┘
          │               │
          └───────┬───────┘
                  ▼
         ┌────────────────┐
         │ Final Decision │
         │   + Report     │
         └────────────────┘
    
    Strategies Available:
    • Weighted Consensus (default)
    • Majority Vote
    • Conservative (any detector flags = phishing)
    • Aggressive (all must agree = phishing)
    • High Confidence First
    """)


if __name__ == "__main__":
    demonstrate_workflow()
    show_workflow_diagram()