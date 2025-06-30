#!/usr/bin/env python3
"""
Test MCP server model loading
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from inbox_sentinel.ml.models import NeuralNetworkDetector
from inbox_sentinel.core.types import Email


async def test_model_loading():
    """Test if model can be loaded"""
    print("Testing model loading...")
    
    # Create detector
    detector = NeuralNetworkDetector()
    
    # Initialize (should load pkl file)
    print("\nInitializing detector...")
    result = await detector.initialize(use_pretrained=True)
    print(f"Initialize result: {result}")
    print(f"Is trained: {detector.is_trained}")
    
    if detector.is_trained:
        # Test analysis
        print("\nTesting analysis...")
        email = Email(
            content="Win $1000000 now! Click here to claim your prize!",
            subject="You're a WINNER!",
            sender="lottery@scam.com"
        )
        
        try:
            prediction = await detector.analyze(email)
            print(f"\nPrediction results:")
            print(f"  Is spam: {prediction.is_spam}")
            print(f"  Confidence: {prediction.confidence:.2%}")
            print(f"  Spam probability: {prediction.spam_probability:.2%}")
        except Exception as e:
            print(f"Analysis error: {e}")
    else:
        print("\nModel not trained, cannot test analysis")


if __name__ == "__main__":
    asyncio.run(test_model_loading())