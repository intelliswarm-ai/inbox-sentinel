#!/usr/bin/env python3
"""
Quick test for RL Enhanced Orchestrator
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.append('/app')

from inbox_sentinel.orchestration.rl_orchestrator import RLEnhancedOrchestrator
from inbox_sentinel.utils.email_parser import parse_gmail_forward
from inbox_sentinel.core.types import Email

async def quick_test():
    print("🚀 Quick RL Orchestrator Test")
    print("=" * 40)
    
    # Initialize orchestrator
    orchestrator = RLEnhancedOrchestrator()
    await orchestrator.initialize()
    
    # Load a few sample emails
    samples = []
    
    # Load 10 legitimate emails
    legitimate_dir = Path("./legitimate")
    for i, email_file in enumerate(legitimate_dir.glob("*.txt")):
        if i >= 10:
            break
        try:
            with open(email_file, 'r', encoding='utf-8') as f:
                email_text = f.read()
                email = parse_gmail_forward(email_text)
                samples.append((email, False))  # False = not spam
        except Exception as e:
            print(f"Error loading {email_file}: {e}")
    
    # Load 10 spam emails
    spam_dir = Path("./spam")
    for i, email_file in enumerate(spam_dir.glob("*.txt")):
        if i >= 10:
            break
        try:
            with open(email_file, 'r', encoding='utf-8') as f:
                email_text = f.read()
                email = parse_gmail_forward(email_text)
                samples.append((email, True))  # True = spam
        except Exception as e:
            print(f"Error loading {email_file}: {e}")
    
    print(f"📊 Loaded {len(samples)} test samples")
    
    # Test prediction before training
    print("\n📈 Testing predictions before training...")
    correct_before = 0
    for email, true_label in samples[:5]:
        result = await orchestrator.predict(email, training=False)
        if result['success']:
            predicted = result['is_spam']
            if predicted == true_label:
                correct_before += 1
            print(f"  Email: {true_label} -> {predicted} {'✅' if predicted == true_label else '❌'}")
    
    accuracy_before = correct_before / 5
    print(f"  Accuracy before training: {accuracy_before:.3f}")
    
    # Quick training (5 episodes)
    print("\n🎯 Quick training (5 episodes)...")
    start_time = time.time()
    await orchestrator.train_on_dataset(samples, episodes=5, validation_split=0.3)
    training_time = time.time() - start_time
    print(f"  Training completed in {training_time:.1f} seconds")
    
    # Test prediction after training
    print("\n📈 Testing predictions after training...")
    correct_after = 0
    for email, true_label in samples[:5]:
        result = await orchestrator.predict(email, training=False)
        if result['success']:
            predicted = result['is_spam']
            if predicted == true_label:
                correct_after += 1
            print(f"  Email: {true_label} -> {predicted} {'✅' if predicted == true_label else '❌'}")
    
    accuracy_after = correct_after / 5
    print(f"  Accuracy after training: {accuracy_after:.3f}")
    
    # Show improvement
    improvement = accuracy_after - accuracy_before
    print(f"\n🏆 Improvement: {improvement:+.3f}")
    
    # Show training stats
    stats = orchestrator.get_training_stats()
    print(f"\n📊 Training Stats:")
    print(f"  Episodes trained: {stats['rl_agent_stats']['episodes']}")
    print(f"  Q-table size: {stats['q_table_size']}")
    print(f"  Final epsilon: {stats['current_epsilon']:.3f}")
    
    print("\n✅ Quick test completed!")

if __name__ == "__main__":
    asyncio.run(quick_test())