#!/usr/bin/env python3
"""
Demo of RL Enhanced Orchestrator functionality
"""

import asyncio
import sys
from pathlib import Path

sys.path.append('/app')

from inbox_sentinel.orchestration.rl_orchestrator import RLEnhancedOrchestrator
from inbox_sentinel.orchestration import SimpleOrchestrator
from inbox_sentinel.utils.email_parser import parse_gmail_forward

async def demo_rl_orchestrator():
    print("🚀 RL Enhanced Orchestrator Demo")
    print("=" * 50)
    
    # Initialize both orchestrators
    rl_orchestrator = RLEnhancedOrchestrator()
    baseline_orchestrator = SimpleOrchestrator()
    
    await rl_orchestrator.initialize()
    
    # Load a few test emails
    test_emails = []
    
    # Load 3 legitimate emails
    legitimate_dir = Path("./legitimate")
    for i, email_file in enumerate(legitimate_dir.glob("*.txt")):
        if i >= 3:
            break
        with open(email_file, 'r') as f:
            email = parse_gmail_forward(f.read())
            test_emails.append((email, False, f"Legitimate #{i+1}"))
    
    # Load 3 spam emails  
    spam_dir = Path("./spam")
    for i, email_file in enumerate(spam_dir.glob("*.txt")):
        if i >= 3:
            break
        with open(email_file, 'r') as f:
            email = parse_gmail_forward(f.read())
            test_emails.append((email, True, f"Spam #{i+1}"))
    
    print(f"📊 Testing with {len(test_emails)} sample emails\n")
    
    # Compare predictions
    for email, true_label, description in test_emails:
        print(f"📧 {description}")
        print(f"   Subject: {email.subject[:60]}...")
        print(f"   True label: {'SPAM' if true_label else 'HAM'}")
        
        # Get baseline prediction
        baseline_result = await baseline_orchestrator.analyze_email(email)
        baseline_prediction = "ERROR"
        if baseline_result['success']:
            detailed = baseline_result.get('detailed_results', {})
            spam_votes = sum(1 for r in detailed.values() if r.get('is_spam', False) and 'error' not in r)
            total_votes = sum(1 for r in detailed.values() if 'error' not in r)
            baseline_prediction = "SPAM" if spam_votes > (total_votes / 2) else "HAM"
        
        # Get RL prediction
        rl_result = await rl_orchestrator.predict(email, training=False)
        rl_prediction = "SPAM" if rl_result.get('is_spam', False) else "HAM"
        rl_confidence = rl_result.get('confidence', 0.0)
        
        # Show state information
        rl_state = rl_result.get('rl_state')
        if rl_state:
            print(f"   📊 RL State Analysis:")
            print(f"      Model consensus: {rl_state.spam_votes}/5 voted spam")
            print(f"      Agreement level: {rl_state.agreement_level:.3f}")
            print(f"      Average confidence: {rl_state.avg_confidence:.3f}")
            print(f"      Email features: length={rl_state.content_length}, urls={rl_state.has_urls}, money={rl_state.has_money_terms}")
        
        # Compare results
        baseline_correct = baseline_prediction == ('SPAM' if true_label else 'HAM')
        rl_correct = rl_prediction == ('SPAM' if true_label else 'HAM')
        
        print(f"   🎯 Predictions:")
        print(f"      Baseline:     {baseline_prediction} {'✅' if baseline_correct else '❌'}")
        print(f"      RL Enhanced:  {rl_prediction} {'✅' if rl_correct else '❌'} (conf: {rl_confidence:.3f})")
        
        print()
    
    # Demonstrate training capability
    print("🎓 Training Demonstration")
    print("-" * 30)
    print("The RL orchestrator can be trained using the training script:")
    print("   docker compose exec inbox-sentinel bash -c \\")
    print("   \"cd /app/data/test_samples && python train_rl_orchestrator_fixed.py\"")
    print()
    print("Key RL Features:")
    print("✨ Dynamic model weighting based on performance")
    print("✨ Context-aware decisions using email features") 
    print("✨ Adaptive thresholds learned from training data")
    print("✨ Q-learning for optimal ensemble strategies")
    print("✨ State representation with 24 features")
    print()
    
    # Show training statistics if available
    training_stats = rl_orchestrator.get_training_stats()
    print("📈 Current Training Status:")
    print(f"   Episodes trained: {training_stats['rl_agent_stats']['episodes']}")
    print(f"   Q-table entries: {training_stats['q_table_size']}")
    print(f"   Exploration rate: {training_stats['current_epsilon']:.3f}")
    
    print("\n✅ Demo completed!")

if __name__ == "__main__":
    asyncio.run(demo_rl_orchestrator())