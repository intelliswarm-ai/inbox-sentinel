#!/usr/bin/env python3
"""
Quick RL orchestrator training with reasonable parameters
"""

import asyncio
import sys
import time
import random
from pathlib import Path
import json
from typing import List, Tuple, Dict

# Add parent directory to path
sys.path.append('/app')

from inbox_sentinel.orchestration.rl_orchestrator import RLEnhancedOrchestrator
from inbox_sentinel.orchestration import SimpleOrchestrator
from inbox_sentinel.utils.email_parser import parse_gmail_forward
from inbox_sentinel.core.types import Email

async def quick_rl_training():
    print("🚀 RL Orchestrator Training - Quick Demo")
    print("=" * 50)
    
    # Configuration for reasonable training time
    MAX_SAMPLES_PER_CLASS = 50  # 50 legitimate + 50 spam = 100 total
    TRAINING_EPISODES = 20      # 20 episodes for quick demo
    
    print(f"📊 Dataset: {MAX_SAMPLES_PER_CLASS * 2} emails total")
    print(f"🎯 Training episodes: {TRAINING_EPISODES}")
    print(f"⏱️ Expected time: ~3-5 minutes\n")
    
    # Load training data
    print("📚 Loading training data...")
    training_data = []
    test_data = []
    
    # Load legitimate emails
    legitimate_samples = []
    legitimate_dir = Path("./legitimate")
    for i, email_file in enumerate(legitimate_dir.glob("*.txt")):
        if i >= MAX_SAMPLES_PER_CLASS:
            break
        if i % 10 == 0:
            print(f"  Loading legitimate email {i+1}...")
        try:
            with open(email_file, 'r', encoding='utf-8') as f:
                email = parse_gmail_forward(f.read())
                legitimate_samples.append((email, False))
        except Exception as e:
            print(f"Error loading {email_file}: {e}")
    
    # Load spam emails
    spam_samples = []
    spam_dir = Path("./spam")
    for i, email_file in enumerate(spam_dir.glob("*.txt")):
        if i >= MAX_SAMPLES_PER_CLASS:
            break
        if i % 10 == 0:
            print(f"  Loading spam email {i+1}...")
        try:
            with open(email_file, 'r', encoding='utf-8') as f:
                email = parse_gmail_forward(f.read())
                spam_samples.append((email, True))
        except Exception as e:
            print(f"Error loading {email_file}: {e}")
    
    # Combine and split
    all_samples = legitimate_samples + spam_samples
    random.shuffle(all_samples)
    
    split_idx = int(len(all_samples) * 0.7)  # 70% training, 30% test
    training_data = all_samples[:split_idx]
    test_data = all_samples[split_idx:]
    
    print(f"✅ Loaded {len(training_data)} training, {len(test_data)} test samples")
    
    # Initialize orchestrators
    print("\n🤖 Initializing orchestrators...")
    rl_orchestrator = RLEnhancedOrchestrator()
    baseline_orchestrator = SimpleOrchestrator()
    
    await rl_orchestrator.initialize()
    
    # Evaluate baseline first
    print("\n📊 Evaluating baseline orchestrator...")
    baseline_correct = 0
    baseline_total = 0
    
    for email, true_label in test_data:
        try:
            result = await baseline_orchestrator.analyze_email(email)
            if result['success']:
                detailed = result.get('detailed_results', {})
                spam_votes = sum(1 for r in detailed.values() if r.get('is_spam', False) and 'error' not in r)
                total_votes = sum(1 for r in detailed.values() if 'error' not in r)
                predicted_spam = spam_votes > (total_votes / 2) if total_votes > 0 else False
                
                if predicted_spam == true_label:
                    baseline_correct += 1
                baseline_total += 1
        except Exception as e:
            print(f"Baseline evaluation error: {e}")
    
    baseline_accuracy = baseline_correct / baseline_total if baseline_total > 0 else 0
    print(f"📈 Baseline accuracy: {baseline_accuracy:.3f} ({baseline_correct}/{baseline_total})")
    
    # Train RL orchestrator
    print(f"\n🎓 Training RL orchestrator for {TRAINING_EPISODES} episodes...")
    training_start = time.time()
    
    await rl_orchestrator.train_on_dataset(
        training_data, 
        episodes=TRAINING_EPISODES, 
        validation_split=0.2
    )
    
    training_time = time.time() - training_start
    print(f"⏱️ Training completed in {training_time:.1f} seconds")
    
    # Evaluate RL orchestrator
    print("\n📊 Evaluating RL orchestrator...")
    rl_correct = 0
    rl_total = 0
    predictions_detail = []
    
    for email, true_label in test_data:
        try:
            result = await rl_orchestrator.predict(email, training=False)
            if result['success']:
                predicted_spam = result['is_spam']
                confidence = result['confidence']
                
                is_correct = predicted_spam == true_label
                if is_correct:
                    rl_correct += 1
                
                predictions_detail.append({
                    'predicted_spam': predicted_spam,
                    'true_label': true_label,
                    'correct': is_correct,
                    'confidence': confidence
                })
                
                rl_total += 1
        except Exception as e:
            print(f"RL evaluation error: {e}")
    
    rl_accuracy = rl_correct / rl_total if rl_total > 0 else 0
    
    # Calculate metrics
    tp = sum(1 for p in predictions_detail if p['predicted_spam'] and p['true_label'])
    tn = sum(1 for p in predictions_detail if not p['predicted_spam'] and not p['true_label'])
    fp = sum(1 for p in predictions_detail if p['predicted_spam'] and not p['true_label'])
    fn = sum(1 for p in predictions_detail if not p['predicted_spam'] and p['true_label'])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"📈 RL accuracy: {rl_accuracy:.3f} ({rl_correct}/{rl_total})")
    print(f"📈 RL precision: {precision:.3f}")
    print(f"📈 RL recall: {recall:.3f}")
    print(f"📈 RL F1-score: {f1_score:.3f}")
    
    # Save the trained model
    model_path = "/app/data/models/rl_orchestrator_trained.pkl"
    rl_orchestrator.save_model(model_path)
    
    # Show comparison results
    print("\n" + "=" * 50)
    print("🏆 FINAL COMPARISON RESULTS")
    print("=" * 50)
    
    accuracy_improvement = rl_accuracy - baseline_accuracy
    improvement_percent = (accuracy_improvement / baseline_accuracy) * 100 if baseline_accuracy > 0 else 0
    
    print(f"\n📊 Results Summary:")
    print(f"   Baseline Orchestrator:  {baseline_accuracy:.3f}")
    print(f"   RL Enhanced:           {rl_accuracy:.3f}")
    print(f"   Improvement:           {accuracy_improvement:+.3f} ({improvement_percent:+.1f}%)")
    
    if accuracy_improvement > 0.02:
        print(f"\n🎉 SUCCESS! RL orchestrator improved accuracy by {accuracy_improvement:.3f}!")
    elif accuracy_improvement > 0:
        print(f"\n✅ Modest improvement. RL orchestrator learned some patterns.")
    else:
        print(f"\n⚠️ RL orchestrator needs more training or hyperparameter tuning.")
    
    # Training statistics
    training_stats = rl_orchestrator.get_training_stats()
    print(f"\n🔧 RL Training Statistics:")
    print(f"   Episodes completed: {training_stats['rl_agent_stats']['episodes']}")
    print(f"   Q-table entries: {training_stats['q_table_size']}")
    print(f"   Final exploration rate: {training_stats['current_epsilon']:.3f}")
    print(f"   Training time: {training_time:.1f} seconds")
    
    # Show some example predictions
    print(f"\n🔍 Example Predictions:")
    for i, pred in enumerate(predictions_detail[:5]):
        status = "✅" if pred['correct'] else "❌"
        label_str = "SPAM" if pred['true_label'] else "HAM"
        pred_str = "SPAM" if pred['predicted_spam'] else "HAM"
        print(f"   {i+1}. {status} True: {label_str}, Predicted: {pred_str} (conf: {pred['confidence']:.3f})")
    
    # Save results
    results_data = {
        'configuration': {
            'max_samples_per_class': MAX_SAMPLES_PER_CLASS,
            'training_episodes': TRAINING_EPISODES,
            'training_time_seconds': training_time
        },
        'baseline': {
            'accuracy': baseline_accuracy,
            'correct': baseline_correct,
            'total': baseline_total
        },
        'rl_enhanced': {
            'accuracy': rl_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'correct': rl_correct,
            'total': rl_total
        },
        'improvement': {
            'accuracy_delta': accuracy_improvement,
            'improvement_percent': improvement_percent
        },
        'training_stats': training_stats,
        'model_path': model_path
    }
    
    with open('quick_rl_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n💾 Results saved to 'quick_rl_results.json'")
    print(f"💾 Trained RL model saved to '{model_path}'")
    print("\n✅ RL training demonstration complete!")
    
    return results_data

if __name__ == "__main__":
    asyncio.run(quick_rl_training())