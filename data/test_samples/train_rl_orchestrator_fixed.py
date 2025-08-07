#!/usr/bin/env python3
"""
Training script for RL Enhanced Orchestrator - Fixed version
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


class RLTrainingManager:
    """Manager for RL orchestrator training"""
    
    def __init__(self):
        self.rl_orchestrator = RLEnhancedOrchestrator()
        self.baseline_orchestrator = SimpleOrchestrator()
        
    def load_training_data(self, max_samples_per_class=100) -> Tuple[List[Tuple[Email, bool]], List[Tuple[Email, bool]]]:
        """Load and prepare training data with limits for faster training"""
        print("📚 Loading training data...")
        
        legitimate_samples = []
        spam_samples = []
        
        # Load legitimate emails (limited)
        legitimate_dir = Path("./legitimate")
        print(f"Loading up to {max_samples_per_class} legitimate emails...")
        
        for i, email_file in enumerate(legitimate_dir.glob("*.txt")):
            if i >= max_samples_per_class:
                break
                
            if i % 25 == 0:
                print(f"  Loaded {i} legitimate emails...")
            
            try:
                with open(email_file, 'r', encoding='utf-8') as f:
                    email_text = f.read()
                    email = parse_gmail_forward(email_text)
                    legitimate_samples.append((email, False))  # False = not spam
            except Exception as e:
                print(f"Error loading {email_file}: {e}")
        
        # Load spam emails (limited)  
        spam_dir = Path("./spam")
        print(f"Loading up to {max_samples_per_class} spam emails...")
        
        for i, email_file in enumerate(spam_dir.glob("*.txt")):
            if i >= max_samples_per_class:
                break
                
            if i % 25 == 0:
                print(f"  Loaded {i} spam emails...")
            
            try:
                with open(email_file, 'r', encoding='utf-8') as f:
                    email_text = f.read()
                    email = parse_gmail_forward(email_text)
                    spam_samples.append((email, True))  # True = spam
            except Exception as e:
                print(f"Error loading {email_file}: {e}")
        
        print(f"✅ Loaded {len(legitimate_samples)} legitimate and {len(spam_samples)} spam samples")
        
        # Combine and shuffle
        all_samples = legitimate_samples + spam_samples
        random.shuffle(all_samples)
        
        # Split into training and test sets
        split_idx = int(len(all_samples) * 0.7)  # 70% training, 30% test
        training_data = all_samples[:split_idx]
        test_data = all_samples[split_idx:]
        
        print(f"📊 Training set: {len(training_data)} samples")
        print(f"📊 Test set: {len(test_data)} samples")
        
        return training_data, test_data
    
    async def evaluate_baseline_orchestrator(self, test_data: List[Tuple[Email, bool]]) -> Dict:
        """Evaluate current simple orchestrator for baseline"""
        print("📈 Evaluating baseline orchestrator...")
        
        correct_predictions = 0
        total_predictions = 0
        
        for email, true_label in test_data:
            try:
                result = await self.baseline_orchestrator.analyze_email(email)
                if result['success']:
                    detailed = result.get('detailed_results', {})
                    spam_votes = sum(1 for r in detailed.values() if r.get('is_spam', False) and 'error' not in r)
                    total_votes = sum(1 for r in detailed.values() if 'error' not in r)
                    predicted_spam = spam_votes > (total_votes / 2) if total_votes > 0 else False
                    
                    if predicted_spam == true_label:
                        correct_predictions += 1
                    total_predictions += 1
            except Exception as e:
                print(f"Error in baseline evaluation: {e}")
        
        baseline_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"📊 Baseline orchestrator accuracy: {baseline_accuracy:.3f} ({correct_predictions}/{total_predictions})")
        
        return {
            'accuracy': baseline_accuracy,
            'correct': correct_predictions,
            'total': total_predictions
        }
    
    async def train_rl_orchestrator(self, training_data: List[Tuple[Email, bool]], 
                                  episodes: int = 25):
        """Train the RL orchestrator"""
        print(f"🚀 Training RL orchestrator for {episodes} episodes...")
        
        # Initialize orchestrator
        await self.rl_orchestrator.initialize()
        
        # Train the model with lower validation split for more training data
        await self.rl_orchestrator.train_on_dataset(
            training_data, 
            episodes=episodes, 
            validation_split=0.1  # Use 90% for training, 10% for validation
        )
        
        # Save the trained model
        model_path = "./rl_orchestrator_trained.pkl"
        self.rl_orchestrator.save_model(model_path)
        
        return model_path
    
    async def evaluate_rl_orchestrator(self, test_data: List[Tuple[Email, bool]]) -> Dict:
        """Evaluate trained RL orchestrator"""
        print("📈 Evaluating RL orchestrator...")
        
        correct_predictions = 0
        total_predictions = 0
        predictions_detail = []
        
        for email, true_label in test_data:
            try:
                result = await self.rl_orchestrator.predict(email, training=False)
                if result['success']:
                    predicted_spam = result['is_spam']
                    confidence = result['confidence']
                    
                    is_correct = predicted_spam == true_label
                    if is_correct:
                        correct_predictions += 1
                    
                    predictions_detail.append({
                        'predicted_spam': predicted_spam,
                        'true_label': true_label,
                        'correct': is_correct,
                        'confidence': confidence
                    })
                    
                    total_predictions += 1
            except Exception as e:
                print(f"Error in RL evaluation: {e}")
        
        rl_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # Calculate additional metrics
        tp = sum(1 for p in predictions_detail if p['predicted_spam'] and p['true_label'])
        tn = sum(1 for p in predictions_detail if not p['predicted_spam'] and not p['true_label'])
        fp = sum(1 for p in predictions_detail if p['predicted_spam'] and not p['true_label'])
        fn = sum(1 for p in predictions_detail if not p['predicted_spam'] and p['true_label'])
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"📊 RL orchestrator accuracy: {rl_accuracy:.3f} ({correct_predictions}/{total_predictions})")
        print(f"📊 RL orchestrator precision: {precision:.3f}")
        print(f"📊 RL orchestrator recall: {recall:.3f}")
        print(f"📊 RL orchestrator F1-score: {f1_score:.3f}")
        
        return {
            'accuracy': rl_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'correct': correct_predictions,
            'total': total_predictions,
            'confusion_matrix': {
                'true_positive': tp,
                'true_negative': tn,
                'false_positive': fp,
                'false_negative': fn
            }
        }
    
    async def run_comparison(self, max_samples_per_class=100, episodes=25):
        """Run complete training and evaluation pipeline"""
        print("🎯 RL vs Baseline Orchestrator Comparison")
        print(f"Max samples per class: {max_samples_per_class}")
        print(f"Training episodes: {episodes}")
        print("=" * 60)
        
        # Load data
        training_data, test_data = self.load_training_data(max_samples_per_class)
        
        # Evaluate baseline first
        baseline_results = await self.evaluate_baseline_orchestrator(test_data)
        
        # Train RL orchestrator
        model_path = await self.train_rl_orchestrator(training_data, episodes=episodes)
        
        # Evaluate RL orchestrator
        rl_results = await self.evaluate_rl_orchestrator(test_data)
        
        # Generate comparison report
        print("\\n" + "=" * 60)
        print("📊 FINAL RESULTS COMPARISON")
        print("=" * 60)
        
        print(f"\\n🎯 Baseline Simple Orchestrator:")
        print(f"   Accuracy: {baseline_results['accuracy']:.3f}")
        
        print(f"\\n🤖 RL Enhanced Orchestrator:")
        print(f"   Accuracy:  {rl_results['accuracy']:.3f}")
        print(f"   Precision: {rl_results['precision']:.3f}")
        print(f"   Recall:    {rl_results['recall']:.3f}")
        print(f"   F1-Score:  {rl_results['f1_score']:.3f}")
        
        # Calculate improvements
        accuracy_improvement = rl_results['accuracy'] - baseline_results['accuracy']
        improvement_percent = (accuracy_improvement / baseline_results['accuracy']) * 100 if baseline_results['accuracy'] > 0 else 0
        
        print(f"\\n📈 IMPROVEMENT:")
        print(f"   Accuracy improvement: {accuracy_improvement:+.3f} ({improvement_percent:+.1f}%)")
        
        if accuracy_improvement > 0.02:  # Improvement threshold of 2%
            print(f"   🎉 Significant improvement! RL orchestrator is better by {accuracy_improvement:.3f}")
        elif accuracy_improvement > 0:
            print(f"   ✅ Modest improvement. RL orchestrator is slightly better.")
        else:
            print(f"   ⚠️ RL orchestrator needs more training or different hyperparameters")
        
        # Get training statistics
        training_stats = self.rl_orchestrator.get_training_stats()
        print(f"\\n🔧 RL Training Details:")
        print(f"   Episodes completed: {training_stats['rl_agent_stats']['episodes']}")
        print(f"   Q-table size: {training_stats['q_table_size']}")
        print(f"   Final exploration rate: {training_stats['current_epsilon']:.3f}")
        
        # Save results
        results_data = {
            'test_configuration': {
                'max_samples_per_class': max_samples_per_class,
                'training_episodes': episodes,
                'test_samples': len(test_data)
            },
            'baseline': baseline_results,
            'rl_enhanced': rl_results,
            'improvement': {
                'accuracy_delta': accuracy_improvement,
                'improvement_percent': improvement_percent
            },
            'training_stats': training_stats,
            'model_path': model_path
        }
        
        with open('rl_comparison_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\\n💾 Results saved to 'rl_comparison_results.json'")
        print(f"💾 Trained RL model saved to '{model_path}'")
        print("\\n✅ Comparison complete!")
        
        return results_data


async def main():
    """Main execution"""
    # Configuration - reasonable values for testing
    MAX_SAMPLES_PER_CLASS = 100  # 100 legitimate + 100 spam = 200 total
    TRAINING_EPISODES = 30       # Moderate training
    
    print("🚀 RL Enhanced Orchestrator Comparison Study")
    print(f"Dataset size: {MAX_SAMPLES_PER_CLASS * 2} emails total")
    print(f"Training episodes: {TRAINING_EPISODES}")
    print("=" * 60)
    
    # Create trainer and run comparison
    trainer = RLTrainingManager()
    
    start_time = time.time()
    results = await trainer.run_comparison(
        max_samples_per_class=MAX_SAMPLES_PER_CLASS,
        episodes=TRAINING_EPISODES
    )
    end_time = time.time()
    
    print(f"\\n⏱️ Total execution time: {end_time - start_time:.1f} seconds")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())