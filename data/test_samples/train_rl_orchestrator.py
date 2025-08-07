#!/usr/bin/env python3
"""
Training script for RL Enhanced Orchestrator
"""

import asyncio
import sys
import time
import random
from pathlib import Path
import matplotlib.pyplot as plt
import json
from typing import List, Tuple

# Add parent directory to path
sys.path.append('/app')

from inbox_sentinel.orchestration.rl_orchestrator import RLEnhancedOrchestrator
from inbox_sentinel.utils.email_parser import parse_gmail_forward
from inbox_sentinel.core.types import Email


class RLTrainingManager:
    """Manager for RL orchestrator training"""
    
    def __init__(self):
        self.orchestrator = RLEnhancedOrchestrator()
        self.training_data = []
        self.test_data = []
    
    def load_training_data(self) -> Tuple[List[Tuple[Email, bool]], List[Tuple[Email, bool]]]:
        """Load and prepare training data"""
        print("📚 Loading training data...")
        
        legitimate_samples = []
        spam_samples = []
        
        # Load legitimate emails
        legitimate_dir = Path("./legitimate")
        print(f"Loading legitimate emails from {legitimate_dir}...")
        
        for i, email_file in enumerate(legitimate_dir.glob("*.txt")):
            if i % 50 == 0:
                print(f"  Loaded {i} legitimate emails...")
            
            try:
                with open(email_file, 'r', encoding='utf-8') as f:
                    email_text = f.read()
                    email = parse_gmail_forward(email_text)
                    legitimate_samples.append((email, False))  # False = not spam
            except Exception as e:
                print(f"Error loading {email_file}: {e}")
        
        # Load spam emails
        spam_dir = Path("./spam")
        print(f"Loading spam emails from {spam_dir}...")
        
        for i, email_file in enumerate(spam_dir.glob("*.txt")):
            if i % 100 == 0:
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
        split_idx = int(len(all_samples) * 0.8)  # 80% training, 20% test
        training_data = all_samples[:split_idx]
        test_data = all_samples[split_idx:]
        
        print(f"📊 Training set: {len(training_data)} samples")
        print(f"📊 Test set: {len(test_data)} samples")
        
        return training_data, test_data
    
    async def evaluate_baseline_orchestrator(self, test_data: List[Tuple[Email, bool]]) -> Dict:
        """Evaluate current simple orchestrator for baseline"""
        from inbox_sentinel.orchestration import SimpleOrchestrator
        
        print("📈 Evaluating baseline orchestrator...")
        baseline_orchestrator = SimpleOrchestrator()
        
        correct_predictions = 0
        total_predictions = 0
        
        for email, true_label in test_data:
            try:
                result = await baseline_orchestrator.analyze_email(email)
                if result['success']:
                    detailed = result.get('detailed_results', {})
                    spam_votes = sum(1 for r in detailed.values() if r.get('is_spam', False) and 'error' not in r)
                    total_votes = sum(1 for r in detailed.values() if 'error' not in r)
                    predicted_spam = spam_votes > (total_votes / 2)
                    
                    if predicted_spam == true_label:
                        correct_predictions += 1
                    total_predictions += 1
            except Exception as e:
                print(f"Error in baseline evaluation: {e}")
        
        baseline_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"📊 Baseline orchestrator accuracy: {baseline_accuracy:.3f}")
        
        return {
            'accuracy': baseline_accuracy,
            'correct': correct_predictions,
            'total': total_predictions
        }
    
    async def train_rl_orchestrator(self, training_data: List[Tuple[Email, bool]], 
                                  episodes: int = 50, validation_split: float = 0.2):
        """Train the RL orchestrator"""
        print(f"🚀 Starting RL orchestrator training...")
        
        # Initialize orchestrator
        await self.orchestrator.initialize()
        
        # Train the model
        await self.orchestrator.train_on_dataset(
            training_data, 
            episodes=episodes, 
            validation_split=validation_split
        )
        
        # Save the trained model
        model_path = "./rl_orchestrator_trained.pkl"
        self.orchestrator.save_model(model_path)
        
        return model_path
    
    async def evaluate_rl_orchestrator(self, test_data: List[Tuple[Email, bool]]) -> Dict:
        """Evaluate trained RL orchestrator"""
        print("📈 Evaluating RL orchestrator...")
        
        correct_predictions = 0
        total_predictions = 0
        predictions_detail = []
        
        for email, true_label in test_data:
            try:
                result = await self.orchestrator.predict(email, training=False)
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
        
        print(f"📊 RL orchestrator accuracy: {rl_accuracy:.3f}")
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
    
    def plot_training_progress(self):
        """Plot training progress"""
        stats = self.orchestrator.get_training_stats()
        
        if not stats['rl_agent_stats']['accuracy_history']:
            print("No training data to plot")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training accuracy
        episodes = range(1, len(stats['rl_agent_stats']['accuracy_history']) + 1)
        ax1.plot(episodes, stats['rl_agent_stats']['accuracy_history'], 'b-', label='Training Accuracy')
        ax1.plot(episodes, stats['orchestrator_stats']['validation_accuracy_history'], 'r-', label='Validation Accuracy')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Training Progress')
        ax1.legend()
        ax1.grid(True)
        
        # Epsilon decay
        ax2.plot(episodes, stats['rl_agent_stats']['epsilon_history'], 'g-')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Epsilon (Exploration Rate)')
        ax2.set_title('Exploration Decay')
        ax2.grid(True)
        
        # Cumulative reward
        cumulative_rewards = []
        total = 0
        for reward in stats['rl_agent_stats'].get('reward_history', []):
            total += reward
            cumulative_rewards.append(total)
        
        if cumulative_rewards:
            ax3.plot(range(1, len(cumulative_rewards) + 1), cumulative_rewards, 'm-')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Cumulative Reward')
            ax3.set_title('Learning Progress')
            ax3.grid(True)
        
        # Final accuracy comparison (if baseline exists)
        final_accuracies = ['Baseline', 'RL Enhanced']
        baseline_acc = getattr(self, 'baseline_accuracy', 0.914)  # From previous test results
        rl_acc = stats['orchestrator_stats']['validation_accuracy_history'][-1] if stats['orchestrator_stats']['validation_accuracy_history'] else 0
        
        accuracies = [baseline_acc, rl_acc]
        colors = ['lightcoral', 'lightblue']
        
        bars = ax4.bar(final_accuracies, accuracies, color=colors)
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Final Accuracy Comparison')
        ax4.set_ylim(0, 1.0)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('rl_training_progress.png', dpi=300, bbox_inches='tight')
        print("📈 Training progress plot saved as 'rl_training_progress.png'")
    
    async def run_full_training_and_evaluation(self, episodes: int = 50):
        """Run complete training and evaluation pipeline"""
        print("🎯 Starting RL Orchestrator Training & Evaluation Pipeline")
        print("=" * 60)
        
        # Load data
        training_data, test_data = self.load_training_data()
        
        # Evaluate baseline
        baseline_results = await self.evaluate_baseline_orchestrator(test_data)
        self.baseline_accuracy = baseline_results['accuracy']
        
        # Train RL orchestrator
        model_path = await self.train_rl_orchestrator(training_data, episodes=episodes)
        
        # Evaluate RL orchestrator
        rl_results = await self.evaluate_rl_orchestrator(test_data)
        
        # Generate comprehensive report
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
        
        # Improvement calculation
        accuracy_improvement = rl_results['accuracy'] - baseline_results['accuracy']
        improvement_percent = (accuracy_improvement / baseline_results['accuracy']) * 100
        
        print(f"\\n📈 IMPROVEMENT:")
        print(f"   Accuracy improvement: {accuracy_improvement:+.3f} ({improvement_percent:+.1f}%)")
        
        if accuracy_improvement > 0:
            print(f"   🎉 RL orchestrator achieved {accuracy_improvement:.3f} better accuracy!")
        else:
            print(f"   ⚠️ RL orchestrator needs more training or hyperparameter tuning")
        
        # Save results
        results_data = {
            'baseline': baseline_results,
            'rl_enhanced': rl_results,
            'improvement': {
                'accuracy_delta': accuracy_improvement,
                'improvement_percent': improvement_percent
            },
            'training_stats': self.orchestrator.get_training_stats(),
            'model_path': model_path
        }
        
        with open('rl_training_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\\n💾 Results saved to 'rl_training_results.json'")
        print(f"💾 Trained model saved to '{model_path}'")
        
        # Plot training progress
        try:
            self.plot_training_progress()
        except Exception as e:
            print(f"⚠️ Could not generate plots: {e}")
        
        print("\\n✅ Training and evaluation complete!")
        return results_data


async def main():
    """Main training execution"""
    # Configuration
    TRAINING_EPISODES = 100  # Adjust based on your needs
    
    print("🚀 RL Enhanced Orchestrator Training")
    print(f"Episodes: {TRAINING_EPISODES}")
    print("=" * 50)
    
    # Create trainer
    trainer = RLTrainingManager()
    
    # Run training and evaluation
    start_time = time.time()
    results = await trainer.run_full_training_and_evaluation(episodes=TRAINING_EPISODES)
    end_time = time.time()
    
    print(f"\\n⏱️ Total execution time: {end_time - start_time:.1f} seconds")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())