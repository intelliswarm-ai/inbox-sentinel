#!/usr/bin/env python3
"""
Enhanced test pipeline with detailed progress logging and intermediate results
"""

import asyncio
import json
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import statistics
import sys

# Add parent directory to path to import inbox_sentinel modules
sys.path.append('/app')

from inbox_sentinel.ml.models.naive_bayes import NaiveBayesDetector
from inbox_sentinel.ml.models.svm import SVMDetector
from inbox_sentinel.ml.models.random_forest import RandomForestDetector
from inbox_sentinel.ml.models.logistic_regression import LogisticRegressionDetector
from inbox_sentinel.ml.models.neural_network import NeuralNetworkDetector
from inbox_sentinel.orchestration import SimpleOrchestrator
from inbox_sentinel.utils.email_parser import parse_gmail_forward
from inbox_sentinel.core.types import Email

class VerboseModelTester:
    """Enhanced test suite with detailed logging"""
    
    def __init__(self):
        self.models = {
            'naive_bayes': NaiveBayesDetector(),
            'svm': SVMDetector(),
            'random_forest': RandomForestDetector(),
            'logistic_regression': LogisticRegressionDetector(),
            'neural_network': NeuralNetworkDetector(),
        }
        self.orchestrator = SimpleOrchestrator()
        
    def log_progress(self, message: str, level: str = "INFO"):
        """Enhanced logging with timestamps"""
        timestamp = time.strftime("%H:%M:%S")
        if level == "INFO":
            prefix = "ℹ️"
        elif level == "SUCCESS":
            prefix = "✅"
        elif level == "WARNING":
            prefix = "⚠️"
        elif level == "ERROR":
            prefix = "❌"
        else:
            prefix = "📊"
        
        print(f"[{timestamp}] {prefix} {message}")
    
    async def initialize_models(self):
        """Initialize all models with progress logging"""
        self.log_progress("Starting model initialization...")
        
        for name, model in self.models.items():
            self.log_progress(f"Initializing {name}...")
            try:
                await model.initialize(use_pretrained=True)
                if model.is_trained:
                    self.log_progress(f"{name} initialized successfully", "SUCCESS")
                else:
                    self.log_progress(f"{name} not trained", "WARNING")
            except Exception as e:
                self.log_progress(f"{name} initialization failed: {e}", "ERROR")
        
        self.log_progress("Model initialization complete!")
    
    def load_test_samples(self) -> Tuple[List[Tuple[str, Email]], List[Tuple[str, Email]]]:
        """Load test samples with progress logging"""
        self.log_progress("Loading test samples...")
        
        legitimate_samples = []
        spam_samples = []
        
        # Load legitimate emails
        legitimate_dir = Path("./legitimate")
        self.log_progress(f"Loading legitimate emails from {legitimate_dir}...")
        
        for i, email_file in enumerate(legitimate_dir.glob("*.txt")):
            if i % 50 == 0:  # Progress update every 50 files
                self.log_progress(f"Loaded {i} legitimate emails so far...")
            
            try:
                with open(email_file, 'r', encoding='utf-8') as f:
                    email_text = f.read()
                    email = parse_gmail_forward(email_text)
                    legitimate_samples.append((email_file.name, email))
            except Exception as e:
                self.log_progress(f"Error loading {email_file}: {e}", "ERROR")
        
        # Load spam emails
        spam_dir = Path("./spam")
        self.log_progress(f"Loading spam emails from {spam_dir}...")
        
        for i, email_file in enumerate(spam_dir.glob("*.txt")):
            if i % 100 == 0:  # Progress update every 100 files
                self.log_progress(f"Loaded {i} spam emails so far...")
            
            try:
                with open(email_file, 'r', encoding='utf-8') as f:
                    email_text = f.read()
                    email = parse_gmail_forward(email_text)
                    spam_samples.append((email_file.name, email))
            except Exception as e:
                self.log_progress(f"Error loading {email_file}: {e}", "ERROR")
        
        self.log_progress(f"Sample loading complete! {len(legitimate_samples)} legitimate, {len(spam_samples)} spam", "SUCCESS")
        return legitimate_samples, spam_samples
    
    async def test_individual_model_with_progress(self, model_name: str, model, samples: List[Tuple[str, Email]], 
                                                true_label: str) -> Dict:
        """Test individual model with detailed progress logging"""
        self.log_progress(f"Testing {model_name} on {len(samples)} {true_label} samples...")
        
        if not model.is_trained:
            self.log_progress(f"{model_name} is not trained, skipping...", "WARNING")
            return {
                'model': model_name,
                'error': 'Model not trained',
                'predictions': [],
                'timing': []
            }
        
        predictions = []
        timing_data = []
        correct_count = 0
        
        start_time = time.time()
        
        for i, (filename, email) in enumerate(samples):
            # Progress updates every 100 samples
            if i % 100 == 0 and i > 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                estimated_remaining = avg_time * (len(samples) - i)
                accuracy_so_far = correct_count / i if i > 0 else 0
                
                self.log_progress(
                    f"{model_name}: {i}/{len(samples)} processed "
                    f"(Accuracy: {accuracy_so_far:.3f}, "
                    f"Avg: {avg_time*1000:.1f}ms, "
                    f"ETA: {estimated_remaining:.1f}s)"
                )
            
            try:
                sample_start = time.time()
                result = await model.analyze(email)
                sample_end = time.time()
                
                is_correct = (result.is_spam and true_label == 'spam') or (not result.is_spam and true_label == 'ham')
                if is_correct:
                    correct_count += 1
                
                prediction = {
                    'filename': filename,
                    'true_label': true_label,
                    'predicted_spam': result.is_spam,
                    'predicted_label': 'spam' if result.is_spam else 'ham',
                    'spam_probability': result.spam_probability,
                    'confidence': result.confidence,
                    'correct': is_correct
                }
                predictions.append(prediction)
                timing_data.append(sample_end - sample_start)
                
            except Exception as e:
                self.log_progress(f"Error processing {filename} with {model_name}: {e}", "ERROR")
                predictions.append({
                    'filename': filename,
                    'true_label': true_label,
                    'error': str(e)
                })
        
        total_time = time.time() - start_time
        final_accuracy = correct_count / len([p for p in predictions if 'error' not in p])
        
        self.log_progress(
            f"{model_name} complete: {correct_count}/{len(samples)} correct "
            f"(Accuracy: {final_accuracy:.3f}, Total time: {total_time:.1f}s)", 
            "SUCCESS"
        )
        
        return {
            'model': model_name,
            'predictions': predictions,
            'timing': timing_data
        }
    
    async def test_orchestrator_with_progress(self, samples: List[Tuple[str, Email]], true_label: str) -> Dict:
        """Test orchestrator with detailed progress logging"""
        self.log_progress(f"Testing orchestrator on {len(samples)} {true_label} samples...")
        
        predictions = []
        timing_data = []
        correct_count = 0
        
        start_time = time.time()
        
        for i, (filename, email) in enumerate(samples):
            # Progress updates every 50 samples (orchestrator is slower)
            if i % 50 == 0 and i > 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                estimated_remaining = avg_time * (len(samples) - i)
                accuracy_so_far = correct_count / i if i > 0 else 0
                
                self.log_progress(
                    f"Orchestrator: {i}/{len(samples)} processed "
                    f"(Accuracy: {accuracy_so_far:.3f}, "
                    f"Avg: {avg_time:.1f}s, "
                    f"ETA: {estimated_remaining:.1f}s)"
                )
            
            try:
                sample_start = time.time()
                result = await self.orchestrator.analyze_email(email)
                sample_end = time.time()
                
                if result['success']:
                    # Extract consensus from detailed results
                    detailed = result.get('detailed_results', {})
                    spam_votes = sum(1 for r in detailed.values() if r.get('is_spam', False) and 'error' not in r)
                    total_votes = sum(1 for r in detailed.values() if 'error' not in r)
                    is_spam = spam_votes > (total_votes / 2)
                    
                    is_correct = (is_spam and true_label == 'spam') or (not is_spam and true_label == 'ham')
                    if is_correct:
                        correct_count += 1
                    
                    prediction = {
                        'filename': filename,
                        'true_label': true_label,
                        'predicted_spam': is_spam,
                        'predicted_label': 'spam' if is_spam else 'ham',
                        'spam_votes': spam_votes,
                        'total_votes': total_votes,
                        'consensus_strength': abs(spam_votes - (total_votes - spam_votes)) / total_votes if total_votes > 0 else 0,
                        'correct': is_correct,
                        'detailed_results': detailed
                    }
                else:
                    prediction = {
                        'filename': filename,
                        'true_label': true_label,
                        'error': result.get('error', 'Unknown error')
                    }
                
                predictions.append(prediction)
                timing_data.append(sample_end - sample_start)
                
            except Exception as e:
                self.log_progress(f"Error processing {filename} with orchestrator: {e}", "ERROR")
                predictions.append({
                    'filename': filename,
                    'true_label': true_label,
                    'error': str(e)
                })
        
        total_time = time.time() - start_time
        valid_predictions = [p for p in predictions if 'error' not in p]
        final_accuracy = correct_count / len(valid_predictions) if valid_predictions else 0
        
        self.log_progress(
            f"Orchestrator complete: {correct_count}/{len(valid_predictions)} correct "
            f"(Accuracy: {final_accuracy:.3f}, Total time: {total_time:.1f}s)", 
            "SUCCESS"
        )
        
        return {
            'model': 'orchestrator',
            'predictions': predictions,
            'timing': timing_data
        }
    
    def show_intermediate_results(self, model_name: str, predictions: List[Dict]):
        """Show intermediate results for a model"""
        valid_predictions = [p for p in predictions if 'error' not in p]
        if not valid_predictions:
            self.log_progress(f"{model_name}: No valid predictions to analyze", "WARNING")
            return
        
        # Basic stats
        total = len(valid_predictions)
        correct = sum(1 for p in valid_predictions if p['correct'])
        accuracy = correct / total
        
        # Confusion matrix components
        tp = sum(1 for p in valid_predictions if p['true_label'] == 'spam' and p['predicted_spam'])
        tn = sum(1 for p in valid_predictions if p['true_label'] == 'ham' and not p['predicted_spam'])
        fp = sum(1 for p in valid_predictions if p['true_label'] == 'ham' and p['predicted_spam'])
        fn = sum(1 for p in valid_predictions if p['true_label'] == 'spam' and not p['predicted_spam'])
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        self.log_progress(
            f"{model_name} Results: "
            f"Accuracy: {accuracy:.3f}, "
            f"Precision: {precision:.3f}, "
            f"Recall: {recall:.3f}, "
            f"F1: {f1:.3f}"
        )
        
        # Show some example predictions
        self.log_progress(f"Sample predictions from {model_name}:")
        
        # Show some correct predictions
        correct_preds = [p for p in valid_predictions if p['correct']][:3]
        for pred in correct_preds:
            label = "✅ CORRECT"
            self.log_progress(f"  {label}: {pred['filename']} -> {pred['predicted_label']} (confidence: {pred.get('confidence', 'N/A')})")
        
        # Show some incorrect predictions
        incorrect_preds = [p for p in valid_predictions if not p['correct']][:3]
        for pred in incorrect_preds:
            label = "❌ INCORRECT"
            self.log_progress(f"  {label}: {pred['filename']} -> {pred['predicted_label']} (should be {pred['true_label']}, confidence: {pred.get('confidence', 'N/A')})")
    
    def calculate_metrics(self, predictions: List[Dict]) -> Dict:
        """Calculate performance metrics"""
        valid_predictions = [p for p in predictions if 'error' not in p]
        
        if not valid_predictions:
            return {'error': 'No valid predictions'}
        
        total = len(valid_predictions)
        correct = sum(1 for p in valid_predictions if p['correct'])
        
        tp = sum(1 for p in valid_predictions if p['true_label'] == 'spam' and p['predicted_spam'])
        tn = sum(1 for p in valid_predictions if p['true_label'] == 'ham' and not p['predicted_spam'])
        fp = sum(1 for p in valid_predictions if p['true_label'] == 'ham' and p['predicted_spam'])
        fn = sum(1 for p in valid_predictions if p['true_label'] == 'spam' and not p['predicted_spam'])
        
        accuracy = correct / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'total_samples': total,
            'correct_predictions': correct,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1_score,
            'confusion_matrix': {
                'true_positive': tp,
                'true_negative': tn,
                'false_positive': fp,
                'false_negative': fn
            }
        }
    
    async def run_verbose_test(self):
        """Run comprehensive test with verbose logging"""
        self.log_progress("🚀 Starting Inbox Sentinel Comprehensive Test Suite")
        self.log_progress("=" * 60)
        
        # Initialize models
        await self.initialize_models()
        
        # Load samples
        legitimate_samples, spam_samples = self.load_test_samples()
        
        self.log_progress("📊 Dataset Summary:")
        self.log_progress(f"   Legitimate emails: {len(legitimate_samples)}")
        self.log_progress(f"   Spam emails: {len(spam_samples)}")
        self.log_progress(f"   Total samples: {len(legitimate_samples) + len(spam_samples)}")
        self.log_progress(f"   Class balance: {len(spam_samples)/(len(legitimate_samples) + len(spam_samples)):.1%} spam")
        
        all_results = {}
        
        # Test individual models
        self.log_progress("🔍 Testing Individual Models")
        self.log_progress("-" * 40)
        
        for model_name, model in self.models.items():
            self.log_progress(f"Starting {model_name} evaluation...")
            
            # Test on legitimate emails
            leg_results = await self.test_individual_model_with_progress(model_name, model, legitimate_samples, 'ham')
            
            # Test on spam emails
            spam_results = await self.test_individual_model_with_progress(model_name, model, spam_samples, 'spam')
            
            # Combine and show intermediate results
            all_predictions = leg_results['predictions'] + spam_results['predictions']
            all_timing = leg_results['timing'] + spam_results['timing']
            
            self.show_intermediate_results(model_name, all_predictions)
            
            all_results[model_name] = {
                'predictions': all_predictions,
                'timing': all_timing,
                'metrics': self.calculate_metrics(all_predictions),
            }
            
            self.log_progress(f"{model_name} evaluation complete!")
            self.log_progress("-" * 20)
        
        # Test orchestrator
        self.log_progress("🎭 Testing Orchestrator (Ensemble)")
        self.log_progress("-" * 40)
        
        orch_leg_results = await self.test_orchestrator_with_progress(legitimate_samples, 'ham')
        orch_spam_results = await self.test_orchestrator_with_progress(spam_samples, 'spam')
        
        orch_predictions = orch_leg_results['predictions'] + orch_spam_results['predictions']
        orch_timing = orch_leg_results['timing'] + orch_spam_results['timing']
        
        self.show_intermediate_results('orchestrator', orch_predictions)
        
        all_results['orchestrator'] = {
            'predictions': orch_predictions,
            'timing': orch_timing,
            'metrics': self.calculate_metrics(orch_predictions),
        }
        
        # Final summary
        self.log_progress("📋 FINAL PERFORMANCE SUMMARY")
        self.log_progress("=" * 60)
        
        for model_name, result in all_results.items():
            metrics = result['metrics']
            if 'error' not in metrics:
                avg_time = statistics.mean(result['timing']) * 1000 if result['timing'] else 0
                self.log_progress(
                    f"{model_name:>15}: Acc={metrics['accuracy']:.3f} "
                    f"Prec={metrics['precision']:.3f} "
                    f"Rec={metrics['recall']:.3f} "
                    f"F1={metrics['f1_score']:.3f} "
                    f"Time={avg_time:.1f}ms"
                )
        
        # Find best model
        best_model = None
        best_f1 = 0
        
        for model_name, result in all_results.items():
            if 'error' not in result['metrics']:
                f1 = result['metrics']['f1_score']
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model_name
        
        if best_model:
            self.log_progress(f"🏆 Best performing model: {best_model.upper()} (F1: {best_f1:.3f})", "SUCCESS")
        
        self.log_progress("✅ Test suite completed successfully!", "SUCCESS")
        return all_results

async def main():
    """Run verbose test suite"""
    tester = VerboseModelTester()
    await tester.run_verbose_test()

if __name__ == "__main__":
    asyncio.run(main())