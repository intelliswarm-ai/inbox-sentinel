#!/usr/bin/env python3
"""
Comprehensive test pipeline for Inbox Sentinel models
Processes all test samples and generates detailed performance statistics
"""

import asyncio
import json
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import statistics

# Add parent directory to path to import inbox_sentinel modules
import sys
sys.path.append('/app')

from inbox_sentinel.ml.models.naive_bayes import NaiveBayesDetector
from inbox_sentinel.ml.models.svm import SVMDetector
from inbox_sentinel.ml.models.random_forest import RandomForestDetector
from inbox_sentinel.ml.models.logistic_regression import LogisticRegressionDetector
from inbox_sentinel.ml.models.neural_network import NeuralNetworkDetector
from inbox_sentinel.orchestration import SimpleOrchestrator
from inbox_sentinel.utils.email_parser import parse_gmail_forward
from inbox_sentinel.core.types import Email

class ModelTester:
    """Test suite for evaluating model performance"""
    
    def __init__(self):
        self.models = {
            'naive_bayes': NaiveBayesDetector(),
            'svm': SVMDetector(),
            'random_forest': RandomForestDetector(),
            'logistic_regression': LogisticRegressionDetector(),
            'neural_network': NeuralNetworkDetector(),
        }
        self.orchestrator = SimpleOrchestrator()
        self.results = defaultdict(list)
        self.performance_stats = {}
        
    async def initialize_models(self):
        """Initialize all models"""
        print("Initializing models...")
        for name, model in self.models.items():
            try:
                await model.initialize(use_pretrained=True)
                if model.is_trained:
                    print(f"✅ {name} initialized successfully")
                else:
                    print(f"❌ {name} not trained")
            except Exception as e:
                print(f"❌ {name} initialization failed: {e}")
    
    def load_test_samples(self) -> Tuple[List[Tuple[str, Email]], List[Tuple[str, Email]]]:
        """Load legitimate and spam test samples"""
        legitimate_samples = []
        spam_samples = []
        
        # Load legitimate emails
        legitimate_dir = Path("./legitimate")
        for email_file in legitimate_dir.glob("*.txt"):
            try:
                with open(email_file, 'r', encoding='utf-8') as f:
                    email_text = f.read()
                    email = parse_gmail_forward(email_text)
                    legitimate_samples.append((email_file.name, email))
            except Exception as e:
                print(f"Error loading {email_file}: {e}")
        
        # Load spam emails
        spam_dir = Path("./spam")
        for email_file in spam_dir.glob("*.txt"):
            try:
                with open(email_file, 'r', encoding='utf-8') as f:
                    email_text = f.read()
                    email = parse_gmail_forward(email_text)
                    spam_samples.append((email_file.name, email))
            except Exception as e:
                print(f"Error loading {email_file}: {e}")
        
        print(f"Loaded {len(legitimate_samples)} legitimate and {len(spam_samples)} spam samples")
        return legitimate_samples, spam_samples
    
    async def test_individual_model(self, model_name: str, model, samples: List[Tuple[str, Email]], 
                                  true_label: str) -> Dict:
        """Test individual model performance"""
        print(f"\nTesting {model_name}...")
        
        if not model.is_trained:
            print(f"❌ {model_name} is not trained, skipping...")
            return {
                'model': model_name,
                'error': 'Model not trained',
                'predictions': [],
                'timing': []
            }
        
        predictions = []
        timing_data = []
        
        for filename, email in samples:
            try:
                start_time = time.time()
                result = await model.analyze(email)
                end_time = time.time()
                
                prediction = {
                    'filename': filename,
                    'true_label': true_label,
                    'predicted_spam': result.is_spam,
                    'predicted_label': 'spam' if result.is_spam else 'ham',
                    'spam_probability': result.spam_probability,
                    'confidence': result.confidence,
                    'correct': (result.is_spam and true_label == 'spam') or (not result.is_spam and true_label == 'ham')
                }
                predictions.append(prediction)
                timing_data.append(end_time - start_time)
                
            except Exception as e:
                print(f"Error processing {filename} with {model_name}: {e}")
                predictions.append({
                    'filename': filename,
                    'true_label': true_label,
                    'error': str(e)
                })
        
        return {
            'model': model_name,
            'predictions': predictions,
            'timing': timing_data
        }
    
    async def test_orchestrator(self, samples: List[Tuple[str, Email]], true_label: str) -> Dict:
        """Test orchestrator performance"""
        print(f"\nTesting orchestrator...")
        
        predictions = []
        timing_data = []
        
        for filename, email in samples:
            try:
                start_time = time.time()
                result = await self.orchestrator.analyze_email(email)
                end_time = time.time()
                
                if result['success']:
                    # Extract consensus from detailed results
                    detailed = result.get('detailed_results', {})
                    spam_votes = sum(1 for r in detailed.values() if r.get('is_spam', False) and 'error' not in r)
                    total_votes = sum(1 for r in detailed.values() if 'error' not in r)
                    is_spam = spam_votes > (total_votes / 2)
                    
                    prediction = {
                        'filename': filename,
                        'true_label': true_label,
                        'predicted_spam': is_spam,
                        'predicted_label': 'spam' if is_spam else 'ham',
                        'spam_votes': spam_votes,
                        'total_votes': total_votes,
                        'consensus_strength': abs(spam_votes - (total_votes - spam_votes)) / total_votes if total_votes > 0 else 0,
                        'correct': (is_spam and true_label == 'spam') or (not is_spam and true_label == 'ham'),
                        'detailed_results': detailed
                    }
                else:
                    prediction = {
                        'filename': filename,
                        'true_label': true_label,
                        'error': result.get('error', 'Unknown error')
                    }
                
                predictions.append(prediction)
                timing_data.append(end_time - start_time)
                
            except Exception as e:
                print(f"Error processing {filename} with orchestrator: {e}")
                predictions.append({
                    'filename': filename,
                    'true_label': true_label,
                    'error': str(e)
                })
        
        return {
            'model': 'orchestrator',
            'predictions': predictions,
            'timing': timing_data
        }
    
    def calculate_metrics(self, predictions: List[Dict]) -> Dict:
        """Calculate performance metrics"""
        # Filter out error predictions
        valid_predictions = [p for p in predictions if 'error' not in p]
        
        if not valid_predictions:
            return {'error': 'No valid predictions'}
        
        # Basic counts
        total = len(valid_predictions)
        correct = sum(1 for p in valid_predictions if p['correct'])
        
        # Confusion matrix
        tp = sum(1 for p in valid_predictions if p['true_label'] == 'spam' and p['predicted_spam'])
        tn = sum(1 for p in valid_predictions if p['true_label'] == 'ham' and not p['predicted_spam'])
        fp = sum(1 for p in valid_predictions if p['true_label'] == 'ham' and p['predicted_spam'])
        fn = sum(1 for p in valid_predictions if p['true_label'] == 'spam' and not p['predicted_spam'])
        
        # Metrics
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
    
    def calculate_timing_stats(self, timing_data: List[float]) -> Dict:
        """Calculate timing statistics"""
        if not timing_data:
            return {'error': 'No timing data'}
        
        return {
            'total_samples': len(timing_data),
            'total_time': sum(timing_data),
            'average_time': statistics.mean(timing_data),
            'median_time': statistics.median(timing_data),
            'min_time': min(timing_data),
            'max_time': max(timing_data),
            'std_deviation': statistics.stdev(timing_data) if len(timing_data) > 1 else 0
        }
    
    async def run_full_test(self):
        """Run comprehensive test suite"""
        print("=" * 80)
        print("INBOX SENTINEL COMPREHENSIVE TEST SUITE")
        print("=" * 80)
        
        # Initialize models
        await self.initialize_models()
        
        # Load test samples
        legitimate_samples, spam_samples = self.load_test_samples()
        all_samples = legitimate_samples + spam_samples
        
        # Test each individual model
        print("\n" + "=" * 60)
        print("INDIVIDUAL MODEL TESTING")
        print("=" * 60)
        
        all_results = {}
        
        for model_name, model in self.models.items():
            # Test on legitimate emails
            leg_results = await self.test_individual_model(model_name, model, legitimate_samples, 'ham')
            # Test on spam emails
            spam_results = await self.test_individual_model(model_name, model, spam_samples, 'spam')
            
            # Combine results
            all_predictions = leg_results['predictions'] + spam_results['predictions']
            all_timing = leg_results['timing'] + spam_results['timing']
            
            all_results[model_name] = {
                'predictions': all_predictions,
                'timing': all_timing,
                'metrics': self.calculate_metrics(all_predictions),
                'timing_stats': self.calculate_timing_stats(all_timing)
            }
        
        # Test orchestrator
        print("\n" + "=" * 60)
        print("ORCHESTRATOR TESTING")
        print("=" * 60)
        
        # Test orchestrator on legitimate emails
        orch_leg_results = await self.test_orchestrator(legitimate_samples, 'ham')
        # Test orchestrator on spam emails
        orch_spam_results = await self.test_orchestrator(spam_samples, 'spam')
        
        # Combine orchestrator results
        orch_predictions = orch_leg_results['predictions'] + orch_spam_results['predictions']
        orch_timing = orch_leg_results['timing'] + orch_spam_results['timing']
        
        all_results['orchestrator'] = {
            'predictions': orch_predictions,
            'timing': orch_timing,
            'metrics': self.calculate_metrics(orch_predictions),
            'timing_stats': self.calculate_timing_stats(orch_timing)
        }
        
        # Generate comprehensive report
        self.generate_report(all_results, len(legitimate_samples), len(spam_samples))
        
        # Save detailed results
        self.save_detailed_results(all_results)
        
        return all_results
    
    def generate_report(self, results: Dict, num_legitimate: int, num_spam: int):
        """Generate comprehensive performance report"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE PERFORMANCE REPORT")
        print("=" * 80)
        
        print(f"\nDataset Overview:")
        print(f"- Legitimate emails: {num_legitimate}")
        print(f"- Spam emails: {num_spam}")
        print(f"- Total samples: {num_legitimate + num_spam}")
        print(f"- Class balance: {num_spam}/{num_legitimate + num_spam:.1%} spam")
        
        # Performance summary table
        print("\n" + "-" * 90)
        print(f"{'Model':<18} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Avg Time (ms)':<12}")
        print("-" * 90)
        
        for model_name, result in results.items():
            metrics = result['metrics']
            timing = result['timing_stats']
            
            if 'error' not in metrics and 'error' not in timing:
                print(f"{model_name:<18} "
                      f"{metrics['accuracy']:<10.3f} "
                      f"{metrics['precision']:<10.3f} "
                      f"{metrics['recall']:<10.3f} "
                      f"{metrics['f1_score']:<10.3f} "
                      f"{timing['average_time']*1000:<12.1f}")
            else:
                print(f"{model_name:<18} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<12}")
        
        # Detailed metrics for each model
        print("\n" + "=" * 60)
        print("DETAILED MODEL METRICS")
        print("=" * 60)
        
        for model_name, result in results.items():
            metrics = result['metrics']
            timing = result['timing_stats']
            
            print(f"\n{model_name.upper().replace('_', ' ')}")
            print("-" * 40)
            
            if 'error' not in metrics:
                cm = metrics['confusion_matrix']
                print(f"Accuracy:     {metrics['accuracy']:.3f}")
                print(f"Precision:    {metrics['precision']:.3f}")
                print(f"Recall:       {metrics['recall']:.3f}")
                print(f"Specificity:  {metrics['specificity']:.3f}")
                print(f"F1-Score:     {metrics['f1_score']:.3f}")
                print(f"\nConfusion Matrix:")
                print(f"  True Positive (Spam → Spam):     {cm['true_positive']}")
                print(f"  True Negative (Ham → Ham):       {cm['true_negative']}")
                print(f"  False Positive (Ham → Spam):     {cm['false_positive']}")
                print(f"  False Negative (Spam → Ham):     {cm['false_negative']}")
            else:
                print(f"Error: {metrics.get('error', 'Unknown error')}")
            
            if 'error' not in timing:
                print(f"\nTiming Statistics:")
                print(f"  Average time:     {timing['average_time']*1000:.1f} ms")
                print(f"  Median time:      {timing['median_time']*1000:.1f} ms")
                print(f"  Min time:         {timing['min_time']*1000:.1f} ms")
                print(f"  Max time:         {timing['max_time']*1000:.1f} ms")
                print(f"  Total time:       {timing['total_time']:.1f} seconds")
        
        # Best performing model
        best_model = None
        best_f1 = 0
        
        for model_name, result in results.items():
            if 'error' not in result['metrics']:
                f1 = result['metrics']['f1_score']
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model_name
        
        if best_model:
            print(f"\n🏆 BEST PERFORMING MODEL: {best_model.upper().replace('_', ' ')}")
            print(f"   F1-Score: {best_f1:.3f}")
        
        # Recommendations
        print(f"\n📊 RECOMMENDATIONS:")
        print(f"- Use {best_model.replace('_', ' ')} for best overall performance")
        print(f"- Use orchestrator for maximum reliability (consensus-based)")
        print(f"- Monitor false positive rates to avoid blocking legitimate emails")
        
    def save_detailed_results(self, results: Dict):
        """Save detailed results to JSON file"""
        # Convert results to JSON-serializable format
        json_results = {}
        
        for model_name, result in results.items():
            json_results[model_name] = {
                'metrics': result['metrics'],
                'timing_stats': result['timing_stats'],
                'sample_count': len([p for p in result['predictions'] if 'error' not in p])
            }
        
        # Save to file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {results_file}")

async def main():
    """Main test execution"""
    tester = ModelTester()
    results = await tester.run_full_test()
    
    print(f"\n✅ Test suite completed successfully!")
    print(f"📈 Check the generated report above for detailed analysis.")

if __name__ == "__main__":
    asyncio.run(main())