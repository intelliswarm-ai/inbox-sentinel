#!/usr/bin/env python3
"""
Test all models on actual emails from the datasets to verify accuracy
"""

import asyncio
import sys
import pandas as pd
import zipfile
import random
from pathlib import Path
from typing import List, Dict, Tuple
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from inbox_sentinel.core.types import Email
from inbox_sentinel.ml.models import (
    NaiveBayesDetector,
    SVMDetector,
    RandomForestDetector,
    LogisticRegressionDetector,
    NeuralNetworkDetector
)

console = Console()


def load_sample_emails(dataset_dir: str = "data/datasets", n_samples: int = 20) -> List[Tuple[Email, bool]]:
    """Load sample emails from datasets"""
    samples = []
    
    # Load from different datasets - ensure we get both spam and ham
    dataset_files = [
        ("SpamAssasin.csv.zip", "mixed"),  # Contains both spam and ham
        ("phishing_email.csv.zip", "spam"),  # Mostly spam/phishing
        ("Enron.csv.zip", "mixed")  # Contains both
    ]
    
    spam_samples = []
    ham_samples = []
    
    for dataset_file, dataset_type in dataset_files:
        file_path = Path(dataset_dir) / dataset_file
        if file_path.exists():
            console.print(f"Loading samples from {dataset_file}...")
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    csv_filename = zip_ref.namelist()[0]
                    with zip_ref.open(csv_filename) as csv_file:
                        df = pd.read_csv(csv_file, nrows=1000)  # Load first 1000 rows
                        
                        # Find columns
                        content_cols = ['text', 'email', 'message', 'content', 'body', 'Body', 'Text', 'text_combined']
                        label_cols = ['label', 'Label', 'spam', 'is_spam', 'class', 'Class']
                        
                        content_col = None
                        label_col = None
                        
                        for col in content_cols:
                            if col in df.columns:
                                content_col = col
                                break
                        
                        for col in label_cols:
                            if col in df.columns:
                                label_col = col
                                break
                        
                        if content_col and label_col:
                            # Sample some emails
                            n_to_sample = min(n_samples // len(dataset_files), len(df))
                            sampled_df = df.sample(n=n_to_sample, random_state=42)
                            
                            for _, row in sampled_df.iterrows():
                                content = str(row[content_col])
                                # Extract subject if possible (simple heuristic)
                                lines = content.split('\\n')
                                subject = lines[0][:100] if lines else "No subject"
                                
                                # Simple sender extraction or default
                                sender = f"test_{dataset_file.split('.')[0]}@example.com"
                                
                                email = Email(
                                    content=content[:1000],  # Limit content length
                                    subject=subject,
                                    sender=sender
                                )
                                
                                # Determine if spam
                                label_value = row[label_col]
                                if isinstance(label_value, str):
                                    is_spam = label_value.lower() in ['spam', '1', 'phishing']
                                else:
                                    is_spam = bool(label_value == 1)
                                
                                if is_spam:
                                    spam_samples.append((email, is_spam))
                                else:
                                    ham_samples.append((email, is_spam))
                        
            except Exception as e:
                console.print(f"[red]Error loading {dataset_file}: {e}[/red]")
    
    # Balance the samples - equal number of spam and ham
    console.print(f"Found {len(spam_samples)} spam and {len(ham_samples)} ham emails")
    
    # Take equal samples of each
    n_each = min(n_samples // 2, len(spam_samples), len(ham_samples))
    
    if n_each > 0:
        samples = random.sample(spam_samples, n_each) + random.sample(ham_samples, n_each)
        random.shuffle(samples)
    else:
        # If we can't balance, just take what we have
        samples = spam_samples[:n_samples//2] + ham_samples[:n_samples//2]
    
    return samples


async def test_model(detector, email: Email, true_label: bool) -> Dict[str, any]:
    """Test a single model on an email"""
    try:
        result = await detector.analyze(email)
        
        # Check if prediction matches true label
        correct = (result.is_spam == true_label)
        
        return {
            'model': detector.algorithm,
            'prediction': result.is_spam,
            'true_label': true_label,
            'correct': correct,
            'confidence': result.confidence,
            'spam_prob': result.spam_probability,
            'error': result.error
        }
    except Exception as e:
        return {
            'model': detector.algorithm,
            'prediction': None,
            'true_label': true_label,
            'correct': False,
            'confidence': 0.0,
            'spam_prob': 0.0,
            'error': str(e)
        }


async def test_all_models():
    """Test all models on sample emails"""
    console.print("[bold cyan]Loading sample emails from datasets...[/bold cyan]")
    samples = load_sample_emails(n_samples=50)
    
    if not samples:
        console.print("[red]No samples loaded![/red]")
        return
    
    console.print(f"[green]Loaded {len(samples)} sample emails[/green]")
    
    # Show distribution
    n_spam = sum(1 for _, is_spam in samples if is_spam)
    n_ham = len(samples) - n_spam
    console.print(f"Distribution: {n_spam} spam ({n_spam/len(samples)*100:.1f}%), {n_ham} ham ({n_ham/len(samples)*100:.1f}%)")
    
    # Initialize all detectors
    detectors = {
        'Naive Bayes': NaiveBayesDetector(),
        'SVM': SVMDetector(),
        'Random Forest': RandomForestDetector(),
        'Logistic Regression': LogisticRegressionDetector(),
        'Neural Network': NeuralNetworkDetector()
    }
    
    # Initialize models
    console.print("\n[bold cyan]Initializing models...[/bold cyan]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        for name, detector in detectors.items():
            task = progress.add_task(f"Initializing {name}...", total=None)
            try:
                await detector.initialize(use_pretrained=True)
                if detector.is_trained:
                    progress.update(task, description=f"✅ {name} loaded")
                else:
                    progress.update(task, description=f"❌ {name} not trained")
            except Exception as e:
                progress.update(task, description=f"❌ {name} error: {e}")
    
    # Test each model
    results = {name: [] for name in detectors.keys()}
    
    console.print("\n[bold cyan]Testing models on sample emails...[/bold cyan]")
    
    # Sample some emails for detailed display
    detailed_samples = random.sample(samples, min(5, len(samples)))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Testing models...", total=len(samples))
        
        for i, (email, true_label) in enumerate(samples):
            for name, detector in detectors.items():
                if detector.is_trained:
                    result = await test_model(detector, email, true_label)
                    results[name].append(result)
            
            progress.update(task, advance=1)
    
    # Calculate accuracy for each model
    console.print("\n[bold cyan]Model Performance Summary:[/bold cyan]")
    
    summary_table = Table(title="Model Accuracy on Dataset Samples")
    summary_table.add_column("Model", style="cyan")
    summary_table.add_column("Accuracy", style="green")
    summary_table.add_column("Correct", style="yellow")
    summary_table.add_column("Total", style="yellow")
    summary_table.add_column("Avg Confidence", style="magenta")
    
    for name, model_results in results.items():
        if model_results:
            correct = sum(1 for r in model_results if r['correct'])
            total = len(model_results)
            accuracy = (correct / total) * 100 if total > 0 else 0
            avg_confidence = sum(r['confidence'] for r in model_results) / total if total > 0 else 0
            
            summary_table.add_row(
                name,
                f"{accuracy:.1f}%",
                str(correct),
                str(total),
                f"{avg_confidence:.1%}"
            )
    
    console.print(summary_table)
    
    # Show detailed results for a few samples
    console.print("\n[bold cyan]Detailed Results for Sample Emails:[/bold cyan]")
    
    for i, (email, true_label) in enumerate(detailed_samples):
        console.print(f"\n[bold]Sample {i+1}:[/bold]")
        console.print(f"Subject: {email.subject[:50]}...")
        console.print(f"True Label: {'SPAM' if true_label else 'HAM'}")
        console.print("Model Predictions:")
        
        pred_table = Table(show_header=False)
        pred_table.add_column("Model", style="cyan", width=20)
        pred_table.add_column("Prediction", style="yellow")
        pred_table.add_column("Confidence", style="magenta")
        pred_table.add_column("Result", style="green")
        
        for name, model_results in results.items():
            if i < len(model_results):
                r = model_results[i]
                if r['prediction'] is not None:
                    pred_label = "SPAM" if r['prediction'] else "HAM"
                    result_icon = "✅" if r['correct'] else "❌"
                    pred_table.add_row(
                        name,
                        pred_label,
                        f"{r['confidence']:.1%}",
                        result_icon
                    )
        
        console.print(pred_table)
    
    # Show confusion matrix for best model
    best_model = max(results.items(), key=lambda x: sum(r['correct'] for r in x[1]) if x[1] else 0)
    
    if best_model[1]:
        console.print(f"\n[bold cyan]Confusion Matrix for {best_model[0]}:[/bold cyan]")
        
        tp = sum(1 for r in best_model[1] if r['prediction'] and r['true_label'])
        fp = sum(1 for r in best_model[1] if r['prediction'] and not r['true_label'])
        tn = sum(1 for r in best_model[1] if not r['prediction'] and not r['true_label'])
        fn = sum(1 for r in best_model[1] if not r['prediction'] and r['true_label'])
        
        conf_table = Table(title=f"{best_model[0]} Confusion Matrix")
        conf_table.add_column("", style="bold")
        conf_table.add_column("Predicted SPAM", style="yellow")
        conf_table.add_column("Predicted HAM", style="green")
        
        conf_table.add_row("Actual SPAM", str(tp), str(fn))
        conf_table.add_row("Actual HAM", str(fp), str(tn))
        
        console.print(conf_table)
        
        if tp + fp > 0:
            precision = tp / (tp + fp)
            console.print(f"Precision: {precision:.2%}")
        if tp + fn > 0:
            recall = tp / (tp + fn)
            console.print(f"Recall: {recall:.2%}")


if __name__ == "__main__":
    asyncio.run(test_all_models())