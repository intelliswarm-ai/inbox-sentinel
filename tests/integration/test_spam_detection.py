#!/usr/bin/env python3
"""
Test models specifically on known spam/phishing examples
"""

import asyncio
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table

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

# Known spam/phishing examples
SPAM_EXAMPLES = [
    {
        "content": "Congratulations! You've won $1,000,000 in the lottery! Click here to claim your prize: http://scam-site.com/claim",
        "subject": "You're a WINNER!",
        "sender": "lottery@fake-prize.com"
    },
    {
        "content": "Dear Customer, Your account has been suspended due to suspicious activity. Please verify your account immediately by clicking this link: http://phishing-site.tk/verify",
        "subject": "Urgent: Account Suspended",
        "sender": "security@paypal-verify.tk"
    },
    {
        "content": "Get rich quick! Make $5000 per week working from home. No experience needed. Limited time offer! Act now!",
        "subject": "Make Money Fast!",
        "sender": "opportunity@get-rich-quick.com"
    },
    {
        "content": "V1AGRA for sale! Cheapest prices online. No prescription needed. Fast discrete shipping. Order now!",
        "subject": "Cheap Medications",
        "sender": "pharmacy@meds-online.ru"
    },
    {
        "content": "I am a Nigerian prince and I need your help to transfer $50 million. You will receive 20% for your assistance.",
        "subject": "Business Proposal",
        "sender": "prince@nigeria-gov.com"
    }
]

# Known legitimate examples
HAM_EXAMPLES = [
    {
        "content": "Hi team, Please find attached the Q3 financial report. Let me know if you have any questions.",
        "subject": "Q3 Financial Report",
        "sender": "cfo@company.com"
    },
    {
        "content": "Thank you for your order. Your package will arrive in 3-5 business days. Order #12345",
        "subject": "Order Confirmation",
        "sender": "orders@amazon.com"
    },
    {
        "content": "Meeting reminder: Project sync tomorrow at 2 PM in conference room B.",
        "subject": "Meeting Reminder",
        "sender": "calendar@company.com"
    },
    {
        "content": "Here's the contract draft we discussed. Please review and let me know if any changes are needed.",
        "subject": "Contract Draft",
        "sender": "legal@lawfirm.com"
    },
    {
        "content": "Your monthly statement is now available. Login to your account to view details.",
        "subject": "Monthly Statement Available",
        "sender": "statements@bank.com"
    }
]


async def test_examples():
    """Test all models on known examples"""
    
    # Initialize all detectors
    detectors = {
        'Naive Bayes': NaiveBayesDetector(),
        'SVM': SVMDetector(),
        'Random Forest': RandomForestDetector(),
        'Logistic Regression': LogisticRegressionDetector(),
        'Neural Network': NeuralNetworkDetector()
    }
    
    # Initialize models
    console.print("[bold cyan]Initializing models...[/bold cyan]")
    for name, detector in detectors.items():
        try:
            await detector.initialize(use_pretrained=True)
            if detector.is_trained:
                console.print(f"✅ {name} loaded")
            else:
                console.print(f"❌ {name} not trained")
        except Exception as e:
            console.print(f"❌ {name} error: {e}")
    
    # Test spam examples
    console.print("\n[bold red]Testing Known SPAM Examples:[/bold red]")
    
    for i, example in enumerate(SPAM_EXAMPLES):
        email = Email(**example)
        console.print(f"\n[bold]Spam Example {i+1}:[/bold] {example['subject']}")
        
        results_table = Table(show_header=True)
        results_table.add_column("Model", style="cyan")
        results_table.add_column("Prediction", style="yellow")
        results_table.add_column("Confidence", style="magenta")
        results_table.add_column("Spam Prob", style="red")
        
        for name, detector in detectors.items():
            if detector.is_trained:
                try:
                    result = await detector.analyze(email)
                    prediction = "SPAM" if result.is_spam else "HAM"
                    style = "green" if result.is_spam else "red"
                    
                    results_table.add_row(
                        name,
                        f"[{style}]{prediction}[/{style}]",
                        f"{result.confidence:.1%}",
                        f"{result.spam_probability:.1%}"
                    )
                except Exception as e:
                    results_table.add_row(name, f"[red]ERROR[/red]", "-", "-")
        
        console.print(results_table)
    
    # Test ham examples
    console.print("\n[bold green]Testing Known HAM Examples:[/bold green]")
    
    for i, example in enumerate(HAM_EXAMPLES):
        email = Email(**example)
        console.print(f"\n[bold]Ham Example {i+1}:[/bold] {example['subject']}")
        
        results_table = Table(show_header=True)
        results_table.add_column("Model", style="cyan")
        results_table.add_column("Prediction", style="yellow")
        results_table.add_column("Confidence", style="magenta")
        results_table.add_column("Spam Prob", style="red")
        
        for name, detector in detectors.items():
            if detector.is_trained:
                try:
                    result = await detector.analyze(email)
                    prediction = "SPAM" if result.is_spam else "HAM"
                    style = "green" if not result.is_spam else "red"
                    
                    results_table.add_row(
                        name,
                        f"[{style}]{prediction}[/{style}]",
                        f"{result.confidence:.1%}",
                        f"{result.spam_probability:.1%}"
                    )
                except Exception as e:
                    results_table.add_row(name, f"[red]ERROR[/red]", "-", "-")
        
        console.print(results_table)
    
    # Summary statistics
    console.print("\n[bold cyan]Summary Statistics:[/bold cyan]")
    
    summary_table = Table(title="Model Performance on Known Examples")
    summary_table.add_column("Model", style="cyan")
    summary_table.add_column("Spam Detection Rate", style="red")
    summary_table.add_column("Ham Detection Rate", style="green")
    summary_table.add_column("Overall Accuracy", style="yellow")
    
    for name, detector in detectors.items():
        if detector.is_trained:
            spam_correct = 0
            ham_correct = 0
            
            # Test spam examples
            for example in SPAM_EXAMPLES:
                try:
                    email = Email(**example)
                    result = await detector.analyze(email)
                    if result.is_spam:
                        spam_correct += 1
                except:
                    pass
            
            # Test ham examples
            for example in HAM_EXAMPLES:
                try:
                    email = Email(**example)
                    result = await detector.analyze(email)
                    if not result.is_spam:
                        ham_correct += 1
                except:
                    pass
            
            spam_rate = spam_correct / len(SPAM_EXAMPLES) * 100
            ham_rate = ham_correct / len(HAM_EXAMPLES) * 100
            overall = (spam_correct + ham_correct) / (len(SPAM_EXAMPLES) + len(HAM_EXAMPLES)) * 100
            
            summary_table.add_row(
                name,
                f"{spam_rate:.0f}%",
                f"{ham_rate:.0f}%",
                f"{overall:.0f}%"
            )
    
    console.print(summary_table)


if __name__ == "__main__":
    asyncio.run(test_examples())