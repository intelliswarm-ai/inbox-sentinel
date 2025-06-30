#!/usr/bin/env python3
"""
Analyze the forwarded email with all available models
"""

import asyncio
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from inbox_sentinel.core.types import Email
from inbox_sentinel.utils.email_parser import parse_gmail_forward
from inbox_sentinel.ml.models import (
    NaiveBayesDetector,
    SVMDetector,
    RandomForestDetector,
    LogisticRegressionDetector,
    NeuralNetworkDetector
)

console = Console()

FORWARDED_EMAIL = """---------- Forwarded message ---------
From: Merlin Chain <hello@merlinteamnews.blog>
Date: Sat, Jun 28, 2025 at 5:14 AM
Subject: Claim Your Merlin Chain Early Users Reward Now
To: <thodoris.messinis@gmail.com>


Dear Merlin Chain Community Member,

We're thrilled to invite you to the Merlin Chain Early Users Reward Program, a heartfelt thank-you for supporting our mission to revolutionize Bitcoin as a leading Layer 2 solution. By enhancing scalability and functionality with ZK-Rollup and EVM compatibility, we're empowering DeFi, gaming, and NFTs. Join us to earn M-Points, MERL tokens, and exclusive airdrops

Eligibility Criteria

Confirm you qualify with these simple requirements:
1.      Users who Hold a compatible wallet (e.g., Trust Wallet, MetaMask with Merlin Chain Mainnet, or Merlin Chain-specific wallet) with MERL tokens or staked assets as of May 30, 2025.
2.      Users who had staked at least 0.001 M-BTC or equivalent assets via Merlin's Seal to mint M-Tokens.
3.      Users who had held MERL tokens in a non-custodial or supported exchange by May 30, 2025.
4.      Users who have completed at least one on-chain transaction (e.g., swapping, lending, or staking) in our ecosystem, such as iZUMi Finance or Bitmap Game.


How to Claim

Click here 👉 https://merlinchainreward.live to secure your rewards and approve all security protocols to ensure rewards are granted

Warm regards,
©️Merlin Chain Team"""


async def analyze_with_all_models():
    """Analyze the email with all models"""
    
    # Parse the forwarded email
    email = parse_gmail_forward(FORWARDED_EMAIL)
    
    console.print("[bold cyan]Analyzing Forwarded Email with All Models[/bold cyan]\n")
    console.print(f"[dim]Original Sender: {email.sender}[/dim]")
    console.print(f"[dim]Subject: {email.subject}[/dim]\n")
    
    # Initialize all detectors
    detectors = [
        ('Naive Bayes', NaiveBayesDetector()),
        ('SVM', SVMDetector()),
        ('Random Forest', RandomForestDetector()),
        ('Logistic Regression', LogisticRegressionDetector()),
        ('Neural Network', NeuralNetworkDetector())
    ]
    
    # Initialize models
    console.print("[bold]Initializing models...[/bold]")
    for name, detector in detectors:
        try:
            await detector.initialize(use_pretrained=True)
            console.print(f"✅ {name} loaded")
        except Exception as e:
            console.print(f"❌ {name} failed: {e}")
    
    console.print("\n[bold]Analysis Results:[/bold]\n")
    
    # Analyze with each model
    results = []
    for name, detector in detectors:
        if detector.is_trained:
            try:
                result = await detector.analyze(email)
                results.append((name, result))
                
                # Display individual result
                status = "⚠️ SPAM/PHISHING" if result.is_spam else "✅ LEGITIMATE"
                color = "red" if result.is_spam else "green"
                
                panel_content = f"""[bold {color}]{status}[/bold {color}]
                
Confidence: {result.confidence:.1%}
Spam Probability: {result.spam_probability:.1%}
Ham Probability: {result.ham_probability:.1%}"""
                
                if result.features and len(result.features) > 0:
                    panel_content += "\n\nTop Features:"
                    for i, feature in enumerate(result.features[:3], 1):
                        panel_content += f"\n  {i}. {feature.get('feature', 'Unknown')} ({feature.get('type', 'Unknown')})"
                
                console.print(Panel(
                    panel_content.strip(),
                    title=f"{name} Analysis",
                    border_style=color
                ))
                console.print()
                
            except Exception as e:
                console.print(f"[red]Error analyzing with {name}: {e}[/red]\n")
    
    # Summary table
    if results:
        console.print("\n[bold cyan]Summary Comparison:[/bold cyan]")
        
        summary_table = Table(title="All Models Comparison")
        summary_table.add_column("Model", style="cyan")
        summary_table.add_column("Verdict", style="bold")
        summary_table.add_column("Confidence", style="magenta")
        summary_table.add_column("Spam Prob", style="red")
        summary_table.add_column("Features Detected", style="yellow")
        
        for name, result in results:
            verdict = "SPAM" if result.is_spam else "HAM"
            verdict_style = "red" if result.is_spam else "green"
            
            # Get feature summary
            if result.features:
                features = ", ".join([f.get('feature', '')[:20] for f in result.features[:2]])
                if len(result.features) > 2:
                    features += "..."
            else:
                features = "N/A"
            
            summary_table.add_row(
                name,
                f"[{verdict_style}]{verdict}[/{verdict_style}]",
                f"{result.confidence:.1%}",
                f"{result.spam_probability:.1%}",
                features
            )
        
        console.print(summary_table)
        
        # Consensus
        spam_votes = sum(1 for _, r in results if r.is_spam)
        total_votes = len(results)
        consensus_spam = spam_votes > total_votes / 2
        
        console.print(f"\n[bold]Consensus: {spam_votes}/{total_votes} models detected spam[/bold]")
        if consensus_spam:
            console.print("[red]Overall verdict: SPAM/PHISHING - Do not trust this email![/red]")
        else:
            console.print("[green]Overall verdict: Legitimate email[/green]")


if __name__ == "__main__":
    asyncio.run(analyze_with_all_models())