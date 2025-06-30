#!/usr/bin/env python3
"""
Compare different orchestration approaches for email analysis
"""

import asyncio
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inbox_sentinel.core.types import Email
from inbox_sentinel.orchestration import SimpleOrchestrator
from inbox_sentinel.utils.email_parser import parse_gmail_forward

console = Console()

# The forwarded email example
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


async def demonstrate_orchestration():
    """Show how orchestration works with MCP servers"""
    
    console.print("[bold cyan]Email Analysis Orchestration Demonstration[/bold cyan]\n")
    
    # Parse the forwarded email
    email = parse_gmail_forward(FORWARDED_EMAIL)
    
    console.print(f"[bold]Analyzing Email:[/bold]")
    console.print(f"Subject: {email.subject}")
    console.print(f"Sender: {email.sender}")
    console.print(f"Content preview: {email.content[:100]}...\n")
    
    # Create orchestrator
    orchestrator = SimpleOrchestrator()
    
    console.print("[bold]How Orchestration Works:[/bold]\n")
    
    console.print("1. [cyan]Initialize All Detection Models[/cyan]")
    console.print("   Each MCP server loads its trained ML model\n")
    
    console.print("2. [cyan]Parallel Analysis[/cyan]")
    console.print("   All models analyze the email simultaneously\n")
    
    console.print("3. [cyan]Collect Results[/cyan]")
    console.print("   Gather predictions from each model\n")
    
    console.print("4. [cyan]Consensus Building[/cyan]")
    console.print("   Combine results using voting/averaging\n")
    
    console.print("5. [cyan]Final Verdict[/cyan]")
    console.print("   Provide unified recommendation\n")
    
    # Run the orchestration
    console.print("[bold yellow]Running Orchestrated Analysis...[/bold yellow]\n")
    
    result = await orchestrator.analyze_email(email)
    
    if result['success']:
        # Show the consensus result
        consensus = result['consensus']
        
        # Create summary panel
        verdict = "SPAM/PHISHING" if consensus['is_spam'] else "LEGITIMATE"
        color = "red" if consensus['is_spam'] else "green"
        
        summary = f"""
[bold {color}]{verdict}[/bold {color}]

Consensus: {consensus['spam_votes']}/{consensus['total_models']} models detected spam
Average Confidence: {consensus['average_confidence']:.1%}
"""
        
        console.print(Panel(summary.strip(), title="Orchestration Result", border_style=color))
        
        # Show detailed breakdown
        console.print("\n[bold]Individual Model Results:[/bold]")
        
        table = Table(show_header=True)
        table.add_column("Model", style="cyan", width=20)
        table.add_column("Verdict", style="bold", width=10)
        table.add_column("Confidence", style="magenta", width=12)
        table.add_column("Spam Prob", style="red", width=10)
        table.add_column("Key Features", style="yellow", width=40)
        
        for model_name, model_result in result['detailed_results'].items():
            if 'error' not in model_result:
                verdict = "SPAM" if model_result['is_spam'] else "HAM"
                verdict_color = "red" if model_result['is_spam'] else "green"
                
                # Get features
                features = []
                if model_result.get('features'):
                    features = [f['feature'] for f in model_result['features'][:2]]
                features_str = ", ".join(features) if features else "N/A"
                
                table.add_row(
                    model_name.replace('_', ' ').title(),
                    f"[{verdict_color}]{verdict}[/{verdict_color}]",
                    f"{model_result['confidence']:.1%}",
                    f"{model_result['spam_probability']:.1%}",
                    features_str
                )
        
        console.print(table)
        
        # Explain the decision process
        console.print("\n[bold]Decision Process:[/bold]")
        
        explanation = f"""
The orchestrator used {consensus['total_models']} different ML models to analyze this email:

1. [cyan]Data Collection[/cyan]: Each model received the same email data
2. [cyan]Independent Analysis[/cyan]: Models analyzed using their trained parameters
3. [cyan]Feature Detection[/cyan]: Each model identified suspicious patterns
4. [cyan]Probability Calculation[/cyan]: Models computed spam probabilities
5. [cyan]Consensus Building[/cyan]: {consensus['spam_votes']} out of {consensus['total_models']} models voted SPAM
6. [cyan]Confidence Aggregation[/cyan]: Average confidence was {consensus['average_confidence']:.1%}

[bold]Why Multiple Models?[/bold]
- [green]Reduces false positives[/green]: One model's mistake is corrected by others
- [green]Improves accuracy[/green]: Different algorithms catch different patterns
- [green]Provides confidence[/green]: Agreement between models increases certainty
- [green]Explains decisions[/green]: Multiple perspectives on why email is suspicious
"""
        
        console.print(Panel(explanation.strip(), title="How Orchestration Improves Detection"))
        
        # Show LLM enhancement possibilities
        console.print("\n[bold]LLM Enhancement (with LangChain):[/bold]")
        
        llm_benefits = """
When using LangChain with a local LLM (like Llama2 via Ollama), the orchestration becomes even smarter:

1. [cyan]Intelligent Tool Selection[/cyan]
   The LLM decides which models to use based on email characteristics

2. [cyan]Contextual Analysis[/cyan]
   The LLM understands the email content and can reason about it

3. [cyan]Adaptive Strategy[/cyan]
   If models disagree, the LLM can request additional analysis

4. [cyan]Natural Language Explanation[/cyan]
   The LLM provides human-readable explanations of its decision

5. [cyan]Pattern Recognition[/cyan]
   The LLM can identify new phishing patterns not in training data

Example LLM reasoning:
"I notice this email mentions cryptocurrency rewards and uses urgency tactics.
Let me check with the Neural Network and Logistic Regression models first,
as they're best at detecting crypto scams. The models show high spam probability
(97.8% and 100%). The suspicious domain and future date confirm this is phishing."
"""
        
        console.print(Panel(llm_benefits.strip(), title="Benefits of LLM Orchestration", border_style="cyan"))
        
    else:
        console.print(f"[red]Orchestration failed: {result.get('error')}[/red]")


if __name__ == "__main__":
    asyncio.run(demonstrate_orchestration())