#!/usr/bin/env python3
"""
Demo of LangChain-based orchestration for email analysis

This demonstrates how to use a local LLM (via Ollama) to orchestrate
multiple MCP servers as tools for sophisticated email analysis.

Prerequisites:
1. Install Ollama: https://ollama.ai/
2. Pull a model: ollama pull llama2 (or mistral, codellama, etc.)
3. Install dependencies: pip install langchain langchain-community
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inbox_sentinel.core.types import Email
from inbox_sentinel.orchestration import LangChainOrchestrator, OrchestrationConfig
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()


# Example emails for testing
TEST_EMAILS = [
    {
        "name": "Phishing - Crypto Scam",
        "email": Email(
            subject="Claim Your Merlin Chain Early Users Reward Now",
            sender="hello@merlinteamnews.blog",
            content="""Dear Merlin Chain Community Member,

We're thrilled to invite you to the Merlin Chain Early Users Reward Program. 
Join us to earn M-Points, MERL tokens, and exclusive airdrops.

Click here 👉 https://merlinchainreward.live to secure your rewards.

Warm regards,
Merlin Chain Team"""
        )
    },
    {
        "name": "Legitimate - Business Email",
        "email": Email(
            subject="Q3 Financial Report",
            sender="cfo@company.com",
            content="""Hi team,

Please find attached the Q3 financial report for review. 
The meeting to discuss these results is scheduled for Monday at 2 PM.

Best regards,
John Smith
CFO"""
        )
    }
]


async def demo_langchain_orchestration():
    """Demonstrate LangChain orchestration with local LLM"""
    
    console.print("[bold cyan]LangChain Email Analysis Orchestration Demo[/bold cyan]\n")
    
    # Check if LangChain is available
    try:
        import langchain
        console.print("[green]✓ LangChain is installed[/green]")
    except ImportError:
        console.print("[red]✗ LangChain not installed[/red]")
        console.print("Install with: pip install langchain langchain-community")
        return
    
    # Configure orchestrator
    console.print("\n[yellow]Configuring LangChain orchestrator with Ollama...[/yellow]")
    
    try:
        config = OrchestrationConfig(
            llm_provider="ollama",
            model_name="llama2",  # Change this to your installed model
            temperature=0.1,
            verbose=True,  # Show agent's reasoning
            max_iterations=10
        )
        
        orchestrator = LangChainOrchestrator(config)
        console.print("[green]✓ Orchestrator configured[/green]\n")
        
    except Exception as e:
        console.print(f"[red]Failed to configure orchestrator: {e}[/red]")
        console.print("\n[yellow]Make sure Ollama is running:[/yellow]")
        console.print("1. Install Ollama from https://ollama.ai/")
        console.print("2. Run: ollama serve")
        console.print("3. Pull a model: ollama pull llama2")
        return
    
    # Test each email
    for test_case in TEST_EMAILS:
        console.print(f"\n[bold]Testing: {test_case['name']}[/bold]")
        console.print(f"Subject: {test_case['email'].subject}")
        console.print(f"Sender: {test_case['email'].sender}\n")
        
        # Run analysis
        result = orchestrator.analyze_email_sync(test_case['email'])
        
        if result['success']:
            console.print(Panel(
                Markdown(result['analysis']),
                title="LLM Analysis",
                border_style="cyan"
            ))
        else:
            console.print(f"[red]Analysis failed: {result['error']}[/red]")
        
        console.print("\n" + "="*80 + "\n")


def show_langchain_setup_instructions():
    """Show instructions for setting up LangChain with Ollama"""
    
    instructions = """
# LangChain + Ollama Setup Instructions

## 1. Install Dependencies

```bash
pip install langchain langchain-community
```

## 2. Install and Setup Ollama

### On macOS/Linux:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### On Windows:
Download from https://ollama.ai/download

## 3. Start Ollama Server

```bash
ollama serve
```

## 4. Pull a Model

```bash
# Recommended models for email analysis:
ollama pull llama2        # 7B parameters, good balance
ollama pull mistral       # 7B parameters, fast
ollama pull phi           # 2.7B parameters, very fast
```

## 5. Test the Setup

```bash
# Test Ollama is working
curl http://localhost:11434/api/generate -d '{
  "model": "llama2",
  "prompt": "Is this email spam: You won $1000000"
}'
```

## 6. Run Orchestrated Analysis

```bash
# Using CLI
inbox-sentinel orchestrate -F email.txt --forwarded --llm-provider ollama

# Or run this demo
python examples/langchain_orchestration_demo.py
```

## Alternative: Using OpenAI

If you prefer using OpenAI instead of local models:

```bash
pip install openai
export OPENAI_API_KEY="your-api-key"

# Then modify the config:
config = OrchestrationConfig(
    llm_provider="openai",
    model_name="gpt-3.5-turbo",
    temperature=0.1
)
```
"""
    
    console.print(Markdown(instructions))


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--setup":
        show_langchain_setup_instructions()
    else:
        try:
            asyncio.run(demo_langchain_orchestration())
        except KeyboardInterrupt:
            console.print("\n[yellow]Demo interrupted[/yellow]")