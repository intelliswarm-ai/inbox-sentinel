"""
Command-line interface for Inbox Sentinel
"""

import click
from rich.console import Console
from rich.table import Table
from pathlib import Path

from inbox_sentinel import __version__
from inbox_sentinel.config import get_settings

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="Inbox Sentinel")
def cli():
    """Inbox Sentinel - Advanced Phishing Detection System"""
    pass


@cli.group()
def server():
    """Manage MCP servers"""
    pass


@server.command()
@click.argument('model', type=click.Choice([
    'naive-bayes', 'svm', 'random-forest', 
    'logistic-regression', 'neural-network', 'orchestrator'
]))
def start(model):
    """Start an MCP server for the specified model"""
    console.print(f"[bold green]Starting {model} server...[/bold green]")
    
    if model == 'naive-bayes':
        from inbox_sentinel.servers.mcp.naive_bayes_server import main
        main()
    elif model == 'svm':
        from inbox_sentinel.servers.mcp.svm_server import main
        main()
    elif model == 'random-forest':
        from inbox_sentinel.servers.mcp.random_forest_server import main
        main()
    elif model == 'logistic-regression':
        from inbox_sentinel.servers.mcp.logistic_regression_server import main
        main()
    elif model == 'neural-network':
        from inbox_sentinel.servers.mcp.neural_network_server import main
        main()
    elif model == 'orchestrator':
        from inbox_sentinel.servers.mcp.orchestrator_server import main
        main()


@cli.group()
def models():
    """Manage ML models"""
    pass


@models.command()
def list():
    """List all available models and their status"""
    settings = get_settings()
    
    table = Table(title="Inbox Sentinel Models")
    table.add_column("Model", style="cyan")
    table.add_column("Algorithm", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("Accuracy", style="yellow")
    table.add_column("File Size", style="blue")
    
    models_info = [
        ("naive-bayes", "Multinomial Naive Bayes", "naive_bayes_model.pkl", "96.25%"),
        ("svm", "Support Vector Machine", "svm_model.pkl", "95.75%"),
        ("random-forest", "Random Forest", "random_forest_model.pkl", "93.95%"),
        ("logistic-regression", "Logistic Regression", "logistic_regression_model.pkl", "95.75%"),
        ("neural-network", "Neural Network (MLP)", "neural_network_model.pkl", "96.60%"),
    ]
    
    for name, algorithm, filename, accuracy in models_info:
        filepath = settings.models_dir / filename
        if filepath.exists():
            status = "✅ Trained"
            size = f"{filepath.stat().st_size / 1024 / 1024:.1f} MB"
        else:
            status = "❌ Not trained"
            size = "N/A"
        
        table.add_row(name, algorithm, status, accuracy, size)
    
    console.print(table)


@models.command()
def train():
    """Train all models with the dataset"""
    console.print("[bold yellow]Training all models...[/bold yellow]")
    
    from inbox_sentinel.scripts.train_models import main as train_main
    train_main()


@models.command()
def verify():
    """Verify all trained models"""
    console.print("[bold cyan]Verifying trained models...[/bold cyan]")
    
    from inbox_sentinel.scripts.verify_models import main as verify_main
    verify_main()


@cli.command()
@click.option('--email-content', '-c', help='Email content')
@click.option('--subject', '-s', help='Email subject')
@click.option('--sender', '-f', help='Sender email address')
@click.option('--file', '-F', type=click.File('r', encoding='utf-8'), help='Read email from file')
@click.option('--forwarded', '-fw', is_flag=True, help='Parse as forwarded email')
@click.option('--model', '-m', 
              type=click.Choice(['naive-bayes', 'svm', 'random-forest', 
                               'logistic-regression', 'neural-network']),
              default='neural-network',
              help='Model to use for analysis')
def analyze(email_content, subject, sender, file, forwarded, model):
    """Analyze an email for phishing/spam"""
    import asyncio
    from inbox_sentinel.core.types import Email
    from inbox_sentinel.utils.email_parser import parse_gmail_forward
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    
    # Handle input sources
    if file:
        email_text = file.read()
        if forwarded:
            # Parse forwarded email
            email = parse_gmail_forward(email_text)
            console.print("[dim]Parsed forwarded email[/dim]")
            console.print(f"[dim]Original sender: {email.sender}[/dim]")
            console.print(f"[dim]Subject: {email.subject}[/dim]\n")
        else:
            # Treat file content as email body
            email = Email(
                content=email_text.strip(),
                subject=subject or "Email from file",
                sender=sender or "unknown@file.com"
            )
    elif email_content:
        if forwarded:
            # Parse the content as a forwarded email
            email = parse_gmail_forward(email_content)
            console.print("[dim]Parsed forwarded email[/dim]")
            console.print(f"[dim]Original sender: {email.sender}[/dim]")
            console.print(f"[dim]Subject: {email.subject}[/dim]\n")
        else:
            # Regular email from command line
            if not subject or not sender:
                console.print("[red]Error: --subject and --sender are required when not using --forwarded[/red]")
                return
            email = Email(content=email_content, subject=subject, sender=sender)
    else:
        console.print("[red]Error: Provide either --email-content or --file[/red]")
        return
    
    async def run_analysis():
        
        # Import the appropriate detector
        if model == 'naive-bayes':
            from inbox_sentinel.ml.models.naive_bayes import NaiveBayesDetector
            detector = NaiveBayesDetector()
        elif model == 'svm':
            from inbox_sentinel.ml.models.svm import SVMDetector
            detector = SVMDetector()
        elif model == 'random-forest':
            from inbox_sentinel.ml.models.random_forest import RandomForestDetector
            detector = RandomForestDetector()
        elif model == 'logistic-regression':
            from inbox_sentinel.ml.models.logistic_regression import LogisticRegressionDetector
            detector = LogisticRegressionDetector()
        elif model == 'neural-network':
            from inbox_sentinel.ml.models.neural_network import NeuralNetworkDetector
            detector = NeuralNetworkDetector()
        
        # Initialize detector
        result = None
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Initializing {model} model...", total=None)
            try:
                await detector.initialize(use_pretrained=True)
                progress.update(task, completed=True)
                
                # Check if model is actually trained
                if not detector.is_trained:
                    raise Exception("Model initialization succeeded but model is not trained")
                    
            except Exception as e:
                progress.update(task, completed=True)
                console.print(f"[red]Model not available: The {model} model needs to be trained[/red]")
                console.print(f"\n[yellow]The ML models need to be trained with the new package structure.[/yellow]")
                console.print("[yellow]Please run: [bold]inbox-sentinel models train[/bold][/yellow]")
                console.print("\n[dim]For now, showing a demo with rule-based detection:[/dim]\n")
                
                # Fall back to rule-based detection for demo
                from inbox_sentinel.core.types import PredictionResult
                import re
                
                # Simple rule-based analysis
                full_text = f"{email.subject} {email.content} {email.sender}".lower()
                spam_keywords = ['winner', 'prize', 'click here', 'urgent', 'suspended', 'million', 'lottery']
                spam_score = sum(1 for kw in spam_keywords if kw in full_text)
                is_spam = spam_score >= 2
                
                result = PredictionResult(
                    model_name="fallback",
                    algorithm="Rule-Based (Fallback)",
                    is_spam=is_spam,
                    spam_probability=min(0.95, spam_score * 0.2),
                    ham_probability=max(0.05, 1 - spam_score * 0.2),
                    confidence=min(0.9, spam_score * 0.3),
                    features=[{'feature': f'keyword: {kw}', 'type': 'rule'} 
                             for kw in spam_keywords if kw in full_text][:5]
                )
                
                progress.update(task, completed=True)
        
        # Analyze email (if not already done via fallback)
        if result is None:
            console.print(f"\n[bold cyan]Analyzing email...[/bold cyan]")
            try:
                result = await detector.analyze(email)
            except Exception as e:
                console.print(f"[red]Error during analysis: {e}[/red]")
                import traceback
                console.print(f"[dim]{traceback.format_exc()}[/dim]")
                return
        
        # Display results
        try:
            if result.error:
                console.print(f"[red]Analysis error: {result.error}[/red]")
                return
            
            # Determine status color and icon
            if result.is_spam:
                status_color = "red"
                status_icon = "⚠️"
                status_text = "SPAM/PHISHING DETECTED"
            else:
                status_color = "green"
                status_icon = "✅"
                status_text = "LEGITIMATE EMAIL"
            
            # Create result panel
            result_content = f"""
{status_icon} [bold {status_color}]{status_text}[/bold {status_color}]

[bold]Model:[/bold] {result.algorithm}
[bold]Confidence:[/bold] {result.confidence:.1%}
[bold]Spam Probability:[/bold] {result.spam_probability:.1%}
[bold]Ham Probability:[/bold] {result.ham_probability:.1%}
"""
            
            # Add top features if available
            if result.features:
                result_content += "\n[bold]Top Indicators:[/bold]"
                for i, feature in enumerate(result.features[:5], 1):
                    result_content += f"\n  {i}. {feature['feature']} ({feature['type']})"
            
            # Add metadata if available
            if result.metadata and model == 'neural-network':
                result_content += f"\n\n[bold]Network Details:[/bold]"
                result_content += f"\n  Hidden Layers: {result.metadata.get('hidden_layers', 'N/A')}"
                result_content += f"\n  Iterations: {result.metadata.get('n_iter', 'N/A')}"
                result_content += f"\n  Final Loss: {result.metadata.get('loss', 'N/A'):.4f}"
            
            console.print(Panel(
                result_content.strip(),
                title="Analysis Results",
                border_style=status_color
            ))
            
            # Add recommendation
            if result.is_spam:
                console.print("\n[yellow]⚠️  Recommendation: This email appears to be spam or phishing. Do not click any links or provide personal information.[/yellow]")
            else:
                console.print("\n[green]✅ This email appears to be legitimate based on the analysis.[/green]")
                
        except Exception as e:
            console.print(f"[red]Error displaying results: {e}[/red]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
    
    # Run the async function
    asyncio.run(run_analysis())


@cli.command()
@click.option('--email-content', '-c', help='Email content')
@click.option('--file', '-F', type=click.File('r', encoding='utf-8'), help='Read email from file')
@click.option('--forwarded', '-fw', is_flag=True, help='Parse as forwarded email')
@click.option('--llm-provider', type=click.Choice(['ollama', 'simple']), default='simple',
              help='LLM provider for orchestration (ollama requires local Ollama server)')
@click.option('--model-name', default='llama2', help='Model name for Ollama')
def orchestrate(email_content, file, forwarded, llm_provider, model_name):
    """Analyze email using LLM-orchestrated ensemble of models"""
    import asyncio
    from inbox_sentinel.core.types import Email
    from inbox_sentinel.utils.email_parser import parse_gmail_forward
    from inbox_sentinel.orchestration import SimpleOrchestrator, LangChainOrchestrator, OrchestrationConfig
    from rich.panel import Panel
    from rich.markdown import Markdown
    
    # Handle input sources (similar to analyze command)
    if file:
        email_text = file.read()
        if forwarded:
            email = parse_gmail_forward(email_text)
        else:
            email = Email(
                content=email_text.strip(),
                subject="Email from file",
                sender="unknown@file.com"
            )
    elif email_content:
        if forwarded:
            email = parse_gmail_forward(email_content)
        else:
            console.print("[red]Error: For direct content, use --forwarded or provide via --file[/red]")
            return
    else:
        console.print("[red]Error: Provide either --email-content or --file[/red]")
        return
    
    console.print(f"[bold cyan]Orchestrated Email Analysis[/bold cyan]\n")
    console.print(f"[dim]Subject: {email.subject}[/dim]")
    console.print(f"[dim]Sender: {email.sender}[/dim]\n")
    
    async def run_orchestration():
        try:
            if llm_provider == 'ollama':
                console.print("[yellow]Using Ollama LLM orchestration (requires Ollama server running)[/yellow]")
                config = OrchestrationConfig(
                    llm_provider="ollama",
                    model_name=model_name,
                    temperature=0.1,
                    verbose=True
                )
                orchestrator = LangChainOrchestrator(config)
                # Since we're in an async context, use the async method
                result = await orchestrator.analyze_email(email)
            else:
                console.print("[green]Using consensus-based orchestration[/green]")
                orchestrator = SimpleOrchestrator()
                result = await orchestrator.analyze_email(email)
            
            if result['success']:
                # Display analysis
                console.print(Panel(
                    Markdown(result['analysis']),
                    title="Orchestrated Analysis Result",
                    border_style="cyan"
                ))
                
                # Show detailed results if available
                if 'detailed_results' in result:
                    console.print("\n[bold]Detailed Model Results:[/bold]")
                    from rich.table import Table
                    
                    table = Table(show_header=True)
                    table.add_column("Model", style="cyan")
                    table.add_column("Verdict", style="bold")
                    table.add_column("Confidence", style="magenta")
                    table.add_column("Spam Prob", style="red")
                    
                    for model, res in result['detailed_results'].items():
                        if 'error' not in res:
                            verdict = "SPAM" if res['is_spam'] else "HAM"
                            color = "red" if res['is_spam'] else "green"
                            table.add_row(
                                model.replace('_', ' ').title(),
                                f"[{color}]{verdict}[/{color}]",
                                f"{res['confidence']:.1%}",
                                f"{res['spam_probability']:.1%}"
                            )
                    
                    console.print(table)
            else:
                console.print(f"[red]Orchestration failed: {result.get('error', 'Unknown error')}[/red]")
                
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
    
    # Run the orchestration
    asyncio.run(run_orchestration())


@cli.command()
def info():
    """Show system information"""
    settings = get_settings()
    
    console.print(f"[bold cyan]Inbox Sentinel v{__version__}[/bold cyan]")
    console.print(f"Project Root: {settings.project_root}")
    console.print(f"Models Directory: {settings.models_dir}")
    console.print(f"Data Directory: {settings.data_dir}")
    console.print(f"Use Pretrained Models: {settings.use_pretrained_models}")
    console.print(f"Max Training Samples: {settings.max_training_samples}")


def main():
    """Main entry point"""
    cli()


if __name__ == "__main__":
    main()