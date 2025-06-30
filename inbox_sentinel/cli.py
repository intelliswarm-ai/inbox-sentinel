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
@click.option('--email-content', '-c', required=True, help='Email content')
@click.option('--subject', '-s', required=True, help='Email subject')
@click.option('--sender', '-f', required=True, help='Sender email address')
@click.option('--model', '-m', 
              type=click.Choice(['naive-bayes', 'svm', 'random-forest', 
                               'logistic-regression', 'neural-network']),
              default='neural-network',
              help='Model to use for analysis')
def analyze(email_content, subject, sender, model):
    """Analyze an email for phishing/spam"""
    console.print(f"[bold]Analyzing email with {model}...[/bold]")
    
    # Import and use the appropriate detector
    # This would be implemented to load the model and analyze
    console.print("[yellow]Analysis feature coming soon![/yellow]")


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