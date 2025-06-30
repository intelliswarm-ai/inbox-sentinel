#!/usr/bin/env python3
"""
Demo analyzer that shows the analyze feature working
Uses rule-based detection as a fallback when models aren't available
"""

import asyncio
from inbox_sentinel.core.types import Email, PredictionResult
from rich.console import Console
from rich.panel import Panel
import re

console = Console()

class SimpleRuleBasedDetector:
    """Simple rule-based detector for demo purposes"""
    
    def __init__(self):
        self.spam_keywords = [
            'winner', 'prize', 'congratulations', 'click here', 'limited time',
            'act now', 'urgent', 'verify account', 'suspended', 'blocked',
            'million', 'lottery', 'inheritance', 'prince', 'claim',
            'guarantee', 'risk free', 'viagra', 'pharmacy', 'casino'
        ]
        
        self.phishing_patterns = [
            r'http[s]?://[^\s]+\.(tk|ml|ga|cf)', # Suspicious TLDs
            r'[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}', # IP addresses
            r'bit\.ly|tinyurl|goo\.gl', # URL shorteners
            r'paypal-.*\.(com|net|org)', # Fake PayPal
            r'amazon-.*\.(com|net|org)', # Fake Amazon
        ]
    
    async def analyze(self, email: Email) -> PredictionResult:
        """Analyze email using simple rules"""
        
        # Combine all text
        full_text = f"{email.subject} {email.content} {email.sender}".lower()
        
        # Count spam keywords
        spam_score = 0
        found_keywords = []
        
        for keyword in self.spam_keywords:
            if keyword in full_text:
                spam_score += 1
                found_keywords.append(keyword)
        
        # Check phishing patterns
        phishing_score = 0
        for pattern in self.phishing_patterns:
            if re.search(pattern, full_text, re.IGNORECASE):
                phishing_score += 2
        
        # Check sender patterns
        if '@' in email.sender:
            domain = email.sender.split('@')[1].lower()
            if any(suspicious in domain for suspicious in ['tk', 'ml', 'ga', 'cf', '-verify', '-secure']):
                spam_score += 3
        
        # Calculate final score
        total_score = spam_score + phishing_score
        is_spam = total_score >= 3
        
        # Calculate probabilities
        spam_probability = min(0.95, total_score * 0.15)
        ham_probability = 1 - spam_probability
        
        # Create features list
        features = []
        if found_keywords:
            for kw in found_keywords[:5]:
                features.append({
                    'feature': f'keyword: {kw}',
                    'type': 'rule',
                    'importance': 0.8
                })
        
        return PredictionResult(
            model_name="rule-based",
            algorithm="Rule-Based Detection",
            is_spam=is_spam,
            spam_probability=spam_probability,
            ham_probability=ham_probability,
            confidence=abs(spam_probability - 0.5) * 2,
            features=features
        )

async def demo_analyze(content: str, subject: str, sender: str):
    """Demo the analyze functionality"""
    
    console.print("[bold cyan]Inbox Sentinel - Email Analysis Demo[/bold cyan]\n")
    
    # Create email object
    email = Email(content=content, subject=subject, sender=sender)
    
    # Create detector
    detector = SimpleRuleBasedDetector()
    
    # Analyze
    console.print("Analyzing email...")
    result = await detector.analyze(email)
    
    # Display results
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
    
    console.print("\n[dim]Note: This is using a simple rule-based detector for demo purposes.")
    console.print("For production use, train the ML models with: inbox-sentinel models train[/dim]")

if __name__ == "__main__":
    # Test with spam
    print("\n" + "="*60)
    print("TEST 1: Spam Email")
    print("="*60)
    asyncio.run(demo_analyze(
        content="Congratulations! You've won $1000000! Click here to claim your prize: http://scam-site.tk/claim",
        subject="You're a WINNER!",
        sender="lottery@fake-domain.tk"
    ))
    
    # Test with legitimate
    print("\n" + "="*60)
    print("TEST 2: Legitimate Email")
    print("="*60)
    asyncio.run(demo_analyze(
        content="Hi, I wanted to follow up on our meeting yesterday. Can we schedule a follow-up call for next week?",
        subject="Meeting follow-up",
        sender="john.doe@company.com"
    ))