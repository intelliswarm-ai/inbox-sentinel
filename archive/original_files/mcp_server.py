#!/usr/bin/env python3
"""
MCP Server for Phishing Detection Tools
Provides multiple ML-based phishing detection tools with varying accuracy rates

PRIVACY NOTICE: This server processes all data locally and does not send any
email content, subjects, or sender information to external services or APIs.
All analysis is performed using local algorithms and pattern matching.
"""

import json
import asyncio
from typing import Dict, Any, List
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.types import Tool, TextContent
import mcp.server.stdio
import random
import re
import base64


class PhishingDetector:
    """Base class for phishing detection algorithms"""
    
    def __init__(self, name: str, accuracy: float, false_positive_rate: float):
        self.name = name
        self.accuracy = accuracy
        self.false_positive_rate = false_positive_rate
    
    def analyze(self, email_content: str, subject: str, sender: str) -> Dict[str, Any]:
        """Analyze email for phishing indicators"""
        raise NotImplementedError


class URLPatternDetector(PhishingDetector):
    """Detects phishing based on suspicious URL patterns"""
    
    def __init__(self):
        super().__init__("URL Pattern Detector", 0.85, 0.12)
        self.suspicious_patterns = [
            r'bit\.ly|tinyurl|short\.link',  # URL shorteners
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',  # IP addresses
            r'[a-z]+-[a-z]+\.(tk|ml|ga|cf)',  # Suspicious TLDs
            r'(paypal|amazon|bank).*\.(tk|ml|ga|cf|xyz)',  # Impersonation
            r'[0-9]{5,}\.com',  # Numeric domains
        ]
    
    def analyze(self, email_content: str, subject: str, sender: str) -> Dict[str, Any]:
        score = 0.0
        indicators = []
        
        # Check for suspicious URLs
        for pattern in self.suspicious_patterns:
            if re.search(pattern, email_content, re.IGNORECASE):
                score += 0.2
                indicators.append(f"Suspicious URL pattern: {pattern}")
        
        # Check sender domain
        if '@' in sender:
            domain = sender.split('@')[1]
            if re.search(r'[0-9]{3,}', domain):
                score += 0.15
                indicators.append("Sender domain contains numbers")
        
        # Simulate some randomness based on accuracy
        if random.random() > self.accuracy:
            score = 1.0 - score  # Flip the result occasionally
        
        is_phishing = score > 0.5
        confidence = min(0.95, abs(score - 0.5) * 2)
        
        return {
            "detector": self.name,
            "is_phishing": is_phishing,
            "confidence": confidence,
            "score": score,
            "indicators": indicators
        }


class ContentAnalyzer(PhishingDetector):
    """Analyzes email content for phishing keywords and patterns"""
    
    def __init__(self):
        super().__init__("Content Analyzer", 0.88, 0.10)
        self.phishing_keywords = [
            "urgent", "verify account", "suspended", "click here immediately",
            "confirm identity", "update payment", "security alert",
            "winner", "congratulations", "claim prize", "act now"
        ]
        self.urgency_phrases = [
            "within 24 hours", "expires today", "immediate action",
            "account will be closed", "last chance"
        ]
    
    def analyze(self, email_content: str, subject: str, sender: str) -> Dict[str, Any]:
        score = 0.0
        indicators = []
        
        content_lower = (email_content + " " + subject).lower()
        
        # Check for phishing keywords
        for keyword in self.phishing_keywords:
            if keyword in content_lower:
                score += 0.1
                indicators.append(f"Phishing keyword: {keyword}")
        
        # Check for urgency
        for phrase in self.urgency_phrases:
            if phrase in content_lower:
                score += 0.15
                indicators.append(f"Urgency phrase: {phrase}")
        
        # Check for ALL CAPS abuse
        caps_ratio = sum(1 for c in email_content if c.isupper()) / max(len(email_content), 1)
        if caps_ratio > 0.3:
            score += 0.1
            indicators.append("Excessive use of capital letters")
        
        # Simulate accuracy
        if random.random() > self.accuracy:
            score = 1.0 - score
        
        is_phishing = score > 0.4
        confidence = min(0.95, abs(score - 0.4) * 2.5)
        
        return {
            "detector": self.name,
            "is_phishing": is_phishing,
            "confidence": confidence,
            "score": score,
            "indicators": indicators
        }


class BayesianFilter(PhishingDetector):
    """Bayesian spam/phishing filter"""
    
    def __init__(self):
        super().__init__("Bayesian Filter", 0.92, 0.08)
        # Simplified token probabilities
        self.phishing_tokens = {
            "click": 0.8, "verify": 0.85, "suspended": 0.9,
            "congratulations": 0.75, "urgent": 0.7, "prize": 0.8,
            "paypal": 0.6, "ebay": 0.6, "bank": 0.65,
            "http": 0.4, "https": 0.3, ".com": 0.3
        }
    
    def analyze(self, email_content: str, subject: str, sender: str) -> Dict[str, Any]:
        indicators = []
        
        # Tokenize content
        tokens = re.findall(r'\b\w+\b', (email_content + " " + subject).lower())
        
        # Calculate spam probability using simplified Bayes
        spam_score = 0.5  # Prior probability
        ham_score = 0.5
        
        for token in tokens[:50]:  # Limit to first 50 tokens
            if token in self.phishing_tokens:
                p_spam = self.phishing_tokens[token]
                p_ham = 1 - p_spam
                
                spam_score *= p_spam
                ham_score *= p_ham
                
                if p_spam > 0.7:
                    indicators.append(f"High-risk token: {token}")
        
        # Normalize
        total = spam_score + ham_score
        if total > 0:
            spam_probability = spam_score / total
        else:
            spam_probability = 0.5
        
        # Simulate accuracy
        if random.random() > self.accuracy:
            spam_probability = 1.0 - spam_probability
        
        is_phishing = spam_probability > 0.6
        
        return {
            "detector": self.name,
            "is_phishing": is_phishing,
            "confidence": min(0.95, abs(spam_probability - 0.5) * 2),
            "score": spam_probability,
            "indicators": indicators
        }


class HeaderAnalyzer(PhishingDetector):
    """Analyzes email headers for spoofing and suspicious patterns"""
    
    def __init__(self):
        super().__init__("Header Analyzer", 0.90, 0.09)
    
    def analyze(self, email_content: str, subject: str, sender: str) -> Dict[str, Any]:
        score = 0.0
        indicators = []
        
        # Check sender spoofing patterns
        if '@' in sender:
            domain = sender.split('@')[1]
            username = sender.split('@')[0]
            
            # Check for lookalike domains
            legitimate_domains = ['paypal.com', 'amazon.com', 'google.com', 'microsoft.com']
            for legit in legitimate_domains:
                if legit.replace('.com', '') in domain and domain != legit:
                    score += 0.3
                    indicators.append(f"Lookalike domain detected: {domain} impersonating {legit}")
            
            # Check for display name tricks
            if 'no-reply' in username or 'noreply' in username:
                score += 0.1
                indicators.append("No-reply sender address")
        
        # Check subject patterns
        if re.search(r'^\[.*\]', subject):
            score += 0.05
            indicators.append("Subject contains bracketed prefix")
        
        if len(subject) > 100:
            score += 0.1
            indicators.append("Unusually long subject line")
        
        # Simulate accuracy
        if random.random() > self.accuracy:
            score = 1.0 - score
        
        is_phishing = score > 0.3
        confidence = min(0.95, score * 2)
        
        return {
            "detector": self.name,
            "is_phishing": is_phishing,
            "confidence": confidence,
            "score": score,
            "indicators": indicators
        }


# Initialize detectors
detectors = {
    "url_pattern": URLPatternDetector(),
    "content": ContentAnalyzer(),
    "bayesian": BayesianFilter(),
    "header": HeaderAnalyzer()
}

# Initialize MCP server
server = Server("phishing-detection")


@server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """List all available phishing detection tools"""
    tools = []
    
    for detector_id, detector in detectors.items():
        tools.append(Tool(
            name=f"analyze_with_{detector_id}",
            description=f"Analyze email using {detector.name} (Accuracy: {detector.accuracy*100:.0f}%, FPR: {detector.false_positive_rate*100:.0f}%)",
            inputSchema={
                "type": "object",
                "properties": {
                    "email_content": {
                        "type": "string",
                        "description": "The email body content"
                    },
                    "subject": {
                        "type": "string", 
                        "description": "The email subject line"
                    },
                    "sender": {
                        "type": "string",
                        "description": "The sender email address"
                    }
                },
                "required": ["email_content", "subject", "sender"]
            }
        ))
    
    # Add ensemble analysis tool
    tools.append(Tool(
        name="analyze_ensemble",
        description="Run all detectors and provide ensemble analysis",
        inputSchema={
            "type": "object",
            "properties": {
                "email_content": {
                    "type": "string",
                    "description": "The email body content"
                },
                "subject": {
                    "type": "string",
                    "description": "The email subject line" 
                },
                "sender": {
                    "type": "string",
                    "description": "The sender email address"
                }
            },
            "required": ["email_content", "subject", "sender"]
        }
    ))
    
    return tools


@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls for phishing analysis"""
    
    email_content = arguments.get("email_content", "")
    subject = arguments.get("subject", "")
    sender = arguments.get("sender", "")
    
    if name == "analyze_ensemble":
        # Run all detectors
        results = []
        for detector_id, detector in detectors.items():
            result = detector.analyze(email_content, subject, sender)
            results.append(result)
        
        # Calculate ensemble decision
        phishing_votes = sum(1 for r in results if r["is_phishing"])
        total_votes = len(results)
        
        # Weighted voting based on confidence
        weighted_score = sum(
            r["score"] * r["confidence"] for r in results
        ) / sum(r["confidence"] for r in results)
        
        ensemble_decision = {
            "ensemble_result": {
                "is_phishing": phishing_votes > total_votes / 2,
                "confidence": sum(r["confidence"] for r in results) / total_votes,
                "weighted_score": weighted_score,
                "votes": f"{phishing_votes}/{total_votes}",
                "consensus_level": "high" if phishing_votes == 0 or phishing_votes == total_votes else "low"
            },
            "individual_results": results
        }
        
        return [TextContent(
            type="text",
            text=json.dumps(ensemble_decision, indent=2)
        )]
    
    # Individual detector analysis
    detector_id = name.replace("analyze_with_", "")
    if detector_id in detectors:
        detector = detectors[detector_id]
        result = detector.analyze(email_content, subject, sender)
        
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    
    return [TextContent(
        type="text",
        text=f"Unknown tool: {name}"
    )]


async def main():
    """Run the MCP server"""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="phishing-detection",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )


if __name__ == "__main__":
    asyncio.run(main())