#!/usr/bin/env python3
"""
AI Agentic Workflow for Phishing Email Detection
Orchestrates multiple ML tools via MCP and handles conflicting analyses
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from mcp.client import StdioServerParameters
from mcp.client.stdio import stdio_client


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalysisStrategy(Enum):
    """Strategies for handling conflicting analyses"""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_CONSENSUS = "weighted_consensus"
    HIGH_CONFIDENCE_FIRST = "high_confidence_first"
    CONSERVATIVE = "conservative"  # If any detector says phishing, flag it
    AGGRESSIVE = "aggressive"  # Only flag if all detectors agree


@dataclass
class EmailAnalysis:
    """Result of email analysis"""
    is_phishing: bool
    confidence: float
    strategy_used: str
    individual_results: List[Dict[str, Any]]
    consensus_details: Dict[str, Any]
    final_indicators: List[str]


class PhishingDetectionWorkflow:
    """
    Orchestrates phishing detection using multiple ML tools via MCP
    Handles tool switching and consensus building
    """
    
    def __init__(self, strategy: AnalysisStrategy = AnalysisStrategy.WEIGHTED_CONSENSUS):
        self.strategy = strategy
        self.session: Optional[ClientSession] = None
        self.available_tools: List[str] = []
        self.tool_performance: Dict[str, Dict[str, float]] = {}
        
    async def connect(self):
        """Connect to the MCP server"""
        server_params = StdioServerParameters(
            command="python",
            args=["mcp_server.py"]
        )
        
        # Use the context manager properly
        self._client_ctx = stdio_client(server_params)
        transport = await self._client_ctx.__aenter__()
        read_stream, write_stream = transport
        
        from mcp.client import ClientSession
        self.session = ClientSession(read_stream, write_stream)
        
        # Initialize the session
        await self.session.initialize()
        
        # Get available tools
        tools_response = await self.session.list_tools()
        self.available_tools = [tool.name for tool in tools_response.tools]
        logger.info(f"Connected to MCP server. Available tools: {self.available_tools}")
        
    async def disconnect(self):
        """Disconnect from the MCP server"""
        if hasattr(self, '_client_ctx'):
            await self._client_ctx.__aexit__(None, None, None)
            
    async def analyze_email(self, email_content: str, subject: str, sender: str) -> EmailAnalysis:
        """
        Analyze an email for phishing using the workflow
        """
        if not self.session:
            raise RuntimeError("Not connected to MCP server")
            
        # First, try ensemble analysis
        ensemble_result = await self._run_ensemble_analysis(email_content, subject, sender)
        
        # Check if we need to switch strategies based on consensus
        if self._should_switch_strategy(ensemble_result):
            logger.info("Low consensus detected, switching to individual analysis approach")
            return await self._adaptive_analysis(email_content, subject, sender, ensemble_result)
        
        # Use standard strategy
        return self._apply_strategy(ensemble_result)
        
    async def _run_ensemble_analysis(self, email_content: str, subject: str, sender: str) -> Dict[str, Any]:
        """Run ensemble analysis using all detectors"""
        result = await self.session.call_tool(
            "analyze_ensemble",
            arguments={
                "email_content": email_content,
                "subject": subject,
                "sender": sender
            }
        )
        
        return json.loads(result.content[0].text)
        
    async def _run_individual_analysis(self, tool_name: str, email_content: str, 
                                     subject: str, sender: str) -> Dict[str, Any]:
        """Run analysis with a specific detector"""
        result = await self.session.call_tool(
            tool_name,
            arguments={
                "email_content": email_content,
                "subject": subject,
                "sender": sender
            }
        )
        
        return json.loads(result.content[0].text)
        
    def _should_switch_strategy(self, ensemble_result: Dict[str, Any]) -> bool:
        """Determine if we should switch to adaptive analysis"""
        ensemble = ensemble_result["ensemble_result"]
        individual = ensemble_result["individual_results"]
        
        # Switch if consensus is low and confidence varies significantly
        if ensemble["consensus_level"] == "low":
            confidences = [r["confidence"] for r in individual]
            confidence_variance = max(confidences) - min(confidences)
            return confidence_variance > 0.3
            
        return False
        
    async def _adaptive_analysis(self, email_content: str, subject: str, sender: str,
                               initial_result: Dict[str, Any]) -> EmailAnalysis:
        """
        Adaptive analysis that switches between tools based on results
        """
        individual_results = initial_result["individual_results"]
        
        # Find detectors with conflicting high-confidence results
        high_confidence_conflicts = []
        for i, result1 in enumerate(individual_results):
            if result1["confidence"] > 0.8:
                for j, result2 in enumerate(individual_results[i+1:], i+1):
                    if result2["confidence"] > 0.8 and result1["is_phishing"] != result2["is_phishing"]:
                        high_confidence_conflicts.append((result1, result2))
                        
        # If we have high-confidence conflicts, run additional targeted analysis
        if high_confidence_conflicts:
            logger.info(f"Found {len(high_confidence_conflicts)} high-confidence conflicts")
            
            # Identify which indicators are causing conflicts
            all_indicators = set()
            for result in individual_results:
                all_indicators.update(result.get("indicators", []))
                
            # Re-run specific detectors with focus on conflicting indicators
            refined_results = []
            for tool_name in ["analyze_with_url_pattern", "analyze_with_content", 
                            "analyze_with_bayesian", "analyze_with_header"]:
                if tool_name in self.available_tools:
                    try:
                        result = await self._run_individual_analysis(
                            tool_name, email_content, subject, sender
                        )
                        refined_results.append(result)
                    except Exception as e:
                        logger.error(f"Error running {tool_name}: {e}")
                        
            # Combine initial and refined results
            all_results = individual_results + refined_results
            
            # Apply conflict resolution
            return self._resolve_conflicts(all_results, all_indicators)
            
        # No high-confidence conflicts, use standard strategy
        return self._apply_strategy(initial_result)
        
    def _resolve_conflicts(self, all_results: List[Dict[str, Any]], 
                         indicators: set) -> EmailAnalysis:
        """Resolve conflicts between detector results"""
        # Group results by detector
        detector_groups = {}
        for result in all_results:
            detector = result["detector"]
            if detector not in detector_groups:
                detector_groups[detector] = []
            detector_groups[detector].append(result)
            
        # Take the most recent result from each detector
        final_results = []
        for detector, results in detector_groups.items():
            final_results.append(results[-1])  # Most recent analysis
            
        # Apply weighted voting with indicator analysis
        weighted_scores = []
        for result in final_results:
            # Boost confidence if multiple indicators agree
            indicator_boost = min(0.2, len(result.get("indicators", [])) * 0.05)
            adjusted_confidence = min(0.95, result["confidence"] + indicator_boost)
            
            weighted_scores.append({
                "score": result["score"],
                "confidence": adjusted_confidence,
                "is_phishing": result["is_phishing"],
                "detector": result["detector"]
            })
            
        # Calculate final decision
        total_weight = sum(ws["confidence"] for ws in weighted_scores)
        weighted_phishing_score = sum(
            ws["score"] * ws["confidence"] for ws in weighted_scores
        ) / total_weight
        
        is_phishing = weighted_phishing_score > 0.5
        
        # Collect all unique indicators
        all_indicators = []
        for result in final_results:
            all_indicators.extend(result.get("indicators", []))
            
        unique_indicators = list(set(all_indicators))
        
        return EmailAnalysis(
            is_phishing=is_phishing,
            confidence=min(0.95, sum(ws["confidence"] for ws in weighted_scores) / len(weighted_scores)),
            strategy_used="adaptive_conflict_resolution",
            individual_results=final_results,
            consensus_details={
                "weighted_score": weighted_phishing_score,
                "total_analyses": len(all_results),
                "unique_detectors": len(detector_groups),
                "conflict_resolved": True
            },
            final_indicators=unique_indicators[:10]  # Top 10 indicators
        )
        
    def _apply_strategy(self, ensemble_result: Dict[str, Any]) -> EmailAnalysis:
        """Apply the configured strategy to determine final result"""
        ensemble = ensemble_result["ensemble_result"]
        individual = ensemble_result["individual_results"]
        
        if self.strategy == AnalysisStrategy.MAJORITY_VOTE:
            is_phishing = ensemble["is_phishing"]
            confidence = ensemble["confidence"]
            
        elif self.strategy == AnalysisStrategy.WEIGHTED_CONSENSUS:
            is_phishing = ensemble["weighted_score"] > 0.5
            confidence = ensemble["confidence"]
            
        elif self.strategy == AnalysisStrategy.HIGH_CONFIDENCE_FIRST:
            # Use the result with highest confidence
            highest_conf = max(individual, key=lambda x: x["confidence"])
            is_phishing = highest_conf["is_phishing"]
            confidence = highest_conf["confidence"]
            
        elif self.strategy == AnalysisStrategy.CONSERVATIVE:
            # Flag as phishing if any detector says so with decent confidence
            is_phishing = any(r["is_phishing"] and r["confidence"] > 0.6 for r in individual)
            confidence = max(r["confidence"] for r in individual if r["is_phishing"]) if is_phishing else ensemble["confidence"]
            
        elif self.strategy == AnalysisStrategy.AGGRESSIVE:
            # Only flag if strong consensus
            phishing_votes = sum(1 for r in individual if r["is_phishing"])
            is_phishing = phishing_votes == len(individual)
            confidence = ensemble["confidence"] if is_phishing else 1.0 - ensemble["confidence"]
            
        # Collect all indicators
        all_indicators = []
        for result in individual:
            all_indicators.extend(result.get("indicators", []))
            
        unique_indicators = list(set(all_indicators))
        
        return EmailAnalysis(
            is_phishing=is_phishing,
            confidence=confidence,
            strategy_used=self.strategy.value,
            individual_results=individual,
            consensus_details=ensemble,
            final_indicators=unique_indicators[:10]
        )
        
    def update_tool_performance(self, tool_name: str, was_correct: bool):
        """Update performance metrics for a tool based on feedback"""
        if tool_name not in self.tool_performance:
            self.tool_performance[tool_name] = {"correct": 0, "total": 0}
            
        self.tool_performance[tool_name]["total"] += 1
        if was_correct:
            self.tool_performance[tool_name]["correct"] += 1
            
        # Calculate accuracy
        perf = self.tool_performance[tool_name]
        perf["accuracy"] = perf["correct"] / perf["total"]
        
        logger.info(f"Updated {tool_name} performance: {perf['accuracy']:.2%} accuracy")


async def main():
    """Example usage of the phishing detection workflow"""
    
    # Example phishing email
    phishing_email = {
        "sender": "security@paypaI.com",  # Note the capital I instead of l
        "subject": "Urgent: Verify Your Account Within 24 Hours",
        "content": """Dear Customer,

We have detected suspicious activity on your PayPal account. Your account has been temporarily suspended for your protection.

To restore access to your account, please click here immediately: http://bit.ly/paypal-verify

You must verify your account within 24 hours or it will be permanently closed.

Thank you,
PayPal Security Team"""
    }
    
    # Example legitimate email
    legitimate_email = {
        "sender": "newsletter@company.com",
        "subject": "Your Monthly Newsletter",
        "content": """Hello,

Here's your monthly update with the latest news and updates from our company.

Check out our new blog post about cybersecurity best practices at https://company.com/blog/security

Best regards,
The Company Team"""
    }
    
    # Create workflow
    workflow = PhishingDetectionWorkflow(strategy=AnalysisStrategy.WEIGHTED_CONSENSUS)
    
    try:
        # Connect to MCP server
        await workflow.connect()
        
        # Analyze phishing email
        print("Analyzing suspected phishing email...")
        phishing_result = await workflow.analyze_email(
            phishing_email["content"],
            phishing_email["subject"],
            phishing_email["sender"]
        )
        
        print(f"\nResult: {'PHISHING' if phishing_result.is_phishing else 'LEGITIMATE'}")
        print(f"Confidence: {phishing_result.confidence:.2%}")
        print(f"Strategy: {phishing_result.strategy_used}")
        print(f"Indicators: {', '.join(phishing_result.final_indicators[:3])}")
        
        # Analyze legitimate email
        print("\n" + "="*50 + "\n")
        print("Analyzing legitimate email...")
        legit_result = await workflow.analyze_email(
            legitimate_email["content"],
            legitimate_email["subject"],
            legitimate_email["sender"]
        )
        
        print(f"\nResult: {'PHISHING' if legit_result.is_phishing else 'LEGITIMATE'}")
        print(f"Confidence: {legit_result.confidence:.2%}")
        print(f"Strategy: {legit_result.strategy_used}")
        
    finally:
        await workflow.disconnect()


if __name__ == "__main__":
    asyncio.run(main())