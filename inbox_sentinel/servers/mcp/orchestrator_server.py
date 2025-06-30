"""
FastMCP server for orchestrating multiple classifiers
"""

from typing import Dict, Any, List, Optional
from fastmcp import FastMCP, Context

from inbox_sentinel.core.types import Email, ConsensusResult
from inbox_sentinel.core.orchestrator import DetectionOrchestrator
from inbox_sentinel.core.constants import (
    STRATEGY_WEIGHTED_AVERAGE,
    STRATEGY_MAJORITY_VOTE,
    STRATEGY_HIGHEST_CONFIDENCE,
    STRATEGY_CONSERVATIVE,
    STRATEGY_AGGRESSIVE
)


class OrchestratorServer:
    """MCP server for ensemble spam detection"""
    
    def __init__(self):
        self.server_name = "Phishing Detection Orchestrator"
        self.mcp = FastMCP(self.server_name)
        self.orchestrator = DetectionOrchestrator()
        
        # Register tools
        self._register_tools()
    
    def _register_tools(self):
        """Register MCP tools"""
        
        @self.mcp.tool()
        async def analyze_email_ensemble(
            ctx: Context,
            email_content: str,
            subject: str,
            sender: str,
            strategy: str = STRATEGY_WEIGHTED_AVERAGE,
            models_to_use: Optional[List[str]] = None
        ) -> Dict[str, Any]:
            """
            Analyze email using ensemble of all available models.
            
            Parameters:
            - email_content: The email body
            - subject: Email subject line
            - sender: Sender email address
            - strategy: Consensus strategy (majority_vote, weighted_average, highest_confidence, conservative, aggressive)
            - models_to_use: List of model names to use (default: all)
            """
            
            email = Email(content=email_content, subject=subject, sender=sender)
            
            # Get ensemble prediction
            result = await self.orchestrator.analyze_ensemble(
                email=email,
                strategy=strategy,
                models_to_use=models_to_use
            )
            
            return {
                'success': True,
                'is_spam': result.is_spam,
                'confidence': result.confidence,
                'spam_probability': result.spam_probability,
                'ham_probability': result.ham_probability,
                'strategy_used': result.strategy,
                'models_used': result.models_used,
                'individual_predictions': result.individual_predictions,
                'errors': result.errors or [],
                'explanation': result.explanation
            }
        
        @self.mcp.tool()
        async def train_all_models(
            ctx: Context,
            training_samples: List[Dict[str, Any]],
            models_to_train: Optional[List[str]] = None
        ) -> Dict[str, Any]:
            """
            Train all specified models with provided samples.
            Each sample should have: email_content, subject, sender, is_spam (bool)
            """
            
            results = await self.orchestrator.train_all_models(
                training_samples=training_samples,
                models_to_train=models_to_train
            )
            
            # Summary statistics
            successful = sum(1 for r in results.values() if r.get('success', False))
            failed = len(results) - successful
            
            return {
                'success': successful > 0,
                'models_trained': successful,
                'models_failed': failed,
                'results': results,
                'summary': f"Successfully trained {successful}/{len(results)} models"
            }
        
        @self.mcp.tool()
        async def initialize_all_models(
            ctx: Context,
            use_pretrained: bool = True,
            models_to_init: Optional[List[str]] = None
        ) -> Dict[str, Any]:
            """
            Initialize all specified models.
            If use_pretrained is True, loads pre-trained models if available.
            """
            
            results = await self.orchestrator.initialize_all_models(
                use_pretrained=use_pretrained,
                models_to_init=models_to_init
            )
            
            return {
                'success': True,
                'models_initialized': len(results),
                'results': results
            }
        
        @self.mcp.tool()
        async def get_orchestrator_info(ctx: Context) -> Dict[str, Any]:
            """Get information about all available models in the orchestrator"""
            
            models_info = self.orchestrator.get_models_info()
            
            # Calculate statistics
            trained_models = sum(1 for info in models_info.values() if info.get('is_trained', False))
            
            return {
                'total_models': len(models_info),
                'trained_models': trained_models,
                'available_strategies': [
                    STRATEGY_MAJORITY_VOTE,
                    STRATEGY_WEIGHTED_AVERAGE,
                    STRATEGY_HIGHEST_CONFIDENCE,
                    STRATEGY_CONSERVATIVE,
                    STRATEGY_AGGRESSIVE
                ],
                'models': models_info
            }
        
        @self.mcp.tool()
        async def compare_models(
            ctx: Context,
            email_content: str,
            subject: str,
            sender: str
        ) -> Dict[str, Any]:
            """
            Compare predictions from all models for a given email.
            Useful for debugging and understanding model behavior.
            """
            
            email = Email(content=email_content, subject=subject, sender=sender)
            
            # Get predictions from all models with different strategies
            strategies = [
                STRATEGY_MAJORITY_VOTE,
                STRATEGY_WEIGHTED_AVERAGE,
                STRATEGY_HIGHEST_CONFIDENCE,
                STRATEGY_CONSERVATIVE,
                STRATEGY_AGGRESSIVE
            ]
            
            comparisons = {}
            
            for strategy in strategies:
                result = await self.orchestrator.analyze_ensemble(
                    email=email,
                    strategy=strategy
                )
                
                comparisons[strategy] = {
                    'is_spam': result.is_spam,
                    'confidence': result.confidence,
                    'spam_probability': result.spam_probability
                }
            
            # Get individual model predictions
            base_result = await self.orchestrator.analyze_ensemble(
                email=email,
                strategy=STRATEGY_MAJORITY_VOTE
            )
            
            return {
                'individual_models': base_result.individual_predictions,
                'strategy_comparisons': comparisons,
                'unanimous': all(p['is_spam'] == base_result.individual_predictions[0]['is_spam'] 
                               for p in base_result.individual_predictions),
                'agreement_rate': sum(1 for p in base_result.individual_predictions 
                                    if p['is_spam'] == base_result.is_spam) / len(base_result.individual_predictions)
            }
    
    def run(self):
        """Run the MCP server"""
        self.mcp.run()


def main():
    """Run the Orchestrator MCP server"""
    server = OrchestratorServer()
    server.run()


if __name__ == "__main__":
    main()