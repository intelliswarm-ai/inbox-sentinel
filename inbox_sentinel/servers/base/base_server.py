"""
Base class for MCP servers
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from fastmcp import FastMCP, Context
import logging

from inbox_sentinel.core.types import Email, PredictionResult
from inbox_sentinel.core.base_detector import BaseDetector


class BaseMCPServer(ABC):
    """Base class for MCP server implementations"""
    
    def __init__(self, server_name: str, detector: BaseDetector):
        self.server_name = server_name
        self.detector = detector
        self.mcp = FastMCP(server_name)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Register tools
        self._register_tools()
    
    def _register_tools(self):
        """Register MCP tools"""
        # Analyze email tool
        @self.mcp.tool()
        async def analyze_email(
            ctx: Context,
            email_content: str,
            subject: str,
            sender: str
        ) -> Dict[str, Any]:
            """Analyze email for spam/phishing detection"""
            email = Email(content=email_content, subject=subject, sender=sender)
            result = await self.detector.analyze(email)
            
            return {
                'algorithm': result.algorithm,
                'is_spam': result.is_spam,
                'spam_probability': result.spam_probability,
                'ham_probability': result.ham_probability,
                'confidence': result.confidence,
                'top_features': result.features if result.features else [],
                'error': result.error
            }
        
        # Train model tool
        @self.mcp.tool()
        async def train_model(
            ctx: Context,
            training_samples: List[Dict[str, Any]]
        ) -> Dict[str, Any]:
            """Train the model with provided samples"""
            return await self.detector.train(training_samples)
        
        # Initialize model tool
        @self.mcp.tool()
        async def initialize_model(
            ctx: Context,
            use_pretrained: bool = True
        ) -> Dict[str, str]:
            """Initialize the model"""
            return await self.detector.initialize(use_pretrained)
        
        # Get model info tool
        @self.mcp.tool()
        async def get_model_info(ctx: Context) -> Dict[str, Any]:
            """Get information about the current model"""
            info = await self.detector.get_info()
            
            return {
                'model_name': info.model_name,
                'algorithm': info.algorithm,
                'is_trained': info.is_trained,
                'parameters': info.parameters
            }
    
    def run(self):
        """Run the MCP server"""
        self.mcp.run()