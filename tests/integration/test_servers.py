#!/usr/bin/env python3
"""
Test script to verify all MCP servers can be imported and initialized
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from inbox_sentinel.servers.mcp import (
    NaiveBayesServer,
    SVMServer,
    RandomForestServer,
    LogisticRegressionServer,
    NeuralNetworkServer,
    OrchestratorServer
)


async def test_servers():
    """Test that all servers can be initialized"""
    servers = [
        ("Naive Bayes", NaiveBayesServer),
        ("SVM", SVMServer),
        ("Random Forest", RandomForestServer),
        ("Logistic Regression", LogisticRegressionServer),
        ("Neural Network", NeuralNetworkServer),
        ("Orchestrator", OrchestratorServer),
    ]
    
    print("Testing MCP server initialization...\n")
    
    for name, server_class in servers:
        try:
            server = server_class()
            print(f"✓ {name} Server: Successfully initialized")
        except Exception as e:
            print(f"✗ {name} Server: Failed to initialize - {e}")
    
    print("\nAll server initialization tests complete!")


if __name__ == "__main__":
    asyncio.run(test_servers())