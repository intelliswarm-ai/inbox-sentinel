#!/usr/bin/env python3
"""
Test MCP server startup and model loading
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from inbox_sentinel.servers.mcp.neural_network_server import NeuralNetworkServer


def test_server_init():
    """Test if MCP server loads model on init"""
    print("Creating Neural Network MCP Server...")
    
    # This should automatically load the model
    server = NeuralNetworkServer()
    
    print(f"Server created: {server.server_name}")
    print(f"Detector algorithm: {server.detector.algorithm}")
    print(f"Model trained: {server.detector.is_trained}")
    
    if server.detector.is_trained:
        print("✅ Model successfully loaded on server startup!")
    else:
        print("❌ Model not loaded on server startup")


if __name__ == "__main__":
    test_server_init()