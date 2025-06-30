"""
FastMCP server for Neural Network classifier
"""

from inbox_sentinel.servers.base import BaseMCPServer
from inbox_sentinel.ml.models import NeuralNetworkDetector


class NeuralNetworkServer(BaseMCPServer):
    """MCP server for Neural Network spam detection"""
    
    def __init__(self):
        detector = NeuralNetworkDetector()
        super().__init__(
            server_name="Neural Network Classifier",
            detector=detector
        )


def main():
    """Run the Neural Network MCP server"""
    server = NeuralNetworkServer()
    server.run()


if __name__ == "__main__":
    main()