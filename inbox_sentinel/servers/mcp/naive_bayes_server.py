"""
FastMCP server for Naive Bayes classifier
"""

from inbox_sentinel.servers.base import BaseMCPServer
from inbox_sentinel.ml.models import NaiveBayesDetector


class NaiveBayesServer(BaseMCPServer):
    """MCP server for Naive Bayes spam detection"""
    
    def __init__(self):
        detector = NaiveBayesDetector()
        super().__init__(
            server_name="Naive Bayes Classifier",
            detector=detector
        )


def main():
    """Run the Naive Bayes MCP server"""
    server = NaiveBayesServer()
    server.run()


if __name__ == "__main__":
    main()