"""
FastMCP server for Logistic Regression classifier
"""

from inbox_sentinel.servers.base import BaseMCPServer
from inbox_sentinel.ml.models import LogisticRegressionDetector


class LogisticRegressionServer(BaseMCPServer):
    """MCP server for Logistic Regression spam detection"""
    
    def __init__(self):
        detector = LogisticRegressionDetector()
        super().__init__(
            server_name="Logistic Regression Classifier",
            detector=detector
        )


def main():
    """Run the Logistic Regression MCP server"""
    server = LogisticRegressionServer()
    server.run()


if __name__ == "__main__":
    main()