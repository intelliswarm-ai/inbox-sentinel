"""
FastMCP server for SVM classifier
"""

from inbox_sentinel.servers.base import BaseMCPServer
from inbox_sentinel.ml.models import SVMDetector


class SVMServer(BaseMCPServer):
    """MCP server for Support Vector Machine spam detection"""
    
    def __init__(self):
        detector = SVMDetector()
        super().__init__(
            server_name="SVM Classifier",
            detector=detector
        )


def main():
    """Run the SVM MCP server"""
    server = SVMServer()
    server.run()


if __name__ == "__main__":
    main()