"""
FastMCP server for Random Forest classifier
"""

from inbox_sentinel.servers.base import BaseMCPServer
from inbox_sentinel.ml.models import RandomForestDetector


class RandomForestServer(BaseMCPServer):
    """MCP server for Random Forest spam detection"""
    
    def __init__(self):
        detector = RandomForestDetector()
        super().__init__(
            server_name="Random Forest Classifier",
            detector=detector
        )


def main():
    """Run the Random Forest MCP server"""
    server = RandomForestServer()
    server.run()


if __name__ == "__main__":
    main()