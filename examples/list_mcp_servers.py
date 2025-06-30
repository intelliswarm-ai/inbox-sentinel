#!/usr/bin/env python3
"""
List all available MCP servers and their endpoints
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("INBOX SENTINEL - MCP SERVERS")
print("=" * 80)
print()

# Individual Model Servers
model_servers = [
    {
        "name": "Naive Bayes Server",
        "file": "inbox_sentinel/servers/mcp/naive_bayes_server.py",
        "description": "FastMCP server for Naive Bayes spam detection",
        "endpoints": [
            "analyze_email - Analyze email for spam/phishing",
            "train_model - Train the model with provided samples",
            "initialize_model - Initialize the model",
            "get_model_info - Get information about the current model"
        ]
    },
    {
        "name": "SVM Server",
        "file": "inbox_sentinel/servers/mcp/svm_server.py",
        "description": "FastMCP server for Support Vector Machine spam detection",
        "endpoints": [
            "analyze_email - Analyze email for spam/phishing",
            "train_model - Train the model with provided samples",
            "initialize_model - Initialize the model",
            "get_model_info - Get information about the current model"
        ]
    },
    {
        "name": "Random Forest Server",
        "file": "inbox_sentinel/servers/mcp/random_forest_server.py",
        "description": "FastMCP server for Random Forest spam detection",
        "endpoints": [
            "analyze_email - Analyze email for spam/phishing",
            "train_model - Train the model with provided samples",
            "initialize_model - Initialize the model",
            "get_model_info - Get information about the current model"
        ]
    },
    {
        "name": "Logistic Regression Server",
        "file": "inbox_sentinel/servers/mcp/logistic_regression_server.py",
        "description": "FastMCP server for Logistic Regression spam detection",
        "endpoints": [
            "analyze_email - Analyze email for spam/phishing",
            "train_model - Train the model with provided samples",
            "initialize_model - Initialize the model",
            "get_model_info - Get information about the current model"
        ]
    },
    {
        "name": "Neural Network Server",
        "file": "inbox_sentinel/servers/mcp/neural_network_server.py",
        "description": "FastMCP server for Neural Network spam detection",
        "endpoints": [
            "analyze_email - Analyze email for spam/phishing",
            "train_model - Train the model with provided samples",
            "initialize_model - Initialize the model",
            "get_model_info - Get information about the current model"
        ]
    }
]

# Orchestrator Server
orchestrator_server = {
    "name": "Orchestrator Server",
    "file": "inbox_sentinel/servers/mcp/orchestrator_server.py",
    "description": "FastMCP server for ensemble spam detection using multiple models",
    "endpoints": [
        "analyze_email_ensemble - Analyze email using ensemble of all available models",
        "train_all_models - Train all specified models with provided samples",
        "initialize_all_models - Initialize all specified models",
        "get_orchestrator_info - Get information about all available models",
        "compare_models - Compare predictions from all models for debugging"
    ]
}

# Print Individual Model Servers
print("INDIVIDUAL MODEL SERVERS")
print("-" * 80)
for server in model_servers:
    print(f"\n{server['name']}:")
    print(f"  File: {server['file']}")
    print(f"  Description: {server['description']}")
    print(f"  Endpoints:")
    for endpoint in server['endpoints']:
        print(f"    - {endpoint}")

# Print Orchestrator Server
print(f"\n{'=' * 80}")
print("ORCHESTRATOR SERVER (ENSEMBLE)")
print("-" * 80)
print(f"\n{orchestrator_server['name']}:")
print(f"  File: {orchestrator_server['file']}")
print(f"  Description: {orchestrator_server['description']}")
print(f"  Endpoints:")
for endpoint in orchestrator_server['endpoints']:
    print(f"    - {endpoint}")

print(f"\n{'=' * 80}")
print("HOW TO RUN THE SERVERS:")
print("-" * 80)
print()
print("To run any server, use one of these methods:")
print()
print("1. Direct execution:")
print("   python inbox_sentinel/servers/mcp/<server_name>.py")
print()
print("2. Module execution:")
print("   python -m inbox_sentinel.servers.mcp.<server_name>")
print()
print("3. Using the run script (if created):")
print("   python run_server.py <server_name>")
print()
print("Example:")
print("   python inbox_sentinel/servers/mcp/orchestrator_server.py")
print()
print("=" * 80)