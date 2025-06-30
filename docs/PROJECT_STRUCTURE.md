# Project Structure

```
inbox-sentinel/
│
├── inbox_sentinel/              # Main package directory
│   ├── __init__.py             # Package initialization
│   ├── cli.py                  # Command-line interface
│   │
│   ├── core/                   # Core functionality
│   │   ├── __init__.py
│   │   ├── base_detector.py    # Abstract base class for detectors
│   │   ├── constants.py        # Project-wide constants
│   │   ├── exceptions.py       # Custom exceptions
│   │   └── types.py           # Type definitions and dataclasses
│   │
│   ├── config/                 # Configuration management
│   │   ├── __init__.py
│   │   ├── settings.py         # Pydantic settings with env support
│   │   └── model_config.py     # Model-specific configurations
│   │
│   ├── ml/                     # Machine learning components
│   │   ├── __init__.py
│   │   ├── preprocessing/      # Data preprocessing
│   │   │   ├── __init__.py
│   │   │   ├── email_preprocessor.py
│   │   │   └── feature_extractors.py
│   │   │
│   │   ├── models/             # ML model implementations
│   │   │   ├── __init__.py
│   │   │   ├── naive_bayes.py
│   │   │   ├── svm.py
│   │   │   ├── random_forest.py
│   │   │   ├── logistic_regression.py
│   │   │   └── neural_network.py
│   │   │
│   │   └── training/           # Training utilities
│   │       ├── __init__.py
│   │       ├── trainer.py
│   │       └── dataset_loader.py
│   │
│   ├── servers/                # MCP server implementations
│   │   ├── __init__.py
│   │   ├── base/               # Base server functionality
│   │   │   ├── __init__.py
│   │   │   └── base_server.py
│   │   │
│   │   └── mcp/                # FastMCP servers
│   │       ├── __init__.py
│   │       ├── naive_bayes_server.py
│   │       ├── svm_server.py
│   │       ├── random_forest_server.py
│   │       ├── logistic_regression_server.py
│   │       ├── neural_network_server.py
│   │       └── orchestrator_server.py
│   │
│   ├── utils/                  # Utility functions
│   │   ├── __init__.py
│   │   ├── logging.py
│   │   └── validators.py
│   │
│   └── scripts/                # CLI scripts
│       ├── __init__.py
│       ├── train_models.py
│       └── verify_models.py
│
├── data/                       # Data directory
│   ├── models/                 # Trained model files
│   │   ├── naive_bayes_model.pkl
│   │   ├── svm_model.pkl
│   │   ├── random_forest_model.pkl
│   │   ├── logistic_regression_model.pkl
│   │   └── neural_network_model.pkl
│   │
│   └── datasets/               # Training datasets
│       ├── SpamAssasin.csv.zip
│       ├── Enron.csv.zip
│       ├── Ling.csv.zip
│       ├── CEAS_08.csv.zip
│       ├── Nazario.csv.zip
│       └── phishing_email.csv.zip
│
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── conftest.py            # Pytest configuration
│   ├── unit/                  # Unit tests
│   │   ├── __init__.py
│   │   ├── test_detectors.py
│   │   └── test_preprocessing.py
│   │
│   ├── integration/           # Integration tests
│   │   ├── __init__.py
│   │   └── test_servers.py
│   │
│   └── fixtures/              # Test data
│       └── sample_emails.json
│
├── docs/                      # Documentation
│   ├── index.md
│   ├── installation.md
│   ├── usage.md
│   ├── api/
│   └── architecture.md
│
├── logs/                      # Log files (gitignored)
│
├── scripts/                   # Development scripts
│   ├── setup_dev.sh
│   └── run_tests.sh
│
├── .env.example              # Environment configuration template
├── .gitignore                # Git ignore file
├── Makefile                  # Common tasks
├── README.md                 # Project documentation
├── pyproject.toml            # Modern Python project configuration
├── setup.py                  # Package installation script
├── requirements.txt          # Direct dependencies
└── requirements-dev.txt      # Development dependencies
```

## Key Design Principles

1. **Separation of Concerns**: Each module has a single, well-defined responsibility
2. **Clean Architecture**: Core business logic is separated from infrastructure
3. **Dependency Injection**: Components depend on abstractions, not concrete implementations
4. **Configuration Management**: Centralized configuration with environment variable support
5. **Type Safety**: Extensive use of type hints and dataclasses
6. **Testability**: All components designed with testing in mind
7. **Modularity**: Easy to add new models or modify existing ones

## Package Organization

- **core/**: Domain models and business logic
- **ml/**: Machine learning specific code
- **servers/**: MCP server implementations
- **config/**: Configuration and settings
- **utils/**: Shared utilities
- **scripts/**: Standalone scripts

This structure follows Python best practices and makes the codebase maintainable, scalable, and easy to understand.