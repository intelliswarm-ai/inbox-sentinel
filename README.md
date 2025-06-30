# Inbox Sentinel - Advanced Phishing Detection System

A professional-grade phishing detection system featuring multiple machine learning algorithms accessible via FastMCP servers. Built with clean architecture principles and trained on 160,000+ real spam/phishing emails.

## 🔒 Privacy Guarantee

**All processing is done locally on your machine. No email content, subjects, or sender information is ever sent to external services, APIs, or cloud providers.**

## 🚀 Quick Start

```bash
# Install the package
pip install -e ".[dev]"

# Start a server (e.g., Neural Network with 96.6% accuracy)
make serve-nn

# Or use the CLI
inbox-sentinel server start neural-network

# Check available models
inbox-sentinel models list
```

## 🏗️ Professional Architecture

```
inbox-sentinel/
├── inbox_sentinel/          # Main package
│   ├── core/               # Base classes, types, exceptions
│   ├── ml/                 # Machine learning components
│   │   ├── models/         # Model implementations
│   │   ├── preprocessing/  # Feature extraction
│   │   └── training/       # Training utilities
│   ├── servers/            # MCP server implementations
│   │   ├── base/          # Base server class
│   │   └── mcp/           # FastMCP servers
│   ├── config/            # Configuration management
│   ├── utils/             # Utilities
│   └── scripts/           # CLI scripts
├── data/                   # Data directory
│   ├── models/            # Trained models (*.pkl)
│   └── datasets/          # Training datasets
├── tests/                 # Test suite
└── docs/                  # Documentation
```

## 📊 Model Performance

| Model | Algorithm | Test Accuracy | Key Features |
|-------|-----------|---------------|--------------|
| `naive-bayes` | Multinomial Naive Bayes | 96.25% | Fast, interpretable, great for text |
| `svm` | Support Vector Machine | 95.75% | RBF kernel, 3,882 support vectors |
| `random-forest` | Random Forest | 93.95% | 100 trees, feature importance |
| `logistic-regression` | Logistic Regression | 95.75% | Linear, highly interpretable |
| `neural-network` | Neural Network (MLP) | 96.60% | 3-layer architecture, best accuracy |

## 🛠️ Features

### Advanced ML Capabilities
- **Pre-trained Models**: All models trained on real spam datasets
- **Feature Engineering**: TF-IDF + 15 manual features (URLs, keywords, patterns)
- **Ensemble Methods**: 5 consensus strategies for combining predictions
- **Real-time Analysis**: Fast inference (<100ms per email)
- **Explainable AI**: Feature importance and confidence scores
- **LLM Orchestration**: Use local LLMs to intelligently coordinate multiple models
- **Forwarded Email Support**: Automatically parse and analyze Gmail forwarded emails

### Professional Development
- **Clean Architecture**: Separation of concerns, SOLID principles
- **Type Safety**: Full type hints with custom types
- **Configuration Management**: Pydantic settings with env support
- **Testing**: Comprehensive test suite with pytest
- **CLI Tools**: Rich CLI interface for all operations
- **Documentation**: Complete API and usage docs

## 💾 Training Data

Models trained on **161,640 emails** from 6 datasets:
- SpamAssassin (5,809 emails)
- Enron Spam (29,767 emails) 
- Ling Spam (2,859 emails)
- CEAS 2008 (39,154 emails)
- Nazario Phishing (1,565 emails)
- Phishing Email Dataset (82,486 emails)

**Distribution**: 51% spam/phishing, 49% legitimate

## 📖 Usage

### Orchestrated Analysis

The orchestration feature runs multiple ML models in parallel and combines their results for more accurate detection:

```bash
# Simple consensus-based orchestration (no dependencies)
inbox-sentinel orchestrate -F email.txt --forwarded

# LLM-powered orchestration with Ollama (requires setup)
inbox-sentinel orchestrate -F email.txt --forwarded --llm-provider ollama --model-name llama2
```

**Two Orchestration Modes:**

1. **Simple Consensus (Default)**
   - Runs all 5 ML models in parallel
   - Uses majority voting (e.g., 4/5 models = spam)
   - Calculates average confidence scores
   - No additional dependencies required
   - Fast and reliable

2. **LLM-Powered (Advanced)**
   - Uses local LLM to coordinate analysis
   - LLM selects which models to query
   - Provides natural language explanations
   - Can adapt strategy based on results
   - Requires Ollama + LangChain setup

**How It Works:**
- Each MCP server (Naive Bayes, SVM, Random Forest, Logistic Regression, Neural Network) is wrapped as a tool
- In simple mode: All tools are called and results are combined
- In LLM mode: The AI agent decides which tools to use and interprets results

### CLI Commands

```bash
# View available models and their status
inbox-sentinel models list

# Train all models
inbox-sentinel models train

# Verify trained models
inbox-sentinel models verify

# Analyze an email
inbox-sentinel analyze -c "Email content" -s "Subject" -f "sender@email.com"

# Analyze a forwarded Gmail email
inbox-sentinel analyze -F forwarded_email.txt --forwarded

# Orchestrate multiple models with consensus
inbox-sentinel orchestrate -F email.txt --forwarded

# Start a specific MCP server
inbox-sentinel server start neural-network
```

### Using MCP Servers

Each server provides these tools:
- `analyze_email` - Analyze an email for spam/phishing
- `train_model` - Train with new data
- `initialize_model` - Initialize/load pre-trained model
- `get_model_info` - Get model information

### LLM-Orchestrated Analysis with Ollama

For advanced analysis using a local LLM to orchestrate multiple detection models:

#### Setup Ollama (One-time setup)

**Windows:**
```bash
# 1. Download and install from https://ollama.ai/download/windows
# 2. Start Ollama server (in a separate terminal)
ollama serve

# 3. Pull a model (in your main terminal)
ollama pull llama2     # 7B parameters, balanced
# Or use smaller/faster models:
ollama pull phi        # 2.7B parameters, very fast
ollama pull mistral    # 7B parameters, fast
```

**macOS/Linux:**
```bash
# 1. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Start Ollama server
ollama serve

# 3. Pull a model
ollama pull llama2
```

#### Install LangChain
```bash
# Option 1: Install LangChain dependencies as an extra
pip install -e ".[langchain]"

# Option 2: Install LangChain dependencies separately
pip install langchain langchain-community langchain-openai

# Option 3: Use the requirements file
pip install -r requirements-langchain.txt
```

#### Run LLM-Orchestrated Analysis
```bash
# Analyze forwarded email with LLM orchestration
inbox-sentinel orchestrate -F email.txt --forwarded --llm-provider ollama --model-name llama2

# Or use simple consensus-based orchestration (no LLM required)
inbox-sentinel orchestrate -F email.txt --forwarded --llm-provider simple
```

**Recommended Models for Tool Use:**
```bash
# Mistral - Better at following tool-use instructions
ollama pull mistral
inbox-sentinel orchestrate -F email.txt --forwarded --llm-provider ollama --model-name mistral

# Mixtral - Excellent at structured outputs
ollama pull mixtral
inbox-sentinel orchestrate -F email.txt --forwarded --llm-provider ollama --model-name mixtral
```

**Troubleshooting LLM Orchestration:**

If the LLM gets stuck or doesn't use tools correctly:
1. **Try a different model** - Mistral and Mixtral are better at tool use than Llama2
2. **Check Ollama is running** - `curl http://localhost:11434/api/tags`
3. **Use simple orchestration** - Works reliably without LLM: `--llm-provider simple`
4. **Install dependencies** - `pip install langchain langchain-community nest-asyncio`

The LLM orchestration provides:
- Intelligent tool selection based on email characteristics
- Natural language explanations of decisions
- Adaptive analysis strategies
- Context-aware reasoning about phishing patterns

**Note:** Some models (like Llama2) may struggle with the structured format required for tool use. If you experience issues, the simple consensus-based orchestration provides excellent results without requiring an LLM.

#### Example Output (Simple Consensus)
```
Orchestrated Email Analysis

Subject: Claim Your Merlin Chain Early Users Reward Now
Sender: hello@merlinteamnews.blog

Using consensus-based orchestration
✅ Initialized all 5 models

╭───────── Orchestrated Analysis Result ─────────╮
│ SPAM/PHISHING                                  │
│                                                │
│ Consensus: 4/5 models detected spam            │
│ Average Confidence: 58.8%                      │
│                                                │
│ Individual Results:                            │
│ • naive_bayes: LEGITIMATE (16.7%)              │
│ • svm: SPAM (53.5%)                           │
│ • random_forest: SPAM (28.4%)                 │
│ • logistic_regression: SPAM (99.9%)           │
│ • neural_network: SPAM (95.5%)                │
│                                                │
│ Recommendation: DO NOT trust this email.       │
╰────────────────────────────────────────────────╯
```

### Python API

```python
from inbox_sentinel.ml.models import NeuralNetworkDetector
from inbox_sentinel.core.types import Email

# Initialize detector
detector = NeuralNetworkDetector()
await detector.initialize(use_pretrained=True)

# Analyze email
email = Email(
    content="Your account will be suspended...",
    subject="Urgent Security Alert",
    sender="security@paypal-verify.tk"
)
result = await detector.analyze(email)

print(f"Is Spam: {result.is_spam}")
print(f"Confidence: {result.confidence:.1%}")
```

## 🔧 Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/your-org/inbox-sentinel.git
cd inbox-sentinel

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
make install-dev

# Run tests
make test

# Format code
make format
```

### Common Tasks (Makefile)

```bash
make help              # Show all available commands
make serve-nn          # Start Neural Network server
make serve-svm         # Start SVM server
make train             # Train all models
make test              # Run test suite
make lint              # Run code quality checks
make format            # Format code with black/isort
make clean             # Clean build artifacts
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=inbox_sentinel

# Run specific test file
pytest tests/unit/test_detectors.py
```

## 🔍 API Reference

### Core Types

- `Email`: Email data structure
- `PredictionResult`: Single model prediction
- `EnsembleResult`: Combined prediction from multiple models
- `ConsensusStrategy`: Enum for ensemble strategies

### Base Classes

- `BaseDetector`: Abstract base for all detectors
- `BaseMCPServer`: Base class for MCP servers

### Configuration

- Environment variables via `.env` file
- Pydantic settings for type-safe configuration
- Model-specific configurations in `config/model_config.py`

## 📚 Additional Resources

- [Full Documentation](docs/)
- [Architecture Guide](docs/PROJECT_STRUCTURE.md)
- [Training Details](docs/TRAINED_MODELS.md)
- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [Model Context Protocol](https://modelcontextprotocol.io/)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Areas for Improvement
- Additional ML algorithms (XGBoost, LightGBM)
- Deep learning models (BERT, Transformers)
- Real-time learning capabilities
- Email header analysis
- Attachment scanning
- Multi-language support

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

This project is for educational and defensive security purposes only. 
