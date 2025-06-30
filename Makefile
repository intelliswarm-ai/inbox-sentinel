.PHONY: help install install-dev test lint format clean train verify serve-nb serve-svm serve-rf serve-lr serve-nn serve-orchestrator

# Default target
help:
	@echo "Inbox Sentinel - Make Commands"
	@echo "=============================="
	@echo "install          Install package in production mode"
	@echo "install-dev      Install package with development dependencies"
	@echo "test             Run test suite"
	@echo "lint             Run linters (flake8, mypy)"
	@echo "format           Format code with black and isort"
	@echo "clean            Clean build artifacts and cache"
	@echo "train            Train all models"
	@echo "verify           Verify trained models"
	@echo ""
	@echo "MCP Servers:"
	@echo "serve-nb         Start Naive Bayes server"
	@echo "serve-svm        Start SVM server"
	@echo "serve-rf         Start Random Forest server"
	@echo "serve-lr         Start Logistic Regression server"
	@echo "serve-nn         Start Neural Network server"
	@echo "serve-orchestrator Start Orchestrator server"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,docs]"
	pre-commit install

# Testing
test:
	pytest tests/ -v --cov=inbox_sentinel --cov-report=html

test-quick:
	pytest tests/ -v -x

# Code quality
lint:
	flake8 inbox_sentinel tests
	mypy inbox_sentinel

format:
	black inbox_sentinel tests
	isort inbox_sentinel tests

check: lint
	black --check inbox_sentinel tests
	isort --check-only inbox_sentinel tests

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Model management
train:
	python -m inbox_sentinel.scripts.train_models

verify:
	python -m inbox_sentinel.scripts.verify_models

# MCP Servers
serve-nb:
	fastmcp dev inbox_sentinel/servers/mcp/naive_bayes_server.py

serve-svm:
	fastmcp dev inbox_sentinel/servers/mcp/svm_server.py

serve-rf:
	fastmcp dev inbox_sentinel/servers/mcp/random_forest_server.py

serve-lr:
	fastmcp dev inbox_sentinel/servers/mcp/logistic_regression_server.py

serve-nn:
	fastmcp dev inbox_sentinel/servers/mcp/neural_network_server.py

serve-orchestrator:
	fastmcp dev inbox_sentinel/servers/mcp/orchestrator_server.py

# Development
dev-setup: install-dev
	mkdir -p logs
	mkdir -p data/models
	cp .env.example .env

# Documentation
docs:
	mkdocs serve

docs-build:
	mkdocs build