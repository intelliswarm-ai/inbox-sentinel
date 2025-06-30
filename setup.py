"""
Setup script for Inbox Sentinel
"""

from setuptools import setup, find_packages

# Read the contents of README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="inbox-sentinel",
    version="1.0.0",
    author="Inbox Sentinel Team",
    author_email="contact@inboxsentinel.ai",
    description="Advanced phishing detection system with multiple ML algorithms via FastMCP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/inboxsentinel/inbox-sentinel",
    packages=find_packages(exclude=["tests", "tests.*", "docs", "docs.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "fastmcp>=2.9.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "click>=8.0.0",
        "rich>=13.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "mkdocs>=1.4.0",
            "mkdocs-material>=9.0.0",
            "mkdocstrings[python]>=0.20.0",
        ],
        "langchain": [
            "langchain>=0.1.0",
            "langchain-community>=0.1.0",
            "langchain-openai>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "inbox-sentinel=inbox_sentinel.cli:main",
            "train-models=inbox_sentinel.scripts.train_models:main",
            "verify-models=inbox_sentinel.scripts.verify_models:main",
        ],
    },
    include_package_data=True,
    package_data={
        "inbox_sentinel": ["config/*.json", "data/models/*.pkl"],
    },
)