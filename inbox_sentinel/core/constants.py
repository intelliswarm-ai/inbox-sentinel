"""
Constants used throughout Inbox Sentinel
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
DATASETS_DIR = DATA_DIR / "datasets"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Model constants
MAX_FEATURES = 5000
DEFAULT_RANDOM_STATE = 42
TEST_SIZE = 0.2

# Model names
MODEL_NAIVE_BAYES = "naive_bayes"
MODEL_SVM = "svm"
MODEL_RANDOM_FOREST = "random_forest"
MODEL_LOGISTIC_REGRESSION = "logistic_regression"
MODEL_NEURAL_NETWORK = "neural_network"

# Model file paths
MODEL_FILES = {
    MODEL_NAIVE_BAYES: MODELS_DIR / "naive_bayes_model.pkl",
    MODEL_SVM: MODELS_DIR / "svm_model.pkl",
    MODEL_RANDOM_FOREST: MODELS_DIR / "random_forest_model.pkl",
    MODEL_LOGISTIC_REGRESSION: MODELS_DIR / "logistic_regression_model.pkl",
    MODEL_NEURAL_NETWORK: MODELS_DIR / "neural_network_model.pkl",
}

# Feature extraction
TFIDF_FEATURES = 5000
MANUAL_FEATURES = [
    'content_length', 'subject_length', 'word_count',
    'exclamation_count', 'question_count', 'uppercase_ratio',
    'url_count', 'short_url_count', 'money_symbols',
    'percentage_symbols', 'urgent_words', 'winner_words',
    'verify_words', 'sender_has_numbers', 'sender_domain_suspicious'
]

# Email classification
SPAM_LABEL = 1
HAM_LABEL = 0

# Consensus strategies
STRATEGY_MAJORITY_VOTE = "majority_vote"
STRATEGY_WEIGHTED_AVERAGE = "weighted_average"
STRATEGY_HIGHEST_CONFIDENCE = "highest_confidence"
STRATEGY_CONSERVATIVE = "conservative"
STRATEGY_AGGRESSIVE = "aggressive"