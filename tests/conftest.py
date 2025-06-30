"""
Pytest configuration and fixtures
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from inbox_sentinel.core.types import Email


@pytest.fixture
def sample_spam_email():
    """Sample spam email for testing"""
    return Email(
        content="URGENT! Your account will be suspended. Click here to verify: http://bit.ly/verify",
        subject="Account Security Alert!!!",
        sender="security@paypal-verify.tk"
    )


@pytest.fixture
def sample_ham_email():
    """Sample legitimate email for testing"""
    return Email(
        content="Hi team, Please review the Q4 report attached. Let me know if you have questions.",
        subject="Q4 Report for Review",
        sender="john@company.com"
    )


@pytest.fixture
def temp_model_dir():
    """Temporary directory for model files"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_training_data():
    """Mock training data for testing"""
    return [
        {
            "email_content": "Win a FREE iPhone! Click here now!",
            "subject": "Congratulations Winner!",
            "sender": "prizes@win.tk",
            "is_spam": True
        },
        {
            "email_content": "Meeting scheduled for tomorrow at 2 PM.",
            "subject": "Meeting Reminder",
            "sender": "calendar@company.com",
            "is_spam": False
        }
    ]