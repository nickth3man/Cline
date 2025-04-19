"""Test configuration for pytest."""

import pytest
import os
import sys
import subprocess # Import subprocess for patching
from unittest.mock import MagicMock, patch # Import MagicMock and patch
from dotenv import load_dotenv # Import load_dotenv

# Add the src directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Load environment variables from .env file for tests
load_dotenv()

# Patch subprocess.run before importing workflow_logic
# This is necessary because check_ffmpeg is called on import
# We use autouse=True in the fixture below, but this patch is needed for the initial import
# when the module is first loaded by pytest.
# This is a bit of a workaround for the import-time side effect in workflow_logic.
# A cleaner approach would be to refactor workflow_logic to not call check_ffmpeg on import.
mock_run_patch = patch("src.utils.workflow_logic.subprocess.run")
mock_run_instance = mock_run_patch.start()
mock_process = MagicMock()
mock_process.stderr = b""
mock_process.stdout = b"ffmpeg version ..."
mock_run_instance.return_value = mock_process

# Import the module *after* setting env vars and patching subprocess.run
# This import is necessary for the fixtures and tests to access workflow_logic
try:
    from src.utils import workflow_logic
except Exception as e:
    print(f"Error importing workflow_logic after patching: {e}")
    # Depending on the severity, you might want to exit or handle differently
    # For now, we'll let the tests potentially fail with import errors if this happens.
    workflow_logic = None # Ensure workflow_logic is defined even on import error



@pytest.fixture(autouse=True)
def mock_env_vars_and_ffmpeg_check(monkeypatch):
    """
    Fixture to mock environment variables and patch subprocess.run for ffmpeg check
    for each test.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-fakekey")
    monkeypatch.setenv("HF_TOKEN", "hf_faketoken")

    # Patch subprocess.run for the duration of each test
    with patch("src.utils.workflow_logic.subprocess.run") as mock_run:
        mock_process = MagicMock()
        mock_process.stderr = b""
        mock_process.stdout = b"ffmpeg version ..."
        mock_run.return_value = mock_process
        yield mock_run # Yield the mock to allow tests to configure it


@pytest.fixture
def mock_openai_client():
    """Provides a mock OpenAI client for testing API interactions."""
    # Use MagicMock to mock the entire OpenAI client
    mock_client = MagicMock()
    # You can configure specific mock responses here if needed for tests
    # e.g., mock_client.audio.transcriptions.create.return_value = ...
    # e.g., mock_client.chat.completions.create.return_value = ...
    return mock_client
