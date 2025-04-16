"""Test configuration for pytest."""

import pytest
import os
import sys

# Add the src directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
