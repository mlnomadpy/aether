"""Test runner for all unit tests."""

import pytest
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    # Run all tests in the unit directory
    exit_code = pytest.main([
        "tests/unit/",
        "-v",
        "--tb=short"
    ])
    sys.exit(exit_code)