"""Test runner for all tests in the Aether framework.

This module provides a comprehensive test runner that can execute:
- Unit tests: Test individual components
- Integration tests: Test component interactions
- Profiling tests: Performance and memory analysis

Usage:
    python tests/run_tests.py                    # Run all tests
    python tests/run_tests.py --unit             # Run only unit tests
    python tests/run_tests.py --integration      # Run only integration tests
    python tests/run_tests.py --profiling        # Run only profiling tests
    python tests/run_tests.py --coverage         # Run with coverage report
"""

import pytest
import sys
import os
import argparse

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Aether Test Runner")
    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run only unit tests"
    )
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run only integration tests"
    )
    parser.add_argument(
        "--profiling",
        action="store_true",
        help="Run only profiling tests"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage report"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "-x", "--exitfirst",
        action="store_true",
        help="Exit on first failure"
    )
    return parser.parse_args()


def get_test_directories(args):
    """Get test directories based on arguments."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.unit:
        return [os.path.join(base_dir, "unit")]
    elif args.integration:
        return [os.path.join(base_dir, "integration")]
    elif args.profiling:
        return [os.path.join(base_dir, "profiling")]
    else:
        # Run all tests
        return [
            os.path.join(base_dir, "unit"),
            os.path.join(base_dir, "integration"),
        ]


def build_pytest_args(test_dirs, args):
    """Build pytest arguments."""
    pytest_args = test_dirs.copy()
    
    if args.verbose:
        pytest_args.append("-v")
    else:
        pytest_args.append("-v")  # Always use verbose for clarity
    
    pytest_args.append("--tb=short")
    
    if args.exitfirst:
        pytest_args.append("-x")
    
    if args.coverage:
        pytest_args.extend([
            "--cov=aether",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov"
        ])
    
    return pytest_args


def print_summary(args):
    """Print a summary of what tests will be run."""
    print("=" * 60)
    print("Aether Test Suite")
    print("=" * 60)
    
    if args.unit:
        print("Running: Unit Tests")
    elif args.integration:
        print("Running: Integration Tests")
    elif args.profiling:
        print("Running: Profiling Tests")
    else:
        print("Running: All Tests (Unit + Integration)")
    
    if args.coverage:
        print("Coverage: Enabled")
    
    print("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    print_summary(args)
    
    test_dirs = get_test_directories(args)
    pytest_args = build_pytest_args(test_dirs, args)
    
    exit_code = pytest.main(pytest_args)
    
    print("\n" + "=" * 60)
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
    print("=" * 60)
    
    sys.exit(exit_code)