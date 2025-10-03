"""Test that README contains the updated performance metrics."""

import os


def test_readme_contains_tpu_performance():
    """Test that README contains TPU v5-8 performance metrics."""
    readme_path = os.path.join(os.path.dirname(__file__), "..", "..", "README.md")
    
    with open(readme_path, 'r') as f:
        content = f.read()
    
    # Check for TPU v5-8 section
    assert "TPU v5-8 (Kaggle)" in content, "README missing TPU v5-8 section"
    
    # Check for YAT GPT Base performance
    assert "YAT GPT Base" in content, "README missing YAT GPT Base performance"
    assert "132,000 tokens/second" in content, "README missing YAT GPT Base performance metric"
    
    # Check for Linear GPT Model performance
    assert "Linear GPT Model" in content, "README missing Linear GPT Model performance"
    assert "138,000 tokens/second" in content, "README missing Linear GPT Model performance metric"
    
    # Check for paper reference
    assert "Deep Learning 2.0.1" in content, "README missing paper reference"
    assert "Taha Bouhsine" in content, "README missing author reference"


def test_readme_preserves_existing_performance():
    """Test that README still contains existing V100 performance metrics."""
    readme_path = os.path.join(os.path.dirname(__file__), "..", "..", "README.md")
    
    with open(readme_path, 'r') as f:
        content = f.read()
    
    # Check that existing V100 metrics are preserved
    assert "V100 GPU" in content, "README missing V100 GPU section"
    assert "~2000 tokens/second" in content, "README missing V100 forward pass metric"
    assert "~1500 tokens/second" in content, "README missing V100 training step metric"
    assert "~8GB for 12-layer, 768-dim model" in content, "README missing V100 memory usage"


if __name__ == "__main__":
    test_readme_contains_tpu_performance()
    test_readme_preserves_existing_performance()
    print("✓ All README performance tests passed!")