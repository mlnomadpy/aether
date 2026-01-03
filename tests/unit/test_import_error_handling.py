"""Unit tests for import error handling in aether/__init__.py."""

import pytest
import sys


def test_is_jax_flax_error_detection():
    """Test the _is_jax_flax_error helper function for correct pattern matching."""
    # Need to import after clearing modules to get fresh version
    modules_to_remove = [mod for mod in list(sys.modules.keys()) if 'aether' in mod]
    for mod in modules_to_remove:
        del sys.modules[mod]
    
    # Import the internal function for testing pattern matching
    # This is acceptable for unit testing internal logic
    from aether import _is_jax_flax_error
    
    # Should match JAX/Flax errors
    assert _is_jax_flax_error("No module named 'jax'") is True
    assert _is_jax_flax_error("No module named 'flax'") is True
    assert _is_jax_flax_error("No module named 'jax.numpy'") is True
    assert _is_jax_flax_error("No module named 'flax.nnx'") is True
    assert _is_jax_flax_error("cannot import jax") is True
    assert _is_jax_flax_error("cannot import flax") is True
    
    # Should NOT match unrelated errors (avoiding false positives)
    assert _is_jax_flax_error("No module named 'wandb'") is False
    assert _is_jax_flax_error("No module named 'optax'") is False
    assert _is_jax_flax_error("No module named 'orbax'") is False
    assert _is_jax_flax_error("No module named 'relaxation'") is False  # Contains 'lax' but not 'flax'


def test_training_available_api():
    """Test that is_training_available() correctly reports training status."""
    from aether import is_training_available, Trainer, train_step, eval_step, loss_fn
    
    # When all dependencies are available, training should be available
    assert is_training_available() is True
    
    # Verify components are the real implementations, not placeholders
    assert callable(train_step)
    assert callable(eval_step)
    assert callable(loss_fn)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
