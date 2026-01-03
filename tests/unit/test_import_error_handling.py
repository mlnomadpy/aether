"""Unit tests for import error handling in aether/__init__.py."""

import pytest
import sys
import builtins


def test_jax_missing_error():
    """Test that missing JAX produces appropriate error message."""
    original_import = builtins.__import__
    
    def mock_import(name, *args, **kwargs):
        if name == 'jax' or name.startswith('jax.'):
            raise ModuleNotFoundError("No module named 'jax'")
        return original_import(name, *args, **kwargs)
    
    builtins.__import__ = mock_import
    
    # Clear cached imports
    modules_to_remove = [mod for mod in sys.modules.keys() 
                        if 'aether' in mod or 'jax' in mod or 'flax' in mod]
    for mod in modules_to_remove:
        del sys.modules[mod]
    
    try:
        from aether import Trainer
        trainer = Trainer(None)
        pytest.fail("Should have raised ImportError")
    except ImportError as e:
        assert 'JAX/Flax' in str(e), f"Error should mention JAX/Flax, got: {e}"
    finally:
        builtins.__import__ = original_import
        # Clean up modules after test
        modules_to_remove = [mod for mod in sys.modules.keys() if 'aether' in mod]
        for mod in modules_to_remove:
            del sys.modules[mod]


def test_flax_missing_error():
    """Test that missing Flax produces appropriate error message."""
    original_import = builtins.__import__
    
    def mock_import(name, *args, **kwargs):
        if name == 'flax' or name.startswith('flax.'):
            raise ModuleNotFoundError("No module named 'flax'")
        return original_import(name, *args, **kwargs)
    
    builtins.__import__ = mock_import
    
    # Clear cached imports
    modules_to_remove = [mod for mod in sys.modules.keys() 
                        if 'aether' in mod or 'flax' in mod]
    for mod in modules_to_remove:
        del sys.modules[mod]
    
    try:
        from aether import Trainer
        trainer = Trainer(None)
        pytest.fail("Should have raised ImportError")
    except ImportError as e:
        assert 'JAX/Flax' in str(e), f"Error should mention JAX/Flax, got: {e}"
    finally:
        builtins.__import__ = original_import
        # Clean up modules after test
        modules_to_remove = [mod for mod in sys.modules.keys() if 'aether' in mod]
        for mod in modules_to_remove:
            del sys.modules[mod]


def test_other_module_missing_error():
    """Test that missing non-JAX/Flax module produces correct error (not JAX/Flax message)."""
    original_import = builtins.__import__
    
    def mock_import(name, *args, **kwargs):
        if name == 'wandb':
            raise ModuleNotFoundError("No module named 'wandb'")
        return original_import(name, *args, **kwargs)
    
    builtins.__import__ = mock_import
    
    # Clear cached imports
    modules_to_remove = [mod for mod in sys.modules.keys() 
                        if 'aether' in mod or 'wandb' in mod]
    for mod in modules_to_remove:
        del sys.modules[mod]
    
    try:
        from aether import Trainer
        pytest.fail("Should have raised ModuleNotFoundError")
    except ModuleNotFoundError as e:
        # Should get the real error about wandb, not JAX/Flax
        assert 'wandb' in str(e), f"Error should mention wandb, got: {e}"
        assert 'JAX/Flax' not in str(e), f"Error should NOT mention JAX/Flax, got: {e}"
    except ImportError as e:
        # If we get an ImportError (which is a parent class), check it's about wandb
        assert 'wandb' in str(e), f"Error should mention wandb, got: {e}"
        assert 'JAX/Flax' not in str(e), f"Error should NOT mention JAX/Flax, got: {e}"
    finally:
        builtins.__import__ = original_import
        # Clean up modules after test
        modules_to_remove = [mod for mod in sys.modules.keys() if 'aether' in mod]
        for mod in modules_to_remove:
            del sys.modules[mod]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
