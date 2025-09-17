"""Simpler test to verify data isolation functionality."""

import sys
import os
sys.path.insert(0, '/home/runner/work/aether/aether')

import unittest.mock as mock


def test_create_validation_iterator_function_exists():
    """Test that create_validation_iterator function exists and can be imported."""
    try:
        from aether.data.dataset import create_validation_iterator
        print("✓ create_validation_iterator function exists")
        return True
    except ImportError as e:
        print(f"✗ create_validation_iterator function not found: {e}")
        return False


def test_create_training_iterator_function_exists():
    """Test that create_training_iterator function exists and can be imported."""
    try:
        from aether.data.dataset import create_training_iterator
        print("✓ create_training_iterator function exists")
        return True
    except ImportError as e:
        print(f"✗ create_training_iterator function not found: {e}")
        return False


def test_trainer_has_separate_reset_methods():
    """Test that trainer has separate reset methods that use individual iterator creators."""
    try:
        from aether.training.trainer import Trainer
        import inspect
        
        # Check if methods exist
        assert hasattr(Trainer, '_reset_validation_iterator'), "Missing _reset_validation_iterator method"
        assert hasattr(Trainer, '_reset_training_iterator'), "Missing _reset_training_iterator method"
        assert hasattr(Trainer, '_create_validation_iterator'), "Missing _create_validation_iterator method"
        assert hasattr(Trainer, '_create_training_iterator'), "Missing _create_training_iterator method"
        
        # Check that reset methods call individual creation methods
        reset_val_source = inspect.getsource(Trainer._reset_validation_iterator)
        reset_train_source = inspect.getsource(Trainer._reset_training_iterator)
        
        assert 'self._create_validation_iterator()' in reset_val_source, "_reset_validation_iterator should call _create_validation_iterator"
        assert 'self._create_training_iterator()' in reset_train_source, "_reset_training_iterator should call _create_training_iterator"
        
        # Check that individual creation methods use specific functions
        create_val_source = inspect.getsource(Trainer._create_validation_iterator)
        create_train_source = inspect.getsource(Trainer._create_training_iterator)
        
        assert 'create_validation_iterator(' in create_val_source, "_create_validation_iterator should call create_validation_iterator"
        assert 'create_training_iterator(' in create_train_source, "_create_training_iterator should call create_training_iterator"
        
        print("✓ Trainer methods correctly use separate iterator creation functions")
        return True
    except Exception as e:
        print(f"✗ Trainer method test failed: {e}")
        return False


def test_function_signatures():
    """Test that the new functions have correct signatures."""
    try:
        from aether.data.dataset import create_validation_iterator, create_training_iterator
        import inspect
        
        # Check function signatures
        val_sig = inspect.signature(create_validation_iterator)
        train_sig = inspect.signature(create_training_iterator)
        
        expected_params = ['dataset_name', 'split', 'streaming', 'maxlen', 'tokenizer', 'val_set_size', 'batch_size']
        
        val_params = list(val_sig.parameters.keys())
        train_params = list(train_sig.parameters.keys())
        
        assert val_params == expected_params, f"create_validation_iterator params: {val_params}, expected: {expected_params}"
        assert train_params == expected_params, f"create_training_iterator params: {train_params}, expected: {expected_params}"
        
        print("✓ Function signatures are correct")
        return True
    except Exception as e:
        print(f"✗ Function signature test failed: {e}")
        return False


if __name__ == "__main__":
    tests = [
        test_create_validation_iterator_function_exists,
        test_create_training_iterator_function_exists,
        test_trainer_has_separate_reset_methods,
        test_function_signatures,
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nPassed {passed}/{len(tests)} tests")
    if passed == len(tests):
        print("✓ All basic isolation functionality tests passed!")
    else:
        print("✗ Some tests failed")
        sys.exit(1)