"""Test validation dataset handling in trainer."""

import pytest
import sys
import os
sys.path.insert(0, '/home/runner/work/aether/aether')

from aether.training.trainer import Trainer


def test_validation_iterator_reset_methods():
    """Test that validation iterator reset methods exist and have correct signatures."""
    # Test method existence
    assert hasattr(Trainer, '_reset_validation_iterator'), "Missing _reset_validation_iterator method"
    assert hasattr(Trainer, '_reset_training_iterator'), "Missing _reset_training_iterator method"
    
    # Test method signatures
    import inspect
    reset_val_sig = inspect.signature(Trainer._reset_validation_iterator)
    reset_train_sig = inspect.signature(Trainer._reset_training_iterator)
    
    assert len(reset_val_sig.parameters) == 1, "Wrong parameter count for _reset_validation_iterator"
    assert len(reset_train_sig.parameters) == 1, "Wrong parameter count for _reset_training_iterator"


def test_evaluate_method_structure():
    """Test that the _evaluate method has the correct structure for complete validation."""
    # Get the source code
    import inspect
    source = inspect.getsource(Trainer._evaluate)
    
    # Check for key improvements
    assert 'batches_processed = 0' in source, "Missing batches_processed tracking"
    assert 'max_eval_steps = self.config.training.eval_steps' in source, "Missing eval_steps configuration"
    assert 'while batches_processed < max_eval_steps' in source, "Missing proper loop structure"
    assert 'self._reset_validation_iterator()' in source, "Missing call to reset validation iterator"
    assert 'val_batches' in source, "Missing val_batches logging"
    assert 'Completed validation pass' in source, "Missing completion message"


def test_evaluate_method_signature():
    """Test that _evaluate method has correct signature."""
    import inspect
    evaluate_sig = inspect.signature(Trainer._evaluate)
    
    assert len(evaluate_sig.parameters) == 2, "Wrong parameter count for _evaluate"
    assert 'step' in evaluate_sig.parameters, "Missing step parameter in _evaluate"


def test_old_inefficient_patterns_removed():
    """Test that old inefficient patterns are removed from the main validation loop."""
    import inspect
    source = inspect.getsource(Trainer._evaluate)
    
    # These patterns should not be in the _evaluate method anymore
    assert 'for _ in range(self.config.training.eval_steps):' not in source, "Old fixed-loop validation still present"
    assert '_, self.val_iterator = self._create_data_iterators()' not in source, "Old direct iterator reset in _evaluate"


if __name__ == "__main__":
    test_validation_iterator_reset_methods()
    test_evaluate_method_structure()
    test_evaluate_method_signature()
    test_old_inefficient_patterns_removed()
    print("✓ All validation iterator tests passed!")