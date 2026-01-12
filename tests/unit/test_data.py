"""Unit tests for data processing module."""

import pytest
import sys
import os
sys.path.insert(0, '/home/runner/work/aether/aether')

import jax.numpy as jnp
from unittest import mock


class TestProcessDatasetFunctions:
    """Tests for dataset processing functions."""
    
    def test_create_validation_iterator_import(self):
        """Test that create_validation_iterator can be imported."""
        from aether.data.dataset import create_validation_iterator
        assert callable(create_validation_iterator)
    
    def test_create_training_iterator_import(self):
        """Test that create_training_iterator can be imported."""
        from aether.data.dataset import create_training_iterator
        assert callable(create_training_iterator)
    
    def test_create_data_iterators_import(self):
        """Test that create_data_iterators can be imported."""
        from aether.data.dataset import create_data_iterators
        assert callable(create_data_iterators)
    
    def test_prepare_batch_import(self):
        """Test that prepare_batch can be imported."""
        from aether.data.dataset import prepare_batch
        assert callable(prepare_batch)
    
    def test_process_dataset_import(self):
        """Test that process_dataset can be imported."""
        from aether.data.dataset import process_dataset
        assert callable(process_dataset)


class TestPrepareBatch:
    """Tests for prepare_batch function."""
    
    def test_prepare_batch_without_mesh(self):
        """Test prepare_batch without mesh sharding."""
        from aether.data.dataset import prepare_batch
        
        # Create a mock batch
        batch = {'tokens': [[1, 2, 3, 4], [5, 6, 7, 8]]}
        
        result = prepare_batch(batch, mesh=None)
        
        assert isinstance(result, jnp.ndarray)
        assert result.shape == (2, 4)
        assert jnp.array_equal(result, jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]]))
    
    def test_prepare_batch_returns_array(self):
        """Test that prepare_batch returns a JAX array."""
        from aether.data.dataset import prepare_batch
        
        batch = {'tokens': [[1, 2, 3], [4, 5, 6]]}
        result = prepare_batch(batch, mesh=None)
        
        assert hasattr(result, 'shape')
        assert hasattr(result, 'dtype')


class TestDataModuleExports:
    """Tests for data module exports."""
    
    def test_data_module_has_all_exports(self):
        """Test that data module exports all expected functions."""
        from aether.data import (
            create_data_iterators,
            create_validation_iterator,
            create_training_iterator,
            prepare_batch,
            process_dataset
        )
        
        assert callable(create_data_iterators)
        assert callable(create_validation_iterator)
        assert callable(create_training_iterator)
        assert callable(prepare_batch)
        assert callable(process_dataset)


class TestFunctionSignatures:
    """Tests for function signatures."""
    
    def test_create_validation_iterator_signature(self):
        """Test create_validation_iterator has correct parameters."""
        import inspect
        from aether.data.dataset import create_validation_iterator
        
        sig = inspect.signature(create_validation_iterator)
        params = list(sig.parameters.keys())
        
        expected = ['dataset_name', 'split', 'streaming', 'maxlen', 
                    'tokenizer', 'val_set_size', 'batch_size']
        assert params == expected
    
    def test_create_training_iterator_signature(self):
        """Test create_training_iterator has correct parameters."""
        import inspect
        from aether.data.dataset import create_training_iterator
        
        sig = inspect.signature(create_training_iterator)
        params = list(sig.parameters.keys())
        
        expected = ['dataset_name', 'split', 'streaming', 'maxlen', 
                    'tokenizer', 'val_set_size', 'batch_size']
        assert params == expected
    
    def test_create_data_iterators_signature(self):
        """Test create_data_iterators has correct parameters."""
        import inspect
        from aether.data.dataset import create_data_iterators
        
        sig = inspect.signature(create_data_iterators)
        params = list(sig.parameters.keys())
        
        expected = ['dataset_name', 'split', 'streaming', 'maxlen', 
                    'tokenizer', 'val_set_size', 'batch_size']
        assert params == expected
    
    def test_prepare_batch_signature(self):
        """Test prepare_batch has correct parameters."""
        import inspect
        from aether.data.dataset import prepare_batch
        
        sig = inspect.signature(prepare_batch)
        params = list(sig.parameters.keys())
        
        assert 'batch' in params
        assert 'mesh' in params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
