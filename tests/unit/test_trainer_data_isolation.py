"""Test that data loading and iterator resets work independently."""

import sys
import os
sys.path.insert(0, '/home/runner/work/aether/aether')

import unittest.mock as mock


def test_validation_reset_does_not_affect_training():
    """Test that resetting validation iterator doesn't recreate training iterator."""
    
    # Mock the data creation functions to track calls - need to mock the whole module first
    with mock.patch.dict('sys.modules', {
        'aether.data.dataset': mock.MagicMock(),
        'aether.training.steps': mock.MagicMock(),
        'aether.registry': mock.MagicMock(),
        'aether.utils': mock.MagicMock(),
    }):
        # Mock specific functions
        with mock.patch('aether.data.dataset.create_data_iterators') as mock_create_both, \
             mock.patch('aether.data.dataset.create_validation_iterator') as mock_create_val, \
             mock.patch('aether.data.dataset.create_training_iterator') as mock_create_train:
            
            # Setup mock return values
            mock_train_iter = mock.Mock()
            mock_val_iter = mock.Mock()
            mock_create_both.return_value = (mock_train_iter, mock_val_iter)
            mock_create_val.return_value = mock.Mock()
            mock_create_train.return_value = mock.Mock()
            
            # Mock other dependencies to avoid initialization issues
            with mock.patch('aether.training.trainer.setup_mesh'), \
                 mock.patch('aether.training.trainer.get_tokenizer'), \
                 mock.patch('aether.training.trainer.wandb'), \
                 mock.patch('aether.training.trainer.orbax'), \
                 mock.patch('aether.training.trainer.create_model'), \
                 mock.patch('aether.training.trainer.nnx'):
                
                from aether.training.trainer import Trainer
                
                # Create a minimal config mock
                config_mock = mock.Mock()
                config_mock.data.dataset_name = "test"
                config_mock.data.split = "train"
                config_mock.data.streaming = True
                config_mock.data.tokenizer_name = "gpt2"
                config_mock.model.maxlen = 512
                config_mock.training.val_set_size = 1000
                config_mock.training.batch_size = 32
                config_mock.training.learning_rate = 0.001
                config_mock.training.optimizer = "adam"
                config_mock.logging.wandb_project = "test"
                config_mock.logging.checkpoint_dir = "/tmp"
                config_mock.get_model_config_dict.return_value = {}
                config_mock.to_dict.return_value = {}
                
                # Create trainer - this should call create_data_iterators once
                trainer = Trainer(config_mock)
                
                # Verify initial setup called create_data_iterators once
                assert mock_create_both.call_count == 1, f"Expected 1 call to create_data_iterators, got {mock_create_both.call_count}"
                
                # Reset the mock call counts
                mock_create_both.reset_mock()
                mock_create_val.reset_mock()
                mock_create_train.reset_mock()
                
                # Reset validation iterator
                trainer._reset_validation_iterator()
                
                # Verify only validation iterator creation was called
                assert mock_create_val.call_count == 1, f"Expected 1 call to create_validation_iterator, got {mock_create_val.call_count}"
                assert mock_create_train.call_count == 0, f"Expected 0 calls to create_training_iterator, got {mock_create_train.call_count}"
                assert mock_create_both.call_count == 0, f"Expected 0 calls to create_data_iterators, got {mock_create_both.call_count}"
                
                print("✓ Validation reset test passed!")


def test_training_reset_does_not_affect_validation():
    """Test that resetting training iterator doesn't recreate validation iterator."""
    
    with mock.patch.dict('sys.modules', {
        'aether.data.dataset': mock.MagicMock(),
        'aether.training.steps': mock.MagicMock(),
        'aether.registry': mock.MagicMock(),
        'aether.utils': mock.MagicMock(),
    }):
        # Mock the data creation functions to track calls
        with mock.patch('aether.data.dataset.create_data_iterators') as mock_create_both, \
             mock.patch('aether.data.dataset.create_validation_iterator') as mock_create_val, \
             mock.patch('aether.data.dataset.create_training_iterator') as mock_create_train:
            
            # Setup mock return values
            mock_train_iter = mock.Mock()
            mock_val_iter = mock.Mock()
            mock_create_both.return_value = (mock_train_iter, mock_val_iter)
            mock_create_val.return_value = mock.Mock()
            mock_create_train.return_value = mock.Mock()
            
            # Mock other dependencies
            with mock.patch('aether.training.trainer.setup_mesh'), \
                 mock.patch('aether.training.trainer.get_tokenizer'), \
                 mock.patch('aether.training.trainer.wandb'), \
                 mock.patch('aether.training.trainer.orbax'), \
                 mock.patch('aether.training.trainer.create_model'), \
                 mock.patch('aether.training.trainer.nnx'):
                
                from aether.training.trainer import Trainer
                
                # Create a minimal config mock
                config_mock = mock.Mock()
                config_mock.data.dataset_name = "test"
                config_mock.data.split = "train"
                config_mock.data.streaming = True
                config_mock.data.tokenizer_name = "gpt2"
                config_mock.model.maxlen = 512
                config_mock.training.val_set_size = 1000
                config_mock.training.batch_size = 32
                config_mock.training.learning_rate = 0.001
                config_mock.training.optimizer = "adam"
                config_mock.logging.wandb_project = "test"
                config_mock.logging.checkpoint_dir = "/tmp"
                config_mock.get_model_config_dict.return_value = {}
                config_mock.to_dict.return_value = {}
                
                # Create trainer
                trainer = Trainer(config_mock)
                
                # Reset the mock call counts
                mock_create_both.reset_mock()
                mock_create_val.reset_mock()
                mock_create_train.reset_mock()
                
                # Reset training iterator
                trainer._reset_training_iterator()
                
                # Verify only training iterator creation was called
                assert mock_create_train.call_count == 1, f"Expected 1 call to create_training_iterator, got {mock_create_train.call_count}"
                assert mock_create_val.call_count == 0, f"Expected 0 calls to create_validation_iterator, got {mock_create_val.call_count}"
                assert mock_create_both.call_count == 0, f"Expected 0 calls to create_data_iterators, got {mock_create_both.call_count}"
                
                print("✓ Training reset test passed!")


if __name__ == "__main__":
    test_validation_reset_does_not_affect_training()
    test_training_reset_does_not_affect_validation()
    print("✓ All data isolation tests passed!")