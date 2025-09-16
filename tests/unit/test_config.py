"""Unit tests for configuration management."""

import pytest
import json
import tempfile
import os
from aether.config import Config, ModelConfig, TrainingConfig, DataConfig, LoggingConfig


def test_config_creation():
    """Test configuration creation with defaults."""
    config = Config()
    
    assert config.model.architecture == "linear"
    assert config.training.batch_size == 32
    assert config.data.dataset_name == "HuggingFaceFW/fineweb"
    assert config.logging.wandb_project == "aether-training"


def test_config_from_dict():
    """Test configuration creation from dictionary."""
    config_dict = {
        "model": {
            "name": "test-model",
            "embed_dim": 512
        },
        "training": {
            "batch_size": 16,
            "learning_rate": 0.001
        }
    }
    
    config = Config.from_dict(config_dict)
    
    assert config.model.name == "test-model"
    assert config.model.embed_dim == 512
    assert config.training.batch_size == 16
    assert config.training.learning_rate == 0.001
    # Defaults should be preserved for other fields
    assert config.data.dataset_name == "HuggingFaceFW/fineweb"


def test_config_file_operations():
    """Test saving and loading configuration files."""
    config = Config()
    config.model.embed_dim = 512
    config.training.batch_size = 16
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "test_config.json")
        
        # Save configuration
        config.save(config_path)
        assert os.path.exists(config_path)
        
        # Load configuration
        loaded_config = Config.from_file(config_path)
        
        assert loaded_config.model.embed_dim == 512
        assert loaded_config.training.batch_size == 16


def test_get_model_config_dict():
    """Test getting model configuration as dictionary."""
    config = Config()
    config.model.name = "test-model"
    config.model.embed_dim = 512
    
    model_config = config.get_model_config_dict()
    
    # Name should be removed
    assert "name" not in model_config
    assert model_config["embed_dim"] == 512
    assert model_config["architecture"] == "linear"


def test_mixed_precision_config():
    """Test mixed precision configuration."""
    config = Config()
    
    # Test default (no mixed precision)
    assert config.training.mixed_precision is None
    
    # Test setting mixed precision modes
    config.training.mixed_precision = "fp16"
    assert config.training.mixed_precision == "fp16"
    
    config.training.mixed_precision = "bfloat16"
    assert config.training.mixed_precision == "bfloat16"
    
    # Test loading from dict with mixed precision
    config_dict = {
        "training": {
            "mixed_precision": "bfloat16",
            "batch_size": 16
        }
    }
    
    config = Config.from_dict(config_dict)
    assert config.training.mixed_precision == "bfloat16"
    assert config.training.batch_size == 16


if __name__ == "__main__":
    pytest.main([__file__])