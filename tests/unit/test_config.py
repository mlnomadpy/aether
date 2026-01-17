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


def test_config_with_yat_epsilon():
    """Test configuration with yat_epsilon parameter."""
    config_dict = {
        "model": {
            "name": "aether-yat",
            "architecture": "aether_yat",
            "yat_epsilon": 0.1
        }
    }
    
    config = Config.from_dict(config_dict)
    
    assert config.model.name == "aether-yat"
    assert config.model.architecture == "aether_yat"
    assert config.model.yat_epsilon == 0.1


def test_config_with_num_random_features():
    """Test configuration with num_random_features parameter."""
    config_dict = {
        "model": {
            "name": "aether-yat-performer",
            "architecture": "aether_yat_performer",
            "yat_epsilon": 0.1,
            "num_random_features": 256
        }
    }
    
    config = Config.from_dict(config_dict)
    
    assert config.model.name == "aether-yat-performer"
    assert config.model.architecture == "aether_yat_performer"
    assert config.model.yat_epsilon == 0.1
    assert config.model.num_random_features == 256


def test_yat_config_file_loading():
    """Test loading actual Yat config files."""
    import os
    
    # Get the path to the configs directory
    test_dir = os.path.dirname(__file__)
    repo_root = os.path.dirname(os.path.dirname(test_dir))
    aether_yat_config_path = os.path.join(repo_root, "configs", "aether_yat_config.json")
    
    if os.path.exists(aether_yat_config_path):
        config = Config.from_file(aether_yat_config_path)
        assert config.model.name == "aether-yat"
        assert config.model.architecture == "aether_yat"
        assert config.model.yat_epsilon == 0.1


def test_yat_performer_config_file_loading():
    """Test loading actual Yat Performer config files."""
    import os
    
    # Get the path to the configs directory
    test_dir = os.path.dirname(__file__)
    repo_root = os.path.dirname(os.path.dirname(test_dir))
    aether_yat_performer_config_path = os.path.join(repo_root, "configs", "aether_yat_performer_config.json")
    
    if os.path.exists(aether_yat_performer_config_path):
        config = Config.from_file(aether_yat_performer_config_path)
        assert config.model.name == "aether-yat-performer"
        assert config.model.architecture == "aether_yat_performer"
        assert config.model.yat_epsilon == 0.1
        assert config.model.num_random_features == 256


if __name__ == "__main__":
    pytest.main([__file__])