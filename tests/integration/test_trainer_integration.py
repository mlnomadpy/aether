"""Integration test for trainer with new optimizers and schedulers."""

import sys
import os
sys.path.insert(0, '/home/runner/work/aether/aether')

import tempfile
import shutil
import json
from aether.config import Config


def test_trainer_with_cosine_scheduler():
    """Test trainer creation with cosine scheduler."""
    
    # Create a temporary config for testing
    config_dict = {
        "model": {
            "name": "minigpt-linear",
            "architecture": "linear",
            "maxlen": 128,
            "vocab_size": 1000,
            "embed_dim": 64,
            "num_heads": 4,
            "feed_forward_dim": 128,
            "num_transformer_blocks": 2,
            "dropout_rate": 0.1
        },
        "training": {
            "batch_size": 4,
            "learning_rate": 0.001,
            "max_tokens_to_process": 1000,
            "eval_interval": 100,
            "eval_steps": 10,
            "val_set_size": 100,
            "checkpoint_interval": 200,
            "optimizer": "adamw",
            "lr_scheduler": "cosine",
            "lr_scheduler_alpha": 0.1,
            "lr_scheduler_warmup_steps": None,
            "momentum": 0.9,
            "weight_decay": 0.01
        },
        "data": {
            "dataset_name": "HuggingFaceFW/fineweb",
            "split": "train",
            "streaming": True,
            "tokenizer_name": "gpt2"
        },
        "logging": {
            "wandb_project": "test-aether-cosine",
            "checkpoint_dir": "./test_checkpoints",
            "log_level": "INFO",
            "log_file": None
        }
    }
    
    config = Config.from_dict(config_dict)
    
    # Test that the config was created properly
    assert config.training.lr_scheduler == "cosine"
    assert config.training.optimizer == "adamw"
    assert config.training.lr_scheduler_alpha == 0.1
    assert config.training.weight_decay == 0.01
    
    print("✓ Config creation with cosine scheduler passed!")


def test_all_optimizers_valid():
    """Test that all supported optimizers can be configured."""
    
    optimizers = ["adam", "adamw", "sgd", "rmsprop", "novograd", 
                  "adagrad", "adadelta", "adamax", "nadam", "lion"]
    
    for optimizer in optimizers:
        config_dict = {
            "model": {
                "name": "minigpt-linear",
                "architecture": "linear",
                "maxlen": 128,
                "vocab_size": 1000,
                "embed_dim": 64,
                "num_heads": 4,
                "feed_forward_dim": 128,
                "num_transformer_blocks": 2,
                "dropout_rate": 0.1
            },
            "training": {
                "batch_size": 4,
                "learning_rate": 0.001,
                "max_tokens_to_process": 1000,
                "eval_interval": 100,
                "eval_steps": 10,
                "val_set_size": 100,
                "checkpoint_interval": 200,
                "optimizer": optimizer,
                "lr_scheduler": "constant",
                "momentum": 0.9,
                "weight_decay": 0.01
            },
            "data": {
                "dataset_name": "HuggingFaceFW/fineweb",
                "split": "train",
                "streaming": True,
                "tokenizer_name": "gpt2"
            },
            "logging": {
                "wandb_project": f"test-aether-{optimizer}",
                "checkpoint_dir": "./test_checkpoints",
                "log_level": "INFO",
                "log_file": None
            }
        }
        
        config = Config.from_dict(config_dict)
        assert config.training.optimizer == optimizer
        print(f"✓ Optimizer {optimizer} configuration valid!")


def test_all_schedulers_valid():
    """Test that all supported schedulers can be configured."""
    
    schedulers = ["constant", "linear", "cosine", "warmup_cosine"]
    
    for scheduler in schedulers:
        config_dict = {
            "model": {
                "name": "minigpt-linear",
                "architecture": "linear",
                "maxlen": 128,
                "vocab_size": 1000,
                "embed_dim": 64,
                "num_heads": 4,
                "feed_forward_dim": 128,
                "num_transformer_blocks": 2,
                "dropout_rate": 0.1
            },
            "training": {
                "batch_size": 4,
                "learning_rate": 0.001,
                "max_tokens_to_process": 1000,
                "eval_interval": 100,
                "eval_steps": 10,
                "val_set_size": 100,
                "checkpoint_interval": 200,
                "optimizer": "adam",
                "lr_scheduler": scheduler,
                "lr_scheduler_alpha": 0.1,
                "lr_scheduler_warmup_steps": 50 if scheduler == "warmup_cosine" else None,
                "momentum": 0.9,
                "weight_decay": 0.01
            },
            "data": {
                "dataset_name": "HuggingFaceFW/fineweb",
                "split": "train",
                "streaming": True,
                "tokenizer_name": "gpt2"
            },
            "logging": {
                "wandb_project": f"test-aether-{scheduler}",
                "checkpoint_dir": "./test_checkpoints",
                "log_level": "INFO",
                "log_file": None
            }
        }
        
        config = Config.from_dict(config_dict)
        assert config.training.lr_scheduler == scheduler
        print(f"✓ Scheduler {scheduler} configuration valid!")


def test_config_file_parsing():
    """Test that the new config files can be parsed correctly."""
    
    # Test cosine config
    with open('/home/runner/work/aether/aether/configs/cosine_adamw_config.json', 'r') as f:
        cosine_config_dict = json.load(f)
    
    cosine_config = Config.from_dict(cosine_config_dict)
    assert cosine_config.training.lr_scheduler == "cosine"
    assert cosine_config.training.optimizer == "adamw"
    assert cosine_config.training.weight_decay == 0.01
    print("✓ Cosine AdamW config file parsing passed!")
    
    # Test SGD warmup cosine config
    with open('/home/runner/work/aether/aether/configs/sgd_warmup_cosine_config.json', 'r') as f:
        sgd_config_dict = json.load(f)
    
    sgd_config = Config.from_dict(sgd_config_dict)
    assert sgd_config.training.lr_scheduler == "warmup_cosine"
    assert sgd_config.training.optimizer == "sgd"
    assert sgd_config.training.lr_scheduler_warmup_steps == 5000
    print("✓ SGD warmup cosine config file parsing passed!")


if __name__ == "__main__":
    test_trainer_with_cosine_scheduler()
    test_all_optimizers_valid()
    test_all_schedulers_valid()
    test_config_file_parsing()
    print("✓ All integration tests passed!")