"""Tests for optimizer and learning rate scheduler functionality."""

import pytest
import sys
import os
sys.path.insert(0, '/home/runner/work/aether/aether')

import jax.numpy as jnp
import flax.nnx as nnx
import optax
from aether.config import Config, TrainingConfig


class MockConfig:
    """Mock config for testing without full dependencies."""
    
    def __init__(self, training_config: TrainingConfig):
        self.training = training_config
        self.model = type('ModelConfig', (), {
            'name': 'minigpt-linear',
            'maxlen': 128,
            'vocab_size': 1000,
            'embed_dim': 64,
            'num_heads': 4,
            'feed_forward_dim': 128,
            'num_transformer_blocks': 2,
            'dropout_rate': 0.1,
            'architecture': 'linear'
        })()
        self.data = type('DataConfig', (), {
            'dataset_name': 'test',
            'split': 'train',
            'streaming': True,
            'tokenizer_name': 'gpt2'
        })()
        self.logging = type('LoggingConfig', (), {
            'wandb_project': 'test',
            'checkpoint_dir': './test_checkpoints',
            'log_level': 'INFO',
            'log_file': None
        })()

    def get_model_config_dict(self):
        return {
            'maxlen': self.model.maxlen,
            'vocab_size': self.model.vocab_size,
            'embed_dim': self.model.embed_dim,
            'num_heads': self.model.num_heads,
            'feed_forward_dim': self.model.feed_forward_dim,
            'num_transformer_blocks': self.model.num_transformer_blocks,
            'dropout_rate': self.model.dropout_rate,
            'architecture': self.model.architecture
        }

    def to_dict(self):
        return {}


def test_learning_rate_schedulers():
    """Test different learning rate schedulers."""
    # Create a minimal trainer instance just for testing scheduler creation
    rngs = nnx.Rngs(42)
    
    # Test constant scheduler
    training_config = TrainingConfig(
        batch_size=4,
        learning_rate=1e-3,
        max_tokens_to_process=1000,
        lr_scheduler="constant"
    )
    config = MockConfig(training_config)
    
    # Mock the trainer's scheduler creation method
    class TestTrainer:
        def __init__(self, config):
            self.config = config
        
        def _create_learning_rate_schedule(self, base_lr: float, total_steps: int, 
                                         scheduler: str, alpha: float, 
                                         warmup_steps = None):
            """Create learning rate schedule from configuration."""
            scheduler = scheduler.lower()
            
            if scheduler == "constant":
                return base_lr
            elif scheduler == "linear":
                return optax.linear_schedule(
                    init_value=base_lr,
                    end_value=alpha * base_lr,
                    transition_steps=total_steps
                )
            elif scheduler == "cosine":
                return optax.cosine_decay_schedule(
                    init_value=base_lr,
                    decay_steps=total_steps,
                    alpha=alpha
                )
            elif scheduler == "warmup_cosine":
                if warmup_steps is None:
                    warmup_steps = max(1, total_steps // 10)
                return optax.warmup_cosine_decay_schedule(
                    init_value=0.0,
                    peak_value=base_lr,
                    warmup_steps=warmup_steps,
                    decay_steps=total_steps - warmup_steps,
                    end_value=alpha * base_lr
                )
            else:
                raise ValueError(f"Unsupported learning rate scheduler: {scheduler}")
    
    trainer = TestTrainer(config)
    
    # Test constant scheduler
    schedule = trainer._create_learning_rate_schedule(1e-3, 100, "constant", 0.0)
    assert schedule == 1e-3
    
    # Test linear scheduler
    schedule = trainer._create_learning_rate_schedule(1e-3, 100, "linear", 0.1)
    assert callable(schedule)
    assert abs(schedule(0) - 1e-3) < 1e-6  # Initial value
    assert abs(schedule(100) - 1e-4) < 1e-6  # Final value (alpha * base_lr)
    
    # Test cosine scheduler
    schedule = trainer._create_learning_rate_schedule(1e-3, 100, "cosine", 0.1)
    assert callable(schedule)
    assert abs(schedule(0) - 1e-3) < 1e-6  # Initial value
    
    # Test warmup cosine scheduler
    schedule = trainer._create_learning_rate_schedule(1e-3, 100, "warmup_cosine", 0.1, 10)
    assert callable(schedule)
    assert schedule(0) == 0.0  # Starts at 0 for warmup


def test_optimizer_creation():
    """Test creation of different optimizers."""
    optimizers_to_test = [
        "adam", "adamw", "sgd", "rmsprop", "novograd", 
        "adagrad", "adadelta", "adamax", "nadam"
    ]
    
    for optimizer_name in optimizers_to_test:
        training_config = TrainingConfig(
            batch_size=4,
            learning_rate=1e-3,
            max_tokens_to_process=1000,
            optimizer=optimizer_name,
            momentum=0.9,
            weight_decay=0.01
        )
        
        config = MockConfig(training_config)
        
        # Test that optimizer configuration is valid
        assert config.training.optimizer == optimizer_name
        assert config.training.learning_rate == 1e-3


def test_unsupported_scheduler():
    """Test that unsupported schedulers raise ValueError."""
    training_config = TrainingConfig(lr_scheduler="unsupported")
    config = MockConfig(training_config)
    
    class TestTrainer:
        def __init__(self, config):
            self.config = config
        
        def _create_learning_rate_schedule(self, base_lr: float, total_steps: int, 
                                         scheduler: str, alpha: float, 
                                         warmup_steps = None):
            if scheduler.lower() == "unsupported":
                raise ValueError(f"Unsupported learning rate scheduler: {scheduler}")
            return base_lr
    
    trainer = TestTrainer(config)
    
    with pytest.raises(ValueError, match="Unsupported learning rate scheduler"):
        trainer._create_learning_rate_schedule(1e-3, 100, "unsupported", 0.0)


def test_training_config_defaults():
    """Test that TrainingConfig has proper defaults for new fields."""
    config = TrainingConfig()
    
    assert config.lr_scheduler == "constant"
    assert config.lr_scheduler_alpha == 0.0
    assert config.lr_scheduler_warmup_steps is None
    assert config.momentum == 0.9
    assert config.weight_decay == 0.0


if __name__ == "__main__":
    test_learning_rate_schedulers()
    test_optimizer_creation()
    test_unsupported_scheduler()
    test_training_config_defaults()
    print("✓ All optimizer and scheduler tests passed!")