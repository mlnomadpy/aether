"""Tests for MLM (Masked Language Modeling) functionality."""

import pytest
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from aether.config import Config, TrainingConfig
from aether.data.dataset import apply_mlm_masking, prepare_batch
from aether.training.steps import mlm_loss_fn, clm_loss_fn, loss_fn
from aether.models import MiniGPT


class MockTokenizer:
    """Mock tokenizer for testing."""
    def __init__(self):
        self.vocab_size = 1000
        self.mask_token_id = 999
        self.pad_token_id = 0
    
    def encode(self, text):
        # Simple mock encoding - just convert to list of ints
        return [i % self.vocab_size for i in range(len(text.split()))]


def test_training_config_mlm_defaults():
    """Test that MLM configuration has correct defaults."""
    config = TrainingConfig()
    
    assert config.training_mode == "clm"  # Default should be CLM
    assert config.mlm_mask_prob == 0.15
    assert config.mlm_replace_prob == 0.8
    assert config.mlm_random_prob == 0.1
    assert config.final_evaluation == False


def test_training_config_mlm_mode():
    """Test MLM mode configuration."""
    config = TrainingConfig(
        training_mode="mlm",
        mlm_mask_prob=0.2,
        final_evaluation=True
    )
    
    assert config.training_mode == "mlm"
    assert config.mlm_mask_prob == 0.2
    assert config.final_evaluation == True


def test_apply_mlm_masking():
    """Test MLM masking function."""
    tokenizer = MockTokenizer()
    tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    masked_tokens, mask_labels = apply_mlm_masking(
        tokens, tokenizer, mask_prob=0.5, replace_prob=0.8, random_prob=0.1
    )
    
    # Check that we got some masking
    assert len(masked_tokens) == len(tokens)
    assert len(mask_labels) == len(tokens)
    
    # Check that some tokens were masked (labels != -100)
    masked_positions = [i for i, label in enumerate(mask_labels) if label != -100]
    assert len(masked_positions) > 0  # Should have some masked tokens
    
    # Check that masked positions have correct original labels
    for i in masked_positions:
        assert mask_labels[i] == tokens[i]


def test_prepare_batch_clm():
    """Test batch preparation for CLM mode."""
    batch = {'tokens': [[1, 2, 3, 4], [5, 6, 7, 8]]}
    
    result = prepare_batch(batch, mesh=None, training_mode="clm")
    
    assert 'tokens' in result
    assert result['tokens'].shape == (2, 4)


def test_prepare_batch_mlm():
    """Test batch preparation for MLM mode."""
    batch = {
        'masked_tokens': [[1, 999, 3, 4], [5, 999, 7, 8]],
        'mask_labels': [[-100, 2, -100, -100], [-100, 6, -100, -100]]
    }
    
    result = prepare_batch(batch, mesh=None, training_mode="mlm")
    
    assert 'masked_tokens' in result
    assert 'mask_labels' in result
    assert result['masked_tokens'].shape == (2, 4)
    assert result['mask_labels'].shape == (2, 4)


def test_clm_loss_fn():
    """Test CLM loss function."""
    # Create a simple model
    rngs = nnx.Rngs(42)
    model = MiniGPT(
        maxlen=10,
        vocab_size=100,
        embed_dim=32,
        num_heads=2,
        feed_forward_dim=32,
        num_transformer_blocks=1,
        rngs=rngs
    )
    
    # Create test batch
    batch = jnp.array([[1, 2, 3, 4, 5]])
    
    loss, logits = clm_loss_fn(model, batch, training=False)
    
    assert loss.shape == ()  # Scalar loss
    assert logits.shape == (1, 4, 100)  # (batch, seq_len-1, vocab_size) due to shifting


def test_mlm_loss_fn():
    """Test MLM loss function."""
    # Create a simple model
    rngs = nnx.Rngs(42)
    model = MiniGPT(
        maxlen=10,
        vocab_size=100,
        embed_dim=32,
        num_heads=2,
        feed_forward_dim=32,
        num_transformer_blocks=1,
        rngs=rngs
    )
    
    # Create test batch with masked tokens
    batch = {
        'masked_tokens': jnp.array([[1, 99, 3, 4, 5]]),  # 99 is mask token
        'mask_labels': jnp.array([[-100, 2, -100, -100, -100]])  # Only position 1 is masked
    }
    
    loss, logits = mlm_loss_fn(model, batch, training=False)
    
    assert loss.shape == ()  # Scalar loss
    assert logits.shape == (1, 5, 100)  # (batch, seq_len, vocab_size)


def test_loss_fn_mode_dispatch():
    """Test that loss_fn correctly dispatches to CLM or MLM functions."""
    # Create a simple model
    rngs = nnx.Rngs(42)
    model = MiniGPT(
        maxlen=10,
        vocab_size=100,
        embed_dim=32,
        num_heads=2,
        feed_forward_dim=32,
        num_transformer_blocks=1,
        rngs=rngs
    )
    
    # Test CLM mode
    clm_batch = jnp.array([[1, 2, 3, 4, 5]])
    clm_loss, clm_logits = loss_fn(model, clm_batch, training=False, training_mode="clm")
    assert clm_loss.shape == ()
    
    # Test MLM mode
    mlm_batch = {
        'masked_tokens': jnp.array([[1, 99, 3, 4, 5]]),
        'mask_labels': jnp.array([[-100, 2, -100, -100, -100]])
    }
    mlm_loss, mlm_logits = loss_fn(model, mlm_batch, training=False, training_mode="mlm")
    assert mlm_loss.shape == ()


def test_config_integration():
    """Test that Config class properly handles MLM configuration."""
    config_dict = {
        "training": {
            "training_mode": "mlm",
            "mlm_mask_prob": 0.2,
            "final_evaluation": True
        }
    }
    
    config = Config.from_dict(config_dict)
    
    assert config.training.training_mode == "mlm"
    assert config.training.mlm_mask_prob == 0.2
    assert config.training.final_evaluation == True
    
    # Test round-trip conversion
    dict_back = config.to_dict()
    assert dict_back["training"]["training_mode"] == "mlm"
    assert dict_back["training"]["mlm_mask_prob"] == 0.2