"""Unit tests for models."""

import pytest
import jax.numpy as jnp
import flax.nnx as nnx
from aether.models import MiniGPT, TransformerBlock, TokenAndPositionEmbedding


def test_token_position_embedding():
    """Test token and position embedding layer."""
    rngs = nnx.Rngs(42)
    embedding = TokenAndPositionEmbedding(
        maxlen=128,
        vocab_size=1000,
        embed_dim=256,
        rngs=rngs
    )
    
    # Test forward pass
    inputs = jnp.array([[1, 2, 3, 4, 5]])  # batch_size=1, seq_len=5
    outputs = embedding(inputs)
    
    assert outputs.shape == (1, 5, 256)


def test_transformer_block_linear():
    """Test transformer block with linear architecture."""
    rngs = nnx.Rngs(42)
    block = TransformerBlock(
        embed_dim=256,
        num_heads=4,
        ff_dim=512,
        rngs=rngs,
        architecture="linear"
    )
    
    # Test forward pass
    inputs = jnp.ones((2, 10, 256))  # batch_size=2, seq_len=10, embed_dim=256
    outputs = block(inputs, training=True)
    
    assert outputs.shape == inputs.shape


def test_minigpt_linear():
    """Test MiniGPT model with linear architecture."""
    rngs = nnx.Rngs(42)
    model = MiniGPT(
        maxlen=128,
        vocab_size=1000,
        embed_dim=256,
        num_heads=4,
        feed_forward_dim=512,
        num_transformer_blocks=2,
        rngs=rngs,
        architecture="linear"
    )
    
    # Test forward pass
    inputs = jnp.array([[1, 2, 3, 4, 5]])  # batch_size=1, seq_len=5
    outputs = model(inputs, training=True)
    
    assert outputs.shape == (1, 5, 1000)  # (batch_size, seq_len, vocab_size)


def test_minigpt_config():
    """Test MiniGPT configuration methods."""
    rngs = nnx.Rngs(42)
    config = {
        "maxlen": 128,
        "vocab_size": 1000,
        "embed_dim": 256,
        "num_heads": 4,
        "feed_forward_dim": 512,
        "num_transformer_blocks": 2,
        "architecture": "linear",
        "mlp_dim_multiplier": 4.0
    }
    
    model = MiniGPT.from_config(config, rngs)
    retrieved_config = model.get_config()
    
    # Check that configurations match
    for key, value in config.items():
        assert retrieved_config[key] == value


def test_transformer_block_mlp_multiplier():
    """Test transformer block with custom MLP multiplier."""
    rngs = nnx.Rngs(42)
    embed_dim = 256
    
    # Test with custom multiplier
    block = TransformerBlock(
        embed_dim=embed_dim,
        num_heads=4,
        ff_dim=512,
        rngs=rngs,
        architecture="linear",
        mlp_dim_multiplier=2.5
    )
    
    # Test forward pass
    inputs = jnp.ones((2, 10, embed_dim))  # batch_size=2, seq_len=10, embed_dim=256
    outputs = block(inputs, training=True)
    
    assert outputs.shape == inputs.shape
    assert block.mlp_dim_multiplier == 2.5


def test_minigpt_custom_mlp_multiplier():
    """Test MiniGPT model with custom MLP multiplier."""
    rngs = nnx.Rngs(42)
    model = MiniGPT(
        maxlen=128,
        vocab_size=1000,
        embed_dim=256,
        num_heads=4,
        feed_forward_dim=512,
        num_transformer_blocks=2,
        rngs=rngs,
        architecture="linear",
        mlp_dim_multiplier=3.0
    )
    
    # Test forward pass
    inputs = jnp.array([[1, 2, 3, 4, 5]])  # batch_size=1, seq_len=5
    outputs = model(inputs, training=True)
    
    assert outputs.shape == (1, 5, 1000)  # (batch_size, seq_len, vocab_size)
    assert model.mlp_dim_multiplier == 3.0
    
    # Check that transformer blocks have the correct multiplier
    for transformer_block in model.transformer_blocks:
        assert transformer_block.mlp_dim_multiplier == 3.0


def test_minigpt_config_with_mlp_multiplier():
    """Test MiniGPT configuration methods with MLP multiplier."""
    rngs = nnx.Rngs(42)
    config = {
        "maxlen": 128,
        "vocab_size": 1000,
        "embed_dim": 256,
        "num_heads": 4,
        "feed_forward_dim": 512,
        "num_transformer_blocks": 2,
        "architecture": "linear",
        "mlp_dim_multiplier": 2.0
    }
    
    model = MiniGPT.from_config(config, rngs)
    retrieved_config = model.get_config()
    
    # Check that configurations match including the new parameter
    for key, value in config.items():
        assert retrieved_config[key] == value
    
    # Verify the multiplier is correctly set
    assert model.mlp_dim_multiplier == 2.0


if __name__ == "__main__":
    pytest.main([__file__])