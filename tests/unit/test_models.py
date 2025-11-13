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


def test_transformer_block_linear_without_norm():
    """Test transformer block with linear architecture without layer normalization."""
    rngs = nnx.Rngs(42)
    block = TransformerBlock(
        embed_dim=256,
        num_heads=4,
        ff_dim=512,
        rngs=rngs,
        architecture="linear",
        use_layer_norm=False
    )
    
    # Verify that layer norm layers are None
    assert block.layer_norm1 is None
    assert block.layer_norm2 is None
    
    # Test forward pass
    inputs = jnp.ones((2, 10, 256))  # batch_size=2, seq_len=10, embed_dim=256
    outputs = block(inputs, training=True)
    
    assert outputs.shape == inputs.shape


def test_transformer_block_linear_with_norm():
    """Test transformer block with linear architecture with layer normalization."""
    rngs = nnx.Rngs(42)
    block = TransformerBlock(
        embed_dim=256,
        num_heads=4,
        ff_dim=512,
        rngs=rngs,
        architecture="linear",
        use_layer_norm=True
    )
    
    # Verify that layer norm layers are created
    assert block.layer_norm1 is not None
    assert block.layer_norm2 is not None
    
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
        "architecture": "linear"
    }
    
    model = MiniGPT.from_config(config, rngs)
    retrieved_config = model.get_config()
    
    # Check that configurations match
    for key, value in config.items():
        assert retrieved_config[key] == value


def test_minigpt_attention_block_reuse():
    """Test MiniGPT with attention block reuse functionality."""
    rngs = nnx.Rngs(42)
    
    # Test with reuse = 1 (default, no reuse)
    model1 = MiniGPT(
        maxlen=128,
        vocab_size=1000,
        embed_dim=256,
        num_heads=4,
        feed_forward_dim=512,
        num_transformer_blocks=2,
        rngs=rngs,
        architecture="linear",
        attention_block_reuse=1
    )
    
    inputs = jnp.array([[1, 2, 3, 4, 5]])
    outputs1 = model1(inputs, training=True)
    assert outputs1.shape == (1, 5, 1000)
    
    # Test with reuse = 3
    rngs2 = nnx.Rngs(42)
    model2 = MiniGPT(
        maxlen=128,
        vocab_size=1000,
        embed_dim=256,
        num_heads=4,
        feed_forward_dim=512,
        num_transformer_blocks=2,
        rngs=rngs2,
        architecture="linear",
        attention_block_reuse=3
    )
    
    outputs2 = model2(inputs, training=True)
    assert outputs2.shape == (1, 5, 1000)
    
    # Verify config contains attention_block_reuse
    config = model2.get_config()
    assert "attention_block_reuse" in config
    assert config["attention_block_reuse"] == 3


if __name__ == "__main__":
    pytest.main([__file__])