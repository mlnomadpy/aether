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


def test_transformer_block_yat():
    """Test transformer block with YAT architecture."""
    pytest.importorskip("nmn", reason="nmn package required for YAT architecture")
    
    rngs = nnx.Rngs(42)
    block = TransformerBlock(
        embed_dim=256,
        num_heads=4,
        ff_dim=512,
        rngs=rngs,
        architecture="yat",
        residual_scale=0.1
    )
    
    # Test forward pass
    inputs = jnp.ones((2, 10, 256))
    outputs = block(inputs, training=True)
    
    assert outputs.shape == inputs.shape
    assert not jnp.any(jnp.isnan(outputs)), "YAT block output contains NaN"


def test_minigpt_yat():
    """Test MiniGPT model with YAT architecture."""
    pytest.importorskip("nmn", reason="nmn package required for YAT architecture")
    
    rngs = nnx.Rngs(42)
    model = MiniGPT(
        maxlen=128,
        vocab_size=1000,
        embed_dim=256,
        num_heads=4,
        feed_forward_dim=512,
        num_transformer_blocks=4,
        rngs=rngs,
        architecture="yat"
    )
    
    # Test forward pass
    inputs = jnp.array([[1, 2, 3, 4, 5]])
    outputs = model(inputs, training=True)
    
    assert outputs.shape == (1, 5, 1000)
    assert not jnp.any(jnp.isnan(outputs)), "YAT model output contains NaN"


def test_minigpt_yat_training_stability():
    """Test that YAT architecture training is stable (no NaN loss)."""
    pytest.importorskip("nmn", reason="nmn package required for YAT architecture")
    import optax
    
    rngs = nnx.Rngs(42)
    model = MiniGPT(
        maxlen=64,
        vocab_size=1000,
        embed_dim=128,
        num_heads=4,
        feed_forward_dim=256,
        num_transformer_blocks=4,
        rngs=rngs,
        architecture="yat"
    )
    
    # Create a simple training setup
    import jax
    key = jax.random.PRNGKey(0)
    batch = jax.random.randint(key, (2, 32), 0, 1000)
    
    # Simple loss function
    def loss_fn(model, batch):
        logits = model(batch, training=True)
        targets = batch[:, 1:]
        logits = logits[:, :-1, :]
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1)
        )
        return jnp.mean(loss)
    
    # Test a few training steps
    optimizer_fn = optax.adam(0.001)
    opt_state = optimizer_fn.init(nnx.state(model, nnx.Param))
    
    for step in range(10):
        grad_fn = nnx.value_and_grad(loss_fn)
        loss, grads = grad_fn(model, batch)
        
        params = nnx.state(model, nnx.Param)
        grad_state = nnx.state(grads, nnx.Param)
        updates, opt_state = optimizer_fn.update(grad_state, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        nnx.update(model, new_params)
        
        assert not jnp.isnan(loss), f"YAT training produced NaN loss at step {step}"
        assert loss < 1e6, f"YAT training loss exploded at step {step}: {loss}"


if __name__ == "__main__":
    pytest.main([__file__])