"""Unit tests for mixed precision training functionality."""

import pytest
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from aether.models import MiniGPT
from aether.training.steps import loss_fn, train_step, eval_step, apply_mixed_precision


def test_apply_mixed_precision():
    """Test mixed precision parameter casting."""
    # Create test parameters
    params = {
        'weights': jnp.ones((10, 5), dtype=jnp.float32),
        'bias': jnp.zeros(5, dtype=jnp.float32),
        'integers': jnp.array([1, 2, 3], dtype=jnp.int32)
    }
    
    # Test fp16 casting
    fp16_params = apply_mixed_precision(params, "fp16")
    assert fp16_params['weights'].dtype == jnp.float16
    assert fp16_params['bias'].dtype == jnp.float16
    assert fp16_params['integers'].dtype == jnp.int32  # Should not change
    
    # Test bfloat16 casting
    bf16_params = apply_mixed_precision(params, "bfloat16")
    assert bf16_params['weights'].dtype == jnp.bfloat16
    assert bf16_params['bias'].dtype == jnp.bfloat16
    assert bf16_params['integers'].dtype == jnp.int32  # Should not change
    
    # Test no mixed precision
    no_cast_params = apply_mixed_precision(params, None)
    assert no_cast_params['weights'].dtype == jnp.float32
    assert no_cast_params['bias'].dtype == jnp.float32


def test_mixed_precision_loss_fn():
    """Test loss function with standard precision."""
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
    
    # Create a small batch
    batch = jnp.ones((2, 64), dtype=jnp.int32)
    
    # Test standard loss function
    loss, logits = loss_fn(model, batch, training=True)
    assert loss.dtype == jnp.float32
    assert jnp.isfinite(loss)
    assert logits.shape == (2, 63, 1000)  # sequence length reduced by 1


def test_mixed_precision_train_step():
    """Test training step works correctly."""
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
    
    # Create optimizer
    optimizer_fn = optax.adam(0.001)
    optimizer = nnx.Optimizer(model, optimizer_fn, wrt=nnx.Param)
    
    # Create a small batch
    batch = jnp.ones((2, 64), dtype=jnp.int32)
    
    # Test training step
    loss, updated_model, updated_optimizer = train_step(
        model, optimizer, batch
    )
    
    assert loss.dtype == jnp.float32
    assert isinstance(updated_model, MiniGPT)
    assert isinstance(updated_optimizer, nnx.Optimizer)


def test_mixed_precision_eval_step():
    """Test evaluation step works correctly."""
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
    
    # Create a small batch
    batch = jnp.ones((2, 64), dtype=jnp.int32)
    
    # Test evaluation step
    loss = eval_step(model, batch)
    
    assert loss.dtype == jnp.float32
    assert jnp.isfinite(loss)


if __name__ == "__main__":
    pytest.main([__file__])