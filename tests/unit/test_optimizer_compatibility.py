"""Tests for Flax NNX optimizer compatibility."""

import jax.numpy as jnp
import flax.nnx as nnx
import optax
from aether.models import MiniGPT


def test_model_is_nnx_module():
    """Test that MiniGPT properly inherits from nnx.Module."""
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
    
    # Model should be a proper nnx.Module
    assert isinstance(model, nnx.Module)
    assert issubclass(MiniGPT, nnx.Module)


def test_optimizer_creation():
    """Test that we can create an optimizer with the model."""
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
    
    # Should be able to create optimizer with wrt parameter
    optimizer_fn = optax.adam(0.001)
    optimizer = nnx.Optimizer(model, optimizer_fn, wrt=nnx.Param)
    
    assert isinstance(optimizer, nnx.Optimizer)


def test_nnx_state_call():
    """Test that nnx.state works with the model."""
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
    
    # Should be able to extract state without errors
    state = nnx.state(model, nnx.Param)
    assert state is not None


def test_model_works_with_nnx_split():
    """Test that the model works with nnx graph operations like split."""
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
    
    # This should not raise "Unsupported type" error
    graph_def, params, other = nnx.split(model, nnx.Param, ...)
    assert graph_def is not None
    assert params is not None
    
    # Should be able to recreate model from split representation
    model_restored = nnx.merge(graph_def, params, other)
    assert isinstance(model_restored, MiniGPT)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])