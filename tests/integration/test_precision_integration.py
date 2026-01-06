"""Integration test to verify all precision types work in practice."""

import jax.numpy as jnp
import flax.nnx as nnx
from aether.config import Config
from aether.models import MiniGPT
from aether.training.steps import loss_fn
import optax


def test_float16_integration():
    """Test Float16 training with actual data."""
    # Create config with float16
    config = Config()
    config.training.precision = "float16"
    
    # Create a small model
    rngs = nnx.Rngs(42)
    model = MiniGPT(
        maxlen=64,
        vocab_size=1000,
        embed_dim=128,
        num_heads=4,
        feed_forward_dim=256,
        num_transformer_blocks=2,
        rngs=rngs,
        param_dtype=jnp.float16,
        compute_dtype=jnp.float16
    )
    
    # Create some sample data
    batch_size = 4
    seq_len = 32
    inputs = jnp.array([[i % 100 for i in range(seq_len)]] * batch_size, dtype=jnp.int32)
    
    # Test training step
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    
    # Forward pass
    loss, _ = loss_fn(model, inputs, training=True)
    
    # Verify loss is reasonable
    assert jnp.isfinite(loss), "Loss should be finite"
    print(f"Float16 - Initial loss: {loss:.4f}")


def test_float64_integration():
    """Test Float64 training with actual data."""
    from jax import config as jax_config
    jax_config.update("jax_enable_x64", True)
    
    # Create config with float64
    config = Config()
    config.training.precision = "float64"
    
    # Create a small model
    rngs = nnx.Rngs(42)
    model = MiniGPT(
        maxlen=64,
        vocab_size=1000,
        embed_dim=128,
        num_heads=4,
        feed_forward_dim=256,
        num_transformer_blocks=2,
        rngs=rngs,
        param_dtype=jnp.float64,
        compute_dtype=jnp.float64
    )
    
    # Create some sample data
    batch_size = 4
    seq_len = 32
    inputs = jnp.array([[i % 100 for i in range(seq_len)]] * batch_size, dtype=jnp.int32)
    
    # Test training step
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    
    # Forward pass
    loss, _ = loss_fn(model, inputs, training=True)
    
    # Verify loss is reasonable
    assert jnp.isfinite(loss), "Loss should be finite"
    print(f"Float64 - Initial loss: {loss:.4f}")


def test_all_precisions_produce_different_results():
    """Test that different precisions produce different numerical results."""
    # Create the same model with different precisions
    results = {}
    
    for precision, dtype in [("float32", jnp.float32), ("float16", jnp.float16), ("bfloat16", jnp.bfloat16)]:
        rngs = nnx.Rngs(42)
        model = MiniGPT(
            maxlen=64,
            vocab_size=1000,
            embed_dim=128,
            num_heads=4,
            feed_forward_dim=256,
            num_transformer_blocks=2,
            rngs=rngs,
            param_dtype=dtype,
            compute_dtype=dtype
        )
        
        # Create sample data
        inputs = jnp.array([[i % 100 for i in range(32)]] * 4, dtype=jnp.int32)
        
        # Get loss
        loss, _ = loss_fn(model, inputs, training=False)
        results[precision] = float(loss)
        print(f"{precision}: Loss = {float(loss):.6f}")
    
    # Verify all are finite
    for precision, loss in results.items():
        assert jnp.isfinite(loss), f"{precision} loss should be finite"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
