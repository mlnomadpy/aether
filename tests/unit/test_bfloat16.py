"""Tests for BFloat16 precision support."""

import jax.numpy as jnp
import flax.nnx as nnx
from aether.config import Config, TrainingConfig
from aether.models import MiniGPT


def test_bfloat16_model_creation():
    """Test that models can be created with BFloat16 precision."""
    # Create model with bfloat16
    rngs = nnx.Rngs(42)
    model = MiniGPT(
        maxlen=64,
        vocab_size=1000,
        embed_dim=128,
        num_heads=4,
        feed_forward_dim=256,
        num_transformer_blocks=2,
        rngs=rngs,
        architecture="linear",
        param_dtype=jnp.bfloat16,
        compute_dtype=jnp.bfloat16
    )
    
    # Test forward pass
    batch_size = 4
    seq_len = 32
    inputs = jnp.array([[1, 2, 3, 4] * (seq_len // 4)] * batch_size, dtype=jnp.int32)
    
    # Forward pass
    outputs = model(inputs, training=False)
    
    # Check output dtype
    assert outputs.dtype == jnp.bfloat16, f"Expected output dtype to be bfloat16, got {outputs.dtype}"
    
    # Check parameter dtypes
    assert model.output_layer.kernel.dtype == jnp.bfloat16, "Expected output layer kernel to be bfloat16"
    assert model.embedding_layer.token_emb.embedding.dtype == jnp.bfloat16, "Expected token embedding to be bfloat16"


def test_float32_model_backward_compatibility():
    """Test that Float32 models still work (backward compatibility)."""
    # Create model with float32 (default behavior)
    rngs = nnx.Rngs(42)
    model = MiniGPT(
        maxlen=64,
        vocab_size=1000,
        embed_dim=128,
        num_heads=4,
        feed_forward_dim=256,
        num_transformer_blocks=2,
        rngs=rngs,
        architecture="linear",
        param_dtype=jnp.float32,
        compute_dtype=jnp.float32
    )
    
    # Test forward pass
    batch_size = 4
    seq_len = 32
    inputs = jnp.array([[1, 2, 3, 4] * (seq_len // 4)] * batch_size, dtype=jnp.int32)
    
    # Forward pass
    outputs = model(inputs, training=False)
    
    # Check output dtype
    assert outputs.dtype == jnp.float32, f"Expected output dtype to be float32, got {outputs.dtype}"
    
    # Check parameter dtypes
    assert model.output_layer.kernel.dtype == jnp.float32, "Expected output layer kernel to be float32"
    assert model.embedding_layer.token_emb.embedding.dtype == jnp.float32, "Expected token embedding to be float32"


def test_training_config_precision_field():
    """Test that TrainingConfig has precision field with correct default."""
    config = TrainingConfig()
    
    # Check default precision
    assert config.precision == "float32", f"Expected default precision to be 'float32', got {config.precision}"
    
    # Test setting to bfloat16
    config.precision = "bfloat16"
    assert config.precision == "bfloat16", f"Expected precision to be 'bfloat16', got {config.precision}"


def test_config_with_bfloat16_precision():
    """Test that Config can be created and saved/loaded with BFloat16 precision."""
    # Create config with bfloat16
    config = Config()
    config.training.precision = "bfloat16"
    
    # Convert to dict and back
    config_dict = config.to_dict()
    assert config_dict["training"]["precision"] == "bfloat16", "Expected precision to be preserved in dict"
    
    # Create new config from dict
    new_config = Config.from_dict(config_dict)
    assert new_config.training.precision == "bfloat16", "Expected precision to be preserved after from_dict"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])