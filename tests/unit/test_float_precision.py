"""Tests for Float16 and Float64 precision support."""

import jax.numpy as jnp
import flax.nnx as nnx
from aether.config import Config, TrainingConfig
from aether.models import MiniGPT


def test_float16_model_creation():
    """Test that models can be created with Float16 precision."""
    # Create model with float16
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
        param_dtype=jnp.float16,
        compute_dtype=jnp.float16
    )
    
    # Test forward pass
    batch_size = 4
    seq_len = 32
    inputs = jnp.array([[1, 2, 3, 4] * (seq_len // 4)] * batch_size, dtype=jnp.int32)
    
    # Forward pass
    outputs = model(inputs, training=False)
    
    # Check output dtype
    assert outputs.dtype == jnp.float16, f"Expected output dtype to be float16, got {outputs.dtype}"
    
    # Check parameter dtypes
    assert model.output_layer.kernel.dtype == jnp.float16, "Expected output layer kernel to be float16"
    assert model.embedding_layer.token_emb.embedding.dtype == jnp.float16, "Expected token embedding to be float16"


def test_float64_model_creation():
    """Test that models can be created with Float64 precision."""
    # Enable x64 for JAX
    from jax import config as jax_config
    jax_config.update("jax_enable_x64", True)
    
    # Create model with float64
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
        param_dtype=jnp.float64,
        compute_dtype=jnp.float64
    )
    
    # Test forward pass
    batch_size = 4
    seq_len = 32
    inputs = jnp.array([[1, 2, 3, 4] * (seq_len // 4)] * batch_size, dtype=jnp.int32)
    
    # Forward pass
    outputs = model(inputs, training=False)
    
    # Check output dtype
    assert outputs.dtype == jnp.float64, f"Expected output dtype to be float64, got {outputs.dtype}"
    
    # Check parameter dtypes
    assert model.output_layer.kernel.dtype == jnp.float64, "Expected output layer kernel to be float64"
    assert model.embedding_layer.token_emb.embedding.dtype == jnp.float64, "Expected token embedding to be float64"


def test_float16_training_config():
    """Test that TrainingConfig can be set to float16."""
    config = TrainingConfig()
    
    # Test setting to float16
    config.precision = "float16"
    assert config.precision == "float16", f"Expected precision to be 'float16', got {config.precision}"


def test_float64_training_config():
    """Test that TrainingConfig can be set to float64."""
    config = TrainingConfig()
    
    # Test setting to float64
    config.precision = "float64"
    assert config.precision == "float64", f"Expected precision to be 'float64', got {config.precision}"


def test_config_with_float16_precision():
    """Test that Config can be created and saved/loaded with Float16 precision."""
    # Create config with float16
    config = Config()
    config.training.precision = "float16"
    
    # Convert to dict and back
    config_dict = config.to_dict()
    assert config_dict["training"]["precision"] == "float16", "Expected precision to be preserved in dict"
    
    # Create new config from dict
    new_config = Config.from_dict(config_dict)
    assert new_config.training.precision == "float16", "Expected precision to be preserved after from_dict"


def test_config_with_float64_precision():
    """Test that Config can be created and saved/loaded with Float64 precision."""
    # Create config with float64
    config = Config()
    config.training.precision = "float64"
    
    # Convert to dict and back
    config_dict = config.to_dict()
    assert config_dict["training"]["precision"] == "float64", "Expected precision to be preserved in dict"
    
    # Create new config from dict
    new_config = Config.from_dict(config_dict)
    assert new_config.training.precision == "float64", "Expected precision to be preserved after from_dict"


def test_all_precision_types_supported():
    """Test that all precision types are supported."""
    precision_types = ["float16", "bfloat16", "float32", "float64"]
    
    for precision in precision_types:
        config = TrainingConfig()
        config.precision = precision
        assert config.precision == precision, f"Expected precision to be '{precision}'"


def test_float16_forward_backward_pass():
    """Test forward and backward pass with Float16 precision."""
    import optax
    
    # Create model with float16
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
        param_dtype=jnp.float16,
        compute_dtype=jnp.float16
    )
    
    # Create optimizer
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    
    # Test forward pass
    batch_size = 4
    seq_len = 32
    inputs = jnp.array([[1, 2, 3, 4] * (seq_len // 4)] * batch_size, dtype=jnp.int32)
    
    # Define loss function
    def loss_fn(model, batch):
        logits = model(batch, training=True)
        # Simple loss for testing
        target_tokens = batch[:, 1:]
        logits = logits[:, :-1, :]
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, logits.shape[-1]),
            target_tokens.reshape(-1)
        )
        return jnp.mean(loss)
    
    # Compute loss and gradients
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, inputs)
    
    # Verify loss is computed
    assert jnp.isfinite(loss), "Loss should be finite"
    
    # Update parameters
    optimizer.update(model, grads)


def test_float64_forward_backward_pass():
    """Test forward and backward pass with Float64 precision."""
    import optax
    from jax import config as jax_config
    
    # Enable x64 for JAX
    jax_config.update("jax_enable_x64", True)
    
    # Create model with float64
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
        param_dtype=jnp.float64,
        compute_dtype=jnp.float64
    )
    
    # Create optimizer
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    
    # Test forward pass
    batch_size = 4
    seq_len = 32
    inputs = jnp.array([[1, 2, 3, 4] * (seq_len // 4)] * batch_size, dtype=jnp.int32)
    
    # Define loss function
    def loss_fn(model, batch):
        logits = model(batch, training=True)
        # Simple loss for testing
        target_tokens = batch[:, 1:]
        logits = logits[:, :-1, :]
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, logits.shape[-1]),
            target_tokens.reshape(-1)
        )
        return jnp.mean(loss)
    
    # Compute loss and gradients
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, inputs)
    
    # Verify loss is computed
    assert jnp.isfinite(loss), "Loss should be finite"
    
    # Update parameters
    optimizer.update(model, grads)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
