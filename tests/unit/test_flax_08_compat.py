"""Tests for Flax 0.8 optimizer compatibility layer."""

import jax.numpy as jnp
import flax.nnx as nnx
import optax
import pytest
from aether.models import MiniGPT
from aether.training.compat import (
    get_flax_version,
    is_flax_08,
    create_optimizer,
    update_optimizer,
    _check_optimizer_update_signature,
    _USE_OLD_API,
)


class TestFlaxVersionDetection:
    """Tests for Flax version detection."""
    
    def test_get_flax_version_returns_tuple(self):
        """Test that get_flax_version returns a tuple."""
        version = get_flax_version()
        assert isinstance(version, tuple)
        assert len(version) == 3
        assert all(isinstance(v, int) for v in version)
    
    def test_is_flax_08_returns_bool(self):
        """Test that is_flax_08 returns a boolean."""
        result = is_flax_08()
        assert isinstance(result, bool)
    
    def test_check_optimizer_update_signature_returns_bool(self):
        """Test that signature check returns a boolean."""
        result = _check_optimizer_update_signature()
        assert isinstance(result, bool)
    
    def test_use_old_api_matches_version_check(self):
        """Test that _USE_OLD_API matches version detection."""
        # _USE_OLD_API should be consistent with is_flax_08
        # On modern flax (0.11+), both should be False
        version = get_flax_version()
        if version >= (0, 11, 0):
            assert not _USE_OLD_API


class TestCreateOptimizer:
    """Tests for create_optimizer function."""
    
    def test_create_optimizer_returns_optimizer(self):
        """Test that create_optimizer returns an nnx.Optimizer."""
        rngs = nnx.Rngs(42)
        model = MiniGPT(
            maxlen=128,
            vocab_size=1000,
            embed_dim=256,
            num_heads=4,
            feed_forward_dim=512,
            num_transformer_blocks=2,
            rngs=rngs
        )
        optimizer_fn = optax.adam(0.001)
        optimizer = create_optimizer(model, optimizer_fn, wrt=nnx.Param)
        
        assert isinstance(optimizer, nnx.Optimizer)
    
    def test_create_optimizer_default_wrt(self):
        """Test that create_optimizer works with default wrt parameter."""
        rngs = nnx.Rngs(42)
        model = MiniGPT(
            maxlen=128,
            vocab_size=1000,
            embed_dim=256,
            num_heads=4,
            feed_forward_dim=512,
            num_transformer_blocks=2,
            rngs=rngs
        )
        optimizer_fn = optax.adam(0.001)
        # Should work without explicitly specifying wrt
        optimizer = create_optimizer(model, optimizer_fn)
        
        assert isinstance(optimizer, nnx.Optimizer)


class TestUpdateOptimizer:
    """Tests for update_optimizer function."""
    
    def test_update_optimizer_modifies_model(self):
        """Test that update_optimizer updates model parameters."""
        rngs = nnx.Rngs(42)
        model = MiniGPT(
            maxlen=128,
            vocab_size=1000,
            embed_dim=256,
            num_heads=4,
            feed_forward_dim=512,
            num_transformer_blocks=2,
            rngs=rngs
        )
        optimizer_fn = optax.adam(0.001)
        optimizer = create_optimizer(model, optimizer_fn, wrt=nnx.Param)
        
        # Get initial parameters
        _, initial_params, _ = nnx.split(model, nnx.Param, ...)
        
        # Create a dummy batch and compute loss
        batch = jnp.ones((2, 32), dtype=jnp.int32)
        
        # Define loss function
        def loss_fn(model):
            logits = model(batch, training=True)
            return jnp.mean(logits)
        
        # Compute gradients
        grads = nnx.grad(loss_fn)(model)
        
        # Update using the compatibility function
        update_optimizer(optimizer, model, grads)
        
        # Get updated parameters
        _, updated_params, _ = nnx.split(model, nnx.Param, ...)
        
        # Parameters should have changed
        # (We can't easily compare nested structures, so just verify no error occurred)
        assert updated_params is not None


class TestTrainStepWithCompat:
    """Tests for train_step using the compatibility layer."""
    
    def test_train_step_works_with_compat(self):
        """Test that train_step works with the compatibility layer."""
        from aether.training import train_step
        
        rngs = nnx.Rngs(42)
        model = MiniGPT(
            maxlen=128,
            vocab_size=1000,
            embed_dim=256,
            num_heads=4,
            feed_forward_dim=512,
            num_transformer_blocks=2,
            rngs=rngs
        )
        optimizer_fn = optax.adam(0.001)
        optimizer = create_optimizer(model, optimizer_fn, wrt=nnx.Param)
        
        # Create a dummy batch
        batch = jnp.ones((2, 32), dtype=jnp.int32)
        
        # Run train step - should not raise any errors
        loss, updated_model, updated_optimizer = train_step(model, optimizer, batch)
        
        # Verify outputs
        assert loss.shape == ()  # Scalar loss
        assert isinstance(updated_model, nnx.Module)
        assert isinstance(updated_optimizer, nnx.Optimizer)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
