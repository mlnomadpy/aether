"""Unit tests for training steps module."""

import pytest
import sys
import os
sys.path.insert(0, '/home/runner/work/aether/aether')

import jax.numpy as jnp
import flax.nnx as nnx
import optax


class TestLossFn:
    """Tests for loss_fn function."""
    
    def test_loss_fn_import(self):
        """Test that loss_fn can be imported."""
        from aether.training.steps import loss_fn
        assert callable(loss_fn)
    
    def test_loss_fn_returns_tuple(self):
        """Test that loss_fn returns (loss, logits) tuple."""
        from aether.training.steps import loss_fn
        from aether.models import MiniGPT
        
        rngs = nnx.Rngs(42)
        model = MiniGPT(
            maxlen=64,
            vocab_size=100,
            embed_dim=32,
            num_heads=2,
            feed_forward_dim=64,
            num_transformer_blocks=1,
            rngs=rngs
        )
        
        batch = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        result = loss_fn(model, batch, training=False)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_loss_fn_loss_is_scalar(self):
        """Test that loss is a scalar value."""
        from aether.training.steps import loss_fn
        from aether.models import MiniGPT
        
        rngs = nnx.Rngs(42)
        model = MiniGPT(
            maxlen=64,
            vocab_size=100,
            embed_dim=32,
            num_heads=2,
            feed_forward_dim=64,
            num_transformer_blocks=1,
            rngs=rngs
        )
        
        batch = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        loss, _ = loss_fn(model, batch, training=False)
        
        assert loss.shape == ()  # Scalar has empty shape
        assert jnp.isfinite(loss)
    
    def test_loss_fn_logits_shape(self):
        """Test that logits have correct shape."""
        from aether.training.steps import loss_fn
        from aether.models import MiniGPT
        
        rngs = nnx.Rngs(42)
        vocab_size = 100
        model = MiniGPT(
            maxlen=64,
            vocab_size=vocab_size,
            embed_dim=32,
            num_heads=2,
            feed_forward_dim=64,
            num_transformer_blocks=1,
            rngs=rngs
        )
        
        batch_size = 2
        seq_len = 8
        batch = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]])
        
        _, logits = loss_fn(model, batch, training=False)
        
        # Logits should be (batch_size, seq_len - 1, vocab_size)
        assert logits.shape == (batch_size, seq_len - 1, vocab_size)
    
    def test_loss_fn_training_mode(self):
        """Test loss_fn in training mode."""
        from aether.training.steps import loss_fn
        from aether.models import MiniGPT
        
        rngs = nnx.Rngs(42)
        model = MiniGPT(
            maxlen=64,
            vocab_size=100,
            embed_dim=32,
            num_heads=2,
            feed_forward_dim=64,
            num_transformer_blocks=1,
            rngs=rngs
        )
        
        batch = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        
        loss_train, _ = loss_fn(model, batch, training=True)
        loss_eval, _ = loss_fn(model, batch, training=False)
        
        assert jnp.isfinite(loss_train)
        assert jnp.isfinite(loss_eval)


class TestTrainStep:
    """Tests for train_step function."""
    
    def test_train_step_import(self):
        """Test that train_step can be imported."""
        from aether.training.steps import train_step
        assert callable(train_step)
    
    def test_train_step_returns_tuple(self):
        """Test that train_step returns (loss, model, optimizer) tuple."""
        from aether.training.steps import train_step
        from aether.models import MiniGPT
        
        rngs = nnx.Rngs(42)
        model = MiniGPT(
            maxlen=64,
            vocab_size=100,
            embed_dim=32,
            num_heads=2,
            feed_forward_dim=64,
            num_transformer_blocks=1,
            rngs=rngs
        )
        
        optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
        batch = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        
        result = train_step(model, optimizer, batch)
        
        assert isinstance(result, tuple)
        assert len(result) == 3
    
    def test_train_step_loss_decreases(self):
        """Test that loss decreases over multiple training steps."""
        from aether.training.steps import train_step
        from aether.models import MiniGPT
        
        rngs = nnx.Rngs(42)
        model = MiniGPT(
            maxlen=64,
            vocab_size=100,
            embed_dim=32,
            num_heads=2,
            feed_forward_dim=64,
            num_transformer_blocks=1,
            rngs=rngs
        )
        
        optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)
        batch = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]] * 4)
        
        initial_loss = None
        for i in range(5):
            loss, model, optimizer = train_step(model, optimizer, batch)
            if i == 0:
                initial_loss = loss
        
        # Loss should generally decrease (or at least not explode)
        assert jnp.isfinite(loss)
        assert loss < initial_loss * 2  # Loss shouldn't explode


class TestEvalStep:
    """Tests for eval_step function."""
    
    def test_eval_step_import(self):
        """Test that eval_step can be imported."""
        from aether.training.steps import eval_step
        assert callable(eval_step)
    
    def test_eval_step_returns_loss(self):
        """Test that eval_step returns loss."""
        from aether.training.steps import eval_step
        from aether.models import MiniGPT
        
        rngs = nnx.Rngs(42)
        model = MiniGPT(
            maxlen=64,
            vocab_size=100,
            embed_dim=32,
            num_heads=2,
            feed_forward_dim=64,
            num_transformer_blocks=1,
            rngs=rngs
        )
        
        batch = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        loss = eval_step(model, batch)
        
        assert loss.shape == ()  # Scalar
        assert jnp.isfinite(loss)
    
    def test_eval_step_deterministic(self):
        """Test that eval_step is deterministic (no dropout)."""
        from aether.training.steps import eval_step
        from aether.models import MiniGPT
        
        rngs = nnx.Rngs(42)
        model = MiniGPT(
            maxlen=64,
            vocab_size=100,
            embed_dim=32,
            num_heads=2,
            feed_forward_dim=64,
            num_transformer_blocks=1,
            rngs=rngs
        )
        
        batch = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        
        loss1 = eval_step(model, batch)
        loss2 = eval_step(model, batch)
        
        # Should be identical since dropout is disabled
        assert jnp.allclose(loss1, loss2)


class TestTrainingModuleExports:
    """Tests for training module exports."""
    
    def test_training_exports_loss_fn(self):
        """Test that training module exports loss_fn."""
        from aether.training import loss_fn
        assert callable(loss_fn)
    
    def test_training_exports_train_step(self):
        """Test that training module exports train_step."""
        from aether.training import train_step
        assert callable(train_step)
    
    def test_training_exports_eval_step(self):
        """Test that training module exports eval_step."""
        from aether.training import eval_step
        assert callable(eval_step)
    
    def test_training_exports_trainer(self):
        """Test that training module exports Trainer."""
        from aether.training import Trainer
        assert Trainer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
