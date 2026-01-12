"""Integration tests for component interactions.

These tests verify that different components of the framework work together
correctly, which is essential for large-scale integration readiness.
"""

import pytest
import sys
import os
sys.path.insert(0, '/home/runner/work/aether/aether')

import jax.numpy as jnp
import flax.nnx as nnx
import optax


class TestModelOptimizerIntegration:
    """Tests for model and optimizer integration."""
    
    def test_minigpt_with_adam_optimizer(self):
        """Test MiniGPT with Adam optimizer."""
        from aether.models import MiniGPT
        
        rngs = nnx.Rngs(42)
        model = MiniGPT(
            maxlen=64,
            vocab_size=100,
            embed_dim=32,
            num_heads=2,
            feed_forward_dim=64,
            num_transformer_blocks=2,
            rngs=rngs
        )
        
        optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
        
        # Run a forward pass
        batch = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        outputs = model(batch, training=True)
        
        assert outputs.shape == (1, 8, 100)
    
    def test_minigpt_with_adamw_optimizer(self):
        """Test MiniGPT with AdamW optimizer."""
        from aether.models import MiniGPT
        
        rngs = nnx.Rngs(42)
        model = MiniGPT(
            maxlen=64,
            vocab_size=100,
            embed_dim=32,
            num_heads=2,
            feed_forward_dim=64,
            num_transformer_blocks=2,
            rngs=rngs
        )
        
        optimizer = nnx.Optimizer(
            model, 
            optax.adamw(learning_rate=1e-3, weight_decay=0.01), 
            wrt=nnx.Param
        )
        
        batch = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        outputs = model(batch, training=True)
        
        assert outputs.shape == (1, 8, 100)
    
    def test_minigpt_with_sgd_optimizer(self):
        """Test MiniGPT with SGD optimizer."""
        from aether.models import MiniGPT
        
        rngs = nnx.Rngs(42)
        model = MiniGPT(
            maxlen=64,
            vocab_size=100,
            embed_dim=32,
            num_heads=2,
            feed_forward_dim=64,
            num_transformer_blocks=2,
            rngs=rngs
        )
        
        optimizer = nnx.Optimizer(
            model, 
            optax.sgd(learning_rate=1e-2, momentum=0.9), 
            wrt=nnx.Param
        )
        
        batch = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        outputs = model(batch, training=True)
        
        assert outputs.shape == (1, 8, 100)


class TestModelTrainingStepsIntegration:
    """Tests for model and training steps integration."""
    
    def test_model_with_loss_fn(self):
        """Test model with loss function."""
        from aether.models import MiniGPT
        from aether.training.steps import loss_fn
        
        rngs = nnx.Rngs(42)
        model = MiniGPT(
            maxlen=64,
            vocab_size=100,
            embed_dim=32,
            num_heads=2,
            feed_forward_dim=64,
            num_transformer_blocks=2,
            rngs=rngs
        )
        
        batch = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        loss, logits = loss_fn(model, batch, training=True)
        
        assert jnp.isfinite(loss)
        assert logits.shape == (1, 7, 100)
    
    def test_model_with_train_step(self):
        """Test model with full training step."""
        from aether.models import MiniGPT
        from aether.training.steps import train_step
        
        rngs = nnx.Rngs(42)
        model = MiniGPT(
            maxlen=64,
            vocab_size=100,
            embed_dim=32,
            num_heads=2,
            feed_forward_dim=64,
            num_transformer_blocks=2,
            rngs=rngs
        )
        
        optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
        batch = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        
        loss, model, optimizer = train_step(model, optimizer, batch)
        
        assert jnp.isfinite(loss)
    
    def test_model_with_eval_step(self):
        """Test model with evaluation step."""
        from aether.models import MiniGPT
        from aether.training.steps import eval_step
        
        rngs = nnx.Rngs(42)
        model = MiniGPT(
            maxlen=64,
            vocab_size=100,
            embed_dim=32,
            num_heads=2,
            feed_forward_dim=64,
            num_transformer_blocks=2,
            rngs=rngs
        )
        
        batch = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        loss = eval_step(model, batch)
        
        assert jnp.isfinite(loss)


class TestConfigModelIntegration:
    """Tests for config and model integration."""
    
    def test_config_to_model_creation(self):
        """Test creating model from config."""
        from aether.config import Config
        from aether.models import MiniGPT
        
        config = Config()
        config.model.maxlen = 64
        config.model.vocab_size = 100
        config.model.embed_dim = 32
        config.model.num_heads = 2
        config.model.feed_forward_dim = 64
        config.model.num_transformer_blocks = 2
        
        rngs = nnx.Rngs(42)
        model_config = config.get_model_config_dict()
        model = MiniGPT.from_config(model_config, rngs)
        
        assert model is not None
        assert model.maxlen == 64
        assert model.vocab_size == 100
    
    def test_config_precision_integration(self):
        """Test config precision setting with model creation."""
        from aether.config import Config
        from aether.models import MiniGPT
        
        config = Config()
        config.training.precision = "float32"
        
        # Verify precision setting
        assert config.training.precision == "float32"
        
        # Create model with matching dtype
        rngs = nnx.Rngs(42)
        model = MiniGPT(
            maxlen=64,
            vocab_size=100,
            embed_dim=32,
            num_heads=2,
            feed_forward_dim=64,
            num_transformer_blocks=2,
            rngs=rngs,
            param_dtype=jnp.float32,
            compute_dtype=jnp.float32
        )
        
        assert model.param_dtype == jnp.float32


class TestRegistryModelIntegration:
    """Tests for registry and model integration."""
    
    def test_registry_create_minigpt(self):
        """Test creating MiniGPT through registry."""
        from aether.registry import ModelRegistry
        from aether.models import MiniGPT
        
        registry = ModelRegistry()
        registry.register_model(
            "test-model",
            MiniGPT,
            default_config={
                "maxlen": 64,
                "vocab_size": 100,
                "embed_dim": 32,
                "num_heads": 2,
                "feed_forward_dim": 64,
                "num_transformer_blocks": 2
            }
        )
        
        rngs = nnx.Rngs(42)
        model = registry.create_model("test-model", {}, rngs)
        
        assert isinstance(model, MiniGPT)
    
    def test_registry_with_custom_config(self):
        """Test registry with custom config override."""
        from aether.registry import ModelRegistry
        from aether.models import MiniGPT
        
        registry = ModelRegistry()
        registry.register_model(
            "base-model",
            MiniGPT,
            default_config={
                "maxlen": 64,
                "vocab_size": 100,
                "embed_dim": 32,
                "num_heads": 2,
                "feed_forward_dim": 64,
                "num_transformer_blocks": 2
            }
        )
        
        rngs = nnx.Rngs(42)
        model = registry.create_model("base-model", {"embed_dim": 64}, rngs)
        
        assert model.embed_dim == 64


class TestTransformerBlockModelIntegration:
    """Tests for transformer block and model integration."""
    
    def test_transformer_block_in_model(self):
        """Test that transformer blocks work correctly in model."""
        from aether.models import MiniGPT, TransformerBlock
        
        rngs = nnx.Rngs(42)
        model = MiniGPT(
            maxlen=64,
            vocab_size=100,
            embed_dim=32,
            num_heads=2,
            feed_forward_dim=64,
            num_transformer_blocks=3,
            rngs=rngs
        )
        
        # Verify transformer blocks exist
        assert len(model.transformer_blocks) == 3
        
        # Verify each block is a TransformerBlock
        for block in model.transformer_blocks:
            assert isinstance(block, TransformerBlock)
    
    def test_transformer_block_with_layer_norm(self):
        """Test transformer block with layer norm in model."""
        from aether.models import MiniGPT
        
        rngs = nnx.Rngs(42)
        model = MiniGPT(
            maxlen=64,
            vocab_size=100,
            embed_dim=32,
            num_heads=2,
            feed_forward_dim=64,
            num_transformer_blocks=2,
            rngs=rngs,
            use_layer_norm=True
        )
        
        batch = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        outputs = model(batch, training=False)
        
        assert outputs.shape == (1, 8, 100)
    
    def test_transformer_block_without_layer_norm(self):
        """Test transformer block without layer norm in model."""
        from aether.models import MiniGPT
        
        rngs = nnx.Rngs(42)
        model = MiniGPT(
            maxlen=64,
            vocab_size=100,
            embed_dim=32,
            num_heads=2,
            feed_forward_dim=64,
            num_transformer_blocks=2,
            rngs=rngs,
            use_layer_norm=False
        )
        
        batch = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        outputs = model(batch, training=False)
        
        assert outputs.shape == (1, 8, 100)


class TestEmbeddingModelIntegration:
    """Tests for embedding and model integration."""
    
    def test_embedding_layer_in_model(self):
        """Test embedding layer integration in model."""
        from aether.models import MiniGPT
        from aether.models.embeddings import TokenAndPositionEmbedding
        
        rngs = nnx.Rngs(42)
        model = MiniGPT(
            maxlen=64,
            vocab_size=100,
            embed_dim=32,
            num_heads=2,
            feed_forward_dim=64,
            num_transformer_blocks=2,
            rngs=rngs
        )
        
        assert isinstance(model.embedding_layer, TokenAndPositionEmbedding)
    
    def test_embedding_output_feeds_transformer(self):
        """Test that embedding output correctly feeds into transformer."""
        from aether.models import MiniGPT
        
        rngs = nnx.Rngs(42)
        model = MiniGPT(
            maxlen=64,
            vocab_size=100,
            embed_dim=32,
            num_heads=2,
            feed_forward_dim=64,
            num_transformer_blocks=2,
            rngs=rngs
        )
        
        # Test full forward pass
        batch = jnp.array([[1, 2, 3, 4, 5]])
        
        # Get embedding output
        embed_output = model.embedding_layer(batch)
        assert embed_output.shape == (1, 5, 32)
        
        # Get full model output
        model_output = model(batch, training=False)
        assert model_output.shape == (1, 5, 100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
