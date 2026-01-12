"""Scalability tests for large-scale integration readiness.

These tests verify that the framework components can handle various scales
of input and configuration, which is essential for production deployment.
"""

import pytest
import sys
import os
sys.path.insert(0, '/home/runner/work/aether/aether')

import jax.numpy as jnp
import flax.nnx as nnx
import optax
import time


class TestModelScalability:
    """Tests for model scalability with different configurations."""
    
    def test_model_with_small_batch(self):
        """Test model with small batch size."""
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
        
        batch = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]])  # Batch size 1
        outputs = model(batch, training=False)
        
        assert outputs.shape == (1, 8, 100)
    
    def test_model_with_larger_batch(self):
        """Test model with larger batch size."""
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
        
        batch_size = 16
        seq_len = 32
        batch = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        outputs = model(batch, training=False)
        
        assert outputs.shape == (batch_size, seq_len, 100)
    
    def test_model_with_variable_sequence_lengths(self):
        """Test model with different sequence lengths."""
        from aether.models import MiniGPT
        
        rngs = nnx.Rngs(42)
        maxlen = 128
        model = MiniGPT(
            maxlen=maxlen,
            vocab_size=100,
            embed_dim=32,
            num_heads=2,
            feed_forward_dim=64,
            num_transformer_blocks=2,
            rngs=rngs
        )
        
        for seq_len in [8, 16, 32, 64, 128]:
            batch = jnp.ones((4, seq_len), dtype=jnp.int32)
            outputs = model(batch, training=False)
            assert outputs.shape == (4, seq_len, 100)
    
    def test_model_with_different_embed_dims(self):
        """Test model creation with different embedding dimensions."""
        from aether.models import MiniGPT
        
        for embed_dim in [32, 64, 128, 256]:
            rngs = nnx.Rngs(42)
            model = MiniGPT(
                maxlen=64,
                vocab_size=100,
                embed_dim=embed_dim,
                num_heads=4,
                feed_forward_dim=embed_dim * 2,
                num_transformer_blocks=2,
                rngs=rngs
            )
            
            batch = jnp.ones((2, 16), dtype=jnp.int32)
            outputs = model(batch, training=False)
            
            assert outputs.shape == (2, 16, 100)
    
    def test_model_with_different_num_blocks(self):
        """Test model with different numbers of transformer blocks."""
        from aether.models import MiniGPT
        
        for num_blocks in [1, 2, 4, 6]:
            rngs = nnx.Rngs(42)
            model = MiniGPT(
                maxlen=64,
                vocab_size=100,
                embed_dim=32,
                num_heads=2,
                feed_forward_dim=64,
                num_transformer_blocks=num_blocks,
                rngs=rngs
            )
            
            assert len(model.transformer_blocks) == num_blocks
            
            batch = jnp.ones((2, 16), dtype=jnp.int32)
            outputs = model(batch, training=False)
            assert outputs.shape == (2, 16, 100)
    
    def test_model_with_attention_block_reuse(self):
        """Test model with attention block reuse for iterative refinement."""
        from aether.models import MiniGPT
        
        for reuse_count in [1, 2, 3, 4]:
            rngs = nnx.Rngs(42)
            model = MiniGPT(
                maxlen=64,
                vocab_size=100,
                embed_dim=32,
                num_heads=2,
                feed_forward_dim=64,
                num_transformer_blocks=2,
                attention_block_reuse=reuse_count,
                rngs=rngs
            )
            
            batch = jnp.ones((2, 16), dtype=jnp.int32)
            outputs = model(batch, training=False)
            
            assert outputs.shape == (2, 16, 100)
            assert model.attention_block_reuse == reuse_count


class TestTrainingScalability:
    """Tests for training scalability."""
    
    def test_multiple_training_steps(self):
        """Test multiple consecutive training steps."""
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
        batch = jnp.ones((4, 16), dtype=jnp.int32)
        
        losses = []
        for _ in range(10):
            loss, model, optimizer = train_step(model, optimizer, batch)
            losses.append(float(loss))
        
        # All losses should be finite
        assert all(jnp.isfinite(l) for l in losses)
    
    def test_training_with_different_batch_sizes(self):
        """Test training with different batch sizes."""
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
        
        for batch_size in [1, 2, 4, 8]:
            batch = jnp.ones((batch_size, 16), dtype=jnp.int32)
            loss, model, optimizer = train_step(model, optimizer, batch)
            assert jnp.isfinite(loss)
    
    def test_training_with_different_optimizers(self):
        """Test training with different optimizers."""
        from aether.models import MiniGPT
        from aether.training.steps import train_step
        
        optimizers = [
            optax.adam(1e-3),
            optax.adamw(1e-3),
            optax.sgd(1e-2),
            optax.novograd(1e-3),
        ]
        
        for opt_fn in optimizers:
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
            
            optimizer = nnx.Optimizer(model, opt_fn, wrt=nnx.Param)
            batch = jnp.ones((4, 16), dtype=jnp.int32)
            
            loss, model, optimizer = train_step(model, optimizer, batch)
            assert jnp.isfinite(loss)


class TestPrecisionScalability:
    """Tests for precision scalability."""
    
    def test_float32_precision(self):
        """Test model with float32 precision."""
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
            param_dtype=jnp.float32,
            compute_dtype=jnp.float32
        )
        
        batch = jnp.ones((4, 16), dtype=jnp.int32)
        outputs = model(batch, training=False)
        
        assert outputs.dtype == jnp.float32
    
    def test_float16_precision(self):
        """Test model with float16 precision."""
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
            param_dtype=jnp.float16,
            compute_dtype=jnp.float16
        )
        
        batch = jnp.ones((4, 16), dtype=jnp.int32)
        outputs = model(batch, training=False)
        
        assert outputs.dtype == jnp.float16
    
    def test_bfloat16_precision(self):
        """Test model with bfloat16 precision."""
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
            param_dtype=jnp.bfloat16,
            compute_dtype=jnp.bfloat16
        )
        
        batch = jnp.ones((4, 16), dtype=jnp.int32)
        outputs = model(batch, training=False)
        
        assert outputs.dtype == jnp.bfloat16


class TestConfigScalability:
    """Tests for configuration scalability."""
    
    def test_config_with_all_model_params(self):
        """Test config with all model parameters."""
        from aether.config import Config
        
        config = Config()
        config.model.name = "test-model"
        config.model.architecture = "linear"
        config.model.maxlen = 1024
        config.model.vocab_size = 50257
        config.model.embed_dim = 768
        config.model.num_heads = 12
        config.model.feed_forward_dim = 3072
        config.model.num_transformer_blocks = 12
        config.model.dropout_rate = 0.1
        config.model.use_layer_norm = True
        config.model.attention_block_reuse = 1
        
        # Verify all settings
        assert config.model.maxlen == 1024
        assert config.model.vocab_size == 50257
        assert config.model.embed_dim == 768
    
    def test_config_with_all_training_params(self):
        """Test config with all training parameters."""
        from aether.config import Config
        
        config = Config()
        config.training.batch_size = 32
        config.training.learning_rate = 1e-3
        config.training.max_tokens_to_process = 1_000_000_000
        config.training.eval_interval = 2000
        config.training.eval_steps = 1000
        config.training.val_set_size = 20000
        config.training.checkpoint_interval = 10000
        config.training.optimizer = "adamw"
        config.training.lr_scheduler = "cosine"
        config.training.lr_scheduler_alpha = 0.1
        config.training.lr_scheduler_warmup_steps = 5000
        config.training.momentum = 0.9
        config.training.weight_decay = 0.01
        config.training.precision = "bfloat16"
        
        # Verify all settings
        assert config.training.batch_size == 32
        assert config.training.optimizer == "adamw"
        assert config.training.precision == "bfloat16"
    
    def test_config_serialization_roundtrip(self):
        """Test config serialization and deserialization."""
        from aether.config import Config
        import tempfile
        import os
        
        config = Config()
        config.model.embed_dim = 512
        config.training.batch_size = 64
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "test_config.json")
            
            # Save and reload
            config.save(config_path)
            loaded_config = Config.from_file(config_path)
            
            # Verify values preserved
            assert loaded_config.model.embed_dim == 512
            assert loaded_config.training.batch_size == 64


class TestRegistryScalability:
    """Tests for registry scalability."""
    
    def test_registry_with_multiple_models(self):
        """Test registry with multiple model registrations."""
        from aether.registry import ModelRegistry
        from aether.models import MiniGPT
        
        registry = ModelRegistry()
        
        # Register multiple model variants
        for i in range(5):
            registry.register_model(
                f"model-{i}",
                MiniGPT,
                default_config={
                    "maxlen": 64 * (i + 1),
                    "vocab_size": 100,
                    "embed_dim": 32,
                    "num_heads": 2,
                    "feed_forward_dim": 64,
                    "num_transformer_blocks": i + 1
                }
            )
        
        # Verify all models registered
        assert len(registry.list_models()) == 5
        
        # Create each model
        for i in range(5):
            rngs = nnx.Rngs(42)
            model = registry.create_model(f"model-{i}", {}, rngs)
            assert model.maxlen == 64 * (i + 1)


class TestPerformanceBaseline:
    """Tests to establish performance baselines."""
    
    def test_forward_pass_timing(self):
        """Test forward pass timing for baseline."""
        from aether.models import MiniGPT
        
        rngs = nnx.Rngs(42)
        model = MiniGPT(
            maxlen=64,
            vocab_size=100,
            embed_dim=64,
            num_heads=4,
            feed_forward_dim=128,
            num_transformer_blocks=2,
            rngs=rngs
        )
        
        batch = jnp.ones((8, 32), dtype=jnp.int32)
        
        # Warmup
        _ = model(batch, training=False)
        
        # Timing
        start = time.time()
        for _ in range(10):
            outputs = model(batch, training=False)
        elapsed = time.time() - start
        
        # Should complete reasonably quickly (< 30s for 10 iterations)
        assert elapsed < 30.0
    
    def test_training_step_timing(self):
        """Test training step timing for baseline."""
        from aether.models import MiniGPT
        from aether.training.steps import train_step
        
        rngs = nnx.Rngs(42)
        model = MiniGPT(
            maxlen=64,
            vocab_size=100,
            embed_dim=64,
            num_heads=4,
            feed_forward_dim=128,
            num_transformer_blocks=2,
            rngs=rngs
        )
        
        optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
        batch = jnp.ones((4, 16), dtype=jnp.int32)
        
        # Warmup
        loss, model, optimizer = train_step(model, optimizer, batch)
        
        # Timing
        start = time.time()
        for _ in range(5):
            loss, model, optimizer = train_step(model, optimizer, batch)
        elapsed = time.time() - start
        
        # Should complete reasonably quickly (< 60s for 5 iterations)
        assert elapsed < 60.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
