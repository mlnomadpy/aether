"""Unit tests for embedding layers."""

import pytest
import sys
import os
sys.path.insert(0, '/home/runner/work/aether/aether')

import jax.numpy as jnp
import flax.nnx as nnx


class TestRMSNorm:
    """Tests for RMSNorm layer."""
    
    def test_rmsnorm_initialization(self):
        """Test RMSNorm initialization."""
        from aether.models.embeddings import RMSNorm
        
        rngs = nnx.Rngs(42)
        norm = RMSNorm(dim=256, rngs=rngs)
        
        assert norm is not None
        assert norm.weight.shape == (256,)
    
    def test_rmsnorm_forward_pass(self):
        """Test RMSNorm forward pass."""
        from aether.models.embeddings import RMSNorm
        
        rngs = nnx.Rngs(42)
        norm = RMSNorm(dim=128, rngs=rngs)
        
        inputs = jnp.ones((2, 10, 128))
        outputs = norm(inputs)
        
        assert outputs.shape == inputs.shape
    
    def test_rmsnorm_preserves_shape(self):
        """Test that RMSNorm preserves input shape."""
        from aether.models.embeddings import RMSNorm
        
        rngs = nnx.Rngs(42)
        norm = RMSNorm(dim=64, rngs=rngs)
        
        test_shapes = [
            (1, 5, 64),
            (4, 16, 64),
            (8, 32, 64),
        ]
        
        for shape in test_shapes:
            inputs = jnp.ones(shape)
            outputs = norm(inputs)
            assert outputs.shape == shape
    
    def test_rmsnorm_epsilon(self):
        """Test RMSNorm with custom epsilon."""
        from aether.models.embeddings import RMSNorm
        
        rngs = nnx.Rngs(42)
        norm = RMSNorm(dim=128, eps=1e-8, rngs=rngs)
        
        assert norm.eps == 1e-8
    
    def test_rmsnorm_dtype(self):
        """Test RMSNorm with different dtypes."""
        from aether.models.embeddings import RMSNorm
        
        rngs = nnx.Rngs(42)
        norm = RMSNorm(dim=64, param_dtype=jnp.float32, rngs=rngs)
        
        assert norm.weight.dtype == jnp.float32


class TestTokenOnlyEmbedding:
    """Tests for TokenOnlyEmbedding layer."""
    
    def test_token_only_embedding_initialization(self):
        """Test TokenOnlyEmbedding initialization."""
        from aether.models.embeddings import TokenOnlyEmbedding
        
        rngs = nnx.Rngs(42)
        embedding = TokenOnlyEmbedding(
            vocab_size=1000,
            embed_dim=256,
            rngs=rngs
        )
        
        assert embedding is not None
        assert embedding.vocab_size == 1000
        assert embedding.embed_dim == 256
    
    def test_token_only_embedding_forward(self):
        """Test TokenOnlyEmbedding forward pass."""
        from aether.models.embeddings import TokenOnlyEmbedding
        
        rngs = nnx.Rngs(42)
        embedding = TokenOnlyEmbedding(
            vocab_size=500,
            embed_dim=128,
            rngs=rngs
        )
        
        inputs = jnp.array([[1, 2, 3, 4, 5]])
        outputs = embedding(inputs)
        
        assert outputs.shape == (1, 5, 128)
    
    def test_token_only_embedding_batch_size(self):
        """Test TokenOnlyEmbedding with different batch sizes."""
        from aether.models.embeddings import TokenOnlyEmbedding
        
        rngs = nnx.Rngs(42)
        embedding = TokenOnlyEmbedding(
            vocab_size=1000,
            embed_dim=64,
            rngs=rngs
        )
        
        for batch_size in [1, 2, 4, 8]:
            inputs = jnp.ones((batch_size, 10), dtype=jnp.int32)
            outputs = embedding(inputs)
            assert outputs.shape == (batch_size, 10, 64)
    
    def test_token_only_embedding_dtype(self):
        """Test TokenOnlyEmbedding with different param dtype."""
        from aether.models.embeddings import TokenOnlyEmbedding
        
        rngs = nnx.Rngs(42)
        embedding = TokenOnlyEmbedding(
            vocab_size=1000,
            embed_dim=64,
            param_dtype=jnp.float16,
            rngs=rngs
        )
        
        assert embedding.token_emb.embedding.dtype == jnp.float16


class TestTokenAndPositionEmbedding:
    """Tests for TokenAndPositionEmbedding layer."""
    
    def test_token_position_embedding_initialization(self):
        """Test TokenAndPositionEmbedding initialization."""
        from aether.models.embeddings import TokenAndPositionEmbedding
        
        rngs = nnx.Rngs(42)
        embedding = TokenAndPositionEmbedding(
            maxlen=128,
            vocab_size=1000,
            embed_dim=256,
            rngs=rngs
        )
        
        assert embedding is not None
        assert embedding.maxlen == 128
        assert embedding.vocab_size == 1000
        assert embedding.embed_dim == 256
    
    def test_token_position_embedding_forward(self):
        """Test TokenAndPositionEmbedding forward pass."""
        from aether.models.embeddings import TokenAndPositionEmbedding
        
        rngs = nnx.Rngs(42)
        embedding = TokenAndPositionEmbedding(
            maxlen=64,
            vocab_size=500,
            embed_dim=128,
            rngs=rngs
        )
        
        inputs = jnp.array([[1, 2, 3, 4, 5]])
        outputs = embedding(inputs)
        
        assert outputs.shape == (1, 5, 128)
    
    def test_token_position_embedding_combines_embeddings(self):
        """Test that token and position embeddings are combined."""
        from aether.models.embeddings import TokenAndPositionEmbedding
        
        rngs = nnx.Rngs(42)
        embedding = TokenAndPositionEmbedding(
            maxlen=128,
            vocab_size=1000,
            embed_dim=64,
            rngs=rngs
        )
        
        # Same tokens at different positions should give different outputs
        inputs1 = jnp.array([[1, 1, 1]])
        outputs1 = embedding(inputs1)
        
        # All outputs should be different due to position embeddings
        assert not jnp.allclose(outputs1[0, 0], outputs1[0, 1])
        assert not jnp.allclose(outputs1[0, 1], outputs1[0, 2])
    
    def test_token_position_embedding_batch_size(self):
        """Test TokenAndPositionEmbedding with different batch sizes."""
        from aether.models.embeddings import TokenAndPositionEmbedding
        
        rngs = nnx.Rngs(42)
        embedding = TokenAndPositionEmbedding(
            maxlen=64,
            vocab_size=1000,
            embed_dim=32,
            rngs=rngs
        )
        
        for batch_size in [1, 2, 4, 8]:
            inputs = jnp.ones((batch_size, 16), dtype=jnp.int32)
            outputs = embedding(inputs)
            assert outputs.shape == (batch_size, 16, 32)
    
    def test_token_position_embedding_dtype(self):
        """Test TokenAndPositionEmbedding with different param dtype."""
        from aether.models.embeddings import TokenAndPositionEmbedding
        
        rngs = nnx.Rngs(42)
        embedding = TokenAndPositionEmbedding(
            maxlen=64,
            vocab_size=1000,
            embed_dim=32,
            param_dtype=jnp.bfloat16,
            rngs=rngs
        )
        
        assert embedding.token_emb.embedding.dtype == jnp.bfloat16
        assert embedding.pos_emb.embedding.dtype == jnp.bfloat16


class TestEmbeddingsModuleExports:
    """Tests for embeddings module exports."""
    
    def test_models_exports_rmsnorm(self):
        """Test that models module exports RMSNorm."""
        from aether.models import RMSNorm
        assert RMSNorm is not None
    
    def test_models_exports_token_only_embedding(self):
        """Test that models module exports TokenOnlyEmbedding."""
        from aether.models import TokenOnlyEmbedding
        assert TokenOnlyEmbedding is not None
    
    def test_models_exports_token_and_position_embedding(self):
        """Test that models module exports TokenAndPositionEmbedding."""
        from aether.models import TokenAndPositionEmbedding
        assert TokenAndPositionEmbedding is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
