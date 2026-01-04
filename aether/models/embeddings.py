"""Token and position embedding layers."""

import jax
import jax.numpy as jnp
import flax.nnx as nnx


class RMSNorm(nnx.Module):
    """Root Mean Square Layer Normalization.
    
    This is a simpler alternative to LayerNorm that normalizes by the
    root mean square of the activations without centering.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6, *, rngs: nnx.Rngs = None, param_dtype: jnp.dtype = jnp.float32):
        """Initialize RMSNorm.
        
        Args:
            dim: The dimension of the input features
            eps: Small constant for numerical stability
            rngs: Random number generators (unused but kept for API consistency)
            param_dtype: Data type for parameters
        """
        self.eps = eps
        self.weight = nnx.Param(jnp.ones(dim, dtype=param_dtype))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply RMS normalization.
        
        Args:
            x: Input tensor of shape (..., dim)
            
        Returns:
            Normalized tensor of same shape
        """
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        return x * jax.lax.rsqrt(var + self.eps) * self.weight


class TokenOnlyEmbedding(nnx.Module):
    """Token embeddings without positional embeddings.
    
    This is used for architectures that use rotary position embeddings (RoPE)
    which are applied directly in the attention layer.
    """
    
    def __init__(self, vocab_size: int, embed_dim: int, *, rngs: nnx.Rngs, param_dtype: jnp.dtype = jnp.float32):
        """Initialize token embeddings.
        
        Args:
            vocab_size: Vocabulary size
            embed_dim: Embedding dimension
            rngs: Random number generators
            param_dtype: Data type for parameters
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = nnx.Embed(num_embeddings=vocab_size, features=embed_dim, param_dtype=param_dtype, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply token embeddings.
        
        Args:
            x: Input token IDs of shape (batch_size, sequence_length)
            
        Returns:
            Embedded tensor of shape (batch_size, sequence_length, embed_dim)
        """
        return self.token_emb(x)


class TokenAndPositionEmbedding(nnx.Module):
    """Combines token and positional embeddings."""
    
    def __init__(self, maxlen: int, vocab_size: int, embed_dim: int, *, rngs: nnx.Rngs, param_dtype: jnp.dtype = jnp.float32):
        """Initialize embeddings.
        
        Args:
            maxlen: Maximum sequence length
            vocab_size: Vocabulary size
            embed_dim: Embedding dimension
            rngs: Random number generators
            param_dtype: Data type for parameters
        """
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        self.token_emb = nnx.Embed(num_embeddings=vocab_size, features=embed_dim, param_dtype=param_dtype, rngs=rngs)
        self.pos_emb = nnx.Embed(num_embeddings=maxlen, features=embed_dim, param_dtype=param_dtype, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply token and position embeddings.
        
        Args:
            x: Input token IDs of shape (batch_size, sequence_length)
            
        Returns:
            Embedded tensor of shape (batch_size, sequence_length, embed_dim)
        """
        positions = jnp.arange(0, x.shape[1])[None, :]
        position_embedding = self.pos_emb(positions)
        token_embedding = self.token_emb(x)
        return token_embedding + position_embedding