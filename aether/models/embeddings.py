"""Token and position embedding layer."""

import jax.numpy as jnp
import flax.nnx as nnx


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