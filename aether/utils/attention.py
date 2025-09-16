"""Attention utilities."""

import jax.numpy as jnp


def causal_attention_mask(seq_len: int) -> jnp.ndarray:
    """Creates a causal attention mask.
    
    Args:
        seq_len: Sequence length
        
    Returns:
        Lower triangular mask of shape (seq_len, seq_len)
    """
    return jnp.tril(jnp.ones((seq_len, seq_len)))