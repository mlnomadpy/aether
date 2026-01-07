"""Transformer block implementations with linear architecture."""

import jax.numpy as jnp
import flax.nnx as nnx
from jax.sharding import NamedSharding, PartitionSpec as P
from typing import Optional, Union

from ..utils.attention import causal_attention_mask


class TransformerBlock(nnx.Module):
    """A transformer block with linear feed-forward architecture.
    
    For YAT architecture, use YatTransformerBlock from yatgpt module instead.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        *,
        rngs: nnx.Rngs,
        rate: float = 0.1,
        mesh: Optional[object] = None,
        use_layer_norm: bool = True,
        param_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        **kwargs
    ):
        """Initialize transformer block.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            rngs: Random number generators
            rate: Dropout rate
            mesh: JAX mesh for sharding
            use_layer_norm: Whether to use layer normalization
            param_dtype: Data type for parameters
            compute_dtype: Data type for computations
            **kwargs: Additional arguments (ignored for compatibility)
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.use_layer_norm = use_layer_norm
        self.param_dtype = param_dtype
        self.compute_dtype = compute_dtype
        
        # Set up partitioning if mesh is provided
        if mesh is not None:
            kernel_init = nnx.with_partitioning(
                nnx.initializers.xavier_uniform(), 
                NamedSharding(mesh, P(None, 'model'))
            )
            bias_init = nnx.with_partitioning(
                nnx.initializers.zeros_init(), 
                NamedSharding(mesh, P('model'))
            )
            layer_norm_scale_init = nnx.with_partitioning(
                nnx.initializers.ones_init(), 
                NamedSharding(mesh, P('model'))
            )
        else:
            kernel_init = nnx.initializers.xavier_uniform()
            bias_init = nnx.initializers.zeros_init()
            layer_norm_scale_init = nnx.initializers.ones_init()
        
        # Multi-head attention
        self.mha = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            kernel_init=kernel_init,
            bias_init=bias_init,
            param_dtype=param_dtype,
            rngs=rngs
        )
        self.dropout1 = nnx.Dropout(rate=rate, rngs=rngs)
        
        # Linear feed-forward network
        self._create_linear_ffn(embed_dim, ff_dim, kernel_init, bias_init, layer_norm_scale_init, rngs, param_dtype, use_layer_norm)
            
        self.dropout2 = nnx.Dropout(rate=rate, rngs=rngs)
    
    def _create_linear_ffn(self, embed_dim, ff_dim, kernel_init, bias_init, layer_norm_scale_init, rngs, param_dtype, use_layer_norm):
        """Create linear feed-forward network."""
        # Conditionally create layer normalization layers
        if use_layer_norm:
            self.layer_norm1 = nnx.LayerNorm(
                epsilon=1e-6,
                num_features=embed_dim,
                scale_init=layer_norm_scale_init,
                bias_init=bias_init,
                param_dtype=param_dtype,
                rngs=rngs
            )
            self.layer_norm2 = nnx.LayerNorm(
                epsilon=1e-6,
                num_features=embed_dim,
                scale_init=layer_norm_scale_init,
                bias_init=bias_init,
                param_dtype=param_dtype,
                rngs=rngs
            )
        else:
            self.layer_norm1 = None
            self.layer_norm2 = None
        
        self.linear1 = nnx.Linear(
            in_features=embed_dim,
            out_features=ff_dim,
            kernel_init=kernel_init,
            bias_init=bias_init,
            param_dtype=param_dtype,
            rngs=rngs
        )
        self.linear2 = nnx.Linear(
            in_features=ff_dim,
            out_features=embed_dim,
            kernel_init=kernel_init,
            bias_init=bias_init,
            param_dtype=param_dtype,
            rngs=rngs
        )

    def __call__(self, inputs: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Forward pass through the transformer block.
        
        Args:
            inputs: Input tensor of shape (batch_size, sequence_length, embed_dim)
            training: Whether in training mode
            
        Returns:
            Output tensor of same shape as inputs
        """
        seq_len = inputs.shape[1]
        mask = causal_attention_mask(seq_len)
        
        # Multi-head attention
        attention_output = self.mha(inputs_q=inputs, mask=mask, decode=False)
        attention_output = self.dropout1(attention_output, deterministic=not training)
        
        # Linear architecture with optional layer normalization
        if self.use_layer_norm:
            out1 = self.layer_norm1(inputs + attention_output)
        else:
            out1 = inputs + attention_output
        
        ffn_output = self.linear1(out1)
        ffn_output = nnx.relu(ffn_output)
        ffn_output = self.linear2(ffn_output)
        ffn_output = self.dropout2(ffn_output, deterministic=not training)
        
        if self.use_layer_norm:
            return self.layer_norm2(out1 + ffn_output)
        else:
            return out1 + ffn_output
