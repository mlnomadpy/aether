"""Transformer block implementations."""

import jax.numpy as jnp
import flax.nnx as nnx
from jax.sharding import NamedSharding, PartitionSpec as P
from typing import Optional, Union

from ..utils.attention import causal_attention_mask


class TransformerBlock(nnx.Module):
    """A transformer block with configurable architecture."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        *,
        rngs: nnx.Rngs,
        rate: float = 0.1,
        mesh: Optional[object] = None,
        architecture: str = "linear",
        use_layer_norm: bool = True,
        param_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        residual_scale: float = 1.0,
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
            architecture: Architecture type ('linear' or 'yat')
            use_layer_norm: Whether to use layer normalization (only for linear architecture)
            param_dtype: Data type for parameters
            compute_dtype: Data type for computations
            residual_scale: Scale factor for residual connections (for YAT architecture stability)
            **kwargs: Additional architecture-specific arguments
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.architecture = architecture
        self.use_layer_norm = use_layer_norm
        self.param_dtype = param_dtype
        self.residual_scale = residual_scale
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
            alpha_init = nnx.with_partitioning(
                nnx.initializers.ones_init(), 
                NamedSharding(mesh, P(None, 'model'))
            )
        else:
            kernel_init = nnx.initializers.xavier_uniform()
            bias_init = nnx.initializers.zeros_init()
            layer_norm_scale_init = nnx.initializers.ones_init()
            alpha_init = nnx.initializers.ones_init()
        
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
        
        # Feed-forward network based on architecture
        if architecture == "linear":
            self._create_linear_ffn(embed_dim, ff_dim, kernel_init, bias_init, layer_norm_scale_init, rngs, param_dtype, use_layer_norm)
        elif architecture == "yat":
            self._create_yat_ffn(embed_dim, ff_dim, kernel_init, bias_init, alpha_init, layer_norm_scale_init, rngs, param_dtype)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
            
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
    
    def _create_yat_ffn(self, embed_dim, ff_dim, kernel_init, bias_init, alpha_init, layer_norm_scale_init, rngs, param_dtype):
        """Create YAT feed-forward network."""
        try:
            from nmn.nnx.nmn import YatNMN
        except ImportError:
            raise ImportError("YatNMN architecture requires the 'nmn' package. Please install it first.")
        
        self.non_linear1 = YatNMN(
            in_features=embed_dim,
            out_features=4 * embed_dim,
            use_dropconnect=False,
            use_bias=False,
            drop_rate=0.,
            kernel_init=kernel_init,
            alpha_init=alpha_init,
            bias_init=bias_init,
            rngs=rngs
        )
        self.out_linear1 = nnx.Linear(
            in_features=4 * embed_dim,
            out_features=embed_dim,
            use_bias=False,
            kernel_init=kernel_init,
            bias_init=bias_init,
            param_dtype=param_dtype,
            rngs=rngs
        )
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
        
        if self.architecture == "linear":
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
        
        elif self.architecture == "yat":
            # YAT architecture without explicit layer normalization
            # Uses scaled residual connections for training stability
            out1 = inputs + self.residual_scale * attention_output
            ffn_output = self.non_linear1(out1)
            ffn_output = self.out_linear1(ffn_output)
            ffn_output = self.dropout2(ffn_output, deterministic=not training)
            return out1 + self.residual_scale * ffn_output
        
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
