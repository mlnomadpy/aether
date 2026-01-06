"""YatGPT transformer model implementation with YAT architecture.

This module provides a decoupled YAT GPT implementation that is separate from the
standard linear MiniGPT model.
"""

import jax.numpy as jnp
import flax.nnx as nnx
from jax.sharding import NamedSharding, PartitionSpec as P
from typing import Dict, Any, Optional

from .base import BaseModel
from .embeddings import TokenAndPositionEmbedding
from ..utils.attention import causal_attention_mask


# Compatibility check for nnx.List (not available in Flax < 0.11.0)
def _create_module_list(modules):
    """Create a list of modules compatible with the current Flax version."""
    if hasattr(nnx, 'List'):
        return nnx.List(modules)
    else:
        return modules


class YatTransformerBlock(nnx.Module):
    """A transformer block with YAT (YatNMN) architecture.
    
    This block uses the YatNMN non-linear layer for the feed-forward network,
    providing a decoupled implementation from the standard linear TransformerBlock.
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
        param_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        **kwargs
    ):
        """Initialize YAT transformer block.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            rngs: Random number generators
            rate: Dropout rate
            mesh: JAX mesh for sharding
            param_dtype: Data type for parameters
            compute_dtype: Data type for computations
            **kwargs: Additional architecture-specific arguments
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
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
        
        # YAT feed-forward network
        self._create_yat_ffn(embed_dim, ff_dim, kernel_init, bias_init, alpha_init, layer_norm_scale_init, rngs, param_dtype)
            
        self.dropout2 = nnx.Dropout(rate=rate, rngs=rngs)
    
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
        
        # YAT architecture
        out1 = inputs + attention_output
        ffn_output = self.non_linear1(out1)
        ffn_output = self.out_linear1(ffn_output)
        ffn_output = self.dropout2(ffn_output, deterministic=not training)
        return out1 + ffn_output


class YatGPT(BaseModel):
    """A YAT GPT transformer model using YatNMN architecture.
    
    This model is decoupled from the standard MiniGPT and uses YatNMN
    non-linear layers instead of standard linear feed-forward networks.
    """
    
    def __init__(
        self,
        maxlen: int,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        feed_forward_dim: int,
        num_transformer_blocks: int,
        rngs: nnx.Rngs,
        mesh: Optional[object] = None,
        param_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        attention_block_reuse: int = 1,
        **kwargs
    ):
        """Initialize YatGPT model.
        
        Args:
            maxlen: Maximum sequence length
            vocab_size: Vocabulary size
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            feed_forward_dim: Feed-forward dimension
            num_transformer_blocks: Number of transformer blocks
            rngs: Random number generators
            mesh: JAX mesh for sharding
            param_dtype: Data type for parameters
            compute_dtype: Data type for computations
            attention_block_reuse: Number of times to reuse attention blocks (1 = no reuse)
            **kwargs: Additional arguments passed to transformer blocks
        """
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.feed_forward_dim = feed_forward_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.param_dtype = param_dtype
        self.compute_dtype = compute_dtype
        self.attention_block_reuse = attention_block_reuse
        
        # Embedding layer
        self.embedding_layer = TokenAndPositionEmbedding(
            maxlen, vocab_size, embed_dim, rngs=rngs, param_dtype=param_dtype
        )
        
        # YAT Transformer blocks
        self.transformer_blocks = _create_module_list([
            YatTransformerBlock(
                embed_dim, 
                num_heads, 
                feed_forward_dim, 
                rngs=rngs,
                mesh=mesh,
                param_dtype=param_dtype,
                compute_dtype=compute_dtype,
                **kwargs
            )
            for _ in range(num_transformer_blocks)
        ])
        
        # Output layer
        if mesh is not None:
            kernel_init = nnx.with_partitioning(
                nnx.initializers.xavier_uniform(), 
                NamedSharding(mesh, P(None, 'model'))
            )
            bias_init = nnx.with_partitioning(
                nnx.initializers.zeros_init(), 
                NamedSharding(mesh, P('model'))
            )
        else:
            kernel_init = nnx.initializers.xavier_uniform()
            bias_init = nnx.initializers.zeros_init()
        
        self.output_layer = nnx.Linear(
            in_features=embed_dim,
            out_features=vocab_size,
            kernel_init=kernel_init,
            bias_init=bias_init,
            param_dtype=param_dtype,
            rngs=rngs
        )

    def __call__(self, inputs: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Forward pass through the model.
        
        Args:
            inputs: Input token IDs of shape (batch_size, sequence_length)
            training: Whether in training mode
            
        Returns:
            Logits of shape (batch_size, sequence_length, vocab_size)
        """
        x = self.embedding_layer(inputs)
        
        # Apply transformer blocks with reuse
        for _ in range(self.attention_block_reuse):
            for transformer_block in self.transformer_blocks:
                x = transformer_block(x, training=training)
        
        return self.output_layer(x)
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration.
        
        Returns:
            Dictionary containing model configuration
        """
        return {
            'maxlen': self.maxlen,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'feed_forward_dim': self.feed_forward_dim,
            'num_transformer_blocks': self.num_transformer_blocks,
            'architecture': 'yat',
            'attention_block_reuse': self.attention_block_reuse
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], rngs: nnx.Rngs, **kwargs) -> "YatGPT":
        """Create model from configuration.
        
        Args:
            config: Model configuration dictionary
            rngs: Random number generators
            **kwargs: Additional arguments (e.g., mesh)
            
        Returns:
            YatGPT model instance
        """
        # Remove 'architecture' from config if present since it's not a constructor arg
        config = {k: v for k, v in config.items() if k != 'architecture'}
        return cls(rngs=rngs, **config, **kwargs)
