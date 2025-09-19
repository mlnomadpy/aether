"""MiniGPT transformer model implementation."""

import jax.numpy as jnp
import flax.nnx as nnx
from jax.sharding import NamedSharding, PartitionSpec as P
from typing import Dict, Any, Optional

from .base import BaseModel
from .transformer_block import TransformerBlock
from .embeddings import TokenAndPositionEmbedding


class MiniGPT(BaseModel):
    """A miniGPT transformer model with configurable architecture."""
    
    def __init__(
        self,
        maxlen: int,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        feed_forward_dim: int,
        num_transformer_blocks: int,
        rngs: nnx.Rngs,
        architecture: str = "linear",
        mesh: Optional[object] = None,
        param_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        mlp_dim_multiplier: float = 4.0,
        **kwargs
    ):
        """Initialize MiniGPT model.
        
        Args:
            maxlen: Maximum sequence length
            vocab_size: Vocabulary size
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            feed_forward_dim: Feed-forward dimension
            num_transformer_blocks: Number of transformer blocks
            rngs: Random number generators
            architecture: Architecture type ('linear' or 'yat')
            mesh: JAX mesh for sharding
            param_dtype: Data type for parameters
            compute_dtype: Data type for computations
            mlp_dim_multiplier: Multiplier for MLP hidden dimension (only used in 'yat' architecture)
            **kwargs: Additional arguments passed to transformer blocks
        """
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.feed_forward_dim = feed_forward_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.architecture = architecture
        self.param_dtype = param_dtype
        self.compute_dtype = compute_dtype
        self.mlp_dim_multiplier = mlp_dim_multiplier
        
        # Embedding layer
        self.embedding_layer = TokenAndPositionEmbedding(
            maxlen, vocab_size, embed_dim, rngs=rngs, param_dtype=param_dtype
        )
        
        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(
                embed_dim, 
                num_heads, 
                feed_forward_dim, 
                rngs=rngs,
                mesh=mesh,
                architecture=architecture,
                param_dtype=param_dtype,
                compute_dtype=compute_dtype,
                mlp_dim_multiplier=mlp_dim_multiplier,
                **kwargs
            )
            for _ in range(num_transformer_blocks)
        ]
        
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
        
        # if architecture == "yat":
        #     try:
        #         from nmn.nnx.nmn import YatNMN
        #         self.output_layer = YatNMN(
        #             in_features=embed_dim,
        #             out_features=vocab_size,
        #             kernel_init=kernel_init,
        #             bias_init=bias_init,
        #             use_bias=False,
        #             rngs=rngs
        #         )
        #     except ImportError:
        #         raise ImportError("YatNMN architecture requires the 'nmn' package.")
        # else:
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
            'architecture': self.architecture,
            'mlp_dim_multiplier': self.mlp_dim_multiplier
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], rngs: nnx.Rngs, **kwargs) -> "MiniGPT":
        """Create model from configuration.
        
        Args:
            config: Model configuration dictionary
            rngs: Random number generators
            **kwargs: Additional arguments (e.g., mesh)
            
        Returns:
            MiniGPT model instance
        """
        return cls(rngs=rngs, **config, **kwargs)
