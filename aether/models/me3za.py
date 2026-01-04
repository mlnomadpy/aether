"""Me3za: Modern BERT with Rotary YAT Performer Attention.

Me3za is a modern BERT-style encoder that uses:
- RMSNorm instead of LayerNorm
- Rotary Position Embeddings (RoPE) 
- YAT attention with optional Performer mode for O(n) linear complexity
- YatNMN feed-forward network
- Pre-norm architecture

The attention uses the YAT formula: (q·k)² / (2(1 - q·k) + ε)
With normalized Q/K, only ONE dot product is needed!
"""

import jax
import jax.numpy as jnp
import flax.nnx as nnx
from jax.sharding import NamedSharding, PartitionSpec as P
from typing import Dict, Any, Optional

from .base import BaseModel
from .embeddings import RMSNorm, TokenOnlyEmbedding


# Compatibility check for nnx.List (not available in Flax < 0.11.0)
def _create_module_list(modules):
    """Create a list of modules compatible with the current Flax version."""
    if hasattr(nnx, 'List'):
        return nnx.List(modules)
    else:
        return modules


class Me3zaTransformerBlock(nnx.Module):
    """Transformer block with RMSNorm and Rotary YAT Performer Attention.
    
    Uses Pre-Norm architecture:
        x = x + Drop(Attn(Norm(x)))
        x = x + Drop(FFN(Norm(x)))
    
    The attention uses O(n) linear complexity with normalized Q/K optimization:
        YAT formula: (q·k)² / (2(1 - q·k) + ε)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        maxlen: int,
        *,
        rngs: nnx.Rngs,
        rate: float = 0.1,
        mesh: Optional[object] = None,
        use_performer: bool = True,
        num_features: Optional[int] = None,
        param_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
    ):
        """Initialize Me3za transformer block.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            maxlen: Maximum sequence length (for RoPE precomputation)
            rngs: Random number generators
            rate: Dropout rate
            mesh: JAX mesh for sharding
            use_performer: Whether to use Performer mode for O(n) complexity
            num_features: Number of random features for Performer (default: embed_dim // 2)
            param_dtype: Data type for parameters
            compute_dtype: Data type for computations
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.maxlen = maxlen
        self.rate = rate
        self.use_performer = use_performer
        self.param_dtype = param_dtype
        self.compute_dtype = compute_dtype
        
        # Store rngs for dropout during training
        self.rngs = rngs
        
        # Set up partitioning if mesh is provided
        if mesh is not None:
            kernel_init = nnx.with_partitioning(
                nnx.initializers.xavier_uniform(), 
                NamedSharding(mesh, P(None, 'model'))
            )
        else:
            kernel_init = nnx.initializers.xavier_uniform()
        
        # Use RotaryYatAttention with optional Performer mode
        try:
            from nmn.nnx.attention import RotaryYatAttention
        except ImportError:
            raise ImportError(
                "Me3za architecture requires the 'nmn' package. "
                "Please install it with: pip install nmn"
            )
        
        num_features = num_features or embed_dim // 2
        
        self.attn = RotaryYatAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_seq_len=maxlen,
            kernel_init=kernel_init,
            use_bias=False,
            dropout_rate=rate,
            # Performer mode for linear complexity
            use_performer=use_performer,
            num_features=num_features,
            performer_normalize=True,  # Optimized: only ONE dot product needed!
            # Alpha scaling for YAT attention
            constant_alpha=True,  # Use sqrt(2) as constant alpha
            rngs=rngs,
        )
        self.norm1 = RMSNorm(embed_dim, rngs=rngs, param_dtype=param_dtype)
        self.dropout1 = nnx.Dropout(rate=rate, rngs=rngs)
        
        # Feed-forward network with YatNMN
        try:
            from nmn.nnx.nmn import YatNMN
        except ImportError:
            raise ImportError(
                "Me3za architecture requires the 'nmn' package. "
                "Please install it with: pip install nmn"
            )
        
        self.ffn_yat = YatNMN(
            in_features=embed_dim,
            out_features=ff_dim,
            kernel_init=kernel_init,
            rngs=rngs,
            use_bias=False
        )
        self.ffn_out = nnx.Linear(
            in_features=ff_dim,
            out_features=embed_dim,
            kernel_init=kernel_init,
            use_bias=False,
            param_dtype=param_dtype,
            rngs=rngs
        )
        self.norm2 = RMSNorm(embed_dim, rngs=rngs, param_dtype=param_dtype)
        self.dropout2 = nnx.Dropout(rate=rate, rngs=rngs)

    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Forward pass through the transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, embed_dim)
            training: Whether in training mode
            
        Returns:
            Output tensor of same shape as input
        """
        # Pre-Norm Architecture
        # x = x + Drop(Attn(Norm(x)))
        h = self.norm1(x)
        attn_out = self.attn(h, deterministic=not training, rngs=self.rngs if training else None)
        x = x + self.dropout1(attn_out, deterministic=not training)
        
        # x = x + Drop(FFN(Norm(x)))
        h = self.norm2(x)
        ffn_out = self.ffn_yat(h)
        ffn_out = self.ffn_out(ffn_out)
        x = x + self.dropout2(ffn_out, deterministic=not training)
        
        return x


class Me3za(BaseModel):
    """Me3za: Modern BERT with Rotary YAT Performer Attention.
    
    This model combines several modern techniques for efficient transformers:
    - Rotary Position Embeddings (RoPE) for position encoding
    - YAT attention with normalized Q/K: (q·k)² / (2(1 - q·k) + ε)
    - Optional FAVOR+ random features for O(n) linear complexity
    - YatNMN in the feed-forward network
    - RMSNorm instead of LayerNorm
    - Pre-norm architecture
    
    The normalization trick: since ||q|| = ||k|| = 1, we only need ONE dot product
    instead of computing separate squared norms.
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
        use_performer: bool = True,
        num_features: Optional[int] = None,
        param_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        """Initialize Me3za model.
        
        Args:
            maxlen: Maximum sequence length
            vocab_size: Vocabulary size
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            feed_forward_dim: Feed-forward dimension
            num_transformer_blocks: Number of transformer blocks
            rngs: Random number generators
            mesh: JAX mesh for sharding
            use_performer: Whether to use Performer mode for O(n) complexity
            num_features: Number of random features for Performer
            param_dtype: Data type for parameters
            compute_dtype: Data type for computations
            dropout_rate: Dropout rate
            **kwargs: Additional arguments (ignored)
        """
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.feed_forward_dim = feed_forward_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.use_performer = use_performer
        self.num_features = num_features
        self.param_dtype = param_dtype
        self.compute_dtype = compute_dtype
        self.dropout_rate = dropout_rate
        
        # Token embeddings only (no positional - RoPE is handled in attention)
        self.embedding_layer = TokenOnlyEmbedding(
            vocab_size, embed_dim, rngs=rngs, param_dtype=param_dtype
        )
        
        # Transformer blocks with RoPE and YAT attention
        self.transformer_blocks = _create_module_list([
            Me3zaTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=feed_forward_dim,
                maxlen=maxlen,
                rngs=rngs,
                rate=dropout_rate,
                mesh=mesh,
                use_performer=use_performer,
                num_features=num_features,
                param_dtype=param_dtype,
                compute_dtype=compute_dtype,
            )
            for _ in range(num_transformer_blocks)
        ])
        
        # Final RMSNorm
        self.norm_final = RMSNorm(embed_dim, rngs=rngs, param_dtype=param_dtype)
        
        # Weight Tying: We'll reuse embedding weights for output projection
        self.head_dim = embed_dim // num_heads

    def __call__(self, inputs: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Forward pass through the model.
        
        Args:
            inputs: Input token IDs of shape (batch_size, sequence_length)
            training: Whether in training mode
            
        Returns:
            Logits of shape (batch_size, sequence_length, vocab_size)
        """
        x = self.embedding_layer(inputs)
        
        # RoPE is handled internally by RotaryYatAttention
        for block in self.transformer_blocks:
            x = block(x, training=training)
        
        x = self.norm_final(x)
        
        # Weight Tying: Reuse embedding weights for output projection
        # x: [Batch, Seq, Dim]
        # emb: [Vocab, Dim]
        # logits = x @ emb.T -> [Batch, Seq, Vocab]
        embedding_weights = self.embedding_layer.token_emb.embedding[...]
        logits = x @ embedding_weights.T
        
        return logits

    def embed(self, inputs: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Gets embeddings before the final output layer.
        
        Useful for sentence embeddings, downstream tasks, etc.
        
        Args:
            inputs: Input token IDs of shape (batch_size, sequence_length)
            training: Whether in training mode
            
        Returns:
            Embeddings of shape (batch_size, sequence_length, embed_dim)
        """
        x = self.embedding_layer(inputs)
        
        for block in self.transformer_blocks:
            x = block(x, training=training)
        
        return self.norm_final(x)

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
            'architecture': 'me3za',
            'use_performer': self.use_performer,
            'num_features': self.num_features,
            'dropout_rate': self.dropout_rate,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any], rngs: nnx.Rngs, **kwargs) -> "Me3za":
        """Create model from configuration.
        
        Args:
            config: Model configuration dictionary
            rngs: Random number generators
            **kwargs: Additional arguments (e.g., mesh)
            
        Returns:
            Me3za model instance
        """
        # Remove 'architecture' from config if present since it's not a constructor arg
        config = {k: v for k, v in config.items() if k != 'architecture'}
        return cls(rngs=rngs, **config, **kwargs)
