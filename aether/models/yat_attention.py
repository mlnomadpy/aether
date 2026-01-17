"""Aether models with Yat Attention mechanisms.

This module provides Aether GPT variants that implement the Yat-product attention
and Yat-Performer (linearized Yat with random features) attention mechanisms.

The Yat kernel formula: (1 + <q,k>)^2 / (epsilon + (1 - <q,k>))
- With normalized Q/K, this provides a non-linear attention kernel
- YatPerformer uses random spherical features for O(n) linear complexity
"""

import jax
import jax.numpy as jnp
import flax.nnx as nnx
from jax.sharding import NamedSharding, PartitionSpec as P
from typing import Dict, Any, Optional

from .base import BaseModel
from .embeddings import TokenAndPositionEmbedding
from nmn.nnx.nmn import YatNMN

# Compatibility check for nnx.List (not available in Flax < 0.11.0)
def _create_module_list(modules):
    """Create a list of modules compatible with the current Flax version."""
    if hasattr(nnx, 'List'):
        return nnx.List(modules)
    else:
        return modules


class YatAttention(nnx.Module):
    """Yat-product attention (exact).
    
    Implements the Yat kernel attention mechanism:
        scores = (1 + <q,k>)^2 / (epsilon + (1 - <q,k>))
    
    Where Q and K are L2-normalized before computing the dot product.
    This provides a non-linear attention kernel that can capture different
    similarity patterns compared to standard softmax attention.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        rngs: nnx.Rngs,
        yat_epsilon: float = 0.1,
        dropout_rate: float = 0.1,
        param_dtype: jnp.dtype = jnp.float32,
        mesh: Optional[object] = None,
    ):
        """Initialize YatAttention.
        
        Args:
            embed_dim: Total embedding dimension
            num_heads: Number of attention heads
            rngs: Random number generators
            yat_epsilon: Epsilon for numerical stability in Yat kernel denominator
            dropout_rate: Dropout rate for attention weights
            param_dtype: Data type for parameters
            mesh: JAX mesh for sharding
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.yat_epsilon = yat_epsilon
        self.param_dtype = param_dtype
        
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
        else:
            kernel_init = nnx.initializers.xavier_uniform()
            bias_init = nnx.initializers.zeros_init()
        
        # Linear projections for Q, K, V
        self.q_proj = nnx.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
            kernel_init=kernel_init,
            bias_init=bias_init,
            param_dtype=param_dtype,
            rngs=rngs
        )
        self.k_proj = nnx.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
            kernel_init=kernel_init,
            bias_init=bias_init,
            param_dtype=param_dtype,
            rngs=rngs
        )
        self.v_proj = nnx.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
            kernel_init=kernel_init,
            bias_init=bias_init,
            param_dtype=param_dtype,
            rngs=rngs
        )
        self.out_proj = nnx.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
            kernel_init=kernel_init,
            bias_init=bias_init,
            param_dtype=param_dtype,
            rngs=rngs
        )
        
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
    
    def __call__(
        self, 
        x: jnp.ndarray, 
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True
    ) -> jnp.ndarray:
        """Forward pass of YatAttention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask of shape (seq_len, seq_len)
            deterministic: Whether to apply dropout
            
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head attention: (batch, seq, heads, head_dim)
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose to (batch, heads, seq, head_dim)
        Q = jnp.transpose(Q, (0, 2, 1, 3))
        K = jnp.transpose(K, (0, 2, 1, 3))
        V = jnp.transpose(V, (0, 2, 1, 3))
        
        # Normalize Q and K
        Q_norm = Q / (jnp.linalg.norm(Q, axis=-1, keepdims=True) + 1e-8)
        K_norm = K / (jnp.linalg.norm(K, axis=-1, keepdims=True) + 1e-8)
        
        # Yat kernel: (1 + <q,k>)^2 / (epsilon + (1 - <q,k>))
        # dot_product shape: (batch, heads, seq_q, seq_k)
        dot_product = jnp.matmul(Q_norm, jnp.transpose(K_norm, (0, 1, 3, 2)))
        
        numerator = (1 + dot_product) ** 2
        denominator = self.yat_epsilon + (1 - dot_product)
        scores = numerator / denominator
        
        # Apply causal mask if provided
        if mask is not None:
            # mask shape: (seq_len, seq_len), expand to (1, 1, seq_len, seq_len)
            # Convert mask to attention mask (0 for attend, -inf for ignore)
            mask_value = jnp.finfo(scores.dtype).min
            attention_mask = jnp.where(mask == 0, mask_value, 0.0)
            scores = scores + attention_mask[None, None, :, :]
        
        # Normalize scores (instead of softmax, we use L1 normalization)
        attn_weights = scores / (jnp.sum(scores, axis=-1, keepdims=True) + 1e-6)
        attn_weights = self.dropout(attn_weights, deterministic=deterministic)
        
        # Apply attention to values
        # attn_weights: (batch, heads, seq_q, seq_k)
        # V: (batch, heads, seq_k, head_dim)
        output = jnp.matmul(attn_weights, V)
        
        # Transpose back: (batch, heads, seq, head_dim) -> (batch, seq, heads, head_dim)
        output = jnp.transpose(output, (0, 2, 1, 3))
        
        # Reshape to (batch, seq, embed_dim)
        output = output.reshape(batch_size, seq_len, self.embed_dim)
        
        # Output projection
        return self.out_proj(output)


class YatPerformerAttention(nnx.Module):
    """Yat-Performer (linearized Yat with random features).
    
    Approximates the Yat attention kernel using random spherical features
    for O(n) linear complexity instead of O(n^2).
    
    Uses random projections to approximate the Yat kernel in feature space,
    enabling linear-time attention computation via the kernel trick.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        rngs: nnx.Rngs,
        num_random_features: int = 256,
        yat_epsilon: float = 0.1,
        dropout_rate: float = 0.1,
        param_dtype: jnp.dtype = jnp.float32,
        mesh: Optional[object] = None,
    ):
        """Initialize YatPerformerAttention.
        
        Args:
            embed_dim: Total embedding dimension
            num_heads: Number of attention heads
            rngs: Random number generators
            num_random_features: Number of random features for kernel approximation
            yat_epsilon: Epsilon for numerical stability
            dropout_rate: Dropout rate
            param_dtype: Data type for parameters
            mesh: JAX mesh for sharding
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_random_features = num_random_features
        self.yat_epsilon = yat_epsilon
        self.param_dtype = param_dtype
        
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
        else:
            kernel_init = nnx.initializers.xavier_uniform()
            bias_init = nnx.initializers.zeros_init()
        
        # Linear projections for Q, K, V
        self.q_proj = nnx.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
            kernel_init=kernel_init,
            bias_init=bias_init,
            param_dtype=param_dtype,
            rngs=rngs
        )
        self.k_proj = nnx.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
            kernel_init=kernel_init,
            bias_init=bias_init,
            param_dtype=param_dtype,
            rngs=rngs
        )
        self.v_proj = nnx.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
            kernel_init=kernel_init,
            bias_init=bias_init,
            param_dtype=param_dtype,
            rngs=rngs
        )
        self.out_proj = nnx.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
            kernel_init=kernel_init,
            bias_init=bias_init,
            param_dtype=param_dtype,
            rngs=rngs
        )
        
        # Random projection matrix for spherical features
        # Shape: (num_features, head_dim) for each head
        projection_key = rngs.params()
        self.projection = nnx.Param(
            jax.random.normal(projection_key, (num_heads, num_random_features, self.head_dim), dtype=param_dtype)
        )
        
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
    
    def _spherical_features(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute spherical random features.
        
        Args:
            x: Input tensor of shape (batch, heads, seq, head_dim)
            
        Returns:
            Feature tensor of shape (batch, heads, seq, num_features)
        """
        # Normalize input
        x_norm = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
        
        # projection shape: (num_heads, num_random_features, head_dim)
        # We want: (batch, heads, seq, head_dim) @ (heads, num_features, head_dim).T
        # -> (batch, heads, seq, num_features)
        # Use einsum: bhsd,hfd->bhsf where d is head_dim and f is num_features
        proj = jnp.einsum('bhsd,hfd->bhsf', x_norm, self.projection.value)
        
        # Apply ReLU + epsilon for positive features
        return nnx.relu(proj) + self.yat_epsilon
    
    def __call__(
        self, 
        x: jnp.ndarray, 
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True
    ) -> jnp.ndarray:
        """Forward pass of YatPerformerAttention.
        
        Uses the kernel trick for linear complexity:
            Attn(Q, K, V) = phi(Q) @ (phi(K)^T @ V) / normalizer
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask (ignored in Performer for efficiency)
            deterministic: Whether to apply dropout
            
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head attention: (batch, seq, heads, head_dim)
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose to (batch, heads, seq, head_dim)
        Q = jnp.transpose(Q, (0, 2, 1, 3))
        K = jnp.transpose(K, (0, 2, 1, 3))
        V = jnp.transpose(V, (0, 2, 1, 3))
        
        # Compute spherical features
        Q_prime = self._spherical_features(Q)  # (batch, heads, seq, num_features)
        K_prime = self._spherical_features(K)  # (batch, heads, seq, num_features)
        
        # Linear attention via kernel trick
        # KV = K_prime^T @ V: (batch, heads, num_features, head_dim)
        KV = jnp.einsum('bhsf,bhsd->bhfd', K_prime, V)
        
        # QKV = Q_prime @ KV: (batch, heads, seq, head_dim)
        QKV = jnp.einsum('bhsf,bhfd->bhsd', Q_prime, KV)
        
        # Normalizer: sum over all K features for each Q position
        # (batch, heads, seq, num_features) @ sum(batch, heads, seq, num_features) -> (batch, heads, seq)
        K_sum = jnp.sum(K_prime, axis=2)  # (batch, heads, num_features)
        normalizer = jnp.einsum('bhsf,bhf->bhs', Q_prime, K_sum)
        normalizer = normalizer[..., None]  # (batch, heads, seq, 1)
        normalizer = jnp.maximum(normalizer, 1e-6)
        
        # Normalize output
        output = QKV / normalizer
        
        # Transpose back: (batch, heads, seq, head_dim) -> (batch, seq, heads, head_dim)
        output = jnp.transpose(output, (0, 2, 1, 3))
        
        # Reshape to (batch, seq, embed_dim)
        output = output.reshape(batch_size, seq_len, self.embed_dim)
        
        # Output projection and dropout
        output = self.out_proj(output)
        output = self.dropout(output, deterministic=deterministic)
        
        return output


class YatAttentionTransformerBlock(nnx.Module):
    """Transformer block using exact YatAttention.
    
    Uses the Yat-product attention mechanism with O(n^2) complexity.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        *,
        rngs: nnx.Rngs,
        rate: float = 0.1,
        yat_epsilon: float = 0.1,
        mesh: Optional[object] = None,
        param_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        **kwargs
    ):
        """Initialize YatAttention transformer block.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            rngs: Random number generators
            rate: Dropout rate
            yat_epsilon: Epsilon for Yat attention kernel
            mesh: JAX mesh for sharding
            param_dtype: Data type for parameters
            compute_dtype: Data type for computations
            **kwargs: Additional arguments (ignored)
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.yat_epsilon = yat_epsilon
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
        
        # YatAttention
        self.attn = YatAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            rngs=rngs,
            yat_epsilon=yat_epsilon,
            dropout_rate=rate,
            param_dtype=param_dtype,
            mesh=mesh,
        )
        self.dropout1 = nnx.Dropout(rate=rate, rngs=rngs)
        
        # Feed-forward network
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
        self.dropout2 = nnx.Dropout(rate=rate, rngs=rngs)
    
    def __call__(self, inputs: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Forward pass through the transformer block.
        
        Args:
            inputs: Input tensor of shape (batch_size, sequence_length, embed_dim)
            training: Whether in training mode
            
        Returns:
            Output tensor of same shape as inputs
        """
        seq_len = inputs.shape[1]
        # Create causal mask
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        
        # YatAttention with residual
        attention_output = self.attn(inputs, mask=mask, deterministic=not training)
        attention_output = self.dropout1(attention_output, deterministic=not training)
        out1 = inputs + attention_output
        
        # Feed-forward with residual
        ffn_output = self.non_linear1(out1)
        ffn_output = self.out_linear1(ffn_output)
        ffn_output = self.dropout2(ffn_output, deterministic=not training)
        
        return out1 + ffn_output


class YatPerformerTransformerBlock(nnx.Module):
    """Transformer block using linearized YatPerformerAttention.
    
    Uses the Yat-Performer attention mechanism with O(n) linear complexity.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        *,
        rngs: nnx.Rngs,
        rate: float = 0.1,
        num_random_features: int = 256,
        yat_epsilon: float = 0.1,
        mesh: Optional[object] = None,
        param_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        **kwargs
    ):
        """Initialize YatPerformer transformer block.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            rngs: Random number generators
            rate: Dropout rate
            num_random_features: Number of random features for kernel approximation
            yat_epsilon: Epsilon for Yat attention kernel
            mesh: JAX mesh for sharding
            param_dtype: Data type for parameters
            compute_dtype: Data type for computations
            **kwargs: Additional arguments (ignored)
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.num_random_features = num_random_features
        self.yat_epsilon = yat_epsilon
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
        
        # YatPerformerAttention
        self.attn = YatPerformerAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            rngs=rngs,
            num_random_features=num_random_features,
            yat_epsilon=yat_epsilon,
            dropout_rate=rate,
            param_dtype=param_dtype,
            mesh=mesh,
        )
        self.dropout1 = nnx.Dropout(rate=rate, rngs=rngs)
        
        
        # Feed-forward network
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
        self.linear2 = nnx.Linear(
            in_features=4 * embed_dim,
            out_features=embed_dim,
            use_bias=False,
            kernel_init=kernel_init,
            bias_init=bias_init,
            param_dtype=param_dtype,
            rngs=rngs
        )

        self.dropout2 = nnx.Dropout(rate=rate, rngs=rngs)
    
    def __call__(self, inputs: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Forward pass through the transformer block.
        
        Args:
            inputs: Input tensor of shape (batch_size, sequence_length, embed_dim)
            training: Whether in training mode
            
        Returns:
            Output tensor of same shape as inputs
        """
        # YatPerformerAttention (no explicit mask - it's a global attention)
        attention_output = self.attn(inputs, deterministic=not training)
        attention_output = self.dropout1(attention_output, deterministic=not training)
        out1 = inputs + attention_output
        
        # Feed-forward with residual
        ffn_output = self.non_linear1(out1)
        ffn_output = self.linear2(ffn_output)
        ffn_output = self.dropout2(ffn_output, deterministic=not training)
        
        return out1 + ffn_output


class AetherYat(BaseModel):
    """Aether GPT model with exact YatAttention.
    
    A transformer language model that uses the Yat-product attention mechanism
    with O(n^2) complexity. The Yat kernel provides a non-linear attention
    pattern that can capture different similarity relationships.
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
        yat_epsilon: float = 0.1,
        dropout_rate: float = 0.1,
        attention_block_reuse: int = 1,
        **kwargs
    ):
        """Initialize AetherYat model.
        
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
            yat_epsilon: Epsilon for Yat attention kernel
            dropout_rate: Dropout rate
            attention_block_reuse: Number of times to reuse attention blocks (1 = no reuse)
            **kwargs: Additional arguments (ignored)
        """
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.feed_forward_dim = feed_forward_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.param_dtype = param_dtype
        self.compute_dtype = compute_dtype
        self.yat_epsilon = yat_epsilon
        self.dropout_rate = dropout_rate
        self.attention_block_reuse = attention_block_reuse
        
        # Embedding layer
        self.embedding_layer = TokenAndPositionEmbedding(
            maxlen, vocab_size, embed_dim, rngs=rngs, param_dtype=param_dtype
        )
        
        # Transformer blocks with YatAttention
        self.transformer_blocks = _create_module_list([
            YatAttentionTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=feed_forward_dim,
                rngs=rngs,
                rate=dropout_rate,
                yat_epsilon=yat_epsilon,
                mesh=mesh,
                param_dtype=param_dtype,
                compute_dtype=compute_dtype,
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
            'architecture': 'aether_yat',
            'yat_epsilon': self.yat_epsilon,
            'dropout_rate': self.dropout_rate,
            'attention_block_reuse': self.attention_block_reuse,
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], rngs: nnx.Rngs, **kwargs) -> "AetherYat":
        """Create model from configuration.
        
        Args:
            config: Model configuration dictionary
            rngs: Random number generators
            **kwargs: Additional arguments (e.g., mesh)
            
        Returns:
            AetherYat model instance
        """
        # Remove 'architecture' from config if present since it's not a constructor arg
        config = {k: v for k, v in config.items() if k != 'architecture'}
        return cls(rngs=rngs, **config, **kwargs)


class AetherYatPerformer(BaseModel):
    """Aether GPT model with linearized YatPerformerAttention.
    
    A transformer language model that uses the Yat-Performer attention mechanism
    with O(n) linear complexity via random spherical features.
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
        num_random_features: int = 256,
        yat_epsilon: float = 0.1,
        dropout_rate: float = 0.1,
        attention_block_reuse: int = 1,
        **kwargs
    ):
        """Initialize AetherYatPerformer model.
        
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
            num_random_features: Number of random features for kernel approximation
            yat_epsilon: Epsilon for Yat attention kernel
            dropout_rate: Dropout rate
            attention_block_reuse: Number of times to reuse attention blocks (1 = no reuse)
            **kwargs: Additional arguments (ignored)
        """
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.feed_forward_dim = feed_forward_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.param_dtype = param_dtype
        self.compute_dtype = compute_dtype
        self.num_random_features = num_random_features
        self.yat_epsilon = yat_epsilon
        self.dropout_rate = dropout_rate
        self.attention_block_reuse = attention_block_reuse
        
        # Embedding layer
        self.embedding_layer = TokenAndPositionEmbedding(
            maxlen, vocab_size, embed_dim, rngs=rngs, param_dtype=param_dtype
        )
        
        # Transformer blocks with YatPerformerAttention
        self.transformer_blocks = _create_module_list([
            YatPerformerTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=feed_forward_dim,
                rngs=rngs,
                rate=dropout_rate,
                num_random_features=num_random_features,
                yat_epsilon=yat_epsilon,
                mesh=mesh,
                param_dtype=param_dtype,
                compute_dtype=compute_dtype,
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
            'architecture': 'aether_yat_performer',
            'num_random_features': self.num_random_features,
            'yat_epsilon': self.yat_epsilon,
            'dropout_rate': self.dropout_rate,
            'attention_block_reuse': self.attention_block_reuse,
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], rngs: nnx.Rngs, **kwargs) -> "AetherYatPerformer":
        """Create model from configuration.
        
        Args:
            config: Model configuration dictionary
            rngs: Random number generators
            **kwargs: Additional arguments (e.g., mesh)
            
        Returns:
            AetherYatPerformer model instance
        """
        # Remove 'architecture' from config if present since it's not a constructor arg
        config = {k: v for k, v in config.items() if k != 'architecture'}
        return cls(rngs=rngs, **config, **kwargs)
