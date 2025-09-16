"""Training step functions and loss computation."""

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from typing import Tuple, Any, Optional


def loss_fn(model: nnx.Module, batch: jnp.ndarray, training: bool = True) -> Tuple[jnp.ndarray, Any]:
    """Compute the cross-entropy loss for language modeling.
    
    Args:
        model: The model to evaluate
        batch: Input batch of token IDs
        training: Whether in training mode
        
    Returns:
        Tuple of (loss, logits)
    """
    # Forward pass
    logits = model(batch, training=training)
    
    # Shift for language modeling: predict next token
    # Input: [BOS, token1, token2, ..., tokenN]
    # Target: [token1, token2, ..., tokenN, EOS]
    input_tokens = batch[:, :-1]  # Remove last token
    target_tokens = batch[:, 1:]  # Remove first token
    
    # Get logits for the input tokens
    logits = logits[:, :-1, :]  # Remove logits for last position
    
    # Compute cross-entropy loss
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, logits.shape[-1]),
        target_tokens.reshape(-1)
    )
    
    return jnp.mean(loss), logits


def apply_mixed_precision(params, mixed_precision: Optional[str]) -> Any:
    """Apply mixed precision casting to model parameters.
    
    Args:
        params: Model parameters
        mixed_precision: Mixed precision mode ("fp16", "bfloat16", or None)
        
    Returns:
        Parameters cast to specified precision
    """
    if mixed_precision is None:
        return params
    
    target_dtype = jnp.bfloat16 if mixed_precision == "bfloat16" else jnp.float16
    
    def cast_to_precision(x):
        if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating):
            return jax.lax.convert_element_type(x, target_dtype)
        return x
    
    return jax.tree_util.tree_map(cast_to_precision, params)


@nnx.jit
def train_step(
    model: nnx.Module, 
    optimizer: nnx.Optimizer, 
    batch: jnp.ndarray
) -> Tuple[jnp.ndarray, nnx.Module, nnx.Optimizer]:
    """Perform a single training step.
    
    Args:
        model: The model to train
        optimizer: The optimizer
        batch: Input batch
        
    Returns:
        Tuple of (loss, updated_model, updated_optimizer)
    """
    # Compute loss and gradients
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = grad_fn(model, batch, training=True)
    
    # Update model parameters
    # Handle compatibility between different Flax versions
    try:
        # New API (Flax >= 0.11.0): update(model, grads)
        optimizer.update(model, grads)
    except TypeError as e:
        if "takes 2 positional arguments but 3 were given" in str(e):
            # Old API (Flax < 0.11.0): update(grads)
            optimizer.update(grads)
        else:
            raise e
    
    return loss, model, optimizer


@nnx.jit
def eval_step(model: nnx.Module, batch: jnp.ndarray) -> jnp.ndarray:
    """Perform a single evaluation step.
    
    Args:
        model: The model to evaluate
        batch: Input batch
        
    Returns:
        Loss value
    """
    # `training=False` ensures dropout and other training-specific layers are disabled
    loss, _ = loss_fn(model, batch, training=False)
    return loss