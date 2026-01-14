"""Training step functions and loss computation."""

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from typing import Tuple, Any

from .compat import update_optimizer


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
    
    # Update model parameters (compatible with both Flax 0.8 and 0.11+)
    update_optimizer(optimizer, model, grads)
    
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