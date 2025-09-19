"""Training step functions and loss computation."""

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from typing import Tuple, Any, Dict, Union


def loss_fn(model: nnx.Module, batch: Union[jnp.ndarray, Dict[str, jnp.ndarray]], training: bool = True, training_mode: str = "clm") -> Tuple[jnp.ndarray, Any]:
    """Compute the cross-entropy loss for language modeling.
    
    Args:
        model: The model to evaluate
        batch: Input batch - either jnp.ndarray for CLM or Dict with keys for MLM
        training: Whether in training mode
        training_mode: Training mode ("clm" or "mlm")
        
    Returns:
        Tuple of (loss, logits)
    """
    if training_mode == "clm":
        return clm_loss_fn(model, batch, training)
    elif training_mode == "mlm":
        return mlm_loss_fn(model, batch, training)
    else:
        raise ValueError(f"Unsupported training mode: {training_mode}")


def clm_loss_fn(model: nnx.Module, batch: jnp.ndarray, training: bool = True) -> Tuple[jnp.ndarray, Any]:
    """Compute the cross-entropy loss for causal language modeling.
    
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


def mlm_loss_fn(model: nnx.Module, batch: Dict[str, jnp.ndarray], training: bool = True) -> Tuple[jnp.ndarray, Any]:
    """Compute the cross-entropy loss for masked language modeling.
    
    Args:
        model: The model to evaluate
        batch: Batch dictionary with 'masked_tokens' and 'mask_labels'
        training: Whether in training mode
        
    Returns:
        Tuple of (loss, logits)
    """
    masked_tokens = batch['masked_tokens']
    mask_labels = batch['mask_labels']
    
    # Forward pass with masked tokens
    logits = model(masked_tokens, training=training)
    
    # Only compute loss on masked positions
    # mask_labels contains -100 for unmasked positions and original token_id for masked positions
    mask = mask_labels != -100
    
    # Get logits and labels for masked positions only
    masked_logits = logits[mask]
    masked_labels = mask_labels[mask]
    
    # Compute cross-entropy loss only on masked tokens
    if masked_logits.shape[0] > 0:  # Ensure we have some masked tokens
        loss = optax.softmax_cross_entropy_with_integer_labels(
            masked_logits,
            masked_labels
        )
        return jnp.mean(loss), logits
    else:
        # No masked tokens in this batch (edge case)
        return jnp.array(0.0), logits


@nnx.jit(static_argnames=('training_mode',))
def train_step(
    model: nnx.Module, 
    optimizer: nnx.Optimizer, 
    batch: Union[jnp.ndarray, Dict[str, jnp.ndarray]],
    training_mode: str = "clm"
) -> Tuple[jnp.ndarray, nnx.Module, nnx.Optimizer]:
    """Perform a single training step.
    
    Args:
        model: The model to train
        optimizer: The optimizer
        batch: Input batch
        training_mode: Training mode ("clm" or "mlm")
        
    Returns:
        Tuple of (loss, updated_model, updated_optimizer)
    """
    # Compute loss and gradients
    grad_fn = nnx.value_and_grad(lambda m, b: loss_fn(m, b, training=True, training_mode=training_mode)[0])
    loss, grads = grad_fn(model, batch)
    
    # Update model parameters
    optimizer.update(model, grads)
    
    return loss, model, optimizer


@nnx.jit(static_argnames=('training_mode',))
def eval_step(model: nnx.Module, batch: Union[jnp.ndarray, Dict[str, jnp.ndarray]], training_mode: str = "clm") -> jnp.ndarray:
    """Perform a single evaluation step.
    
    Args:
        model: The model to evaluate
        batch: Input batch
        training_mode: Training mode ("clm" or "mlm")
        
    Returns:
        Loss value
    """
    # `training=False` ensures dropout and other training-specific layers are disabled
    loss, _ = loss_fn(model, batch, training=False, training_mode=training_mode)
    return loss