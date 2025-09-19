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


def clm_loss_fn(model: nnx.Module, batch: Union[jnp.ndarray, Dict[str, jnp.ndarray]], training: bool = True) -> Tuple[jnp.ndarray, Any]:
    """Compute the cross-entropy loss for causal language modeling.
    
    Args:
        model: The model to evaluate
        batch: Input batch of token IDs (jnp.ndarray) or dict with 'tokens' key
        training: Whether in training mode
        
    Returns:
        Tuple of (loss, logits)
    """
    # Handle both array and dictionary batch formats
    if isinstance(batch, dict):
        if 'tokens' in batch:
            tokens = batch['tokens']
        else:
            raise ValueError("CLM mode expects either a jnp.ndarray or a dict with 'tokens' key")
    else:
        tokens = batch
    
    # Forward pass
    logits = model(tokens, training=training)
    
    # Shift for language modeling: predict next token
    # Input: [BOS, token1, token2, ..., tokenN]
    # Target: [token1, token2, ..., tokenN, EOS]
    input_tokens = tokens[:, :-1]  # Remove last token
    target_tokens = tokens[:, 1:]  # Remove first token
    
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
    
    # Use JAX-compatible method to compute loss only on masked tokens
    # Instead of boolean indexing, we'll compute loss on all positions and mask out non-masked ones
    # Reshape for cross-entropy computation
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = mask_labels.reshape(-1)
    flat_mask = mask.reshape(-1)
    
    # Compute cross-entropy loss on all positions
    all_losses = optax.softmax_cross_entropy_with_integer_labels(
        flat_logits,
        jnp.where(flat_mask, flat_labels, 0)  # Use 0 for non-masked positions (will be masked out)
    )
    
    # Apply mask to only consider losses from masked positions
    masked_losses = jnp.where(flat_mask, all_losses, 0.0)
    
    # Compute mean loss over masked positions only
    num_masked = jnp.sum(flat_mask)
    loss = jnp.where(num_masked > 0, jnp.sum(masked_losses) / num_masked, 0.0)
    
    return loss, logits


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