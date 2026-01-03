"""Aether: Modular JAX/Flax Transformer Training Framework."""

import re
from .config import Config, ModelConfig, TrainingConfig, DataConfig, LoggingConfig


def _is_jax_flax_error(error_msg: str) -> bool:
    """Check if an error message is about missing JAX or Flax.
    
    Uses word boundary matching to avoid false positives
    (e.g., 'relaxation' should not match 'lax').
    """
    error_lower = error_msg.lower()
    # Match 'jax' or 'flax' as complete module names (with word boundaries or quotes)
    # Typical error: "No module named 'jax'" or "No module named 'flax.nnx'"
    return bool(re.search(r"['\"]jax['\"\.]", error_lower) or 
                re.search(r"['\"]flax['\"\.]", error_lower) or
                re.search(r"\bjax\b", error_lower) or
                re.search(r"\bflax\b", error_lower))


# Try to import training components, but make them optional
try:
    from .training import Trainer, train_step, eval_step, loss_fn
    _TRAINING_AVAILABLE = True
except ImportError as e:
    # Check if the error is specifically about JAX or Flax
    if _is_jax_flax_error(str(e)):
        _TRAINING_AVAILABLE = False
        # Create placeholder classes/functions that raise informative errors
        class Trainer:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "Training functionality requires JAX/Flax. Please install: "
                    "pip install jax flax"
                )
        
        def train_step(*args, **kwargs):
            raise ImportError("Training functionality requires JAX/Flax")
        
        def eval_step(*args, **kwargs):
            raise ImportError("Training functionality requires JAX/Flax")
        
        def loss_fn(*args, **kwargs):
            raise ImportError("Training functionality requires JAX/Flax")
    else:
        # Re-raise the original error if it's not about JAX/Flax
        raise

__all__ = [
    "Config", 
    "ModelConfig", 
    "TrainingConfig", 
    "DataConfig", 
    "LoggingConfig",
    "Trainer", 
    "train_step", 
    "eval_step", 
    "loss_fn"
]

__version__ = "0.1.0"
