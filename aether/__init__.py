"""Aether: Modular JAX/Flax Transformer Training Framework."""

from .config import Config, ModelConfig, TrainingConfig, DataConfig, LoggingConfig

# Try to import training components, but make them optional
try:
    from .training import Trainer, train_step, eval_step, loss_fn
    _TRAINING_AVAILABLE = True
except ImportError as e:
    # Check if the error is specifically about JAX or Flax
    error_msg = str(e).lower()
    if 'jax' in error_msg or 'flax' in error_msg:
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
