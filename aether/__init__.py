"""Aether: Modular JAX/Flax Transformer Training Framework."""

from .config import Config, ModelConfig, TrainingConfig, DataConfig, LoggingConfig

# Try to import training components, but make them optional
try:
    from .training import Trainer, train_step, eval_step, loss_fn
    _TRAINING_AVAILABLE = True
except ImportError:
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
