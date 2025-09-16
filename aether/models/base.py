"""Base model class for all Aether models."""

import abc
from typing import Any, Dict, Optional
import jax.numpy as jnp
try:
    import flax.nnx as nnx
except ImportError:
    nnx = None


class BaseModel(nnx.Module, abc.ABC):
    """Abstract base class for all Aether models."""
    
    @abc.abstractmethod
    def __call__(self, inputs: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Forward pass of the model.
        
        Args:
            inputs: Input tensor of shape (batch_size, sequence_length)
            training: Whether the model is in training mode
            
        Returns:
            Output tensor of shape (batch_size, sequence_length, vocab_size)
        """
        pass
    
    @abc.abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration.
        
        Returns:
            Dictionary containing model configuration
        """
        pass
    
    @classmethod
    @abc.abstractmethod
    def from_config(cls, config: Dict[str, Any], rngs: Any) -> "BaseModel":
        """Create model from configuration.
        
        Args:
            config: Model configuration dictionary
            rngs: Random number generators
            
        Returns:
            Model instance
        """
        pass