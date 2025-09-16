"""Model registry for dynamic model creation and management."""

from typing import Dict, Any, Type, Callable, Optional
try:
    import flax.nnx as nnx
except ImportError:
    nnx = None
    print("Warning: flax.nnx not available. Some functionality may be limited.")

from ..models.base import BaseModel


class ModelRegistry:
    """Registry for managing different model architectures."""
    
    def __init__(self):
        self._models: Dict[str, Type[BaseModel]] = {}
        self._factories: Dict[str, Callable] = {}
        self._configs: Dict[str, Dict[str, Any]] = {}
    
    def register_model(
        self, 
        name: str, 
        model_class: Type[BaseModel], 
        factory_fn: Optional[Callable] = None,
        default_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a model class.
        
        Args:
            name: Model name identifier
            model_class: Model class that inherits from BaseModel
            factory_fn: Optional factory function for custom initialization
            default_config: Default configuration for the model
        """
        if not issubclass(model_class, BaseModel):
            raise ValueError(f"Model class {model_class} must inherit from BaseModel")
        
        self._models[name] = model_class
        if factory_fn is not None:
            self._factories[name] = factory_fn
        if default_config is not None:
            self._configs[name] = default_config
    
    def create_model(
        self, 
        name: str, 
        config: Dict[str, Any], 
        rngs: Any,  # nnx.Rngs
        **kwargs
    ) -> BaseModel:
        """Create a model instance.
        
        Args:
            name: Registered model name
            config: Model configuration
            rngs: Random number generators
            **kwargs: Additional arguments passed to model creation
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model name is not registered
        """
        if name not in self._models:
            raise ValueError(f"Model '{name}' not found. Available models: {list(self._models.keys())}")
        
        # Merge default config with provided config
        final_config = {}
        if name in self._configs:
            final_config.update(self._configs[name])
        final_config.update(config)
        
        # Use factory function if available, otherwise use from_config
        if name in self._factories:
            return self._factories[name](final_config, rngs, **kwargs)
        else:
            return self._models[name].from_config(final_config, rngs, **kwargs)
    
    def get_model_class(self, name: str) -> Type[BaseModel]:
        """Get model class by name.
        
        Args:
            name: Registered model name
            
        Returns:
            Model class
            
        Raises:
            ValueError: If model name is not registered
        """
        if name not in self._models:
            raise ValueError(f"Model '{name}' not found. Available models: {list(self._models.keys())}")
        return self._models[name]
    
    def get_default_config(self, name: str) -> Dict[str, Any]:
        """Get default configuration for a model.
        
        Args:
            name: Registered model name
            
        Returns:
            Default configuration dictionary
            
        Raises:
            ValueError: If model name is not registered
        """
        if name not in self._models:
            raise ValueError(f"Model '{name}' not found. Available models: {list(self._models.keys())}")
        return self._configs.get(name, {}).copy()
    
    def list_models(self) -> list[str]:
        """List all registered model names.
        
        Returns:
            List of registered model names
        """
        return list(self._models.keys())
    
    def unregister_model(self, name: str) -> None:
        """Unregister a model.
        
        Args:
            name: Model name to unregister
            
        Raises:
            ValueError: If model name is not registered
        """
        if name not in self._models:
            raise ValueError(f"Model '{name}' not found")
        
        del self._models[name]
        if name in self._factories:
            del self._factories[name]
        if name in self._configs:
            del self._configs[name]


# Global model registry instance
_global_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    """Get the global model registry instance.
    
    Returns:
        Global ModelRegistry instance
    """
    return _global_registry


def register_model(*args, **kwargs) -> None:
    """Register a model in the global registry.
    
    Args:
        *args, **kwargs: Arguments passed to ModelRegistry.register_model
    """
    _global_registry.register_model(*args, **kwargs)


def create_model(*args, **kwargs) -> BaseModel:
    """Create a model using the global registry.
    
    Args:
        *args, **kwargs: Arguments passed to ModelRegistry.create_model
        
    Returns:
        Model instance
    """
    return _global_registry.create_model(*args, **kwargs)