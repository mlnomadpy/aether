"""Setup and register default models."""

from aether.registry import register_model
from aether.models import MiniGPT


def register_default_models():
    """Register the default models in the registry."""
    
    # Linear architecture MiniGPT
    register_model(
        name="minigpt-linear",
        model_class=MiniGPT,
        default_config={
            "architecture": "linear",
            "maxlen": 1024,
            "vocab_size": 50257,
            "embed_dim": 768,
            "num_heads": 12,
            "feed_forward_dim": 768,
            "num_transformer_blocks": 12,
            "dropout_rate": 0.1
        }
    )
    
    # YAT architecture MiniGPT
    register_model(
        name="minigpt-yat",
        model_class=MiniGPT,
        default_config={
            "architecture": "yat",
            "maxlen": 1024,
            "vocab_size": 50257,
            "embed_dim": 768,
            "num_heads": 12,
            "feed_forward_dim": 768,
            "num_transformer_blocks": 12,
            "dropout_rate": 0.1
        }
    )


if __name__ == "__main__":
    register_default_models()
    from aether.registry import get_registry
    registry = get_registry()
    print("Registered models:", registry.list_models())