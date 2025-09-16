"""Unit tests for model registry."""

import pytest
import flax.nnx as nnx
from aether.registry import ModelRegistry, get_registry, register_model
from aether.models import MiniGPT


def test_model_registry_registration():
    """Test model registration and retrieval."""
    registry = ModelRegistry()
    
    # Test registration
    registry.register_model(
        "test-model",
        MiniGPT,
        default_config={"embed_dim": 256}
    )
    
    assert "test-model" in registry.list_models()
    assert registry.get_model_class("test-model") == MiniGPT
    assert registry.get_default_config("test-model")["embed_dim"] == 256


def test_model_creation():
    """Test model creation through registry."""
    registry = ModelRegistry()
    
    registry.register_model(
        "test-minigpt",
        MiniGPT,
        default_config={
            "architecture": "linear",
            "maxlen": 128,
            "vocab_size": 1000,
            "embed_dim": 256,
            "num_heads": 4,
            "feed_forward_dim": 256,
            "num_transformer_blocks": 2,
            "dropout_rate": 0.1
        }
    )
    
    rngs = nnx.Rngs(42)
    model = registry.create_model(
        "test-minigpt",
        {},
        rngs
    )
    
    assert isinstance(model, MiniGPT)
    assert model.embed_dim == 256
    assert model.maxlen == 128


def test_global_registry():
    """Test global registry functions."""
    register_model(
        "global-test",
        MiniGPT,
        default_config={"embed_dim": 128}
    )
    
    registry = get_registry()
    assert "global-test" in registry.list_models()


if __name__ == "__main__":
    pytest.main([__file__])