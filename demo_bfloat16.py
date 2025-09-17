#!/usr/bin/env python3
"""
Demo script showing how to use BFloat16 precision in Aether.

This script demonstrates:
1. Creating a config with BFloat16 precision
2. Loading a model with BFloat16 precision
3. Running inference with BFloat16
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax.numpy as jnp
import jax
import flax.nnx as nnx
from aether import Config, ModelConfig, TrainingConfig
from aether.models.minigpt import MiniGPT

def demo_bfloat16_config():
    """Demonstrate creating and using BFloat16 configuration."""
    print("=== BFloat16 Configuration Demo ===")
    
    # Create configuration with BFloat16 precision
    config = Config()
    config.training.precision = "bfloat16"
    config.model.embed_dim = 384  # Divisible by 12 heads
    config.model.num_transformer_blocks = 4
    config.model.maxlen = 128
    
    print(f"Training precision: {config.training.precision}")
    print(f"Model configuration: {config.model.__dict__}")
    
    # Save config to file
    config.save("bfloat16_demo_config.json")
    print("✓ Saved BFloat16 configuration to bfloat16_demo_config.json")
    
    # Load config from file
    loaded_config = Config.from_file("bfloat16_demo_config.json")
    print(f"✓ Loaded configuration with precision: {loaded_config.training.precision}")
    
    return config

def demo_bfloat16_model(config):
    """Demonstrate creating and using a BFloat16 model."""
    print("\n=== BFloat16 Model Demo ===")
    
    # Determine dtypes based on precision
    if config.training.precision == "bfloat16":
        param_dtype = jnp.bfloat16
        compute_dtype = jnp.bfloat16
    else:
        param_dtype = jnp.float32
        compute_dtype = jnp.float32
    
    # Create model with BFloat16 precision
    rngs = nnx.Rngs(42)
    model = MiniGPT(
        maxlen=config.model.maxlen,
        vocab_size=config.model.vocab_size,
        embed_dim=config.model.embed_dim,
        num_heads=config.model.num_heads,
        feed_forward_dim=config.model.feed_forward_dim,
        num_transformer_blocks=config.model.num_transformer_blocks,
        rngs=rngs,
        architecture=config.model.architecture,
        param_dtype=param_dtype,
        compute_dtype=compute_dtype
    )
    
    print(f"Model created with param_dtype: {param_dtype}")
    print(f"Model output layer kernel dtype: {model.output_layer.kernel.dtype}")
    print(f"Model embedding dtype: {model.embedding_layer.token_emb.embedding.dtype}")
    
    return model

def demo_bfloat16_inference(model):
    """Demonstrate running inference with BFloat16."""
    print("\n=== BFloat16 Inference Demo ===")
    
    # Create sample input
    batch_size = 2
    seq_len = 64
    vocab_size = 1000
    
    # Generate random token IDs
    inputs = jnp.array([[i % vocab_size for i in range(j, j + seq_len)] 
                       for j in range(batch_size)], dtype=jnp.int32)
    
    print(f"Input shape: {inputs.shape}, dtype: {inputs.dtype}")
    
    # Run inference
    outputs = model(inputs, training=False)
    
    print(f"Output shape: {outputs.shape}, dtype: {outputs.dtype}")
    print(f"Output stats - min: {float(outputs.min()):.4f}, max: {float(outputs.max()):.4f}, mean: {float(outputs.mean()):.4f}")
    
    # Compare model sizes (parameters)
    total_params = sum(param.size for param in jax.tree.leaves(nnx.state(model, nnx.Param)))
    param_dtype_size = 2 if outputs.dtype == jnp.bfloat16 else 4  # bytes per parameter
    model_size_mb = (total_params * param_dtype_size) / (1024 * 1024)
    
    print(f"Model has {total_params:,} parameters")
    print(f"Model size: {model_size_mb:.2f} MB ({outputs.dtype})")
    
    return outputs

def memory_comparison():
    """Compare memory usage between float32 and bfloat16."""
    print("\n=== Memory Usage Comparison ===")
    
    config = Config()
    config.model.embed_dim = 768  # Divisible by 12 heads
    config.model.num_transformer_blocks = 6
    config.model.maxlen = 1024
    
    # Test both precisions
    for precision in ["float32", "bfloat16"]:
        config.training.precision = precision
        param_dtype = jnp.bfloat16 if precision == "bfloat16" else jnp.float32
        
        rngs = nnx.Rngs(42)
        model = MiniGPT(
            maxlen=config.model.maxlen,
            vocab_size=config.model.vocab_size,
            embed_dim=config.model.embed_dim,
            num_heads=config.model.num_heads,
            feed_forward_dim=config.model.feed_forward_dim,
            num_transformer_blocks=config.model.num_transformer_blocks,
            rngs=rngs,
            architecture=config.model.architecture,
            param_dtype=param_dtype,
            compute_dtype=param_dtype
        )
        
        # Calculate model size
        total_params = sum(param.size for param in jax.tree.leaves(nnx.state(model, nnx.Param)))
        param_dtype_size = 2 if precision == "bfloat16" else 4
        model_size_mb = (total_params * param_dtype_size) / (1024 * 1024)
        
        print(f"{precision:8}: {total_params:,} params, {model_size_mb:.2f} MB")
    
    print(f"Memory savings with BFloat16: ~50%")

def main():
    """Main demo function."""
    print("BFloat16 Training Support Demo for Aether")
    print("=" * 50)
    
    # Demo configuration
    config = demo_bfloat16_config()
    
    # Demo model creation
    model = demo_bfloat16_model(config)
    
    # Demo inference
    outputs = demo_bfloat16_inference(model)
    
    # Demo memory comparison
    memory_comparison()
    
    print("\n" + "=" * 50)
    print("✓ BFloat16 demo completed successfully!")
    print("\nTo use BFloat16 in your training:")
    print("1. Set config.training.precision = 'bfloat16'")
    print("2. Or use --config configs/bfloat16_example.json")
    print("3. Or add 'precision': 'bfloat16' to your JSON/YAML config")

if __name__ == "__main__":
    main()