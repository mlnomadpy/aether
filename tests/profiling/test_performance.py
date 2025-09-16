"""Profiling tests for model performance."""

import time
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from aether.models import MiniGPT
from aether.training.steps import loss_fn, train_step, eval_step


def profile_model_forward_pass():
    """Profile model forward pass performance."""
    print("=== Model Forward Pass Profiling ===")
    
    # Create model
    rngs = nnx.Rngs(42)
    model = MiniGPT(
        maxlen=1024,
        vocab_size=50257,
        embed_dim=768,
        num_heads=12,
        feed_forward_dim=768,
        num_transformer_blocks=12,
        rngs=rngs,
        architecture="linear"
    )
    
    # Create test input
    batch_size = 8
    seq_len = 1024
    inputs = jax.random.randint(rngs.default(), (batch_size, seq_len), 0, 1000)
    
    # Warm up
    print("Warming up...")
    for _ in range(5):
        _ = model(inputs, training=False)
    
    # Profile forward pass
    print("Profiling forward pass...")
    times = []
    num_runs = 20
    
    for i in range(num_runs):
        start_time = time.time()
        outputs = model(inputs, training=False)
        outputs.block_until_ready()  # Ensure computation is complete
        end_time = time.time()
        times.append(end_time - start_time)
        
        if i % 5 == 0:
            print(f"  Run {i+1}/{num_runs}: {times[-1]:.4f}s")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\nForward Pass Results:")
    print(f"  Average time: {avg_time:.4f}s")
    print(f"  Min time: {min_time:.4f}s")
    print(f"  Max time: {max_time:.4f}s")
    print(f"  Tokens/second: {(batch_size * seq_len) / avg_time:.0f}")
    print(f"  Output shape: {outputs.shape}")


def profile_training_step():
    """Profile training step performance."""
    print("\n=== Training Step Profiling ===")
    
    # Create model and optimizer
    rngs = nnx.Rngs(42)
    model = MiniGPT(
        maxlen=512,  # Smaller for faster training
        vocab_size=50257,
        embed_dim=768,
        num_heads=12,
        feed_forward_dim=768,
        num_transformer_blocks=6,  # Fewer layers for faster training
        rngs=rngs,
        architecture="linear"
    )
    
    import optax
    optimizer = nnx.Optimizer(model, optax.adam(0.001))
    
    # Create test batch
    batch_size = 4
    seq_len = 512
    batch = jax.random.randint(rngs.default(), (batch_size, seq_len), 0, 1000)
    
    # Compile training step
    print("Compiling training step...")
    compiled_train_step = nnx.jit(train_step)
    
    # Warm up
    print("Warming up...")
    for _ in range(3):
        loss, model, optimizer = compiled_train_step(model, optimizer, batch)
        loss.block_until_ready()
    
    # Profile training step
    print("Profiling training step...")
    times = []
    num_runs = 10
    
    for i in range(num_runs):
        start_time = time.time()
        loss, model, optimizer = compiled_train_step(model, optimizer, batch)
        loss.block_until_ready()
        end_time = time.time()
        times.append(end_time - start_time)
        
        print(f"  Run {i+1}/{num_runs}: {times[-1]:.4f}s, Loss: {loss.item():.4f}")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\nTraining Step Results:")
    print(f"  Average time: {avg_time:.4f}s")
    print(f"  Min time: {min_time:.4f}s")
    print(f"  Max time: {max_time:.4f}s")
    print(f"  Tokens/second: {(batch_size * seq_len) / avg_time:.0f}")


def profile_memory_usage():
    """Profile memory usage of the model."""
    print("\n=== Memory Usage Profiling ===")
    
    try:
        import psutil
        import os
        
        def get_memory_usage():
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB
        
        initial_memory = get_memory_usage()
        print(f"Initial memory usage: {initial_memory:.1f} MB")
        
        # Create small model
        rngs = nnx.Rngs(42)
        model = MiniGPT(
            maxlen=256,
            vocab_size=10000,
            embed_dim=512,
            num_heads=8,
            feed_forward_dim=512,
            num_transformer_blocks=4,
            rngs=rngs,
            architecture="linear"
        )
        
        after_model_memory = get_memory_usage()
        print(f"After model creation: {after_model_memory:.1f} MB")
        print(f"Model memory usage: {after_model_memory - initial_memory:.1f} MB")
        
        # Create batch and run forward pass
        batch = jax.random.randint(rngs.default(), (8, 256), 0, 1000)
        outputs = model(batch, training=False)
        
        after_forward_memory = get_memory_usage()
        print(f"After forward pass: {after_forward_memory:.1f} MB")
        print(f"Forward pass memory overhead: {after_forward_memory - after_model_memory:.1f} MB")
        
    except ImportError:
        print("psutil not available, skipping memory profiling")


def main():
    """Run all profiling tests."""
    print("Aether Model Profiling")
    print("======================")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Available devices: {len(jax.devices())}")
    print(f"Device types: {[d.device_kind for d in jax.devices()]}")
    print()
    
    try:
        profile_model_forward_pass()
        profile_training_step()
        profile_memory_usage()
        
        print("\n=== Profiling Complete ===")
        print("All profiling tests completed successfully!")
        
    except Exception as e:
        print(f"Profiling failed with error: {e}")
        raise


if __name__ == "__main__":
    main()