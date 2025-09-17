"""
Learning Rate Scheduler Visualization Demo

This script demonstrates how to use the new learning rate schedulers in Aether,
following the pattern shown in the problem statement.
"""

import sys
import os
sys.path.insert(0, '/home/runner/work/aether/aether')

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import optax
from aether.config import Config


def visualize_learning_rate_schedules():
    """Visualize different learning rate schedules."""
    
    # Training configuration
    learning_rate = 0.002
    num_epochs = 10
    batch_size = 32
    maxlen = 1024
    max_tokens = 100_000_000
    
    # Calculate total steps
    tokens_per_iteration = batch_size * maxlen
    total_steps = max_tokens // tokens_per_iteration
    
    print(f"Total training steps: {total_steps}")
    print(f"Tokens per iteration: {tokens_per_iteration}")
    
    # Create different schedules
    schedules = {
        'Constant': learning_rate,
        'Linear': optax.linear_schedule(learning_rate, 0.0, total_steps),
        'Cosine': optax.cosine_decay_schedule(learning_rate, total_steps, alpha=0.1),
        'Warmup Cosine': optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=total_steps // 10,
            decay_steps=total_steps - (total_steps // 10),
            end_value=0.1 * learning_rate
        )
    }
    
    # Create sample points for visualization
    iterate_subsample = np.linspace(0, total_steps, 200).astype(int)
    step_to_epoch = lambda step: step * tokens_per_iteration / (max_tokens / num_epochs)
    
    # Plot all schedules
    plt.figure(figsize=(12, 8))
    
    for name, schedule in schedules.items():
        if callable(schedule):
            lr_values = [schedule(i) for i in iterate_subsample]
        else:
            lr_values = [schedule] * len(iterate_subsample)
        
        epoch_values = [step_to_epoch(step) for step in iterate_subsample]
        plt.plot(epoch_values, lr_values, lw=3, label=name)
    
    plt.title("Learning Rate Schedules Comparison", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Learning Rate", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim((0, num_epochs))
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('/tmp/learning_rate_schedules.png', dpi=150, bbox_inches='tight')
    print("✓ Learning rate schedule visualization saved to /tmp/learning_rate_schedules.png")
    
    return schedules


def demo_optimizer_creation():
    """Demonstrate how different optimizers can be created with schedules."""
    
    learning_rate = 0.002
    total_steps = 10000
    
    # Example: Creating optimizer with cosine schedule (as in problem statement)
    lr_schedule = optax.cosine_decay_schedule(learning_rate, total_steps, alpha=0.1)
    momentum = 0.9
    
    # This is equivalent to what the new trainer does internally
    optimizer_configs = [
        ("AdamW with Cosine", optax.adamw(lr_schedule, weight_decay=0.01)),
        ("SGD with Cosine", optax.sgd(lr_schedule, momentum=momentum)),
        ("Adam with Cosine", optax.adam(lr_schedule)),
        ("Novograd with Cosine", optax.novograd(lr_schedule)),
    ]
    
    print("\n=== Optimizer Creation Demo ===")
    for name, optimizer in optimizer_configs:
        print(f"✓ {name}: {type(optimizer).__name__}")
        
    # Show what the schedule looks like at different steps
    print(f"\nLearning rate at different steps:")
    test_steps = [0, total_steps//4, total_steps//2, 3*total_steps//4, total_steps]
    for step in test_steps:
        lr = lr_schedule(step)
        print(f"  Step {step:5d}: {lr:.6f}")


def demo_config_usage():
    """Demonstrate how to use the new configuration options."""
    
    print("\n=== Configuration Demo ===")
    
    # Example config with cosine scheduler and AdamW
    config_dict = {
        "model": {
            "name": "minigpt-linear",
            "architecture": "linear",
            "maxlen": 1024,
            "vocab_size": 50257,
            "embed_dim": 768,
            "num_heads": 12,
            "feed_forward_dim": 768,
            "num_transformer_blocks": 12,
            "dropout_rate": 0.1
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 0.002,
            "max_tokens_to_process": 1000000000,
            "eval_interval": 2000,
            "eval_steps": 1000,
            "val_set_size": 20000,
            "checkpoint_interval": 10000,
            "optimizer": "adamw",
            "lr_scheduler": "cosine",
            "lr_scheduler_alpha": 0.1,
            "lr_scheduler_warmup_steps": None,
            "momentum": 0.9,
            "weight_decay": 0.01
        },
        "data": {
            "dataset_name": "HuggingFaceFW/fineweb",
            "split": "train",
            "streaming": True,
            "tokenizer_name": "gpt2"
        },
        "logging": {
            "wandb_project": "aether-cosine-demo",
            "checkpoint_dir": "./checkpoints",
            "log_level": "INFO"
        }
    }
    
    config = Config.from_dict(config_dict)
    
    print(f"Optimizer: {config.training.optimizer}")
    print(f"LR Scheduler: {config.training.lr_scheduler}")
    print(f"Base Learning Rate: {config.training.learning_rate}")
    print(f"LR Alpha (min factor): {config.training.lr_scheduler_alpha}")
    print(f"Weight Decay: {config.training.weight_decay}")
    print(f"Momentum: {config.training.momentum}")


def main():
    """Run the complete demo."""
    print("=== Aether Learning Rate Scheduler & Optimizer Demo ===\n")
    
    # Visualize schedules
    schedules = visualize_learning_rate_schedules()
    
    # Demo optimizer creation
    demo_optimizer_creation()
    
    # Demo configuration usage
    demo_config_usage()
    
    print("\n✓ Demo completed successfully!")
    print("\nNew Features Added:")
    print("  • Support for cosine and warmup-cosine learning rate schedules")
    print("  • Extended optimizer support (SGD, RMSprop, Lion, AdaGrad, etc.)")
    print("  • Configurable momentum and weight decay")
    print("  • Learning rate schedule visualization capabilities")
    print("  • Example configurations for different setups")


if __name__ == "__main__":
    main()