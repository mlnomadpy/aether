#!/usr/bin/env python3
"""
Demo script showing MLM (Masked Language Modeling) training.

This script demonstrates how to use the new MLM functionality in Aether.
"""

import argparse
import sys
import os

# Add the current directory to the path to import aether
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aether import Config, Trainer
from setup_models import register_default_models


def create_mlm_config():
    """Create a configuration for MLM training."""
    config = Config()
    
    # Set up for MLM training
    config.training.training_mode = "mlm"
    config.training.mlm_mask_prob = 0.15
    config.training.mlm_replace_prob = 0.8
    config.training.mlm_random_prob = 0.1
    config.training.final_evaluation = True
    
    # Use smaller settings for demo
    config.training.batch_size = 8
    config.training.max_tokens_to_process = 100_000  # Much smaller for demo
    config.training.eval_interval = 100
    config.training.eval_steps = 10
    config.training.checkpoint_interval = 500
    config.training.val_set_size = 1000
    
    # Use smaller model for demo
    config.model.embed_dim = 256
    config.model.num_heads = 4
    config.model.feed_forward_dim = 256
    config.model.num_transformer_blocks = 4
    config.model.maxlen = 128
    
    return config


def create_clm_config():
    """Create a configuration for CLM training for comparison."""
    config = create_mlm_config()
    config.training.training_mode = "clm"  # Override to CLM
    return config


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="MLM Training Demo")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["mlm", "clm"],
        default="mlm",
        help="Training mode: mlm (masked language modeling) or clm (causal language modeling)"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file (optional)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="minigpt-linear",
        help="Model name to train (default: minigpt-linear)"
    )
    
    args = parser.parse_args()
    
    # Register default models
    register_default_models()
    
    # Load or create configuration
    if args.config:
        config = Config.from_file(args.config)
    else:
        if args.mode == "mlm":
            config = create_mlm_config()
        else:
            config = create_clm_config()
    
    # Update model name if specified
    if args.model:
        config.model.name = args.model
    
    print(f"=== {args.mode.upper()} Training Demo ===")
    print(f"Training model: {config.model.name}")
    print(f"Training mode: {config.training.training_mode}")
    print(f"Architecture: {config.model.architecture}")
    
    if config.training.training_mode == "mlm":
        print(f"MLM mask probability: {config.training.mlm_mask_prob}")
        print(f"MLM replace probability: {config.training.mlm_replace_prob}")
        print(f"MLM random probability: {config.training.mlm_random_prob}")
    
    print(f"Final evaluation: {config.training.final_evaluation}")
    print(f"Configuration: {config.to_dict()}")
    print()
    
    # Create trainer
    trainer = Trainer(config)
    
    # Start training
    try:
        trainer.train()
        print(f"\n=== {args.mode.upper()} Training Demo Completed Successfully ===")
    except KeyboardInterrupt:
        print(f"\n{args.mode.upper()} training interrupted by user.")
    except Exception as e:
        print(f"{args.mode.upper()} training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()