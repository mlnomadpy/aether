#!/usr/bin/env python3
"""
Unified training script for Aether models.

This script replaces the original linear_main.py and yat_main.py with a unified,
configurable training interface.
"""

import argparse
import sys
import os

# Add the current directory to the path to import aether
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aether import Config, Trainer
from setup_models import register_default_models


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Aether transformer models")
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file (JSON or YAML)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="minigpt-linear",
        help="Model name to train (default: minigpt-linear)"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    # Register default models
    register_default_models()
    
    # Load configuration
    if args.config:
        config = Config.from_file(args.config)
    else:
        # Use default configuration
        config = Config()
        config.model.name = args.model
    
    # Update model name if specified
    if args.model:
        config.model.name = args.model
    
    print(f"Training model: {config.model.name}")
    print(f"Architecture: {config.model.architecture}")
    print(f"Configuration: {config.to_dict()}")
    
    # Create trainer
    trainer = Trainer(config)
    
    # Load checkpoint if specified
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()