"""Main training orchestration."""

import os
import time
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import wandb
import orbax.checkpoint as orbax
from typing import Dict, Any, Optional

from .steps import train_step, eval_step
from .compat import create_optimizer
from ..config import Config
from ..registry import create_model
from ..data import create_data_iterators, create_validation_iterator, create_training_iterator, prepare_batch
from ..utils import setup_mesh, get_tokenizer


class Trainer:
    """Main trainer class for managing the training process."""

    def __init__(self, config: Config):
        """Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.mesh = setup_mesh(device_config=config.device)
        self.tokenizer = get_tokenizer(config.data.tokenizer_name)

        # Set up precision policy
        self._setup_precision()

        # Setup logging
        self._setup_logging()

        # Initialize model
        self.rngs = nnx.Rngs(0)
        self.model = self._create_model()

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Setup checkpointing
        self.checkpointer = orbax.PyTreeCheckpointer(config.logging.checkpoint_dir)

        # Setup data
        self.train_iterator, self.val_iterator = self._create_data_iterators()

    def _setup_logging(self) -> None:
        """Setup logging and experiment tracking."""
        wandb.init(project=self.config.logging.wandb_project, config=self.config.to_dict())

    def _setup_precision(self) -> None:
        """Setup precision policy for training."""
        precision = self.config.training.precision.lower()
        
        if precision == "bfloat16":
            # Configure JAX to use bfloat16 for computations
            from jax import config as jax_config
            jax_config.update('jax_default_matmul_precision', 'bfloat16')
            
            # Set the default dtype for parameters
            self._param_dtype = jnp.bfloat16
            self._compute_dtype = jnp.bfloat16
            print("Using BFloat16 precision for training")
        elif precision == "float16":
            # Configure JAX to use float16 for computations
            from jax import config as jax_config
            jax_config.update('jax_default_matmul_precision', 'float16')
            
            # Set the default dtype for parameters
            self._param_dtype = jnp.float16
            self._compute_dtype = jnp.float16
            print("Using Float16 precision for training")
        elif precision == "float32":
            self._param_dtype = jnp.float32
            self._compute_dtype = jnp.float32
            print("Using Float32 precision for training")
        elif precision == "float64":
            # Enable 64-bit precision in JAX (disabled by default)
            from jax import config as jax_config
            jax_config.update("jax_enable_x64", True)
            
            # Set the default dtype for parameters
            self._param_dtype = jnp.float64
            self._compute_dtype = jnp.float64
            print("Using Float64 precision for training")
        else:
            raise ValueError(
                f"Unsupported precision: {precision}. "
                f"Options are 'float16', 'bfloat16', 'float32', or 'float64'"
            )

    def _create_model(self) -> nnx.Module:
        """Create model from configuration."""
        model_config = self.config.get_model_config_dict()
        return create_model(
            self.config.model.name, 
            model_config, 
            self.rngs, 
            mesh=self.mesh,
            param_dtype=self._param_dtype,
            compute_dtype=self._compute_dtype
        )

    def _create_learning_rate_schedule(self, base_lr: float, total_steps: int, 
                                     scheduler: str, alpha: float, 
                                     warmup_steps: Optional[int] = None):
        """Create learning rate schedule from configuration."""
        scheduler = scheduler.lower()
        
        if scheduler == "constant":
            return base_lr
        elif scheduler == "linear":
            return optax.linear_schedule(
                init_value=base_lr,
                end_value=alpha * base_lr,
                transition_steps=total_steps
            )
        elif scheduler == "cosine":
            return optax.cosine_decay_schedule(
                init_value=base_lr,
                decay_steps=total_steps,
                alpha=alpha
            )
        elif scheduler == "warmup_cosine":
            if warmup_steps is None:
                # Default to 10% of total steps for warmup
                warmup_steps = max(1, total_steps // 10)
            return optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=base_lr,
                warmup_steps=warmup_steps,
                decay_steps=total_steps - warmup_steps,
                end_value=alpha * base_lr
            )
        else:
            raise ValueError(f"Unsupported learning rate scheduler: {scheduler}")

    def _create_optimizer(self) -> nnx.Optimizer:
        """Create optimizer from configuration."""
        # Calculate total training steps for schedulers
        tokens_per_iteration = self.config.training.batch_size * self.config.model.maxlen
        total_steps = self.config.training.max_tokens_to_process // tokens_per_iteration
        
        # Create learning rate schedule
        learning_rate = self._create_learning_rate_schedule(
            base_lr=self.config.training.learning_rate,
            total_steps=total_steps,
            scheduler=self.config.training.lr_scheduler,
            alpha=self.config.training.lr_scheduler_alpha,
            warmup_steps=self.config.training.lr_scheduler_warmup_steps
        )
        
        # Create optimizer
        optimizer_name = self.config.training.optimizer.lower()
        
        if optimizer_name == "novograd":
            optimizer_fn = optax.novograd(learning_rate)
        elif optimizer_name == "adam":
            optimizer_fn = optax.adam(learning_rate)
        elif optimizer_name == "adamw":
            optimizer_fn = optax.adamw(
                learning_rate=learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif optimizer_name == "sgd":
            optimizer_fn = optax.sgd(
                learning_rate=learning_rate,
                momentum=self.config.training.momentum
            )
        elif optimizer_name == "rmsprop":
            optimizer_fn = optax.rmsprop(learning_rate)
        elif optimizer_name == "lion":
            optimizer_fn = optax.lion(learning_rate)
        elif optimizer_name == "adagrad":
            optimizer_fn = optax.adagrad(learning_rate)
        elif optimizer_name == "adadelta":
            optimizer_fn = optax.adadelta(learning_rate)
        elif optimizer_name == "adamax":
            optimizer_fn = optax.adamax(learning_rate)
        elif optimizer_name == "nadam":
            optimizer_fn = optax.nadam(learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.training.optimizer}")

        return create_optimizer(self.model, optimizer_fn, wrt=nnx.Param)

    def _create_data_iterators(self):
        """Create data iterators."""
        return create_data_iterators(
            dataset_name=self.config.data.dataset_name,
            split=self.config.data.split,
            streaming=self.config.data.streaming,
            maxlen=self.config.model.maxlen,
            tokenizer=self.tokenizer,
            val_set_size=self.config.training.val_set_size,
            batch_size=self.config.training.batch_size,
        )

    def _create_validation_iterator(self):
        """Create only the validation iterator."""
        return create_validation_iterator(
            dataset_name=self.config.data.dataset_name,
            split=self.config.data.split,
            streaming=self.config.data.streaming,
            maxlen=self.config.model.maxlen,
            tokenizer=self.tokenizer,
            val_set_size=self.config.training.val_set_size,
            batch_size=self.config.training.batch_size,
        )

    def _create_training_iterator(self):
        """Create only the training iterator."""
        return create_training_iterator(
            dataset_name=self.config.data.dataset_name,
            split=self.config.data.split,
            streaming=self.config.data.streaming,
            maxlen=self.config.model.maxlen,
            tokenizer=self.tokenizer,
            val_set_size=self.config.training.val_set_size,
            batch_size=self.config.training.batch_size,
        )

    def _reset_validation_iterator(self):
        """Reset only the validation iterator."""
        self.val_iterator = self._create_validation_iterator()

    def _reset_training_iterator(self):
        """Reset only the training iterator."""
        self.train_iterator = self._create_training_iterator()

    def train(self) -> None:
        """Main training loop."""
        # Calculate training parameters
        tokens_per_iteration = self.config.training.batch_size * self.config.model.maxlen
        max_iterations = self.config.training.max_tokens_to_process // tokens_per_iteration

        print(f"Starting training for {max_iterations} iterations...")
        print(f"Total tokens to process: {self.config.training.max_tokens_to_process:,}")

        start_time = time.time()

        for step in range(max_iterations):
            # Training step
            try:
                batch = next(self.train_iterator)
                sharded_batch = prepare_batch(batch, self.mesh)

                loss, self.model, self.optimizer = train_step(
                    self.model, self.optimizer, sharded_batch
                )

                wandb.log({"train_loss": loss.item()}, step=step)

            except StopIteration:
                print("Restarting training iterator.")
                self._reset_training_iterator()
                continue

            # Evaluation
            if (step + 1) % self.config.training.eval_interval == 0:
                self._evaluate(step)

            # Checkpointing
            if (step + 1) % self.config.training.checkpoint_interval == 0:
                self._save_checkpoint(step + 1)

            # Progress logging
            if (step + 1) % 100 == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = (step + 1) * tokens_per_iteration / elapsed
                print(
                    f"Step {step + 1}/{max_iterations}, "
                    f"Loss: {loss.item():.4f}, "
                    f"Tokens/sec: {tokens_per_sec:.0f}"
                )

        print("Training completed!")
        wandb.finish()

    def _evaluate(self, step: int) -> None:
        """Run evaluation."""
        val_losses = []
        batches_processed = 0
        max_eval_steps = self.config.training.eval_steps

        # Continue until we either reach eval_steps limit or exhaust the validation dataset
        while batches_processed < max_eval_steps:
            try:
                val_batch = next(self.val_iterator)
                sharded_val_batch = prepare_batch(val_batch, self.mesh)

                val_loss = eval_step(self.model, sharded_val_batch)
                val_losses.append(val_loss)
                batches_processed += 1

            except StopIteration:
                # We've completed a full pass through the validation dataset
                print(
                    f"Completed validation pass with {batches_processed} batches. Resetting validation iterator."
                )
                self._reset_validation_iterator()
                break

        if val_losses:
            avg_val_loss = jnp.mean(jnp.array(val_losses))
            wandb.log(
                {"val_loss": avg_val_loss.item(), "val_batches": batches_processed}, step=step
            )
            print(
                f"Validation loss at step {step + 1}: {avg_val_loss.item():.4f} "
                f"(computed over {batches_processed} batches)"
            )

    def _save_checkpoint(self, step: int) -> None:
        """Save model checkpoint."""
        # Extract model parameters
        _, params_state, _ = nnx.split(self.model, nnx.Param, ...)

        # Save checkpoint with absolute path (required by Orbax)
        checkpoint_path = os.path.join(self.config.logging.checkpoint_dir, f"step_{step}")
        checkpoint_path = os.path.abspath(checkpoint_path)
        self.checkpointer.save(checkpoint_path, params_state)
        print(f"Checkpoint saved at step {step}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
        """
        # Ensure absolute path for consistency with save_checkpoint
        checkpoint_path = os.path.abspath(checkpoint_path)
        
        # Load parameters
        params_state = self.checkpointer.restore(checkpoint_path)

        # Merge parameters back into model
        graph_def, _, other_state = nnx.split(self.model, nnx.Param, ...)
        self.model = nnx.merge(graph_def, params_state, other_state)

        print(f"Loaded checkpoint from {checkpoint_path}")
