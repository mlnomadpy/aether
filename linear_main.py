# Install necessary libraries
# The following command ensures JAX is installed correctly for a TPU environment,
# which is critical to avoid XlaRuntimeError. It also installs/upgrades other dependencies.
!pip install -Uq tiktoken datasets wandb

import os
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
import flax.nnx as nnx
import optax
from dataclasses import dataclass
import pandas as pd
import tiktoken
import time
from datasets import load_dataset
import wandb
import orbax.checkpoint as orbax

# --- JAX Device and Mesh Setup ---
# Initialize JAX devices
if jax.default_backend() == 'tpu':
    # For 4-way data parallel and 2-way tensor parallel on TPU v2/v3
    mesh = Mesh(mesh_utils.create_device_mesh((4, 2)), ('batch', 'model'))
else:
    # Fallback for GPUs or other setups
    # This will use 8-way data parallelism if 8 devices are available.
    # Adjust the mesh shape according to your hardware.
    num_devices = len(jax.devices())
    mesh_shape = (num_devices, 1)
    mesh = Mesh(mesh_utils.create_device_mesh(mesh_shape), ('batch', 'model'))

# --- Tokenizer ---
tokenizer = tiktoken.get_encoding("gpt2")

# --- Causal Attention Mask ---
def causal_attention_mask(seq_len):
    """Creates a causal attention mask."""
    return jnp.tril(jnp.ones((seq_len, seq_len)))

# --- Model Definition ---
class TransformerBlock(nnx.Module):
    """A single Transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, *, rngs: nnx.Rngs, rate: float = 0.1):
        self.mha = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), NamedSharding(mesh, P(None, 'model'))),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model'))),
            rngs=rngs
        )
        self.dropout1 = nnx.Dropout(rate=rate, rngs=rngs)
        self.layer_norm1 = nnx.LayerNorm(
            epsilon=1e-6,
            num_features=embed_dim,
            scale_init=nnx.with_partitioning(nnx.initializers.ones_init(), NamedSharding(mesh, P('model'))),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model'))),
            rngs=rngs
        )
        self.linear1 = nnx.Linear(
            in_features=embed_dim,
            out_features=ff_dim,
            kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), NamedSharding(mesh, P(None, 'model'))),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model'))),
            rngs=rngs
        )
        self.linear2 = nnx.Linear(
            in_features=ff_dim,
            out_features=embed_dim,
            kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), NamedSharding(mesh, P(None, 'model'))),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model'))),
            rngs=rngs
        )
        self.dropout2 = nnx.Dropout(rate=rate, rngs=rngs)
        self.layer_norm2 = nnx.LayerNorm(
            epsilon=1e-6,
            num_features=embed_dim,
            scale_init=nnx.with_partitioning(nnx.initializers.ones_init(), NamedSharding(mesh, P(None, 'model'))),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model'))),
            rngs=rngs
        )

    def __call__(self, inputs, training: bool = False):
        seq_len = inputs.shape[1]
        mask = causal_attention_mask(seq_len)
        attention_output = self.mha(inputs_q=inputs, mask=mask, decode=False)
        attention_output = self.dropout1(attention_output, deterministic=not training)
        out1 = self.layer_norm1(inputs + attention_output)
        ffn_output = self.linear1(out1)
        ffn_output = nnx.relu(ffn_output)
        ffn_output = self.linear2(ffn_output)
        ffn_output = self.dropout2(ffn_output, deterministic=not training)
        return self.layer_norm2(out1 + ffn_output)

class TokenAndPositionEmbedding(nnx.Module):
    """Combines token and positional embeddings."""
    def __init__(self, maxlen: int, vocab_size: int, embed_dim: int, *, rngs: nnx.Rngs):
        self.token_emb = nnx.Embed(num_embeddings=vocab_size, features=embed_dim, rngs=rngs)
        self.pos_emb = nnx.Embed(num_embeddings=maxlen, features=embed_dim, rngs=rngs)

    def __call__(self, x):
        positions = jnp.arange(0, x.shape[1])[None, :]
        position_embedding = self.pos_emb(positions)
        token_embedding = self.token_emb(x)
        return token_embedding + position_embedding

class MiniGPT(nnx.Module):
    """A miniGPT transformer model."""
    def __init__(self, maxlen: int, vocab_size: int, embed_dim: int, num_heads: int, feed_forward_dim: int, num_transformer_blocks: int, rngs: nnx.Rngs):
        self.embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim, rngs=rngs)
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, feed_forward_dim, rngs=rngs)
            for _ in range(num_transformer_blocks)
        ]
        self.output_layer = nnx.Linear(
            in_features=embed_dim,
            out_features=vocab_size,
            kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), NamedSharding(mesh, P(None, 'model'))),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model'))),
            rngs=rngs
        )

    def __call__(self, inputs, training: bool = False):
        x = self.embedding_layer(inputs)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)
        return self.output_layer(x)

def create_model(rngs, config):
    """Creates the MiniGPT model."""
    return MiniGPT(
        maxlen=config['maxlen'],
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        feed_forward_dim=config['feed_forward_dim'],
        num_transformer_blocks=config['num_transformer_blocks'],
        rngs=rngs
    )

# --- Data Loading and Preprocessing ---
def process_dataset(dataset, maxlen):
    """Tokenizes, pads, and shuffles a streaming dataset."""
    def tokenize_and_pad(examples):
        # Tokenize and pad each text to maxlen
        tokenized_texts = []
        for text in examples['text']:
            tokens = tokenizer.encode(text, allowed_special={'<|endoftext|>'})[:maxlen]
            padded_tokens = tokens + [0] * (maxlen - len(tokens))
            tokenized_texts.append(padded_tokens)
        return {'tokens': tokenized_texts}

    # The batch_size here is for the mapping operation, not for training.
    # A larger size can be more efficient.
    dataset = dataset.map(tokenize_and_pad, batched=True, batch_size=1000)
    dataset = dataset.shuffle(buffer_size=10_000, seed=42)
    return dataset

# --- Loss and Step Functions ---
def loss_fn(model, batch, training: bool):
    """Calculates the cross-entropy loss."""
    # The `training` flag is passed to the model to control behaviors like dropout.
    logits = model(batch[:, :-1], training=training)
    targets = batch[:, 1:]
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=targets).mean()
    return loss, logits

@nnx.jit
def train_step(model: MiniGPT, optimizer: nnx.Optimizer, batch):
    """Performs a single training step."""
    # A lambda function is used to pass the `training=True` argument to the loss function.
    grad_fn = nnx.value_and_grad(lambda m, b: loss_fn(m, b, training=True), has_aux=True)
    (loss, _), grads = grad_fn(model, batch)
    optimizer.update(grads)
    return loss, model, optimizer

@nnx.jit
def eval_step(model: MiniGPT, batch):
    """Performs a single evaluation step."""
    # `training=False` ensures dropout and other training-specific layers are disabled.
    loss, _ = loss_fn(model, batch, training=False)
    return loss

# --- Training Loop ---
def main():
    # --- Configuration ---
    config = {
        'vocab_size': tokenizer.n_vocab,
        'num_transformer_blocks': 12,
        'maxlen': 1024,
        'embed_dim': 768,
        'num_heads': 12,
        'feed_forward_dim': 768,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'max_tokens_to_process': 1_000_000_000, # 600M tokens
        'eval_interval': 2000,
        'eval_steps': 1000,
        'val_set_size': 20000, # Number of samples to hold out for validation
        'checkpoint_interval': 10000,
        'checkpoint_dir': '/kaggle/working/checkpoints', # Absolute path for checkpoints
        'wandb_project': 'fineweb-training-demo'
    }

    # Calculate max iterations
    tokens_per_iteration = config['batch_size'] * config['maxlen']
    max_iterations = config['max_tokens_to_process'] // tokens_per_iteration

    # --- Initialization ---
    wandb.init(project=config['wandb_project'], config=config)
    
    rngs = nnx.Rngs(0)
    model = create_model(rngs, config)
    optimizer = nnx.Optimizer(model, optax.novograd(config['learning_rate']))

    # --- Checkpointing Setup ---
    # Using an absolute path for the checkpoint directory
    checkpointer = orbax.PyTreeCheckpointer(config['checkpoint_dir'])


    # --- Data Loading ---
    # The FineWeb dataset only has a 'train' split. We'll create a validation set from it.
    print("Loading and splitting the dataset...")
    full_dataset = load_dataset('HuggingFaceFW/fineweb', split='train', streaming=True)

    # Create a validation set by taking the first N examples and a training set from the rest.
    val_dataset_raw = full_dataset.take(config['val_set_size'])
    train_dataset_raw = full_dataset.skip(config['val_set_size'])

    print("Processing training and validation datasets...")
    train_dataset = process_dataset(train_dataset_raw, config['maxlen'])
    val_dataset = process_dataset(val_dataset_raw, config['maxlen'])
    
    # Use the .iter() method to get batches from the streaming dataset
    train_iterator = train_dataset.iter(batch_size=config['batch_size'], drop_last_batch=True)
    val_iterator = val_dataset.iter(batch_size=config['batch_size'], drop_last_batch=True)

    # --- Training ---
    start_time = time.time()
    for step in range(max_iterations):
        # --- Training Step ---
        try:
            batch = next(train_iterator)
            input_batch = jnp.array(batch['tokens'])
            # Shard the data across devices
            sharded_batch = jax.device_put(input_batch, NamedSharding(mesh, P('batch', None)))
            
            loss, model, optimizer = train_step(model, optimizer, sharded_batch)
            wandb.log({"train_loss": loss.item()}, step=step)

        except StopIteration:
            print("Restarting training iterator.")
            train_iterator = train_dataset.iter(batch_size=config['batch_size'], drop_last_batch=True)
            continue

        # --- Evaluation and Logging ---
        if (step + 1) % config['eval_interval'] == 0:
            # --- Validation Step ---
            val_losses = []
            for _ in range(config['eval_steps']):
                try:
                    val_batch = next(val_iterator)
                    val_input_batch = jnp.array(val_batch['tokens'])
                    sharded_val_batch = jax.device_put(val_input_batch, NamedSharding(mesh, P('batch', None)))
                    
                    val_loss = eval_step(model, sharded_val_batch)
                    val_losses.append(val_loss)
                except StopIteration:
                    print("Restarting validation iterator.")
                    val_iterator = val_dataset.iter(batch_size=config['batch_size'], drop_last_batch=True)
                    break # Break out of eval loop
            
            if val_losses:
                avg_val_loss = jnp.mean(jnp.array(val_losses))
                wandb.log({"val_loss": avg_val_loss.item()}, step=step)
                elapsed_time = time.time() - start_time
                print(f"Step {step + 1}/{max_iterations}, Train Loss: {loss.item():.4f}, Val Loss: {avg_val_loss.item():.4f}, Elapsed Time: {elapsed_time:.2f}s")
                start_time = time.time()

        # --- Checkpointing ---
        if (step + 1) % config['checkpoint_interval'] == 0:
            # We only want to save the model's learnable parameters.
            # nnx.split returns the model structure (GraphDef) and a State object for each filter.
            # Here, we get the state for `nnx.Param` and the state for everything else (`...`).
            _, params_state, _ = nnx.split(model, nnx.Param, ...)
            
            # We save the State object containing only the parameters.
            checkpointer.save(
                os.path.join(config['checkpoint_dir'], f'step_{step+1}'),
                params_state
            )
            print(f"Checkpoint saved at step {step+1}")

    wandb.finish()

if __name__ == '__main__':
    # You will need to login to wandb first
    # from google.colab import userdata
    # wandb.login(key=userdata.get('WANDB_API_KEY'))
    # Or run `wandb login` in your terminal
    main()
