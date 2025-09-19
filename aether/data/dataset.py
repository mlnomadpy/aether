"""Data processing utilities."""

from datasets import load_dataset
from typing import Any, Iterator, Dict, Optional
import jax.numpy as jnp
import jax


def process_dataset(dataset: Any, maxlen: int, tokenizer: Any, training_mode: str = "clm", **mlm_kwargs) -> Any:
    """Process and tokenize a streaming dataset.
    
    Args:
        dataset: Input dataset
        maxlen: Maximum sequence length
        tokenizer: Tokenizer to use
        training_mode: Training mode ("clm" or "mlm")
        **mlm_kwargs: MLM-specific parameters (mask_prob, replace_prob, random_prob)
        
    Returns:
        Processed dataset
    """
    def tokenize_and_pad(example):
        # Tokenize the text
        tokens = tokenizer.encode(example['text'])
        
        # Truncate or pad to maxlen
        if len(tokens) >= maxlen:
            tokens = tokens[:maxlen]
        else:
            # Pad with zeros (or tokenizer.pad_token_id if available)
            pad_id = getattr(tokenizer, 'pad_token_id', 0)
            tokens = tokens + [pad_id] * (maxlen - len(tokens))
        
        result = {'tokens': tokens}
        
        # Add MLM masking if in MLM mode
        if training_mode == "mlm":
            masked_tokens, mask_labels = apply_mlm_masking(
                tokens, tokenizer, **mlm_kwargs
            )
            result.update({
                'masked_tokens': masked_tokens,
                'mask_labels': mask_labels
            })
        
        return result
    
    # Apply tokenization and padding
    processed = dataset.map(
        tokenize_and_pad,
        remove_columns=['text']
    )
    
    # Shuffle the dataset for better training
    processed = processed.shuffle(buffer_size=10000, seed=42)
    
    return processed


def apply_mlm_masking(tokens: list, tokenizer: Any, mask_prob: float = 0.15, 
                      replace_prob: float = 0.8, random_prob: float = 0.1) -> tuple[list, list]:
    """Apply MLM masking to tokens.
    
    Args:
        tokens: List of token IDs
        tokenizer: Tokenizer instance
        mask_prob: Probability of masking each token
        replace_prob: Probability of replacing masked tokens with [MASK] token
        random_prob: Probability of replacing masked tokens with random tokens
        
    Returns:
        Tuple of (masked_tokens, mask_labels)
        mask_labels: -100 for unmasked, original_token_id for masked positions
    """
    # Get special token IDs
    mask_token_id = getattr(tokenizer, 'mask_token_id', None)
    pad_token_id = getattr(tokenizer, 'pad_token_id', 0)
    vocab_size = getattr(tokenizer, 'vocab_size', 50257)
    
    # If no mask token available, use a placeholder (this is common for GPT-style tokenizers)
    if mask_token_id is None:
        # For GPT-2 style tokenizers, we'll use token_id 50256 as a pseudo-mask token
        mask_token_id = vocab_size - 1
    
    tokens = tokens.copy()
    mask_labels = [-100] * len(tokens)  # -100 means "ignore" in loss computation
    
    # Create random states
    import random
    random.seed(42)  # For reproducible masking
    
    for i, token in enumerate(tokens):
        # Don't mask padding tokens
        if token == pad_token_id:
            continue
            
        # Randomly decide whether to mask this token
        if random.random() < mask_prob:
            mask_labels[i] = token  # Store original token for loss computation
            
            rand_val = random.random()
            if rand_val < replace_prob:
                # Replace with [MASK] token
                tokens[i] = mask_token_id
            elif rand_val < replace_prob + random_prob:
                # Replace with random token
                tokens[i] = random.randint(0, vocab_size - 1)
            # Else: keep original token (10% of masked tokens)
    
    return tokens, mask_labels


def create_data_iterators(
    dataset_name: str,
    split: str,
    streaming: bool,
    maxlen: int,
    tokenizer: Any,
    val_set_size: int,
    batch_size: int,
    training_mode: str = "clm",
    **mlm_kwargs
) -> tuple[Iterator[Dict], Iterator[Dict]]:
    """Create training and validation data iterators.
    
    Args:
        dataset_name: Name of the dataset to load
        split: Dataset split to use
        streaming: Whether to use streaming mode
        maxlen: Maximum sequence length
        tokenizer: Tokenizer to use
        val_set_size: Number of samples for validation
        batch_size: Batch size for iterators
        training_mode: Training mode ("clm" or "mlm")
        **mlm_kwargs: MLM-specific parameters
        
    Returns:
        Tuple of (train_iterator, val_iterator)
    """
    # Load the full dataset
    print("Loading and splitting the dataset...")
    full_dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    
    # Create validation and training splits
    val_dataset_raw = full_dataset.take(val_set_size)
    train_dataset_raw = full_dataset.skip(val_set_size)
    
    print("Processing training and validation datasets...")
    train_dataset = process_dataset(train_dataset_raw, maxlen, tokenizer, training_mode, **mlm_kwargs)
    val_dataset = process_dataset(val_dataset_raw, maxlen, tokenizer, training_mode, **mlm_kwargs)
    
    # Create iterators
    train_iterator = train_dataset.iter(batch_size=batch_size, drop_last_batch=True)
    val_iterator = val_dataset.iter(batch_size=batch_size, drop_last_batch=True)
    
    return train_iterator, val_iterator


def create_validation_iterator(
    dataset_name: str,
    split: str,
    streaming: bool,
    maxlen: int,
    tokenizer: Any,
    val_set_size: int,
    batch_size: int,
    training_mode: str = "clm",
    **mlm_kwargs
) -> Iterator[Dict]:
    """Create only the validation data iterator.
    
    Args:
        dataset_name: Name of the dataset to load
        split: Dataset split to use
        streaming: Whether to use streaming mode
        maxlen: Maximum sequence length
        tokenizer: Tokenizer to use
        val_set_size: Number of samples for validation
        batch_size: Batch size for iterators
        training_mode: Training mode ("clm" or "mlm")
        **mlm_kwargs: MLM-specific parameters
        
    Returns:
        Validation iterator
    """
    # Load the full dataset
    print("Loading validation dataset...")
    full_dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    
    # Create validation split
    val_dataset_raw = full_dataset.take(val_set_size)
    
    print("Processing validation dataset...")
    val_dataset = process_dataset(val_dataset_raw, maxlen, tokenizer, training_mode, **mlm_kwargs)
    
    # Create iterator
    val_iterator = val_dataset.iter(batch_size=batch_size, drop_last_batch=True)
    
    return val_iterator


def create_training_iterator(
    dataset_name: str,
    split: str,
    streaming: bool,
    maxlen: int,
    tokenizer: Any,
    val_set_size: int,
    batch_size: int,
    training_mode: str = "clm",
    **mlm_kwargs
) -> Iterator[Dict]:
    """Create only the training data iterator.
    
    Args:
        dataset_name: Name of the dataset to load
        split: Dataset split to use
        streaming: Whether to use streaming mode
        maxlen: Maximum sequence length
        tokenizer: Tokenizer to use
        val_set_size: Number of samples for validation
        batch_size: Batch size for iterators
        training_mode: Training mode ("clm" or "mlm")
        **mlm_kwargs: MLM-specific parameters
        
    Returns:
        Training iterator
    """
    # Load the full dataset
    print("Loading training dataset...")
    full_dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    
    # Create training split (skip validation set)
    train_dataset_raw = full_dataset.skip(val_set_size)
    
    print("Processing training dataset...")
    train_dataset = process_dataset(train_dataset_raw, maxlen, tokenizer, training_mode, **mlm_kwargs)
    
    # Create iterator
    train_iterator = train_dataset.iter(batch_size=batch_size, drop_last_batch=True)
    
    return train_iterator


def prepare_batch(batch: Dict[str, Any], mesh: Any = None, training_mode: str = "clm") -> Dict[str, jnp.ndarray]:
    """Prepare a batch for training/evaluation.
    
    Args:
        batch: Batch dictionary containing 'tokens' and optionally 'masked_tokens', 'mask_labels'
        mesh: Optional JAX mesh for sharding
        training_mode: Training mode ("clm" or "mlm")
        
    Returns:
        Prepared batch dictionary with JAX arrays
    """
    from jax.sharding import NamedSharding, PartitionSpec as P
    import jax
    
    result = {}
    
    if training_mode == "clm":
        input_batch = jnp.array(batch['tokens'])
        if mesh is not None:
            input_batch = jax.device_put(input_batch, NamedSharding(mesh, P('batch', None)))
        result['tokens'] = input_batch
    elif training_mode == "mlm":
        # For MLM, we need both masked tokens and mask labels
        masked_tokens = jnp.array(batch['masked_tokens'])
        mask_labels = jnp.array(batch['mask_labels'])
        
        if mesh is not None:
            masked_tokens = jax.device_put(masked_tokens, NamedSharding(mesh, P('batch', None)))
            mask_labels = jax.device_put(mask_labels, NamedSharding(mesh, P('batch', None)))
        
        result['masked_tokens'] = masked_tokens
        result['mask_labels'] = mask_labels
    
    return result