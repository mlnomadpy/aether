"""Data processing utilities."""

from datasets import load_dataset
from typing import Any, Iterator, Dict
import jax.numpy as jnp


def process_dataset(dataset: Any, maxlen: int, tokenizer: Any) -> Any:
    """Process and tokenize a streaming dataset.
    
    Args:
        dataset: Input dataset
        maxlen: Maximum sequence length
        tokenizer: Tokenizer to use
        
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
        
        return {'tokens': tokens}
    
    # Apply tokenization and padding
    processed = dataset.map(
        tokenize_and_pad,
        remove_columns=['text']
    )
    
    # Shuffle the dataset for better training
    processed = processed.shuffle(buffer_size=10000, seed=42)
    
    return processed


def create_data_iterators(
    dataset_name: str,
    split: str,
    streaming: bool,
    maxlen: int,
    tokenizer: Any,
    val_set_size: int,
    batch_size: int
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
    train_dataset = process_dataset(train_dataset_raw, maxlen, tokenizer)
    val_dataset = process_dataset(val_dataset_raw, maxlen, tokenizer)
    
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
    batch_size: int
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
        
    Returns:
        Validation iterator
    """
    # Load the full dataset
    print("Loading validation dataset...")
    full_dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    
    # Create validation split
    val_dataset_raw = full_dataset.take(val_set_size)
    
    print("Processing validation dataset...")
    val_dataset = process_dataset(val_dataset_raw, maxlen, tokenizer)
    
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
    batch_size: int
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
        
    Returns:
        Training iterator
    """
    # Load the full dataset
    print("Loading training dataset...")
    full_dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    
    # Create training split (skip validation set)
    train_dataset_raw = full_dataset.skip(val_set_size)
    
    print("Processing training dataset...")
    train_dataset = process_dataset(train_dataset_raw, maxlen, tokenizer)
    
    # Create iterator
    train_iterator = train_dataset.iter(batch_size=batch_size, drop_last_batch=True)
    
    return train_iterator


def prepare_batch(batch: Dict[str, Any], mesh: Any = None) -> jnp.ndarray:
    """Prepare a batch for training/evaluation.
    
    Args:
        batch: Batch dictionary containing 'tokens'
        mesh: Optional JAX mesh for sharding
        
    Returns:
        Prepared batch tensor
    """
    from jax.sharding import NamedSharding, PartitionSpec as P
    import jax
    
    input_batch = jnp.array(batch['tokens'])
    
    if mesh is not None:
        # Shard the data across devices
        sharded_batch = jax.device_put(input_batch, NamedSharding(mesh, P('batch', None)))
        return sharded_batch
    
    return input_batch