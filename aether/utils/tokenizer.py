"""Tokenizer utilities."""

import tiktoken
from typing import Any


def get_tokenizer(name: str = "gpt2") -> Any:
    """Get a tokenizer by name.
    
    Args:
        name: Tokenizer name
        
    Returns:
        Tokenizer instance
    """
    return tiktoken.get_encoding(name)