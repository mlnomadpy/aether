"""Flax optimizer compatibility layer for Flax 0.8 and 0.11+.

In Flax 0.8:
    - nnx.Optimizer(model, tx, wrt=nnx.Param) - wrt has default value
    - optimizer.update(grads) - only takes grads

In Flax 0.11+:
    - nnx.Optimizer(model, tx, wrt=nnx.Param) - wrt is keyword-only
    - optimizer.update(model, grads) - takes model and grads

This module provides a unified interface that works with both versions.
"""

import inspect
from importlib.metadata import version

import flax.nnx as nnx


def get_flax_version() -> tuple:
    """Get the installed Flax version as a tuple.
    
    Returns:
        Tuple of (major, minor, patch) version numbers
    """
    flax_version = version("flax")
    parts = flax_version.split(".")
    return tuple(int(p) for p in parts[:3])


def is_flax_08() -> bool:
    """Check if the installed Flax version is 0.8.x (uses old optimizer API).
    
    Returns:
        True if Flax 0.8.x or earlier, False for 0.9+ (new API)
    """
    flax_ver = get_flax_version()
    # Flax 0.8.x and earlier use the old API (update takes only grads)
    # Flax 0.9+ uses the new API (update takes model and grads)
    return flax_ver < (0, 9, 0)


def _check_optimizer_update_signature() -> bool:
    """Inspect the Optimizer.update signature to determine API version.
    
    Returns:
        True if old API (grads only), False if new API (model, grads)
    """
    try:
        sig = inspect.signature(nnx.Optimizer.update)
        params = list(sig.parameters.keys())
        # Old API: update(self, grads) - exactly 2 params
        # New API: update(self, model, grads, /, **kwargs) - 3+ params with 'model' second
        # In old API, the second param (after 'self') is 'grads'
        # In new API, the second param is 'model'
        return len(params) == 2 and params[1] == "grads"
    except Exception:
        # Fall back to version check if signature inspection fails
        return is_flax_08()


# Determine API version at import time using signature inspection
_USE_OLD_API = _check_optimizer_update_signature()


def create_optimizer(model: nnx.Module, optimizer_fn, wrt=nnx.Param) -> nnx.Optimizer:
    """Create an optimizer with Flax version-compatible API.
    
    Args:
        model: The model to optimize
        optimizer_fn: The optax optimizer transformation
        wrt: Variable type filter for parameters to optimize
        
    Returns:
        An nnx.Optimizer instance
    """
    # The constructor API is the same for both Flax 0.8 and 0.11+
    # The difference is in the update() method, not the constructor
    return nnx.Optimizer(model, optimizer_fn, wrt=wrt)


def update_optimizer(optimizer: nnx.Optimizer, model: nnx.Module, grads) -> None:
    """Update model parameters using the optimizer with Flax version-compatible API.
    
    This function mutates the model parameters in-place.
    
    Args:
        optimizer: The optimizer instance
        model: The model (needed for Flax 0.11+ API)
        grads: The computed gradients
    """
    if _USE_OLD_API:
        # Flax 0.8: optimizer.update(grads) - only takes grads
        # The optimizer stores a reference to the model internally
        optimizer.update(grads)
    else:
        # Flax 0.11+: optimizer.update(model, grads) - takes model and grads
        optimizer.update(model, grads)
