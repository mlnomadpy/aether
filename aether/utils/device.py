"""Device and mesh setup utilities."""

import jax
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from typing import Tuple


def setup_mesh(mesh_shape: Tuple[int, int] = None) -> Mesh:
    """Set up JAX mesh for distributed training.
    
    Args:
        mesh_shape: Custom mesh shape (batch_dim, model_dim). If None, auto-detect.
        
    Returns:
        JAX Mesh object
    """
    if mesh_shape is None:
        if jax.default_backend() == 'tpu':
            # For 4-way data parallel and 2-way tensor parallel on TPU v2/v3
            mesh_shape = (4, 2)
        else:
            # Fallback for GPUs or other setups
            num_devices = len(jax.devices())
            mesh_shape = (num_devices, 1)
    
    return Mesh(mesh_utils.create_device_mesh(mesh_shape), ('batch', 'model'))