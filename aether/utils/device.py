"""Device and mesh setup utilities."""

import jax
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from typing import Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import DeviceConfig


def setup_mesh(mesh_shape: Tuple[int, int] = None, device_config: Optional["DeviceConfig"] = None) -> Mesh:
    """Set up JAX mesh for distributed training.
    
    Args:
        mesh_shape: Custom mesh shape (batch_dim, model_dim). If None, use device_config or auto-detect.
        device_config: Device configuration object with mesh settings.
        
    Returns:
        JAX Mesh object
    """
    # Priority: explicit mesh_shape > device_config.mesh_shape > auto-detection
    if mesh_shape is None and device_config is not None:
        mesh_shape = device_config.mesh_shape
        
    if mesh_shape is None:
        if jax.default_backend() == 'tpu':
            # For 4-way data parallel and 2-way tensor parallel on TPU v2/v3
            mesh_shape = (4, 2)
        else:
            # Fallback for GPUs or other setups
            num_devices = len(jax.devices())
            mesh_shape = (num_devices, 1)
    
    return Mesh(mesh_utils.create_device_mesh(mesh_shape), ('batch', 'model'))