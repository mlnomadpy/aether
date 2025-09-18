# Device Mesh Configuration

This document explains how to configure device mesh settings in Aether to optimize training for larger models and multi-device setups.

## Overview

Aether now supports customizable device mesh configuration through the config system. This allows you to:

- **Customize tensor and data parallelism** by specifying mesh dimensions
- **Optimize memory distribution** across available devices  
- **Configure mesh topology** for better communication patterns
- **Support multi-device training setups** more effectively

## Configuration Options

### DeviceConfig

The `DeviceConfig` class provides two main options:

```python
@dataclass
class DeviceConfig:
    """Device and mesh configuration."""
    mesh_shape: Optional[Tuple[int, int]] = None  # (batch_dim, model_dim)
    auto_detect_mesh: bool = True  # Whether to auto-detect mesh if mesh_shape is None
```

- **`mesh_shape`**: Tuple specifying `(batch_dimension, model_dimension)` for the device mesh
  - `batch_dimension`: Number of devices for data parallelism
  - `model_dimension`: Number of devices for model parallelism
  - Must multiply to equal the total number of available devices
  - Set to `None` to use auto-detection
  
- **`auto_detect_mesh`**: Whether to automatically detect optimal mesh shape when `mesh_shape` is `None`

## Configuration Methods

### 1. JSON/YAML Configuration Files

Add a `device` section to your configuration file:

```json
{
  "model": {
    "name": "minigpt-linear",
    "embed_dim": 1024,
    ...
  },
  "training": {
    "batch_size": 64,
    ...
  },
  "device": {
    "mesh_shape": [2, 4],
    "auto_detect_mesh": false
  }
}
```

### 2. Programmatic Configuration

```python
from aether.config import Config, DeviceConfig

# Create custom device config
device_config = DeviceConfig(
    mesh_shape=(2, 4),  # 2-way data parallel, 4-way model parallel
    auto_detect_mesh=False
)

# Add to main config
config = Config()
config.device = device_config
```

### 3. Dictionary Configuration

```python
config_dict = {
    "device": {
        "mesh_shape": [2, 4],
        "auto_detect_mesh": False
    }
}
config = Config.from_dict(config_dict)
```

## Usage Examples

### Default Auto-Detection

```python
# Uses auto-detection (default behavior)
config = Config()
# config.device.mesh_shape = None
# config.device.auto_detect_mesh = True
```

### Custom Mesh for 8-Device Setup

```python
# For 8 devices total
config = Config()
config.device.mesh_shape = (2, 4)  # 2-way data parallel, 4-way model parallel
config.device.auto_detect_mesh = False
```

### Single Device Setup

```python
# For single device (CPU/GPU)
config = Config()
config.device.mesh_shape = (1, 1)
```

## Mesh Shape Guidelines

### For Data Parallelism
- Use `(num_devices, 1)` to maximize data parallelism
- Example: `(8, 1)` for 8-way data parallel

### For Model Parallelism
- Use `(1, num_devices)` to maximize model parallelism
- Example: `(1, 8)` for 8-way model parallel

### For Mixed Parallelism
- Balance data and model parallelism
- Example: `(2, 4)` for 2-way data + 4-way model parallel

### Device Requirements
- The product of mesh dimensions must equal available devices
- For `(2, 4)` mesh shape, you need exactly 8 devices
- Auto-detection helps find valid configurations

## Priority System

The mesh configuration follows a priority order:

1. **Explicit `mesh_shape` parameter** (highest priority)
2. **Device config `mesh_shape`**
3. **Auto-detection** (lowest priority)

```python
# Explicit shape takes priority over config
mesh = setup_mesh(mesh_shape=(1, 4), device_config=device_config)

# Config shape used when no explicit shape
mesh = setup_mesh(device_config=device_config)

# Auto-detection when no config
mesh = setup_mesh()
```

## Benefits for Large Models

### Memory Optimization
- **Model parallelism** distributes model parameters across devices
- **Reduces per-device memory requirements** for large models
- **Enables training of models larger than single-device memory**

### Communication Efficiency
- **Optimized mesh topology** reduces communication overhead
- **Balanced parallelism** minimizes data movement
- **Better utilization** of available bandwidth

### Scalability
- **Easy scaling** to multi-device setups
- **Flexible configuration** for different hardware configurations
- **Future-proof** for larger model architectures

## Example Configurations

See the `configs/` directory for example configurations:

- `configs/linear_config.json` - Default auto-detection
- `configs/large_model_custom_mesh.json` - Custom 2x4 mesh for large models

## Troubleshooting

### Device Count Mismatch
```
ValueError: Number of devices 4 must equal the product of mesh_shape (2, 4)
```
- Ensure mesh shape product matches available devices
- Use auto-detection if unsure about device count

### Invalid Mesh Shape
```
TypeError: mesh_shape must be a tuple of integers
```
- Ensure mesh_shape is a tuple/list of two integers
- Example: `[2, 4]` or `(2, 4)`

### Single Device Limitations
- Use `(1, 1)` mesh shape for single-device setups
- Model parallelism requires multiple devices
- Data parallelism works with any number of devices

## Migration from Previous Versions

Existing configurations work without changes - device configuration is optional and backwards compatible:

```python
# Old config (still works)
config = Config()
# No device section needed

# New config (with device control)
config = Config()
config.device.mesh_shape = (2, 4)
```