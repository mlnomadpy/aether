# Aether Demo Scripts

This folder contains demonstration scripts that showcase various features of the Aether framework.

## Demo Scripts

### `demo.py`
Main demonstration script showing the complete Aether framework capabilities:
- Framework architecture overview
- Configuration system demo
- Model registry demo
- Usage examples
- Migration guide from legacy scripts

```bash
python demo/demo.py
```

### `demo_bfloat16.py`
Demonstrates BFloat16 precision training support:
- BFloat16 configuration
- Model creation with BFloat16 precision
- Memory usage comparison
- Performance benefits

```bash
python demo/demo_bfloat16.py
```

### `demo_device_mesh.py`
Shows device mesh configuration capabilities:
- Auto-detection of device configuration
- Custom mesh configuration
- Priority system demonstration

```bash
python demo/demo_device_mesh.py
```

### `demo_schedulers.py`
Demonstrates learning rate schedulers and optimizers:
- Cosine and warmup-cosine schedules
- Extended optimizer support
- Learning rate schedule visualization
- Configuration examples

```bash
python demo/demo_schedulers.py
```

## Additional Files

- `learning_rate_schedules_demo.png`: Visualization of different learning rate schedules

## Requirements

Some demo scripts require JAX/Flax dependencies. Install them with:

```bash
pip install jax flax
```