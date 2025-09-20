# Aether: Modular JAX/Flax Transformer Training Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-latest-green.svg)](https://jax.readthedocs.io/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Aether is a production-ready, modular framework for training transformer models using JAX and Flax. It provides a clean, extensible architecture with support for multiple model architectures, efficient distributed training, and comprehensive experiment management.

## 🌟 Features

- **Modular Architecture**: Clean separation of concerns with pluggable components
- **Multiple Model Support**: Built-in support for different transformer architectures (Linear, YAT)
- **Model Registry**: Dynamic model registration and creation system
- **Mixed Precision Training**: BFloat16 support for 50% memory reduction and faster training
- **Distributed Training**: JAX mesh-based sharding for efficient multi-device training
- **Configuration Management**: Flexible JSON/YAML configuration system
- **Experiment Tracking**: Integrated Weights & Biases (wandb) support
- **Production Ready**: Comprehensive error handling, logging, and checkpointing
- **Extensible**: Easy to add new models, optimizers, and training strategies

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for GPU acceleration)
- TPU access (optional, for TPU acceleration)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Optional: Install YAT Components

For YAT architecture support, install the nmn package:

```bash
# Install nmn package for YAT architecture
pip install nmn  # or follow nmn package installation instructions
```

## 🚀 Quick Start

### Basic Training

```bash
# Train with default linear architecture
python train.py --model minigpt-linear

# Train with YAT architecture
python train.py --model minigpt-yat

# Train with custom configuration
python train.py --config configs/custom_config.json
```

### Using the Python API

```python
from aether import Config, Trainer
from aether.registry import register_model
from aether.models import MiniGPT

# Create configuration
config = Config()
config.model.name = "minigpt-linear"
config.training.batch_size = 16

# Initialize and run training
trainer = Trainer(config)
trainer.train()
```

## 📁 Project Structure

```
aether/
├── aether/                 # Main package
│   ├── __init__.py
│   ├── config/            # Configuration management
│   ├── data/              # Data processing utilities
│   ├── models/            # Model architectures
│   ├── registry/          # Model registry system
│   ├── training/          # Training orchestration
│   └── utils/             # Utility functions
├── configs/               # Example configurations
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── profiling/        # Performance tests
├── train.py              # Main training script
├── setup_models.py       # Model registration
└── requirements.txt      # Dependencies
```

## 🔧 Configuration

Aether uses a hierarchical configuration system with support for JSON and YAML formats.

### Configuration Structure

```json
{
  "model": {
    "name": "minigpt-linear",
    "architecture": "linear",
    "maxlen": 1024,
    "vocab_size": 50257,
    "embed_dim": 768,
    "num_heads": 12,
    "feed_forward_dim": 768,
    "num_transformer_blocks": 12,
    "dropout_rate": 0.1
  },
  "training": {
    "batch_size": 32,
    "learning_rate": 0.002,
    "max_tokens_to_process": 1000000000,
    "eval_interval": 2000,
    "eval_steps": 1000,
    "val_set_size": 20000,
    "checkpoint_interval": 10000,
    "optimizer": "adamw",
    "lr_scheduler": "cosine",
    "lr_scheduler_alpha": 0.1,
    "lr_scheduler_warmup_steps": null,
    "momentum": 0.9,
    "weight_decay": 0.01,
    "precision": "float32"
  },
  "data": {
    "dataset_name": "HuggingFaceFW/fineweb",
    "split": "train",
    "streaming": true,
    "tokenizer_name": "gpt2"
  },
  "logging": {
    "wandb_project": "aether-training",
    "checkpoint_dir": "./checkpoints",
    "log_level": "INFO"
  }
}
```

### Pre-configured Examples

- `configs/linear_config.json`: Standard linear transformer
- `configs/yat_config.json`: YAT architecture transformer
- `configs/cosine_adamw_config.json`: AdamW optimizer with cosine learning rate decay
- `configs/sgd_warmup_cosine_config.json`: SGD optimizer with warmup cosine schedule
- `configs/bfloat16_example.json`: BFloat16 mixed precision training configuration

### Optimizers and Learning Rate Schedulers

Aether supports multiple optimizers and learning rate schedules for flexible training:

#### Supported Optimizers
- `adam`: Adam optimizer
- `adamw`: AdamW with weight decay
- `sgd`: Stochastic Gradient Descent with momentum
- `rmsprop`: RMSprop optimizer
- `novograd`: Novograd optimizer (default)
- `lion`: Lion optimizer
- `adagrad`: AdaGrad optimizer
- `adadelta`: AdaDelta optimizer
- `adamax`: AdaMax optimizer
- `nadam`: Nesterov Adam optimizer

#### Supported Learning Rate Schedulers
- `constant`: Fixed learning rate (default)
- `linear`: Linear decay from initial to final value
- `cosine`: Cosine decay schedule
- `warmup_cosine`: Cosine decay with linear warmup

#### Configuration Example
```json
{
  "training": {
    "optimizer": "adamw",
    "learning_rate": 0.002,
    "lr_scheduler": "cosine",
    "lr_scheduler_alpha": 0.1,
    "lr_scheduler_warmup_steps": 5000,
    "momentum": 0.9,
    "weight_decay": 0.01
  }
}
```

### Mixed Precision Training with BFloat16

Aether supports BFloat16 mixed precision training for improved memory efficiency and training speed:

#### BFloat16 Benefits
- **50% Memory Reduction**: Model parameters and activations use half the memory
- **Faster Training**: Improved throughput on modern hardware (TPUs, newer GPUs)
- **Maintained Accuracy**: BFloat16 provides better numerical stability than Float16
- **Easy Configuration**: Single parameter controls entire precision policy

#### Configuration
```json
{
  "training": {
    "precision": "bfloat16",  // Enable BFloat16 precision
    "optimizer": "adamw",
    "learning_rate": 0.001
  }
}
```

#### Python API Usage
```python
from aether import Config, Trainer

# Configure BFloat16 training
config = Config()
config.training.precision = "bfloat16"
config.training.optimizer = "adamw"

# Train with BFloat16
trainer = Trainer(config)
trainer.train()
```

#### Memory Usage Comparison
| Precision | Model Size (GPT-2 768d, 6 layers) | Memory Savings |
|-----------|-----------------------------------|----------------|
| Float32   | ~378 MB                          | Baseline       |
| BFloat16  | ~189 MB                          | ~50%           |

#### Example Configuration
See `configs/bfloat16_example.json` for a complete BFloat16 training configuration.

#### Demo Script
```bash
# Run BFloat16 demonstration
python demo_bfloat16.py
```

The demo script showcases:
- BFloat16 model creation and inference
- Memory usage comparison
- Configuration management
- Performance benefits

## 🏗️ Architecture Overview

### Model Registry System

The model registry enables dynamic model creation and management:

```python
from aether.registry import register_model, create_model
from aether.models import MiniGPT

# Register a new model variant
register_model(
    name="custom-model",
    model_class=MiniGPT,
    default_config={
        "architecture": "linear",
        "embed_dim": 1024,
        # ... other config
    }
)

# Create model instance
model = create_model("custom-model", config, rngs)
```

### Available Architectures

1. **Linear Architecture** (`minigpt-linear`):
   - Standard transformer with linear feed-forward networks
   - Layer normalization and residual connections
   - Suitable for most general-purpose tasks

2. **YAT Architecture** (`minigpt-yat`):
   - Uses YetAnotherTransformer components
   - Advanced non-linear mappings
   - Requires `nmn` package installation

### Adding Custom Models

```python
from aether.models.base import BaseModel
from aether.registry import register_model

class CustomModel(BaseModel):
    def __init__(self, config, rngs):
        # Model implementation
        pass
    
    def __call__(self, inputs, training=False):
        # Forward pass
        pass
    
    @classmethod
    def from_config(cls, config, rngs):
        return cls(config, rngs)

# Register the custom model
register_model("custom", CustomModel)
```

## 🧪 Testing

### Running Tests

```bash
# Run all unit tests
python tests/run_tests.py

# Run specific test file
python -m pytest tests/unit/test_models.py -v

# Run with coverage
python -m pytest tests/unit/ --cov=aether --cov-report=html
```

### Performance Profiling

```bash
# Run performance profiling
python tests/profiling/test_performance.py
```

### Test Categories

- **Unit Tests**: Test individual components and functions
- **Integration Tests**: Test component interactions
- **Profiling Tests**: Performance and memory usage analysis

## 🔬 Distributed Training

Aether supports efficient distributed training across multiple devices:

### Automatic Device Detection

```python
# Automatic mesh setup based on available devices
mesh = setup_mesh()  # Auto-detects TPU/GPU configuration

# Custom mesh configuration
mesh = setup_mesh(mesh_shape=(4, 2))  # 4-way data parallel, 2-way model parallel
```

### Sharding Strategy

- **Data Parallelism**: Batch dimension sharded across devices
- **Model Parallelism**: Model parameters sharded across devices
- **Automatic**: Framework automatically handles device placement

## 📊 Experiment Tracking

### Weights & Biases Integration

```python
# Configure experiment tracking
config.logging.wandb_project = "my-experiment"

# Automatic logging of:
# - Training/validation loss
# - Model configuration
# - System metrics
# - Checkpoints
```

### Checkpointing

```python
# Automatic checkpointing
config.training.checkpoint_interval = 10000  # Save every 10k steps

# Manual checkpoint loading
trainer.load_checkpoint("./checkpoints/step_50000")
```

## 🛠️ Development

### Code Style

```bash
# Format code
black aether/ tests/

# Sort imports
isort aether/ tests/

# Lint code
flake8 aether/ tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📈 Performance

### Benchmarks

On a V100 GPU with the default configuration:
- **Forward Pass**: ~2000 tokens/second
- **Training Step**: ~1500 tokens/second
- **Memory Usage**: ~8GB for 12-layer, 768-dim model

On TPU v5-8 (Kaggle):
- **YAT GPT Base**: ~132,000 tokens/second
- **Linear GPT Model**: ~138,000 tokens/second

*Performance metrics referenced from "Deep Learning 2.0.1" by Taha Bouhsine.*

### Optimization Tips

1. **Batch Size**: Increase for better GPU utilization
2. **Sequence Length**: Shorter sequences = faster training
3. **Model Size**: Reduce layers/dimensions for faster iteration
4. **Mixed Precision**: Enable for memory savings (future feature)

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or sequence length
2. **Slow Training**: Check device utilization and mesh configuration
3. **Import Errors**: Ensure all dependencies are installed
4. **YAT Models**: Install `nmn` package for YAT architecture support

### Getting Help

- Check the [issues](https://github.com/mlnomadpy/aether/issues) page
- Create a new issue with:
  - System information (GPU/TPU, JAX version)
  - Configuration used
  - Error messages and stack traces

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- JAX team for the excellent framework
- Flax team for the neural network library
- HuggingFace for datasets and tokenizers
- The open-source ML community

## 🔄 Migration from Legacy Scripts

### From `linear_main.py`

```bash
# Old way
python linear_main.py

# New way
python train.py --model minigpt-linear
# or
python train.py --config configs/linear_config.json
```

### From `yat_main.py`

```bash
# Old way
python yat_main.py

# New way
python train.py --model minigpt-yat
# or
python train.py --config configs/yat_config.json
```

The new framework provides the same functionality with improved modularity, better error handling, and more configuration options.