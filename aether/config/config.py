"""Configuration management for Aether."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import os
import json
import yaml


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "minigpt-linear"
    architecture: str = "linear"
    maxlen: int = 1024
    vocab_size: int = 50257
    embed_dim: int = 768
    num_heads: int = 12
    feed_forward_dim: int = 768
    num_transformer_blocks: int = 12
    dropout_rate: float = 0.1


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    learning_rate: float = 2e-3
    max_tokens_to_process: int = 1_000_000_000
    eval_interval: int = 2000
    eval_steps: int = 1000
    val_set_size: int = 20000
    checkpoint_interval: int = 10000
    optimizer: str = "novograd"
    # Learning rate scheduler settings
    lr_scheduler: str = "constant"  # Options: "constant", "linear", "cosine", "warmup_cosine"
    lr_scheduler_alpha: float = 0.0  # Minimum learning rate multiplier for cosine schedules
    lr_scheduler_warmup_steps: Optional[int] = None  # Warmup steps for warmup schedules
    # Optimizer-specific settings
    momentum: float = 0.9  # For optimizers that support momentum (SGD, etc.)
    weight_decay: float = 0.0  # Weight decay for regularization
    # Precision settings
    precision: str = "float32"  # Options: "float32", "bfloat16"


@dataclass
class DataConfig:
    """Data configuration."""
    dataset_name: str = "HuggingFaceFW/fineweb"
    split: str = "train"
    streaming: bool = True
    tokenizer_name: str = "gpt2"


@dataclass 
class LoggingConfig:
    """Logging configuration."""
    wandb_project: str = "aether-training"
    checkpoint_dir: str = "./checkpoints"
    log_level: str = "INFO"
    log_file: Optional[str] = None


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create config from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Config instance
        """
        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            data=DataConfig(**config_dict.get("data", {})),
            logging=LoggingConfig(**config_dict.get("logging", {}))
        )
    
    @classmethod
    def from_file(cls, config_path: str) -> "Config":
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file (JSON or YAML)
            
        Returns:
            Config instance
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.endswith('.json'):
                config_dict = json.load(f)
            elif config_path.endswith(('.yaml', '.yml')):
                config_dict = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path}")
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.
        
        Returns:
            Configuration dictionary
        """
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
            "logging": self.logging.__dict__
        }
    
    def save(self, config_path: str) -> None:
        """Save configuration to file.
        
        Args:
            config_path: Path to save configuration file
        """
        config_dict = self.to_dict()
        
        os.makedirs(os.path.dirname(config_path) if os.path.dirname(config_path) else ".", exist_ok=True)
        
        with open(config_path, 'w') as f:
            if config_path.endswith('.json'):
                json.dump(config_dict, f, indent=2)
            elif config_path.endswith(('.yaml', '.yml')):
                yaml.dump(config_dict, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported config file format: {config_path}")
    
    def get_model_config_dict(self) -> Dict[str, Any]:
        """Get model configuration as dictionary compatible with model creation.
        
        Returns:
            Model configuration dictionary
        """
        model_dict = self.model.__dict__.copy()
        # Remove name field as it's used for registry lookup
        model_dict.pop("name", None)
        return model_dict