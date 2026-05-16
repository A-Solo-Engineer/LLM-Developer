"""
Configuration Management for LLM Developer Studio

Centralized configuration with presets for different use cases.
Supports device detection, batch/block size recommendations, and hyperparameter presets.
"""

import os
import torch
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class PresetConfig:
    """Configuration preset for different use cases"""
    name: str
    batch_size: int
    block_size: int
    learning_rate: float
    epochs: int
    n_embd: int
    n_head: int
    n_layer: int
    dropout: float
    description: str

class Config:
    """Centralized configuration manager"""
    
    # Presets for different use cases
    PRESETS: Dict[str, PresetConfig] = {
        'light': PresetConfig(
            name='light',
            batch_size=16,
            block_size=32,
            learning_rate=0.001,
            epochs=20,
            n_embd=32,
            n_head=2,
            n_layer=2,
            dropout=0.05,
            description='Lightweight preset for testing and CPU-constrained systems'
        ),
        'medium': PresetConfig(
            name='medium',
            batch_size=32,
            block_size=64,
            learning_rate=0.001,
            epochs=50,
            n_embd=64,
            n_head=4,
            n_layer=4,
            dropout=0.1,
            description='Balanced preset for standard use cases with Phase 2 optimizations'
        ),
        'heavy': PresetConfig(
            name='heavy',
            batch_size=64,
            block_size=128,
            learning_rate=0.0005,
            epochs=100,
            n_embd=128,
            n_head=8,
            n_layer=6,
            dropout=0.1,
            description='Advanced preset for larger datasets and powerful CPUs with Phase 2 optimizations'
        )
    }
    
    # Default values
    DEFAULT_PRESET = 'medium'
    DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DEFAULT_EVAL_ITERS = 200
    DEFAULT_MAX_NEW_TOKENS = 100
    DEFAULT_TEMPERATURE = 0.8
    DEFAULT_TOP_P = 0.9
    
    # Model paths
    MODEL_SAVE_PATH = 'model.pt'
    TOKENIZER_SAVE_PATH = 'tokenizer.pkl'
    METADATA_SAVE_PATH = 'model_metadata.json'
    
    # Validation constraints
    BATCH_SIZE_RANGE = (1, 256)
    BLOCK_SIZE_RANGE = (8, 512)
    LEARNING_RATE_RANGE = (1e-5, 0.1)
    EPOCHS_RANGE = (1, 1000)
    EMBEDDING_DIM_RANGE = (16, 512)
    NUM_HEADS_RANGE = (1, 16)
    NUM_LAYERS_RANGE = (1, 12)
    DROPOUT_RANGE = (0.0, 0.5)
    
    # Safety constraints
    MAX_INPUT_SIZE_MB = 50  # Maximum input file size
    MAX_VOCAB_SIZE = 10000  # Maximum vocabulary size
    MIN_TRAINING_SAMPLES = 5  # Minimum training samples required
    
    @staticmethod
    def get_preset(preset_name: str = DEFAULT_PRESET) -> PresetConfig:
        """Get a configuration preset"""
        if preset_name not in Config.PRESETS:
            raise ValueError(
                f"Unknown preset '{preset_name}'. "
                f"Available presets: {', '.join(Config.PRESETS.keys())}"
            )
        return Config.PRESETS[preset_name]
    
    @staticmethod
    def list_presets() -> Dict[str, str]:
        """List all available presets with descriptions"""
        return {name: preset.description for name, preset in Config.PRESETS.items()}
    
    @staticmethod
    def get_recommended_config(dataset_size_bytes: int) -> str:
        """Recommend a preset based on dataset size"""
        if dataset_size_bytes < 1_000_000:  # < 1MB
            return 'light'
        elif dataset_size_bytes < 10_000_000:  # < 10MB
            return 'medium'
        else:
            return 'heavy'
    
    @staticmethod
    def detect_device() -> str:
        """Detect available device (CUDA or CPU)"""
        if torch.cuda.is_available():
            return 'cuda'
        return 'cpu'
    
    @staticmethod
    def get_device_info() -> Dict[str, Any]:
        """Get device information"""
        device = Config.detect_device()
        info = {
            'device': device,
            'cuda_available': torch.cuda.is_available(),
        }
        if device == 'cuda':
            info['cuda_version'] = torch.version.cuda
            info['cudnn_version'] = torch.backends.cudnn.version()
            info['gpu_count'] = torch.cuda.device_count()
            if torch.cuda.device_count() > 0:
                info['gpu_name'] = torch.cuda.get_device_name(0)
        return info
