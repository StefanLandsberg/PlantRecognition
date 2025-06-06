#!/usr/bin/env python3
"""
Curse-Resistant Feature Learning Package

An approach to combat the curse of dimensionality through real-time 
feature quality learning and adaptive enhancement.

Author: Plant Classification Research Team
"""

__version__ = "1.0.0"
__author__ = "Plant Classification Research Team"
__description__ = "Curse-resistant feature learning for high-dimensional data"

# Core components (implemented)
from .foundation.multi_extractor import MultiExtractorFoundation
from .metrics.curse_metrics import CurseResistanceMetrics
from .processor.curse_processor import CurseResistantProcessor
from .training.config import CurseResistantTrainingConfig, create_training_config
from .training.loss_functions import CurseResistantLoss, AdaptiveLossWeighting
from .training.trainer import CurseResistantTrainer

__all__ = [
    # Core classes (implemented)
    'MultiExtractorFoundation',
    'CurseResistanceMetrics', 
    'CurseResistantProcessor',
    
    # Training components
    'CurseResistantTrainingConfig',
    'create_training_config',
    'CurseResistantLoss',
    'AdaptiveLossWeighting',
    'CurseResistantTrainer',
    
    # Package info
    '__version__',
    '__author__',
    '__description__'
] 