#!/usr/bin/env python3
"""
Foundation Model Fusion Framework - Curse-Resistant Feature Learning
Copyright (C) 2024 Stefan (PlantRecognition Project)

PROPRIETARY AND CONFIDENTIAL
This file contains proprietary algorithms for foundation model fusion and 
curse-resistant feature learning. The core methodology represents novel 
intellectual property in the field of multi-model ensemble learning.

ALL RIGHTS RESERVED - NO MODIFICATIONS PERMITTED

This software is proprietary and confidential. Any unauthorized use,
modification, copying, or distribution is strictly prohibited and will
be prosecuted to the full extent of the law.

Viewing permitted for educational purposes only.
See LICENSE_PROPRIETARY for complete terms.

PATENT NOTICE: This software may be covered by pending patent applications.
Commercial use requires explicit licensing agreement.

Original Development: December 2024
Contact: [Your Contact Information]
"""

"""
Curse-Resistant Feature Learning Training System

Implements overfitting prevention techniques for plant classification:
- Gaussian noise injection during training
- Feature dropout regularisation
- Network dropout regularisation
- Strong weight decay optimisation
- Mixed precision training for efficiency
- Stratified data splitting for balanced validation

This system processes foundation features (4352-dim) to produce curse-resistant
features (1024-dim) whilst maintaining classification performance.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
import argparse
import logging
import json
import random
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from typing import Dict, Optional, Tuple
import time

# Add the curse_resistant_feature_learning to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / "curse_resistant_feature_learning"))

from foundation.multi_extractor import MultiExtractorFoundation
from metrics.curse_metrics import CurseResistanceMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CurseResistantProcessor(nn.Module):
    """
    Feature processor with overfitting prevention techniques.
    
    Applies Gaussian noise injection, feature dropout, and network dropout
    to transform foundation features into curse-resistant representations.
    """
    
    def __init__(self, 
                 input_dim: int = 4352, 
                 target_dim: int = 1024,
                 noise_std: float = 0.05,
                 feature_dropout_rate: float = 0.15,
                 network_dropout_rate: float = 0.4):
        """
        Initialise curse-resistant processor.
        
        Args:
            input_dim: Input feature dimension
            target_dim: Output feature dimension
            noise_std: Standard deviation for Gaussian noise injection
            feature_dropout_rate: Feature dropout probability
            network_dropout_rate: Network dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.noise_std = noise_std
        self.feature_dropout_rate = feature_dropout_rate
        self.network_dropout_rate = network_dropout_rate
        
        # Feature processing network
        self.processor = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(network_dropout_rate),
            
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(network_dropout_rate),
            
            nn.Linear(1024, target_dim),
            nn.BatchNorm1d(target_dim),
            nn.ReLU()
        )
        
        # Curse resistance evaluator for metrics
        self.curse_evaluator = nn.Sequential(
            nn.Linear(target_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(network_dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(network_dropout_rate),
            
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self._initialise_weights()
    
    def _initialise_weights(self):
        """Initialise network weights using Xavier normalisation."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.8)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def _apply_training_augmentation(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply training-time feature augmentation techniques.
        
        Args:
            features: Input features [batch_size, input_dim]            
        Returns:
            Augmented features [batch_size, input_dim]

        """
        if not self.training:
            return features
        
        # Gaussian noise injection
        noise = torch.randn_like(features) * self.noise_std
        augmented_features = features + noise
        
        # Feature dropout
        dropout_mask = torch.rand_like(features) > self.feature_dropout_rate
        augmented_features = augmented_features * dropout_mask.float()
        
        return augmented_features
    
    def forward(self, features: torch.Tensor, 
                return_metrics: bool = False) -> torch.Tensor:
        """
        Forward pass through curse-resistant processor.
        
        Args:
            features: Input features [batch_size, input_dim]
            return_metrics: Whether to return curse resistance metrics
            
        Returns:
            Enhanced features [batch_size, target_dim] or (features, metrics)
        """
        # Handle single sample batches for BatchNorm compatibility
        single_sample = features.size(0) == 1
        if single_sample and self.training:
            was_training = True
            self.eval()
        else:
            was_training = False
        
        # Apply training augmentation
        augmented_features = self._apply_training_augmentation(features)
        
        # Process through network
        enhanced_features = self.processor(augmented_features)
        
        # Restore training mode if needed
        if was_training:
            self.train()
        
        if return_metrics:
            curse_score = self.curse_evaluator(enhanced_features)
            
            metrics = {
                'curse_resistance_score': curse_score,
                'feature_norm': torch.norm(enhanced_features, dim=1).mean(),
                'feature_std': torch.std(enhanced_features, dim=1).mean()
            }
            return enhanced_features, metrics
        
        return enhanced_features


class CurseResistantTrainer:
    """Training system for curse-resistant feature learning."""
    
    def __init__(self, input_dim: int = 4352, target_dim: int = 1024):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialise processor
        self.processor = CurseResistantProcessor(
            input_dim=input_dim,
            target_dim=target_dim,
            noise_std=0.05,
            feature_dropout_rate=0.15,
            network_dropout_rate=0.4
        ).to(self.device)
        
        # Mixed precision scaler for GPU acceleration
        self.scaler = GradScaler() if torch.cuda.is_available() else None
        
        # Training state
        self.best_val_curse_score = 0.0
        self.best_model_state = None
        self.patience_counter = 0
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_curse_score': [],
            'val_curse_score': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def setup_training(self, num_classes: int, learning_rate: float = 1e-3):
        """Configure optimiser and classifier for training."""
        
        # Setup classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.processor.target_dim, num_classes)
        ).to(self.device)
        
        # Setup optimiser with strong weight decay
        self.optimiser = torch.optim.AdamW(
            list(self.processor.parameters()) + list(self.classifier.parameters()),
            lr=learning_rate,
            weight_decay=1e-2,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Classification loss
        self.classification_loss = nn.CrossEntropyLoss()
    
    def train_epoch(self, train_loader, curse_metrics):
        """Execute one training epoch."""
        self.processor.train()
        self.classifier.train()
        
        total_loss = 0.0
        total_curse_score = 0.0
        total_accuracy = 0.0
        num_batches = 0
        curse_scores = []  # Store curse scores from sampled batches
        
        epoch_start = time.time()
        
        for batch_idx, (foundation_features, labels) in enumerate(train_loader):
            batch_start = time.time()
            foundation_features = foundation_features.to(self.device)
            labels = labels.to(self.device)

            self.optimiser.zero_grad()
            
            # Calculate curse metrics only every 10 batches for speed
            calculate_curse = (batch_idx % 10 == 0)
            
            # Mixed precision forward pass
            if self.scaler is not None:
                with autocast():
                    enhanced_features, processor_metrics = self.processor(
                        foundation_features, return_metrics=True
                    )
                    
                    predictions = self.classifier(enhanced_features)
                    
                    # Calculate losses
                    cls_loss = self.classification_loss(predictions, labels)
                    curse_loss = -torch.mean(processor_metrics['curse_resistance_score'])
                    
                    # Combined loss
                    total_loss_batch = cls_loss + 0.4 * curse_loss
                
                # Calculate curse metrics only periodically for speed
                if calculate_curse:
                    with torch.no_grad():
                        batch_curse_metrics = curse_metrics.evaluate_batch(enhanced_features, labels)
                        curse_scores.append(batch_curse_metrics['curse_resistance_score'])
                
                # Mixed precision backward pass
                self.scaler.scale(total_loss_batch).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimiser)
                torch.nn.utils.clip_grad_norm_(
                    list(self.processor.parameters()) + list(self.classifier.parameters()),
                    max_norm=1.0
                )
                
                self.scaler.step(self.optimiser)
                self.scaler.update()
            else:
                # Standard precision fallback
                enhanced_features, processor_metrics = self.processor(
                    foundation_features, return_metrics=True
                )
                
                predictions = self.classifier(enhanced_features)
                
                # Calculate curse metrics only periodically
                if calculate_curse:
                    batch_curse_metrics = curse_metrics.evaluate_batch(enhanced_features, labels)
                    curse_scores.append(batch_curse_metrics['curse_resistance_score'])
                
                # Calculate losses
                cls_loss = self.classification_loss(predictions, labels)
                curse_loss = -torch.mean(processor_metrics['curse_resistance_score'])
                
                # Combined loss
                total_loss_batch = cls_loss + 0.4 * curse_loss
                
                # Standard backward pass
                total_loss_batch.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    list(self.processor.parameters()) + list(self.classifier.parameters()),
                    max_norm=1.0
                )
                
                self.optimiser.step()
            
            # Calculate accuracy
            _, predicted = torch.max(predictions, 1)
            accuracy = (predicted == labels).float().mean()
            
            # Update metrics
            total_loss += total_loss_batch.item()
            total_accuracy += accuracy.item()
            num_batches += 1
            
            # Log batch timing every 10 batches
            if batch_idx % 10 == 0:
                batch_time = time.time() - batch_start
                print(f"  Batch {batch_idx+1}: {batch_time:.2f}s")
                
                # Memory cleanup every 10 batches
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Final memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch training time: {epoch_time:.1f}s")
        
        # Use average of sampled curse scores
        avg_curse_score = np.mean(curse_scores) if curse_scores else 0.0
        
        return {
            'loss': total_loss / num_batches,
            'curse_score': avg_curse_score,
            'accuracy': total_accuracy / num_batches
        }
    
    def validate_epoch(self, val_loader, curse_metrics):
        """Execute one validation epoch."""
        self.processor.eval()
        self.classifier.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        curse_scores = []  # Store curse scores from sampled batches
        
        val_start = time.time()

        with torch.no_grad():
            for batch_idx, (foundation_features, labels) in enumerate(val_loader):
                foundation_features = foundation_features.to(self.device)
                labels = labels.to(self.device)
                
                # Calculate curse metrics only every 5 batches for speed
                calculate_curse = (batch_idx % 5 == 0)
                
                # Mixed precision validation
                if self.scaler is not None:
                    with autocast():
                        enhanced_features = self.processor(foundation_features)
                        predictions = self.classifier(enhanced_features)
                        cls_loss = self.classification_loss(predictions, labels)
                else:
                    # Standard precision validation
                    enhanced_features = self.processor(foundation_features)
                    predictions = self.classifier(enhanced_features)
                    cls_loss = self.classification_loss(predictions, labels)
                
                # Calculate curse metrics only periodically
                if calculate_curse:
                    batch_curse_metrics = curse_metrics.evaluate_batch(enhanced_features, labels)
                    curse_scores.append(batch_curse_metrics['curse_resistance_score'])
                
                # Calculate accuracy
                _, predicted = torch.max(predictions, 1)
                accuracy = (predicted == labels).float().mean()
                
                # Update metrics
                total_loss += cls_loss.item()
                total_accuracy += accuracy.item()
                num_batches += 1
        
        val_time = time.time() - val_start
        print(f"Validation time: {val_time:.1f}s")
        
        # Use average of sampled curse scores
        avg_curse_score = np.mean(curse_scores) if curse_scores else 0.0
        
        return {
            'loss': total_loss / num_batches,
            'curse_score': avg_curse_score,
            'accuracy': total_accuracy / num_batches
        }
    
    def train(self, train_loader, val_loader, num_classes: int, 
              epochs: int = 50, patience: int = 10):
        """Execute complete training process with overfitting prevention."""
        
        print("Starting Curse-Resistant Feature Learning")
        print("=" * 60)
        print("Overfitting prevention techniques:")
        print(f"  - Gaussian noise injection (std={self.processor.noise_std})")
        print(f"  - Feature dropout ({self.processor.feature_dropout_rate*100:.0f}%)")
        print(f"  - Network dropout ({self.processor.network_dropout_rate*100:.0f}%)")
        print(f"  - Strong weight decay (1e-2)")
        if self.scaler is not None:
            print(f"  - Mixed precision training (GPU acceleration)")
        print("=" * 60)
        print()
        
        # Setup training components
        self.setup_training(num_classes)
        curse_metrics = CurseResistanceMetrics()
        
        # Training loop
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            print("-" * 40)
            
            # Train and validate
            train_results = self.train_epoch(train_loader, curse_metrics)
            val_results = self.validate_epoch(val_loader, curse_metrics)
            
            # Update scheduler
            self.scheduler.step(val_results['curse_score'])
            
            # Log results
            print(f"Train - Loss: {train_results['loss']:.4f}, "
                  f"Curse: {train_results['curse_score']:.4f}, "
                  f"Acc: {train_results['accuracy']:.4f}")
            print(f"Val   - Loss: {val_results['loss']:.4f}, "
                  f"Curse: {val_results['curse_score']:.4f}, "
                  f"Acc: {val_results['accuracy']:.4f}")
            
            # Check for overfitting
            train_val_gap = train_results['accuracy'] - val_results['accuracy']
            if train_val_gap > 0.2:
                print(f"Warning: Overfitting detected! Train-Val gap: {train_val_gap:.3f}")
            else:
                print(f"Good generalisation. Train-Val gap: {train_val_gap:.3f}")
            
            # Update history
            self.history['train_loss'].append(train_results['loss'])
            self.history['val_loss'].append(val_results['loss'])
            self.history['train_curse_score'].append(train_results['curse_score'])
            self.history['val_curse_score'].append(val_results['curse_score'])
            self.history['train_acc'].append(train_results['accuracy'])
            self.history['val_acc'].append(val_results['accuracy'])
            
            # Early stopping based on curse resistance score
            if val_results['curse_score'] > self.best_val_curse_score:
                self.best_val_curse_score = val_results['curse_score']
                self.best_model_state = self.processor.state_dict().copy()
                self.patience_counter = 0
                print(f"New best validation curse score: {self.best_val_curse_score:.4f}")
            else:
                self.patience_counter += 1
                print(f"No improvement. Patience: {self.patience_counter}/{patience}")
            
            if self.patience_counter >= patience:
                print("Early stopping triggered!")
                break
            
            print()
        
        # Load best model
        if self.best_model_state is not None:
            self.processor.load_state_dict(self.best_model_state)
            print(f"Loaded best model (curse score: {self.best_val_curse_score:.4f})")
        
        return {
            'best_curse_score': self.best_val_curse_score,
            'history': self.history,
            'model': self.processor
        }


def load_plant_data(data_dir: str, max_samples_per_class: int = None) -> tuple:
    """Load plant images for curse-resistant feature learning."""
    data_path = Path(data_dir)
    
    # Check if data directory exists
    if not data_path.exists():
        logger.error(f"Data directory does not exist: {data_path}")
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    logger.info(f"Looking for plant classes in: {data_path.absolute()}")
    
    # Image preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Find class directories
    class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    class_dirs.sort()
    class_names = [d.name for d in class_dirs]
    
    logger.info(f"Found {len(class_names)} plant classes")
    if len(class_names) > 0:
        logger.info(f"Class names: {class_names[:10]}...")  # Show first 10
    
    if len(class_names) == 0:
        logger.error(f"No class directories found in {data_path}")
        # List what's actually in the directory
        all_items = list(data_path.iterdir())
        logger.error(f"Directory contents: {[item.name for item in all_items]}")
        raise ValueError("No plant class directories found!")
    
    images = []
    labels = []
    
    for class_idx, class_dir in enumerate(class_dirs):
        logger.info(f"Processing class {class_idx + 1}/{len(class_dirs)}: {class_dir.name}")
        
        # Find image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        class_images = [
            f for f in class_dir.iterdir() 
            if f.suffix.lower() in image_extensions
        ]
        
        logger.info(f"  Found {len(class_images)} images in {class_dir.name}")
        
        if len(class_images) == 0:
            logger.warning(f"  No images found in class directory: {class_dir}")
            # List what's actually in the class directory
            all_files = list(class_dir.iterdir())
            logger.warning(f"  Directory contents: {[f.name for f in all_files[:5]]}")
            continue
        
        # Sample images if necessary
        if max_samples_per_class and len(class_images) > max_samples_per_class:
            class_images = random.sample(class_images, max_samples_per_class)
            logger.info(f"  Sampled {len(class_images)} images from {class_dir.name}")
        
        # Load and process images
        loaded_count = 0
        for img_path in class_images:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                images.append(img_tensor)
                labels.append(class_idx)
                loaded_count += 1
            except Exception as e:
                logger.warning(f"Failed to load {img_path}: {e}")
                continue
        
        logger.info(f"  Successfully loaded {loaded_count} images from {class_dir.name}")
    
    if len(images) == 0:
        logger.error("No images were successfully loaded!")
        raise ValueError("No images could be loaded from any class!")
    
    image_tensor = torch.stack(images)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    
    logger.info(f"Total: {len(images)} images from {len(class_names)} classes")
    return image_tensor, label_tensor, class_names


def extract_foundation_features(images: torch.Tensor, batch_size: int = 64) -> torch.Tensor:
    """Extract foundation features from plant images."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialise foundation feature extractor
    foundation_extractor = MultiExtractorFoundation(
        pretrained=True, freeze_extractors=True
    ).to(device)
    foundation_extractor.eval()
    
    all_features = []
    
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size].to(device)
            batch_features = foundation_extractor.extract_foundation_features(batch_images)
            all_features.append(batch_features.cpu())
    
    return torch.cat(all_features, dim=0)


def create_data_loaders(features: torch.Tensor, labels: torch.Tensor, 
                       batch_size: int = 64) -> tuple:
    """Create stratified train and validation data loaders."""
    # Convert to numpy for stratified split
    features_np = features.numpy()
    labels_np = labels.numpy()
    
    # Stratified train/validation split (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        features_np, labels_np, 
        test_size=0.2, 
        stratify=labels_np,
        random_state=42
    )
    
    # Convert back to tensors
    train_features = torch.from_numpy(X_train)
    val_features = torch.from_numpy(X_val)
    train_labels = torch.from_numpy(y_train)
    val_labels = torch.from_numpy(y_val)
    
    # Create datasets
    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Log class distributions
    train_class_counts = torch.bincount(train_labels)
    val_class_counts = torch.bincount(val_labels)
    logger.info(f"Training: {len(train_dataset)} samples, class distribution: {train_class_counts.tolist()}")
    logger.info(f"Validation: {len(val_dataset)} samples, class distribution: {val_class_counts.tolist()}")
    
    return train_loader, val_loader, train_features, train_labels


def evaluate_curse_resistance(foundation_features: torch.Tensor, labels: torch.Tensor,
                             processor: CurseResistantProcessor) -> dict:
    """Evaluate curse resistance improvement after processing."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    curse_metrics = CurseResistanceMetrics()
    
    foundation_features = foundation_features.to(device)
    labels = labels.to(device)
    processor.eval()
    
    # Evaluate foundation features
    foundation_metrics = curse_metrics.evaluate_batch(foundation_features, labels)
    
    # Evaluate enhanced features
    with torch.no_grad():
        enhanced_features = processor(foundation_features)
    enhanced_metrics = curse_metrics.evaluate_batch(enhanced_features, labels)
    
    # Calculate improvement
    curse_improvement = enhanced_metrics['curse_resistance_score'] - foundation_metrics['curse_resistance_score']
    
    return {
        'foundation_score': foundation_metrics['curse_resistance_score'],
        'enhanced_score': enhanced_metrics['curse_resistance_score'],
        'improvement': curse_improvement,
        'improvement_percent': (curse_improvement / foundation_metrics['curse_resistance_score']) * 100
    }


def main():
    parser = argparse.ArgumentParser(description='Curse-Resistant Feature Learning Training')
    parser.add_argument('--data-dir', type=str, default='../../data/plant_images',
                       help='Directory containing plant images')
    parser.add_argument('--max-samples-per-class', type=int, default=None,
                       help='Maximum samples per class (None = use all images)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=5e-4,
                       help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='./training_output',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    print("\n" + "="*70)
    print("CURSE-RESISTANT FEATURE LEARNING FOR PLANT CLASSIFICATION")
    print("="*70)
    print("Overfitting prevention techniques:")
    print("  - Gaussian noise injection (5% std)")
    print("  - Feature dropout (15% dropout)")
    print("  - Network dropout (40% dropout)")
    print("  - Strong weight decay (1e-2)")
    print("  - Mixed precision training")
    print("  - Stratified data splitting")
    print("="*70)
    print()
    
    # Load data
    logger.info("Loading plant image data...")
    images, labels, class_names = load_plant_data(
        args.data_dir, args.max_samples_per_class
    )
    
    # Extract foundation features
    logger.info("Extracting foundation features...")
    # NOTE: Foundation features are extracted BEFORE train/val split
    # This is correct - we're not using any labels during feature extraction
    # The pre-trained models are frozen and don't learn from our data
    foundation_features = extract_foundation_features(images, args.batch_size)
    
    # Create data loaders with stratified splitting
    # CRITICAL: Data split happens AFTER feature extraction to prevent leakage
    train_loader, val_loader, train_features_only, train_labels_only = create_data_loaders(
        foundation_features, labels, args.batch_size
    )
    
    # Initialise trainer
    trainer = CurseResistantTrainer()
    
    # Execute training
    logger.info("Starting curse-resistant feature learning...")
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=len(class_names),
        epochs=args.epochs,
        patience=20
    )
    
    # Evaluate final curse resistance
    logger.info("Evaluating curse resistance improvement...")
    evaluation = evaluate_curse_resistance(
        foundation_features, labels, trainer.processor
    )
    
    # Display results
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best validation curse score: {results['best_curse_score']:.4f}")
    print(f"Foundation features curse score: {evaluation['foundation_score']:.4f}")
    print(f"Enhanced features curse score: {evaluation['enhanced_score']:.4f}")
    print(f"Curse resistance improvement: {evaluation['improvement']:+.4f}")
    print(f"Relative improvement: {evaluation['improvement_percent']:+.2f}%")
    
    # Grade the results based on curse resistance score
    if evaluation['enhanced_score'] >= 0.9:
        grade = "A+ (Excellent)"
    elif evaluation['enhanced_score'] >= 0.85:
        grade = "A (Very Good)"
    elif evaluation['enhanced_score'] >= 0.8:
        grade = "B+ (Good)"
    elif evaluation['enhanced_score'] >= 0.75:
        grade = "B (Satisfactory)"
    else:
        grade = "C (Needs Improvement)"
    
    print(f"Final grade: {grade}")
    print("="*60)
    
    # Save results
    save_data = {
        'best_curse_score': results['best_curse_score'],
        'foundation_score': evaluation['foundation_score'],
        'enhanced_score': evaluation['enhanced_score'],
        'improvement': evaluation['improvement'],
        'improvement_percent': evaluation['improvement_percent'],
        'grade': grade,
        'history': results['history'],
        'args': vars(args)
    }
    
    # Save model and results
    torch.save(trainer.processor.state_dict(), output_dir / 'curse_resistant_processor.pth')
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")
    
    return results


if __name__ == "__main__":
    main() 