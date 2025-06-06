#!/usr/bin/env python3
"""
Speed-Optimized Curse-Resistant Feature Learning Training System

Key optimizations:
1. Reduced curse metrics calculation frequency (every 10 batches)
2. Better memory management and cleanup
3. Batch timing monitoring  
4. Error-resistant mutual information calculation
5. Fallback mechanisms for edge cases

This version should achieve 2-3 minute epochs instead of 30 minutes.
"""

import os
import sys
import torch
import torch.nn as nn
import argparse
import logging
from pathlib import Path

# Import the original training system
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from train_curse_resistant import (
    CurseResistantTrainer, 
    load_plant_data, 
    extract_foundation_features,
    create_data_loaders
)

def optimized_training_run():
    """Run optimized training with performance monitoring."""
    
    print("\n" + "="*70)
    print("SPEED-OPTIMIZED CURSE-RESISTANT FEATURE LEARNING")
    print("="*70)
    print("Performance optimizations:")
    print("  - Curse metrics calculated every 10 training batches")
    print("  - Curse metrics calculated every 5 validation batches")
    print("  - Aggressive memory cleanup after each epoch")
    print("  - Batch timing monitoring")
    print("  - Error-resistant mutual information calculation")
    print("  - Small batch fallback for edge cases")
    print("="*70)
    print()
    
    # Configuration for speed
    config = {
        'data_dir': '../../data/plant_images',
        'max_samples_per_class': None,  # Use all available
        'epochs': 100,
        'batch_size': 32,  # Keep reasonable batch size
        'learning_rate': 5e-4,
        'output_dir': './training_output_optimized'
    }
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading plant image data...")
    images, labels, class_names = load_plant_data(
        config['data_dir'], config['max_samples_per_class']
    )
    
    print(f"Loaded {len(images)} images from {len(class_names)} classes")
    
    # Extract foundation features
    print("Extracting foundation features...")
    import time
    feature_start = time.time()
    foundation_features = extract_foundation_features(images, config['batch_size'])
    feature_time = time.time() - feature_start
    print(f"Feature extraction completed in {feature_time:.1f}s")
    
    # Clean up images to free memory
    del images
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Create data loaders
    print("Creating stratified data loaders...")
    train_loader, val_loader, _, _ = create_data_loaders(
        foundation_features, labels, config['batch_size']
    )
    
    # Initialize trainer
    trainer = CurseResistantTrainer()
    
    # Execute training with timing
    print("Starting optimized training...")
    training_start = time.time()
    
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=len(class_names),
        epochs=config['epochs'],
        patience=20
    )
    
    training_time = time.time() - training_start
    
    # Display performance results
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Total training time: {training_time:.1f}s ({training_time/60:.1f} minutes)")
    print(f"Average time per epoch: {training_time/len(results['history']['train_loss']):.1f}s")
    print(f"Feature extraction time: {feature_time:.1f}s")
    print(f"Best validation curse score: {results['best_curse_score']:.4f}")
    
    # Performance grade
    if training_time < 600:  # Less than 10 minutes
        perf_grade = "Excellent - Under 10 minutes"
    elif training_time < 1200:  # Less than 20 minutes  
        perf_grade = "Good - Under 20 minutes"
    elif training_time < 1800:  # Less than 30 minutes
        perf_grade = "Fair - Under 30 minutes"
    else:
        perf_grade = "Poor - Over 30 minutes"
    
    print(f"Performance grade: {perf_grade}")
    print("="*60)
    
    # Save optimized results
    torch.save(trainer.processor.state_dict(), output_dir / 'optimized_processor.pth')
    
    import json
    perf_data = {
        'total_time_seconds': training_time,
        'total_time_minutes': training_time / 60,
        'avg_epoch_time': training_time / len(results['history']['train_loss']),
        'feature_extraction_time': feature_time,
        'best_curse_score': results['best_curse_score'],
        'performance_grade': perf_grade,
        'config': config
    }
    
    with open(output_dir / 'performance_results.json', 'w') as f:
        json.dump(perf_data, f, indent=2)
    
    print(f"Optimized results saved to {output_dir}")
    
    return results, perf_data

if __name__ == "__main__":
    optimized_training_run() 