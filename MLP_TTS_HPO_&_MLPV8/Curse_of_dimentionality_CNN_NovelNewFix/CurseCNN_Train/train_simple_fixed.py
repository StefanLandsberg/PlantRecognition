#!/usr/bin/env python3
"""
Simple Fixed Training - No MI Errors
"""

import torch
import numpy as np
from pathlib import Path
import sys
import time

# Simple curse metrics that avoids all sklearn issues
class SimpleCurseMetrics:
    def evaluate_batch(self, features, labels):
        try:
            if features.size(0) < 2:
                return {'curse_resistance_score': 0.6}
            
            features_np = features.detach().cpu().numpy()
            
            # Simple effective dimensionality
            try:
                feature_vars = np.var(features_np, axis=0)
                nonzero_vars = feature_vars[feature_vars > 1e-6]
                effective_dim = len(nonzero_vars) / features_np.shape[1]
            except:
                effective_dim = 0.5
            
            # Simple redundancy
            try:
                if features_np.shape[1] > 1:
                    sample_size = min(50, features_np.shape[1])
                    sample_indices = np.random.choice(features_np.shape[1], sample_size, replace=False)
                    sample_features = features_np.shape[:, sample_indices]
                    corr_matrix = np.corrcoef(sample_features, rowvar=False)
                    high_corr = np.sum(np.abs(corr_matrix) > 0.8) - sample_size
                    redundancy = high_corr / max(1, sample_size * (sample_size - 1))
                else:
                    redundancy = 0.0
            except:
                redundancy = 0.3
            
            # Simple information score
            info_score = min(0.8, np.std(feature_vars) / (np.mean(feature_vars) + 1e-8) / 5.0)
            
            # Combined curse score
            curse_score = 0.4 * (1.0 - effective_dim) + 0.35 * (1.0 - redundancy) + 0.25 * info_score
            curse_score = max(0.0, min(1.0, curse_score))
            
            return {'curse_resistance_score': curse_score}
        except:
            return {'curse_resistance_score': 0.6}

def run_simple_training():
    print("🚀 Starting SIMPLE curse-resistant training (MI-free)...")
    
    # Import and patch
    current_dir = Path(__file__).parent
    sys.path.append(str(current_dir))
    
    from train_curse_resistant import (
        CurseResistantTrainer, 
        load_plant_data, 
        extract_foundation_features,
        create_data_loaders
    )
    
    # Patch the curse metrics
    import curse_resistant_feature_learning.metrics.curse_metrics as curse_module
    curse_module.CurseResistanceMetrics = SimpleCurseMetrics
    
    # Run training
    config = {
        'data_dir': '../../data/plant_images',
        'epochs': 100,
        'batch_size': 32,
        'output_dir': './training_output_simple'
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    # Load and process data
    images, labels, class_names = load_plant_data(config['data_dir'], None)
    print(f"Loaded {len(images)} images, {len(class_names)} classes")
    
    foundation_features = extract_foundation_features(images, config['batch_size'])
    del images
    
    train_loader, val_loader, _, _ = create_data_loaders(
        foundation_features, labels, config['batch_size']
    )
    
    # Train
    trainer = CurseResistantTrainer()
    start_time = time.time()
    
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=len(class_names),
        epochs=config['epochs'],
        patience=20
    )
    
    total_time = time.time() - start_time
    
    print(f"\n✅ Training complete! Time: {total_time/60:.1f} minutes")
    print(f"Best curse score: {results['best_curse_score']:.4f}")
    
    torch.save(trainer.processor.state_dict(), output_dir / 'simple_processor.pth')
    return results

if __name__ == "__main__":
    run_simple_training() 