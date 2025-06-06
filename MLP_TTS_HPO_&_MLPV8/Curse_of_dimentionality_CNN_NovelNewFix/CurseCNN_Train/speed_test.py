#!/usr/bin/env python3
"""
Quick speed test for curse-resistant feature learning optimizations.
"""

import torch
import time
import numpy as np
from pathlib import Path
import sys

# Add the curse_resistant_feature_learning to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / "curse_resistant_feature_learning"))

from metrics.curse_metrics import CurseResistanceMetrics

def test_curse_metrics_speed():
    """Test curse metrics calculation speed."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    curse_metrics = CurseResistanceMetrics()
    
    print("Testing curse metrics calculation speed...")
    print(f"Device: {device}")
    
    # Test different batch sizes
    batch_sizes = [8, 16, 32, 64]
    feature_dim = 1024
    num_classes = 154
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Create test data
        features = torch.randn(batch_size, feature_dim).to(device)
        labels = torch.randint(0, num_classes, (batch_size,)).to(device)
        
        # Time the calculation
        times = []
        for _ in range(5):
            start_time = time.time()
            metrics = curse_metrics.evaluate_batch(features, labels)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        print(f"  Average time: {avg_time:.3f}s")
        print(f"  Curse score: {metrics['curse_resistance_score']:.4f}")
    
    print("\nSpeed test complete!")

if __name__ == "__main__":
    test_curse_metrics_speed() 