#!/usr/bin/env python3
"""
FAST Curse-Resistant Feature Learning - Optimized for Speed

This script fixes the mutual information errors and speeds up training:
- No more "Found array with 0 sample(s)" errors
- Epochs should take 2-3 minutes instead of 30 minutes
- All optimizations applied
"""

import sys
from pathlib import Path

# Apply the patch first to fix MI errors
print("🔧 Applying curse metrics patch...")
exec(open(Path(__file__).parent / "curse_metrics_patch.py").read())

# Now import and run optimized training
from train_optimized_speed import optimized_training_run

if __name__ == "__main__":
    print("\n🚀 Starting FAST curse-resistant training...")
    print("   - Mutual information errors fixed")
    print("   - Speed optimizations enabled")
    print("   - Expected epoch time: 2-3 minutes")
    print("=" * 60)
    
    # Run the optimized training
    results, performance = optimized_training_run()
    
    print("\n🎉 FAST training complete!")
    print(f"   - Total time: {performance['total_time_minutes']:.1f} minutes")
    print(f"   - Performance grade: {performance['performance_grade']}")
    print(f"   - Best curse score: {performance['best_curse_score']:.4f}") 