import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import time
from pathlib import Path
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from plant_feature_extractor import create_feature_extractor, PlantFeatureExtractor, HandcraftedFeatureExtractor
from unified_plant_augmentation import UnifiedPlantAugmentationEngine

def test_feature_extractor():
    """Test the feature extractor with dummy data"""
    print("Testing Feature Extractor...")
    
    # Create dummy image
    dummy_image = torch.randn(3, 224, 224)
    
    # Test CNN extractor
    print("  Testing CNN extractor...")
    cnn_extractor = create_feature_extractor('cnn', out_features=1024)
    cnn_extractor.eval()
    
    with torch.no_grad():
        start_time = time.time()
        features = cnn_extractor(dummy_image)
        end_time = time.time()
        
        print(f"    CNN features shape: {features.shape}")
        print(f"    CNN extraction time: {(end_time - start_time)*1000:.2f}ms")
        print(f"    CNN feature stats - Mean: {features.mean():.4f}, Std: {features.std():.4f}")
    
    # Test handcrafted extractor
    print("  Testing handcrafted extractor...")
    handcrafted_extractor = create_feature_extractor('handcrafted', out_features=1024)
    
    # Convert to PIL for handcrafted
    dummy_pil = Image.fromarray((dummy_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    
    start_time = time.time()
    features = handcrafted_extractor.extract(dummy_pil)
    end_time = time.time()
    
    print(f"    Handcrafted features shape: {features.shape}")
    print(f"    Handcrafted extraction time: {(end_time - start_time)*1000:.2f}ms")
    print(f"    Handcrafted feature stats - Mean: {features.mean():.4f}, Std: {features.std():.4f}")
    
    return True

def test_augmentation():
    """Test the augmentation engine"""
    print("\nTesting Augmentation Engine...")
    
    # Create dummy image
    dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    # Create augmentation engine
    aug_engine = UnifiedPlantAugmentationEngine(image_size=224)
    
    # Test realistic augmentation
    print("  Testing realistic augmentation...")
    start_time = time.time()
    realistic_aug = aug_engine.generate_single_augmentation(dummy_image, "realistic")
    end_time = time.time()
    
    print(f"    Realistic aug shape: {realistic_aug.shape}")
    print(f"    Realistic aug time: {(end_time - start_time)*1000:.2f}ms")
    
    # Test seasonal augmentation
    print("  Testing seasonal augmentation...")
    start_time = time.time()
    seasonal_aug = aug_engine.generate_single_augmentation(dummy_image, "seasonal")
    end_time = time.time()
    
    print(f"    Seasonal aug shape: {seasonal_aug.shape}")
    print(f"    Seasonal aug time: {(end_time - start_time)*1000:.2f}ms")
    
    return True

def test_training_pipeline():
    """Test the training pipeline components"""
    print("\nTesting Training Pipeline...")
    
    # Test MLP head
    from training import MLPHead
    
    # Create dummy features
    dummy_features = torch.randn(8, 1024)  # Batch of 8, 1024 features
    dummy_labels = torch.randint(0, 100, (8,))  # 100 classes
    
    # Create MLP head
    mlp_head = MLPHead(in_dim=1024, num_classes=100)
    
    # Test forward pass
    print("  Testing MLP head...")
    start_time = time.time()
    logits = mlp_head(dummy_features)
    end_time = time.time()
    
    print(f"    MLP output shape: {logits.shape}")
    print(f"    MLP forward time: {(end_time - start_time)*1000:.2f}ms")
    
    # Test loss calculation
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, dummy_labels)
    print(f"    Loss value: {loss.item():.4f}")
    
    return True

def test_data_paths():
    """Test if data paths exist"""
    print("\nTesting Data Paths...")
    
    # Check various possible data paths
    possible_paths = [
        "../data/plant_images",
        "data/plant_images", 
        "../data_preprocessing/data",
        "data_preprocessing/data"
    ]
    
    for path in possible_paths:
        path_obj = Path(path)
        if path_obj.exists():
            print(f"  ✅ Found data path: {path}")
            if path_obj.is_dir():
                # Count subdirectories (classes)
                subdirs = [d for d in path_obj.iterdir() if d.is_dir()]
                print(f"    Classes found: {len(subdirs)}")
                if subdirs:
                    # Count images in first class
                    first_class = subdirs[0]
                    images = list(first_class.glob("*.jpg")) + list(first_class.glob("*.png"))
                    print(f"    Images in first class: {len(images)}")
        else:
            print(f"  ❌ Data path not found: {path}")
    
    return True

def main():
    """Run all tests"""
    print("=" * 50)
    print("PLANT RECOGNITION MODEL TESTING")
    print("=" * 50)
    
    try:
        # Test feature extractor
        test_feature_extractor()
        
        # Test augmentation
        test_augmentation()
        
        # Test training pipeline
        test_training_pipeline()
        
        # Test data paths
        test_data_paths()
        
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main() 