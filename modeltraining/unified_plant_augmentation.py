import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import random
import numpy as np

class UnifiedPlantAugmentationEngine:
    """
    Simplified augmentation engine using torchvision transforms
    """
    def __init__(self, image_size=512, device='cuda'):
        self.image_size = image_size
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Realistic augmentations (60%)
        self.realistic_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomResizedCrop((image_size, image_size), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Seasonal/edge augmentations (40%)
        self.seasonal_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.2),
            transforms.RandomResizedCrop((image_size, image_size), scale=(0.7, 1.0), ratio=(0.8, 1.2)),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def generate_single_augmentation(self, img, method="realistic"):
        """
        Generate a single augmented image
        
        Args:
            img: PIL Image or numpy array
            method: "realistic" or "seasonal"
        
        Returns:
            Augmented image as tensor
        """
        try:
            # Validate input image
            if img is None:
                raise ValueError("Input image is None")
            
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            elif not isinstance(img, Image.Image):
                raise ValueError("Input must be PIL Image or numpy array")
            
            # Validate method
            if method not in ["realistic", "seasonal"]:
                print(f"Warning: Invalid method '{method}', using 'realistic'")
                method = "realistic"
            
            if method == "realistic":
                tensor_img = self.realistic_transforms(img)
            elif method == "seasonal":
                tensor_img = self.seasonal_transforms(img)
            else:
                # Default to realistic
                tensor_img = self.realistic_transforms(img)
            
            # Move to specified device
            return tensor_img.to(self.device)
        except Exception as e:
            print(f"Error in generate_single_augmentation: {e}")
            # Return a zero tensor as fallback on the correct device
            return torch.zeros(3, self.image_size, self.image_size, device=self.device)
    
    def generate_batch_augmentations(self, img, count=30, realistic_frac=0.6):
        """
        Generate multiple augmentations for an image
        
        Args:
            img: PIL Image or numpy array
            count: Number of augmentations to generate
            realistic_frac: Fraction of realistic vs seasonal augmentations
        
        Returns:
            List of augmented images as tensors
        """
        try:
            # Validate parameters
            if not isinstance(count, int) or count <= 0:
                raise ValueError("Count must be a positive integer")
            
            if not isinstance(realistic_frac, (int, float)) or realistic_frac < 0 or realistic_frac > 1:
                raise ValueError("realistic_frac must be between 0 and 1")
            
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            
            augmented = []
            realistic_count = int(count * realistic_frac)
            seasonal_count = count - realistic_count
            
            # Generate realistic augmentations
            for _ in range(realistic_count):
                aug = self.generate_single_augmentation(img, "realistic")
                augmented.append(aug)
            
            # Generate seasonal augmentations
            for _ in range(seasonal_count):
                aug = self.generate_single_augmentation(img, "seasonal")
                augmented.append(aug)
            
            return augmented
        except Exception as e:
            print(f"Error in generate_batch_augmentations: {e}")
            return []

def create_augmentation_engine(use_gpu=True, enable_all_features=True, **kwargs):
    """
    Factory function to create augmentation engine
    """
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    return UnifiedPlantAugmentationEngine(device=device, **kwargs) 