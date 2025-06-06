#!/usr/bin/env python3
"""
Multi-Extractor Foundation

Combines proven architectures (EfficientNet B4, ViT Base, ConvNeXt Large) 
for robust feature extraction as the foundation for curse-resistant learning.
"""

import torch
import torch.nn as nn
import timm
import logging
from typing import Dict, List, Tuple, Optional
import warnings
import torch.nn.functional as F

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class MultiExtractorFoundation(nn.Module):
    """
    Foundation feature extraction using multiple proven architectures.
    
    Combines:
    - EfficientNet B4: 1792 features
    - ViT Large Patch16: 1024 features (512x512 support)
    - ConvNeXt Large: 1536 features
    Total: 4352 features
    """
    
    def __init__(self, pretrained: bool = True, freeze_extractors: bool = True):
        super().__init__()
        
        self.freeze_extractors = freeze_extractors
        
        # Feature dimensions for each extractor (known dimensions)
        self.feature_dims = {
            'efficientnet_b4': 1792,
            'vit_large_patch16_224': 1024,  # ViT Large - can be resized for 512x512
            'convnext_large': 1536
        }
        self.total_features = sum(self.feature_dims.values())  # 4352
        
        # Initialize extractors
        self.extractors = nn.ModuleDict()
        self._create_extractors(pretrained)
        
        # Freeze extractors if requested
        if freeze_extractors:
            self._freeze_extractors()
        
        logging.info(f"MultiExtractorFoundation initialized with {self.total_features} total features")
        logging.info(f"Feature dimensions: {self.feature_dims}")
    
    def _create_extractors(self, pretrained: bool):
        """Create the three foundation extractors."""
        
        try:
            # EfficientNet B4
            logging.info("Loading EfficientNet B4...")
            self.extractors['efficientnet_b4'] = timm.create_model(
                'efficientnet_b4',
                pretrained=pretrained,
                num_classes=0,  # Remove classifier head
                global_pool='avg'
            )
            
            # ViT Large Patch16 (can handle variable input sizes)
            logging.info("Loading ViT Large Patch16...")
            self.extractors['vit_large_patch16_224'] = timm.create_model(
                'vit_large_patch16_224',
                pretrained=pretrained,
                num_classes=0,  # Remove classifier head
                img_size=512,   # Set to handle 512x512 images
                dynamic_img_size=True  # Allow dynamic image sizes
            )
            
            # ConvNeXt Large
            logging.info("Loading ConvNeXt Large...")
            self.extractors['convnext_large'] = timm.create_model(
                'convnext_large',
                pretrained=pretrained,
                num_classes=0,  # Remove classifier head
                global_pool='avg'
            )
            
            logging.info("All extractors loaded successfully!")
            
        except Exception as e:
            logging.error(f"Error loading extractors: {e}")
            raise
    
    def _freeze_extractors(self):
        """Freeze all extractor parameters."""
        for name, extractor in self.extractors.items():
            for param in extractor.parameters():
                param.requires_grad = False
            logging.info(f"Frozen {name} parameters")
    
    def extract_foundation_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from all foundation models.
        
        Args:
            images: Input images tensor [batch_size, channels, height, width]
            
        Returns:
            Concatenated features tensor [batch_size, total_features]
        """
        if images.dim() != 4:
            raise ValueError(f"Expected 4D input tensor, got {images.dim()}D")
        
        features = {}
        
        # Extract features from each model
        for name, extractor in self.extractors.items():
            try:
                with torch.no_grad():
                    feat = extractor(images)
                    
                    # Ensure 2D output
                    if feat.dim() > 2:
                        feat = feat.view(feat.size(0), -1)
                    
                    features[name] = feat
                    
            except Exception as e:
                logging.error(f"Error extracting features from {name}: {e}")
                raise
        
        # Concatenate all features
        feature_list = [features[name] for name in ['efficientnet_b4', 'vit_large_patch16_224', 'convnext_large']]
        concatenated_features = torch.cat(feature_list, dim=1)
        
        return concatenated_features
    
    def extract_individual_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features from each model individually.
        
        Args:
            images: Input images tensor
            
        Returns:
            Dictionary of features from each extractor
        """
        features = {}
        
        for name, extractor in self.extractors.items():
            with torch.no_grad():
                feat = extractor(images)
                if feat.dim() > 2:
                    feat = feat.view(feat.size(0), -1)
                features[name] = feat
        
        return features
    
    def get_feature_info(self) -> Dict[str, int]:
        """Get information about feature dimensions."""
        return {
            'individual_dims': self.feature_dims.copy(),
            'total_dims': self.total_features,
            'models': list(self.extractors.keys())
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - alias for extract_foundation_features."""
        return self.extract_foundation_features(x)


def create_foundation_extractor(pretrained: bool = True, freeze: bool = True) -> MultiExtractorFoundation:
    """
    Convenience function to create a MultiExtractorFoundation.
    
    Args:
        pretrained: Whether to use pretrained weights
        freeze: Whether to freeze extractor parameters
        
    Returns:
        Initialized MultiExtractorFoundation
    """
    return MultiExtractorFoundation(pretrained=pretrained, freeze_extractors=freeze)


def test_foundation_extractor():
    """Test the foundation extractor with dummy data."""
    print("Testing MultiExtractorFoundation...")
    
    # Create dummy input
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    
    # Create foundation
    foundation = create_foundation_extractor(pretrained=False)
    
    # Test extraction
    with torch.no_grad():
        features = foundation.extract_foundation_features(dummy_images)
        individual_features = foundation.extract_individual_features(dummy_images)
    
    print(f"Input shape: {dummy_images.shape}")
    print(f"Combined features shape: {features.shape}")
    print(f"Individual feature shapes:")
    for name, feat in individual_features.items():
        print(f"  {name}: {feat.shape}")
    
    # Test feature info
    info = foundation.get_feature_info()
    print(f"Feature info: {info}")
    
    print("Foundation extractor test passed!")


if __name__ == "__main__":
    test_foundation_extractor() 