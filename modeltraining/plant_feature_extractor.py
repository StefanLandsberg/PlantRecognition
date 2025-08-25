import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import torchvision.models as models
import torchvision.transforms as transforms

class PlantFeatureExtractor(nn.Module):
    """
    Lightweight CNN-based feature extractor for plant recognition
    Optimized for 1024 features with 20ms processing time
    """
    
    def __init__(self, out_features=1024, input_channels=3):
        super(PlantFeatureExtractor, self).__init__()
        self.out_features = out_features
        
        # Efficient CNN backbone
        self.features = nn.Sequential(
            # Block 1: 3 -> 32
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Feature projection to exact output size
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, out_features)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for better training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Extract features from input images"""
        # Ensure input is in correct format
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # Add batch dimension
        
        # Ensure input is on the same device as the model
        if next(self.parameters()).is_cuda and not x.is_cuda:
            x = x.cuda()
        
        # Extract features
        features = self.features(x)
        features = features.view(features.size(0), -1)  # Flatten
        
        # Project to final feature space
        output = self.classifier(features)
        
        return output
    
    def extract_features(self, x):
        """Extract features without final classification layer"""
        # Ensure input is on the same device as the model
        if next(self.parameters()).is_cuda and not x.is_cuda:
            x = x.cuda()
            
        features = self.features(x)
        features = features.view(features.size(0), -1)
        return features

class HandcraftedFeatureExtractor:
    """
    Fast handcrafted feature extractor as fallback
    Extracts 1024 features using traditional computer vision techniques
    """
    
    def __init__(self, out_features=1024):
        self.out_features = out_features
    
    def extract(self, image):
        """
        Extract handcrafted features from image
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            numpy array of features
        """
        import cv2
        
        # Convert to numpy if needed
        if hasattr(image, 'numpy'):
            image = image.numpy()
        if len(image.shape) == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))  # CHW -> HWC
        
        # Convert to uint8 for OpenCV
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Convert to grayscale for some features
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        features = []
        
        # 1. Color features (256)
        # RGB histograms
        for i in range(3):
            hist = cv2.calcHist([image], [i], None, [64], [0, 256]).flatten()
            features.extend(hist)
        
        # 2. Texture features (256)
        # Sobel edge detection
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Magnitude histogram
        mag_hist = np.histogram(magnitude, bins=128, range=(0, 1000))[0]
        features.extend(mag_hist)
        
        # Direction histogram
        direction = np.arctan2(sobely, sobelx)
        dir_hist = np.histogram(direction, bins=128, range=(-np.pi, np.pi))[0]
        features.extend(dir_hist)
        
        # 3. Shape features (256)
        # Threshold for shape analysis
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Use largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Shape features
            shape_features = [area, perimeter]
            
            # Compactness
            if area > 0:
                compactness = (perimeter**2) / (4 * np.pi * area)
                shape_features.append(compactness)
            else:
                shape_features.append(0)
            
            # Fill remaining with zeros
            shape_features.extend([0] * 253)
        else:
            shape_features = [0] * 256
        
        features.extend(shape_features)
        
        # 4. Pattern features (256)
        # Local Binary Pattern approximation
        lbp = np.zeros_like(gray)
        for i in range(1, gray.shape[0]-1):
            for j in range(1, gray.shape[1]-1):
                center = gray[i, j]
                code = 0
                for k, (di, dj) in enumerate([(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]):
                    if gray[i+di, j+dj] > center:
                        code += 2**k
                lbp[i, j] = code
        
        # LBP histogram
        lbp_hist = np.histogram(lbp, bins=256, range=(0, 256))[0]
        features.extend(lbp_hist)
        
        # Convert to numpy array and ensure correct size
        features = np.array(features, dtype=np.float32)
        
        # Pad or truncate to exact size
        if len(features) < self.out_features:
            padding = np.zeros(self.out_features - len(features))
            features = np.concatenate([features, padding])
        elif len(features) > self.out_features:
            features = features[:self.out_features]
        
        return features

class MultiModelEnsembleExtractor(nn.Module):
    """Multi-model ensemble for comprehensive feature extraction"""
    
    def __init__(self):
        super().__init__()
        
        # Load pre-trained models
        self.resnet50 = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        self.efficientnet = models.efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT')
        self.densenet = models.densenet121(weights='DenseNet121_Weights.DEFAULT')
        self.vgg16 = models.vgg16(weights='VGG16_Weights.DEFAULT')
        
        # Remove final classification layers
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])
        self.efficientnet = nn.Sequential(*list(self.efficientnet.children())[:-1])
        self.densenet = nn.Sequential(*list(self.densenet.children())[:-1])
        self.vgg16 = nn.Sequential(*list(self.vgg16.children())[:-1])
        
        # Feature dimensions from each model
        self.feature_dims = {
            'resnet50': 2048,
            'efficientnet': 1280,
            'densenet': 1024,
            'vgg16': 512
        }
        
        self.total_features = sum(self.feature_dims.values())  # 2048+1280+1024+512 = 4864
        
        # Set all models to eval mode
        for model in [self.resnet50, self.efficientnet, self.densenet, self.vgg16]:
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
    
    def extract_features(self, x):
        """Extract features from all models and concatenate"""
        with torch.no_grad():
            # ResNet50 features
            resnet_features = self.resnet50(x)
            resnet_features = F.adaptive_avg_pool2d(resnet_features, (1, 1)).flatten(1)
            
            # EfficientNet features
            eff_features = self.efficientnet(x)
            eff_features = F.adaptive_avg_pool2d(eff_features, (1, 1)).flatten(1)
            
            # DenseNet features
            dense_features = self.densenet(x)
            dense_features = F.adaptive_avg_pool2d(dense_features, (1, 1)).flatten(1)
            
            # VGG16 features
            vgg_features = self.vgg16(x)
            vgg_features = F.adaptive_avg_pool2d(vgg_features, (1, 1)).flatten(1)
            
            # Concatenate all features
            all_features = torch.cat([
                resnet_features,
                eff_features,
                dense_features,
                vgg_features
            ], dim=1)
            
            return all_features

def create_feature_extractor(extractor_type='ensemble', **kwargs):
    """
    Factory function to create feature extractor
    
    Args:
        extractor_type: 'ensemble', 'cnn' or 'handcrafted'
        **kwargs: Additional arguments for the extractor
        
    Returns:
        Feature extractor instance
    """
    if extractor_type == 'ensemble':
        return MultiModelEnsembleExtractor()
    elif extractor_type == 'cnn':
        return PlantFeatureExtractor(**kwargs)
    elif extractor_type == 'handcrafted':
        return HandcraftedFeatureExtractor(**kwargs)
    else:
        raise ValueError(f"Unknown extractor type: {extractor_type}") 