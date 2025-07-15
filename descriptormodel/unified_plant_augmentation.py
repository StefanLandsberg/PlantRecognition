"""
Unified Plant Augmentation System - Complete Modular Augmentation Engine

This script consolidates ALL augmentation methods from both main scripts into a single,
modular system that provides sophisticated plant-specific augmentations.

Features:
- All complex augmentation methods from the original scripts
- Modular design for easy integration
- GPU acceleration support
- memory management
- plant-specific transformations
"""

import numpy as np
import cv2
import torch
import random
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import concurrent.futures
import threading
from dataclasses import dataclass

# Configuration for GPU usage
GPU_AVAILABLE = torch.cuda.is_available()

@dataclass
class AugmentationConfig:
    """Configuration for augmentation parameters"""
    use_gpu: bool = GPU_AVAILABLE
    max_cache_size: int = 1000
    enable_advanced_transforms: bool = True
    enable_mixing_methods: bool = True
    enable_realistic_angles: bool = True
    enable_plant_specific: bool = True

class UnifiedPlantAugmentationEngine:
    """
    Unified Plant Augmentation Engine - Complete augmentation system
    
    Consolidates all augmentation methods from both scripts:
    - Seasonal effects (summer, autumn, winter, spring, drought, overwatered)
    - Lighting conditions (shadow, sunflare, over/under exposed, indoor, golden/blue hour)
    - Plant-specific transforms (wilt, curl, focus blur, growth stages, diseases, deficiencies)
    - Weather effects (rain, wind, dust, dew)
    - mixing (MixUp, CutMix, AugMix)
    - Realistic angle variations (natural rotation, camera perspective, botanical angles)
    - Mobile photography simulation
    - Delta-based feature augmentation
    """
    
    def __init__(self, config: AugmentationConfig = None):
        self.config = config or AugmentationConfig()
        self.device = torch.device('cuda')
        
        # Augmentation categories
        self.seasonal_effects = ["summer", "autumn", "winter", "spring", "drought", "overwatered"]
        self.lighting_conditions = ["shadow", "sunflare", "overexposed", "underexposed", "indoor", "golden_hour", "blue_hour"]
        self.plant_transforms = ["leaf_wilt", "leaf_curl", "focus_blur", "growth_stage", "disease_spots", "nutrient_deficiency", "pest_damage"]
        self.weather_effects = ["rain_drops", "wind_blur", "dust_particles", "morning_dew"]
        
        # Cache for augmentation
        self.augmentation_cache = {}
        self._lock = threading.Lock()
        
    def generate_single_augmentation(self, image: np.ndarray, method: str = "auto") -> np.ndarray:
        """
        Generate a single augmented image using the specified method
        
        Args:
            image: Input image (numpy array)
            method: Augmentation method ('auto', 'seasonal', 'lighting', 'plant', 'weather', 'angle', 'mix')
        
        Returns:
            Augmented image
        """
        if method == "auto":
            method = random.choice(['seasonal', 'lighting', 'plant', 'weather', 'angle', 'basic'])
        
        if method == 'seasonal':
            effect = random.choice(self.seasonal_effects)
            return self._apply_seasonal_effect(image, effect)
        elif method == 'lighting':
            effect = random.choice(self.lighting_conditions)
            return self._apply_lighting_condition(image, effect)
        elif method == 'plant':
            effect = random.choice(self.plant_transforms)
            return self._apply_plant_specific_transform(image, effect)
        elif method == 'weather':
            effect = random.choice(self.weather_effects)
            return self._apply_weather_effects(image, effect)
        elif method == 'angle':
            return self._apply_realistic_angle_transform(image)
        elif method == 'basic':
            return self._apply_basic_transform(image)
        else:
            return self._apply_basic_transform(image)
    
    def _apply_seasonal_effect(self, image: np.ndarray, season_type: str) -> np.ndarray:
        """Apply realistic seasonal effects to plant images"""
        img = image.astype(np.float32)
        
        if season_type == "summer":
            img = img * 1.15
            img[:,:,2] = img[:,:,2] * 0.92
            img[:,:,1] = img[:,:,1] * 1.08
        elif season_type == "autumn":
            img[:,:,0] = img[:,:,0] * 1.25
            img[:,:,1] = img[:,:,1] * 0.88
            img[:,:,2] = img[:,:,2] * 0.75
        elif season_type == "winter":
            img = img * 0.82
            img[:,:,0] = img[:,:,0] * 0.78
            img[:,:,1] = img[:,:,1] * 0.85
            img[:,:,2] = img[:,:,2] * 1.18
        elif season_type == "spring":
            img[:,:,1] = img[:,:,1] * 1.22
            img = img * 1.08
        elif season_type == "drought":
            img[:,:,0] = img[:,:,0] * 1.18
            img[:,:,1] = img[:,:,1] * 0.92
            img[:,:,2] = img[:,:,2] * 0.72
        elif season_type == "overwatered":
            img[:,:,0] = img[:,:,0] * 0.82
            img[:,:,1] = img[:,:,1] * 0.96
            img[:,:,2] = img[:,:,2] * 1.15
        
        return np.clip(img, 0, 255).astype(np.uint8)
    
    def _apply_lighting_condition(self, image: np.ndarray, light_type: str) -> np.ndarray:
        """Apply realistic lighting conditions"""
        img = image.astype(np.float32)
        h, w = img.shape[:2]
        
        if light_type == "shadow":
            for _ in range(np.random.randint(2, 6)):
                x, y = np.random.randint(0, w), np.random.randint(0, h)
                size = np.random.randint(20, min(h,w)//2)
                mask = np.zeros((h, w), dtype=np.float32)
                cv2.ellipse(mask, (x, y), (size, size//2), np.random.randint(0, 180), 0, 360, 1, -1)
                mask = cv2.GaussianBlur(mask, (15, 15), 0)
                for c in range(3):
                    img[:,:,c] = img[:,:,c] * (1 - 0.4 * mask)
        elif light_type == "sunflare":
            center_x, center_y = np.random.randint(w//4, 3*w//4), np.random.randint(h//4, 3*h//4)
            y, x = np.ogrid[-center_y:h-center_y, -center_x:w-center_x]
            dist = np.sqrt(x*x + y*y)
            flare_mask = np.exp(-(dist**2) / (2 * (min(h,w)/8)**2))
            for c in range(3):
                img[:,:,c] = np.clip(img[:,:,c] + flare_mask * 60, 0, 255)
        elif light_type == "overexposed":
            img = np.clip(img * 1.4 + 30, 0, 255)
        elif light_type == "underexposed":
            img = img * 0.6 - 20
        elif light_type == "indoor":
            img[:,:,0] = img[:,:,0] * 1.05
            img[:,:,1] = img[:,:,1] * 1.02
            img[:,:,2] = img[:,:,2] * 0.94
        elif light_type == "golden_hour":
            img[:,:,0] = img[:,:,0] * 1.15
            img[:,:,1] = img[:,:,1] * 1.08
            img[:,:,2] = img[:,:,2] * 0.88
        elif light_type == "blue_hour":
            img[:,:,0] = img[:,:,0] * 0.85
            img[:,:,1] = img[:,:,1] * 0.92
            img[:,:,2] = img[:,:,2] * 1.18
        
        return np.clip(img, 0, 255).astype(np.uint8)
    
    def _apply_plant_specific_transform(self, image: np.ndarray, transform_type: str) -> np.ndarray:
        """Apply plant biology-specific transformations"""
        img = image.copy()
        h, w = img.shape[:2]
        
        if transform_type == "leaf_wilt":
            rows, cols = img.shape[:2]
            map_y, map_x = np.mgrid[0:rows, 0:cols].astype(np.float32)
            center_col = cols // 2
            for i in range(cols):
                dist_factor = abs(i - center_col) / center_col
                map_y[:, i] += dist_factor * 12 * np.sin(np.pi * i / cols)
            img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
        elif transform_type == "leaf_curl":
            rows, cols = img.shape[:2]
            map_y, map_x = np.mgrid[0:rows, 0:cols].astype(np.float32)
            for i in range(rows):
                map_x[i,:] += 8 * np.sin(i/12)
            img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
        elif transform_type == "focus_blur":
            center_x = np.random.randint(w//3, 2*w//3)
            center_y = np.random.randint(h//3, 2*h//3)
            y, x = np.ogrid[-center_y:h-center_y, -center_x:w-center_x]
            dist = np.sqrt(x*x + y*y)
            max_dist = min(h, w) / 3
            focus_mask = np.clip(dist / max_dist, 0, 1)
            focus_mask = np.stack([focus_mask] * 3, axis=2)
            blurred_img = cv2.GaussianBlur(img, (17, 17), 0)
            img = (img * (1 - focus_mask) + blurred_img * focus_mask).astype(np.uint8)
        elif transform_type == "growth_stage":
            growth_factor = np.random.uniform(0.75, 1.3)
            img[:,:,1] = np.clip(img[:,:,1] * growth_factor, 0, 255)
        elif transform_type == "disease_spots":
            num_spots = np.random.randint(3, 18)
            for _ in range(num_spots):
                spot_x = np.random.randint(0, w-1)
                spot_y = np.random.randint(0, h-1)
                spot_size = np.random.randint(2, 12)
                spot_color = (np.random.randint(50, 110), np.random.randint(35, 85), np.random.randint(35, 95))
                cv2.circle(img, (spot_x, spot_y), spot_size, spot_color, -1)
        elif transform_type == "nutrient_deficiency":
            img[:,:,0] = np.clip(img[:,:,0] * 1.1, 0, 255)
            img[:,:,1] = np.clip(img[:,:,1] * 0.85, 0, 255)
        elif transform_type == "pest_damage":
            num_holes = np.random.randint(2, 8)
            for _ in range(num_holes):
                hole_x = np.random.randint(10, w-10)
                hole_y = np.random.randint(10, h-10)
                hole_size = np.random.randint(3, 8)
                cv2.circle(img, (hole_x, hole_y), hole_size, (0, 0, 0), -1)
        
        return img
    
    def _apply_weather_effects(self, image: np.ndarray, weather_type: str) -> np.ndarray:
        """Apply weather-related effects"""
        img = image.copy()
        h, w = img.shape[:2]
        
        if weather_type == "rain_drops":
            num_drops = np.random.randint(5, 20)
            for _ in range(num_drops):
                drop_x = np.random.randint(0, w)
                drop_y = np.random.randint(0, h)
                drop_size = np.random.randint(2, 6)
                cv2.circle(img, (drop_x, drop_y), drop_size, (255, 255, 255), -1)
        elif weather_type == "wind_blur":
            kernel_size = np.random.randint(3, 8)
            angle = np.random.uniform(0, 360)
            kernel = cv2.getRotationMatrix2D((kernel_size//2, kernel_size//2), angle, 1.0)
            img = cv2.filter2D(img, -1, kernel[:2,:2])
        elif weather_type == "dust_particles":
            num_particles = np.random.randint(10, 40)
            for _ in range(num_particles):
                x, y = np.random.randint(0, w), np.random.randint(0, h)
                size = np.random.randint(1, 3)
                color = (np.random.randint(200, 255), np.random.randint(180, 230), np.random.randint(120, 180))
                cv2.circle(img, (x, y), size, color, -1)
        elif weather_type == "morning_dew":
            num_drops = np.random.randint(8, 25)
            for _ in range(num_drops):
                x, y = np.random.randint(0, w), np.random.randint(0, h)
                size = np.random.randint(1, 4)
                overlay = img.copy()
                cv2.circle(overlay, (x, y), size, (255, 250, 200), -1)
                img = cv2.addWeighted(img, 0.9, overlay, 0.1, 0)
        
        return img
    
    def _apply_realistic_angle_transform(self, image: np.ndarray) -> np.ndarray:
        """Apply realistic angle transformations"""
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        angle_type = random.choice(['natural_rotation', 'camera_perspective', 'botanical_standard'])
        
        if angle_type == 'natural_rotation':
            common_angles = [15, 30, 45, 90, 135, 180, 225, 270, 315]
            angle = random.choice(common_angles) + np.random.uniform(-5, 5)
            rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
            return cv2.warpAffine(image, rotation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        elif angle_type == 'camera_perspective':
            tilt_angle = np.random.uniform(-12, 12)
            h_shift = np.random.uniform(-0.15, 0.15)
            v_shift = np.random.uniform(-0.15, 0.15)
            
            rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), tilt_angle, 1.0)
            tilted = cv2.warpAffine(image, rotation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
            
            translation_matrix = np.float32([[1, 0, w * h_shift], [0, 1, h * v_shift]])
            return cv2.warpAffine(tilted, translation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        else:  # botanical_standard
            angle = np.random.uniform(-20, 20)
            scale = np.random.uniform(0.8, 1.3)
            rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, scale)
            return cv2.warpAffine(image, rotation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    def _apply_basic_transform(self, image: np.ndarray) -> np.ndarray:
        """Apply transformations (rotation, brightness, flip)"""
        h, w = image.shape[:2]
        variant = image.copy()
        
        # Random rotation (-15 to +15 degrees)
        angle = np.random.uniform(-15, 15)
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        variant = cv2.warpAffine(variant, rotation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Random brightness adjustment
        brightness = np.random.uniform(0.8, 1.2)
        variant = np.clip(variant * brightness, 0, 255).astype(np.uint8)
        
        # Random horizontal flip (50% chance)
        if np.random.random() > 0.5:
            variant = cv2.flip(variant, 1)
        
        return variant
    
    def mixup(self, img1: np.ndarray, img2: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """MixUp augmentation - blend two images"""
        lam = np.random.beta(alpha, alpha)
        mixed_img = lam * img1 + (1 - lam) * img2
        return mixed_img.astype(np.uint8)
    
    def cutmix(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """CutMix augmentation - cut and paste regions"""
        h, w = img1.shape[:2]
        
        cut_w = np.random.randint(w//6, w//3)
        cut_h = np.random.randint(h//6, h//3)
        cut_x = np.random.randint(0, w - cut_w)
        cut_y = np.random.randint(0, h - cut_h)
        
        result = img1.copy()
        result[cut_y:cut_y+cut_h, cut_x:cut_x+cut_w] = img2[cut_y:cut_y+cut_h, cut_x:cut_x+cut_w]
        return result
    
    def augmix(self, image: np.ndarray, num_ops: int = 3) -> np.ndarray:
        """AugMix - multiple augmentation chains with mixing"""
        chains = []
        
        for _ in range(num_ops):
            aug_img = image.copy()
            ops = np.random.choice(['seasonal', 'lighting', 'plant', 'weather'], size=2, replace=False)
            
            for op in ops:
                if op == 'seasonal':
                    effect = np.random.choice(self.seasonal_effects)
                    aug_img = self._apply_seasonal_effect(aug_img, effect)
                elif op == 'lighting':
                    effect = np.random.choice(self.lighting_conditions)
                    aug_img = self._apply_lighting_condition(aug_img, effect)
                elif op == 'plant':
                    effect = np.random.choice(self.plant_transforms)
                    aug_img = self._apply_plant_specific_transform(aug_img, effect)
                elif op == 'weather':
                    effect = np.random.choice(self.weather_effects)
                    aug_img = self._apply_weather_effects(aug_img, effect)
            
            chains.append(aug_img)
        
        mixed = image.astype(np.float32)
        for chain in chains:
            weight = np.random.uniform(0.1, 0.3)
            mixed = mixed * (1 - weight) + chain.astype(np.float32) * weight
        
        return mixed.astype(np.uint8)
    
    def clear_cache(self):
        """Clear augmentation cache"""
        with self._lock:
            self.augmentation_cache.clear()
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        with self._lock:
            return {
                'cache_size': len(self.augmentation_cache),
                'max_cache_size': self.config.max_cache_size,
                'cache_usage': len(self.augmentation_cache) / self.config.max_cache_size * 100
            }
    
    def generate_augmented_variants(self, image: np.ndarray, num_variants: int = 12) -> List[np.ndarray]:
        """Generate multiple augmented variants efficiently - returns ONLY augmentations, not original"""
        variants = []  # Don't include original - caller will handle that
        
        # Resize to standard size if needed
        if image.shape[:2] != (512, 512):
            image = cv2.resize(image, (512, 512))
        
        for i in range(num_variants):  # Generate exactly num_variants augmentations
            # Apply different augmentation methods
            aug_type = np.random.choice(['basic', 'seasonal', 'lighting', 'plant', 'weather', 'angle'], 
                                       p=[0.3, 0.15, 0.15, 0.15, 0.15, 0.1])
            variant = self.generate_single_augmentation(image, method=aug_type)
            variants.append(variant)
        
        return variants
    
    def generate_realistic_angle_augmentations(self, image: np.ndarray, num_angle_variants: int = 50) -> List[np.ndarray]:
        """
        Generate realistic angle variations that simulate natural plant photography angles
        
        Args:
            image: Input image
            num_angle_variants: Number of angle variants to generate
        
        Returns:
            List of angle-augmented images
        """
        variants = []
        for i in range(num_angle_variants):
            variant = self._apply_realistic_angle_transform(image)
            variants.append(variant)
        return variants

    def create_augmented_batch_gpu(self, image_tensor: torch.Tensor, 
                                 num_augmentations: int = 10) -> List[torch.Tensor]:
        """
        Create augmented versions staying entirely on GPU for ultra-parallel processing
        Moved from GPUAugmentationEngine to consolidate all augmentation functionality
        
        Args:
            image_tensor: Input image tensor on GPU
            num_augmentations: Number of augmentations to generate
            
        Returns:
            List of GPU tensors with augmentations
        """
        augmented_tensors = []
        
        # Ensure input is on GPU
        image_tensor = image_tensor.to(self.device)
        
        for i in range(num_augmentations):
            # Apply random augmentation directly on GPU tensor
            augmented = self._apply_gpu_augmentation(image_tensor, i)
            augmented_tensors.append(augmented)
        
        return augmented_tensors
    
    def _apply_gpu_augmentation(self, tensor: torch.Tensor, variant_idx: int) -> torch.Tensor:
        """
        Apply augmentation directly on GPU tensor for processing
        Moved from GPUAugmentationEngine to consolidate all augmentation functionality
        
        Args:
            tensor: Input tensor on GPU
            variant_idx: Variant index to determine transform type
            
        Returns:
            Augmented tensor on GPU
        """
        # Simple GPU-based augmentations for now
        augmented = tensor.clone()
        
        # Random transformations based on variant index
        transform_type = variant_idx % 6
        
        if transform_type == 0:  # Brightness
            brightness = 0.7 + (variant_idx % 8) * 0.1
            augmented = torch.clamp(augmented * brightness, 0, 1)
        elif transform_type == 1:  # Contrast
            contrast = 0.7 + (variant_idx % 8) * 0.1
            mean_val = torch.mean(augmented)
            augmented = torch.clamp((augmented - mean_val) * contrast + mean_val, 0, 1)
        elif transform_type == 2:  # Noise
            noise = torch.randn_like(augmented) * 0.03
            augmented = torch.clamp(augmented + noise, 0, 1)
        elif transform_type == 3:  # Color shift
            color_shift = 0.9 + (variant_idx % 4) * 0.05
            if len(augmented.shape) == 3:  # HWC
                augmented[:,:,variant_idx % 3] *= color_shift
            augmented = torch.clamp(augmented, 0, 1)
        elif transform_type == 4:  # Saturation (HSV-like)
            saturation = 0.8 + (variant_idx % 6) * 0.1
            gray = torch.mean(augmented, dim=2, keepdim=True)
            augmented = torch.clamp(gray + (augmented - gray) * saturation, 0, 1)
        else:  # Gamma correction
            gamma = 0.8 + (variant_idx % 6) * 0.1
            augmented = torch.clamp(torch.pow(augmented, gamma), 0, 1)
        
        return augmented

# Factory function for easy import
def create_augmentation_engine(use_gpu: bool = True, enable_all_features: bool = True) -> UnifiedPlantAugmentationEngine:
    """Create a unified augmentation engine with specified configuration"""
    config = AugmentationConfig(
        use_gpu=use_gpu,
        enable_advanced_transforms=enable_all_features,
        enable_mixing_methods=enable_all_features,
        enable_realistic_angles=enable_all_features,
        enable_plant_specific=enable_all_features
    )
    return UnifiedPlantAugmentationEngine(config) 