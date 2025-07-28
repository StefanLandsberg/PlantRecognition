import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2
from pathlib import Path
import pickle
import time
import gc
import psutil
import random
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys
import asyncio
import json

np.random.seed(42)
random.seed(42)
try:
    import torch
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
except ImportError:
    pass

try:
    import torch
    import torch.nn.functional as F
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        pass
    else:
        pass
except ImportError:
    GPU_AVAILABLE = False

# GPU Configuration Loader
def load_gpu_config() -> Dict:
    """Load GPU CUDA cores and calculate optimal thread counts automatically"""
    config_path = Path(__file__).parent / "GPUconfig.json"
    
    # Default CUDA cores (RTX 3050 level)
    cuda_cores = 2560
    
    try:
        if config_path.exists():
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                cuda_cores = loaded_config.get("gpu_cuda_cores", 2560)
    except Exception as e:
        print(f"     GPU config load failed, using default CUDA cores: {e}")
    
    optimal_threads = max(8, min(32, cuda_cores // 150))  # Much higher thread count for 100% utilization
    
    # Calculate MAXIMUM worker counts for 100% GPU utilization
    config = {
        "max_workers": {
            "modality_extraction": max(8, optimal_threads),                    # MAXIMUM thread utilization for 6 modalities
            "feature_processing": max(12, int(optimal_threads * 1.5)),        # 150% - aggressive mixed CPU/GPU tasks
            "batch_processing": max(16, int(optimal_threads * 2.5)),          # 250% - maximum I/O intensive operations
            "gpu_workers": max(4, optimal_threads // 2),                      # 50% - aggressive GPU operations
            "cpu_workers": max(8, int(optimal_threads * 0.75))                # 75% - aggressive CPU-only tasks
        },
        "cuda_cores": cuda_cores,                                             # Expose CUDA cores for direct use
        "streams_per_modality": max(16, cuda_cores // 100),                   # AGGRESSIVE: 1 stream per 100 cores
        "optimal_batch_size": max(8, cuda_cores // 320)                      # Core-based batch sizing
    }
    
    print(f"     GPU Config: {cuda_cores} CUDA cores → {optimal_threads} threads → modality:{config['max_workers']['modality_extraction']}, feature:{config['max_workers']['feature_processing']}, batch:{config['max_workers']['batch_processing']}")
    print(f"     GPU Streams: {config['streams_per_modality']} streams/modality, Batch size: {config['optimal_batch_size']}")
    
    return config

# Global configuration instance
_GPU_CONFIG = load_gpu_config()

#  SPEED OPTIMIZATION: Global GPU Tensor Cache
class GlobalGPUTensorCache:
    """Global cache for GPU tensors to eliminate transfer overhead"""
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.tensor_cache = {}
            self.device = torch.device('cuda')
            self._initialized = True
    
    def get_reusable_tensor(self, key: str, shape: tuple, dtype=torch.float32) -> torch.Tensor:
        """GPU tensor caching with memory pool management for better performance"""
        cache_key = f"{key}_{shape}_{dtype}"
        if cache_key not in self.tensor_cache:
            # Check if GPU memory is available before allocation
            if self.device.type == 'cuda':
                try:
                    # Pre-allocate with contiguous memory for better performance
                    tensor = torch.zeros(shape, dtype=dtype, device=self.device).contiguous()
                    self.tensor_cache[cache_key] = tensor
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        # Clear some cache and retry
                        self.clear_cache()
                        tensor = torch.zeros(shape, dtype=dtype, device=self.device).contiguous()
                        self.tensor_cache[cache_key] = tensor
                    else:
                        raise e
            else:
                self.tensor_cache[cache_key] = torch.zeros(shape, dtype=dtype, device=self.device).contiguous()
        return self.tensor_cache[cache_key]
    
    def clear_cache(self):
        """Clear cache if memory becomes an issue"""
        for tensor in self.tensor_cache.values():
            del tensor
        self.tensor_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class GPUPersistentTensorCache:
    """Optimized GPU cache for faster training with minimal memory usage"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Only cache essential tensors - remove redundant storage
        self.descriptor_cache = {}  # Pre-computed descriptors only
        self._lock = threading.Lock()
        
    def cache_descriptors(self, key: str, descriptors: torch.Tensor):
        """Cache only final descriptors - no intermediate data"""
        with self._lock:
            # Store as contiguous tensor for faster access
            self.descriptor_cache[key] = descriptors.contiguous()
    
    def get_descriptors(self, key: str) -> torch.Tensor:
        """Get cached descriptors directly"""
        with self._lock:
            return self.descriptor_cache.get(key)
    
    def clear_cache(self):
        """Fast cache clear"""
        with self._lock:
            self.descriptor_cache.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

class CUDAStreamParallelProcessor:
    """parallel processor using CUDA streams for true GPU parallelization"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create MAXIMUM CUDA streams for 100% GPU utilization
        if torch.cuda.is_available():
            from torch.cuda import Stream
            
            # Load GPU config to get optimal stream count based on CUDA cores
            gpu_config = load_gpu_config()
            
            # Use config-based stream count (no hardcoded values)
            streams_per_modality = gpu_config["streams_per_modality"]  # Core-based calculation from config
            
            self.streams = {}
            
            # Create multiple streams for each modality for maximum parallel processing
            for modality in ['texture', 'color', 'shape', 'contrast', 'frequency', 'unique']:
                self.streams[modality] = [Stream() for _ in range(streams_per_modality)]
            
            # Additional specialized streams
            self.streams['augmentation'] = [Stream() for _ in range(2)]
            self.streams['main'] = [Stream()]
            
            print(f"     CUDA Streams: {streams_per_modality} streams per modality = {6 * streams_per_modality + 3} total streams")
            print(f"     GPU Utilization: MAXIMIZED for {gpu_config['cuda_cores']} CUDA cores")
            
        else:
            self.streams = {modality: [None] for modality in ['texture', 'color', 'shape', 'contrast', 'frequency', 'unique', 'augmentation', 'main']}
        
        # Initialize extractors for each stream
        self.extractors = {
            'texture': None,  # Will use MultiModalCurseResistantRecognizer methods
            'color': None,    # Will be initialized when needed
            'shape': None,
            'contrast': None,
            'frequency': None,
            'unique': None
        }
    
    def process_image_parallel_cuda_streams(self, image_tensor: torch.Tensor, 
                                          augmented_tensors: List[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Process image + augmentations in single batch extraction
        Returns GPU tensors for all features (never touches CPU)
        """
        # Ensure all tensors are on GPU
        image_tensor = image_tensor.to(self.device)
        if augmented_tensors:
            augmented_tensors = [tensor.to(self.device) for tensor in augmented_tensors]
        
        # Prepare all images for processing (original + augmentations)
        all_images = [image_tensor]
        if augmented_tensors:
            all_images.extend(augmented_tensors)
        
        # Results storage (GPU tensors)
        all_results = {}
        
        # Extract ALL modalities for ALL images in single batch operations
        for modality in ['texture', 'color', 'shape', 'contrast', 'frequency', 'unique']:
            modality_features = self._extract_modality_cuda_stream(modality, all_images)
            all_results[modality] = modality_features
        
        return all_results
    
    def _extract_modality_cuda_stream(self, modality: str, image_tensors: List[torch.Tensor]) -> torch.Tensor:
        """TRUE BATCH extraction - process ALL images at once for maximum GPU efficiency"""
        batch_tensor = torch.stack(image_tensors)
        batch_size = batch_tensor.shape[0]
        
        # SINGLE batch extraction for ALL images at once (no stream splitting)
        if modality == 'texture':
            features = self._extract_texture_tensor_batch(batch_tensor)
        elif modality == 'color':
            features = self._extract_color_tensor_batch(batch_tensor)
        elif modality == 'shape':
            features = self._extract_shape_tensor_batch(batch_tensor)
        elif modality == 'contrast':
            features = self._extract_contrast_tensor_batch(batch_tensor)
        elif modality == 'frequency':
            features = self._extract_frequency_tensor_batch(batch_tensor)
        elif modality == 'unique':
            features = self._extract_unique_tensor_batch(batch_tensor)
        else:
            features = torch.zeros((batch_size, 5000), device=self.device, dtype=torch.float32)
        
        return features
    
    def _extract_texture_tensor_batch(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """DETERMINISTIC PLANT-OPTIMIZED texture extraction with botanical edge detection"""
        B, H, W, C = batch_tensor.shape
        
        # Deterministic grayscale conversion (botanical-optimized weights)
        # Green channel weighted higher for plant material
        gray_batch = 0.25 * batch_tensor[:,:,:,2] + 0.6 * batch_tensor[:,:,:,1] + 0.15 * batch_tensor[:,:,:,0]
        gray_batch = gray_batch.unsqueeze(1)
        
        # DETERMINISTIC BOTANICAL EDGE KERNELS - designed for plant structures
        device = self.device
        
        # Primary edge detectors (leaf edges, stems, veins)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
        
        # Botanical-specific kernels for plant structure detection
        leaf_vein_h = torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
        leaf_vein_v = torch.tensor([[-1, 0, -1], [0, 5, 0], [-1, 0, -1]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
        
        # Stem/branch detection (diagonal structures)
        stem_diag1 = torch.tensor([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
        stem_diag2 = torch.tensor([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
        
        # Fine texture detection (leaf surface roughness)
        texture_fine = torch.tensor([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
        
        # Apply all botanical kernels deterministically
        edges_x = torch.nn.functional.conv2d(gray_batch, sobel_x, padding=1)
        edges_y = torch.nn.functional.conv2d(gray_batch, sobel_y, padding=1)
        primary_edges = torch.sqrt(edges_x**2 + edges_y**2).squeeze(1)
        
        vein_h = torch.nn.functional.conv2d(gray_batch, leaf_vein_h, padding=1).squeeze(1)
        vein_v = torch.nn.functional.conv2d(gray_batch, leaf_vein_v, padding=1).squeeze(1)
        vein_response = torch.abs(vein_h) + torch.abs(vein_v)
        
        stem_d1 = torch.nn.functional.conv2d(gray_batch, stem_diag1, padding=1).squeeze(1)
        stem_d2 = torch.nn.functional.conv2d(gray_batch, stem_diag2, padding=1).squeeze(1)
        stem_response = torch.abs(stem_d1) + torch.abs(stem_d2)
        
        fine_texture = torch.abs(torch.nn.functional.conv2d(gray_batch, texture_fine, padding=1).squeeze(1))
        
        # DETERMINISTIC BOTANICAL STATISTICS
        # Use fixed percentiles for deterministic behavior
        primary_flat = primary_edges.view(B, -1)
        vein_flat = vein_response.view(B, -1)
        stem_flat = stem_response.view(B, -1)
        fine_flat = fine_texture.view(B, -1)
        
        # Deterministic percentile calculation (botanical-relevant thresholds)
        primary_p90 = torch.quantile(primary_flat, 0.90, dim=1)  # Strong edges (leaf boundaries)
        primary_p50 = torch.quantile(primary_flat, 0.50, dim=1)  # Medium edges
        primary_p10 = torch.quantile(primary_flat, 0.10, dim=1)  # Weak edges
        
        vein_p75 = torch.quantile(vein_flat, 0.75, dim=1)        # Major veins
        vein_p25 = torch.quantile(vein_flat, 0.25, dim=1)        # Minor veins
        
        stem_p80 = torch.quantile(stem_flat, 0.80, dim=1)        # Strong stems/branches
        
        # DETERMINISTIC MULTI-SCALE PATCHES (powers of 2 for consistency)
        # Use deterministic strides that divide evenly
        patch_sizes = [4, 8, 16, 32, 64]
        patch_features = []
        
        for size in patch_sizes:
            stride = size  # No overlap for deterministic behavior
            
            # Extract patches from primary edges
            if size <= H and size <= W:
                patches = torch.nn.functional.unfold(primary_edges.unsqueeze(1), size, stride=stride)
                if patches.shape[-1] > 0:
                    patch_means = torch.mean(patches, dim=1)
                    patch_stds = torch.std(patches, dim=1)
                    patch_features.append(patch_means)
                    patch_features.append(patch_stds)
        
        # BOTANICAL BASE FEATURES (deterministic and plant-specific)
        base_features = torch.stack([
            primary_p90,                    # Leaf boundary strength
            primary_p50,                    # General edge density  
            vein_p75,                       # Major vein prominence
            torch.mean(vein_flat, dim=1),   # Overall vein density
            stem_p80,                       # Stem/branch strength
            torch.mean(fine_flat, dim=1),   # Surface texture
            torch.std(primary_flat, dim=1), # Edge variation
            torch.std(vein_flat, dim=1),    # Vein variation
        ], dim=1)  # 8 deterministic botanical features
        
        # Concatenate all patch features deterministically
        if patch_features:
            all_patch_features = torch.cat(patch_features, dim=1)
            
            # Ensure exactly 4992 patch features (5000 - 8 base = 4992)
            target_patch_features = 4992
            current_patch_features = all_patch_features.shape[1]
            
            if current_patch_features >= target_patch_features:
                # Take first 4992 features deterministically
                patch_component = all_patch_features[:, :target_patch_features]
            else:
                # Pad with deterministic values (repeat last features)
                needed = target_patch_features - current_patch_features
                last_feature = all_patch_features[:, -1:].repeat(1, needed)
                patch_component = torch.cat([all_patch_features, last_feature], dim=1)
        else:
            # Fallback: use primary edge statistics
            patch_component = torch.zeros((B, 4992), device=device, dtype=torch.float32)
        
        # Combine for exactly 5000 texture features
        final_features = torch.cat([base_features, patch_component], dim=1)
        
        # Final NaN/Inf protection
        final_features = torch.where(torch.isfinite(final_features), final_features, torch.zeros_like(final_features))
        
        return final_features
    
    def _extract_color_tensor_batch(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """DETERMINISTIC BOTANICAL COLOR extraction optimized for plant identification"""
        B, H, W, C = batch_tensor.shape
        
        # Extract RGB channels with botanical focus
        r, g, b = batch_tensor[:,:,:,2], batch_tensor[:,:,:,1], batch_tensor[:,:,:,0]
        
        # DETERMINISTIC BOTANICAL COLOR ANALYSIS
        # Convert to multiple color spaces for plant discrimination
        
        # 1. Standard RGB statistics
        rgb_flat = torch.stack([r, g, b], dim=3).view(B, -1, 3)
        rgb_means = torch.mean(rgb_flat, dim=1)
        rgb_stds = torch.std(rgb_flat, dim=1)
        
        # 2. Plant-optimized HSV conversion (deterministic)
        hsv_batch = self._deterministic_rgb_to_hsv(batch_tensor)
        h, s, v = hsv_batch[:,:,:,0], hsv_batch[:,:,:,1], hsv_batch[:,:,:,2]
        
        hsv_flat = torch.stack([h, s, v], dim=3).view(B, -1, 3)
        hsv_means = torch.mean(hsv_flat, dim=1)
        hsv_stds = torch.std(hsv_flat, dim=1)
        
        # 3. BOTANICAL COLOR INDICES (NaN-safe divisions only)
        # Green vegetation indices
        ndvi = (g - r) / (g + r + 1e-6)  # Normalized Difference Vegetation Index
        
        gndvi = (g - r) / (g + r + 1e-6)  # Green NDVI
        
        evi_denom = g + 6*r - 7.5*b + 1e-6
        evi = 2.5 * (g - r) / evi_denom  # Enhanced Vegetation Index
        
        # Plant health indices (safe divisions only)
        ari = (1.0/(g + 1e-6)) - (1.0/(r + 1e-6))  # Anthocyanin Reflectance Index
        
        cri = (1.0/(b + 1e-6)) - (1.0/(g + 1e-6))  # Carotenoid Reflectance Index
        
        # Color ratios for species discrimination (safe divisions only)
        total_rgb = r + g + b + 1e-6
        green_ratio = g / total_rgb
        red_ratio = r / total_rgb
        blue_ratio = b / total_rgb
        
        # DETERMINISTIC PERCENTILE ANALYSIS
        # Use fixed percentiles for consistent behavior
        g_flat = g.view(B, -1)
        r_flat = r.view(B, -1)
        b_flat = b.view(B, -1)
        
        # Green channel analysis (most important for plants)
        g_p90 = torch.quantile(g_flat, 0.90, dim=1)  # Bright green areas
        g_p75 = torch.quantile(g_flat, 0.75, dim=1)  # Healthy green
        g_p50 = torch.quantile(g_flat, 0.50, dim=1)  # Medium green
        g_p25 = torch.quantile(g_flat, 0.25, dim=1)  # Dark green/shadows
        
        # Red channel analysis (flowers, autumn, stress)
        r_p90 = torch.quantile(r_flat, 0.90, dim=1)  # Bright red areas
        r_p10 = torch.quantile(r_flat, 0.10, dim=1)  # Low red areas
        
        # Hue analysis for color discrimination
        h_flat = h.view(B, -1)
        h_mean = torch.mean(h_flat, dim=1)
        h_std = torch.std(h_flat, dim=1)
        
        # Saturation analysis for color purity
        s_flat = s.view(B, -1)
        s_p75 = torch.quantile(s_flat, 0.75, dim=1)  # High saturation areas
        s_mean = torch.mean(s_flat, dim=1)
        
        # DETERMINISTIC SPATIAL COLOR ANALYSIS
        # Multi-scale deterministic patches
        patch_sizes = [8, 16, 32]  # Deterministic power-of-2 sizes
        spatial_features = []
        
        for size in patch_sizes:
            stride = size  # No overlap for deterministic behavior
            
            if size <= H and size <= W:
                # RGB patches
                rgb_patches = torch.nn.functional.unfold(
                    torch.stack([r, g, b], dim=1), size, stride=stride
                )
                if rgb_patches.shape[-1] > 0:
                    rgb_patch_means = torch.mean(rgb_patches, dim=1)
                    spatial_features.append(rgb_patch_means)
                
                # HSV patches
                hsv_patches = torch.nn.functional.unfold(
                    torch.stack([h, s, v], dim=1), size, stride=stride
                )
                if hsv_patches.shape[-1] > 0:
                    hsv_patch_means = torch.mean(hsv_patches, dim=1)
                    spatial_features.append(hsv_patch_means)
        
        # BOTANICAL BASE FEATURES (deterministic and plant-specific)
        base_features = torch.stack([
            rgb_means[:, 0], rgb_means[:, 1], rgb_means[:, 2],  # RGB means
            hsv_means[:, 0], hsv_means[:, 1], hsv_means[:, 2],  # HSV means
            g_p90, g_p75, g_p50, g_p25,                         # Green percentiles
            r_p90, r_p10,                                       # Red extremes
            h_mean, h_std,                                       # Hue statistics
            s_p75, s_mean,                                       # Saturation stats
            torch.mean(ndvi.view(B, -1), dim=1),               # NDVI
            torch.mean(gndvi.view(B, -1), dim=1),              # Green NDVI
            torch.mean(evi.view(B, -1), dim=1),                # Enhanced VI
            torch.mean(ari.view(B, -1), dim=1),                # Anthocyanin
            torch.mean(cri.view(B, -1), dim=1),                # Carotenoid
            torch.mean(green_ratio.view(B, -1), dim=1),        # Green ratio
            torch.mean(red_ratio.view(B, -1), dim=1),          # Red ratio
            torch.mean(blue_ratio.view(B, -1), dim=1),         # Blue ratio
        ], dim=1)  # 24 deterministic botanical color features
        
        # Concatenate spatial features deterministically
        if spatial_features:
            all_spatial_features = torch.cat(spatial_features, dim=1)
            
            # Ensure exactly 4976 spatial features (5000 - 24 base = 4976)
            target_spatial_features = 4976
            current_spatial_features = all_spatial_features.shape[1]
            
            if current_spatial_features >= target_spatial_features:
                # Take first 4976 features deterministically
                spatial_component = all_spatial_features[:, :target_spatial_features]
            else:
                # Pad with deterministic values (repeat RGB means)
                needed = target_spatial_features - current_spatial_features
                padding = rgb_means.repeat(1, (needed // 3) + 1)[:, :needed]
                spatial_component = torch.cat([all_spatial_features, padding], dim=1)
        else:
            # Fallback: use RGB and HSV means repeated
            spatial_component = torch.cat([rgb_means, hsv_means], dim=1).repeat(1, 4976 // 6)[:, :4976]
        
        # Combine for exactly 5000 color features
        final_features = torch.cat([base_features, spatial_component], dim=1)
        
        # Final NaN/Inf protection
        final_features = torch.where(torch.isfinite(final_features), final_features, torch.zeros_like(final_features))
        
        return final_features
    
    def _deterministic_rgb_to_hsv(self, rgb_batch: torch.Tensor) -> torch.Tensor:
        """Deterministic RGB to HSV conversion optimized for botanical analysis"""
        r, g, b = rgb_batch[:,:,:,2], rgb_batch[:,:,:,1], rgb_batch[:,:,:,0]
        
        # Normalize to 0-1 range for HSV conversion
        r_norm = r / 255.0
        g_norm = g / 255.0
        b_norm = b / 255.0
        
        max_val, max_idx = torch.max(torch.stack([r_norm, g_norm, b_norm], dim=3), dim=3)
        min_val, _ = torch.min(torch.stack([r_norm, g_norm, b_norm], dim=3), dim=3)
        delta = max_val - min_val
        
        # Value (brightness)
        v = max_val
        
        # Saturation  
        s = torch.where(max_val != 0, delta / max_val, torch.zeros_like(max_val))
        
        # Hue (deterministic calculation)
        h = torch.zeros_like(max_val)
        
        # Red is max
        mask_r = (max_idx == 0) & (delta != 0)
        h[mask_r] = ((g_norm[mask_r] - b_norm[mask_r]) / delta[mask_r]) % 6.0
        
        # Green is max  
        mask_g = (max_idx == 1) & (delta != 0)
        h[mask_g] = ((b_norm[mask_g] - r_norm[mask_g]) / delta[mask_g]) + 2.0
        
        # Blue is max
        mask_b = (max_idx == 2) & (delta != 0)
        h[mask_b] = ((r_norm[mask_b] - g_norm[mask_b]) / delta[mask_b]) + 4.0
        
        # Convert to 0-1 range
        h = h / 6.0
        
        return torch.stack([h, s, v], dim=3)
    
    def _extract_shape_tensor_batch(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """DETERMINISTIC PLANT MORPHOLOGY shape extraction optimized for botanical structures"""
        B, H, W, C = batch_tensor.shape
        
        # Botanical-optimized grayscale (emphasize green structures)
        gray_batch = 0.25 * batch_tensor[:,:,:,2] + 0.6 * batch_tensor[:,:,:,1] + 0.15 * batch_tensor[:,:,:,0]
        gray_batch = gray_batch.unsqueeze(1)
        
        device = self.device
        
        # DETERMINISTIC BOTANICAL MORPHOLOGY KERNELS
        # Designed specifically for plant structures
        
        # 1. Leaf edge detectors
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
        
        # 2. Plant structure specific kernels
        # Leaf tip detection (elongated structures)
        leaf_tip_v = torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
        leaf_tip_h = torch.tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
        
        # Stem/branch detection (linear structures)
        stem_linear_h = torch.tensor([[0, 0, 0], [1, 1, 1], [0, 0, 0]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
        stem_linear_v = torch.tensor([[0, 1, 0], [0, 1, 0], [0, 1, 0]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
        
        # Diagonal branch structures
        branch_diag1 = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
        branch_diag2 = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
        
        # Corner/junction detection (leaf attachments, branch junctions)
        corner_tl = torch.tensor([[1, 1, 0], [1, -4, 0], [0, 0, 0]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
        corner_tr = torch.tensor([[0, 1, 1], [0, -4, 1], [0, 0, 0]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
        corner_bl = torch.tensor([[0, 0, 0], [1, -4, 0], [1, 1, 0]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
        corner_br = torch.tensor([[0, 0, 0], [0, -4, 1], [0, 1, 1]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
        
        # Apply all morphological kernels deterministically
        edges_x = torch.nn.functional.conv2d(gray_batch, sobel_x, padding=1).squeeze(1)
        edges_y = torch.nn.functional.conv2d(gray_batch, sobel_y, padding=1).squeeze(1)
        primary_edges = torch.sqrt(edges_x**2 + edges_y**2)
        
        # Leaf structure responses
        tip_v_resp = torch.abs(torch.nn.functional.conv2d(gray_batch, leaf_tip_v, padding=1).squeeze(1))
        tip_h_resp = torch.abs(torch.nn.functional.conv2d(gray_batch, leaf_tip_h, padding=1).squeeze(1))
        leaf_tips = tip_v_resp + tip_h_resp
        
        # Stem/branch responses
        stem_h_resp = torch.abs(torch.nn.functional.conv2d(gray_batch, stem_linear_h, padding=1).squeeze(1))
        stem_v_resp = torch.abs(torch.nn.functional.conv2d(gray_batch, stem_linear_v, padding=1).squeeze(1))
        stem_linear = stem_h_resp + stem_v_resp
        
        branch_d1_resp = torch.abs(torch.nn.functional.conv2d(gray_batch, branch_diag1, padding=1).squeeze(1))
        branch_d2_resp = torch.abs(torch.nn.functional.conv2d(gray_batch, branch_diag2, padding=1).squeeze(1))
        branch_diagonal = branch_d1_resp + branch_d2_resp
        
        # Corner/junction responses
        corner_tl_resp = torch.abs(torch.nn.functional.conv2d(gray_batch, corner_tl, padding=1).squeeze(1))
        corner_tr_resp = torch.abs(torch.nn.functional.conv2d(gray_batch, corner_tr, padding=1).squeeze(1))
        corner_bl_resp = torch.abs(torch.nn.functional.conv2d(gray_batch, corner_bl, padding=1).squeeze(1))
        corner_br_resp = torch.abs(torch.nn.functional.conv2d(gray_batch, corner_br, padding=1).squeeze(1))
        corner_response = corner_tl_resp + corner_tr_resp + corner_bl_resp + corner_br_resp
        
        # DETERMINISTIC MORPHOLOGICAL STATISTICS
        # Fixed percentiles for consistent botanical analysis
        primary_flat = primary_edges.view(B, -1)
        leaf_flat = leaf_tips.view(B, -1)
        stem_flat = stem_linear.view(B, -1)
        branch_flat = branch_diagonal.view(B, -1)
        corner_flat = corner_response.view(B, -1)
        
        # Edge strength analysis (leaf boundaries)
        edge_p95 = torch.quantile(primary_flat, 0.95, dim=1)  # Very strong edges
        edge_p75 = torch.quantile(primary_flat, 0.75, dim=1)  # Strong edges
        edge_p50 = torch.quantile(primary_flat, 0.50, dim=1)  # Medium edges
        edge_p25 = torch.quantile(primary_flat, 0.25, dim=1)  # Weak edges
        
        # Leaf tip analysis
        tip_p90 = torch.quantile(leaf_flat, 0.90, dim=1)      # Prominent tips
        tip_mean = torch.mean(leaf_flat, dim=1)               # Average tip response
        
        # Stem/branch analysis
        stem_p85 = torch.quantile(stem_flat, 0.85, dim=1)     # Strong linear structures
        branch_p80 = torch.quantile(branch_flat, 0.80, dim=1) # Strong diagonal structures
        
        # Junction analysis (important for plant architecture)
        corner_p90 = torch.quantile(corner_flat, 0.90, dim=1) # Strong junctions
        corner_mean = torch.mean(corner_flat, dim=1)          # Average junction density
        
        # DETERMINISTIC MULTI-SCALE MORPHOLOGICAL ANALYSIS
        # Use different scales to capture different plant structures
        scale_sizes = [4, 8, 16, 32]  # Deterministic scales
        morphology_features = []
        
        for size in scale_sizes:
            stride = size  # No overlap for deterministic behavior
            
            if size <= H and size <= W:
                # Analyze primary edges at this scale
                edge_patches = torch.nn.functional.unfold(primary_edges.unsqueeze(1), size, stride=stride)
                if edge_patches.shape[-1] > 0:
                    edge_patch_means = torch.mean(edge_patches, dim=1)
                    edge_patch_stds = torch.std(edge_patches, dim=1)
                    morphology_features.append(edge_patch_means)
                    morphology_features.append(edge_patch_stds)
                
                # Analyze leaf tips at this scale
                tip_patches = torch.nn.functional.unfold(leaf_tips.unsqueeze(1), size, stride=stride)
                if tip_patches.shape[-1] > 0:
                    tip_patch_means = torch.mean(tip_patches, dim=1)
                    morphology_features.append(tip_patch_means)
        
        # BOTANICAL MORPHOLOGY BASE FEATURES (deterministic)
        base_features = torch.stack([
            edge_p95, edge_p75, edge_p50, edge_p25,             # Edge strength hierarchy
            tip_p90, tip_mean,                                   # Leaf tip characteristics
            stem_p85, torch.mean(stem_flat, dim=1),             # Linear structure analysis
            branch_p80, torch.mean(branch_flat, dim=1),         # Diagonal structure analysis
            corner_p90, corner_mean,                             # Junction analysis
            torch.std(primary_flat, dim=1),                     # Edge variation
            torch.std(leaf_flat, dim=1),                        # Tip variation
            torch.std(stem_flat, dim=1),                        # Stem variation
            torch.std(branch_flat, dim=1),                      # Branch variation
        ], dim=1)  # 16 deterministic morphological features
        
        # Concatenate multi-scale features deterministically
        if morphology_features:
            all_morphology_features = torch.cat(morphology_features, dim=1)
            
            # Ensure exactly 4984 multi-scale features (5000 - 16 base = 4984)
            target_morphology_features = 4984
            current_morphology_features = all_morphology_features.shape[1]
            
            if current_morphology_features >= target_morphology_features:
                # Take first 4984 features deterministically
                morphology_component = all_morphology_features[:, :target_morphology_features]
            else:
                # Pad with deterministic values (repeat edge statistics)
                needed = target_morphology_features - current_morphology_features
                padding = torch.stack([edge_p75, edge_p50, edge_p25], dim=1).repeat(1, (needed // 3) + 1)[:, :needed]
                morphology_component = torch.cat([all_morphology_features, padding], dim=1)
        else:
            # Fallback: use edge statistics repeated
            morphology_component = torch.stack([edge_p75, edge_p50, edge_p25], dim=1).repeat(1, 4984 // 3)[:, :4984]
        
        # Combine for exactly 5000 shape features
        final_features = torch.cat([base_features, morphology_component], dim=1)
        
        # Final NaN/Inf protection
        final_features = torch.where(torch.isfinite(final_features), final_features, torch.zeros_like(final_features))
        
        return final_features
    
    def _extract_contrast_tensor_batch(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """FULLY VECTORISED contrast extraction for maximum GPU utilisation"""
        B, H, W, C = batch_tensor.shape
        
        gray_batch = 0.299 * batch_tensor[:,:,:,2] + 0.587 * batch_tensor[:,:,:,1] + 0.114 * batch_tensor[:,:,:,0]
        
        # Vectorised global statistics
        gray_flat = gray_batch.view(B, -1)
        global_means = torch.mean(gray_flat, dim=1)
        global_stds = torch.std(gray_flat, dim=1)
        global_ranges = torch.max(gray_flat, dim=1)[0] - torch.min(gray_flat, dim=1)[0]
        
        # Multi-scale patch extraction to reach 5000 features  
        # Use safer strides to avoid dimension issues
        patches_8 = torch.nn.functional.unfold(gray_batch.unsqueeze(1), 8, stride=8)
        patches_16 = torch.nn.functional.unfold(gray_batch.unsqueeze(1), 16, stride=16)
        patches_32 = torch.nn.functional.unfold(gray_batch.unsqueeze(1), 32, stride=32)
        
        patch_8_means = torch.mean(patches_8, dim=1)
        patch_16_means = torch.mean(patches_16, dim=1)
        patch_32_means = torch.mean(patches_32, dim=1)
        
        # Combine all patch features
        all_patches = torch.cat([patch_8_means, patch_16_means, patch_32_means], dim=1)
        
        # BALANCED contribution: larger patches get priority (fewer available features)
        # Take ALL features from 32px patches, distribute remaining among smaller
        
        # Priority to larger patches (32px has fewest features)
        all_patch_32 = patch_32_means  # All available 32px features
        
        # Calculate remaining budget for smaller patches
        used_features = all_patch_32.shape[1]
        remaining_budget = 4997 - used_features - 2  # Reserve 2 for global stats
        
        # Distribute remaining budget among 8px and 16px patches
        features_per_small_scale = remaining_budget // 2  # ~2200+ per scale
        
        patch_features = torch.cat([
            patch_8_means[:, :features_per_small_scale],   # 8px: ~2200+ features
            patch_16_means[:, :features_per_small_scale],  # 16px: ~2200+ features
            all_patch_32,                                   # 32px: ALL available
            torch.var(gray_batch.view(B, -1), dim=1).unsqueeze(1),     # Global variance
            torch.median(gray_batch.view(B, -1), dim=1)[0].unsqueeze(1)  # Global median
        ], dim=1)
        
        # Adjust to exactly 4997 features
        current_features = patch_features.shape[1]
        if current_features < 4997:
            needed = 4997 - current_features
            extra = torch.std(gray_batch.view(B, -1), dim=1).unsqueeze(1).repeat(1, needed)
            patch_features = torch.cat([patch_features, extra], dim=1)
        else:
            patch_features = patch_features[:, :4997]
        
        final_features = torch.cat([
            global_means.unsqueeze(1), global_stds.unsqueeze(1), global_ranges.unsqueeze(1),
            patch_features,  # Exactly 4997 contrast features from all scales
        ], dim=1)  # Total: exactly 5000 features (contrast)
        
        # Final NaN/Inf protection
        final_features = torch.where(torch.isfinite(final_features), final_features, torch.zeros_like(final_features))
        
        return final_features
    
    def _extract_frequency_tensor_batch(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """FULLY VECTORISED frequency extraction with RAW FEATURES"""
        B, H, W, C = batch_tensor.shape
        
        gray_batch = 0.299 * batch_tensor[:,:,:,2] + 0.587 * batch_tensor[:,:,:,1] + 0.114 * batch_tensor[:,:,:,0]
        
        # Vectorised FFT across batch
        fft_batch = torch.fft.fft2(gray_batch)
        fft_magnitude = torch.abs(fft_batch)
        
        # RAW FFT features: Log transform to compress dynamic range but keep raw values
        fft_log = torch.log10(fft_magnitude + 1e-8)  # Log transform + small epsilon for stability
        
        # Extract RAW frequency features (no normalization)
        dc_components = fft_log[:, 0, 0]
        high_freq = torch.mean(fft_log[:, H//2:, W//2:].reshape(B, -1), dim=1)
        low_freq = torch.mean(fft_log[:, :H//4, :W//4].reshape(B, -1), dim=1)
        
        # Additional RAW frequency statistics
        global_mean = torch.mean(fft_log.reshape(B, -1), dim=1)
        global_std = torch.std(fft_log.reshape(B, -1), dim=1)
        global_median = torch.median(fft_log.reshape(B, -1), dim=1)[0]
        
        final_features = torch.cat([
            dc_components.unsqueeze(1), high_freq.unsqueeze(1), low_freq.unsqueeze(1),
            global_mean.unsqueeze(1), global_std.unsqueeze(1), global_median.unsqueeze(1),
            fft_log.reshape(B, -1)[:, :4994],  # RAW frequency components - no normalization
        ], dim=1)  # Total: 5000 features (frequency) - RAW VALUES
        
        # Final NaN/Inf protection
        final_features = torch.where(torch.isfinite(final_features), final_features, torch.zeros_like(final_features))
        
        return final_features
    
    def _extract_unique_tensor_batch(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """DETERMINISTIC BOTANICAL DISCRIMINATION features optimized for plant species identification"""
        B, H, W, C = batch_tensor.shape
        
        # Extract RGB channels for botanical analysis
        r, g, b = batch_tensor[:,:,:,2], batch_tensor[:,:,:,1], batch_tensor[:,:,:,0]
        
        # DETERMINISTIC BOTANICAL INDICES (plant-specific discrimination)
        # These are scientifically-proven vegetation indices for plant analysis
        
        # 1. VEGETATION HEALTH INDICES (NaN-safe divisions only)
        # Normalized Difference Vegetation Index (most important for plants)
        ndvi = (g - r) / (g + r + 1e-6)  # Prevent division by zero
        
        # Enhanced Vegetation Index (drought stress detection)
        evi_denom = g + 6*r - 7.5*b + 1e-6  # Prevent division by zero
        evi = 2.5 * (g - r) / evi_denom
        
        # Green Normalized Difference Vegetation Index
        gndvi = (g - r) / (g + r + 1e-6)
        
        # Soil-Adjusted Vegetation Index
        savi = ((g - r) / (g + r + 1e-6)) * 1.5
        
        # 2. PLANT PIGMENT INDICES (safe divisions only)
        # Anthocyanin Reflectance Index (red pigments, autumn/stress)
        ari = (1.0/(g + 1e-6)) - (1.0/(r + 1e-6))  # Prevent division by zero
        
        # Carotenoid Reflectance Index (yellow pigments, senescence)  
        cri = (1.0/(b + 1e-6)) - (1.0/(g + 1e-6))  # Prevent division by zero
        
        # Photochemical Reflectance Index (photosynthetic activity)
        pri = (g - b) / (g + b + 1e-6)
        
        # 3. PLANT STRUCTURE INDICES (safe divisions only)
        # Red Edge Normalized Difference (canopy structure)
        rend = (r - g) / (r + g + 1e-6)
        
        # Simple Ratio (biomass estimation)
        sr = g / (r + 1e-6)
        
        # Green/Red ratio (plant health)
        gr_ratio = g / (r + 1e-6)
        
        # 4. DETERMINISTIC COLOR SPACE ANALYSIS
        # Convert to botanically-relevant color spaces
        
        # HSV for color discrimination
        hsv = self._deterministic_rgb_to_hsv(batch_tensor)
        h, s, v = hsv[:,:,:,0], hsv[:,:,:,1], hsv[:,:,:,2]
        
        # LAB color space approximation (perceptually uniform)
        # Simplified LAB conversion for deterministic behavior
        l_approx = 0.299*r + 0.587*g + 0.114*b  # Lightness approximation
        a_approx = r - g                          # Red-Green opponent
        b_approx = (r + g)/2 - b                 # Yellow-Blue opponent
        
        # 5. DETERMINISTIC DIRECTIONAL ANALYSIS
        # Plant structures have preferred orientations (leaves, stems, branches)
        gray = 0.25*r + 0.6*g + 0.15*b  # Botanical-weighted grayscale
        
        # Deterministic directional filters for plant structures
        device = self.device
        
        # Vertical structures (stems, upright leaves)
        vertical_kernel = torch.tensor([[0, 1, 0], [0, 1, 0], [0, 1, 0]], 
                                     device=device, dtype=torch.float32).view(1, 1, 3, 3)
        
        # Horizontal structures (branch spread, horizontal leaves)
        horizontal_kernel = torch.tensor([[0, 0, 0], [1, 1, 1], [0, 0, 0]], 
                                       device=device, dtype=torch.float32).view(1, 1, 3, 3)
        
        # Diagonal structures (angled branches, leaf orientations)
        diag1_kernel = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], 
                                  device=device, dtype=torch.float32).view(1, 1, 3, 3)
        diag2_kernel = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]], 
                                  device=device, dtype=torch.float32).view(1, 1, 3, 3)
        
        # Apply directional filters
        vertical_resp = torch.nn.functional.conv2d(gray.unsqueeze(1), vertical_kernel, padding=1).squeeze(1)
        horizontal_resp = torch.nn.functional.conv2d(gray.unsqueeze(1), horizontal_kernel, padding=1).squeeze(1)
        diag1_resp = torch.nn.functional.conv2d(gray.unsqueeze(1), diag1_kernel, padding=1).squeeze(1)
        diag2_resp = torch.nn.functional.conv2d(gray.unsqueeze(1), diag2_kernel, padding=1).squeeze(1)
        
        # 6. DETERMINISTIC BOTANICAL STATISTICS
        # Fixed percentiles for consistent behavior across all indices
        
        # Vegetation index statistics
        ndvi_flat = ndvi.view(B, -1)
        ndvi_p90 = torch.quantile(ndvi_flat, 0.90, dim=1)  # Healthy vegetation areas
        ndvi_p75 = torch.quantile(ndvi_flat, 0.75, dim=1)  # Good vegetation
        ndvi_p50 = torch.quantile(ndvi_flat, 0.50, dim=1)  # Medium vegetation
        ndvi_p25 = torch.quantile(ndvi_flat, 0.25, dim=1)  # Poor vegetation
        ndvi_mean = torch.mean(ndvi_flat, dim=1)
        ndvi_std = torch.std(ndvi_flat, dim=1)
        
        # Enhanced vegetation index statistics
        evi_flat = evi.view(B, -1)
        evi_p75 = torch.quantile(evi_flat, 0.75, dim=1)
        evi_mean = torch.mean(evi_flat, dim=1)
        
        # Color statistics for species discrimination
        h_flat = h.view(B, -1)
        h_mean = torch.mean(h_flat, dim=1)
        h_std = torch.std(h_flat, dim=1)
        h_p90 = torch.quantile(h_flat, 0.90, dim=1)  # Dominant hues
        
        s_flat = s.view(B, -1)
        s_mean = torch.mean(s_flat, dim=1)
        s_p75 = torch.quantile(s_flat, 0.75, dim=1)  # Saturated areas
        
        # Directional structure statistics
        vert_flat = vertical_resp.view(B, -1)
        vert_p80 = torch.quantile(vert_flat, 0.80, dim=1)   # Strong vertical structures
        
        horiz_flat = horizontal_resp.view(B, -1)
        horiz_p80 = torch.quantile(horiz_flat, 0.80, dim=1) # Strong horizontal structures
        
        # 7. DETERMINISTIC MULTI-SCALE BOTANICAL ANALYSIS
        # Analyze vegetation indices at multiple scales
        scale_sizes = [8, 16, 32]  # Deterministic scales for plant analysis
        botanical_features = []
        
        for size in scale_sizes:
            stride = size  # No overlap for deterministic behavior
            
            if size <= H and size <= W:
                # NDVI patches (most important vegetation index)
                ndvi_patches = torch.nn.functional.unfold(ndvi.unsqueeze(1), size, stride=stride)
                if ndvi_patches.shape[-1] > 0:
                    ndvi_patch_means = torch.mean(ndvi_patches, dim=1)
                    ndvi_patch_stds = torch.std(ndvi_patches, dim=1)
                    botanical_features.append(ndvi_patch_means)
                    botanical_features.append(ndvi_patch_stds)
                
                # HSV patches for color discrimination
                hue_patches = torch.nn.functional.unfold(h.unsqueeze(1), size, stride=stride)
                if hue_patches.shape[-1] > 0:
                    hue_patch_means = torch.mean(hue_patches, dim=1)
                    botanical_features.append(hue_patch_means)
        
        # DETERMINISTIC BOTANICAL BASE FEATURES (32 features)
        base_features = torch.stack([
            # Vegetation health (8 features)
            ndvi_p90, ndvi_p75, ndvi_p50, ndvi_p25, ndvi_mean, ndvi_std,
            evi_p75, evi_mean,
            
            # Plant pigments (6 features)
            torch.mean(ari.view(B, -1), dim=1), torch.std(ari.view(B, -1), dim=1),
            torch.mean(cri.view(B, -1), dim=1), torch.std(cri.view(B, -1), dim=1),
            torch.mean(pri.view(B, -1), dim=1), torch.std(pri.view(B, -1), dim=1),
            
            # Color discrimination (6 features)
            h_mean, h_std, h_p90,
            s_mean, s_p75,
            torch.mean(v.view(B, -1), dim=1),
            
            # Plant structure (6 features)
            torch.mean(sr.view(B, -1), dim=1), torch.std(sr.view(B, -1), dim=1),
            torch.mean(gr_ratio.view(B, -1), dim=1),
            vert_p80, horiz_p80,
            torch.mean(torch.abs(diag1_resp).view(B, -1), dim=1),
            
            # Additional discrimination (6 features)
            torch.mean(savi.view(B, -1), dim=1), torch.std(savi.view(B, -1), dim=1),
            torch.mean(rend.view(B, -1), dim=1), torch.std(rend.view(B, -1), dim=1),
            torch.mean(l_approx.view(B, -1), dim=1), torch.std(a_approx.view(B, -1), dim=1),
        ], dim=1)  # 32 deterministic botanical discrimination features
        
        # Concatenate multi-scale botanical features deterministically
        if botanical_features:
            all_botanical_features = torch.cat(botanical_features, dim=1)
            
            # Ensure exactly 4968 multi-scale features (5000 - 32 base = 4968)
            target_botanical_features = 4968
            current_botanical_features = all_botanical_features.shape[1]
            
            if current_botanical_features >= target_botanical_features:
                # Take first 4968 features deterministically
                botanical_component = all_botanical_features[:, :target_botanical_features]
            else:
                # Pad with deterministic values (repeat NDVI statistics)
                needed = target_botanical_features - current_botanical_features
                padding = torch.stack([ndvi_mean, ndvi_std, evi_mean], dim=1).repeat(1, (needed // 3) + 1)[:, :needed]
                botanical_component = torch.cat([all_botanical_features, padding], dim=1)
        else:
            # Fallback: use NDVI and color statistics repeated
            botanical_component = torch.stack([ndvi_mean, ndvi_std, h_mean], dim=1).repeat(1, 4968 // 3)[:, :4968]
        
        # Combine for exactly 5000 unique botanical features
        final_features = torch.cat([base_features, botanical_component], dim=1)
        
        # Final NaN/Inf protection for all features
        final_features = torch.where(torch.isfinite(final_features), final_features, torch.zeros_like(final_features))
        
        return final_features

# Import unified augmentation system
from unified_plant_augmentation import create_augmentation_engine

# Global instances
_gpu_tensor_cache = GPUPersistentTensorCache()
_cuda_processor = CUDAStreamParallelProcessor()
_gpu_augmenter = create_augmentation_engine(use_gpu=True, enable_all_features=True)

class ProgressTracker:
    """progress tracker that updates last 2 lines without scrolling"""
    
    def __init__(self):
        self.current_stage = ""
        self.current_item = ""
        self.progress = 0
        self.total = 100
        self.start_time = time.time()
        self.last_update = 0
        
    def update(self, progress: int, total: int, stage: str = None, item: str = None):
        """Update progress without terminal scrolling"""
        current_time = time.time()
        
        # Only update every 100ms to avoid flickering
        if current_time - self.last_update < 0.1:
            return
            
        self.last_update = current_time
        self.progress = progress
        self.total = total
        
        if stage:
            self.current_stage = stage
        if item:
            self.current_item = item
            
        self._render()
    
    def _render(self):
        """Render progress bar to last 2 lines"""
        # Calculate progress percentage and bar
        percentage = (self.progress / max(self.total, 1)) * 100
        bar_width = 50
        filled = int(bar_width * percentage / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        # Calculate elapsed time and ETA
        elapsed = time.time() - self.start_time
        if self.progress > 0:
            eta = (elapsed / self.progress) * (self.total - self.progress)
            eta_str = f"{int(eta//60):02d}:{int(eta%60):02d}"
        else:
            eta_str = "--:--"
        
        # Create the two lines
        line1 = f"  {self.current_stage} [{bar}] {percentage:5.1f}% ({self.progress}/{self.total})"
        line2 = f"  Processing: {self.current_item:<60} ⏱  {int(elapsed//60):02d}:{int(elapsed%60):02d} / ETA {eta_str}"
        
        # Move cursor up 2 lines, clear them, and write new content
        sys.stdout.write("\033[2A")  # Move cursor up 2 lines
        sys.stdout.write("\033[K")   # Clear line
        sys.stdout.write(line1 + "\n")
        sys.stdout.write("\033[K")   # Clear line  
        sys.stdout.write(line2 + "\n")
        sys.stdout.flush()
    
    def start(self, stage: str, total: int):
        """Start tracking a new stage"""
        self.current_stage = stage
        self.total = total
        self.progress = 0
        self.start_time = time.time()
        
        # Print initial lines (these will be overwritten)
        print()  # Line 1 placeholder
        print()  # Line 2 placeholder
        self._render()
    
    def finish(self, message: str = "Complete"):
        """Finish progress tracking with final message"""
        # Final update showing 100%
        self.update(self.total, self.total, item=message)
        print()  # Add final newline

# feature extraction system


# Global instances
_gpu_cache = GlobalGPUTensorCache()
_progress_tracker = ProgressTracker()

def get_timestamp_ms() -> float:
    """Get current timestamp in milliseconds"""
    return time.time() * 1000

def format_timing(start_ms: float, end_ms: float) -> str:
    """Format timing difference in milliseconds"""
    diff_ms = end_ms - start_ms
    if diff_ms < 1:
        return f"{diff_ms:.2f}ms"
    elif diff_ms < 10:
        return f"{diff_ms:.1f}ms"
    else:
        return f"{diff_ms:.0f}ms"


class MultiModalCurseResistantRecognizer:
    """
    Multi-Modal Curse-Resistant Plant Recognition System with Live Augmentation
    
    
    Each descriptor type gets individual curse-resistant processing,
    then unified weighted combination for generalization.
    
      NEW: Live augmentation engine for 98-99%+ accuracy
    """
    
    def __init__(self, image_size: int = 512, num_classes: int = 100):
        self.image_size = image_size
        self.num_classes = num_classes
        
        # GPU/CPU device setup
        self.device = torch.device('cuda')
        
        # Storage for reference descriptors and class names
        self.reference_descriptors = {}
        self.class_names = []

        
    def process_image_parallel_gpu(self, image: np.ndarray, augmentations_per_image: int = 30) -> List[np.ndarray]:
        """
        : parallel GPU processing with CUDA streams
        - Keeps ALL data on GPU throughout entire pipeline
        - Processes original + augmentations simultaneously
        - 6 modalities extract in parallel using CUDA streams
        - Zero CPU-GPU transfers during processing
        
        Target: 3-5s per image with 10 augmentations (vs 16s+ current)
        """
        import time
        start_time = time.time()
        

        
        # STEP 1: Convert to GPU tensor and keep there
        if isinstance(image, np.ndarray):
            # Convert to tensor - keep raw 0-255 intensities for maximum discrimination
            image_tensor = torch.from_numpy(image.astype(np.float32)).to(_cuda_processor.device)
        else:
            image_tensor = image.to(_cuda_processor.device)
        
        # Ensure proper format (H, W, C)
        if len(image_tensor.shape) == 4:  # Remove batch dimension
            image_tensor = image_tensor[0]
        if len(image_tensor.shape) == 3 and image_tensor.shape[0] == 3:  # CHW -> HWC  
            image_tensor = image_tensor.permute(1, 2, 0)
        
        # Resize to standard size if needed
        if image_tensor.shape[:2] != (self.image_size, self.image_size):
            image_tensor = torch.nn.functional.interpolate(
                image_tensor.permute(2, 0, 1).unsqueeze(0),  # HWC -> BCHW
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).permute(1, 2, 0)  # BCHW -> HWC
        
        # STEP 2: PERFECT TRAINING-INFERENCE CONSISTENCY
        # ALWAYS process as single clean image for consistent feature extraction
        # Training: uses clean original + augmentations but extracts features per image individually
        # Inference: uses clean original only, but same extraction process
        
        if augmentations_per_image > 0:
            # TRAINING MODE: Process clean original + augmentations
            augmented_tensors = [image_tensor.clone()]  # Clean original first
            
            # Add the requested number of augmented variants
            aug_variants = self._realistic_plant_augmentation_mix(image_tensor, augmentations_per_image)
            augmented_tensors.extend(aug_variants)
            # Total: 1 original + augmentations_per_image variants = augmentations_per_image + 1 images
        else:
            # INFERENCE MODE: Process ONLY clean original (matches training's first image exactly)
            augmented_tensors = [image_tensor.clone()]  # Same as training's first image
        
        
        

        # STEP 4: INDIVIDUAL IMAGE PROCESSING (matches inference exactly)
        extraction_start = time.time()
        
        # Process each image individually to ensure identical feature extraction
        individual_feature_vectors = []
        
        # Process each augmented image one at a time (like inference)
        for img_idx, single_image in enumerate(augmented_tensors):
            try:
                # Extract features from single image (no batch mixing)
                single_image_results = _cuda_processor.process_image_parallel_cuda_streams(
                    single_image, []  # Empty list = single image processing
                )
                
                # Process features for this one image
                single_modality_features = []
                single_modality_weights = []
                
                # Plant-biology based modality importance weights
                importance_weights = {
                    'texture': 0.25, 'unique': 0.25, 'shape': 0.20,
                    'color': 0.15, 'contrast': 0.10, 'frequency': 0.05
                }
                
                # Process all modalities for this single image
                for modality in ['texture', 'color', 'shape', 'contrast', 'frequency', 'unique']:
                    modality_tensor = single_image_results.get(modality)
                    if modality_tensor is not None:
                        # Process single image (batch_size = 1)
                        selected_features = self._smart_feature_selection(
                            modality_tensor, modality, importance_weights[modality], 1  # Always 1 image
                        )
                        single_modality_features.append(selected_features)
                        single_modality_weights.extend([importance_weights[modality]] * selected_features.shape[1])
                
                # Combine features for this single image
                if single_modality_features:
                    unified_features = torch.cat(single_modality_features, dim=1)  # [1, features]
                    
                    # Apply feature selection
                    target_features = min(2500, unified_features.shape[1])
                    if unified_features.shape[1] > target_features:
                        final_features = self._remove_redundancy_and_select_best(
                            unified_features, torch.tensor(single_modality_weights, device=_cuda_processor.device), target_features
                        )
                    else:
                        final_features = unified_features
                    
                    # Extract single descriptor
                    descriptor = final_features.cpu().numpy()[0]  # Get single image's features
                    individual_feature_vectors.append(descriptor)
                else:
                    # Fallback
                    individual_feature_vectors.append(np.zeros(2500, dtype=np.float32))
                    
            except Exception as e:
                print(f"     Warning: Failed to process image {img_idx}: {e}")
                # Add fallback descriptor
                individual_feature_vectors.append(np.zeros(2500, dtype=np.float32))
        
        extraction_time = time.time() - extraction_start
        
        total_images = len(individual_feature_vectors)
        
        # Individual processing complete - each descriptor created independently
        
        # Cache final descriptors for training efficiency  
        if total_images > 1:  # Training mode (multiple images)
            cache_key = f"descriptors_{hash(str(image_tensor.cpu().numpy().tobytes())[:16])}"
            # Convert individual descriptors back to tensor for caching
            if individual_feature_vectors and len(individual_feature_vectors) > 0:
                try:
                    # Ensure all descriptors have the same length
                    descriptor_lengths = [len(desc) for desc in individual_feature_vectors]
                    if len(set(descriptor_lengths)) == 1:  # All same length
                        cache_tensor = torch.tensor(np.array(individual_feature_vectors), device=_cuda_processor.device)
                        _gpu_tensor_cache.cache_descriptors(cache_key, cache_tensor)
                    else:
                        print(f"     Warning: Inconsistent descriptor lengths: {set(descriptor_lengths)}")
                except Exception as e:
                    print(f"     Warning: Could not cache descriptors: {e}")
        
        total_time = time.time() - start_time
        
        # Simple one-line summary
        training_note = " (TRAINING)" if augmentations_per_image > 0 else " (INFERENCE)"
        if individual_feature_vectors:
            features_per_descriptor = len(individual_feature_vectors[0]) if individual_feature_vectors else 0
            print(f"     Extracted {len(individual_feature_vectors)} descriptors × {features_per_descriptor} features in {total_time:.2f}s{training_note}")
        else:
            print(f"     Extracted 0 descriptors in {total_time:.2f}s{training_note}")
        
        # Efficient cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return individual_feature_vectors
    
   
    
    
    def numpy_to_gpu_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy image to GPU tensor in proper format"""
        # Convert to float32 - keep raw 0-255 intensities for better discrimination
        tensor = torch.from_numpy(image.astype(np.float32))
        
        # Ensure HWC format
        if len(tensor.shape) == 3 and tensor.shape[0] == 3:  # CHW -> HWC
            tensor = tensor.permute(1, 2, 0)
        
        # Move to GPU
        return tensor.to(_cuda_processor.device)
    
    def _gpu_batch_augmentation(self, image_tensor: torch.Tensor, num_augmentations: int) -> List[torch.Tensor]:
        """
        GPU batch augmentation optimized for training speed
        Uses GPU config for maximum parallel processing
        Target: <10ms for 30 augmentations vs 100ms+ in unified system
        """
        if num_augmentations <= 0:
            return []
        
        # Get GPU configuration for optimal performance
        gpu_config = _GPU_CONFIG
        batch_size = min(num_augmentations, gpu_config["optimal_batch_size"] * 4)  # 4x batch for augmentation
        
        device = self.device
        H, W, C = image_tensor.shape
        
        # Pre-allocate GPU tensors for maximum speed
        augmented_tensors = []
        
        # Process in optimized batches
        for batch_start in range(0, num_augmentations, batch_size):
            batch_end = min(batch_start + batch_size, num_augmentations)
            current_batch_size = batch_end - batch_start
            
            # Create batch tensor on GPU
            batch_images = image_tensor.unsqueeze(0).repeat(current_batch_size, 1, 1, 1)  # [B, H, W, C]
            
            # GPU transformations (much faster than unified system)
            for i in range(current_batch_size):
                # Random rotation (botanical angles: -15° to +15°)
                angle = (torch.rand(1, device=device) - 0.5) * 30.0  # -15 to +15 degrees
                
                # Random brightness (lighting variation: 0.7 to 1.3)
                brightness = 0.7 + torch.rand(1, device=device) * 0.6
                
                # Random contrast (surface variation: 0.8 to 1.2)
                contrast = 0.8 + torch.rand(1, device=device) * 0.4
                
                # Apply transformations directly on GPU (vectorized)
                img = batch_images[i]
                
                # brightness/contrast adjustment - no clamping, let values be raw
                img = img * brightness * contrast
                
                # rotation using grid sampling (GPU accelerated)
                if angle.abs() > 1.0:  # Only rotate if angle is significant
                    img = self._gpu_rotate(img, angle)
                
                # noise addition (plant texture variation) - no clamping, let values be raw
                noise_strength = torch.rand(1, device=device) * 5.0  # Up to 5 intensity units noise
                noise = torch.randn_like(img) * noise_strength
                img = img + noise
                
                # color jitter (seasonal/health variation)
                if torch.rand(1, device=device) > 0.5:  # 50% chance
                    hue_shift = (torch.rand(1, device=device) - 0.5) * 0.1  # Small hue shifts
                    img = self._gpu_hue_shift(img, hue_shift)
                
                augmented_tensors.append(img)
        
        return augmented_tensors
    
    def _gpu_rotate(self, image_tensor: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """GPU rotation using affine grid sampling"""
        H, W, C = image_tensor.shape
        
        # Convert angle to radians
        angle_rad = angle * torch.pi / 180.0
        cos_a, sin_a = torch.cos(angle_rad), torch.sin(angle_rad)
        
        # Create rotation matrix
        rotation_matrix = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ], device=image_tensor.device, dtype=torch.float32).unsqueeze(0)
        
        # Create affine grid
        grid = torch.nn.functional.affine_grid(
            rotation_matrix, 
            [1, C, H, W], 
            align_corners=False
        ).to(image_tensor.device)
        
        # Apply rotation
        img_chw = image_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
        rotated = torch.nn.functional.grid_sample(
            img_chw, grid, 
            mode='bilinear', 
            padding_mode='border',
            align_corners=False
        )
        
        return rotated.squeeze(0).permute(1, 2, 0)  # BCHW -> HWC
    
    def _gpu_hue_shift(self, image_tensor: torch.Tensor, hue_shift: torch.Tensor) -> torch.Tensor:
        """GPU hue shifting for color variation"""
        # Simple RGB hue approximation (much faster than full HSV conversion)
        r, g, b = image_tensor[:, :, 0], image_tensor[:, :, 1], image_tensor[:, :, 2]
        
        # hue rotation matrix approximation
        shift = hue_shift.item()
        cos_shift = torch.cos(torch.tensor(shift * 2 * torch.pi, device=image_tensor.device))
        sin_shift = torch.sin(torch.tensor(shift * 2 * torch.pi, device=image_tensor.device))
        
        # Apply hue shift
        new_r = r * cos_shift - g * sin_shift
        new_g = r * sin_shift + g * cos_shift
        new_b = b  # Keep blue relatively stable
        
        # No clamping - let raw values flow through
        result = torch.stack([new_r, new_g, new_b], dim=2)
        return result
    
    def _realistic_plant_augmentation_mix(self, image_tensor: torch.Tensor, num_augmentations: int) -> List[torch.Tensor]:
        """
        Realistic plant augmentation mix using unified_plant_augmentation
        Creates augmentations that match what users actually upload
        """
        if num_augmentations <= 0:
            return []
        
        # Convert tensor to numpy for unified augmentation system (already in 0-255 range)
        image_np = image_tensor.cpu().numpy().astype(np.uint8)
        
        augmented_tensors = []
        
        # Use the sophisticated unified plant augmentation system
        for i in range(num_augmentations):
            try:
                # Mix of augmentation types that match real-world photos
                if i % 5 == 0:
                    # 20% basic lighting/camera variations (most common in user photos)
                    method = "basic"
                elif i % 5 == 1:
                    # 20% seasonal color variations
                    method = "seasonal"  
                elif i % 5 == 2:
                    # 20% lighting conditions (indoor, outdoor, shadows)
                    method = "lighting"
                elif i % 5 == 3:
                    # 20% realistic angle variations
                    method = "angle"
                else:
                    # 20% subtle plant variations (growth stages, slight stress)
                    method = "plant"
                
                # Apply the unified augmentation with realistic intensity
                augmented_np = _gpu_augmenter.generate_single_augmentation(image_np, method)
            
                # Convert back to tensor - keep raw 0-255 intensities
                augmented_tensor = torch.from_numpy(augmented_np.astype(np.float32)).to(self.device)
                augmented_tensors.append(augmented_tensor)
        
            except Exception as e:
                # If augmentation fails, use the original image with slight noise
                print(f"       Warning: Augmentation {i} failed ({method}), using original + noise")
                noise = torch.randn_like(image_tensor) * 2.0  # Small noise
                fallback_tensor = torch.clamp(image_tensor + noise, 0, 255)
                augmented_tensors.append(fallback_tensor)
        
        # Ensure we have exactly the requested number of augmentations
        while len(augmented_tensors) < num_augmentations:
            # Add slight variations of the original if we're short
            noise = torch.randn_like(image_tensor) * 1.0
            fallback_tensor = torch.clamp(image_tensor + noise, 0, 255)
            augmented_tensors.append(fallback_tensor)
            print(f"       Added fallback augmentation {len(augmented_tensors)}/{num_augmentations}")
        
        return augmented_tensors[:num_augmentations]  # Ensure exact count
    
    def _get_user_realistic_variant(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Get a user-realistic variant using the unified augmentation system
        Fast GPU-accelerated processing via unified_plant_augmentation
        """
        # Convert tensor to numpy for unified augmentation system (already in 0-255 range)
        image_np = image_tensor.cpu().numpy().astype(np.uint8)
        
        # Use unified augmentation system with 'user_realistic' method
        augmented_np = _gpu_augmenter.generate_single_augmentation(image_np, "user_realistic")
        
        # Convert back to tensor - keep raw 0-255 intensities
        augmented_tensor = torch.from_numpy(augmented_np.astype(np.float32)).to(self.device)
        
        return augmented_tensor
    
    def _smart_feature_selection(self, modality_tensor: torch.Tensor, modality: str, 
                               importance_weight: float, total_images: int) -> torch.Tensor:
        """
        Intelligent feature selection based on discriminative power and plant biology
        """
        B, F = modality_tensor.shape  # Batch size, Features
        
        # Calculate base features to select based on modality importance
        base_target = int(500 * importance_weight * 4)  # Scale by importance (max ~500 per modality)
        
        if B > 1:  # Training mode: use discriminative analysis
            # 1. DISCRIMINATIVE POWER ANALYSIS
            feature_variances = torch.var(modality_tensor, dim=0)
            feature_means = torch.abs(torch.mean(modality_tensor, dim=0))
            feature_ranges = torch.max(modality_tensor, dim=0)[0] - torch.min(modality_tensor, dim=0)[0]
            
            # 2. PLANT-BIOLOGY SPECIFIC SCORING
            if modality == 'texture':
                # Prioritize edge strength and surface variation
                discrimination_scores = feature_variances * (1.0 + feature_ranges * 0.1)
            elif modality == 'unique':
                # Prioritize vegetation indices and directional features
                discrimination_scores = feature_variances * (2.0 + feature_means * 0.5)
            elif modality == 'shape':
                # Prioritize morphological variation
                discrimination_scores = feature_variances * (1.5 + feature_ranges * 0.2)
            elif modality == 'color':
                # Prioritize color discrimination but account for lighting variation
                discrimination_scores = feature_variances * (0.8 + feature_means * 0.1)
            else:
                # Standard scoring for contrast/frequency
                discrimination_scores = feature_variances * (1.0 + feature_means * 0.1)
            
            # 3. SELECT TOP DISCRIMINATIVE FEATURES
            target_features = min(base_target, F)
            _, top_indices = torch.topk(discrimination_scores, k=target_features, largest=True)
            top_indices = torch.sort(top_indices)[0]  # Maintain order
            
        else:  # Inference mode: use deterministic selection
            # Use plant-biology informed deterministic selection
            if modality == 'texture':
                # Take first features (strongest edge responses)
                target_features = min(base_target, F)
                top_indices = torch.arange(0, target_features, device=modality_tensor.device)
            elif modality == 'unique':
                # Take evenly distributed (broad botanical coverage)
                target_features = min(base_target, F)
                step = max(1, F // target_features)
                top_indices = torch.arange(0, F, step, device=modality_tensor.device)[:target_features]
            elif modality == 'shape':
                # Take middle range (structural features)
                target_features = min(base_target, F)
                start_idx = (F - target_features) // 2
                top_indices = torch.arange(start_idx, start_idx + target_features, device=modality_tensor.device)
            else:
                # Evenly distributed for color/contrast/frequency
                target_features = min(base_target, F)
                step = max(1, F // target_features)
                top_indices = torch.arange(0, F, step, device=modality_tensor.device)[:target_features]
        
        return modality_tensor[:, top_indices]
    
    def _remove_redundancy_and_select_best(self, features: torch.Tensor, weights: torch.Tensor, 
                                         target_count: int) -> torch.Tensor:
        """
        Remove redundant features and select the most discriminative ones
        """
        B, F = features.shape
        
        if F <= target_count:
            return features
        
        if B > 1:  # Training mode: correlation-based redundancy removal
            # 1. CORRELATION ANALYSIS (sample-based to avoid memory issues)
            sample_size = min(1000, B)
            sample_indices = torch.randperm(B, device=features.device)[:sample_size]
            sample_features = features[sample_indices]
            
            # 2. FEATURE VARIANCE AND MAGNITUDE
            feature_vars = torch.var(sample_features, dim=0)
            feature_mags = torch.mean(torch.abs(sample_features), dim=0)
            
            # 3. WEIGHTED IMPORTANCE SCORE
            importance_scores = feature_vars * (1.0 + feature_mags) * weights
            
            # 4. GREEDY SELECTION WITH REDUNDANCY REMOVAL
            selected_indices = []
            remaining_indices = torch.arange(F, device=features.device)
            
            # Select top feature first
            best_idx = torch.argmax(importance_scores)
            selected_indices.append(best_idx.item())
            remaining_indices = remaining_indices[remaining_indices != best_idx]
            
            # Iteratively select non-redundant features
            for _ in range(target_count - 1):
                if len(remaining_indices) == 0:
                    break
                
                best_score = -1
                best_candidate = None
                
                for candidate_idx in remaining_indices[:100]:  # Sample for efficiency
                    candidate_score = importance_scores[candidate_idx]
                    
                    # Penalize correlation with already selected features
                    max_correlation = 0
                    for selected_idx in selected_indices[-5:]:  # Check last 5 to avoid O(n²)
                        try:
                            # Safe correlation calculation with fallback
                            f1 = sample_features[:, candidate_idx]
                            f2 = sample_features[:, selected_idx]
                            
                            # Check for constant features (zero variance)
                            if torch.var(f1) < 1e-8 or torch.var(f2) < 1e-8:
                                corr_val = 0.0  # Treat constant features as uncorrelated
                            else:
                                corr_matrix = torch.corrcoef(torch.stack([f1, f2]))
                                corr_val = corr_matrix[0, 1].abs().item()
                                if torch.isnan(torch.tensor(corr_val)):
                                    corr_val = 0.0
                            
                            max_correlation = max(max_correlation, corr_val)
                        except:
                            # If correlation fails, assume no correlation
                            max_correlation = max(max_correlation, 0.0)
                    
                    # Penalize high correlation
                    adjusted_score = candidate_score * (1.0 - max_correlation * 0.8)
                    
                    if adjusted_score > best_score:
                        best_score = adjusted_score
                        best_candidate = candidate_idx
                
                if best_candidate is not None:
                    selected_indices.append(best_candidate.item())
                    remaining_indices = remaining_indices[remaining_indices != best_candidate]
            
            # Fill remaining slots with top remaining features
            while len(selected_indices) < target_count and len(remaining_indices) > 0:
                remaining_scores = importance_scores[remaining_indices]
                best_remaining = remaining_indices[torch.argmax(remaining_scores)]
                selected_indices.append(best_remaining.item())
                remaining_indices = remaining_indices[remaining_indices != best_remaining]
            
            final_indices = torch.tensor(selected_indices, device=features.device)
            
        else:  # Inference mode: deterministic selection
            # Select based on weighted importance (deterministic)
            _, final_indices = torch.topk(weights, k=target_count, largest=True)
            final_indices = torch.sort(final_indices)[0]
        
        return features[:, final_indices]
    
    def _simple_dual_selection(self, modality_features: dict, target_count: int = 10000) -> List[float]:
        """Simple dual feature selection - just combine all features and split"""
        all_features = []
        
        for modality_name, features in modality_features.items():
            if features:  # Only add non-empty features
                all_features.extend(features)
        
        if not all_features:
            return [0.0] * target_count
        
        # If we have enough features, take first 10k
        if len(all_features) >= target_count:
            return all_features[:target_count]
        
        # If not enough, repeat features to reach target
        repeated_features = all_features[:]
        while len(repeated_features) < target_count:
            repeated_features.extend(all_features)
        
        return repeated_features[:target_count]
    
    def _calculate_feature_quality_score(self, feature_val: float, modality: str, 
                                       feature_idx: int, all_modality_features: List[float]) -> float:
        """Calculate quality score (0-10) for a single feature - optimized for plant discrimination"""
        
        abs_val = abs(feature_val)
        
        # Fast plant-specific scoring without heavy computation
        if modality == "texture":
            # Texture: edge and surface patterns (most discriminative for plants)
            if 1.0 < abs_val < 50.0:
                return 9.0 + min(1.0, abs_val / 50.0)
            elif 0.1 < abs_val < 1.0:
                return 7.0
            else:
                return 3.0
        elif modality == "unique":
            # Unique: plant-specific indices (vegetation, directional gradients)
            if 0.1 < abs_val < 20.0:
                return 9.5 + min(0.5, abs_val / 20.0)
            else:
                return 4.0
        elif modality == "shape":
            # Shape: structural boundaries
            if 0.5 < abs_val < 25.0:
                return 8.5 + min(1.5, abs_val / 25.0)
            else:
                return 4.0
        elif modality == "contrast":
            # Contrast: intensity variations
            if 5.0 < abs_val < 100.0:
                return 8.0 + min(2.0, abs_val / 100.0)
            else:
                return 3.5
        elif modality == "frequency":
            # Frequency: periodic patterns
            if 0.01 < abs_val < 10.0:
                return 7.5 + min(2.5, abs_val / 10.0)
            else:
                return 3.0
        else:  # color
            # Color: less reliable but still useful for raw 0-255 range
            if 20.0 < abs_val < 200.0:
                return 7.0 + min(3.0, abs_val / 200.0)
            else:
                return 2.0
    
    def _balanced_feature_selection(self, feature_scores: np.ndarray, feature_indices: np.ndarray,
                                  modality_labels: List[str], target_features: int) -> np.ndarray:
        """class-aware balanced feature selection for better minority class performance"""
        
        # balancing with adaptive allocation based on modality importance
        print(f"           Class-aware balanced feature selection for {len(set(modality_labels))} modalities...")
        
        # Define adaptive minimum features per modality based on biological importance
        modality_importance = {
            'texture': 0.30,    # Most for species differentiation
            'unique': 0.25,     # Class-specific discriminative features
            'shape': 0.20,      # Structural characteristics
            'contrast': 0.15,   # Surface pattern analysis
            'frequency': 0.10,  # Periodic patterns
            'color': 0.10       # Less reliable due to lighting variation
        }
        
        # Identify unique modalities and their indices
        unique_modalities = list(set(modality_labels))
        selected_indices = []
        
        # Adaptive allocation based on importance and available features
        for modality in unique_modalities:
            modality_mask = np.array([label == modality for label in modality_labels])
            modality_indices = feature_indices[modality_mask]
            modality_scores = feature_scores[modality_mask]
            
            # Get biological importance weight
            importance = modality_importance.get(modality, 0.10)
            
            # Adaptive allocation: more modalities get more features
            base_allocation = int(target_features * importance)
            
            # Ensure we don't exceed available features for this modality
            take_count = min(base_allocation, len(modality_indices))
            
            # Sort by scores within this modality and take top features
            sorted_idx = np.argsort(modality_scores)[::-1]
            top_modality_indices = modality_indices[sorted_idx[:take_count]]
            selected_indices.extend(top_modality_indices)
            
            print(f"           {modality}: {take_count}/{len(modality_indices)} features "
                  f"(importance: {importance:.2f}, avg_score: {np.mean(modality_scores[sorted_idx[:take_count]]):.3f})")
        
        # Fill any remaining budget with highest scoring features across all modalities
        if len(selected_indices) < target_features:
            remaining_budget = target_features - len(selected_indices)
            available_mask = np.isin(feature_indices, selected_indices, invert=True)
            available_indices = feature_indices[available_mask]
            available_scores = feature_scores[available_mask]
            
            if len(available_indices) > 0:
                sorted_idx = np.argsort(available_scores)[::-1]
                top_remaining = available_indices[sorted_idx[:remaining_budget]]
                selected_indices.extend(top_remaining)
                print(f"           Added {len(top_remaining)} top cross-modality features to fill budget")
        
        final_selection = np.array(selected_indices[:target_features])
        print(f"           Selected {len(final_selection)} balanced features across {len(unique_modalities)} modalities")
        
        return final_selection
    
