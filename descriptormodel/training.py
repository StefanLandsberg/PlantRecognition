import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle
import os
import sys
import time
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import classification_report, confusion_matrix
from concurrent.futures import ThreadPoolExecutor
import threading

from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_

from hyper_detailed_pattern_recognition import MultiModalCurseResistantRecognizer

from unified_plant_augmentation import UnifiedPlantAugmentationEngine, AugmentationConfig
import random
import json
import multiprocessing

from hyper_detailed_pattern_recognition import load_gpu_config

_TRAINING_GPU_CONFIG = load_gpu_config()

# Direct Classification Network for One-Shot Learning
class DirectClassificationNetwork(nn.Module):
    """
    Direct classification network optimized for one-shot plant recognition.
    Instead of prototypes, directly learns to classify plant features into classes.
    Works well with limited data per class.
    """
    
    def __init__(self, feature_dim: int, num_classes: int, hidden_dim: int = 512, dropout: float = 0.5, actual_samples: int = None):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        # ADAPTIVE PARAMETER SCALING: Mathematical equation based on dataset size
        # Use actual sample count if provided, otherwise estimate
        if actual_samples is not None:
            estimated_samples = actual_samples
        else:
            estimated_samples = num_classes * 50 * 30  # Conservative estimate
        
        # ADAPTIVE EQUATION: Hidden size scales with sqrt(samples) for optimal param/data ratio
        # Formula: hidden = base_size * sqrt(samples / base_samples) * feature_scaling
        
        base_samples = 1000           # Reference point for scaling
        base_hidden = 64              # Hidden size for base_samples
        feature_scaling = min(2.0, feature_dim / 2500)  # Scale with feature complexity
        
        # Square root scaling provides good balance between capacity and overfitting
        sample_scaling = (estimated_samples / base_samples) ** 0.5
        
        # Complete adaptive formula
        calculated_hidden = int(base_hidden * sample_scaling * feature_scaling)
        
        # Smooth bounds using tanh for natural limiting (no hard clamps)
        import math
        
        # Minimum bound: approaches 16 for very small datasets
        min_hidden = 16 + 48 * math.tanh((estimated_samples - 500) / 1000)
        
        # Maximum bound: approaches 2048 for very large datasets  
        max_hidden = 2048 * math.tanh(estimated_samples / 50000)
        
        # Apply smooth bounds
        self.hidden_dim = max(int(min_hidden), min(calculated_hidden, int(max_hidden), hidden_dim))
        
        # For very large datasets, consider adding depth instead of just width
        self.use_extra_depth = num_classes > 150
        if self.use_extra_depth:
            # Cap width and add depth for better parameter efficiency
            self.hidden_dim = min(self.hidden_dim, 2048)
            print(f"   Large dataset detected: Using extra depth for {num_classes} classes")
        
        print(f"  DIRECT CLASSIFICATION NETWORK:")
        print(f"   Input: {feature_dim:,} features")
        print(f"   Hidden: {self.hidden_dim} (adaptive: {estimated_samples:,} samples → {calculated_hidden} calculated → {feature_scaling:.2f} feature scaling)")
        print(f"   Output: {num_classes} classes")
        arch_desc = "Adaptive classifier with residual connection"
        if self.use_extra_depth:
            arch_desc += " + extra depth"
        print(f"   Architecture: {arch_desc}")
        
        # OPTIMAL ARCHITECTURE: Scale capacity with classes and features
        # More classes need more capacity, but prevent overfitting
        base_capacity = max(256, num_classes * 32)  # 32 neurons per class minimum
        feature_scaling = max(512, feature_dim // 4)  # Scale with feature count
        self.hidden_dim = min(base_capacity, feature_scaling)  # Take the reasonable limit
        
        print(f"   ANTI-OVERFITTING ARCHITECTURE: Hidden {self.hidden_dim}")
        
        # Balanced architecture with moderate regularization
        self.hidden1 = nn.Linear(feature_dim, self.hidden_dim)
        self.hidden1_norm = nn.BatchNorm1d(self.hidden_dim, momentum=0.1)  # Normal updates
        self.hidden1_dropout = nn.Dropout(0.3)  # Moderate dropout
        
        self.hidden2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.hidden2_norm = nn.BatchNorm1d(self.hidden_dim // 2, momentum=0.1)
        self.hidden2_dropout = nn.Dropout(0.4)  # Slightly higher dropout
        
        # Remove third layer to reduce capacity
        self.hidden3 = None
        self.hidden3_norm = None
        self.hidden3_dropout = None
        
        self.classifier = nn.Linear(self.hidden_dim // 2, num_classes)
        
        # Enable learning components
        self.input_norm = None
        self.input_dropout = None
        self.residual_proj = None
        
        # Temperature parameter for confidence calibration (much higher to reduce overconfidence)
        self.temperature = nn.Parameter(torch.tensor(20.0))  # Much higher to combat overconfidence
        
        self._initialize_weights()
        
        # Calculate parameters for overfitting assessment
        total_params = sum(p.numel() for p in self.parameters())
        print(f"   Total parameters: {total_params:,}")
        print(f"   Adaptive design for {num_classes}-class plant recognition")
        print(f"   No bottleneck - maintains {self.hidden_dim} features throughout")
    
    def _initialize_weights(self):
        """Initialize weights for better one-shot learning with adaptive architecture"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier initialization for better few-shot performance
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Special initialization for residual projection if it exists
        if self.residual_proj is not None:
            # Initialize residual projection with smaller weights
            nn.init.xavier_uniform_(self.residual_proj.weight, gain=0.1)
            nn.init.constant_(self.residual_proj.bias, 0)
    
    def forward(self, features):
        """
        Forward pass - direct classification with adaptive architecture
        Args:
            features: Plant features [N, feature_dim]
        Returns:
            logits: Class logits [N, num_classes]
        """
        # Anti-overfitting forward pass
        x = features
        
        # First hidden layer
        x = self.hidden1(x)
        x = self.hidden1_norm(x)
        x = F.relu(x)
        x = self.hidden1_dropout(x)
        
        # Second hidden layer
        x = self.hidden2(x)
        x = self.hidden2_norm(x)
        x = F.relu(x)
        x = self.hidden2_dropout(x)
        
        # No third layer - directly to classifier
        
        # Final classification layer
        logits = self.classifier(x)
        
        # Apply temperature scaling for better confidence calibration
        scaled_logits = logits / self.temperature
        return scaled_logits
    
    def predict_with_confidence(self, features):
        """
        Predict with confidence scores
        Args:
            features: Plant features [N, feature_dim]
        Returns:
            predictions: Class predictions [N]
            probabilities: Class probabilities [N, num_classes]
            confidences: Confidence scores [N]
        """
        self.eval()
        with torch.no_grad():
            logits = self(features)
            probabilities = F.softmax(logits, dim=1)
            
            # Get predictions
            max_probs, predictions = torch.max(probabilities, dim=1)
            
            # Calculate properly calibrated confidence scores
            if probabilities.shape[1] > 1:
                sorted_probs, _ = torch.sort(probabilities, dim=1, descending=True)
                top1_prob = sorted_probs[:, 0]
                top2_prob = sorted_probs[:, 1] if sorted_probs.shape[1] > 1 else torch.zeros_like(top1_prob)
                
                # Improved confidence calculation with entropy consideration
                margin = top1_prob - top2_prob  # Gap between top 2
                entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1)
                max_entropy = torch.log(torch.tensor(probabilities.shape[1], dtype=torch.float32))
                normalized_entropy = entropy / max_entropy
                
                # Confidence = top probability - entropy penalty - uncertainty penalty
                confidences = top1_prob * (1.0 - 0.3 * normalized_entropy) * (0.7 + 0.3 * margin)
            else:
                confidences = max_probs
            
            # More conservative confidence bounds to prevent overconfidence
            confidences = confidences  # Max 85% confidence
            
            return predictions, probabilities, confidences

class ProgressTracker:
    """progress tracker that updates last 2 lines without scrolling"""
    
    def __init__(self):
        self.current_stage = ""
        self.current_item = ""
        self.progress = 0
        self.total = 100
        self.start_time = time.time()
        self.last_update = 0
        self._lock = threading.Lock()  # Thread-safe updates
        
    def update(self, progress: int, total: int, stage: str = None, item: str = None):
        """Update progress without terminal scrolling"""
        with self._lock:  # Thread-safe
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
        
        try:
            # Move cursor up 2 lines, clear them, and write new content
            sys.stdout.write("\033[2A")  # Move cursor up 2 lines
            sys.stdout.write("\033[K")   # Clear line
            sys.stdout.write(line1 + "\n")
            sys.stdout.write("\033[K")   # Clear line  
            sys.stdout.write(line2 + "\n")
            sys.stdout.flush()
        except:
            # Fallback for terminals that don't support cursor movement
            print(f"\r{line1}", end="", flush=True)
    
    def start(self, stage: str, total: int):
        """Start tracking a new stage"""
        self.current_stage = stage
        self.total = total
        self.progress = 0
        self.start_time = time.time()
        self.current_item = "Initializing..."
        
        # Print initial lines (these will be overwritten)
        print()  # Line 1 placeholder
        print()  # Line 2 placeholder
        self._render()
    
    def finish(self, message: str = "Complete"):
        """Finish progress tracking with final message"""
        # Final update showing 100%
        self.update(self.total, self.total, item=message)
        print()  # Add final newline
    
    def reset(self):
        """Reset progress tracker for new stage"""
        with self._lock:
            self.current_stage = ""
            self.current_item = ""
            self.progress = 0
            self.total = 100
            self.start_time = time.time()
            self.last_update = 0

# Global progress tracker
_progress_tracker = ProgressTracker()

# Removed AugmentationCounter - using ProgressTracker directly for cleaner code

class FeatureExtractionDataset:
    """Dataset for storing extracted features after parallel processing"""
    
    def __init__(self, features: List[np.ndarray], labels: List[int], class_names: List[str]):
        self.features = features
        self.labels = labels
        self.class_names = class_names
        
        # Calculate statistics (handle empty case)
        if features:
            self.feature_sizes = [len(f) for f in features]
            self.min_size = min(self.feature_sizes)
            self.max_size = max(self.feature_sizes)
            self.avg_size = np.mean(self.feature_sizes)
        else:
            self.feature_sizes = []
            self.min_size = 0
            self.max_size = 0
            self.avg_size = 0
        
        print(f"  EXTRACTED FEATURE DATASET CREATED (PLANT-FOCUSED):")
        print(f"   Total samples: {len(features):,}")
        print(f"   Classes: {len(class_names)}")
        if features:
            print(f"   Feature sizes: {self.min_size:,} to {self.max_size:,}")
            print(f"   Average features: {self.avg_size:,.0f}")
            print(f"     Using plant-focused extraction (vegetation-biased sampling)")
        else:
            print(f"      No features extracted - check extraction process")

class DynamicFeatureDataset(Dataset):
    """PyTorch dataset for variable-size features"""
    
    def __init__(self, features: List[np.ndarray], labels: List[int]):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), self.labels[idx]

def collate_dynamic_features(batch):
    """Collate function for raw descriptors - NO TRUNCATION"""
    features, labels = zip(*batch)
    
    # Convert to tensors - descriptors are already padded to same length
    feature_tensors = []
    for f in features:
        # Ensure features are 1D arrays
        if len(f.shape) > 1:
            f = f.flatten()
        feature_tensors.append(torch.FloatTensor(f))
    
    # All descriptors should now be same length (padded in stage2)
    # Stack into batch tensor [batch_size, descriptor_dim]
    batch_features = torch.stack(feature_tensors)
    
    return batch_features, torch.LongTensor(labels)

class TwoStageTrainer:
    """Two-stage trainer: 1) Feature extraction, 2) Neural network training"""
    
    def __init__(self, use_gpu: bool = True, max_parallel_images: int = None):
        print(f"  TWO-STAGE PLANT RECOGNITION TRAINER")
        print(f"   Stage 1: Simple feature extraction with augmentation")
        print(f"   Stage 2: Neural network training with full metrics")
        print(f"   GPU: {use_gpu and torch.cuda.is_available()}")
        
        self.use_gpu = use_gpu
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        
        # Initialize simple processor
        self.processor = SimpleImageProcessor()
        
        print(f"     Using SimpleImageProcessor with GPU optimization")
        print(f"     Target: 500ms per extraction with parallel modality processing")
        
        # Training metrics storage - INITIALIZE PROPERLY
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': [],
            'final_accuracy': 0.0,
            'total_training_time': 0.0,
            'epochs_completed': 0,
            'training_method': 'not_started'
        }
    
    def stage1_extract_all_features(self, training_data: Dict[str, List[str]], 
                                   augmentations_per_image: int = 30, force_extract_all: bool = False) -> FeatureExtractionDataset:
        """Stage 1: Check cache first, then extract if needed"""
        
        print(f"\n  STAGE 1: CACHE CHECK & FEATURE EXTRACTION")
        print(f"   Augmentations per image: {augmentations_per_image}")
        print(f"   Force extract all: {force_extract_all}")
        
        class_names = list(training_data.keys())
        
        # Step 1: Check if cache exists and can be bulk loaded
        if not force_extract_all:
            cache_dir = Path("data/plant_images/.descriptor_cache")
            print(f"     Checking cache directory: {cache_dir}")
            print(f"     Directory exists: {cache_dir.exists()}")
            
            if cache_dir.exists():
                cache_files = list(cache_dir.glob("*.npy"))
                print(f"     Cache files found: {len(cache_files)}")
                
                if cache_files:
                    print(f"     CACHE FOUND: {len(cache_files)} files - will bulk load ALL cached features")
                    
                    # Bulk load all cached features instantly
                    all_features, all_labels = self._bulk_load_all_cache(class_names)
                    
                    if all_features:
                        print(f"     CACHE LOADING SUCCESS: {len(all_features):,} features loaded")
                        print(f"     SKIPPING EXTRACTION: Using cached features")
                        
                        # Create feature dataset from cache
                        return FeatureExtractionDataset(all_features, all_labels, class_names)
                    else:
                        print(f"     Cache loading failed - will extract features")
                else:
                    print(f"     No .npy files in cache directory - will extract features")
            else:
                print(f"     No cache directory found - will extract features")
        else:
            print(f"     🔧 Forced extraction mode - ignoring cache")
        
        # Step 2: Extract features (only if cache not available or forced)
        print(f"\n  EXTRACTING FEATURES: {len(training_data)} classes")
        
        # Prepare image paths for processing
        all_image_paths_and_classes = []
        total_original_images = 0
        
        for class_idx, (class_name, image_paths) in enumerate(training_data.items()):
            for image_path in image_paths:
                all_image_paths_and_classes.append((image_path, class_idx, class_name))
            total_original_images += len(image_paths)
        
        print(f"     {total_original_images:,} images → {total_original_images * (augmentations_per_image + 1):,} augmented samples")
        
        start_time = time.time()
        
        # Extract features from all images
        extracted_data = self.processor.process_images_batch(
            all_image_paths_and_classes, 
            augmentations_per_image
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Convert extracted data to features and labels
        features = [data[0] for data in extracted_data]
        labels = [data[1] for data in extracted_data]
        
        print(f"\n  STAGE 1 COMPLETE:")
        print(f"     Extraction time: {total_time:.1f}s")
        print(f"     Features extracted: {len(features):,}")
        print(f"     Features cached for future use")
        
        # Create feature dataset
        return FeatureExtractionDataset(features, labels, class_names)
    

    
    def _bulk_load_all_cache(self, class_names: List[str]) -> Tuple[List[np.ndarray], List[int]]:
        """GPU-ACCELERATED BULK LOAD: Load ALL cache files using GPU memory mapping"""
        cache_dir = Path("data/plant_images/.descriptor_cache")
        
        print(f"\n  GPU-ACCELERATED BULK LOADING")
        
        if not cache_dir.exists():
            print(f"     No cache directory found at {cache_dir}")
            return [], []
        
        cache_files = list(cache_dir.glob("*.npy"))
        if not cache_files:
            print(f"     Cache directory empty at {cache_dir}")
            return [], []
        
        print(f"     GPU BULK LOAD: {len(cache_files)} cache files...")
        print(f"     Using GPU tensor operations for MAXIMUM speed")
        
        start_time = time.time()
        
        # GPU-ACCELERATED loading strategy
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"     Loading on: {device}")
        
        # CHUNKED loading for GPU memory efficiency (load 5000 files at a time)
        chunk_size = 5000
        all_features = []
        all_labels = []
        successful_loads = 0
        
        # Process cache files in GPU-optimized chunks
        for chunk_start in range(0, len(cache_files), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(cache_files))
            chunk_files = cache_files[chunk_start:chunk_end]
            
            print(f"     GPU Chunk {chunk_start//chunk_size + 1}: Loading {len(chunk_files)} files...")
            chunk_start_time = time.time()
            
            # GPU-optimized parallel loading
            import concurrent.futures
            
            def load_with_gpu_prep(cache_file):
                try:
                    # Use memory mapping for loading of large files
                    features = np.load(cache_file, mmap_mode='r')
                    if len(features) >= 2500:  # Valid feature vector
                        # Extract class info from filename
                        filename = cache_file.stem
                        
                        # CRITICAL FIX: Extract full class name properly
                        # Cache filename format: {class_name}_{image_name}_{suffix}
                        # Need to find the known class name that matches the filename start
                        class_name = None
                        for known_class in class_names:
                            if filename.startswith(known_class + '_'):
                                class_name = known_class
                                break
                        
                        if class_name is None:
                            # Fallback: try old method for backwards compatibility
                            first_part = filename.split('_')[0]
                            if first_part in class_names:
                                class_name = first_part
                            else:
                                return None  # Skip unknown class files
                        
                        try:
                            class_idx = class_names.index(class_name)
                            # Copy to avoid memory mapping issues
                            return (features.copy(), class_idx)
                        except ValueError:
                            return None
                except Exception:
                    pass
                return None
            
            # Use GPU-optimized worker count from config
            max_workers = min(len(chunk_files), _TRAINING_GPU_CONFIG["max_workers"]["batch_processing"])
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                chunk_results = list(executor.map(load_with_gpu_prep, chunk_files))
            
            # GPU batch processing for validation and optimization
            chunk_features = []
            chunk_labels = []
            
            for result in chunk_results:
                if result is not None:
                    chunk_features.append(result[0])
                    chunk_labels.append(result[1])
                    successful_loads += 1
            
            # GPU tensor validation for large chunks
            if chunk_features and torch.cuda.is_available() and len(chunk_features) > 500:
                print(f"       GPU validation on {len(chunk_features)} features...")
                try:
                    # Convert to GPU tensors for validation
                    feature_tensors = [torch.from_numpy(f).to(device, dtype=torch.float32) for f in chunk_features]
                    tensor_batch = torch.stack(feature_tensors)
                    
                    # GPU-accelerated validation (check for NaN/Inf)
                    valid_mask = torch.isfinite(tensor_batch).all(dim=1)
                    valid_indices = torch.where(valid_mask)[0].cpu().numpy()
                    
                    # Keep only valid features
                    for idx in valid_indices:
                        all_features.append(chunk_features[idx])
                        all_labels.append(chunk_labels[idx])
                    
                    # Clear GPU memory
                    del feature_tensors, tensor_batch, valid_mask
                    torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    print(f"       GPU validation failed ({e}), using CPU fallback")
                    # CPU fallback
                    all_features.extend(chunk_features)
                    all_labels.extend(chunk_labels)
            else:
                # Direct append for small chunks or CPU mode
                all_features.extend(chunk_features)
                all_labels.extend(chunk_labels)
            
            chunk_time = time.time() - chunk_start_time
            print(f"       Chunk complete: {chunk_time:.2f}s ({len(chunk_features)} features)")
        
        end_time = time.time()
        load_time = end_time - start_time
        
        print(f"     GPU BULK LOADING COMPLETE!")
        print(f"       Loaded: {successful_loads}/{len(cache_files)} files")
        print(f"       Valid features: {len(all_features):,} vectors")
        print(f"       Time: {load_time:.2f}s")
        print(f"       Speed: {len(all_features)/load_time:.0f} vectors/second")
        print(f"       GPU acceleration: {torch.cuda.is_available()}")
        
        return all_features, all_labels



    def _robust_normalize_features(self, X: np.ndarray) -> tuple[np.ndarray, object]:
        """
        Robust feature normalization for plant recognition
        Handles extreme values and ensures stable training
        
        Returns:
            tuple: (normalized_features, fitted_scaler)
        """
        print(f"     Normalizing features: {X.shape}")
        
        # Handle NaN and infinite values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Clip extreme outliers (beyond 5 standard deviations)
        for i in range(X.shape[1]):
            col = X[:, i]
            mean_val = np.mean(col)
            std_val = np.std(col)
            if std_val > 0:
                # Clip to 5 standard deviations
                lower_bound = mean_val - 5 * std_val
                upper_bound = mean_val + 5 * std_val
                X[:, i] = np.clip(col, lower_bound, upper_bound)
        
        # Standardize features (zero mean, unit variance)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)
        
        # Ensure no remaining NaN values
        X_normalized = np.nan_to_num(X_normalized, nan=0.0)
        
        print(f"       Normalized shape: {X_normalized.shape}")
        print(f"       Feature range: [{X_normalized.min():.3f}, {X_normalized.max():.3f}]")
        print(f"       Feature mean: {X_normalized.mean():.6f}")
        print(f"       Feature std: {X_normalized.std():.6f}")
        print(f"        SCALER FITTED: Will be saved with model for consistent inference")
        
        return X_normalized, scaler
    
    def stage2_train_neural_network(self, feature_dataset: FeatureExtractionDataset,
                               epochs: int = 60, batch_size: int = 32, 
                               learning_rate: float = 0.001, hidden_dim: int = 512) -> DirectClassificationNetwork:
        """
        Stage 2: Train neural network using direct classification approach
        
        UPDATED: Uses Direct Classification for one-shot learning (no prototypes)
        Perfect for personalized plant recognition with limited reference images
        """
        
        print(f"\n  STAGE 2: NEURAL NETWORK TRAINING")
        print(f"     Training samples: {len(feature_dataset.features):,}")
        print(f"     Classes: {len(feature_dataset.class_names)}")
        print(f"     Samples per class: ~{len(feature_dataset.features) / len(feature_dataset.class_names):.1f}")
        
        # Use direct classification approach for one-shot learning
        samples_per_class = len(feature_dataset.features) / len(feature_dataset.class_names)
        
        # Auto-configure calibration based on dataset size
        total_samples = len(feature_dataset.features)
        auto_disable_calibration = total_samples > 50000  # Disable for very large datasets
        calibration_sample_size = min(2000, total_samples // 10)  # Use 10% or max 2000 samples
        
        if auto_disable_calibration:
            print(f"     LARGE DATASET: Calibration check auto-disabled for speed ({total_samples:,} samples)")
            enable_calibration = False
        else:
            print(f"     Calibration check enabled ({calibration_sample_size:,} sample size)")
            enable_calibration = True
        
        print(f"     APPROACH: Direct Classification for One-Shot Learning")
        print(f"     DATA USAGE: 100% training, 0% validation waste")
        print(f"     PERFECT FOR: One-shot personalized plant recognition")
        
        return self.stage2_train_direct_classifier(
            feature_dataset,
            epochs=60,
            batch_size=32,
            learning_rate=0.001,
            hidden_dim=8
        )
    
    def stage2_train_direct_classifier(self, feature_dataset: FeatureExtractionDataset,
                                      epochs: int = 60, batch_size: int = 32,
                                      learning_rate: float = 0.001, hidden_dim: int = 512) -> DirectClassificationNetwork:
        """
        Stage 2: Train direct classifier for one-shot plant recognition
        """
        # Set default values for removed parameters
        enable_calibration_check = True
        calibration_sample_size = 1000
        
        print(f"\n  STAGE 2: DIRECT CLASSIFICATION TRAINING")
        print(f"     Training samples: {len(feature_dataset.features)} (ALL DATA USED)")
        print(f"     Classes: {len(feature_dataset.class_names)}")
        print(f"     Samples per class: ~{len(feature_dataset.features) / len(feature_dataset.class_names):.1f}")
        print(f"     Method: Direct classification for one-shot learning")
        print(f"     No prototypes - simple neural network")
        
        # Prepare data
        X = np.array(feature_dataset.features)
        y = np.array(feature_dataset.labels)
        
        # NO NORMALIZATION - Use raw features like inference
        print(f"     Raw features: mean={X.mean():.3f}, std={X.std():.3f}")
        fitted_scaler = None  # No scaler needed
        
        # Create model with actual sample count for adaptive sizing
        feature_dim = X.shape[1]
        num_classes = len(feature_dataset.class_names)
        actual_samples = len(feature_dataset.features)
        
        model = DirectClassificationNetwork(
            feature_dim=feature_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=0.5,
            actual_samples=actual_samples
        ).to(self.device)
        
        # BALANCED TRAINING SETUP
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,  # Normal learning rate for 10 classes
            weight_decay=0.001,  # Moderate L2 regularization
            betas=(0.9, 0.999)
        )
        
        # Learning rate with cosine annealing for better convergence
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=learning_rate * 0.01
        )
        
        # VALIDATION SPLIT: Essential for overfitting detection
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"     ANTI-OVERFITTING: Train={len(X_train)}, Val={len(X_val)}")
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )
        
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size*2, shuffle=False, drop_last=False
        )
        
        # Training loop
        model.train()
        training_history = {
            'train_losses': [],
            'train_accuracies': [],
            'confidence_scores': [],
            'epochs_completed': 0
        }
        
        print(f"     Training for up to {epochs} epochs with early stopping...")
        
        # VALIDATION-BASED EARLY STOPPING - prevent overfitting
        best_val_loss = float('inf')
        best_val_accuracy = 0.0
        best_model_state = None
        patience_counter = 0
        patience = 8  # Moderate patience to catch overfitting
        min_delta = 0.001  # Require meaningful improvement
        
        for epoch in range(epochs):
            # TRAINING PHASE
            model.train()
            train_losses = []
            train_accuracies = []
            
            for batch_X, batch_y in train_dataloader:
                optimizer.zero_grad()
                
                # Simple forward pass - no label smoothing
                logits = model(batch_X)
                loss = F.cross_entropy(logits, batch_y)  # Standard cross entropy
                
                # Aggressive learning backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # Higher clipping to allow bigger updates
                optimizer.step()
                
                # Calculate training metrics
                with torch.no_grad():
                    probabilities = F.softmax(logits, dim=1)
                    _, predictions = torch.max(probabilities, dim=1)
                    accuracy = (predictions == batch_y).float().mean().item()
                    
                    train_losses.append(loss.item())
                    train_accuracies.append(accuracy)
            
            # VALIDATION PHASE
            model.eval()
            val_losses = []
            val_accuracies = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_dataloader:
                    logits = model(batch_X)
                    val_loss = F.cross_entropy(logits, batch_y)
                    
                    probabilities = F.softmax(logits, dim=1)
                    _, predictions = torch.max(probabilities, dim=1)
                    val_accuracy = (predictions == batch_y).float().mean().item()
                    
                    val_losses.append(val_loss.item())
                    val_accuracies.append(val_accuracy)
            
            # Record epoch metrics
            avg_train_loss = np.mean(train_losses)
            avg_train_accuracy = np.mean(train_accuracies)
            avg_val_loss = np.mean(val_losses)
            avg_val_accuracy = np.mean(val_accuracies)
            
            # Update learning rate every epoch
            old_lr = optimizer.param_groups[0]['lr']
            # Update learning rate
            scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
            
            # Track LR changes
            lr_changed = abs(new_lr - old_lr) > 1e-8
            
            training_history['train_losses'].append(avg_train_loss)
            training_history['train_accuracies'].append(avg_train_accuracy)
            training_history['epochs_completed'] = epoch + 1
            
            # Simple early stopping
            improved = False
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                best_val_accuracy = avg_val_accuracy
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                improved = True
            else:
                patience_counter += 1
            
            # Progress reporting with validation metrics
            if epoch < 10 or epoch >= epochs - 5 or epoch % 5 == 4 or improved:
                status = "NEW BEST" if improved else ""
                lr_status = f"LR {new_lr:.6f}"
                if lr_changed:
                    lr_status += f" ({old_lr:.6f})"
                print(f"   Epoch {epoch+1:3d}: Train {avg_train_loss:.3f}/{avg_train_accuracy:.3f} | "
                      f"Val {avg_val_loss:.3f}/{avg_val_accuracy:.3f} | {lr_status} {status}")
            
            # Stop if validation doesn't improve
            if patience_counter >= patience:
                print(f"   Early stopping at epoch {epoch+1} (val loss no improvement for {patience} epochs)")
                break
        
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"   Restored best model (val loss: {best_val_loss:.4f}, val accuracy: {best_val_accuracy:.3f})")
        
        # Final evaluation and temperature calibration
        model.eval()
        print(f"\n  PERFORMING POST-TRAINING CONFIDENCE CALIBRATION...")
        
        # Use validation set for calibration (more realistic than training set)
        total_samples = len(X_val_tensor)
        if total_samples > 5000:
            # Large dataset: use sampling for speed
            max_sample_size = min(calibration_sample_size, total_samples // 5)  # Max 20% of data
            auto_sampling = True
        else:
            # Small dataset: use all data
            max_sample_size = total_samples
            auto_sampling = False
        
        if not enable_calibration_check:
            print(f"     Calibration check DISABLED - skipping confidence validation")
            # Skip calibration check entirely
            final_accuracy = None
            final_confidence = None
        else:
            calibration_start_time = time.time()
            
            if auto_sampling and total_samples > max_sample_size:
                # Sample representative subset for large datasets
                sample_indices = torch.randperm(total_samples)[:max_sample_size]
                X_sample = X_val_tensor[sample_indices]
                y_sample = y_val_tensor[sample_indices]
                print(f"     Using {max_sample_size:,} validation samples (of {total_samples:,}) for calibration check")
            else:
                # Use all validation data for calibration
                X_sample = X_val_tensor
                y_sample = y_val_tensor
                print(f"     Using all {total_samples:,} validation samples for calibration check")
            
            # Step 1: Get initial predictions and check for overconfidence
            with torch.no_grad():
                # Use simple forward pass instead of complex predict_with_confidence
                initial_logits = model(X_sample)
                initial_probs = F.softmax(initial_logits, dim=1)
                initial_predictions = torch.argmax(initial_probs, dim=1)
                initial_confidences = torch.max(initial_probs, dim=1)[0]  # Simple max probability
                
                initial_accuracy = (initial_predictions == y_sample).float().mean().item()
                
                # Quick overconfidence check
                wrong_predictions = (initial_predictions != y_sample)
                if wrong_predictions.any():
                    wrong_confidences = initial_confidences[wrong_predictions]
                    overconfident_errors = (wrong_confidences > 0.7).float().mean().item() if len(wrong_confidences) > 0 else 0.0
                    print(f"     Initial overconfidence rate: {overconfident_errors:.3f} (errors with >70% confidence)")
        
        # Step 2: Advanced temperature scaling for plant recognition (prevents same-plant overfitting)
        num_classes = len(feature_dataset.class_names)
        
        # RAW FEATURE TEMPERATURE SCALING (different from normalized features)
        # Raw features have much larger magnitude (0-255) vs normalized (-3 to +3)
        print(f"     RAW FEATURE temperature scaling for {num_classes} classes...")
        original_temp = model.temperature.item()
        
        if num_classes >= 50:
            # Large plant dataset: moderate temperature for raw features
            if num_classes >= 100:
                # Very large dataset: prevent overconfidence but not too aggressive
                print(f"     Very large dataset: Moderate scaling for raw features")
                model.temperature.data = torch.tensor(3.5)  # Higher temp for raw features
                print(f"     Temperature adjusted (raw features): {original_temp:.2f} → {model.temperature.item():.2f}")
            else:
                # Large dataset: balanced scaling for raw features
                print(f"     Large dataset: Balanced scaling for raw features")
                model.temperature.data = torch.tensor(3.0)  # Moderate temp for raw features
                print(f"     Temperature adjusted (raw features): {original_temp:.2f} → {model.temperature.item():.2f}")
        elif num_classes >= 20:
            # Medium datasets: light scaling for raw features
            print(f"     Medium dataset: Light scaling for raw features")
            model.temperature.data = torch.tensor(2.5)  # Light scaling for raw features
            print(f"     Temperature adjusted (raw features): {original_temp:.2f} → {model.temperature.item():.2f}")
        else:
            # Small datasets: minimal scaling for raw features
            print(f"     Small dataset: Minimal scaling for raw features")
            model.temperature.data = torch.tensor(2.0)  # Minimal scaling for raw features
            print(f"     Temperature adjusted (raw features): {original_temp:.2f} → {model.temperature.item():.2f}")
        
        # Step 3: Final calibrated evaluation
        if enable_calibration_check:
            with torch.no_grad():
                # Use simple forward pass for speed
                final_logits = model(X_sample)
                final_probs = F.softmax(final_logits, dim=1)
                final_predictions = torch.argmax(final_probs, dim=1)
                final_confidences = torch.max(final_probs, dim=1)[0]  # Simple confidence
                
                final_accuracy = (final_predictions == y_sample).float().mean().item()
                final_confidence = final_confidences.mean().item()
                
                # Calculate calibration metrics 
                wrong_predictions = (final_predictions != y_sample)
                if wrong_predictions.any():
                    wrong_confidences = final_confidences[wrong_predictions].mean().item()
                    correct_confidences = final_confidences[~wrong_predictions].mean().item()
                    print(f"     Calibration check - Correct predictions avg confidence: {correct_confidences:.3f}")
                    print(f"     Calibration check - Wrong predictions avg confidence: {wrong_confidences:.3f}")
                
            calibration_time = time.time() - calibration_start_time
            print(f"     Calibration check completed in {calibration_time:.2f}s")
        else:
            # Set defaults when calibration is disabled
            final_accuracy = initial_accuracy if 'initial_accuracy' in locals() else 0.0
            final_confidence = 0.5  # Default neutral confidence
        
        training_history['final_accuracy'] = final_accuracy
        training_history['final_confidence'] = final_confidence
        training_history['temperature_value'] = model.temperature.item()
        
        print(f"\n  DIRECT CLASSIFICATION TRAINING COMPLETE!")
        print(f"     Final accuracy: {final_accuracy:.3f}")
        print(f"     Final confidence: {final_confidence:.3f}")
        print(f"     Temperature parameter: {model.temperature.item():.3f}")
        print(f"     Model ready for calibrated plant recognition")
        
        # Save model with fitted scaler
        self._save_direct_classifier_model(model, feature_dataset, training_history, fitted_scaler)
        
        return model

    def _save_direct_classifier_model(self, model: DirectClassificationNetwork, 
                                     feature_dataset: FeatureExtractionDataset,
                                     training_history: dict,
                                     fitted_scaler: object,
                                     filepath: str = "trained_plant_model.pt"):
        """
        Save direct classifier model with metadata and fitted scaler
        """
        model_data = {
            'model_state_dict': model.state_dict(),
            'model_config': {
                'feature_dim': model.feature_dim,
                'num_classes': model.num_classes,
                'hidden_dim': model.hidden_dim,
            },
            'class_names': feature_dataset.class_names,
            'training_history': training_history,
            'model_type': 'direct_classification_network',
            'training_samples': len(feature_dataset.features),
            'num_classes': len(feature_dataset.class_names),
            'training_approach': 'direct_classification_one_shot',
            'data_utilization': '100% training, 0% validation waste',
            'feature_scaler': fitted_scaler  #  CRITICAL FIX: Save the fitted scaler
        }
        
        torch.save(model_data, filepath)
        
        print(f"  DIRECT CLASSIFIER MODEL SAVED: {filepath}")
        print(f"     Model type: Direct classification approach")
        print(f"     Classes: {len(feature_dataset.class_names)}")
        print(f"     Training method: {model_data['training_approach']}")
        print(f"     Data efficiency: {model_data['data_utilization']}")
        print(f"      SCALER SAVED: Feature normalization will be consistent in inference")
        
        return filepath

class SimpleImageProcessor:
    """Simple single-threaded image processor with GPU-accelerated feature extraction"""
    
    def __init__(self):
        """Initialize simple processor with GPU optimization"""
        self.recognizer = None
        self._cache_dir = None
        
        # Initialize sophisticated augmentation engine
        print("  Initializing UnifiedPlantAugmentationEngine...")
        aug_config = AugmentationConfig(
            use_gpu=torch.cuda.is_available(),
            enable_advanced_transforms=True,
            enable_mixing_methods=True,
            enable_realistic_angles=True,
            enable_plant_specific=True
        )
        self.augmentation_engine = UnifiedPlantAugmentationEngine(aug_config)
        print("      Advanced plant-specific augmentations enabled")
        
    def _get_recognizer(self):
        """Get or create the recognizer"""
        if self.recognizer is None:
            try:
                print("  Initializing GPU-accelerated feature recognizer...")
                from hyper_detailed_pattern_recognition import MultiModalCurseResistantRecognizer
                
                # Initialize with GPU optimization settings
                self.recognizer = MultiModalCurseResistantRecognizer()
                print("     Recognizer initialized successfully")
                
                # Enable GPU acceleration if available
                if hasattr(self.recognizer, 'enable_gpu_acceleration'):
                    self.recognizer.enable_gpu_acceleration(max_threads=True)
                    print("     GPU acceleration enabled - targeting 500ms extraction")
                
            except Exception as e:
                print(f"     ERROR: Failed to initialize recognizer: {e}")
                raise e
            
        return self.recognizer
    
    def _get_cache_path(self, image_path: str, variant_idx = None):
        """Get cache file path for 1.5K selected features"""
        # Ensure cache directory is set
        if self._cache_dir is None:
            data_dir = Path(image_path).parent.parent
            self._cache_dir = data_dir / ".descriptor_cache"
            self._cache_dir.mkdir(exist_ok=True)
        
        image_name = Path(image_path).stem
        class_name = Path(image_path).parent.name
        
        if variant_idx is not None:
            if isinstance(variant_idx, int):
                cache_filename = f"{class_name}_{image_name}_aug{variant_idx:02d}_adaptive.npy"
            else:
                # Handle string variant_idx (like "batch_30")
                cache_filename = f"{class_name}_{image_name}_{variant_idx}_adaptive.npy"
        else:
            cache_filename = f"{class_name}_{image_name}_base_adaptive.npy"
        
        return self._cache_dir / cache_filename
    
    def _load_cached_descriptors(self, cache_path):
        """Load cached adaptive-size features if they exist"""
        try:
            if cache_path.exists():
                descriptors = np.load(cache_path, allow_pickle=True)
                
                # Handle both single feature vector and batch of feature vectors
                if descriptors.ndim == 1:
                    # Single feature vector (any size now - adaptive)
                    print(f"            CACHE HIT: {len(descriptors)} features")
                    return descriptors
                elif descriptors.ndim == 2:
                    # Batch of feature vectors (any size now - adaptive)
                    print(f"            BATCH CACHE HIT: {descriptors.shape[0]} vectors × {descriptors.shape[1]} features")
                    return descriptors
                else:
                    print(f"            Cache invalid: unexpected shape {descriptors.shape}")
        except Exception as e:
            print(f"   Cache load failed: {str(e)[:50]}")
        return None
    
    def _save_cached_descriptors(self, descriptors, cache_path):
        """Save adaptive-size selected features to cache"""
        try:
            # Ensure cache directory exists
            cache_path.parent.mkdir(exist_ok=True)
            
            # Handle both single feature vector and list of feature vectors
            if isinstance(descriptors, list):
                # List of feature vectors (batch processing) - any size now
                if len(descriptors) > 0:
                    np.save(cache_path, descriptors)
            else:
                # Single feature vector - any size now
                np.save(cache_path, descriptors)
        except Exception as e:
            print(f"       ERROR: Cache save failed: {e}")  # Show errors instead of silent failure
    

    
    def _process_single_image_complete(self, image_path: Path, class_idx: int, image_idx: int, 
                                     augmentations_per_image: int = 30) -> List[Tuple[np.ndarray, int]]:
        """Process single image with augmentations and return QUAD descriptors"""
        
        # Load and validate image
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Resize to standard size for consistency
            image = cv2.resize(image, (512, 512))
            
        except Exception as e:
            print(f"     Error loading {image_path}: {e}")
            return []
        
        # Check cache first for faster training
        variant_count = (augmentations_per_image + 1)  # +1 for original image
        cache_path = self._get_cache_path(image_path, f"batch_{variant_count}")
        
        # Try to load from cache first
        cached_descriptors = self._load_cached_descriptors(cache_path)
        if cached_descriptors is not None:
            print(f"     CACHE HIT: Loaded {len(cached_descriptors)} cached descriptors")
            
            # Convert cached descriptors to grouped features format (single descriptor per image now)
            grouped_features = []
            for descriptor in cached_descriptors:
                # Each descriptor is now a single feature vector - add directly to flat list
                grouped_features.append((descriptor, class_idx))
            return grouped_features
        
        try:
            # Get the recognizer (initialize if needed)
            recognizer = self._get_recognizer()
            
            if recognizer is None:
                print(f"     ERROR: Recognizer is None after initialization")
                return []
            
            # Extract features for original + augmentations (QUAD descriptors)
            feature_vectors = recognizer.process_image_parallel_gpu(
                image, augmentations_per_image=augmentations_per_image
            )
            
            # Process single descriptors (no more A,B,C,D grouping)
            grouped_features = []
            cache_descriptors = []  # For caching raw feature vectors
            
            for feature_vector in feature_vectors:
                # Each variant now produces 1 single descriptor - add directly to flat list
                grouped_features.append((feature_vector, class_idx))
                
                # Store raw feature vector for caching (without class_idx)
                cache_descriptors.append(feature_vector)
            
            # Save to cache for future use
            if cache_descriptors:
                variant_count = len(cache_descriptors)  # Number of single descriptors (no division by 4)
                cache_path = self._get_cache_path(image_path, f"batch_{variant_count}")
                self._save_cached_descriptors(cache_descriptors, cache_path)
                print(f"     Cached {len(cache_descriptors)} descriptors to {cache_path.name}")
            
            return grouped_features
            
        except Exception as e:
            print(f"     Feature extraction failed for {image_path}: {e}")
            return []
    

    
    def process_images_batch(self, image_paths_and_classes: List[Tuple[str, int, str]], 
                           augmentations_per_image: int = 30) -> List[Tuple[np.ndarray, int]]:
        """EXTRACTION ONLY: Process images and create features (no cache checking)"""
        
        print(f"\n  GPU-ACCELERATED EXTRACTION: {len(image_paths_and_classes)} images")
        print(f"     Creating: {augmentations_per_image} augmentations per image")
        print(f"   Pipeline: Load → Live Augmentation → GPU Feature Extraction → Cache")
        
        # Set up cache directory
        if image_paths_and_classes:
            data_dir = Path(image_paths_and_classes[0][0]).parent.parent
            cache_dir = data_dir / ".descriptor_cache"
            cache_dir.mkdir(exist_ok=True)
            self._cache_dir = cache_dir
            print(f"  Cache directory: {cache_dir}")
        
        results = []
        total_images = len(image_paths_and_classes)
        
        # Start progress tracking
        total_augmented_samples = total_images * (augmentations_per_image + 1)  # +1 for base image
        _progress_tracker.start("GPU Feature Extraction", total_augmented_samples)
        
        # Process images sequentially (no cache checking here)
        for img_idx, (image_path, class_idx, class_name) in enumerate(image_paths_and_classes):
            print(f"    Image {img_idx+1}/{total_images}: {Path(image_path).name}")
            
            try:
                # Extract features from this image (optimized processing)
                image_results = self._process_single_image_complete(
                    Path(image_path), class_idx, img_idx, augmentations_per_image
                )
                results.extend(image_results)
                
                print(f"     Extracted {len(image_results)} feature vectors")
                
                # Force GPU memory cleanup between images
                import torch
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                print(f"     GPU memory cleared")
                
            except Exception as e:
                print(f"     Failed: {str(e)[:100]}")
                # Clear GPU memory even on failure
                import torch
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        
        # Finish progress tracking
        _progress_tracker.finish("GPU Feature Extraction Complete")
        
        print(f"\n  GPU EXTRACTION COMPLETE")
        print(f"     Total feature vectors: {len(results):,}")
        print(f"     All features cached for future use")
             
        return results

def load_plant_dataset(data_path: str = "data/plant_images") -> Dict[str, List[str]]:
    """Load plant dataset from directory structure"""
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_path}")
    
    dataset = {}
    
    for class_dir in data_path.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            image_files = []
            
            for image_file in class_dir.iterdir():
                if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image_files.append(str(image_file))
            
            if image_files:
                dataset[class_name] = image_files
                print(f"  Found {len(image_files)} images for class: {class_name}")
    
    print(f"  Dataset loaded: {len(dataset)} classes, {sum(len(files) for files in dataset.values())} total images")
    return dataset

def main_two_stage_training():
    """Main two-stage training function"""
    
    print("TWO-STAGE QUAD DESCRIPTOR PLANT RECOGNITION TRAINING")
    
    print("   Stage 1: Smart feature extraction with plant-specific selection - 3k adaptive features per image")
    print("   Stage 2: Neural network training with single unified descriptor per image")
    
    
    # Load dataset
    try:
        training_data = load_plant_dataset("data/plant_images")
        if not training_data:
            print("  No training data found!")
            return
    except FileNotFoundError as e:
        print(f"  {e}")
        return
    
    # Initialize two-stage trainer with GPU acceleration
    trainer = TwoStageTrainer(use_gpu=True)
    
    # Stage 1: Extract all features with augmentation
    print(f"\n  Starting Stage 1: Feature Extraction")
    feature_dataset = trainer.stage1_extract_all_features(
        training_data, 
        augmentations_per_image=30  # Using sophisticated plant-specific augmentations
    )
    
    if not feature_dataset.features:
        print("  No features extracted! Check your data directory and extraction process.")
        return
    
    print(f"\n  Stage 1 Complete: {len(feature_dataset.features):,} features extracted")
    
    # Stage 2: Train neural network with state-of-the-art techniques
    print(f"\n  Starting Stage 2: Neural Network Training")
    model = trainer.stage2_train_neural_network(
        feature_dataset,
        epochs=60,
        batch_size=32,
        learning_rate=0.001,
        hidden_dim=8
    )
    
    if model is None:
        print("  Training failed! Check your feature extraction and model setup.")
        return
    
    # Model is already saved by blind prediction trainer
    # trainer.save_complete_model(model, feature_dataset, "trained_plant_model.pt")
    
    print(f"\n  TWO-STAGE TRAINING PIPELINE COMPLETE!")
    print(f"     Stage 1: {len(feature_dataset.features):,} features extracted")
    print(f"     Stage 2: Neural network trained with full metrics")
    print(f"     Model saved: trained_plant_model.pt")
    print(f"     Training history available for analysis")
    
    # Safe access to training history
    if hasattr(trainer, 'training_history') and trainer.training_history:
        final_acc = trainer.training_history.get('final_accuracy', 0.0)
        total_time = trainer.training_history.get('total_training_time', 0.0)
        epochs = trainer.training_history.get('epochs_completed', 0)
        method = trainer.training_history.get('training_method', 'unknown')
        
        print(f"     Final accuracy: {final_acc:.1f}%")
        print(f"     Training time: {total_time:.1f}s")
        print(f"     Epochs completed: {epochs}")
        print(f"     Training method: {method}")
    else:
        print(f"      Training history not available")
    
    print(f"     Ready for production use!")

if __name__ == "__main__":
    try:
        main_two_stage_training() 
    except KeyboardInterrupt:
        print("\n  Training interrupted by user")
    except Exception as e:
        print(f"\n  Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc() 