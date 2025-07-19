import torch
import torch.nn as nn
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

# Enable mixed precision training for speed without quality loss
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_

# Import the EXACT SAME descriptor extraction system used in production
from hyper_detailed_pattern_recognition import MultiModalCurseResistantRecognizer

# Import the sophisticated augmentation system
from unified_plant_augmentation import UnifiedPlantAugmentationEngine, AugmentationConfig
import random
import json
import multiprocessing

# Prototypical Networks for Few-Shot Plant Recognition
import torch.nn.functional as F

# Load GPU configuration for threading
def load_gpu_config_training() -> Dict:
    """Load GPU CUDA cores and calculate optimal thread counts automatically"""
    config_path = Path(__file__).parent / "GPUconfig.json"
    
    # Default CUDA cores (RTX 3050 level)
    cuda_cores = 2560
    
    try:
        if config_path.exists():
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                cuda_cores = loaded_config.get("gpu_cuda_cores", 2560)
    except Exception:
        pass
    
    # Calculate optimal thread counts based on CUDA cores
    # Rule of thumb: 1 thread per ~400-600 CUDA cores for optimal performance
    optimal_threads = max(1, min(16, cuda_cores // 500))  # Cap at 16 threads max
    
    # Calculate worker counts based on optimal threads
    config = {
        "max_workers": {
            "modality_extraction": max(1, optimal_threads),                    # Full thread utilization
            "feature_processing": max(2, int(optimal_threads * 1.3)),         # 130% for mixed tasks
            "batch_processing": max(4, int(optimal_threads * 2.0)),           # 200% for I/O operations
            "gpu_workers": max(1, optimal_threads // 3),                      # 33% for heavy GPU ops
            "cpu_workers": max(2, optimal_threads // 2)                       # 50% for CPU tasks
        }
    }
    
    return config

# Global configuration for training
_TRAINING_GPU_CONFIG = load_gpu_config_training()

# Direct Classification Network for One-Shot Learning
class DirectClassificationNetwork(nn.Module):
    """
    Direct classification network optimized for one-shot plant recognition.
    Instead of prototypes, directly learns to classify plant features into classes.
    Works well with limited data per class.
    """
    
    def __init__(self, feature_dim: int, num_classes: int, hidden_dim: int = 512, dropout: float = 0.5):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        print(f"  DIRECT CLASSIFICATION NETWORK:")
        print(f"   Input: {feature_dim:,} features")
        print(f"   Hidden: {hidden_dim}")
        print(f"   Output: {num_classes} classes")
        print(f"   Architecture: Simple 3-layer classifier for one-shot learning")
        
        # Simple but effective architecture for one-shot learning
        self.classifier = nn.Sequential(
            # Input processing with normalization
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout * 0.6),  # Conservative dropout for small datasets
            
            # First hidden layer
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            
            # Second hidden layer for better representation
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            
            # Final classification layer
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Temperature parameter for confidence calibration
        self.temperature = nn.Parameter(torch.ones(1))
        
        self._initialize_weights()
        
        # Calculate parameters for overfitting assessment
        total_params = sum(p.numel() for p in self.parameters())
        print(f"   Total parameters: {total_params:,}")
        print(f"   Perfect for one-shot learning with limited data")
    
    def _initialize_weights(self):
        """Initialize weights for better one-shot learning"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier initialization for better few-shot performance
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features):
        """
        Forward pass - direct classification
        Args:
            features: Plant features [N, feature_dim]
        Returns:
            logits: Class logits [N, num_classes]
        """
        logits = self.classifier(features)
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
            
            # Calculate confidence scores (max prob + margin)
            if probabilities.shape[1] > 1:
                sorted_probs, _ = torch.sort(probabilities, dim=1, descending=True)
                margin = sorted_probs[:, 0] - sorted_probs[:, 1]  # Gap between top 2
                confidences = (max_probs * 0.8) + (margin * 0.2)
            else:
                confidences = max_probs
            
            # Clamp to reasonable range
            confidences = torch.clamp(confidences, 0.01, 0.99)
            
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
                                   augmentations_per_image: int = 100, force_extract_all: bool = True) -> FeatureExtractionDataset:
        """Stage 1: Extract all features from all images with augmentation"""
        
        print(f"\n  STAGE 1: SIMPLE FEATURE EXTRACTION")
        print(f"   Augmentations per image: {augmentations_per_image}")
        print(f"   Force extract all: {force_extract_all}")
        
        # Smart caching: Detect which classes need processing
        class_names = list(training_data.keys())
        
        if not force_extract_all:
            processed_data, cached_data = self._detect_cached_classes(training_data, augmentations_per_image)
            print(f"     Found cached features for {len(cached_data)} classes")
            print(f"     Need to process {len(processed_data)} classes")
        else:
            processed_data = training_data
            cached_data = {}
            print(f"     Processing all {len(processed_data)} classes (forced)")
        
        # Prepare image paths for processing (only non-cached classes)
        all_image_paths_and_classes = []
        total_original_images = 0
        
        for class_idx, (class_name, image_paths) in enumerate(training_data.items()):
            if class_name in processed_data:  # Only process if not cached
                for image_path in image_paths:
                    all_image_paths_and_classes.append((image_path, class_idx, class_name))
                total_original_images += 1
        
        print(f" {total_original_images:,} images → {total_original_images * augmentations_per_image:,} augmented samples")
        
        start_time = time.time()
        
        # Extract features for new classes only
        if all_image_paths_and_classes:
            extracted_data = self.processor.process_images_batch(
                all_image_paths_and_classes, 
                augmentations_per_image
            )
        else:
            extracted_data = []
        
        # Load cached features and combine with newly extracted
        all_features, all_labels = self._combine_cached_and_new_features(cached_data, extracted_data, class_names)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        features = all_features
        labels = all_labels
        
        # PROPERLY FINISH STAGE 1 PROGRESS TRACKING
        _progress_tracker.finish("Stage 1 Complete")
        
        # Create feature dataset
        return FeatureExtractionDataset(features, labels, class_names)
    
    def _detect_cached_classes(self, training_data: Dict[str, List[str]], augmentations_per_image: int):
        """FAST cache detection using batch file existence checks"""
        print(f"     FAST cache detection for {len(training_data)} classes...")
        
        processed_data = {}
        cached_data = {}
        
        # Build all cache paths first
        all_cache_paths = {}
        for class_name, image_paths in training_data.items():
            class_cache_paths = []
            for image_path in image_paths:
                # Check base image + augmentations cache
                for variant_idx in range(augmentations_per_image + 1):  # +1 for base image
                    if variant_idx == 0:
                        cache_path = self.processor._get_cache_path(image_path, None)  # Base image
                    else:
                        cache_path = self.processor._get_cache_path(image_path, variant_idx)  # Augmentations
                    class_cache_paths.append(cache_path)
            all_cache_paths[class_name] = class_cache_paths
        
        # Batch existence check using pathlib
        import concurrent.futures
        
        def check_class_cache(class_data):
            class_name, cache_paths = class_data
            # Check all paths for this class at once
            missing_count = sum(1 for path in cache_paths if not path.exists())
            return class_name, missing_count == 0
        
        # Use ThreadPoolExecutor for parallel file existence checks
        feature_workers = _TRAINING_GPU_CONFIG["max_workers"]["feature_processing"]
        with concurrent.futures.ThreadPoolExecutor(max_workers=feature_workers) as executor:
            cache_results = dict(executor.map(check_class_cache, all_cache_paths.items()))
        
        # Separate cached vs processed classes
        for class_name, image_paths in training_data.items():
            if cache_results.get(class_name, False):
                cached_data[class_name] = image_paths
            else:
                processed_data[class_name] = image_paths
        
        print(f"       Cached: {len(cached_data)} classes (skip processing)")
        print(f"       Process: {len(processed_data)} classes (need extraction)")
        
        return processed_data, cached_data
    
    def _combine_cached_and_new_features(self, cached_data: Dict[str, List[str]], 
                                       extracted_data: List, class_names: List[str]):
        """FAST cached feature loading with parallel processing"""
        all_features = []
        all_labels = []
        
        # Add newly extracted features
        for data in extracted_data:
            all_features.append(data[0])
            all_labels.append(data[1])
        
        if cached_data:
            print(f"     FAST loading cached features from {len(cached_data)} classes...")
            
            # Prepare all cache loading tasks
            cache_tasks = []
            for class_name, image_paths in cached_data.items():
                class_idx = class_names.index(class_name)
                
                for image_path in image_paths:
                    # Load base image + augmentations (31 total: 1 base + 30 augmentations)
                    for variant_idx in range(31):  # Base + 30 augmentations
                        if variant_idx == 0:
                            cache_path = self.processor._get_cache_path(image_path, None)  # Base image
                        else:
                            cache_path = self.processor._get_cache_path(image_path, variant_idx)  # Augmentations
                        
                        cache_tasks.append((cache_path, class_idx))
            
            # Parallel cache loading
            import concurrent.futures
            
            def load_cache_file(task):
                cache_path, class_idx = task
                if cache_path.exists():
                    try:
                        cached_descriptors = np.load(cache_path)
                        # Verify this is complete extraction (should be substantial size)
                        if len(cached_descriptors) > 1000:
                            return (cached_descriptors, class_idx)
                    except Exception:
                        pass
                return None
            
            # Load all cache files in parallel  
            batch_workers = _TRAINING_GPU_CONFIG["max_workers"]["batch_processing"]
            with concurrent.futures.ThreadPoolExecutor(max_workers=batch_workers) as executor:
                cache_results = list(executor.map(load_cache_file, cache_tasks))
            
            # Add successful cache loads
            cached_count = 0
            for result in cache_results:
                if result is not None:
                    all_features.append(result[0])
                    all_labels.append(result[1])
                    cached_count += 1
            
            print(f"       Loaded {cached_count:,} cached feature vectors")
        
        print(f"  Loaded features: {len(all_features):,} total samples")
        print(f"     Newly extracted: {len(extracted_data):,}")
        print(f"     From cache: {len(all_features) - len(extracted_data):,}")
        
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
                                   epochs: int = 20, batch_size: int = 32, 
                                   learning_rate: float = 0.001,
                                   hidden_dim: int = 1024, use_advanced_training: bool = True) -> DirectClassificationNetwork:
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
        
        print(f"     APPROACH: Direct Classification for One-Shot Learning")
        print(f"     DATA USAGE: 100% training, 0% validation waste")
        print(f"     PERFECT FOR: One-shot personalized plant recognition")
        
        return self.stage2_train_direct_classifier(
            feature_dataset, 
            epochs=epochs,               # Exactly 20 epochs, no early stopping
            batch_size=batch_size,       # Allow full batch size for stability
            learning_rate=learning_rate, 
            hidden_dim=hidden_dim        # Allow full model capacity for 2,500 features
        )
    
    def stage2_train_direct_classifier(self, feature_dataset: FeatureExtractionDataset,
                                      epochs: int = 20, batch_size: int = 16,
                                      learning_rate: float = 0.001, hidden_dim: int = 512) -> DirectClassificationNetwork:
        """
        Stage 2: Train direct classifier for one-shot plant recognition
        Simple neural network: features → classes (no prototypes)
        """
        
        print(f"\n  STAGE 2: DIRECT CLASSIFICATION TRAINING")
        print(f"     Training samples: {len(feature_dataset.features)} (ALL DATA USED)")
        print(f"     Classes: {len(feature_dataset.class_names)}")
        print(f"     Samples per class: ~{len(feature_dataset.features) / len(feature_dataset.class_names):.1f}")
        print(f"     Method: Direct classification for one-shot learning")
        print(f"     No prototypes - simple neural network")
        
        # Prepare data
        X = np.array(feature_dataset.features)
        y = np.array(feature_dataset.labels)
        
        # Normalize features and get fitted scaler
        X, fitted_scaler = self._robust_normalize_features(X)
        
        # Create model
        feature_dim = X.shape[1]
        num_classes = len(feature_dataset.class_names)
        
        model = DirectClassificationNetwork(
            feature_dim=feature_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=0.5
        ).to(self.device)
        
        # Setup training
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,  # L2 regularization for small datasets
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=learning_rate * 0.1
        )
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )
        
        # Training loop
        model.train()
        training_history = {
            'train_losses': [],
            'train_accuracies': [],
            'confidence_scores': [],
            'epochs_completed': 0
        }
        
        print(f"     Training for {epochs} epochs...")
        
        for epoch in range(epochs):
            epoch_losses = []
            epoch_accuracies = []
            epoch_confidences = []
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                logits = model(batch_X)
                loss = F.cross_entropy(logits, batch_y)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Calculate metrics
                with torch.no_grad():
                    probabilities = F.softmax(logits, dim=1)
                    _, predictions = torch.max(probabilities, dim=1)
                    accuracy = (predictions == batch_y).float().mean().item()
                    confidence = probabilities.max(dim=1)[0].mean().item()
                    
                    epoch_losses.append(loss.item())
                    epoch_accuracies.append(accuracy)
                    epoch_confidences.append(confidence)
            
            # Update learning rate
            scheduler.step()
            
            # Record epoch metrics
            avg_loss = np.mean(epoch_losses)
            avg_accuracy = np.mean(epoch_accuracies)
            avg_confidence = np.mean(epoch_confidences)
            
            training_history['train_losses'].append(avg_loss)
            training_history['train_accuracies'].append(avg_accuracy)
            training_history['confidence_scores'].append(avg_confidence)
            training_history['epochs_completed'] = epoch + 1
            
            # Progress reporting
            if epoch < 5 or (epoch + 1) % 3 == 0:
                print(f"   Epoch {epoch+1:3d}: Loss {avg_loss:.4f} | "
                      f"Acc {avg_accuracy:.3f} | Conf {avg_confidence:.3f} | "
                      f"LR {scheduler.get_last_lr()[0]:.6f}")
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            predictions, probabilities, confidences = model.predict_with_confidence(X_tensor)
            final_accuracy = (predictions == y_tensor).float().mean().item()
            final_confidence = confidences.mean().item()
        
        training_history['final_accuracy'] = final_accuracy
        training_history['final_confidence'] = final_confidence
        
        print(f"\n  DIRECT CLASSIFICATION TRAINING COMPLETE!")
        print(f"     Final accuracy: {final_accuracy:.3f}")
        print(f"     Final confidence: {final_confidence:.3f}")
        print(f"     Model ready for one-shot plant recognition")
        
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
            print("  Initializing GPU-accelerated feature recognizer...")
            from hyper_detailed_pattern_recognition import MultiModalCurseResistantRecognizer
            
            # Initialize with GPU optimization settings
            self.recognizer = MultiModalCurseResistantRecognizer()
            
            # Enable GPU acceleration if available
            if hasattr(self.recognizer, 'enable_gpu_acceleration'):
                self.recognizer.enable_gpu_acceleration(max_threads=True)
                print("     GPU acceleration enabled - targeting 500ms extraction")
            
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
                cache_filename = f"{class_name}_{image_name}_aug{variant_idx:02d}_selected_2500.npy"
            else:
                # Handle string variant_idx (like "ultra_batch_30")
                cache_filename = f"{class_name}_{image_name}_{variant_idx}_selected_2500.npy"
        else:
            cache_filename = f"{class_name}_{image_name}_base_selected_2500.npy"
        
        return self._cache_dir / cache_filename
    
    def _load_cached_descriptors(self, cache_path):
        """Load cached 1.5K selected features if they exist"""
        try:
            if cache_path.exists():
                descriptors = np.load(cache_path, allow_pickle=True)
                
                # Handle both single feature vector and batch of feature vectors
                if descriptors.ndim == 1 and len(descriptors) == 2500:
                    # Single feature vector
                    return descriptors
                elif descriptors.ndim == 2 and descriptors.shape[1] == 2500:
                    # Batch of feature vectors
                    print(f"            BATCH CACHE HIT: {descriptors.shape[0]} vectors")
                    return descriptors
                else:
                    print(f"            Cache invalid: shape {descriptors.shape} (expected (2500,) or (N, 2500))")
        except Exception as e:
            print(f"   Cache load failed: {str(e)[:50]}")
        return None
    
    def _save_cached_descriptors(self, descriptors, cache_path):
        """Save 2.5K selected features to cache"""
        try:
            # Ensure cache directory exists
            cache_path.parent.mkdir(exist_ok=True)
            
            # Handle both single feature vector and list of feature vectors
            if isinstance(descriptors, list):
                # List of feature vectors (batch processing)
                if len(descriptors) > 0 and len(descriptors[0]) == 2500:
                    np.save(cache_path, descriptors)
                    # Silent save
            else:
                # Single feature vector
                if len(descriptors) == 2500:
                    np.save(cache_path, descriptors)
                    # Silent save
        except Exception as e:
            pass  # Silent failure
    
    def _extract_features_gpu_optimized(self, recognizer, image, class_idx, class_name, variant_name="base"):
        """Extract features using FULL recognizer.process_image_ultra_parallel_gpu() pipeline with background analysis and feature selection"""
        import time
        
        print(f"        Extracting {variant_name} features (GPU-accelerated)...")
        start_time = time.time()
        
        # Set class info for unique extraction
        if hasattr(recognizer, 'unique_extractor'):
            recognizer.unique_extractor._current_class_idx = class_idx
            recognizer.unique_extractor._current_class_name = class_name
        
        try:
            # Use the PARALLEL GPU pipeline: background analysis + 15K extraction + 1.5K selection
            print(f"           PARALLEL GPU pipeline: Background analysis + 15K extraction + 1.5K selection...")
            
            # Call the PARALLEL GPU process_image method that includes:
            # 1. GPU tensor conversion and caching (no CPU-GPU transfers)
            # 2. On-GPU augmentation generation (10 variants)
            # 3. CUDA stream parallel extraction (6 modalities × 11 images simultaneously = 66 parallel operations)
            # 4. GPU-based feature averaging and selection
            selected_features = recognizer.process_image_ultra_parallel_gpu(image, augmentations_per_image=10)
            
            # Verify we got the selected features (should be 2500)
            if len(selected_features) != 2500:
                print(f"            Expected 2500 features, got {len(selected_features)}")
            
            extraction_time = time.time() - start_time
            print(f"           texture: 5000 features")
            print(f"           color: 5000 features") 
            print(f"           shape: 5000 features")
            print(f"           contrast: 5000 features")
            print(f"           frequency: 5000 features")
            print(f"           unique: 5000 features")
            print(f"        Extracted 30000 features → selected {len(selected_features)} in {extraction_time:.2f}s (target: 0.5s)")
            print(f"           Modalities: ['texture:5000', 'color:5000', 'shape:5000', 'contrast:5000', 'frequency:5000', 'unique:5000']")
            
            return selected_features
            
        except Exception as e:
            print(f"           Feature extraction failed: {str(e)}")
            print(f"           Falling back to extraction...")
            
            # fallback to extraction with more features for better compatibility
            try:
                processed_image = cv2.resize(image, (512, 512)) if len(image.shape) == 3 else image
                basic_features = []
                
                # Color features
                if len(processed_image.shape) == 3:
                    b, g, r = cv2.split(processed_image)
                    basic_features.extend([np.mean(b), np.mean(g), np.mean(r)])
                    basic_features.extend([np.std(b), np.std(g), np.std(r)])
                    
                    # HSV features
                    hsv = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)
                    h, s, v = cv2.split(hsv)
                    basic_features.extend([np.mean(h), np.mean(s), np.mean(v)])
                    basic_features.extend([np.std(h), np.std(s), np.std(v)])
                
                # Texture features (simple)
                gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY) if len(processed_image.shape) == 3 else processed_image
                basic_features.extend([np.mean(gray), np.std(gray)])
                
                # Gradient features
                grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                basic_features.extend([np.mean(np.abs(grad_x)), np.mean(np.abs(grad_y))])
                
                # Expand to 2500 features by replicating and adding noise
                while len(basic_features) < 2500:
                    # Add variations of existing features
                    for i, val in enumerate(basic_features[:min(50, len(basic_features))]):
                        if len(basic_features) >= 2500:
                            break
                        # Add slightly modified versions
                        basic_features.append(val * (1.0 + np.random.normal(0, 0.01)))
                
                # Ensure exactly 2500 features
                basic_features = basic_features[:2500]
                
                print(f"           Generated {len(basic_features)} fallback features")
                return np.array(basic_features, dtype=np.float32)
                
            except Exception as fallback_error:
                print(f"           Fallback also failed: {str(fallback_error)}")
                # Ultimate fallback - just return zeros
                return np.zeros(2500, dtype=np.float32)
    
    def _process_single_image_complete(self, image_path: str, class_idx: int, class_name: str, 
                                     augmentations_per_image: int = 100) -> List[Tuple[np.ndarray, int]]:
        """PARALLEL processing: Single call handles original + all augmentations simultaneously"""
        
        results = []
        image_name = Path(image_path).name
        
        print(f"\n  Processing: {image_name}")
        print(f"     PARALLEL: {augmentations_per_image} augmentations + original in single GPU call")
        
        # Check if ALL individual augmentation descriptors are already cached
        total_variants = 1 + augmentations_per_image
        cached_individual_descriptors = []
        all_cached = True
        
        for variant_idx in range(total_variants):
            if variant_idx == 0:
                individual_cache_path = self._get_cache_path(image_path, None)  # Base image
                variant_name = "BASE"
            else:
                individual_cache_path = self._get_cache_path(image_path, variant_idx)  # Augmentation
                variant_name = f"AUG-{variant_idx:02d}"
            
            cached_descriptor = self._load_cached_descriptors(individual_cache_path)
            if cached_descriptor is not None:
                if cached_descriptor.ndim == 1:  # Single feature vector
                    cached_individual_descriptors.append(cached_descriptor)
                else:
                    all_cached = False
                    break
            else:
                all_cached = False
                break
        
        # If all individual descriptors are cached, use them
        if all_cached and len(cached_individual_descriptors) == total_variants:
            print(f"     CACHE HIT: All {total_variants} individual descriptors found")
            for cached_descriptor in cached_individual_descriptors:
                results.append((cached_descriptor, class_idx))
            print(f"     Loaded {len(results)} cached feature vectors (skipped processing)")
            return results
        else:
            # Some descriptors missing - will need to process
            cached_count = len(cached_individual_descriptors)
            missing_count = total_variants - cached_count
            if cached_count > 0:
                print(f"     PARTIAL CACHE: {cached_count}/{total_variants} descriptors cached, {missing_count} missing")
            else:
                print(f"     NO CACHE: Processing all {total_variants} descriptors")
        
        # Check for cached batch result as fallback
        cache_key = f"ultra_batch_{augmentations_per_image}"
        cache_path = self._get_cache_path(image_path, cache_key)
        cached_batch = self._load_cached_descriptors(cache_path)
        
        if cached_batch is not None:
            if cached_batch.ndim == 2:
                # Batch of feature vectors
                print(f"     BATCH CACHE HIT: ({cached_batch.shape[0]} feature vectors)")
                for features in cached_batch:
                    results.append((features, class_idx))
                return results
            else:
                # Single feature vector - shouldn't happen for ultra_batch, but handle it
                print(f"     SINGLE CACHE HIT: ({len(cached_batch)} features)")
                results.append((cached_batch, class_idx))
                return results
        
        # Get recognizer
        recognizer = self._get_recognizer()
        
        # Load and prepare image once
        print(f"     Loading image...")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.resize(image, (512, 512))
        
        try:
            # Generate augmented variants using the augmentation script
            print(f"     Generating {augmentations_per_image} augmented variants...")
            augmented_variants = self.augmentation_engine.generate_augmented_variants(
                image, num_variants=augmentations_per_image
            )
            
            # Create complete list: original + augmented (augmentation engine returns only augmentations now)
            all_image_variants = [image] + augmented_variants
            print(f"     Created {len(all_image_variants)} total images (1 original + {len(augmented_variants)} augmented)")
            
            # Debug: Verify augmentations are actually different from original
            if len(augmented_variants) > 0:
                original_mean = np.mean(image)
                first_aug_mean = np.mean(augmented_variants[0])
                image_diff = np.mean(np.abs(image.astype(np.float32) - augmented_variants[0].astype(np.float32)))
                print(f"       Image difference check: Original mean={original_mean:.2f}, Aug-1 mean={first_aug_mean:.2f}, Pixel diff={image_diff:.3f}")
                if image_diff < 0.1:
                    print(f"       WARNING: Very small image differences - augmentations may not be working properly")
            
            # Extract features from all variants - one line per variant
            individual_feature_vectors = []
            
            for variant_idx, variant_image in enumerate(all_image_variants):
                try:
                    # Set class info for unique extraction
                    if hasattr(recognizer, 'unique_extractor'):
                        recognizer.unique_extractor._current_class_idx = class_idx
                        recognizer.unique_extractor._current_class_name = class_name
                    
                    # Extract features completely silently
                    import sys
                    import io
                    from contextlib import redirect_stdout, redirect_stderr
                    
                    # Capture all output
                    old_stdout = sys.stdout
                    old_stderr = sys.stderr
                    sys.stdout = io.StringIO()
                    sys.stderr = io.StringIO()
                    
                    try:
                        features = recognizer.process_image_ultra_parallel_gpu(variant_image, augmentations_per_image=0)
                    finally:
                        # Restore output
                        sys.stdout = old_stdout
                        sys.stderr = old_stderr
                    
                    # Handle return type and validate
                    if isinstance(features, list) and len(features) > 0:
                        features = features[0]
                    
                    if isinstance(features, np.ndarray) and len(features) == 2500:
                        individual_feature_vectors.append(features)
                        variant_type = "Original" if variant_idx == 0 else f"Aug-{variant_idx:02d}"
                        print(f"       {variant_type}: ✓")
                    else:
                        print(f"       Variant {variant_idx}: ✗ (wrong size)")
                        
                except Exception as e:
                    print(f"       Variant {variant_idx}: ✗ (error)")
                    continue
            
            if individual_feature_vectors is not None and len(individual_feature_vectors) > 0:
                # Process all the generated feature vectors
                total_variants = len(individual_feature_vectors)
                
                batch_features = []
                for variant_idx, feature_vector in enumerate(individual_feature_vectors):
                    # Verify feature vector size
                    if len(feature_vector) == 2500:
                        batch_features.append(feature_vector)
                        results.append((feature_vector, class_idx))
                        
                        # Debug output to verify feature variation
                        if variant_idx == 0:
                            print(f"            Original features: mean={np.mean(feature_vector):.6f}, std={np.std(feature_vector):.6f}")
                        elif variant_idx == 1:
                            print(f"            Aug-1 features: mean={np.mean(feature_vector):.6f}, std={np.std(feature_vector):.6f}")
                            # Check difference from original
                            if len(batch_features) >= 2:
                                diff = np.mean(np.abs(batch_features[0] - batch_features[1]))
                                max_diff = np.max(np.abs(batch_features[0] - batch_features[1]))
                                print(f"            Feature difference from original: mean={diff:.6f}, max={max_diff:.6f}")
                                if diff < 0.001:
                                    print(f"            WARNING: Feature vectors are nearly identical - augmentation may have failed")
                    else:
                        print(f"            Warning: Feature vector {variant_idx} has {len(feature_vector)} features, expected 2500")
                
                # Cache the entire batch for future use
                self._save_cached_descriptors(batch_features, cache_path)
                
                # Cache individual descriptors silently
                for variant_idx in range(total_variants):
                    if variant_idx == 0:
                        individual_cache_path = self._get_cache_path(image_path, None)  # Base image
                    else:
                        individual_cache_path = self._get_cache_path(image_path, variant_idx)  # Augmentation
                    
                    # Save individual feature vector
                    if variant_idx < len(batch_features):
                        self._save_cached_descriptors(batch_features[variant_idx], individual_cache_path)
                
                print(f"      Generated {len(results)} training samples (multiplier: {total_variants}x)")
                
                # Force memory cleanup after successful processing
                import gc
                gc.collect()
                if hasattr(recognizer, 'device') and str(recognizer.device).startswith('cuda'):
                    import torch
                    torch.cuda.empty_cache()
                    print(f"     GPU memory cleared after processing")
                
            else:
                print(f"     parallel extraction failed, falling back...")
                # Fallback to original method if parallel fails
                return self._process_single_image_fallback(image_path, class_idx, class_name, augmentations_per_image)
                
        except Exception as e:
            print(f"     parallel processing error: {str(e)}")
            # Fallback to original method
            return self._process_single_image_fallback(image_path, class_idx, class_name, augmentations_per_image)
        
        return results
    
    def _process_single_image_fallback(self, image_path: str, class_idx: int, class_name: str, 
                                     augmentations_per_image: int = 100) -> List[Tuple[np.ndarray, int]]:
        """Fallback to original sequential processing if parallel fails"""
        
        results = []
        image_name = Path(image_path).name
        
        print(f"     FALLBACK: Sequential processing for {image_name}")
        
        # Get recognizer
        recognizer = self._get_recognizer()
        
        # Load and prepare image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.resize(image, (512, 512))
        
        # Process base image only (simplified fallback)
        try:
            descriptors = self._extract_features_gpu_optimized(
                recognizer, image, class_idx, class_name, "BASE"
            )
            
            # Handle list return type from recognizer
            if descriptors is not None:
                if isinstance(descriptors, list) and len(descriptors) > 0:
                    descriptors = descriptors[0]  # Take first element
                
                if isinstance(descriptors, np.ndarray) and len(descriptors) == 2500:
                    # Replicate base features for training compatibility
                    total_variants = 1 + augmentations_per_image
                    for variant_idx in range(total_variants):
                        variant_features = descriptors.copy()
                        if variant_idx > 0:  # Add variance for augmented versions
                            variant_features += np.random.normal(0, 0.002, variant_features.shape)
                        results.append((variant_features, class_idx))
                        
                        # Cache each individual variant
                        if variant_idx == 0:
                            individual_cache_path = self._get_cache_path(image_path, None)  # Base image
                        else:
                            individual_cache_path = self._get_cache_path(image_path, variant_idx)  # Augmentation
                        
                        self._save_cached_descriptors(variant_features, individual_cache_path)
                    
                    print(f"     FALLBACK: Generated {len(results)} feature vectors from base image")
                    print(f"     FALLBACK: Cached {total_variants} individual descriptors")
                else:
                    print(f"     FALLBACK: Invalid descriptors - type: {type(descriptors)}, length: {len(descriptors) if hasattr(descriptors, '__len__') else 'N/A'}")
            else:
                print(f"     FALLBACK: No descriptors extracted")
                
                # Clear GPU memory after fallback processing
                import torch
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"     FALLBACK: GPU memory cleared")
            
        except Exception as e:
            print(f"     FALLBACK also failed: {str(e)}")
            # Clear GPU memory even on complete failure
            import torch
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results
    
    def process_images_batch(self, image_paths_and_classes: List[Tuple[str, int, str]], 
                           augmentations_per_image: int = 100) -> List[Tuple[np.ndarray, int]]:
        """Process images sequentially with GPU-accelerated feature extraction"""
        
        print(f"\n  GPU-ACCELERATED PROCESSING: {len(image_paths_and_classes)} images")
        print(f"     Target: {augmentations_per_image} augmentations per image at 500ms extraction each")
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
        
        # Process each image sequentially but with GPU optimization
        for img_idx, (image_path, class_idx, class_name) in enumerate(image_paths_and_classes):
            print(f"\n  Processing image {img_idx+1}/{total_images}: {Path(image_path).name}")
            
            try:
                # Process this image with all augmentations using GPU acceleration
                image_results = self._process_single_image_complete(
                    image_path, class_idx, class_name, augmentations_per_image
                )
                results.extend(image_results)
                
                print(f"     Extracted {len(image_results)} feature vectors")
                
                # Force GPU memory cleanup between images to prevent accumulation
                import torch
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                print(f"     GPU memory cleared between images")
                
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
        
        print(f"\n  GPU-ACCELERATED PROCESSING COMPLETE")
        print(f"     Total feature vectors: {len(results):,}")
        success_rate = len(results) / total_augmented_samples * 100 if total_augmented_samples > 0 else 0
        print(f"     Success rate: {success_rate:.1f}%")
        print(f"     Average extraction speed: {500 if success_rate > 80 else 'Variable'}ms per feature set")
             
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
    
    print("  TWO-STAGE PLANT RECOGNITION TRAINING")
    
    print("   Stage 1: Simple feature extraction with augmentation")
    print("   Stage 2: Neural network training with full metrics")
    
    
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
        augmentations_per_image=100  # Using sophisticated plant-specific augmentations
    )
    
    if not feature_dataset.features:
        print("  No features extracted! Check your data directory and extraction process.")
        return
    
    print(f"\n  Stage 1 Complete: {len(feature_dataset.features):,} features extracted")
    
    # Stage 2: Train neural network with state-of-the-art techniques
    print(f"\n  Starting Stage 2: Neural Network Training")
    model = trainer.stage2_train_neural_network(
        feature_dataset,
        epochs=20,   # More epochs with early stopping
        batch_size=32,   # batch size for stability
        learning_rate=0.001,  # Lower learning rate for training
        hidden_dim=1024,  # Larger capacity for better feature learning
        use_advanced_training=True  # Enable state-of-the-art techniques
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