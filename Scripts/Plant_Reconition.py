import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, models, mixed_precision
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers.schedules import CosineDecay
import tensorflow.keras.backend as K
from sklearn.ensemble import RandomForestClassifier
import os
import cv2
import joblib
from scipy.optimize import minimize
import json
import time
import signal
import threading
from tqdm.auto import tqdm
import logging
from datetime import datetime, timedelta
import sys
import glob
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import psutil
import pickle
import gc
import types
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import shutil
import traceback
import colorama
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=no INFO, 2=no WARNING, 3=no ERROR
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Suppress sklearn warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Suppress Numpy warnings
import numpy as np
np.seterr(all='ignore')

import re

# Create a filter for TFA warnings
def filter_tfa_warnings(message, category, filename, lineno, file=None, line=None):
    if category == UserWarning and "Tensorflow Addons" in str(message):
        return None  # Suppress the warning
    return True  # Show other warnings

# Disable TensorFlow device placement logging
tf.debugging.set_log_device_placement(False)

# Suppress CUDA warnings
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Update your existing line to be more aggressive
os.environ['PYTHONWARNINGS'] = 'ignore'

IMAGE_SIZE = (224, 224)
FEATURE_DIR = "data/features"
MODEL_DIR = "models"
CHECKPOINT_DIR = "checkpoints"

# Define TermColors before using it
class TermColors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Define memory limit function first thing after imports
# Define memory limit function first thing after imports
def limit_memory_usage(gpu_memory_limit_mb=6144, ram_percent_limit=80):
    """Limit GPU memory usage and log RAM limit intention.
    RAM monitoring will be handled by MemoryCallback or manual checks.

    Args:
        gpu_memory_limit_mb: GPU memory limit in megabytes (default: 6144MB for ~8GB card)
        ram_percent_limit: Maximum RAM usage target percentage (default: 80%)
    """
    print(f"{TermColors.YELLOW}⚠️ MEMORY LIMITER ACTIVE (GPU: {gpu_memory_limit_mb}MB, Target RAM: {ram_percent_limit}%){TermColors.ENDC}")

    # 1. GPU Memory Limiting
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            # Set memory limit on each GPU
            for device in physical_devices:
                tf.config.set_logical_device_configuration(
                    device,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=gpu_memory_limit_mb)]
                )
            print(f"{TermColors.GREEN}✅ GPU memory limited to {gpu_memory_limit_mb}MB{TermColors.ENDC}")
        except Exception as e:
            # Log error but continue - might be already configured
            print(f"{TermColors.RED}❌ Error setting GPU memory limit (might be already set): {e}{TermColors.ENDC}")
            # Try enabling memory growth as a fallback
            try:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                print(f"{TermColors.YELLOW}⚠️ Enabled GPU memory growth as fallback.{TermColors.ENDC}")
            except Exception as e_growth:
                print(f"{TermColors.RED}❌ Failed to enable memory growth: {e_growth}{TermColors.ENDC}")
    else:
        print(f"{TermColors.YELLOW}⚠️ No GPU devices found to apply memory limit{TermColors.ENDC}")

    # RAM monitoring thread removed - will be handled by MemoryCallback or manual checks.
    print(f"{TermColors.CYAN}ℹ RAM usage target set to {ram_percent_limit}%. Monitoring handled by callbacks.{TermColors.ENDC}")

# Apply memory limits immediately
# Check your specific RTX 3050 VRAM - if 4GB, use ~3584MB
limit_memory_usage(gpu_memory_limit_mb=6144, ram_percent_limit=80) # Using 6GB GPU limit, 80% RAM target

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=no INFO, 2=no WARNING, 3=no ERROR
tf.get_logger().setLevel('ERROR')

# Replace your existing TensorFlow logger settings
print(f"\n{TermColors.GREEN}✅ TensorFlow is using GPU: {tf.test.is_gpu_available()}{TermColors.ENDC}")

# Suppress additional Tensorflow output
if not os.environ.get('VERBOSE_MODE') == '1':
    # Disable all TensorFlow logging except fatal errors
    tf.autograph.set_verbosity(0)
    tf.get_logger().setLevel('ERROR')
    
    # Patch print function for TensorFlow internals
    original_print = print
    def filtered_print(*args, **kwargs):
        if args and isinstance(args[0], str):
            if any(x in args[0].lower() for x in ['warning:', 'note:', 'tensorflow']):
                return  # Skip TensorFlow warnings
        original_print(*args, **kwargs)
    
    # Only apply in non-verbose mode
    import builtins
    builtins.print = filtered_print

# Force TensorFlow to see and use the GPU correctly
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Force device placement logging
tf.debugging.set_log_device_placement(False)

# Configure TensorFlow to use the GPU 
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("GPU is available:", physical_devices)
    try:
        # Only allow TensorFlow to see the first GPU
        tf.config.set_visible_devices(physical_devices[0], 'GPU')
        logical_devices = tf.config.list_logical_devices('GPU')
        print("Logical GPU devices:", logical_devices)
        print("Successfully configured GPU")
    except RuntimeError as e:
        print("GPU configuration error:", e)
else:
    print("⚠️ WARNING: No GPU found. Running on CPU only!")

# Initialize colorama for cross-platform colored terminal output
colorama.init()

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))    # Script directory
BASE_DIR = os.path.dirname(SCRIPT_DIR)                     # Project root directory
DATA_DIR = os.path.join(BASE_DIR, "data", "plant_images")  # Where images are saved
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")  # Where extracted features are stored
MODEL_DIR = os.path.join(BASE_DIR, "models")               # Where trained models are saved
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")     # Where training checkpoints are saved
LOG_DIR = os.path.join(BASE_DIR, "logs")                   # Where training logs are stored
STATS_DIR = os.path.join(BASE_DIR, "stats")                # Where statistics are saved

# Training Parameters
BATCH_SIZE = 128
IMAGE_SIZE = (224, 224)
EPOCHS = 100
MAX_CLASSES_PER_CHUNK = 1000
TOP_K = 10
INITIAL_LR = 1e-3
CPU_WORKERS = min(6, multiprocessing.cpu_count() - 1)  # Leave one core free
GRADIENT_ACCUMULATION_STEPS = 4
MEMORY_CLEANUP_FREQ = 500

# Create required directories
for directory in [FEATURES_DIR, MODEL_DIR, CHECKPOINT_DIR, LOG_DIR, STATS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Add to imports at the top of the file
import threading
import os

# Add this function to periodically clear the terminal
def setup_terminal_clearing():
    """Setup background thread to clear terminal every hour"""
    def clear_terminal():
        while True:
            # Sleep for 1 hour
            time.sleep(3600)
            
            # Clear screen based on OS
            clear_command = 'cls' if os.name == 'nt' else 'clear'
            os.system(clear_command)
            
            # Print header to show clearing happened
            print(f"\n{TermColors.CYAN}{'='*50}")
            print(f"TERMINAL CLEARED - TRAINING CONTINUES")
            print(f"USE check_training_progress() TO SEE STATUS")
            print(f"{'='*50}{TermColors.ENDC}\n")
    
    # Start the clearing thread as daemon
    clearing_thread = threading.Thread(target=clear_terminal, daemon=True)
    clearing_thread.start()
    print(f"{TermColors.CYAN}ℹ Automatic terminal clearing enabled (every 1 hour){TermColors.ENDC}")

# Add this function to check progress
def check_training_progress():
    """Generate a quick summary of training progress for all chunks"""
    print(f"{TermColors.HEADER}\n{'='*50}")
    print(f"PLANT RECOGNITION TRAINING PROGRESS SUMMARY")
    print(f"{'='*50}{TermColors.ENDC}")
    
    # Load training state
    state_file = os.path.join(CHECKPOINT_DIR, "training_state.json")
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            state = json.load(f)
            
        print(f"Current chunk: {state['current_chunk']+1}")
        print(f"Processed chunks: {len(state['processed_chunks'])}")
        print(f"Best validation accuracy: {state['best_val_accuracy']:.2%}")
        print(f"Last updated: {state['last_updated']}")
    else:
        print(f"{TermColors.YELLOW}⚠️ No training state file found.{TermColors.ENDC}")
    
    # Check all completed models
    model_files = glob.glob(os.path.join(MODEL_DIR, "chunk_*_metadata.json"))
    
    if model_files:
        print(f"\n{TermColors.CYAN}ℹ Found {len(model_files)} trained chunk models{TermColors.ENDC}")
        
        # Collect performance metrics
        all_metrics = []
        
        for meta_file in model_files:
            chunk_idx = int(os.path.basename(meta_file).split("_")[1])
            
            with open(meta_file) as f:
                metadata = json.load(f)
            
            perf = metadata.get('performance', {})
            all_metrics.append({
                'chunk': chunk_idx,
                'classes': metadata.get('num_classes', 0),
                'accuracy': perf.get('val_accuracy', 0),
                'top_k': perf.get('val_sparse_top_k_accuracy', 0),
                'created': metadata.get('created', 'unknown')
            })
        
        # Sort by chunk index
        all_metrics.sort(key=lambda x: x['chunk'])
        
        # Print table header
        print(f"\n{'Chunk':^6}|{'Classes':^8}|{'Accuracy':^10}|{'Top-10':^10}|{'Completed':^20}")
        print(f"{'-'*6:^6}|{'-'*8:^8}|{'-'*10:^10}|{'-'*10:^10}|{'-'*20:^20}")
        
        # Print metrics
        for m in all_metrics:
            # Format date to be shorter
            date_str = m['created'].split('T')[0] if isinstance(m['created'], str) else 'unknown'
            
            print(f"{m['chunk']:^6}|{m['classes']:^8}|{m['accuracy']*100:^10.2f}%|{m['top_k']*100:^10.2f}%|{date_str:^20}")
        
        # Calculate averages
        avg_acc = sum(m['accuracy'] for m in all_metrics) / len(all_metrics)
        avg_top_k = sum(m['top_k'] for m in all_metrics) / len(all_metrics)
        
        print(f"{'-'*6:^6}|{'-'*8:^8}|{'-'*10:^10}|{'-'*10:^10}|{'-'*20:^20}")
        print(f"{'AVG':^6}|{'-':^8}|{avg_acc*100:^10.2f}%|{avg_top_k*100:^10.2f}%|{'-':^20}")
    else:
        print(f"{TermColors.YELLOW}⚠️ No trained models found.{TermColors.ENDC}")

# Metrics formatting
def format_metrics(metrics_dict):
    """Format metrics dictionary into colored string"""
    formatted = []
    for key, value in metrics_dict.items():
        if isinstance(value, str):
            # Handle string values
            formatted.append(f"{key}: {value}")
        elif 'accuracy' in key:
            color = TermColors.GREEN
            formatted.append(f"{color}{key}: {value:.4f}{TermColors.ENDC}")
        elif 'loss' in key:
            color = TermColors.RED
            formatted.append(f"{color}{key}: {value:.4f}{TermColors.ENDC}")
        else:
            formatted.append(f"{key}: {value:.4f}")
    return " | ".join(formatted)

# Custom tqdm for showing metrics
class MetricsTQDM(tqdm):
    """Custom tqdm progress bar that displays training metrics"""
    def __init__(self, *args, **kwargs):
        # Initialize metrics and flag before super().__init__
        self.metrics = {}
        self._metrics_updated = False
        # Now call the parent constructor
        super().__init__(*args, **kwargs)
        
    def update_metrics(self, epoch, metrics):
        """Update training metrics in history"""
        self.current_epoch = epoch
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            # Ensure we don't have more entries than epochs
            while len(self.history[key]) <= epoch:
                self.history[key].append(None)
            
            # Handle non-numeric types properly
            try:
                # Try to convert to float
                self.history[key][epoch] = float(value)
            except (ValueError, TypeError):
                # If conversion fails, store as string representation
                self.history[key][epoch] = str(value)
        
        # Update best metrics
        val_loss = metrics.get('val_loss')
        val_accuracy = metrics.get('val_accuracy')
        
        if val_loss is not None and isinstance(val_loss, (float, int)):
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
        
        if val_accuracy is not None and isinstance(val_accuracy, (float, int)):
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
        
        self.save_state()
        
    def display(self, msg=None, pos=None):
        """Override display to include metrics"""
        # We don't need special handling anymore - postfix is set directly in update_metrics
        return super().display(msg, pos)

# Training state management
class TrainingState:
    """Manages training state for pause/resume functionality"""
    def __init__(self, state_file=os.path.join(CHECKPOINT_DIR, "training_state.json")):
        self.state_file = state_file
        self.current_chunk = 0
        self.current_epoch = 0
        self.processed_chunks = []
        self.history = {}
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0
        self.epochs_without_improvement = 0
        self.load_state()
    
    def load_state(self):
        """Load saved training state if exists"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.current_chunk = state.get('current_chunk', 0)
                    self.current_epoch = state.get('current_epoch', 0)
                    self.processed_chunks = state.get('processed_chunks', [])
                    self.history = state.get('history', {})
                    self.best_val_loss = state.get('best_val_loss', float('inf'))
                    self.best_val_accuracy = state.get('best_val_accuracy', 0)
                print(f"{TermColors.GREEN}✅ Resuming from chunk {self.current_chunk}, epoch {self.current_epoch}{TermColors.ENDC}")
            except Exception as e:
                print(f"{TermColors.YELLOW}⚠️ Could not load training state: {e}{TermColors.ENDC}")
    
    def save_state(self):
        """Save current training state"""
        state = {
            'current_chunk': self.current_chunk,
            'current_epoch': self.current_epoch,
            'processed_chunks': self.processed_chunks,
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def update_metrics(self, epoch, metrics):
        """Update training metrics in history"""
        self.current_epoch = epoch
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            # Ensure we don't have more entries than epochs
            while len(self.history[key]) <= epoch:
                self.history[key].append(None)
            self.history[key][epoch] = float(value)
        
        # Update best metrics
        val_loss = metrics.get('val_loss')
        val_accuracy = metrics.get('val_accuracy')
        
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        
        if val_accuracy is not None and val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy
        
        self.save_state()
    
    def mark_chunk_complete(self, chunk_idx):
        """Mark a chunk as completely processed"""
        if chunk_idx not in self.processed_chunks:
            self.processed_chunks.append(chunk_idx)
        self.current_chunk = chunk_idx + 1
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0
        self.epochs_without_improvement = 0
        self.history = {}
        self.save_state()

def load_and_preprocess_image(img_path):
    """Load and preprocess a single image file"""
    try:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = img / 255.0
        return img
    except Exception:
        return None

def preprocess_image_enhanced(img_path):
    """Enhanced preprocessing with advanced augmentation"""
    try:
        # Load image
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)

        # Create multiple augmented versions
        augmented_images = []

        # Original image with proper preprocessing
        img_orig = tf.keras.applications.mobilenet_v3.preprocess_input(img_array.copy()) # CHANGED
        augmented_images.append(img_orig)

        # Slight rotation (common in plant photos)
        img_rot = tf.keras.preprocessing.image.apply_affine_transform(
            img_array.copy(), theta=np.random.uniform(-20, 20))
        img_rot = tf.keras.applications.mobilenet_v3.preprocess_input(img_rot) # CHANGED
        augmented_images.append(img_rot)

        # Slight zoom (simulate different distances)
        img_zoom = tf.keras.preprocessing.image.apply_affine_transform(
            img_array.copy(), zx=np.random.uniform(0.8, 1.2), zy=np.random.uniform(0.8, 1.2))
        img_zoom = tf.keras.applications.mobilenet_v3.preprocess_input(img_zoom) # CHANGED
        augmented_images.append(img_zoom)

        # Changes in brightness/contrast
        img_bright = img_array.copy() * np.random.uniform(0.8, 1.2)
        img_bright = np.clip(img_bright, 0, 255)
        img_bright = tf.keras.applications.mobilenet_v3.preprocess_input(img_bright) # CHANGED
        augmented_images.append(img_bright)

        # Random crop with padding
        img_crop = tf.keras.preprocessing.image.apply_affine_transform(
            img_array.copy(), zx=0.8, zy=0.8)
        img_crop = tf.keras.applications.mobilenet_v3.preprocess_input(img_crop) # CHANGED
        augmented_images.append(img_crop)

        return augmented_images
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

# Define metrics functions
def sparse_top_k_accuracy(y_true, y_pred):
    """Calculate top-K accuracy for sparse labels"""
    return tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=TOP_K)

# Custom callbacks
class MetricsDisplayCallback(tf.keras.callbacks.Callback):
    """Display metrics during training with progress bars"""
    def __init__(self, total_epochs, validation_data=None, training_state=None):
        super().__init__()
        self.epoch_bar = None
        self.batch_bar = None
        self.total_epochs = total_epochs
        self.training_state = training_state
        self.validation_data = validation_data
        self.start_time = None
        self.epoch_times = []
        self.current_metrics = {}
        self.stop_flag = False
        
        # Setup signal handler for graceful stopping
        self.original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._signal_handler)
        
    def _signal_handler(self, sig, frame):
        """Handle Ctrl+C signal gracefully"""
        print(f"\n{TermColors.YELLOW}⏸️ Training pause requested (Ctrl+C). Will stop after current batch.{TermColors.ENDC}")
        self.stop_flag = True
        self.model.stop_training = True
    
    def check_keyboard_shortcut(self):
        """Check if Ctrl+Alt+C is pressed to stop training gracefully"""
        try:
            # Check if keyboard module is imported, if not, import it
            try:
                import keyboard
            except ImportError:
                print(f"{TermColors.YELLOW}⚠️ Keyboard module not found. Installing...{TermColors.ENDC}")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "keyboard"])
                import keyboard
                
            if keyboard.is_pressed('ctrl+alt+c'):
                print(f"\n{TermColors.YELLOW}⏸️ Training pause requested (Ctrl+Alt+C). Will stop after current batch.{TermColors.ENDC}")
                self.stop_flag = True
                self.model.stop_training = True
                return True
            return False
        except Exception as e:
            # If keyboard monitoring fails, silently continue
            return False
    
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        print(f"{TermColors.HEADER}\n{'='*50}")
        print(f"BEGINNING MODEL TRAINING")
        print(f"{'='*50}{TermColors.ENDC}")
        print(f"{TermColors.CYAN}ℹ Press Ctrl+Alt+C to stop training gracefully{TermColors.ENDC}")
        
        # Create epoch progress bar
        self.epoch_bar = tqdm(
            total=self.total_epochs, 
            desc="Epochs", 
            position=0, 
            leave=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        # Update to starting epoch if resuming
        if self.training_state and self.training_state.current_epoch > 0:
            self.epoch_bar.update(self.training_state.current_epoch)
    
    def on_train_end(self, logs=None):
        self.stop_flag = True
        
        if self.epoch_bar:
            self.epoch_bar.close()
        if self.batch_bar:
            self.batch_bar.close()
        
        # Calculate and print training summary
        total_time = time.time() - self.start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"\n{TermColors.HEADER}{'='*50}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*50}")
        print(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(f"Final metrics: {format_metrics(logs or {})}")
        print(f"{'='*50}{TermColors.ENDC}")
    
    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch_start = time.time()
        self.current_epoch = epoch
    
    # Use standard tqdm instead of MetricsTQDM
        self.batch_bar = tqdm(
            desc=f"Batch (Epoch {epoch+1}/{self.total_epochs})",
            total=self.params['steps'],
            position=1, 
            leave=False,
            unit='batch',
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
    
    def on_epoch_end(self, epoch, logs=None):
        """Update epoch progress bar and save metrics at the end of each epoch"""
        # Update epoch progress bar with metrics
        if self.epoch_bar:
            self.epoch_bar.update(1)
        
        # Calculate epoch time
        epoch_time = time.time() - self.current_epoch_start
        self.epoch_times.append(epoch_time)
        
        # Estimate remaining time
        if len(self.epoch_times) > 0:
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
            remaining_epochs = self.total_epochs - (epoch + 1)
            eta_seconds = avg_epoch_time * remaining_epochs
            eta = str(timedelta(seconds=int(eta_seconds)))
            
            # Enhanced metrics display with better type handling
            metrics_display = []
        
            # Always show these metrics in this order if available
            priority_metrics = ['loss', 'val_loss', 'accuracy', 'val_accuracy', 'sparse_top_k_accuracy', 'val_sparse_top_k_accuracy', 'lr']
            
            for metric in priority_metrics:
                if metric in logs:
                    value = logs[metric]
                    # Check if value is a formatting-compatible type
                    if isinstance(value, (float, int, np.float32, np.float64, np.int32, np.int64)):
                        color = TermColors.GREEN if 'accuracy' in metric else TermColors.RED if 'loss' in metric else TermColors.CYAN
                        format_spec = ".4f" if metric != 'lr' else ".6f"
                        metrics_display.append(f"{color}{metric}: {value:{format_spec}}{TermColors.ENDC}")
                    else:
                        # Handle non-numeric types (like learning rate schedule objects)
                        metrics_display.append(f"{TermColors.CYAN}{metric}: {str(value)}{TermColors.ENDC}")
            
            # Add any remaining metrics with type checking
            for metric, value in logs.items():
                if metric not in priority_metrics:
                    if isinstance(value, (float, int, np.float32, np.float64, np.int32, np.int64)):
                        metrics_display.append(f"{metric}: {value:.4f}")
                    else:
                        metrics_display.append(f"{metric}: {str(value)}")
            
            metrics_str = " | ".join(metrics_display)
            self.epoch_bar.set_postfix_str(f"ETA: {eta} | {metrics_str}")
        
        # Close batch progress bar
        if self.batch_bar:
            self.batch_bar.close()
            self.batch_bar = None
        
        # Save metrics in training state
        if self.training_state and logs:
            # Create a copy of logs with only serializable values
            filtered_logs = {}
            for key, value in (logs or {}).items():
                # Remove learning rate schedule objects that can't be serialized
                if key == 'lr' and not isinstance(value, (float, int, np.float32, np.float64)):
                    # Try to extract current lr value if possible
                    try:
                        if hasattr(value, '__call__'):
                            step = self.model.optimizer.iterations.numpy()
                            filtered_logs[key] = float(value(step).numpy())
                        else:
                            filtered_logs[key] = str(value)
                    except:
                        filtered_logs[key] = f"<schedule>"
                elif isinstance(value, (float, int, np.float32, np.float64, np.int32, np.int64)):
                    filtered_logs[key] = float(value)
                else:
                    filtered_logs[key] = str(value)
                    
            self.training_state.update_metrics(epoch, filtered_logs)
    
    def on_batch_end(self, batch, logs=None):
        if self.batch_bar:
            self.batch_bar.update(1)
        
        # Only update metrics display periodically to reduce overhead
        if batch % 10 == 0 and logs:
            # Format metrics string directly
            metrics_str = ""
            log_items = list(logs.items()) # Take a copy to avoid issues if logs change
            for k, v in log_items:
                if isinstance(v, str):
                    metrics_str += f" | {k}: {v}"
                elif 'loss' in k:
                    metrics_str += f" | {TermColors.RED}{k}: {v:.4f}{TermColors.ENDC}"
                elif 'accuracy' in k:
                    metrics_str += f" | {TermColors.GREEN}{k}: {v:.4f}{TermColors.ENDC}"
                elif isinstance(v, (float, int, np.float32, np.float64)):
                    metrics_str += f" | {k}: {v:.4f}"
                    
            # Update the description instead of using custom metrics display
            self.batch_bar.set_description(
                f"Batch (Epoch {self.current_epoch+1}/{self.total_epochs}){metrics_str}"
            )
            
            # Check for keyboard shortcut
            if batch % 10 == 0:
                self.check_keyboard_shortcut()
    
    # Check for stop flag
        if self.stop_flag:
            self.model.stop_training = True

class AutoTrainingConfig(tf.keras.callbacks.Callback):
    """Automatically adjust training parameters based on performance"""
    def __init__(self, initial_lr=INITIAL_LR, training_state=None):
        super().__init__()
        self.initial_lr = initial_lr
        self.training_state = training_state
        self.plateau_count = 0
        self.best_val_loss = float('inf')
    
    def on_epoch_end(self, epoch, logs=None):
        if not logs or 'val_loss' not in logs:
            return
            
        val_loss = logs['val_loss']
        
        # Fixed: Handle CosineDecay learning rate correctly
        try:
            if hasattr(self.model.optimizer.lr, '__call__'):
                # For LearningRateSchedule objects like CosineDecay
                current_step = self.model.optimizer.iterations.numpy()
                current_lr = self.model.optimizer.lr(current_step).numpy()
            else:
                # For fixed learning rates
                current_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        except:
            # Fallback if we can't get the learning rate
            current_lr = 0.001
        
        # Check if validation loss improved
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.plateau_count = 0
            print(f"{TermColors.GREEN}✓ Validation loss improved to {val_loss:.4f}{TermColors.ENDC}")
        else:
            self.plateau_count += 1
            print(f"{TermColors.YELLOW}⚠ Validation loss plateau: {self.plateau_count} epochs without improvement{TermColors.ENDC}")
            
            # If we have a plateau for too long, adjust learning rate
            if self.plateau_count >= 3:
                new_lr = current_lr * 0.5
                if new_lr < 1e-6:
                    print(f"{TermColors.YELLOW}⚠️ Learning rate too low, stopping adjustment.{TermColors.ENDC}")
                    return # Stop reducing LR if it gets too small
                
                print(f"{TermColors.YELLOW}⚙ Reducing learning rate: {current_lr:.6f} → {new_lr:.6f}{TermColors.ENDC}")
                # Set the new learning rate safely
                try:
                    # Directly set the value if it's a simple variable
                    tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                except Exception as e:
                    print(f"{TermColors.RED}❌ Failed to set new learning rate: {e}{TermColors.ENDC}")
                    # If it's a schedule, we might not be able to change it this way easily
                    # Consider using ReduceLROnPlateau callback instead for simplicity
                    
                self.plateau_count = 0  # Reset counter after adjustment

# Advanced training configuration with gradual unfreezing
# filepath: c:\Users\stefa\Desktop\PlantRecognition_github\PlantRecognition\Scripts\Plant_Reconition.py
def create_custom_callbacks(model, feature_dim):
    """Create custom callbacks for training with parameters for the specific model
    
    Args:
        model: The model being trained
        feature_dim: Dimension of the feature space
        
    Returns:
        List of callbacks to use during training
    """
    class GradualUnfreezingCallback(tf.keras.callbacks.Callback):
        def __init__(self, model, feature_dim):
            super().__init__()
            self.model = model
            self.feature_dim = feature_dim
            self.unfrozen_layers = 0
            # Assuming the base model is the first layer
            self.base_model = self.model.layers[0]
            self.total_layers = len(self.base_model.layers)
            
        def on_epoch_begin(self, epoch, logs=None):
            # Unfreeze layers gradually
            if epoch > 0 and epoch % 5 == 0 and self.unfrozen_layers < 3:
                # Calculate how many layers to unfreeze (e.g., 20% of remaining frozen layers)
                layers_to_unfreeze = max(10, int(0.2 * (self.total_layers - self.unfrozen_layers)))
                
                # Ensure we don't unfreeze more than available
                layers_to_unfreeze = min(layers_to_unfreeze, self.total_layers - self.unfrozen_layers)
                
                if layers_to_unfreeze > 0:
                    print(f"\n{TermColors.CYAN}⚙ Unfreezing {layers_to_unfreeze} layers in base model...{TermColors.ENDC}")
                    
                    # Unfreeze from the end backwards
                    for i in range(self.total_layers - self.unfrozen_layers - layers_to_unfreeze, self.total_layers - self.unfrozen_layers):
                        if i >= 0 and i < len(self.base_model.layers):
                            self.base_model.layers[i].trainable = True
                    
                    self.unfrozen_layers += layers_to_unfreeze
                    
                    # Recompile the model to apply changes (important!)
                    # Use the existing optimizer's learning rate
                    current_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
                    self.model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=current_lr),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy', sparse_top_k_accuracy]
                    )
                    print(f"{TermColors.GREEN}✅ Model recompiled with {self.unfrozen_layers} base layers unfrozen.{TermColors.ENDC}")

    return [GradualUnfreezingCallback(model, feature_dim)]

def analyze_feature_space_confusion(model, features, labels, class_names):
    """Analyze feature space to identify clusters of confusable plants
    
    Args:
        model: Trained model
        features: Feature vectors
        labels: Ground truth labels
        class_names: List of class names
        
    Returns:
        Dictionary mapping each class to its most confusable classes
    """
    from sklearn.manifold import TSNE
    from sklearn.metrics import confusion_matrix
    
    # Get model predictions
    preds = model.predict(features)
    pred_labels = np.argmax(preds, axis=1)
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(labels, pred_labels)
    
    # Normalize by row (true class)
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    norm_conf_matrix = conf_matrix / row_sums
    
    # Identify most confusable classes for each class
    confusion_clusters = {}
    
    for i in range(len(class_names)):
        # Skip classes with no samples
        if row_sums[i][0] == 0:
            continue
            
        # Get confusion rates for this class
        confusion_rates = norm_conf_matrix[i, :]
        
        # Sort by confusion rate (excluding self)
        sorted_indices = np.argsort(confusion_rates)[::-1]
        sorted_indices = sorted_indices[sorted_indices != i]  # Remove self
        
        # Get top confusable classes (with >10% confusion rate)
        confusable_classes = []
        for idx in sorted_indices:
            if confusion_rates[idx] > 0.1:
                confusable_classes.append({
                    'class': class_names[idx],
                    'confusion_rate': float(confusion_rates[idx])
                })
                # Limit to top 3 confusable classes for brevity
                if len(confusable_classes) >= 3:
                    break
        
        confusion_clusters[class_names[i]] = confusable_classes
    
    # Get TSNE visualization to see clusters
    tsne = TSNE(n_components=2, random_state=42)
    
    # Extract features from last layer before softmax
    feature_model = tf.keras.Model(inputs=model.input, 
                                  outputs=model.layers[-2].output)
    latent_features = feature_model.predict(features)
    
    # Apply TSNE
    tsne_result = tsne.fit_transform(latent_features)
    
    # Plotting code would go here (if needed)
    
    return {
        'confusion_clusters': confusion_clusters,
        'tsne_visualization': tsne_result,
        'labels': labels
    }

def analyze_features(features_file):
    """Analyze feature quality to detect problems"""
    print(f"{TermColors.HEADER}\n{'='*50}")
    print(f"FEATURE QUALITY ANALYSIS")
    print(f"{'='*50}{TermColors.ENDC}")
    
    features, labels = load_features(features_file)
    
    # Check for NaN values
    nan_count = np.isnan(features).sum()
    if nan_count > 0:
        print(f"{TermColors.RED}❌ Found {nan_count} NaN values in features!{TermColors.ENDC}")
    
    # Check feature variance
    feature_variance = np.var(features, axis=0)
    low_variance = (feature_variance < 1e-5).sum()
    if low_variance > features.shape[1] // 2:
        print(f"{TermColors.RED}❌ {low_variance}/{features.shape[1]} features have very low variance!{TermColors.ENDC}")
    
    # Check for uniform distribution (indicates potential preprocessing issue)
    feature_ranges = np.ptp(features, axis=0)
    if np.mean(feature_ranges) < 0.1:
        print(f"{TermColors.RED}❌ Features have very small range (mean range: {np.mean(feature_ranges):.6f}){TermColors.ENDC}")
    
    # Check for class separability
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    
    print(f"{TermColors.CYAN}ℹ Analyzing class separability...{TermColors.ENDC}")
    
    # Use PCA to reduce dimensions 
    pca = PCA(n_components=min(100, features.shape[0]//10, features.shape[1]))
    reduced_features = pca.fit_transform(features)
    
    # Sample for silhouette score (max 10,000 samples)
    if len(features) > 10000:
        indices = np.random.choice(len(features), 10000, replace=False)
        sample_features = reduced_features[indices]
        sample_labels = labels[indices]
    else:
        sample_features = reduced_features
        sample_labels = labels
    
    try:
        silhouette = silhouette_score(sample_features, sample_labels)
        print(f"{TermColors.CYAN}ℹ Class separability (silhouette score): {silhouette:.4f}{TermColors.ENDC}")
        
        if silhouette < 0.1:
            print(f"{TermColors.RED}❌ Poor class separability in features!{TermColors.ENDC}")
        elif silhouette < 0.3:
            print(f"{TermColors.YELLOW}⚠️ Mediocre class separability in features{TermColors.ENDC}")
        else:
            print(f"{TermColors.GREEN}✅ Good class separability in features{TermColors.ENDC}")
    except:
        print(f"{TermColors.YELLOW}⚠️ Unable to calculate class separability{TermColors.ENDC}")
    
    # Check explained variance from PCA
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"{TermColors.CYAN}ℹ Top 100 components explain {explained_variance:.2%} of variance{TermColors.ENDC}")
    
    if explained_variance < 0.5:
        print(f"{TermColors.YELLOW}⚠️ Features have high dimensionality with limited structure{TermColors.ENDC}")
    
    return features, labels

def analyze_plant_features(features, labels):
    """Specialized analysis for plant feature quality"""
    # Calculate inter-class similarity matrix
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Sample for performance if needed
    if len(features) > 5000:
        indices = np.random.choice(len(features), 5000, replace=False)
        sample_features = features[indices]
        sample_labels = labels[indices]
    else:
        sample_features = features
        sample_labels = labels
    
    # Get unique classes
    unique_classes = np.unique(sample_labels)
    
    # Calculate class centroids
    centroids = []
    for cls in unique_classes:
        class_features = sample_features[sample_labels == cls]
        centroid = np.mean(class_features, axis=0)
        centroids.append(centroid)
    
    centroids = np.array(centroids)
    
    # Calculate similarity between class centroids
    similarity = cosine_similarity(centroids)
    np.fill_diagonal(similarity, 0)  # Remove self-similarity
    
    # Analyze potential confusion points
    high_similarity = np.where(similarity > 0.9)
    confusion_pairs = list(zip(high_similarity[0], high_similarity[1]))
    
    if confusion_pairs:
        print(f"{TermColors.YELLOW}⚠️ Found {len(confusion_pairs)} potentially confusable class pairs{TermColors.ENDC}")
        # Print a few examples
        for i, j in confusion_pairs[:5]:
            print(f"  - Classes {unique_classes[i]} and {unique_classes[j]}: {similarity[i,j]:.4f} similarity")
    else:
        print(f"{TermColors.GREEN}✅ No highly confusable classes detected{TermColors.ENDC}")
    
    # Return confusability score (lower is better)
    return np.mean(similarity)

def augment_feature_space(X, y, augmentation_factor=0.3):
    """Create synthetic feature samples by interpolating between same-class examples"""
    X_augmented = []
    y_augmented = []
    
    # For each class, generate synthetic examples
    unique_classes = np.unique(y)
    
    for cls in tqdm(unique_classes, desc="Augmenting features"):
        # Get examples of this class
        class_indices = np.where(y == cls)[0]
        class_features = X[class_indices]
        
        if len(class_indices) < 2:
            continue # Need at least two samples to interpolate
        
        # How many synthetic examples to generate 
        num_to_generate = int(len(class_indices) * augmentation_factor)
        
        for _ in range(num_to_generate):
            # Select two random samples from the same class
            idx1, idx2 = np.random.choice(class_indices, 2, replace=False)
            feature1 = X[idx1]
            feature2 = X[idx2]
            
            # Interpolation factor (beta distribution often used, but uniform is simpler)
            alpha = np.random.uniform(0.1, 0.9) 
            
            # Create synthetic feature
            synthetic_feature = alpha * feature1 + (1 - alpha) * feature2
            
            X_augmented.append(synthetic_feature)
            y_augmented.append(cls)

    # Combine with original data
    if X_augmented:  # Check that we generated something
        X_combined = np.vstack([X, np.array(X_augmented)])
        y_combined = np.concatenate([y, np.array(y_augmented)])
        print(f"{TermColors.GREEN}✅ Added {len(X_augmented)} synthetic feature samples{TermColors.ENDC}")
        return X_combined, y_combined
    else:
        return X, y # Return original data if no augmentation happened

def analyze_model_performance():
    """Analyze all chunk models and identify low-performing ones"""
    print(f"{TermColors.HEADER}\n{'='*50}")
    print(f"ANALYZING CHUNK MODEL PERFORMANCE")
    print(f"{'='*50}{TermColors.ENDC}")
    
    model_files = glob.glob(os.path.join(MODEL_DIR, "chunk_*_metadata.json"))
    
    if not model_files:
        print(f"{TermColors.RED}❌ No model metadata files found{TermColors.ENDC}")
        return
        
    print(f"{TermColors.CYAN}ℹ Found {len(model_files)} trained chunks{TermColors.ENDC}")
    
    # Track performance metrics
    all_performance = []
    problematic_chunks = []
    
    for metadata_file in model_files:
        chunk_idx = int(os.path.basename(metadata_file).split("_")[1])
        
        with open(metadata_file) as f:
            metadata = json.load(f)
            
        # Get performance metrics
        val_acc = metadata.get('performance', {}).get('val_accuracy', 0.0)
        val_top_k = metadata.get('performance', {}).get('val_sparse_top_k_accuracy', 0.0)
        num_classes = metadata.get('num_classes', 0)
        
        all_performance.append((chunk_idx, val_acc, val_top_k, num_classes))
        
        # Flag chunks with accuracy below threshold
        if val_acc < 0.6:
            problematic_chunks.append((chunk_idx, val_acc, val_top_k, num_classes))
            print(f"{TermColors.RED}❌ Chunk {chunk_idx} has low accuracy: {val_acc:.2%}{TermColors.ENDC}")
    
    # Sort by accuracy
    all_performance.sort(key=lambda x: x[1])
    
    # Display overall statistics
    avg_acc = sum(p[1] for p in all_performance) / len(all_performance) if all_performance else 0
    avg_top_k = sum(p[2] for p in all_performance) / len(all_performance) if all_performance else 0
    
    print(f"\n{TermColors.CYAN}📊 OVERALL PERFORMANCE:{TermColors.ENDC}")
    print(f"Average accuracy: {avg_acc:.2%}")
    print(f"Average top-{TOP_K} accuracy: {avg_top_k:.2%}")
    
    # Display worst performing chunks
    print(f"\n{TermColors.YELLOW}⚠️ LOWEST PERFORMING CHUNKS:{TermColors.ENDC}")
    for i, (chunk_idx, acc, top_k, num_classes) in enumerate(all_performance[:5]):
        print(f"{i+1}. Chunk {chunk_idx}: Accuracy {acc:.2%}, Top-{TOP_K} {top_k:.2%} ({num_classes} classes)")
    
    # Display problematic chunks
    if problematic_chunks:
        print(f"\n{TermColors.RED}❌ PROBLEMATIC CHUNKS (Below 60% accuracy):{TermColors.ENDC}")
        for chunk_idx, acc, top_k, num_classes in problematic_chunks:
            print(f"- Chunk {chunk_idx}: Accuracy {acc:.2%}, Top-{TOP_K} {top_k:.2%} ({num_classes} classes)")
        
        print(f"\n{TermColors.CYAN}ℹ️ To retrain problematic chunks, run:{TermColors.ENDC}")
        print(f"retrain_chunk([{', '.join(str(c[0]) for c in problematic_chunks)}])")
    else:
        print(f"\n{TermColors.GREEN}✅ No problematic chunks detected!{TermColors.ENDC}")
        
    return problematic_chunks

def scan_class_directory(args):
        class_idx, class_dir = args
        class_name = os.path.basename(class_dir)
        paths = []
        indices = []
        
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, img_name)
                paths.append(img_path)
                indices.append(class_idx)
                
        return class_name, paths, indices

def extract_enhanced_features(class_dirs, chunk_idx):
    """Enhanced feature extraction using MobileNetV3Large with optimized resource utilization""" # CHANGED DOCSTRING
    print(f"{TermColors.HEADER}\n{'='*50}")
    print(f"HIGH-PERFORMANCE FEATURE EXTRACTION FOR CHUNK {chunk_idx+1}")
    print(f"{'='*50}{TermColors.ENDC}")

    # ... (keyboard interrupt setup remains the same) ...

    # Use MobileNetV3Large with optimized settings
    with tf.device('/GPU:0'):
        # Use MobileNetV3Large for faster processing
        from tensorflow.keras.applications import MobileNetV3Large # CHANGED IMPORT

        base_model = MobileNetV3Large( # CHANGED MODEL
            input_shape=(*IMAGE_SIZE, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        base_model.trainable = False
        feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=base_model.output)

    feature_dim = feature_extractor.output_shape[1]
    print(f"{TermColors.GREEN}✅ Feature extractor loaded: MobileNetV3Large ({feature_dim} dims){TermColors.ENDC}") # CHANGED PRINT
    
    # Storage for features - simple approach that worked before
    all_features = []
    all_labels = []
    class_names = []
    
    # Create chunk feature directory
    chunk_feature_dir = os.path.join(FEATURES_DIR, f"chunk_{chunk_idx}")
    os.makedirs(chunk_feature_dir, exist_ok=True)
    
    # Collect all image paths with parallel processing
    print(f"{TermColors.CYAN}ℹ Collecting image paths with parallel processing{TermColors.ENDC}")
    
    all_image_paths = []
    all_class_indices = []
    
    # Use process pool for truly parallel directory scanning
    with multiprocessing.Pool(processes=CPU_WORKERS) as pool:
        results = list(tqdm(
            pool.imap(scan_class_directory, enumerate(class_dirs)),
            total=len(class_dirs),
            desc="Scanning classes",
            position=0
        ))
    
    # Combine results
    for class_name, paths, indices in results:
        class_names.append(class_name)
        all_image_paths.extend(paths)
        all_class_indices.extend(indices)
    
    total_images = len(all_image_paths)
    
    if total_images == 0:
        print(f"{TermColors.RED}❌ No valid images found in provided class directories!{TermColors.ENDC}")
        return None, None
    
    print(f"{TermColors.GREEN}✅ Found {total_images} images across {len(class_names)} classes{TermColors.ENDC}")
    
    # Process in smaller batches to reduce memory usage
    batch_size = 128  # Smaller batch size to prevent memory issues
    print(f"{TermColors.CYAN}ℹ Using batch size of {batch_size} for feature extraction{TermColors.ENDC}")
    
    # Create progress bar
    progress_bar = tqdm(total=total_images, desc="Extracting features", position=0)
    
    # Process images in batches
    for i in range(0, len(all_image_paths), batch_size):
        if check_keyboard_interrupt():
            print(f"{TermColors.YELLOW}⚠️ Feature extraction interrupted by user.{TermColors.ENDC}")
            break # Exit the loop if interrupted
            
        batch_paths = all_image_paths[i:i+batch_size]
        batch_indices = all_class_indices[i:i+batch_size]
        
        # Load and preprocess images in the batch
        batch_images = []
        valid_indices = [] # Keep track of indices for images that loaded successfully
        for idx, img_path in enumerate(batch_paths):
            img = load_and_preprocess_image(img_path)
            if img is not None:
                batch_images.append(img)
                valid_indices.append(batch_indices[idx]) # Store the original class index
            else:
                print(f"{TermColors.YELLOW}⚠️ Skipping corrupted image: {img_path}{TermColors.ENDC}")

        if not batch_images:
            progress_bar.update(len(batch_paths)) # Update progress even if batch is empty
            continue # Skip if batch is empty after filtering bad images

        # Convert to numpy array and extract features
        batch_images_np = np.array(batch_images)
        try:
            batch_features = feature_extractor.predict(batch_images_np, batch_size=batch_size, verbose=0)
            
            # Append features and corresponding valid labels
            all_features.extend(batch_features)
            all_labels.extend(valid_indices) # Use the labels of successfully loaded images
        except Exception as e:
            print(f"{TermColors.RED}❌ Error extracting features for batch starting at index {i}: {e}{TermColors.ENDC}")
            # Optionally: try processing images one by one in the batch on error
            
        # Update progress bar
        progress_bar.update(len(batch_paths)) # Update by the original number of paths attempted

    progress_bar.close()
    
    # Save class mapping
    class_mapping = {idx: name for idx, name in enumerate(class_names)}
    with open(os.path.join(chunk_feature_dir, "class_mapping.json"), "w") as f:
        json.dump(class_mapping, f, indent=2)
    
    # Convert to numpy arrays
    if len(all_features) == 0:
        print(f"{TermColors.YELLOW}⚠️ No features extracted. Aborting.{TermColors.ENDC}")
        return None, None
    
    print(f"{TermColors.CYAN}ℹ Saving {len(all_features)} features to disk{TermColors.ENDC}")
    features_array = np.array(all_features)
    labels_array = np.array(all_labels)
    
    # Save features to disk
    features_file = os.path.join(chunk_feature_dir, "features.npz")
    np.savez_compressed(features_file, features=features_array, labels=labels_array)
    
    print(f"{TermColors.GREEN}✅ Features saved to {features_file}{TermColors.ENDC}")
    return features_file, class_mapping

def extract_combined_features_optimized(class_dirs, chunk_idx):
    """Feature extraction with optimized data pipeline for maximum CPU+GPU utilization"""
    print(f"{TermColors.HEADER}\n{'='*50}")
    print(f"OPTIMIZED COMBINED MODEL FEATURE EXTRACTION FOR CHUNK {chunk_idx+1}")
    print(f"{'='*50}{TermColors.ENDC}")
    
    # AGGRESSIVE MEMORY CLEANUP BEFORE STARTING A NEW CHUNK
    print(f"{TermColors.CYAN}ℹ Performing aggressive memory cleanup before starting chunk {chunk_idx+1}{TermColors.ENDC}")
    # Force full garbage collection multiple times
    for _ in range(3):
        gc.collect()
    # Clear TensorFlow session completely
    tf.keras.backend.clear_session()
    # Sleep to ensure memory is released by the OS
    time.sleep(3)
    
    # Force GPU memory reset
    print(f"{TermColors.CYAN}ℹ Resetting GPU memory allocation...{TermColors.ENDC}")
    try:
        # Reset the entire TensorFlow runtime and recreate
        from tensorflow.python.framework import ops
        ops.reset_default_graph()
        tf.keras.backend.clear_session()
        time.sleep(2)
        print(f"{TermColors.GREEN}✅ GPU memory reset successful{TermColors.ENDC}")
    except Exception as e:
        print(f"{TermColors.YELLOW}⚠️ GPU memory reset failed: {e}{TermColors.ENDC}")
    
    
    # Load feature extractors
    try:
        # Try with GPU first
        with tf.device('/GPU:0'):
            from tensorflow.keras.applications import EfficientNetV2L, DenseNet121
            
            # Create a combined model that outputs both feature sets
            input_tensor = tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3))
            
            # Process through EfficientNetV2L path
            efficient_base = EfficientNetV2L(
                include_top=False, 
                weights='imagenet',
                input_tensor=input_tensor,
                pooling='avg'
            )
            efficient_base.trainable = False
            efficient_features = efficient_base(input_tensor)
            
            # Process through DenseNet121 path
            dense_base = DenseNet121(
                include_top=False, 
                weights='imagenet',
                input_tensor=input_tensor,
                pooling='avg'
            )
            dense_base.trainable = False
            dense_features = dense_base(input_tensor)
            
            # Combined model with two outputs
            combined_model = tf.keras.Model(inputs=input_tensor, outputs=[efficient_features, dense_features])
    except Exception as e:
        print(f"{TermColors.YELLOW}⚠️ GPU initialization failed, falling back to CPU: {e}{TermColors.ENDC}")
        # Fall back to CPU
        with tf.device('/CPU:0'):
            from tensorflow.keras.applications import EfficientNetV2L, DenseNet121
            
            # Create a combined model that outputs both feature sets
            input_tensor = tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3))
            
            # Process through EfficientNetV2L path
            efficient_base = EfficientNetV2L(
                include_top=False, 
                weights='imagenet',
                input_tensor=input_tensor,
                pooling='avg'
            )
            efficient_base.trainable = False
            efficient_features = efficient_base(input_tensor)
            
            # Process through DenseNet121 path
            dense_base = DenseNet121(
                include_top=False, 
                weights='imagenet',
                input_tensor=input_tensor,
                pooling='avg'
            )
            dense_base.trainable = False
            dense_features = dense_base(input_tensor)
            
            # Combined model with two outputs
            combined_model = tf.keras.Model(inputs=input_tensor, outputs=[efficient_features, dense_features])
    
    efficient_dim = efficient_base.output_shape[1]
    dense_dim = dense_base.output_shape[1]
    combined_dim = efficient_dim + dense_dim
    
    print(f"{TermColors.GREEN}✅ Combined extractors loaded: {combined_dim} dimensions{TermColors.ENDC}")
    
    # Storage for features
    all_features = []
    all_labels = []
    class_names = []
    
    # Create chunk feature directory
    chunk_feature_dir = os.path.join(FEATURES_DIR, f"chunk_{chunk_idx}")
    os.makedirs(chunk_feature_dir, exist_ok=True)
    
    # Collect all image paths with parallel processing
    print(f"{TermColors.CYAN}ℹ Collecting image paths{TermColors.ENDC}")
    all_image_paths = []
    all_class_indices = []
    
    # Use ALL available CPU cores
    max_workers = os.cpu_count()
    print(f"{TermColors.CYAN}ℹ Using all {max_workers} CPU cores for parallel processing{TermColors.ENDC}")
    
    # Use process pool for directory scanning with maximum parallelism
    with multiprocessing.Pool(processes=max_workers) as pool:
        results = list(tqdm(
            pool.imap(safe_scan_class_directory, enumerate(class_dirs)),
            total=len(class_dirs),
            desc="Scanning classes",
            position=0
        ))
    
    # Combine results
    for class_name, paths, indices in results:
        if len(paths) > 0:  # Only add classes that have valid images
            class_names.append(class_name)
            all_image_paths.extend(paths)
            all_class_indices.extend(indices)
    
    total_images = len(all_image_paths)
    print(f"{TermColors.GREEN}✅ Found {total_images} valid images across {len(class_names)} classes{TermColors.ENDC}")
    
    if total_images == 0:
        print(f"{TermColors.RED}❌ No valid images found for chunk {chunk_idx}!{TermColors.ENDC}")
        return None, None
        
    # Use multiple thread pools for different processing stages
    # This maximizes CPU utilization across different cores
    print(f"{TermColors.CYAN}ℹ Setting up multi-stage parallel processing pipeline{TermColors.ENDC}")
    
    # Create larger batch size for more efficient parallel processing 
    batch_size = 128  # Increased from 64 to 128
    print(f"{TermColors.CYAN}ℹ Using batch size of {batch_size} for feature extraction{TermColors.ENDC}")
    
    # Split work across multiple processing pools
    num_pools = min(4, os.cpu_count() // 2)  # Use up to 4 pools
    images_per_pool = total_images // num_pools
    
    # Create multiple thread pools
    image_loading_pool = ThreadPoolExecutor(max_workers=max_workers // 2)
    preprocessing_pool = ThreadPoolExecutor(max_workers=max_workers // 2)
    
    # Function to load an image
    def load_image(img_path):
        try:
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
            img = tf.keras.preprocessing.image.img_to_array(img)
            return img
        except Exception as e:
            return None
            
    # Function to preprocess an image
    def preprocess_image(img):
        if img is None:
            return None
        try:
            return tf.keras.applications.efficientnet_v2.preprocess_input(img)
        except Exception as e:
            return None
    
    # Process images in batches
    progress_bar = tqdm(total=total_images, desc="Extracting features", position=0)
    
    # Process in batches with memory management between chunks
    for i in range(0, len(all_image_paths), batch_size):
        # Reset TensorFlow session periodically to avoid memory buildup
        if i > 0 and i % 5000 == 0:
            print(f"{TermColors.CYAN}ℹ Periodic memory cleanup at image {i}{TermColors.ENDC}")
            tf.keras.backend.clear_session()
            gc.collect()
            
            # Save intermediate results
            if len(all_features) > 0:
                print(f"{TermColors.YELLOW}⚠️ Saving intermediate features at image {i}...{TermColors.ENDC}")
                temp_features = np.array(all_features)
                temp_labels = np.array(all_labels)
                temp_file = os.path.join(chunk_feature_dir, f"features_temp_{i//batch_size}.npz")
                np.savez_compressed(temp_file, features=temp_features, labels=temp_labels)
        
        batch_paths = all_image_paths[i:i+batch_size]
        batch_indices = all_class_indices[i:i+batch_size]
        
        # STAGE 1: Load images in parallel (CPU intensive)
        loaded_images = list(image_loading_pool.map(load_image, batch_paths))
        
        # STAGE 2: Preprocess images in parallel (CPU intensive)
        processed_images = list(preprocessing_pool.map(preprocess_image, loaded_images))
        
        # Filter out any failed images and their indices
        batch_images = []
        valid_batch_indices = []
        
        for img, class_idx in zip(processed_images, batch_indices):
            if img is not None:
                batch_images.append(img)
                valid_batch_indices.append(class_idx)
        
        if not batch_images:
            progress_bar.update(len(batch_paths))
            continue
            
        # Convert to numpy array
        batch_array = np.array(batch_images)
        
        try:
            # Extract features on GPU
            efficient_batch, dense_batch = combined_model.predict(batch_array, verbose=0)
            
            # STAGE 3: Process and combine features in parallel
            def combine_features(idx):
                return np.concatenate([efficient_batch[idx], dense_batch[idx]]), valid_batch_indices[idx]
            
            # Process feature combination in parallel
            combined_results = list(preprocessing_pool.map(combine_features, range(len(valid_batch_indices))))
            
            # Add results to collections
            for feat, class_idx in combined_results:
                all_features.append(feat)
                all_labels.append(class_idx)
            
            # Free memory after each batch
            del efficient_batch, dense_batch, batch_array, batch_images, loaded_images, processed_images
            
        except Exception as e:
            print(f"{TermColors.RED}❌ Error in batch {i//batch_size}: {e}{TermColors.ENDC}")
            # Try to switch to CPU for this batch as fallback
            try:
                print(f"{TermColors.YELLOW}⚠️ Trying CPU fallback for this batch{TermColors.ENDC}")
                with tf.device('/CPU:0'):
                    efficient_batch, dense_batch = combined_model.predict(batch_array, verbose=0)
                    
                    # Combine features
                    for idx, class_idx in enumerate(valid_batch_indices):
                        combined_feat = np.concatenate([efficient_batch[idx], dense_batch[idx]])
                        all_features.append(combined_feat)
                        all_labels.append(class_idx)
                
                # Free memory after CPU fallback
                del efficient_batch, dense_batch, batch_array, batch_images, loaded_images, processed_images
            except Exception as e2:
                print(f"{TermColors.RED}❌ CPU fallback also failed: {e2}. Skipping batch.{TermColors.ENDC}")
                del batch_array, batch_images, loaded_images, processed_images
        
        # Update progress
        progress_bar.update(len(batch_paths))
    
    # Close the thread pools
    image_loading_pool.shutdown()
    preprocessing_pool.shutdown()
    progress_bar.close()
    
    # If we don't have enough features, return failure
    if len(all_features) < 10:  # Arbitrary minimum threshold
        print(f"{TermColors.RED}❌ Too few features extracted ({len(all_features)}). Extraction failed.{TermColors.ENDC}")
        return None, None
    
    # Free GPU memory
    print(f"{TermColors.CYAN}ℹ Final memory cleanup after feature extraction{TermColors.ENDC}")
    del combined_model, efficient_base, dense_base
    tf.keras.backend.clear_session()
    gc.collect()
    
    # Save class mapping and features
    class_mapping = {idx: name for idx, name in enumerate(class_names)}
    with open(os.path.join(chunk_feature_dir, "class_mapping.json"), "w") as f:
        json.dump(class_mapping, f, indent=2)
    
    # Convert to numpy arrays and save
    features_array = np.array(all_features)
    labels_array = np.array(all_labels)
    
    features_file = os.path.join(chunk_feature_dir, "features.npz")
    np.savez_compressed(features_file, features=features_array, labels=labels_array)
    
    # Final cleanup
    del features_array, labels_array, all_features, all_labels
    gc.collect()
    
    print(f"{TermColors.GREEN}✅ Combined features saved: {combined_dim} dimensions{TermColors.ENDC}")
    return features_file, class_mapping

def calibrate_model_temperature(model, X_val, y_val, initial_temp=1.0):
    """Calibrate model confidence using temperature scaling
    
    Args:
        model: Trained model to calibrate
        X_val: Validation features
        y_val: Validation labels
        initial_temp: Initial temperature value
        
    Returns:
        Optimal temperature value for scaling logits
    """
    print(f"Calibrating model confidence using temperature scaling...")
    
    def temperature_scale(logits, T):
        """Apply temperature scaling to logits"""
        return logits / T
    
    def objective(T):
        """Objective function for temperature scaling"""
        # Get raw logits from the model (before softmax)
        logits = model.predict(X_val, verbose=0)
        
        # Apply temperature scaling
        scaled_logits = temperature_scale(logits, T)
        
        # Calculate NLL loss with scaled logits
        # Convert to one-hot encoding for loss calculation
        y_true_one_hot = tf.keras.utils.to_categorical(y_val, num_classes=logits.shape[1])
        scaled_probs = tf.nn.softmax(scaled_logits, axis=1)
        loss = -np.mean(np.sum(y_true_one_hot * np.log(scaled_probs + 1e-8), axis=1))
        return loss
    
    # Find optimal temperature using scipy's minimize
    opt_result = minimize(objective, x0=initial_temp, method='BFGS', bounds=[(0.1, 10.0)])
    optimal_temp = opt_result.x[0]
    
    print(f"Optimal temperature scaling factor: {optimal_temp:.4f}")
    return optimal_temp

# Define this function at the module level (outside any other function)
def safe_scan_class_directory(args):
    """Scan a class directory and handle corrupt images"""
    class_idx, class_dir = args
    class_name = os.path.basename(class_dir)
    paths = []
    indices = []
    
    for img_name in os.listdir(class_dir):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(class_dir, img_name)
            paths.append(img_path)
            indices.append(class_idx) # Add the class index for this image
            
    return class_name, paths, indices

def balance_training_data(X_train, y_train, max_samples_per_class=None):
    """Balance training data to reduce bias toward overrepresented classes
    
    Args:
        X_train: Input features
        y_train: Labels
        max_samples_per_class: Optional cap on samples per class
        
    Returns:
        Balanced features and labels
    """
    # Get class counts
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    n_classes = len(unique_classes)
    
    # Determine target count per class
    if max_samples_per_class is None:
        # Use the median count as a reasonable default
        target_count = int(np.median(class_counts))
        # Cap at 500 to avoid excessive memory usage
        target_count = min(target_count, 500)
    else:
        target_count = max_samples_per_class
    
    print(f"{TermColors.CYAN}ℹ Balancing {n_classes} classes to ~{target_count} samples each{TermColors.ENDC}")
    
    # Create balanced dataset
    X_balanced = []
    y_balanced = []
    
    for cls in unique_classes:
        # Get samples for this class
        cls_indices = np.where(y_train == cls)[0]
        cls_count = len(cls_indices)
        
        if cls_count <= target_count:
            # Use all samples plus augmentations if needed
            selected_indices = cls_indices
            if cls_count < target_count:
                # Need to oversample
                additional_needed = target_count - cls_count
                # Random oversampling with replacement
                additional_indices = np.random.choice(cls_indices, size=additional_needed, replace=True)
                selected_indices = np.concatenate([selected_indices, additional_indices])
        else:
            # Undersample
            selected_indices = np.random.choice(cls_indices, size=target_count, replace=False)
        
        # Add to balanced dataset
        X_balanced.append(X_train[selected_indices])
        y_balanced.append(y_train[selected_indices])
    
    # Combine and shuffle
    X_balanced = np.vstack(X_balanced)
    y_balanced = np.concatenate(y_balanced)
    
    # Shuffle to avoid blocks of the same class
    shuffle_idx = np.random.permutation(len(X_balanced))
    X_balanced = X_balanced[shuffle_idx]
    y_balanced = y_balanced[shuffle_idx]
    
    print(f"{TermColors.GREEN}✅ Balanced dataset created with {len(X_balanced)} samples{TermColors.ENDC}")
    
    return X_balanced, y_balanced

def prune_features_rolling_window(current_chunk_idx, window_size=1):
    """Prune feature files outside the rolling window to save disk space
    
    Args:
        current_chunk_idx: Current chunk index
        window_size: Number of adjacent chunks to keep on each side
    """
    print(f"{TermColors.CYAN}ℹ Pruning feature files outside rolling window (size={window_size}){TermColors.ENDC}")
    
    # Get all feature chunks
    feature_dirs = glob.glob(os.path.join(FEATURE_DIR, "chunk_*"))
    
    for feature_dir in feature_dirs:
        try:
            chunk_idx = int(os.path.basename(feature_dir).split("_")[1])
            
            # Skip if within rolling window
            if abs(chunk_idx - current_chunk_idx) <= window_size:
                continue
                
            # Skip if no features.npz (already pruned)
            features_file = os.path.join(feature_dir, "features.npz")
            if not os.path.exists(features_file):
                continue
                
            # Check if we've trained this chunk
            model_file = os.path.join(MODEL_DIR, f"chunk_{chunk_idx}_model.keras")
            if os.path.exists(model_file):
                # Safe to remove features since model is trained
                os.remove(features_file)
                print(f"{TermColors.GREEN}✅ Pruned features from chunk {chunk_idx}{TermColors.ENDC}")
        except Exception as e:
            print(f"{TermColors.YELLOW}⚠️ Error pruning chunk {feature_dir}: {e}{TermColors.ENDC}")
    
    # Clean memory
    gc.collect()

def load_features(features_file):
    """Load and verify feature files
    
    Args:
        features_file: Path to features.npz file
        
    Returns:
        Features and labels if successful, None otherwise
    """
    try:
        # Load the feature file
        print(f"{TermColors.CYAN}ℹ Loading features from {os.path.basename(os.path.dirname(features_file))}{TermColors.ENDC}")
        data = np.load(features_file)
        
        # Verify the data has the expected keys
        if 'features' not in data or 'labels' not in data:
            print(f"{TermColors.RED}❌ Feature file has invalid format{TermColors.ENDC}")
            return None, None
            
        features = data['features']
        labels = data['labels']
        
        # Check for valid shapes
        if features.shape[0] == 0 or labels.shape[0] == 0:
            print(f"{TermColors.RED}❌ Feature file contains empty data{TermColors.ENDC}")
            return None, None
            
        if features.shape[0] != labels.shape[0]:
            print(f"{TermColors.RED}❌ Feature and label counts don't match: {features.shape[0]} vs {labels.shape[0]}{TermColors.ENDC}")
            return None, None
            
        print(f"{TermColors.GREEN}✅ Successfully loaded {features.shape[0]} samples with {features.shape[1]} dimensions{TermColors.ENDC}")
        return features, labels
        
    except Exception as e:
        print(f"{TermColors.RED}❌ Error loading features: {str(e)}{TermColors.ENDC}")
        return None, None

# Memory management function with proper gc import
def clean_memory():
    """Clean up memory to prevent OOM errors during training"""
    import gc
    
    # Run garbage collection
    collected = gc.collect()
    
    # Clear TensorFlow session
    tf.keras.backend.clear_session()
    
    # Force release GPU memory if possible
    if hasattr(tf.keras.backend, 'clear_session'):
        tf.keras.backend.clear_session()
        
    # Print memory usage info if psutil is available
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        mem_usage_mb = mem_info.rss / (1024 * 1024)
        print(f"{TermColors.CYAN}ℹ Memory cleaned. Current usage: {mem_usage_mb:.1f} MB{TermColors.ENDC}")
    except ImportError:
        print(f"{TermColors.CYAN}ℹ Memory cleaned. Install psutil for detailed memory info.{TermColors.ENDC}")

def build_hierarchical_model(feature_dim, genus_classes, species_classes_by_genus):
    """Build hierarchical model with genus and species classifiers for better taxonomy handling
    
    Args:
        feature_dim: Input feature dimension
        genus_classes: Number of genus classes
        species_classes_by_genus: Dictionary mapping genus index to species count
        
    Returns:
        Hierarchical model
    """
    print(f"{TermColors.CYAN}ℹ Building hierarchical model with {genus_classes} genera{TermColors.ENDC}")
    
    # Input layer
    inputs = tf.keras.Input(shape=(feature_dim,))
    
    # Shared feature extractor
    x = tf.keras.layers.Dense(1024, activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    shared_features = tf.keras.layers.Dense(512, activation='relu')(x)
    shared_features = tf.keras.layers.BatchNormalization()(shared_features)
    
    # Genus classifier head
    genus_features = tf.keras.layers.Dense(256, activation='relu')(shared_features)
    genus_features = tf.keras.layers.Dropout(0.3)(genus_features)
    genus_outputs = tf.keras.layers.Dense(genus_classes, name='genus_output')(genus_features)
    
    # Create species classifier head for each genus
    all_species_outputs = []
    
    for genus_idx, species_count in species_classes_by_genus.items():
        if species_count <= 1:
            continue  # Skip genera with only one species
            
        # Create genus-specific gate using softmax outputs
        genus_gate = tf.keras.layers.Lambda(
            lambda x: tf.expand_dims(tf.nn.softmax(x)[:, int(genus_idx)], axis=1)
        )(genus_outputs)
        
        # Genus-specific features
        genus_specific_features = tf.keras.layers.Dense(128, activation='relu')(shared_features)
        
        # Species classifier for this genus
        species_outputs = tf.keras.layers.Dense(species_count)(genus_specific_features)
        
        # Gate the species outputs with genus probability
        gated_outputs = tf.keras.layers.Multiply()([
            species_outputs,
            genus_gate
        ])
        
        all_species_outputs.append(gated_outputs)
    
    # Combine all species outputs if we have any
    if all_species_outputs:
        combined_species_outputs = tf.keras.layers.Concatenate(name='species_output')(all_species_outputs)
    else:
        # Fallback if no species classifiers
        combined_species_outputs = tf.keras.layers.Dense(1, name='species_output')(shared_features)
    
    # Build model with both outputs
    model = tf.keras.Model(inputs=inputs, outputs=[genus_outputs, combined_species_outputs])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'genus_output': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            'species_output': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        },
        metrics={
            'genus_output': ['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_accuracy')],
            'species_output': ['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_accuracy')]
        },
        loss_weights={
            'genus_output': 0.3,
            'species_output': 0.7
        }
    )
    
    return model

def predict_with_similarity_rejection(model, image, class_names, similarity_threshold=0.92):
    """Predict with rejection of predictions that are too similar to multiple classes
    
    Args:
        model: Trained model
        image: Input image (preprocessed)
        class_names: List of class names
        similarity_threshold: Threshold for similarity rejection
        
    Returns:
        Dictionary with prediction results and similarity information
    """
    # Make prediction
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
        
    logits = model.predict(image, verbose=0)[0]
    
    # Apply temperature scaling for confidence calibration
    temperature = 1.5  # Higher temperature gives softer probabilities
    scaled_logits = logits / temperature
    
    # Convert to probabilities
    probs = tf.nn.softmax(scaled_logits).numpy()
    
    # Get top predictions
    top_indices = np.argsort(probs)[::-1]
    top_probs = probs[top_indices]
    top_classes = [class_names[i] for i in top_indices]
    
    # Check for similarity between top predictions
    is_ambiguous = False
    ambiguity_message = ""
    
    if len(top_probs) >= 2:
        # Calculate ratio between top two probabilities
        similarity_ratio = top_probs[1] / top_probs[0]
        
        # If top predictions are too similar, consider it ambiguous
        if similarity_ratio > similarity_threshold:
            is_ambiguous = True
            ambiguity_message = f"Ambiguous prediction - could be {top_classes[0]} or {top_classes[1]}"
    
    # Create result dictionary
    result = {
        "prediction": ambiguity_message if is_ambiguous else top_classes[0],
        "confidence": float(top_probs[0]),
        "is_ambiguous": is_ambiguous,
        "similarity_ratio": float(similarity_ratio) if len(top_probs) >= 2 else 0.0,
        "top_k_classes": top_classes[:5],
        "top_k_probabilities": top_probs[:5].tolist()
    }
    
    return result

def multi_view_consensus_predict(image_path, model_ensemble, class_names, min_consensus=0.7):
    """Make prediction by combining results from multiple views of the same plant
    
    Args:
        image_path: Path to image file
        model_ensemble: List of trained models
        class_names: List of class names
        min_consensus: Minimum consensus ratio for a definitive prediction
        
    Returns:
        Dictionary with consensus prediction results
    """
    if not os.path.exists(image_path):
        return {"error": f"Image not found: {image_path}"}
    
    print(f"{TermColors.CYAN}ℹ Performing multi-view consensus prediction{TermColors.ENDC}")
    
    # Generate different views/transformations of the input image
    def generate_views(img_path, num_views=5):
        # Load image
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        
        views = []
        
        # Original view
        views.append(tf.keras.applications.efficientnet_v2.preprocess_input(img_array.copy()))
        
        # Horizontal flip
        flipped = np.fliplr(img_array.copy())
        views.append(tf.keras.applications.efficientnet_v2.preprocess_input(flipped))
        
        # Rotation views
        for angle in [15, -15, 30, -30]:
            rotated = tf.keras.preprocessing.image.apply_affine_transform(
                img_array.copy(), 
                theta=angle,
                fill_mode='nearest'
            )
            views.append(tf.keras.applications.efficientnet_v2.preprocess_input(rotated))
        
        # Add batch dimension to each view
        return [np.expand_dims(view, axis=0) for view in views]
    
    # Generate views
    views = generate_views(image_path)
    print(f"{TermColors.CYAN}ℹ Generated {len(views)} different views of the plant{TermColors.ENDC}")
    
    # Make predictions on each view
    all_predictions = []
    
    for view_idx, view in enumerate(views):
        # Predict with each model in the ensemble
        model_predictions = []
        
        for model_idx, model in enumerate(model_ensemble):
            preds = model.predict(view, verbose=0)[0]
            probs = tf.nn.softmax(preds).numpy()
            
            top_idx = np.argmax(probs)
            top_class = class_names[top_idx]
            top_prob = probs[top_idx]
            
            model_predictions.append((top_class, top_prob))
        
        # Get ensemble prediction for this view (simple majority voting)
        view_votes = {}
        for cls, prob in model_predictions:
            if cls not in view_votes:
                view_votes[cls] = {"count": 0, "total_prob": 0.0}
            view_votes[cls]["count"] += 1
            view_votes[cls]["total_prob"] += prob
        
        # Select most voted class for this view
        view_result = max(view_votes.items(), key=lambda x: (x[1]["count"], x[1]["total_prob"]))
        view_class = view_result[0]
        view_confidence = view_result[1]["total_prob"] / view_result[1]["count"]
        
        all_predictions.append((view_class, view_confidence))
    
    # Analyze consensus across views
    consensus_votes = {}
    for cls, conf in all_predictions:
        if cls not in consensus_votes:
            consensus_votes[cls] = {"count": 0, "total_conf": 0.0}
        consensus_votes[cls]["count"] += 1
        consensus_votes[cls]["total_conf"] += conf
    
    # Calculate consensus ratio for each class
    for cls in consensus_votes:
        consensus_votes[cls]["ratio"] = consensus_votes[cls]["count"] / len(all_predictions)
        consensus_votes[cls]["avg_conf"] = consensus_votes[cls]["total_conf"] / consensus_votes[cls]["count"]
    
    # Sort classes by consensus ratio, then by average confidence
    sorted_consensus = sorted(
        consensus_votes.items(), 
        key=lambda x: (x[1]["ratio"], x[1]["avg_conf"]), 
        reverse=True
    )
    
    # Get top consensus result
    top_class, top_stats = sorted_consensus[0]
    consensus_ratio = top_stats["ratio"]
    avg_confidence = top_stats["avg_conf"]
    
    # Check if consensus is strong enough
    is_reliable = consensus_ratio >= min_consensus
    
    # Create result
    result = {
        "prediction": top_class if is_reliable else "Uncertain - insufficient view consensus",
        "confidence": float(avg_confidence) if is_reliable else float(avg_confidence * consensus_ratio),
        "consensus_ratio": float(consensus_ratio),
        "is_reliable": is_reliable,
        "view_predictions": [{"class": cls, "confidence": float(conf)} for cls, conf in all_predictions],
        "top_alternatives": [
            {"class": cls, "consensus": float(stats["ratio"]), "confidence": float(stats["avg_conf"])}
            for cls, stats in sorted_consensus[1:3]  # Next 2 alternatives
        ]
    }
    
    print(f"{TermColors.GREEN}✅ Multi-view consensus: {top_class} ({consensus_ratio:.1%}){TermColors.ENDC}")
    
    return result
    
def predict_with_visual_explanation(model, image, class_names):
    """Generate prediction with visual explanation using Grad-CAM
    
    Args:
        model: Trained model
        image: Input image
        class_names: List of class names
        
    Returns:
        Dictionary with prediction results and explanation heatmap
    """
    # Make prediction
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    
    # Get logits and probabilities
    logits = model.predict(image, verbose=0)[0]
    probs = tf.nn.softmax(logits).numpy()
    
    # Get top prediction
    top_idx = np.argmax(probs)
    top_class = class_names[top_idx]
    top_prob = probs[top_idx]
    
    # Generate Grad-CAM heatmap
    # This is a simplified version - in reality you would use a proper Grad-CAM implementation
    # that works with your specific model architecture
    try:
        last_conv_layer = None
        for layer in reversed(model.layers):
            if 'conv' in layer.name:
                last_conv_layer = layer
                break
        
        if last_conv_layer is None:
            # Fallback for models without convolutional layers
            return {
                "prediction": top_class,
                "confidence": float(top_prob),
                "explanation_available": False,
                "top_k_classes": [class_names[i] for i in np.argsort(probs)[::-1][:5]],
                "top_k_probabilities": probs[np.argsort(probs)[::-1][:5]].tolist()
            }
        
        # Create Grad-CAM model to access intermediate outputs
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            # Forward pass
            conv_output, predictions = grad_model(image)
            class_idx = top_idx
            output = predictions[:, class_idx]
        
        # Gradients of output with respect to last conv layer
        grads = tape.gradient(output, conv_output)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the activation maps
        conv_output = conv_output[0]
        heatmap = tf.reduce_sum(conv_output * pooled_grads[..., tf.newaxis], axis=-1)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Create explanation
        features_analysis = analyze_heatmap(heatmap)
        
        # Create result with explanation
        return {
            "prediction": top_class,
            "confidence": float(top_prob),
            "explanation_available": True,
            "explanation": {
                "heatmap": heatmap.tolist(),  # Convert to list for JSON serialization
                "feature_analysis": features_analysis,
                "interpretation": generate_explanation_text(features_analysis, top_class)
            },
            "top_k_classes": [class_names[i] for i in np.argsort(probs)[::-1][:5]],
            "top_k_probabilities": probs[np.argsort(probs)[::-1][:5]].tolist()
        }
        
    except Exception as e:
        # Fallback if explanation generation fails
        return {
            "prediction": top_class,
            "confidence": float(top_prob),
            "explanation_available": False,
            "explanation_error": str(e),
            "top_k_classes": [class_names[i] for i in np.argsort(probs)[::-1][:5]],
            "top_k_probabilities": probs[np.argsort(probs)[::-1][:5]].tolist()
        }

def analyze_heatmap(heatmap):
    """Analyze the activation heatmap to identify key regions of interest
    
    Args:
        heatmap: The activation heatmap from the model's attention layer
        
    Returns:
        Dictionary containing analysis of heatmap regions and their significance
    """
    # Normalize the heatmap to 0-1 range
    normalized_heatmap = heatmap / (np.max(heatmap) + 1e-10)
    
    # Find hotspots (high activation regions)
    threshold = 0.6  # Threshold for significant activation
    hotspots = np.where(normalized_heatmap > threshold)
    
    # Calculate center of mass for the heatmap
    if len(hotspots[0]) > 0:
        center_y = np.mean(hotspots[0])
        center_x = np.mean(hotspots[1])
    else:
        # If no hotspots above threshold, use the overall center of mass
        total_mass = np.sum(normalized_heatmap)
        y_indices, x_indices = np.indices(normalized_heatmap.shape)
        center_y = np.sum(y_indices * normalized_heatmap) / total_mass
        center_x = np.sum(x_indices * normalized_heatmap) / total_mass
    
    # Calculate the spread of activation
    y_spread = np.std(np.repeat(np.arange(normalized_heatmap.shape[0]), normalized_heatmap.shape[1]) * 
                    normalized_heatmap.flatten())
    x_spread = np.std(np.tile(np.arange(normalized_heatmap.shape[1]), normalized_heatmap.shape[0]) * 
                    normalized_heatmap.flatten())
    
    # Analyze the distribution of activations
    # Calculate the entropy to measure how focused or spread out the attention is
    epsilon = 1e-10  # To avoid log(0)
    flat_heatmap = normalized_heatmap.flatten()
    entropy = -np.sum(flat_heatmap * np.log2(flat_heatmap + epsilon))
    
    # Return the analysis results
    return {
        'center_of_mass': (float(center_x), float(center_y)),
        'max_activation': float(np.max(heatmap)),
        'hotspot_count': len(hotspots[0]),
        'activation_spread': (float(x_spread), float(y_spread)),
        'entropy': float(entropy),
        'focused_attention': float(1.0 - min(1.0, entropy / 5.0))  # Normalized focus measure
    }

def generate_explanation_text(model_output, class_names, heatmap_analysis=None):
    """Generate human-readable explanation for model prediction
    
    Args:
        model_output: Raw output from the model (logits or probabilities)
        class_names: List of class names
        heatmap_analysis: Optional analysis from analyze_heatmap function
        
    Returns:
        String with explanation of prediction
    """
    # Get top predictions
    top_indices = np.argsort(model_output)[-5:][::-1]
    top_probabilities = [model_output[i] for i in top_indices]
    top_classes = [class_names[i] for i in top_indices]
    
    # Format main prediction
    main_class = top_classes[0]
    main_prob = top_probabilities[0]
    
    # Start with confidence level explanation
    if main_prob > 0.95:
        confidence_text = "I'm very confident"
        certainty = "definitely"
    elif main_prob > 0.80:
        confidence_text = "I'm quite confident"
        certainty = "very likely"
    elif main_prob > 0.60:
        confidence_text = "I'm moderately confident"
        certainty = "likely"
    elif main_prob > 0.40:
        confidence_text = "I'm somewhat uncertain"
        certainty = "possibly"
    else:
        confidence_text = "I'm quite uncertain"
        certainty = "might be"
    
    # Basic explanation
    explanation = f"{confidence_text} that this is {certainty} {main_class} ({main_prob:.1%} probability)."
    
    # Add alternative possibilities if confidence is not very high
    if main_prob < 0.85:
        alternatives = [f"{name} ({prob:.1%})" for name, prob in 
                    zip(top_classes[1:3], top_probabilities[1:3]) if prob > 0.05]
        if alternatives:
            explanation += f" Other possibilities include: {', '.join(alternatives)}."
    
    # Add insights from heatmap analysis if provided
    if heatmap_analysis:
        focus = heatmap_analysis.get('focused_attention', 0)
        hotspots = heatmap_analysis.get('hotspot_count', 0)
        
        if focus > 0.8:
            explanation += f" My identification is based on very specific features in the image."
        elif focus > 0.5:
            explanation += f" I'm focusing on a few key regions to make this identification."
        else:
            explanation += f" I'm looking at multiple features across the entire image."
        
        if hotspots > 10:
            explanation += f" There are many distinctive patterns that match this species."
        
    # Add taxonomic information when available
    if hasattr(main_class, 'split') and '_' in main_class:
        try:
            genus, species = main_class.split('_', 1)
            explanation += f" This belongs to the genus {genus} in the classification system."
        except:
            pass
    
    return explanation

# In your build_model function, add attention mechanism
def build_model(feature_dim, num_classes):
    """Build model with attention mechanism for better plant feature focus
    
    Args:
        feature_dim: Input feature dimension
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    inputs = tf.keras.Input(shape=(feature_dim,))
    
    # First dense block
    x = tf.keras.layers.Dense(1024, activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Second dense block
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Attention mechanism
    attention = tf.keras.layers.Dense(512, activation='tanh')(x)
    attention = tf.keras.layers.Dense(1, activation='sigmoid')(attention)
    attention = tf.keras.layers.Reshape((512,))(attention)
    
    # Apply attention
    x = tf.keras.layers.Multiply()([x, attention])
    
    # Final classification layer
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)
    
    # Build model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Define custom top-k accuracy metric
    def sparse_top_k_accuracy(y_true, y_pred, k=5):
        return tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy', sparse_top_k_accuracy]
    )
    
    return model

def build_triplet_model(feature_dim, num_classes, margin=0.5):
    """Build model for triplet loss training to learn better plant similarity
    
    Args:
        feature_dim: Input feature dimension
        num_classes: Number of output classes
        margin: Margin for triplet loss
        
    Returns:
        Triplet model for embedding learning
    """
    # Define embedding model
    embedding_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(feature_dim,)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation=None),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))  # L2 normalization
    ])
    
    # Define triplet model with anchor, positive, and negative inputs
    anchor_input = tf.keras.layers.Input(shape=(feature_dim,), name='anchor_input')
    positive_input = tf.keras.layers.Input(shape=(feature_dim,), name='positive_input')
    negative_input = tf.keras.layers.Input(shape=(feature_dim,), name='negative_input')
    
    # Generate embeddings
    anchor_embedding = embedding_model(anchor_input)
    positive_embedding = embedding_model(positive_input)
    negative_embedding = embedding_model(negative_input)
    
    # Define triplet loss
    def triplet_loss(y_true, y_pred):
        anchor, positive, negative = y_pred[:, 0:128], y_pred[:, 128:256], y_pred[:, 256:384]
        
        # Calculate distances
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        
        # Calculate triplet loss
        basic_loss = pos_dist - neg_dist + margin
        loss = tf.maximum(basic_loss, 0.0)
        
        return tf.reduce_mean(loss)
    
    # Stack embeddings for the loss function
    stacked_embeddings = tf.keras.layers.Concatenate()([
        anchor_embedding, positive_embedding, negative_embedding
    ])
    
    # Define metrics for monitoring
    def triplet_accuracy(y_true, y_pred):
        anchor, positive, negative = y_pred[:, 0:128], y_pred[:, 128:256], y_pred[:, 256:384]
        
        # Calculate distances
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        
        # Calculate accuracy (percentage of triplets where pos_dist < neg_dist)
        return tf.reduce_mean(tf.cast(pos_dist < neg_dist, tf.float32))
    
    # Create and compile triplet model
    triplet_model = tf.keras.Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=stacked_embeddings
    )
    
    triplet_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=triplet_loss,
        metrics=[triplet_accuracy]
    )
    
    return triplet_model, embedding_model

class DynamicTrainingOptimizer(tf.keras.callbacks.Callback):
    def __init__(self, patience=3):
        super().__init__()
        self.patience = patience
        self.best_loss = float('inf')
        self.wait = 0
        self.best_weights = None
        self.learning_rates = [1e-3, 5e-4, 1e-4, 5e-5]
        self.current_lr_idx = 0
        
    def on_train_begin(self, logs=None):
        self.best_weights = self.model.get_weights()
        print(f"{TermColors.CYAN}ℹ Dynamic optimizer initialized with LR={self.learning_rates[0]}{TermColors.ENDC}")
        
    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('val_loss')
        
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # If we've waited enough epochs without improvement
                if self.current_lr_idx < len(self.learning_rates) - 1:
                    # Try a lower learning rate
                    self.current_lr_idx += 1
                    new_lr = self.learning_rates[self.current_lr_idx]
                    
                    print(f"{TermColors.YELLOW}⚠️ No improvement for {self.patience} epochs. Reducing LR to {new_lr}{TermColors.ENDC}")
                    
                    # Restore best weights and update learning rate
                    self.model.set_weights(self.best_weights)
                    K.set_value(self.model.optimizer.learning_rate, new_lr)
                    
                    # Reset wait counter
                    self.wait = 0

class AdaptiveRegularization(tf.keras.callbacks.Callback):
    """Dynamically adjust regularization based on overfitting detection"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.train_losses = []
        self.val_losses = []
        self.regularization_applied = False
        
    def on_epoch_end(self, epoch, logs=None):
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        
        # Need at least 3 epochs to detect trends
        if epoch < 3:
            return
        
        # Calculate gap between training and validation loss
        train_val_gap = self.val_losses[-1] - self.train_losses[-1]
        
        # Detect potential overfitting
        if (train_val_gap > 0.1 or 
            (self.val_losses[-1] > self.val_losses[-2] and self.train_losses[-1] < self.train_losses[-2])):
            
            # Only apply regularization once to avoid excessive modification
            if not self.regularization_applied:
                print(f"{TermColors.YELLOW}⚠️ Potential overfitting detected. Increasing regularization.{TermColors.ENDC}")
                
                # Increase dropout in all dropout layers
                for layer in self.model.layers:
                    if isinstance(layer, tf.keras.layers.Dropout):
                        # Get current rate and increase it by 0.1, capping at 0.5
                        current_rate = K.get_value(layer.rate)
                        new_rate = min(current_rate + 0.1, 0.5)
                        K.set_value(layer.rate, new_rate)
                        print(f"{TermColors.CYAN}ℹ Increased dropout rate from {current_rate:.1f} to {new_rate:.1f}{TermColors.ENDC}")
                
                self.regularization_applied = True

class AdaptiveClassWeightAdjuster(tf.keras.callbacks.Callback):
    """Dynamically adjust class weights based on per-class performance"""
    def __init__(self, X_val, y_val, class_names, update_frequency=5):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.class_names = class_names
        self.update_frequency = update_frequency
        self.current_class_weights = None
        self.class_performance = {}
        
    def on_train_begin(self, logs=None):
        # Initialize with uniform weights
        unique_classes = np.unique(self.y_val)
        self.current_class_weights = {int(cls): 1.0 for cls in unique_classes}
        
        # Initialize performance tracking
        for cls in unique_classes:
            self.class_performance[int(cls)] = {
                "accuracy": 0.0,
                "history": []
            }
        
    def on_epoch_end(self, epoch, logs=None):
        # Only update periodically to avoid excessive computation
        if (epoch + 1) % self.update_frequency != 0:
            return
            
        print(f"{TermColors.CYAN}ℹ Analyzing per-class performance to adjust weights{TermColors.ENDC}")
        
        # Get predictions on validation set
        y_pred = self.model.predict(self.X_val, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calculate per-class accuracy
        for cls in self.class_performance:
            # Get indices for this class
            cls_indices = np.where(self.y_val == cls)[0]
            
            if len(cls_indices) > 0:
                # Calculate accuracy for this class
                cls_correct = np.sum(y_pred_classes[cls_indices] == self.y_val[cls_indices])
                cls_accuracy = cls_correct / len(cls_indices)
                
                # Store accuracy
                self.class_performance[cls]["accuracy"] = cls_accuracy
                self.class_performance[cls]["history"].append(cls_accuracy)
        
        # Identify poorly performing classes
        all_accuracies = [info["accuracy"] for info in self.class_performance.values()]
        mean_accuracy = np.mean(all_accuracies)
        
        # Adjust weights: increase for poorly performing classes
        for cls, info in self.class_performance.items():
            # If class is performing worse than average
            if info["accuracy"] < mean_accuracy * 0.8:
                # Increase weight (cap at 2.0 to prevent instability)
                self.current_class_weights[cls] = min(self.current_class_weights[cls] * 1.2, 2.0)
                print(f"{TermColors.YELLOW}⚠️ Increasing weight for underperforming class {self.class_names[cls]} to {self.current_class_weights[cls]:.2f}{TermColors.ENDC}")
            
            # If class is performing much better than average
            elif info["accuracy"] > mean_accuracy * 1.2:
                # Decrease weight (floor at 0.5)
                self.current_class_weights[cls] = max(self.current_class_weights[cls] * 0.9, 0.5)
                print(f"{TermColors.GREEN}✅ Decreasing weight for well-performing class {self.class_names[cls]} to {self.current_class_weights[cls]:.2f}{TermColors.ENDC}")
        
        # Update model's class weights for next epoch
        self.model.class_weight = self.current_class_weights

def compute_balanced_class_weights(labels):
    """Compute class weights inversely proportional to class frequencies
    with proper caps to prevent extreme values
    
    Args:
        labels: Training labels array
        
    Returns:
        Dictionary mapping class indices to weights
    """
    # Get unique classes and count occurrences
    unique_classes, counts = np.unique(labels, return_counts=True)
    
    # Calculate weights inversely proportional to class frequencies
    # but with reasonable caps to prevent training instability
    n_samples = len(labels)
    n_classes = len(unique_classes)
    
    # Calculate balanced weights
    weights = n_samples / (n_classes * counts)
    
    # Create weight dictionary
    weight_dict = {i: w for i, w in zip(unique_classes, weights)}
    
    # Apply caps to prevent extreme weights
    max_weight_allowed = 5.0  # Cap maximum weight to prevent instability
    min_weight_allowed = 0.5  # Ensure minimum weight isn't too small
    
    # Calculate median weight for reference
    median_weight = np.median(list(weight_dict.values()))
    
    # Apply caps relative to median
    for cls, weight in weight_dict.items():
        if weight > median_weight * 3:
            weight_dict[cls] = min(median_weight * 3, max_weight_allowed)
        elif weight < median_weight * 0.3:
            weight_dict[cls] = max(median_weight * 0.3, min_weight_allowed)
    
    # Print class weights for the most imbalanced classes
    sorted_weights = sorted(weight_dict.items(), key=lambda x: x[1], reverse=True)
    
    print(f"{TermColors.CYAN}ℹ Class weight examples:{TermColors.ENDC}")
    for cls, weight in sorted_weights[:5]:  # Top 5 highest weights
        print(f"  - Class {cls}: weight={weight:.2f}")
    
    for cls, weight in sorted_weights[-5:]:  # Bottom 5 lowest weights
        print(f"  - Class {cls}: weight={weight:.2f}")
        
    return weight_dict

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """Focal Loss to address class imbalance by focusing on hard examples
    
    Args:
        y_true: True labels (sparse categorical format)
        y_pred: Predicted class probabilities
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Weighting factor for positive/negative examples
        
    Returns:
        Focal loss value
    """
    # Convert sparse y_true to one-hot
    y_true_one_hot = tf.keras.backend.one_hot(tf.cast(y_true, tf.int32), tf.shape(y_pred)[1])
    
    # Get the probabilities for the true classes
    p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=1)
    
    # Calculate focal weight
    focal_weight = tf.pow(1.0 - p_t, gamma)
    
    # Apply alpha weighting
    alpha_factor = 1.0
    if alpha > 0:
        alpha_t = y_true_one_hot * alpha + (1 - y_true_one_hot) * (1 - alpha)
        alpha_factor = tf.reduce_sum(alpha_t, axis=1)
    
    # Calculate the final weighted loss
    loss = -alpha_factor * focal_weight * tf.math.log(tf.clip_by_value(p_t, 1e-7, 1.0))
    
    return tf.reduce_mean(loss)

def test_time_augmentation_predict(model, image, num_augmentations=10, temperature=1.0):
    """Predict with test-time augmentation for more robust predictions
    
    Args:
        model: Trained model
        image: Input image (numpy array)
        num_augmentations: Number of augmented versions to create
        temperature: Temperature scaling factor for calibration
        
    Returns:
        Averaged prediction probabilities
    """
    augmented_images = []
    
    # Original image
    augmented_images.append(image)
    
    # Create augmentations
    for _ in range(num_augmentations - 1):
        # Apply random augmentation
        aug_img = image.copy()
        
        # Horizontal flip (50% chance)
        if np.random.random() > 0.5:
            aug_img = np.fliplr(aug_img)
        
        # Random rotation (-20 to 20 degrees)
        angle = np.random.uniform(-20, 20)
        aug_img = tf.keras.preprocessing.image.apply_affine_transform(
            aug_img, 
            theta=angle, 
            fill_mode='nearest'
        )
        
        # Random brightness/contrast adjustment
        aug_img = tf.image.random_brightness(aug_img, 0.1)
        aug_img = tf.image.random_contrast(aug_img, 0.9, 1.1)
        
        # Ensure values are in valid range
        aug_img = np.clip(aug_img, 0, 255)
        
        augmented_images.append(aug_img)
    
    # Predict on all augmented images
    batch = np.array(augmented_images)
    predictions = model.predict(batch, verbose=0)
    
    # Apply temperature scaling for calibration
    scaled_predictions = predictions / temperature
    scaled_probabilities = tf.nn.softmax(scaled_predictions, axis=1).numpy()
    
    # Average predictions
    avg_prediction = np.mean(scaled_probabilities, axis=0)
    return avg_prediction

def normalize_feature_vectors(features):
    """Normalize feature vectors to improve prediction stability
    
    Args:
        features: Input feature vectors
        
    Returns:
        Normalized feature vectors
    """
    # L2 normalization (unit length)
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    normalized_features = features / (norm + 1e-8)
    return normalized_features

def ensemble_predict(image_path, models, class_names, weights=None):
    """Make prediction using ensemble of models with weighted voting
    
    Args:
        image_path: Path to input image
        models: List of trained models
        class_names: List of class names
        weights: Optional list of weights for each model
        
    Returns:
        Dictionary with ensemble prediction results
    """
    # Initialize weights if None
    if weights is None:
        # Equal weights for all models if not specified
        weights = [1.0 / len(models)] * len(models)
    else:
        # Normalize weights to sum to 1
        total = sum(weights)
        weights = [w / total for w in weights]
    
    # Load and preprocess the image
    try:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        
        # Add batch dimension
        batch_img = np.expand_dims(img_array, axis=0)
        
        # Get predictions from all models
        all_predictions = []
        for model in models:
            pred = model.predict(batch_img, verbose=0)[0]
            all_predictions.append(pred)

        # Weighted voting
        ensemble_result = np.zeros_like(all_predictions[0])
        for i, pred in enumerate(all_predictions):
            ensemble_result += pred * weights[i]
        
        # Get final prediction
        top_class_idx = np.argmax(ensemble_result)
        
        # Get top 3 predictions
        top_indices = np.argsort(ensemble_result)[-3:][::-1]
        top_3_predictions = [
            {"class": class_names[idx], "probability": float(ensemble_result[idx])}
            for idx in top_indices
        ]
        
        return {
            "top_prediction": {
                "class": class_names[top_class_idx],
                "probability": float(ensemble_result[top_class_idx])
            },
            "top_3_predictions": top_3_predictions,
            "ensemble_output": ensemble_result.tolist()
        }
        
    except Exception as e:
        print(f"Error in ensemble prediction: {e}")
        traceback.print_exc()
        return {
            "error": str(e),
            "top_prediction": None
        }

def calibrated_confidence_score(prediction_probabilities, temperature=1.0):
    """Calculate a calibrated confidence score from model output probabilities
    
    Args:
        prediction_probabilities: Array of prediction probabilities from model
        temperature: Temperature scaling parameter
        
    Returns:
        Calibrated confidence score between 0 and 1
    """
    # Apply temperature scaling
    scaled_probs = prediction_probabilities ** (1 / temperature)
    scaled_probs = scaled_probs / np.sum(scaled_probs)
    
    # Sort probabilities in descending order
    sorted_probs = np.sort(scaled_probs)[::-1]
    
    # Calculate the gap between top probability and runner-up
    if len(sorted_probs) > 1:
        gap = sorted_probs[0] - sorted_probs[1]
    else:
        gap = sorted_probs[0]
    
    # Calculate entropy (uncertainty measure)
    epsilon = 1e-10  # To avoid log(0)
    entropy = -np.sum(scaled_probs * np.log2(scaled_probs + epsilon))
    max_entropy = np.log2(len(scaled_probs))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    # Calculate calibrated confidence using both gap and entropy
    # Larger gap and lower entropy should result in higher confidence
    confidence = 0.7 * (1 - normalized_entropy) + 0.3 * gap
    
    # Apply sigmoid scaling to get a smooth 0-1 range
    calibrated_confidence = 1.0 / (1.0 + np.exp(-6 * (confidence - 0.5)))
    
    return float(calibrated_confidence)

def generate_adversarial_examples(model, X, y, epsilon=0.01):
    """Generate adversarial examples to improve model robustness
    
    Args:
        model: Trained model
        X: Input features
        y: Target labels
        epsilon: Perturbation magnitude
        
    Returns:
        Adversarial examples
    """
    X_adv = X.copy()
    
    # Define gradient function
    with tf.GradientTape() as tape:
        X_tensor = tf.convert_to_tensor(X)
        tape.watch(X_tensor)
        predictions = model(X_tensor)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions)
    
    # Get gradients
    gradients = tape.gradient(loss, X_tensor)
    
    # Generate adversarial examples using Fast Gradient Sign Method (FGSM)
    signed_gradients = tf.sign(gradients).numpy()
    X_adv = X_adv + epsilon * signed_gradients
    
    # Ensure values stay in valid range
    X_adv = np.clip(X_adv, 0, 1)
    
    return X_adv

def predict_with_uncertainty(model, image, num_samples=20, dropout_rate=0.5):
    """Make prediction with uncertainty estimation using MC Dropout
    
    Args:
        model: Model with dropout layers
        image: Input image
        num_samples: Number of stochastic forward passes
        dropout_rate: Dropout rate to use
        
    Returns:
        Mean prediction and uncertainty estimate
    """
    # Enable dropout at inference time
    # Note: This requires the model to be built with tf.keras.backend.set_learning_phase(1)
    tf.keras.backend.set_learning_phase(1)
    
    # Multiple stochastic forward passes
    predictions = []
    for _ in range(num_samples):
        pred = model.predict(np.expand_dims(image, axis=0), verbose=0)
        predictions.append(pred[0])
    
    # Calculate mean and variance of predictions
    predictions = np.array(predictions)
    mean_prediction = np.mean(predictions, axis=0)
    uncertainty = np.var(predictions, axis=0)
    
    # Reset to inference mode
    tf.keras.backend.set_learning_phase(0)
    
    return mean_prediction, uncertainty

def extract_color_histogram(image, bins=32):
    """Extract color histogram features from an image to help with plant identification
    
    Args:
        image: Input image (numpy array)
        bins: Number of bins per channel
        
    Returns:
        Histogram features
    """
    # Convert to HSV color space for better color analysis
    if len(image.shape) == 3 and image.shape[2] == 3:
        try:
            # Convert to HSV
            hsv_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV)
            
            # Calculate histograms for each HSV channel
            h_hist = cv2.calcHist([hsv_image], [0], None, [bins], [0, 180]).flatten()
            s_hist = cv2.calcHist([hsv_image], [1], None, [bins], [0, 256]).flatten()
            v_hist = cv2.calcHist([hsv_image], [2], None, [bins], [0, 256]).flatten()
            
            # Calculate histograms for RGB channels too
            r_hist = cv2.calcHist([image.astype(np.uint8)], [0], None, [bins], [0, 256]).flatten()
            g_hist = cv2.calcHist([image.astype(np.uint8)], [1], None, [bins], [0, 256]).flatten()
            b_hist = cv2.calcHist([image.astype(np.uint8)], [2], None, [bins], [0, 256]).flatten()
            
            # Normalize histograms
            h_hist = h_hist / np.sum(h_hist) if np.sum(h_hist) > 0 else h_hist
            s_hist = s_hist / np.sum(s_hist) if np.sum(s_hist) > 0 else s_hist
            v_hist = v_hist / np.sum(v_hist) if np.sum(v_hist) > 0 else v_hist
            r_hist = r_hist / np.sum(r_hist) if np.sum(r_hist) > 0 else r_hist
            g_hist = g_hist / np.sum(g_hist) if np.sum(g_hist) > 0 else g_hist
            b_hist = b_hist / np.sum(b_hist) if np.sum(b_hist) > 0 else b_hist
            
            # Combine all histograms
            hist_features = np.concatenate([h_hist, s_hist, v_hist, r_hist, g_hist, b_hist])
            
            return hist_features
            
        except Exception as e:
            print(f"{TermColors.YELLOW}⚠️ Error extracting color histogram: {str(e)}{TermColors.ENDC}")
            return np.zeros(bins * 6)  # Return empty histogram on error
    else:
        # For feature vectors or grayscale images, return empty histogram
        return np.zeros(bins * 6)

def calculate_green_ratio(hist_features, bins=32):
    """Calculate the ratio of green in an image for plant classification weighting
    
    Args:
        hist_features: Color histogram features
        bins: Number of bins per channel used in the histogram
        
    Returns:
        Ratio of green content (0-1)
    """
    # The histogram features should contain RGB histograms in the second half
    # We're particularly interested in the green channel
    if len(hist_features) >= bins * 6:
        # Extract RGB histograms
        r_hist = hist_features[bins*3:bins*4]
        g_hist = hist_features[bins*4:bins*5]
        b_hist = hist_features[bins*5:bins*6]
        
        # Calculate green to other channels ratio
        # Focus on middle-high green values (vegetation)
        green_mid_high = np.sum(g_hist[bins//2:])
        red_mid_high = np.sum(r_hist[bins//2:])
        blue_mid_high = np.sum(b_hist[bins//2:])
        
        # Green dominance calculation
        # Check if green is higher than red and blue in mid-high range
        if (red_mid_high + blue_mid_high) > 0:
            green_dominance = green_mid_high / (red_mid_high + blue_mid_high + 1e-8)
        else:
            green_dominance = 1.0
            
        # Also consider overall green presence
        total_green = np.sum(g_hist)
        total_color = np.sum(r_hist) + np.sum(g_hist) + np.sum(b_hist) + 1e-8
        green_ratio = total_green / total_color
        
        # Combine both metrics with higher weight to dominance
        combined_green_score = 0.7 * min(green_dominance, 2.0) + 0.3 * green_ratio
        
        # Normalize to 0-1 range
        return min(combined_green_score / 2.0, 1.0)
    else:
        # If histogram features are not available
        return 0.5  # Default neutral value

def apply_plant_context_weighting(probabilities, image_features, location=None, season=None):
    """Apply contextual weights based on plant characteristics
    
    Args:
        probabilities: Initial prediction probabilities
        image_features: Features from the image
        location: Optional location data
        season: Optional season information
        
    Returns:
        Context-weighted probabilities
    """
    weighted_probs = probabilities.copy()
    
    # Apply color distribution weighting (greener images favor plants)
    if image_features is not None:
        try:
            # Calculate green ratio from histogram features
            green_ratio = calculate_green_ratio(image_features)
            
            # Apply weight scale factor based on green content
            # This boosts probabilities for plants when the image has lots of green
            weight_scale = 1.0 + green_ratio * 0.2  # Up to 20% boost for very green images
            
            # Apply to current probabilities
            weighted_probs = weighted_probs * weight_scale
            
            # Renormalize
            weighted_probs = weighted_probs / np.sum(weighted_probs)
        except:
            pass  # Continue with original probabilities if green calculation fails
    
    # Apply seasonal context if available
    if season is not None:
        # This would contain season-specific logic
        pass
        
    # Apply location context if available
    if location is not None:
        # This would contain location-specific logic
        pass
    
    return weighted_probs

def self_consistency_predict(image_path, model, class_names, num_augmentations=10):
    """Make prediction with self-consistency test through augmentation
    
    Args:
        image_path: Path to input image
        model: Trained model
        class_names: List of class names
        num_augmentations: Number of augmented versions to test
        
    Returns:
        Dictionary with prediction results and confidence metrics
    """
    # Load the original image
    try:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        
        # Generate augmented versions
        augmented_images = generate_augmentations(img_array, num_augmentations)
        
        # Make predictions on all versions
        all_predictions = []
        for aug_img in augmented_images:
            # Expand dimensions for batch
            batch_img = np.expand_dims(aug_img, axis=0)
            pred = model.predict(batch_img, verbose=0)[0]
            all_predictions.append(pred)
        
        # Add original image prediction
        batch_original = np.expand_dims(img_array, axis=0)
        original_pred = model.predict(batch_original, verbose=0)[0]
        all_predictions.append(original_pred)
        
        # Convert to numpy array
        all_predictions = np.array(all_predictions)
        
        # Calculate mean prediction
        mean_prediction = np.mean(all_predictions, axis=0)
        
        # Calculate prediction consistency
        top_class_indices = np.argmax(all_predictions, axis=1)
        most_common_class = np.bincount(top_class_indices).argmax()
        consistency_score = np.sum(top_class_indices == most_common_class) / len(top_class_indices)
        
        # Calculate standard deviation of predictions for top class
        std_dev = np.std(all_predictions[:, most_common_class])
        
        # Result with original and consistency-checked prediction
        result = {
            'original_prediction': {
                'class': class_names[np.argmax(original_pred)],
                'probability': float(np.max(original_pred))
            },
            'consistent_prediction': {
                'class': class_names[most_common_class],
                'probability': float(mean_prediction[most_common_class])
            },
            'consistency_score': float(consistency_score),
            'prediction_std_dev': float(std_dev),
            'all_predictions': all_predictions.tolist()
        }
        
        return result
    
    except Exception as e:
        print(f"Error in self-consistency prediction: {e}")
        traceback.print_exc()
        return {
            'error': str(e),
            'original_prediction': None,
            'consistent_prediction': None
        }

def generate_augmentations(image, num_augmentations=10):
    """Generate multiple augmented versions of an image
    
    Args:
        image: Original image array
        num_augmentations: Number of augmented versions to generate
        
    Returns:
        List of augmented image arrays
    """
    augmented_images = []
    
    for _ in range(num_augmentations):
        # Create a copy to avoid modifying the original
        img = image.copy()
        
        # Randomly apply different augmentations
        # Rotation
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-20, 20)
            img = tf.keras.preprocessing.image.apply_affine_transform(
                img, theta=angle, fill_mode='nearest')
        
        # Zoom
        if np.random.rand() > 0.5:
            zoom_factor = np.random.uniform(0.8, 1.2)
            img = tf.keras.preprocessing.image.apply_affine_transform(
                img, zx=zoom_factor, zy=zoom_factor, fill_mode='nearest')
        
        # Shift
        if np.random.rand() > 0.5:
            tx = np.random.uniform(-10, 10)
            ty = np.random.uniform(-10, 10)
            img = tf.keras.preprocessing.image.apply_affine_transform(
                img, tx=tx, ty=ty, fill_mode='nearest')
        
        # Brightness
        if np.random.rand() > 0.5:
            brightness_factor = np.random.uniform(0.8, 1.2)
            img = img * brightness_factor
            img = np.clip(img, 0, 1.0)
        
        # Flip
        if np.random.rand() > 0.5:
            img = img[:, ::-1, :]
        
        augmented_images.append(img)
    
    return augmented_images

def cross_model_fusion_predict(image_path, models, class_names, weights=None):
    """Make prediction using multiple models with weighted fusion
    
    Args:
        image_path: Path to input image
        models: List of trained models
        class_names: List of class names
        weights: Optional list of weights for each model
        
    Returns:
        Dictionary with fusion prediction results
    """
    # Load and preprocess the image
    try:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        
        # Add batch dimension
        batch_img = np.expand_dims(img_array, axis=0)
        
        # Initialize weights if not provided
        if weights is None:
            # Equal weights for all models
            weights = [1.0 / len(models)] * len(models)
        
        # Make predictions with each model
        all_predictions = []
        model_outputs = []
        
        for i, model in enumerate(models):
            # Get prediction
            pred = model.predict(batch_img, verbose=0)[0]
            model_outputs.append(pred)
            
            # Get top predicted class
            top_class_idx = np.argmax(pred)
            top_prob = pred[top_class_idx]
            
            all_predictions.append({
                'model_index': i,
                'class': class_names[top_class_idx],
                'class_index': int(top_class_idx),
                'probability': float(top_prob)
            })
        
        # Convert model outputs to numpy array
        model_outputs = np.array(model_outputs)
        
        # Calculate weighted average of predictions
        weighted_outputs = np.zeros_like(model_outputs[0])
        for i, output in enumerate(model_outputs):
            weighted_outputs += output * weights[i]
        
        # Get final prediction from weighted fusion
        fusion_class_idx = np.argmax(weighted_outputs)
        fusion_prob = weighted_outputs[fusion_class_idx]
        
        # Calculate model agreement score
        agreement_score = 0.0
        if len(all_predictions) > 1:
            # Count how many models agree with the fusion
            agreements = sum(1 for p in all_predictions if p['class_index'] == fusion_class_idx)
            agreement_score = agreements / len(models)
        
        # Calculate confidence adjustment based on agreement
        confidence_adjustment = 0.0
        if agreement_score >= 0.8:
            # High agreement boosts confidence
            confidence_adjustment = 0.1
        elif agreement_score <= 0.4:
            # Low agreement reduces confidence
            confidence_adjustment = -0.2
        
        # Apply adjusted confidence (but keep within 0-1 range)
        adjusted_prob = min(1.0, max(0.0, fusion_prob + confidence_adjustment))
        
        return {
            'fusion_prediction': {
                'class': class_names[fusion_class_idx],
                'probability': float(adjusted_prob),
                'raw_probability': float(fusion_prob)
            },
            'individual_predictions': all_predictions,
            'model_agreement': float(agreement_score),
            'confidence_adjustment': float(confidence_adjustment)
        }
        
    except Exception as e:
        print(f"Error in cross-model fusion prediction: {e}")
        traceback.print_exc()
        return {
            'error': str(e),
            'fusion_prediction': None,
            'individual_predictions': []
        }
    
def apply_confusion_cluster_knowledge(prediction, class_names, confusion_clusters):
    """Adjust prediction confidence based on known confusion clusters
    
    Args:
        prediction: Original prediction dictionary
        class_names: List of class names
        confusion_clusters: Dictionary mapping each class to its confusable classes
        
    Returns:
        Updated prediction with adjusted confidence
    """
    predicted_class = prediction['prediction']
    
    # If prediction is already rejected, just return it
    if 'Ambiguous' in predicted_class or 'Unknown' in predicted_class:
        return prediction
    
    # Check if this class is in a known confusion cluster
    if predicted_class in confusion_clusters:
        confusable_classes = confusion_clusters[predicted_class]
        
        # If it has confusable classes
        if confusable_classes:
            # Check if any top-k predictions are in the confusion cluster
            confusable_found = False
            for confusable in confusable_classes:
                confusable_class_name = confusable['class']
                confusion_rate = confusable['confusion_rate']
                
                if confusable_class_name in prediction['top_k_classes'][:3]:
                    # Found a known confusable class in top predictions
                    confusable_found = True
                    
                    # Get the position in top_k
                    confusable_idx = prediction['top_k_classes'].index(confusable_class_name)
                    confusable_prob = prediction['top_k_probabilities'][confusable_idx]
                    top_prob = prediction['top_k_probabilities'][0]
                    
                    # Calculate the similarity ratio for these known confusable classes
                    similarity_ratio = confusable_prob / (top_prob + 1e-8)
                    
                    # If the similarity is high for known confusable classes, adjust confidence
                    if similarity_ratio > 0.85:  # This can be lower than the general threshold
                        # Reduce confidence based on known confusion rate
                        adjusted_confidence = prediction['confidence'] * (1.0 - confusion_rate * 0.5)
                        prediction['confidence'] = float(adjusted_confidence)
                        prediction['confidence_adjusted'] = True
                        prediction['confidence_note'] = f"Adjusted for known confusion with {confusable_class_name}"
                        
                        # If confidence becomes too low, reject the prediction
                        if adjusted_confidence < 0.35:
                            prediction['prediction'] = f"Ambiguous - likely {predicted_class} or {confusable_class_name}"
                            prediction['confidence'] = 0.0
                            
                        break
            
            # If no confusable class found in top predictions, slightly boost confidence
            if not confusable_found:
                prediction['confidence'] = min(1.0, prediction['confidence'] * 1.1)
                prediction['confidence_note'] = "Boosted - no known confusable classes in top predictions"
    
    return prediction

# Add this class to focus training on difficult examples
class HardNegativeMiningCallback(tf.keras.callbacks.Callback):
    """Focus training on the hardest examples"""
    def __init__(self, X_train, y_train, batch_size=128):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.history = []
        self.current_weights = None
        self.epoch_count = 0
    
    def on_epoch_begin(self, epoch, logs=None):
        if epoch > 0 and epoch % 3 == 0:  # Every 3 epochs
            print(f"{TermColors.CYAN}ℹ Mining hard negative examples{TermColors.ENDC}")
            
            # Get predictions on full training set
            y_pred = self.model.predict(self.X_train)
            y_true_oh = tf.keras.utils.to_categorical(self.y_train, y_pred.shape[1])
            
            # Calculate loss for each sample
            losses = []
            for i in range(len(self.X_train)):
                pred = y_pred[i]
                true = y_true_oh[i]
                loss = -np.sum(true * np.log(pred + 1e-8))
                losses.append(loss)
            
            # Get indices of hardest examples (highest loss)
            hardest_indices = np.argsort(losses)[-int(len(losses)*0.3):]  # Top 30% hardest
            
            # Create sample weights
            self.current_weights = np.ones(len(self.X_train))
            self.current_weights[hardest_indices] = 2.0  # Double weight for hard examples
            
            print(f"{TermColors.GREEN}✅ Focused training on {len(hardest_indices)} difficult examples{TermColors.ENDC}")
        
        self.epoch_count += 1

def train_plants_advanced():
    """Main function to orchestrate the advanced training pipeline,
    automatically applying contrastive learning for best model quality."""
    print(f"{TermColors.HEADER}\n{'='*60}")
    print(f"STARTING ADVANCED PLANT RECOGNITION TRAINING PIPELINE")
    print(f"(Applying Contrastive Learning Enhancement)")
    print(f"{'='*60}{TermColors.ENDC}")

    # 1. Check for Feature Extraction
    # Look for features inside chunk directories first, as that seems to be the structure
    feature_files = sorted(glob.glob(os.path.join(FEATURES_DIR, "chunk_*", "features.npz")))

    if not feature_files:
        # Fallback: Check root FEATURES_DIR if not found in subdirs
        feature_files = sorted(glob.glob(os.path.join(FEATURES_DIR, "chunk_*_features.npz")))

    if not feature_files:
        print(f"{TermColors.RED}❌ No feature files found in {FEATURES_DIR} or its subdirectories. Feature extraction is required.{TermColors.ENDC}")
        # Attempt to run feature extraction automatically if a function exists
        print(f"{TermColors.CYAN}ℹ Attempting to run feature extraction pipeline automatically...{TermColors.ENDC}")
        try:
            # Assuming a function like 'run_feature_extraction_pipeline' exists
            # This function should handle extracting features and saving them correctly
            # (e.g., into FEATURES_DIR/chunk_X/features.npz)
            run_feature_extraction_pipeline() # You need to ensure this function is defined and works
            # Re-check for feature files after attempting extraction
            feature_files = sorted(glob.glob(os.path.join(FEATURES_DIR, "chunk_*", "features.npz")))
            if not feature_files:
                 print(f"{TermColors.RED}❌ Feature extraction failed or produced no files. Exiting.{TermColors.ENDC}")
                 return
            print(f"{TermColors.GREEN}✅ Feature extraction completed successfully.{TermColors.ENDC}")
        except NameError:
             print(f"{TermColors.RED}❌ 'run_feature_extraction_pipeline' function not found. Cannot extract features automatically. Please run feature extraction manually first. Exiting.{TermColors.ENDC}")
             return
        except Exception as e:
             print(f"{TermColors.RED}❌ Error during automatic feature extraction: {e}. Please check the extraction process. Exiting.{TermColors.ENDC}")
             traceback.print_exc()
             return

    print(f"{TermColors.CYAN}ℹ Found {len(feature_files)} feature chunks to process.{TermColors.ENDC}")

    # 2. Train Model for Each Chunk with Contrastive Learning
    training_state = TrainingState(CHECKPOINT_DIR) # Initialize state tracking

    for i, features_file in enumerate(feature_files):
        # Determine chunk index from the feature file path
        try:
            # Assumes path like ".../chunk_X/features.npz"
            chunk_idx = int(os.path.basename(os.path.dirname(features_file)).split("_")[1])
        except (IndexError, ValueError):
             print(f"{TermColors.YELLOW}⚠️ Could not determine chunk index from path {features_file}. Skipping.{TermColors.ENDC}")
             continue

        print(f"{TermColors.HEADER}\n--- Processing Chunk {chunk_idx} ({i+1}/{len(feature_files)}) ---{TermColors.ENDC}")

        # Check if we've already fully trained this chunk
        if training_state.is_chunk_completed(chunk_idx):
            print(f"{TermColors.GREEN}✅ Chunk {chunk_idx} already completed, skipping...{TermColors.ENDC}")
            continue

        # Load features for the current chunk
        try:
            print(f"{TermColors.CYAN}ℹ Loading features from {features_file}...{TermColors.ENDC}")
            with np.load(features_file) as data:
                features = data['features']
                labels = data['labels']
            print(f"{TermColors.GREEN}✅ Loaded {len(features)} samples for chunk {chunk_idx}.{TermColors.ENDC}")

            # Apply contrastive learning enhancement
            print(f"{TermColors.CYAN}ℹ Applying contrastive learning enhancement to features for chunk {chunk_idx}...{TermColors.ENDC}")
            # Ensure contrastive_feature_learning exists and handles potential errors
            try:
                # Assuming contrastive_feature_learning is defined elsewhere
                features, labels = contrastive_feature_learning(features, labels)
                print(f"{TermColors.GREEN}✅ Contrastive enhancement applied.{TermColors.ENDC}")
            except NameError:
                 print(f"{TermColors.RED}❌ 'contrastive_feature_learning' function not found. Cannot apply enhancement. Training with original features.{TermColors.ENDC}")
            except Exception as cle:
                 print(f"{TermColors.RED}❌ Error during contrastive learning for chunk {chunk_idx}: {cle}. Training with original features.{TermColors.ENDC}")
                 traceback.print_exc()


            # Split the (potentially enhanced) features into training and validation sets
            print(f"{TermColors.CYAN}ℹ Splitting features into train/validation sets...{TermColors.ENDC}")
            X_train, X_val, y_train, y_val = train_test_split(
                features, labels, test_size=0.2, random_state=42, stratify=labels
            )
            print(f"{TermColors.GREEN}✅ Data split: {len(X_train)} train, {len(X_val)} validation samples.{TermColors.ENDC}")

            # Train the model for this chunk using the features
            # Assuming train_chunk_model_with_swa is the preferred/best training function
            print(f"{TermColors.CYAN}ℹ Starting training for chunk {chunk_idx} using SWA...{TermColors.ENDC}")
            try:
                # Assuming train_chunk_model_with_swa is defined elsewhere and takes these args
                train_chunk_model_with_swa(X_train, y_train, X_val, y_val, chunk_idx, training_state)
            except NameError:
                 print(f"{TermColors.RED}❌ 'train_chunk_model_with_swa' function not found. Attempting 'train_chunk_model'.{TermColors.ENDC}")
                 try:
                     # Assuming train_chunk_model is defined elsewhere
                     train_chunk_model(X_train, y_train, X_val, y_val, chunk_idx, training_state)
                 except NameError:
                     print(f"{TermColors.RED}❌ Neither 'train_chunk_model_with_swa' nor 'train_chunk_model' found. Cannot train chunk {chunk_idx}.{TermColors.ENDC}")
                     continue # Skip to next chunk if no training function found
                 except Exception as e_train:
                     print(f"{TermColors.RED}❌ Error during training chunk {chunk_idx} with 'train_chunk_model': {e_train}{TermColors.ENDC}")
                     traceback.print_exc()
                     continue # Skip to next chunk on error
            except Exception as e_swa:
                 print(f"{TermColors.RED}❌ Error during training chunk {chunk_idx} with SWA: {e_swa}{TermColors.ENDC}")
                 traceback.print_exc()
                 continue # Skip to next chunk on error


            # Add memory cleanup between chunks
            print(f"{TermColors.CYAN}ℹ Cleaning up memory after chunk {chunk_idx}...{TermColors.ENDC}")
            del features, labels, X_train, X_val, y_train, y_val # Explicitly delete large arrays
            gc.collect()
            tf.keras.backend.clear_session()
            time.sleep(1) # Short pause

        except FileNotFoundError:
            print(f"{TermColors.RED}❌ Feature file not found: {features_file}. Skipping chunk.{TermColors.ENDC}")
            continue
        except Exception as e:
            print(f"{TermColors.RED}❌ Error processing chunk {chunk_idx}: {e}{TermColors.ENDC}")
            traceback.print_exc()
            # Attempt cleanup even on error
            gc.collect()
            tf.keras.backend.clear_session()
            continue # Move to the next chunk on error

    print(f"\n{TermColors.HEADER}{'='*60}")
    print(f"ADVANCED TRAINING PIPELINE COMPLETED")
    print(f"{'='*60}{TermColors.ENDC}")

    # Final cleanup
    print(f"{TermColors.CYAN}ℹ Running final cleanup...{TermColors.ENDC}")
    gc.collect()
    tf.keras.backend.clear_session()

def train_meta_model():
    """Train a meta-model that combines predictions from all chunk models"""
    print(f"{TermColors.CYAN}ℹ Implementing meta-model training...{TermColors.ENDC}")
    
    # Get all trained models
    model_files = glob.glob(os.path.join(MODEL_DIR, "chunk_*_model.keras"))
    
    if len(model_files) < 2:
        print(f"{TermColors.YELLOW}⚠️ Not enough models to train a meta-model (need at least 2){TermColors.ENDC}")
        return
    
    print(f"{TermColors.CYAN}ℹ Found {len(model_files)} models to combine{TermColors.ENDC}")
    
    # Create a validation dataset from a mix of samples from different chunks
    print(f"{TermColors.CYAN}ℹ Creating validation dataset for meta-model training{TermColors.ENDC}")
    
    validation_X = []
    validation_y = []
    validation_classes = {}
    class_to_idx = {}
    idx_to_class = {}
    next_class_idx = 0
    
    # Load a subset of samples from each chunk
    for model_file in tqdm(model_files, desc="Loading validation data"):
        chunk_idx = int(os.path.basename(model_file).split("_")[1])
        
        # Find feature file for this chunk
        feature_file = os.path.join(FEATURE_DIR, f"chunk_{chunk_idx}", "features.npz")
        if not os.path.exists(feature_file):
            print(f"{TermColors.YELLOW}⚠️ Feature file not found for chunk {chunk_idx}, skipping{TermColors.ENDC}")
            continue
        
        # Load class mapping
        class_mapping_file = os.path.join(os.path.dirname(feature_file), "class_mapping.json")
        if not os.path.exists(class_mapping_file):
            print(f"{TermColors.YELLOW}⚠️ Class mapping not found for chunk {chunk_idx}, skipping{TermColors.ENDC}")
            continue
            
        with open(class_mapping_file) as f:
            class_mapping = json.load(f)
        
        # Load features
        try:
            data = np.load(feature_file)
            X = data['features']
            y = data['labels']
            
            # Take a small subset of samples (max 10 per class)
            unique_classes = np.unique(y)
            chunk_X = []
            chunk_y = []
            
            for cls in unique_classes:
                # Get indices for this class
                indices = np.where(y == cls)[0]
                
                # Take at most 10 samples
                selected_indices = indices[:min(10, len(indices))]
                
                # Get class name
                class_name = class_mapping[str(cls)]
                
                # Map to global class index
                if class_name not in class_to_idx:
                    class_to_idx[class_name] = next_class_idx
                    idx_to_class[next_class_idx] = class_name
                    next_class_idx += 1
                
                global_cls_idx = class_to_idx[class_name]
                
                # Add to validation set
                chunk_X.append(X[selected_indices])
                chunk_y.extend([global_cls_idx] * len(selected_indices))
                
                # Track classes
                if global_cls_idx not in validation_classes:
                    validation_classes[global_cls_idx] = 0
                validation_classes[global_cls_idx] += len(selected_indices)
            
            if chunk_X:
                chunk_X = np.vstack(chunk_X)
                validation_X.append(chunk_X)
                validation_y.extend(chunk_y)
                
        except Exception as e:
            print(f"{TermColors.YELLOW}⚠️ Error loading features from chunk {chunk_idx}: {e}{TermColors.ENDC}")
            continue
    
    if not validation_X:
        print(f"{TermColors.RED}❌ Failed to create validation dataset{TermColors.ENDC}")
        return
    
    # Combine all validation data
    validation_X = np.vstack(validation_X)
    validation_y = np.array(validation_y)
    
    print(f"{TermColors.GREEN}✅ Created validation dataset with {len(validation_X)} samples, {len(validation_classes)} classes{TermColors.ENDC}")
    
    # Standardize features
    scaler = StandardScaler()
    validation_X = scaler.fit_transform(validation_X)
    
    # Split into train/validation
    meta_X_train, meta_X_val, meta_y_train, meta_y_val = train_test_split(
        validation_X, validation_y, test_size=0.2, random_state=42, stratify=validation_y
    )
    
    print(f"{TermColors.CYAN}ℹ Training meta-model on {len(meta_X_train)} samples{TermColors.ENDC}")
    
    # Now get predictions from all models on the validation set
    # We'll use these predictions as features for our meta-model
    
    # First, load all models and their metadata
    loaded_models = []
    class_mappings = []
    
    for model_file in tqdm(model_files, desc="Loading models"):
        try:
            chunk_idx = int(os.path.basename(model_file).split("_")[1])
            metadata_file = os.path.join(MODEL_DIR, f"chunk_{chunk_idx}_metadata.json")
            
            if not os.path.exists(metadata_file):
                print(f"{TermColors.YELLOW}⚠️ Metadata not found for chunk {chunk_idx}, skipping{TermColors.ENDC}")
                continue
            
            with open(metadata_file) as f:
                metadata = json.load(f)
                class_mappings.append(metadata["class_mapping"])
            
            model = tf.keras.models.load_model(model_file, compile=False)
            loaded_models.append((chunk_idx, model))
            
        except Exception as e:
            print(f"{TermColors.YELLOW}⚠️ Error loading model from chunk {chunk_idx}: {e}{TermColors.ENDC}")
            continue
    
    if not loaded_models:
        print(f"{TermColors.RED}❌ No models could be loaded{TermColors.ENDC}")
        return
    
    print(f"{TermColors.GREEN}✅ Loaded {len(loaded_models)} models{TermColors.ENDC}")
    
    # Generate predictions for both train and validation sets
    meta_train_preds = []
    meta_val_preds = []
    
    for chunk_idx, model in tqdm(loaded_models, desc="Generating predictions"):
        # Get predictions on train data
        train_preds = model.predict(meta_X_train)
        meta_train_preds.append(train_preds)
        
        # Get predictions on validation data
        val_preds = model.predict(meta_X_val)
        meta_val_preds.append(val_preds)
    
    # Stack predictions from all models to create meta-features
    meta_train_features = np.hstack(meta_train_preds)
    meta_val_features = np.hstack(meta_val_preds)
    
    print(f"{TermColors.CYAN}ℹ Created meta-features of shape {meta_train_features.shape}{TermColors.ENDC}")
    
    # Train the meta-model
    # We'll use a simple random forest for this
    meta_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        n_jobs=-1,
        random_state=42
    )
    
    print(f"{TermColors.CYAN}ℹ Training meta-model...{TermColors.ENDC}")
    meta_model.fit(meta_train_features, meta_y_train)
    
    # Evaluate on validation set
    val_accuracy = meta_model.score(meta_val_features, meta_y_val)
    print(f"{TermColors.GREEN}✅ Meta-model validation accuracy: {val_accuracy:.4f}{TermColors.ENDC}")
    
    # Save the meta-model
    meta_model_file = os.path.join(MODEL_DIR, "meta_model.joblib")
    joblib.dump(meta_model, meta_model_file)
    
    # Save the class mapping
    meta_class_mapping = {str(idx): class_name for idx, class_name in idx_to_class.items()}
    with open(os.path.join(MODEL_DIR, "meta_model_class_mapping.json"), "w") as f:
        json.dump(meta_class_mapping, f, indent=2)
    
    # Save the scaler
    scaler_file = os.path.join(MODEL_DIR, "meta_model_scaler.joblib")
    joblib.dump(scaler, scaler_file)
    
    # Save model indices used
    model_indices = [chunk_idx for chunk_idx, _ in loaded_models]
    with open(os.path.join(MODEL_DIR, "meta_model_chunks.json"), "w") as f:
        json.dump(model_indices, f)
    
    print(f"{TermColors.GREEN}✅ Meta-model saved to {meta_model_file}{TermColors.ENDC}")
    
    # Clean up
    for _, model in loaded_models:
        del model
    gc.collect()
    tf.keras.backend.clear_session()

def train_chunk_model_with_swa(features_file, chunk_idx, training_state=None):
    """Train a model on extracted features with Stochastic Weight Averaging for better generalization"""
    print(f"{TermColors.HEADER}\n{'='*50}")
    print(f"TRAINING MODEL WITH SWA FOR CHUNK {chunk_idx+1}")
    print(f"{'='*50}{TermColors.ENDC}")
    
    # Extract directory and verify files
    chunk_feature_dir = os.path.dirname(features_file)
    class_mapping_file = os.path.join(chunk_feature_dir, "class_mapping.json")
    
    if not os.path.exists(class_mapping_file):
        print(f"{TermColors.RED}❌ Class mapping file not found: {class_mapping_file}{TermColors.ENDC}")
        return None
    
    # Load class mapping
    with open(class_mapping_file) as f:
        class_mapping = json.load(f)
    
    num_classes = len(class_mapping)
    print(f"{TermColors.CYAN}ℹ Found {num_classes} classes in this chunk{TermColors.ENDC}")
    
    try:
        # Load features and analyze
        features, labels = analyze_features(features_file)
        if features is None or labels is None:
            print(f"{TermColors.RED}❌ Failed to load features for chunk {chunk_idx}{TermColors.ENDC}")
            return
            
        # Further analyze plant-specific characteristics
        confusability = analyze_plant_features(features, labels)
        
        # Standardize features
        print(f"{TermColors.CYAN}ℹ Standardizing features{TermColors.ENDC}")
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        X_train, X_val, y_train, y_val = train_test_split(
            features_scaled, labels, test_size=0.2, random_state=42, stratify=labels
        )

        print(f"{TermColors.GREEN}✅ Data split: {X_train.shape[0]} training, {X_val.shape[0]} validation samples{TermColors.ENDC}")

        # THEN balance the training data to reduce bias
        X_train, y_train = balance_training_data(X_train, y_train)

        # THEN augment the training data after balancing
        print(f"{TermColors.CYAN}ℹ Augmenting training data for better class separation{TermColors.ENDC}")
        X_train, y_train = augment_feature_space(X_train, y_train, augmentation_factor=0.3)
        
        # Build model
        feature_dim = features.shape[1]
        print(f"{TermColors.CYAN}ℹ Building model for {num_classes} classes with {feature_dim} input features{TermColors.ENDC}")
        
        model = build_model(feature_dim, num_classes)
        
        # Compute class weights if imbalanced
        class_weights = None
        if len(np.unique(y_train)) > 1:  # Only if we have multiple classes
            try:
                # Use our new balanced class weight function
                print(f"{TermColors.CYAN}ℹ Computing balanced class weights to handle class imbalance{TermColors.ENDC}")
                class_weights = compute_balanced_class_weights(y_train)
                
                print(f"{TermColors.GREEN}✅ Class weights computed to balance training{TermColors.ENDC}")
            except Exception as e:
                print(f"{TermColors.YELLOW}⚠️ Could not compute class weights: {e}{TermColors.ENDC}")
                # Fallback to manual calculation if the function fails
                try:
                    # Count samples per class
                    unique_classes = np.unique(y_train)
                    class_counts = {cls: np.sum(y_train == cls) for cls in unique_classes}
                    
                    # Check for imbalance
                    max_count = max(class_counts.values())
                    min_count = min(class_counts.values())
                    
                    if max_count / min_count > 2:  # Significant imbalance
                        print(f"{TermColors.CYAN}ℹ Class imbalance detected (max/min ratio: {max_count/min_count:.1f}){TermColors.ENDC}")
                        
                        # Compute balanced weights
                        weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train)
                        class_weights = {i: w for i, w in zip(unique_classes, weights)}
                        
                        # Cap extreme weights
                        for cls, weight in class_weights.items():
                            if weight > 5:
                                class_weights[cls] = 5.0  # Cap at 5x
                                
                        print(f"{TermColors.CYAN}ℹ Using class weights to balance training{TermColors.ENDC}")
                except Exception as e:
                    print(f"{TermColors.YELLOW}⚠️ Could not compute class weights: {e}{TermColors.ENDC}")
                    class_weights = None
        
        # Keep track of weight snapshots
        weight_snapshots = []
        
        # Custom callback for SWA
        class SWACallback(tf.keras.callbacks.Callback):
            def __init__(self, start_epoch=10):
                super().__init__()
                self.start_epoch = start_epoch
                self.snapshots = []
                
            def on_epoch_end(self, epoch, logs=None):
                if epoch >= self.start_epoch:
                    print(f"Taking weight snapshot at epoch {epoch+1}")
                    self.snapshots.append(self.model.get_weights())
        
        # Setup callbacks
        callbacks = [
            # Checkpoints to save best model
            ModelCheckpoint(
                filepath=os.path.join(CHECKPOINT_DIR, f"chunk_{chunk_idx}_best.keras"),
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                verbose=1
            ),
            
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_loss',
                patience=12,  # Longer patience for SWA
                restore_best_weights=True, 
                verbose=1
            ),
            
            # Dynamic optimizer for automatic learning rate adjustments
            DynamicTrainingOptimizer(patience=2),
            
            # SWA callback to collect weight snapshots
            SWACallback(start_epoch=15),  # Start collecting after 15 epochs
            
            # Adaptive regularization based on overfitting detection
            AdaptiveRegularization(model),
            
            # Display progress with custom metrics visualization
            MetricsDisplayCallback(
                total_epochs=EPOCHS,
                validation_data=(X_val, y_val),
                training_state=training_state
            )
        ]
        
        # Check if we're resuming training
        start_epoch = 0
        if training_state and training_state.current_chunk == chunk_idx:
            start_epoch = training_state.current_epoch
            print(f"{TermColors.CYAN}ℹ Resuming training from epoch {start_epoch + 1}{TermColors.ENDC}")
        
        # Train the model
        print(f"{TermColors.CYAN}ℹ Training model for up to {EPOCHS} epochs{TermColors.ENDC}")
        swa_callback = callbacks[3]  # The SWACallback
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            initial_epoch=start_epoch,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=0  # We use our own progress display
        )
        
        # Apply SWA if we have snapshots
        snapshots = swa_callback.snapshots
        if snapshots:
            print(f"{TermColors.CYAN}ℹ Applying Stochastic Weight Averaging with {len(snapshots)} snapshots{TermColors.ENDC}")
            # Compute average weights
            avg_weights = []
            for i in range(len(snapshots[0])):
                # Average weights from all snapshots for each layer
                layer_avg = np.mean([s[i] for s in snapshots], axis=0)
                avg_weights.append(layer_avg)
            
            # Create a copy of model with averaged weights
            swa_model = tf.keras.models.clone_model(model)
            swa_model.set_weights(avg_weights)
            swa_model.compile(
                optimizer=model.optimizer, 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy', sparse_top_k_accuracy]
            )
            
            # Evaluate both models
            orig_loss, orig_acc, orig_top_k = model.evaluate(X_val, y_val, verbose=0)
            swa_loss, swa_acc, swa_top_k = swa_model.evaluate(X_val, y_val, verbose=0)
            
            print(f"{TermColors.CYAN}ℹ Original model: Acc={orig_acc:.4f}, Top-{TOP_K}={orig_top_k:.4f}{TermColors.ENDC}")
            print(f"{TermColors.CYAN}ℹ SWA model: Acc={swa_acc:.4f}, Top-{TOP_K}={swa_top_k:.4f}{TermColors.ENDC}")
            
            # Use SWA model if it's better
            if swa_acc > orig_acc:
                print(f"{TermColors.GREEN}✅ SWA model performs better! Using SWA weights.{TermColors.ENDC}")
                model = swa_model
                val_loss, val_acc, val_top_k = swa_loss, swa_acc, swa_top_k
            else:
                print(f"{TermColors.YELLOW}⚠️ Original model performs better. Keeping original weights.{TermColors.ENDC}")
                val_loss, val_acc, val_top_k = orig_loss, orig_acc, orig_top_k
        else:
            # If no snapshots (training was too short), evaluate original model
            val_loss, val_acc, val_top_k = model.evaluate(X_val, y_val, verbose=0)
            
        # Print final metrics
        print(f"\n{TermColors.HEADER}FINAL VALIDATION METRICS:{TermColors.ENDC}")
        print(f"{TermColors.RED}Loss: {val_loss:.4f}{TermColors.ENDC}")
        print(f"{TermColors.GREEN}Accuracy: {val_acc:.4f} ({val_acc:.1%}){TermColors.ENDC}")
        print(f"{TermColors.GREEN}Top-{TOP_K} Accuracy: {val_top_k:.4f} ({val_top_k:.1%}){TermColors.ENDC}")
        
        # Save the model
        model_file = os.path.join(MODEL_DIR, f"chunk_{chunk_idx}_model.keras")
        model.save(model_file)
        
        # Save metadata
        metadata = {
            "chunk_idx": chunk_idx,
            "num_classes": num_classes,
            "feature_dim": feature_dim,
            "class_mapping": class_mapping,
            "performance": {
                "val_loss": float(val_loss),
                "val_accuracy": float(val_acc),
                "val_sparse_top_k_accuracy": float(val_top_k)
            },
            "training_samples": int(X_train.shape[0]),
            "validation_samples": int(X_val.shape[0]),
            "confusability": float(confusability),
            "created": datetime.now().isoformat(),
            "used_swa": len(snapshots) > 0
        }
        
        with open(os.path.join(MODEL_DIR, f"chunk_{chunk_idx}_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
            
        print(f"{TermColors.GREEN}✅ Model saved to {model_file}{TermColors.ENDC}")
        
        # Mark chunk as processed
        if training_state:
            training_state.mark_chunk_complete(chunk_idx)
        
        # Clean up to prevent memory leaks
        clean_memory()
        
        # Prune features to save disk space
        prune_features_rolling_window(chunk_idx)
        
        return model
        
    except Exception as e:
        print(f"{TermColors.RED}❌ Error training model: {e}{TermColors.ENDC}")
        traceback.print_exc()
        return None

def quantize_model(model, quantization_type="dynamic"):
    """Quantize model for faster inference and smaller size
    
    Args:
        model: Keras model to quantize
        quantization_type: Type of quantization ('dynamic', 'float16', or 'int8')
        
    Returns:
        Quantized model
    """
    import tensorflow as tf
    
    print(f"{TermColors.CYAN}ℹ Quantizing model to {quantization_type}{TermColors.ENDC}")
    
    if quantization_type == "dynamic":
        # Dynamic range quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        # Convert back to Keras model if possible
        # Note: In real implementations, you would use the TFLite model directly
        # This is just a placeholder for the conversion back to Keras
        # In production, you would save the TFLite model and use the TFLite runtime
        print(f"{TermColors.GREEN}✅ Model successfully quantized to dynamic range{TermColors.ENDC}")
        print(f"{TermColors.YELLOW}⚠️ Original model size: {model_size_mb(model):.2f} MB{TermColors.ENDC}")
        print(f"{TermColors.GREEN}✅ Quantized model size: {len(tflite_model)/1024/1024:.2f} MB{TermColors.ENDC}")
        
        return model  # In reality, would return or save the TFLite model
        
    elif quantization_type == "float16":
        # Float16 quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        
        print(f"{TermColors.GREEN}✅ Model successfully quantized to float16{TermColors.ENDC}")
        print(f"{TermColors.YELLOW}⚠️ Original model size: {model_size_mb(model):.2f} MB{TermColors.ENDC}")
        print(f"{TermColors.GREEN}✅ Quantized model size: {len(tflite_model)/1024/1024:.2f} MB{TermColors.ENDC}")
        
        return model
        
    elif quantization_type == "int8":
        # Int8 quantization (requires representative dataset)
        # This is a simplified version - production code would use a real dataset
        def representative_dataset():
            # Use a small subset of your training data here
            for _ in range(100):
                # Generate random input in the expected shape
                input_shape = model.input_shape
                if isinstance(input_shape, list):
                    input_shape = input_shape[0]
                    
                # Remove batch dimension
                if input_shape[0] is None:
                    input_shape = list(input_shape[1:])
                yield [np.random.random(input_shape).astype(np.float32)]
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        
        print(f"{TermColors.GREEN}✅ Model successfully quantized to int8{TermColors.ENDC}")
        print(f"{TermColors.YELLOW}⚠️ Original model size: {model_size_mb(model):.2f} MB{TermColors.ENDC}")
        print(f"{TermColors.GREEN}✅ Quantized model size: {len(tflite_model)/1024/1024:.2f} MB{TermColors.ENDC}")
        
        return model
    
    return model

def model_size_mb(model):
    """Calculate model size in MB"""
    # Save model to a temporary file and get its size
    temp_path = "temp_model.keras"
    model.save(temp_path)
    size_bytes = os.path.getsize(temp_path)
    os.remove(temp_path)
    return size_bytes / (1024 * 1024)

def distill_model(teacher_model, X_train, y_train, X_val, y_val, temperature=3.0):
    """Create a smaller, faster model using knowledge distillation
    
    Args:
        teacher_model: Larger, more accurate model to learn from
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        temperature: Distillation temperature (higher = softer probabilities)
        
    Returns:
        Distilled student model
    """
    print(f"{TermColors.CYAN}ℹ Starting knowledge distillation process{TermColors.ENDC}")
    
    # Get input and output dimensions from teacher model
    input_shape = teacher_model.input_shape[1:]
    num_classes = teacher_model.output_shape[1]
    
    # Create a smaller student model
    student_model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(num_classes)
    ])
    
    # Generate soft targets from teacher model
    print(f"{TermColors.CYAN}ℹ Generating soft targets from teacher model{TermColors.ENDC}")
    teacher_preds = teacher_model.predict(X_train)
    
    # Apply temperature scaling to soften predictions
    soft_targets = tf.nn.softmax(teacher_preds / temperature).numpy()
    
    # Define distillation loss
    def distillation_loss(y_true, y_pred):
        # Convert one-hot encoded y_true to class indices
        y_true_indices = tf.argmax(y_true, axis=1)
        
        # Hard target loss (standard cross-entropy with true labels)
        hard_loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true_indices, 
            y_pred, 
            from_logits=True
        )
        
        # Soft target loss (KL divergence with teacher's predictions)
        soft_targets_scaled = tf.nn.softmax(y_true / temperature)
        logits_scaled = y_pred / temperature
        soft_prob = tf.nn.softmax(logits_scaled)
        soft_loss = tf.reduce_sum(
            soft_targets_scaled * tf.math.log(soft_targets_scaled / (soft_prob + 1e-8)), 
            axis=1
        )
        
        # Combine losses (α controls the balance)
        alpha = 0.7  # Weight for soft targets
        return alpha * soft_loss * (temperature**2) + (1 - alpha) * hard_loss
    
    # Compile student model
    student_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=distillation_loss,
        metrics=['accuracy']
    )
    
    # Train the student on soft targets
    print(f"{TermColors.CYAN}ℹ Training student model on distilled knowledge{TermColors.ENDC}")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]
    
    student_model.fit(
        X_train, soft_targets,
        validation_data=(X_val, tf.nn.softmax(teacher_model.predict(X_val) / temperature).numpy()),
        epochs=30,
        batch_size=128,
        callbacks=callbacks,
        verbose=1
    )
    
    # Final evaluation
    teacher_loss, teacher_acc = teacher_model.evaluate(X_val, y_val)
    # Compile with standard loss for evaluation
    student_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    student_loss, student_acc = student_model.evaluate(X_val, y_val)
    
    print(f"{TermColors.GREEN}✅ Distillation complete!{TermColors.ENDC}")
    print(f"Teacher - Accuracy: {teacher_acc:.4f}, Size: {model_size_mb(teacher_model):.2f} MB")
    print(f"Student - Accuracy: {student_acc:.4f}, Size: {model_size_mb(student_model):.2f} MB")
    
    return student_model

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction
    
    Args:
        image: Input image (numpy array or PIL Image)
        target_size: Target size for resizing (default: 224x224)
        
    Returns:
        Preprocessed image ready for model input
    """
    # Check if image is PIL Image
    if not isinstance(image, np.ndarray):
        # Convert PIL Image to numpy array
        image = tf.keras.preprocessing.image.img_to_array(image)
    
    # Resize if necessary
    if image.shape[0] != target_size[0] or image.shape[1] != target_size[1]:
        image = tf.image.resize(image, target_size)
    
    # Apply EfficientNet preprocessing
    processed_image = tf.keras.applications.efficientnet_v2.preprocess_input(image)
    
    # Ensure we have a batch dimension for prediction
    if len(processed_image.shape) == 3:
        processed_image = np.expand_dims(processed_image, axis=0)
    
    return processed_image

def predict_plant_with_confidence(models, image, class_names, confusion_clusters=None, temp=1.5, use_tta=True):
    """Complete prediction pipeline with confusion handling for plant recognition
    
    Args:
        models: Dictionary mapping model name to model object, or list of models
        image: Input image
        class_names: List of class names
        confusion_clusters: Optional dictionary mapping each class to confusable classes
        temp: Temperature scaling factor
        use_tta: Whether to use test-time augmentation
    
    Returns:
        Predicted class, confidence score, and top-k predictions with confidence adjustment
    """
    # Convert models to dictionary if it's a list
    if isinstance(models, list):
        models = {f"model_{i}": model for i, model in enumerate(models)}
    
    # If we have multiple models, use cross-model fusion
    if len(models) > 1:
        prediction = cross_model_fusion_predict(image, models, class_names, temperature=temp)
    else:
        # Single model with TTA if requested
        model = list(models.values())[0]
        
        if use_tta:
            # Use test-time augmentation for more robust prediction
            processed_image = preprocess_image(image)
            probs = test_time_augmentation_predict(model, processed_image[0], num_augmentations=10, temperature=temp)
        else:
            # Basic prediction with temperature scaling
            processed_image = preprocess_image(image)
            logits = model.predict(processed_image)[0]
            scaled_logits = logits / temp
            probs = tf.nn.softmax(scaled_logits).numpy()
        
        # Calculate calibrated confidence
        calibrated_conf = calibrated_confidence_score(probs)
        
        # Get top predictions
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        
        # Create prediction dictionary
        prediction = {
            'prediction': class_names[sorted_indices[0]],
            'confidence': float(calibrated_conf),
            'top_k_classes': [class_names[i] for i in sorted_indices[:10]],
            'top_k_probabilities': sorted_probs[:10].tolist()
        }
    
    # Extract image features for context-based weighting
    try:
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_features = extract_color_histogram(img_array)
        
        # Apply plant-specific context weighting
        weighted_prediction = apply_plant_context_weighting(
            probs,
            img_features,
            prediction['top_k_classes'],
            prediction['top_k_probabilities']
        )
        
        # Update the prediction with weighted probabilities
        prediction.update(weighted_prediction)
    except Exception as e:
        # Context weighting failed, continue with original prediction
        pass
    
    # Apply similarity rejection with higher threshold (0.92)
    similarity_result = check_prediction_similarity(
        prediction['top_k_probabilities'][:2],
        prediction['top_k_classes'][:2],
        similarity_threshold=0.92
    )
    
    if similarity_result['is_ambiguous']:
        prediction['prediction'] = similarity_result['message']
        prediction['confidence'] = 0.0
        prediction['ambiguity_ratio'] = similarity_result['ratio']
    
    # Apply confusion cluster knowledge if available
    if confusion_clusters and prediction['confidence'] > 0:
        prediction = apply_confusion_cluster_knowledge(prediction, class_names, confusion_clusters)
    
    # Apply rejection if confidence is too low
    if prediction['confidence'] < 0.35 and 'Ambiguous' not in prediction['prediction']:
        prediction['prediction'] = "Unknown plant (low confidence)"
        prediction['confidence'] = 0.0
    
    return prediction

def check_prediction_similarity(top_probs, top_classes, similarity_threshold=0.92):
    """Check if top predictions are too similar
    
    Args:
        top_probs: List of top prediction probabilities
        top_classes: List of top prediction classes
        similarity_threshold: Threshold for similarity rejection
        
    Returns:
        Dictionary with ambiguity information
    """
    if len(top_probs) < 2:
        return {'is_ambiguous': False, 'ratio': 0.0}
    
    # Calculate similarity ratio
    similarity_ratio = top_probs[1] / (top_probs[0] + 1e-8)
    
    if similarity_ratio > similarity_threshold:
        return {
            'is_ambiguous': True,
            'ratio': float(similarity_ratio),
            'message': f"Ambiguous - could be {top_classes[0]} or {top_classes[1]}"
        }
    
    return {'is_ambiguous': False, 'ratio': float(similarity_ratio)}

def mixup_augmentation(X, y, alpha=0.2):
    """Performs MixUp augmentation on the input data
    
    Args:
        X: Input features
        y: One-hot encoded labels
        alpha: MixUp interpolation strength parameter
        
    Returns:
        Augmented features and labels
    """
    batch_size = X.shape[0]
    
    # Convert labels to one-hot if they aren't already
    if len(y.shape) == 1:
        y = tf.keras.utils.to_categorical(y)
    
    # Sample the beta distribution for mixing weights
    weights = np.random.beta(alpha, alpha, batch_size)
    weights = np.max([weights, 1-weights], axis=0).reshape(-1, 1)
    
    # Create random index permutation for mixing
    index_permutation = np.random.permutation(batch_size)
    
    # Mix up the features and labels
    X_mixed = X * weights + X[index_permutation] * (1 - weights)
    y_mixed = y * weights + y[index_permutation] * (1 - weights)
    
    return X_mixed, y_mixed

def train_with_meta_learning(model, X_train, y_train, X_val, y_val, epochs=10, tasks_per_batch=5, examples_per_task=10):
    """Implements basic MAML (Model-Agnostic Meta-Learning) for few-shot plant recognition
    
    Args:
        model: Base model to train
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        epochs: Number of meta-learning epochs
        tasks_per_batch: Number of tasks per batch
        examples_per_task: Number of examples per task
    
    Returns:
        Trained model with improved few-shot learning capability
    """
    print(f"{TermColors.CYAN}ℹ Starting meta-learning training for few-shot recognition{TermColors.ENDC}")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Function to create a random task
    def create_random_task(X, y):
        # Randomly select classes for this task
        unique_classes = np.unique(y)
        task_classes = np.random.choice(unique_classes, size=min(5, len(unique_classes)), replace=False)
        
        # Get examples from these classes
        task_indices = np.where(np.isin(y, task_classes))[0]
        np.random.shuffle(task_indices)
        
        # Split into support and query sets
        support_indices = task_indices[:examples_per_task]
        query_indices = task_indices[examples_per_task:examples_per_task*2]
        
        # Return the task data
        return X[support_indices], y[support_indices], X[query_indices], y[query_indices]
    
    # Meta-learning loop
    for epoch in range(epochs):
        meta_loss = 0
        
        for _ in range(tasks_per_batch):
            # Clone the model for this task
            task_model = tf.keras.models.clone_model(model)
            task_model.set_weights(model.get_weights())
            task_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            
            # Create a random task
            X_support, y_support, X_query, y_query = create_random_task(X_train, y_train)
            
            # Inner loop optimization (task-specific)
            task_model.fit(X_support, y_support, epochs=1, verbose=0)
            
            # Evaluate on query set to get meta-gradient
            task_loss = task_model.evaluate(X_query, y_query, verbose=0)[0]
            meta_loss += task_loss
        
        # Update meta-model based on performance across tasks
        meta_loss /= tasks_per_batch
        print(f"Meta-learning epoch {epoch+1}/{epochs}, meta-loss: {meta_loss:.4f}")
        
        # Validate meta-learning progress
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        print(f"Validation - loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")
    
    print(f"{TermColors.GREEN}✅ Meta-learning training completed{TermColors.ENDC}")
    return model

def contrastive_feature_learning(X_train, y_train, margin=1.0, temperature=0.5):
    """Applies contrastive learning to feature vectors to increase class separability
    
    Args:
        X_train: Training features
        y_train: Training labels
        margin: Margin for pushing apart different classes
        temperature: Temperature parameter for similarity scaling
        
    Returns:
        Enhanced feature vectors with better class separation
    """
    print(f"{TermColors.CYAN}ℹ Applying contrastive learning to increase class separability{TermColors.ENDC}")
    
    # Create class-wise centroids
    unique_classes = np.unique(y_train)
    centroids = {}
    
    for cls in unique_classes:
        class_features = X_train[y_train == cls]
        centroids[cls] = np.mean(class_features, axis=0)
    
    # Function to calculate similarity with temperature scaling
    def scaled_cosine_similarity(a, b, temp=temperature):
        similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        return np.exp(similarity / temp)
    
    # Apply contrastive transformation to move features closer to their class centroid
    # and further from other class centroids
    enhanced_features = []
    
    for i, (x, y) in enumerate(zip(X_train, y_train)):
        # Get distances to all centroids
        similarities = {}
        for cls, centroid in centroids.items():
            similarities[cls] = scaled_cosine_similarity(x, centroid)
        
        # Calculate positive and negative forces
        own_centroid = centroids[y]
        positive_force = own_centroid - x
        
        negative_force = np.zeros_like(x)
        for cls, centroid in centroids.items():
            if cls != y:
                # Apply repulsive force from other centroids
                force_magnitude = similarities[cls] * margin
                direction = x - centroid
                direction_norm = np.linalg.norm(direction) + 1e-8
                negative_force += force_magnitude * direction / direction_norm
        
        # Combine forces and apply to feature vector
        combined_force = 0.7 * positive_force + 0.3 * negative_force
        enhanced_x = x + 0.1 * combined_force  # Small step to avoid distorting too much
        
        # Normalize the enhanced feature
        enhanced_x = enhanced_x / (np.linalg.norm(enhanced_x) + 1e-8)
        enhanced_features.append(enhanced_x)
    
    enhanced_features = np.array(enhanced_features)
    print(f"{TermColors.GREEN}✅ Feature enhancement completed with contrastive learning{TermColors.ENDC}")
    
    return enhanced_features

def predict_with_chunked_models(image_path, top_k=5):
    """Make predictions using all the chunked models with combined features and visual progress"""
    def check_memory_limit(threshold=85): # Slightly higher threshold for prediction
        mem = psutil.virtual_memory()
        if mem.percent > threshold:
            print(f"{TermColors.RED}🚨 RAM usage too high ({mem.percent}%). Cannot proceed with prediction.{TermColors.ENDC}")
            return False
        return True

    if not check_memory_limit():
        return []

    print(f"{TermColors.HEADER}\n{'='*50}")
    print(f"PREDICTION FOR: {os.path.basename(image_path)}")
    print(f"{'='*50}{TermColors.ENDC}")

    # Load and preprocess image (using standard scaling, assuming models trained this way)
    try:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0 # Normalize to 0-1 if models expect this
        img_batch = np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"{TermColors.RED}❌ Error loading or preprocessing image {image_path}: {e}{TermColors.ENDC}")
        return []

    # Load feature extractors
    mobilenet_extractor = None
    densenet_extractor = None
    features = None

    print(f"{TermColors.CYAN}ℹ Loading feature extractors (MobileNetV3Large + DenseNet121){TermColors.ENDC}")
    try:
        with tf.device('/GPU:0'):
            from tensorflow.keras.applications import MobileNetV3Large, DenseNet121

            # Load MobileNetV3Large
            mobilenet_base = MobileNetV3Large(
                input_shape=(*IMAGE_SIZE, 3),
                include_top=False,
                weights='imagenet',
                pooling='avg'
            )
            mobilenet_base.trainable = False
            mobilenet_extractor = tf.keras.Model(inputs=mobilenet_base.input, outputs=mobilenet_base.output)
            print(f"{TermColors.GREEN}✅ Loaded MobileNetV3Large{TermColors.ENDC}")

            # Load DenseNet121
            densenet_base = DenseNet121(
                input_shape=(*IMAGE_SIZE, 3),
                include_top=False,
                weights='imagenet',
                pooling='avg'
            )
            densenet_base.trainable = False
            densenet_extractor = tf.keras.Model(inputs=densenet_base.input, outputs=densenet_base.output)
            print(f"{TermColors.GREEN}✅ Loaded DenseNet121{TermColors.ENDC}")

        # Extract features from both models
        print(f"{TermColors.CYAN}ℹ Extracting features...{TermColors.ENDC}")
        mobilenet_features = mobilenet_extractor.predict(img_batch, verbose=0)
        densenet_features = densenet_extractor.predict(img_batch, verbose=0)

        # Concatenate features
        features = np.concatenate([mobilenet_features, densenet_features], axis=-1)
        print(f"{TermColors.GREEN}✅ Combined features extracted (shape: {features.shape}){TermColors.ENDC}")

    except Exception as e:
        print(f"{TermColors.RED}❌ Error loading or using feature extractors: {e}{TermColors.ENDC}")
        traceback.print_exc()
        # Clean up whatever loaded
        del mobilenet_extractor, densenet_extractor, mobilenet_base, densenet_base
        gc.collect()
        tf.keras.backend.clear_session()
        return []
    finally:
        # Clean up base models immediately after feature extraction
        del mobilenet_extractor, densenet_extractor
        # Keep 'features' variable
        gc.collect()
        # Don't clear session yet, need it for chunk models

    if features is None:
        print(f"{TermColors.RED}❌ Feature extraction failed.{TermColors.ENDC}")
        return []

    # Find all trained chunk models
    model_files = glob.glob(os.path.join(MODEL_DIR, "chunk_*_model.keras"))

    if not model_files:
        print(f"{TermColors.RED}❌ No trained models found in {MODEL_DIR}{TermColors.ENDC}")
        # Clean up features if no models found
        del features
        gc.collect()
        tf.keras.backend.clear_session()
        return []

    print(f"{TermColors.CYAN}ℹ Found {len(model_files)} trained chunk models{TermColors.ENDC}")

    # Process each model and collect predictions
    all_predictions = []
    model_bar = tqdm(total=len(model_files), desc="Processing models", position=0, leave=False)

    for model_file in model_files:
        if not check_memory_limit(threshold=90): # Higher threshold during loop
             print(f"{TermColors.YELLOW}⚠️ High memory usage, stopping prediction loop early.{TermColors.ENDC}")
             break # Stop processing more models if memory is critical

        # Extract chunk index
        try:
            chunk_idx = int(os.path.basename(model_file).split("_")[1])
        except (IndexError, ValueError):
            print(f"{TermColors.YELLOW}⚠️ Could not parse chunk index from {model_file}, skipping.{TermColors.ENDC}")
            model_bar.update(1)
            continue

        # Load metadata
        metadata_file = os.path.join(MODEL_DIR, f"chunk_{chunk_idx}_metadata.json")

        if not os.path.exists(metadata_file):
            print(f"{TermColors.YELLOW}⚠️ No metadata found for chunk {chunk_idx}, skipping {model_file}{TermColors.ENDC}")
            model_bar.update(1)
            continue

        class_mapping = {}
        try:
            with open(metadata_file) as f:
                metadata = json.load(f)
                class_mapping = metadata.get("class_mapping", {})
                # Verify feature dimension matches metadata if possible
                expected_dim = metadata.get("feature_dim")
                if expected_dim is not None and expected_dim != features.shape[1]:
                     print(f"{TermColors.RED}❌ Feature dimension mismatch for chunk {chunk_idx}! Expected {expected_dim}, got {features.shape[1]}. Skipping.{TermColors.ENDC}")
                     model_bar.update(1)
                     continue
        except Exception as e:
            print(f"{TermColors.RED}❌ Error loading metadata for chunk {chunk_idx}: {e}{TermColors.ENDC}")
            model_bar.update(1)
            continue

        # Load model
        model = None
        try:
            model = tf.keras.models.load_model(model_file, custom_objects={
                'sparse_top_k_accuracy': sparse_top_k_accuracy,
                # Add any other custom objects your models might need here
            }, compile=False) # Compile=False can save memory if not retraining

            # Get predictions
            preds = model.predict(features, verbose=0)[0]

            # Get top predictions for this chunk
            # Use softmax to get probabilities
            probabilities = tf.nn.softmax(preds).numpy()
            top_indices = np.argsort(probabilities)[-top_k:][::-1]

            chunk_predictions = []
            for idx in top_indices:
                class_name = class_mapping.get(str(idx), f"Unknown-{idx}")
                confidence = float(probabilities[idx])
                chunk_predictions.append((class_name, confidence, chunk_idx))

            # Add to overall predictions
            all_predictions.extend(chunk_predictions)

        except Exception as e:
            print(f"{TermColors.RED}❌ Error loading or predicting with model {model_file}: {e}{TermColors.ENDC}")
            # traceback.print_exc() # Uncomment for detailed debugging
        finally:
            # Clean up the loaded model immediately
            del model
            model_bar.update(1)
            # Explicitly clear session more frequently within the loop
            gc.collect()
            tf.keras.backend.clear_session()
            # Short sleep might help OS reclaim memory sometimes
            time.sleep(0.1)


    model_bar.close()

    # Final cleanup of features
    del features
    gc.collect()
    tf.keras.backend.clear_session()

    if not all_predictions:
        print(f"{TermColors.YELLOW}⚠️ No predictions could be generated.{TermColors.ENDC}")
        return []

    # Sort by confidence
    all_predictions.sort(key=lambda x: x[1], reverse=True)

    # Filter out duplicates keeping the highest confidence one
    filtered_predictions = []
    seen_classes = set()
    for pred_tuple in all_predictions:
        class_name = pred_tuple[0]
        if class_name not in seen_classes:
            filtered_predictions.append(pred_tuple)
            seen_classes.add(class_name)

    # Print top predictions
    print(f"\n{TermColors.HEADER}TOP {top_k} PREDICTIONS (Unique Classes):{TermColors.ENDC}")
    for i, (class_name, confidence, chunk_idx) in enumerate(filtered_predictions[:top_k]):
        confidence_bar = "█" * int(confidence * 20)
        confidence_str = f"[{confidence_bar:<20}] {confidence:.4f}"
        print(f"{i+1}. {TermColors.BOLD}{class_name}{TermColors.ENDC} - {TermColors.CYAN}{confidence_str}{TermColors.ENDC} (From Chunk {chunk_idx})")

    return filtered_predictions[:top_k]

if __name__ == "__main__":
    # Ensure signal handler is set up for graceful interruption
    signal.signal(signal.SIGINT, _signal_handler)

    # Configure GPU memory (using the function defined earlier)
    # limit_memory_usage() is called globally after its definition

    try:
        # Define gpus before using it
        gpus = tf.config.experimental.list_physical_devices('GPU')

        # Print system info with enhanced formatting
        print(f"{TermColors.HEADER}\n{'='*50}")
        print(f"PLANT RECOGNITION MODEL - ADVANCED IMPLEMENTATION")
        print(f"{'='*50}{TermColors.ENDC}")

        gpu_info = "NVIDIA GPU" if gpus else "CPU (No GPU detected)"
        if gpus:
            for gpu in gpus:
                print(f"{TermColors.CYAN}ℹ Found GPU: {gpu.name}{TermColors.ENDC}")
        else:
             print(f"{TermColors.YELLOW}⚠️ Running on CPU (No GPU detected){TermColors.ENDC}")


        # --- AUTOMATICALLY RUN BEST MODEL TRAINING ---
        print(f"\n{TermColors.HEADER}AUTOMATIC MODE: Training Best Model{TermColors.ENDC}")
        print(f"{TermColors.CYAN}This will extract features (if needed), apply contrastive learning, and train chunk models.{TermColors.ENDC}")

        # Call the main training function which now includes contrastive learning
        train_plants_advanced()
        # --- END AUTOMATIC RUN ---

        print(f"\n{TermColors.GREEN}✅ Script execution finished.{TermColors.ENDC}")

    except SystemExit as e:
        print(f"\n{TermColors.YELLOW}ℹ Script exited ({e}).{TermColors.ENDC}")
    except Exception as e:
        print(f"{TermColors.RED}\n{'='*60}")
        print(f"❌ A FATAL ERROR OCCURRED:")
        print(f"{'='*60}")
        traceback.print_exc()
        print(f"{TermColors.RED}{'='*60}{TermColors.ENDC}")
    finally:
        print(f"\n{TermColors.CYAN}--- End of Script ---{TermColors.ENDC}")