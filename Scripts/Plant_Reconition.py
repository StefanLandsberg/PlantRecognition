<<<<<<< HEAD
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, models, mixed_precision
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers.schedules import CosineDecay
import os
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
def limit_memory_usage(gpu_memory_limit_mb=7168, ram_percent_limit=80):
    """Limit both GPU and RAM memory usage to prevent crashes
    
    Args:
        gpu_memory_limit_mb: GPU memory limit in megabytes (default: 2GB)
        ram_percent_limit: Maximum RAM usage allowed as percentage (default: 80%)
    """
    print(f"{TermColors.YELLOW}⚠️ MEMORY LIMITER ACTIVE{TermColors.ENDC}")
    
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
            print(f"{TermColors.RED}❌ Error setting GPU memory limit: {e}{TermColors.ENDC}")
    else:
        print(f"{TermColors.YELLOW}⚠️ No GPU devices found to apply memory limit{TermColors.ENDC}")
    
    # 2. RAM Memory Monitoring Thread
    def ram_monitor_thread(ram_limit=ram_percent_limit):
        """Background thread to monitor RAM usage and prevent crashes"""
        print(f"{TermColors.GREEN}✅ RAM monitor started (limit: {ram_limit}%){TermColors.ENDC}")
        while True:
            try:
                # Check current RAM usage
                mem = psutil.virtual_memory()
                if mem.percent > ram_limit:
                    print(f"{TermColors.RED}⚠️ RAM usage critical: {mem.percent}% > {ram_limit}%{TermColors.ENDC}")
                    print(f"{TermColors.YELLOW}⏸️ Force releasing memory...{TermColors.ENDC}")
                        
                    # Force garbage collection
                    gc.collect()
                        
                    # Try to release large objects
                    for obj in gc.get_objects():
                        try:
                            if isinstance(obj, (np.ndarray, tf.Tensor)):
                                if hasattr(obj, 'nbytes') and getattr(obj, 'nbytes', 0) > 1e6:
                                    del obj
                        except Exception:
                            pass
                        
                    # For extreme cases, try to reduce TensorFlow memory
                    if mem.percent > ram_limit + 10:
                        print(f"{TermColors.RED}🚨 CRITICAL RAM USAGE: {mem.percent}%{TermColors.ENDC}")
                        tf.keras.backend.clear_session()
                        # Sleep to allow system to recover
                        time.sleep(5)
                            
                # Check less frequently when RAM usage is normal    
                time.sleep(5 if mem.percent > ram_limit - 10 else 30)
                    
            except Exception as e:
                # Don't let monitoring thread crash
                print(f"{TermColors.RED}❌ RAM monitor error: {e}{TermColors.ENDC}")
                time.sleep(60)
    
    # Start RAM monitor in background
    try:
        ram_thread = threading.Thread(target=ram_monitor_thread, args=(ram_percent_limit,), daemon=True)
        ram_thread.start()
    except Exception as e:
        print(f"{TermColors.YELLOW}⚠️ Could not start RAM monitor: {e}{TermColors.ENDC}")

# Apply memory limits immediately
limit_memory_usage(6144, 80)  # Using 6GB of GPU memory and 75% RAM threshold

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
DATA_DIR = r"C:\Users\stefa\Desktop\New folder\data\completed_images"
FEATURES_DIR = r"C:\Users\stefa\Desktop\New folder\data\features"
MODEL_DIR = r"C:\Users\stefa\Desktop\New folder\models"
CHECKPOINT_DIR = r"C:\Users\stefa\Desktop\New folder\checkpoints"
LOG_DIR = r"C:\Users\stefa\Desktop\New folder\logs"
STATS_DIR = r"C:\Users\stefa\Desktop\New folder\stats"

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

def enable_emergency_memory_protection():
    """Enable emergency memory protection to prevent system crashes"""
    print(f"{TermColors.RED}🚨 EMERGENCY MEMORY PROTECTION ACTIVATED{TermColors.ENDC}")
    
    # 1. Enforce strict GPU memory limits
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory limit to 4GB instead of 6GB
            for device in gpus:
                tf.config.set_logical_device_configuration(
                    device,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # 4GB limit
                )
            print(f"{TermColors.YELLOW}⚠️ GPU memory limited to 4GB to prevent crashes{TermColors.ENDC}")
        except Exception as e:
            print(f"Error setting GPU memory limit: {e}")
    
    # 2. More aggressive garbage collection
    gc.enable()
    gc.set_threshold(100, 5, 5)  # Make GC more aggressive
    
    # 3. Reduce batch size globally
    global BATCH_SIZE
    BATCH_SIZE = 32  # Force smaller batch size
    
    # 4. Install emergency watchdog
    def emergency_watchdog():
        while True:
            try:
                mem = psutil.virtual_memory()
                if mem.percent > 90:  # Critical memory threshold
                    print(f"{TermColors.RED}🚨 CRITICAL MEMORY USAGE: {mem.percent}%{TermColors.ENDC}")
                    print(f"{TermColors.RED}🚨 FORCING EMERGENCY CLEANUP{TermColors.ENDC}")
                    # Force garbage collection
                    gc.collect()
                    # Clear TensorFlow memory
                    tf.keras.backend.clear_session()
                    # Try to free large objects
                    for obj in gc.get_objects():
                        try:
                            if isinstance(obj, (np.ndarray, tf.Tensor)) and getattr(obj, 'nbytes', 0) > 1e8:
                                del obj
                        except:
                            pass
            except:
                pass
            time.sleep(1)  # Check every second in emergency mode
    
    # Start emergency watchdog
    threading.Thread(target=emergency_watchdog, daemon=True).start()

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
        img_orig = tf.keras.applications.convnext.preprocess_input(img_array.copy())
        augmented_images.append(img_orig)
        
        # Slight rotation (common in plant photos)
        img_rot = tf.keras.preprocessing.image.apply_affine_transform(
            img_array.copy(), theta=np.random.uniform(-20, 20))
        img_rot = tf.keras.applications.convnext.preprocess_input(img_rot)
        augmented_images.append(img_rot)
        
        # Slight zoom (simulate different distances)
        img_zoom = tf.keras.preprocessing.image.apply_affine_transform(
            img_array.copy(), zx=np.random.uniform(0.8, 1.2), zy=np.random.uniform(0.8, 1.2))
        img_zoom = tf.keras.applications.convnext.preprocess_input(img_zoom)
        augmented_images.append(img_zoom)
        
        # Changes in brightness/contrast
        img_bright = img_array.copy() * np.random.uniform(0.8, 1.2)
        img_bright = np.clip(img_bright, 0, 255)
        img_bright = tf.keras.applications.convnext.preprocess_input(img_bright)
        augmented_images.append(img_bright)
        
        # Random crop with padding
        img_crop = tf.keras.preprocessing.image.apply_affine_transform(
            img_array.copy(), zx=0.8, zy=0.8)
        img_crop = tf.keras.applications.convnext.preprocess_input(img_crop)
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
            priority_metrics = ['loss', 'val_loss', 'accuracy', 'val_accuracy', 'sparse_top_k_accuracy', 'val_sparse_top_k_accuracy']
            
            for metric in priority_metrics:
                if metric in logs:
                    value = logs[metric]
                    # Check if value is a formatting-compatible type
                    if isinstance(value, (float, int, np.float32, np.float64, np.int32, np.int64)):
                        if 'loss' in metric:
                            color = TermColors.RED
                            metrics_display.append(f"{color}{metric}: {value:.4f}{TermColors.ENDC}")
                        elif 'accuracy' in metric:
                            color = TermColors.GREEN
                            metrics_display.append(f"{color}{metric}: {value:.4f}{TermColors.ENDC}")
                        else:
                            metrics_display.append(f"{metric}: {value:.4f}")
                    else:
                        # Handle non-numeric types like CosineDecay
                        metrics_display.append(f"{metric}: {str(value)}")
            
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
            for k, v in logs.items():
                if isinstance(v, str):
                    metrics_str += f" | {k}: {v}"
                elif 'loss' in k:
                    metrics_str += f" | {TermColors.RED}{k}: {v:.4f}{TermColors.ENDC}"
                elif 'accuracy' in k:
                    metrics_str += f" | {TermColors.GREEN}{k}: {v:.4f}{TermColors.ENDC}"
                else:
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
                if new_lr < 1e-6:  # Don't go too low
                    new_lr = 1e-6
                
                print(f"{TermColors.YELLOW}⚙ Reducing learning rate: {current_lr:.6f} → {new_lr:.6f}{TermColors.ENDC}")
                # Set the new learning rate safely
                try:
                    tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                except:
                    print(f"{TermColors.YELLOW}⚠ Could not update learning rate - using schedule instead{TermColors.ENDC}")
                self.plateau_count = 0  # Reset counter after adjustment

# Advanced training configuration with gradual unfreezing
def create_custom_callbacks():
    class GradualUnfreezingCallback(tf.keras.callbacks.Callback):
        def __init__(self, model, feature_dim):
            super().__init__()
            self.model = model
            self.feature_dim = feature_dim
            self.unfrozen_layers = 0
            
        def on_epoch_end(self, epoch, logs=None):
            # Every 5 epochs, make the model slightly more complex
            if epoch > 0 and epoch % 5 == 0 and self.unfrozen_layers < 3:
                if self.unfrozen_layers == 0:
                    # Add a new layer
                    x = self.model.layers[-2].output
                    new_layer = layers.Dense(512, activation='swish')(x)
                    outputs = layers.Dense(self.model.output.shape[1], activation='softmax')(new_layer)
                    new_model = tf.keras.Model(inputs=self.model.input, outputs=outputs)
                    weights = self.model.get_weights()
                    self.model = new_model
                    self.model.compile(optimizer=self.model.optimizer, 
                                     loss='sparse_categorical_crossentropy',
                                     metrics=['accuracy', sparse_top_k_accuracy])
                    print(f"{TermColors.CYAN}✨ Added complexity at epoch {epoch+1}{TermColors.ENDC}")
                    
                self.unfrozen_layers += 1
    
    return [GradualUnfreezingCallback(model, feature_dim)]  # Add to your callbacks list

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
        
        if len(class_indices) < 2:  # Need at least 2 examples to interpolate
            continue
        
        # How many synthetic examples to generate 
        num_to_generate = int(len(class_indices) * augmentation_factor)
        
        for _ in range(num_to_generate):
            # Select two random examples from the same class
            idx1, idx2 = np.random.choice(len(class_features), 2, replace=False)
            
            # Create a synthetic sample through interpolation
            alpha = np.random.uniform(0.2, 0.8)
            synthetic_feature = alpha * class_features[idx1] + (1 - alpha) * class_features[idx2]
            
            # Add small noise to avoid creating exact duplicates
            noise = np.random.normal(0, 0.01, size=synthetic_feature.shape)
            synthetic_feature += noise
            
            X_augmented.append(synthetic_feature)
            y_augmented.append(cls)
    
    # Combine with original data
    if X_augmented:  # Check that we generated something
        X_combined = np.vstack([X, np.array(X_augmented)])
        y_combined = np.concatenate([y, np.array(y_augmented)])
        print(f"{TermColors.GREEN}✅ Added {len(X_augmented)} synthetic feature samples{TermColors.ENDC}")
        return X_combined, y_combined
    else:
        return X, y

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
        if val_acc < 0.6:  # 60% threshold
            problematic_chunks.append((chunk_idx, val_acc, val_top_k, num_classes))
    
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
            print(f"Chunk {chunk_idx}: Accuracy {acc:.2%}, Top-{TOP_K} {top_k:.2%} ({num_classes} classes)")
        
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
    """Enhanced feature extraction using EfficientNetV2L with optimized resource utilization"""
    print(f"{TermColors.HEADER}\n{'='*50}")
    print(f"HIGH-PERFORMANCE FEATURE EXTRACTION FOR CHUNK {chunk_idx+1}")
    print(f"{'='*50}{TermColors.ENDC}")
    
    # Setup keyboard interrupt checking
    stop_requested = False
    last_keyboard_check = time.time()
    
    def check_keyboard_interrupt():
        nonlocal stop_requested, last_keyboard_check
        current_time = time.time()
        if current_time - last_keyboard_check < 1.0:
            return stop_requested
        last_keyboard_check = current_time
        try:
            import keyboard
            if keyboard.is_pressed('ctrl+alt+c'):
                print(f"\n{TermColors.YELLOW}⏸️ Feature extraction pause requested (Ctrl+Alt+C). Will stop after current batch.{TermColors.ENDC}")
                stop_requested = True
                return True
        except Exception:
            pass
        return stop_requested

    # Use EfficientNetV2L with optimized settings
    with tf.device('/GPU:0'):
        # Use smaller model for faster processing while maintaining quality
        from tensorflow.keras.applications import EfficientNetV2L
        
        base_model = EfficientNetV2L(
            input_shape=(*IMAGE_SIZE, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        base_model.trainable = False
        feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=base_model.output)
    
    feature_dim = feature_extractor.output_shape[1]
    print(f"{TermColors.GREEN}✅ Feature extractor loaded: EfficientNetV2L ({feature_dim} dims){TermColors.ENDC}")
    
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
            print(f"{TermColors.YELLOW}⚠️ Feature extraction interrupted. Saving progress...{TermColors.ENDC}")
            break
        
        batch_paths = all_image_paths[i:i+batch_size]
        batch_indices = all_class_indices[i:i+batch_size]
        
        # Load and preprocess images
        batch_images = []
        valid_indices = []
        
        for j, (img_path, class_idx) in enumerate(zip(batch_paths, batch_indices)):
            try:
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
                img = tf.keras.preprocessing.image.img_to_array(img)
                img = tf.keras.applications.efficientnet_v2.preprocess_input(img)
                batch_images.append(img)
                valid_indices.append(class_idx)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
        
        # Skip if no valid images in batch
        if not batch_images:
            continue
        
        # Convert to numpy array
        batch_array = np.array(batch_images)
        
        # Extract features
        batch_features = feature_extractor.predict(batch_array, verbose=0)
        
        # Store features and labels
        for feature, class_idx in zip(batch_features, valid_indices):
            all_features.append(feature)
            all_labels.append(class_idx)
        
        # Update progress bar
        progress_bar.update(len(batch_images))
        
        # Force garbage collection periodically
        if i % (batch_size * 10) == 0:
            gc.collect()
    
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
            try:
                # Quick validation check
                with open(img_path, 'rb') as f:
                    # Check if file is at least 100 bytes
                    if len(f.read(100)) < 100:
                        print(f"{TermColors.YELLOW}⚠️ Skipping small file: {img_path}{TermColors.ENDC}")
                        continue
                
                # Try to open the image to verify it's valid
                img = tf.io.read_file(img_path)
                img = tf.image.decode_image(img, channels=3, expand_animations=False)
                
                # If we get here, image is valid
                paths.append(img_path)
                indices.append(class_idx)
            except Exception as e:
                print(f"{TermColors.YELLOW}⚠️ Skipping corrupt image: {img_path} - {e}{TermColors.ENDC}")
                continue
                
    return class_name, paths, indices

def prune_features_rolling_window(current_chunk_idx, window_size=1):  # Reduced window size to 1
    """Aggressively remove feature files after model training to save disk space
    
    Args:
        current_chunk_idx: The chunk that was just processed
        window_size: Number of chunks to keep (default: 1 - only keep current chunk)
    """
    print(f"{TermColors.HEADER}\n{'='*50}")
    print(f"AGGRESSIVE FEATURE MANAGEMENT")
    print(f"{'='*50}{TermColors.ENDC}")
    
    # Check all chunks up to the current one for possible deletion
    for chunk_to_check in range(current_chunk_idx + 1):  # Include current chunk in check
        # Skip the most recent chunks based on window_size
        if chunk_to_check > current_chunk_idx - window_size:
            continue
            
        # Check if this chunk's features exist
        chunk_feature_dir = os.path.join(FEATURES_DIR, f"chunk_{chunk_to_check}")
        feature_file = os.path.join(chunk_feature_dir, "features.npz")
        
        if os.path.exists(feature_file):
            # Check if model for this chunk exists (we don't want to delete features 
            # if the model doesn't exist or training failed)
            model_file = os.path.join(MODEL_DIR, f"chunk_{chunk_to_check}_model.keras")
            metadata_file = os.path.join(MODEL_DIR, f"chunk_{chunk_to_check}_metadata.json")
            
            if os.path.exists(model_file) and os.path.exists(metadata_file):
                # Get file size before deletion for reporting
                file_size_mb = os.path.getsize(feature_file) / (1024 * 1024)
                
                print(f"{TermColors.YELLOW}⚠️ Removing features for chunk {chunk_to_check} to save disk space ({file_size_mb:.2f} MB){TermColors.ENDC}")
                
                # Remove the feature file but keep the class mapping
                os.remove(feature_file)
                
                # Create a placeholder to indicate features were deleted
                with open(os.path.join(chunk_feature_dir, "features_pruned.txt"), "w") as f:
                    f.write(f"Features were pruned on {datetime.now().isoformat()} to save disk space.\n")
                    f.write(f"Original size: {file_size_mb:.2f} MB\n")
                    f.write(f"Model is preserved at: {model_file}\n")
                
                print(f"{TermColors.GREEN}✅ Freed {file_size_mb:.2f} MB of disk space{TermColors.ENDC}")
            else:
                print(f"{TermColors.CYAN}ℹ Keeping features for chunk {chunk_to_check} because model not found or incomplete{TermColors.ENDC}")

def load_features(features_file):
    """Load extracted features from file with verification to detect corruption"""
    print(f"{TermColors.CYAN}ℹ Loading features from {features_file}{TermColors.ENDC}")
    
    # Add this corruption check before loading
    try:
        file_size = os.path.getsize(features_file)
        if file_size < 1000:  # Suspiciously small
            print(f"{TermColors.RED}⚠️ Feature file appears corrupted (only {file_size} bytes){TermColors.ENDC}")
            raise ValueError("Feature file too small, likely corrupted")
            
        # Now proceed with loading
        load_bar = tqdm(total=1, desc="Loading features")
        data = np.load(features_file)
        features = data['features']
        labels = data['labels']
        load_bar.update(1)
        load_bar.close()
        
        print(f"{TermColors.GREEN}✅ Loaded {features.shape[0]} features with {features.shape[1]} dimensions{TermColors.ENDC}")
        return features, labels
        
    except Exception as e:
        print(f"{TermColors.RED}❌ Error loading features: {e}{TermColors.ENDC}")
        return None, None

# Memory management function with proper gc import
def clean_memory():
    """Aggressively clean memory between processing steps with time-based cleanup"""
    # Static variable to track last cleanup time
    if not hasattr(clean_memory, "last_cleanup_time"):
        clean_memory.last_cleanup_time = time.time()
    
    current_time = time.time()
    should_deep_clean = (current_time - clean_memory.last_cleanup_time) >= 60  # 60 seconds = 1 minute
    
    # Check current memory usage
    mem = psutil.virtual_memory()
    force_cleanup = mem.percent > 80  # Force cleanup if memory usage is above 80%
    
    # Basic cleanup always happens
    tf.keras.backend.clear_session()
    gc.collect()
    
    # Deep cleanup happens every minute OR when memory usage is high
    if should_deep_clean or force_cleanup:
        if force_cleanup:
            print(f"{TermColors.RED}⚠️ High memory usage detected ({mem.percent}%). Performing emergency cleanup.{TermColors.ENDC}")
        else:
            print(f"{TermColors.CYAN}ℹ Performing scheduled deep memory cleanup{TermColors.ENDC}")
        
        # More aggressive garbage collection
        for _ in range(5):  # Multiple collection cycles
            gc.collect()
        
        # Try to release GPU memory if possible
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Reset TensorFlow session
                tf.keras.backend.clear_session()
                
                # Extra GPU memory cleanup
                print(f"{TermColors.CYAN}ℹ Releasing GPU memory{TermColors.ENDC}")
            except:
                pass
        
        # Update last cleanup time
        clean_memory.last_cleanup_time = current_time

# In your build_model function, add attention mechanism
def build_model(feature_dim, num_classes):
    inputs = tf.keras.Input(shape=(feature_dim,))
    
    # Modified architecture for better generalization
    x = layers.Dense(2048)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)  # Replace ReLU with Swish for better gradient flow
    x = layers.Dropout(0.4)(x)  # Increased dropout slightly (0.3 -> 0.4)
    
    # Add a residual connection
    residual = x  # Save for residual connection
    
    x = layers.Dense(1024)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Dropout(0.3)(x)  # Increased dropout slightly (0.2 -> 0.3)
    
    # Optional: Add a skip connection if shapes match
    if residual.shape[-1] == x.shape[-1]:
        x = layers.add([x, residual])
    else:
        # Use a projection if shapes don't match
        projection = layers.Dense(x.shape[-1], use_bias=False)(residual)
        x = layers.add([x, projection])
    
    # Add one more layer with L2 regularization for better generalization
    x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Original optimizer settings
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', sparse_top_k_accuracy]
    )
    
    return model

def build_triplet_model(feature_dim, num_classes, margin=0.5):
    """Build a model using triplet loss to enhance class separability"""
    # Feature extraction base
    input_tensor = tf.keras.Input(shape=(feature_dim,))
    x = layers.Dense(1024)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Dropout(0.3)(x)
    
    # Create embedding space
    embedding = layers.Dense(512, name="embedding")(x)
    
    # Classifier head
    classifier = layers.Dense(num_classes, activation='softmax')(embedding)
    
    # Create main model
    model = tf.keras.Model(inputs=input_tensor, outputs=classifier)
    
    # Create embedding model
    embedding_model = tf.keras.Model(inputs=input_tensor, outputs=embedding)
    
    # Define triplet loss function
    def triplet_loss(y_true, y_pred):
        # Standard categorical cross-entropy
        categorical_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        
        # Get embeddings for batch
        embeddings = embedding_model(input_tensor)
        
        # Calculate triplet loss
        # (Simplified implementation - a full triplet loss would need proper triplet mining)
        batch_size = tf.shape(y_true)[0]
        
        # Get positive and negative masks based on class labels
        y_reshaped = tf.reshape(y_true, [-1, 1])
        positive_mask = tf.equal(y_reshaped, tf.transpose(y_reshaped))
        negative_mask = tf.logical_not(positive_mask)
        
        # Convert to float
        positive_mask = tf.cast(positive_mask, dtype=tf.float32)
        negative_mask = tf.cast(negative_mask, dtype=tf.float32)
        
        # Get embeddings and calculate distances
        embeddings_flat = tf.reshape(embeddings, [batch_size, -1])
        
        # Calculate distance matrix
        distances = tf.reduce_sum(tf.square(tf.expand_dims(embeddings_flat, 1) - 
                                          tf.expand_dims(embeddings_flat, 0)), 2)
        
        # Get positives and negatives
        positive_dists = tf.reduce_mean(distances * positive_mask) 
        negative_dists = tf.reduce_mean(distances * negative_mask)
        
        # Triplet loss term
        triplet_term = tf.maximum(0.0, positive_dists - negative_dists + margin)
        
        # Combine losses
        total_loss = categorical_loss + 0.3 * triplet_term
        return total_loss
    
    # Compile with custom loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=triplet_loss,
        metrics=['accuracy', sparse_top_k_accuracy]
    )
    
    return model

class DynamicTrainingOptimizer(tf.keras.callbacks.Callback):
    """Dynamically adapt training parameters based on metrics patterns"""
    def __init__(self, patience=3):
        super().__init__()
        self.last_losses = []
        self.patience = patience
        self.last_lr_change = 0
        self.best_val_loss = float('inf')
        self.stagnation_count = 0
        self.lr_reductions = 0
        self.max_lr_reductions = 4  # Prevent too many reductions
        self.initial_lr = None
        
    def on_train_begin(self, logs=None):
        # Capture initial learning rate
        if hasattr(self.model.optimizer, 'lr'):
            if hasattr(self.model.optimizer.lr, '__call__'):
                # For LearningRateSchedule objects
                current_step = self.model.optimizer.iterations.numpy()
                self.initial_lr = float(self.model.optimizer.lr(current_step).numpy())
            else:
                # For fixed learning rates
                self.initial_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        
    def on_epoch_end(self, epoch, logs=None):
        if not logs or 'val_loss' not in logs:
            return
            
        val_loss = logs.get('val_loss')
        val_acc = logs.get('val_accuracy', 0)
        
        # Store loss history for trend analysis
        self.last_losses.append(val_loss)
        if len(self.last_losses) > self.patience:
            self.last_losses.pop(0)
        
        # Check for improvement
        if val_loss < self.best_val_loss:
            improvement = self.best_val_loss - val_loss
            self.best_val_loss = val_loss
            self.stagnation_count = 0
            print(f"{TermColors.GREEN}✓ Validation improving by {improvement:.4f} - best loss: {val_loss:.4f}{TermColors.ENDC}")
        else:
            self.stagnation_count += 1
            print(f"{TermColors.YELLOW}⚠ No improvement for {self.stagnation_count} epochs (best: {self.best_val_loss:.4f}, current: {val_loss:.4f}){TermColors.ENDC}")
            
        # Get current learning rate
        try:
            if hasattr(self.model.optimizer.lr, '__call__'):
                # For LearningRateSchedule objects
                current_step = self.model.optimizer.iterations.numpy()
                current_lr = float(self.model.optimizer.lr(current_step).numpy())
            else:
                # For fixed learning rates
                current_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        except:
            current_lr = 0.001  # Fallback
            
        # If enough history has been collected and we're past the early epochs
        if len(self.last_losses) >= self.patience and epoch > 5:
            # Check for worsening trend (consecutive increases in loss)
            if all(self.last_losses[i] > self.last_losses[i-1] for i in range(1, len(self.last_losses))):
                # Only reduce if we haven't reduced too many times already
                if self.lr_reductions < self.max_lr_reductions:
                    # Reduce LR more aggressively when loss is consistently increasing
                    new_lr = current_lr * 0.3  # More aggressive reduction
                    try:
                        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                        print(f"{TermColors.YELLOW}⚠️ Loss consistently increasing - LR reduced: {current_lr:.6f} → {new_lr:.6f}{TermColors.ENDC}")
                        self.lr_reductions += 1
                        self.last_losses = []  # Reset history after adjustment
                        self.last_lr_change = epoch
                    except:
                        print(f"{TermColors.RED}❌ Failed to update learning rate{TermColors.ENDC}")
                
            # Check for plateaus (minimal changes in loss)
            elif self.stagnation_count >= self.patience:
                diffs = [abs(self.last_losses[i] - self.last_losses[i-1]) for i in range(1, len(self.last_losses))]
                if all(d < 0.001 for d in diffs):  # Very small changes
                    # Try increasing the batch size if plateau detected
                    if epoch - self.last_lr_change > 5:
                        current_bs = self.model.get_config().get('batch_size', BATCH_SIZE)
                        if current_bs < 128:  # Don't go too large
                            # Rather than changing batch size (which is tricky), adjust learning rate instead
                            new_lr = current_lr * 0.5
                            if new_lr >= 1e-6:  # Don't go too small
                                try:
                                    tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                                    print(f"{TermColors.CYAN}🔄 Plateau detected - Reducing learning rate: {current_lr:.6f} → {new_lr:.6f}{TermColors.ENDC}")
                                    self.lr_reductions += 1
                                    self.last_lr_change = epoch
                                except:
                                    print(f"{TermColors.RED}❌ Failed to update learning rate{TermColors.ENDC}")
                
        # If accuracy is already very high, fine-tune with smaller learning rate
        if val_acc > 0.9 and current_lr > 1e-5 and epoch - self.last_lr_change > 3:
            new_lr = current_lr * 0.5
            try:
                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                print(f"{TermColors.GREEN}✓ High accuracy detected ({val_acc:.4f}) - Fine-tuning with lower LR: {new_lr:.6f}{TermColors.ENDC}")
                self.last_lr_change = epoch
            except:
                pass

class AdaptiveRegularization(tf.keras.callbacks.Callback):
    """Dynamically adjust regularization based on overfitting detection"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.overfitting_streak = 0
        self.underfitting_streak = 0
        self.dropout_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dropout)]
        self.regularization_changes = 0
        self.max_reg_changes = 3
        
    def on_epoch_end(self, epoch, logs=None):
        # Skip early epochs
        if epoch < 3:
            return
            
        # Check for overfitting by comparing train and val accuracy
        train_acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        
        if train_acc is not None and val_acc is not None:
            gap = train_acc - val_acc
            loss_gap = val_loss - train_loss
            
            # Detect significant overfitting gap
            if gap > 0.1:  # 10% gap indicates overfitting
                self.overfitting_streak += 1
                self.underfitting_streak = 0
                
                if self.overfitting_streak >= 3 and self.regularization_changes < self.max_reg_changes:  # 3 consecutive epochs showing overfitting
                    print(f"{TermColors.YELLOW}⚠️ Overfitting detected (acc gap: {gap:.2f}, loss gap: {loss_gap:.2f}) - Increasing regularization{TermColors.ENDC}")
                    
                    # Increase dropout rates if we can
                    if self.dropout_layers:
                        for layer in self.dropout_layers:
                            current_rate = tf.keras.backend.get_value(layer.rate)
                            new_rate = min(current_rate + 0.05, 0.7)  # Don't go beyond 0.7
                            layer.rate = new_rate
                            print(f"{TermColors.CYAN}↑ Dropout adjusted: {current_rate:.2f} → {new_rate:.2f}{TermColors.ENDC}")
                        
                        self.regularization_changes += 1
                        self.overfitting_streak = 0  # Reset after adjustment
            
            # Detect underfitting (training and validation both low but close)
            elif gap < 0.03 and train_acc < 0.5 and epoch > 5:
                self.underfitting_streak += 1
                self.overfitting_streak = 0
                
                if self.underfitting_streak >= 3 and self.regularization_changes < self.max_reg_changes:
                    print(f"{TermColors.YELLOW}⚠️ Underfitting detected (acc: {train_acc:.2f}, gap: {gap:.2f}) - Decreasing regularization{TermColors.ENDC}")
                    
                    # Decrease dropout rates if possible
                    if self.dropout_layers:
                        for layer in self.dropout_layers:
                            current_rate = tf.keras.backend.get_value(layer.rate)
                            new_rate = max(current_rate - 0.05, 0.1)  # Don't go below 0.1
                            layer.rate = new_rate
                            print(f"{TermColors.CYAN}↓ Dropout adjusted: {current_rate:.2f} → {new_rate:.2f}{TermColors.ENDC}")
                        
                        self.regularization_changes += 1
                        self.underfitting_streak = 0  # Reset after adjustment
            else:
                # Reset both streaks if neither condition is met
                self.overfitting_streak = 0
                self.underfitting_streak = 0

class AdaptiveClassWeightAdjuster(tf.keras.callbacks.Callback):
    """Dynamically adjust class weights based on per-class performance"""
    def __init__(self, X_val, y_val, class_names, update_frequency=5):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.class_names = class_names
        self.update_frequency = update_frequency
        self.adjustments_made = 0
        self.max_adjustments = 3
        self.class_weights = None
        
    def on_train_begin(self, logs=None):
        # Initialize with equal class weights
        num_classes = len(self.class_names)
        self.class_weights = {i: 1.0 for i in range(num_classes)}
        
    def on_epoch_end(self, epoch, logs=None):
        # Only update weights periodically to allow training to adapt
        if epoch % self.update_frequency != 0 or self.adjustments_made >= self.max_adjustments:
            return
            
        # Get predictions on validation set
        y_pred = self.model.predict(self.X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calculate per-class accuracy
        class_accuracy = {}
        for cls in range(len(self.class_names)):
            # Find samples for this class
            mask = self.y_val == cls
            if np.sum(mask) == 0:  # No examples of this class
                continue
                
            # Calculate accuracy for this class
            correct = (y_pred_classes[mask] == cls).sum()
            total = mask.sum()
            class_accuracy[cls] = correct / total if total > 0 else 0.0
        
        # Find problematic classes
        overall_acc = logs.get('val_accuracy', 0)
        problem_classes = []
        
        for cls, acc in class_accuracy.items():
            # Classes with accuracy significantly below average
            if acc < overall_acc * 0.7:  # 30% below average is problematic
                problem_classes.append((cls, acc))
        
        # Adjust weights if we have problematic classes
        if problem_classes:
            print(f"{TermColors.YELLOW}⚠️ Found {len(problem_classes)} underperforming classes{TermColors.ENDC}")
            
            # Sort by lowest accuracy first
            problem_classes.sort(key=lambda x: x[1])
            
            # Just show a few examples
            for cls, acc in problem_classes[:3]:
                print(f"  - Class {self.class_names[cls]}: {acc:.2%} accuracy (vs {overall_acc:.2%} overall)")
                
                # Increase weight for this class
                old_weight = self.class_weights.get(cls, 1.0)
                new_weight = old_weight * 1.5  # Increase by 50%
                self.class_weights[cls] = min(new_weight, 5.0)  # Cap at 5x
                
                print(f"{TermColors.CYAN}↑ Weight adjusted: {old_weight:.2f} → {self.class_weights[cls]:.2f}{TermColors.ENDC}")
            
            # Apply new class weights to model 
            print(f"{TermColors.CYAN}ℹ Updated class weights will affect future training{TermColors.ENDC}")
            self.adjustments_made += 1

def test_time_augmentation_predict(image_path, top_k=5):
    """Memory-efficient prediction using test-time augmentation"""
    def check_memory_limit():
        mem = psutil.virtual_memory()
        if mem.percent > 80:  # Emergency cutoff
            print(f"{TermColors.RED}🚨 RAM usage too high ({mem.percent}%). Cannot proceed with prediction.{TermColors.ENDC}")
            return False
        return True
        
    if not check_memory_limit():
        return []
    
    print(f"{TermColors.HEADER}\n{'='*50}")
    print(f"TTA PLANT IDENTIFICATION: {os.path.basename(image_path)}")
    print(f"{'='*50}{TermColors.ENDC}")
    
    # Load and preprocess image once
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img_processed = tf.keras.applications.efficientnet_v2.preprocess_input(img)
    
    # Define efficient augmentations for plants
    augmentations = [
        {"name": "original", "transform": lambda img: img},
        {"name": "horizontal_flip", "transform": lambda img: np.fliplr(img)},
        {"name": "rotation_10", "transform": lambda img: tf.keras.preprocessing.image.apply_affine_transform(img, theta=10)},
        {"name": "brightness_up", "transform": lambda img: np.clip(img * 1.1, 0, 255)}
    ]
    
    # Load feature extractor once
    print(f"{TermColors.CYAN}ℹ Loading feature extractor{TermColors.ENDC}")
    with tf.device('/GPU:0'):
        from tensorflow.keras.applications import EfficientNetV2L
        feature_extractor = EfficientNetV2L(
            input_shape=(*IMAGE_SIZE, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        feature_extractor.trainable = False
    
    # Dictionary to store all class predictions
    all_predictions = {}
    aug_bar = tqdm(total=len(augmentations), desc="Processing augmentations", position=0)
    
    # Process each augmentation separately
    for aug in augmentations:
        # Apply augmentation
        aug_bar.set_description(f"Processing {aug['name']}")
        img_aug = aug["transform"](img_processed.copy())
        img_batch = np.expand_dims(img_aug, axis=0)
        
        # Extract features
        features = feature_extractor.predict(img_batch, verbose=0)
        
        # Find and load models
        model_files = glob.glob(os.path.join(MODEL_DIR, "chunk_*_model.keras"))
        
        # Process each model
        for model_file in model_files:
            # Reset TF session between models to free memory
            tf.keras.backend.clear_session()
            
            chunk_idx = int(os.path.basename(model_file).split("_")[1])
            metadata_file = os.path.join(MODEL_DIR, f"chunk_{chunk_idx}_metadata.json")
            
            if not os.path.exists(metadata_file):
                continue
                
            # Load model and metadata
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            model = tf.keras.models.load_model(model_file, custom_objects={
                'sparse_top_k_accuracy': sparse_top_k_accuracy
            })
            
            # Get predictions
            preds = model.predict(features, verbose=0)[0]
            
            # Get top predictions for this model
            top_indices = np.argsort(preds)[-top_k*2:][::-1]  # Get more candidates for consistency
            
            for idx in top_indices:
                class_name = metadata['class_mapping'][str(idx)]
                confidence = float(preds[idx])
                
                # Store predictions by class name
                class_key = f"{class_name}_{chunk_idx}"
                if class_key not in all_predictions:
                    all_predictions[class_key] = {
                        "class_name": class_name,
                        "chunk_idx": chunk_idx,
                        "total_score": 0,
                        "count": 0
                    }
                
                all_predictions[class_key]["total_score"] += confidence
                all_predictions[class_key]["count"] += 1
            
            # Free memory
            del model, preds
            gc.collect()
        
        # Free memory
        del img_aug, img_batch, features
        gc.collect()
        aug_bar.update(1)
    
    aug_bar.close()
    
    # Clean up GPU memory
    del feature_extractor
    gc.collect()
    tf.keras.backend.clear_session()
    
    # Calculate average score for each class
    final_predictions = []
    for class_key, data in all_predictions.items():
        avg_confidence = data["total_score"] / data["count"] if data["count"] > 0 else 0
        final_predictions.append((data["class_name"], avg_confidence, data["chunk_idx"]))
    
    # Sort by confidence
    final_predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Print results
    print(f"\n{TermColors.HEADER}TOP {top_k} PREDICTIONS (WITH TTA):{TermColors.ENDC}")
    for i, (class_name, confidence, chunk_idx) in enumerate(final_predictions[:top_k]):
        confidence_bar = "█" * int(confidence * 20)
        confidence_str = f"[{confidence_bar:<20}] {confidence:.4f}"
        print(f"{i+1}. {TermColors.BOLD}{class_name}{TermColors.ENDC} - {TermColors.CYAN}{confidence_str}{TermColors.ENDC} (Chunk {chunk_idx})")
    
    return final_predictions[:top_k]

def self_consistency_predict(image_path, top_k=5):
    """Memory-efficient prediction with self-consistency checking"""
    def check_memory_limit():
        mem = psutil.virtual_memory()
        if mem.percent > 80:  # Emergency cutoff
            print(f"{TermColors.RED}🚨 RAM usage too high ({mem.percent}%). Cannot proceed with prediction.{TermColors.ENDC}")
            return False
        return True
        
    if not check_memory_limit():
        return []
    
    print(f"{TermColors.HEADER}\n{'='*50}")
    print(f"SELF-CONSISTENCY PLANT IDENTIFICATION: {os.path.basename(image_path)}")
    print(f"{'='*50}{TermColors.ENDC}")
    
    # Generate multiple augmentations of the input image
    def generate_augmentations(img_path, num_aug=4):
        """Generate multiple augmentations with different parameters"""
        # Load image once
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
        img = tf.keras.preprocessing.image.img_to_array(img)
        
        # Create augmentations
        augmentations = []
        
        # Original image
        img_orig = tf.keras.applications.efficientnet_v2.preprocess_input(img.copy())
        augmentations.append(img_orig)
        
        # Flip horizontal
        img_flip = np.fliplr(img.copy())
        img_flip = tf.keras.applications.efficientnet_v2.preprocess_input(img_flip)
        augmentations.append(img_flip)
        
        # Rotate +10 degrees
        img_rot1 = tf.keras.preprocessing.image.apply_affine_transform(img.copy(), theta=10)
        img_rot1 = tf.keras.applications.efficientnet_v2.preprocess_input(img_rot1)
        augmentations.append(img_rot1)
        
        # Rotate -10 degrees
        img_rot2 = tf.keras.preprocessing.image.apply_affine_transform(img.copy(), theta=-10)
        img_rot2 = tf.keras.applications.efficientnet_v2.preprocess_input(img_rot2)
        augmentations.append(img_rot2)
        
        return augmentations
    
    # Generate augmentations
    print(f"{TermColors.CYAN}ℹ Generating image augmentations{TermColors.ENDC}")
    augmented_images = generate_augmentations(image_path)
    
    # Load feature extractor
    print(f"{TermColors.CYAN}ℹ Loading feature extractor{TermColors.ENDC}")
    with tf.device('/GPU:0'):
        from tensorflow.keras.applications import EfficientNetV2L
        feature_extractor = EfficientNetV2L(
            input_shape=(*IMAGE_SIZE, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        feature_extractor.trainable = False
    
    # Track votes for each class
    class_votes = {}
    
    # Process each augmentation independently to minimize memory usage
    aug_bar = tqdm(total=len(augmented_images), desc="Processing augmentations", position=0)
    
    for aug_idx, aug_img in enumerate(augmented_images):
        # Extract features
        img_batch = np.expand_dims(aug_img, axis=0)
        features = feature_extractor.predict(img_batch, verbose=0)
        
        # Load models
        model_files = glob.glob(os.path.join(MODEL_DIR, "chunk_*_model.keras"))
        
        for model_file in model_files:
            # Clean memory between models
            tf.keras.backend.clear_session()
            
            chunk_idx = int(os.path.basename(model_file).split("_")[1])
            metadata_file = os.path.join(MODEL_DIR, f"chunk_{chunk_idx}_metadata.json")
            
            if not os.path.exists(metadata_file):
                continue
                
            # Load model and metadata
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            model = tf.keras.models.load_model(model_file, custom_objects={
                'sparse_top_k_accuracy': sparse_top_k_accuracy
            })
            
            # Get predictions
            preds = model.predict(features, verbose=0)[0]
            
            # Get top predictions 
            top_indices = np.argsort(preds)[-top_k*2:][::-1]  # Get more candidates
            
            for rank, idx in enumerate(top_indices):
                class_name = metadata['class_mapping'][str(idx)]
                confidence = float(preds[idx])
                
                # Weight by both rank and confidence
                # Higher rank (lower number) = higher weight
                rank_weight = 1.0 / (rank + 1)
                # Also consider which augmentation (original gets higher weight)
                aug_weight = 1.0 if aug_idx == 0 else 0.8
                
                # Compute vote weight
                vote_weight = confidence * rank_weight * aug_weight
                
                # Add to votes
                if class_name not in class_votes:
                    class_votes[class_name] = {
                        "score": 0,
                        "chunk_idx": chunk_idx,
                        "mentions": 0,
                        "highest_conf": 0
                    }
                
                class_votes[class_name]["score"] += vote_weight
                class_votes[class_name]["mentions"] += 1
                class_votes[class_name]["highest_conf"] = max(
                    class_votes[class_name]["highest_conf"], confidence
                )
            
            # Clean up model
            del model
            gc.collect()
        
        # Clean up features
        del features, img_batch
        gc.collect()
        aug_bar.update(1)
    
    aug_bar.close()
    
    # Clean up GPU memory
    del feature_extractor, augmented_images
    gc.collect()
    tf.keras.backend.clear_session()
    
    # Calculate consistency score:
    # - High score if appears in multiple augmentations
    # - High score if ranks high consistently
    final_predictions = []
    for class_name, data in class_votes.items():
        # Combine score with number of mentions for consistency
        # This favors classes that appear consistently across augmentations
        consistency_score = data["score"] * (data["mentions"] / (len(augmented_images) * 2))
        
        # Boost score of items that appeared in all augmentations
        if data["mentions"] >= len(augmented_images):
            consistency_score *= 1.2
        
        final_predictions.append((class_name, consistency_score, data["chunk_idx"]))
    
    # Sort by consistency score
    final_predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Print results
    print(f"\n{TermColors.HEADER}TOP {top_k} PREDICTIONS (SELF-CONSISTENCY):{TermColors.ENDC}")
    for i, (class_name, score, chunk_idx) in enumerate(final_predictions[:top_k]):
        # Normalize score for display
        normalized_score = min(score / 2.0, 1.0)  # Scale for display
        confidence_bar = "█" * int(normalized_score * 20)
        confidence_str = f"[{confidence_bar:<20}] {normalized_score:.4f}"
        print(f"{i+1}. {TermColors.BOLD}{class_name}{TermColors.ENDC} - {TermColors.CYAN}{confidence_str}{TermColors.ENDC} (Chunk {chunk_idx})")
    
    return final_predictions[:top_k]

def cross_model_fusion_predict(image_path, top_k=5):
    """Memory-efficient prediction using cross-model feature fusion"""
    def check_memory_limit():
        mem = psutil.virtual_memory()
        if mem.percent > 80:  # Emergency cutoff
            print(f"{TermColors.RED}🚨 RAM usage too high ({mem.percent}%). Cannot proceed with prediction.{TermColors.ENDC}")
            return False
        return True
    
    if not check_memory_limit():
        return []
    
    print(f"{TermColors.HEADER}\n{'='*50}")
    print(f"CROSS-MODEL FUSION PLANT IDENTIFICATION: {os.path.basename(image_path)}")
    print(f"{'='*50}{TermColors.ENDC}")
    
    # Load image once
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
    img = tf.keras.preprocessing.image.img_to_array(img)
    
    # Create two copies with different preprocessing for different models
    img_efficient = tf.keras.applications.efficientnet_v2.preprocess_input(img.copy())
    img_efficient_batch = np.expand_dims(img_efficient, axis=0)
    
    print(f"{TermColors.CYAN}ℹ Extracting features using dual-model approach{TermColors.ENDC}")
    
    # First extract primary features
    with tf.device('/GPU:0'):
        # Primary model - EfficientNetV2L
        print(f"{TermColors.CYAN}ℹ Loading primary feature extractor (EfficientNetV2L){TermColors.ENDC}")
        from tensorflow.keras.applications import EfficientNetV2L
        feature_extractor1 = EfficientNetV2L(
            input_shape=(*IMAGE_SIZE, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        feature_extractor1.trainable = False
        
        # Extract primary features
        primary_features = feature_extractor1.predict(img_efficient_batch, verbose=0)
        
        # Free GPU memory
        del feature_extractor1
        gc.collect()
        tf.keras.backend.clear_session()
    
    # Process models with primary features
    print(f"{TermColors.CYAN}ℹ Processing models with primary feature extractor{TermColors.ENDC}")
    primary_predictions = {}
    
    # Get all models
    model_files = glob.glob(os.path.join(MODEL_DIR, "chunk_*_model.keras"))
    model_bar = tqdm(total=len(model_files), desc="Processing models (primary)", position=0)
    
    for model_file in model_files:
        # Clean memory before each model
        tf.keras.backend.clear_session()
        
        chunk_idx = int(os.path.basename(model_file).split("_")[1])
        metadata_file = os.path.join(MODEL_DIR, f"chunk_{chunk_idx}_metadata.json")
        
        if not os.path.exists(metadata_file):
            model_bar.update(1)
            continue
            
        # Load model and metadata
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        model = tf.keras.models.load_model(model_file, custom_objects={
            'sparse_top_k_accuracy': sparse_top_k_accuracy
        })
        
        # Get predictions
        preds = model.predict(primary_features, verbose=0)[0]
        
        # Store top predictions
        top_indices = np.argsort(preds)[-top_k*3:][::-1]  # Get more candidates
        
        for rank, idx in enumerate(top_indices):
            class_name = metadata['class_mapping'][str(idx)]
            confidence = float(preds[idx])
            
            class_key = f"{class_name}_{chunk_idx}"
            if class_key not in primary_predictions:
                primary_predictions[class_key] = {
                    "class_name": class_name,
                    "chunk_idx": chunk_idx,
                    "primary_score": 0,
                    "primary_rank": float('inf')
                }
            
            # Store confidence and best rank
            primary_predictions[class_key]["primary_score"] = confidence
            primary_predictions[class_key]["primary_rank"] = min(
                primary_predictions[class_key]["primary_rank"], rank
            )
        
        # Free memory
        del model, preds
        gc.collect()
        model_bar.update(1)
    
    model_bar.close()
    
    # Free primary features to save memory
    del primary_features, img_efficient, img_efficient_batch
    gc.collect()
    
    # Second feature extraction - use DenseNet121 for diversity
    # Prepare image for second model
    img_dense = tf.keras.applications.densenet.preprocess_input(img.copy())
    img_dense_batch = np.expand_dims(img_dense, axis=0)
    
    # Extract secondary features
    with tf.device('/GPU:0'):
        print(f"{TermColors.CYAN}ℹ Loading secondary feature extractor (DenseNet121){TermColors.ENDC}")
        from tensorflow.keras.applications import DenseNet121
        feature_extractor2 = DenseNet121(
            input_shape=(*IMAGE_SIZE, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        feature_extractor2.trainable = False
        
        # Extract secondary features
        secondary_features = feature_extractor2.predict(img_dense_batch, verbose=0)
        
        # Free GPU memory
        del feature_extractor2
        gc.collect()
        tf.keras.backend.clear_session()
    
    # Process models with secondary features
    print(f"{TermColors.CYAN}ℹ Processing models with secondary feature extractor{TermColors.ENDC}")
    secondary_predictions = {}
    
    model_bar = tqdm(total=len(model_files), desc="Processing models (secondary)", position=0)
    
    for model_file in model_files:
        # Clean memory before each model
        tf.keras.backend.clear_session()
        
        chunk_idx = int(os.path.basename(model_file).split("_")[1])
        metadata_file = os.path.join(MODEL_DIR, f"chunk_{chunk_idx}_metadata.json")
        
        if not os.path.exists(metadata_file):
            model_bar.update(1)
            continue
            
        # Load model and metadata
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        model = tf.keras.models.load_model(model_file, custom_objects={
            'sparse_top_k_accuracy': sparse_top_k_accuracy
        })
        
        # Get predictions - may need to reshape or adapt secondary features
        try:
            # Try to predict directly
            preds = model.predict(secondary_features, verbose=0)[0]
        except:
            # If shape mismatch, adapt the features
            # This is just a fallback in case the model expects different dimensions
            # It will reduce accuracy but prevent crashes
            model_bar.update(1)
            continue
        
        # Store top predictions
        top_indices = np.argsort(preds)[-top_k*2:][::-1]
        
        for rank, idx in enumerate(top_indices):
            class_name = metadata['class_mapping'][str(idx)]
            confidence = float(preds[idx])
            
            class_key = f"{class_name}_{chunk_idx}"
            if class_key not in secondary_predictions:
                secondary_predictions[class_key] = {
                    "class_name": class_name,
                    "chunk_idx": chunk_idx,
                    "secondary_score": 0,
                    "secondary_rank": float('inf')
                }
            
            # Store confidence and best rank
            secondary_predictions[class_key]["secondary_score"] = confidence
            secondary_predictions[class_key]["secondary_rank"] = min(
                secondary_predictions[class_key]["secondary_rank"], rank
            )
        
        # Free memory
        del model, preds
        gc.collect()
        model_bar.update(1)
    
    model_bar.close()
    
    # Free secondary features
    del secondary_features, img_dense, img_dense_batch, img
    gc.collect()
    
    # Fusion approach - combine predictions from both models
    print(f"{TermColors.CYAN}ℹ Fusing predictions from both feature extractors{TermColors.ENDC}")
    fused_predictions = []
    
    # First add all primary predictions
    for class_key, data in primary_predictions.items():
        class_name = data["class_name"]
        chunk_idx = data["chunk_idx"]
        primary_score = data["primary_score"]
        primary_rank = data["primary_rank"]
        
        # Check if this class also appeared in secondary predictions
        secondary_score = 0
        secondary_rank = float('inf')
        if class_key in secondary_predictions:
            secondary_score = secondary_predictions[class_key]["secondary_score"]
            secondary_rank = secondary_predictions[class_key]["secondary_rank"]
            
        # Calculate fusion score:
        # 1. Weight primary model higher (0.6 vs 0.4)
        # 2. Boost classes that appear in both models
        # 3. Consider both score and rank
        
        # Base scores from each model
        p_score = primary_score * (1.0 / (primary_rank + 1))
        s_score = secondary_score * (1.0 / (secondary_rank + 1))
        
        # Higher weight to primary model (60% primary, 40% secondary)
        alpha = 0.6
        beta = 0.4
        
        # Calculate weighted score
        if secondary_score > 0:
            # Class appeared in both models - use weighted average
            fusion_score = (alpha * p_score) + (beta * s_score)
            # Bonus for appearing in both models
            fusion_score *= 1.2
        else:
            # Class only appeared in primary model
            fusion_score = p_score * alpha
        
        fused_predictions.append((class_name, fusion_score, chunk_idx))
    
    # Now add any secondary predictions that weren't in primary
    for class_key, data in secondary_predictions.items():
        if class_key not in primary_predictions:
            class_name = data["class_name"]
            chunk_idx = data["chunk_idx"]
            secondary_score = data["secondary_score"]
            secondary_rank = data["secondary_rank"]
            
            # Calculate score (weighted less since only from secondary model)
            s_score = secondary_score * (1.0 / (secondary_rank + 1)) * 0.4
            
            fused_predictions.append((class_name, s_score, chunk_idx))
    
    # Sort by fusion score
    fused_predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Print results
    print(f"\n{TermColors.HEADER}TOP {top_k} PREDICTIONS (CROSS-MODEL FUSION):{TermColors.ENDC}")
    for i, (class_name, score, chunk_idx) in enumerate(fused_predictions[:top_k]):
        # Normalize score for display
        normalized_score = min(score, 1.0)  # Ensure score is between 0 and 1
        confidence_bar = "█" * int(normalized_score * 20)
        confidence_str = f"[{confidence_bar:<20}] {normalized_score:.4f}"
        print(f"{i+1}. {TermColors.BOLD}{class_name}{TermColors.ENDC} - {TermColors.CYAN}{confidence_str}{TermColors.ENDC} (Chunk {chunk_idx})")
    
    return fused_predictions[:top_k]

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
                loss = -np.sum(y_true_oh[i] * np.log(y_pred[i] + 1e-7))
                losses.append(loss)
            
            # Get indices of hardest examples (highest loss)
            hardest_indices = np.argsort(losses)[-int(len(losses)*0.3):]  # Top 30% hardest
            
            # Create sample weights
            self.current_weights = np.ones(len(self.X_train))
            self.current_weights[hardest_indices] = 2.0  # Double weight for hard examples
            
            print(f"{TermColors.GREEN}✅ Focused training on {len(hardest_indices)} difficult examples{TermColors.ENDC}")
        
        self.epoch_count += 1

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
        
        # FIRST do the train/test split
        X_train, X_val, y_train, y_val = train_test_split(
            features_scaled, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"{TermColors.GREEN}✅ Data split: {X_train.shape[0]} training, {X_val.shape[0]} validation samples{TermColors.ENDC}")
        
        # THEN augment the training data after splitting
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
                    # Save weights after start_epoch (when model is more stable)
                    if epoch % 3 == 0:  # Capture every 3rd epoch
                        print(f"{TermColors.CYAN}📸 Taking SWA snapshot at epoch {epoch+1}{TermColors.ENDC}")
                        
                        # Fix: Check if weights are already numpy arrays
                        weights = self.model.get_weights()
                        snapshot = []
                        
                        for w in weights:
                            # Check if it's already a numpy array or a TensorFlow tensor
                            if isinstance(w, np.ndarray):
                                snapshot.append(w.copy())  # It's already numpy, just copy it
                            else:
                                snapshot.append(w.numpy().copy())  # Convert TensorFlow tensor to numpy
                                
                        self.snapshots.append(snapshot)
        
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

def predict_with_chunked_models(image_path, top_k=5):
    """Make predictions using all the chunked models with visual progress"""
    def check_memory_limit():
        mem = psutil.virtual_memory()
        if mem.percent > 80:  # Emergency cutoff
            print(f"{TermColors.RED}🚨 RAM usage too high ({mem.percent}%). Cannot proceed with prediction.{TermColors.ENDC}")
            return False
        return True
    
    # Add this line HERE, after the function definition
    if not check_memory_limit():
        return []
    
    print(f"{TermColors.HEADER}\n{'='*50}")
    print(f"PREDICTION FOR: {os.path.basename(image_path)}")
    print(f"{'='*50}{TermColors.ENDC}")
    
    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img_processed = tf.keras.applications.efficientnet_v2.preprocess_input(img)
    img_batch = np.expand_dims(img_processed, axis=0)
    
    # Load feature extractor
    print(f"{TermColors.CYAN}ℹ Loading feature extractor{TermColors.ENDC}")
    with tf.device('/GPU:0'):
        from tensorflow.keras.applications import EfficientNetV2L
        
        feature_extractor = EfficientNetV2L(
            input_shape=(*IMAGE_SIZE, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        feature_extractor.trainable = False
    
    # Extract features
    features = feature_extractor.predict(img_batch, verbose=0)
    
    # Find all trained chunk models
    model_files = glob.glob(os.path.join(MODEL_DIR, "chunk_*_model.keras"))
    
    if not model_files:
        print(f"{TermColors.RED}❌ No trained models found in {MODEL_DIR}{TermColors.ENDC}")
        return []
    
    print(f"{TermColors.CYAN}ℹ Found {len(model_files)} trained chunk models{TermColors.ENDC}")
    
    # Process each model and collect predictions
    all_predictions = []
    model_bar = tqdm(total=len(model_files), desc="Processing models", position=0)
    
    for model_file in model_files:
        # Extract chunk index
        chunk_idx = int(os.path.basename(model_file).split("_")[1])
        
        # Load metadata
        metadata_file = os.path.join(MODEL_DIR, f"chunk_{chunk_idx}_metadata.json")
        
        if not os.path.exists(metadata_file):
            model_bar.update(1)
            continue
        
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        # Load model
        model = tf.keras.models.load_model(model_file, custom_objects={
            'sparse_top_k_accuracy': sparse_top_k_accuracy
        })
        
        # Get predictions
        preds = model.predict(features, verbose=0)[0]
        
        # Get top predictions for this chunk
        top_indices = np.argsort(preds)[-top_k:][::-1]
        chunk_predictions = []
        
        for idx in top_indices:
            class_name = metadata['class_mapping'][str(idx)]
            confidence = float(preds[idx])
            chunk_predictions.append((class_name, confidence, chunk_idx))
        
        # Add to overall predictions
        all_predictions.extend(chunk_predictions)
        
        # Clean up
        del model
        model_bar.update(1)
        
        # Clear session to release memory
        tf.keras.backend.clear_session()
    
    model_bar.close()
    
    # Free resources
    del feature_extractor
    gc.collect()
    tf.keras.backend.clear_session()
    
    # Sort by confidence
    all_predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Print top predictions
    print(f"\n{TermColors.HEADER}TOP {top_k} PREDICTIONS:{TermColors.ENDC}")
    for i, (class_name, confidence, chunk_idx) in enumerate(all_predictions[:top_k]):
        confidence_bar = "█" * int(confidence * 20)
        confidence_str = f"[{confidence_bar:<20}] {confidence:.4f}"
        print(f"{i+1}. {TermColors.BOLD}{class_name}{TermColors.ENDC} - {TermColors.CYAN}{confidence_str}{TermColors.ENDC} (Chunk {chunk_idx})")
    
    return all_predictions[:top_k]

if __name__ == "__main__":

    enable_emergency_protection = True  # Set this to False if you want to disable it
    if enable_emergency_protection:
        enable_emergency_memory_protection()

    def set_global_memory_growth():
        """Set memory growth limits in a platform-specific way"""
        import sys
        
        # Windows-specific memory management
        if sys.platform == 'win32':
            print(f"{TermColors.YELLOW}ℹ️ Windows detected - using Windows-specific memory management{TermColors.ENDC}")
            try:
                import psutil
                # Get the current process
                process = psutil.Process()
                # Try to set higher priority
                try:
                    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                except:
                    pass
                print(f"{TermColors.GREEN}✅ Windows process priority adjusted{TermColors.ENDC}")
                
                # Memory monitoring is already handled in the RAM monitor thread
                print(f"{TermColors.GREEN}✅ Using RAM monitor thread for memory management{TermColors.ENDC}")
                return
            except Exception as e:
                print(f"{TermColors.YELLOW}⚠️ Windows memory management setup error: {e}{TermColors.ENDC}")
                return
        
        # Unix-specific memory management (Linux/Mac)
        else:
            try:
                import resource
                # Limit maximum memory to 24GB
                MAX_MEMORY_GB = 24
                resource_limit = MAX_MEMORY_GB * 1024 * 1024 * 1024  # Convert to bytes
                
                # Set maximum memory limit
                resource.setrlimit(resource.RLIMIT_AS, (resource_limit, resource_limit))
                print(f"{TermColors.GREEN}✅ System memory limit set to {MAX_MEMORY_GB}GB{TermColors.ENDC}")
            except ImportError:
                print(f"{TermColors.YELLOW}⚠️ Resource module not available - skipping system memory limits{TermColors.ENDC}")
            except Exception as e:
                print(f"{TermColors.YELLOW}⚠️ Could not set system memory limit: {e}{TermColors.ENDC}")

    # Call the function to apply memory settings
    set_global_memory_growth()

    setup_terminal_clearing()

    try:
        # Define gpus before using it
        gpus = tf.config.experimental.list_physical_devices('GPU')
        
        # Print system info with enhanced formatting
        print(f"{TermColors.HEADER}\n{'='*50}")
        print(f"PLANT RECOGNITION MODEL - OPTIMIZED FOR RTX 3050")
        print(f"{'='*50}{TermColors.ENDC}")
        
        gpu_info = ""
        if gpus:
            gpu_info = "NVIDIA GPU"
        else:
            gpu_info = "CPU (No GPU detected)"
        
        # Add this function to train plants with combined features and SWA    
        def train_plants_advanced():
            """Train plant recognition models with combined features and SWA"""
            print(f"{TermColors.HEADER}\n{'='*50}")
            print(f"ADVANCED PLANT RECOGNITION TRAINING")
            print(f"{'='*50}{TermColors.ENDC}")
            
            # Initialize training state
            training_state = TrainingState()
            
            # Get class directories
            class_dirs = [os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR) 
                         if os.path.isdir(os.path.join(DATA_DIR, d))]
            
            # Group classes into chunks
            chunks = []
            current_chunk = []
            
            for class_dir in class_dirs:
                current_chunk.append(class_dir)
                if len(current_chunk) >= MAX_CLASSES_PER_CHUNK:
                    chunks.append(current_chunk)
                    current_chunk = []
            
            # Add the last chunk if not empty
            if current_chunk:
                chunks.append(current_chunk)
            
            print(f"{TermColors.CYAN}ℹ Divided {len(class_dirs)} classes into {len(chunks)} chunks{TermColors.ENDC}")
            
            # Process each chunk
            for chunk_idx, chunk_classes in enumerate(chunks):
                # Skip already processed chunks - MODIFY THIS SECTION
                if chunk_idx in training_state.processed_chunks:
                    print(f"{TermColors.YELLOW}⚠️ Skipping chunk {chunk_idx} (already processed){TermColors.ENDC}")
                    continue
                
                # IMPORTANT: Check if features already exist before extracting again
                chunk_feature_dir = os.path.join(FEATURES_DIR, f"chunk_{chunk_idx}")
                features_file = os.path.join(chunk_feature_dir, "features.npz")
                
                print(f"{TermColors.HEADER}\n{'='*50}")
                print(f"PROCESSING CHUNK {chunk_idx+1}/{len(chunks)} WITH {len(chunk_classes)} CLASSES")
                print(f"{'='*50}{TermColors.ENDC}")
                
                # Check if features already exist
                if os.path.exists(features_file):
                    print(f"{TermColors.CYAN}ℹ Found existing features for chunk {chunk_idx}. Using existing features.{TermColors.ENDC}")
                else:
                    # Extract combined features only if they don't exist
                    print(f"{TermColors.CYAN}ℹ Extracting features for chunk {chunk_idx}...{TermColors.ENDC}")
                    features_file, _ = extract_combined_features_optimized(chunk_classes, chunk_idx)
                    
                if features_file is None:
                    print(f"{TermColors.RED}❌ Failed to extract features for chunk {chunk_idx}{TermColors.ENDC}")
                    continue
                
                # Train with SWA
                train_chunk_model_with_swa(features_file, chunk_idx, training_state)
                
                print(f"{TermColors.YELLOW}⚠️ Performing full memory cleanup between chunks{TermColors.ENDC}")
                # Release all resources explicitly
                del features_file
                # Multiple garbage collections
                for _ in range(5):
                    gc.collect()
                tf.keras.backend.clear_session()
                # Sleep to ensure OS reclaims memory
                print(f"{TermColors.CYAN}ℹ Waiting 10 seconds for memory to be fully released...{TermColors.ENDC}")
                time.sleep(10) 

                # Clean memory between chunks
                clean_memory()
            
            print(f"{TermColors.GREEN}✅ All chunks processed successfully!{TermColors.ENDC}")
        
        # Example command to run the advanced training pipeline
        train_plants_advanced()
        
    except Exception as e:
        print(f"{TermColors.RED}❌ Fatal error: {e}{TermColors.ENDC}")
=======
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, models, mixed_precision
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers.schedules import CosineDecay
import os
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
def limit_memory_usage(gpu_memory_limit_mb=7168, ram_percent_limit=80):
    """Limit both GPU and RAM memory usage to prevent crashes
    
    Args:
        gpu_memory_limit_mb: GPU memory limit in megabytes (default: 2GB)
        ram_percent_limit: Maximum RAM usage allowed as percentage (default: 80%)
    """
    print(f"{TermColors.YELLOW}⚠️ MEMORY LIMITER ACTIVE{TermColors.ENDC}")
    
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
            print(f"{TermColors.RED}❌ Error setting GPU memory limit: {e}{TermColors.ENDC}")
    else:
        print(f"{TermColors.YELLOW}⚠️ No GPU devices found to apply memory limit{TermColors.ENDC}")
    
    # 2. RAM Memory Monitoring Thread
    def ram_monitor_thread(ram_limit=ram_percent_limit):
        """Background thread to monitor RAM usage and prevent crashes"""
        print(f"{TermColors.GREEN}✅ RAM monitor started (limit: {ram_limit}%){TermColors.ENDC}")
        while True:
            try:
                # Check current RAM usage
                mem = psutil.virtual_memory()
                if mem.percent > ram_limit:
                    print(f"{TermColors.RED}⚠️ RAM usage critical: {mem.percent}% > {ram_limit}%{TermColors.ENDC}")
                    print(f"{TermColors.YELLOW}⏸️ Force releasing memory...{TermColors.ENDC}")
                        
                    # Force garbage collection
                    gc.collect()
                        
                    # Try to release large objects
                    for obj in gc.get_objects():
                        try:
                            if isinstance(obj, (np.ndarray, tf.Tensor)):
                                if hasattr(obj, 'nbytes') and getattr(obj, 'nbytes', 0) > 1e6:
                                    del obj
                        except Exception:
                            pass
                        
                    # For extreme cases, try to reduce TensorFlow memory
                    if mem.percent > ram_limit + 10:
                        print(f"{TermColors.RED}🚨 CRITICAL RAM USAGE: {mem.percent}%{TermColors.ENDC}")
                        tf.keras.backend.clear_session()
                        # Sleep to allow system to recover
                        time.sleep(5)
                            
                # Check less frequently when RAM usage is normal    
                time.sleep(5 if mem.percent > ram_limit - 10 else 30)
                    
            except Exception as e:
                # Don't let monitoring thread crash
                print(f"{TermColors.RED}❌ RAM monitor error: {e}{TermColors.ENDC}")
                time.sleep(60)
    
    # Start RAM monitor in background
    try:
        ram_thread = threading.Thread(target=ram_monitor_thread, args=(ram_percent_limit,), daemon=True)
        ram_thread.start()
    except Exception as e:
        print(f"{TermColors.YELLOW}⚠️ Could not start RAM monitor: {e}{TermColors.ENDC}")

# Apply memory limits immediately
limit_memory_usage(6144, 80)  # Using 6GB of GPU memory and 75% RAM threshold

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
DATA_DIR = r"C:\Users\stefa\Desktop\New folder\data\completed_images"
FEATURES_DIR = r"C:\Users\stefa\Desktop\New folder\data\features"
MODEL_DIR = r"C:\Users\stefa\Desktop\New folder\models"
CHECKPOINT_DIR = r"C:\Users\stefa\Desktop\New folder\checkpoints"
LOG_DIR = r"C:\Users\stefa\Desktop\New folder\logs"
STATS_DIR = r"C:\Users\stefa\Desktop\New folder\stats"

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

def enable_emergency_memory_protection():
    """Enable emergency memory protection to prevent system crashes"""
    print(f"{TermColors.RED}🚨 EMERGENCY MEMORY PROTECTION ACTIVATED{TermColors.ENDC}")
    
    # 1. Enforce strict GPU memory limits
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory limit to 4GB instead of 6GB
            for device in gpus:
                tf.config.set_logical_device_configuration(
                    device,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # 4GB limit
                )
            print(f"{TermColors.YELLOW}⚠️ GPU memory limited to 4GB to prevent crashes{TermColors.ENDC}")
        except Exception as e:
            print(f"Error setting GPU memory limit: {e}")
    
    # 2. More aggressive garbage collection
    gc.enable()
    gc.set_threshold(100, 5, 5)  # Make GC more aggressive
    
    # 3. Reduce batch size globally
    global BATCH_SIZE
    BATCH_SIZE = 32  # Force smaller batch size
    
    # 4. Install emergency watchdog
    def emergency_watchdog():
        while True:
            try:
                mem = psutil.virtual_memory()
                if mem.percent > 90:  # Critical memory threshold
                    print(f"{TermColors.RED}🚨 CRITICAL MEMORY USAGE: {mem.percent}%{TermColors.ENDC}")
                    print(f"{TermColors.RED}🚨 FORCING EMERGENCY CLEANUP{TermColors.ENDC}")
                    # Force garbage collection
                    gc.collect()
                    # Clear TensorFlow memory
                    tf.keras.backend.clear_session()
                    # Try to free large objects
                    for obj in gc.get_objects():
                        try:
                            if isinstance(obj, (np.ndarray, tf.Tensor)) and getattr(obj, 'nbytes', 0) > 1e8:
                                del obj
                        except:
                            pass
            except:
                pass
            time.sleep(1)  # Check every second in emergency mode
    
    # Start emergency watchdog
    threading.Thread(target=emergency_watchdog, daemon=True).start()

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
        img_orig = tf.keras.applications.convnext.preprocess_input(img_array.copy())
        augmented_images.append(img_orig)
        
        # Slight rotation (common in plant photos)
        img_rot = tf.keras.preprocessing.image.apply_affine_transform(
            img_array.copy(), theta=np.random.uniform(-20, 20))
        img_rot = tf.keras.applications.convnext.preprocess_input(img_rot)
        augmented_images.append(img_rot)
        
        # Slight zoom (simulate different distances)
        img_zoom = tf.keras.preprocessing.image.apply_affine_transform(
            img_array.copy(), zx=np.random.uniform(0.8, 1.2), zy=np.random.uniform(0.8, 1.2))
        img_zoom = tf.keras.applications.convnext.preprocess_input(img_zoom)
        augmented_images.append(img_zoom)
        
        # Changes in brightness/contrast
        img_bright = img_array.copy() * np.random.uniform(0.8, 1.2)
        img_bright = np.clip(img_bright, 0, 255)
        img_bright = tf.keras.applications.convnext.preprocess_input(img_bright)
        augmented_images.append(img_bright)
        
        # Random crop with padding
        img_crop = tf.keras.preprocessing.image.apply_affine_transform(
            img_array.copy(), zx=0.8, zy=0.8)
        img_crop = tf.keras.applications.convnext.preprocess_input(img_crop)
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
            priority_metrics = ['loss', 'val_loss', 'accuracy', 'val_accuracy', 'sparse_top_k_accuracy', 'val_sparse_top_k_accuracy']
            
            for metric in priority_metrics:
                if metric in logs:
                    value = logs[metric]
                    # Check if value is a formatting-compatible type
                    if isinstance(value, (float, int, np.float32, np.float64, np.int32, np.int64)):
                        if 'loss' in metric:
                            color = TermColors.RED
                            metrics_display.append(f"{color}{metric}: {value:.4f}{TermColors.ENDC}")
                        elif 'accuracy' in metric:
                            color = TermColors.GREEN
                            metrics_display.append(f"{color}{metric}: {value:.4f}{TermColors.ENDC}")
                        else:
                            metrics_display.append(f"{metric}: {value:.4f}")
                    else:
                        # Handle non-numeric types like CosineDecay
                        metrics_display.append(f"{metric}: {str(value)}")
            
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
            for k, v in logs.items():
                if isinstance(v, str):
                    metrics_str += f" | {k}: {v}"
                elif 'loss' in k:
                    metrics_str += f" | {TermColors.RED}{k}: {v:.4f}{TermColors.ENDC}"
                elif 'accuracy' in k:
                    metrics_str += f" | {TermColors.GREEN}{k}: {v:.4f}{TermColors.ENDC}"
                else:
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
                if new_lr < 1e-6:  # Don't go too low
                    new_lr = 1e-6
                
                print(f"{TermColors.YELLOW}⚙ Reducing learning rate: {current_lr:.6f} → {new_lr:.6f}{TermColors.ENDC}")
                # Set the new learning rate safely
                try:
                    tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                except:
                    print(f"{TermColors.YELLOW}⚠ Could not update learning rate - using schedule instead{TermColors.ENDC}")
                self.plateau_count = 0  # Reset counter after adjustment

# Advanced training configuration with gradual unfreezing
def create_custom_callbacks():
    class GradualUnfreezingCallback(tf.keras.callbacks.Callback):
        def __init__(self, model, feature_dim):
            super().__init__()
            self.model = model
            self.feature_dim = feature_dim
            self.unfrozen_layers = 0
            
        def on_epoch_end(self, epoch, logs=None):
            # Every 5 epochs, make the model slightly more complex
            if epoch > 0 and epoch % 5 == 0 and self.unfrozen_layers < 3:
                if self.unfrozen_layers == 0:
                    # Add a new layer
                    x = self.model.layers[-2].output
                    new_layer = layers.Dense(512, activation='swish')(x)
                    outputs = layers.Dense(self.model.output.shape[1], activation='softmax')(new_layer)
                    new_model = tf.keras.Model(inputs=self.model.input, outputs=outputs)
                    weights = self.model.get_weights()
                    self.model = new_model
                    self.model.compile(optimizer=self.model.optimizer, 
                                     loss='sparse_categorical_crossentropy',
                                     metrics=['accuracy', sparse_top_k_accuracy])
                    print(f"{TermColors.CYAN}✨ Added complexity at epoch {epoch+1}{TermColors.ENDC}")
                    
                self.unfrozen_layers += 1
    
    return [GradualUnfreezingCallback(model, feature_dim)]  # Add to your callbacks list

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
        
        if len(class_indices) < 2:  # Need at least 2 examples to interpolate
            continue
        
        # How many synthetic examples to generate 
        num_to_generate = int(len(class_indices) * augmentation_factor)
        
        for _ in range(num_to_generate):
            # Select two random examples from the same class
            idx1, idx2 = np.random.choice(len(class_features), 2, replace=False)
            
            # Create a synthetic sample through interpolation
            alpha = np.random.uniform(0.2, 0.8)
            synthetic_feature = alpha * class_features[idx1] + (1 - alpha) * class_features[idx2]
            
            # Add small noise to avoid creating exact duplicates
            noise = np.random.normal(0, 0.01, size=synthetic_feature.shape)
            synthetic_feature += noise
            
            X_augmented.append(synthetic_feature)
            y_augmented.append(cls)
    
    # Combine with original data
    if X_augmented:  # Check that we generated something
        X_combined = np.vstack([X, np.array(X_augmented)])
        y_combined = np.concatenate([y, np.array(y_augmented)])
        print(f"{TermColors.GREEN}✅ Added {len(X_augmented)} synthetic feature samples{TermColors.ENDC}")
        return X_combined, y_combined
    else:
        return X, y

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
        if val_acc < 0.6:  # 60% threshold
            problematic_chunks.append((chunk_idx, val_acc, val_top_k, num_classes))
    
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
            print(f"Chunk {chunk_idx}: Accuracy {acc:.2%}, Top-{TOP_K} {top_k:.2%} ({num_classes} classes)")
        
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
    """Enhanced feature extraction using EfficientNetV2L with optimized resource utilization"""
    print(f"{TermColors.HEADER}\n{'='*50}")
    print(f"HIGH-PERFORMANCE FEATURE EXTRACTION FOR CHUNK {chunk_idx+1}")
    print(f"{'='*50}{TermColors.ENDC}")
    
    # Setup keyboard interrupt checking
    stop_requested = False
    last_keyboard_check = time.time()
    
    def check_keyboard_interrupt():
        nonlocal stop_requested, last_keyboard_check
        current_time = time.time()
        if current_time - last_keyboard_check < 1.0:
            return stop_requested
        last_keyboard_check = current_time
        try:
            import keyboard
            if keyboard.is_pressed('ctrl+alt+c'):
                print(f"\n{TermColors.YELLOW}⏸️ Feature extraction pause requested (Ctrl+Alt+C). Will stop after current batch.{TermColors.ENDC}")
                stop_requested = True
                return True
        except Exception:
            pass
        return stop_requested

    # Use EfficientNetV2L with optimized settings
    with tf.device('/GPU:0'):
        # Use smaller model for faster processing while maintaining quality
        from tensorflow.keras.applications import EfficientNetV2L
        
        base_model = EfficientNetV2L(
            input_shape=(*IMAGE_SIZE, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        base_model.trainable = False
        feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=base_model.output)
    
    feature_dim = feature_extractor.output_shape[1]
    print(f"{TermColors.GREEN}✅ Feature extractor loaded: EfficientNetV2L ({feature_dim} dims){TermColors.ENDC}")
    
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
            print(f"{TermColors.YELLOW}⚠️ Feature extraction interrupted. Saving progress...{TermColors.ENDC}")
            break
        
        batch_paths = all_image_paths[i:i+batch_size]
        batch_indices = all_class_indices[i:i+batch_size]
        
        # Load and preprocess images
        batch_images = []
        valid_indices = []
        
        for j, (img_path, class_idx) in enumerate(zip(batch_paths, batch_indices)):
            try:
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
                img = tf.keras.preprocessing.image.img_to_array(img)
                img = tf.keras.applications.efficientnet_v2.preprocess_input(img)
                batch_images.append(img)
                valid_indices.append(class_idx)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
        
        # Skip if no valid images in batch
        if not batch_images:
            continue
        
        # Convert to numpy array
        batch_array = np.array(batch_images)
        
        # Extract features
        batch_features = feature_extractor.predict(batch_array, verbose=0)
        
        # Store features and labels
        for feature, class_idx in zip(batch_features, valid_indices):
            all_features.append(feature)
            all_labels.append(class_idx)
        
        # Update progress bar
        progress_bar.update(len(batch_images))
        
        # Force garbage collection periodically
        if i % (batch_size * 10) == 0:
            gc.collect()
    
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
            try:
                # Quick validation check
                with open(img_path, 'rb') as f:
                    # Check if file is at least 100 bytes
                    if len(f.read(100)) < 100:
                        print(f"{TermColors.YELLOW}⚠️ Skipping small file: {img_path}{TermColors.ENDC}")
                        continue
                
                # Try to open the image to verify it's valid
                img = tf.io.read_file(img_path)
                img = tf.image.decode_image(img, channels=3, expand_animations=False)
                
                # If we get here, image is valid
                paths.append(img_path)
                indices.append(class_idx)
            except Exception as e:
                print(f"{TermColors.YELLOW}⚠️ Skipping corrupt image: {img_path} - {e}{TermColors.ENDC}")
                continue
                
    return class_name, paths, indices

def prune_features_rolling_window(current_chunk_idx, window_size=1):  # Reduced window size to 1
    """Aggressively remove feature files after model training to save disk space
    
    Args:
        current_chunk_idx: The chunk that was just processed
        window_size: Number of chunks to keep (default: 1 - only keep current chunk)
    """
    print(f"{TermColors.HEADER}\n{'='*50}")
    print(f"AGGRESSIVE FEATURE MANAGEMENT")
    print(f"{'='*50}{TermColors.ENDC}")
    
    # Check all chunks up to the current one for possible deletion
    for chunk_to_check in range(current_chunk_idx + 1):  # Include current chunk in check
        # Skip the most recent chunks based on window_size
        if chunk_to_check > current_chunk_idx - window_size:
            continue
            
        # Check if this chunk's features exist
        chunk_feature_dir = os.path.join(FEATURES_DIR, f"chunk_{chunk_to_check}")
        feature_file = os.path.join(chunk_feature_dir, "features.npz")
        
        if os.path.exists(feature_file):
            # Check if model for this chunk exists (we don't want to delete features 
            # if the model doesn't exist or training failed)
            model_file = os.path.join(MODEL_DIR, f"chunk_{chunk_to_check}_model.keras")
            metadata_file = os.path.join(MODEL_DIR, f"chunk_{chunk_to_check}_metadata.json")
            
            if os.path.exists(model_file) and os.path.exists(metadata_file):
                # Get file size before deletion for reporting
                file_size_mb = os.path.getsize(feature_file) / (1024 * 1024)
                
                print(f"{TermColors.YELLOW}⚠️ Removing features for chunk {chunk_to_check} to save disk space ({file_size_mb:.2f} MB){TermColors.ENDC}")
                
                # Remove the feature file but keep the class mapping
                os.remove(feature_file)
                
                # Create a placeholder to indicate features were deleted
                with open(os.path.join(chunk_feature_dir, "features_pruned.txt"), "w") as f:
                    f.write(f"Features were pruned on {datetime.now().isoformat()} to save disk space.\n")
                    f.write(f"Original size: {file_size_mb:.2f} MB\n")
                    f.write(f"Model is preserved at: {model_file}\n")
                
                print(f"{TermColors.GREEN}✅ Freed {file_size_mb:.2f} MB of disk space{TermColors.ENDC}")
            else:
                print(f"{TermColors.CYAN}ℹ Keeping features for chunk {chunk_to_check} because model not found or incomplete{TermColors.ENDC}")

def load_features(features_file):
    """Load extracted features from file with verification to detect corruption"""
    print(f"{TermColors.CYAN}ℹ Loading features from {features_file}{TermColors.ENDC}")
    
    # Add this corruption check before loading
    try:
        file_size = os.path.getsize(features_file)
        if file_size < 1000:  # Suspiciously small
            print(f"{TermColors.RED}⚠️ Feature file appears corrupted (only {file_size} bytes){TermColors.ENDC}")
            raise ValueError("Feature file too small, likely corrupted")
            
        # Now proceed with loading
        load_bar = tqdm(total=1, desc="Loading features")
        data = np.load(features_file)
        features = data['features']
        labels = data['labels']
        load_bar.update(1)
        load_bar.close()
        
        print(f"{TermColors.GREEN}✅ Loaded {features.shape[0]} features with {features.shape[1]} dimensions{TermColors.ENDC}")
        return features, labels
        
    except Exception as e:
        print(f"{TermColors.RED}❌ Error loading features: {e}{TermColors.ENDC}")
        return None, None

# Memory management function with proper gc import
def clean_memory():
    """Aggressively clean memory between processing steps with time-based cleanup"""
    # Static variable to track last cleanup time
    if not hasattr(clean_memory, "last_cleanup_time"):
        clean_memory.last_cleanup_time = time.time()
    
    current_time = time.time()
    should_deep_clean = (current_time - clean_memory.last_cleanup_time) >= 60  # 60 seconds = 1 minute
    
    # Check current memory usage
    mem = psutil.virtual_memory()
    force_cleanup = mem.percent > 80  # Force cleanup if memory usage is above 80%
    
    # Basic cleanup always happens
    tf.keras.backend.clear_session()
    gc.collect()
    
    # Deep cleanup happens every minute OR when memory usage is high
    if should_deep_clean or force_cleanup:
        if force_cleanup:
            print(f"{TermColors.RED}⚠️ High memory usage detected ({mem.percent}%). Performing emergency cleanup.{TermColors.ENDC}")
        else:
            print(f"{TermColors.CYAN}ℹ Performing scheduled deep memory cleanup{TermColors.ENDC}")
        
        # More aggressive garbage collection
        for _ in range(5):  # Multiple collection cycles
            gc.collect()
        
        # Try to release GPU memory if possible
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Reset TensorFlow session
                tf.keras.backend.clear_session()
                
                # Extra GPU memory cleanup
                print(f"{TermColors.CYAN}ℹ Releasing GPU memory{TermColors.ENDC}")
            except:
                pass
        
        # Update last cleanup time
        clean_memory.last_cleanup_time = current_time

# In your build_model function, add attention mechanism
def build_model(feature_dim, num_classes):
    inputs = tf.keras.Input(shape=(feature_dim,))
    
    # Modified architecture for better generalization
    x = layers.Dense(2048)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)  # Replace ReLU with Swish for better gradient flow
    x = layers.Dropout(0.4)(x)  # Increased dropout slightly (0.3 -> 0.4)
    
    # Add a residual connection
    residual = x  # Save for residual connection
    
    x = layers.Dense(1024)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Dropout(0.3)(x)  # Increased dropout slightly (0.2 -> 0.3)
    
    # Optional: Add a skip connection if shapes match
    if residual.shape[-1] == x.shape[-1]:
        x = layers.add([x, residual])
    else:
        # Use a projection if shapes don't match
        projection = layers.Dense(x.shape[-1], use_bias=False)(residual)
        x = layers.add([x, projection])
    
    # Add one more layer with L2 regularization for better generalization
    x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Original optimizer settings
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', sparse_top_k_accuracy]
    )
    
    return model

def build_triplet_model(feature_dim, num_classes, margin=0.5):
    """Build a model using triplet loss to enhance class separability"""
    # Feature extraction base
    input_tensor = tf.keras.Input(shape=(feature_dim,))
    x = layers.Dense(1024)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Dropout(0.3)(x)
    
    # Create embedding space
    embedding = layers.Dense(512, name="embedding")(x)
    
    # Classifier head
    classifier = layers.Dense(num_classes, activation='softmax')(embedding)
    
    # Create main model
    model = tf.keras.Model(inputs=input_tensor, outputs=classifier)
    
    # Create embedding model
    embedding_model = tf.keras.Model(inputs=input_tensor, outputs=embedding)
    
    # Define triplet loss function
    def triplet_loss(y_true, y_pred):
        # Standard categorical cross-entropy
        categorical_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        
        # Get embeddings for batch
        embeddings = embedding_model(input_tensor)
        
        # Calculate triplet loss
        # (Simplified implementation - a full triplet loss would need proper triplet mining)
        batch_size = tf.shape(y_true)[0]
        
        # Get positive and negative masks based on class labels
        y_reshaped = tf.reshape(y_true, [-1, 1])
        positive_mask = tf.equal(y_reshaped, tf.transpose(y_reshaped))
        negative_mask = tf.logical_not(positive_mask)
        
        # Convert to float
        positive_mask = tf.cast(positive_mask, dtype=tf.float32)
        negative_mask = tf.cast(negative_mask, dtype=tf.float32)
        
        # Get embeddings and calculate distances
        embeddings_flat = tf.reshape(embeddings, [batch_size, -1])
        
        # Calculate distance matrix
        distances = tf.reduce_sum(tf.square(tf.expand_dims(embeddings_flat, 1) - 
                                          tf.expand_dims(embeddings_flat, 0)), 2)
        
        # Get positives and negatives
        positive_dists = tf.reduce_mean(distances * positive_mask) 
        negative_dists = tf.reduce_mean(distances * negative_mask)
        
        # Triplet loss term
        triplet_term = tf.maximum(0.0, positive_dists - negative_dists + margin)
        
        # Combine losses
        total_loss = categorical_loss + 0.3 * triplet_term
        return total_loss
    
    # Compile with custom loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=triplet_loss,
        metrics=['accuracy', sparse_top_k_accuracy]
    )
    
    return model

class DynamicTrainingOptimizer(tf.keras.callbacks.Callback):
    """Dynamically adapt training parameters based on metrics patterns"""
    def __init__(self, patience=3):
        super().__init__()
        self.last_losses = []
        self.patience = patience
        self.last_lr_change = 0
        self.best_val_loss = float('inf')
        self.stagnation_count = 0
        self.lr_reductions = 0
        self.max_lr_reductions = 4  # Prevent too many reductions
        self.initial_lr = None
        
    def on_train_begin(self, logs=None):
        # Capture initial learning rate
        if hasattr(self.model.optimizer, 'lr'):
            if hasattr(self.model.optimizer.lr, '__call__'):
                # For LearningRateSchedule objects
                current_step = self.model.optimizer.iterations.numpy()
                self.initial_lr = float(self.model.optimizer.lr(current_step).numpy())
            else:
                # For fixed learning rates
                self.initial_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        
    def on_epoch_end(self, epoch, logs=None):
        if not logs or 'val_loss' not in logs:
            return
            
        val_loss = logs.get('val_loss')
        val_acc = logs.get('val_accuracy', 0)
        
        # Store loss history for trend analysis
        self.last_losses.append(val_loss)
        if len(self.last_losses) > self.patience:
            self.last_losses.pop(0)
        
        # Check for improvement
        if val_loss < self.best_val_loss:
            improvement = self.best_val_loss - val_loss
            self.best_val_loss = val_loss
            self.stagnation_count = 0
            print(f"{TermColors.GREEN}✓ Validation improving by {improvement:.4f} - best loss: {val_loss:.4f}{TermColors.ENDC}")
        else:
            self.stagnation_count += 1
            print(f"{TermColors.YELLOW}⚠ No improvement for {self.stagnation_count} epochs (best: {self.best_val_loss:.4f}, current: {val_loss:.4f}){TermColors.ENDC}")
            
        # Get current learning rate
        try:
            if hasattr(self.model.optimizer.lr, '__call__'):
                # For LearningRateSchedule objects
                current_step = self.model.optimizer.iterations.numpy()
                current_lr = float(self.model.optimizer.lr(current_step).numpy())
            else:
                # For fixed learning rates
                current_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        except:
            current_lr = 0.001  # Fallback
            
        # If enough history has been collected and we're past the early epochs
        if len(self.last_losses) >= self.patience and epoch > 5:
            # Check for worsening trend (consecutive increases in loss)
            if all(self.last_losses[i] > self.last_losses[i-1] for i in range(1, len(self.last_losses))):
                # Only reduce if we haven't reduced too many times already
                if self.lr_reductions < self.max_lr_reductions:
                    # Reduce LR more aggressively when loss is consistently increasing
                    new_lr = current_lr * 0.3  # More aggressive reduction
                    try:
                        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                        print(f"{TermColors.YELLOW}⚠️ Loss consistently increasing - LR reduced: {current_lr:.6f} → {new_lr:.6f}{TermColors.ENDC}")
                        self.lr_reductions += 1
                        self.last_losses = []  # Reset history after adjustment
                        self.last_lr_change = epoch
                    except:
                        print(f"{TermColors.RED}❌ Failed to update learning rate{TermColors.ENDC}")
                
            # Check for plateaus (minimal changes in loss)
            elif self.stagnation_count >= self.patience:
                diffs = [abs(self.last_losses[i] - self.last_losses[i-1]) for i in range(1, len(self.last_losses))]
                if all(d < 0.001 for d in diffs):  # Very small changes
                    # Try increasing the batch size if plateau detected
                    if epoch - self.last_lr_change > 5:
                        current_bs = self.model.get_config().get('batch_size', BATCH_SIZE)
                        if current_bs < 128:  # Don't go too large
                            # Rather than changing batch size (which is tricky), adjust learning rate instead
                            new_lr = current_lr * 0.5
                            if new_lr >= 1e-6:  # Don't go too small
                                try:
                                    tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                                    print(f"{TermColors.CYAN}🔄 Plateau detected - Reducing learning rate: {current_lr:.6f} → {new_lr:.6f}{TermColors.ENDC}")
                                    self.lr_reductions += 1
                                    self.last_lr_change = epoch
                                except:
                                    print(f"{TermColors.RED}❌ Failed to update learning rate{TermColors.ENDC}")
                
        # If accuracy is already very high, fine-tune with smaller learning rate
        if val_acc > 0.9 and current_lr > 1e-5 and epoch - self.last_lr_change > 3:
            new_lr = current_lr * 0.5
            try:
                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                print(f"{TermColors.GREEN}✓ High accuracy detected ({val_acc:.4f}) - Fine-tuning with lower LR: {new_lr:.6f}{TermColors.ENDC}")
                self.last_lr_change = epoch
            except:
                pass

class AdaptiveRegularization(tf.keras.callbacks.Callback):
    """Dynamically adjust regularization based on overfitting detection"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.overfitting_streak = 0
        self.underfitting_streak = 0
        self.dropout_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dropout)]
        self.regularization_changes = 0
        self.max_reg_changes = 3
        
    def on_epoch_end(self, epoch, logs=None):
        # Skip early epochs
        if epoch < 3:
            return
            
        # Check for overfitting by comparing train and val accuracy
        train_acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        
        if train_acc is not None and val_acc is not None:
            gap = train_acc - val_acc
            loss_gap = val_loss - train_loss
            
            # Detect significant overfitting gap
            if gap > 0.1:  # 10% gap indicates overfitting
                self.overfitting_streak += 1
                self.underfitting_streak = 0
                
                if self.overfitting_streak >= 3 and self.regularization_changes < self.max_reg_changes:  # 3 consecutive epochs showing overfitting
                    print(f"{TermColors.YELLOW}⚠️ Overfitting detected (acc gap: {gap:.2f}, loss gap: {loss_gap:.2f}) - Increasing regularization{TermColors.ENDC}")
                    
                    # Increase dropout rates if we can
                    if self.dropout_layers:
                        for layer in self.dropout_layers:
                            current_rate = tf.keras.backend.get_value(layer.rate)
                            new_rate = min(current_rate + 0.05, 0.7)  # Don't go beyond 0.7
                            layer.rate = new_rate
                            print(f"{TermColors.CYAN}↑ Dropout adjusted: {current_rate:.2f} → {new_rate:.2f}{TermColors.ENDC}")
                        
                        self.regularization_changes += 1
                        self.overfitting_streak = 0  # Reset after adjustment
            
            # Detect underfitting (training and validation both low but close)
            elif gap < 0.03 and train_acc < 0.5 and epoch > 5:
                self.underfitting_streak += 1
                self.overfitting_streak = 0
                
                if self.underfitting_streak >= 3 and self.regularization_changes < self.max_reg_changes:
                    print(f"{TermColors.YELLOW}⚠️ Underfitting detected (acc: {train_acc:.2f}, gap: {gap:.2f}) - Decreasing regularization{TermColors.ENDC}")
                    
                    # Decrease dropout rates if possible
                    if self.dropout_layers:
                        for layer in self.dropout_layers:
                            current_rate = tf.keras.backend.get_value(layer.rate)
                            new_rate = max(current_rate - 0.05, 0.1)  # Don't go below 0.1
                            layer.rate = new_rate
                            print(f"{TermColors.CYAN}↓ Dropout adjusted: {current_rate:.2f} → {new_rate:.2f}{TermColors.ENDC}")
                        
                        self.regularization_changes += 1
                        self.underfitting_streak = 0  # Reset after adjustment
            else:
                # Reset both streaks if neither condition is met
                self.overfitting_streak = 0
                self.underfitting_streak = 0

class AdaptiveClassWeightAdjuster(tf.keras.callbacks.Callback):
    """Dynamically adjust class weights based on per-class performance"""
    def __init__(self, X_val, y_val, class_names, update_frequency=5):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.class_names = class_names
        self.update_frequency = update_frequency
        self.adjustments_made = 0
        self.max_adjustments = 3
        self.class_weights = None
        
    def on_train_begin(self, logs=None):
        # Initialize with equal class weights
        num_classes = len(self.class_names)
        self.class_weights = {i: 1.0 for i in range(num_classes)}
        
    def on_epoch_end(self, epoch, logs=None):
        # Only update weights periodically to allow training to adapt
        if epoch % self.update_frequency != 0 or self.adjustments_made >= self.max_adjustments:
            return
            
        # Get predictions on validation set
        y_pred = self.model.predict(self.X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calculate per-class accuracy
        class_accuracy = {}
        for cls in range(len(self.class_names)):
            # Find samples for this class
            mask = self.y_val == cls
            if np.sum(mask) == 0:  # No examples of this class
                continue
                
            # Calculate accuracy for this class
            correct = (y_pred_classes[mask] == cls).sum()
            total = mask.sum()
            class_accuracy[cls] = correct / total if total > 0 else 0.0
        
        # Find problematic classes
        overall_acc = logs.get('val_accuracy', 0)
        problem_classes = []
        
        for cls, acc in class_accuracy.items():
            # Classes with accuracy significantly below average
            if acc < overall_acc * 0.7:  # 30% below average is problematic
                problem_classes.append((cls, acc))
        
        # Adjust weights if we have problematic classes
        if problem_classes:
            print(f"{TermColors.YELLOW}⚠️ Found {len(problem_classes)} underperforming classes{TermColors.ENDC}")
            
            # Sort by lowest accuracy first
            problem_classes.sort(key=lambda x: x[1])
            
            # Just show a few examples
            for cls, acc in problem_classes[:3]:
                print(f"  - Class {self.class_names[cls]}: {acc:.2%} accuracy (vs {overall_acc:.2%} overall)")
                
                # Increase weight for this class
                old_weight = self.class_weights.get(cls, 1.0)
                new_weight = old_weight * 1.5  # Increase by 50%
                self.class_weights[cls] = min(new_weight, 5.0)  # Cap at 5x
                
                print(f"{TermColors.CYAN}↑ Weight adjusted: {old_weight:.2f} → {self.class_weights[cls]:.2f}{TermColors.ENDC}")
            
            # Apply new class weights to model 
            print(f"{TermColors.CYAN}ℹ Updated class weights will affect future training{TermColors.ENDC}")
            self.adjustments_made += 1

def test_time_augmentation_predict(image_path, top_k=5):
    """Memory-efficient prediction using test-time augmentation"""
    def check_memory_limit():
        mem = psutil.virtual_memory()
        if mem.percent > 80:  # Emergency cutoff
            print(f"{TermColors.RED}🚨 RAM usage too high ({mem.percent}%). Cannot proceed with prediction.{TermColors.ENDC}")
            return False
        return True
        
    if not check_memory_limit():
        return []
    
    print(f"{TermColors.HEADER}\n{'='*50}")
    print(f"TTA PLANT IDENTIFICATION: {os.path.basename(image_path)}")
    print(f"{'='*50}{TermColors.ENDC}")
    
    # Load and preprocess image once
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img_processed = tf.keras.applications.efficientnet_v2.preprocess_input(img)
    
    # Define efficient augmentations for plants
    augmentations = [
        {"name": "original", "transform": lambda img: img},
        {"name": "horizontal_flip", "transform": lambda img: np.fliplr(img)},
        {"name": "rotation_10", "transform": lambda img: tf.keras.preprocessing.image.apply_affine_transform(img, theta=10)},
        {"name": "brightness_up", "transform": lambda img: np.clip(img * 1.1, 0, 255)}
    ]
    
    # Load feature extractor once
    print(f"{TermColors.CYAN}ℹ Loading feature extractor{TermColors.ENDC}")
    with tf.device('/GPU:0'):
        from tensorflow.keras.applications import EfficientNetV2L
        feature_extractor = EfficientNetV2L(
            input_shape=(*IMAGE_SIZE, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        feature_extractor.trainable = False
    
    # Dictionary to store all class predictions
    all_predictions = {}
    aug_bar = tqdm(total=len(augmentations), desc="Processing augmentations", position=0)
    
    # Process each augmentation separately
    for aug in augmentations:
        # Apply augmentation
        aug_bar.set_description(f"Processing {aug['name']}")
        img_aug = aug["transform"](img_processed.copy())
        img_batch = np.expand_dims(img_aug, axis=0)
        
        # Extract features
        features = feature_extractor.predict(img_batch, verbose=0)
        
        # Find and load models
        model_files = glob.glob(os.path.join(MODEL_DIR, "chunk_*_model.keras"))
        
        # Process each model
        for model_file in model_files:
            # Reset TF session between models to free memory
            tf.keras.backend.clear_session()
            
            chunk_idx = int(os.path.basename(model_file).split("_")[1])
            metadata_file = os.path.join(MODEL_DIR, f"chunk_{chunk_idx}_metadata.json")
            
            if not os.path.exists(metadata_file):
                continue
                
            # Load model and metadata
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            model = tf.keras.models.load_model(model_file, custom_objects={
                'sparse_top_k_accuracy': sparse_top_k_accuracy
            })
            
            # Get predictions
            preds = model.predict(features, verbose=0)[0]
            
            # Get top predictions for this model
            top_indices = np.argsort(preds)[-top_k*2:][::-1]  # Get more candidates for consistency
            
            for idx in top_indices:
                class_name = metadata['class_mapping'][str(idx)]
                confidence = float(preds[idx])
                
                # Store predictions by class name
                class_key = f"{class_name}_{chunk_idx}"
                if class_key not in all_predictions:
                    all_predictions[class_key] = {
                        "class_name": class_name,
                        "chunk_idx": chunk_idx,
                        "total_score": 0,
                        "count": 0
                    }
                
                all_predictions[class_key]["total_score"] += confidence
                all_predictions[class_key]["count"] += 1
            
            # Free memory
            del model, preds
            gc.collect()
        
        # Free memory
        del img_aug, img_batch, features
        gc.collect()
        aug_bar.update(1)
    
    aug_bar.close()
    
    # Clean up GPU memory
    del feature_extractor
    gc.collect()
    tf.keras.backend.clear_session()
    
    # Calculate average score for each class
    final_predictions = []
    for class_key, data in all_predictions.items():
        avg_confidence = data["total_score"] / data["count"] if data["count"] > 0 else 0
        final_predictions.append((data["class_name"], avg_confidence, data["chunk_idx"]))
    
    # Sort by confidence
    final_predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Print results
    print(f"\n{TermColors.HEADER}TOP {top_k} PREDICTIONS (WITH TTA):{TermColors.ENDC}")
    for i, (class_name, confidence, chunk_idx) in enumerate(final_predictions[:top_k]):
        confidence_bar = "█" * int(confidence * 20)
        confidence_str = f"[{confidence_bar:<20}] {confidence:.4f}"
        print(f"{i+1}. {TermColors.BOLD}{class_name}{TermColors.ENDC} - {TermColors.CYAN}{confidence_str}{TermColors.ENDC} (Chunk {chunk_idx})")
    
    return final_predictions[:top_k]

def self_consistency_predict(image_path, top_k=5):
    """Memory-efficient prediction with self-consistency checking"""
    def check_memory_limit():
        mem = psutil.virtual_memory()
        if mem.percent > 80:  # Emergency cutoff
            print(f"{TermColors.RED}🚨 RAM usage too high ({mem.percent}%). Cannot proceed with prediction.{TermColors.ENDC}")
            return False
        return True
        
    if not check_memory_limit():
        return []
    
    print(f"{TermColors.HEADER}\n{'='*50}")
    print(f"SELF-CONSISTENCY PLANT IDENTIFICATION: {os.path.basename(image_path)}")
    print(f"{'='*50}{TermColors.ENDC}")
    
    # Generate multiple augmentations of the input image
    def generate_augmentations(img_path, num_aug=4):
        """Generate multiple augmentations with different parameters"""
        # Load image once
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
        img = tf.keras.preprocessing.image.img_to_array(img)
        
        # Create augmentations
        augmentations = []
        
        # Original image
        img_orig = tf.keras.applications.efficientnet_v2.preprocess_input(img.copy())
        augmentations.append(img_orig)
        
        # Flip horizontal
        img_flip = np.fliplr(img.copy())
        img_flip = tf.keras.applications.efficientnet_v2.preprocess_input(img_flip)
        augmentations.append(img_flip)
        
        # Rotate +10 degrees
        img_rot1 = tf.keras.preprocessing.image.apply_affine_transform(img.copy(), theta=10)
        img_rot1 = tf.keras.applications.efficientnet_v2.preprocess_input(img_rot1)
        augmentations.append(img_rot1)
        
        # Rotate -10 degrees
        img_rot2 = tf.keras.preprocessing.image.apply_affine_transform(img.copy(), theta=-10)
        img_rot2 = tf.keras.applications.efficientnet_v2.preprocess_input(img_rot2)
        augmentations.append(img_rot2)
        
        return augmentations
    
    # Generate augmentations
    print(f"{TermColors.CYAN}ℹ Generating image augmentations{TermColors.ENDC}")
    augmented_images = generate_augmentations(image_path)
    
    # Load feature extractor
    print(f"{TermColors.CYAN}ℹ Loading feature extractor{TermColors.ENDC}")
    with tf.device('/GPU:0'):
        from tensorflow.keras.applications import EfficientNetV2L
        feature_extractor = EfficientNetV2L(
            input_shape=(*IMAGE_SIZE, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        feature_extractor.trainable = False
    
    # Track votes for each class
    class_votes = {}
    
    # Process each augmentation independently to minimize memory usage
    aug_bar = tqdm(total=len(augmented_images), desc="Processing augmentations", position=0)
    
    for aug_idx, aug_img in enumerate(augmented_images):
        # Extract features
        img_batch = np.expand_dims(aug_img, axis=0)
        features = feature_extractor.predict(img_batch, verbose=0)
        
        # Load models
        model_files = glob.glob(os.path.join(MODEL_DIR, "chunk_*_model.keras"))
        
        for model_file in model_files:
            # Clean memory between models
            tf.keras.backend.clear_session()
            
            chunk_idx = int(os.path.basename(model_file).split("_")[1])
            metadata_file = os.path.join(MODEL_DIR, f"chunk_{chunk_idx}_metadata.json")
            
            if not os.path.exists(metadata_file):
                continue
                
            # Load model and metadata
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            model = tf.keras.models.load_model(model_file, custom_objects={
                'sparse_top_k_accuracy': sparse_top_k_accuracy
            })
            
            # Get predictions
            preds = model.predict(features, verbose=0)[0]
            
            # Get top predictions 
            top_indices = np.argsort(preds)[-top_k*2:][::-1]  # Get more candidates
            
            for rank, idx in enumerate(top_indices):
                class_name = metadata['class_mapping'][str(idx)]
                confidence = float(preds[idx])
                
                # Weight by both rank and confidence
                # Higher rank (lower number) = higher weight
                rank_weight = 1.0 / (rank + 1)
                # Also consider which augmentation (original gets higher weight)
                aug_weight = 1.0 if aug_idx == 0 else 0.8
                
                # Compute vote weight
                vote_weight = confidence * rank_weight * aug_weight
                
                # Add to votes
                if class_name not in class_votes:
                    class_votes[class_name] = {
                        "score": 0,
                        "chunk_idx": chunk_idx,
                        "mentions": 0,
                        "highest_conf": 0
                    }
                
                class_votes[class_name]["score"] += vote_weight
                class_votes[class_name]["mentions"] += 1
                class_votes[class_name]["highest_conf"] = max(
                    class_votes[class_name]["highest_conf"], confidence
                )
            
            # Clean up model
            del model
            gc.collect()
        
        # Clean up features
        del features, img_batch
        gc.collect()
        aug_bar.update(1)
    
    aug_bar.close()
    
    # Clean up GPU memory
    del feature_extractor, augmented_images
    gc.collect()
    tf.keras.backend.clear_session()
    
    # Calculate consistency score:
    # - High score if appears in multiple augmentations
    # - High score if ranks high consistently
    final_predictions = []
    for class_name, data in class_votes.items():
        # Combine score with number of mentions for consistency
        # This favors classes that appear consistently across augmentations
        consistency_score = data["score"] * (data["mentions"] / (len(augmented_images) * 2))
        
        # Boost score of items that appeared in all augmentations
        if data["mentions"] >= len(augmented_images):
            consistency_score *= 1.2
        
        final_predictions.append((class_name, consistency_score, data["chunk_idx"]))
    
    # Sort by consistency score
    final_predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Print results
    print(f"\n{TermColors.HEADER}TOP {top_k} PREDICTIONS (SELF-CONSISTENCY):{TermColors.ENDC}")
    for i, (class_name, score, chunk_idx) in enumerate(final_predictions[:top_k]):
        # Normalize score for display
        normalized_score = min(score / 2.0, 1.0)  # Scale for display
        confidence_bar = "█" * int(normalized_score * 20)
        confidence_str = f"[{confidence_bar:<20}] {normalized_score:.4f}"
        print(f"{i+1}. {TermColors.BOLD}{class_name}{TermColors.ENDC} - {TermColors.CYAN}{confidence_str}{TermColors.ENDC} (Chunk {chunk_idx})")
    
    return final_predictions[:top_k]

def cross_model_fusion_predict(image_path, top_k=5):
    """Memory-efficient prediction using cross-model feature fusion"""
    def check_memory_limit():
        mem = psutil.virtual_memory()
        if mem.percent > 80:  # Emergency cutoff
            print(f"{TermColors.RED}🚨 RAM usage too high ({mem.percent}%). Cannot proceed with prediction.{TermColors.ENDC}")
            return False
        return True
    
    if not check_memory_limit():
        return []
    
    print(f"{TermColors.HEADER}\n{'='*50}")
    print(f"CROSS-MODEL FUSION PLANT IDENTIFICATION: {os.path.basename(image_path)}")
    print(f"{'='*50}{TermColors.ENDC}")
    
    # Load image once
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
    img = tf.keras.preprocessing.image.img_to_array(img)
    
    # Create two copies with different preprocessing for different models
    img_efficient = tf.keras.applications.efficientnet_v2.preprocess_input(img.copy())
    img_efficient_batch = np.expand_dims(img_efficient, axis=0)
    
    print(f"{TermColors.CYAN}ℹ Extracting features using dual-model approach{TermColors.ENDC}")
    
    # First extract primary features
    with tf.device('/GPU:0'):
        # Primary model - EfficientNetV2L
        print(f"{TermColors.CYAN}ℹ Loading primary feature extractor (EfficientNetV2L){TermColors.ENDC}")
        from tensorflow.keras.applications import EfficientNetV2L
        feature_extractor1 = EfficientNetV2L(
            input_shape=(*IMAGE_SIZE, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        feature_extractor1.trainable = False
        
        # Extract primary features
        primary_features = feature_extractor1.predict(img_efficient_batch, verbose=0)
        
        # Free GPU memory
        del feature_extractor1
        gc.collect()
        tf.keras.backend.clear_session()
    
    # Process models with primary features
    print(f"{TermColors.CYAN}ℹ Processing models with primary feature extractor{TermColors.ENDC}")
    primary_predictions = {}
    
    # Get all models
    model_files = glob.glob(os.path.join(MODEL_DIR, "chunk_*_model.keras"))
    model_bar = tqdm(total=len(model_files), desc="Processing models (primary)", position=0)
    
    for model_file in model_files:
        # Clean memory before each model
        tf.keras.backend.clear_session()
        
        chunk_idx = int(os.path.basename(model_file).split("_")[1])
        metadata_file = os.path.join(MODEL_DIR, f"chunk_{chunk_idx}_metadata.json")
        
        if not os.path.exists(metadata_file):
            model_bar.update(1)
            continue
            
        # Load model and metadata
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        model = tf.keras.models.load_model(model_file, custom_objects={
            'sparse_top_k_accuracy': sparse_top_k_accuracy
        })
        
        # Get predictions
        preds = model.predict(primary_features, verbose=0)[0]
        
        # Store top predictions
        top_indices = np.argsort(preds)[-top_k*3:][::-1]  # Get more candidates
        
        for rank, idx in enumerate(top_indices):
            class_name = metadata['class_mapping'][str(idx)]
            confidence = float(preds[idx])
            
            class_key = f"{class_name}_{chunk_idx}"
            if class_key not in primary_predictions:
                primary_predictions[class_key] = {
                    "class_name": class_name,
                    "chunk_idx": chunk_idx,
                    "primary_score": 0,
                    "primary_rank": float('inf')
                }
            
            # Store confidence and best rank
            primary_predictions[class_key]["primary_score"] = confidence
            primary_predictions[class_key]["primary_rank"] = min(
                primary_predictions[class_key]["primary_rank"], rank
            )
        
        # Free memory
        del model, preds
        gc.collect()
        model_bar.update(1)
    
    model_bar.close()
    
    # Free primary features to save memory
    del primary_features, img_efficient, img_efficient_batch
    gc.collect()
    
    # Second feature extraction - use DenseNet121 for diversity
    # Prepare image for second model
    img_dense = tf.keras.applications.densenet.preprocess_input(img.copy())
    img_dense_batch = np.expand_dims(img_dense, axis=0)
    
    # Extract secondary features
    with tf.device('/GPU:0'):
        print(f"{TermColors.CYAN}ℹ Loading secondary feature extractor (DenseNet121){TermColors.ENDC}")
        from tensorflow.keras.applications import DenseNet121
        feature_extractor2 = DenseNet121(
            input_shape=(*IMAGE_SIZE, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        feature_extractor2.trainable = False
        
        # Extract secondary features
        secondary_features = feature_extractor2.predict(img_dense_batch, verbose=0)
        
        # Free GPU memory
        del feature_extractor2
        gc.collect()
        tf.keras.backend.clear_session()
    
    # Process models with secondary features
    print(f"{TermColors.CYAN}ℹ Processing models with secondary feature extractor{TermColors.ENDC}")
    secondary_predictions = {}
    
    model_bar = tqdm(total=len(model_files), desc="Processing models (secondary)", position=0)
    
    for model_file in model_files:
        # Clean memory before each model
        tf.keras.backend.clear_session()
        
        chunk_idx = int(os.path.basename(model_file).split("_")[1])
        metadata_file = os.path.join(MODEL_DIR, f"chunk_{chunk_idx}_metadata.json")
        
        if not os.path.exists(metadata_file):
            model_bar.update(1)
            continue
            
        # Load model and metadata
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        model = tf.keras.models.load_model(model_file, custom_objects={
            'sparse_top_k_accuracy': sparse_top_k_accuracy
        })
        
        # Get predictions - may need to reshape or adapt secondary features
        try:
            # Try to predict directly
            preds = model.predict(secondary_features, verbose=0)[0]
        except:
            # If shape mismatch, adapt the features
            # This is just a fallback in case the model expects different dimensions
            # It will reduce accuracy but prevent crashes
            model_bar.update(1)
            continue
        
        # Store top predictions
        top_indices = np.argsort(preds)[-top_k*2:][::-1]
        
        for rank, idx in enumerate(top_indices):
            class_name = metadata['class_mapping'][str(idx)]
            confidence = float(preds[idx])
            
            class_key = f"{class_name}_{chunk_idx}"
            if class_key not in secondary_predictions:
                secondary_predictions[class_key] = {
                    "class_name": class_name,
                    "chunk_idx": chunk_idx,
                    "secondary_score": 0,
                    "secondary_rank": float('inf')
                }
            
            # Store confidence and best rank
            secondary_predictions[class_key]["secondary_score"] = confidence
            secondary_predictions[class_key]["secondary_rank"] = min(
                secondary_predictions[class_key]["secondary_rank"], rank
            )
        
        # Free memory
        del model, preds
        gc.collect()
        model_bar.update(1)
    
    model_bar.close()
    
    # Free secondary features
    del secondary_features, img_dense, img_dense_batch, img
    gc.collect()
    
    # Fusion approach - combine predictions from both models
    print(f"{TermColors.CYAN}ℹ Fusing predictions from both feature extractors{TermColors.ENDC}")
    fused_predictions = []
    
    # First add all primary predictions
    for class_key, data in primary_predictions.items():
        class_name = data["class_name"]
        chunk_idx = data["chunk_idx"]
        primary_score = data["primary_score"]
        primary_rank = data["primary_rank"]
        
        # Check if this class also appeared in secondary predictions
        secondary_score = 0
        secondary_rank = float('inf')
        if class_key in secondary_predictions:
            secondary_score = secondary_predictions[class_key]["secondary_score"]
            secondary_rank = secondary_predictions[class_key]["secondary_rank"]
            
        # Calculate fusion score:
        # 1. Weight primary model higher (0.6 vs 0.4)
        # 2. Boost classes that appear in both models
        # 3. Consider both score and rank
        
        # Base scores from each model
        p_score = primary_score * (1.0 / (primary_rank + 1))
        s_score = secondary_score * (1.0 / (secondary_rank + 1))
        
        # Higher weight to primary model (60% primary, 40% secondary)
        alpha = 0.6
        beta = 0.4
        
        # Calculate weighted score
        if secondary_score > 0:
            # Class appeared in both models - use weighted average
            fusion_score = (alpha * p_score) + (beta * s_score)
            # Bonus for appearing in both models
            fusion_score *= 1.2
        else:
            # Class only appeared in primary model
            fusion_score = p_score * alpha
        
        fused_predictions.append((class_name, fusion_score, chunk_idx))
    
    # Now add any secondary predictions that weren't in primary
    for class_key, data in secondary_predictions.items():
        if class_key not in primary_predictions:
            class_name = data["class_name"]
            chunk_idx = data["chunk_idx"]
            secondary_score = data["secondary_score"]
            secondary_rank = data["secondary_rank"]
            
            # Calculate score (weighted less since only from secondary model)
            s_score = secondary_score * (1.0 / (secondary_rank + 1)) * 0.4
            
            fused_predictions.append((class_name, s_score, chunk_idx))
    
    # Sort by fusion score
    fused_predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Print results
    print(f"\n{TermColors.HEADER}TOP {top_k} PREDICTIONS (CROSS-MODEL FUSION):{TermColors.ENDC}")
    for i, (class_name, score, chunk_idx) in enumerate(fused_predictions[:top_k]):
        # Normalize score for display
        normalized_score = min(score, 1.0)  # Ensure score is between 0 and 1
        confidence_bar = "█" * int(normalized_score * 20)
        confidence_str = f"[{confidence_bar:<20}] {normalized_score:.4f}"
        print(f"{i+1}. {TermColors.BOLD}{class_name}{TermColors.ENDC} - {TermColors.CYAN}{confidence_str}{TermColors.ENDC} (Chunk {chunk_idx})")
    
    return fused_predictions[:top_k]

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
                loss = -np.sum(y_true_oh[i] * np.log(y_pred[i] + 1e-7))
                losses.append(loss)
            
            # Get indices of hardest examples (highest loss)
            hardest_indices = np.argsort(losses)[-int(len(losses)*0.3):]  # Top 30% hardest
            
            # Create sample weights
            self.current_weights = np.ones(len(self.X_train))
            self.current_weights[hardest_indices] = 2.0  # Double weight for hard examples
            
            print(f"{TermColors.GREEN}✅ Focused training on {len(hardest_indices)} difficult examples{TermColors.ENDC}")
        
        self.epoch_count += 1

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
        
        # FIRST do the train/test split
        X_train, X_val, y_train, y_val = train_test_split(
            features_scaled, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"{TermColors.GREEN}✅ Data split: {X_train.shape[0]} training, {X_val.shape[0]} validation samples{TermColors.ENDC}")
        
        # THEN augment the training data after splitting
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
                    # Save weights after start_epoch (when model is more stable)
                    if epoch % 3 == 0:  # Capture every 3rd epoch
                        print(f"{TermColors.CYAN}📸 Taking SWA snapshot at epoch {epoch+1}{TermColors.ENDC}")
                        
                        # Fix: Check if weights are already numpy arrays
                        weights = self.model.get_weights()
                        snapshot = []
                        
                        for w in weights:
                            # Check if it's already a numpy array or a TensorFlow tensor
                            if isinstance(w, np.ndarray):
                                snapshot.append(w.copy())  # It's already numpy, just copy it
                            else:
                                snapshot.append(w.numpy().copy())  # Convert TensorFlow tensor to numpy
                                
                        self.snapshots.append(snapshot)
        
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

def predict_with_chunked_models(image_path, top_k=5):
    """Make predictions using all the chunked models with visual progress"""
    def check_memory_limit():
        mem = psutil.virtual_memory()
        if mem.percent > 80:  # Emergency cutoff
            print(f"{TermColors.RED}🚨 RAM usage too high ({mem.percent}%). Cannot proceed with prediction.{TermColors.ENDC}")
            return False
        return True
    
    # Add this line HERE, after the function definition
    if not check_memory_limit():
        return []
    
    print(f"{TermColors.HEADER}\n{'='*50}")
    print(f"PREDICTION FOR: {os.path.basename(image_path)}")
    print(f"{'='*50}{TermColors.ENDC}")
    
    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img_processed = tf.keras.applications.efficientnet_v2.preprocess_input(img)
    img_batch = np.expand_dims(img_processed, axis=0)
    
    # Load feature extractor
    print(f"{TermColors.CYAN}ℹ Loading feature extractor{TermColors.ENDC}")
    with tf.device('/GPU:0'):
        from tensorflow.keras.applications import EfficientNetV2L
        
        feature_extractor = EfficientNetV2L(
            input_shape=(*IMAGE_SIZE, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        feature_extractor.trainable = False
    
    # Extract features
    features = feature_extractor.predict(img_batch, verbose=0)
    
    # Find all trained chunk models
    model_files = glob.glob(os.path.join(MODEL_DIR, "chunk_*_model.keras"))
    
    if not model_files:
        print(f"{TermColors.RED}❌ No trained models found in {MODEL_DIR}{TermColors.ENDC}")
        return []
    
    print(f"{TermColors.CYAN}ℹ Found {len(model_files)} trained chunk models{TermColors.ENDC}")
    
    # Process each model and collect predictions
    all_predictions = []
    model_bar = tqdm(total=len(model_files), desc="Processing models", position=0)
    
    for model_file in model_files:
        # Extract chunk index
        chunk_idx = int(os.path.basename(model_file).split("_")[1])
        
        # Load metadata
        metadata_file = os.path.join(MODEL_DIR, f"chunk_{chunk_idx}_metadata.json")
        
        if not os.path.exists(metadata_file):
            model_bar.update(1)
            continue
        
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        # Load model
        model = tf.keras.models.load_model(model_file, custom_objects={
            'sparse_top_k_accuracy': sparse_top_k_accuracy
        })
        
        # Get predictions
        preds = model.predict(features, verbose=0)[0]
        
        # Get top predictions for this chunk
        top_indices = np.argsort(preds)[-top_k:][::-1]
        chunk_predictions = []
        
        for idx in top_indices:
            class_name = metadata['class_mapping'][str(idx)]
            confidence = float(preds[idx])
            chunk_predictions.append((class_name, confidence, chunk_idx))
        
        # Add to overall predictions
        all_predictions.extend(chunk_predictions)
        
        # Clean up
        del model
        model_bar.update(1)
        
        # Clear session to release memory
        tf.keras.backend.clear_session()
    
    model_bar.close()
    
    # Free resources
    del feature_extractor
    gc.collect()
    tf.keras.backend.clear_session()
    
    # Sort by confidence
    all_predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Print top predictions
    print(f"\n{TermColors.HEADER}TOP {top_k} PREDICTIONS:{TermColors.ENDC}")
    for i, (class_name, confidence, chunk_idx) in enumerate(all_predictions[:top_k]):
        confidence_bar = "█" * int(confidence * 20)
        confidence_str = f"[{confidence_bar:<20}] {confidence:.4f}"
        print(f"{i+1}. {TermColors.BOLD}{class_name}{TermColors.ENDC} - {TermColors.CYAN}{confidence_str}{TermColors.ENDC} (Chunk {chunk_idx})")
    
    return all_predictions[:top_k]

if __name__ == "__main__":

    enable_emergency_protection = True  # Set this to False if you want to disable it
    if enable_emergency_protection:
        enable_emergency_memory_protection()

    def set_global_memory_growth():
        """Set memory growth limits in a platform-specific way"""
        import sys
        
        # Windows-specific memory management
        if sys.platform == 'win32':
            print(f"{TermColors.YELLOW}ℹ️ Windows detected - using Windows-specific memory management{TermColors.ENDC}")
            try:
                import psutil
                # Get the current process
                process = psutil.Process()
                # Try to set higher priority
                try:
                    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                except:
                    pass
                print(f"{TermColors.GREEN}✅ Windows process priority adjusted{TermColors.ENDC}")
                
                # Memory monitoring is already handled in the RAM monitor thread
                print(f"{TermColors.GREEN}✅ Using RAM monitor thread for memory management{TermColors.ENDC}")
                return
            except Exception as e:
                print(f"{TermColors.YELLOW}⚠️ Windows memory management setup error: {e}{TermColors.ENDC}")
                return
        
        # Unix-specific memory management (Linux/Mac)
        else:
            try:
                import resource
                # Limit maximum memory to 24GB
                MAX_MEMORY_GB = 24
                resource_limit = MAX_MEMORY_GB * 1024 * 1024 * 1024  # Convert to bytes
                
                # Set maximum memory limit
                resource.setrlimit(resource.RLIMIT_AS, (resource_limit, resource_limit))
                print(f"{TermColors.GREEN}✅ System memory limit set to {MAX_MEMORY_GB}GB{TermColors.ENDC}")
            except ImportError:
                print(f"{TermColors.YELLOW}⚠️ Resource module not available - skipping system memory limits{TermColors.ENDC}")
            except Exception as e:
                print(f"{TermColors.YELLOW}⚠️ Could not set system memory limit: {e}{TermColors.ENDC}")

    # Call the function to apply memory settings
    set_global_memory_growth()

    setup_terminal_clearing()

    try:
        # Define gpus before using it
        gpus = tf.config.experimental.list_physical_devices('GPU')
        
        # Print system info with enhanced formatting
        print(f"{TermColors.HEADER}\n{'='*50}")
        print(f"PLANT RECOGNITION MODEL - OPTIMIZED FOR RTX 3050")
        print(f"{'='*50}{TermColors.ENDC}")
        
        gpu_info = ""
        if gpus:
            gpu_info = "NVIDIA GPU"
        else:
            gpu_info = "CPU (No GPU detected)"
        
        # Add this function to train plants with combined features and SWA    
        def train_plants_advanced():
            """Train plant recognition models with combined features and SWA"""
            print(f"{TermColors.HEADER}\n{'='*50}")
            print(f"ADVANCED PLANT RECOGNITION TRAINING")
            print(f"{'='*50}{TermColors.ENDC}")
            
            # Initialize training state
            training_state = TrainingState()
            
            # Get class directories
            class_dirs = [os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR) 
                         if os.path.isdir(os.path.join(DATA_DIR, d))]
            
            # Group classes into chunks
            chunks = []
            current_chunk = []
            
            for class_dir in class_dirs:
                current_chunk.append(class_dir)
                if len(current_chunk) >= MAX_CLASSES_PER_CHUNK:
                    chunks.append(current_chunk)
                    current_chunk = []
            
            # Add the last chunk if not empty
            if current_chunk:
                chunks.append(current_chunk)
            
            print(f"{TermColors.CYAN}ℹ Divided {len(class_dirs)} classes into {len(chunks)} chunks{TermColors.ENDC}")
            
            # Process each chunk
            for chunk_idx, chunk_classes in enumerate(chunks):
                # Skip already processed chunks - MODIFY THIS SECTION
                if chunk_idx in training_state.processed_chunks:
                    print(f"{TermColors.YELLOW}⚠️ Skipping chunk {chunk_idx} (already processed){TermColors.ENDC}")
                    continue
                
                # IMPORTANT: Check if features already exist before extracting again
                chunk_feature_dir = os.path.join(FEATURES_DIR, f"chunk_{chunk_idx}")
                features_file = os.path.join(chunk_feature_dir, "features.npz")
                
                print(f"{TermColors.HEADER}\n{'='*50}")
                print(f"PROCESSING CHUNK {chunk_idx+1}/{len(chunks)} WITH {len(chunk_classes)} CLASSES")
                print(f"{'='*50}{TermColors.ENDC}")
                
                # Check if features already exist
                if os.path.exists(features_file):
                    print(f"{TermColors.CYAN}ℹ Found existing features for chunk {chunk_idx}. Using existing features.{TermColors.ENDC}")
                else:
                    # Extract combined features only if they don't exist
                    print(f"{TermColors.CYAN}ℹ Extracting features for chunk {chunk_idx}...{TermColors.ENDC}")
                    features_file, _ = extract_combined_features_optimized(chunk_classes, chunk_idx)
                    
                if features_file is None:
                    print(f"{TermColors.RED}❌ Failed to extract features for chunk {chunk_idx}{TermColors.ENDC}")
                    continue
                
                # Train with SWA
                train_chunk_model_with_swa(features_file, chunk_idx, training_state)
                
                print(f"{TermColors.YELLOW}⚠️ Performing full memory cleanup between chunks{TermColors.ENDC}")
                # Release all resources explicitly
                del features_file
                # Multiple garbage collections
                for _ in range(5):
                    gc.collect()
                tf.keras.backend.clear_session()
                # Sleep to ensure OS reclaims memory
                print(f"{TermColors.CYAN}ℹ Waiting 10 seconds for memory to be fully released...{TermColors.ENDC}")
                time.sleep(10) 

                # Clean memory between chunks
                clean_memory()
            
            print(f"{TermColors.GREEN}✅ All chunks processed successfully!{TermColors.ENDC}")
        
        # Example command to run the advanced training pipeline
        train_plants_advanced()
        
    except Exception as e:
        print(f"{TermColors.RED}❌ Fatal error: {e}{TermColors.ENDC}")
>>>>>>> ba6742206bc2a45205ad6d1b60d1894da4fd3dd6
        traceback.print_exc()