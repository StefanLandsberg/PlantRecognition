import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision, callbacks
from tensorflow.keras.applications import MobileNetV3Large, DenseNet121, VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import os
import gc
import time
import json
import glob
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier # For meta-model
from sklearn.utils import class_weight as sk_class_weight, compute_class_weight
from tqdm.auto import tqdm
from datetime import datetime, timedelta
import signal
import traceback
import random
import multiprocessing
import builtins # For print patching
import colorama # For colored output
import sys # For checking keyboard module
import subprocess # For installing keyboard module if needed
import warnings
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Attempt to import tensorflow-addons
try:
    import tensorflow_addons as tfa
    ADAMW_AVAILABLE = True
except ImportError:
    print("tensorflow-addons not found. AdamW optimizer unavailable. Falling back to Adam.")
    tfa = None
    ADAMW_AVAILABLE = False

# Attempt to import keyboard module for graceful stop shortcut
try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    keyboard = None
    KEYBOARD_AVAILABLE = False

# Suppress TensorFlow/System Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
np.seterr(all='ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' # Allow GPU memory growth

# --- Constants and Setup ---
class TermColors:
    HEADER = '\033[95m'; BLUE = '\033[94m'; CYAN = '\033[96m'; GREEN = '\033[92m'
    YELLOW = '\033[93m'; RED = '\033[91m'; ENDC = '\033[0m'; BOLD = '\033[1m'; UNDERLINE = '\033[4m'
    MAGENTA = '\033[35m' # Added for SWA


# Enable Mixed Precision Training
mixed_precision.set_global_policy('mixed_float16')
print(f"{TermColors.GREEN}✅ Mixed precision training enabled (mixed_float16).{TermColors.ENDC}")


colorama.init(autoreset=True) # Automatically reset color after each print

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR) # Assumes Scripts is one level down
DATA_DIR = os.path.join(BASE_DIR, "data", "plant_images")
FEATURES_DIR = os.path.join(DATA_DIR, "features_v2") # Use separate dir for PEFT features
MODEL_DIR = os.path.join(BASE_DIR, "models_v2")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints_v2")
LOG_DIR = os.path.join(BASE_DIR, "logs_v2")
STATS_DIR = os.path.join(BASE_DIR, "stats_v2")

# Training Hyperparameters
SEED = 42
IMAGE_SIZE = (224, 224) # Standard size for many models
PEFT_BATCH_SIZE = 16 # Lower for PEFT fine-tuning VRAM
FEATURE_EXTRACT_BATCH_SIZE = 32 # Can be higher for inference
CHUNK_TRAIN_BATCH_SIZE = 64 # Batch size for MLP training
PEFT_EPOCHS = 5 # Short fine-tuning for adapters
CHUNK_EPOCHS = 75 # Epochs for chunk classifiers
MAX_CLASSES_PER_CHUNK = 1000 # Adjust based on RAM/feature size
TOP_K = 10 # For top-k accuracy metric
INITIAL_LR = 1e-3 # For chunk classifier
PEFT_LR = 1e-4 # Lower LR for fine-tuning adapters
L2_REG = 1e-5 # Reduced L2 regularization
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25 # Standard value, can be tuned
SWA_START_EPOCH = 15 # Start SWA after initial convergence
CONTRASTIVE_MARGIN = 1.0
CONTRASTIVE_PULL = 0.1
CONTRASTIVE_PUSH = 0.05
CONTRASTIVE_MAX_SHIFT = 0.5
FEATURE_AUG_FACTOR = 0.2 # Percentage of samples to augment in feature space
FEATURE_AUG_NOISE = 0.02 # Noise level for feature augmentation

# System Config
CPU_WORKERS = min(8, multiprocessing.cpu_count())
MEMORY_CLEANUP_FREQ = 500 # Batches between aggressive cleanup
GPU_MEMORY_LIMIT_MB = 7168 # ~7GB for RTX 3050 8GB, leaving some headroom
RAM_PERCENT_LIMIT = 85 # Target RAM usage limit

# PEFT Config
LORA_RANK = 16
LORA_ALPHA = 32
# Specify target layer names (User needs to inspect model summaries)
# Example: Find dense/attention layers. If None, LoRA injection placeholder won't target specific layers.
# --- Start Change: Add example target layer names ---
MOBILENET_LORA_TARGETS = ['Conv_1', 'expanded_conv_6/project', 'expanded_conv_15/project'] # Example Conv layers
DENSENET_LORA_TARGETS = ['conv3_block12_2_conv', 'conv4_block24_2_conv', 'conv5_block16_2_conv'] # Example Conv layers in dense blocks
# --- End Change ---

# Create directories
for directory in [FEATURES_DIR, MODEL_DIR, CHECKPOINT_DIR, LOG_DIR, STATS_DIR]:
    os.makedirs(directory, exist_ok=True)

# --- Utility Functions ---
def limit_memory_usage(gpu_memory_limit_mb=GPU_MEMORY_LIMIT_MB, ram_percent_limit=RAM_PERCENT_LIMIT):
    print(f"{TermColors.YELLOW}⚠️ MEMORY LIMITER ACTIVE (GPU: {gpu_memory_limit_mb}MB, Target RAM: {ram_percent_limit}%){TermColors.ENDC}")
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.set_logical_device_configuration(
                physical_devices[0], [tf.config.LogicalDeviceConfiguration(memory_limit=gpu_memory_limit_mb)]
            )
            print(f"{TermColors.GREEN}✅ GPU memory limited to {gpu_memory_limit_mb}MB{TermColors.ENDC}")
        except Exception as e:
            print(f"{TermColors.RED}❌ Error setting GPU memory limit: {e}{TermColors.ENDC}")
            try:
                for device in physical_devices: tf.config.experimental.set_memory_growth(device, True)
                print(f"{TermColors.YELLOW}⚠️ Enabled GPU memory growth as fallback.{TermColors.ENDC}")
            except Exception as e_growth: print(f"{TermColors.RED}❌ Failed to enable memory growth: {e_growth}{TermColors.ENDC}")
    else: print(f"{TermColors.YELLOW}⚠️ No GPU devices found.{TermColors.ENDC}")
    print(f"{TermColors.CYAN}ℹ RAM usage target set to {ram_percent_limit}%.{TermColors.ENDC}")

limit_memory_usage() # Apply limits at startup

def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # print(f"{TermColors.CYAN}ℹ Random seeds set to {seed}{TermColors.ENDC}") # Reduced verbosity

STOP_FLAG = False
def _signal_handler(sig, frame):
    global STOP_FLAG
    if not STOP_FLAG:
        print(f"\n{TermColors.RED}🛑 SIGINT received. Attempting graceful stop... Press Ctrl+C again to force exit.{TermColors.ENDC}")
        STOP_FLAG = True
    else:
        print(f"\n{TermColors.RED}🛑 Force exiting now.{TermColors.ENDC}")
        exit(1)

signal.signal(signal.SIGINT, _signal_handler)

# Optional: Keyboard listener for 'q' to stop
if KEYBOARD_AVAILABLE:
    def check_stop_key():
        global STOP_FLAG
        if keyboard.is_pressed('q'):
            if not STOP_FLAG:
                print(f"\n{TermColors.RED}🛑 'q' pressed. Attempting graceful stop...{TermColors.ENDC}")
                STOP_FLAG = True
    # Note: keyboard listener might need to run in a separate thread or use hooks,
    # which adds complexity. Simple check within loops is less reliable.
    # For simplicity, relying primarily on Ctrl+C.
    pass
else:
    # print(f"{TermColors.YELLOW}⚠️ 'keyboard' module not found. 'q' to stop disabled. Use Ctrl+C.{TermColors.ENDC}")
    pass

def model_size_mb(model):
    try:
        temp_path = os.path.join(CHECKPOINT_DIR, "_temp_model_size.keras")
        model.save(temp_path)
        size_bytes = os.path.getsize(temp_path)
        os.remove(temp_path)
        return size_bytes / (1024 * 1024)
    except Exception: return 0

def cleanup_memory():
    gc.collect()
    tf.keras.backend.clear_session()
    # print(f"{TermColors.BLUE}🧹 Memory cleanup executed.{TermColors.ENDC}") # Optional: Verbose cleanup message

# --- PEFT: LoRA Implementation ---
class LoRALayer(layers.Layer):
    def __init__(self, original_layer, rank=8, alpha=16, trainable=True, **kwargs):
        super().__init__(name=f"{original_layer.name}_lora", trainable=trainable, **kwargs)
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self._supported_layers = (layers.Dense, layers.Conv2D)
        if not isinstance(original_layer, self._supported_layers):
            raise ValueError(f"LoRA only supports {self._supported_layers}.")
        self.input_dim = None
        self.original_units = None
        self.is_conv = isinstance(original_layer, layers.Conv2D)
        self.original_layer.trainable = False

    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        if self.is_conv:
            self.original_units = self.original_layer.filters
            lora_A_shape = (self.input_dim, self.rank)
            lora_B_shape = (self.rank, self.original_units)
        else:
            self.original_units = self.original_layer.units
            lora_A_shape = (self.input_dim, self.rank)
            lora_B_shape = (self.rank, self.original_units)
        self.lora_A = self.add_weight(name="lora_A", shape=lora_A_shape, initializer=tf.keras.initializers.RandomNormal(stddev=1/self.rank), trainable=self.trainable)
        self.lora_B = self.add_weight(name="lora_B", shape=lora_B_shape, initializer=tf.keras.initializers.Zeros(), trainable=self.trainable)
        super().build(input_shape)

    def call(self, inputs):
        original_output = self.original_layer(inputs)
        if not self.trainable: return original_output
        if self.is_conv: lora_adjustment = 0 # Placeholder: Needs proper Conv LoRA
        else: lora_adjustment = tf.matmul(tf.matmul(inputs, self.lora_A), self.lora_B) * (self.alpha / self.rank)
        return original_output + lora_adjustment

    def get_config(self):
        config = super().get_config(); config.update({"rank": self.rank, "alpha": self.alpha}); return config

def add_lora_to_model(model, rank=LORA_RANK, alpha=LORA_ALPHA, target_layer_names=None):
    print(f"{TermColors.CYAN}ℹ Attempting LoRA injection (rank={rank}, alpha={alpha})...{TermColors.ENDC}")
    if not isinstance(model, tf.keras.Model) or not model.built: raise ValueError("Model must be built Keras Functional model.")
    if target_layer_names is None: print(f"{TermColors.YELLOW}⚠️ No target_layer_names specified. No LoRA adapters added.{TermColors.ENDC}"); return model
    print(f"{TermColors.YELLOW}⚠️ LoRA injection placeholder: Robust implementation requires graph surgery or library.{TermColors.ENDC}")
    print(f"{TermColors.YELLOW}⚠️ Suggestion: Manually modify model definition or use a PEFT library.{TermColors.ENDC}")
    return model # Return original structure - User must ensure LoRA layers are present

def set_lora_trainable(model, trainable=True):
    model.trainable = False; found_lora = False
    for layer in model.layers:
        if isinstance(layer, LoRALayer): layer.trainable = trainable; found_lora = True
        elif isinstance(layer, tf.keras.Model): set_lora_trainable(layer, trainable)
        else: layer.trainable = False

def get_preprocessing_function(model_name):
    model_name_lower = model_name.lower()
    if "mobilenetv3large" in model_name_lower: return tf.keras.applications.mobilenet_v3.preprocess_input
    elif "densenet121" in model_name_lower: return tf.keras.applications.densenet.preprocess_input
    else: print(f"{TermColors.YELLOW}⚠️ Unknown model name '{model_name}' for preprocessing.{TermColors.ENDC}"); return None

# --- Base Model Loaders ---
def load_mobilenetv3(include_top=False, pooling='avg'):
    return MobileNetV3Large(input_shape=(*IMAGE_SIZE, 3), include_top=include_top, weights='imagenet', pooling=pooling)
def load_densenet121(include_top=False, pooling='avg'):
    return DenseNet121(input_shape=(*IMAGE_SIZE, 3), include_top=include_top, weights='imagenet', pooling=pooling)

# --- PEFT Fine-tuning ---
def run_peft_fine_tuning(base_model_loader, model_name, lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA,
                         peft_epochs=PEFT_EPOCHS, peft_batch_size=PEFT_BATCH_SIZE, peft_lr=PEFT_LR, target_layer_names=None):
    global STOP_FLAG
    print(f"{TermColors.HEADER}\n{'='*50}\nSTARTING PEFT (LoRA) FINE-TUNING for {model_name}\n{'='*50}{TermColors.ENDC}")
    set_seeds(); cleanup_memory()
    try: base_model = base_model_loader(include_top=False, pooling='avg'); base_model.trainable = False
    except Exception as e: print(f"{TermColors.RED}❌ Error loading base model {model_name}: {e}{TermColors.ENDC}"); return False

    peft_model = add_lora_to_model(base_model, rank=lora_rank, alpha=lora_alpha, target_layer_names=target_layer_names)
    set_lora_trainable(peft_model, trainable=True)

    try:
        img_height, img_width = IMAGE_SIZE
        image_dir = DATA_DIR # Use DATA_DIR directly, which points to .../data/plant_images

        # --- Start Change: Use flow_from_directory ---
        print(f"{TermColors.CYAN}ℹ️ Setting up ImageDataGenerators...{TermColors.ENDC}")
        # Training generator with augmentation AND validation split
        train_datagen = ImageDataGenerator(
            rescale=1./255, # Option 2: Basic rescale if preprocessing done later
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2 # Use built-in validation split (20%)
        )

        # Validation generator with only rescaling (or model-specific preprocessing)
        validation_datagen = ImageDataGenerator(
            rescale=1./255, # Option 2
            validation_split=0.2 # Must match train_datagen
        )

        # Flow from directory
        print(f"{TermColors.CYAN}ℹ️ Loading training data from: {image_dir}{TermColors.ENDC}")
        train_ds = train_datagen.flow_from_directory(
            directory=image_dir,
            target_size=(img_height, img_width),
            batch_size=peft_batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=SEED, # Add seed for reproducibility
            subset='training', # Specify this is the training subset
            workers=CPU_WORKERS,       # Use multiple workers
            use_multiprocessing=True # Enable multiprocessing
        )

        print(f"{TermColors.CYAN}ℹ️ Loading validation data from: {image_dir}{TermColors.ENDC}")
        val_ds = validation_datagen.flow_from_directory(
            directory=image_dir,
            target_size=(img_height, img_width),
            batch_size=peft_batch_size,
            class_mode='categorical',
            shuffle=False, # No need to shuffle validation data
            seed=SEED, # Add seed for reproducibility
            subset='validation', # Specify this is the validation subset
            workers=CPU_WORKERS,       # Use multiple workers
            use_multiprocessing=True # Enable multiprocessing
        )

        if not train_ds.classes.size > 0:
             raise ValueError(f"No images found by flow_from_directory in {image_dir}. Check directory structure and permissions.")

        num_classes = len(train_ds.class_indices)
        print(f"{TermColors.GREEN}✅ Loaded image data: {train_ds.samples} training, {val_ds.samples} validation samples across {num_classes} classes.{TermColors.ENDC}")
        # --- End Change ---

        preprocess_input = get_preprocessing_function(model_name)
        if preprocess_input is None: return False

    except Exception as e: print(f"{TermColors.RED}❌ Error loading/preprocessing image data: {e}{TermColors.ENDC}"); return False

    inputs = peft_model.input; x = peft_model.output
    if len(x.shape) > 2: x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x) # Added Dropout layer for regularization
    outputs = layers.Dense(num_classes, name="peft_temp_head")(x)
    model_with_head = tf.keras.Model(inputs=inputs, outputs=outputs)

    set_lora_trainable(model_with_head, trainable=True)
    try: model_with_head.get_layer('peft_temp_head').trainable = True
    except ValueError: pass

    trainable_count = sum(tf.keras.backend.count_params(w) for w in model_with_head.trainable_weights)
    non_trainable_count = sum(tf.keras.backend.count_params(w) for w in model_with_head.non_trainable_weights)
    print(f"{TermColors.GREEN}✅ Trainable params: {trainable_count:,}{TermColors.ENDC}")
    print(f"{TermColors.YELLOW}⚠️ Non-trainable params: {non_trainable_count:,}{TermColors.ENDC}")
    if trainable_count == 0 or trainable_count > non_trainable_count * 0.1: print(f"{TermColors.RED}❌ Trainable param count seems wrong. LoRA injection likely failed. Aborting PEFT.{TermColors.ENDC}"); return False

    optimizer = tf.keras.optimizers.Adam(learning_rate=peft_lr)
    loss_fn = focal_loss
    model_with_head.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    # Calculate class weights to handle imbalance
    class_labels = train_ds.classes
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(class_labels),
        y=class_labels
    )
    class_weight_dict = dict(enumerate(class_weights))
    print("Calculated Class Weights:", class_weight_dict)

    # Define callbacks
    checkpoint = callbacks.ModelCheckpoint(os.path.join(CHECKPOINT_DIR, f"{model_name}_best.keras"), monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='min', min_lr=0.00001)

    callbacks_list = [checkpoint, early_stopping, reduce_lr]

    print(f"{TermColors.CYAN}ℹ Fine-tuning LoRA adapters for {peft_epochs} epochs... (Press 'q' or Ctrl+C to stop){TermColors.ENDC}")
    history = model_with_head.fit(
        train_ds, validation_data=val_ds, epochs=peft_epochs,
        callbacks=callbacks_list, class_weight=class_weight_dict
    )

    lora_weights_path = os.path.join(MODEL_DIR, f"{model_name}_lora_weights.npz")
    lora_weights_dict = {}
    model_to_save_from = peft_model
    for layer in model_to_save_from.layers:
         if isinstance(layer, LoRALayer): lora_weights_dict[layer.name] = layer.get_weights()
         elif isinstance(layer, tf.keras.Model):
             for sub_layer in layer.layers:
                 if isinstance(sub_layer, LoRALayer): lora_weights_dict[sub_layer.name] = sub_layer.get_weights()

    if not lora_weights_dict: print(f"{TermColors.YELLOW}⚠️ No LoRA layers found to save weights for.{TermColors.ENDC}"); return False
    else:
         try: np.savez(lora_weights_path, **lora_weights_dict); print(f"{TermColors.GREEN}✅ LoRA adapter weights saved to {lora_weights_path}{TermColors.ENDC}")
         except Exception as e: print(f"{TermColors.RED}❌ Error saving LoRA weights: {e}{TermColors.ENDC}"); return False

    cleanup_memory()
    print(f"{TermColors.GREEN}✅ PEFT fine-tuning completed for {model_name}.{TermColors.ENDC}")
    return True

# --- Feature Extraction Helper ---
def load_image_array(image_path, target_size):
    """Loads an image, resizes it, converts to array. Returns None on error."""
    try:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        return img_array
    except Exception as img_err:
        # Use print directly as this runs in separate processes
        print(f"\n{TermColors.YELLOW}⚠️ Error loading image {os.path.basename(image_path)}: {img_err}. Skipping.{TermColors.ENDC}")
        return None

# --- Feature Extraction with PEFT ---
def extract_features_with_peft(chunk_size=MAX_CLASSES_PER_CHUNK, batch_size=FEATURE_EXTRACT_BATCH_SIZE):
    global STOP_FLAG
    print(f"{TermColors.HEADER}\n{'='*50}\nSTARTING FEATURE EXTRACTION with PEFT Adapters\n{'='*50}{TermColors.ENDC}")
    set_seeds(); cleanup_memory()
    models_to_load = {"mobilenetv3large": load_mobilenetv3, "densenet121": load_densenet121}
    feature_extractors = {}; all_features_dim = 0
    lora_rank = LORA_RANK; lora_alpha = LORA_ALPHA

    for model_name, loader_func in models_to_load.items():
        print(f"{TermColors.CYAN}ℹ Loading base model {model_name} for feature extraction...{TermColors.ENDC}")
        try:
            base_model = loader_func(include_top=False, pooling='avg'); base_model.trainable = False
            peft_model = add_lora_to_model(base_model, rank=lora_rank, alpha=lora_alpha) # Placeholder injection
            lora_weights_path = os.path.join(MODEL_DIR, f"{model_name}_lora_weights.npz")
            if not os.path.exists(lora_weights_path): print(f"{TermColors.YELLOW}⚠️ LoRA weights not found at {lora_weights_path}. Using base model without PEFT.{TermColors.ENDC}")
            else:
                try:
                    lora_weights_dict = np.load(lora_weights_path, allow_pickle=True)
                    print(f"{TermColors.CYAN}ℹ Loading LoRA weights from {lora_weights_path}...{TermColors.ENDC}")
                    loaded_count = 0
                    for layer in peft_model.layers:
                        if isinstance(layer, LoRALayer):
                            if layer.name in lora_weights_dict: layer.set_weights(lora_weights_dict[layer.name]); loaded_count += 1
                        elif isinstance(layer, tf.keras.Model):
                             for sub_layer in layer.layers:
                                 if isinstance(sub_layer, LoRALayer):
                                     if sub_layer.name in lora_weights_dict: sub_layer.set_weights(lora_weights_dict[sub_layer.name]); loaded_count += 1
                    if loaded_count > 0: print(f"{TermColors.GREEN}✅ Loaded weights for {loaded_count} LoRA layers.{TermColors.ENDC}")
                    elif len(lora_weights_dict) > 0: print(f"{TermColors.YELLOW}⚠️ LoRA weights file found, but no matching LoRA layers in model!{TermColors.ENDC}")
                except Exception as e: print(f"{TermColors.RED}❌ Error loading LoRA weights: {e}. Using base model without PEFT.{TermColors.ENDC}")
            feature_extractors[model_name] = peft_model
            all_features_dim += peft_model.output_shape[-1]
            print(f"{TermColors.GREEN}✅ Prepared feature extractor: {model_name}{TermColors.ENDC}")
        except Exception as e: print(f"{TermColors.RED}❌ Error setting up feature extractor {model_name}: {e}{TermColors.ENDC}"); return False
    if not feature_extractors: return False
    print(f"{TermColors.CYAN}ℹ Combined feature dimension: {all_features_dim}{TermColors.ENDC}")

    try:
        img_height, img_width = IMAGE_SIZE
        image_dir = DATA_DIR # Use DATA_DIR directly
        image_paths = []; labels = []
        class_names = sorted([d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))])
        if not class_names: raise ValueError(f"No subdirectories found in {image_dir}")
        class_indices = {name: i for i, name in enumerate(class_names)}
        print(f"{TermColors.CYAN}ℹ Scanning image files...{TermColors.ENDC}")
        for class_name in tqdm(class_names, desc="Scanning Classes", unit="class"):
            class_dir = os.path.join(image_dir, class_name)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    image_paths.append(os.path.join(class_dir, fname))
                    labels.append(class_indices[class_name])
        if not image_paths: raise ValueError("No image files found.")
        print(f"{TermColors.GREEN}✅ Found {len(image_paths)} images across {len(class_names)} classes.{TermColors.ENDC}")
        image_paths = np.array(image_paths); labels = np.array(labels)

        num_classes = len(class_names)
        num_chunks = (num_classes + chunk_size - 1) // chunk_size
        print(f"{TermColors.CYAN}ℹ Splitting {num_classes} classes into {num_chunks} chunks of size ~{chunk_size}...{TermColors.ENDC}")
        os.makedirs(FEATURES_DIR, exist_ok=True)

        for chunk_idx in range(num_chunks):
            if STOP_FLAG: print(f"{TermColors.YELLOW}⚠️ Stop requested during feature extraction. Exiting chunk loop.{TermColors.ENDC}"); break
            start_class_idx = chunk_idx * chunk_size; end_class_idx = min((chunk_idx + 1) * chunk_size, num_classes)
            chunk_class_indices = list(range(start_class_idx, end_class_idx))
            chunk_class_names = [class_names[i] for i in chunk_class_indices]
            print(f"{TermColors.HEADER}\n--- Processing Chunk {chunk_idx}/{num_chunks-1} ({len(chunk_class_names)} classes) ---{TermColors.ENDC}")

            chunk_mask = np.isin(labels, chunk_class_indices)
            chunk_image_paths = image_paths[chunk_mask]
            chunk_global_labels = labels[chunk_mask]
            if len(chunk_image_paths) == 0: print(f"{TermColors.YELLOW}⚠️ No images for chunk {chunk_idx}. Skipping.{TermColors.ENDC}"); continue

            local_class_map = {global_idx: local_idx for local_idx, global_idx in enumerate(chunk_class_indices)}
            chunk_local_labels = np.array([local_class_map[global_idx] for global_idx in chunk_global_labels])
            chunk_class_mapping_json = {str(local_idx): class_names[global_idx] for global_idx, local_idx in local_class_map.items()}

            chunk_features_list = []; valid_indices_in_chunk = []
            num_batches = (len(chunk_image_paths) + batch_size - 1) // batch_size

            # Use multiprocessing pool for image loading within the chunk loop
            with multiprocessing.Pool(processes=CPU_WORKERS) as pool:
                for i in tqdm(range(num_batches), desc=f"Extract Features Chunk {chunk_idx}", unit="batch", leave=False):
                    if STOP_FLAG: break
                    start_idx = i * batch_size
                    end_idx = (i + 1) * batch_size
                    batch_paths = chunk_image_paths[start_idx:end_idx]
                    original_indices_in_chunk = list(range(start_idx, min(end_idx, len(chunk_image_paths)))) # Indices relative to the chunk

                    # Parallel image loading and resizing
                    load_args = [(path, (img_height, img_width)) for path in batch_paths]
                    results = pool.starmap(load_image_array, load_args)

                    # Filter out failed loads and keep track of successful original indices
                    batch_images = []
                    current_batch_valid_indices_relative_to_chunk = []
                    for original_chunk_idx, img_array in zip(original_indices_in_chunk, results):
                        if img_array is not None:
                            batch_images.append(img_array)
                            current_batch_valid_indices_relative_to_chunk.append(original_chunk_idx) # Store original index within chunk

                    if not batch_images: continue # Skip if batch loading failed completely

                    # Store the valid original indices from the chunk's perspective
                    valid_indices_in_chunk.extend(current_batch_valid_indices_relative_to_chunk)

                    # Apply preprocessing after parallel loading
                    batch_images_np = np.array(batch_images)
                    batch_features_concat = []
                    for model_name, extractor in feature_extractors.items():
                        preprocess_input = get_preprocessing_function(model_name)
                        if preprocess_input is None: continue
                        # Preprocess the successfully loaded batch
                        processed_batch = preprocess_input(batch_images_np.copy()) # Use copy just in case
                        batch_features = extractor.predict(processed_batch, verbose=0)
                        batch_features_concat.append(batch_features)

                    if not batch_features_concat: continue # Skip if no features were extracted (e.g., preprocessing failed)

                    concatenated_features = np.concatenate(batch_features_concat, axis=-1)
                    chunk_features_list.append(concatenated_features)
                    if i % MEMORY_CLEANUP_FREQ == 0 and i > 0: cleanup_memory() # Periodic cleanup

            if STOP_FLAG: print(f"{TermColors.YELLOW}⚠️ Stop requested. Skipping save for chunk {chunk_idx}.{TermColors.ENDC}"); continue
            if not chunk_features_list: print(f"{TermColors.YELLOW}⚠️ No features extracted for chunk {chunk_idx}. Skipping save.{TermColors.ENDC}"); continue

            chunk_features_np = np.concatenate(chunk_features_list, axis=0)
            chunk_local_labels_filtered = chunk_local_labels[valid_indices_in_chunk] # Filter labels to match extracted features

            if len(chunk_features_np) != len(chunk_local_labels_filtered):
                 print(f"{TermColors.RED}❌ Feature length ({len(chunk_features_np)}) mismatch filtered label length ({len(chunk_local_labels_filtered)}) for chunk {chunk_idx}. Skipping save.{TermColors.ENDC}")
                 continue

            chunk_dir = os.path.join(FEATURES_DIR, f"chunk_{chunk_idx}")
            os.makedirs(chunk_dir, exist_ok=True)
            features_path = os.path.join(chunk_dir, "features.npz")
            class_mapping_path = os.path.join(chunk_dir, "class_mapping.json")
            np.savez_compressed(features_path, features=chunk_features_np, labels=chunk_local_labels_filtered)
            with open(class_mapping_path, 'w') as f: json.dump(chunk_class_mapping_json, f, indent=2)
            print(f"{TermColors.GREEN}✅ Saved features ({chunk_features_np.shape}) and mapping for chunk {chunk_idx}.{TermColors.ENDC}")
            del chunk_features_np, chunk_local_labels_filtered, chunk_features_list; cleanup_memory()

    except Exception as e: print(f"{TermColors.RED}❌ Error during PEFT feature extraction: {e}{TermColors.ENDC}"); traceback.print_exc(); return False
    cleanup_memory()
    print(f"{TermColors.GREEN}✅ PEFT feature extraction completed.{TermColors.ENDC}")
    return True

# --- Loss Functions ---
def focal_loss(y_true, y_pred_logits, gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA):
    y_true = tf.cast(y_true, tf.int32)
    if len(y_true.shape) > 1 and y_true.shape[-1] == 1: y_true = tf.squeeze(y_true, axis=-1)
    probs = tf.nn.softmax(y_pred_logits, axis=-1)
    indices = tf.stack([tf.range(tf.shape(y_true)[0], dtype=tf.int32), y_true], axis=1)
    p_t = tf.gather_nd(probs, indices)
    focal_weight = tf.pow(1.0 - p_t, gamma)
    cross_entropy = -tf.math.log(tf.clip_by_value(p_t, 1e-8, 1.0))
    alpha_factor = tf.ones_like(y_true, dtype=tf.float32) * alpha
    alpha_weighted_focal_weight = alpha_factor * focal_weight
    loss = alpha_weighted_focal_weight * cross_entropy
    return tf.reduce_mean(loss)

# --- Accuracy Metric ---
@tf.keras.utils.register_keras_serializable(package="Custom", name="sparse_top_k_accuracy")
def sparse_top_k_accuracy(y_true, y_pred_logits, k=TOP_K):
    return tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred_logits, k=k)

# --- Model Building ---
def build_model(input_dim, num_classes, l2_reg=L2_REG):
    model = models.Sequential([
        layers.Dense(1024, activation='relu', input_shape=(input_dim,), kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name="dense_1"),
        layers.BatchNormalization(name="bn_1"), layers.Dropout(0.4, name="dropout_1"),
        layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name="dense_2"),
        layers.BatchNormalization(name="bn_2"), layers.Dropout(0.4, name="dropout_2"),
        layers.Dense(num_classes, name="output") # Logits
    ], name=f"MLP_{input_dim}in_{num_classes}out")
    if ADAMW_AVAILABLE: optimizer = tfa.optimizers.AdamW(weight_decay=1e-5, learning_rate=INITIAL_LR)
    else: optimizer = tf.keras.optimizers.Adam(learning_rate=INITIAL_LR)
    model.compile(optimizer=optimizer, loss=focal_loss, metrics=['accuracy', sparse_top_k_accuracy])
    return model

# --- Data Handling Utilities ---
def compute_balanced_class_weights(y):
    """Computes class weights inversely proportional to class frequencies."""
    try:
        weights = sk_class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
        return dict(enumerate(weights))
    except Exception as e:
        print(f"{TermColors.YELLOW}⚠️ Failed to compute class weights: {e}. Returning None.{TermColors.ENDC}")
        return None

def balance_training_data(X_train, y_train, strategy='over'):
    """Balances training data using simple random over/under-sampling."""
    print(f"{TermColors.CYAN}ℹ Balancing training data using random '{strategy}' sampling...{TermColors.ENDC}")
    unique, counts = np.unique(y_train, return_counts=True)
    if len(unique) <= 1: return X_train, y_train # Cannot balance single class
    target_count = int(np.median(counts)) if strategy == 'over' else int(np.min(counts))
    if strategy == 'over': target_count = max(target_count, int(np.max(counts) * 0.8)) # Don't oversample too extremely

    X_resampled, y_resampled = [], []
    for cls in unique:
        cls_mask = (y_train == cls)
        X_cls = X_train[cls_mask]
        y_cls = y_train[cls_mask]
        current_count = len(X_cls)

        if strategy == 'over':
            if current_count < target_count:
                indices = np.random.choice(current_count, target_count - current_count, replace=True)
                X_resampled.append(np.vstack([X_cls, X_cls[indices]]))
                y_resampled.append(np.hstack([y_cls, y_cls[indices]]))
            else: # Keep original if already above target
                X_resampled.append(X_cls)
                y_resampled.append(y_cls)
        elif strategy == 'under':
            if current_count > target_count:
                indices = np.random.choice(current_count, target_count, replace=False)
                X_resampled.append(X_cls[indices])
                y_resampled.append(y_cls[indices])
            else: # Keep original if already below target
                X_resampled.append(X_cls)
                y_resampled.append(y_cls)

    X_resampled = np.vstack(X_resampled)
    y_resampled = np.hstack(y_resampled)
    print(f"{TermColors.GREEN}✅ Resampled data shape: {X_resampled.shape}{TermColors.ENDC}")
    return X_resampled, y_resampled

def augment_feature_space(X, y, augmentation_factor=FEATURE_AUG_FACTOR, noise_level=FEATURE_AUG_NOISE):
    """Adds Gaussian noise to a fraction of feature vectors."""
    if augmentation_factor <= 0: return X, y
    print(f"{TermColors.CYAN}ℹ Augmenting {augmentation_factor*100:.1f}% of features with noise level {noise_level}...{TermColors.ENDC}")
    num_samples = X.shape[0]
    num_to_augment = int(num_samples * augmentation_factor)
    if num_to_augment == 0: return X, y

    indices_to_augment = np.random.choice(num_samples, num_to_augment, replace=False)
    X_augmented = np.copy(X)
    noise = np.random.normal(loc=0.0, scale=noise_level, size=(num_to_augment, X.shape[1]))
    X_augmented[indices_to_augment] += noise
    print(f"{TermColors.GREEN}✅ Feature augmentation applied.{TermColors.ENDC}")
    return X_augmented, y # Labels remain the same

def contrastive_feature_learning(X_train, y_train, margin=CONTRASTIVE_MARGIN, pull_factor=CONTRASTIVE_PULL, push_factor=CONTRASTIVE_PUSH, max_shift_norm=CONTRASTIVE_MAX_SHIFT):
    """Applies centroid-based contrastive learning to feature vectors."""
    print(f"{TermColors.CYAN}ℹ Applying contrastive learning to features...{TermColors.ENDC}")
    set_seeds()
    X_norm = X_train # Assume features are already scaled
    enhanced_features = np.copy(X_norm)
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2: return X_train

    centroids = {cls: np.mean(X_norm[y_train == cls], axis=0) for cls in unique_classes if np.sum(y_train == cls) > 0}
    if not centroids: return X_train # No valid centroids

    for i in tqdm(range(X_norm.shape[0]), desc="Contrastive Enhancement", unit="sample", leave=False):
        x = X_norm[i]; y = y_train[i]
        if y not in centroids: continue # Skip if class has no centroid (e.g., single sample)
        own_centroid = centroids[y]; shift_vector = np.zeros_like(x)
        shift_vector += pull_factor * (own_centroid - x) # Pull
        push_force_total = np.zeros_like(x); num_other_centroids = 0
        for other_cls, other_centroid in centroids.items():
            if other_cls != y:
                push_direction = x - other_centroid; distance_sq = np.sum(push_direction**2)
                push_force = push_direction / (distance_sq + 1e-6)
                push_force_total += push_force; num_other_centroids += 1
        if num_other_centroids > 0: shift_vector += push_factor * (push_force_total / num_other_centroids) # Push
        shift_norm = np.linalg.norm(shift_vector)
        if shift_norm > max_shift_norm: shift_vector = (shift_vector / shift_norm) * max_shift_norm
        enhanced_features[i] = x + shift_vector

    if np.allclose(X_train, enhanced_features, atol=1e-4): print(f"{TermColors.YELLOW}⚠️ Contrastive learning did not significantly alter features.{TermColors.ENDC}")
    else: print(f"{TermColors.GREEN}✅ Feature enhancement completed.{TermColors.ENDC}")
    return enhanced_features

# --- Callbacks & State ---
class TrainingState:
    """Manages training state persistence."""
    def __init__(self, checkpoint_dir=CHECKPOINT_DIR):
        self.state_file = os.path.join(checkpoint_dir, "training_state.json")
        self.state = self._load_state()

    def _load_state(self):
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    return json.load(f)
        except json.JSONDecodeError:
            print(f"{TermColors.YELLOW}⚠️ Error reading state file. Starting fresh.{TermColors.ENDC}")
        except Exception as e:
            print(f"{TermColors.YELLOW}⚠️ Could not load training state: {e}. Starting fresh.{TermColors.ENDC}")
        return {"completed_chunks": []}

    def _save_state(self):
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            print(f"{TermColors.RED}❌ Error saving training state: {e}{TermColors.ENDC}")

    def mark_chunk_completed(self, chunk_idx):
        if chunk_idx not in self.state["completed_chunks"]:
            self.state["completed_chunks"].append(chunk_idx)
            self._save_state()

    def is_chunk_completed(self, chunk_idx):
        return chunk_idx in self.state["completed_chunks"]

class SWACallback(callbacks.Callback):
    """Stochastic Weight Averaging callback."""
    def __init__(self, start_epoch, swa_model_path):
        super().__init__()
        self.start_epoch = start_epoch
        self.swa_model_path = swa_model_path
        self.swa_weights = None
        self.n_averaged = tf.Variable(0, dtype=tf.int64)
        self.swa_model_saved = False # Flag to prevent saving multiple times if training continues

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.start_epoch and not self.swa_model_saved:
            current_weights = self.model.get_weights()
            if self.swa_weights is None: self.swa_weights = [np.copy(w) for w in current_weights]
            else: self.swa_weights = [(swa_w * tf.cast(self.n_averaged, swa_w.dtype) + current_w) / tf.cast(self.n_averaged + 1, swa_w.dtype) for swa_w, current_w in zip(self.swa_weights, current_weights)]
            self.n_averaged.assign_add(1)

    def on_train_end(self, logs=None):
        if self.n_averaged > 0 and not self.swa_model_saved:
            print(f"\n{TermColors.MAGENTA}Applying and saving SWA model (averaged over {self.n_averaged.numpy()} epochs)...{TermColors.ENDC}")
            self.model.set_weights(self.swa_weights)
            try:
                self.model.save(self.swa_model_path)
                print(f"{TermColors.GREEN}✅ SWA model saved to {self.swa_model_path}{TermColors.ENDC}")
                self.swa_model_saved = True
            except Exception as e:
                print(f"{TermColors.RED}❌ Error saving SWA model: {e}{TermColors.ENDC}")
        elif self.swa_model_saved:
             print(f"{TermColors.YELLOW}⚠️ SWA model already saved.{TermColors.ENDC}")
        else:
            print(f"{TermColors.YELLOW}⚠️ SWA did not run or average weights. Final model is the best checkpoint.{TermColors.ENDC}")

class MetricsDisplayCallback(callbacks.Callback):
    """Displays training progress and metrics in a structured way."""
    def __init__(self, total_epochs, validation_data, chunk_idx, update_freq=1):
        super().__init__()
        self.total_epochs = total_epochs
        self.validation_data = validation_data # Should be (X_val, y_val)
        self.chunk_idx = chunk_idx
        self.update_freq = update_freq
        self.epoch_start_time = None
        self.progbar = None

    def on_train_begin(self, logs=None):
        print(f"{TermColors.CYAN}Starting training for chunk {self.chunk_idx}...{TermColors.ENDC}")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        print(f"\n--- Chunk {self.chunk_idx} | Epoch {epoch + 1}/{self.total_epochs} ---")
        self.progbar = tf.keras.utils.Progbar(self.params['steps'], stateful_metrics=['lr'])

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        lr = self.model.optimizer.learning_rate
        if isinstance(lr, tf.Variable): lr_val = lr.numpy()
        elif hasattr(lr, '_current_learning_rate'): lr_val = lr._current_learning_rate # For schedules
        else: lr_val = tf.keras.backend.get_value(lr)
        self.progbar.update(batch + 1, values=[('loss', logs.get('loss')), ('acc', logs.get('accuracy')), ('lr', f"{lr_val:.1e}")])

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_time = time.time() - self.epoch_start_time

        val_loss = logs.get('val_loss')
        val_acc = logs.get('val_accuracy')
        val_top_k = logs.get(f'val_{sparse_top_k_accuracy.__name__}')

        metrics_str = f"Time: {epoch_time:.1f}s - Loss: {logs.get('loss'):.4f} - Acc: {logs.get('accuracy'):.4f}"
        if val_loss is not None: metrics_str += f" - Val Loss: {val_loss:.4f}"
        if val_acc is not None: metrics_str += f" - Val Acc: {val_acc:.4f}"
        if val_top_k is not None: metrics_str += f" - Val Top-{TOP_K}: {val_top_k:.4f}"

        print(f"\n{metrics_str}")

    def on_train_end(self, logs=None):
         print(f"{TermColors.GREEN}Training finished for chunk {self.chunk_idx}.{TermColors.ENDC}")

# --- Chunk Training ---
def train_chunk_model_with_swa(X_train, y_train, X_val, y_val, chunk_idx, training_state, class_mapping, feature_dim, confusability=0.0):
    global STOP_FLAG
    print(f"{TermColors.HEADER}\n{'='*50}\nTRAINING MODEL (SWA, Focal Loss) FOR CHUNK {chunk_idx}\n{'='*50}{TermColors.ENDC}")
    set_seeds(); cleanup_memory()
    if class_mapping is None or feature_dim is None: print(f"{TermColors.RED}❌ Missing class_mapping/feature_dim. Cannot train.{TermColors.ENDC}"); return None
    num_classes = len(class_mapping)
    print(f"{TermColors.CYAN}ℹ Training for {num_classes} classes, {feature_dim} features. Train: {X_train.shape}, Val: {X_val.shape}{TermColors.ENDC}")

    try:
        model = build_model(feature_dim, num_classes)

        class_weights = compute_balanced_class_weights(y_train)
        if class_weights: print(f"{TermColors.GREEN}✅ Class weights computed.{TermColors.ENDC}")

        model_base_path = os.path.join(MODEL_DIR, f"chunk_{chunk_idx}")
        best_model_path = f"{model_base_path}_best.keras"
        swa_model_path = f"{model_base_path}_swa.keras" # Path for SWA model

        swa_callback = SWACallback(start_epoch=SWA_START_EPOCH, swa_model_path=swa_model_path)
        metrics_display = MetricsDisplayCallback(total_epochs=CHUNK_EPOCHS, validation_data=(X_val, y_val), chunk_idx=chunk_idx)

        callbacks_list = [
            callbacks.ModelCheckpoint(filepath=best_model_path, save_best_only=True, monitor='val_loss', mode='min', verbose=0),
            callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1),
            swa_callback,
            metrics_display,
        ]

        print(f"{TermColors.CYAN}ℹ Training model for up to {CHUNK_EPOCHS} epochs (SWA starts epoch {SWA_START_EPOCH})... (Press 'q' or Ctrl+C to stop){TermColors.ENDC}")
        history = model.fit(
            X_train, y_train, validation_data=(X_val, y_val), epochs=CHUNK_EPOCHS,
            batch_size=CHUNK_TRAIN_BATCH_SIZE, callbacks=callbacks_list, class_weight=class_weights, verbose=0
        )
        final_epoch = len(history.history['loss'])

        final_model_path = swa_model_path if swa_callback.swa_model_saved else best_model_path
        if os.path.exists(final_model_path):
             print(f"{TermColors.CYAN}ℹ Evaluating final model from {os.path.basename(final_model_path)}...{TermColors.ENDC}")
             try:
                 eval_model = models.load_model(final_model_path, custom_objects={
                     'focal_loss': focal_loss,
                     'sparse_top_k_accuracy': sparse_top_k_accuracy
                 })
                 val_loss, val_acc, val_top_k = eval_model.evaluate(X_val, y_val, verbose=0)
                 print(f"\n{TermColors.HEADER}FINAL VALIDATION METRICS (Chunk {chunk_idx}):{TermColors.ENDC}")
                 print(f"{TermColors.RED}Loss: {val_loss:.4f}{TermColors.ENDC}")
                 print(f"{TermColors.GREEN}Accuracy: {val_acc:.4f} ({val_acc:.1%}){TermColors.ENDC}")
                 print(f"{TermColors.GREEN}Top-{TOP_K} Accuracy: {val_top_k:.4f} ({val_top_k:.1%}){TermColors.ENDC}")
                 del eval_model
             except Exception as eval_err:
                 print(f"{TermColors.RED}❌ Error evaluating final model: {eval_err}{TermColors.ENDC}")
                 val_loss, val_acc, val_top_k = -1, -1, -1
        else:
             print(f"{TermColors.RED}❌ Final model file not found at {final_model_path}. Cannot evaluate.{TermColors.ENDC}")
             val_loss, val_acc, val_top_k = -1, -1, -1

        metadata = {
            "chunk_idx": chunk_idx, "num_classes": num_classes, "feature_dim": feature_dim,
            "class_mapping": class_mapping,
            "performance": {"val_loss": float(val_loss), "val_accuracy": float(val_acc), f"val_sparse_top_{TOP_K}_accuracy": float(val_top_k)},
            "training_samples": int(X_train.shape[0]), "validation_samples": int(X_val.shape[0]),
            "confusability_metric": float(confusability), "epochs_trained": final_epoch,
            "stopped_early": final_epoch < CHUNK_EPOCHS, "used_swa": swa_callback.swa_model_saved,
            "n_swa_averaged": int(swa_callback.n_averaged.numpy()), "l2_regularization": L2_REG,
            "optimizer": model.optimizer.get_config()['name'], "loss": "focal_loss",
            "created": datetime.now().isoformat(), "seed": SEED
        }
        metadata_file = os.path.join(MODEL_DIR, f"chunk_{chunk_idx}_metadata.json")
        try:
            with open(metadata_file, "w") as f: json.dump(metadata, f, indent=2)
            print(f"{TermColors.GREEN}✅ Metadata saved to {metadata_file}{TermColors.ENDC}")
        except Exception as json_err: print(f"{TermColors.RED}❌ Error saving metadata: {json_err}{TermColors.ENDC}")

        if not STOP_FLAG: training_state.mark_chunk_completed(chunk_idx)
        cleanup_memory()
        return final_model_path if os.path.exists(final_model_path) else None

    except Exception as e:
        print(f"{TermColors.RED}❌ Error training model for chunk {chunk_idx}: {e}{TermColors.ENDC}")
        traceback.print_exc(); cleanup_memory(); return None

# --- Main Training Orchestration ---
def train_plants_advanced():
    global STOP_FLAG
    print(f"{TermColors.HEADER}\n{'='*60}\nSTARTING ADVANCED PLANT RECOGNITION TRAINING PIPELINE\n(PEFT Features, Contrastive, SWA, Focal Loss)\n{'='*60}{TermColors.ENDC}")
    set_seeds()
    feature_files = sorted(glob.glob(os.path.join(FEATURES_DIR, "chunk_*", "features.npz")))
    if not feature_files: print(f"{TermColors.RED}❌ No PEFT feature files found in {FEATURES_DIR}. Run PEFT tuning & extraction first. Exiting.{TermColors.ENDC}"); return
    print(f"{TermColors.CYAN}ℹ Found {len(feature_files)} PEFT feature chunks to process.{TermColors.ENDC}")

    training_state = TrainingState(CHECKPOINT_DIR)
    for i, features_file in enumerate(feature_files):
        if STOP_FLAG: print(f"{TermColors.YELLOW}⚠️ Stop requested during chunk training loop. Exiting.{TermColors.ENDC}"); break
        try:
            chunk_idx = int(os.path.basename(os.path.dirname(features_file)).split("_")[1])
            chunk_feature_dir = os.path.dirname(features_file)
            class_mapping_file = os.path.join(chunk_feature_dir, "class_mapping.json")
        except (IndexError, ValueError): print(f"{TermColors.YELLOW}⚠️ Invalid chunk path {features_file}. Skipping.{TermColors.ENDC}"); continue
        print(f"{TermColors.HEADER}\n--- Processing Chunk {chunk_idx} ({i+1}/{len(feature_files)}) ---{TermColors.ENDC}")
        if training_state.is_chunk_completed(chunk_idx): print(f"{TermColors.GREEN}✅ Chunk {chunk_idx} already completed, skipping...{TermColors.ENDC}"); continue

        try:
            print(f"{TermColors.CYAN}ℹ Loading features and metadata from {features_file}...{TermColors.ENDC}")
            with np.load(features_file) as data: features = data['features']; labels = data['labels']
            if not os.path.exists(class_mapping_file): raise FileNotFoundError(f"Mapping not found: {class_mapping_file}")
            with open(class_mapping_file) as f: class_mapping = json.load(f)
            print(f"{TermColors.GREEN}✅ Loaded {len(features)} samples for chunk {chunk_idx}.{TermColors.ENDC}")
            feature_dim = features.shape[1]

            print(f"{TermColors.CYAN}ℹ Applying contrastive learning enhancement...{TermColors.ENDC}")
            try: features_enhanced = contrastive_feature_learning(features, labels)
            except Exception as cle: print(f"{TermColors.RED}❌ Contrastive learning error: {cle}. Using original features.{TermColors.ENDC}"); features_enhanced = features

            print(f"{TermColors.CYAN}ℹ Standardizing features...{TermColors.ENDC}")
            scaler = StandardScaler(); features_scaled = scaler.fit_transform(features_enhanced)

            print(f"{TermColors.CYAN}ℹ Splitting features...{TermColors.ENDC}")
            X_train, X_val, y_train, y_val = train_test_split(features_scaled, labels, test_size=0.2, random_state=SEED, stratify=labels)
            print(f"{TermColors.GREEN}✅ Data split: {len(X_train)} train, {len(X_val)} val.{TermColors.ENDC}")

            print(f"{TermColors.CYAN}ℹ Balancing training data...{TermColors.ENDC}")
            X_train_balanced, y_train_balanced = balance_training_data(X_train, y_train, strategy='over')
            print(f"{TermColors.CYAN}ℹ Augmenting training features...{TermColors.ENDC}")
            X_train_aug, y_train_aug = augment_feature_space(X_train_balanced, y_train_balanced, augmentation_factor=FEATURE_AUG_FACTOR, noise_level=FEATURE_AUG_NOISE)

            confusability = 0.0

            print(f"{TermColors.CYAN}ℹ Starting training for chunk {chunk_idx}...{TermColors.ENDC}")
            train_chunk_model_with_swa(
                X_train=X_train_aug, y_train=y_train_aug, X_val=X_val, y_val=y_val,
                chunk_idx=chunk_idx, training_state=training_state, class_mapping=class_mapping,
                feature_dim=feature_dim, confusability=confusability
            )
            cleanup_memory()

        except FileNotFoundError as e: print(f"{TermColors.RED}❌ File not found error for chunk {chunk_idx}: {e}. Skipping.{TermColors.ENDC}"); continue
        except Exception as e: print(f"{TermColors.RED}❌ Error processing chunk {chunk_idx}: {e}{TermColors.ENDC}"); traceback.print_exc(); cleanup_memory(); continue

    print(f"\n{TermColors.HEADER}{'='*60}\nADVANCED TRAINING PIPELINE COMPLETED\n{'='*60}{TermColors.ENDC}")
    cleanup_memory()

# --- Future Stage Placeholders ---
def train_meta_model(): print(f"{TermColors.YELLOW}⚠️ Meta-model training not implemented.{TermColors.ENDC}")
def run_distillation_process(): print(f"{TermColors.YELLOW}⚠️ Distillation process not implemented.{TermColors.ENDC}"); return None
def run_quantization_process(model_path, quantization_type="int8"): print(f"{TermColors.YELLOW}⚠️ Quantization process not implemented.{TermColors.ENDC}"); return None

# --- Main Execution ---
if __name__ == "__main__":
    STOP_FLAG = False # Reset stop flag at start
    start_time = time.time()
    set_seeds() # Set seeds early

    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print(f"{TermColors.HEADER}\n{'='*50}\nPLANT RECOGNITION V2 - PEFT + ADVANCED TRAINING\n{'='*50}{TermColors.ENDC}")
        if gpus: print(f"{TermColors.CYAN}ℹ Found GPU: {tf.config.experimental.get_device_details(gpus[0])['device_name']}{TermColors.ENDC}")
        else: print(f"{TermColors.YELLOW}⚠️ Running on CPU.{TermColors.ENDC}")

        # --- STEP 0: PEFT Fine-tuning (Optional, run if LoRA weights don't exist or need update) ---
        run_peft = True # Set to False to skip PEFT fine-tuning
        if run_peft:
            print(f"\n{TermColors.HEADER}--- STEP 0: PEFT Fine-tuning ---{TermColors.ENDC}")
            peft_success_mobilenet = run_peft_fine_tuning(load_mobilenetv3, "mobilenetv3large", target_layer_names=MOBILENET_LORA_TARGETS)
            if STOP_FLAG: raise KeyboardInterrupt("Stop requested during PEFT.")
            peft_success_densenet = run_peft_fine_tuning(load_densenet121, "densenet121", target_layer_names=DENSENET_LORA_TARGETS)
            if STOP_FLAG: raise KeyboardInterrupt("Stop requested during PEFT.")
            if not peft_success_mobilenet and not peft_success_densenet:
                 print(f"{TermColors.YELLOW}⚠️ PEFT fine-tuning failed or was skipped for both models. Proceeding without PEFT adapters.{TermColors.ENDC}")
            else:
                 print(f"{TermColors.GREEN}✅ PEFT fine-tuning step completed (or skipped).{TermColors.ENDC}")
        else:
            print(f"{TermColors.YELLOW}⚠️ Skipping PEFT fine-tuning step.{TermColors.ENDC}")

        # --- STEP 1: Feature Extraction ---
        run_extraction = not os.path.exists(FEATURES_DIR) or not any(glob.glob(os.path.join(FEATURES_DIR, "chunk_*")))
        if run_extraction:
            print(f"\n{TermColors.HEADER}--- STEP 1: Feature Extraction ---{TermColors.ENDC}")
            extraction_success = extract_features_with_peft()
            if STOP_FLAG: raise KeyboardInterrupt("Stop requested during feature extraction.")
            if not extraction_success:
                print(f"{TermColors.RED}❌ Feature extraction failed. Cannot proceed with training. Exiting.{TermColors.ENDC}")
                exit(1)
            print(f"{TermColors.GREEN}✅ Feature extraction step completed.{TermColors.ENDC}")
        else:
             print(f"{TermColors.YELLOW}⚠️ Skipping feature extraction step (features found in {FEATURES_DIR}).{TermColors.ENDC}")

        # --- STEP 2: Train Chunk Classifiers ---
        print(f"\n{TermColors.HEADER}--- STEP 2: Train Chunk Classifiers ---{TermColors.ENDC}")
        train_plants_advanced()
        if STOP_FLAG: raise KeyboardInterrupt("Stop requested during chunk training.")

        # --- Optional Further Steps (Placeholders) ---
        # print(f"\n{TermColors.HEADER}--- Optional Steps ---{TermColors.ENDC}")
        # train_meta_model()
        # final_model = run_distillation_process()
        # if final_model: run_quantization_process(final_model)

    except KeyboardInterrupt:
        print(f"\n{TermColors.RED}🛑 Training interrupted by user (Ctrl+C or 'q').{TermColors.ENDC}")
    except Exception as e:
        print(f"\n{TermColors.RED}❌ An unexpected error occurred during the main execution:{TermColors.ENDC}")
        print(f"{TermColors.RED}{traceback.format_exc()}{TermColors.ENDC}")
    finally:
        end_time = time.time()
        duration = timedelta(seconds=end_time - start_time)
        print(f"\n{TermColors.HEADER}{'='*50}\nPIPELINE FINISHED\n{'='*50}{TermColors.ENDC}")
        print(f"{TermColors.CYAN}Total execution time: {str(duration)}{TermColors.ENDC}")
        cleanup_memory() # Final cleanup call
