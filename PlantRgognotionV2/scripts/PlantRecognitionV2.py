# --- Standard Library Imports ---
import os
import sys
import random
import numpy as np
import pandas as pd
import glob # For finding image files
import json
import time
import gc
import signal
import traceback
import multiprocessing
import warnings
import math
from enum import Enum
from datetime import datetime, timedelta
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import csv
from collections import defaultdict

# --- PyTorch Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, random_split, WeightedRandomSampler
from torchvision import transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR, LinearLR, ConstantLR, StepLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.optim.swa_utils import AveragedModel, SWALR

# --- Scikit-learn Imports ---
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import class_weight as sk_class_weight
from sklearn.linear_model import LogisticRegression # For Stacking
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
SAM_AVAILABLE = True

# --- Utility Imports ---
import colorama
from PIL import Image
import cv2
import joblib # For saving sklearn models

# --- Optional/Advanced Libraries ---
# Define global flags first
TIMM_AVAILABLE = False
TORCHMETRICS_AVAILABLE = False
ALBUMENTATIONS_AVAILABLE = False
ADAMP_AVAILABLE = False
SAM_AVAILABLE = False

# Now import each library once
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    print("FATAL: timm library not found. Required. Install with 'pip install timm'.")
    sys.exit(1)

try:
    import torchmetrics
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    print("WARN: torchmetrics library not found. Some metrics might be unavailable. Install with 'pip install torchmetrics'.")

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    print("WARN: albumentations library not found. Using torchvision transforms. Install with 'pip install albumentations'.")

try:
    from adamp import AdamP
    ADAMP_AVAILABLE = True
except ImportError:
    print("WARN: adamp optimizer not found. Will use AdamW instead. Install with 'pip install adamp'.")

try:
    from sam_optimizer.sam import SAM
    SAM_AVAILABLE = True
except ImportError:
    print("WARN: sam_optimizer not found. SAM optimizer will be disabled. Install from 'https://github.com/davda54/sam'.")

# Print library info just once at startup
_INITIALIZED = False

def print_library_info():
    global _INITIALIZED
    if _INITIALIZED:
        return
    
    print("INFO: timm library found.")
    if TORCHMETRICS_AVAILABLE: 
        print("INFO: torchmetrics library found.")
    if ALBUMENTATIONS_AVAILABLE:
        print("INFO: albumentations library found.")
    if ADAMP_AVAILABLE:
        print("INFO: adamp optimizer found.")
    if SAM_AVAILABLE:
        print("INFO: sam_optimizer found.")
    
    _INITIALIZED = True

# --- Terminal Colors ---
colorama.init(autoreset=True)
class TermColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    # Custom Colors
    INFO = '\033[94m' # Blue for info
    DEBUG = '\033[90m' # Grey for debug
    TRACE = '\033[90m' # Grey for trace
    ERROR = '\033[91m' # Red for errors
    SUCCESS = '\033[92m' # Green for success
    WARN = '\033[93m' # Yellow for warnings
    CRITICAL = '\033[91m' + '\033[1m' # Bold Red for critical
    # Simplified aliases
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    DIM = '\033[2m'

# --- Configuration ---
SEED = 42
DEBUG_MODE = False # Set to True for small dataset and fewer epochs

# --- Path Configuration ---
# Assume this script is in PlantRgognotionV2/scripts/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V2_DIR = os.path.dirname(SCRIPT_DIR)
# Go up one level from 'PlantRgognotionV2' to the project root 'PlantRecognition'
PROJECT_ROOT = os.path.dirname(V2_DIR)

# Define data and output directories based on calculated roots
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "plant_images") # Base directory containing species subfolders
CSV_PATH = os.path.join(DATA_DIR, "observations-561226.csv")

# Output directories will be inside the V2_DIR (PlantRgognotionV2)
BASE_CHECKPOINT_DIR = os.path.join(V2_DIR, "checkpoints_v2_pytorch")
BASE_LOG_DIR = os.path.join(V2_DIR, "logs_v2_pytorch")
BASE_MODEL_SAVE_DIR = os.path.join(V2_DIR, "models_v2_pytorch")
BASE_ERROR_ANALYSIS_DIR = os.path.join(V2_DIR, "error_analysis_pytorch")

# Create directories if they don't exist
os.makedirs(BASE_CHECKPOINT_DIR, exist_ok=True)
os.makedirs(BASE_LOG_DIR, exist_ok=True)
os.makedirs(BASE_MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(BASE_ERROR_ANALYSIS_DIR, exist_ok=True)

# --- Training Config (Defaults, potentially tuned by HPO) ---
# Define BATCH_SIZE before it's used in HPO_BATCH_SIZE
BATCH_SIZE = 32 if not DEBUG_MODE else 8
GRADIENT_ACCUMULATION_STEPS = 2 if not DEBUG_MODE else 1
LEARNING_RATE = 1e-4; WEIGHT_DECAY = 1e-5
OPTIMIZER_TYPE = 'AdamP' if ADAMP_AVAILABLE else 'AdamW'
USE_SAM = True if SAM_AVAILABLE else False
SAM_RHO = 0.05; SAM_ADAPTIVE = True
GRADIENT_CLIP_VAL = 1.0
PROGRESSIVE_RESIZING_STAGES = [
    (10 if not DEBUG_MODE else 1, (224, 224)),
    (15 if not DEBUG_MODE else 1, (384, 384)),
    (5 if not DEBUG_MODE else 1, (448, 448)), 
]
TOTAL_EPOCHS_PER_FOLD = sum(s[0] for s in PROGRESSIVE_RESIZING_STAGES)
CURRENT_IMAGE_SIZE = None # Will be set per stage

# --- Cross-Validation Config ---
N_FOLDS = 5 if not DEBUG_MODE else 2

# --- Hardware Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = multiprocessing.cpu_count() // 2 if multiprocessing.cpu_count() > 1 else 0
MIXED_PRECISION = True if DEVICE.type == 'cuda' else False
USE_TORCH_COMPILE = False # Set to True to try torch.compile (requires PyTorch 2.0+)

# --- Model Config (Defaults, will be updated by HPO if RUN_HPO=True) ---
MODEL_NAMES = ["tf_efficientnetv2_l_in21ft1k", "convnext_large_in22ft1k"]
DROP_PATH_RATE = 0.1; PRETRAINED = True; NUM_CLASSES = -1 # NUM_CLASSES updated after data loading
EMBEDDING_SIZE = 1024; DROPOUT_RATE = 0.3; GLOBAL_POOLING = 'avg'

# --- Metric Learning Config (Defaults, potentially tuned by HPO) ---
METRIC_LEARNING_TYPE = 'ArcFace'
ARCFACE_S = 30.0; ARCFACE_M = 0.6

# --- Loss Function & Imbalance Handling ---
LOSS_TYPE = 'CrossEntropy' # 'CrossEntropy', 'FocalLoss'
LABEL_SMOOTHING = 0.1
FOCAL_ALPHA = 0.25; FOCAL_GAMMA = 2.0
IMBALANCE_STRATEGY = 'WeightedSampler' # 'WeightedLoss', 'LogitAdjust', 'WeightedSampler', 'None'
LOGIT_ADJUSTMENT_TAU = 1.0
CLASS_FREQUENCIES = None; CLASS_PRIORS = None; CLASS_WEIGHTS = None # Calculated after data loading

# --- Learning Rate Scheduler Config ---
SCHEDULER_TYPE = 'CosineWarmRestarts' # 'CosineWarmRestarts', 'StepLR', 'ReduceLROnPlateau'
WARMUP_EPOCHS = 3; LR_MAX = LEARNING_RATE; LR_MIN = LEARNING_RATE * 0.01
T_0 = 5; T_MULT = 1 # For CosineWarmRestarts
STEP_LR_STEP_SIZE = 5; STEP_LR_GAMMA = 0.1 # For StepLR
PLATEAU_FACTOR = 0.1; PLATEAU_PATIENCE = 3; PLATEAU_MIN_LR = 1e-6 # For ReduceLROnPlateau
PLATEAU_MODE = 'min'; PLATEAU_MONITOR = 'val_loss'

# --- Augmentation Config ---
USE_RAND_AUGMENT = False # Set to True to use torchvision RandAugment (if not using Albumentations)
RAND_AUGMENT_N = 2; RAND_AUGMENT_M = 9
MIXUP_ALPHA = 0.8; CUTMIX_ALPHA = 1.0; AUG_PROBABILITY = 0.5 # Probability of applying Mixup OR Cutmix

# --- Averaging Config ---
USE_SWA = True
SWA_START_EPOCH_GLOBAL_FACTOR = 0.75 # Start SWA after 75% of total epochs per fold
SWA_LR_FACTOR = 0.05; SWA_ANNEAL_EPOCHS = 5
USE_EMA = True; EMA_DECAY = 0.999

# --- Checkpointing Config ---
CHECKPOINT_MONITOR = 'val_acc' # 'val_loss' or 'val_acc'
CHECKPOINT_MODE = 'max' if CHECKPOINT_MONITOR == 'val_acc' else 'min' # 'min' for loss, 'max' for accuracy
SAVE_TOP_K = 1 # How many best checkpoints to keep (currently saves 'best' and 'latest')

# --- Error Analysis Config ---
ERROR_LOG_BATCH_SIZE = 64
LOG_MISCLASSIFIED_IMAGES = True # Set to False to disable logging misclassified images

# --- Test Time Augmentation (TTA) Config ---
USE_TTA = True # Enable/disable TTA during validation
TTA_TRANSFORMS = None # Will be set in get_transforms

# --- Stacking Config ---
RUN_STACKING = True
STACKING_META_MODEL_PATH = os.path.join(BASE_MODEL_SAVE_DIR, "stacking_meta_model.joblib")
STACKING_OOF_PREDS_PATH = os.path.join(BASE_MODEL_SAVE_DIR, "oof_predictions.npz")

# --- Knowledge Distillation Config ---
RUN_KNOWLEDGE_DISTILLATION = True
KD_STUDENT_MODEL_NAME = "mobilenetv3_small_100" # Example student
KD_STUDENT_IMAGE_SIZE = (224, 224) # Student typically uses smaller input
KD_STUDENT_EMBEDDING_SIZE = 512 # Adjust based on student model
KD_STUDENT_DROPOUT = 0.2
KD_TEACHER_FOLD = 0 # Which fold's best model to use as teacher (0-based index)
KD_EPOCHS = 15 if not DEBUG_MODE else 2
KD_BATCH_SIZE = BATCH_SIZE * 2 # Can often use larger batch for smaller student
KD_LR = 1e-4
KD_ALPHA = 0.5 # Weight for KD loss (KLDiv) vs CrossEntropy loss (1-alpha)
KD_TEMPERATURE = 4.0 # Softening temperature for logits
KD_STUDENT_MODEL_SAVE_PATH = os.path.join(BASE_MODEL_SAVE_DIR, f"kd_student_{KD_STUDENT_MODEL_NAME}.pth")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- Global Variables ---
stop_requested = False; label_encoder = None; class_names = None

# --- Utility Functions ---
def set_seed(seed=SEED):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic algorithms are used where possible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"{TermColors.INFO}ℹ Seed set to {seed}{TermColors.ENDC}")

def handle_interrupt(signal, frame):
    """Handles Ctrl+C interrupts gracefully."""
    global stop_requested
    if stop_requested: # If Ctrl+C is pressed twice, force exit
        print(f"\n{TermColors.CRITICAL}🚨 Force exiting...{TermColors.ENDC}")
        sys.exit(1)
    print(f"\n{TermColors.WARNING}⚠️ Interrupt received. Finishing current epoch and saving state... Press Ctrl+C again to force exit.{TermColors.ENDC}")
    stop_requested = True

def check_keyboard_stop():
    """Checks if stop_requested flag is set."""
    if stop_requested:
        print(f"{TermColors.WARNING}Stop request detected. Breaking loop...{TermColors.ENDC}")
    return stop_requested

# --- Checkpointing, Saving, Logging ---
def save_checkpoint(fold, global_epoch, stage_idx, stage_epoch, model, optimizer, scheduler, scaler, best_metric, filename="checkpoint.pth.tar"):
    """Saves training checkpoint."""
    checkpoint_dir = os.path.join(BASE_CHECKPOINT_DIR, f"fold_{fold}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)

    model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    # Handle SAM optimizer state saving
    opt_state_dict = optimizer.base_optimizer.state_dict() if USE_SAM and hasattr(optimizer, 'base_optimizer') else optimizer.state_dict()

    state = {
        'fold': fold,
        'global_epoch': global_epoch,
        'stage_idx': stage_idx,
        'stage_epoch': stage_epoch,
        'image_size': CURRENT_IMAGE_SIZE, # Save current image size
        'state_dict': model_state_dict,
        'optimizer': opt_state_dict,
        'scheduler': scheduler.state_dict() if scheduler else None,
        'scaler': scaler.state_dict() if scaler else None,
        'best_metric': best_metric,
        'label_encoder_classes': list(label_encoder.classes_) if label_encoder else None, # Save label mapping
        'class_frequencies': CLASS_FREQUENCIES # Save class frequencies if needed
    }
    try:
        torch.save(state, filepath)
        print(f"{TermColors.GREEN}✅ Ckpt Fold {fold} saved: {filename} (GlobEp {global_epoch}, Best {CHECKPOINT_MONITOR}: {best_metric:.4f}){TermColors.ENDC}")
    except Exception as e:
        print(f"{TermColors.RED}❌ Error saving checkpoint {filepath}: {e}{TermColors.ENDC}")


def load_checkpoint(fold, model, optimizer, scheduler, scaler, filename="checkpoint.pth.tar"):
    """Loads training checkpoint."""
    checkpoint_dir = os.path.join(BASE_CHECKPOINT_DIR, f"fold_{fold}")
    filepath = os.path.join(checkpoint_dir, filename)

    start_global_epoch, start_stage_idx, start_stage_epoch = 0, 0, 0
    loaded_image_size, loaded_class_frequencies, loaded_label_classes = None, None, None
    best_metric = float('-inf') if CHECKPOINT_MODE == 'max' else float('inf') # Initialize best_metric correctly

    if os.path.isfile(filepath):
        print(f"{TermColors.CYAN}ℹ Loading Fold {fold} checkpoint '{filename}'...{TermColors.ENDC}")
        try:
            ckpt = torch.load(filepath, map_location=DEVICE)
            if ckpt.get('fold', -1) != fold: print(f"{TermColors.YELLOW}Warn: Checkpoint fold ({ckpt.get('fold')}) mismatch!{TermColors.ENDC}")

            start_global_epoch = ckpt.get('global_epoch', 0)
            start_stage_idx = ckpt.get('stage_idx', 0)
            start_stage_epoch = ckpt.get('stage_epoch', 0)
            loaded_image_size = ckpt.get('image_size', None)
            loaded_class_frequencies = ckpt.get('class_frequencies', None)
            best_metric = ckpt.get('best_metric', best_metric) # Load saved best metric
            loaded_label_classes = ckpt.get('label_encoder_classes', None)

            # --- Model State Loading ---
            state_dict = ckpt['state_dict']
            # Handle potential prefixes (module., _orig_mod.)
            new_state_dict = {}
            is_compiled = hasattr(model, '_orig_mod') # Check if model was compiled
            for k, v in state_dict.items():
                name = k
                if name.startswith('module.'): name = name[len('module.'):]
                # If model is compiled now, but checkpoint wasn't, add prefix
                if is_compiled and not name.startswith('_orig_mod.'): name = '_orig_mod.' + name
                # If model is NOT compiled now, but checkpoint was, remove prefix
                if not is_compiled and name.startswith('_orig_mod.'): name = name[len('_orig_mod.'):]
                new_state_dict[name] = v

            try:
                # Use strict=False to allow loading even if some keys mismatch (e.g., FC layer size change)
                model.load_state_dict(new_state_dict, strict=False)
                print(f"{TermColors.GREEN}  Model state loaded.{TermColors.ENDC}")
            except RuntimeError as e:
                 print(f"{TermColors.YELLOW}⚠️ Model Load Warning (strict=False): {e}{TermColors.ENDC}")
            except Exception as e:
                 print(f"{TermColors.RED}❌ Model Load Failed: {e}{TermColors.ENDC}")


            # --- Optimizer State Loading ---
            if optimizer and 'optimizer' in ckpt and ckpt['optimizer']:
                opt_to_load = optimizer.base_optimizer if USE_SAM and hasattr(optimizer, 'base_optimizer') else optimizer
                try:
                    opt_to_load.load_state_dict(ckpt['optimizer'])
                    print(f"{TermColors.GREEN}  Optimizer state loaded.{TermColors.ENDC}")
                    # Optionally move optimizer state to device if needed (usually handled by optimizer creation)
                    # for state in opt_to_load.state.values():
                    #     for k, v in state.items():
                    #         if isinstance(v, torch.Tensor):
                    #             state[k] = v.to(DEVICE)
                except Exception as e: print(f"{TermColors.YELLOW}⚠️ Optim Load Failed: {e}{TermColors.ENDC}")

            # --- Scheduler State Loading ---
            # Scheduler state is often loaded per-stage if resuming mid-stage
            if scheduler and 'scheduler' in ckpt and ckpt['scheduler']:
                 try:
                     scheduler.load_state_dict(ckpt['scheduler'])
                     print(f"{TermColors.GREEN}  Scheduler state loaded (initial).{TermColors.ENDC}")
                 except Exception as e: print(f"{TermColors.YELLOW}⚠️ Scheduler Load Failed (initial): {e}{TermColors.ENDC}")


            # --- Scaler State Loading ---
            if scaler and 'scaler' in ckpt and ckpt['scaler']:
                try:
                    scaler.load_state_dict(ckpt['scaler'])
                    print(f"{TermColors.GREEN}  Scaler state loaded.{TermColors.ENDC}")
                except Exception as e: print(f"{TermColors.YELLOW}⚠️ Scaler Load Failed: {e}{TermColors.ENDC}")

            print(f"{TermColors.GREEN}✅ Ckpt Fold {fold} loaded. Resume GlobEp {start_global_epoch}. Best {CHECKPOINT_MONITOR}: {best_metric:.4f}{TermColors.ENDC}")

            # --- Validation Checks ---
            if loaded_label_classes and label_encoder and list(label_encoder.classes_) != loaded_label_classes:
                print(f"{TermColors.CRITICAL}🚨 Label mapping mismatch! Checkpoint labels: {len(loaded_label_classes)}, Current labels: {len(label_encoder.classes_)}. Exiting.{TermColors.ENDC}")
                sys.exit(1)
            # Image size check happens per stage

        except Exception as e:
            print(f"{TermColors.RED}❌ Error loading checkpoint {filepath}: {e}{TermColors.ENDC}")
            traceback.print_exc()
            # Reset values if loading fails
            start_global_epoch, start_stage_idx, start_stage_epoch = 0, 0, 0
            best_metric = float('-inf') if CHECKPOINT_MODE == 'max' else float('inf')

    else:
        print(f"{TermColors.YELLOW}⚠️ No checkpoint found for Fold {fold} at {filepath}. Starting fresh.{TermColors.ENDC}")

    return start_global_epoch, start_stage_idx, start_stage_epoch, best_metric, loaded_label_classes, loaded_image_size, loaded_class_frequencies


def save_model(fold, model, filename="final_model.pth"):
    """Saves the model state_dict."""
    model_dir = os.path.join(BASE_MODEL_SAVE_DIR, f"fold_{fold}")
    os.makedirs(model_dir, exist_ok=True)
    filepath = os.path.join(model_dir, filename)

    # Get the underlying model if wrapped (e.g., by DP, DDP, compile)
    model_to_save = model
    if hasattr(model_to_save, 'module'): model_to_save = model_to_save.module
    if hasattr(model_to_save, '_orig_mod'): model_to_save = model_to_save._orig_mod # For torch.compile

    try:
        torch.save(model_to_save.state_dict(), filepath)
        print(f"{TermColors.GREEN}✅ Fold {fold} model state_dict saved: {filename}{TermColors.ENDC}")
    except Exception as e:
        print(f"{TermColors.RED}❌ Error saving model state_dict {filepath}: {e}{TermColors.ENDC}")


def log_misclassified(fold, model, dataloader, criterion, device, global_epoch, writer, num_classes, max_images=20):
    """Logs misclassified images and details to CSV and TensorBoard."""
    if not LOG_MISCLASSIFIED_IMAGES: return

    error_dir = os.path.join(BASE_ERROR_ANALYSIS_DIR, f"fold_{fold}")
    os.makedirs(error_dir, exist_ok=True)
    error_log_file = os.path.join(error_dir, f"epoch_{global_epoch}_errors.csv")

    model.eval(); misclassified_count = 0; logged_images = 0
    print(f"{TermColors.CYAN}ℹ Fold {fold} Logging misclassified images for global epoch {global_epoch}...{TermColors.ENDC}")

    # Prepare CSV logging
    try:
        with open(error_log_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['image_path', 'true_label', 'predicted_label', 'confidence', 'loss', 'logits_raw', 'logits_adjusted']
            writer_csv = csv.DictWriter(csvfile, fieldnames=fieldnames); writer_csv.writeheader()

            with torch.no_grad():
                for batch_data in tqdm(dataloader, desc=f"Logging Errors Fold {fold}", leave=False):
                    # Expect dataloader to yield (inputs, labels, paths) when include_paths=True
                    if len(batch_data) != 3:
                        print(f"{TermColors.YELLOW}Warn: Skipping error logging batch, expected 3 items (inputs, labels, paths), got {len(batch_data)}.{TermColors.ENDC}")
                        continue
                    inputs, labels, paths = batch_data
                    inputs, labels = inputs.to(device), labels.to(device)

                    with torch.amp.autocast('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda')):
                        embeddings = model(inputs)
                        # Handle potential mismatch if model doesn't have metric_fc
                        if hasattr(model, 'metric_fc'):
                            outputs = model.metric_fc(embeddings, labels) # ArcFace needs labels
                        elif hasattr(model, 'fc'): # Assume standard fc layer
                             outputs = model(inputs) # If model directly outputs logits
                        else: # Fallback for simple timm models
                            outputs = model(inputs) # Assume model(inputs) gives logits

                        # Apply logit adjustment if used
                        adj_outputs = outputs
                        if IMBALANCE_STRATEGY == 'LogitAdjust' and CLASS_PRIORS is not None:
                            logit_adj = LOGIT_ADJUSTMENT_TAU * torch.log(CLASS_PRIORS + 1e-12)
                            adj_outputs = outputs + logit_adj.unsqueeze(0)

                        loss = criterion(adj_outputs, labels) # Use adjusted outputs for loss calc? Or raw? Using adjusted.
                        preds = torch.argmax(adj_outputs, dim=1)
                        probs = F.softmax(adj_outputs, dim=1)

                    # Find misclassified samples in the batch
                    misclassified_mask = (preds != labels)
                    misclassified_indices = torch.where(misclassified_mask)[0]

                    for idx in misclassified_indices:
                        misclassified_count += 1
                        true_label_idx = labels[idx].item()
                        pred_label_idx = preds[idx].item()
                        confidence = probs[idx, pred_label_idx].item()
                        item_loss = F.cross_entropy(adj_outputs[idx].unsqueeze(0), labels[idx].unsqueeze(0)).item() # Loss for this item
                        img_path = paths[idx]
                        true_n = class_names[true_label_idx] if class_names and 0 <= true_label_idx < len(class_names) else str(true_label_idx)
                        pred_n = class_names[pred_label_idx] if class_names and 0 <= pred_label_idx < len(class_names) else str(pred_label_idx)

                        # Log to CSV
                        writer_csv.writerow({
                            'image_path': os.path.basename(img_path), # Log only filename to CSV
                            'true_label': true_n,
                            'predicted_label': pred_n,
                            'confidence': f"{confidence:.4f}",
                            'loss': f"{item_loss:.4f}",
                            'logits_raw': outputs[idx].cpu().numpy().round(2).tolist(),
                            'logits_adjusted': adj_outputs[idx].cpu().numpy().round(2).tolist()
                        })

                        # Log images to TensorBoard (limited number)
                        if writer and logged_images < max_images:
                            try:
                                # Denormalize and prepare image for TensorBoard
                                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
                                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
                                img_tensor = inputs[idx] * std + mean
                                img_tensor = torch.clamp(img_tensor, 0, 1)
                                writer.add_image(f"Misclassified/Fold_{fold}/True_{true_n}_Pred_{pred_n}_{os.path.basename(img_path)}",
                                                 img_tensor, global_epoch)
                                logged_images += 1
                            except Exception as img_e:
                                print(f"{TermColors.YELLOW}Warn: Failed to log image {img_path} to TensorBoard: {img_e}{TermColors.ENDC}")

                    if stop_requested: break
            if stop_requested: return # Exit if interrupted

        print(f"{TermColors.CYAN}ℹ Fold {fold} Misclassified images logged ({misclassified_count} total errors). CSV: {error_log_file}{TermColors.ENDC}")

    except Exception as e:
        print(f"{TermColors.RED}❌ Error during misclassified logging for fold {fold}: {e}{TermColors.ENDC}")
        traceback.print_exc()


# --- Dataset and Transforms ---
class PlantDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None, label_encoder=None, include_paths=False, image_size=None):
        # Input dataframe should have 'scientificName', 'id', 'label'
        self.input_df = dataframe.copy()
        self.image_dir = image_dir
        self.transform = transform
        # label_encoder is not directly used here anymore, labels come from input_df
        self.include_paths = include_paths
        self.image_size = image_size if image_size else PROGRESSIVE_RESIZING_STAGES[0][1]

        self.image_data = [] # List to store {'scientificName': ..., 'label': ..., 'image_path': ..., 'original_index': ...}

        print(f"{TermColors.CYAN}ℹ Scanning for images based on input dataframe (Size: {self.image_size})...{TermColors.ENDC}")

        # Check required columns in the input dataframe
        required_cols = ['scientificName', 'id', 'label']
        if not all(col in self.input_df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in self.input_df.columns]
            print(f"{TermColors.RED}❌ PlantDataset input dataframe missing required columns: {missing}. Found: {self.input_df.columns.tolist()}{TermColors.ENDC}")
            self.dataframe = pd.DataFrame(self.image_data) # Create empty dataframe
            return # Stop init

        # Store original index from input_df
        self.input_df['original_index'] = self.input_df.index

        # Iterate through the original dataframe entries
        for idx, row in tqdm(self.input_df.iterrows(), total=len(self.input_df), desc="Finding image files", leave=False):
            try:
                species_name = str(row['scientificName'])
                obs_id = str(row['id'])
                label = row['label'] # Get the pre-encoded label
                original_index = row['original_index'] # Get the original index

                # Construct species directory path
                species_dir_name = species_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                species_dir_path = os.path.join(self.image_dir, species_dir_name)

                if os.path.isdir(species_dir_path):
                    # Construct glob pattern to find original and augmented images for this ID
                    # Pattern: SPECIES_NAME_ID_*.[jJ][pP]*[gG] (handles .jpg, .JPG, .jpeg, .JPEG)
                    glob_pattern = os.path.join(species_dir_path, f"{species_dir_name}_{obs_id}_*.[jJ][pP]*[gG]")
                    found_files = glob.glob(glob_pattern)

                    if not found_files: # Also try finding the base image if glob fails (e.g., no _0 suffix)
                         base_pattern = os.path.join(species_dir_path, f"{species_dir_name}_{obs_id}.[jJ][pP]*[gG]")
                         found_files = glob.glob(base_pattern)


                    for full_path in found_files:
                        if os.path.getsize(full_path) > 0:
                            self.image_data.append({
                                'scientificName': species_name,
                                'label': label,
                                'image_path': full_path, # Store the full path to the image file
                                'original_index': original_index # Store original index for OOF mapping
                            })
                        # else: # Optional: log skipped empty files
                        #    print(f"{TermColors.YELLOW}Warn: Skipping empty file: {full_path}{TermColors.ENDC}")

                # else: # Optional: log missing species directories
                #    if idx < 10: # Log only a few times
                #        print(f"{TermColors.YELLOW}Warn: Species directory not found: {species_dir_path}{TermColors.ENDC}")

            except Exception as e:
                print(f"{TermColors.RED}Error processing row {idx} (ID: {row.get('id', 'N/A')}): {e}{TermColors.ENDC}")

        # Create the final dataframe for the dataset from the found image data
        self.dataframe = pd.DataFrame(self.image_data)

        found_count = len(self.dataframe)
        if found_count == 0 and len(self.input_df) > 0:
             print(f"{TermColors.RED}❌ Found 0 image files after scanning. Check image paths, filenames, and CSV IDs.{TermColors.ENDC}")
             # Provide an example of the expected structure
             example_row = self.input_df.iloc[0]
             example_species_dir = str(example_row['scientificName']).replace(' ', '_').replace('/', '_').replace('\\', '_')
             example_id = str(example_row['id'])
             print(f"Example expected file pattern: {os.path.join(self.image_dir, example_species_dir, f'{example_species_dir}_{example_id}_*.jpg')}")
        else:
             print(f"{TermColors.GREEN}✅ Dataset initialized with {found_count} image files (including augmentations).{TermColors.ENDC}")


    def __len__(self):
        # Length is now the number of found image files
        return len(self.dataframe)

    def get_labels(self):
        # Get labels from the final internal dataframe
        if 'label' in self.dataframe.columns:
            return self.dataframe['label'].tolist()
        else:
            print(f"{TermColors.RED}❌ 'label' column missing in internal dataframe for get_labels!{TermColors.ENDC}")
            return []

    def __getitem__(self, idx):
        if idx >= len(self.dataframe):
             print(f"{TermColors.RED}❌ Index {idx} out of bounds for PlantDataset (len: {len(self.dataframe)}). Returning dummy.{TermColors.ENDC}")
             dummy_img = torch.zeros((3, *self.image_size), dtype=torch.float32); label = -1
             # Return original index as -1 or None for error case if needed for OOF
             return (dummy_img, label, "ERROR_INDEX_OOB", -1) if self.include_paths else (dummy_img, label)

        # Get data for the specific image file
        row = self.dataframe.iloc[idx]
        img_path = row['image_path'] # This is the full path now
        label = row['label']
        original_index = row['original_index'] # Get original index for OOF
        # scientificName = row['scientificName'] # Available if needed

        try:
            # Load the image using the full path
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
            if image is None: raise IOError("Image loading returned None")

        except Exception as e:
            print(f"{TermColors.RED}Err load {img_path}: {e}. Dummy.{TermColors.ENDC}")
            dummy_img = torch.zeros((3, *self.image_size), dtype=torch.float32); label = -1
            error_filename = os.path.basename(img_path)
            # Return original index as -1 or None for error case
            return (dummy_img, label, f"ERROR_LOAD_{error_filename}", original_index) if self.include_paths else (dummy_img, label)

        # Apply transforms
        if self.transform:
            try:
                augmented = self.transform(image=image)
                image = augmented['image']
            except Exception as e:
                print(f"{TermColors.RED}Err transform {img_path}: {e}. Fallback.{TermColors.ENDC}")
                try:
                    image = Image.fromarray(image) # Back to PIL
                    fallback_transform = transforms.Compose([
                        transforms.Resize(self.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    image = fallback_transform(image)
                except Exception as fb_e:
                    print(f"{TermColors.RED}Fallback failed {img_path}: {fb_e}. Dummy.{TermColors.ENDC}")
                    dummy_img = torch.zeros((3, *self.image_size), dtype=torch.float32); label = -1
                    error_filename = os.path.basename(img_path)
                    # Return original index as -1 or None for error case
                    return (dummy_img, label, f"ERROR_TF_{error_filename}", original_index) if self.include_paths else (dummy_img, label)

        # Return image, label, and potentially path and original_index
        if self.include_paths:
            return image, label, img_path, original_index
        else:
            return image, label


def get_transforms(image_size=(224, 224), augmentations_config=None):
    """
    Generates simplified train and validation image transformations,
    suitable for use with pre-augmented offline data.
    """
    print(f"{TermColors.CYAN}ℹ Generating SIMPLIFIED transforms for image size: {image_size}{TermColors.ENDC}")
    h = int(image_size[0])
    w = int(image_size[1])

    # --- Simplified Training Transforms ---
    train_transform_list = [
        A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
    train_transform = A.Compose(train_transform_list)
    print(f"{TermColors.CYAN}  Using minimal online training augmentations: Resize, HFlip, Mild ColorJitter, Normalize, ToTensorV2.{TermColors.ENDC}")


    # --- Validation Transforms ---
    val_transform = A.Compose([
        A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    # --- TTA Transforms (Example: Horizontal Flip + Resize) ---
    global TTA_TRANSFORMS 
    if USE_TTA:
        TTA_TRANSFORMS = A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
        print(f"{TermColors.CYAN}ℹ TTA transforms defined (Example: HFlip).{TermColors.ENDC}")
    else:
        TTA_TRANSFORMS = None

    return train_transform, val_transform


# --- Model Architecture ---
class ArcFace(nn.Module):
    """ ArcFace (Additive Angular Margin Loss) module """
    def __init__(self, in_features, out_features, s=ARCFACE_S, m=ARCFACE_M, easy_margin=False, ls_eps=0.0):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        # If labels are None, we're in inference mode - just return cosine similarity * scale
        if label is None:
            return cosine * self.s
            
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).clamp(0, 1) # Sine component
        phi = cosine * self.cos_m - sine * self.sin_m # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device=DEVICE)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0: # Apply label smoothing
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s # Scale the logits

        return output


class CombinedModel(nn.Module):
    """Combines multiple backbones with optional ArcFace head."""
    def __init__(self, model_names, num_classes, pretrained=True, global_pool='avg', dropout_rate=0.3, embedding_size=512,
                 drop_path_rate=0.1, arcface_s=ARCFACE_S, arcface_m=ARCFACE_M, metric_learning=METRIC_LEARNING_TYPE):
        super().__init__()
        self.model_names = model_names
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.metric_learning = metric_learning
        self.backbones = nn.ModuleList()
        self.total_features = 0
        self.feature_dims = {}  # To store special case feature dimensions

        print(f"{TermColors.INFO}ℹ Building Combined Model: {', '.join(model_names)}{TermColors.ENDC}")

        for name in model_names:
            print(f"  Loading backbone: {name}...")
            try:
                kwargs = {}
                supported_families = ['efficientnet', 'convnext', 'vit', 'swin', 'beit', 'deit']
                model_family = name.split('_')[0].lower() # Basic way to get family

                if any(family in model_family for family in supported_families) and drop_path_rate > 0:
                    kwargs['drop_path_rate'] = drop_path_rate
                    print(f"    Applying drop_path_rate: {drop_path_rate}")

                backbone = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool=global_pool, **kwargs)

                # Get the correct feature dimension
                # Handle special cases for models with mismatched num_features
                if 'mobilenetv3' in name:
                    # MobileNetV3 may return 1280 features instead of 960
                    backbone_features = 1280
                    print(f"    {name} loaded. Features: {backbone_features} (overriding reported {backbone.num_features})")
                    self.feature_dims[name] = backbone_features
                elif 'efficientnetv2' in name:
                    # EfficientNetV2 models may also have mismatched feature dimensions
                    if 's' in name:  # Small variant
                        backbone_features = 1280
                    elif 'm' in name:  # Medium variant
                        backbone_features = 1280
                    elif 'l' in name:  # Large variant
                        backbone_features = 1280
                    else:
                        backbone_features = backbone.num_features
                    print(f"    {name} loaded. Features: {backbone_features} (overriding reported {backbone.num_features})")
                    self.feature_dims[name] = backbone_features
                else:
                    # For other models, use standard num_features attribute
                    backbone_features = backbone.num_features
                    print(f"    {name} loaded. Features: {backbone_features}")
                    self.feature_dims[name] = backbone_features
                
                self.backbones.append(backbone)
                self.total_features += backbone_features

            except Exception as e:
                print(f"{TermColors.RED}❌ Backbone Load Fail {name}: {e}{TermColors.ENDC}")
                # traceback.print_exc() # Uncomment for debugging
                raise e # Re-raise the exception to stop execution if a backbone fails

        print(f"  Total backbone features: {self.total_features}, Target Embedding: {self.embedding_size}")

        self.embedding_layer = nn.Sequential(
            nn.Linear(self.total_features, self.embedding_size),
            nn.BatchNorm1d(self.embedding_size),
            nn.ReLU(inplace=True)
        )
        current_features = self.embedding_size
        print(f"  Added embedding layer: {self.total_features} -> {self.embedding_size}")

        # Dropout Layer
        self.dropout = nn.Dropout(dropout_rate)

        # Final Classification Head (ArcFace or Linear)
        if self.metric_learning == 'ArcFace':
            print(f"  Using ArcFace head (S={arcface_s}, M={arcface_m})")
            self.metric_fc = ArcFace(current_features, num_classes, s=arcface_s, m=arcface_m)
        else:
            print(f"  Using standard Linear head")
            self.metric_fc = nn.Linear(current_features, num_classes)


    def forward(self, x, labels=None):
        """Forward pass through the combined model."""
        # Extract features from all backbones
        all_features = []
        for i, backbone in enumerate(self.backbones):
            features = backbone(x)
            all_features.append(features)

        # Concatenate features if multiple backbones exist
        if len(all_features) > 1:
            combined_features = torch.cat(all_features, dim=1)
        else:
            combined_features = all_features[0]

        # Pass through embedding layer (if exists) and dropout
        embedding = self.embedding_layer(combined_features)
        embedding = self.dropout(embedding)

        # Pass through final classification head
        if self.metric_learning == 'ArcFace':
            # ArcFace requires labels during training and potentially eval
            if labels is None:
                output = self.metric_fc(embedding, labels) # Pass None labels
            else:
                output = self.metric_fc(embedding, labels)
        else:
            # Standard linear layer
            output = self.metric_fc(embedding)

        return output


def build_model(model_names=MODEL_NAMES, num_classes=NUM_CLASSES, pretrained=PRETRAINED, dropout_rate=DROPOUT_RATE,
                embedding_size=EMBEDDING_SIZE, drop_path_rate=DROP_PATH_RATE, global_pool=GLOBAL_POOLING,
                arcface_s=ARCFACE_S, arcface_m=ARCFACE_M, metric_learning=METRIC_LEARNING_TYPE):
    """Builds the combined model."""
    # Print exact model configuration for debugging
    print(f"Building model with: model_names={model_names}, embedding_size={embedding_size}, num_classes={num_classes}")

    # Create model instance
    model = CombinedModel(model_names, num_classes, pretrained, global_pool, dropout_rate, embedding_size,
                         drop_path_rate, arcface_s, arcface_m, metric_learning)
    
    return model

# --- Loss Functions ---
class FocalLoss(nn.Module):
    """ Focal Loss for imbalanced datasets """
    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_criterion(loss_type=LOSS_TYPE, label_smoothing=LABEL_SMOOTHING, class_weights=CLASS_WEIGHTS):
    """Gets the appropriate loss function."""
    weights = class_weights.to(DEVICE) if class_weights is not None and IMBALANCE_STRATEGY == 'WeightedLoss' else None
    if weights is not None:
        print(f"{TermColors.INFO}  Applying class weights to loss function.{TermColors.ENDC}")

    if loss_type == 'FocalLoss':
        print(f"{TermColors.INFO}  Using Focal Loss (alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA}){TermColors.ENDC}")
        return FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    elif loss_type == 'CrossEntropy':
        print(f"{TermColors.INFO}  Using Cross Entropy Loss (Smoothing={label_smoothing}){TermColors.ENDC}")
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=weights)
    else:
        print(f"{TermColors.WARNING}Warn: Unknown loss type '{loss_type}'. Defaulting to CrossEntropy.{TermColors.ENDC}")
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=weights)


# --- Optimizer and Scheduler ---
def get_optimizer(model, optimizer_type=OPTIMIZER_TYPE, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, use_sam=USE_SAM, sam_rho=SAM_RHO, sam_adaptive=SAM_ADAPTIVE):
    """Gets the optimizer."""
    # Get only trainable parameters
    params = [p for p in model.parameters() if p.requires_grad]
    
    # Check if we have trainable parameters
    if len(params) == 0:
        print(f"{TermColors.RED}❌ Error: No trainable parameters found in model!{TermColors.ENDC}")
        raise ValueError("No trainable parameters in model")
        
    print(f"{TermColors.INFO}  Using Optimizer: {optimizer_type} (LR={lr:.1E}, WD={weight_decay:.1E}, SAM={use_sam}){TermColors.ENDC}")

    if use_sam and SAM_AVAILABLE:
        print(f"    SAM settings: rho={sam_rho}, adaptive={sam_adaptive}")
        
        # Use partial to properly create optimizer function for SAM
        from functools import partial
        
        if optimizer_type == 'AdamP' and ADAMP_AVAILABLE:
            base_optimizer = partial(AdamP, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), nesterov=True)
        elif optimizer_type == 'AdamW':
            base_optimizer = partial(optim.AdamW, lr=lr, weight_decay=weight_decay)
        else:
            base_optimizer = partial(optim.AdamW, lr=lr, weight_decay=weight_decay)
            
        return SAM(params, base_optimizer, rho=sam_rho, adaptive=sam_adaptive)
    else:
        # Standard optimizers - no change needed
        if optimizer_type == 'AdamP' and ADAMP_AVAILABLE:
            return AdamP(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), nesterov=True)
        elif optimizer_type == 'AdamW':
            return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'SGD':
            return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)
        else:
            print(f"{TermColors.YELLOW}⚠️ Unknown optimizer '{optimizer_type}'. Using AdamW.{TermColors.ENDC}")
            return optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def get_scheduler(optimizer, scheduler_type=SCHEDULER_TYPE, total_epochs=TOTAL_EPOCHS_PER_FOLD, warmup_epochs=WARMUP_EPOCHS, lr_max=LEARNING_RATE, lr_min=LR_MIN, t_0=T_0, t_mult=T_MULT, step_size=STEP_LR_STEP_SIZE, gamma=STEP_LR_GAMMA, plateau_factor=PLATEAU_FACTOR, plateau_patience=PLATEAU_PATIENCE, plateau_min_lr=PLATEAU_MIN_LR, plateau_mode=PLATEAU_MODE):
    """Gets the learning rate scheduler."""
    print(f"{TermColors.INFO}  Using LR Scheduler: {scheduler_type}{TermColors.ENDC}")

    # Handle SAM optimizer - scheduler should wrap the base optimizer's param groups
    opt_for_scheduler = optimizer.base_optimizer if hasattr(optimizer, 'base_optimizer') else optimizer

    if scheduler_type == 'CosineWarmRestarts':
        print(f"    CosineWarmRestarts: T_0={t_0}, T_mult={t_mult}, LR_min={lr_min:.1E}")
        # Simple Cosine Annealing with Warm Restarts (no separate warmup phase here)
        return CosineAnnealingWarmRestarts(opt_for_scheduler, T_0=t_0, T_mult=t_mult, eta_min=lr_min)
    elif scheduler_type == 'StepLR':
        print(f"    StepLR: step_size={step_size}, gamma={gamma}")
        return StepLR(opt_for_scheduler, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'ReduceLROnPlateau':
        print(f"    ReduceLROnPlateau: factor={plateau_factor}, patience={plateau_patience}, min_lr={plateau_min_lr:.1E}, mode={plateau_mode}")
        return ReduceLROnPlateau(opt_for_scheduler, mode=plateau_mode, factor=plateau_factor, patience=plateau_patience, min_lr=plateau_min_lr, verbose=True)
    else:
        print(f"{TermColors.WARNING}Warn: Unknown scheduler type '{scheduler_type}'. No scheduler used.{TermColors.ENDC}")
        return None


# --- Data Augmentation Helpers ---
def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    mixed_x = x.clone()
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    # Adjust lambda to match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# --- EMA Helper ---
@torch.no_grad()
def ema_avg_fn(averaged_model_parameter, model_parameter, num_averaged):
    """ Exponential Moving Average update rule. """
    return EMA_DECAY * averaged_model_parameter + (1 - EMA_DECAY) * model_parameter

# --- Training & Validation Loops ---
def train_one_epoch(model, dataloader, criterion, optimizer, scaler, scheduler, global_epoch, stage_idx, stage_epoch, stage_total_epochs, device, writer, num_classes, ema_model,
                    mixup_alpha=MIXUP_ALPHA, cutmix_alpha=CUTMIX_ALPHA, aug_probability=AUG_PROBABILITY, grad_clip_val=GRADIENT_CLIP_VAL,
                    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS, use_sam=USE_SAM, fold_num=0):
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0
    total_samples = 0
    all_preds, all_labels = [], []
    is_sam = hasattr(optimizer, 'base_optimizer') and use_sam # Check if SAM is active

    # Simplified description - minimal string operations
    pbar_desc = f"HPO Train Ep {global_epoch+1}/{stage_total_epochs}" if fold_num == "HPO" else f"F{fold_num} S{stage_idx+1} E{stage_epoch+1}/{stage_total_epochs} Tr"

    # Use less verbose progress bar
    progress_bar = tqdm(dataloader, desc=pbar_desc, leave=True, disable=fold_num == "HPO", 
                       bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}')  # Enhanced progress bar format
    optimizer.zero_grad() # Zero grad at the beginning
    
    # Epoch progress bar to track overall epoch progress
    total_batches = len(dataloader)
    epoch_progress = tqdm(total=total_batches, desc=f"Epoch {global_epoch+1} Progress", 
                         leave=True, position=0, bar_format='{desc}: {percentage:3.0f}%|{bar:50}{r_bar}')

    for batch_idx, batch_data in enumerate(progress_bar):
        if check_keyboard_stop():
            break
            
        # Expect dataloader to yield (inputs, labels)
        if len(batch_data) != 2:
            continue

        inputs, labels = batch_data
        inputs, labels = inputs.to(device), labels.to(device)
        batch_size = inputs.size(0)

        # --- Apply Mixup/Cutmix ---
        use_mixup, use_cutmix = False, False
        r = np.random.rand()
        if mixup_alpha > 0 and cutmix_alpha > 0 and r < aug_probability:
            if np.random.rand() < 0.5:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, mixup_alpha, device)
                use_mixup = True
            else:
                inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, cutmix_alpha, device)
                use_cutmix = True
        elif mixup_alpha > 0 and r < aug_probability:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, mixup_alpha, device)
            use_mixup = True
        elif cutmix_alpha > 0 and r < aug_probability:
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, cutmix_alpha, device)
            use_cutmix = True
        else:
            lam = 1.0
            targets_a, targets_b = labels, labels

        # --- Forward Pass ---
        with torch.amp.autocast('cuda', enabled=(MIXED_PRECISION and device.type == 'cuda')):
            # Helper function definitions moved outside loop for efficiency
            if is_sam:
                # First forward-backward pass
                embeddings = model(inputs)
                outputs = model.metric_fc(embeddings, targets_a) if hasattr(model, 'metric_fc') and METRIC_LEARNING_TYPE == 'ArcFace' else model.metric_fc(embeddings) if hasattr(model, 'metric_fc') else embeddings
                
                # Apply logit adjustment if needed
                adj_outputs = outputs
                if IMBALANCE_STRATEGY == 'LogitAdjust' and CLASS_PRIORS is not None:
                    logit_adj = LOGIT_ADJUSTMENT_TAU * torch.log(CLASS_PRIORS + 1e-12)
                    adj_outputs = outputs + logit_adj.unsqueeze(0)
                
                # Calculate loss
                if use_mixup or use_cutmix:
                    loss1 = mixup_criterion(criterion, adj_outputs, targets_a, targets_b, lam)
                else:
                    loss1 = criterion(adj_outputs, targets_a)
                    
                loss1_scaled = loss1 / gradient_accumulation_steps
                scaler.scale(loss1_scaled).backward()
                
                # First optimizer step
                optimizer.first_step(zero_grad=True)
                
                # Second forward-backward pass
                embeddings_perturbed = model(inputs)
                outputs_perturbed = model.metric_fc(embeddings_perturbed, targets_a) if hasattr(model, 'metric_fc') and METRIC_LEARNING_TYPE == 'ArcFace' else model.metric_fc(embeddings_perturbed) if hasattr(model, 'metric_fc') else embeddings_perturbed
                
                # Apply logit adjustment again
                adj_outputs_final = outputs_perturbed
                if IMBALANCE_STRATEGY == 'LogitAdjust' and CLASS_PRIORS is not None:
                    adj_outputs_final = outputs_perturbed + logit_adj.unsqueeze(0)
                
                # Calculate second loss
                if use_mixup or use_cutmix:
                    loss2 = mixup_criterion(criterion, adj_outputs_final, targets_a, targets_b, lam)
                else:
                    loss2 = criterion(adj_outputs_final, targets_a)
                    
                loss2_scaled = loss2 / gradient_accumulation_steps
                scaler.scale(loss2_scaled).backward()
                
                # Second optimizer step
                optimizer.second_step(zero_grad=True)
                
                loss_final = loss2  # Use loss from second step for reporting
            else:
                # Standard training path
                embeddings = model(inputs)
                outputs = model.metric_fc(embeddings, targets_a) if hasattr(model, 'metric_fc') and METRIC_LEARNING_TYPE == 'ArcFace' else model.metric_fc(embeddings) if hasattr(model, 'metric_fc') else embeddings
                
                # Apply logit adjustment if needed
                adj_outputs_final = outputs
                if IMBALANCE_STRATEGY == 'LogitAdjust' and CLASS_PRIORS is not None:
                    logit_adj = LOGIT_ADJUSTMENT_TAU * torch.log(CLASS_PRIORS + 1e-12)
                    adj_outputs_final = outputs + logit_adj.unsqueeze(0)
                
                # Calculate loss
                if use_mixup or use_cutmix:
                    loss = mixup_criterion(criterion, adj_outputs_final, targets_a, targets_b, lam)
                else:
                    loss = criterion(adj_outputs_final, targets_a)
                    
                loss_final = loss  # For reporting
                loss_scaled = loss / gradient_accumulation_steps
                
                # Backward pass
                scaler.scale(loss_scaled).backward()

        # --- Gradient Accumulation & Optimizer Step ---
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient Clipping (Applied before optimizer step)
            if grad_clip_val > 0 and not is_sam: # SAM handles clipping internally
                scaler.unscale_(optimizer) # Unscale gradients before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)

            # Optimizer Step (Standard optimizer)
            if not is_sam:
                scaler.step(optimizer)
                scaler.update()

            optimizer.zero_grad() # Zero gradients for the next accumulation cycle

            # --- EMA Update ---
            if USE_EMA and ema_model:
                ema_model.update() # Update EMA model after optimizer step

        # --- Loss and Accuracy Tracking (more efficient) ---
        if not torch.isnan(loss_final) and not torch.isinf(loss_final):
            current_step_loss = loss_final.item()
            running_loss += current_step_loss * batch_size
            total_samples += batch_size
            
            # Use adjusted outputs for accuracy calculation
            preds_for_acc = torch.argmax(adj_outputs_final, dim=1)
            all_preds.append(preds_for_acc.detach().cpu())
            all_labels.append(labels.detach().cpu())
            
            # Update progress bar less frequently to reduce overhead
            if batch_idx % 5 == 0:
                progress_bar.set_postfix(loss=f"{current_step_loss:.3f}", 
                                        lr=f"{optimizer.param_groups[0]['lr']:.1E}",
                                        batch=f"{batch_idx+1}/{total_batches}")
        else:
            # If loss is NaN/Inf and we're accumulating gradients, zero them
            if (batch_idx + 1) % gradient_accumulation_steps != 0:
                optimizer.zero_grad()
        
        # Update epoch progress bar
        epoch_progress.update(1)

    # Close the epoch progress bar
    epoch_progress.close()
    
    # --- End of Epoch ---
    if stop_requested:
        return None, None # Indicate interruption

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0

    # Calculate accuracy from collected predictions and labels
    epoch_acc = 0.0
    if all_preds and all_labels:
        all_preds_tensor = torch.cat(all_preds)
        all_labels_tensor = torch.cat(all_labels)
        epoch_acc = (all_preds_tensor == all_labels_tensor).sum().item() / total_samples if total_samples > 0 else 0

    # --- Learning Rate Scheduler Step ---
    # Step the scheduler after the epoch (except for ReduceLROnPlateau which needs metrics)
    if scheduler and not isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step()

    # --- Logging ---
    # Only log to writer if it's provided (not during HPO objective usually)
    if writer and fold_num != "HPO":
        writer.add_scalar('Loss/train', epoch_loss, global_epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, global_epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], global_epoch)

    return epoch_loss, epoch_acc


def validate_one_epoch(model, dataloader, criterion, device, global_epoch, writer, num_classes, scheduler=None, swa_model=None, ema_model=None, return_preds=False, fold_num=0): # Added scheduler=None
    """Validates the model for one epoch."""
    model.eval() # Set model to evaluation mode
    if swa_model: swa_model.eval()
    if ema_model: ema_model.eval()

    running_loss = 0.0; total_samples = 0
    all_labels_list, all_preds_list, all_original_indices_list = [], [], [] # For OOF

    models_to_eval = {'base': model}
    if swa_model: models_to_eval['swa'] = swa_model
    if ema_model: models_to_eval['ema'] = ema_model

    results = {} # Store results like {'base': (loss, acc), 'swa': (loss, acc), ...}
    oof_data = {'preds': [], 'indices': []} # Store OOF preds and indices from the *base* model run

    with torch.no_grad():
        for model_key, current_model in models_to_eval.items():
            current_model.eval() # Ensure eval mode for each model
            model_running_loss = 0.0; model_total_samples = 0
            model_all_preds, model_all_labels, model_all_indices = [], [], []

            # Determine if TTA should be used for this model type (e.g., only for base/swa/ema, not during HPO)
            apply_tta = USE_TTA and fold_num != "HPO" and TTA_TRANSFORMS is not None

            pbar_desc = f"Fold {fold_num} Validate GlobEp {global_epoch+1} ({model_key})"
            if fold_num == "HPO": pbar_desc = f"HPO Trial Validate Ep {global_epoch+1}"
            progress_bar = tqdm(dataloader, desc=pbar_desc, leave=False)

            for batch_data in progress_bar:
                # Expect (inputs, labels) or (inputs, labels, paths, original_indices) if include_paths=True
                original_indices_batch = None
                if len(batch_data) == 4 and return_preds: # Need original indices for OOF
                    inputs, labels, _, original_indices_batch = batch_data
                elif len(batch_data) == 2:
                    inputs, labels = batch_data
                else:
                    print(f"{TermColors.YELLOW}Warn: Skipping validation batch, unexpected data format (len={len(batch_data)}).{TermColors.ENDC}")
                    continue

                inputs, labels = inputs.to(device), labels.to(device); batch_size = inputs.size(0)

                # --- Standard Inference ---
                with torch.amp.autocast('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda')):
                    embeddings = current_model(inputs)
                    # Handle potential mismatch if model doesn't have metric_fc
                    if hasattr(current_model, 'metric_fc') and METRIC_LEARNING_TYPE == 'ArcFace':
                        outputs = current_model.metric_fc(embeddings, labels) # ArcFace needs labels even for eval if model defined that way
                    elif hasattr(current_model, 'metric_fc'): # Standard linear head
                         outputs = current_model.metric_fc(embeddings)
                    else: # Fallback
                        outputs = current_model(inputs)

                    # Apply logit adjustment if used (consistent with training)
                    adj_outputs = outputs
                    if IMBALANCE_STRATEGY == 'LogitAdjust' and CLASS_PRIORS is not None:
                        logit_adj = LOGIT_ADJUSTMENT_TAU * torch.log(CLASS_PRIORS + 1e-12)
                        adj_outputs = outputs + logit_adj.unsqueeze(0)

                    loss = criterion(adj_outputs, labels)

                # --- TTA Inference (Optional) ---
                tta_adj_outputs = None
                if apply_tta:
                    try:
                        # Apply TTA transform - assumes TTA_TRANSFORMS is an Albumentations Compose object
                        # Need to handle potential batch application or loop
                        inputs_tta_list = []
                        for i in range(inputs.size(0)):
                             img_np = inputs[i].cpu().permute(1, 2, 0).numpy() # HWC format
                             # Denormalize? No, TTA transform should include normalization
                             augmented = TTA_TRANSFORMS(image=img_np)
                             inputs_tta_list.append(augmented['image'])
                        inputs_tta = torch.stack(inputs_tta_list).to(device)

                        with torch.amp.autocast('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda')):
                            embed_tta = current_model(inputs_tta)
                            if hasattr(current_model, 'metric_fc') and METRIC_LEARNING_TYPE == 'ArcFace':
                                out_tta = current_model.metric_fc(embed_tta, labels) # Pass labels
                            elif hasattr(current_model, 'metric_fc'):
                                out_tta = current_model.metric_fc(embed_tta)
                            else:
                                out_tta = current_model(inputs_tta)

                            tta_adj_outputs = out_tta # Apply logit adjust to TTA outputs too? Yes.
                            if IMBALANCE_STRATEGY == 'LogitAdjust' and CLASS_PRIORS is not None:
                                tta_adj_outputs = out_tta + logit_adj.unsqueeze(0)

                    except Exception as tta_e:
                        print(f"{TermColors.YELLOW}Warn: TTA failed for batch: {tta_e}. Skipping TTA for this batch.{TermColors.ENDC}")
                        tta_adj_outputs = None # Ensure it's None if TTA fails

                # --- Combine Original and TTA Predictions ---
                final_outputs = adj_outputs
                if tta_adj_outputs is not None:
                    # Average logits before softmax
                    final_outputs = (adj_outputs + tta_adj_outputs) / 2.0

                # --- Loss and Accuracy ---
                if not torch.isnan(loss) and not torch.isinf(loss):
                    model_running_loss += loss.item() * batch_size; model_total_samples += batch_size
                    preds = torch.argmax(final_outputs, dim=1) # Use combined outputs for prediction
                    model_all_preds.append(preds.detach().cpu())
                    model_all_labels.append(labels.detach().cpu())
                    if return_preds and original_indices_batch is not None and model_key == 'base': # Store OOF only for base model run
                         oof_data['preds'].append(preds.detach().cpu())
                         oof_data['indices'].append(original_indices_batch.detach().cpu()) # Store original indices

                else:
                     print(f"{TermColors.RED}Warn: NaN/Inf validation loss detected. Skipping batch.{TermColors.ENDC}")

                progress_bar.set_postfix(loss=f"{loss.item():.4f}")
                if stop_requested: break
            if stop_requested: return None, None, None, None # Indicate interruption

            # --- Calculate metrics for this model type ---
            epoch_loss = model_running_loss / model_total_samples if model_total_samples > 0 else 0
            if model_all_preds and model_all_labels:
                preds_tensor = torch.cat(model_all_preds)
                labels_tensor = torch.cat(model_all_labels)
                epoch_acc = (preds_tensor == labels_tensor).sum().item() / model_total_samples if model_total_samples > 0 else 0
            else:
                epoch_acc = 0.0

            results[model_key] = (epoch_loss, epoch_acc)

            # --- Logging for this model type ---
            if writer and fold_num != "HPO":
                writer.add_scalar(f'Loss/val_{model_key}', epoch_loss, global_epoch)
                writer.add_scalar(f'Accuracy/val_{model_key}', epoch_acc, global_epoch)

    # --- Prepare return values ---
    base_loss, base_acc = results.get('base', (float('inf'), 0.0))

    # Concatenate OOF predictions and indices if collected
    oof_preds_concat = torch.cat(oof_data['preds']).numpy() if oof_data['preds'] else None
    oof_indices_concat = torch.cat(oof_data['indices']).numpy() if oof_data['indices'] else None

    # Step ReduceLROnPlateau scheduler if used
    if scheduler and isinstance(scheduler, ReduceLROnPlateau): # Check if scheduler exists and is the correct type
        metric_to_monitor = base_loss if PLATEAU_MONITOR == 'val_loss' else base_acc
        scheduler.step(metric_to_monitor)
        print(f"  ReduceLROnPlateau stepped with {PLATEAU_MONITOR}={metric_to_monitor:.4f}")

    # Return base model's loss/acc and the OOF data
    return base_loss, base_acc, oof_preds_concat, oof_indices_concat


# --- Stacking ---
def train_stacking_meta_model(oof_preds, oof_labels, save_path):
    """Trains a simple meta-model (Logistic Regression) on OOF predictions."""
    print(f"{TermColors.CYAN}ℹ Training Stacking Meta-Model...{TermColors.ENDC}")
    print(f"  Input OOF preds shape: {oof_preds.shape}, Labels shape: {oof_labels.shape}")

    if len(oof_preds.shape) > 2: # If preds are logits/probs per class
        # Option 1: Use probabilities directly
        # Option 2: Use predicted class index (argmax) - simpler
        oof_features = np.argmax(oof_preds, axis=1).reshape(-1, 1) # Use predicted class as feature
        print(f"  Using predicted class index as feature for meta-model.")
    else: # If preds are already class indices
        oof_features = oof_preds.reshape(-1, 1)

    # Simple Logistic Regression as meta-model
    meta_model = LogisticRegression(max_iter=1000, random_state=SEED, C=0.1) # Added C for regularization

    try:
        meta_model.fit(oof_features, oof_labels)
        # Evaluate meta-model on the same OOF data (for info only)
        meta_preds = meta_model.predict(oof_features)
        meta_acc = accuracy_score(oof_labels, meta_preds)
        print(f"{TermColors.GREEN}✅ Stacking meta-model trained. OOF Accuracy: {meta_acc:.4f}{TermColors.ENDC}")

        # Save the trained meta-model
        joblib.dump(meta_model, save_path)
        print(f"  Meta-model saved to: {save_path}")

    except Exception as e:
        print(f"{TermColors.RED}❌ Error training stacking meta-model: {e}{TermColors.ENDC}")
        traceback.print_exc()

# --- Auto Training Configuration ---
class AutoTrainingConfig:
    """Automatically adjusts training parameters based on performance to prevent underfitting and overfitting."""
    def __init__(self, initial_lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY):
        self.initial_lr = initial_lr
        self.weight_decay = weight_decay
        self.plateau_count = 0
        self.overfit_count = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.lr_history = []
        self.wd_history = []
        self.adjustment_log = []
        
    def update(self, train_loss, train_acc, val_loss, val_acc, optimizer):
        """Monitor metrics and adjust training parameters as needed."""
        # Store metrics for trend analysis
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        
        # Get current LR and WD
        current_opt = optimizer.base_optimizer if hasattr(optimizer, 'base_optimizer') else optimizer
        current_lr = current_opt.param_groups[0]['lr']
        current_wd = current_opt.param_groups[0]['weight_decay']
        self.lr_history.append(current_lr)
        self.wd_history.append(current_wd)
        
        # Check if validation metrics improved
        val_loss_improved = val_loss < self.best_val_loss
        val_acc_improved = val_acc > self.best_val_acc
        
        if val_loss_improved:
            self.best_val_loss = val_loss
            self.plateau_count = 0
            print(f"{TermColors.GREEN}✅ Val loss improved to {val_loss:.4f}{TermColors.ENDC}")
        else:
            self.plateau_count += 1
            print(f"{TermColors.YELLOW}⚠ Val loss plateau: {self.plateau_count} epochs without improvement{TermColors.ENDC}")
        
        if val_acc_improved:
            self.best_val_acc = val_acc
            print(f"{TermColors.GREEN}✅ Val accuracy improved to {val_acc:.4f}{TermColors.ENDC}")
        
        # Detect overfitting
        is_overfitting = False
        if len(self.train_losses) >= 3:
            # Signs of overfitting:
            # 1. Train loss decreases while val loss increases
            # 2. Gap between train and val loss is growing
            train_decreasing = self.train_losses[-1] < self.train_losses[-2]
            val_increasing = self.val_losses[-1] > self.val_losses[-2]
            gap_growing = (self.val_losses[-1] - self.train_losses[-1]) > (self.val_losses[-2] - self.train_losses[-2])
            
            if train_decreasing and (val_increasing or gap_growing):
                self.overfit_count += 1
                is_overfitting = True
                print(f"{TermColors.YELLOW}⚠ Potential overfitting detected (count: {self.overfit_count}){TermColors.ENDC}")
            else:
                self.overfit_count = max(0, self.overfit_count - 0.5)  # Gradually reduce counter if not overfitting
        
        # Apply adjustments based on detected issues
        adjustments_made = False
        
        # Handle plateau (underfitting)
        if self.plateau_count >= 3:
            # If we're plateauing, try increasing learning rate slightly or reducing regularization
            if current_lr < self.initial_lr * 1.5:  # Only increase up to 150% of initial LR
                new_lr = current_lr * 1.2
                new_wd = max(current_wd * 0.8, 1e-6)  # Reduce weight decay but keep it reasonable
                
                self._adjust_optimizer(optimizer, new_lr, new_wd)
                self.adjustment_log.append(f"Increased LR to {new_lr:.2e} due to plateau")
                adjustments_made = True
                
            self.plateau_count = 0  # Reset counter after adjustment
        
        # Handle overfitting
        elif self.overfit_count >= 2:
            # If we're overfitting, reduce learning rate and increase regularization
            new_lr = current_lr * 0.7
            new_wd = current_wd * 1.5
            
            self._adjust_optimizer(optimizer, new_lr, new_wd)
            self.adjustment_log.append(f"Reduced LR to {new_lr:.2e}, increased WD to {new_wd:.2e} due to overfitting")
            adjustments_made = True
            
            self.overfit_count = 0  # Reset counter after adjustment
        
        # For SAM specifically - adjust rho parameter based on detected patterns
        if hasattr(optimizer, 'rho') and isinstance(optimizer, SAM):
            self._adjust_sam_parameters(optimizer, is_overfitting)
        
        return adjustments_made
    
    def _adjust_optimizer(self, optimizer, new_lr, new_wd):
        """Apply parameter adjustments to the optimizer."""
        # Handle SAM optimizer which wraps another optimizer
        opt_to_adjust = optimizer.base_optimizer if hasattr(optimizer, 'base_optimizer') else optimizer
        
        for i, param_group in enumerate(opt_to_adjust.param_groups):
            old_lr = param_group['lr']
            old_wd = param_group['weight_decay']
            param_group['lr'] = new_lr
            param_group['weight_decay'] = new_wd
            
            # Log the changes
            print(f"{TermColors.CYAN}⚙ Group {i} - LR: {old_lr:.2e} → {new_lr:.2e}, WD: {old_wd:.2e} → {new_wd:.2e}{TermColors.ENDC}")
    
    def _adjust_sam_parameters(self, optimizer, is_overfitting):
        """Adjust SAM-specific parameters based on training behavior."""
        if not is_overfitting:
            return  # Only adjust when overfitting is detected
        
        current_rho = optimizer.rho
        # If overfitting, increase SAM's rho to find flatter minima
        if is_overfitting and current_rho < 0.1:  # Cap at reasonable value
            new_rho = min(current_rho * 1.2, 0.1)
            if new_rho != current_rho:
                optimizer.rho = new_rho
                print(f"{TermColors.CYAN}⚙ Adjusted SAM rho: {current_rho:.4f} → {new_rho:.4f}{TermColors.ENDC}")
                self.adjustment_log.append(f"Increased SAM rho to {new_rho:.4f}")
    
    def get_status_report(self):
        """Get a summary report of training progression and adjustments."""
        status = []
        status.append(f"Best val loss: {self.best_val_loss:.4f}, Best val acc: {self.best_val_acc:.4f}")
        status.append(f"Current learning rate: {self.lr_history[-1] if self.lr_history else 'N/A'}")
        status.append(f"Current weight decay: {self.wd_history[-1] if self.wd_history else 'N/A'}")
        
        if self.adjustment_log:
            status.append("Recent adjustments:")
            for adj in self.adjustment_log[-3:]:  # Show last 3 adjustments
                status.append(f"  - {adj}")
        
        return "\n".join(status)

# Initialize auto training config
auto_config = AutoTrainingConfig(initial_lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# --- Knowledge Distillation ---
class DistillationLoss(nn.Module):
    """ Combines KL Divergence loss (teacher vs student) and standard CrossEntropy loss (student vs labels). """
    def __init__(self, alpha=KD_ALPHA, temperature=KD_TEMPERATURE, base_criterion=nn.CrossEntropyLoss()):
        super().__init__()
        self.alpha = alpha
        self.T = temperature
        self.base_criterion = base_criterion
        self.KLDiv = nn.KLDivLoss(reduction='batchmean') # Use batchmean reduction

    def forward(self, student_outputs, teacher_outputs, labels):
        # Soften probabilities for KL divergence
        soft_teacher_log_probs = F.log_softmax(teacher_outputs / self.T, dim=1)
        soft_student_log_probs = F.log_softmax(student_outputs / self.T, dim=1)

        # Calculate KL divergence loss (how well student matches softened teacher)
        # Note: KLDivLoss expects log-probabilities for input, probabilities for target
        # We use log_softmax for both and it works, or use softmax for teacher. Let's stick to log_softmax for both.
        distillation_loss = self.KLDiv(soft_student_log_probs, soft_teacher_log_probs) * (self.alpha * self.T * self.T)
        # The (T*T) scaling factor is common practice in KD literature

        # Calculate standard cross-entropy loss (how well student predicts true labels)
        student_loss = self.base_criterion(student_outputs, labels)

        # Combine the losses
        total_loss = distillation_loss + (1. - self.alpha) * student_loss
        return total_loss


def train_student_model(teacher_model_path, student_model_name, student_save_path, df_train, df_val, image_dir, num_classes):
    """Trains a student model using knowledge distillation."""
    print(f"\n{TermColors.HEADER}--- Knowledge Distillation ---{TermColors.ENDC}")
    print(f"Teacher Path: {teacher_model_path}")
    print(f"Student Arch: {student_model_name}, Save Path: {student_save_path}")

    # --- Load Teacher Model ---
    print(f"{TermColors.CYAN}ℹ Loading teacher model...{TermColors.ENDC}")
    try:
        # Assume teacher uses the same CombinedModel structure (adjust if needed)
        # Need to know teacher's config (e.g., backbone names) - try loading from checkpoint if possible
        teacher_ckpt = torch.load(teacher_model_path, map_location=DEVICE)
        # Infer teacher config if possible (this is brittle)
        # TODO: Ideally, save config with checkpoint or load full model object
        teacher_model_names = MODEL_NAMES # Assume same as global default/HPO best - NEEDS REFINEMENT
        teacher_embedding_size = EMBEDDING_SIZE
        teacher_arcface_m = ARCFACE_M
        print(f"{TermColors.YELLOW}Warn: Assuming teacher config matches current global/HPO settings. This might be incorrect.{TermColors.ENDC}")

        teacher_model = build_model(
            model_names=teacher_model_names,
            num_classes=num_classes,
            embedding_size=teacher_embedding_size,
            arcface_m=teacher_arcface_m
        )
        # Load state dict, handling potential prefixes
        state_dict = teacher_ckpt['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '').replace('_orig_mod.', '') # Remove common prefixes
            new_state_dict[name] = v
        teacher_model.load_state_dict(new_state_dict, strict=False)
        teacher_model = teacher_model.to(DEVICE)
        teacher_model.eval() # Teacher is only used for inference
        print(f"{TermColors.GREEN}✅ Teacher model loaded and set to eval mode.{TermColors.ENDC}")
    except Exception as e:
        print(f"{TermColors.RED}❌ Failed to load teacher model: {e}. Skipping KD.{TermColors.ENDC}")
        traceback.print_exc()
        return

    # --- Build Student Model ---
    print(f"{TermColors.CYAN}ℹ Building student model: {student_model_name}...{TermColors.ENDC}")
    try:
        # Assume student also uses CombinedModel structure
        student_model = build_model(
            model_names=[student_model_name], # Student uses single backbone
            num_classes=num_classes,
            pretrained=True, # Usually start student from pretrained weights
            dropout_rate=KD_STUDENT_DROPOUT,
            embedding_size=KD_STUDENT_EMBEDDING_SIZE,
            # Student might not use ArcFace, or use different settings
            metric_learning='None' # Example: Student uses standard linear head
        )
        student_model = student_model.to(DEVICE)
    except Exception as e:
        print(f"{TermColors.RED}❌ Failed to build student model: {e}. Skipping KD.{TermColors.ENDC}")
        traceback.print_exc()
        return

    # --- KD Dataloaders ---
    print(f"{TermColors.CYAN}ℹ Creating KD dataloaders (Size: {KD_STUDENT_IMAGE_SIZE})...{TermColors.ENDC}")
    try:
        train_tf_kd, val_tf_kd = get_transforms(image_size=KD_STUDENT_IMAGE_SIZE)
        # Pass df_train, df_val directly (they have id, scientificName, label)
        train_ds_kd = PlantDataset(df_train, image_dir, train_tf_kd, None, False, KD_STUDENT_IMAGE_SIZE)
        val_ds_kd = PlantDataset(df_val, image_dir, val_tf_kd, None, False, KD_STUDENT_IMAGE_SIZE)
        train_ds_kd.fold = "KD"; val_ds_kd.fold = "KD" # Add identifier

        if len(train_ds_kd) == 0 or len(val_ds_kd) == 0:
             print(f"{TermColors.RED}❌ KD Train or Val dataset is empty. Skipping KD.{TermColors.ENDC}")
             return

        # Optional: Add weighted sampler for KD training too
        sampler_kd = None
        # if IMBALANCE_STRATEGY == 'WeightedSampler' and CLASS_WEIGHTS is not None: ... (add sampler logic if needed)

        train_loader_kd = DataLoader(train_ds_kd, KD_BATCH_SIZE, sampler=sampler_kd, shuffle=(sampler_kd is None), num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
        val_loader_kd = DataLoader(val_ds_kd, KD_BATCH_SIZE*2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        print(f"{TermColors.GREEN}✅ KD Dataloaders ready.{TermColors.ENDC}")

    except Exception as e:
        print(f"{TermColors.RED}❌ Error creating KD dataloaders: {e}. Skipping KD.{TermColors.ENDC}")
        traceback.print_exc()
        return

    # --- KD Training Setup ---
    criterion_kd = DistillationLoss(alpha=KD_ALPHA, temperature=KD_TEMPERATURE, base_criterion=nn.CrossEntropyLoss())
    optimizer_kd = optim.AdamW(student_model.parameters(), lr=KD_LR, weight_decay=WEIGHT_DECAY) # Simple AdamW for student
    # scaler_kd = GradScaler(enabled=MIXED_PRECISION) # Old way
    scaler_kd = torch.amp.GradScaler('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda')) # Updated way
    # Simple scheduler for KD
    scheduler_kd = CosineAnnealingLR(optimizer_kd, T_max=KD_EPOCHS, eta_min=KD_LR * 0.01)

    # --- KD Training Loop ---
    print(f"{TermColors.CYAN}ℹ Starting KD Training ({KD_EPOCHS} epochs)...{TermColors.ENDC}")
    best_kd_val_acc = 0.0
    kd_stop_requested = False

    for epoch in range(KD_EPOCHS):
        if kd_stop_requested: break
        print(f"\n--- KD Epoch {epoch+1}/{KD_EPOCHS} ---")
        student_model.train()
        running_loss_kd = 0.0; total_samples_kd = 0

        progress_bar_kd = tqdm(train_loader_kd, desc=f"KD Train Epoch {epoch+1}", leave=False)
        for batch_data in progress_bar_kd:
            if len(batch_data) != 2: # Simple check for (inputs, labels)
                print(f"{TermColors.YELLOW}Warn: Skipping KD batch, expected 2 items, got {len(batch_data)}.{TermColors.ENDC}")
                continue
            inputs, labels = batch_data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE); batch_size = inputs.size(0)

            # Get teacher outputs (no grad)
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs) # Assume teacher doesn't need labels for inference

            # Get student outputs and calculate loss
            with torch.amp.autocast('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda')):
                student_outputs = student_model(inputs) # Assume student doesn't need labels for inference
                loss = criterion_kd(student_outputs, teacher_outputs, labels)

            # Backward pass and optimizer step
            optimizer_kd.zero_grad()
            scaler_kd.scale(loss).backward()
            scaler_kd.step(optimizer_kd)
            scaler_kd.update()

            if not torch.isnan(loss) and not torch.isinf(loss):
                running_loss_kd += loss.item() * batch_size; total_samples_kd += batch_size
                progress_bar_kd.set_postfix(loss=f"{loss.item():.4f}")
            else:
                print(f"{TermColors.RED}Warn: NaN/Inf loss detected in KD training. Skipping batch.{TermColors.ENDC}")

            # Check for interrupt
            if signal.getsignal(signal.SIGINT) != handle_interrupt: # Check if handler is still active
                 print(f"{TermColors.CRITICAL}SIGINT handler detached! Exiting KD loop.{TermColors.ENDC}")
                 kd_stop_requested = True; break
            if stop_requested: kd_stop_requested = True; break


        epoch_loss_kd = running_loss_kd / total_samples_kd if total_samples_kd > 0 else 0
        print(f"KD Epoch {epoch+1} Train Loss: {epoch_loss_kd:.4f}")

        # --- KD Validation ---
        student_model.eval()
        running_val_loss_kd = 0.0; total_val_samples_kd = 0
        all_preds_kd, all_labels_kd = [], []
        with torch.no_grad():
            progress_bar_val_kd = tqdm(val_loader_kd, desc=f"KD Validate Epoch {epoch+1}", leave=False)
            for batch_data in progress_bar_val_kd:
                if len(batch_data) != 2: continue # Skip if not (inputs, labels)
                inputs, labels = batch_data
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE); batch_size = inputs.size(0)

                with torch.amp.autocast('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda')):
                    outputs = student_model(inputs)
                    # Use standard CE loss for validation metric (not KD loss)
                    val_loss = F.cross_entropy(outputs, labels)

                if not torch.isnan(val_loss) and not torch.isinf(val_loss):
                    running_val_loss_kd += val_loss.item() * batch_size; total_val_samples_kd += batch_size
                    preds = torch.argmax(outputs, dim=1)
                    all_preds_kd.append(preds.cpu())
                    all_labels_kd.append(labels.cpu())
                else:
                    print(f"{TermColors.RED}Warn: NaN/Inf loss detected in KD validation. Skipping batch.{TermColors.ENDC}")


        epoch_val_loss_kd = running_val_loss_kd / total_val_samples_kd if total_val_samples_kd > 0 else 0
        epoch_val_acc_kd = 0.0
        if all_preds_kd and all_labels_kd:
             preds_tensor = torch.cat(all_preds_kd)
             labels_tensor = torch.cat(all_labels_kd)
             epoch_val_acc_kd = (preds_tensor == labels_tensor).sum().item() / total_val_samples_kd if total_val_samples_kd > 0 else 0

        print(f"KD Epoch {epoch+1} Val Loss: {epoch_val_loss_kd:.4f}, Val Acc: {epoch_val_acc_kd:.4f}")

        # Save best student model based on validation accuracy
        if epoch_val_acc_kd > best_kd_val_acc:
            best_kd_val_acc = epoch_val_acc_kd
            print(f"{TermColors.OKGREEN}🏆 New best KD validation accuracy: {best_kd_val_acc:.4f}. Saving student model...{TermColors.ENDC}")
            try:
                # Save only the state_dict
                torch.save(student_model.state_dict(), student_save_path)
                print(f"  Student model state_dict saved to: {student_save_path}")
            except Exception as e:
                print(f"{TermColors.RED}❌ Error saving student model state_dict: {e}{TermColors.ENDC}")

        scheduler_kd.step() # Step scheduler

    print(f"{TermColors.OKGREEN}✅ Knowledge Distillation finished. Best student validation accuracy: {best_kd_val_acc:.4f}{TermColors.ENDC}")
    # Clean up KD resources
    del teacher_model, student_model, train_loader_kd, val_loader_kd, optimizer_kd, criterion_kd, scaler_kd, scheduler_kd
    if DEVICE.type == 'cuda': torch.cuda.empty_cache()
    gc.collect()

# --- Main Execution ---
def main():
    """Main function to run the training pipeline."""
    # Make globals modifiable where needed (updated by HPO or data loading)
    global stop_requested, label_encoder, class_names, NUM_CLASSES
    global CLASS_FREQUENCIES, CLASS_PRIORS, CLASS_WEIGHTS, CURRENT_IMAGE_SIZE
    global LEARNING_RATE, WEIGHT_DECAY, DROPOUT_RATE, ARCFACE_M, OPTIMIZER_TYPE, MODEL_NAMES, EMBEDDING_SIZE
    global RUN_STACKING, RUN_KNOWLEDGE_DISTILLATION # Allow modification if HPO fails

    set_seed(SEED); signal.signal(signal.SIGINT, handle_interrupt)
    print(f"{TermColors.HEADER}===== Plant Recognition Training (PyTorch - V2) ===={TermColors.ENDC}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Device: {DEVICE}")
    print(f"Debug Mode: {DEBUG_MODE}")
    # Initial config print will show defaults
    print(f"Initial Config: Folds={N_FOLDS}, Models={MODEL_NAMES}, Stages={len(PROGRESSIVE_RESIZING_STAGES)}, TotalEpPerFold={TOTAL_EPOCHS_PER_FOLD}, Batch={BATCH_SIZE}x{GRADIENT_ACCUMULATION_STEPS}, LR={LEARNING_RATE}, Optim={OPTIMIZER_TYPE}(SAM:{USE_SAM}), Loss={LOSS_TYPE}(LS:{LABEL_SMOOTHING}), Imbalance={IMBALANCE_STRATEGY}, Sched={SCHEDULER_TYPE}, Device={DEVICE}, MP={MIXED_PRECISION}, Compile={USE_TORCH_COMPILE}, Augs=Mixup/Cutmix/RandAug/Albu, SWA={USE_SWA}, EMA={USE_EMA}, TTA={USE_TTA}, Stacking={RUN_STACKING}, KD={RUN_KNOWLEDGE_DISTILLATION}")


    # --- Data Loading (Load full dataset info once) ---
    print(f"\n{TermColors.HEADER}--- STEP 1: Load Full Dataset Info ---{TermColors.ENDC}")
    df_full = None # Initialize df_full
    try:
        print(f"{TermColors.CYAN}ℹ Reading CSV: {CSV_PATH}{TermColors.ENDC}")
        df_full = pd.read_csv(
            CSV_PATH,
            sep=',',
            low_memory=False,
            on_bad_lines='skip'
        )
        print(f"{TermColors.GREEN}✅ CSV loaded initially. Shape: {df_full.shape}{TermColors.ENDC}")
        # print(f"Columns found: {df_full.columns.tolist()}") # Less verbose

        # --- Data Cleaning and Preprocessing ---
        print(f"{TermColors.CYAN}ℹ Cleaning and preprocessing data...{TermColors.ENDC}")

        # Check for necessary source columns
        required_source_cols = ['id', 'scientific_name'] # Need these from CSV
        if not all(col in df_full.columns for col in required_source_cols):
            if 'scientificName' in df_full.columns and 'scientific_name' not in df_full.columns:
                print(f"{TermColors.YELLOW}⚠️ Found 'scientificName' instead of 'scientific_name'. Renaming.{TermColors.ENDC}")
                df_full.rename(columns={'scientificName': 'scientific_name'}, inplace=True)
            if not all(col in df_full.columns for col in required_source_cols):
                missing = [col for col in required_source_cols if col not in df_full.columns]
                print(f"{TermColors.RED}❌ Error: Missing source columns needed: {missing}. Found: {df_full.columns.tolist()}{TermColors.ENDC}")
                sys.exit(1)

        # Keep only necessary columns initially and drop rows with missing values in these
        df_full = df_full[required_source_cols].dropna().astype({'id': str}) # Ensure ID is string
        print(f"Shape after keeping required cols & dropna: {df_full.shape}")

        # Rename 'scientific_name' -> 'scientificName' for consistency within this script/Dataset class
        df_full.rename(columns={'scientific_name': 'scientificName'}, inplace=True)

        # --- Filter classes by min_samples ---
        min_samples = 1
        print(f"Filtering classes with less than {min_samples} samples...")
        class_counts = df_full['scientificName'].value_counts()
        valid_classes = class_counts[class_counts >= min_samples].index
        df_full = df_full[df_full['scientificName'].isin(valid_classes)].reset_index(drop=True)
        print(f"Shape after filtering by min_samples: {df_full.shape}, Unique classes: {len(valid_classes)}")
        if len(df_full) == 0:
            print(f"{TermColors.RED}❌ Error: Dataframe is empty after filtering by min_samples ({min_samples}). Check data or filter criteria.{TermColors.ENDC}")
            sys.exit(1)

        # --- Encode Labels ---
        print("Encoding labels...")
        label_encoder = LabelEncoder(); df_full['label'] = label_encoder.fit_transform(df_full['scientificName'])
        class_names = list(label_encoder.classes_); NUM_CLASSES = len(class_names) # Update global NUM_CLASSES
        print(f"Classes after encoding: {NUM_CLASSES}")
        if NUM_CLASSES == 0:
            print(f"{TermColors.RED}❌ Error: Zero classes found after encoding. Check data filtering.{TermColors.ENDC}")
            sys.exit(1)

        # Save label mapping
        mapping_path = os.path.join(BASE_MODEL_SAVE_DIR, "label_mapping.json");
        os.makedirs(BASE_MODEL_SAVE_DIR, exist_ok=True)
        with open(mapping_path, 'w') as f: json.dump(dict(zip(range(NUM_CLASSES), class_names)), f, indent=4)
        print(f"Label mapping saved to {mapping_path}")

        # Apply DEBUG mode subset *after* label encoding if needed
        if DEBUG_MODE:
            print(f"{TermColors.YELLOW}DEBUG MODE: Sampling subset...{TermColors.ENDC}")
            # Ensure stratification if sampling
            _, df_full = train_test_split(df_full, test_size=min(2000, len(df_full)), random_state=SEED, stratify=df_full['label'])
            df_full = df_full.reset_index(drop=True)
            print(f"Shape after DEBUG sampling: {df_full.shape}")


        # --- Calculate imbalance stats (using the potentially smaller df_full) ---
        print("Calculating imbalance statistics...")
        label_counts = df_full['label'].value_counts().sort_index(); total_samples_final = len(df_full)
        if total_samples_final == 0:
             print(f"{TermColors.RED}❌ Error: Dataframe is empty before calculating imbalance stats (possibly due to DEBUG mode?).{TermColors.ENDC}")
             sys.exit(1)

        freqs = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        # Use reindex to ensure all classes are present, fill missing with 0
        label_counts_reindexed = label_counts.reindex(range(NUM_CLASSES), fill_value=0)
        for i in range(NUM_CLASSES): freqs[i] = label_counts_reindexed.get(i, 0)

        CLASS_FREQUENCIES = freqs.to(DEVICE)
        # Avoid division by zero if total_samples_final is somehow zero
        CLASS_PRIORS = (CLASS_FREQUENCIES / total_samples_final).to(DEVICE) if total_samples_final > 0 else torch.zeros_like(CLASS_FREQUENCIES)

        try:
            # Ensure labels passed to compute_class_weight cover all expected classes if possible
            unique_labels_in_data = np.unique(df_full['label'])
            if len(unique_labels_in_data) < NUM_CLASSES:
                 print(f"{TermColors.YELLOW}Warn: Data subset contains only {len(unique_labels_in_data)}/{NUM_CLASSES} classes. Class weights might be affected.{TermColors.ENDC}")

            class_weights_array = sk_class_weight.compute_class_weight(
                'balanced',
                classes=np.arange(NUM_CLASSES), # Provide all possible class indices
                y=df_full['label']
            )
            CLASS_WEIGHTS = torch.tensor(class_weights_array, dtype=torch.float32)
            print(f"Class priors and weights calculated for imbalance handling.")
            if IMBALANCE_STRATEGY == 'LogitAdjust': print(f"  Using Logit Adjustment (tau={LOGIT_ADJUSTMENT_TAU})")
            elif IMBALANCE_STRATEGY == 'WeightedLoss': print(f"  Using Weighted Loss")
            elif IMBALANCE_STRATEGY == 'WeightedSampler': print(f"  Using Weighted Sampler")
            else: print(f"  No specific imbalance strategy selected.")
        except ValueError as e:
            print(f"{TermColors.RED}❌ Error calculating class weights: {e}.{TermColors.ENDC}")
            if IMBALANCE_STRATEGY in ['WeightedLoss', 'WeightedSampler']:
                 print(f"{TermColors.RED}Exiting due to class weight calculation error with strategy '{IMBALANCE_STRATEGY}'.{TermColors.ENDC}")
                 sys.exit(1)
            else:
                 print(f"{TermColors.YELLOW}Proceeding without class weights due to calculation error.{TermColors.ENDC}")
                 CLASS_WEIGHTS = None # Ensure it's None if calculation fails


    except FileNotFoundError:
        print(f"{TermColors.RED}❌ Data Error: CSV file not found at {CSV_PATH}{TermColors.ENDC}")
        traceback.print_exc(); sys.exit(1)
    except pd.errors.EmptyDataError:
         print(f"{TermColors.RED}❌ Data Error: CSV file is empty at {CSV_PATH}{TermColors.ENDC}")
         traceback.print_exc(); sys.exit(1)
    except Exception as e:
        print(f"{TermColors.RED}❌ Data Loading/Preprocessing Error: {e}{TermColors.ENDC}")
        traceback.print_exc(); sys.exit(1)

    # --- Cross-Validation Loop ---
    print(f"\n{TermColors.HEADER}--- STEP 3: Main K-Fold Cross-Validation ({N_FOLDS} Folds) ---{TermColors.ENDC}")
    # Use df_full (potentially reduced by DEBUG mode) for splitting
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_results = defaultdict(list) # Stores best metrics per fold

    # Initialize OOF lists based on the length of df_full *before* augmentation discovery
    # Store predictions and true labels corresponding to the original df_full indices
    oof_preds_array = np.full((len(df_full), NUM_CLASSES), np.nan, dtype=np.float32) # Store logits/probs
    oof_labels_array = np.full(len(df_full), -1, dtype=np.int32) # Store true labels

    # Pass df_full (with id, scientificName, label) to the split
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_full, df_full['label'])):
        if stop_requested: break
        print(f"\n{TermColors.HEADER}===== Starting Fold {fold+1}/{N_FOLDS} ====={TermColors.ENDC}")
        # train_df/val_df now contain id, scientificName, label for the fold split
        train_df = df_full.iloc[train_idx]; val_df = df_full.iloc[val_idx]
        print(f"Fold {fold+1} - Train DF: {len(train_df)}, Val DF: {len(val_df)}")

        # --- Build Model, Loss, Optimizer, Scaler for each fold ---
        print(f"\n{TermColors.BOLD}--- Fold {fold+1}: Setup Training Components ---{TermColors.ENDC}")
        model = optimizer = scheduler = scaler = criterion = swa_model = ema_model = None # Ensure cleanup from previous fold
        gc.collect(); torch.cuda.empty_cache()
        try:
            # Build model using potentially updated MODEL_NAMES, DROPOUT_RATE, ARCFACE_M etc.
            model = build_model(model_names=MODEL_NAMES, num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE, arcface_m=ARCFACE_M)
            model = model.to(DEVICE)
            compiled_model_applied = False # Reset compile flag per fold
            if USE_TORCH_COMPILE and hasattr(torch, 'compile'): # Attempt compile per fold
                try:
                    # Check PyTorch version for compile API
                    pt_version = torch.__version__.split('.')
                    if int(pt_version[0]) >= 2:
                        print(f"{TermColors.YELLOW}⏳ Fold {fold+1}: Applying torch.compile()...{TermColors.ENDC}");
                        model = torch.compile(model, mode='default') # or 'reduce-overhead' or 'max-autotune'
                        compiled_model_applied = True
                        print(f"{TermColors.GREEN}✅ torch.compile() applied.{TermColors.ENDC}")
                    else: print(f"{TermColors.YELLOW}Warn: torch.compile requires PyTorch 2.0+. Skipping.{TermColors.ENDC}")
                except Exception as compile_e: print(f"{TermColors.RED}❌ torch.compile() failed: {compile_e}. Continuing without compile.{TermColors.ENDC}")

            criterion = get_criterion(class_weights=CLASS_WEIGHTS) # Get loss function (potentially with weights)
            optimizer = get_optimizer(model, optimizer_type=OPTIMIZER_TYPE, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, use_sam=USE_SAM)
            scaler = torch.amp.GradScaler('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda'))
            scheduler = None # Scheduler is created per stage
            # SWA Model (reinitialized per fold)
            swa_model = AveragedModel(model, avg_fn=ema_avg_fn if USE_EMA else None) if USE_SWA else None # Use EMA decay for SWA if EMA is also on
            # EMA Model (reinitialized per fold)
            ema_model = AveragedModel(model, avg_fn=ema_avg_fn) if USE_EMA else None

        except Exception as e: print(f"{TermColors.RED}❌ Fold {fold+1} Setup Error: {e}{TermColors.ENDC}"); traceback.print_exc(); continue # Skip fold on error

        # --- Checkpoint Loading for the fold ---
        print(f"\n{TermColors.BOLD}--- Fold {fold+1}: Load Checkpoint ---{TermColors.ENDC}")
        # Pass the potentially updated optimizer to load_checkpoint
        start_glob_ep, start_stg_idx, start_stg_ep, best_metric, _, loaded_size, _ = load_checkpoint(
            fold, model, optimizer, None, scaler, filename="latest_checkpoint.pth.tar" # Try loading latest first
        )
        # If best_metric is still initial value after loading 'latest', try loading 'best'
        initial_best = float('-inf') if CHECKPOINT_MONITOR == 'max' else float('inf')
        if best_metric == initial_best:
             print(f"{TermColors.CYAN}ℹ No valid 'latest' checkpoint found or best metric invalid. Trying 'best_model.pth.tar'...{TermColors.ENDC}")
             start_glob_ep, start_stg_idx, start_stg_ep, best_metric, _, loaded_size, _ = load_checkpoint(
                 fold, model, optimizer, None, scaler, filename="best_model.pth.tar"
             )


        # --- TensorBoard Writer for the fold ---
        fold_log_dir = os.path.join(BASE_LOG_DIR, f"fold_{fold}")
        writer = SummaryWriter(log_dir=fold_log_dir)

        # --- Fold Training Loop (Progressive Resizing) ---
        print(f"\n{TermColors.BOLD}--- Fold {fold+1}: Start Training ---{TermColors.ENDC}")
        global_epoch_counter = start_glob_ep
        fold_stop_requested = False # Use a fold-specific flag
        fold_best_val_loss = float('inf')
        fold_best_val_acc = 0.0
        # Initialize stage variables and scheduler before the loop in case of early interruption
        stage_idx = start_stg_idx
        stage_epoch = start_stg_ep
        scheduler = None

        for stage_idx, (stage_epochs, stage_image_size) in enumerate(PROGRESSIVE_RESIZING_STAGES):
            if fold_stop_requested or stop_requested: break # Check global and fold flags

            # --- Stage Setup ---
            # Check if resuming mid-training and skip completed stages/epochs
            if stage_idx < start_stg_idx:
                print(f"{TermColors.YELLOW}⏩ Fold {fold+1} Skipping Stage {stage_idx+1} (Resuming from Stage {start_stg_idx+1}).{TermColors.ENDC}")
                continue
            current_stage_start_epoch = start_stg_ep if stage_idx == start_stg_idx else 0
            if current_stage_start_epoch >= stage_epochs:
                 print(f"{TermColors.YELLOW}⏩ Fold {fold+1} Skipping Stage {stage_idx+1} (Already completed).{TermColors.ENDC}")
                 continue # Skip stage if already completed

            print(f"\n{TermColors.MAGENTA}===== Fold {fold+1} Stage {stage_idx+1}/{len(PROGRESSIVE_RESIZING_STAGES)}: {stage_epochs} Epochs @ {stage_image_size} ====={TermColors.ENDC}")
            CURRENT_IMAGE_SIZE = stage_image_size # Update global current size

            # --- Check loaded checkpoint image size ---
            if loaded_size and stage_idx == start_stg_idx and loaded_size != CURRENT_IMAGE_SIZE:
                 print(f"{TermColors.CRITICAL}🚨 Image size mismatch! Checkpoint size {loaded_size} != Stage size {CURRENT_IMAGE_SIZE}. Exiting.{TermColors.ENDC}")
                 fold_stop_requested = True; break # Stop fold

            # --- Dataloaders for Stage ---
            train_transform, val_transform = get_transforms(image_size=CURRENT_IMAGE_SIZE)
            print(f"{TermColors.CYAN}ℹ Fold {fold+1} Creating dataloaders size {CURRENT_IMAGE_SIZE}...{TermColors.ENDC}")
            train_loader, val_loader, err_loader = None, None, None # Ensure cleanup
            gc.collect()
            try:
                # Pass the fold-specific dataframes (train_df, val_df) to PlantDataset
                train_ds = PlantDataset(train_df, IMAGE_DIR, train_transform, None, False, CURRENT_IMAGE_SIZE)
                # For validation and error analysis, include paths and original indices
                val_ds = PlantDataset(val_df, IMAGE_DIR, val_transform, None, True, CURRENT_IMAGE_SIZE)
                err_ds = PlantDataset(val_df, IMAGE_DIR, val_transform, None, True, CURRENT_IMAGE_SIZE) # Use val_df for errors

                if len(train_ds) == 0 or len(val_ds) == 0:
                     print(f"{TermColors.RED}❌ Fold {fold+1} Stage {stage_idx+1}: Train ({len(train_ds)}) or Val ({len(val_ds)}) dataset is empty. Skipping fold.{TermColors.ENDC}")
                     fold_stop_requested = True; break # Stop fold

                train_ds.fold = fold+1; val_ds.fold = fold+1; err_ds.fold = fold+1

                sampler = None
                if IMBALANCE_STRATEGY == 'WeightedSampler' and CLASS_WEIGHTS is not None:
                    print(f"  Using WeightedRandomSampler for training data.")
                    labels_list = train_ds.get_labels()
                    if labels_list:
                        class_sample_count = np.array([labels_list.count(l) for l in range(NUM_CLASSES)])
                        class_sample_count = np.maximum(class_sample_count, 1)
                        weight = 1. / class_sample_count
                        samples_weight = np.array([weight[t] for t in labels_list])
                        samples_weight = torch.from_numpy(samples_weight).double()
                        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
                    else: print(f"{TermColors.YELLOW}Warn: Cannot apply WeightedSampler as no labels found in train_ds.{TermColors.ENDC}")

                train_loader = DataLoader(train_ds, BATCH_SIZE, sampler=sampler, shuffle=(sampler is None), num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
                val_loader = DataLoader(val_ds, BATCH_SIZE*2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
                err_loader = DataLoader(err_ds, ERROR_LOG_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
                print(f"{TermColors.GREEN}✅ Fold {fold+1} Dataloaders ready (Train batches: {len(train_loader)}, Val batches: {len(val_loader)}).{TermColors.ENDC}")

            except Exception as e: print(f"{TermColors.RED}❌ Fold {fold+1} Dataloader Error: {e}{TermColors.ENDC}"); traceback.print_exc(); fold_stop_requested = True; break # Stop fold

            # --- Setup Scheduler for Stage ---
            # Calculate total steps for this stage for warmup if needed
            stage_total_steps = len(train_loader) * stage_epochs
            # Use stage_epochs for T_max in CosineAnnealingLR if resetting per stage
            current_t_0 = stage_epochs if SCHEDULER_TYPE == 'CosineWarmRestarts' else T_0 # Adjust T_0 per stage?
            scheduler = get_scheduler(optimizer, total_epochs=stage_epochs, t_0=current_t_0) # Pass stage epochs

            # --- Load Scheduler State if resuming mid-stage ---
            if stage_idx == start_stg_idx and current_stage_start_epoch > 0:
                 ckpt_path_sched = os.path.join(BASE_CHECKPOINT_DIR, f"fold_{fold}", "latest_checkpoint.pth.tar")
                 if scheduler and os.path.isfile(ckpt_path_sched):
                     print(f"{TermColors.CYAN}ℹ Fold {fold+1} Reloading scheduler state from latest ckpt...{TermColors.ENDC}")
                     ckpt_sched = torch.load(ckpt_path_sched, map_location=DEVICE)
                     if 'scheduler' in ckpt_sched and ckpt_sched['scheduler']:
                         try: scheduler.load_state_dict(ckpt_sched['scheduler'])
                         except Exception as e: print(f"{TermColors.YELLOW}⚠️ Fold {fold+1} Scheduler reload failed: {e}.{TermColors.ENDC}")

            # --- SWA Scheduler ---
            swa_scheduler = None
            swa_start_epoch_stage = max(0, int(stage_epochs * SWA_START_EPOCH_GLOBAL_FACTOR)) # SWA start relative to stage
            if USE_SWA and swa_model:
                 print(f"  SWA will start at stage epoch {swa_start_epoch_stage+1}")
                 swa_lr = LEARNING_RATE * SWA_LR_FACTOR
                 swa_scheduler = SWALR(optimizer, swa_lr=swa_lr, anneal_epochs=SWA_ANNEAL_EPOCHS, anneal_strategy='cos')


            # --- Epoch Loop for Stage ---
            for stage_epoch in range(current_stage_start_epoch, stage_epochs):
                if fold_stop_requested or stop_requested: break
                print(f"\n{TermColors.CYAN}--- Fold {fold+1} GlobEp {global_epoch_counter+1}/{TOTAL_EPOCHS_PER_FOLD} (Stg {stage_idx+1}: Ep {stage_epoch+1}/{stage_epochs}) ---{TermColors.ENDC}")

                # --- Training ---
                train_loss, train_acc = train_one_epoch(
                    model, train_loader, criterion, optimizer, scaler, scheduler if not (USE_SWA and stage_epoch >= swa_start_epoch_stage) else None, # Don't step main scheduler during SWA anneal
                    global_epoch_counter, stage_idx, stage_epoch, stage_epochs, DEVICE, writer, NUM_CLASSES, ema_model, use_sam=USE_SAM, fold_num=fold+1
                )
                if train_loss is None: fold_stop_requested = True; break # Handle interruption
            
                # --- Validation ---
                # Pass val_loader which yields original indices
                val_loss, val_acc, current_oof_preds, current_oof_indices = validate_one_epoch(
                    model, val_loader, criterion, DEVICE, global_epoch_counter, writer, NUM_CLASSES,
                    scheduler=scheduler, # Pass the scheduler here
                    swa_model=swa_model, ema_model=ema_model, return_preds=True, fold_num=fold+1
                )
                if val_loss is None: fold_stop_requested = True; break # Stop fold if validation interrupted

                auto_config.update(train_loss, train_acc, val_loss, val_acc, optimizer)

                # Print status report every few epochs
                if global_epoch_counter % 5 == 0:
                    print(f"\n{TermColors.CYAN}=== Training Status Report ==={TermColors.ENDC}")
                    print(auto_config.get_status_report())
                    print(f"{TermColors.CYAN}==========================={TermColors.ENDC}\n")

                print(f"Fold {fold+1} GlobEp {global_epoch_counter+1}: Train L={train_loss:.4f} A={train_acc:.4f} | Val L={val_loss:.4f} A={val_acc:.4f}")

                # --- SWA Update ---
                if USE_SWA and swa_model and stage_epoch >= swa_start_epoch_stage:
                    swa_model.update()
                    swa_scheduler.step()
                    print(f"  SWA Model updated. SWA LR: {swa_scheduler.get_last_lr()[0]:.1E}")
                elif scheduler and not isinstance(scheduler, ReduceLROnPlateau): # Step other schedulers (like Cosine) here
                    scheduler.step()
                    print(f"  Scheduler stepped. LR: {optimizer.param_groups[0]['lr']:.1E}")


                # --- Checkpointing and Best OOF Preds Saving ---
                current_metric = val_acc if CHECKPOINT_MONITOR == 'val_acc' else val_loss
                is_best = False
                if (CHECKPOINT_MODE == 'max' and current_metric > best_metric) or \
                   (CHECKPOINT_MODE == 'min' and current_metric < best_metric):
                    best_metric = current_metric; is_best = True
                    fold_best_val_loss = val_loss # Store best metrics for fold summary
                    fold_best_val_acc = val_acc

                    print(f"{TermColors.OKGREEN}🏆 Fold {fold+1} New Best {CHECKPOINT_MONITOR}: {best_metric:.4f}. Saving best models & OOF preds.{TermColors.ENDC}")
                    save_checkpoint(fold, global_epoch_counter + 1, stage_idx, stage_epoch + 1, model, optimizer, scheduler, scaler, best_metric, "best_model.pth.tar")
                    save_model(fold, model, "best_model_state_dict.pth")
                    if USE_EMA and ema_model: save_model(fold, ema_model, "best_ema_model_state_dict.pth")
                    if USE_SWA and swa_model and stage_epoch >= swa_start_epoch_stage: save_model(fold, swa_model, "best_swa_model_state_dict.pth") # Save best SWA if active

                    # Store OOF predictions from the best epoch for this fold
                    if current_oof_preds is not None and current_oof_indices is not None:
                        print(f"  Storing OOF predictions for {len(current_oof_indices)} samples...")
                        for i, original_idx in enumerate(current_oof_indices):
                            if 0 <= original_idx < len(oof_preds_array):
                                oof_preds_array[original_idx] = current_oof_preds[i] # Store logits/probs
                                oof_labels_array[original_idx] = df_full.loc[original_idx, 'label'] # Store true label
                            else:
                                print(f"{TermColors.YELLOW}Warn: OOF Original Index {original_idx} out of bounds (max: {len(oof_preds_array)-1}). Skipping.{TermColors.ENDC}")
                    else:
                         print(f"{TermColors.YELLOW}Warn: No OOF predictions returned from validation to save for best epoch.{TermColors.ENDC}")


                # Save latest checkpoint periodically or always
                save_checkpoint(fold, global_epoch_counter + 1, stage_idx, stage_epoch + 1, model, optimizer, scheduler, scaler, best_metric, "latest_checkpoint.pth.tar")

                # --- Log Misclassified ---
                if LOG_MISCLASSIFIED_IMAGES and ((global_epoch_counter + 1) % 5 == 0 or is_best): # Log every 5 epochs or on best
                    log_misclassified(fold, model, err_loader, criterion, DEVICE, global_epoch_counter + 1, writer, NUM_CLASSES)

                global_epoch_counter += 1
                if DEVICE.type == 'cuda': torch.cuda.empty_cache(); gc.collect()
                start_stg_ep = 0 # Ensure we don't skip epochs on next stage start after resuming

            # --- End Stage ---
            if fold_stop_requested or stop_requested: break
            # Clean up stage-specific resources
            # Note: scheduler is cleaned up here, but we need its value if interrupted
            del train_loader, val_loader, err_loader, train_ds, val_ds, err_ds, train_transform, val_transform, scheduler, swa_scheduler
            scheduler = None # Explicitly set to None after deletion for clarity
            if DEVICE.type == 'cuda': torch.cuda.empty_cache(); gc.collect()

        # --- End Fold ---
        if fold_stop_requested or stop_requested:
             print(f"{TermColors.WARNING}Fold {fold+1} interrupted.{TermColors.ENDC}")
             # Save final state even if interrupted mid-fold
             # scheduler will have the value from the last completed stage setup or None if interrupted before first stage
             save_checkpoint(fold, global_epoch_counter, stage_idx, stage_epoch, model, optimizer, scheduler, scaler, best_metric, "interrupted_checkpoint.pth.tar")
             break

        # --- SWA Final Evaluation ---
        if USE_SWA and swa_model and global_epoch_counter >= int(TOTAL_EPOCHS_PER_FOLD * SWA_START_EPOCH_GLOBAL_FACTOR):
            print(f"{TermColors.CYAN}ℹ Fold {fold+1} Updating SWA BN stats...{TermColors.ENDC}")
            # Need to recreate dataloader with final stage size and train transforms for BN update
            final_stage_size = PROGRESSIVE_RESIZING_STAGES[-1][1]
            final_train_tf, _ = get_transforms(image_size=final_stage_size)
            try:
                # Use train_df for BN update loader
                final_train_ds_bn = PlantDataset(train_df, IMAGE_DIR, final_train_tf, None, False, final_stage_size)
                if len(final_train_ds_bn) > 0:
                    bn_loader = DataLoader(final_train_ds_bn, BATCH_SIZE*2, shuffle=True, num_workers=NUM_WORKERS)
                    torch.optim.swa_utils.update_bn(bn_loader, swa_model, device=DEVICE)
                    print(f"{TermColors.GREEN}✅ Fold {fold+1} SWA BN updated.{TermColors.ENDC}")

                    # Evaluate final SWA model
                    print(f"{TermColors.CYAN}ℹ Fold {fold+1} Evaluating final SWA model...{TermColors.ENDC}")
                    _, final_val_tf = get_transforms(image_size=final_stage_size)
                    final_val_ds_eval = PlantDataset(val_df, IMAGE_DIR, final_val_tf, None, False, final_stage_size) # No paths needed
                    if len(final_val_ds_eval) > 0:
                        final_val_loader = DataLoader(final_val_ds_eval, BATCH_SIZE*2, shuffle=False, num_workers=NUM_WORKERS)
                        # Use validate_one_epoch to evaluate SWA model (pass swa_model as the model)
                        swa_val_loss, swa_val_acc, _, _ = validate_one_epoch(
                            swa_model, final_val_loader, criterion, DEVICE, global_epoch_counter, writer, NUM_CLASSES, fold_num=f"{fold+1}-SWA"
                        )
                        print(f"Fold {fold+1} Final SWA Val Loss: {swa_val_loss:.4f}, Val Acc: {swa_val_acc:.4f}")
                        fold_results['swa_acc'].append(swa_val_acc)
                        fold_results['swa_loss'].append(swa_val_loss)
                        save_model(fold, swa_model, "final_swa_model_state_dict.pth") # Save final SWA model
                    else: print(f"{TermColors.YELLOW}Warn: SWA validation dataset empty. Skipping final SWA eval.{TermColors.ENDC}")
                    del final_val_ds_eval, final_val_loader
                else: print(f"{TermColors.YELLOW}Warn: SWA BN update dataset empty. Skipping BN update and SWA eval.{TermColors.ENDC}")
                del final_train_ds_bn, bn_loader
            except Exception as e: print(f"{TermColors.RED}❌ Fold {fold+1} SWA BN/Eval Error: {e}{TermColors.ENDC}"); traceback.print_exc()


        # Save final standard and EMA models
        save_model(fold, model, "final_model_state_dict.pth")
        if USE_EMA and ema_model: save_model(fold, ema_model, "final_ema_model_state_dict.pth")

        writer.close() # Close writer for this fold

        # Store fold results
        fold_results['best_metric'].append(best_metric) # Store the best monitored metric value
        fold_results['best_val_loss'].append(fold_best_val_loss)
        fold_results['best_val_acc'].append(fold_best_val_acc)


        # --- Explicit Memory Cleanup for Fold ---
        print(f"{TermColors.DIM}Cleaning up resources for fold {fold+1}...{TermColors.ENDC}")
        del model, optimizer, scaler, criterion, swa_model, ema_model, train_df, val_df, writer
        # Dataloaders/datasets cleaned up per stage
        if DEVICE.type == 'cuda': torch.cuda.empty_cache()
        gc.collect()
        print(f"{TermColors.DIM}Fold {fold+1} cleanup complete.{TermColors.ENDC}")


    # --- End of Cross-Validation Loop ---
    print(f"\n{TermColors.HEADER}===== Cross-Validation Finished ====={TermColors.ENDC}")
    if not stop_requested:
        # Calculate and print average results across folds
        avg_best_metric = np.mean([m for m in fold_results['best_metric'] if m is not None and not np.isinf(m)]) if fold_results['best_metric'] else 0
        std_best_metric = np.std([m for m in fold_results['best_metric'] if m is not None and not np.isinf(m)]) if fold_results['best_metric'] else 0
        avg_best_acc = np.mean(fold_results['best_val_acc']) if fold_results['best_val_acc'] else 0
        std_best_acc = np.std(fold_results['best_val_acc']) if fold_results['best_val_acc'] else 0
        print(f"Average Best Val Acc across {len(fold_results['best_val_acc'])} folds: {avg_best_acc:.4f} +/- {std_best_acc:.4f}")
        print(f"Average Best {CHECKPOINT_MONITOR} across {len(fold_results['best_metric'])} folds: {avg_best_metric:.4f} +/- {std_best_metric:.4f}")
        if fold_results['swa_acc']:
             avg_swa_acc = np.mean(fold_results['swa_acc'])
             std_swa_acc = np.std(fold_results['swa_acc'])
             print(f"Average Final SWA Val Acc across {len(fold_results['swa_acc'])} folds: {avg_swa_acc:.4f} +/- {std_swa_acc:.4f}")

        # --- Prepare data for Stacking ---
        if RUN_STACKING:
            print(f"\n{TermColors.CYAN}ℹ Preparing data for Stacking...{TermColors.ENDC}")
            # Check which indices have valid predictions (not NaN or initial -1)
            valid_oof_indices = np.where(oof_labels_array != -1)[0]

            if len(valid_oof_indices) < len(df_full):
                 print(f"{TermColors.YELLOW}Warning: Only collected OOF predictions for {len(valid_oof_indices)} out of {len(df_full)} original samples. Stacking might be suboptimal.{TermColors.ENDC}")
            elif len(valid_oof_indices) == 0:
                 print(f"{TermColors.RED}Error: No valid OOF predictions collected. Skipping Stacking.{TermColors.ENDC}")
                 RUN_STACKING = False # Disable stacking if no data

            if RUN_STACKING:
                final_oof_preds = oof_preds_array[valid_oof_indices]
                final_oof_labels = oof_labels_array[valid_oof_indices]

                if len(final_oof_preds) > 0 and len(final_oof_preds) == len(final_oof_labels):
                    # Save OOF predictions (logits/probs) and labels
                    np.savez_compressed(STACKING_OOF_PREDS_PATH, preds=final_oof_preds, labels=final_oof_labels)
                    print(f"OOF predictions saved to {STACKING_OOF_PREDS_PATH}")
                    # Train Stacking Meta-Model
                    train_stacking_meta_model(final_oof_preds, final_oof_labels, STACKING_META_MODEL_PATH)
                else:
                     print(f"{TermColors.RED}Error preparing stacking data after filtering. Length mismatch or no data. Skipping stacking.{TermColors.ENDC}")
                del final_oof_preds, final_oof_labels # Cleanup stacking data
                gc.collect()

    else:
        print(f"{TermColors.YELLOW}Training interrupted. Results may be incomplete. Skipping Stacking and KD.{TermColors.ENDC}")
        RUN_STACKING = False
        RUN_KNOWLEDGE_DISTILLATION = False

    # Clean up OOF arrays
    del oof_preds_array, oof_labels_array
    gc.collect()

    # --- Knowledge Distillation (after CV) ---
    if RUN_KNOWLEDGE_DISTILLATION and not stop_requested:
        print(f"\n{TermColors.HEADER}--- STEP 4: Knowledge Distillation ---{TermColors.ENDC}")
        try:
            # Find the best teacher model path (using fold KD_TEACHER_FOLD's best model)
            teacher_path = os.path.join(BASE_CHECKPOINT_DIR, f"fold_{KD_TEACHER_FOLD}", "best_model.pth.tar") # Use checkpoint tar file
            if os.path.exists(teacher_path):
                 # Use the full dataset (df_full before CV split) for KD training/validation? Or define specific splits?
                 # Re-split full data for KD train/val to avoid using OOF splits directly
                 # Use df_full which contains id, scientificName, label
                 df_kd_train, df_kd_val = train_test_split(df_full, test_size=0.15, random_state=SEED+1, stratify=df_full['label']) # Simple split for KD
                 print(f"KD Data Split: Train={len(df_kd_train)}, Val={len(df_kd_val)}")
                 train_student_model(teacher_path, KD_STUDENT_MODEL_NAME, KD_STUDENT_MODEL_SAVE_PATH,
                                     df_kd_train, df_kd_val, IMAGE_DIR, NUM_CLASSES)
                 del df_kd_train, df_kd_val # Cleanup KD dataframes
                 gc.collect()
            else:
                 print(f"{TermColors.RED}❌ Teacher model checkpoint not found at {teacher_path}. Skipping Knowledge Distillation.{TermColors.ENDC}")
        except Exception as e:
            print(f"{TermColors.RED}❌ Error during Knowledge Distillation setup or execution: {e}{TermColors.ENDC}")
            traceback.print_exc()

    print(f"\n{TermColors.OKGREEN}🎉 All processes complete. Models/logs saved per fold. Stacking/KD models saved if run.{TermColors.ENDC}")

if __name__ == "__main__":
    # Print library info at script startup - only once
    print_library_info()
    
    # Call your main function
    main()