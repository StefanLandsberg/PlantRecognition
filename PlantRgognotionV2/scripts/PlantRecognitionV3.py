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
SAM_AVAILABLE = False # This will be set to True if the import succeeds

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
    from sam_optimizer.sam import SAM # Ensure this path is correct if you have a local copy
    SAM_AVAILABLE = True
except ImportError:
    print("WARN: sam_optimizer not found. SAM optimizer will be disabled. Install from 'https://github.com/davda54/sam' or ensure it's in your PYTHONPATH.")

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
        print("INFO: sam_optimizer found and available for use.")
    else:
        print("WARN: sam_optimizer not found, SAM will be disabled even if USE_SAM is True in config.")
    
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
    INFO = '\033[94m' 
    DEBUG = '\033[90m' 
    TRACE = '\033[90m' 
    ERROR = '\033[91m' 
    SUCCESS = '\033[92m'
    WARN = '\033[93m' 
    CRITICAL = '\033[91m' + '\033[1m'
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

# --- Run Mode Configuration ---
# Options: "FULL_TRAINING", "TRAIN_MLP_ON_FEATURES"
RUN_MODE = "TRAIN_MLP_ON_FEATURES" 

# --- Path Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V3_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(V3_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "plant_images")
CSV_PATH = os.path.join(DATA_DIR, "observations-568783.csv")

BASE_CHECKPOINT_DIR = os.path.join(V3_DIR, "checkpoints_v3_pytorch")
BASE_LOG_DIR = os.path.join(V3_DIR, "logs_v3_pytorch")
BASE_MODEL_SAVE_DIR = os.path.join(V3_DIR, "models_v3_pytorch")
BASE_ERROR_ANALYSIS_DIR = os.path.join(V3_DIR, "error_analysis_pytorch")

# --- Feature Extraction Config ---
FEATURES_NPZ_PATH = os.path.join(DATA_DIR, "extracted_features_v3.npz") # Path to save/load features
FEATURE_EXTRACTION_IMAGE_SIZE_CONFIG = (512, 512) # Image size for feature extraction
FEATURE_EXTRACTOR_BATCH_SIZE = 64 # Batch size for extracting features
FEATURE_EXTRACTOR_CHECKPOINT_PATH = None # Optional: Path to a specific checkpoint for the CombinedModel feature extractor

os.makedirs(BASE_CHECKPOINT_DIR, exist_ok=True)
os.makedirs(BASE_LOG_DIR, exist_ok=True)
os.makedirs(BASE_MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(BASE_ERROR_ANALYSIS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True) # Ensure data directory exists for features

# --- Training Config (Defaults, potentially tuned by HPO) ---
BATCH_SIZE = 32 if not DEBUG_MODE else 4 # Reduced default for OOM potential
GRADIENT_ACCUMULATION_STEPS = 2 if not DEBUG_MODE else 1
LEARNING_RATE = 1e-4; WEIGHT_DECAY = 1e-5
OPTIMIZER_TYPE = 'AdamP' if ADAMP_AVAILABLE else 'AdamW'

# SAM Configuration
USE_SAM = False # Set to True to use SAM, False for standard optimizer with AutoTrainingConfig
SAM_RHO = 0.05; SAM_ADAPTIVE = True
# AutoTrainingConfig will be used if USE_SAM is False
USE_AUTO_TRAIN_CONFIG = not USE_SAM

GRADIENT_CLIP_VAL = 1.0
PROGRESSIVE_RESIZING_STAGES = [
    (10 if not DEBUG_MODE else 1, (224, 224)),
    (15 if not DEBUG_MODE else 1, (384, 384)),
    (5 if not DEBUG_MODE else 1, (448, 448)), 
    (5 if not DEBUG_MODE else 1, (512, 512)), 
]
TOTAL_EPOCHS_PER_FOLD = sum(s[0] for s in PROGRESSIVE_RESIZING_STAGES)
CURRENT_IMAGE_SIZE = None 
EARLY_STOPPING_PATIENCE = 10 if not DEBUG_MODE else 3 # Patience for early stopping

# --- Cross-Validation Config ---
N_FOLDS = 5 if not DEBUG_MODE else 2

# --- Hardware Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = multiprocessing.cpu_count() // 2 if multiprocessing.cpu_count() > 1 else 0 # Adjusted default
MIXED_PRECISION = True if DEVICE.type == 'cuda' else False
USE_TORCH_COMPILE = False

# --- Model Config (for CombinedModel / Feature Extractor) ---
MODEL_NAMES = ["tf_efficientnetv2_l_in21ft1k", "convnext_large_in22ft1k"] # Dual model
DROP_PATH_RATE = 0.1; PRETRAINED = True; NUM_CLASSES = -1 
EMBEDDING_SIZE = 2048; DROPOUT_RATE = 0.3; GLOBAL_POOLING = 'avg'

# --- Metric Learning Config ---
METRIC_LEARNING_TYPE = 'ArcFace' # Applies to CombinedModel and optionally to MLP
ARCFACE_S = 30.0; ARCFACE_M = 0.6

# --- MLP Model Config (for "TRAIN_MLP_ON_FEATURES" mode) ---
MLP_HIDDEN_DIMS = [1024, 512, 256] # Hidden layer dimensions for the MLP
MLP_DROPOUT_RATE = 0.4
MLP_USE_ARCFACE = True # Whether the MLP should also use an ArcFace head
MLP_LEARNING_RATE = 1e-3
MLP_WEIGHT_DECAY = 1e-5
MLP_EPOCHS = 50 if not DEBUG_MODE else 5
MLP_BATCH_SIZE = 128 if not DEBUG_MODE else 32
MLP_EARLY_STOPPING_PATIENCE = 10 if not DEBUG_MODE else 3
MLP_OPTIMIZER_TYPE = 'AdamW'
MLP_SCHEDULER_TYPE = 'CosineWarmRestarts'
MLP_SWA_START_EPOCH_FACTOR = 0.75
MLP_USE_SWA = True
MLP_USE_EMA = True
MLP_EMA_DECAY = 0.999


# --- Loss Function & Imbalance Handling ---
LOSS_TYPE = 'CrossEntropy' 
LABEL_SMOOTHING = 0.1
FOCAL_ALPHA = 0.25; FOCAL_GAMMA = 2.0
IMBALANCE_STRATEGY = 'WeightedSampler' # For CombinedModel training and MLP training
LOGIT_ADJUSTMENT_TAU = 1.0
CLASS_FREQUENCIES = None; CLASS_PRIORS = None; CLASS_WEIGHTS = None

# --- Learning Rate Scheduler Config (for CombinedModel) ---
SCHEDULER_TYPE = 'CosineWarmRestarts' 
WARMUP_EPOCHS = 3; LR_MAX = LEARNING_RATE; LR_MIN = LEARNING_RATE * 0.01
T_0 = 10; T_MULT = 1 # Adjusted T_0 for CosineWarmRestarts
STEP_LR_STEP_SIZE = 5; STEP_LR_GAMMA = 0.1
PLATEAU_FACTOR = 0.2; PLATEAU_PATIENCE = 5; PLATEAU_MIN_LR = 1e-6 
PLATEAU_MODE = 'min'; PLATEAU_MONITOR = 'val_loss'

# --- Augmentation Config (Primarily for CombinedModel training/feature extraction) ---
USE_RAND_AUGMENT = False 
RAND_AUGMENT_N = 2; RAND_AUGMENT_M = 9
MIXUP_ALPHA = 0.8; CUTMIX_ALPHA = 1.0; AUG_PROBABILITY = 0.5

# --- Averaging Config (for CombinedModel) ---
USE_SWA = True
SWA_START_EPOCH_GLOBAL_FACTOR = 0.75 
SWA_LR_FACTOR = 0.05; SWA_ANNEAL_EPOCHS = 5
USE_EMA = True; EMA_DECAY = 0.999

# --- Checkpointing Config ---
CHECKPOINT_MONITOR = 'val_acc' 
CHECKPOINT_MODE = 'max' if CHECKPOINT_MONITOR == 'val_acc' else 'min'
SAVE_TOP_K = 1

# --- Error Analysis Config ---
ERROR_LOG_BATCH_SIZE = 64
LOG_MISCLASSIFIED_IMAGES = True

# --- Test Time Augmentation (TTA) Config ---
USE_TTA = True 
TTA_TRANSFORMS = None 

# --- Stacking Config ---
RUN_STACKING = True # Can run after MLP training too
STACKING_META_MODEL_PATH = os.path.join(BASE_MODEL_SAVE_DIR, "stacking_meta_model.joblib")
STACKING_OOF_PREDS_PATH = os.path.join(BASE_MODEL_SAVE_DIR, "oof_predictions.npz")

# --- Knowledge Distillation Config ---
RUN_KNOWLEDGE_DISTILLATION = True # Can run with MLP as teacher
KD_STUDENT_MODEL_NAME = "mobilenetv3_small_100" 
KD_STUDENT_IMAGE_SIZE = (224, 224) 
KD_STUDENT_EMBEDDING_SIZE = 512 
KD_STUDENT_DROPOUT = 0.2
KD_TEACHER_FOLD = 0 
KD_EPOCHS = 15 if not DEBUG_MODE else 2
KD_BATCH_SIZE = BATCH_SIZE * 2 
KD_LR = 1e-4
KD_ALPHA = 0.5 
KD_TEMPERATURE = 4.0 
KD_STUDENT_MODEL_SAVE_PATH = os.path.join(BASE_MODEL_SAVE_DIR, f"kd_student_{KD_STUDENT_MODEL_NAME}.pth")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- Global Variables ---
stop_requested = False; label_encoder = None; class_names = None

# --- Utility Functions ---
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # Set to True if input sizes are fixed for potential speedup
    print(f"{TermColors.INFO}ℹ Seed set to {seed}{TermColors.ENDC}")

def handle_interrupt(signal, frame):
    global stop_requested
    if stop_requested: 
        print(f"\n{TermColors.CRITICAL}🚨 Force exiting...{TermColors.ENDC}")
        sys.exit(1)
    print(f"\n{TermColors.WARNING}⚠️ Interrupt received. Finishing current epoch and saving state... Press Ctrl+C again to force exit.{TermColors.ENDC}")
    stop_requested = True

def check_keyboard_stop():
    if stop_requested:
        print(f"{TermColors.WARNING}Stop request detected. Breaking loop...{TermColors.ENDC}")
    return stop_requested

# --- Checkpointing, Saving, Logging ---
def save_checkpoint(fold, global_epoch, stage_idx, stage_epoch, model, optimizer, scheduler, scaler, best_metric, filename="checkpoint.pth.tar", is_mlp_checkpoint=False):
    checkpoint_prefix = "mlp_" if is_mlp_checkpoint else ""
    checkpoint_dir = os.path.join(BASE_CHECKPOINT_DIR, f"{checkpoint_prefix}fold_{fold}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    is_sam_optimizer = hasattr(optimizer, 'base_optimizer') and SAM_AVAILABLE and (USE_SAM if not is_mlp_checkpoint else False) # SAM not typical for MLP
    opt_state_dict = optimizer.base_optimizer.state_dict() if is_sam_optimizer else optimizer.state_dict()

    state = {
        'fold': fold,
        'global_epoch': global_epoch,
        'stage_idx': stage_idx if not is_mlp_checkpoint else -1, # Stage info less relevant for MLP
        'stage_epoch': stage_epoch if not is_mlp_checkpoint else global_epoch,
        'image_size': CURRENT_IMAGE_SIZE if not is_mlp_checkpoint else None,
        'state_dict': model_state_dict,
        'optimizer': opt_state_dict,
        'scheduler': scheduler.state_dict() if scheduler else None,
        'scaler': scaler.state_dict() if scaler else None,
        'best_metric': best_metric,
        'label_encoder_classes': list(label_encoder.classes_) if label_encoder else None,
        'class_frequencies': CLASS_FREQUENCIES if not is_mlp_checkpoint else None # Store if relevant
    }
    try:
        torch.save(state, filepath)
        print(f"{TermColors.GREEN}✅ Ckpt {'MLP ' if is_mlp_checkpoint else ''}Fold {fold} saved: {filename} (GlobEp {global_epoch}, Best {CHECKPOINT_MONITOR}: {best_metric:.4f}){TermColors.ENDC}")
    except Exception as e:
        print(f"{TermColors.RED}❌ Error saving checkpoint {filepath}: {e}{TermColors.ENDC}")

def load_checkpoint(fold, model, optimizer, scheduler, scaler, filename="checkpoint.pth.tar", is_mlp_checkpoint=False):
    checkpoint_prefix = "mlp_" if is_mlp_checkpoint else ""
    checkpoint_dir = os.path.join(BASE_CHECKPOINT_DIR, f"{checkpoint_prefix}fold_{fold}")
    filepath = os.path.join(checkpoint_dir, filename)
    start_global_epoch, start_stage_idx, start_stage_epoch = 0, 0, 0
    loaded_image_size, loaded_class_frequencies, loaded_label_classes = None, None, None
    best_metric = float('-inf') if CHECKPOINT_MODE == 'max' else float('inf')

    if os.path.isfile(filepath):
        print(f"{TermColors.CYAN}ℹ Loading {'MLP ' if is_mlp_checkpoint else ''}Fold {fold} checkpoint '{filename}'...{TermColors.ENDC}")
        try:
            ckpt = torch.load(filepath, map_location=DEVICE)
            start_global_epoch = ckpt.get('global_epoch', 0)
            if not is_mlp_checkpoint:
                start_stage_idx = ckpt.get('stage_idx', 0)
                start_stage_epoch = ckpt.get('stage_epoch', 0)
                loaded_image_size = ckpt.get('image_size', None)
                loaded_class_frequencies = ckpt.get('class_frequencies', None)
            
            best_metric = ckpt.get('best_metric', best_metric)
            loaded_label_classes = ckpt.get('label_encoder_classes', None)

            state_dict = ckpt['state_dict']
            new_state_dict = {}
            is_compiled = hasattr(model, '_orig_mod') 
            
            # Handle state dict loading for compiled vs non-compiled, module vs no-module
            current_model_is_module = any(k.startswith('module.') for k in model.state_dict().keys())
            ckpt_is_module = any(k.startswith('module.') for k in state_dict.keys())

            for k, v in state_dict.items():
                name = k
                # Strip 'module.' from ckpt if model isn't currently a module
                if ckpt_is_module and not current_model_is_module:
                    if name.startswith('module.'): name = name[len('module.'):]
                # Add 'module.' to key if model is a module but ckpt isn't
                elif not ckpt_is_module and current_model_is_module:
                    name = 'module.' + name
                
                # Handle torch.compile's _orig_mod prefix
                if is_compiled and not name.startswith('_orig_mod.'): name = '_orig_mod.' + name
                if not is_compiled and name.startswith('_orig_mod.'): name = name[len('_orig_mod.'):]
                new_state_dict[name] = v
            
            try:
                model.load_state_dict(new_state_dict, strict=False)
                print(f"{TermColors.GREEN}  Model state loaded.{TermColors.ENDC}")
            except RuntimeError as e: print(f"{TermColors.YELLOW}⚠️ Model Load Warning (strict=False): {e}{TermColors.ENDC}")
            except Exception as e: print(f"{TermColors.RED}❌ Model Load Failed: {e}{TermColors.ENDC}")


            if optimizer and 'optimizer' in ckpt and ckpt['optimizer']:
                is_sam_optimizer_runtime = hasattr(optimizer, 'base_optimizer') and SAM_AVAILABLE and (USE_SAM if not is_mlp_checkpoint else False)
                opt_to_load = optimizer.base_optimizer if is_sam_optimizer_runtime else optimizer
                try:
                    opt_to_load.load_state_dict(ckpt['optimizer'])
                    print(f"{TermColors.GREEN}  Optimizer state loaded.{TermColors.ENDC}")
                except Exception as e: print(f"{TermColors.YELLOW}⚠️ Optim Load Failed: {e}{TermColors.ENDC}")
            
            if scheduler and 'scheduler' in ckpt and ckpt['scheduler']:
                 try:
                     scheduler.load_state_dict(ckpt['scheduler'])
                     print(f"{TermColors.GREEN}  Scheduler state loaded (initial).{TermColors.ENDC}")
                 except Exception as e: print(f"{TermColors.YELLOW}⚠️ Scheduler Load Failed (initial): {e}{TermColors.ENDC}")

            if scaler and 'scaler' in ckpt and ckpt['scaler']:
                try:
                    scaler.load_state_dict(ckpt['scaler'])
                    print(f"{TermColors.GREEN}  Scaler state loaded.{TermColors.ENDC}")
                except Exception as e: print(f"{TermColors.YELLOW}⚠️ Scaler Load Failed: {e}{TermColors.ENDC}")

            print(f"{TermColors.GREEN}✅ Ckpt {'MLP ' if is_mlp_checkpoint else ''}Fold {fold} loaded. Resume GlobEp {start_global_epoch}. Best {CHECKPOINT_MONITOR}: {best_metric:.4f}{TermColors.ENDC}")
            if loaded_label_classes and label_encoder and list(label_encoder.classes_) != loaded_label_classes:
                print(f"{TermColors.CRITICAL}🚨 Label mapping mismatch! Exiting.{TermColors.ENDC}"); sys.exit(1)
        except Exception as e:
            print(f"{TermColors.RED}❌ Error loading checkpoint {filepath}: {e}{TermColors.ENDC}"); traceback.print_exc()
            start_global_epoch, start_stage_idx, start_stage_epoch = 0, 0, 0
            best_metric = float('-inf') if CHECKPOINT_MODE == 'max' else float('inf')
    else:
        print(f"{TermColors.YELLOW}⚠️ No checkpoint found for {'MLP ' if is_mlp_checkpoint else ''}Fold {fold} at {filepath}. Starting fresh.{TermColors.ENDC}")
    
    if is_mlp_checkpoint:
        return start_global_epoch, best_metric, loaded_label_classes
    else:
        return start_global_epoch, start_stage_idx, start_stage_epoch, best_metric, loaded_label_classes, loaded_image_size, loaded_class_frequencies

def save_model(fold, model, filename="final_model.pth", is_mlp_model=False):
    model_prefix = "mlp_" if is_mlp_model else ""
    model_dir = os.path.join(BASE_MODEL_SAVE_DIR, f"{model_prefix}fold_{fold}")
    os.makedirs(model_dir, exist_ok=True)
    filepath = os.path.join(model_dir, filename)
    model_to_save = model
    if hasattr(model_to_save, 'module'): model_to_save = model_to_save.module
    if hasattr(model_to_save, '_orig_mod'): model_to_save = model_to_save._orig_mod
    try:
        torch.save(model_to_save.state_dict(), filepath)
        print(f"{TermColors.GREEN}✅ {'MLP ' if is_mlp_model else ''}Fold {fold} model state_dict saved: {filename}{TermColors.ENDC}")
    except Exception as e:
        print(f"{TermColors.RED}❌ Error saving model state_dict {filepath}: {e}{TermColors.ENDC}")

def log_misclassified(fold, model, dataloader, criterion, device, global_epoch, writer, num_classes, max_images=20, is_mlp_logging=False):
    if not LOG_MISCLASSIFIED_IMAGES: return
    log_prefix = "mlp_" if is_mlp_logging else ""
    error_dir = os.path.join(BASE_ERROR_ANALYSIS_DIR, f"{log_prefix}fold_{fold}")
    os.makedirs(error_dir, exist_ok=True)
    error_log_file = os.path.join(error_dir, f"epoch_{global_epoch}_errors.csv")
    model.eval(); misclassified_count = 0; logged_images = 0
    print(f"{TermColors.CYAN}ℹ {'MLP ' if is_mlp_logging else ''}Fold {fold} Logging misclassified images for global epoch {global_epoch}...{TermColors.ENDC}")
    try:
        with open(error_log_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['image_path_or_feature_idx', 'true_label', 'predicted_label', 'confidence', 'loss', 'logits_raw', 'logits_adjusted']
            writer_csv = csv.DictWriter(csvfile, fieldnames=fieldnames); writer_csv.writeheader()
            with torch.no_grad():
                for batch_data in tqdm(dataloader, desc=f"Logging Errors {'MLP ' if is_mlp_logging else ''}Fold {fold}", leave=False):
                    paths_or_indices = None
                    if is_mlp_logging: # Expecting (features, labels, original_indices)
                        if len(batch_data) != 3: continue
                        inputs, labels, paths_or_indices = batch_data # paths_or_indices are original data indices
                    else: # Expecting (images, labels, paths)
                        if len(batch_data) != 3: continue
                        inputs, labels, paths_or_indices = batch_data # paths_or_indices are image paths

                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    with torch.amp.autocast('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda')):
                        outputs_raw = model(inputs) if is_mlp_logging or METRIC_LEARNING_TYPE != 'ArcFace' else model(inputs, labels=labels)
                        adj_outputs = outputs_raw
                        # Logit adjustment (can apply to MLP too if CLASS_PRIORS are available and strategy is LogitAdjust)
                        if IMBALANCE_STRATEGY == 'LogitAdjust' and CLASS_PRIORS is not None:
                            logit_adj_val = LOGIT_ADJUSTMENT_TAU * torch.log(CLASS_PRIORS + 1e-12)
                            adj_outputs = outputs_raw + logit_adj_val.unsqueeze(0)
                        
                        final_outputs_for_loss_pred = adj_outputs
                        if not is_mlp_logging and hasattr(model, 'metric_fc') and isinstance(model.metric_fc, ArcFace):
                             final_outputs_for_loss_pred = model.metric_fc(outputs_raw, labels) # Pass labels to ArcFace
                             if IMBALANCE_STRATEGY == 'LogitAdjust' and CLASS_PRIORS is not None:
                                 final_outputs_for_loss_pred = final_outputs_for_loss_pred + logit_adj_val.unsqueeze(0)
                        elif is_mlp_logging and MLP_USE_ARCFACE and hasattr(model, 'metric_fc') and isinstance(model.metric_fc, ArcFace):
                             final_outputs_for_loss_pred = model.metric_fc(outputs_raw, labels) # Pass labels to ArcFace for MLP
                             if IMBALANCE_STRATEGY == 'LogitAdjust' and CLASS_PRIORS is not None:
                                 final_outputs_for_loss_pred = final_outputs_for_loss_pred + logit_adj_val.unsqueeze(0)
                        
                        loss = criterion(final_outputs_for_loss_pred, labels)
                        preds = torch.argmax(final_outputs_for_loss_pred, dim=1)
                        probs = F.softmax(final_outputs_for_loss_pred, dim=1)

                    misclassified_mask = (preds != labels)
                    misclassified_indices = torch.where(misclassified_mask)[0]

                    for idx in misclassified_indices:
                        misclassified_count += 1
                        true_label_idx = labels[idx].item(); pred_label_idx = preds[idx].item()
                        confidence = probs[idx, pred_label_idx].item()
                        item_loss = F.cross_entropy(final_outputs_for_loss_pred[idx].unsqueeze(0), labels[idx].unsqueeze(0)).item()
                        
                        identifier = paths_or_indices[idx]
                        if isinstance(identifier, torch.Tensor): identifier = identifier.item() # For feature indices
                        elif isinstance(identifier, str): identifier = os.path.basename(identifier) # For image paths

                        true_n = class_names[true_label_idx] if class_names and 0 <= true_label_idx < len(class_names) else str(true_label_idx)
                        pred_n = class_names[pred_label_idx] if class_names and 0 <= pred_label_idx < len(class_names) else str(pred_label_idx)
                        
                        writer_csv.writerow({
                            'image_path_or_feature_idx': identifier, 
                            'true_label': true_n, 'predicted_label': pred_n, 
                            'confidence': f"{confidence:.4f}", 'loss': f"{item_loss:.4f}", 
                            'logits_raw': outputs_raw[idx].cpu().numpy().round(2).tolist(), 
                            'logits_adjusted': adj_outputs[idx].cpu().numpy().round(2).tolist()
                        })

                        if writer and logged_images < max_images and not is_mlp_logging: # Tensorboard image logging only for image-based models
                            try:
                                mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).to(device); std = torch.tensor(IMAGENET_STD).view(3, 1, 1).to(device)
                                img_tensor = inputs[idx] * std + mean; img_tensor = torch.clamp(img_tensor, 0, 1)
                                writer.add_image(f"Misclassified/Fold_{fold}/True_{true_n}_Pred_{pred_n}_{identifier}", img_tensor, global_epoch)
                                logged_images += 1
                            except Exception as img_e: print(f"{TermColors.YELLOW}Warn: Failed to log image {identifier} to TensorBoard: {img_e}{TermColors.ENDC}")
                    if stop_requested: break
            if stop_requested: return
        print(f"{TermColors.CYAN}ℹ {'MLP ' if is_mlp_logging else ''}Fold {fold} Misclassified images logged ({misclassified_count} total errors). CSV: {error_log_file}{TermColors.ENDC}")
    except Exception as e: print(f"{TermColors.RED}❌ Error during misclassified logging for fold {fold}: {e}{TermColors.ENDC}"); traceback.print_exc()

# --- Dataset and Transforms ---
class PlantDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None, label_encoder=None, include_paths=False, image_size=None):
        self.input_df = dataframe.copy()
        self.image_dir = image_dir
        self.transform = transform
        self.include_paths = include_paths
        self.image_size = image_size if image_size else PROGRESSIVE_RESIZING_STAGES[0][1] # Fallback for safety
        self.image_data = []
        
        required_cols = ['scientificName', 'id', 'label']
        if not all(col in self.input_df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in self.input_df.columns]
            print(f"{TermColors.RED}❌ PlantDataset input dataframe missing required columns: {missing}. Found: {self.input_df.columns.tolist()}{TermColors.ENDC}")
            self.dataframe = pd.DataFrame(self.image_data); return

        image_path_lookup = defaultdict(list)
        if not os.path.isdir(self.image_dir):
            print(f"{TermColors.RED}❌ Image directory not found: {self.image_dir}{TermColors.ENDC}")
            self.dataframe = pd.DataFrame(self.image_data); return
        
        species_folder_names_in_fs = []
        try: species_folder_names_in_fs = [d for d in os.listdir(self.image_dir) if os.path.isdir(os.path.join(self.image_dir, d))]
        except OSError as e: print(f"{TermColors.RED}❌ Error listing directories in {self.image_dir}: {e}{TermColors.ENDC}"); self.dataframe = pd.DataFrame(self.image_data); return

        for species_dir_name_from_fs in species_folder_names_in_fs:
            current_species_dir_path = os.path.join(self.image_dir, species_dir_name_from_fs)
            prefix_to_strip = species_dir_name_from_fs + "_"
            try:
                for filename in os.listdir(current_species_dir_path):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        full_path = os.path.join(current_species_dir_path, filename)
                        try:
                            if os.path.getsize(full_path) > 0:
                                if filename.startswith(prefix_to_strip):
                                    id_part = filename[len(prefix_to_strip):]; obs_id_from_file = id_part.split('_')[0]
                                    if '.' in obs_id_from_file: obs_id_from_file = obs_id_from_file.split('.')[0]
                                    if obs_id_from_file: image_path_lookup[(species_dir_name_from_fs, obs_id_from_file)].append(full_path)
                        except OSError: pass # Skip if getsize fails (e.g. broken symlink)
            except OSError: print(f"{TermColors.YELLOW}Warn: Could not read directory {current_species_dir_path}. Skipping.{TermColors.ENDC}"); continue
        
        self.input_df['original_index'] = self.input_df.index
        for idx, row in self.input_df.iterrows():
            try:
                species_name_csv = str(row['scientificName']); obs_id_csv = str(row['id']); label = row['label']; original_index = row['original_index']
                species_dir_name_csv_derived = species_name_csv.replace(' ', '_').replace('/', '_').replace('\\', '_')
                found_files = image_path_lookup.get((species_dir_name_csv_derived, obs_id_csv), [])
                for full_path in found_files:
                    self.image_data.append({'scientificName': species_name_csv, 'label': label, 'image_path': full_path, 'original_index': original_index})
            except Exception as e: print(f"{TermColors.RED}Error processing CSV row {idx} (ID: {row.get('id', 'N/A')}): {e}{TermColors.ENDC}")
        
        self.dataframe = pd.DataFrame(self.image_data)
        found_count = len(self.dataframe)
        if found_count == 0 and len(self.input_df) > 0:
             print(f"{TermColors.RED}❌ Found 0 image files after matching. Check paths/IDs.{TermColors.ENDC}")

    def __len__(self): return len(self.dataframe)
    def get_labels(self):
        if 'label' in self.dataframe.columns: return self.dataframe['label'].tolist()
        else: print(f"{TermColors.RED}❌ 'label' column missing in internal dataframe!{TermColors.ENDC}"); return []

    def __getitem__(self, idx):
        if idx >= len(self.dataframe):
             print(f"{TermColors.RED}❌ Index {idx} OOB. Dummy.{TermColors.ENDC}")
             dummy_img = torch.zeros((3, *self.image_size), dtype=torch.float32); label = -1
             return (dummy_img, label, "ERROR_INDEX_OOB", -1) if self.include_paths else (dummy_img, label)
        
        row = self.dataframe.iloc[idx]; img_path = row['image_path']; label = row['label']; original_index = row['original_index']
        try:
            image = Image.open(img_path).convert('RGB'); image = np.array(image)
            if image is None: raise IOError("Image loading returned None")
        except Exception as e:
            print(f"{TermColors.RED}Err load {img_path}: {e}. Dummy.{TermColors.ENDC}")
            dummy_img = torch.zeros((3, *self.image_size), dtype=torch.float32); label = -1; error_filename = os.path.basename(img_path)
            return (dummy_img, label, f"ERROR_LOAD_{error_filename}", original_index) if self.include_paths else (dummy_img, label)

        if self.transform:
            try: augmented = self.transform(image=image); image = augmented['image']
            except Exception as e:
                print(f"{TermColors.RED}Err transform {img_path}: {e}. Fallback.{TermColors.ENDC}")
                try:
                    image = Image.fromarray(image); fallback_transform = transforms.Compose([transforms.Resize(self.image_size), transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
                    image = fallback_transform(image)
                except Exception as fb_e:
                    print(f"{TermColors.RED}Fallback failed {img_path}: {fb_e}. Dummy.{TermColors.ENDC}")
                    dummy_img = torch.zeros((3, *self.image_size), dtype=torch.float32); label = -1; error_filename = os.path.basename(img_path)
                    return (dummy_img, label, f"ERROR_TF_{error_filename}", original_index) if self.include_paths else (dummy_img, label)
        
        if self.include_paths: return image, label, img_path, original_index
        else: return image, label

class FeatureDataset(Dataset):
    def __init__(self, features, labels, original_indices=None):
        self.features = features
        self.labels = labels
        self.original_indices = original_indices
        if self.original_indices is None:
            self.original_indices = np.arange(len(features))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.original_indices[idx]
    
    def get_labels(self): # For WeightedRandomSampler
        return self.labels.tolist() if isinstance(self.labels, np.ndarray) else self.labels


def get_transforms(image_size=(224, 224), augmentations_config=None, for_feature_extraction=False):
    h, w = int(image_size[0]), int(image_size[1])
    
    if for_feature_extraction: # Minimal transforms for feature extraction, usually no heavy aug
        transform_list = [A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR), A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
        return A.Compose(transform_list)

    # Default train/val transforms for end-to-end training
    train_transform_list = [A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR), A.HorizontalFlip(p=0.5), A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5), A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    train_transform = A.Compose(train_transform_list)
    val_transform = A.Compose([A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR), A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])
    
    global TTA_TRANSFORMS 
    if USE_TTA: TTA_TRANSFORMS = A.Compose([A.HorizontalFlip(p=1.0), A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR), A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])
    else: TTA_TRANSFORMS = None
    return train_transform, val_transform

# --- Model Architecture ---
class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=ARCFACE_S, m=ARCFACE_M, easy_margin=False, ls_eps=0.0):
        super(ArcFace, self).__init__(); self.in_features = in_features; self.out_features = out_features; self.s = s; self.m = m; self.ls_eps = ls_eps
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features)); nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin; self.cos_m = math.cos(m); self.sin_m = math.sin(m); self.th = math.cos(math.pi - m); self.mm = math.sin(math.pi - m) * m
    def forward(self, input_features, label): 
        cosine = F.linear(F.normalize(input_features), F.normalize(self.weight))
        if label is None: return cosine * self.s 
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).clamp(0, 1); phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin: phi = torch.where(cosine > 0, phi, cosine)
        else: phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=input_features.device); one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0: one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine); output *= self.s
        return output

class CombinedModel(nn.Module):
    def __init__(self, model_names, num_classes, pretrained=True, global_pool='avg', dropout_rate=0.3, embedding_size=512, drop_path_rate=0.1, arcface_s=ARCFACE_S, arcface_m=ARCFACE_M, metric_learning=METRIC_LEARNING_TYPE):
        super().__init__(); self.model_names = model_names; self.num_classes = num_classes; self.embedding_size = embedding_size; self.metric_learning = metric_learning
        self.backbones = nn.ModuleList(); self.total_features = 0; self.feature_dims = {}
        for name in model_names:
            try:
                kwargs = {}; supported_families = ['efficientnet', 'convnext', 'vit', 'swin', 'beit', 'deit']; model_family = name.split('_')[0].lower()
                if any(family in model_family for family in supported_families) and drop_path_rate > 0: kwargs['drop_path_rate'] = drop_path_rate
                backbone = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool=global_pool, **kwargs)
                backbone_features = backbone.num_features
                if 'mobilenetv3_large_100' in name and global_pool == 'avg': backbone_features = 960
                elif 'efficientnetv2_l' in name and global_pool == 'avg': backbone_features = 1280
                self.feature_dims[name] = backbone_features; self.backbones.append(backbone); self.total_features += backbone_features
            except Exception as e: print(f"{TermColors.RED}❌ Backbone Load Fail {name}: {e}{TermColors.ENDC}"); raise e
        self.embedding_layer = nn.Sequential(nn.Linear(self.total_features, self.embedding_size), nn.BatchNorm1d(self.embedding_size), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(dropout_rate)
        if self.metric_learning == 'ArcFace': self.metric_fc = ArcFace(self.embedding_size, num_classes, s=arcface_s, m=arcface_m)
        else: self.metric_fc = nn.Linear(self.embedding_size, num_classes)
    
    def forward(self, x, labels=None, return_embedding=False): # Added return_embedding flag
        all_features = [backbone(x) for backbone in self.backbones]
        combined_features = torch.cat(all_features, dim=1) if len(all_features) > 1 else all_features[0]
        embedding = self.embedding_layer(combined_features)
        
        if return_embedding:
            return embedding # Return embedding before dropout and metric_fc
            
        embedding_dropped = self.dropout(embedding)
        if self.metric_learning == 'ArcFace': output = self.metric_fc(embedding_dropped, labels)
        else: output = self.metric_fc(embedding_dropped)
        return output

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_dims, num_classes, dropout_rate=MLP_DROPOUT_RATE, use_arcface=MLP_USE_ARCFACE, arcface_s=ARCFACE_S, arcface_m=ARCFACE_M):
        super().__init__()
        self.use_arcface = use_arcface
        layers = []
        current_dim = input_size
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        
        if self.use_arcface:
            self.metric_fc = ArcFace(current_dim, num_classes, s=arcface_s, m=arcface_m)
        else:
            self.metric_fc = nn.Linear(current_dim, num_classes)

    def forward(self, x, labels=None): # labels for ArcFace
        x = self.hidden_layers(x)
        if self.use_arcface:
            x = self.metric_fc(x, labels)
        else:
            x = self.metric_fc(x)
        return x

def build_model(model_names=MODEL_NAMES, num_classes=NUM_CLASSES, pretrained=PRETRAINED, dropout_rate=DROPOUT_RATE, embedding_size=EMBEDDING_SIZE, drop_path_rate=DROP_PATH_RATE, global_pool=GLOBAL_POOLING, arcface_s=ARCFACE_S, arcface_m=ARCFACE_M, metric_learning=METRIC_LEARNING_TYPE):
    model = CombinedModel(model_names, num_classes, pretrained, global_pool, dropout_rate, embedding_size, drop_path_rate, arcface_s, arcface_m, metric_learning)
    return model

def build_mlp_model(input_size, num_classes):
    return SimpleMLP(input_size, MLP_HIDDEN_DIMS, num_classes, MLP_DROPOUT_RATE, MLP_USE_ARCFACE, ARCFACE_S, ARCFACE_M)


# --- Loss Functions ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, reduction='mean'):
        super().__init__(); self.alpha = alpha; self.gamma = gamma; self.reduction = reduction
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none'); pt = torch.exp(-ce_loss); focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss

def get_criterion(loss_type=LOSS_TYPE, label_smoothing=LABEL_SMOOTHING, class_weights=CLASS_WEIGHTS):
    weights = class_weights.to(DEVICE) if class_weights is not None and IMBALANCE_STRATEGY == 'WeightedLoss' else None
    if loss_type == 'FocalLoss': 
        return FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=weights)

# --- Optimizer and Scheduler ---
def get_optimizer(model, optimizer_type=OPTIMIZER_TYPE, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, use_sam_flag=USE_SAM, sam_rho=SAM_RHO, sam_adaptive=SAM_ADAPTIVE, is_mlp_optimizer=False):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params: raise ValueError("No trainable parameters in model")
    
    current_lr = lr if not is_mlp_optimizer else MLP_LEARNING_RATE
    current_wd = weight_decay if not is_mlp_optimizer else MLP_WEIGHT_DECAY
    current_optim_type = optimizer_type if not is_mlp_optimizer else MLP_OPTIMIZER_TYPE
    current_use_sam = use_sam_flag if not is_mlp_optimizer else False # SAM typically not for MLP

    actually_use_sam = current_use_sam and SAM_AVAILABLE

    if actually_use_sam:
        from functools import partial
        if current_optim_type == 'AdamP' and ADAMP_AVAILABLE: base_optimizer_fn = partial(AdamP, lr=current_lr, weight_decay=current_wd, betas=(0.9, 0.999), nesterov=True)
        else: base_optimizer_fn = partial(optim.AdamW, lr=current_lr, weight_decay=current_wd)
        return SAM(params, base_optimizer_fn, rho=sam_rho, adaptive=sam_adaptive)
    else:
        if current_optim_type == 'AdamP' and ADAMP_AVAILABLE: return AdamP(params, lr=current_lr, weight_decay=current_wd, betas=(0.9, 0.999), nesterov=True)
        if current_optim_type == 'AdamW': return optim.AdamW(params, lr=current_lr, weight_decay=current_wd)
        if current_optim_type == 'SGD': return optim.SGD(params, lr=current_lr, weight_decay=current_wd, momentum=0.9, nesterov=True)
        return optim.AdamW(params, lr=current_lr, weight_decay=current_wd)

def get_scheduler(optimizer, scheduler_type=SCHEDULER_TYPE, total_epochs=TOTAL_EPOCHS_PER_FOLD, warmup_epochs=WARMUP_EPOCHS, lr_max=LEARNING_RATE, lr_min=LR_MIN, t_0=T_0, t_mult=T_MULT, step_size=STEP_LR_STEP_SIZE, gamma=STEP_LR_GAMMA, plateau_factor=PLATEAU_FACTOR, plateau_patience=PLATEAU_PATIENCE, plateau_min_lr=PLATEAU_MIN_LR, plateau_mode=PLATEAU_MODE, plateau_monitor=PLATEAU_MONITOR, is_mlp_scheduler=False):
    opt_for_scheduler = optimizer.base_optimizer if hasattr(optimizer, 'base_optimizer') else optimizer
    
    current_scheduler_type = scheduler_type if not is_mlp_scheduler else MLP_SCHEDULER_TYPE
    current_total_epochs = total_epochs if not is_mlp_scheduler else MLP_EPOCHS
    current_lr_max = lr_max if not is_mlp_scheduler else MLP_LEARNING_RATE
    current_lr_min = lr_min if not is_mlp_scheduler else MLP_LEARNING_RATE * 0.01 # Example for MLP
    current_t_0 = t_0 if not is_mlp_scheduler else current_total_epochs // 2 if current_total_epochs > 1 else 1 # Example for MLP
    current_warmup_epochs = warmup_epochs if not is_mlp_scheduler else 1 # Example for MLP

    if current_scheduler_type == 'CosineWarmRestarts':
        main_scheduler = CosineAnnealingWarmRestarts(opt_for_scheduler, T_0=current_t_0, T_mult=t_mult, eta_min=current_lr_min)
    elif current_scheduler_type == 'StepLR':
        main_scheduler = StepLR(opt_for_scheduler, step_size=step_size, gamma=gamma)
    elif current_scheduler_type == 'ReduceLROnPlateau':
        main_scheduler = ReduceLROnPlateau(opt_for_scheduler, mode=plateau_mode, factor=plateau_factor, patience=plateau_patience, min_lr=plateau_min_lr, verbose=False, monitor=plateau_monitor)
    else:
        return None
    
    if current_warmup_epochs > 0 and current_scheduler_type != 'ReduceLROnPlateau':
        warmup_scheduler = LinearLR(opt_for_scheduler, start_factor=0.01, end_factor=1.0, total_iters=current_warmup_epochs)
        return SequentialLR(opt_for_scheduler, schedulers=[warmup_scheduler, main_scheduler], milestones=[current_warmup_epochs])
    return main_scheduler

# --- Data Augmentation Helpers ---
def mixup_data(x, y, alpha=1.0, device='cuda'):
    if alpha > 0: lam = np.random.beta(alpha, alpha)
    else: lam = 1
    batch_size = x.size()[0]; index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]; y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0, device='cuda'):
    if alpha > 0: lam = np.random.beta(alpha, alpha)
    else: lam = 1
    batch_size = x.size()[0]; index = torch.randperm(batch_size).to(device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam); mixed_x = x.clone()
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[2]; H = size[3]; cut_rat = np.sqrt(1. - lam); cut_w = int(W * cut_rat); cut_h = int(H * cut_rat)
    cx = np.random.randint(W); cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W); bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W); bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# --- EMA Helper ---
@torch.no_grad()
def ema_avg_fn(averaged_model_parameter, model_parameter, num_averaged):
    decay = EMA_DECAY # Default for CombinedModel
    if hasattr(averaged_model_parameter, '_is_mlp_ema') and averaged_model_parameter._is_mlp_ema: # Check if MLP EMA
        decay = MLP_EMA_DECAY
    return decay * averaged_model_parameter + (1 - decay) * model_parameter

# --- Training & Validation Loops ---
def train_one_epoch(model, dataloader, criterion, optimizer, scaler, scheduler, global_epoch, stage_idx, stage_epoch, stage_total_epochs, device, writer, num_classes, ema_model,
                    mixup_alpha=MIXUP_ALPHA, cutmix_alpha=CUTMIX_ALPHA, aug_probability=AUG_PROBABILITY, grad_clip_val=GRADIENT_CLIP_VAL,
                    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS, use_sam_flag=USE_SAM, fold_num=0, is_mlp_training=False):
    model.train(); running_loss = 0.0; total_samples = 0; all_preds, all_labels = [], []
    is_sam_active_for_optim = hasattr(optimizer, 'base_optimizer') and SAM_AVAILABLE and use_sam_flag and not is_mlp_training

    pbar_prefix = "MLP " if is_mlp_training else ""
    pbar_desc = f"{pbar_prefix}F{fold_num} E{global_epoch+1}/{stage_total_epochs} Tr" if not is_mlp_training or isinstance(fold_num, int) else f"HPO {pbar_prefix}Train Ep {global_epoch+1}"
    
    progress_bar = tqdm(dataloader, desc=pbar_desc, leave=True, disable=(fold_num == "HPO" and not is_mlp_training), bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}')
    if not is_sam_active_for_optim: optimizer.zero_grad() 
    
    total_batches = len(dataloader)
    # epoch_progress = tqdm(total=total_batches, desc=f"{pbar_prefix}Epoch {global_epoch+1} Progress", leave=True, position=0, bar_format='{desc}: {percentage:3.0f}%|{bar:50}{r_bar}')
    
    for batch_idx, batch_data in enumerate(progress_bar):
        if check_keyboard_stop(): break
        
        original_indices_batch = None
        if is_mlp_training: # features, labels, original_indices
            if len(batch_data) != 3: continue
            inputs, labels_orig, original_indices_batch = batch_data
        else: # images, labels
            if len(batch_data) != 2: continue
            inputs, labels_orig = batch_data
        
        inputs, labels_orig = inputs.to(device), labels_orig.to(device); batch_size = inputs.size(0)

        use_mixup, use_cutmix = False, False
        if not is_mlp_training: # Mixup/Cutmix for image data only
            r = np.random.rand()
            if mixup_alpha > 0 and cutmix_alpha > 0 and r < aug_probability:
                if np.random.rand() < 0.5: inputs, targets_a, targets_b, lam = mixup_data(inputs, labels_orig, mixup_alpha, device); use_mixup = True
                else: inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels_orig, cutmix_alpha, device); use_cutmix = True
            elif mixup_alpha > 0 and r < aug_probability: inputs, targets_a, targets_b, lam = mixup_data(inputs, labels_orig, mixup_alpha, device); use_mixup = True
            elif cutmix_alpha > 0 and r < aug_probability: inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels_orig, cutmix_alpha, device); use_cutmix = True
            else: lam = 1.0; targets_a, targets_b = labels_orig, labels_orig
        else: # No mixup/cutmix for MLP features
            lam = 1.0; targets_a, targets_b = labels_orig, labels_orig
        
        labels_for_arcface = targets_a # For ArcFace, pass the (potentially mixed) targets_a.

        with torch.amp.autocast('cuda', enabled=(MIXED_PRECISION and device.type == 'cuda')):
            if is_sam_active_for_optim:
                outputs1 = model(inputs, labels=labels_for_arcface if METRIC_LEARNING_TYPE == 'ArcFace' else None) 
                adj_outputs1 = outputs1
                if IMBALANCE_STRATEGY == 'LogitAdjust' and CLASS_PRIORS is not None:
                    logit_adj = LOGIT_ADJUSTMENT_TAU * torch.log(CLASS_PRIORS + 1e-12)
                    adj_outputs1 = outputs1 + logit_adj.unsqueeze(0)
                loss1 = mixup_criterion(criterion, adj_outputs1, targets_a, targets_b, lam) if use_mixup or use_cutmix else criterion(adj_outputs1, targets_a)
                scaler.scale(loss1 / gradient_accumulation_steps).backward()
                optimizer.first_step(zero_grad=True)

                outputs2 = model(inputs, labels=labels_for_arcface if METRIC_LEARNING_TYPE == 'ArcFace' else None)
                adj_outputs_final = outputs2
                if IMBALANCE_STRATEGY == 'LogitAdjust' and CLASS_PRIORS is not None:
                    adj_outputs_final = outputs2 + logit_adj.unsqueeze(0)
                loss2 = mixup_criterion(criterion, adj_outputs_final, targets_a, targets_b, lam) if use_mixup or use_cutmix else criterion(adj_outputs_final, targets_a)
                loss_final = loss2
                scaler.scale(loss_final / gradient_accumulation_steps).backward()
            else: # Standard or MLP training path
                current_metric_learning_type = METRIC_LEARNING_TYPE if not is_mlp_training else ('ArcFace' if MLP_USE_ARCFACE else 'None')
                outputs = model(inputs, labels=labels_for_arcface if current_metric_learning_type == 'ArcFace' else None)
                adj_outputs_final = outputs
                if IMBALANCE_STRATEGY == 'LogitAdjust' and CLASS_PRIORS is not None:
                    logit_adj = LOGIT_ADJUSTMENT_TAU * torch.log(CLASS_PRIORS + 1e-12)
                    adj_outputs_final = outputs + logit_adj.unsqueeze(0)
                loss = mixup_criterion(criterion, adj_outputs_final, targets_a, targets_b, lam) if (use_mixup or use_cutmix) and not is_mlp_training else criterion(adj_outputs_final, targets_a)
                loss_final = loss
                scaler.scale(loss_final / gradient_accumulation_steps).backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if is_sam_active_for_optim:
                optimizer.second_step(zero_grad=True)
            else:
                if grad_clip_val > 0 and not is_mlp_training : # Grad clip mainly for large models
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            if (USE_EMA and not is_mlp_training and ema_model) or \
               (MLP_USE_EMA and is_mlp_training and ema_model):
                ema_model.update_parameters(model)
        
        if not torch.isnan(loss_final) and not torch.isinf(loss_final):
            current_step_loss = loss_final.item(); running_loss += current_step_loss * batch_size; total_samples += batch_size
            preds_for_acc = torch.argmax(adj_outputs_final, dim=1)
            all_preds.append(preds_for_acc.detach().cpu()); all_labels.append(labels_orig.detach().cpu())
            if batch_idx % 20 == 0 or batch_idx == total_batches -1:
                current_acc = 0.0
                if all_preds and all_labels:
                    temp_preds_tensor = torch.cat(all_preds); temp_labels_tensor = torch.cat(all_labels)
                    current_acc = (temp_preds_tensor == temp_labels_tensor).sum().item() / len(temp_labels_tensor) if len(temp_labels_tensor) > 0 else 0.0
                progress_bar.set_postfix(loss=f"{current_step_loss:.3f}", acc=f"{current_acc:.3f}", lr=f"{optimizer.param_groups[0]['lr']:.1E}")
        else:
            if (batch_idx + 1) % gradient_accumulation_steps != 0:
                if not is_sam_active_for_optim: optimizer.zero_grad()
        # epoch_progress.update(1)
    # epoch_progress.close()
    if stop_requested: return None, None
    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_acc = 0.0
    if all_preds and all_labels:
        all_preds_tensor = torch.cat(all_preds); all_labels_tensor = torch.cat(all_labels)
        epoch_acc = (all_preds_tensor == temp_labels_tensor).sum().item() / total_samples if total_samples > 0 else 0
    
    current_use_swa = USE_SWA if not is_mlp_training else MLP_USE_SWA
    swa_start_epoch_factor = SWA_START_EPOCH_GLOBAL_FACTOR if not is_mlp_training else MLP_SWA_START_EPOCH_FACTOR
    
    # Determine swa_start_epoch based on context (stage or global for MLP)
    if not is_mlp_training:
        swa_start_epoch_val = globals().get('swa_start_epoch_stage', float('inf'))
    else: # For MLP, swa_start_epoch is based on total MLP epochs
        swa_start_epoch_val = int(stage_total_epochs * swa_start_epoch_factor)


    if scheduler and not isinstance(scheduler, ReduceLROnPlateau) and not (current_use_swa and stage_epoch >= swa_start_epoch_val):
        scheduler.step()

    if writer and fold_num != "HPO":
        log_prefix = "MLP/" if is_mlp_training else ""
        writer.add_scalar(f'{log_prefix}Loss/train', epoch_loss, global_epoch)
        writer.add_scalar(f'{log_prefix}Accuracy/train', epoch_acc, global_epoch)
        writer.add_scalar(f'{log_prefix}LearningRate', optimizer.param_groups[0]['lr'], global_epoch)
    return epoch_loss, epoch_acc

def validate_one_epoch(model, dataloader, criterion, device, global_epoch, writer, num_classes, scheduler=None, swa_model=None, ema_model=None, return_preds=False, fold_num=0, is_mlp_validation=False):
    model.eval(); 
    if swa_model: swa_model.eval()
    if ema_model: ema_model.eval()
    results = {}; oof_data = {'preds': [], 'indices': []}
    
    models_to_eval = {'base': model}
    current_use_swa = USE_SWA if not is_mlp_validation else MLP_USE_SWA
    current_use_ema = USE_EMA if not is_mlp_validation else MLP_USE_EMA

    if current_use_swa and swa_model: models_to_eval['swa'] = swa_model
    if current_use_ema and ema_model: models_to_eval['ema'] = ema_model

    with torch.no_grad():
        for model_key, current_model in models_to_eval.items():
            current_model.eval(); model_running_loss = 0.0; model_total_samples = 0
            model_all_preds, model_all_labels = [], []
            
            # TTA only for image-based models, not MLP features
            apply_tta = USE_TTA and not is_mlp_validation and fold_num != "HPO" and TTA_TRANSFORMS is not None
            
            pbar_prefix = "MLP " if is_mlp_validation else ""
            pbar_desc = f"{pbar_prefix}Fold {fold_num} Validate GlobEp {global_epoch+1} ({model_key})"
            if fold_num == "HPO": pbar_desc = f"HPO {pbar_prefix}Trial Validate Ep {global_epoch+1}"
            
            progress_bar = tqdm(dataloader, desc=pbar_desc, leave=False)
            for batch_data in progress_bar:
                original_indices_batch = None
                if is_mlp_validation: # features, labels, original_indices
                    if len(batch_data) != 3: continue
                    inputs, labels, original_indices_batch = batch_data
                elif len(batch_data) == 4 and return_preds and not is_mlp_validation: # images, labels, paths, original_indices
                    inputs, labels, _, original_indices_batch = batch_data
                elif len(batch_data) == 2 and not is_mlp_validation: # images, labels
                    inputs, labels = batch_data
                else: continue
                
                inputs, labels = inputs.to(device), labels.to(device); batch_size = inputs.size(0)
                
                current_metric_learning_type = METRIC_LEARNING_TYPE if not is_mlp_validation else ('ArcFace' if MLP_USE_ARCFACE else 'None')

                with torch.amp.autocast('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda')):
                    outputs_orig = current_model(inputs, labels=labels if current_metric_learning_type == 'ArcFace' else None)
                    adj_outputs = outputs_orig
                    if IMBALANCE_STRATEGY == 'LogitAdjust' and CLASS_PRIORS is not None:
                        logit_adj_val = LOGIT_ADJUSTMENT_TAU * torch.log(CLASS_PRIORS + 1e-12)
                        adj_outputs = outputs_orig + logit_adj_val.unsqueeze(0)
                    loss = criterion(adj_outputs, labels)

                tta_adj_outputs = None
                if apply_tta: # This block is for non-MLP TTA
                    try:
                        inputs_tta_list = []
                        for i in range(inputs.size(0)):
                             img_np = inputs[i].cpu().permute(1, 2, 0).numpy() * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
                             img_np = (img_np * 255).astype(np.uint8)
                             augmented = TTA_TRANSFORMS(image=img_np); inputs_tta_list.append(augmented['image'])
                        inputs_tta = torch.stack(inputs_tta_list).to(device)
                        with torch.amp.autocast('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda')):
                            outputs_tta_raw = current_model(inputs_tta, labels=labels if current_metric_learning_type == 'ArcFace' else None)
                            tta_adj_outputs = outputs_tta_raw
                            if IMBALANCE_STRATEGY == 'LogitAdjust' and CLASS_PRIORS is not None:
                                tta_adj_outputs = outputs_tta_raw + logit_adj_val.unsqueeze(0)
                    except Exception as tta_e: print(f"{TermColors.YELLOW}Warn: TTA failed: {tta_e}{TermColors.ENDC}"); tta_adj_outputs = None
                
                final_outputs_for_preds = adj_outputs
                if tta_adj_outputs is not None: final_outputs_for_preds = (adj_outputs + tta_adj_outputs) / 2.0
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    model_running_loss += loss.item() * batch_size; model_total_samples += batch_size
                    preds = torch.argmax(final_outputs_for_preds, dim=1)
                    model_all_preds.append(preds.detach().cpu()); model_all_labels.append(labels.detach().cpu())
                    if return_preds and original_indices_batch is not None and model_key == 'base':
                         oof_data['preds'].append(F.softmax(final_outputs_for_preds, dim=1).detach().cpu())
                         oof_data['indices'].append(original_indices_batch.detach().cpu()) # original_indices_batch is already on CPU if from FeatureDataset
                else: print(f"{TermColors.RED}Warn: NaN/Inf validation loss. Skipping batch.{TermColors.ENDC}")
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")
                if stop_requested: return None, None, None, None
            
            epoch_loss = model_running_loss / model_total_samples if model_total_samples > 0 else 0
            epoch_acc = 0.0
            if model_all_preds and model_all_labels:
                preds_tensor = torch.cat(model_all_preds); labels_tensor = torch.cat(model_all_labels)
                epoch_acc = (preds_tensor == labels_tensor).sum().item() / model_total_samples if model_total_samples > 0 else 0
            results[model_key] = (epoch_loss, epoch_acc)
            if writer and fold_num != "HPO":
                log_prefix = "MLP/" if is_mlp_validation else ""
                writer.add_scalar(f'{log_prefix}Loss/val_{model_key}', epoch_loss, global_epoch)
                writer.add_scalar(f'{log_prefix}Accuracy/val_{model_key}', epoch_acc, global_epoch)
    
    base_loss, base_acc = results.get('base', (float('inf'), 0.0))
    oof_preds_concat = torch.cat(oof_data['preds']).numpy() if oof_data['preds'] else None
    oof_indices_concat = torch.cat(oof_data['indices']).numpy() if oof_data['indices'] else None
    
    if scheduler and isinstance(scheduler, ReduceLROnPlateau):
        metric_to_monitor_for_plateau = base_loss if PLATEAU_MONITOR == 'val_loss' else base_acc
        scheduler.step(metric_to_monitor_for_plateau)
    return base_loss, base_acc, oof_preds_concat, oof_indices_concat

# --- Stacking ---
def train_stacking_meta_model(oof_preds, oof_labels, save_path):
    print(f"{TermColors.CYAN}ℹ Training Stacking Meta-Model...{TermColors.ENDC}")
    if oof_preds.ndim > 1 and oof_preds.shape[1] > 1: oof_features = oof_preds 
    else: oof_features = np.argmax(oof_preds, axis=1).reshape(-1,1) if oof_preds.ndim > 1 else oof_preds.reshape(-1,1)
    meta_model = LogisticRegression(max_iter=1000, random_state=SEED, C=0.1, solver='liblinear')
    try:
        meta_model.fit(oof_features, oof_labels)
        meta_preds = meta_model.predict(oof_features)
        meta_acc = accuracy_score(oof_labels, meta_preds)
        print(f"{TermColors.GREEN}✅ Stacking meta-model trained. OOF Accuracy: {meta_acc:.4f}{TermColors.ENDC}")
        joblib.dump(meta_model, save_path); print(f"  Meta-model saved to: {save_path}")
    except Exception as e: print(f"{TermColors.RED}❌ Error training stacking meta-model: {e}{TermColors.ENDC}"); traceback.print_exc()

# --- Auto Training Configuration ---
class AutoTrainingConfig: # Primarily for CombinedModel end-to-end training
    def __init__(self, initial_lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY):
        self.initial_lr = initial_lr; self.weight_decay = weight_decay; self.plateau_count = 0; self.overfit_count = 0
        self.best_val_loss = float('inf'); self.best_val_acc = 0.0; self.train_losses = []; self.val_losses = []
        self.train_accs = []; self.val_accs = []; self.lr_history = []; self.wd_history = []; self.adjustment_log = []
    def update(self, train_loss, train_acc, val_loss, val_acc, optimizer, is_sam_active_runtime):
        self.train_losses.append(train_loss); self.val_losses.append(val_loss); self.train_accs.append(train_acc); self.val_accs.append(val_acc)
        current_opt_for_lr_wd = optimizer.base_optimizer if is_sam_active_runtime else optimizer
        current_lr = current_opt_for_lr_wd.param_groups[0]['lr']; current_wd = current_opt_for_lr_wd.param_groups[0]['weight_decay']
        self.lr_history.append(current_lr); self.wd_history.append(current_wd)
        val_loss_improved = val_loss < self.best_val_loss; val_acc_improved = val_acc > self.best_val_acc
        if val_loss_improved: self.best_val_loss = val_loss; self.plateau_count = 0
        else: self.plateau_count += 1
        if val_acc_improved: self.best_val_acc = val_acc
        is_overfitting = False
        if len(self.train_losses) >= 3:
            train_decreasing = self.train_losses[-1] < self.train_losses[-2]; val_increasing = self.val_losses[-1] > self.val_losses[-2]
            gap_growing = (self.val_losses[-1] - self.train_losses[-1]) > (self.val_losses[-2] - self.train_losses[-2])
            if train_decreasing and (val_increasing or gap_growing): self.overfit_count += 1; is_overfitting = True
            else: self.overfit_count = max(0, self.overfit_count - 0.5)
        adjustments_made = False
        if self.plateau_count >= 3:
            if current_lr < self.initial_lr * 1.5:
                new_lr = current_lr * 1.2; new_wd = max(current_wd * 0.8, 1e-6)
                self._adjust_optimizer_params(optimizer, new_lr, new_wd, is_sam_active_runtime)
                self.adjustment_log.append(f"Increased LR to {new_lr:.2e} (plateau)"); adjustments_made = True
            self.plateau_count = 0
        elif self.overfit_count >= 2:
            new_lr = current_lr * 0.7; new_wd = current_wd * 1.5
            self._adjust_optimizer_params(optimizer, new_lr, new_wd, is_sam_active_runtime)
            self.adjustment_log.append(f"Reduced LR to {new_lr:.2e}, Inc WD to {new_wd:.2e} (overfit)"); adjustments_made = True
            self.overfit_count = 0
        if is_sam_active_runtime and hasattr(optimizer, 'rho'): self._adjust_sam_parameters(optimizer, is_overfitting)
        return adjustments_made
    def _adjust_optimizer_params(self, optimizer, new_lr, new_wd, is_sam_active_runtime):
        opt_to_adjust = optimizer.base_optimizer if is_sam_active_runtime else optimizer
        for i, param_group in enumerate(opt_to_adjust.param_groups):
            param_group['lr'] = new_lr; param_group['weight_decay'] = new_wd
    def _adjust_sam_parameters(self, sam_optimizer, is_overfitting):
        if not is_overfitting: return
        current_rho = sam_optimizer.rho
        if is_overfitting and current_rho < 0.1:
            new_rho = min(current_rho * 1.2, 0.1)
            if new_rho != current_rho: sam_optimizer.rho = new_rho; self.adjustment_log.append(f"Inc SAM rho to {new_rho:.4f}")
    def get_status_report(self):
        status = [f"Best val_loss: {self.best_val_loss:.4f}, Best val_acc: {self.best_val_acc:.4f}", f"LR: {self.lr_history[-1] if self.lr_history else 'N/A'}, WD: {self.wd_history[-1] if self.wd_history else 'N/A'}"]
        if self.adjustment_log: status.append("Recent adjustments:"); status.extend([f"  - {adj}" for adj in self.adjustment_log[-3:]])
        return "\n".join(status)

# --- Knowledge Distillation ---
class DistillationLoss(nn.Module):
    def __init__(self, alpha=KD_ALPHA, temperature=KD_TEMPERATURE, base_criterion=nn.CrossEntropyLoss()):
        super().__init__(); self.alpha = alpha; self.T = temperature; self.base_criterion = base_criterion; self.KLDiv = nn.KLDivLoss(reduction='batchmean')
    def forward(self, student_outputs, teacher_outputs, labels):
        soft_teacher_log_probs = F.log_softmax(teacher_outputs / self.T, dim=1); soft_student_log_probs = F.log_softmax(student_outputs / self.T, dim=1)
        distillation_loss = self.KLDiv(soft_student_log_probs, soft_teacher_log_probs) * (self.alpha * self.T * self.T)
        student_loss = self.base_criterion(student_outputs, labels); total_loss = distillation_loss + (1. - self.alpha) * student_loss
        return total_loss

def train_student_model(teacher_model_path, student_model_name, student_save_path, df_train, df_val, image_dir, num_classes, is_teacher_mlp=False, teacher_input_size=None):
    print(f"\n{TermColors.HEADER}--- Knowledge Distillation ---{TermColors.ENDC}")
    print(f"Teacher: {teacher_model_path}, Student: {student_model_name}, Teacher MLP: {is_teacher_mlp}")
    
    # Load Teacher Model
    try:
        teacher_model = None
        if is_teacher_mlp:
            if teacher_input_size is None: raise ValueError("teacher_input_size must be provided for MLP teacher.")
            teacher_model = build_mlp_model(input_size=teacher_input_size, num_classes=num_classes)
            teacher_model.load_state_dict(torch.load(teacher_model_path, map_location=DEVICE)) # Assuming state_dict saved directly
        else: # CombinedModel teacher
            teacher_ckpt = torch.load(teacher_model_path, map_location=DEVICE)
            teacher_model_names_cfg = teacher_ckpt.get('model_config', {}).get('model_names', MODEL_NAMES)
            teacher_embedding_size_cfg = teacher_ckpt.get('model_config', {}).get('embedding_size', EMBEDDING_SIZE)
            teacher_arcface_m_cfg = teacher_ckpt.get('model_config', {}).get('arcface_m', ARCFACE_M)
            teacher_model = build_model(model_names=teacher_model_names_cfg, num_classes=num_classes, embedding_size=teacher_embedding_size_cfg, arcface_m=teacher_arcface_m_cfg)
            state_dict = teacher_ckpt['state_dict']; new_state_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            teacher_model.load_state_dict(new_state_dict, strict=False)
        
        teacher_model = teacher_model.to(DEVICE); teacher_model.eval()
        print(f"{TermColors.GREEN}✅ Teacher model loaded.{TermColors.ENDC}")
    except Exception as e: print(f"{TermColors.RED}❌ Failed to load teacher: {e}. Skip KD.{TermColors.ENDC}"); traceback.print_exc(); return

    # Build Student Model (always image-based for this example)
    try:
        student_model = build_model(model_names=[student_model_name], num_classes=num_classes, pretrained=True, dropout_rate=KD_STUDENT_DROPOUT, embedding_size=KD_STUDENT_EMBEDDING_SIZE, metric_learning='None')
        student_model = student_model.to(DEVICE)
    except Exception as e: print(f"{TermColors.RED}❌ Failed to build student: {e}. Skip KD.{TermColors.ENDC}"); traceback.print_exc(); return

    # Dataloaders (always image-based for student)
    try:
        train_tf_kd, val_tf_kd = get_transforms(image_size=KD_STUDENT_IMAGE_SIZE)
        train_ds_kd = PlantDataset(df_train, image_dir, train_tf_kd, None, False, KD_STUDENT_IMAGE_SIZE)
        val_ds_kd = PlantDataset(df_val, image_dir, val_tf_kd, None, False, KD_STUDENT_IMAGE_SIZE)
        if not train_ds_kd or not val_ds_kd or len(train_ds_kd) == 0 or len(val_ds_kd) == 0:
            print(f"{TermColors.RED}❌ KD Dataset empty. Skip KD.{TermColors.ENDC}"); return
        train_loader_kd = DataLoader(train_ds_kd, KD_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
        val_loader_kd = DataLoader(val_ds_kd, KD_BATCH_SIZE*2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        print(f"{TermColors.GREEN}✅ KD Dataloaders ready.{TermColors.ENDC}")
    except Exception as e: print(f"{TermColors.RED}❌ Error KD dataloaders: {e}. Skip KD.{TermColors.ENDC}"); traceback.print_exc(); return

    criterion_kd = DistillationLoss(); optimizer_kd = optim.AdamW(student_model.parameters(), lr=KD_LR, weight_decay=WEIGHT_DECAY)
    scaler_kd = torch.amp.GradScaler('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda'))
    scheduler_kd = CosineAnnealingLR(optimizer_kd, T_max=KD_EPOCHS, eta_min=KD_LR * 0.01)
    print(f"{TermColors.CYAN}ℹ Starting KD Training ({KD_EPOCHS} epochs)...{TermColors.ENDC}"); best_kd_val_acc = 0.0
    
    for epoch in range(KD_EPOCHS):
        if stop_requested: break; print(f"\n--- KD Epoch {epoch+1}/{KD_EPOCHS} ---"); student_model.train(); running_loss_kd = 0.0; total_samples_kd = 0
        progress_bar_kd = tqdm(train_loader_kd, desc=f"KD Train Epoch {epoch+1}", leave=False)
        for batch_data in progress_bar_kd:
            if len(batch_data) != 2: continue; inputs, labels = batch_data; inputs, labels = inputs.to(DEVICE), labels.to(DEVICE); batch_size = inputs.size(0)
            
            with torch.no_grad(): 
                # Teacher forward pass depends on if it's MLP or CombinedModel
                if is_teacher_mlp:
                    # If teacher is MLP, it expects features. This KD setup assumes student is image-based.
                    # This part needs careful design: how to get features for MLP teacher if student gets images?
                    # For now, this example assumes teacher is image-based if student is image-based.
                    # If teacher is MLP, this KD setup would need features as input, not images.
                    # This implies the student would also need to be feature-based or a separate feature extractor is used.
                    # For simplicity, this example will assume teacher is image-based if student is.
                    # A more robust KD would handle teacher_model(features_from_inputs) if teacher is MLP.
                    # This current setup is more for CombinedModel as teacher.
                    teacher_metric_learning = METRIC_LEARNING_TYPE if not is_teacher_mlp else ('ArcFace' if MLP_USE_ARCFACE else 'None')
                    teacher_outputs = teacher_model(inputs, labels=labels if teacher_metric_learning == 'ArcFace' else None)
                else: # CombinedModel teacher
                     teacher_outputs = teacher_model(inputs, labels=labels if METRIC_LEARNING_TYPE == 'ArcFace' else None)


            with torch.amp.autocast('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda')):
                student_outputs = student_model(inputs) # Student (image-based)
                loss = criterion_kd(student_outputs, teacher_outputs, labels)
            optimizer_kd.zero_grad(); scaler_kd.scale(loss).backward(); scaler_kd.step(optimizer_kd); scaler_kd.update()
            if not torch.isnan(loss): running_loss_kd += loss.item() * batch_size; total_samples_kd += batch_size; progress_bar_kd.set_postfix(loss=f"{loss.item():.4f}")
            else: print(f"{TermColors.RED}Warn: NaN/Inf loss in KD train.{TermColors.ENDC}")
            if stop_requested: break
        epoch_loss_kd = running_loss_kd / total_samples_kd if total_samples_kd > 0 else 0; print(f"KD Epoch {epoch+1} Train Loss: {epoch_loss_kd:.4f}")
        
        student_model.eval(); running_val_loss_kd = 0.0; total_val_samples_kd = 0; all_preds_kd, all_labels_kd = [], []
        with torch.no_grad():
            for batch_data in tqdm(val_loader_kd, desc=f"KD Validate Epoch {epoch+1}", leave=False):
                if len(batch_data) != 2: continue; inputs, labels = batch_data; inputs, labels = inputs.to(DEVICE), labels.to(DEVICE); batch_size = inputs.size(0)
                with torch.amp.autocast('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda')): outputs = student_model(inputs); val_loss = F.cross_entropy(outputs, labels)
                if not torch.isnan(val_loss): running_val_loss_kd += val_loss.item()*batch_size; total_val_samples_kd += batch_size; all_preds_kd.append(torch.argmax(outputs,dim=1).cpu()); all_labels_kd.append(labels.cpu())
        epoch_val_loss_kd = running_val_loss_kd / total_val_samples_kd if total_val_samples_kd > 0 else 0; epoch_val_acc_kd = 0.0
        if all_preds_kd: preds_tensor = torch.cat(all_preds_kd); labels_tensor = torch.cat(all_labels_kd); epoch_val_acc_kd = (preds_tensor == labels_tensor).sum().item() / total_val_samples_kd if total_val_samples_kd > 0 else 0.0
        print(f"KD Epoch {epoch+1} Val Loss: {epoch_val_loss_kd:.4f}, Val Acc: {epoch_val_acc_kd:.4f}")
        if epoch_val_acc_kd > best_kd_val_acc: best_kd_val_acc = epoch_val_acc_kd; print(f"{TermColors.OKGREEN}🏆 New best KD val acc: {best_kd_val_acc:.4f}. Saving...{TermColors.ENDC}"); torch.save(student_model.state_dict(), student_save_path)
        scheduler_kd.step()
    print(f"{TermColors.OKGREEN}✅ KD finished. Best student val_acc: {best_kd_val_acc:.4f}{TermColors.ENDC}")
    del teacher_model, student_model, train_loader_kd, val_loader_kd; gc.collect(); torch.cuda.empty_cache()


# --- Feature Extraction Function ---
def extract_all_features(df_full, num_classes_val, label_encoder_val):
    global NUM_CLASSES, label_encoder, class_names, CURRENT_IMAGE_SIZE, CLASS_FREQUENCIES, CLASS_PRIORS, CLASS_WEIGHTS
    NUM_CLASSES = num_classes_val
    label_encoder = label_encoder_val
    class_names = list(label_encoder.classes_)

    print(f"\n{TermColors.HEADER}--- STEP: Feature Extraction ---{TermColors.ENDC}")
    print(f"Extracting features for {len(df_full)} images at size {FEATURE_EXTRACTION_IMAGE_SIZE_CONFIG}.")
    print(f"Features will be saved to: {FEATURES_NPZ_PATH}")

    CURRENT_IMAGE_SIZE = FEATURE_EXTRACTION_IMAGE_SIZE_CONFIG
    feature_extractor_model = build_model(
        model_names=MODEL_NAMES, num_classes=NUM_CLASSES, pretrained=PRETRAINED,
        dropout_rate=0, embedding_size=EMBEDDING_SIZE, # Dropout not needed for feature extraction
        drop_path_rate=DROP_PATH_RATE, global_pool=GLOBAL_POOLING,
        metric_learning='None' # We want raw embeddings
    )
    
    if FEATURE_EXTRACTOR_CHECKPOINT_PATH and os.path.exists(FEATURE_EXTRACTOR_CHECKPOINT_PATH):
        print(f"{TermColors.CYAN}ℹ Loading feature extractor checkpoint from: {FEATURE_EXTRACTOR_CHECKPOINT_PATH}{TermColors.ENDC}")
        try:
            ckpt = torch.load(FEATURE_EXTRACTOR_CHECKPOINT_PATH, map_location=DEVICE)
            state_dict = ckpt['state_dict']
            new_state_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            feature_extractor_model.load_state_dict(new_state_dict, strict=False)
            print(f"{TermColors.GREEN}✅ Feature extractor checkpoint loaded.{TermColors.ENDC}")
        except Exception as e:
            print(f"{TermColors.RED}❌ Failed to load feature extractor checkpoint: {e}. Using pretrained weights.{TermColors.ENDC}")
            traceback.print_exc()
    else:
        print(f"{TermColors.INFO}ℹ Using {'pretrained' if PRETRAINED else 'randomly initialized'} weights for feature extractor.{TermColors.ENDC}")

    feature_extractor_model = feature_extractor_model.to(DEVICE)
    feature_extractor_model.eval()

    # Use minimal transforms for feature extraction
    extraction_transform = get_transforms(image_size=FEATURE_EXTRACTION_IMAGE_SIZE_CONFIG, for_feature_extraction=True)
    
    # Create a dataset for all images
    full_plant_dataset = PlantDataset(df_full, IMAGE_DIR, extraction_transform, label_encoder, include_paths=True, image_size=FEATURE_EXTRACTION_IMAGE_SIZE_CONFIG)
    if len(full_plant_dataset) == 0:
        print(f"{TermColors.RED}❌ Dataset for feature extraction is empty. Aborting feature extraction.{TermColors.ENDC}")
        return

    dataloader = DataLoader(full_plant_dataset, batch_size=FEATURE_EXTRACTOR_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    all_features_list = []
    all_labels_list = []
    all_original_indices_list = []

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Extracting Features"):
            if len(batch_data) != 4: continue # images, labels, paths, original_indices
            inputs, labels_batch, _, original_indices_batch = batch_data
            inputs = inputs.to(DEVICE)
            
            with torch.amp.autocast('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda')):
                # Get embeddings from the CombinedModel
                embeddings = feature_extractor_model(inputs, return_embedding=True) 
            
            all_features_list.append(embeddings.detach().cpu().numpy())
            all_labels_list.append(labels_batch.cpu().numpy())
            all_original_indices_list.append(original_indices_batch.cpu().numpy())

    if not all_features_list:
        print(f"{TermColors.RED}❌ No features were extracted. Aborting.{TermColors.ENDC}")
        return

    all_features_np = np.concatenate(all_features_list, axis=0)
    all_labels_np = np.concatenate(all_labels_list, axis=0)
    all_original_indices_np = np.concatenate(all_original_indices_list, axis=0)

    print(f"{TermColors.GREEN}✅ Feature extraction complete. Extracted {all_features_np.shape[0]} features with dimension {all_features_np.shape[1]}.{TermColors.ENDC}")
    
    try:
        np.savez_compressed(FEATURES_NPZ_PATH, 
                            features=all_features_np, 
                            labels=all_labels_np, 
                            original_indices=all_original_indices_np,
                            label_encoder_classes=np.array(list(label_encoder.classes_))) # Save LE classes
        print(f"{TermColors.GREEN}✅ Features saved to {FEATURES_NPZ_PATH}{TermColors.ENDC}")
    except Exception as e:
        print(f"{TermColors.RED}❌ Error saving features: {e}{TermColors.ENDC}")
        traceback.print_exc()
    
    del feature_extractor_model, dataloader, full_plant_dataset, all_features_list, all_labels_list, all_original_indices_list
    gc.collect()
    torch.cuda.empty_cache()


# --- Main MLP Training Loop ---
def train_mlp_on_features_main_loop(num_classes_val, label_encoder_val, class_names_val, class_frequencies_val, class_priors_val, class_weights_val):
    global NUM_CLASSES, label_encoder, class_names, CLASS_FREQUENCIES, CLASS_PRIORS, CLASS_WEIGHTS
    NUM_CLASSES = num_classes_val
    label_encoder = label_encoder_val
    class_names = class_names_val
    CLASS_FREQUENCIES = class_frequencies_val
    CLASS_PRIORS = class_priors_val
    CLASS_WEIGHTS = class_weights_val

    print(f"\n{TermColors.HEADER}--- STEP: Train MLP on Extracted Features ---{TermColors.ENDC}")
    if not os.path.exists(FEATURES_NPZ_PATH):
        print(f"{TermColors.RED}❌ Features file not found: {FEATURES_NPZ_PATH}. Please run in 'EXTRACT_FEATURES' mode first.{TermColors.ENDC}")
        return
    
    try:
        print(f"{TermColors.CYAN}ℹ Loading features from {FEATURES_NPZ_PATH}...{TermColors.ENDC}")
        data = np.load(FEATURES_NPZ_PATH, allow_pickle=True)
        features_all = data['features']
        labels_all = data['labels']
        original_indices_all = data['original_indices']
        loaded_le_classes = data['label_encoder_classes']
        
        if not np.array_equal(list(label_encoder.classes_), loaded_le_classes):
            print(f"{TermColors.RED}❌ Label encoder mismatch between current run and saved features! Aborting MLP training.{TermColors.ENDC}")
            print(f"  Current LE: {list(label_encoder.classes_)}")
            print(f"  Features LE: {list(loaded_le_classes)}")
            return
        print(f"{TermColors.GREEN}✅ Features loaded: {features_all.shape[0]} samples.{TermColors.ENDC}")
    except Exception as e:
        print(f"{TermColors.RED}❌ Error loading features: {e}{TermColors.ENDC}")
        traceback.print_exc()
        return

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_results = defaultdict(list)
    # OOF for MLP will be based on original data indices
    oof_preds_array = np.full((len(original_indices_all), NUM_CLASSES), np.nan, dtype=np.float32) # Size of original dataset
    oof_labels_array = np.full(len(original_indices_all), -1, dtype=np.int32)


    for fold, (train_feature_idx, val_feature_idx) in enumerate(skf.split(features_all, labels_all)):
        if stop_requested: break
        print(f"\n{TermColors.HEADER}===== Starting MLP Training Fold {fold+1}/{N_FOLDS} ====={TermColors.ENDC}")
        
        train_features, val_features = features_all[train_feature_idx], features_all[val_feature_idx]
        train_labels, val_labels = labels_all[train_feature_idx], labels_all[val_feature_idx]
        train_original_indices, val_original_indices = original_indices_all[train_feature_idx], original_indices_all[val_feature_idx]

        print(f"MLP Fold {fold+1} - Train Features: {len(train_features)}, Val Features: {len(val_features)}")

        mlp_model = optimizer = scheduler = scaler = criterion = swa_mlp_model = ema_mlp_model = None
        gc.collect(); torch.cuda.empty_cache()

        try:
            mlp_model = build_mlp_model(input_size=features_all.shape[1], num_classes=NUM_CLASSES)
            mlp_model = mlp_model.to(DEVICE)
            # torch.compile for MLP might be overkill but can be tried
            if USE_TORCH_COMPILE and hasattr(torch, 'compile') and int(torch.__version__.split('.')[0]) >= 2:
                try: mlp_model = torch.compile(mlp_model, mode='default'); print(f"{TermColors.GREEN}✅ torch.compile() applied to MLP.{TermColors.ENDC}")
                except Exception as compile_e: print(f"{TermColors.RED}❌ torch.compile() for MLP failed: {compile_e}.{TermColors.ENDC}")

            criterion = get_criterion(class_weights=CLASS_WEIGHTS) # Can use same criterion logic
            optimizer = get_optimizer(mlp_model, lr=MLP_LEARNING_RATE, weight_decay=MLP_WEIGHT_DECAY, optimizer_type=MLP_OPTIMIZER_TYPE, is_mlp_optimizer=True)
            scaler = torch.amp.GradScaler('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda'))
            
            if MLP_USE_SWA:
                swa_mlp_model = AveragedModel(mlp_model, avg_fn=ema_avg_fn if MLP_USE_EMA else None)
                if MLP_USE_EMA and hasattr(swa_mlp_model, '_is_mlp_ema'): swa_mlp_model._is_mlp_ema = True # Mark for decay rate
            if MLP_USE_EMA:
                ema_mlp_model = AveragedModel(mlp_model, avg_fn=ema_avg_fn)
                if hasattr(ema_mlp_model, '_is_mlp_ema'): ema_mlp_model._is_mlp_ema = True # Mark for decay rate


        except Exception as e: print(f"{TermColors.RED}❌ MLP Fold {fold+1} Setup Error: {e}{TermColors.ENDC}"); traceback.print_exc(); continue

        start_mlp_epoch, best_metric, _ = load_checkpoint(fold, mlp_model, optimizer, None, scaler, filename="latest_mlp_checkpoint.pth.tar", is_mlp_checkpoint=True)
        initial_best_val = float('-inf') if CHECKPOINT_MODE == 'max' else float('inf')
        if best_metric == initial_best_val:
             start_mlp_epoch, best_metric, _ = load_checkpoint(fold, mlp_model, optimizer, None, scaler, filename="best_mlp_model.pth.tar", is_mlp_checkpoint=True)


        fold_log_dir = os.path.join(BASE_LOG_DIR, f"mlp_fold_{fold}"); writer = SummaryWriter(log_dir=fold_log_dir)
        fold_stop_requested = False
        fold_best_val_loss = float('inf'); fold_best_val_acc = 0.0
        epochs_without_improvement = 0
        best_metric_for_early_stopping = best_metric

        train_feature_ds = FeatureDataset(train_features, train_labels, train_original_indices)
        val_feature_ds = FeatureDataset(val_features, val_labels, val_original_indices)

        sampler = None
        if IMBALANCE_STRATEGY == 'WeightedSampler' and CLASS_WEIGHTS is not None:
            labels_list = train_feature_ds.get_labels()
            if labels_list:
                class_sample_count = np.array([labels_list.count(l) for l in range(NUM_CLASSES)]); class_sample_count = np.maximum(class_sample_count, 1)
                weight = 1. / class_sample_count; samples_weight = torch.from_numpy(np.array([weight[t] for t in labels_list])).double()
                sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        
        train_loader = DataLoader(train_feature_ds, MLP_BATCH_SIZE, sampler=sampler, shuffle=(sampler is None), num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_feature_ds, MLP_BATCH_SIZE*2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        err_loader = DataLoader(val_feature_ds, ERROR_LOG_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS) # For error logging on val features

        print(f"{TermColors.GREEN}✅ MLP Fold {fold+1} Feature Dataloaders ready (Train: {len(train_loader)}, Val: {len(val_loader)}).{TermColors.ENDC}")
        
        scheduler = get_scheduler(optimizer, scheduler_type=MLP_SCHEDULER_TYPE, total_epochs=MLP_EPOCHS, plateau_monitor=CHECKPOINT_MONITOR, plateau_mode=CHECKPOINT_MODE, is_mlp_scheduler=True)
        
        if start_mlp_epoch > 0: # Reload scheduler state if resuming
             ckpt_path_sched = os.path.join(BASE_CHECKPOINT_DIR, f"mlp_fold_{fold}", "latest_mlp_checkpoint.pth.tar")
             if scheduler and os.path.isfile(ckpt_path_sched):
                 ckpt_sched = torch.load(ckpt_path_sched, map_location=DEVICE)
                 if 'scheduler' in ckpt_sched and ckpt_sched['scheduler']:
                     try: scheduler.load_state_dict(ckpt_sched['scheduler'])
                     except Exception as e_sch: print(f"{TermColors.YELLOW}⚠️ MLP Scheduler reload failed: {e_sch}.{TermColors.ENDC}")

        swa_scheduler_mlp = None
        if MLP_USE_SWA and swa_mlp_model:
            swa_scheduler_mlp = SWALR(optimizer, swa_lr=(MLP_LEARNING_RATE * SWA_LR_FACTOR), anneal_epochs=SWA_ANNEAL_EPOCHS, anneal_strategy='cos')
        
        mlp_swa_start_epoch = int(MLP_EPOCHS * MLP_SWA_START_EPOCH_FACTOR)

        for epoch in range(start_mlp_epoch, MLP_EPOCHS):
            if fold_stop_requested or stop_requested: break
            print(f"\n{TermColors.CYAN}--- MLP Fold {fold+1} Epoch {epoch+1}/{MLP_EPOCHS} ---{TermColors.ENDC}")
            
            # Pass MLP_EPOCHS as stage_total_epochs for train_one_epoch's SWA logic
            train_loss, train_acc = train_one_epoch(
                mlp_model, train_loader, criterion, optimizer, scaler, scheduler, 
                epoch, stage_idx=0, stage_epoch=epoch, stage_total_epochs=MLP_EPOCHS, 
                device=DEVICE, writer=writer, num_classes=NUM_CLASSES, ema_model=ema_mlp_model,
                use_sam_flag=False, fold_num=fold+1, is_mlp_training=True
            )
            if train_loss is None: fold_stop_requested = True; break
            
            val_loss, val_acc, current_oof_preds, current_oof_indices = validate_one_epoch(
                mlp_model, val_loader, criterion, DEVICE, epoch, writer, NUM_CLASSES, 
                scheduler=scheduler, swa_model=swa_mlp_model, ema_model=ema_mlp_model, 
                return_preds=True, fold_num=fold+1, is_mlp_validation=True
            )
            if val_loss is None: fold_stop_requested = True; break

            print(f"MLP Fold {fold+1} Ep {epoch+1}: Train L={train_loss:.4f} A={train_acc:.4f} | Val L={val_loss:.4f} A={val_acc:.4f}")

            if MLP_USE_SWA and swa_mlp_model and epoch >= mlp_swa_start_epoch:
                swa_mlp_model.update_parameters(mlp_model)
                if swa_scheduler_mlp: swa_scheduler_mlp.step()
            
            current_metric_for_stopping = val_loss if CHECKPOINT_MONITOR == 'val_loss' else val_acc
            improved_early_stop_metric = False
            if CHECKPOINT_MODE == 'min':
                if current_metric_for_stopping < best_metric_for_early_stopping: best_metric_for_early_stopping = current_metric_for_stopping; improved_early_stop_metric = True
            else:
                if current_metric_for_stopping > best_metric_for_early_stopping: best_metric_for_early_stopping = current_metric_for_stopping; improved_early_stop_metric = True
            
            if improved_early_stop_metric: epochs_without_improvement = 0
            else: epochs_without_improvement += 1
            
            if epochs_without_improvement >= MLP_EARLY_STOPPING_PATIENCE:
                print(f"{TermColors.WARNING}⚠️ Early stopping for MLP Fold {fold+1} after {epochs_without_improvement} epochs on {CHECKPOINT_MONITOR}.{TermColors.ENDC}")
                fold_stop_requested = True; break

            current_metric_for_checkpoint = val_acc if CHECKPOINT_MONITOR == 'val_acc' else val_loss
            is_best = False
            if (CHECKPOINT_MODE == 'max' and current_metric_for_checkpoint > best_metric) or \
               (CHECKPOINT_MODE == 'min' and current_metric_for_checkpoint < best_metric):
                best_metric = current_metric_for_checkpoint; is_best = True; fold_best_val_loss = val_loss; fold_best_val_acc = val_acc
                print(f"{TermColors.OKGREEN}🏆 MLP Fold {fold+1} New Best {CHECKPOINT_MONITOR}: {best_metric:.4f}. Saving...{TermColors.ENDC}")
                save_checkpoint(fold, epoch + 1, -1, epoch + 1, mlp_model, optimizer, scheduler, scaler, best_metric, "best_mlp_model.pth.tar", is_mlp_checkpoint=True)
                save_model(fold, mlp_model, "best_mlp_model_state_dict.pth", is_mlp_model=True)
                if MLP_USE_EMA and ema_mlp_model: save_model(fold, ema_mlp_model, "best_ema_mlp_model_state_dict.pth", is_mlp_model=True)
                if MLP_USE_SWA and swa_mlp_model and epoch >= mlp_swa_start_epoch: save_model(fold, swa_mlp_model, "best_swa_mlp_model_state_dict.pth", is_mlp_model=True)
                
                if current_oof_preds is not None and current_oof_indices is not None:
                    # Map current_oof_indices (which are indices within the val_feature_ds) back to original dataset indices
                    for i, val_ds_idx in enumerate(current_oof_indices): # val_ds_idx is the index from val_feature_ds
                        original_data_idx = val_original_indices[val_ds_idx] # Get the true original index
                        if 0 <= original_data_idx < len(oof_preds_array):
                             oof_preds_array[original_data_idx] = current_oof_preds[i]
                             oof_labels_array[original_data_idx] = labels_all[original_data_idx] # Use labels_all with original_data_idx

            save_checkpoint(fold, epoch + 1, -1, epoch + 1, mlp_model, optimizer, scheduler, scaler, best_metric, "latest_mlp_checkpoint.pth.tar", is_mlp_checkpoint=True)
            if LOG_MISCLASSIFIED_IMAGES and ((epoch + 1) % 5 == 0 or is_best):
                log_misclassified(fold, mlp_model, err_loader, criterion, DEVICE, epoch + 1, writer, NUM_CLASSES, is_mlp_logging=True)
            
            gc.collect(); torch.cuda.empty_cache()
        
        if fold_stop_requested or stop_requested:
            print(f"{TermColors.WARNING}MLP Fold {fold+1} interrupted.{TermColors.ENDC}")
            save_checkpoint(fold, epoch, -1, epoch, mlp_model, optimizer, scheduler, scaler, best_metric, "interrupted_mlp_checkpoint.pth.tar", is_mlp_checkpoint=True)
            break # Break from K-Fold loop if interrupted

        if MLP_USE_SWA and swa_mlp_model and epoch >= mlp_swa_start_epoch: # epoch here is the last completed epoch
            print(f"{TermColors.CYAN}ℹ MLP Fold {fold+1} Updating SWA BN stats...{TermColors.ENDC}")
            # BN update for SWA MLP needs features, not images
            bn_loader_mlp = DataLoader(train_feature_ds, MLP_BATCH_SIZE*2, shuffle=True, num_workers=NUM_WORKERS) # Use train features
            try:
                torch.optim.swa_utils.update_bn(bn_loader_mlp, swa_mlp_model, device=DEVICE)
                print(f"{TermColors.GREEN}✅ MLP SWA BN updated.{TermColors.ENDC}")
                
                # Evaluate SWA MLP model
                swa_val_loss, swa_val_acc, _, _ = validate_one_epoch(swa_mlp_model, val_loader, criterion, DEVICE, epoch, writer, NUM_CLASSES, fold_num=f"{fold+1}-SWA-MLP", is_mlp_validation=True)
                print(f"MLP Fold {fold+1} Final SWA Val L={swa_val_loss:.4f}, A={swa_val_acc:.4f}")
                fold_results['mlp_swa_acc'].append(swa_val_acc); fold_results['mlp_swa_loss'].append(swa_val_loss)
                save_model(fold, swa_mlp_model, "final_swa_mlp_model_state_dict.pth", is_mlp_model=True)
            except Exception as e_swa_mlp: print(f"{TermColors.RED}❌ MLP SWA BN/Eval Error: {e_swa_mlp}{TermColors.ENDC}"); traceback.print_exc()
            del bn_loader_mlp

        save_model(fold, mlp_model, "final_mlp_model_state_dict.pth", is_mlp_model=True)
        if MLP_USE_EMA and ema_mlp_model: save_model(fold, ema_mlp_model, "final_ema_mlp_model_state_dict.pth", is_mlp_model=True)
        
        writer.close()
        fold_results['mlp_best_metric'].append(best_metric)
        fold_results['mlp_best_val_loss'].append(fold_best_val_loss)
        fold_results['mlp_best_val_acc'].append(fold_best_val_acc)
        del mlp_model, optimizer, scheduler, scaler, criterion, swa_mlp_model, ema_mlp_model, train_loader, val_loader, err_loader, train_feature_ds, val_feature_ds
        gc.collect(); torch.cuda.empty_cache()

    print(f"\n{TermColors.HEADER}===== MLP Cross-Validation Finished ====={TermColors.ENDC}")
    if not stop_requested:
        avg_best_acc = np.mean(fold_results['mlp_best_val_acc']) if fold_results['mlp_best_val_acc'] else 0.0
        std_best_acc = np.std(fold_results['mlp_best_val_acc']) if fold_results['mlp_best_val_acc'] else 0.0
        print(f"Avg MLP Best Val Acc: {avg_best_acc:.4f} +/- {std_best_acc:.4f}")
        if fold_results['mlp_swa_acc']: print(f"Avg MLP Final SWA Val Acc: {np.mean(fold_results['mlp_swa_acc']):.4f} +/- {np.std(fold_results['mlp_swa_acc']):.4f}")

        if RUN_STACKING: # Stacking on MLP OOF predictions
            valid_oof_indices = np.where(oof_labels_array != -1)[0]
            if len(valid_oof_indices) < len(original_indices_all) * 0.5: print(f"{TermColors.YELLOW}Warning: MLP OOF collected for only {len(valid_oof_indices)}/{len(original_indices_all)} samples. Stacking may be suboptimal.{TermColors.ENDC}")
            if len(valid_oof_indices) > 0:
                final_oof_preds = oof_preds_array[valid_oof_indices]; final_oof_labels = oof_labels_array[valid_oof_indices]
                if len(final_oof_preds) > 0 and len(final_oof_preds) == len(final_oof_labels):
                    mlp_oof_path = FEATURES_NPZ_PATH.replace(".npz", "_mlp_oof.npz")
                    np.savez_compressed(mlp_oof_path, preds=final_oof_preds, labels=final_oof_labels); print(f"MLP OOF preds saved: {mlp_oof_path}")
                    mlp_stacking_model_path = STACKING_META_MODEL_PATH.replace(".joblib", "_mlp.joblib")
                    train_stacking_meta_model(final_oof_preds, final_oof_labels, mlp_stacking_model_path)
                else: print(f"{TermColors.RED}Error preparing MLP stacking data. Skipping.{TermColors.ENDC}")
                del final_oof_preds, final_oof_labels
            else: print(f"{TermColors.RED}No valid MLP OOF predictions for Stacking. Skipping.{TermColors.ENDC}")
    else:
        print(f"{TermColors.YELLOW}MLP Training interrupted. Stacking/KD skipped.{TermColors.ENDC}")
        # Ensure these are globally false if MLP training was interrupted before completion
        global RUN_STACKING_AFTER_MLP, RUN_KD_AFTER_MLP 
        RUN_STACKING_AFTER_MLP = False # Placeholder for specific control
        RUN_KD_AFTER_MLP = False      # Placeholder for specific control

    del features_all, labels_all, original_indices_all, oof_preds_array, oof_labels_array
    gc.collect()


# --- Main Execution ---
def main_full_training_loop(df_full, num_classes_val, label_encoder_val, class_names_val, class_frequencies_val, class_priors_val, class_weights_val):
    global NUM_CLASSES, label_encoder, class_names, CLASS_FREQUENCIES, CLASS_PRIORS, CLASS_WEIGHTS, CURRENT_IMAGE_SIZE
    NUM_CLASSES = num_classes_val; label_encoder = label_encoder_val; class_names = class_names_val
    CLASS_FREQUENCIES = class_frequencies_val; CLASS_PRIORS = class_priors_val; CLASS_WEIGHTS = class_weights_val
    
    print(f"\n{TermColors.HEADER}--- STEP 3: Main K-Fold Cross-Validation ({N_FOLDS} Folds) ---{TermColors.ENDC}")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_results = defaultdict(list)
    oof_preds_array = np.full((len(df_full), NUM_CLASSES), np.nan, dtype=np.float32)
    oof_labels_array = np.full(len(df_full), -1, dtype=np.int32)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_full, df_full['label'])):
        if stop_requested: break
        print(f"\n{TermColors.HEADER}===== Starting Fold {fold+1}/{N_FOLDS} ====={TermColors.ENDC}")
        train_df = df_full.iloc[train_idx]; val_df = df_full.iloc[val_idx]
        print(f"Fold {fold+1} - Train: {len(train_df)}, Val: {len(val_df)}")
        model = optimizer = scheduler = scaler = criterion = swa_model = ema_model = None
        gc.collect(); torch.cuda.empty_cache()
        try:
            model = build_model(model_names=MODEL_NAMES, num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE, arcface_m=ARCFACE_M, embedding_size=EMBEDDING_SIZE)
            model = model.to(DEVICE)
            if USE_TORCH_COMPILE and hasattr(torch, 'compile') and int(torch.__version__.split('.')[0]) >= 2:
                try: model = torch.compile(model, mode='default'); print(f"{TermColors.GREEN}✅ torch.compile() applied.{TermColors.ENDC}")
                except Exception as compile_e: print(f"{TermColors.RED}❌ torch.compile() failed: {compile_e}.{TermColors.ENDC}")
            criterion = get_criterion(class_weights=CLASS_WEIGHTS)
            optimizer = get_optimizer(model, optimizer_type=OPTIMIZER_TYPE, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, use_sam_flag=USE_SAM, sam_rho=SAM_RHO, sam_adaptive=SAM_ADAPTIVE)
            scaler = torch.amp.GradScaler('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda'))
            if USE_SWA: 
                swa_model = AveragedModel(model, avg_fn=ema_avg_fn if USE_EMA else None)
                if USE_EMA and hasattr(swa_model, '_is_mlp_ema'): swa_model._is_mlp_ema = False # Ensure it uses CombinedModel EMA decay
            if USE_EMA: 
                ema_model = AveragedModel(model, avg_fn=ema_avg_fn)
                if hasattr(ema_model, '_is_mlp_ema'): ema_model._is_mlp_ema = False # Ensure it uses CombinedModel EMA decay

        except Exception as e: print(f"{TermColors.RED}❌ Fold {fold+1} Setup Error: {e}{TermColors.ENDC}"); traceback.print_exc(); continue
        
        start_glob_ep, start_stg_idx, start_stg_ep, best_metric, _, loaded_size, _ = load_checkpoint(fold, model, optimizer, None, scaler, filename="latest_checkpoint.pth.tar")
        initial_best_val = float('-inf') if CHECKPOINT_MODE == 'max' else float('inf')
        if best_metric == initial_best_val:
            start_glob_ep, start_stg_idx, start_stg_ep, best_metric, _, loaded_size, _ = load_checkpoint(fold, model, optimizer, None, scaler, filename="best_model.pth.tar")

        fold_log_dir = os.path.join(BASE_LOG_DIR, f"fold_{fold}"); writer = SummaryWriter(log_dir=fold_log_dir)
        global_epoch_counter = start_glob_ep; fold_stop_requested = False
        fold_best_val_loss = float('inf'); fold_best_val_acc = 0.0
        epochs_without_improvement = 0 
        best_metric_for_early_stopping = best_metric 

        current_auto_config = AutoTrainingConfig(initial_lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        for stage_idx_loop, (stage_epochs, stage_image_size) in enumerate(PROGRESSIVE_RESIZING_STAGES):
            if fold_stop_requested or stop_requested: break
            if stage_idx_loop < start_stg_idx: continue
            current_stage_start_epoch = start_stg_ep if stage_idx_loop == start_stg_idx else 0
            if current_stage_start_epoch >= stage_epochs: continue
            print(f"\n{TermColors.MAGENTA}===== Fold {fold+1} Stage {stage_idx_loop+1}/{len(PROGRESSIVE_RESIZING_STAGES)}: {stage_epochs} Epochs @ {stage_image_size} ====={TermColors.ENDC}")
            CURRENT_IMAGE_SIZE = stage_image_size
            if loaded_size and stage_idx_loop == start_stg_idx and loaded_size != CURRENT_IMAGE_SIZE: print(f"{TermColors.CRITICAL}🚨 Image size mismatch! Ckpt {loaded_size} != Stage {CURRENT_IMAGE_SIZE}. Exit.{TermColors.ENDC}"); fold_stop_requested = True; break
            
            train_transform, val_transform = get_transforms(image_size=CURRENT_IMAGE_SIZE)
            train_loader, val_loader, err_loader = None, None, None; gc.collect()
            try:
                train_ds = PlantDataset(train_df, IMAGE_DIR, train_transform, None, False, CURRENT_IMAGE_SIZE)
                val_ds = PlantDataset(val_df, IMAGE_DIR, val_transform, None, True, CURRENT_IMAGE_SIZE) # include_paths=True for OOF indices
                err_ds = PlantDataset(val_df, IMAGE_DIR, val_transform, None, True, CURRENT_IMAGE_SIZE) # include_paths=True for error logging
                if not train_ds or not val_ds or len(train_ds) == 0 or len(val_ds) == 0 : print(f"{TermColors.RED}❌ Fold {fold+1} Stage {stage_idx_loop+1}: Dataset empty. Skip fold.{TermColors.ENDC}"); fold_stop_requested = True; break
                sampler = None
                if IMBALANCE_STRATEGY == 'WeightedSampler' and CLASS_WEIGHTS is not None:
                    labels_list = train_ds.get_labels()
                    if labels_list:
                        class_sample_count = np.array([labels_list.count(l) for l in range(NUM_CLASSES)]); class_sample_count = np.maximum(class_sample_count, 1)
                        weight = 1. / class_sample_count; samples_weight = torch.from_numpy(np.array([weight[t] for t in labels_list])).double()
                        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
                train_loader = DataLoader(train_ds, BATCH_SIZE, sampler=sampler, shuffle=(sampler is None), num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
                val_loader = DataLoader(val_ds, BATCH_SIZE*2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
                err_loader = DataLoader(err_ds, ERROR_LOG_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
                print(f"{TermColors.GREEN}✅ Fold {fold+1} Dataloaders ready (Train: {len(train_loader)}, Val: {len(val_loader)}).{TermColors.ENDC}")
            except Exception as e: print(f"{TermColors.RED}❌ Fold {fold+1} Dataloader Error: {e}{TermColors.ENDC}"); traceback.print_exc(); fold_stop_requested = True; break
            
            current_t_0_val = stage_epochs if SCHEDULER_TYPE == 'CosineWarmRestarts' else T_0
            scheduler = get_scheduler(optimizer, total_epochs=stage_epochs, t_0=current_t_0_val, plateau_monitor=CHECKPOINT_MONITOR, plateau_mode=CHECKPOINT_MODE)
            
            if stage_idx_loop == start_stg_idx and current_stage_start_epoch > 0:
                 ckpt_path_sched = os.path.join(BASE_CHECKPOINT_DIR, f"fold_{fold}", "latest_checkpoint.pth.tar")
                 if scheduler and os.path.isfile(ckpt_path_sched):
                     ckpt_sched = torch.load(ckpt_path_sched, map_location=DEVICE)
                     if 'scheduler' in ckpt_sched and ckpt_sched['scheduler']:
                         try: scheduler.load_state_dict(ckpt_sched['scheduler'])
                         except Exception as e_sch: print(f"{TermColors.YELLOW}⚠️ Scheduler reload failed: {e_sch}.{TermColors.ENDC}")
            
            globals()['swa_start_epoch_stage'] = max(0, int(stage_epochs * SWA_START_EPOCH_GLOBAL_FACTOR))
            swa_scheduler = None
            if USE_SWA and swa_model: swa_scheduler = SWALR(optimizer, swa_lr=(LEARNING_RATE * SWA_LR_FACTOR), anneal_epochs=SWA_ANNEAL_EPOCHS, anneal_strategy='cos')

            for stage_epoch_loop in range(current_stage_start_epoch, stage_epochs):
                if fold_stop_requested or stop_requested: break
                print(f"\n{TermColors.CYAN}--- Fold {fold+1} GlobEp {global_epoch_counter+1}/{TOTAL_EPOCHS_PER_FOLD} (Stg {stage_idx_loop+1}: Ep {stage_epoch_loop+1}/{stage_epochs}) ---{TermColors.ENDC}")
                train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, scheduler, global_epoch_counter, stage_idx_loop, stage_epoch_loop, stage_epochs, DEVICE, writer, NUM_CLASSES, ema_model, use_sam_flag=USE_SAM, fold_num=fold+1)
                if train_loss is None: fold_stop_requested = True; break
                
                val_loss, val_acc, current_oof_preds, current_oof_indices = validate_one_epoch(model, val_loader, criterion, DEVICE, global_epoch_counter, writer, NUM_CLASSES, scheduler=scheduler, swa_model=swa_model, ema_model=ema_model, return_preds=True, fold_num=fold+1)
                if val_loss is None: fold_stop_requested = True; break
                
                if USE_AUTO_TRAIN_CONFIG:
                    is_sam_active_runtime = hasattr(optimizer, 'base_optimizer') and SAM_AVAILABLE and USE_SAM
                    current_auto_config.update(train_loss, train_acc, val_loss, val_acc, optimizer, is_sam_active_runtime)
                    if global_epoch_counter % 5 == 0: print(f"\n{TermColors.CYAN}AutoConfig Status:\n{current_auto_config.get_status_report()}{TermColors.ENDC}\n")

                print(f"Fold {fold+1} GlobEp {global_epoch_counter+1}: Train L={train_loss:.4f} A={train_acc:.4f} | Val L={val_loss:.4f} A={val_acc:.4f}")

                if USE_SWA and swa_model and stage_epoch_loop >= globals()['swa_start_epoch_stage']:
                    swa_model.update_parameters(model)
                    if swa_scheduler: swa_scheduler.step()
                
                current_metric_for_stopping = val_loss if CHECKPOINT_MONITOR == 'val_loss' else val_acc
                improved_early_stop_metric = False
                if CHECKPOINT_MODE == 'min':
                    if current_metric_for_stopping < best_metric_for_early_stopping: best_metric_for_early_stopping = current_metric_for_stopping; improved_early_stop_metric = True
                else: 
                    if current_metric_for_stopping > best_metric_for_early_stopping: best_metric_for_early_stopping = current_metric_for_stopping; improved_early_stop_metric = True
                
                if improved_early_stop_metric: epochs_without_improvement = 0
                else: epochs_without_improvement += 1
                
                if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                    print(f"{TermColors.WARNING}⚠️ Early stopping for Fold {fold+1} after {epochs_without_improvement} epochs on {CHECKPOINT_MONITOR}.{TermColors.ENDC}")
                    fold_stop_requested = True; break 

                current_metric_for_checkpoint = val_acc if CHECKPOINT_MONITOR == 'val_acc' else val_loss
                is_best = False
                if (CHECKPOINT_MODE == 'max' and current_metric_for_checkpoint > best_metric) or \
                   (CHECKPOINT_MODE == 'min' and current_metric_for_checkpoint < best_metric):
                    best_metric = current_metric_for_checkpoint; is_best = True; fold_best_val_loss = val_loss; fold_best_val_acc = val_acc
                    print(f"{TermColors.OKGREEN}🏆 Fold {fold+1} New Best {CHECKPOINT_MONITOR}: {best_metric:.4f}. Saving...{TermColors.ENDC}")
                    save_checkpoint(fold, global_epoch_counter + 1, stage_idx_loop, stage_epoch_loop + 1, model, optimizer, scheduler, scaler, best_metric, "best_model.pth.tar")
                    save_model(fold, model, "best_model_state_dict.pth")
                    if USE_EMA and ema_model: save_model(fold, ema_model, "best_ema_model_state_dict.pth")
                    if USE_SWA and swa_model and stage_epoch_loop >= globals()['swa_start_epoch_stage']: save_model(fold, swa_model, "best_swa_model_state_dict.pth")
                    
                    if current_oof_preds is not None and current_oof_indices is not None:
                        # current_oof_indices are original indices from df_full for this validation batch
                        for i, original_idx_val in enumerate(current_oof_indices):
                            if 0 <= original_idx_val < len(oof_preds_array): 
                                oof_preds_array[original_idx_val] = current_oof_preds[i]
                                oof_labels_array[original_idx_val] = df_full.loc[original_idx_val, 'label']
                
                save_checkpoint(fold, global_epoch_counter + 1, stage_idx_loop, stage_epoch_loop + 1, model, optimizer, scheduler, scaler, best_metric, "latest_checkpoint.pth.tar")
                if LOG_MISCLASSIFIED_IMAGES and ((global_epoch_counter + 1) % 5 == 0 or is_best): log_misclassified(fold, model, err_loader, criterion, DEVICE, global_epoch_counter + 1, writer, NUM_CLASSES)
                global_epoch_counter += 1; gc.collect(); torch.cuda.empty_cache(); start_stg_ep = 0 # Reset for next stage
            if fold_stop_requested or stop_requested: break
            del train_loader, val_loader, err_loader, train_ds, val_ds, err_ds, train_transform, val_transform, scheduler, swa_scheduler; scheduler = None; gc.collect(); torch.cuda.empty_cache()
        
        if fold_stop_requested or stop_requested: print(f"{TermColors.WARNING}Fold {fold+1} interrupted.{TermColors.ENDC}"); save_checkpoint(fold, global_epoch_counter, stage_idx_loop, stage_epoch_loop, model, optimizer, scheduler, scaler, best_metric, "interrupted_checkpoint.pth.tar"); break
        
        if USE_SWA and swa_model and global_epoch_counter >= int(TOTAL_EPOCHS_PER_FOLD * SWA_START_EPOCH_GLOBAL_FACTOR):
            print(f"{TermColors.CYAN}ℹ Fold {fold+1} Updating SWA BN stats...{TermColors.ENDC}")
            final_stage_size_swa = PROGRESSIVE_RESIZING_STAGES[-1][1]; final_train_tf_swa, _ = get_transforms(image_size=final_stage_size_swa)
            try:
                final_train_ds_bn_swa = PlantDataset(train_df, IMAGE_DIR, final_train_tf_swa, None, False, final_stage_size_swa)
                if len(final_train_ds_bn_swa) > 0:
                    bn_loader_swa = DataLoader(final_train_ds_bn_swa, BATCH_SIZE*2, shuffle=True, num_workers=NUM_WORKERS)
                    torch.optim.swa_utils.update_bn(bn_loader_swa, swa_model, device=DEVICE); print(f"{TermColors.GREEN}✅ SWA BN updated.{TermColors.ENDC}")
                    _, final_val_tf_swa = get_transforms(image_size=final_stage_size_swa)
                    final_val_ds_eval_swa = PlantDataset(val_df, IMAGE_DIR, final_val_tf_swa, None, False, final_stage_size_swa) # No paths needed for SWA eval
                    if len(final_val_ds_eval_swa) > 0:
                        final_val_loader_swa = DataLoader(final_val_ds_eval_swa, BATCH_SIZE*2, shuffle=False, num_workers=NUM_WORKERS)
                        swa_val_loss, swa_val_acc, _, _ = validate_one_epoch(swa_model, final_val_loader_swa, criterion, DEVICE, global_epoch_counter, writer, NUM_CLASSES, fold_num=f"{fold+1}-SWA")
                        print(f"Fold {fold+1} Final SWA Val L={swa_val_loss:.4f}, A={swa_val_acc:.4f}"); fold_results['swa_acc'].append(swa_val_acc); fold_results['swa_loss'].append(swa_val_loss)
                        save_model(fold, swa_model, "final_swa_model_state_dict.pth")
                    del final_val_ds_eval_swa, final_val_loader_swa
                del final_train_ds_bn_swa, bn_loader_swa
            except Exception as e_swa: print(f"{TermColors.RED}❌ SWA BN/Eval Error: {e_swa}{TermColors.ENDC}"); traceback.print_exc()
        save_model(fold, model, "final_model_state_dict.pth")
        if USE_EMA and ema_model: save_model(fold, ema_model, "final_ema_model_state_dict.pth")
        writer.close(); fold_results['best_metric'].append(best_metric); fold_results['best_val_loss'].append(fold_best_val_loss); fold_results['best_val_acc'].append(fold_best_val_acc)
        del model, optimizer, scaler, criterion, swa_model, ema_model, train_df, val_df, writer; gc.collect(); torch.cuda.empty_cache()

    print(f"\n{TermColors.HEADER}===== Cross-Validation Finished (Full Training) ====={TermColors.ENDC}")
    if not stop_requested:
        avg_best_acc = np.mean(fold_results['best_val_acc']) if fold_results['best_val_acc'] else 0.0
        std_best_acc = np.std(fold_results['best_val_acc']) if fold_results['best_val_acc'] else 0.0
        print(f"Avg Best Val Acc: {avg_best_acc:.4f} +/- {std_best_acc:.4f}")
        if fold_results['swa_acc']: print(f"Avg Final SWA Val Acc: {np.mean(fold_results['swa_acc']):.4f} +/- {np.std(fold_results['swa_acc']):.4f}")
        if RUN_STACKING:
            valid_oof_indices = np.where(oof_labels_array != -1)[0]
            if len(valid_oof_indices) < len(df_full) * 0.5: print(f"{TermColors.YELLOW}Warning: OOF collected for only {len(valid_oof_indices)}/{len(df_full)} samples. Stacking may be suboptimal.{TermColors.ENDC}")
            if len(valid_oof_indices) > 0:
                final_oof_preds = oof_preds_array[valid_oof_indices]; final_oof_labels = oof_labels_array[valid_oof_indices]
                if len(final_oof_preds) > 0 and len(final_oof_preds) == len(final_oof_labels):
                    np.savez_compressed(STACKING_OOF_PREDS_PATH, preds=final_oof_preds, labels=final_oof_labels); print(f"OOF preds saved: {STACKING_OOF_PREDS_PATH}")
                    train_stacking_meta_model(final_oof_preds, final_oof_labels, STACKING_META_MODEL_PATH)
                else: print(f"{TermColors.RED}Error preparing stacking data. Skipping.{TermColors.ENDC}")
                del final_oof_preds, final_oof_labels
            else: print(f"{TermColors.RED}No valid OOF predictions for Stacking. Skipping.{TermColors.ENDC}")
    else: print(f"{TermColors.YELLOW}Training interrupted. Stacking/KD skipped.{TermColors.ENDC}"); 
    del oof_preds_array, oof_labels_array; gc.collect()
    return not stop_requested # Return True if completed, False if interrupted


def main():
    global stop_requested, label_encoder, class_names, NUM_CLASSES, CLASS_FREQUENCIES, CLASS_PRIORS, CLASS_WEIGHTS
    global LEARNING_RATE, WEIGHT_DECAY, DROPOUT_RATE, ARCFACE_M, OPTIMIZER_TYPE, MODEL_NAMES, EMBEDDING_SIZE, USE_SAM
    global RUN_STACKING, RUN_KNOWLEDGE_DISTILLATION, USE_AUTO_TRAIN_CONFIG

    set_seed(SEED); signal.signal(signal.SIGINT, handle_interrupt); print_library_info()
    print(f"{TermColors.HEADER}===== Plant Recognition V3 (PyTorch) - Mode: {RUN_MODE} ===={TermColors.ENDC}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, PyTorch: {torch.__version__}, Device: {DEVICE}, Debug: {DEBUG_MODE}")
    
    # Common Data Loading and Preprocessing
    df_full = None
    try:
        print(f"\n{TermColors.HEADER}--- STEP 1: Load Full Dataset Info ---{TermColors.ENDC}")
        df_full = pd.read_csv(CSV_PATH, sep=',', low_memory=False, on_bad_lines='skip')
        required_source_cols = ['id', 'scientific_name']
        if 'scientificName' in df_full.columns and 'scientific_name' not in df_full.columns: df_full.rename(columns={'scientificName': 'scientific_name'}, inplace=True)
        if not all(col in df_full.columns for col in required_source_cols): print(f"{TermColors.RED}❌ Missing source columns: {[c for c in required_source_cols if c not in df_full.columns]}. Found: {df_full.columns.tolist()}{TermColors.ENDC}"); sys.exit(1)
        df_full = df_full[required_source_cols].dropna().astype({'id': str}); df_full.rename(columns={'scientific_name': 'scientificName'}, inplace=True)
        min_samples = 1; class_counts = df_full['scientificName'].value_counts(); valid_classes = class_counts[class_counts >= min_samples].index
        df_full = df_full[df_full['scientificName'].isin(valid_classes)].reset_index(drop=True)
        if len(df_full) == 0: print(f"{TermColors.RED}❌ Dataframe empty after min_samples filter. Exiting.{TermColors.ENDC}"); sys.exit(1)
        
        # Initialize LabelEncoder and transform labels
        label_encoder = LabelEncoder(); df_full['label'] = label_encoder.fit_transform(df_full['scientificName'])
        class_names = list(label_encoder.classes_); NUM_CLASSES = len(class_names)
        if NUM_CLASSES == 0: print(f"{TermColors.RED}❌ Zero classes after encoding. Exiting.{TermColors.ENDC}"); sys.exit(1)
        
        mapping_path = os.path.join(BASE_MODEL_SAVE_DIR, "label_mapping.json"); os.makedirs(BASE_MODEL_SAVE_DIR, exist_ok=True)
        with open(mapping_path, 'w') as f: json.dump(dict(zip(range(NUM_CLASSES), class_names)), f, indent=4)
        
        if DEBUG_MODE: 
            # Ensure stratification is possible with small debug dataset
            min_debug_samples_per_class = 2 
            stratify_col = df_full['label']
            # Filter classes with fewer than min_debug_samples_per_class for stratification
            class_counts_debug = df_full['label'].value_counts()
            valid_labels_for_stratify = class_counts_debug[class_counts_debug >= min_debug_samples_per_class].index
            df_debug_stratify_possible = df_full[df_full['label'].isin(valid_labels_for_stratify)]

            if len(df_debug_stratify_possible) >= min(100, len(df_full)) and len(valid_labels_for_stratify) > 1 :
                 _, df_full = train_test_split(df_debug_stratify_possible, test_size=min(100, len(df_debug_stratify_possible)), random_state=SEED, stratify=df_debug_stratify_possible['label'])
            else: # Fallback if stratification is not possible
                 df_full = df_full.sample(n=min(100, len(df_full)), random_state=SEED)
            df_full = df_full.reset_index(drop=True)
        
        label_counts = df_full['label'].value_counts().sort_index(); total_samples_final = len(df_full)
        if total_samples_final == 0: print(f"{TermColors.RED}❌ Dataframe empty before imbalance stats. Exiting.{TermColors.ENDC}"); sys.exit(1)
        freqs = torch.zeros(NUM_CLASSES, dtype=torch.float32); label_counts_reindexed = label_counts.reindex(range(NUM_CLASSES), fill_value=0)
        for i in range(NUM_CLASSES): freqs[i] = label_counts_reindexed.get(i, 0)
        CLASS_FREQUENCIES = freqs.to(DEVICE); CLASS_PRIORS = (CLASS_FREQUENCIES / total_samples_final if total_samples_final > 0 else torch.zeros_like(CLASS_FREQUENCIES)).to(DEVICE)
        try:
            class_weights_array = sk_class_weight.compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=df_full['label'])
            CLASS_WEIGHTS = torch.tensor(class_weights_array, dtype=torch.float32)
        except ValueError as e:
            print(f"{TermColors.RED}❌ Error calculating class weights: {e}.{TermColors.ENDC}")
            if IMBALANCE_STRATEGY in ['WeightedLoss', 'WeightedSampler']: print(f"{TermColors.RED}Exiting due to class weight error with strategy '{IMBALANCE_STRATEGY}'.{TermColors.ENDC}"); sys.exit(1)
            CLASS_WEIGHTS = None
        print(f"{TermColors.GREEN}✅ Data loaded: {len(df_full)} samples, {NUM_CLASSES} classes. Imbalance strategy: {IMBALANCE_STRATEGY}.{TermColors.ENDC}")
    except Exception as e: print(f"{TermColors.RED}❌ Data Load/Prep Error: {e}{TermColors.ENDC}"); traceback.print_exc(); sys.exit(1)

    # Execute based on RUN_MODE
    training_completed_successfully = False
    if RUN_MODE == "TRAIN_MLP_ON_FEATURES":
        if not os.path.exists(FEATURES_NPZ_PATH):
            print(f"{TermColors.YELLOW}⚠️ Features file not found: {FEATURES_NPZ_PATH}. Attempting to extract features first.{TermColors.ENDC}")
            extract_all_features(df_full, NUM_CLASSES, label_encoder)
            if not os.path.exists(FEATURES_NPZ_PATH):
                print(f"{TermColors.RED}❌ Feature extraction failed or did not produce the file: {FEATURES_NPZ_PATH}. Aborting MLP training.{TermColors.ENDC}")
                sys.exit(1)
            else:
                print(f"{TermColors.GREEN}✅ Features extracted successfully. Proceeding with MLP training.{TermColors.ENDC}")
        
        train_mlp_on_features_main_loop(NUM_CLASSES, label_encoder, class_names, CLASS_FREQUENCIES, CLASS_PRIORS, CLASS_WEIGHTS)
        training_completed_successfully = not stop_requested # Check if MLP training was interrupted
    elif RUN_MODE == "FULL_TRAINING":
        print(f"Initial Config: Folds={N_FOLDS}, Models={MODEL_NAMES}, TotalEpPerFold={TOTAL_EPOCHS_PER_FOLD}, Batch={BATCH_SIZE}x{GRADIENT_ACCUMULATION_STEPS}, LR={LEARNING_RATE}, Optim={OPTIMIZER_TYPE}(SAM:{USE_SAM}, AutoConfig:{USE_AUTO_TRAIN_CONFIG}), Loss={LOSS_TYPE}(LS:{LABEL_SMOOTHING}), Imbalance={IMBALANCE_STRATEGY}, Sched={SCHEDULER_TYPE}, MP={MIXED_PRECISION}, Compile={USE_TORCH_COMPILE}, SWA={USE_SWA}, EMA={USE_EMA}, TTA={USE_TTA}, Stacking={RUN_STACKING}, KD={RUN_KNOWLEDGE_DISTILLATION}, EarlyStopPat={EARLY_STOPPING_PATIENCE}")
        training_completed_successfully = main_full_training_loop(df_full, NUM_CLASSES, label_encoder, class_names, CLASS_FREQUENCIES, CLASS_PRIORS, CLASS_WEIGHTS)
    else:
        print(f"{TermColors.RED}❌ Invalid RUN_MODE: '{RUN_MODE}'. Choose from 'FULL_TRAINING', 'TRAIN_MLP_ON_FEATURES'.{TermColors.ENDC}")
        sys.exit(1)

    # Knowledge Distillation (if full training or MLP training completed successfully)
    # KD for MLP is handled within its loop, this is for CombinedModel teacher
    if RUN_KNOWLEDGE_DISTILLATION and training_completed_successfully and RUN_MODE == "FULL_TRAINING":
        print(f"\n{TermColors.HEADER}--- STEP 4: Knowledge Distillation (Post Full Training) ---{TermColors.ENDC}")
        try:
            # Assuming best model from fold 0 of CombinedModel as teacher
            teacher_path = os.path.join(BASE_CHECKPOINT_DIR, f"fold_{KD_TEACHER_FOLD}", "best_model.pth.tar")
            if os.path.exists(teacher_path):
                 df_kd_train, df_kd_val = train_test_split(df_full, test_size=0.15, random_state=SEED+1, stratify=df_full['label'])
                 train_student_model(teacher_path, KD_STUDENT_MODEL_NAME, KD_STUDENT_MODEL_SAVE_PATH, df_kd_train, df_kd_val, IMAGE_DIR, NUM_CLASSES, is_teacher_mlp=False)
                 del df_kd_train, df_kd_val
            else: print(f"{TermColors.RED}❌ Teacher model (CombinedModel) not found: {teacher_path}. Skip KD.{TermColors.ENDC}")
        except Exception as e_kd_main: print(f"{TermColors.RED}❌ KD Error (Post Full Training): {e_kd_main}{TermColors.ENDC}"); traceback.print_exc()
    
    elif RUN_KNOWLEDGE_DISTILLATION and training_completed_successfully and RUN_MODE == "TRAIN_MLP_ON_FEATURES":
        print(f"\n{TermColors.HEADER}--- STEP 4: Knowledge Distillation (Post MLP Training) ---{TermColors.ENDC}")
        try:
            # Assuming best MLP model from fold 0 as teacher
            teacher_path = os.path.join(BASE_MODEL_SAVE_DIR, f"mlp_fold_{KD_TEACHER_FOLD}", "best_mlp_model_state_dict.pth")
            if os.path.exists(teacher_path):
                 df_kd_train, df_kd_val = train_test_split(df_full, test_size=0.15, random_state=SEED+1, stratify=df_full['label'])
                 # Student is image-based, teacher is MLP. This KD setup needs careful thought.
                 # The current train_student_model expects teacher to process images if student does.
                 # For an MLP teacher, it would need features.
                 print(f"{TermColors.YELLOW}WARN: KD with MLP teacher and image-based student is complex and not fully implemented in this generic setup. Student will be trained from scratch or with a different image-based teacher if available.{TermColors.ENDC}")
                 # Fallback: try to use CombinedModel teacher if available, or skip.
                 fallback_teacher_path = os.path.join(BASE_CHECKPOINT_DIR, f"fold_{KD_TEACHER_FOLD}", "best_model.pth.tar")
                 if os.path.exists(fallback_teacher_path):
                     print(f"{TermColors.INFO}  Using CombinedModel from {fallback_teacher_path} as teacher for KD instead of MLP.{TermColors.ENDC}")
                     train_student_model(fallback_teacher_path, KD_STUDENT_MODEL_NAME, KD_STUDENT_MODEL_SAVE_PATH, df_kd_train, df_kd_val, IMAGE_DIR, NUM_CLASSES, is_teacher_mlp=False)
                 else:
                     print(f"{TermColors.RED}❌ MLP Teacher for KD selected, but image-based student requires image-based teacher or feature pipeline. Fallback CombinedModel teacher not found. Skipping KD.{TermColors.ENDC}")

                 del df_kd_train, df_kd_val
            else: print(f"{TermColors.RED}❌ Teacher model (MLP) not found: {teacher_path}. Skip KD.{TermColors.ENDC}")
        except Exception as e_kd_main: print(f"{TermColors.RED}❌ KD Error (Post MLP Training): {e_kd_main}{TermColors.ENDC}"); traceback.print_exc()


    print(f"\n{TermColors.OKGREEN} All processes complete for RUN_MODE: {RUN_MODE}.{TermColors.ENDC}")

if __name__ == "__main__":
    main()