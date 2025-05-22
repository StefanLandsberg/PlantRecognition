# --- Standard Library Imports ---
import csv
import gc
import glob # For finding image files
import json
import math
import pandas as pd
import numpy as np
import multiprocessing
import os
import random
import signal
import sys
import seaborn as sns
import traceback
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from tqdm.auto import tqdm

# --- PyTorch Imports ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import (ConstantLR, CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts, LambdaLR,
                                      LinearLR, ReduceLROnPlateau,
                                      SequentialLR, StepLR)
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import (DataLoader, Dataset, WeightedRandomSampler,
                              random_split, Subset)
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms

# --- Scikit-learn Imports ---
from sklearn.linear_model import LogisticRegression # For Stacking
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import class_weight as sk_class_weight
try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

# --- Utility Imports ---
import colorama
import cv2
import joblib # For saving sklearn models
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns

# --- Optional/Advanced Libraries ---
TIMM_AVAILABLE = False
TORCHMETRICS_AVAILABLE = False
ALBUMENTATIONS_AVAILABLE = False
ADAMP_AVAILABLE = False
SAM_AVAILABLE = False
_INITIALIZED = False
LGBM_PRINTED_INFO = False
OPTUNA_AVAILABLE = False
WANDB_AVAILABLE = False

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    print("FATAL: timm library not found. Required. Install with 'pip install timm'.")
    sys.exit(1)

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    print(f"WARN: optuna library not found. MLP HPO will be disabled. Install with 'pip install optuna'.")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("WARN: wandb library not found. Experiment tracking with Weights & Biases will be disabled. Install with 'pip install wandb'.")

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
    print("WARN: albumentations library not found. Using torchvision transforms for feature extraction. Install with 'pip install albumentations'.")

try:
    from adamp import AdamP
    ADAMP_AVAILABLE = True
except ImportError:
    print("WARN: adamp optimizer not found. Will use AdamW instead. Install with 'pip install adamp'.")

try:
    from sam_optimizer.sam import SAM
    SAM_AVAILABLE = True
except ImportError:
    print("WARN: sam_optimizer not found. SAM optimizer will be disabled for MLP even if MLP_USE_SAM is True. Install from 'https://github.com/davda54/sam' or ensure it's in your PYTHONPATH.")

def print_library_info():
    global _INITIALIZED, LGBM_PRINTED_INFO
    if _INITIALIZED:
        return
    
    print("INFO: timm library found.")
    if TORCHMETRICS_AVAILABLE: print("INFO: torchmetrics library found.")
    if ALBUMENTATIONS_AVAILABLE: print("INFO: albumentations library found.")
    else: print("WARN: Albumentations not found. Feature extraction will use basic transforms.")
    if ADAMP_AVAILABLE: print("INFO: adamp optimizer found.")
    if SAM_AVAILABLE: print("INFO: sam_optimizer found and available for MLP.")
    else: print("WARN: sam_optimizer not found, SAM will be disabled for MLP.")
    
    if LGBM_AVAILABLE and not LGBM_PRINTED_INFO:
        print("INFO: lightgbm library found and available for stacking.")
        LGBM_PRINTED_INFO = True
    elif not LGBM_AVAILABLE and not LGBM_PRINTED_INFO:
        print(f"{TermColors.WARN}WARN: lightgbm library not found. Stacking with LGBM will not be available. Install with 'pip install lightgbm'.{TermColors.ENDC}")
        LGBM_PRINTED_INFO = True

    if OPTUNA_AVAILABLE: print("INFO: optuna library found and available for MLP HPO.")
    if WANDB_AVAILABLE: print("INFO: wandb library found.")
    _INITIALIZED = True

# --- Terminal Colors ---
colorama.init(autoreset=True)
class TermColors:
    HEADER = '\033[95m'; OKBLUE = '\033[94m'; OKCYAN = '\033[96m'; OKGREEN = '\033[92m'
    WARNING = '\033[93m'; FAIL = '\033[91m'; ENDC = '\033[0m'; BOLD = '\033[1m'
    UNDERLINE = '\033[4m'; INFO = '\033[94m'; DEBUG = '\033[90m'; TRACE = '\033[90m'
    ERROR = '\033[91m'; SUCCESS = '\033[92m'; WARN = '\033[93m'
    CRITICAL = '\033[91m' + '\033[1m'; BLUE = '\033[94m'; CYAN = '\033[96m'
    GREEN = '\033[92m'; YELLOW = '\033[93m'; RED = '\033[91m'; MAGENTA = '\033[95m'
    DIM = '\033[2m'

# --- Configuration ---
SEED = 42
DEBUG_MODE = False # Set to True for small dataset and fewer epochs

# --- Path Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V3_DIR = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "scripts" else SCRIPT_DIR # Adjust if script is not in "scripts"
PROJECT_ROOT = os.path.dirname(V3_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "plant_images")
CSV_PATH = os.path.join(DATA_DIR, "observations-561226.csv") # Update with your CSV file name

BASE_CHECKPOINT_DIR = os.path.join(V3_DIR, "checkpoints_mlp_pytorch")
BASE_LOG_DIR = os.path.join(V3_DIR, "logs_mlp_pytorch")
BASE_MODEL_SAVE_DIR = os.path.join(V3_DIR, "models_mlp_pytorch")
BASE_ERROR_ANALYSIS_DIR = os.path.join(V3_DIR, "error_analysis_mlp_pytorch")

# --- Feature Extraction Config ---
FEATURES_NPZ_PATH = os.path.join(DATA_DIR, "extracted_features.npz") # Path to save/load features
FEATURE_EXTRACTION_IMAGE_SIZE_CONFIG = (512, 512) # Image size for feature extraction
FEATURE_EXTRACTOR_BATCH_SIZE = 32 # Batch size for extracting features
FEATURE_EXTRACTOR_CHECKPOINT_PATH = None # Optional: Path to a specific checkpoint for the CombinedModel feature extractor
FEATURE_EXTRACTION_MIXED_PRECISION = True if torch.cuda.is_available() else False 

os.makedirs(BASE_CHECKPOINT_DIR, exist_ok=True)
os.makedirs(BASE_LOG_DIR, exist_ok=True)
os.makedirs(BASE_MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(BASE_ERROR_ANALYSIS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# --- Cross-Validation Config ---
N_FOLDS = 5 if not DEBUG_MODE else 2

# --- Hardware Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = min(8, multiprocessing.cpu_count()) if DEVICE.type == 'cuda' else 0
MIXED_PRECISION = False if DEVICE.type == 'cuda' else False
USE_TORCH_COMPILE = False # Set to True if using PyTorch 2.0+ and want to try compilation

# --- Model Config (for Feature Extractor - CombinedModel) ---
MODEL_NAMES = ["tf_efficientnetv2_l_in21ft1k", "convnext_large_in22ft1k"] # Do test with ["tf_efficientnetv2_l_in21ft1k", "vit_large_patch14_dinov2.lvd142m"]
DROP_PATH_RATE = 0.1; PRETRAINED = True; NUM_CLASSES = -1 # NUM_CLASSES will be updated
EMBEDDING_SIZE = 2048; GLOBAL_POOLING = 'avg' # DROPOUT_RATE for feature extractor is 0 (set in extract_all_features)
ARCFACE_S = 15.0 
ARCFACE_M = 0.25 

# --- MLP Model Config ---
MLP_HIDDEN_DIMS = [1024, 512] # Hidden layer dimensions for the MLP
MLP_DROPOUT_RATE = 0.3  # Reduced from 0.5 to 0.3
MLP_USE_ARCFACE = True # Whether the MLP should also use an ArcFace head
MLP_LEARNING_RATE = 1e-3
MLP_WEIGHT_DECAY = 1e-4
MLP_EPOCHS = 150 if not DEBUG_MODE else 5
MLP_BATCH_SIZE = 256 if not DEBUG_MODE else 32
MLP_EARLY_STOPPING_PATIENCE = 7 if not DEBUG_MODE else 3
MLP_OPTIMIZER_TYPE = 'AdamW' # AdamP or AdamW
MLP_SCHEDULER_TYPE = 'CosineWarmRestarts' # e.g., CosineWarmRestarts, ReduceLROnPlateau
MLP_SWA_START_EPOCH_FACTOR = 0.70
MLP_USE_SWA = True
MLP_USE_EMA = True
MLP_EMA_DECAY = 0.999
MLP_USE_SAM = False  # Disabled SAM as it might be too aggressive
MLP_SAM_RHO = 0.007
MLP_SAM_ADAPTIVE = True

# --- MLP Advanced Features ---
MLP_ACTIVATION = 'ReLU'  # Options: 'ReLU', 'SiLU', 'GELU', 'Mish'
MLP_USE_FEATURE_NORMALIZATION = True  # Normalize features before feeding to MLP
MLP_FEATURE_AUGMENTATION = 'none'  # Changed from 'dropout' to 'none' to reduce regularization
MLP_FEATURE_DROPOUT_RATE = 0.05  # Feature dropout rate (for feature augmentation)
MLP_FEATURE_NOISE_SIGMA = 0.01  # Std of Gaussian noise to add (for feature augmentation)
MLP_MULTI_INIT_ENSEMBLE = 2  # Number of models to initialize with different seeds (1 = no ensemble)

# --- MLP Feature Mixup ---
MLP_USE_MIXUP = True  # Keep mixup as it's a good regularizer
MLP_MIXUP_ALPHA = 0.2  # Alpha parameter for Beta distribution in mixup
MLP_MIXUP_PROB = 0.5  # Probability of applying mixup to a batch


# --- MLP Learning Rate Schedule ---
MLP_LR_WARMUP_EPOCHS = 3  # Number of warmup epochs
MLP_LR_WARMUP_FACTOR = 0.1  # Start with this fraction of target LR

# --- MLP Hyperparameter Optimization (HPO) Config ---
MLP_DO_HPO = True
MLP_HPO_N_TRIALS = 100 if not DEBUG_MODE else 3
MLP_HPO_EPOCHS = 30 if not DEBUG_MODE else 2  # Increased from 30 to 60 for better convergence
MLP_HPO_PATIENCE = 5 if not DEBUG_MODE else 2
MLP_HPO_DATA_SPLIT_RATIO = 0.20
MLP_HPO_INTERNAL_VAL_SPLIT = 0.25
MLP_HPO_STUDY_DB_PATH = os.path.join(BASE_CHECKPOINT_DIR, "mlp_hpo_study.db") # Path for HPO study database
MLP_HPO_LOAD_BEST_ONLY = False  # If True, only load best from DB, don't run new trials unless DB is empty/invalid
                                # Flase, do HPO trails then load best from DB

# --- Loss Function & Imbalance Handling ---
LOSS_TYPE = 'CrossEntropy' 
USE_LABEL_SMOOTHING = False  # Flag to control whether to use label smoothing
LABEL_SMOOTHING = 0.1 if USE_LABEL_SMOOTHING else 0.0  # Only apply smoothing if enabled
FOCAL_ALPHA = 0.25; FOCAL_GAMMA = 2.0
IMBALANCE_STRATEGY = 'WeightedSampler' # Options: 'None', 'WeightedLoss', 'WeightedSampler', 'LogitAdjust'
LOGIT_ADJUSTMENT_TAU = 1.0
CLASS_FREQUENCIES = None; CLASS_PRIORS = None; CLASS_WEIGHTS = None

# --- Advanced Loss Options ---
USE_CLASS_FREQUENCY_SMOOTHING = False  # Disable this to avoid double smoothing
CLASS_FREQUENCY_SMOOTHING_BINS = 3
CLASS_FREQUENCY_SMOOTHING_VALUES = [0.05, 0.1, 0.2]

# --- Checkpointing Config ---
CHECKPOINT_MONITOR = 'val_acc' # 'val_loss' or 'val_acc'
CHECKPOINT_MODE = 'max' if CHECKPOINT_MONITOR == 'val_acc' else 'min'

# --- Error Analysis Config ---
ERROR_LOG_BATCH_SIZE = FEATURE_EXTRACTOR_BATCH_SIZE # Can be same as feature extractor
LOG_MISCLASSIFIED_IMAGES = False # Set to True to log misclassified images from feature extractor (if run)

# --- Stacking Config ---
RUN_STACKING = True
STACKING_META_MODEL_PATH = os.path.join(BASE_MODEL_SAVE_DIR, "stacking_meta_model_mlp.joblib")
STACKING_OOF_PREDS_PATH = os.path.join(BASE_MODEL_SAVE_DIR, "oof_predictions_mlp.npz")
STACKING_DO_HPO = True # Keep True to enable Optuna HPO
STACKING_HPO_CV_FOLDS = 3 if not DEBUG_MODE else 2 # Used within Optuna objective
STACKING_HPO_N_TRIALS = 20 if not DEBUG_MODE else 5 
STACKING_HPO_STUDY_DB_PATH = os.path.join(BASE_CHECKPOINT_DIR, "stacking_hpo_study.db") 

# --- OOF Collection Config ---
OOF_START_COLLECTION_EPOCH = 10  # Start collecting OOF predictions from this epoch
OOF_TRACK_PER_CLASS_PERFORMANCE = True  # Whether to track and keep best per-class OOF predictions
OOF_METRIC_FOR_BEST = 'f1'  # Metric to use for determining best epoch per class: 'f1', 'precision', 'recall', or 'accuracy'

# --- Knowledge Distillation Config ---
RUN_KNOWLEDGE_DISTILLATION = True
KD_STUDENT_MODEL_NAME = "mobilenetv3_small_100" # Student model (image-based)
KD_STUDENT_IMAGE_SIZE = (512, 512)
KD_STUDENT_EMBEDDING_SIZE = 512
KD_STUDENT_DROPOUT = 0.2
KD_EPOCHS = 20 if not DEBUG_MODE else 2
KD_BATCH_SIZE = FEATURE_EXTRACTOR_BATCH_SIZE # Batch size for student training
KD_LR = 1e-4
KD_ALPHA = 0.5 # Weight for distillation loss
KD_TEMPERATURE = 4.0 # Softening temperature for teacher logits
KD_STUDENT_MODEL_SAVE_PATH = os.path.join(BASE_MODEL_SAVE_DIR, f"final_distilled_student_{KD_STUDENT_MODEL_NAME}_base.pth")
KD_STUDENT_SWA_MODEL_SAVE_PATH = os.path.join(BASE_MODEL_SAVE_DIR, f"final_distilled_student_{KD_STUDENT_MODEL_NAME}_swa.pth")
KD_STUDENT_EMA_MODEL_SAVE_PATH = os.path.join(BASE_MODEL_SAVE_DIR, f"final_distilled_student_{KD_STUDENT_MODEL_NAME}_ema.pth")
KD_STUDENT_USE_SWA = True
KD_STUDENT_SWA_START_EPOCH_FACTOR = 0.70
KD_STUDENT_SWA_LR_FACTOR = 0.05
KD_STUDENT_SWA_ANNEAL_EPOCHS = 5
KD_STUDENT_USE_EMA = True
KD_STUDENT_EMA_DECAY = 0.999

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- Global Variables ---
stop_requested = False; label_encoder = None; class_names = None

# --- Utility Functions ---
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    print(f"{TermColors.INFO}Seed set to {seed}{TermColors.ENDC}")

def feature_mixup(features, labels, alpha=0.2):
    """
    Apply mixup to feature vectors
    
    Args:
        features: Input features tensor [batch_size, feature_dim]
        labels: Labels tensor [batch_size]
        alpha: Parameter for Beta distribution
    
    Returns:
        mixed_features: Mixed feature vectors
        labels_a: Original labels
        labels_b: Mixed-in labels
        lam: Mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0  # No mixup
        
    batch_size = features.size(0)
    if batch_size > 1:  # Only apply if batch has more than 1 sample
        index = torch.randperm(batch_size).to(features.device)
        mixed_features = lam * features + (1 - lam) * features[index]
        return mixed_features, labels, labels[index], lam
    else:
        return features, labels, labels, 1.0

def diagnose_oof_class_distribution(oof_preds, oof_labels, original_indices, fold_indices=None, fold_specific_labels=None, log_prefix=""):
    """
    Diagnose class distribution issues in OOF predictions.
    
    Args:
        oof_preds: Array of OOF predictions (or None for some diagnostics)
        oof_labels: Array of OOF labels 
        original_indices: Original indices for the OOF samples
        fold_indices: Optional list of fold indices for each sample
        fold_specific_labels: Optional list of per-fold labels for detailed analysis
        log_prefix: Prefix for log messages
    """
    print(f"\n{TermColors.CYAN}===== OOF DIAGNOSTICS {log_prefix} ====={TermColors.ENDC}")
    
    # Check for NaN or uninitialized values
    if oof_labels is not None:
        nan_count = np.sum(oof_labels == -1)
        if nan_count > 0:
            print(f"{TermColors.RED}ERROR: Found {nan_count} uninitialized labels (-1) in oof_labels{TermColors.ENDC}")
    
    if oof_preds is not None:
        nan_count_preds = np.sum(np.isnan(oof_preds))
        if nan_count_preds > 0:
            print(f"{TermColors.RED}ERROR: Found {nan_count_preds} NaN values in oof_preds{TermColors.ENDC}")
    
    # Overall class distribution
    if oof_labels is not None:
        unique_labels, counts = np.unique(oof_labels, return_counts=True)
        print(f"Overall class distribution: {len(unique_labels)} unique classes")
        print(f"Min count per class: {np.min(counts)} (class {unique_labels[np.argmin(counts)]})")
        print(f"Max count per class: {np.max(counts)} (class {unique_labels[np.argmax(counts)]})")
        
        # List classes with very few samples
        low_count_threshold = 5  # Adjust as needed
        low_count_classes = [(lbl, cnt) for lbl, cnt in zip(unique_labels, counts) if cnt <= low_count_threshold]
        if low_count_classes:
            print(f"{TermColors.YELLOW}Classes with <= {low_count_threshold} samples: {low_count_classes}{TermColors.ENDC}")
    
    # Per-fold analysis if fold info is provided
    if fold_indices is not None and fold_specific_labels is not None:
        print("\nPer-fold class distribution:")
        for fold, (fold_idx, fold_labels) in enumerate(zip(fold_indices, fold_specific_labels)):
            if fold_labels is None or len(fold_labels) == 0:
                print(f"  Fold {fold}: No labels")
                continue
                
            unique_fold_labels, fold_counts = np.unique(fold_labels, return_counts=True)
            print(f"  Fold {fold}: {len(unique_fold_labels)} unique classes")
            low_count_fold_classes = [(lbl, cnt) for lbl, cnt in zip(unique_fold_labels, fold_counts) if cnt <= low_count_threshold]
            if low_count_fold_classes:
                print(f"    Low count classes in fold {fold}: {low_count_fold_classes}")
    
    # Check for mismatches between indices
    if original_indices is not None:
        unique_indices = np.unique(original_indices)
        duplicate_indices = [idx for idx, count in zip(*np.unique(original_indices, return_counts=True)) if count > 1]
        if duplicate_indices:
            print(f"{TermColors.RED}ERROR: Found {len(duplicate_indices)} duplicate original indices{TermColors.ENDC}")
    
    print(f"{TermColors.CYAN}===== END OOF DIAGNOSTICS ====={TermColors.ENDC}\n")

def handle_interrupt(signal, frame):
    global stop_requested
    if stop_requested: print(f"\n{TermColors.CRITICAL}Force exiting...{TermColors.ENDC}"); sys.exit(1)
    print(f"\n{TermColors.WARNING}Interrupt received. Finishing current operation... Press Ctrl+C again to force exit.{TermColors.ENDC}")
    stop_requested = True

def check_keyboard_stop():
    if stop_requested: print(f"{TermColors.WARNING}Stop request detected. Breaking loop...{TermColors.ENDC}")
    return stop_requested

# --- Checkpointing, Saving, Logging ---
def save_checkpoint(fold, global_epoch, model, optimizer, scheduler, scaler, best_metric, filename="checkpoint.pth.tar", is_mlp_checkpoint=False):
    checkpoint_prefix = "mlp_" if is_mlp_checkpoint else "fe_" # "fe_" for feature extractor if we were to checkpoint it
    checkpoint_dir = os.path.join(BASE_CHECKPOINT_DIR, f"{checkpoint_prefix}fold_{fold}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    optimizer_is_sam_instance = hasattr(optimizer, 'base_optimizer') and SAM_AVAILABLE
    should_treat_as_sam_for_saving = optimizer_is_sam_instance and MLP_USE_SAM if is_mlp_checkpoint else False
        
    opt_state_dict = optimizer.base_optimizer.state_dict() if should_treat_as_sam_for_saving else optimizer.state_dict()

    state = {
        'fold': fold, 'global_epoch': global_epoch,
        'state_dict': model_state_dict, 'optimizer': opt_state_dict,
        'scheduler': scheduler.state_dict() if scheduler else None,
        'scaler': scaler.state_dict() if scaler else None,
        'best_metric': best_metric,
        'label_encoder_classes': list(label_encoder.classes_) if label_encoder else None,
        'is_sam_optimizer': should_treat_as_sam_for_saving
    }
    try:
        torch.save(state, filepath)
    except Exception as e:
        print(f"{TermColors.RED}Error saving checkpoint {filepath}: {e}{TermColors.ENDC}")

def load_checkpoint(fold, model, optimizer, scheduler, scaler, filename="checkpoint.pth.tar", is_mlp_checkpoint=False):
    checkpoint_prefix = "mlp_" if is_mlp_checkpoint else "fe_"
    checkpoint_dir = os.path.join(BASE_CHECKPOINT_DIR, f"{checkpoint_prefix}fold_{fold}")
    filepath = os.path.join(checkpoint_dir, filename)
    start_global_epoch = 0
    best_metric = float('-inf') if CHECKPOINT_MODE == 'max' else float('inf')
    loaded_label_classes = None

    if os.path.isfile(filepath):
        print(f"{TermColors.CYAN}Loading {'MLP ' if is_mlp_checkpoint else 'FE '}Fold {fold} checkpoint '{filename}'...{TermColors.ENDC}")
        try:
            ckpt = torch.load(filepath, map_location=DEVICE)
            start_global_epoch = ckpt.get('global_epoch', 0)
            best_metric = ckpt.get('best_metric', best_metric)
            loaded_label_classes = ckpt.get('label_encoder_classes', None)
            was_sam_optimizer_in_ckpt = ckpt.get('is_sam_optimizer', False)

            state_dict = ckpt['state_dict']
            new_state_dict = {}
            is_compiled = hasattr(model, '_orig_mod') 
            current_model_is_module = any(k.startswith('module.') for k in model.state_dict().keys())
            ckpt_is_module = any(k.startswith('module.') for k in state_dict.keys())

            for k, v in state_dict.items():
                name = k
                if ckpt_is_module and not current_model_is_module:
                    if name.startswith('module.'): name = name[len('module.'):]
                elif not ckpt_is_module and current_model_is_module:
                    name = 'module.' + name
                if is_compiled and not name.startswith('_orig_mod.'): name = '_orig_mod.' + name
                elif not is_compiled and name.startswith('_orig_mod.'): name = name[len('_orig_mod.'):]
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict, strict=False)

            if optimizer and 'optimizer' in ckpt and ckpt['optimizer']:
                current_optimizer_should_be_sam = MLP_USE_SAM and SAM_AVAILABLE and hasattr(optimizer, 'base_optimizer') if is_mlp_checkpoint else False
                opt_to_load_state_into = optimizer.base_optimizer if current_optimizer_should_be_sam and was_sam_optimizer_in_ckpt else optimizer
                if (current_optimizer_should_be_sam and not was_sam_optimizer_in_ckpt) or \
                   (not current_optimizer_should_be_sam and was_sam_optimizer_in_ckpt):
                    print(f"{TermColors.YELLOW}WARN: Optimizer SAM type mismatch. Ckpt SAM: {was_sam_optimizer_in_ckpt}, Current SAM: {current_optimizer_should_be_sam}. Load may fail.{TermColors.ENDC}")
                try: opt_to_load_state_into.load_state_dict(ckpt['optimizer'])
                except Exception as e: print(f"{TermColors.YELLOW}Optim Load Failed: {e}{TermColors.ENDC}")
            
            if scheduler and 'scheduler' in ckpt and ckpt['scheduler']:
                 try: scheduler.load_state_dict(ckpt['scheduler'])
                 except Exception as e: print(f"{TermColors.YELLOW}Scheduler Load Failed: {e}{TermColors.ENDC}")
            if scaler and 'scaler' in ckpt and ckpt['scaler']:
                try: scaler.load_state_dict(ckpt['scaler'])
                except Exception as e: print(f"{TermColors.YELLOW}Scaler Load Failed: {e}{TermColors.ENDC}")
            print(f"{TermColors.GREEN}Ckpt {'MLP ' if is_mlp_checkpoint else ''}Fold {fold} loaded. Resume GlobEp {start_global_epoch}. Best {CHECKPOINT_MONITOR}: {best_metric:.4f}{TermColors.ENDC}")
            if loaded_label_classes and label_encoder and list(label_encoder.classes_) != loaded_label_classes:
                print(f"{TermColors.CRITICAL}Label mapping mismatch! Exiting.{TermColors.ENDC}"); sys.exit(1)
        except Exception as e:
            print(f"{TermColors.RED}Error loading checkpoint {filepath}: {e}{TermColors.ENDC}"); traceback.print_exc()
            start_global_epoch = 0; best_metric = float('-inf') if CHECKPOINT_MODE == 'max' else float('inf')
    else:
        print(f"{TermColors.YELLOW}No checkpoint for {'MLP ' if is_mlp_checkpoint else ''}Fold {fold} at {filepath}. Starting fresh.{TermColors.ENDC}")
    
    return start_global_epoch, best_metric, loaded_label_classes

def save_model(fold, model, filename="final_model.pth", is_mlp_model=False):
    model_prefix = "mlp_" if is_mlp_model else "fe_"
    model_dir = os.path.join(BASE_MODEL_SAVE_DIR, f"{model_prefix}fold_{fold}")
    os.makedirs(model_dir, exist_ok=True)
    filepath = os.path.join(model_dir, filename)
    model_to_save = model
    if hasattr(model_to_save, 'module'): model_to_save = model_to_save.module
    if hasattr(model_to_save, '_orig_mod'): model_to_save = model_to_save._orig_mod # For torch.compile
    try: torch.save(model_to_save.state_dict(), filepath)
    except Exception as e: print(f"{TermColors.RED}Error saving model {filepath}: {e}{TermColors.ENDC}")

def log_misclassified(fold, model, dataloader, criterion, device, global_epoch, writer, num_classes, is_mlp_logging=False):
    if not LOG_MISCLASSIFIED_IMAGES and not is_mlp_logging: return # Only log for MLP errors if LOG_MISCLASSIFIED_IMAGES is False
    if is_mlp_logging and not LOG_MISCLASSIFIED_IMAGES : # Allow MLP error logging even if image logging is off
        pass # Proceed to log MLP errors to CSV
    elif not LOG_MISCLASSIFIED_IMAGES: return


    log_prefix = "mlp_" if is_mlp_logging else "fe_"
    error_dir = os.path.join(BASE_ERROR_ANALYSIS_DIR, f"{log_prefix}fold_{fold}")
    os.makedirs(error_dir, exist_ok=True)
    error_log_file = os.path.join(error_dir, f"epoch_{global_epoch}_errors.csv")
    model.eval(); misclassified_count = 0; logged_images_tb = 0
    
    print(f"{TermColors.CYAN}{'MLP ' if is_mlp_logging else 'FE '}Fold {fold} Logging misclassified for global epoch {global_epoch}...{TermColors.ENDC}")
    try:
        with open(error_log_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['identifier', 'true_label', 'predicted_label', 'confidence', 'loss', 'logits_raw', 'logits_adjusted']
            writer_csv = csv.DictWriter(csvfile, fieldnames=fieldnames); writer_csv.writeheader()
            with torch.no_grad():
                for batch_data in tqdm(dataloader, desc=f"Logging Errors {'MLP ' if is_mlp_logging else 'FE '}F{fold}", leave=False):
                    paths_or_indices = None
                    if is_mlp_logging: # (features, labels, original_indices)
                        if len(batch_data) != 3: continue
                        inputs, labels, paths_or_indices = batch_data # paths_or_indices are original data indices
                    else: # (images, labels, paths) - for feature extractor error logging
                        if len(batch_data) != 3: continue
                        inputs, labels, paths_or_indices = batch_data 

                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    with torch.amp.autocast('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda')):
                        current_metric_learning_type = 'ArcFace' if is_mlp_logging and MLP_USE_ARCFACE else 'None'
                        outputs_raw = model(inputs, labels=labels if current_metric_learning_type == 'ArcFace' else None)
                        adj_outputs = outputs_raw
                        if IMBALANCE_STRATEGY == 'LogitAdjust' and CLASS_PRIORS is not None:
                            logit_adj_val = LOGIT_ADJUSTMENT_TAU * torch.log(CLASS_PRIORS + 1e-12)
                            adj_outputs = outputs_raw + logit_adj_val.unsqueeze(0)
                        
                        loss = criterion(adj_outputs, labels)
                        preds = torch.argmax(adj_outputs, dim=1)
                        probs = F.softmax(adj_outputs, dim=1)

                    misclassified_mask = (preds != labels)
                    misclassified_indices = torch.where(misclassified_mask)[0]

                    for idx in misclassified_indices:
                        misclassified_count += 1
                        true_label_idx = labels[idx].item(); pred_label_idx = preds[idx].item()
                        confidence = probs[idx, pred_label_idx].item()
                        item_loss = F.cross_entropy(adj_outputs[idx].unsqueeze(0), labels[idx].unsqueeze(0)).item()
                        
                        identifier_val = paths_or_indices[idx]
                        if isinstance(identifier_val, torch.Tensor): identifier_val = identifier_val.item()
                        elif isinstance(identifier_val, str): identifier_val = os.path.basename(identifier_val)

                        true_n = class_names[true_label_idx] if class_names and 0 <= true_label_idx < len(class_names) else str(true_label_idx)
                        pred_n = class_names[pred_label_idx] if class_names and 0 <= pred_label_idx < len(class_names) else str(pred_label_idx)
                        
                        writer_csv.writerow({
                            'identifier': identifier_val, 'true_label': true_n, 'predicted_label': pred_n, 
                            'confidence': f"{confidence:.4f}", 'loss': f"{item_loss:.4f}", 
                            'logits_raw': outputs_raw[idx].cpu().numpy().round(2).tolist(), 
                            'logits_adjusted': adj_outputs[idx].cpu().numpy().round(2).tolist()
                        })

                        if writer and logged_images_tb < 20 and not is_mlp_logging and LOG_MISCLASSIFIED_IMAGES: # Tensorboard image logging only for image-based models
                            try:
                                mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).to(device); std = torch.tensor(IMAGENET_STD).view(3, 1, 1).to(device)
                                img_tensor = inputs[idx] * std + mean; img_tensor = torch.clamp(img_tensor, 0, 1)
                                writer.add_image(f"Misclassified/Fold_{fold}/True_{true_n}_Pred_{pred_n}_{identifier_val}", img_tensor, global_epoch)
                                logged_images_tb += 1
                            except Exception as img_e: print(f"{TermColors.YELLOW}Warn: Failed to log image {identifier_val} to TensorBoard: {img_e}{TermColors.ENDC}")
                    if stop_requested: break
            if stop_requested: return
        print(f"{TermColors.CYAN}{'MLP ' if is_mlp_logging else 'FE '}Fold {fold} Misclassified logged ({misclassified_count} errors). CSV: {error_log_file}{TermColors.ENDC}")
    except Exception as e: print(f"{TermColors.RED}Error during misclassified logging for fold {fold}: {e}{TermColors.ENDC}"); traceback.print_exc()

# --- Dataset and Transforms ---
class PlantDataset(Dataset): # Used for Feature Extraction
    def __init__(self, dataframe, image_dir, transform=None, label_encoder_instance=None, include_paths=False, image_size=None):
        self.input_df = dataframe.copy()
        self.image_dir = image_dir
        self.transform = transform
        self.include_paths = include_paths
        self.image_size = image_size if image_size else FEATURE_EXTRACTION_IMAGE_SIZE_CONFIG

        required_cols = ['scientificName', 'id', 'label']
        if not all(col in self.input_df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in self.input_df.columns]
            print(f"{TermColors.RED}PlantDataset input missing: {missing}. Found: {self.input_df.columns.tolist()}{TermColors.ENDC}")
            self.dataframe = pd.DataFrame()
            return

        image_file_lookup = defaultdict(list)
        if not os.path.isdir(self.image_dir):
            print(f"{TermColors.RED}Image dir not found: {self.image_dir}{TermColors.ENDC}")
            self.dataframe = pd.DataFrame()
            return
            
        print(f"{TermColors.INFO}Scanning image directory to build file lookup: {self.image_dir}{TermColors.ENDC}")
        for root, _, files in os.walk(self.image_dir):
            species_dir_name_from_path = os.path.basename(root) # e.g., "Acacia_stricta"
            for filename in files:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root, filename)
                    try:
                        if os.path.getsize(full_path) > 0:
                            # Filename example: Acacia_stricta_63808394_0.jpg
                            # Or augmented: Acacia_stricta_63808394_0_aug0.jpg
                            name_part = os.path.splitext(filename)[0] # "Acacia_stricta_63808394_0" or "Acacia_stricta_63808394_0_aug0"
                            
                            # Expected prefix: "Species_Dir_Name_"
                            expected_prefix = species_dir_name_from_path + "_"
                            if name_part.startswith(expected_prefix):
                                id_and_suffix_part = name_part[len(expected_prefix):] # "63808394_0" or "63808394_0_aug0"
                                
                                # The CSV 'id' should be the numeric part like "63808394"
                                # We assume the CSV 'id' is the first numeric part after species name.
                                parts = id_and_suffix_part.split('_')
                                if len(parts) >= 1: # Should have at least the ID part
                                    obs_id_from_file = parts[0] # This should be "63808394"
                                    if obs_id_from_file.isdigit():
                                        # Store all files (originals and augmentations) under the base CSV ID
                                        image_file_lookup[(species_dir_name_from_path, obs_id_from_file)].append(full_path)
                    except OSError:
                        pass 
        
        self.input_df['original_index'] = self.input_df.index
        
        all_image_entries = []

        for _, row in tqdm(self.input_df.iterrows(), total=len(self.input_df), desc="Matching CSV to image files (incl. augmentations)", leave=False, disable=DEBUG_MODE):
            try:
                species_name_csv = str(row['scientificName'])
                obs_id_csv = str(row['id']) # This is the key ID from CSV, e.g., "63808394"
                label = row['label']
                original_idx_in_df = row['original_index']
                
                species_dir_name_csv_derived = species_name_csv.replace(' ', '_').replace('/', '_').replace('\\', '_')

                candidate_files = image_file_lookup.get((species_dir_name_csv_derived, obs_id_csv), [])
                
                found_at_least_one_for_csv_entry = False
                for file_path in candidate_files:
                    all_image_entries.append({
                        'scientificName': species_name_csv, 
                        'label': label, 
                        'image_path': file_path, 
                        'original_index': original_idx_in_df 
                    })
                    found_at_least_one_for_csv_entry = True
            except Exception as e:
                print(f"{TermColors.RED}Error processing CSV row (ID: {row.get('id', 'N/A')}): {e}{TermColors.ENDC}")
        
        self.dataframe = pd.DataFrame(all_image_entries)
        
        if len(self.dataframe) == 0 and len(self.input_df) > 0:
            print(f"{TermColors.RED}Found 0 image files. Check CSV IDs, image paths, and naming conventions.{TermColors.ENDC}")
        elif len(self.dataframe) > 0 :
            # Simple count for logging
            num_unique_original_indices = self.dataframe['original_index'].nunique()
            print(f"{TermColors.INFO}PlantDataset: Processed {len(self.input_df)} CSV rows (unique original samples).")
            print(f"{TermColors.INFO}  Mapped to {num_unique_original_indices} original samples that had image files.")
            print(f"{TermColors.INFO}  Total usable image entries (originals + their pre-augmentations): {len(self.dataframe)}.")
            if len(self.input_df) > num_unique_original_indices:
                 print(f"{TermColors.YELLOW}  Note: {len(self.input_df) - num_unique_original_indices} CSV entries did not match any image files.{TermColors.ENDC}")
        else: 
             print(f"{TermColors.INFO}PlantDataset: Input CSV was empty or no images matched. Dataset is empty.{TermColors.ENDC}")

    def __len__(self): return len(self.dataframe)
    def get_labels(self): return self.dataframe['label'].tolist() if 'label' in self.dataframe.columns and not self.dataframe.empty else []
    def __getitem__(self, idx):
        if idx >= len(self.dataframe):
             dummy_img = torch.zeros((3, *self.image_size), dtype=torch.float32); label = -1; original_index = -1
             return (dummy_img, label, "ERROR_IDX_OOB", original_index) if self.include_paths else (dummy_img, label)
        
        row = self.dataframe.iloc[idx]; img_path = row['image_path']; label = row['label']; original_index = row['original_index']
        try:
            image = Image.open(img_path).convert('RGB'); image = np.array(image)
        except Exception as e:
            print(f"{TermColors.RED}Error loading {img_path}: {e}. Dummy data.{TermColors.ENDC}")
            dummy_img = torch.zeros((3, *self.image_size), dtype=torch.float32)
            err_file = os.path.basename(img_path) if isinstance(img_path, str) else "UNKNOWN"
            return (dummy_img, label if isinstance(label, int) else -1, f"ERROR_LOAD_{err_file}", original_index) if self.include_paths else (dummy_img, label if isinstance(label, int) else -1)

        if self.transform:
            try: augmented = self.transform(image=image); image = augmented['image']
            except Exception as e:
                print(f"{TermColors.RED}Transform error {img_path}: {e}. Fallback.{TermColors.ENDC}")
                try: # Fallback basic transform
                    pil_image = Image.fromarray(image)
                    fallback_tf = transforms.Compose([transforms.Resize(self.image_size), transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
                    image = fallback_tf(pil_image)
                except Exception as fb_e:
                    print(f"{TermColors.RED}Fallback failed {img_path}: {fb_e}. Dummy.{TermColors.ENDC}")
                    dummy_img = torch.zeros((3, *self.image_size), dtype=torch.float32)
                    err_file = os.path.basename(img_path) if isinstance(img_path, str) else "UNKNOWN"
                    return (dummy_img, label if isinstance(label, int) else -1, f"ERROR_TF_{err_file}", original_index) if self.include_paths else (dummy_img, label if isinstance(label, int) else -1)
        
        return (image, label, img_path, original_index) if self.include_paths else (image, label)

class FeatureDataset(Dataset): # Used for MLP Training
    def __init__(self, features, labels, original_indices=None):
        self.features = features; self.labels = labels
        self.original_indices = original_indices if original_indices is not None else np.arange(len(features))
    def __len__(self): return len(self.features)
    def __getitem__(self, idx): 
        # Ensure features are returned as float32 to prevent dtype mismatches
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        return features, self.labels[idx], self.original_indices[idx]
    def get_labels(self): return self.labels.tolist() if isinstance(self.labels, np.ndarray) else self.labels

def get_transforms(image_size=(224, 224), for_feature_extraction=False): # Simplified for MLP script
    h, w = int(image_size[0]), int(image_size[1])
    if for_feature_extraction:
        if ALBUMENTATIONS_AVAILABLE:
            return A.Compose([A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR), A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])
        else: # Fallback for feature extraction if Albumentations not available
            return transforms.Compose([transforms.Resize((h,w)), transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    # This function in MLP_only script is primarily for feature extraction.
    # If called with for_feature_extraction=False, it means an error or unexpected use.
    print(f"{TermColors.RED}get_transforms called without for_feature_extraction=True in MLP script. This is not expected. Returning basic eval transform.{TermColors.ENDC}")
    if ALBUMENTATIONS_AVAILABLE:
        return A.Compose([A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR), A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])
    else:
        return transforms.Compose([transforms.Resize((h,w)), transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])


# --- Model Architecture ---
class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=ARCFACE_S, m=ARCFACE_M): # Removed ls_eps, easy_margin for simplicity here
        super().__init__(); self.in_features = in_features; self.out_features = out_features; self.s = s; self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features)); nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m); self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m) # threshold to determine if angular margin can be added
        self.mm = math.sin(math.pi - m) * m # penalty for samples where margin cannot be added
    
    def forward(self, input_features, label): 
        # Normalize input features and weights
        normalized_input = F.normalize(input_features)
        normalized_weight = F.normalize(self.weight)
        cosine = F.linear(normalized_input, normalized_weight)

        # Prevent cosine from going slightly out of [-1, 1] due to numerical instability
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        if label is None: return cosine * self.s # For inference if labels not available
        
        # Calculate sine
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)) # Clamped cosine ensures 1.0 - cosine^2 >= 0   
        phi_target_angle = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi_target_angle, cosine - self.mm) 
        
        # Convert label to one-hot
        one_hot = torch.zeros(cosine.size(), device=input_features.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # Apply the modified cosine to the target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s # Scale the output
        return output

class CombinedModel(nn.Module): # Used as Feature Extractor
    def __init__(self, model_names, num_classes, pretrained=True, global_pool='avg', dropout_rate=0.0, embedding_size=512, drop_path_rate=0.1, arcface_s=ARCFACE_S, arcface_m=ARCFACE_M, metric_learning='None'):
        super().__init__(); self.model_names = model_names; self.num_classes = num_classes; self.embedding_size = embedding_size; self.metric_learning = metric_learning
        self.backbones = nn.ModuleList(); self.total_features = 0
        for name in model_names:
            try:
                kwargs = {'drop_path_rate': drop_path_rate} if drop_path_rate > 0 else {}
                backbone = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool=global_pool, **kwargs)
                self.total_features += backbone.num_features; self.backbones.append(backbone)
            except Exception as e: print(f"{TermColors.RED}Backbone Load Fail {name}: {e}{TermColors.ENDC}"); raise e
        self.embedding_layer = nn.Sequential(nn.Linear(self.total_features, self.embedding_size), nn.BatchNorm1d(self.embedding_size), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(dropout_rate) # Dropout is 0 for feature extraction
        if self.metric_learning == 'ArcFace': self.metric_fc = ArcFace(self.embedding_size, num_classes, s=arcface_s, m=arcface_m)
        else: self.metric_fc = nn.Linear(self.embedding_size, num_classes) # Not used if return_embedding=True
    
    def forward(self, x, labels=None, return_embedding=False):
        all_features = [backbone(x) for backbone in self.backbones]
        combined_features = torch.cat(all_features, dim=1) if len(all_features) > 1 else all_features[0]
        embedding = self.embedding_layer(combined_features)
        if return_embedding: return embedding
        embedding_dropped = self.dropout(embedding) # Dropout is 0.0 for feature extraction
        if self.metric_learning == 'ArcFace': output = self.metric_fc(embedding_dropped, labels)
        else: output = self.metric_fc(embedding_dropped)
        return output

class ModelExponentialMovingAverage:
    def __init__(self, decay_rate):
        self.decay_rate = decay_rate
        if not (0.0 <= self.decay_rate <= 1.0):
            raise ValueError(f"Decay rate must be between 0 and 1, got {self.decay_rate}")

    @torch.no_grad()
    def __call__(self, averaged_model_parameter, model_parameter, num_averaged):

        return self.decay_rate * averaged_model_parameter + (1 - self.decay_rate) * model_parameter

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_dims, num_classes, dropout_rate=MLP_DROPOUT_RATE, use_arcface=MLP_USE_ARCFACE, arcface_s=ARCFACE_S, arcface_m=ARCFACE_M, activation=MLP_ACTIVATION):
        super().__init__()
        self.use_arcface = use_arcface
        self.use_feature_normalization = MLP_USE_FEATURE_NORMALIZATION
        self.feature_augmentation = MLP_FEATURE_AUGMENTATION
        self.feature_dropout_rate = MLP_FEATURE_DROPOUT_RATE
        self.feature_noise_sigma = MLP_FEATURE_NOISE_SIGMA
        
        # Initialize activation function based on parameter
        if activation == 'ReLU':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'SiLU':
            self.activation = nn.SiLU(inplace=True)
        elif activation == 'GELU':
            self.activation = nn.GELU()
        elif activation == 'Mish':
            self.activation = nn.Mish(inplace=True) if hasattr(nn, 'Mish') else nn.SiLU(inplace=True)
        else:
            print(f"{TermColors.YELLOW}Unknown activation '{activation}', defaulting to ReLU{TermColors.ENDC}")
            self.activation = nn.ReLU(inplace=True)
            
        # Feature normalization layer (BatchNorm without affine transformation)
        if self.use_feature_normalization:
            self.feature_norm = nn.BatchNorm1d(input_size, affine=False)
            
        # Feature dropout for augmentation
        if self.feature_augmentation in ['dropout', 'both']:
            self.feature_dropout = nn.Dropout(self.feature_dropout_rate)
        
        # Build hidden layers
        layers = []
        current_dim = input_size
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(current_dim, h_dim), nn.BatchNorm1d(h_dim), self.activation, nn.Dropout(dropout_rate)])
            current_dim = h_dim
            
        self.hidden_layers = nn.Sequential(*layers)
        
        if self.use_arcface:
            self.metric_fc = ArcFace(current_dim, num_classes, s=arcface_s, m=arcface_m)
        else:
            self.metric_fc = nn.Linear(current_dim, num_classes)

    def _augment_features(self, x, is_training=True):
        # Apply feature normalization if enabled
        if self.use_feature_normalization:
            x = self.feature_norm(x)
            
        # Only apply augmentation during training
        if is_training:
            # Apply feature dropout if enabled
            if self.feature_augmentation in ['dropout', 'both']:
                x = self.feature_dropout(x)
                
            # Apply noise augmentation if enabled
            if self.feature_augmentation in ['noise', 'both'] and self.feature_noise_sigma > 0:
                noise = torch.randn_like(x) * self.feature_noise_sigma
                x = x + noise
                
        return x

    def forward(self, x, labels=None):
        # Apply feature normalization and augmentation
        x = self._augment_features(x, self.training)
        
        # Pass through hidden layers
        x = self.hidden_layers(x)
        
        # Pass through final classification layer
        if self.use_arcface:
            x = self.metric_fc(x, labels)  # ArcFace requires labels during training
        else:
            x = self.metric_fc(x)
            
        return x

def build_feature_extractor_model(num_classes_fe): # For feature extraction
    return CombinedModel(MODEL_NAMES, num_classes_fe, PRETRAINED, GLOBAL_POOLING, 0.0, EMBEDDING_SIZE, DROP_PATH_RATE, ARCFACE_S, ARCFACE_M, 'None')

def build_mlp_model(input_size, num_classes_mlp): # For MLP training
    return SimpleMLP(input_size, MLP_HIDDEN_DIMS, num_classes_mlp, MLP_DROPOUT_RATE, MLP_USE_ARCFACE, ARCFACE_S, ARCFACE_M, MLP_ACTIVATION)

# --- Loss Functions ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, reduction='mean'):
        super().__init__(); self.alpha = alpha; self.gamma = gamma; self.reduction = reduction
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none'); pt = torch.exp(-ce_loss); focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum() if self.reduction == 'sum' else focal_loss

class AdaptiveLabelSmoothingLoss(nn.Module):
    def __init__(self, class_frequencies, num_bins=CLASS_FREQUENCY_SMOOTHING_BINS, 
                 smoothing_values=CLASS_FREQUENCY_SMOOTHING_VALUES, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        
        # Group classes based on frequency
        if class_frequencies is None:
            print(f"{TermColors.YELLOW}AdaptiveLabelSmoothingLoss: No class frequencies provided, using uniform smoothing{TermColors.ENDC}")
            self.smoothing_per_class = None
            self.default_smoothing = smoothing_values[1] if len(smoothing_values) > 1 else 0.1
            return
        
        # Sort classes by frequency
        class_freq_np = class_frequencies.cpu().numpy()
        sorted_indices = np.argsort(class_freq_np)
        
        # Divide into bins
        bin_size = max(1, len(sorted_indices) // num_bins)
        
        # Create smoothing vector
        self.smoothing_per_class = torch.zeros_like(class_frequencies)
        
        for i in range(num_bins):
            start_idx = i * bin_size
            end_idx = None if i == num_bins - 1 else (i + 1) * bin_size
            
            bin_classes = sorted_indices[start_idx:end_idx]
            smoothing_value = smoothing_values[i] if i < len(smoothing_values) else smoothing_values[-1]
            
            self.smoothing_per_class[bin_classes] = smoothing_value
        
        self.default_smoothing = smoothing_values[1] if len(smoothing_values) > 1 else 0.1
        print(f"{TermColors.INFO}AdaptiveLabelSmoothingLoss: Created smoothing bins with values: {smoothing_values[:num_bins]}{TermColors.ENDC}")
    
    def forward(self, inputs, targets):
        num_classes = inputs.size(-1)
        
        if self.smoothing_per_class is not None:
            # Get smoothing value for each target
            smoothing = self.smoothing_per_class[targets].to(inputs.device)
            
            # Create one-hot encoding
            one_hot = torch.zeros_like(inputs)
            one_hot.scatter_(1, targets.unsqueeze(1), 1)
            
            # Apply different smoothing for each sample based on its class
            smoothed_targets = torch.zeros_like(inputs)
            
            for i, target in enumerate(targets):
                current_smoothing = smoothing[i].item()
                # Create smoothed target for this sample
                smoothed_targets[i] = one_hot[i] * (1 - current_smoothing) + current_smoothing / num_classes
        else:
            # Fallback to uniform smoothing
            smoothing = self.default_smoothing
            log_probs = F.log_softmax(inputs, dim=-1)
            with torch.no_grad():
                one_hot = torch.zeros_like(inputs)
                one_hot.scatter_(1, targets.unsqueeze(1), 1)
                smoothed_targets = one_hot * (1 - smoothing) + smoothing / num_classes
            
            loss = -(smoothed_targets * log_probs).sum(dim=-1)
            
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss
        
        log_probs = F.log_softmax(inputs, dim=-1)
        loss = -(smoothed_targets * log_probs).sum(dim=-1)
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def get_criterion(class_weights_tensor=None, current_label_smoothing=LABEL_SMOOTHING, is_mlp_criterion=False):
    weights = class_weights_tensor.to(DEVICE) if class_weights_tensor is not None else None
    # For MLP, label smoothing is 0 if MLP_USE_ARCFACE is True or if USE_LABEL_SMOOTHING is False
    actual_label_smoothing = 0.0 if (is_mlp_criterion and MLP_USE_ARCFACE) or not USE_LABEL_SMOOTHING else current_label_smoothing
    return nn.CrossEntropyLoss(label_smoothing=actual_label_smoothing, weight=weights)

# --- Optimizer and Scheduler ---
def get_optimizer(model, is_mlp_optimizer=False):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params: raise ValueError("No trainable parameters in model")
    
    lr = MLP_LEARNING_RATE if is_mlp_optimizer else None # FE doesn't train here
    wd = MLP_WEIGHT_DECAY if is_mlp_optimizer else None
    optim_type = MLP_OPTIMIZER_TYPE if is_mlp_optimizer else None
    use_sam = MLP_USE_SAM if is_mlp_optimizer else False
    sam_rho_val = MLP_SAM_RHO if is_mlp_optimizer else 0.0
    sam_adaptive_val = MLP_SAM_ADAPTIVE if is_mlp_optimizer else False

    if lr is None: raise ValueError("LR not set for optimizer (not MLP context?)")

    if use_sam and SAM_AVAILABLE:
        from functools import partial
        base_opt_fn = None
        if optim_type == 'AdamP' and ADAMP_AVAILABLE: base_opt_fn = partial(AdamP, lr=lr, weight_decay=wd, betas=(0.9, 0.999), nesterov=True)
        elif optim_type == 'AdamW': base_opt_fn = partial(optim.AdamW, lr=lr, weight_decay=wd)
        else: base_opt_fn = partial(optim.AdamW, lr=lr, weight_decay=wd); print(f"{TermColors.YELLOW}WARN: SAM base for MLP defaulting to AdamW.{TermColors.ENDC}")
        print(f"{TermColors.BLUE}INFO: Using SAM for MLP (rho={sam_rho_val}, adaptive={sam_adaptive_val}). Base: {optim_type}{TermColors.ENDC}")
        return SAM(params, base_opt_fn, rho=sam_rho_val, adaptive=sam_adaptive_val)
    else:
        if optim_type == 'AdamP' and ADAMP_AVAILABLE: return AdamP(params, lr=lr, weight_decay=wd, betas=(0.9, 0.999), nesterov=True)
        if optim_type == 'AdamW': return optim.AdamW(params, lr=lr, weight_decay=wd)
        if optim_type == 'SGD': return optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9, nesterov=True)
        print(f"{TermColors.YELLOW}WARN: MLP Optimizer '{optim_type}' not AdamP/SGD or not avail. Defaulting to AdamW.{TermColors.ENDC}")
        return optim.AdamW(params, lr=lr, weight_decay=wd)

def get_scheduler(optimizer, is_mlp_scheduler=False, total_epochs_for_sched=None):
    if not is_mlp_scheduler: return None # No scheduler for feature extractor here
    
    opt_for_scheduler = optimizer.base_optimizer if hasattr(optimizer, 'base_optimizer') else optimizer
    sched_type = MLP_SCHEDULER_TYPE
    epochs = total_epochs_for_sched if total_epochs_for_sched is not None else MLP_EPOCHS
    lr = MLP_LEARNING_RATE
    
    # Create the main scheduler
    if sched_type == 'CosineWarmRestarts':
        main_scheduler = CosineAnnealingWarmRestarts(opt_for_scheduler, T_0=epochs, T_mult=1, eta_min=lr * 0.01) 
    elif sched_type == 'ReduceLROnPlateau':
        main_scheduler = ReduceLROnPlateau(opt_for_scheduler, mode=CHECKPOINT_MODE, factor=0.2, patience=5, min_lr=lr * 0.001, monitor=CHECKPOINT_MONITOR)
    else:
        return None  # Default no scheduler if type not matched
        
    # Apply warmup if enabled
    if MLP_LR_WARMUP_EPOCHS > 0 and not isinstance(main_scheduler, ReduceLROnPlateau):
        # LR warmup doesn't work well with ReduceLROnPlateau since it needs validation metrics
        warmup_scheduler = LinearLR(
            opt_for_scheduler, 
            start_factor=MLP_LR_WARMUP_FACTOR, 
            end_factor=1.0, 
            total_iters=MLP_LR_WARMUP_EPOCHS
        )
        combined_scheduler = SequentialLR(
            opt_for_scheduler, 
            schedulers=[warmup_scheduler, main_scheduler], 
            milestones=[MLP_LR_WARMUP_EPOCHS]
        )
        return combined_scheduler
        
    return main_scheduler

# --- Training & Validation Loops (Simplified for MLP) ---
def train_one_epoch_mlp(model, dataloader, criterion, optimizer, scaler, scheduler, global_epoch, fold_num, device, writer, num_classes, ema_model):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Track per-epoch metrics
    epoch_metrics = {
        'loss': [],
        'acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    pbar = tqdm(dataloader, desc=f'Fold {fold_num} - Epoch {global_epoch}')
    for batch_idx, (features, labels, _) in enumerate(pbar):  # Unpack all three values
        features, labels = features.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(features, labels=labels if MLP_USE_ARCFACE else None)
        loss = criterion(outputs, labels)
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{total_loss/(batch_idx+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    # Calculate epoch metrics
    epoch_loss = total_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    # Log metrics
    if writer is not None:
        writer.add_scalar(f'fold_{fold_num}/train_loss', epoch_loss, global_epoch)
        writer.add_scalar(f'fold_{fold_num}/train_acc', epoch_acc, global_epoch)
    
    # Store metrics
    epoch_metrics['loss'].append(epoch_loss)
    epoch_metrics['acc'].append(epoch_acc)
    
    return epoch_metrics

def validate_one_epoch_mlp(model, dataloader, criterion, device, global_epoch, writer, num_classes, scheduler=None, swa_model=None, ema_model=None, return_preds=False, fold_num=0):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    all_indices = []
    
    with torch.no_grad():
        for features, labels, indices in tqdm(dataloader, desc=f'Validation - Fold {fold_num}'):
            features, labels = features.to(device), labels.to(device)
            features = features.float()  # Ensure features are float32
            
            # Forward pass
            outputs = model(features, labels=labels if MLP_USE_ARCFACE else None)
            loss = criterion(outputs, labels)
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if return_preds:
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_indices.append(indices.cpu().numpy())
    
    # Calculate validation metrics
    val_loss = total_loss / len(dataloader)
    val_acc = 100. * correct / total
    
    # Log metrics
    if writer is not None:
        writer.add_scalar(f'fold_{fold_num}/val_loss', val_loss, global_epoch)
        writer.add_scalar(f'fold_{fold_num}/val_acc', val_acc, global_epoch)
    
    # NOTE: Skip the train-val gap calculation since we can't easily retrieve the train metrics
    # SummaryWriter doesn't have a get_scalar method, and we don't have a good way to
    # access previously logged train metrics here
    
    if return_preds and all_preds:
        # Concatenate predictions, labels, and indices
        all_preds_concat = np.vstack(all_preds)
        all_labels_concat = np.concatenate(all_labels)
        all_indices_concat = np.concatenate(all_indices)
        return val_loss, val_acc, all_preds_concat, all_indices_concat, all_labels_concat
    
    return val_loss, val_acc

def plot_learning_curves(metrics, save_path):
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(metrics['acc'], label='Train')
    plt.plot(metrics['val_acc'], label='Validation')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(metrics['loss'], label='Train')
    plt.plot(metrics['val_loss'], label='Validation')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# --- Stacking ---
def stacking_hpo_objective(trial, X, y):
    if not LGBM_AVAILABLE:
        raise optuna.exceptions.TrialPruned("LGBM not available for stacking HPO.")

    param = {
        'objective': 'multiclass',
        'metric': 'multi_logloss', 
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': SEED,
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150, step=5),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50, step=5),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.05),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.05),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
    }

    # First, perform deep diagnostic on the input data
    print(f"\n{TermColors.CYAN}HPO Trial {trial.number}: Input data diagnostics{TermColors.ENDC}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Check class distribution
    unique_labels, counts = np.unique(y, return_counts=True)
    print(f"Class distribution: {len(unique_labels)} unique classes")
    
    if len(counts) == 0:
        print(f"{TermColors.RED}No labels found in y for HPO. Pruning trial.{TermColors.ENDC}")
        raise optuna.exceptions.TrialPruned("No labels found in y for HPO.")
    
    # Calculate appropriate number of splits based on data
    min_actual_class_count = np.min(counts)
    max_possible_splits = min(min_actual_class_count, 5)  # At most 5 splits
    n_splits_hpo_internal = min(max(2, max_possible_splits-1), STACKING_HPO_CV_FOLDS)
    
    print(f"{TermColors.INFO}Class count stats - Min: {min_actual_class_count}, Max: {np.max(counts)}, Mean: {np.mean(counts):.1f}{TermColors.ENDC}")
    
    # Show problematic classes
    if min_actual_class_count < STACKING_HPO_CV_FOLDS:
        low_count_classes = [(label, count) for label, count in zip(unique_labels, counts) if count < STACKING_HPO_CV_FOLDS]
        print(f"{TermColors.YELLOW}Classes with < {STACKING_HPO_CV_FOLDS} samples: {low_count_classes[:10]}{TermColors.ENDC}")
        print(f"{TermColors.YELLOW}Using {n_splits_hpo_internal} splits instead of {STACKING_HPO_CV_FOLDS} due to min class count {min_actual_class_count}{TermColors.ENDC}")
    
    # Use the possibly adjusted n_splits
    skf_hpo = StratifiedKFold(n_splits=n_splits_hpo_internal, shuffle=True, random_state=SEED + trial.number)
    cv_scores = []

    with warnings.catch_warnings():
        warnings.filterwarnings(
            action='ignore', 
            message="The least populated class in y has only.*members, which is less than n_splits=.*", # Regex to catch the specific warning
            category=UserWarning,
            module='sklearn.model_selection._split' # Target the module emitting the warning
        )
        
        try:
            for fold_idx, (train_idx, val_idx) in enumerate(skf_hpo.split(X, y)):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                # Check if all classes are represented in both train and validation
                train_classes = np.unique(y_train_fold)
                val_classes = np.unique(y_val_fold)
                print(f"Fold {fold_idx}: Train classes: {len(train_classes)}, Val classes: {len(val_classes)}")
                
                # Continue even if some classes are missing from validation set
                model = LGBMClassifier(**param)
                try:
                    model.fit(X_train_fold, y_train_fold,
                              eval_set=[(X_val_fold, y_val_fold)],
                              callbacks=[optuna.integration.LightGBMPruningCallback(trial, 'multi_logloss')])
                    preds = model.predict(X_val_fold)
                    accuracy = accuracy_score(y_val_fold, preds)
                    cv_scores.append(accuracy)
                except Exception as e_fit: 
                    print(f"{TermColors.YELLOW}Fitting failed in fold {fold_idx}: {e_fit}. Skipping fold.{TermColors.ENDC}")
                    continue # Skip this fold but continue with others
                    
        except ValueError as e_split: 
            print(f"{TermColors.RED}Error during StratifiedKFold split: {e_split}{TermColors.ENDC}")
            # Less drastic: try with a smaller number of folds instead of pruning
            try:
                print(f"{TermColors.YELLOW}Trying with 2 folds as fallback...{TermColors.ENDC}")
                skf_hpo_fallback = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED + trial.number + 100)
                
                for fold_idx, (train_idx, val_idx) in enumerate(skf_hpo_fallback.split(X, y)):
                    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                    
                    model = LGBMClassifier(**param)
                    model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)])
                    preds = model.predict(X_val_fold)
                    accuracy = accuracy_score(y_val_fold, preds)
                    cv_scores.append(accuracy)
            except Exception as e_fallback:
                print(f"{TermColors.RED}Fallback also failed: {e_fallback}{TermColors.ENDC}")
                return -float('inf')  # Last resort: return a poor score

    if not cv_scores: 
        print(f"{TermColors.RED}No CV scores collected for trial {trial.number}.{TermColors.ENDC}")
        return -float('inf')
        
    mean_score = np.mean(cv_scores)
    print(f"{TermColors.GREEN}Trial {trial.number} completed with mean CV score: {mean_score:.4f}{TermColors.ENDC}")
    return mean_score

def train_stacking_meta_model(oof_preds, oof_labels, save_path):
    print(f"{TermColors.CYAN}Training Stacking Meta-Model (MLP OOFs)...{TermColors.ENDC}")
    if oof_preds.ndim == 1: oof_features = oof_preds.reshape(-1, 1)
    else: oof_features = oof_preds
        
    meta_model = None; best_params_from_hpo = None; meta_model_name_for_log = "LGBM"

    if LGBM_AVAILABLE and STACKING_DO_HPO and OPTUNA_AVAILABLE:
        print(f"{TermColors.DEBUG}  Optuna HPO for LGBM stacker (CV Folds for objective: {STACKING_HPO_CV_FOLDS}, Trials: {STACKING_HPO_N_TRIALS})...{TermColors.ENDC}")
        os.makedirs(os.path.dirname(STACKING_HPO_STUDY_DB_PATH), exist_ok=True)
        study_name_stacking = "stacking_mlp_plant_hpo_study"
        storage_name_stacking = f"sqlite:///{STACKING_HPO_STUDY_DB_PATH}"
        
        try:
            study_stacking = optuna.create_study(
                study_name=study_name_stacking,
                storage=storage_name_stacking,
                direction='maximize',
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=max(1, STACKING_HPO_N_TRIALS // 4)),
                load_if_exists=True
            )
            
            # Check if enough trials are already completed if MLP_HPO_LOAD_BEST_ONLY logic were to be applied here
            # For now, always run STACKING_HPO_N_TRIALS new ones if STACKING_DO_HPO is True
            
            objective_with_data = lambda trial: stacking_hpo_objective(trial, oof_features, oof_labels)
            study_stacking.optimize(objective_with_data, n_trials=STACKING_HPO_N_TRIALS, callbacks=[lambda study, trial: gc.collect()])

            if study_stacking.best_trial:
                best_params_from_hpo = study_stacking.best_trial.params
                print(f"{TermColors.GREEN}  Stacking HPO (Optuna) complete. Best CV score from study: {study_stacking.best_trial.value:.4f}{TermColors.ENDC}")
                print(f"    Best HPO Params for Stacking: {best_params_from_hpo}")
                meta_model = LGBMClassifier(**best_params_from_hpo, random_state=SEED, verbosity=-1, n_jobs=-1)
                meta_model_name_for_log = "LGBM_Optuna_HPO"
            else:
                print(f"{TermColors.YELLOW}  Stacking HPO (Optuna) did not find a best trial. Using default LGBM.{TermColors.ENDC}")
        except Exception as e:
            print(f"{TermColors.RED}  Stacking HPO (Optuna) Error: {e}. Default LGBM.{TermColors.ENDC}"); traceback.print_exc()
            meta_model = None # Ensure it falls back

    # Fallback or if HPO is off/failed
    if meta_model is None and LGBM_AVAILABLE:
        print(f"{TermColors.DEBUG}  Using default LGBM parameters for stacking.{TermColors.ENDC}")
        meta_model = LGBMClassifier(random_state=SEED, n_jobs=-1, verbosity=-1)
        meta_model_name_for_log = "LGBM_Default"
    
    if meta_model: # Try fitting the selected LGBM model (either from HPO or default)
        try:
            meta_model.fit(oof_features, oof_labels)
        except Exception as e:
            print(f"{TermColors.RED}  Error fitting {meta_model_name_for_log}: {e}. Trying LogisticRegression.{TermColors.ENDC}")
            meta_model = None # Force fallback to LogisticRegression

    if meta_model is None: # Fallback to Logistic Regression if LGBM fails or not available
        print(f"{TermColors.DEBUG}  Using LogisticRegression for stacking.{TermColors.ENDC}")
        meta_model = LogisticRegression(max_iter=1000, random_state=SEED, n_jobs=-1)
        meta_model_name_for_log = "LogisticRegression"
        try:
            meta_model.fit(oof_features, oof_labels)
        except Exception as e:
            print(f"{TermColors.RED}  LogisticRegression Error: {e}. Stacking failed.{TermColors.ENDC}")
            return # Critical failure if even LogisticRegression fails

    # Calculate accuracy of the final fitted meta_model
    final_meta_acc = accuracy_score(oof_labels, meta_model.predict(oof_features))
    print(f"{TermColors.GREEN}Stacking meta-model ({meta_model_name_for_log}) trained. Final OOF Accuracy: {final_meta_acc:.4f}{TermColors.ENDC}")
    joblib.dump({"model": meta_model, "scaler": None}, save_path)
    print(f"  Meta-model saved: {save_path}")


# --- MLP HPO Objective Function & Runner ---
def mlp_hpo_objective(trial, features_hpo_train, labels_hpo_train, features_hpo_val, labels_hpo_val, input_size_mlp, num_classes_mlp):
    global MLP_LEARNING_RATE, MLP_WEIGHT_DECAY, MLP_DROPOUT_RATE, MLP_HIDDEN_DIMS
    global MLP_OPTIMIZER_TYPE, MLP_SCHEDULER_TYPE, LOSS_TYPE, MLP_ACTIVATION 
    global MLP_USE_FEATURE_NORMALIZATION, MLP_FEATURE_AUGMENTATION, MLP_FEATURE_DROPOUT_RATE, MLP_FEATURE_NOISE_SIGMA
    
    # Store original global values to restore them later
    original_mlp_lr = MLP_LEARNING_RATE
    original_mlp_wd = MLP_WEIGHT_DECAY
    original_mlp_dropout = MLP_DROPOUT_RATE
    original_mlp_hidden_dims = MLP_HIDDEN_DIMS
    original_mlp_optimizer = MLP_OPTIMIZER_TYPE
    original_mlp_scheduler = MLP_SCHEDULER_TYPE
    original_loss_type = LOSS_TYPE
    original_mlp_activation = MLP_ACTIVATION
    original_mlp_use_feature_norm = MLP_USE_FEATURE_NORMALIZATION
    original_mlp_feature_aug = MLP_FEATURE_AUGMENTATION
    original_mlp_feature_dropout = MLP_FEATURE_DROPOUT_RATE
    original_mlp_feature_noise = MLP_FEATURE_NOISE_SIGMA

    try:
        n_layers = trial.suggest_int('mlp_n_layers', 1, 3)
        temp_hidden_dims = []
        for i in range(n_layers):
            lower_b = max(32, num_classes_mlp // 2)
            upper_b = min(input_size_mlp // 2, 1024)
            if upper_b <= lower_b:
                upper_b = lower_b + 64 # Ensure a minimal range if cap is too aggressive
            if upper_b < 64: # Ensure upper_b is at least a reasonable minimum like 64
                upper_b = 64
            if lower_b >= upper_b : # Final safety for lower_b if it somehow ended up too high
                lower_b = max(32, upper_b // 2)
            temp_hidden_dims.append(trial.suggest_int(f'mlp_h_dim_l{i}', lower_b, upper_b, step=32))
        
        MLP_HIDDEN_DIMS = temp_hidden_dims
        MLP_DROPOUT_RATE = trial.suggest_float('mlp_dropout_rate', 0.4, 0.75, step=0.05) 
        MLP_LEARNING_RATE = trial.suggest_float('mlp_lr', 1e-5, 5e-3, log=True) 
        MLP_WEIGHT_DECAY = trial.suggest_float('mlp_wd', 5e-5, 5e-3, log=True) 
        
        # Add loss type and activation to HPO search
        LOSS_TYPE = trial.suggest_categorical('loss_type', ['CrossEntropy', 'FocalLoss', 'AdaptiveLoss'])
        MLP_ACTIVATION = trial.suggest_categorical('activation', ['ReLU', 'SiLU', 'GELU'])
        
        # Add feature engineering parameters to HPO search
        MLP_USE_FEATURE_NORMALIZATION = trial.suggest_categorical('use_feature_normalization', [True, False])
        MLP_FEATURE_AUGMENTATION = trial.suggest_categorical('feature_augmentation', ['None', 'dropout', 'noise', 'both'])
        
        if MLP_FEATURE_AUGMENTATION in ['dropout', 'both']:
            MLP_FEATURE_DROPOUT_RATE = trial.suggest_float('feature_dropout_rate', 0.01, 0.1)
        
        if MLP_FEATURE_AUGMENTATION in ['noise', 'both']:
            MLP_FEATURE_NOISE_SIGMA = trial.suggest_float('feature_noise_sigma', 0.001, 0.05)

        temp_mlp_model = build_mlp_model(input_size_mlp, num_classes_mlp).to(DEVICE)
        temp_optimizer = get_optimizer(temp_mlp_model, is_mlp_optimizer=True)
        temp_scaler = torch.amp.GradScaler('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda'))
        temp_criterion = get_criterion(class_weights_tensor=CLASS_WEIGHTS, current_label_smoothing=LABEL_SMOOTHING, is_mlp_criterion=True)

        # Convert features to float32 to avoid dtype issues
        features_hpo_train = features_hpo_train.astype(np.float32)
        features_hpo_val = features_hpo_val.astype(np.float32)

        hpo_train_ds = FeatureDataset(features_hpo_train, labels_hpo_train)
        hpo_val_ds = FeatureDataset(features_hpo_val, labels_hpo_val)
        
        hpo_sampler = None
        if IMBALANCE_STRATEGY == 'WeightedSampler' and CLASS_WEIGHTS is not None:
            hpo_labels = hpo_train_ds.get_labels()
            if hpo_labels:
                counts = np.maximum(np.array([hpo_labels.count(l) for l in range(num_classes_mlp)]), 1)
                weights = torch.from_numpy(np.array([1.0 / counts[t] for t in hpo_labels])).double()
                hpo_sampler = WeightedRandomSampler(weights, len(weights))

        hpo_train_loader = DataLoader(hpo_train_ds, MLP_BATCH_SIZE, sampler=hpo_sampler, shuffle=(hpo_sampler is None), num_workers=0, pin_memory=True, drop_last=True)
        hpo_val_loader = DataLoader(hpo_val_ds, MLP_BATCH_SIZE * 2, shuffle=False, num_workers=0, pin_memory=True)
        temp_scheduler = get_scheduler(temp_optimizer, is_mlp_scheduler=True, total_epochs_for_sched=MLP_HPO_EPOCHS)

        best_hpo_trial_val_acc = -float('inf'); epochs_no_improve = 0

        for epoch in range(MLP_HPO_EPOCHS):
            if stop_requested: raise optuna.exceptions.TrialPruned("Global stop")
            
            # Call with explicit return value unpack
            epoch_metrics = train_one_epoch_mlp(temp_mlp_model, hpo_train_loader, temp_criterion, temp_optimizer, temp_scaler, temp_scheduler, 
                            epoch, "HPO", DEVICE, None, num_classes_mlp, None)
            
            # Extract metrics from return value
            if epoch_metrics is None:
                return -float('inf')  # Interrupted
                
            train_loss = epoch_metrics['loss'][0] if 'loss' in epoch_metrics and epoch_metrics['loss'] else float('inf')
            train_acc = epoch_metrics['acc'][0] if 'acc' in epoch_metrics and epoch_metrics['acc'] else 0.0

            val_loss, val_acc = validate_one_epoch_mlp(temp_mlp_model, hpo_val_loader, temp_criterion, DEVICE, epoch, None, num_classes_mlp, 
                                                scheduler=temp_scheduler, fold_num="HPO")
            if val_loss is None: return -float('inf') # Interrupted

            # With single-objective optimization, we can use trial.report
            trial.report(val_acc, epoch)
            
            # Check if the trial should be pruned
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if val_acc > best_hpo_trial_val_acc: best_hpo_trial_val_acc = val_acc; epochs_no_improve = 0
            else: epochs_no_improve += 1
            if epochs_no_improve >= MLP_HPO_PATIENCE: break
        
        del temp_mlp_model, temp_optimizer, temp_scaler, temp_criterion, hpo_train_ds, hpo_val_ds, hpo_train_loader, hpo_val_loader, temp_scheduler
        gc.collect(); torch.cuda.empty_cache()
        
        # Return only the validation accuracy for single-objective optimization
        return best_hpo_trial_val_acc
    finally: # Restore original global MLP parameters
        MLP_LEARNING_RATE = original_mlp_lr
        MLP_WEIGHT_DECAY = original_mlp_wd
        MLP_DROPOUT_RATE = original_mlp_dropout
        MLP_HIDDEN_DIMS = original_mlp_hidden_dims
        MLP_OPTIMIZER_TYPE = original_mlp_optimizer
        MLP_SCHEDULER_TYPE = original_mlp_scheduler
        LOSS_TYPE = original_loss_type
        MLP_ACTIVATION = original_mlp_activation
        MLP_USE_FEATURE_NORMALIZATION = original_mlp_use_feature_norm
        MLP_FEATURE_AUGMENTATION = original_mlp_feature_aug
        MLP_FEATURE_DROPOUT_RATE = original_mlp_feature_dropout
        MLP_FEATURE_NOISE_SIGMA = original_mlp_feature_noise


def run_mlp_hpo(features_hpo, labels_hpo, original_indices_hpo, input_size, num_classes_hpo):
    print(f"\n{TermColors.CYAN}--- Running MLP Hyperparameter Optimization ({MLP_HPO_N_TRIALS} trials) ---{TermColors.ENDC}")
    
    # Create or load study - Change to single objective optimization for simplicity
    study = optuna.create_study(
        direction='maximize',  # Changed from multi-objective to single objective
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=max(1, MLP_HPO_EPOCHS // 4)),
        study_name="mlp_hpo",
        storage=f"sqlite:///{MLP_HPO_STUDY_DB_PATH}",
        load_if_exists=True  # Load existing study if it exists
    )
    
    # Get best trials from previous runs
    best_trials = study.best_trials
    if best_trials:
        print(f"\n{TermColors.GREEN}Found {len(best_trials)} previous best trials in database{TermColors.ENDC}")
        for i, trial in enumerate(best_trials[:3]):  # Show top 3 trials
            print(f"\nBest Trial {i+1}:")
            print(f"  Accuracy: {trial.value:.4f}")
            print("  Hyperparameters:")
            for key, value in trial.params.items():
                print(f"    {key}: {value}")
    
    # Calculate remaining trials needed
    completed_trials = len(study.trials)
    remaining_trials = max(0, MLP_HPO_N_TRIALS - completed_trials)
    
    # Check if we should skip running new trials based on MLP_HPO_LOAD_BEST_ONLY
    should_run_trials = True
    if MLP_HPO_LOAD_BEST_ONLY and best_trials:
        should_run_trials = False
        print(f"\n{TermColors.CYAN}MLP_HPO_LOAD_BEST_ONLY=True and found existing best trials. Skipping new trials.{TermColors.ENDC}")
    elif MLP_HPO_LOAD_BEST_ONLY and not best_trials:
        print(f"\n{TermColors.YELLOW}MLP_HPO_LOAD_BEST_ONLY=True but no best trials found. Will run new trials.{TermColors.ENDC}")
    
    if remaining_trials > 0 and should_run_trials:
        print(f"\n{TermColors.CYAN}Running {remaining_trials} new trials...{TermColors.ENDC}")
        
        # If we have best trials, suggest their parameters for new trials
        if best_trials:
            print(f"{TermColors.GREEN}Using previous best trials as starting points for new trials.{TermColors.ENDC}")
            
            # FIX: Update callback function to accept both study and trial arguments
            def suggest_with_best_trials(study, trial):
                # 70% chance to use best trial parameters as base
                if random.random() < 0.7 and best_trials:
                    best_trial = random.choice(best_trials)
                    params = best_trial.params.copy()
                    
                    # Add smaller variations to the best parameters (5% instead of 10%)
                    for key in params:
                        if isinstance(params[key], (int, float)):
                            params[key] *= random.uniform(0.95, 1.05)
                    
                    return params
                return None
            
            # Create a wrapper function that includes all required arguments
            def hpo_objective_wrapper(trial):
                return mlp_hpo_objective(
                    trial=trial,
                    features_hpo_train=features_hpo,
                    labels_hpo_train=labels_hpo,
                    features_hpo_val=features_hpo,  # Using same data for validation in HPO
                    labels_hpo_val=labels_hpo,      # Using same data for validation in HPO
                    input_size_mlp=input_size,
                    num_classes_mlp=num_classes_hpo
                )
            
            try:
                study.optimize(
                    hpo_objective_wrapper,
                    n_trials=remaining_trials,
                    callbacks=[suggest_with_best_trials]
                )
            except Exception as e:
                print(f"{TermColors.RED}Error during HPO optimization: {e}{TermColors.ENDC}")
        else:
            # Create a wrapper function that includes all required arguments
            def hpo_objective_wrapper(trial):
                return mlp_hpo_objective(
                    trial=trial,
                    features_hpo_train=features_hpo,
                    labels_hpo_train=labels_hpo,
                    features_hpo_val=features_hpo,  # Using same data for validation in HPO
                    labels_hpo_val=labels_hpo,      # Using same data for validation in HPO
                    input_size_mlp=input_size,
                    num_classes_mlp=num_classes_hpo
                )
            
            try:
                study.optimize(
                    hpo_objective_wrapper,
                    n_trials=remaining_trials
                )
            except Exception as e:
                print(f"{TermColors.RED}Error during HPO optimization: {e}{TermColors.ENDC}")
    elif remaining_trials <= 0:
        print(f"\n{TermColors.YELLOW}Already completed {completed_trials} trials (requested {MLP_HPO_N_TRIALS}). Skipping new trials.{TermColors.ENDC}")
    
    # Get best trial
    if study.best_trial:
        best_trial = study.best_trial
        print(f"\n{TermColors.GREEN}Best trial:{TermColors.ENDC}")
        print(f"  Accuracy: {best_trial.value:.4f}")
        print("  Hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
        
        return best_trial.params
    else:
        print(f"\n{TermColors.RED}No best trial found. Using default parameters.{TermColors.ENDC}")
        return None

# --- Knowledge Distillation ---
class DistillationLoss(nn.Module):
    def __init__(self, alpha=KD_ALPHA, temperature=KD_TEMPERATURE, base_criterion=nn.CrossEntropyLoss()):
        super().__init__(); self.alpha = alpha; self.T = temperature; self.base_criterion = base_criterion
        self.KLDiv = nn.KLDivLoss(reduction='batchmean')
    def forward(self, student_outputs, teacher_outputs, labels):
        soft_teacher = F.log_softmax(teacher_outputs / self.T, dim=1)
        soft_student = F.log_softmax(student_outputs / self.T, dim=1)
        distill_loss = self.KLDiv(soft_student, soft_teacher) * (self.alpha * self.T * self.T)
        student_loss = self.base_criterion(student_outputs, labels)
        return distill_loss + (1. - self.alpha) * student_loss

def train_student_model(df_full_kd, num_classes_kd, label_encoder_kd, stacking_teacher_info):
    print(f"\n{TermColors.HEADER}--- Knowledge Distillation with Stacking Teacher (MLP OOFs) ---{TermColors.ENDC}")
    actual_stacking_model, stacking_scaler, feature_extractor_teacher, base_mlps_teacher = None, None, None, []
    teacher_loaded = False
    try:
        print(f"{TermColors.INFO}Loading Stacking Teacher components (MLP-based)...{TermColors.ENDC}")
        stack_pkg = joblib.load(stacking_teacher_info["stacking_model_path"])
        actual_stacking_model = stack_pkg['model']; stacking_scaler = stack_pkg.get('scaler') # Scaler might be None
        
        feature_extractor_teacher = build_feature_extractor_model(num_classes_kd) # Build fresh for teacher
        if stacking_teacher_info["feature_extractor_checkpoint_path"] and os.path.exists(stacking_teacher_info["feature_extractor_checkpoint_path"]):
            ckpt = torch.load(stacking_teacher_info["feature_extractor_checkpoint_path"], map_location=DEVICE)
            fe_state = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in ckpt['state_dict'].items()}
            feature_extractor_teacher.load_state_dict(fe_state, strict=False)
        feature_extractor_teacher = feature_extractor_teacher.to(DEVICE).eval()

        for mlp_path in stacking_teacher_info["base_mlp_model_paths"]:
            if not os.path.exists(mlp_path): print(f"{TermColors.RED}  Base MLP for KD teacher missing: {mlp_path}{TermColors.ENDC}"); continue
            mlp = build_mlp_model(stacking_teacher_info["base_mlp_model_input_size"], num_classes_kd)
            mlp.load_state_dict(torch.load(mlp_path, map_location=DEVICE)); base_mlps_teacher.append(mlp.to(DEVICE).eval())
        
        if actual_stacking_model and feature_extractor_teacher and len(base_mlps_teacher) == stacking_teacher_info["n_folds_for_stacking"]:
            teacher_loaded = True; print(f"{TermColors.GREEN}  All teacher components loaded.{TermColors.ENDC}")
        else: print(f"{TermColors.RED}  Failed to load all teacher components.{TermColors.ENDC}")
    except Exception as e: print(f"{TermColors.RED}Failed to load teacher: {e}. Skip KD.{TermColors.ENDC}"); traceback.print_exc()
    if not teacher_loaded: return

    student_model = CombinedModel([KD_STUDENT_MODEL_NAME], num_classes_kd, True, GLOBAL_POOLING, KD_STUDENT_DROPOUT, KD_STUDENT_EMBEDDING_SIZE, metric_learning='None').to(DEVICE)
    
    df_kd_train, df_kd_val = train_test_split(df_full_kd, test_size=0.2, random_state=SEED + 42, stratify=df_full_kd['label'])
    # Student uses its own image size for transforms
    train_tf_kd = get_transforms(image_size=KD_STUDENT_IMAGE_SIZE, for_feature_extraction=True) # Use FE transforms for student input
    val_tf_kd = get_transforms(image_size=KD_STUDENT_IMAGE_SIZE, for_feature_extraction=True)

    train_ds_kd = PlantDataset(df_kd_train, IMAGE_DIR, train_tf_kd, label_encoder_kd, False, KD_STUDENT_IMAGE_SIZE)
    val_ds_kd = PlantDataset(df_kd_val, IMAGE_DIR, val_tf_kd, label_encoder_kd, False, KD_STUDENT_IMAGE_SIZE)
    if not train_ds_kd or len(train_ds_kd) == 0: print(f"{TermColors.RED}KD Train Dataset empty. Skip KD.{TermColors.ENDC}"); return

    kd_sampler = None # Simplified: no weighted sampler for KD student for now
    train_loader_kd = DataLoader(train_ds_kd, KD_BATCH_SIZE, sampler=kd_sampler, shuffle=(kd_sampler is None), num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader_kd = DataLoader(val_ds_kd, KD_BATCH_SIZE * 2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    criterion_kd = DistillationLoss(base_criterion=nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING))
    optimizer_kd = optim.AdamW(student_model.parameters(), lr=KD_LR, weight_decay=1e-5) # Example WD
    scaler_kd = torch.amp.GradScaler('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda'))
    scheduler_kd = CosineAnnealingLR(optimizer_kd, T_max=KD_EPOCHS, eta_min=KD_LR * 0.01)

    best_kd_val_acc = 0.0
    swa_student_model = None
    ema_student_model = None

    kd_student_ema_decay_fn_instance = None
    if KD_STUDENT_USE_EMA or (KD_STUDENT_USE_SWA and KD_STUDENT_USE_EMA):
        kd_student_ema_decay_fn_instance = ModelExponentialMovingAverage(KD_STUDENT_EMA_DECAY)
    kd_swa_start_ep = int(KD_EPOCHS * KD_STUDENT_SWA_START_EPOCH_FACTOR) if KD_STUDENT_USE_SWA else KD_EPOCHS 
    if KD_STUDENT_USE_SWA:
        current_kd_swa_avg_fn = kd_student_ema_decay_fn_instance if KD_STUDENT_USE_EMA else None
        swa_student_model = AveragedModel(student_model, avg_fn=current_kd_swa_avg_fn)
    if KD_STUDENT_USE_EMA:
        ema_student_model = AveragedModel(student_model, avg_fn=kd_student_ema_decay_fn_instance)
    swa_scheduler_kd = SWALR(optimizer_kd, swa_lr=(KD_LR * KD_STUDENT_SWA_LR_FACTOR), anneal_epochs=KD_STUDENT_SWA_ANNEAL_EPOCHS) if KD_STUDENT_USE_SWA else None

    print(f"{TermColors.CYAN}Starting KD Student Training ({KD_EPOCHS} epochs)...{TermColors.ENDC}")
    for epoch in range(KD_EPOCHS):
        if stop_requested: break
        student_model.train(); running_loss_kd = 0.0; total_samples_kd = 0
        progress_bar = tqdm(train_loader_kd, desc=f"KD Student Train E{epoch+1}", leave=False)
        for inputs, labels in progress_bar:
            if stop_requested: break
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE); batch_size = inputs.size(0)
            teacher_logits = None
            with torch.no_grad():
                img_feats_teacher = feature_extractor_teacher(inputs, return_embedding=True)
                mlp_preds_teacher = [mlp(img_feats_teacher) for mlp in base_mlps_teacher]
                # Assuming stacking model takes concatenated probabilities or logits. Here, using probabilities.
                stacker_input_probs = torch.cat([F.softmax(p, dim=1) for p in mlp_preds_teacher], dim=1)
                stacker_input_np = stacker_input_probs.cpu().numpy()
                if stacking_scaler: stacker_input_np = stacking_scaler.transform(stacker_input_np)
                teacher_probs_np = actual_stacking_model.predict_proba(stacker_input_np)
                teacher_logits = torch.log(torch.tensor(teacher_probs_np, dtype=torch.float32).to(DEVICE) + 1e-9)

            if teacher_logits is None: continue
            with torch.amp.autocast('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda')):
                student_logits = student_model(inputs)
                loss = criterion_kd(student_logits, teacher_logits, labels)
            
            optimizer_kd.zero_grad(); scaler_kd.scale(loss).backward(); scaler_kd.step(optimizer_kd); scaler_kd.update()
            if KD_STUDENT_USE_EMA and ema_student_model: ema_student_model.update_parameters(student_model)
            if not torch.isnan(loss): running_loss_kd += loss.item() * batch_size; total_samples_kd += batch_size
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        if stop_requested: break
        print(f"KD Student E{epoch+1} Train Loss: {running_loss_kd/total_samples_kd if total_samples_kd > 0 else 0:.4f}")

        # Validation
        student_model.eval()
        if swa_student_model: swa_student_model.eval()
        if ema_student_model: ema_student_model.eval()
        eval_models_kd = {'base_kd': student_model}
        if KD_STUDENT_USE_SWA and swa_student_model and epoch >= kd_swa_start_ep: eval_models_kd['swa_kd'] = swa_student_model
        if KD_STUDENT_USE_EMA and ema_student_model: eval_models_kd['ema_kd'] = ema_student_model
        
        best_model_key_this_epoch_kd = 'base_kd'; current_best_val_acc_this_epoch_kd = 0.0
        for key, eval_model in eval_models_kd.items():
            eval_model.eval(); val_loss_kd, val_acc_kd = 0,0; val_samples = 0; all_p, all_l = [],[]
            with torch.no_grad():
                for inputs_v, labels_v in tqdm(val_loader_kd, desc=f"KD Val E{epoch+1} ({key})", leave=False):
                    inputs_v, labels_v = inputs_v.to(DEVICE), labels_v.to(DEVICE)
                    with torch.amp.autocast('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda')):
                        outputs_v = eval_model(inputs_v)
                        val_loss_kd += F.cross_entropy(outputs_v, labels_v).item() * inputs_v.size(0)
                    all_p.append(torch.argmax(outputs_v, dim=1).cpu()); all_l.append(labels_v.cpu()); val_samples += inputs_v.size(0)
            val_loss_kd /= val_samples if val_samples > 0 else 1
            val_acc_kd = (torch.cat(all_p) == torch.cat(all_l)).sum().item() / val_samples if val_samples > 0 else 0
            print(f"  KD Student Val ({key}) E{epoch+1}: Loss={val_loss_kd:.4f}, Acc={val_acc_kd:.4f}")
            if val_acc_kd > current_best_val_acc_this_epoch_kd: current_best_val_acc_this_epoch_kd = val_acc_kd; best_model_key_this_epoch_kd = key
        
        if current_best_val_acc_this_epoch_kd > best_kd_val_acc:
            best_kd_val_acc = current_best_val_acc_this_epoch_kd
            print(f"{TermColors.OKGREEN}New best KD student val acc: {best_kd_val_acc:.4f} (from {best_model_key_this_epoch_kd}). Saving...{TermColors.ENDC}")
            save_path = KD_STUDENT_MODEL_SAVE_PATH
            model_state_to_save = student_model.state_dict()
            if best_model_key_this_epoch_kd == 'swa_kd' and swa_student_model: save_path=KD_STUDENT_SWA_MODEL_SAVE_PATH; model_state_to_save=swa_student_model.module.state_dict()
            elif best_model_key_this_epoch_kd == 'ema_kd' and ema_student_model: save_path=KD_STUDENT_EMA_MODEL_SAVE_PATH; model_state_to_save=ema_student_model.module.state_dict()
            torch.save(model_state_to_save, save_path)

        if KD_STUDENT_USE_SWA and swa_student_model and epoch >= kd_swa_start_ep:
            swa_student_model.update_parameters(student_model)
            if swa_scheduler_kd: swa_scheduler_kd.step()
        elif scheduler_kd: scheduler_kd.step()
        gc.collect(); torch.cuda.empty_cache()

    if KD_STUDENT_USE_SWA and swa_student_model and not stop_requested:
        print(f"{TermColors.CYAN}KD Student: Updating FINAL SWA BN...{TermColors.ENDC}")
        bn_loader = DataLoader(train_ds_kd, KD_BATCH_SIZE*2, shuffle=True, num_workers=NUM_WORKERS)
        try: torch.optim.swa_utils.update_bn(bn_loader, swa_student_model.to(DEVICE), device=DEVICE); torch.save(swa_student_model.module.state_dict(), KD_STUDENT_SWA_MODEL_SAVE_PATH)
        except Exception as e: print(f"{TermColors.RED}KD SWA BN/Save Error: {e}{TermColors.ENDC}")
    if KD_STUDENT_USE_EMA and ema_student_model and not stop_requested:
        try: torch.save(ema_student_model.module.state_dict(), KD_STUDENT_EMA_MODEL_SAVE_PATH)
        except Exception as e: print(f"{TermColors.RED}KD EMA Save Error: {e}{TermColors.ENDC}")
    if not os.path.exists(KD_STUDENT_MODEL_SAVE_PATH) and not stop_requested: torch.save(student_model.state_dict(), KD_STUDENT_MODEL_SAVE_PATH)
    print(f"{TermColors.OKGREEN}KD finished. Best student val_acc: {best_kd_val_acc:.4f}{TermColors.ENDC}")
    del student_model, train_loader_kd, val_loader_kd, feature_extractor_teacher, base_mlps_teacher; gc.collect(); torch.cuda.empty_cache()

# --- Feature Extraction Function ---
def extract_all_features(df_full_fe, num_classes_fe, label_encoder_fe):
    global NUM_CLASSES, label_encoder, class_names # These are used by PlantDataset if label_encoder_instance is None
    NUM_CLASSES = num_classes_fe; label_encoder = label_encoder_fe; class_names = list(label_encoder.classes_)

    print(f"\n{TermColors.HEADER}--- STEP: Feature Extraction ---{TermColors.ENDC}")
    print(f"{TermColors.INFO}Feature Extraction Mixed Precision: {FEATURE_EXTRACTION_MIXED_PRECISION}{TermColors.ENDC}") # Log FE specific setting
    feature_extractor = build_feature_extractor_model(NUM_CLASSES) # Dropout 0, MetricLearning None by default
    if FEATURE_EXTRACTOR_CHECKPOINT_PATH and os.path.exists(FEATURE_EXTRACTOR_CHECKPOINT_PATH):
        print(f"{TermColors.CYAN}Loading FE checkpoint: {FEATURE_EXTRACTOR_CHECKPOINT_PATH}{TermColors.ENDC}")
        try:
            ckpt = torch.load(FEATURE_EXTRACTOR_CHECKPOINT_PATH, map_location=DEVICE)
            state_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in ckpt['state_dict'].items()}
            feature_extractor.load_state_dict(state_dict, strict=False)
        except Exception as e: print(f"{TermColors.RED}FE Ckpt Load Fail: {e}. Using pretrained.{TermColors.ENDC}")
    feature_extractor = feature_extractor.to(DEVICE).eval()

    extraction_transform = get_transforms(image_size=FEATURE_EXTRACTION_IMAGE_SIZE_CONFIG, for_feature_extraction=True)
    full_ds = PlantDataset(df_full_fe, IMAGE_DIR, extraction_transform, label_encoder, True, FEATURE_EXTRACTION_IMAGE_SIZE_CONFIG)
    if len(full_ds) == 0: print(f"{TermColors.RED}FE Dataset empty. Abort FE.{TermColors.ENDC}"); return
    pin_memory_fe = True if DEVICE.type == 'cuda' else False
    dataloader = DataLoader(full_ds, FEATURE_EXTRACTOR_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin_memory_fe)


    all_feats, all_lbls, all_orig_idx = [], [], []
    with torch.no_grad():
        for inputs, lbls_batch, _, orig_idx_batch in tqdm(dataloader, desc="Extracting Features"):
            inputs = inputs.to(DEVICE)
            # Use FEATURE_EXTRACTION_MIXED_PRECISION for autocast
            with torch.amp.autocast('cuda', enabled=(FEATURE_EXTRACTION_MIXED_PRECISION and DEVICE.type == 'cuda')):
                embeddings = feature_extractor(inputs, return_embedding=True) 
            all_feats.append(embeddings.cpu().numpy()); all_lbls.append(lbls_batch.numpy()); all_orig_idx.append(orig_idx_batch.numpy())

    if not all_feats: print(f"{TermColors.RED}No features extracted. Abort.{TermColors.ENDC}"); return
    feats_np = np.concatenate(all_feats); lbls_np = np.concatenate(all_lbls); orig_idx_np = np.concatenate(all_orig_idx)
    print(f"{TermColors.GREEN}FE complete. Extracted {feats_np.shape[0]} features, dim {feats_np.shape[1]}.{TermColors.ENDC}")
    try:
        np.savez_compressed(FEATURES_NPZ_PATH, features=feats_np, labels=lbls_np, original_indices=orig_idx_np, label_encoder_classes=np.array(list(label_encoder.classes_)))
        print(f"{TermColors.GREEN}Features saved: {FEATURES_NPZ_PATH}{TermColors.ENDC}")
    except Exception as e: print(f"{TermColors.RED}Error saving features: {e}{TermColors.ENDC}")
    del feature_extractor, dataloader, full_ds; gc.collect(); torch.cuda.empty_cache()

def regenerate_oof_from_fold_models(num_classes_regen, label_encoder_regen, class_names_regen, class_weight_regen):
    global NUM_CLASSES, label_encoder, class_names, CLASS_WEIGHTS
    global MLP_HIDDEN_DIMS, MLP_DROPOUT_RATE, MLP_LEARNING_RATE, MLP_WEIGHT_DECAY 
    global STACKING_HPO_CV_FOLDS  # Add global declaration at the top of the function
    
    NUM_CLASSES = num_classes_regen
    label_encoder = label_encoder_regen
    class_names = class_names_regen
    CLASS_WEIGHTS = class_weight_regen

    print(f"\n{TermColors.HEADER}--- STEP: Regenerate OOF Predictions from Existing Fold Models ---{TermColors.ENDC}")

    original_mlp_hidden_dims = MLP_HIDDEN_DIMS[:] # Use slicing for lists to copy
    original_mlp_dropout_rate = MLP_DROPOUT_RATE
    original_mlp_lr = MLP_LEARNING_RATE         # Store if HPO modified it
    original_mlp_wd = MLP_WEIGHT_DECAY          # Store if HPO modified it
    successful_hpo_param_load = False
    if MLP_DO_HPO and OPTUNA_AVAILABLE and os.path.exists(MLP_HPO_STUDY_DB_PATH):
        print(f"{TermColors.INFO}Attempting to load HPO-derived MLP parameters for model architecture...{TermColors.ENDC}")
        try:
            study_name_hpo_load = "mlp_plant_recognition_hpo_study"
            storage_name_hpo_load = f"sqlite:///{MLP_HPO_STUDY_DB_PATH}"
            hpo_study = optuna.load_study(study_name=study_name_hpo_load, storage=storage_name_hpo_load)
            if hpo_study.best_trial:
                best_hpo_params = hpo_study.best_trial.params
                # Update global MLP parameters with HPO results for the duration of this function
                MLP_HIDDEN_DIMS = [best_hpo_params[f'mlp_h_dim_l{i}'] for i in range(best_hpo_params['mlp_n_layers'])]
                MLP_DROPOUT_RATE = best_hpo_params['mlp_dropout_rate']
                MLP_LEARNING_RATE = best_hpo_params.get('mlp_lr', original_mlp_lr) # .get in case 'mlp_lr' wasn't in this specific study's params
                MLP_WEIGHT_DECAY = best_hpo_params.get('mlp_wd', original_mlp_wd) # .get in case 'mlp_wd' wasn't in this specific study's params
                
                print(f"{TermColors.GREEN}  Successfully loaded and TEMPORARILY applied HPO params for regeneration:{TermColors.ENDC}")
                print(f"{TermColors.GREEN}    HiddenDims={MLP_HIDDEN_DIMS}, Dropout={MLP_DROPOUT_RATE}, LR={MLP_LEARNING_RATE}, WD={MLP_WEIGHT_DECAY}{TermColors.ENDC}")
                successful_hpo_param_load = True
            else:
                print(f"{TermColors.YELLOW}  MLP HPO study found, but no best trial. Using current global MLP params for regeneration.{TermColors.ENDC}")
        except Exception as e_hpo_load:
            print(f"{TermColors.YELLOW}  Could not load MLP HPO study/params for regeneration: {e_hpo_load}. Using current global MLP params.{TermColors.ENDC}")
    else:
        print(f"{TermColors.INFO}MLP HPO not enabled or study DB not found. Using current global MLP params for regeneration.{TermColors.ENDC}")
    if not os.path.exists(FEATURES_NPZ_PATH):
        print(f"{TermColors.RED}Features file not found: {FEATURES_NPZ_PATH}. Cannot regenerate OOFs. Aborting this step.{TermColors.ENDC}")
        # Restore globals before returning
        MLP_HIDDEN_DIMS = original_mlp_hidden_dims; MLP_DROPOUT_RATE = original_mlp_dropout_rate
        MLP_LEARNING_RATE = original_mlp_lr; MLP_WEIGHT_DECAY = original_mlp_wd
        return None
    
    try:
        data = np.load(FEATURES_NPZ_PATH, allow_pickle=True)
        feats_all, lbls_all, orig_idx_all = data['features'], data['labels'], data['original_indices']
        
        # Explicitly convert features to float32 to avoid dtype mismatches
        feats_all = feats_all.astype(np.float32)
        
        if not np.array_equal(list(label_encoder.classes_), data['label_encoder_classes']):
            print(f"{TermColors.RED}Label encoder mismatch in features file! Aborting OOF regeneration.{TermColors.ENDC}")
            return None
        print(f"{TermColors.GREEN}Features loaded for OOF regeneration: {feats_all.shape[0]} samples.{TermColors.ENDC}")
    except Exception as e:
        print(f"{TermColors.RED}Error loading features for OOF regeneration: {e}{TermColors.ENDC}")
        return None

    feats_kfold, lbls_kfold, orig_idx_kfold = feats_all, lbls_all, orig_idx_all
    
    if len(feats_kfold) < N_FOLDS * 2 : # Basic check
        print(f"{TermColors.RED}Not enough data in features file for K-Fold ({len(feats_kfold)}). Abort OOF regeneration.{TermColors.ENDC}")
        return None

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds = np.full((len(orig_idx_kfold), NUM_CLASSES), np.nan, dtype=np.float32)
    oof_lbls = np.full(len(orig_idx_kfold), -1, dtype=np.int32)
    
    all_fold_models_loaded_and_validated = True

    for fold, (train_idx, val_idx) in enumerate(skf.split(feats_kfold, lbls_kfold)):
        if stop_requested: break
        print(f"\n{TermColors.CYAN}Regenerating OOF for Fold {fold+1}/{N_FOLDS}...{TermColors.ENDC}")
        
        val_feats = feats_kfold[val_idx]
        val_lbls = lbls_kfold[val_idx] # True labels for this validation fold
        val_orig_ids_fold = orig_idx_kfold[val_idx]
        
        model_path = os.path.join(BASE_MODEL_SAVE_DIR, f"mlp_fold_{fold}", "best_mlp_model_state_dict.pth")
        if not os.path.exists(model_path):
            print(f"{TermColors.RED}Model for fold {fold+1} not found at {model_path}. Cannot regenerate OOFs.{TermColors.ENDC}")
            all_fold_models_loaded_and_validated = False
            break
            
        try:
            mlp_model = build_mlp_model(feats_kfold.shape[1], NUM_CLASSES).to(DEVICE)
            mlp_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            mlp_model.eval()

            crit_regen = get_criterion(CLASS_WEIGHTS, LABEL_SMOOTHING, is_mlp_criterion=True) # For validate_one_epoch_mlp
            
            val_ds_mlp = FeatureDataset(val_feats, val_lbls, val_orig_ids_fold)
            val_loader_mlp = DataLoader(val_ds_mlp, MLP_BATCH_SIZE * 2, shuffle=False, num_workers=0, pin_memory=True)
            _, _, fold_oof_preds, fold_oof_indices, fold_oof_labels = validate_one_epoch_mlp(
                mlp_model, val_loader_mlp, crit_regen, DEVICE, 0, None, NUM_CLASSES, return_preds=True, fold_num=f"{fold+1}-OOF_Regen"
            )

            if fold_oof_preds is not None and fold_oof_indices is not None:
                for i, single_orig_id_val in enumerate(fold_oof_indices):
                    # Find where this original_id maps to in the global oof_preds/oof_lbls arrays
                    global_oof_match_idx = np.where(orig_idx_kfold == single_orig_id_val)[0]
                    if len(global_oof_match_idx) > 0:
                        global_idx = global_oof_match_idx[0]
                        oof_preds[global_idx] = fold_oof_preds[i]
                        
                        # Find the matching validation index for the correct label
                        val_idx_match = np.where(val_orig_ids_fold == single_orig_id_val)[0]
                        if len(val_idx_match) > 0:
                            # Use the correct validation label 
                            oof_lbls[global_idx] = val_lbls[val_idx_match[0]]
                        else:
                            # Fallback to global label
                            oof_lbls[global_idx] = lbls_kfold[global_idx]
                            print(f"{TermColors.YELLOW}Regeneration: Val ID {single_orig_id_val} not in val_orig_ids_fold. Using global label.{TermColors.ENDC}")
                    else:
                        print(f"{TermColors.RED}Warning: Regeneration - original ID {single_orig_id_val} not found in global orig_idx_kfold.{TermColors.ENDC}")
                print(f"{TermColors.GREEN}  OOF for Fold {fold+1} regenerated.{TermColors.ENDC}")
            else:
                print(f"{TermColors.RED}  Failed to get OOF predictions for Fold {fold+1}.{TermColors.ENDC}")
                all_fold_models_loaded_and_validated = False; break
            
            del mlp_model, val_ds_mlp, val_loader_mlp, crit_regen; gc.collect(); torch.cuda.empty_cache()

        except Exception as e:
            print(f"{TermColors.RED}Error regenerating OOF for Fold {fold+1}: {e}{TermColors.ENDC}"); traceback.print_exc()
            all_fold_models_loaded_and_validated = False; break
            
    if stop_requested:
        print(f"{TermColors.YELLOW}OOF regeneration interrupted.{TermColors.ENDC}")
        return None

    # Restore original global MLP parameters
    MLP_HIDDEN_DIMS = original_mlp_hidden_dims
    MLP_DROPOUT_RATE = original_mlp_dropout_rate
    MLP_LEARNING_RATE = original_mlp_lr
    MLP_WEIGHT_DECAY = original_mlp_wd
    if successful_hpo_param_load:
        print(f"{TermColors.INFO}Restored original global MLP parameters after OOF regeneration.{TermColors.ENDC}")

    if not all_fold_models_loaded_and_validated:
        print(f"{TermColors.RED}OOF regeneration failed for one or more folds. Cannot proceed with stacking.{TermColors.ENDC}")
        return None

    valid_oof_idx = np.where(oof_lbls != -1)[0]
    if len(valid_oof_idx) > 0:
        final_oof_p = oof_preds[valid_oof_idx]
        final_oof_l = oof_lbls[valid_oof_idx]
        final_oof_orig_id = orig_idx_kfold[valid_oof_idx]
        
        print(f"{TermColors.CYAN}--- OOF Label Diagnostics (After Regeneration) ---{TermColors.ENDC}")
        unique_final_labels, counts_final_labels = np.unique(final_oof_l, return_counts=True)
        print(f"{TermColors.INFO}Total unique classes in final_oof_l: {len(unique_final_labels)} of {NUM_CLASSES} expected{TermColors.ENDC}")
        print(f"{TermColors.INFO}Class counts (first 10 classes): {list(zip(unique_final_labels[:10], counts_final_labels[:10]))}{TermColors.ENDC}")
        
        if len(unique_final_labels) < NUM_CLASSES:
            missing_classes = set(range(NUM_CLASSES)) - set(unique_final_labels)
            print(f"{TermColors.WARNING}WARNING: Missing classes in regenerated OOF: {missing_classes}{TermColors.ENDC}")
            
        min_count_in_oof = np.min(counts_final_labels) if len(counts_final_labels) > 0 else 0
        print(f"{TermColors.INFO}Min samples for any class in final_oof_l: {min_count_in_oof}, STACKING_HPO_CV_FOLDS: {STACKING_HPO_CV_FOLDS}{TermColors.ENDC}")
        
        # If any class has fewer than needed CV folds, adjust the global parameter
        if min_count_in_oof < STACKING_HPO_CV_FOLDS:
            old_cv_folds = STACKING_HPO_CV_FOLDS
            STACKING_HPO_CV_FOLDS = max(2, min_count_in_oof-1)  # Adjust to be at most min_count-1, minimum 2
            print(f"{TermColors.WARNING}WARNING: Adjusting STACKING_HPO_CV_FOLDS from {old_cv_folds} to {STACKING_HPO_CV_FOLDS} based on minimum class count {min_count_in_oof}{TermColors.ENDC}")
        
        np.savez_compressed(STACKING_OOF_PREDS_PATH, preds=final_oof_p, labels=final_oof_l, original_indices=final_oof_orig_id)
        print(f"{TermColors.GREEN}Regenerated MLP OOF preds saved: {STACKING_OOF_PREDS_PATH} ({len(final_oof_p)} samples from {len(unique_final_labels)} classes){TermColors.ENDC}")
        return STACKING_OOF_PREDS_PATH
    else:
        print(f"{TermColors.RED}No valid OOFs generated after regeneration attempt.{TermColors.ENDC}")
        return None

# --- Main MLP Training Loop ---
def train_mlp_on_features_main_loop(num_classes_mlp, label_encoder_mlp, class_names_mlp, class_freq_mlp, class_prior_mlp, class_weight_mlp):
    global NUM_CLASSES, label_encoder, class_names, CLASS_FREQUENCIES, CLASS_PRIORS, CLASS_WEIGHTS
    global STACKING_HPO_CV_FOLDS  # Add global declaration here
    
    NUM_CLASSES=num_classes_mlp; label_encoder=label_encoder_mlp; class_names=class_names_mlp
    CLASS_FREQUENCIES=class_freq_mlp; CLASS_PRIORS=class_prior_mlp; CLASS_WEIGHTS=class_weight_mlp

    print(f"\n{TermColors.HEADER}--- Train MLP on Extracted Features ---{TermColors.ENDC}")
    final_stacking_path = None
    
    try:
        data = np.load(FEATURES_NPZ_PATH, allow_pickle=True)
        feats_all, lbls_all, orig_idx_all = data['features'], data['labels'], data['original_indices']
        
        # Explicitly convert features to float32 to avoid dtype mismatches
        feats_all = feats_all.astype(np.float32)
        
        if not np.array_equal(list(label_encoder.classes_), data['label_encoder_classes']):
            print(f"{TermColors.RED}Label encoder mismatch in features file! Abort.{TermColors.ENDC}"); return final_stacking_path
        print(f"{TermColors.GREEN}Features loaded: {feats_all.shape[0]} samples.{TermColors.ENDC}")
    except Exception as e: print(f"{TermColors.RED}Error loading features: {e}{TermColors.ENDC}"); return final_stacking_path

    feats_kfold, lbls_kfold, orig_idx_kfold = feats_all, lbls_all, orig_idx_all
    if MLP_DO_HPO and OPTUNA_AVAILABLE:
        min_hpo_samples = int(N_FOLDS / (1.0 - MLP_HPO_DATA_SPLIT_RATIO)) + N_FOLDS
        if len(feats_all) < min_hpo_samples or len(feats_all) * MLP_HPO_DATA_SPLIT_RATIO < 20:
            print(f"{TermColors.YELLOW}Not enough data for robust MLP HPO split ({len(feats_all)}). Skip HPO.{TermColors.ENDC}")
        else:
            try:
                feats_kfold, feats_hpo, lbls_kfold, lbls_hpo, orig_idx_kfold, orig_idx_hpo = train_test_split(
                    feats_all, lbls_all, orig_idx_all, test_size=MLP_HPO_DATA_SPLIT_RATIO, random_state=SEED+99, stratify=lbls_all)
                print(f"{TermColors.INFO}Data split for MLP: HPO set: {len(feats_hpo)}, K-Fold set: {len(feats_kfold)}{TermColors.ENDC}")
                run_mlp_hpo(feats_hpo, lbls_hpo, orig_idx_hpo, feats_all.shape[1], NUM_CLASSES)
            except Exception as e: print(f"{TermColors.RED}MLP HPO split/run error: {e}. Default params.{TermColors.ENDC}")
    
    # Make sure features are float32 after the split as well
    feats_kfold = feats_kfold.astype(np.float32)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_results = defaultdict(list)
    
    # Initialize OOF arrays with the right size
    oof_preds = np.full((len(orig_idx_kfold), NUM_CLASSES), np.nan, dtype=np.float32) 
    oof_lbls = np.full(len(orig_idx_kfold), -1, dtype=np.int32) 
    
    # Track multi-initialization ensembles if enabled
    multi_init_models = {}  # Format: {fold_num: [models]}
    
    # NEW: Track per-class metrics across all folds and epochs
    per_class_metrics = {}  # Format: {fold_num: {epoch: {class_id: {'precision': val, 'recall': val, 'f1': val, 'accuracy': val, 'predictions': array}}}}
    best_epoch_per_class = {}  # Format: {fold_num: {class_id: best_epoch}}
    
    # Track predictions from all folds for deeper analysis
    fold_predictions = []
    fold_indices = []
    fold_labels = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(feats_kfold, lbls_kfold)):
        if stop_requested: break
        print(f"\n{TermColors.HEADER} MLP Fold {fold+1}/{N_FOLDS} {TermColors.ENDC}")
        train_feats, val_feats = feats_kfold[train_idx], feats_kfold[val_idx]
        train_lbls, val_lbls = lbls_kfold[train_idx], lbls_kfold[val_idx] 
        val_orig_ids_fold = orig_idx_kfold[val_idx]
        
        # Initialize per-class metrics for this fold
        per_class_metrics[fold] = {}
        best_epoch_per_class[fold] = {}
        
        # Initialize list for multiple model initializations for this fold
        multi_init_models[fold] = []
        
        # Diagnostics: check class distribution in validation fold
        val_unique_labels, val_counts = np.unique(val_lbls, return_counts=True)
        print(f"{TermColors.CYAN}Fold {fold+1} validation set: {len(val_lbls)} samples, {len(val_unique_labels)} classes{TermColors.ENDC}")
        print(f"  Min class count in validation: {np.min(val_counts)}")
        low_count_val_classes = [(label, count) for label, count in zip(val_unique_labels, val_counts) if count <= 2]
        if low_count_val_classes:
            print(f"{TermColors.YELLOW}  Classes with <= 2 samples in validation fold {fold+1}: {low_count_val_classes[:10]}{TermColors.ENDC}")

        # Train each initialization of the model with a different seed
        for init_idx in range(MLP_MULTI_INIT_ENSEMBLE):
            gc.collect(); torch.cuda.empty_cache()
            
            if MLP_MULTI_INIT_ENSEMBLE > 1:
                print(f"\n{TermColors.CYAN}Fold {fold+1} Model Initialization {init_idx+1}/{MLP_MULTI_INIT_ENSEMBLE}{TermColors.ENDC}")
                # Set a different seed for each initialization
                init_seed = SEED + (fold * 100) + (init_idx * 10)
                torch.manual_seed(init_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(init_seed)
            else:
                # Use the main seed if not doing multiple initializations
                init_seed = SEED
                
            mlp, opt, sched, scaler_mlp, crit, swa_mlp, ema_mlp = None,None,None,None,None,None,None
            try:
                mlp = build_mlp_model(feats_kfold.shape[1], NUM_CLASSES).to(DEVICE)
                if USE_TORCH_COMPILE and hasattr(torch, 'compile'):
                    try: mlp = torch.compile(mlp, mode='default')
                    except: print(f"{TermColors.RED}torch.compile for MLP failed.{TermColors.ENDC}")
                crit = get_criterion(CLASS_WEIGHTS, LABEL_SMOOTHING, is_mlp_criterion=True)
                opt = get_optimizer(mlp, is_mlp_optimizer=True)
                scaler_mlp = torch.amp.GradScaler('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda'))
                
                mlp_ema_decay_fn_instance = None
                if MLP_USE_EMA or (MLP_USE_SWA and MLP_USE_EMA): # Create instance if EMA is used anywhere for MLP
                    mlp_ema_decay_fn_instance = ModelExponentialMovingAverage(MLP_EMA_DECAY)
    
                if MLP_USE_SWA:
                    current_swa_avg_fn = mlp_ema_decay_fn_instance if MLP_USE_EMA else None
                    swa_mlp = AveragedModel(mlp, avg_fn=current_swa_avg_fn)
                if MLP_USE_EMA:
                    ema_mlp = AveragedModel(mlp, avg_fn=mlp_ema_decay_fn_instance)
            except Exception as e: 
                print(f"{TermColors.RED}MLP Fold {fold+1} Init {init_idx+1} Setup Error: {e}{TermColors.ENDC}")
                continue
    
            start_ep, best_met, _ = load_checkpoint(fold, mlp, opt, None, scaler_mlp, f"latest_mlp_checkpoint_init{init_idx}.pth.tar", True)
            if best_met == (float('-inf') if CHECKPOINT_MODE == 'max' else float('inf')): # if latest was not good
                start_ep, best_met, _ = load_checkpoint(fold, mlp, opt, None, scaler_mlp, f"best_mlp_model_init{init_idx}.pth.tar", True)
    
            writer_mlp = SummaryWriter(log_dir=os.path.join(BASE_LOG_DIR, f"mlp_fold_{fold}_init{init_idx}"))
            fold_stop, best_val_loss, best_val_acc, best_ep, no_improve_epochs = False, float('inf'), 0.0, -1, 0
            best_met_early_stop = best_met
            
            train_ds_mlp = FeatureDataset(train_feats, train_lbls, orig_idx_kfold[train_idx])
            val_ds_mlp = FeatureDataset(val_feats, val_lbls, val_orig_ids_fold)
            sampler_mlp = None
            if IMBALANCE_STRATEGY == 'WeightedSampler' and CLASS_WEIGHTS is not None:
                lbls_list = train_ds_mlp.get_labels()
                if lbls_list:
                    counts = np.maximum(np.array([lbls_list.count(l) for l in range(NUM_CLASSES)]), 1)
                    w = torch.from_numpy(np.array([1.0/counts[t] for t in lbls_list])).double()
                    sampler_mlp = WeightedRandomSampler(w, len(w))
            
            train_loader_mlp = DataLoader(train_ds_mlp, MLP_BATCH_SIZE, sampler=sampler_mlp, shuffle=(sampler_mlp is None), num_workers=0, pin_memory=True, drop_last=True)
            val_loader_mlp = DataLoader(val_ds_mlp, MLP_BATCH_SIZE*2, shuffle=False, num_workers=0, pin_memory=True)
            err_loader_mlp = DataLoader(val_ds_mlp, ERROR_LOG_BATCH_SIZE, shuffle=False, num_workers=0)
            sched = get_scheduler(opt, is_mlp_scheduler=True, total_epochs_for_sched=MLP_EPOCHS)
            
            # Reload scheduler state if resuming
            if start_ep > 0 and sched:
                ckpt_path_s = os.path.join(BASE_CHECKPOINT_DIR, f"mlp_fold_{fold}", f"latest_mlp_checkpoint_init{init_idx}.pth.tar")
                if os.path.isfile(ckpt_path_s):
                    ckpt_s = torch.load(ckpt_path_s, map_location=DEVICE)
                    if 'scheduler' in ckpt_s and ckpt_s['scheduler']:
                        try: sched.load_state_dict(ckpt_s['scheduler'])
                        except: print(f"{TermColors.YELLOW}MLP Sched reload failed.{TermColors.ENDC}")
    
            swa_sched_mlp = SWALR(opt, swa_lr=(MLP_LEARNING_RATE*0.05), anneal_epochs=5) if MLP_USE_SWA and swa_mlp else None
            mlp_swa_start = int(MLP_EPOCHS * MLP_SWA_START_EPOCH_FACTOR)
            last_ep_fold = start_ep - 1
    
            for epoch in range(start_ep, MLP_EPOCHS):
                if fold_stop or stop_requested: break
                print(f"\n{TermColors.CYAN} MLP F{fold+1} I{init_idx+1} Ep {epoch+1}/{MLP_EPOCHS} {TermColors.ENDC}")
                
                # Call train_one_epoch_mlp and handle its dictionary return value
                epoch_metrics = train_one_epoch_mlp(mlp, train_loader_mlp, crit, opt, scaler_mlp, sched, epoch, 
                                                   f"{fold+1}-I{init_idx+1}", DEVICE, writer_mlp, NUM_CLASSES, ema_mlp)
                
                if epoch_metrics is None: 
                    fold_stop = True
                    break
                
                # Extract metrics from the returned dictionary
                tr_loss = epoch_metrics['loss'][0] if 'loss' in epoch_metrics and epoch_metrics['loss'] else float('inf')
                tr_acc = epoch_metrics['acc'][0] if 'acc' in epoch_metrics and epoch_metrics['acc'] else 0.0
                
                # Always collect OOF predictions during validation
                v_loss, v_acc, oof_p_curr, oof_idx_curr, oof_labels_curr = validate_one_epoch_mlp(
                    mlp, val_loader_mlp, crit, DEVICE, epoch, writer_mlp, NUM_CLASSES, 
                    sched, swa_mlp, ema_mlp, True, f"{fold+1}-I{init_idx+1}"
                )
                
                if v_loss is None: fold_stop = True; break
                print(f"MLP F{fold+1} I{init_idx+1} Ep {epoch+1}: Tr L={tr_loss:.4f} A={tr_acc:.4f} | Val L={v_loss:.4f} A={v_acc:.4f} (Best {CHECKPOINT_MONITOR}: {best_met:.4f})")
    
                # Multi-initialization ensemble metrics tracking
                if init_idx == 0:  # Only track per-class metrics for the first initialization
                    # NEW: Process and update per-class metrics if predictions were collected this epoch
                    if oof_p_curr is not None and oof_labels_curr is not None:
                        # Calculate per-class metrics
                        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
                        
                        # Initialize metrics for this epoch
                        per_class_metrics[fold][epoch] = {}
                        
                        # Process predictions for each class
                        hard_preds = np.argmax(oof_p_curr, axis=1)
                        
                        # Get unique classes in this validation fold
                        unique_classes = np.unique(oof_labels_curr)
                        
                        for class_id in unique_classes:
                            # Create binary classification problem for this class
                            class_mask = (oof_labels_curr == class_id)
                            
                            if np.sum(class_mask) == 0:
                                continue  # Skip if no samples for this class
                            
                            # Get binary predictions for this class
                            binary_true = class_mask
                            binary_pred = (hard_preds == class_id)
                            
                            # Calculate metrics
                            precision, recall, f1, _ = precision_recall_fscore_support(
                                binary_true, binary_pred, average='binary', zero_division=0
                            )
                            
                            # Calculate class-specific accuracy (correct predictions for this class / total samples of this class)
                            accuracy = accuracy_score(binary_true, binary_pred)
                            
                            # Store metrics and predictions for this class
                            per_class_metrics[fold][epoch][class_id] = {
                                'precision': precision,
                                'recall': recall,
                                'f1': f1,
                                'accuracy': accuracy,
                                'predictions': oof_p_curr[class_mask],
                                'indices': oof_idx_curr[class_mask]
                            }
                            
                            # Update best epoch for this class if this is a new best
                            metric_to_track = per_class_metrics[fold][epoch][class_id][OOF_METRIC_FOR_BEST]
                            
                            if class_id not in best_epoch_per_class[fold] or metric_to_track > per_class_metrics[fold][best_epoch_per_class[fold][class_id]][class_id][OOF_METRIC_FOR_BEST]:
                                best_epoch_per_class[fold][class_id] = epoch
                                
                        print(f"{TermColors.INFO}Updated per-class metrics for epoch {epoch+1}. Classes with metrics: {len(unique_classes)}{TermColors.ENDC}")
                    
                    # Store current fold predictions for later analysis (only for first initialization)
                    fold_predictions.append(oof_p_curr if oof_p_curr is not None else None)
                    fold_indices.append(oof_idx_curr if oof_idx_curr is not None else None)
                    fold_labels.append(val_lbls)
    
                # Now continue with SWA, early stopping, etc.
                if MLP_USE_SWA and swa_mlp and epoch >= mlp_swa_start:
                    swa_mlp.update_parameters(mlp); 
                    if swa_sched_mlp: swa_sched_mlp.step()
                
                curr_met_stop = v_loss if CHECKPOINT_MONITOR == 'val_loss' else v_acc
                improved_early = (CHECKPOINT_MODE == 'min' and curr_met_stop < best_met_early_stop) or \
                                 (CHECKPOINT_MODE == 'max' and curr_met_stop > best_met_early_stop)
                if improved_early: best_met_early_stop = curr_met_stop; no_improve_epochs = 0
                else: no_improve_epochs +=1
                if no_improve_epochs >= MLP_EARLY_STOPPING_PATIENCE: fold_stop = True; print(f"{TermColors.WARNING}Early stop MLP F{fold+1} I{init_idx+1}.{TermColors.ENDC}")
                
                curr_met_ckpt = v_acc if CHECKPOINT_MONITOR == 'val_acc' else v_loss
                is_best = (CHECKPOINT_MODE == 'max' and curr_met_ckpt > best_met) or \
                          (CHECKPOINT_MODE == 'min' and curr_met_ckpt < best_met)
                if is_best:
                    best_met = curr_met_ckpt; best_val_loss = v_loss; best_val_acc = v_acc; best_ep = epoch
                    print(f"{TermColors.OKGREEN}MLP F{fold+1} I{init_idx+1} New Best {CHECKPOINT_MONITOR}: {best_met:.4f}. Save...{TermColors.ENDC}")
                    save_checkpoint(fold, epoch+1, mlp, opt, sched, scaler_mlp, best_met, f"best_mlp_model_init{init_idx}.pth.tar", True)
                    save_model(fold, mlp, f"best_mlp_model_state_dict_init{init_idx}.pth", True)
                    if MLP_USE_EMA and ema_mlp: save_model(fold, ema_mlp, f"best_ema_mlp_model_state_dict_init{init_idx}.pth", True)
                    if MLP_USE_SWA and swa_mlp and epoch >= mlp_swa_start: save_model(fold, swa_mlp, f"best_swa_mlp_model_state_dict_init{init_idx}.pth", True)
    
                save_checkpoint(fold, epoch+1, mlp, opt, sched, scaler_mlp, best_met, f"latest_mlp_checkpoint_init{init_idx}.pth.tar", True)
                if LOG_MISCLASSIFIED_IMAGES and ((epoch+1)%5==0 or is_best): log_misclassified(fold, mlp, err_loader_mlp, crit, DEVICE, epoch+1, writer_mlp, NUM_CLASSES, True)
                last_ep_fold = epoch
                
                gc.collect(); torch.cuda.empty_cache()
                
            print(f"MLP F{fold+1} I{init_idx+1} done. Best Base Val Acc: {best_val_acc:.4f} at Ep {best_ep+1 if best_ep!=-1 else 'N/A'}")
            
            if fold_stop and not stop_requested: 
                save_checkpoint(fold, last_ep_fold+1, mlp, opt, sched, scaler_mlp, best_met, f"interrupted_mlp_ckpt_init{init_idx}.pth.tar", True)
            if stop_requested: 
                save_checkpoint(fold, last_ep_fold+1, mlp, opt, sched, scaler_mlp, best_met, f"interrupted_mlp_ckpt_init{init_idx}.pth.tar", True)
                break
            else:
                if MLP_USE_SWA and swa_mlp and last_ep_fold >= mlp_swa_start:
                    print(f"{TermColors.CYAN}MLP F{fold+1} I{init_idx+1} Final SWA BN...{TermColors.ENDC}")
                    bn_loader_final = DataLoader(train_ds_mlp, MLP_BATCH_SIZE*2, shuffle=True, num_workers=0)
                    try:
                        torch.optim.swa_utils.update_bn(bn_loader_final, swa_mlp.to(DEVICE), device=DEVICE)
                        swa_v_loss, swa_v_acc, _, _, _ = validate_one_epoch_mlp(swa_mlp, val_loader_mlp, crit, DEVICE, last_ep_fold+1, writer_mlp, NUM_CLASSES, fold_num=f"{fold+1}-I{init_idx+1}-SWA-Final")
                        print(f"{TermColors.SUCCESS} MLP F{fold+1} I{init_idx+1} FINAL SWA Val L={swa_v_loss:.4f} A={swa_v_acc:.4f}{TermColors.ENDC}")
                        if init_idx == 0:  # Only track metrics for first initialization
                            fold_results['mlp_swa_acc'].append(swa_v_acc)
                        save_model(fold, swa_mlp, f"final_swa_mlp_model_state_dict_init{init_idx}.pth", True)
                    except Exception as e: print(f"{TermColors.RED}MLP SWA BN/Eval Error: {e}{TermColors.ENDC}")
                if MLP_USE_EMA and ema_mlp:
                    try:
                        ema_v_loss, ema_v_acc, _, _, _ = validate_one_epoch_mlp(ema_mlp.to(DEVICE), val_loader_mlp, crit, DEVICE, last_ep_fold+1, writer_mlp, NUM_CLASSES, fold_num=f"{fold+1}-I{init_idx+1}-EMA-Final")
                        print(f"{TermColors.SUCCESS} MLP F{fold+1} I{init_idx+1} FINAL EMA Val L={ema_v_loss:.4f} A={ema_v_acc:.4f}{TermColors.ENDC}")
                        if init_idx == 0:  # Only track metrics for first initialization
                            fold_results['mlp_ema_acc'].append(ema_v_acc)
                        save_model(fold, ema_mlp, f"final_ema_mlp_model_state_dict_init{init_idx}.pth", True)
                    except Exception as e: print(f"{TermColors.RED}MLP EMA Eval Error: {e}{TermColors.ENDC}")
            
            # Save the base model
            if mlp: 
                save_model(fold, mlp, f"final_mlp_model_state_dict_init{init_idx}.pth", True)
                # Store the model for ensemble prediction later
                multi_init_models[fold].append({
                    'model': mlp,
                    'val_acc': best_val_acc
                })
                
            if writer_mlp: writer_mlp.close()
            if init_idx == 0:  # Only track metrics for first initialization
                fold_results['mlp_best_val_acc'].append(best_val_acc)
            
            # Clean up after each initialization except for the stored models
            del opt, sched, scaler_mlp, crit, swa_mlp, ema_mlp, train_loader_mlp, val_loader_mlp, err_loader_mlp
            gc.collect(); torch.cuda.empty_cache()
        
        # After all initializations, if we're doing multi-initialization ensemble, combine predictions
        if MLP_MULTI_INIT_ENSEMBLE > 1 and len(multi_init_models[fold]) > 0:
            print(f"{TermColors.CYAN}Fold {fold+1}: Creating ensemble predictions from {len(multi_init_models[fold])} model initializations...{TermColors.ENDC}")
            
            # Create a validation loader for ensemble prediction
            val_ds_ensemble = FeatureDataset(val_feats, val_lbls, val_orig_ids_fold)
            val_loader_ensemble = DataLoader(val_ds_ensemble, MLP_BATCH_SIZE*2, shuffle=False, num_workers=0, pin_memory=True)
            
            # For each sample, we'll average predictions from all initializations
            ensemble_preds = None
            ensemble_indices = None
            
            with torch.no_grad():
                for model_data in multi_init_models[fold]:
                    model = model_data['model']
                    model.eval()
                    
                    model_preds = []
                    model_indices = []
                    
                    for inputs, labels, indices in tqdm(val_loader_ensemble, desc=f"Ensemble Predictions", leave=False):
                        inputs = inputs.to(DEVICE).float()
                        outputs = model(inputs)
                        probs = F.softmax(outputs, dim=1)
                        
                        model_preds.append(probs.cpu().numpy())
                        model_indices.append(indices.cpu().numpy())
                    
                    if model_preds:
                        model_preds_concat = np.vstack(model_preds)
                        model_indices_concat = np.concatenate(model_indices)
                        
                        if ensemble_preds is None:
                            ensemble_preds = model_preds_concat
                            ensemble_indices = model_indices_concat
                        else:
                            # Check if indices match
                            if np.array_equal(ensemble_indices, model_indices_concat):
                                ensemble_preds += model_preds_concat
                            else:
                                print(f"{TermColors.RED}ERROR: Indices don't match across model initializations. Cannot ensemble.{TermColors.ENDC}")
                                break
            
            if ensemble_preds is not None:
                # Average the predictions
                ensemble_preds /= len(multi_init_models[fold])
                
                # Calculate ensemble accuracy
                hard_preds = np.argmax(ensemble_preds, axis=1)
                ensemble_acc = accuracy_score(val_lbls, hard_preds[np.argsort(ensemble_indices)])
                print(f"{TermColors.GREEN}Ensemble Accuracy: {ensemble_acc:.4f} (vs. Best Single Model: {max([m['val_acc'] for m in multi_init_models[fold]]):.4f}){TermColors.ENDC}")
                
                # Update OOF predictions with ensemble predictions
                for i, orig_id in enumerate(ensemble_indices):
                    # Find this sample in the global OOF array
                    global_idx = np.where(orig_idx_kfold == orig_id)[0]
                    if len(global_idx) > 0:
                        # Update with ensemble prediction
                        oof_preds[global_idx[0]] = ensemble_preds[i]
                        # Find the corresponding label
                        val_idx_match = np.where(val_orig_ids_fold == orig_id)[0]
                        if len(val_idx_match) > 0:
                            oof_lbls[global_idx[0]] = val_lbls[val_idx_match[0]]
                        else:
                            oof_lbls[global_idx[0]] = lbls_kfold[global_idx[0]]
            
            # Clean up ensemble resources
            del val_ds_ensemble, val_loader_ensemble
        
        # If not doing multi-initialization ensemble or if it failed, use the first initialization's per-class best
        elif len(multi_init_models[fold]) > 0:  # Make sure we have at least one model
            # After all epochs, use the best predictions for each class
            print(f"{TermColors.CYAN}Fold {fold+1}: Using best epoch predictions for each class...{TermColors.ENDC}")
            
            # Get unique classes in this fold's validation set
            fold_classes = np.unique(val_lbls)
            
            for class_id in fold_classes:
                # Skip if we don't have metrics for this class
                if class_id not in best_epoch_per_class[fold]:
                    print(f"{TermColors.YELLOW}Warning: No best epoch found for class {class_id} in fold {fold+1}.{TermColors.ENDC}")
                    continue
                    
                best_class_epoch = best_epoch_per_class[fold][class_id]
                
                if best_class_epoch not in per_class_metrics[fold] or class_id not in per_class_metrics[fold][best_class_epoch]:
                    print(f"{TermColors.YELLOW}Warning: Metrics missing for class {class_id} at epoch {best_class_epoch} in fold {fold+1}.{TermColors.ENDC}")
                    continue
                    
                # Get the best predictions for this class
                class_metrics = per_class_metrics[fold][best_class_epoch][class_id]
                class_predictions = class_metrics['predictions']
                class_indices = class_metrics['indices']
                
                if len(class_indices) == 0:
                    print(f"{TermColors.YELLOW}Warning: No samples for class {class_id} in fold {fold+1}.{TermColors.ENDC}")
                    continue
                    
                # Update the OOF predictions for this class
                print(f"  Class {class_id}: Using predictions from epoch {best_class_epoch+1} (F1: {class_metrics['f1']:.4f})")
                
                for i, orig_id in enumerate(class_indices):
                    # Find this sample in the global OOF array
                    global_idx = np.where(orig_idx_kfold == orig_id)[0]
                    if len(global_idx) > 0:
                        # Update the OOF prediction
                        oof_preds[global_idx[0]] = class_predictions[i]
                        oof_lbls[global_idx[0]] = class_id
        
        # After processing best epochs or ensemble, check if any validation samples are missing from OOF
        fold_valid_oof_idx = np.where(~np.isnan(oof_preds[val_idx]).any(axis=1))[0]
        missing_ratio = 1.0 - (len(fold_valid_oof_idx) / len(val_idx)) if len(val_idx) > 0 else 0
        
        if missing_ratio > 0.01:  # More than 1% missing
            missing_val_idx = set(range(len(val_idx))) - set(fold_valid_oof_idx)
            missing_classes = np.unique([val_lbls[i] for i in missing_val_idx])
            print(f"{TermColors.YELLOW}WARNING: Missing {len(missing_val_idx)} validation samples in fold {fold+1} OOF predictions{TermColors.ENDC}")
            print(f"{TermColors.YELLOW}Missing classes: {missing_classes}{TermColors.ENDC}")
            
            # Try to recover these missing samples from our saved fold predictions
            if fold_predictions and fold_indices:
                print(f"{TermColors.INFO}Attempting to recover missing samples from saved fold predictions...{TermColors.ENDC}")
                recovered = 0
                for miss_idx in missing_val_idx:
                    orig_id = val_orig_ids_fold[miss_idx]
                    label_val = val_lbls[miss_idx]
                    
                    # Look through all saved predictions (newest to oldest)
                    for ep_idx, (ep_preds, ep_indices) in enumerate(zip(reversed(fold_predictions), reversed(fold_indices))):
                        if ep_preds is None or ep_indices is None:
                            continue
                            
                        match_idx = np.where(ep_indices == orig_id)[0]
                        if len(match_idx) > 0:
                            # Found this sample in a previous epoch's validation
                            global_oof_match_idx = np.where(orig_idx_kfold == orig_id)[0]
                            if len(global_oof_match_idx) > 0:
                                global_idx = global_oof_match_idx[0]
                                oof_preds[global_idx] = ep_preds[match_idx[0]]
                                oof_lbls[global_idx] = label_val
                                recovered += 1
                            break
                print(f"{TermColors.GREEN}Recovered {recovered}/{len(missing_val_idx)} missing samples{TermColors.ENDC}")
                
        # Clean up all stored models for this fold
        for model_data in multi_init_models[fold]:
            if 'model' in model_data:
                del model_data['model']
        del multi_init_models[fold]
        gc.collect(); torch.cuda.empty_cache()
        if stop_requested: break

    if not stop_requested:
        print(f"Avg MLP Best Base Val Acc: {np.mean(fold_results['mlp_best_val_acc']):.4f} +/- {np.std(fold_results['mlp_best_val_acc']):.4f}")
        if fold_results['mlp_swa_acc']: print(f"Avg MLP Final SWA Val Acc: {np.mean(fold_results['mlp_swa_acc']):.4f} +/- {np.std(fold_results['mlp_swa_acc']):.4f}")
        if fold_results['mlp_ema_acc']: print(f"Avg MLP Final EMA Val Acc: {np.mean(fold_results['mlp_ema_acc']):.4f} +/- {np.std(fold_results['mlp_ema_acc']):.4f}")
    else: 
        print(f"{TermColors.YELLOW}MLP Training interrupted. Stacking/KD skipped.{TermColors.ENDC}")

    # After all folds, use the diagnostic function to check final OOF
    valid_oof_idx = np.where(~np.isnan(oof_preds).any(axis=1))[0]
    if len(valid_oof_idx) > 0:
        global STACKING_HPO_CV_FOLDS  # Move global declaration to the beginning
        
        final_oof_p = oof_preds[valid_oof_idx]
        final_oof_l = oof_lbls[valid_oof_idx]
        final_oof_orig_id = orig_idx_kfold[valid_oof_idx]
        
        diagnose_oof_class_distribution(final_oof_p, final_oof_l, final_oof_orig_id, 
                                      log_prefix="FINAL OOF")
        
        # If we still detect problems with class counts, try to fix them
        unique_oof_labels, oof_label_counts = np.unique(final_oof_l, return_counts=True)
        min_count_in_oof = np.min(oof_label_counts) if len(oof_label_counts) > 0 else 0
        
        if len(unique_oof_labels) < NUM_CLASSES:
            print(f"{TermColors.RED}WARNING: OOF predictions missing {NUM_CLASSES - len(unique_oof_labels)} classes!{TermColors.ENDC}")
            # If STACKING_HPO_CV_FOLDS needs adjustment, do it here
            if min_count_in_oof < STACKING_HPO_CV_FOLDS:
                old_folds = STACKING_HPO_CV_FOLDS
                STACKING_HPO_CV_FOLDS = max(2, min_count_in_oof-1)
                print(f"{TermColors.YELLOW}Adjusted STACKING_HPO_CV_FOLDS from {old_folds} to {STACKING_HPO_CV_FOLDS} based on minimum class count.{TermColors.ENDC}")
        
        np.savez_compressed(STACKING_OOF_PREDS_PATH, preds=final_oof_p, labels=final_oof_l, original_indices=final_oof_orig_id)
        print(f"{TermColors.GREEN}MLP OOF predictions saved: {STACKING_OOF_PREDS_PATH} ({len(final_oof_p)} samples across {len(unique_oof_labels)} classes){TermColors.ENDC}")
        
        if RUN_STACKING:
            final_stacking_path = STACKING_META_MODEL_PATH
            train_stacking_meta_model(final_oof_p, final_oof_l, final_stacking_path)
    else:
        print(f"{TermColors.RED}ERROR: No valid MLP OOF predictions. Stacking cannot proceed.{TermColors.ENDC}")
    
    return final_stacking_path

# --- Main Execution ---
def main():
    global stop_requested, label_encoder, class_names, NUM_CLASSES, CLASS_FREQUENCIES, CLASS_PRIORS, CLASS_WEIGHTS
    global RUN_STACKING, RUN_KNOWLEDGE_DISTILLATION 

    set_seed(SEED); signal.signal(signal.SIGINT, handle_interrupt); print_library_info()
    print(f"{TermColors.HEADER}===== Plant Recognition MLP Only Mode (PyTorch) ===={TermColors.ENDC}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, Device: {DEVICE}, Debug: {DEBUG_MODE}")
    
    df_full = None
    try: # Data Loading and Preprocessing
        print(f"\n{TermColors.HEADER}--- STEP 1: Load Full Dataset Info ---{TermColors.ENDC}")
        df_full = pd.read_csv(CSV_PATH, sep=',', low_memory=False, on_bad_lines='skip')
        if 'scientificName' in df_full.columns and 'scientific_name' not in df_full.columns: df_full.rename(columns={'scientificName': 'scientific_name'}, inplace=True)
        required_cols = ['id', 'scientific_name']; df_full = df_full[required_cols].dropna().astype({'id': str}); df_full.rename(columns={'scientific_name': 'scientificName'}, inplace=True)
    
        min_samples = 1
        print(f"{TermColors.INFO}Using min_samples_per_class={min_samples} for initial CSV filtering (expecting PlantDataset to find augmentations).{TermColors.ENDC}")

        class_counts = df_full['scientificName'].value_counts()
        valid_classes = class_counts[class_counts >= min_samples].index
        
        original_row_count = len(df_full)
        df_full = df_full[df_full['scientificName'].isin(valid_classes)].reset_index(drop=True)
        print(f"{TermColors.INFO}Filtered dataframe from {original_row_count} to {len(df_full)} rows based on min_samples_per_class.{TermColors.ENDC}")
        
        if len(df_full) == 0: print(f"{TermColors.RED}Dataframe empty after filtering by min_samples_per_class. Exit.{TermColors.ENDC}"); sys.exit(1)
        
        label_encoder = LabelEncoder(); df_full['label'] = label_encoder.fit_transform(df_full['scientificName'])
        class_names = list(label_encoder.classes_); NUM_CLASSES = len(class_names)
        if NUM_CLASSES == 0: print(f"{TermColors.RED}Zero classes. Exit.{TermColors.ENDC}"); sys.exit(1)
        
        os.makedirs(BASE_MODEL_SAVE_DIR, exist_ok=True)
        with open(os.path.join(BASE_MODEL_SAVE_DIR, "label_mapping.json"), 'w') as f: json.dump(dict(zip(range(NUM_CLASSES), class_names)), f, indent=4)
        
        if DEBUG_MODE: 
            min_debug_samples = 2; strat_col = df_full['label']
            counts_debug = df_full['label'].value_counts(); valid_lbls_debug = counts_debug[counts_debug >= min_debug_samples].index
            df_debug_poss = df_full[df_full['label'].isin(valid_lbls_debug)]
            if len(df_debug_poss) >= min(100, len(df_full)) and len(valid_lbls_debug) > 1 :
                 _, df_full = train_test_split(df_debug_poss, test_size=min(100, len(df_debug_poss)), random_state=SEED, stratify=df_debug_poss['label'])
            else: df_full = df_full.sample(n=min(100, len(df_full)), random_state=SEED)
            df_full = df_full.reset_index(drop=True)
        
        lbl_counts = df_full['label'].value_counts().sort_index(); total_s = len(df_full)
        if total_s == 0: print(f"{TermColors.RED}Dataframe empty before imbalance. Exit.{TermColors.ENDC}"); sys.exit(1)
        freqs = torch.zeros(NUM_CLASSES, dtype=torch.float32); lbl_counts_reidx = lbl_counts.reindex(range(NUM_CLASSES), fill_value=0)
        for i in range(NUM_CLASSES): freqs[i] = lbl_counts_reidx.get(i, 0)
        CLASS_FREQUENCIES = freqs.to(DEVICE); CLASS_PRIORS = (CLASS_FREQUENCIES / total_s if total_s > 0 else torch.zeros_like(CLASS_FREQUENCIES)).to(DEVICE)
        try:
            weights_arr = sk_class_weight.compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=df_full['label'])
            CLASS_WEIGHTS = torch.tensor(weights_arr, dtype=torch.float32)
        except ValueError as e:
            print(f"{TermColors.RED}Class weights error: {e}.{TermColors.ENDC}")
            if IMBALANCE_STRATEGY in ['WeightedLoss', 'WeightedSampler']: print(f"{TermColors.RED}Exit due to weights error with strategy '{IMBALANCE_STRATEGY}'.{TermColors.ENDC}"); sys.exit(1)
        print(f"{TermColors.GREEN}Data loaded: {len(df_full)} samples, {NUM_CLASSES} classes. Imbalance: {IMBALANCE_STRATEGY}.{TermColors.ENDC}")
    except Exception as e: print(f"{TermColors.RED}Data Load/Prep Error: {e}{TermColors.ENDC}"); traceback.print_exc(); sys.exit(1)

    mlp_stacking_model_path = None 
    training_successful = False 

    if os.path.exists(STACKING_META_MODEL_PATH) and RUN_STACKING:
        print(f"{TermColors.INFO}MLP stacking model exists: {STACKING_META_MODEL_PATH}. Skip MLP K-fold & stacking.{TermColors.ENDC}")
        mlp_stacking_model_path = STACKING_META_MODEL_PATH; training_successful = True
    elif os.path.exists(STACKING_OOF_PREDS_PATH) and RUN_STACKING:
        print(f"{TermColors.INFO}MLP OOF predictions found: {STACKING_OOF_PREDS_PATH}. Attempt stacking only.{TermColors.ENDC}")
        try:
            oof_data = np.load(STACKING_OOF_PREDS_PATH)
            if len(oof_data['preds']) > 0 and len(oof_data['preds']) == len(oof_data['labels']):
                train_stacking_meta_model(oof_data['preds'], oof_data['labels'], STACKING_META_MODEL_PATH)
                if os.path.exists(STACKING_META_MODEL_PATH):
                    mlp_stacking_model_path = STACKING_META_MODEL_PATH; training_successful = True
                    print(f"{TermColors.GREEN}Stacking from existing MLP OOFs complete.{TermColors.ENDC}")
                else: print(f"{TermColors.RED}Stacking from OOFs failed. Will attempt full MLP train or OOF regeneration.{TermColors.ENDC}")
            else: print(f"{TermColors.YELLOW}MLP OOF file invalid. Will attempt full MLP train or OOF regeneration.{TermColors.ENDC}")
        except Exception as e: print(f"{TermColors.RED}Error with OOFs: {e}. Will attempt full MLP train or OOF regeneration.{TermColors.ENDC}")
    
    # --- logic for OOF regeneration if STACKING_OOF_PREDS_PATH is missing but fold models exist ---
    if not training_successful and RUN_STACKING and not os.path.exists(STACKING_OOF_PREDS_PATH):
        print(f"{TermColors.INFO}Consolidated OOF file not found. Checking for individual fold models to regenerate OOFs...{TermColors.ENDC}")
        all_fold_models_exist = True
        if N_FOLDS <= 0: all_fold_models_exist = False # Should not happen with valid config
        
        for f_idx in range(N_FOLDS):
            fold_model_path = os.path.join(BASE_MODEL_SAVE_DIR, f"mlp_fold_{f_idx}", "best_mlp_model_state_dict.pth")
            if not os.path.exists(fold_model_path):
                all_fold_models_exist = False
                print(f"{TermColors.YELLOW}  MLP model for fold {f_idx+1} not found at {fold_model_path}. Cannot regenerate OOFs.{TermColors.ENDC}")
                break
        
        if all_fold_models_exist:
            print(f"{TermColors.INFO}All individual MLP fold models found. Attempting to regenerate OOF predictions...{TermColors.ENDC}")
            regenerated_oof_path = regenerate_oof_from_fold_models(
                NUM_CLASSES, label_encoder, class_names, CLASS_WEIGHTS
            )
            if regenerated_oof_path and os.path.exists(regenerated_oof_path):
                print(f"{TermColors.INFO}OOF predictions regenerated. Attempting stacking...{TermColors.ENDC}")
                try:
                    oof_data = np.load(regenerated_oof_path)
                    train_stacking_meta_model(oof_data['preds'], oof_data['labels'], STACKING_META_MODEL_PATH)
                    if os.path.exists(STACKING_META_MODEL_PATH):
                        mlp_stacking_model_path = STACKING_META_MODEL_PATH; training_successful = True
                        print(f"{TermColors.GREEN}Stacking from regenerated MLP OOFs complete.{TermColors.ENDC}")
                    else: print(f"{TermColors.RED}Stacking from regenerated OOFs failed.{TermColors.ENDC}")
                except Exception as e: print(f"{TermColors.RED}Error stacking from regenerated OOFs: {e}{TermColors.ENDC}")
            else:
                print(f"{TermColors.RED}OOF regeneration failed. Proceeding to full MLP training if necessary.{TermColors.ENDC}")
        else:
            print(f"{TermColors.INFO}Not all fold models found. Proceeding to full MLP training if necessary.{TermColors.ENDC}")

    # Fallback to full training if no prior success
    if not training_successful:
        print(f"{TermColors.INFO}Starting full MLP K-fold training and potentially stacking.{TermColors.ENDC}")
        if not os.path.exists(FEATURES_NPZ_PATH):
            print(f"{TermColors.YELLOW}Base features not found: {FEATURES_NPZ_PATH}. Extracting features...{TermColors.ENDC}")
            extract_all_features(df_full, NUM_CLASSES, label_encoder) # df_full from initial load
            if not os.path.exists(FEATURES_NPZ_PATH): 
                print(f"{TermColors.RED}Feature extraction failed. Aborting MLP training.{TermColors.ENDC}")
                sys.exit(1) # Critical failure if FE fails and is needed
        train_mlp_on_features_main_loop(
            NUM_CLASSES, label_encoder, class_names, CLASS_FREQUENCIES, CLASS_PRIORS, CLASS_WEIGHTS
        )
        # Check if stacking model was created by train_mlp_on_features_main_loop
        if os.path.exists(STACKING_META_MODEL_PATH):
            mlp_stacking_model_path = STACKING_META_MODEL_PATH
            training_successful = True # Assume success if stacking model is now present
        elif not RUN_STACKING: # If stacking was not supposed to run, consider MLP training successful
             training_successful = not stop_requested # Check if it was interrupted

    # --- Knowledge Distillation ---
    if RUN_KNOWLEDGE_DISTILLATION and training_successful and mlp_stacking_model_path and os.path.exists(mlp_stacking_model_path):
        print(f"\n{TermColors.HEADER}--- FINAL STEP: Distill MLP Stacking Ensemble ---{TermColors.ENDC}")
        if not os.path.exists(FEATURES_NPZ_PATH):
            print(f"{TermColors.RED}Base features {FEATURES_NPZ_PATH} not found for KD. Skip KD.{TermColors.ENDC}")
        else:
            feats_data_kd = np.load(FEATURES_NPZ_PATH, allow_pickle=True)
            mlp_input_size_kd = feats_data_kd['features'].shape[1]; del feats_data_kd
            
            
            # Teacher components are MLP-based
            base_mlp_paths_kd = [os.path.join(BASE_MODEL_SAVE_DIR, f"mlp_fold_{f}", "best_mlp_model_state_dict.pth") for f in range(N_FOLDS)]
            all_base_mlps_exist = all(os.path.exists(p) for p in base_mlp_paths_kd)
            
            if not all_base_mlps_exist:
                print(f"{TermColors.RED}Not all base MLP models for KD teacher found. Skip KD.{TermColors.ENDC}")
            else:
                teacher_info_kd = {
                    "stacking_model_path": mlp_stacking_model_path,
                    "feature_extractor_model_names": MODEL_NAMES, 
                    "feature_extractor_embedding_size": EMBEDDING_SIZE, 
                    "feature_extractor_arcface_m": ARCFACE_M, 
                    "feature_extractor_metric_learning": 'None', # FE for teacher provides raw embeddings
                    "feature_extractor_checkpoint_path": FEATURE_EXTRACTOR_CHECKPOINT_PATH, 
                    "base_mlp_model_paths": base_mlp_paths_kd,
                    "base_mlp_model_input_size": mlp_input_size_kd,
                    "n_folds_for_stacking": N_FOLDS, 
                    "num_classes": NUM_CLASSES 
                }
                try: train_student_model(df_full, NUM_CLASSES, label_encoder, teacher_info_kd)
                except Exception as e: print(f"{TermColors.RED}Error during KD: {e}{TermColors.ENDC}"); traceback.print_exc()
    elif RUN_KNOWLEDGE_DISTILLATION:
        status = "training not successful" if not training_successful else "stacking model path missing"
        print(f"{TermColors.YELLOW}KD enabled, but {status}. Skip KD.{TermColors.ENDC}")

    print(f"\n{TermColors.OKGREEN}All MLP Mode processes complete.{TermColors.ENDC}")

if __name__ == "__main__":
    main()