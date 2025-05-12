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
import time
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

# --- Optional/Advanced Libraries ---
# Define global flags first
TIMM_AVAILABLE = False
TORCHMETRICS_AVAILABLE = False
ALBUMENTATIONS_AVAILABLE = False
ADAMP_AVAILABLE = False
SAM_AVAILABLE = False
_INITIALIZED = False
LGBM_PRINTED_INFO = False
OPTUNA_AVAILABLE = False

# Now import each library once
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
    WANDB_AVAILABLE = False

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
    global _INITIALIZED, LGBM_PRINTED_INFO
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
    
    if LGBM_AVAILABLE and not LGBM_PRINTED_INFO:
        print("INFO: lightgbm library found and available for stacking.")
        LGBM_PRINTED_INFO = True
    elif not LGBM_AVAILABLE and not LGBM_PRINTED_INFO:
        print(f"{TermColors.WARN}WARN: lightgbm library not found. Stacking with LGBM will not be available. Install with 'pip install lightgbm'.{TermColors.ENDC}")
        LGBM_PRINTED_INFO = True

    if OPTUNA_AVAILABLE:
        print("INFO: optuna library found and available for MLP HPO.")

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
    (12 if not DEBUG_MODE else 1, (224, 224)),
    (10 if not DEBUG_MODE else 1, (384, 384)),
    (8 if not DEBUG_MODE else 1, (448, 448)),
    (7 if not DEBUG_MODE else 1, (512, 512)),
]
TOTAL_EPOCHS_PER_FOLD = sum(s[0] for s in PROGRESSIVE_RESIZING_STAGES)
CURRENT_IMAGE_SIZE = None
EARLY_STOPPING_PATIENCE = 10 if not DEBUG_MODE else 3 # Patience for early stopping

# --- Cross-Validation Config ---
N_FOLDS = 5 if not DEBUG_MODE else 2

# --- Hardware Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 8
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
MLP_HIDDEN_DIMS = [256, 128] # Hidden layer dimensions for the MLP
MLP_DROPOUT_RATE = 0.6
MLP_USE_ARCFACE = True # Whether the MLP should also use an ArcFace head
MLP_LEARNING_RATE = 1e-3
MLP_WEIGHT_DECAY = 5e-3
MLP_EPOCHS = 200 if not DEBUG_MODE else 5
MLP_BATCH_SIZE = 128 if not DEBUG_MODE else 32
MLP_EARLY_STOPPING_PATIENCE = 10 if not DEBUG_MODE else 3
MLP_OPTIMIZER_TYPE = 'AdamW'
MLP_SCHEDULER_TYPE = 'CosineWarmRestarts'
MLP_SWA_START_EPOCH_FACTOR = 0.75
MLP_USE_SWA = True
MLP_USE_EMA = True
MLP_EMA_DECAY = 0.999
MLP_USE_AUTO_TRAIN_CONFIG = False

# --- MLP Hyperparameter Optimization (HPO) Config ---
MLP_DO_HPO = True  # Set to True to run HPO for MLP parameters
MLP_HPO_N_TRIALS = 20 if not DEBUG_MODE else 3 # Number of HPO trials
MLP_HPO_EPOCHS = 60 if not DEBUG_MODE else 2 # Max epochs for each HPO trial model
MLP_HPO_PATIENCE = 8 if not DEBUG_MODE else 2  # Early stopping patience for HPO trials
MLP_HPO_DATA_SPLIT_RATIO = 0.20 # Proportion of *all* feature data to use for HPO.
                               # This HPO data is then split internally into HPO-train and HPO-val.
MLP_HPO_INTERNAL_VAL_SPLIT = 0.25
MLP_HPO_OVERFIT_PENALTY_FACTOR = 0.8

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
LOG_MISCLASSIFIED_IMAGES = False # for max speed

# --- Test Time Augmentation (TTA) Config ---
USE_TTA = True
TTA_TRANSFORMS = None

# --- Stacking Config ---
RUN_STACKING = True # Can run after MLP training too
STACKING_META_MODEL_PATH = os.path.join(BASE_MODEL_SAVE_DIR, "stacking_meta_model.joblib")
STACKING_OOF_PREDS_PATH = os.path.join(BASE_MODEL_SAVE_DIR, "oof_predictions.npz")
STACKING_DO_HPO = True
STACKING_HPO_CV_FOLDS = 3
STACKING_LGBM_PARAM_GRID = {
    'n_estimators': [100, 200, 400],       # Number of boosting rounds
    'learning_rate': [0.02, 0.05, 0.1],   # Step size shrinkage
    'num_leaves': [20, 31, 40],           # Max number of leaves in one tree
    'max_depth': [-1, 10, 15],            # Max tree depth (-1 means no limit)
    'min_child_samples': [15, 20, 30],    # Minimum number of data needed in a child
    'subsample': [0.7, 0.8, 0.9],         # Subsample ratio of the training instance
    'colsample_bytree': [0.7, 0.8, 0.9],  # Subsample ratio of columns
    'reg_alpha': [0, 0.01, 0.1],        # L1 regularization
    'reg_lambda': [0, 0.01, 0.1],       # L2 regularization
}

# --- Knowledge Distillation Config ---
RUN_KNOWLEDGE_DISTILLATION = True
KD_STUDENT_MODEL_NAME = "mobilenetv3_small_100"
KD_STUDENT_IMAGE_SIZE = (224, 224)
KD_STUDENT_EMBEDDING_SIZE = 512
KD_STUDENT_DROPOUT = 0.2
KD_EPOCHS = 15 if not DEBUG_MODE else 2
KD_BATCH_SIZE = BATCH_SIZE * 2
KD_LR = 1e-4
# --- TUNABLE KD PARAMETERS ---
KD_ALPHA = 0.5 # Weight for distillation loss vs student's own loss. Range: [0.0, 1.0]. Higher alpha emphasizes mimicking the teacher.
KD_TEMPERATURE = 4.0 # Softening temperature for teacher logits. Range: [1.0, ~10.0]. Higher T produces softer labels.
# --- END TUNABLE KD PARAMETERS ---
KD_STUDENT_MODEL_SAVE_PATH = os.path.join(BASE_MODEL_SAVE_DIR, f"final_distilled_stacked_student_{KD_STUDENT_MODEL_NAME}_base.pth")
KD_STUDENT_SWA_MODEL_SAVE_PATH = os.path.join(BASE_MODEL_SAVE_DIR, f"final_distilled_stacked_student_{KD_STUDENT_MODEL_NAME}_swa.pth")
KD_STUDENT_EMA_MODEL_SAVE_PATH = os.path.join(BASE_MODEL_SAVE_DIR, f"final_distilled_stacked_student_{KD_STUDENT_MODEL_NAME}_ema.pth")

# SWA/EMA specific to the student model's training during distillation
KD_STUDENT_USE_SWA = True
KD_STUDENT_SWA_START_EPOCH_FACTOR = 0.75 # Start SWA for student after this fraction of KD_EPOCHS
KD_STUDENT_SWA_LR_FACTOR = 0.05 # SWA learning rate factor for student
KD_STUDENT_SWA_ANNEAL_EPOCHS = 5 # SWA annealing epochs for student

KD_STUDENT_USE_EMA = True
KD_STUDENT_EMA_DECAY = 0.999 # EMA decay for student model

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
    print(f"{TermColors.INFO}Seed set to {seed}{TermColors.ENDC}")

def handle_interrupt(signal, frame):
    global stop_requested
    if stop_requested: 
        print(f"\n{TermColors.CRITICAL}Force exiting...{TermColors.ENDC}")
        sys.exit(1)
    print(f"\n{TermColors.WARNING}Interrupt received. Finishing current epoch and saving state... Press Ctrl+C again to force exit.{TermColors.ENDC}")
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
    
    is_sam_optimizer = hasattr(optimizer, 'base_optimizer') and SAM_AVAILABLE and (USE_SAM if not is_mlp_checkpoint else False)
    opt_state_dict = optimizer.base_optimizer.state_dict() if is_sam_optimizer else optimizer.state_dict()

    state = {
        'fold': fold,
        'global_epoch': global_epoch,
        'stage_idx': stage_idx if not is_mlp_checkpoint else -1,
        'stage_epoch': stage_epoch if not is_mlp_checkpoint else global_epoch,
        'image_size': CURRENT_IMAGE_SIZE if not is_mlp_checkpoint else None,
        'state_dict': model_state_dict,
        'optimizer': opt_state_dict,
        'scheduler': scheduler.state_dict() if scheduler else None,
        'scaler': scaler.state_dict() if scaler else None,
        'best_metric': best_metric,
        'label_encoder_classes': list(label_encoder.classes_) if label_encoder else None,
        'class_frequencies': CLASS_FREQUENCIES if not is_mlp_checkpoint else None
    }
    try:
        torch.save(state, filepath)
        # More concise print statement
        metric_display = f"(Best {CHECKPOINT_MONITOR}: {best_metric:.4f})" if "best" in filename.lower() else ""
        epoch_display = global_epoch # For MLP, global_epoch is the epoch number
        if not is_mlp_checkpoint: # For CombinedModel, use stage_epoch if relevant, or global_epoch
            epoch_display = stage_epoch if stage_idx != -1 else global_epoch

    except Exception as e:
        print(f"{TermColors.RED}Error saving checkpoint {filepath}: {e}{TermColors.ENDC}")

def load_checkpoint(fold, model, optimizer, scheduler, scaler, filename="checkpoint.pth.tar", is_mlp_checkpoint=False):
    checkpoint_prefix = "mlp_" if is_mlp_checkpoint else ""
    checkpoint_dir = os.path.join(BASE_CHECKPOINT_DIR, f"{checkpoint_prefix}fold_{fold}")
    filepath = os.path.join(checkpoint_dir, filename)
    start_global_epoch, start_stage_idx, start_stage_epoch = 0, 0, 0
    loaded_image_size, loaded_class_frequencies, loaded_label_classes = None, None, None
    best_metric = float('-inf') if CHECKPOINT_MODE == 'max' else float('inf')

    if os.path.isfile(filepath):
        print(f"{TermColors.CYAN}Loading {'MLP ' if is_mlp_checkpoint else ''}Fold {fold} checkpoint '{filename}'...{TermColors.ENDC}")
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
            except RuntimeError as e: print(f"{TermColors.YELLOW}Model Load Warning (strict=False): {e}{TermColors.ENDC}")
            except Exception as e: print(f"{TermColors.RED}Model Load Failed: {e}{TermColors.ENDC}")


            if optimizer and 'optimizer' in ckpt and ckpt['optimizer']:
                is_sam_optimizer_runtime = hasattr(optimizer, 'base_optimizer') and SAM_AVAILABLE and (USE_SAM if not is_mlp_checkpoint else False)
                opt_to_load = optimizer.base_optimizer if is_sam_optimizer_runtime else optimizer
                try:
                    opt_to_load.load_state_dict(ckpt['optimizer'])
                    print(f"{TermColors.GREEN}  Optimizer state loaded.{TermColors.ENDC}")
                except Exception as e: print(f"{TermColors.YELLOW}Optim Load Failed: {e}{TermColors.ENDC}")
            
            if scheduler and 'scheduler' in ckpt and ckpt['scheduler']:
                 try:
                     scheduler.load_state_dict(ckpt['scheduler'])
                     print(f"{TermColors.GREEN}  Scheduler state loaded (initial).{TermColors.ENDC}")
                 except Exception as e: print(f"{TermColors.YELLOW}Scheduler Load Failed (initial): {e}{TermColors.ENDC}")

            if scaler and 'scaler' in ckpt and ckpt['scaler']:
                try:
                    scaler.load_state_dict(ckpt['scaler'])
                    print(f"{TermColors.GREEN}  Scaler state loaded.{TermColors.ENDC}")
                except Exception as e: print(f"{TermColors.YELLOW}Scaler Load Failed: {e}{TermColors.ENDC}")

            print(f"{TermColors.GREEN}Ckpt {'MLP ' if is_mlp_checkpoint else ''}Fold {fold} loaded. Resume GlobEp {start_global_epoch}. Best {CHECKPOINT_MONITOR}: {best_metric:.4f}{TermColors.ENDC}")
            if loaded_label_classes and label_encoder and list(label_encoder.classes_) != loaded_label_classes:
                print(f"{TermColors.CRITICAL}Label mapping mismatch! Exiting.{TermColors.ENDC}"); sys.exit(1)
        except Exception as e:
            print(f"{TermColors.RED}Error loading checkpoint {filepath}: {e}{TermColors.ENDC}"); traceback.print_exc()
            start_global_epoch, start_stage_idx, start_stage_epoch = 0, 0, 0
            best_metric = float('-inf') if CHECKPOINT_MODE == 'max' else float('inf')
    else:
        print(f"{TermColors.YELLOW}No checkpoint found for {'MLP ' if is_mlp_checkpoint else ''}Fold {fold} at {filepath}. Starting fresh.{TermColors.ENDC}")
    
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
    except Exception as e:
        print(f"{TermColors.RED}Error saving model state_dict {filepath}: {e}{TermColors.ENDC}")

def log_misclassified(fold, model, dataloader, criterion, device, global_epoch, writer, num_classes, max_images=20, is_mlp_logging=False):
    if not LOG_MISCLASSIFIED_IMAGES: return
    log_prefix = "mlp_" if is_mlp_logging else ""
    error_dir = os.path.join(BASE_ERROR_ANALYSIS_DIR, f"{log_prefix}fold_{fold}")
    os.makedirs(error_dir, exist_ok=True)
    error_log_file = os.path.join(error_dir, f"epoch_{global_epoch}_errors.csv")
    model.eval(); misclassified_count = 0; logged_images = 0
    print(f"{TermColors.CYAN}{'MLP ' if is_mlp_logging else ''}Fold {fold} Logging misclassified images for global epoch {global_epoch}...{TermColors.ENDC}")
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
        print(f"{TermColors.CYAN}{'MLP ' if is_mlp_logging else ''}Fold {fold} Misclassified images logged ({misclassified_count} total errors). CSV: {error_log_file}{TermColors.ENDC}")
    except Exception as e: print(f"{TermColors.RED}Error during misclassified logging for fold {fold}: {e}{TermColors.ENDC}"); traceback.print_exc()

# --- Dataset and Transforms ---
class PlantDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None, label_encoder=None, include_paths=False, image_size=None):
        self.input_df = dataframe.copy()
        self.image_dir = image_dir
        self.transform = transform
        self.include_paths = include_paths
        self.image_size = image_size if image_size else PROGRESSIVE_RESIZING_STAGES[0][1] # Fallback
        self.image_data = []
        
        required_cols = ['scientificName', 'id', 'label']
        if not all(col in self.input_df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in self.input_df.columns]
            print(f"{TermColors.RED}PlantDataset input dataframe missing required columns: {missing}. Found: {self.input_df.columns.tolist()}{TermColors.ENDC}")
            self.dataframe = pd.DataFrame(self.image_data); return

        # Optimized file discovery using os.walk
        image_file_lookup = defaultdict(list) # Stores {(species_dir_name, obs_id_from_file): [list_of_paths]}
        print(f"{TermColors.DEBUG}Scanning {self.image_dir} for images...{TermColors.ENDC}")
        if not os.path.isdir(self.image_dir):
            print(f"{TermColors.RED}Image directory not found: {self.image_dir}{TermColors.ENDC}")
            self.dataframe = pd.DataFrame(self.image_data); return
            
        for root, _, files in os.walk(self.image_dir):
            species_dir_name_from_path = os.path.basename(root) # Assumes species name is the immediate parent folder
            prefix_to_strip = species_dir_name_from_path + "_"
            for filename in files:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root, filename)
                    try:
                        if os.path.getsize(full_path) > 0: # Check if file is not empty
                            # Attempt to parse observation ID from filename
                            # Example: "Species_Name_12345_variant.jpg" -> obs_id = "12345"
                            if filename.startswith(prefix_to_strip):
                                id_part = filename[len(prefix_to_strip):]
                                obs_id_from_file = id_part.split('_')[0]
                                if '.' in obs_id_from_file: # Handle cases like "12345.jpg"
                                    obs_id_from_file = obs_id_from_file.split('.')[0]
                                
                                if obs_id_from_file: # Ensure an ID was parsed
                                    image_file_lookup[(species_dir_name_from_path, obs_id_from_file)].append(full_path)
                    except OSError:
                        # print(f"{TermColors.YELLOW}Warn: Could not get size for {full_path}. Skipping.{TermColors.ENDC}")
                        pass # Skip if getsize fails (e.g. broken symlink or permission issue)
        print(f"{TermColors.DEBUG}Finished scanning. Found {sum(len(v) for v in image_file_lookup.values())} potential image files in lookup.{TermColors.ENDC}")

        self.input_df['original_index'] = self.input_df.index # Keep track of original df index
        for idx, row in tqdm(self.input_df.iterrows(), total=len(self.input_df), desc="Matching CSV to image files", leave=False, disable=DEBUG_MODE):
            try:
                species_name_csv = str(row['scientificName'])
                obs_id_csv = str(row['id']) # Ensure obs_id from CSV is string for matching
                label = row['label']
                original_idx_in_df = row['original_index']

                # Derive the directory name format used in the file system from the CSV's scientificName
                # This needs to consistently match how species_dir_name_from_path was derived during os.walk
                species_dir_name_csv_derived = species_name_csv.replace(' ', '_').replace('/', '_').replace('\\', '_')
                
                found_files = image_file_lookup.get((species_dir_name_csv_derived, obs_id_csv), [])
                
                for full_path in found_files:
                    self.image_data.append({
                        'scientificName': species_name_csv, 
                        'label': label, 
                        'image_path': full_path,
                        'original_index': original_idx_in_df # Store original index from input_df
                    })
            except Exception as e:
                print(f"{TermColors.RED}Error processing CSV row {idx} (ID: {row.get('id', 'N/A')}): {e}{TermColors.ENDC}")
        
        self.dataframe = pd.DataFrame(self.image_data)
        found_count = len(self.dataframe)
        if found_count == 0 and len(self.input_df) > 0:
             print(f"{TermColors.RED}Found 0 image files after matching CSV to file system. Check paths/IDs and naming conventions.{TermColors.ENDC}")
        else:
            print(f"{TermColors.INFO}PlantDataset initialized. Matched {found_count} image entries from {len(self.input_df)} CSV rows.{TermColors.ENDC}")


    def __len__(self): return len(self.dataframe)
    def get_labels(self):
        if 'label' in self.dataframe.columns: return self.dataframe['label'].tolist()
        else: print(f"{TermColors.RED}'label' column missing in internal dataframe!{TermColors.ENDC}"); return []

    def __getitem__(self, idx):
        if idx >= len(self.dataframe):
             # This case should ideally not be hit if DataLoader and sampler are correct
             print(f"{TermColors.RED}Index {idx} out of bounds for PlantDataset (len {len(self.dataframe)}). Returning dummy data.{TermColors.ENDC}")
             dummy_img = torch.zeros((3, *self.image_size), dtype=torch.float32); label = -1 # Or a specific error label
             original_index = -1 # Indicate error
             return (dummy_img, label, "ERROR_INDEX_OUT_OF_BOUNDS", original_index) if self.include_paths else (dummy_img, label)
        
        row = self.dataframe.iloc[idx]; img_path = row['image_path']; label = row['label']
        original_index = row['original_index'] # Get the original index from the input_df

        try:
            # Reading image with PIL and converting to numpy array
            image = Image.open(img_path).convert('RGB')
            image = np.array(image) # Convert PIL Image to numpy array for Albumentations
            if image is None: # Should not happen with PIL if file is valid image
                raise IOError("Image loading returned None")
        except Exception as e:
            print(f"{TermColors.RED}Error loading image {img_path}: {e}. Returning dummy data.{TermColors.ENDC}")
            dummy_img = torch.zeros((3, *self.image_size), dtype=torch.float32)
            error_filename = os.path.basename(img_path) if isinstance(img_path, str) else "UNKNOWN_FILE"
            # Return original_index even for dummy data to maintain batch structure if include_paths is True
            return (dummy_img, label if isinstance(label, int) else -1, f"ERROR_LOAD_{error_filename}", original_index) if self.include_paths else (dummy_img, label if isinstance(label, int) else -1)

        if self.transform:
            try:
                augmented = self.transform(image=image)
                image = augmented['image']
            except Exception as e:
                print(f"{TermColors.RED}Error applying transform to {img_path}: {e}. Attempting fallback basic transform.{TermColors.ENDC}")
                try:
                    # Fallback to basic torchvision transform if albumentations fails
                    pil_image = Image.fromarray(image) # Convert back to PIL for torchvision
                    fallback_transform = transforms.Compose([
                        transforms.Resize(self.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                    ])
                    image = fallback_transform(pil_image)
                except Exception as fb_e:
                    print(f"{TermColors.RED}Fallback transform also failed for {img_path}: {fb_e}. Returning dummy data.{TermColors.ENDC}")
                    dummy_img = torch.zeros((3, *self.image_size), dtype=torch.float32)
                    error_filename = os.path.basename(img_path) if isinstance(img_path, str) else "UNKNOWN_FILE"
                    return (dummy_img, label if isinstance(label, int) else -1, f"ERROR_TRANSFORM_{error_filename}", original_index) if self.include_paths else (dummy_img, label if isinstance(label, int) else -1)
        
        if self.include_paths:
            return image, label, img_path, original_index
        else:
            return image, label

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
        if label is None: 
            return cosine * self.s
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
            except Exception as e: print(f"{TermColors.RED}Backbone Load Fail {name}: {e}{TermColors.ENDC}"); raise e
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
            if labels is not None:
                x = self.metric_fc(x, labels)
            else:
                # During inference when labels unavailable (should rarely happen)
                print(f"{TermColors.YELLOW}Warning: No labels provided to ArcFace during forward pass, using cosine outputs.{TermColors.ENDC}")
                # Use the internal weight of ArcFace for cosine similarity
                x = self.metric_fc(x, None) 
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
    if hasattr(averaged_model_parameter, '_is_mlp_ema_param') and averaged_model_parameter._is_mlp_ema_param:
        decay = MLP_EMA_DECAY
    elif hasattr(averaged_model_parameter, '_is_kd_student_ema_param') and averaged_model_parameter._is_kd_student_ema_param:
        decay = KD_STUDENT_EMA_DECAY
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
                    # Explicitly check if it's MLP validation with ArcFace
                    use_arcface_for_mlp = is_mlp_validation and MLP_USE_ARCFACE and hasattr(current_model, 'metric_fc') and isinstance(current_model.metric_fc, ArcFace)
                    
                    # Pass labels when using ArcFace
                    outputs_orig = current_model(inputs, labels=labels if use_arcface_for_mlp or current_metric_learning_type == 'ArcFace' else None)
                    
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
    print(f"{TermColors.CYAN}Training Stacking Meta-Model...{TermColors.ENDC}")

    # --- Debugging: Print info about inputs ---
    print(f"{TermColors.DEBUG}  Stacking Input: oof_preds shape: {oof_preds.shape}, oof_labels shape: {oof_labels.shape}{TermColors.ENDC}")
    if oof_preds.size > 0:
        print(f"{TermColors.DEBUG}  oof_preds example (first 2 rows, up to 5 cols):\n{oof_preds[:2, :5]}{TermColors.ENDC}")
        print(f"{TermColors.DEBUG}  oof_preds stats: min={np.nanmin(oof_preds):.4f}, max={np.nanmax(oof_preds):.4f}, mean={np.nanmean(oof_preds):.4f}, NaNs={np.isnan(oof_preds).sum()}{TermColors.ENDC}")
    if oof_labels.size > 0:
        print(f"{TermColors.DEBUG}  oof_labels example (first 5): {oof_labels[:5]}{TermColors.ENDC}")
        print(f"{TermColors.DEBUG}  oof_labels unique: {np.unique(oof_labels, return_counts=True)}{TermColors.ENDC}")
    # --- End Debugging ---

    if oof_preds.ndim == 2 and oof_preds.shape[1] > 1: 
        oof_features = oof_preds
        print(f"{TermColors.DEBUG}  Using raw oof_preds as features for stacking. Shape: {oof_features.shape}{TermColors.ENDC}")
    elif oof_preds.ndim == 2 and oof_preds.shape[1] == 1: 
        oof_features = oof_preds
        print(f"{TermColors.DEBUG}  Using single column oof_preds as features. Shape: {oof_features.shape}{TermColors.ENDC}")
    elif oof_preds.ndim == 1: 
        oof_features = oof_preds.reshape(-1, 1)
        print(f"{TermColors.DEBUG}  Reshaped 1D oof_preds to be features. Shape: {oof_features.shape}{TermColors.ENDC}")
    else: 
        print(f"{TermColors.YELLOW}Warning: Unexpected oof_preds shape {oof_preds.shape}. Using argmax as a fallback (may be suboptimal).{TermColors.ENDC}")
        oof_features = np.argmax(oof_preds, axis=1).reshape(-1, 1)
        
    # --- CHOOSE AND TRAIN META MODEL ---
    meta_model_name_for_log = "LGBM" # Default
    meta_model = None
    hpo_best_cv_score = None
    print_accuracy_on = "OOF training data" # Default

    if LGBM_AVAILABLE:
        if STACKING_DO_HPO:
            print(f"{TermColors.DEBUG}  Attempting Hyperparameter Optimization for LGBMClassifier stacker (CV Folds: {STACKING_HPO_CV_FOLDS})...{TermColors.ENDC}")
            lgbm_hpo_model = GridSearchCV(LGBMClassifier(random_state=SEED, n_jobs=-1, verbosity=-1), 
                                          STACKING_LGBM_PARAM_GRID,
                                          cv=STACKING_HPO_CV_FOLDS, scoring='accuracy', n_jobs=-1, verbose=2) # Changed verbose to 2
            try:
                lgbm_hpo_model.fit(oof_features, oof_labels)
                meta_model = lgbm_hpo_model.best_estimator_ # This is already fitted on the full data by GridSearchCV
                hpo_best_cv_score = lgbm_hpo_model.best_score_
                meta_model_name_for_log = "LGBM_HPO"
                print_accuracy_on = f"OOF CV (HPO {STACKING_HPO_CV_FOLDS}-fold)"
                print(f"{TermColors.GREEN}  LGBM HPO complete. Best CV score: {hpo_best_cv_score:.4f}{TermColors.ENDC}")
                print(f"{TermColors.DEBUG}  Best LGBM HPO params: {lgbm_hpo_model.best_params_}{TermColors.ENDC}")
            except Exception as hpo_e:
                print(f"{TermColors.RED}  Error during LGBM HPO: {hpo_e}. Falling back to default LGBM.{TermColors.ENDC}")
                traceback.print_exc()
                meta_model = None # Ensure meta_model is None so default LGBM is tried next
        
        if meta_model is None: # If HPO was disabled or failed
            meta_model_name_for_log = "LGBM_Default"
            print(f"{TermColors.DEBUG}  Using default LGBMClassifier (HPO disabled or failed).{TermColors.ENDC}")
            meta_model = LGBMClassifier(
                n_estimators=500, learning_rate=0.05, num_leaves=31, max_depth=-1,
                min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                random_state=SEED, n_jobs=-1, verbosity=-1 # Added verbosity=-1
            )
            try:
                meta_model.fit(oof_features, oof_labels)
            except Exception as fit_e:
                print(f"{TermColors.RED}  Error fitting default LGBM: {fit_e}. Falling back to LogisticRegression.{TermColors.ENDC}")
                meta_model = None

    if meta_model is None: # Fallback to LogisticRegression if LGBM not available or all LGBM attempts failed
        if LGBM_AVAILABLE: # This means LGBM was available but fitting failed
             print(f"{TermColors.YELLOW}LGBM fitting failed. Falling back to LogisticRegression.{TermColors.ENDC}")
        else: # LGBM was not available from the start
            print(f"{TermColors.YELLOW}LGBM not available. Falling back to LogisticRegression.{TermColors.ENDC}")
        
        meta_model_name_for_log = "LogisticRegression" 
        meta_model = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0, solver='lbfgs', multi_class='multinomial', n_jobs=-1)
        print(f"{TermColors.DEBUG}  Using LogisticRegression.{TermColors.ENDC}")
        try:
            meta_model.fit(oof_features, oof_labels)
        except Exception as lr_fit_e:
            print(f"{TermColors.RED}  Error fitting LogisticRegression: {lr_fit_e}. Stacking model training failed.{TermColors.ENDC}")
            traceback.print_exc()
            return # Cannot proceed if even LogisticRegression fails
     
    # --- Evaluate and Save ---
    try:
        meta_acc = 0.0
        if hpo_best_cv_score is not None and meta_model_name_for_log == "LGBM_HPO":
            meta_acc = hpo_best_cv_score
            # print_accuracy_on is already set
        elif meta_model is not None: # For default LGBM or LogisticRegression
            meta_preds = meta_model.predict(oof_features) 
            meta_acc = accuracy_score(oof_labels, meta_preds)
            # print_accuracy_on is "OOF training data"
        else: # Should not happen if the logic above is correct
            print(f"{TermColors.RED}Meta model is None before evaluation. This is unexpected.{TermColors.ENDC}")
            return

        print(f"{TermColors.GREEN}Stacking meta-model ({meta_model_name_for_log}) trained. Accuracy ({print_accuracy_on}): {meta_acc:.4f}{TermColors.ENDC}")
        
        stacking_package = {"model": meta_model, "scaler": None} 
        joblib.dump(stacking_package, save_path)
        print(f"  Meta-model (and scaler if used) saved to: {save_path}")
    except Exception as e: 
        print(f"{TermColors.RED}Error evaluating or saving stacking meta-model ({meta_model_name_for_log}): {e}{TermColors.ENDC}")
        traceback.print_exc()

# --- Auto Training Configuration ---
class AutoTrainingConfig: # Primarily for CombinedModel end-to-end training
    def __init__(self, initial_lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                 plateau_patience=5, overfit_patience=3, lr_reduce_factor=0.5, wd_increase_factor=1.5,
                 min_lr_factor=0.01, max_wd_factor=10.0, sam_rho_adjust_factor=1.2, max_sam_rho=0.1):
        self.initial_lr = initial_lr
        self.initial_wd = weight_decay 
        self.current_lr = initial_lr
        self.current_wd = weight_decay 
        
        self.plateau_counter = 0  # Counts epochs without val_loss improvement
        self.overfit_counter = 0  # Counts epochs showing overfitting signs
        self.no_improvement_counter = 0 # Counts epochs without val_acc improvement (general stagnation)

        self.plateau_patience = plateau_patience
        self.overfit_patience = overfit_patience
        self.general_stagnation_patience = plateau_patience + 2 # Allow a bit longer if only acc stagnates

        self.lr_reduce_factor = lr_reduce_factor
        self.wd_increase_factor = wd_increase_factor
        self.min_lr = initial_lr * min_lr_factor
        self.max_wd = self.initial_wd * max_wd_factor 
        
        self.sam_rho_adjust_factor = sam_rho_adjust_factor
        self.max_sam_rho = max_sam_rho

        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.epoch_of_best_val_loss = -1
        self.epoch_of_best_val_acc = -1
        
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.lr_history = []
        self.wd_history = []
        self.adjustment_log = []
        self.last_adjustment_epoch = -self.plateau_patience # Allow immediate adjustment at start if needed

    def update(self, train_loss, train_acc, val_loss, val_acc, optimizer, scheduler, is_sam_active_runtime, current_epoch):
        self.train_losses.append(train_loss); self.val_losses.append(val_loss)
        self.train_accs.append(train_acc); self.val_accs.append(val_acc)

        opt_to_query = optimizer.base_optimizer if is_sam_active_runtime else optimizer
        self.lr_history.append(opt_to_query.param_groups[0]['lr']);
        self.wd_history.append(opt_to_query.param_groups[0]['weight_decay'])

        val_loss_improved = val_loss < self.best_val_loss
        val_acc_improved = val_acc > self.best_val_acc 
        
        made_adjustment_this_epoch = False

        if val_loss_improved:
            self.best_val_loss = val_loss
            self.epoch_of_best_val_loss = current_epoch + 1
            self.plateau_counter = 0
        else:
            self.plateau_counter += 1

        if val_acc_improved:
            self.best_val_acc = val_acc
            self.epoch_of_best_val_acc = current_epoch + 1
            self.no_improvement_counter = 0
        else:
            if (CHECKPOINT_MONITOR == 'val_acc' and not val_acc_improved) or \
               (CHECKPOINT_MONITOR != 'val_acc' and not val_loss_improved and not val_acc_improved):
                 self.no_improvement_counter +=1
            elif CHECKPOINT_MONITOR == 'val_loss' and not val_loss_improved and not val_acc_improved:
                 self.no_improvement_counter +=1

        is_overfitting = False
        if len(self.train_losses) >= 2 and len(self.val_losses) >=2: 
            train_loss_decreasing = self.train_losses[-1] < self.train_losses[-2]
            # Check for a more significant validation loss increase
            val_loss_significantly_increasing = self.val_losses[-1] > (self.val_losses[-2] * 1.03) # e.g., >3% increase

            current_gap = self.val_losses[-1] - self.train_losses[-1]
            prev_gap = self.val_losses[-2] - self.train_losses[-2]
            # Gap must be positive and widen by a noticeable margin
            gap_widening_condition = current_gap > 0.01 and current_gap > (prev_gap + 0.02) # Gap positive, and widens by at least 0.02

            # Minimum validation accuracy to consider these signs as true overfitting.
            # This ideally depends on NUM_CLASSES. For example, 1.5 / NUM_CLASSES.
            # Using a fixed small threshold for now.
            min_accuracy_for_overfit_detection = 0.05 # e.g., 5% accuracy. Adjust as needed.
            # If NUM_CLASSES is available to AutoTrainingConfig, you could set:
            # min_accuracy_for_overfit_detection = 1.5 / self.num_classes if hasattr(self, 'num_classes') and self.num_classes > 0 else 0.05


            if self.val_accs[-1] >= min_accuracy_for_overfit_detection and \
               train_loss_decreasing and \
               (val_loss_significantly_increasing or gap_widening_condition):
                self.overfit_counter += 1
                is_overfitting = True
            
            # Reset overfit_counter if conditions improve or not clearly overfitting
            # If val_loss improved OR (gap is not positive or gap is narrowing or not widening significantly)
            if not is_overfitting:
                if val_loss_improved or not (current_gap > 0.01 and current_gap > (prev_gap + 0.01)): # Less strict reset condition
                    self.overfit_counter = 0
        
        if current_epoch - self.last_adjustment_epoch < self.plateau_patience // 2 and self.last_adjustment_epoch >= 0:
            return False 

        # Get current LR from optimizer before deciding on new_lr
        current_optimizer_lr = opt_to_query.param_groups[0]['lr']
        current_optimizer_wd = opt_to_query.param_groups[0]['weight_decay']

        if self.overfit_counter >= self.overfit_patience:
            new_lr = max(current_optimizer_lr * self.lr_reduce_factor, self.min_lr) 
            new_wd = min(current_optimizer_wd * self.wd_increase_factor, self.max_wd) 
            
            if new_lr < current_optimizer_lr or new_wd > current_optimizer_wd:
                self._adjust_optimizer_params(optimizer, new_lr, new_wd, scheduler, is_sam_active_runtime)
                self.adjustment_log.append(f"Ep{current_epoch+1}: Overfit! LR -> {new_lr:.2e}, WD -> {new_wd:.2e}")
                made_adjustment_this_epoch = True
            self.overfit_counter = 0
            self.plateau_counter = 0 
            self.no_improvement_counter = 0
            self.last_adjustment_epoch = current_epoch
            if is_sam_active_runtime and hasattr(optimizer, 'rho'): self._adjust_sam_parameters(optimizer, is_overfitting)

        elif self.plateau_counter >= self.plateau_patience or self.no_improvement_counter >= self.general_stagnation_patience:
            new_lr = max(current_optimizer_lr * self.lr_reduce_factor, self.min_lr)
            new_wd = current_optimizer_wd 
            if current_optimizer_lr <= self.min_lr * 2: 
                new_wd = max(current_optimizer_wd * 0.9, self.initial_wd * 0.5) 

            if new_lr < current_optimizer_lr or new_wd < current_optimizer_wd :
                self._adjust_optimizer_params(optimizer, new_lr, new_wd, scheduler, is_sam_active_runtime)
                self.adjustment_log.append(f"Ep{current_epoch+1}: Plateau! LR -> {new_lr:.2e}, WD -> {new_wd:.2e}")
                made_adjustment_this_epoch = True
            
            self.plateau_counter = 0
            self.no_improvement_counter = 0
            self.last_adjustment_epoch = current_epoch
        
        return made_adjustment_this_epoch

    def _adjust_optimizer_params(self, optimizer, new_lr, new_wd, scheduler, is_sam_active_runtime):
        opt_to_adjust = optimizer.base_optimizer if is_sam_active_runtime else optimizer
        lr_changed_for_optimizer = False
        
        for i, param_group in enumerate(opt_to_adjust.param_groups):
            if param_group['lr'] != new_lr:
                param_group['lr'] = new_lr
                lr_changed_for_optimizer = True
            param_group['weight_decay'] = new_wd
        
        self.current_lr = new_lr # Update internal state after applying to optimizer
        self.current_wd = new_wd

        if lr_changed_for_optimizer and scheduler:
            target_scheduler = scheduler
            if isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR):
                if scheduler._schedulers and len(scheduler._schedulers) > 0:
                    # Assuming the main scheduler is the last one after warmup
                    target_scheduler = scheduler._schedulers[-1] 
            
            if hasattr(target_scheduler, 'base_lrs'):
                # Ensure base_lrs is a list (it should be)
                if isinstance(target_scheduler.base_lrs, list):
                    current_base_lrs_count = len(target_scheduler.base_lrs)
                    target_scheduler.base_lrs = [new_lr] * current_base_lrs_count
                    print(f"{TermColors.DEBUG}    AutoConfig: Updated scheduler base_lrs to {new_lr:.2e}{TermColors.ENDC}")
                else:
                    print(f"{TermColors.YELLOW}    AutoConfig Warn: scheduler.base_lrs is not a list, type: {type(target_scheduler.base_lrs)}. Cannot update.{TermColors.ENDC}")


    def _adjust_sam_parameters(self, sam_optimizer, is_overfitting):
        # This SAM rho adjustment is a simple heuristic, might need tuning
        current_rho = sam_optimizer.rho
        if is_overfitting and current_rho < self.max_sam_rho: # If overfitting, try increasing rho to find flatter minima
            new_rho = min(current_rho * self.sam_rho_adjust_factor, self.max_sam_rho)
            if new_rho > current_rho: 
                sam_optimizer.rho = new_rho
                self.adjustment_log.append(f"SAM rho -> {new_rho:.4f} (overfit)")
        # Could add logic to decrease rho if underfitting, but that's less common for SAM's role.

    def get_status_report(self):
        status = [
            f"Best Val Loss: {self.best_val_loss:.4f} (Epoch {self.epoch_of_best_val_loss if self.epoch_of_best_val_loss > 0 else 'N/A'})",
            f"Best Val Acc: {self.best_val_acc:.4f} (Epoch {self.epoch_of_best_val_acc if self.epoch_of_best_val_acc > 0 else 'N/A'})",
        ]
        
        num_recent_metrics = min(3, len(self.val_losses))

        if self.adjustment_log:
            status.append("Recent adjustments:")
            status.extend([f"  - {adj}" for adj in self.adjustment_log[-min(3, len(self.adjustment_log)):]]) # Show last up to 3 adjustments
        return "\n".join(status)

# --- MLP HPO Objective Function & Runner (NEW) ---
def mlp_hpo_objective(trial, features_hpo_train, labels_hpo_train, features_hpo_val, labels_hpo_val, input_size_mlp, num_classes_mlp):
    # Suggest Hyperparameters
    n_layers = trial.suggest_int('mlp_n_layers', 1, 3)
    hidden_dims_trial = []
    for i in range(n_layers):
        lower_bound_dim = max(32, num_classes_mlp // 2)
        upper_bound_dim = max(input_size_mlp // 2, 512)
        if lower_bound_dim >= upper_bound_dim: upper_bound_dim = lower_bound_dim + 32
        hidden_dims_trial.append(trial.suggest_int(f'mlp_h_dim_l{i}', lower_bound_dim, upper_bound_dim, step=32))
    
    dropout_rate_trial = trial.suggest_float('mlp_dropout_rate', 0.1, 0.7, step=0.05)
    lr_trial = trial.suggest_float('mlp_lr', 5e-5, 5e-3, log=True)
    wd_trial = trial.suggest_float('mlp_wd', 1e-6, 1e-2, log=True)
    
    optimizer_type_trial = 'AdamW'
    scheduler_type_trial = 'CosineWarmRestarts'

    temp_mlp_model = SimpleMLP(input_size_mlp, hidden_dims_trial, num_classes_mlp, dropout_rate_trial, MLP_USE_ARCFACE)
    temp_mlp_model = temp_mlp_model.to(DEVICE)

    temp_optimizer = get_optimizer(temp_mlp_model, optimizer_type=optimizer_type_trial, lr=lr_trial, weight_decay=wd_trial, is_mlp_optimizer=True)
    temp_scaler = torch.amp.GradScaler('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda'))
    temp_criterion = get_criterion(class_weights=CLASS_WEIGHTS, label_smoothing=(0.0 if MLP_USE_ARCFACE else LABEL_SMOOTHING))

    hpo_train_ds = FeatureDataset(features_hpo_train, labels_hpo_train)
    hpo_val_ds = FeatureDataset(features_hpo_val, labels_hpo_val)
    
    hpo_sampler = None
    if IMBALANCE_STRATEGY == 'WeightedSampler' and CLASS_WEIGHTS is not None:
        hpo_labels_list = hpo_train_ds.get_labels()
        if hpo_labels_list:
            hpo_class_sample_count = np.array([hpo_labels_list.count(l) for l in range(num_classes_mlp)])
            hpo_class_sample_count = np.maximum(hpo_class_sample_count, 1)
            hpo_weight = 1. / hpo_class_sample_count
            hpo_samples_weight = torch.from_numpy(np.array([hpo_weight[t] for t in hpo_labels_list])).double()
            hpo_sampler = WeightedRandomSampler(hpo_samples_weight, len(hpo_samples_weight))

    hpo_train_loader = DataLoader(hpo_train_ds, MLP_BATCH_SIZE, sampler=hpo_sampler, shuffle=(hpo_sampler is None), num_workers=0, pin_memory=True, drop_last=True)
    hpo_val_loader = DataLoader(hpo_val_ds, MLP_BATCH_SIZE * 2, shuffle=False, num_workers=0, pin_memory=True)
    temp_scheduler = get_scheduler(temp_optimizer, scheduler_type=scheduler_type_trial, total_epochs=MLP_HPO_EPOCHS, is_mlp_scheduler=True)

    best_hpo_trial_val_acc = -float('inf') 
    best_objectives_for_trial = (-float('inf'), -float('inf')) 
    epochs_no_improve_val_acc = 0

    for epoch in range(MLP_HPO_EPOCHS):
        if stop_requested: raise optuna.exceptions.TrialPruned("Global stop requested")
        
        train_loss, train_acc = train_one_epoch(temp_mlp_model, hpo_train_loader, temp_criterion, temp_optimizer, temp_scaler, temp_scheduler, 
                        epoch, 0, epoch, MLP_HPO_EPOCHS, DEVICE, None, num_classes_mlp, None,
                        use_sam_flag=False, fold_num="HPO", is_mlp_training=True)
        
        if train_loss is None:
            # If training is interrupted, we can't get a valid objective, so prune.
            # However, for multi-objective, it's better to return poor values or let it error out if Optuna expects a tuple.
            # For simplicity, let's return a very poor objective value.
            print(f"{TermColors.YELLOW}Trial {trial.number} training interrupted. Returning poor objectives.{TermColors.ENDC}")
            return (-float('inf'), -float('inf'))


        val_loss, val_acc, _, _ = validate_one_epoch(temp_mlp_model, hpo_val_loader, temp_criterion, DEVICE, epoch, None, num_classes_mlp, 
                                              scheduler=temp_scheduler, fold_num="HPO", is_mlp_validation=True)
        
        if val_loss is None:
            print(f"{TermColors.YELLOW}Trial {trial.number} validation interrupted. Returning poor objectives.{TermColors.ENDC}")
            return (-float('inf'), -float('inf'))

        overfitting_gap = train_acc - val_acc 
        current_objectives = (val_acc, -overfitting_gap) 

        if trial.number % 10 == 0 and epoch == 0 :
             print(f"{TermColors.DIM}  Trial {trial.number} Ep {epoch+1}: TrAcc={train_acc:.4f}, ValAcc={val_acc:.4f}, Gap={overfitting_gap:.4f}, Objs=({current_objectives[0]:.4f}, {current_objectives[1]:.4f}){TermColors.ENDC}")

        if val_acc > best_hpo_trial_val_acc:
            best_hpo_trial_val_acc = val_acc
            best_objectives_for_trial = current_objectives 
            epochs_no_improve_val_acc = 0
        else:
            epochs_no_improve_val_acc += 1
        
        # trial.report(val_acc, epoch) # REMOVED: Not supported for multi-objective
        # if trial.should_prune():     # REMOVED: Relies on trial.report()
        #     raise optuna.exceptions.TrialPruned()

        if epochs_no_improve_val_acc >= MLP_HPO_PATIENCE:
            # print(f"{TermColors.DIM}  Trial {trial.number}: Early stopping HPO trial at epoch {epoch+1}.{TermColors.ENDC}")
            break
    
    del temp_mlp_model, temp_optimizer, temp_scaler, temp_criterion, hpo_train_ds, hpo_val_ds, hpo_train_loader, hpo_val_loader, temp_scheduler
    gc.collect(); torch.cuda.empty_cache()
    return best_objectives_for_trial

def run_mlp_hpo(features_for_hpo_main, labels_for_hpo_main, original_indices_hpo_main, input_size_mlp, num_classes_mlp):
    global MLP_HIDDEN_DIMS, MLP_DROPOUT_RATE, MLP_LEARNING_RATE, MLP_WEIGHT_DECAY

    if not OPTUNA_AVAILABLE:
        print(f"{TermColors.YELLOW}Optuna not available. Skipping MLP HPO.{TermColors.ENDC}")
        return

    print(f"\n{TermColors.CYAN}--- Running MLP Hyperparameter Optimization ({MLP_HPO_N_TRIALS} trials) ---{TermColors.ENDC}")
    print(f"{TermColors.DEBUG}  HPO using {len(features_for_hpo_main)} samples. Internal Val Split: {MLP_HPO_INTERNAL_VAL_SPLIT * 100}%{TermColors.ENDC}")
    
    hpo_train_feats, hpo_val_feats, hpo_train_labels, hpo_val_labels = None, None, None, None
    try:
        hpo_train_feats, hpo_val_feats, hpo_train_labels, hpo_val_labels, _, _ = train_test_split(
            features_for_hpo_main, labels_for_hpo_main, original_indices_hpo_main,
            test_size=MLP_HPO_INTERNAL_VAL_SPLIT, 
            random_state=SEED + 100, 
            stratify=labels_for_hpo_main
        )
    except ValueError as e: 
        print(f"{TermColors.YELLOW}Warning: Stratified split for MLP HPO's internal validation set failed ({e}). Using non-stratified split.{TermColors.ENDC}")
        try:
            hpo_train_feats, hpo_val_feats, hpo_train_labels, hpo_val_labels, _, _ = train_test_split(
                features_for_hpo_main, labels_for_hpo_main, original_indices_hpo_main,
                test_size=MLP_HPO_INTERNAL_VAL_SPLIT, random_state=SEED + 100
            )
        except Exception as e_non_strat:
            print(f"{TermColors.RED}Non-stratified split for MLP HPO also failed: {e_non_strat}. Skipping HPO.{TermColors.ENDC}")
            return # Cannot proceed if data splitting fails completely
    
    if hpo_train_feats is None or len(hpo_train_feats) == 0 or hpo_val_feats is None or len(hpo_val_feats) == 0:
        print(f"{TermColors.RED}MLP HPO internal data split resulted in empty train or validation set. Skipping HPO.{TermColors.ENDC}")
        return

    # For multi-objective, specify directions for each objective
    # Pruner might not be effective without trial.report, but Optuna will still optimize.
    # Consider removing pruner if it causes issues or if trials are short enough.
    study = optuna.create_study(directions=['maximize', 'maximize'], pruner=optuna.pruners.MedianPruner(n_warmup_steps=max(1, MLP_HPO_EPOCHS // 5)))
    
    objective_fn = lambda trial: mlp_hpo_objective(trial, hpo_train_feats, hpo_train_labels, hpo_val_feats, hpo_val_labels, input_size_mlp, num_classes_mlp)
    
    try:
        study.optimize(objective_fn, 
                       n_trials=MLP_HPO_N_TRIALS, 
                       callbacks=[lambda study, trial: gc.collect() or torch.cuda.empty_cache()])
    except Exception as e_opt: # Catch errors during study.optimize itself
        print(f"{TermColors.RED}Error during Optuna study.optimize: {e_opt}{TermColors.ENDC}")
        traceback.print_exc()
        print(f"{TermColors.YELLOW}MLP HPO run failed. Using default MLP parameters.{TermColors.ENDC}")
        return


    pareto_optimal_trials = []
    try:
        pareto_optimal_trials = study.best_trials
    except Exception as e_best_trials:
        print(f"{TermColors.RED}Error retrieving best_trials from Optuna study: {e_best_trials}{TermColors.ENDC}")
        # Fall through to check if any trials completed successfully at all
    
    if not pareto_optimal_trials:
        # Check if there are *any* completed trials, even if not "best_trials" (e.g., if all failed)
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
            print(f"{TermColors.RED}MLP HPO: No successful trials found. Using default MLP parameters.{TermColors.ENDC}")
            return
        else:
            # If there are completed trials but best_trials is empty (should be rare if study.optimize didn't error out),
            # we could try to pick the best from all completed ones, but this indicates an issue.
            # For now, let's assume if best_trials is empty, something went wrong.
            print(f"{TermColors.YELLOW}MLP HPO: No Pareto optimal trials identified by Optuna (study.best_trials was empty), though {len(completed_trials)} trials completed. Using default MLP parameters.{TermColors.ENDC}")
            return


    best_trial_overall = pareto_optimal_trials[0]
    for trial_info in pareto_optimal_trials:
        if trial_info.values is None: # Should not happen for completed trials in best_trials
            print(f"{TermColors.YELLOW}Warning: Trial {trial_info.number} in Pareto front has no values. Skipping.{TermColors.ENDC}")
            continue
        if best_trial_overall.values is None or trial_info.values[0] > best_trial_overall.values[0]: 
            best_trial_overall = trial_info
        elif trial_info.values[0] == best_trial_overall.values[0]: 
            if best_trial_overall.values is None or trial_info.values[1] > best_trial_overall.values[1]:
                best_trial_overall = trial_info
                
    if best_trial_overall.values is None:
        print(f"{TermColors.RED}MLP HPO: Could not determine a best trial from Pareto front (all had None values). Using default MLP parameters.{TermColors.ENDC}")
        return

    best_params = best_trial_overall.params
    best_values = best_trial_overall.values

    print(f"{TermColors.GREEN}MLP HPO Complete. Selected Pareto Optimal Trial with Val Acc: {best_values[0]:.4f}, Neg Overfitting Gap: {best_values[1]:.4f}{TermColors.ENDC}")

# --- Knowledge Distillation ---
class DistillationLoss(nn.Module):
    def __init__(self, alpha=KD_ALPHA, temperature=KD_TEMPERATURE, base_criterion=nn.CrossEntropyLoss()):
        super().__init__(); self.alpha = alpha; self.T = temperature; self.base_criterion = base_criterion; self.KLDiv = nn.KLDivLoss(reduction='batchmean')
    def forward(self, student_outputs, teacher_outputs, labels):
        soft_teacher_log_probs = F.log_softmax(teacher_outputs / self.T, dim=1); soft_student_log_probs = F.log_softmax(student_outputs / self.T, dim=1)
        distillation_loss = self.KLDiv(soft_student_log_probs, soft_teacher_log_probs) * (self.alpha * self.T * self.T)
        student_loss = self.base_criterion(student_outputs, labels); total_loss = distillation_loss + (1. - self.alpha) * student_loss
        return total_loss

def train_student_model(
    student_model_name,
    student_base_save_path, # Path for the base student model
    student_swa_save_path,  # Path for the SWA student model
    student_ema_save_path,  # Path for the EMA student model
    df_full_for_kd, # Full dataframe to be split for KD train/val
    image_dir,
    num_classes,
    stacking_teacher_components # Dictionary with paths and configs for stacking teacher
    ):
    print(f"\n{TermColors.HEADER}--- Knowledge Distillation with Stacking Teacher ---{TermColors.ENDC}")

    # --- Teacher Setup (Stacking Model) ---
    actual_stacking_model = None
    stacking_scaler = None
    feature_extractor_for_teacher = None
    base_mlp_models_for_teacher = []
    teacher_loaded_successfully = False

    try:
        print(f"{TermColors.INFO}Loading Stacking Teacher components...{TermColors.ENDC}")
        # 1. Load Stacking Model (scikit-learn) and Scaler
        stacking_package = joblib.load(stacking_teacher_components["stacking_model_path"])
        actual_stacking_model = stacking_package['model']
        stacking_scaler = stacking_package['scaler']
        print(f"{TermColors.GREEN}  Stacking teacher model and scaler loaded from: {stacking_teacher_components['stacking_model_path']}{TermColors.ENDC}")

        # 2. Load Feature Extractor (CombinedModel used for MLPs)
        fe_names = stacking_teacher_components["feature_extractor_model_names"]
        fe_embed_size = stacking_teacher_components["feature_extractor_embedding_size"]
        fe_arc_m = stacking_teacher_components["feature_extractor_arcface_m"]
        fe_metric_learning = stacking_teacher_components["feature_extractor_metric_learning"]
        fe_ckpt_path = stacking_teacher_components.get("feature_extractor_checkpoint_path") # Optional

        feature_extractor_for_teacher = build_model(
            model_names=fe_names, num_classes=num_classes, pretrained=PRETRAINED, # num_classes for FE doesn't strictly matter if return_embedding
            embedding_size=fe_embed_size, arcface_m=fe_arc_m, metric_learning=fe_metric_learning
        )
        if fe_ckpt_path and os.path.exists(fe_ckpt_path):
            fe_ckpt = torch.load(fe_ckpt_path, map_location=DEVICE)
            # Handle potential 'module.' prefix if model was saved with DataParallel
            fe_state_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in fe_ckpt['state_dict'].items()}
            feature_extractor_for_teacher.load_state_dict(fe_state_dict, strict=False)
            print(f"{TermColors.GREEN}  Feature extractor for teacher loaded from checkpoint: {fe_ckpt_path}{TermColors.ENDC}")
        else:
            print(f"{TermColors.INFO}  Using {'pretrained' if PRETRAINED else 'randomly initialized'} weights for feature extractor for teacher.{TermColors.ENDC}")
        feature_extractor_for_teacher = feature_extractor_for_teacher.to(DEVICE); feature_extractor_for_teacher.eval()

        # 3. Load Base MLP Models (PyTorch)
        base_mlp_input_size = stacking_teacher_components["base_mlp_model_input_size"]
        expected_mlp_folds = stacking_teacher_components["n_folds_for_stacking"]
        for mlp_path in stacking_teacher_components["base_mlp_model_paths"]:
            if not os.path.exists(mlp_path):
                print(f"{TermColors.RED}  Base MLP model path not found: {mlp_path}. Skipping this MLP.{TermColors.ENDC}")
                continue
            mlp = build_mlp_model(input_size=base_mlp_input_size, num_classes=num_classes) # Use global num_classes
            mlp.load_state_dict(torch.load(mlp_path, map_location=DEVICE))
            mlp = mlp.to(DEVICE); mlp.eval()
            base_mlp_models_for_teacher.append(mlp)
        
        if len(base_mlp_models_for_teacher) == expected_mlp_folds:
             print(f"{TermColors.GREEN}  All {len(base_mlp_models_for_teacher)} base MLP models for teacher loaded.{TermColors.ENDC}")
             teacher_loaded_successfully = True
        else:
            print(f"{TermColors.RED}  Failed to load all base MLP models for stacking teacher. Expected {expected_mlp_folds}, Got {len(base_mlp_models_for_teacher)}.{TermColors.ENDC}")

    except Exception as e:
        print(f"{TermColors.RED}Failed to load components for stacking teacher: {e}. Skipping KD.{TermColors.ENDC}"); traceback.print_exc()
        teacher_loaded_successfully = False

    if not teacher_loaded_successfully:
        print(f"{TermColors.RED}Stacking teacher setup failed. Aborting Knowledge Distillation.{TermColors.ENDC}")
        return

    # --- Student Model Setup (Image-based) ---
    try:
        # Student model does not use metric learning head for distillation typically, just plain CE
        student_model = build_model(model_names=[student_model_name], num_classes=num_classes, pretrained=True,
                                    dropout_rate=KD_STUDENT_DROPOUT, embedding_size=KD_STUDENT_EMBEDDING_SIZE,
                                    metric_learning='None') # Student uses standard CE head
        student_model = student_model.to(DEVICE)
        print(f"{TermColors.GREEN}Student model '{student_model_name}' built.{TermColors.ENDC}")
    except Exception as e:
        print(f"{TermColors.RED}Failed to build student model: {e}. Skipping KD.{TermColors.ENDC}"); traceback.print_exc(); return

    # --- Dataloaders for Student (Image-based) ---
    try:
        df_kd_train, df_kd_val = train_test_split(df_full_for_kd, test_size=0.2, random_state=SEED + 42, stratify=df_full_for_kd['label']) # Use a different split for KD
        
        train_tf_kd, val_tf_kd = get_transforms(image_size=KD_STUDENT_IMAGE_SIZE) # Student uses its own image size
        train_ds_kd = PlantDataset(df_kd_train, image_dir, train_tf_kd, label_encoder, False, KD_STUDENT_IMAGE_SIZE)
        val_ds_kd = PlantDataset(df_kd_val, image_dir, val_tf_kd, label_encoder, False, KD_STUDENT_IMAGE_SIZE)

        if not train_ds_kd or not val_ds_kd or len(train_ds_kd) == 0 or len(val_ds_kd) == 0:
            print(f"{TermColors.RED}KD Dataset is empty after split. Train: {len(train_ds_kd)}, Val: {len(val_ds_kd)}. Skipping KD.{TermColors.ENDC}"); return

        # Sampler for KD student training if needed
        kd_sampler = None
        if IMBALANCE_STRATEGY == 'WeightedSampler' and CLASS_WEIGHTS is not None: # Reuse global strategy
            kd_train_labels_list = train_ds_kd.get_labels()
            if kd_train_labels_list:
                kd_class_sample_count = np.array([kd_train_labels_list.count(l) for l in range(num_classes)])
                kd_class_sample_count = np.maximum(kd_class_sample_count, 1) # Avoid division by zero
                kd_weight = 1. / kd_class_sample_count
                kd_samples_weight = torch.from_numpy(np.array([kd_weight[t] for t in kd_train_labels_list])).double()
                kd_sampler = WeightedRandomSampler(kd_samples_weight, len(kd_samples_weight))

        train_loader_kd = DataLoader(train_ds_kd, KD_BATCH_SIZE, sampler=kd_sampler, shuffle=(kd_sampler is None), num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
        val_loader_kd = DataLoader(val_ds_kd, KD_BATCH_SIZE * 2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        print(f"{TermColors.GREEN}KD Dataloaders ready. Train batches: {len(train_loader_kd)}, Val batches: {len(val_loader_kd)}{TermColors.ENDC}")
    except Exception as e:
        print(f"{TermColors.RED}Error creating KD dataloaders: {e}. Skipping KD.{TermColors.ENDC}"); traceback.print_exc(); return

    # --- Student Training Loop ---
    # Student uses standard CE loss for its own predictions, KLDiv for matching teacher
    student_base_criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING if LOSS_TYPE == 'CrossEntropy' else 0.0) # Student's own loss
    criterion_kd = DistillationLoss(alpha=KD_ALPHA, temperature=KD_TEMPERATURE, base_criterion=student_base_criterion)
    
    optimizer_kd = optim.AdamW(student_model.parameters(), lr=KD_LR, weight_decay=WEIGHT_DECAY) # Student uses KD_LR
    scaler_kd = torch.amp.GradScaler('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda'))
    scheduler_kd = CosineAnnealingLR(optimizer_kd, T_max=KD_EPOCHS, eta_min=KD_LR * 0.01) # Simple scheduler for student

    # SWA and EMA for the student model
    swa_student_model = None
    swa_student_scheduler = None
    kd_student_swa_start_epoch = int(KD_EPOCHS * KD_STUDENT_SWA_START_EPOCH_FACTOR)
    if KD_STUDENT_USE_SWA:
        swa_student_model = AveragedModel(student_model, avg_fn=ema_avg_fn if KD_STUDENT_USE_EMA else None)
        if KD_STUDENT_USE_EMA and hasattr(swa_student_model, 'module'): # Check if AveragedModel has wrapped the student_model
            for p_swa_stud in swa_student_model.module.parameters(): p_swa_stud._is_kd_student_ema_param = True
        elif KD_STUDENT_USE_EMA: # If not wrapped (e.g. if student_model itself is AveragedModel, though unlikely here)
             for p_swa_stud in swa_student_model.parameters(): p_swa_stud._is_kd_student_ema_param = True
        swa_student_scheduler = SWALR(optimizer_kd, swa_lr=(KD_LR * KD_STUDENT_SWA_LR_FACTOR), anneal_epochs=KD_STUDENT_SWA_ANNEAL_EPOCHS, anneal_strategy='cos')

    ema_student_model = None
    if KD_STUDENT_USE_EMA:
        ema_student_model = AveragedModel(student_model, avg_fn=ema_avg_fn)
        if hasattr(ema_student_model, 'module'):
            for p_ema_stud in ema_student_model.module.parameters(): p_ema_stud._is_kd_student_ema_param = True
        else:
            for p_ema_stud in ema_student_model.parameters(): p_ema_stud._is_kd_student_ema_param = True


    print(f"{TermColors.CYAN}Starting KD Student Training ({KD_EPOCHS} epochs)...{TermColors.ENDC}")
    best_kd_val_acc = 0.0

    for epoch in range(KD_EPOCHS):
        if stop_requested: break
        print(f"\n{TermColors.MAGENTA}--- KD Student Epoch {epoch+1}/{KD_EPOCHS} ---{TermColors.ENDC}")
        student_model.train()
        running_loss_kd = 0.0
        total_samples_kd = 0
        
        progress_bar_kd = tqdm(train_loader_kd, desc=f"KD Student Train E{epoch+1}", leave=False, bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}')
        for batch_idx, (inputs, labels) in enumerate(progress_bar_kd):
            if stop_requested: break
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            batch_size = inputs.size(0)

            # 1. Get Teacher's "soft labels" from the stacking ensemble
            teacher_outputs_logits = None
            with torch.no_grad():
                # a. Extract features from student's input images using the loaded feature extractor
                img_features_for_teacher = feature_extractor_for_teacher(inputs, return_embedding=True)
                
                # b. Pass these features through each base MLP model
                mlp_predictions_for_stacker_list = []
                for mlp_model_teacher in base_mlp_models_for_teacher:
                    mlp_logits = mlp_model_teacher(img_features_for_teacher)
                    mlp_probs = F.softmax(mlp_logits, dim=1) # Stacking model was likely trained on probabilities
                    mlp_predictions_for_stacker_list.append(mlp_probs)
                
                # c. Concatenate MLP probabilities (ensure order and format matches stacking training)
                stacker_input_features_tensor = torch.cat(mlp_predictions_for_stacker_list, dim=1)
                stacker_input_np = stacker_input_features_tensor.cpu().numpy()
                
                # d. Scale these features using the stacking model's scaler
                scaled_stacker_input_np = stacking_scaler.transform(stacker_input_np)
                
                # e. Use the scikit-learn stacking model to predict probabilities
                teacher_output_probs_np = actual_stacking_model.predict_proba(scaled_stacker_input_np)
                
                # f. Convert probabilities to logits for DistillationLoss (add small epsilon for stability)
                teacher_outputs_logits = torch.log(torch.tensor(teacher_output_probs_np, dtype=torch.float32).to(DEVICE) + 1e-9)

            if teacher_outputs_logits is None:
                print(f"{TermColors.RED}KD: Teacher outputs not generated for batch. Skipping batch.{TermColors.ENDC}")
                continue

            # 2. Student forward pass and loss calculation
            with torch.amp.autocast('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda')):
                student_outputs_logits = student_model(inputs) # Student model is image-based
                loss = criterion_kd(student_outputs_logits, teacher_outputs_logits, labels)
            
            optimizer_kd.zero_grad()
            scaler_kd.scale(loss).backward()
            scaler_kd.step(optimizer_kd)
            scaler_kd.update()

            if KD_STUDENT_USE_EMA and ema_student_model:
                ema_student_model.update_parameters(student_model)

            if not torch.isnan(loss):
                running_loss_kd += loss.item() * batch_size
                total_samples_kd += batch_size
                if batch_idx % 20 == 0 or batch_idx == len(train_loader_kd) -1 :
                    progress_bar_kd.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer_kd.param_groups[0]['lr']:.1E}")
            else:
                print(f"{TermColors.RED}Warn: NaN/Inf loss in KD student train. Skipping update.{TermColors.ENDC}")
        
        if stop_requested: break
        epoch_loss_kd = running_loss_kd / total_samples_kd if total_samples_kd > 0 else float('nan')
        print(f"KD Student Epoch {epoch+1} Train Loss: {epoch_loss_kd:.4f}")

        # Validation for the student model
        student_model.eval()
        if swa_student_model: swa_student_model.eval()
        if ema_student_model: ema_student_model.eval()

        val_models_to_eval_kd = {'base_student': student_model}
        if KD_STUDENT_USE_SWA and swa_student_model and epoch >= kd_student_swa_start_epoch : val_models_to_eval_kd['swa_student'] = swa_student_model
        if KD_STUDENT_USE_EMA and ema_student_model: val_models_to_eval_kd['ema_student'] = ema_student_model
        
        current_best_val_acc_this_epoch = 0.0
        best_model_key_this_epoch = 'base_student'

        for model_key, current_eval_model in val_models_to_eval_kd.items():
            current_eval_model.eval() # Ensure correct mode
            # Removed .apply_shadow() and .restore() for ema_student_model

            running_val_loss_kd = 0.0; total_val_samples_kd = 0
            all_preds_kd, all_labels_kd = [], []
            with torch.no_grad():
                for batch_data_val in tqdm(val_loader_kd, desc=f"KD Student Val E{epoch+1} ({model_key})", leave=False):
                    inputs_val, labels_val = batch_data_val
                    inputs_val, labels_val = inputs_val.to(DEVICE), labels_val.to(DEVICE)
                    batch_size_val = inputs_val.size(0)

                    with torch.amp.autocast('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda')):
                        outputs_val = current_eval_model(inputs_val)
                        # For validation, use simple CrossEntropy against true labels
                        val_loss = F.cross_entropy(outputs_val, labels_val) 
                    
                    if not torch.isnan(val_loss):
                        running_val_loss_kd += val_loss.item() * batch_size_val
                        total_val_samples_kd += batch_size_val
                        all_preds_kd.append(torch.argmax(outputs_val, dim=1).cpu())
                        all_labels_kd.append(labels_val.cpu())
            
            epoch_val_loss_kd = running_val_loss_kd / total_val_samples_kd if total_val_samples_kd > 0 else float('nan')
            epoch_val_acc_kd = 0.0
            if all_preds_kd and total_val_samples_kd > 0:
                preds_tensor = torch.cat(all_preds_kd); labels_tensor = torch.cat(all_labels_kd)
                epoch_val_acc_kd = (preds_tensor == labels_tensor).sum().item() / total_val_samples_kd
            
            print(f"  KD Student Val ({model_key}) - Epoch {epoch+1}: Loss: {epoch_val_loss_kd:.4f}, Acc: {epoch_val_acc_kd:.4f}")

            if epoch_val_acc_kd > current_best_val_acc_this_epoch:
                current_best_val_acc_this_epoch = epoch_val_acc_kd
                best_model_key_this_epoch = model_key
        
        # Save the best performing model variant (base, SWA, or EMA) from this epoch
        if current_best_val_acc_this_epoch > best_kd_val_acc:
            best_kd_val_acc = current_best_val_acc_this_epoch
            print(f"{TermColors.OKGREEN}New best KD student val acc: {best_kd_val_acc:.4f} (from {best_model_key_this_epoch}). Saving...{TermColors.ENDC}")
            
            model_to_save_state = None
            save_path_for_best = student_base_save_path # Default to base student path

            if best_model_key_this_epoch == 'base_student':
                model_to_save_state = student_model.state_dict()
            elif best_model_key_this_epoch == 'swa_student' and swa_student_model:
                model_to_save_state = swa_student_model.module.state_dict() # Save underlying model from SWA
                save_path_for_best = student_swa_save_path
            elif best_model_key_this_epoch == 'ema_student' and ema_student_model:
                # ema_student_model already has the EMA weights
                model_to_save_state = ema_student_model.module.state_dict() # Save underlying model from EMA
                save_path_for_best = student_ema_save_path
            
            if model_to_save_state:
                torch.save(model_to_save_state, save_path_for_best)
                print(f"  Saved best student model ({best_model_key_this_epoch}) to {save_path_for_best}")
            else:
                print(f"{TermColors.YELLOW}Warn: Could not determine model to save for best KD student.{TermColors.ENDC}")


        # SWA update for student model
        if KD_STUDENT_USE_SWA and swa_student_model and epoch >= kd_student_swa_start_epoch:
            swa_student_model.update_parameters(student_model)
            if swa_student_scheduler: swa_student_scheduler.step()
        elif scheduler_kd: # Regular scheduler step if not in SWA phase or SWA not used
            scheduler_kd.step()
        
        gc.collect(); torch.cuda.empty_cache()
    
    # Final SWA BN update and save if SWA was used for student
    if KD_STUDENT_USE_SWA and swa_student_model and not stop_requested:
        print(f"{TermColors.CYAN}KD Student: Updating FINAL SWA BN stats...{TermColors.ENDC}")
        try:
            # Create a new dataloader for BN update to ensure full pass over training data
            bn_loader_kd_final = DataLoader(train_ds_kd, KD_BATCH_SIZE * 2, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
            swa_student_model.to(DEVICE) # Ensure SWA model is on device
            torch.optim.swa_utils.update_bn(bn_loader_kd_final, swa_student_model, device=DEVICE)
            print(f"{TermColors.GREEN}  KD Student SWA BN updated.{TermColors.ENDC}")
            torch.save(swa_student_model.module.state_dict(), student_swa_save_path) # Save the SWA model's underlying parameters
            print(f"  Final SWA student model saved to: {student_swa_save_path}")
            del bn_loader_kd_final
        except Exception as e_swa_final_kd:
            print(f"{TermColors.RED}KD Student FINAL SWA BN Update/Save Error: {e_swa_final_kd}{TermColors.ENDC}"); traceback.print_exc()

    # Final EMA model save if EMA was used for student
    if KD_STUDENT_USE_EMA and ema_student_model and not stop_requested:
        try:
            ema_student_model.to(DEVICE) # Ensure EMA model is on device
            # ema_student_model already has the EMA weights, no apply_shadow needed
            torch.save(ema_student_model.module.state_dict(), student_ema_save_path) # Save the EMA model's underlying parameters
            # No restore needed
            print(f"  Final EMA student model saved to: {student_ema_save_path}")
        except Exception as e_ema_final_kd:
            print(f"{TermColors.RED}KD Student FINAL EMA Save Error: {e_ema_final_kd}{TermColors.ENDC}"); traceback.print_exc()

    # Save the base student model at the very end if not saved as best
    if not os.path.exists(student_base_save_path) and not stop_requested : # If base wasn't the best at any point
         torch.save(student_model.state_dict(), student_base_save_path)
         print(f"  Final base student model saved to: {student_base_save_path}")


    print(f"{TermColors.OKGREEN}Knowledge Distillation for student finished. Best student val_acc during KD: {best_kd_val_acc:.4f}{TermColors.ENDC}")
    
    # Cleanup
    del student_model, train_loader_kd, val_loader_kd, train_ds_kd, val_ds_kd
    del actual_stacking_model, stacking_scaler, feature_extractor_for_teacher, base_mlp_models_for_teacher
    if swa_student_model: del swa_student_model
    if ema_student_model: del ema_student_model
    gc.collect(); torch.cuda.empty_cache()


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
        print(f"{TermColors.CYAN}Loading feature extractor checkpoint from: {FEATURE_EXTRACTOR_CHECKPOINT_PATH}{TermColors.ENDC}")
        try:
            ckpt = torch.load(FEATURE_EXTRACTOR_CHECKPOINT_PATH, map_location=DEVICE)
            state_dict = ckpt['state_dict']
            new_state_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            feature_extractor_model.load_state_dict(new_state_dict, strict=False)
            print(f"{TermColors.GREEN}Feature extractor checkpoint loaded.{TermColors.ENDC}")
        except Exception as e:
            print(f"{TermColors.RED}Failed to load feature extractor checkpoint: {e}. Using pretrained weights.{TermColors.ENDC}")
            traceback.print_exc()
    else:
        print(f"{TermColors.INFO}Using {'pretrained' if PRETRAINED else 'randomly initialized'} weights for feature extractor.{TermColors.ENDC}")

    feature_extractor_model = feature_extractor_model.to(DEVICE)
    feature_extractor_model.eval()

    # Use minimal transforms for feature extraction
    extraction_transform = get_transforms(image_size=FEATURE_EXTRACTION_IMAGE_SIZE_CONFIG, for_feature_extraction=True)
    
    # Create a dataset for all images
    full_plant_dataset = PlantDataset(df_full, IMAGE_DIR, extraction_transform, label_encoder, include_paths=True, image_size=FEATURE_EXTRACTION_IMAGE_SIZE_CONFIG)
    if len(full_plant_dataset) == 0:
        print(f"{TermColors.RED}Dataset for feature extraction is empty. Aborting feature extraction.{TermColors.ENDC}")
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
        print(f"{TermColors.RED}No features were extracted. Aborting.{TermColors.ENDC}")
        return

    all_features_np = np.concatenate(all_features_list, axis=0)
    all_labels_np = np.concatenate(all_labels_list, axis=0)
    all_original_indices_np = np.concatenate(all_original_indices_list, axis=0)

    print(f"{TermColors.GREEN}Feature extraction complete. Extracted {all_features_np.shape[0]} features with dimension {all_features_np.shape[1]}.{TermColors.ENDC}")
    
    try:
        np.savez_compressed(FEATURES_NPZ_PATH, 
                            features=all_features_np, 
                            labels=all_labels_np, 
                            original_indices=all_original_indices_np,
                            label_encoder_classes=np.array(list(label_encoder.classes_))) # Save LE classes
        print(f"{TermColors.GREEN}Features saved to {FEATURES_NPZ_PATH}{TermColors.ENDC}")
    except Exception as e:
        print(f"{TermColors.RED}Error saving features: {e}{TermColors.ENDC}")
        traceback.print_exc()
    
    del feature_extractor_model, dataloader, full_plant_dataset, all_features_list, all_labels_list, all_original_indices_list
    gc.collect()
    torch.cuda.empty_cache()


# --- Main MLP Training Loop ---
def train_mlp_on_features_main_loop(num_classes_val, label_encoder_val, class_names_val, class_frequencies_val, class_priors_val, class_weights_val):
    global NUM_CLASSES, label_encoder, class_names, CLASS_FREQUENCIES, CLASS_PRIORS, CLASS_WEIGHTS
    # Global MLP HPO parameters (MLP_HIDDEN_DIMS, etc.) will be used by build_mlp_model and get_optimizer
    NUM_CLASSES = num_classes_val; label_encoder = label_encoder_val; class_names = class_names_val
    CLASS_FREQUENCIES = class_frequencies_val; CLASS_PRIORS = class_priors_val; CLASS_WEIGHTS = class_weights_val

    print(f"\n{TermColors.HEADER}--- STEP: Train MLP on Extracted Features ---{TermColors.ENDC}")
    final_mlp_stacking_model_path = None 

    if not os.path.exists(FEATURES_NPZ_PATH):
        print(f"{TermColors.RED}Features file not found: {FEATURES_NPZ_PATH}. Please run in 'EXTRACT_FEATURES' mode first or ensure features are generated.{TermColors.ENDC}")
        return final_mlp_stacking_model_path
    
    try:
        print(f"{TermColors.CYAN}Loading features from {FEATURES_NPZ_PATH}...{TermColors.ENDC}")
        data = np.load(FEATURES_NPZ_PATH, allow_pickle=True)
        features_all_loaded = data['features']; labels_all_loaded = data['labels']; original_indices_all_loaded = data['original_indices']
        loaded_le_classes = data['label_encoder_classes']
        if np.isnan(features_all_loaded).sum() > 0: print(f"{TermColors.RED}FATAL: NaNs found in features! Aborting.{TermColors.ENDC}"); return final_mlp_stacking_model_path
        if np.all(features_all_loaded == 0): print(f"{TermColors.RED}FATAL: All features are zero! Aborting.{TermColors.ENDC}"); return final_mlp_stacking_model_path
        if not np.array_equal(list(label_encoder.classes_), loaded_le_classes):
            print(f"{TermColors.RED}Label encoder mismatch! Aborting MLP training.{TermColors.ENDC}"); return final_mlp_stacking_model_path
        print(f"{TermColors.GREEN}Features loaded: {features_all_loaded.shape[0]} samples.{TermColors.ENDC}")
    except Exception as e: print(f"{TermColors.RED}Error loading features: {e}{TermColors.ENDC}"); traceback.print_exc(); return final_mlp_stacking_model_path

    features_for_kfold = features_all_loaded
    labels_for_kfold = labels_all_loaded
    original_indices_for_kfold = original_indices_all_loaded

    if MLP_DO_HPO and OPTUNA_AVAILABLE:
        min_samples_for_hpo_then_kfold = int(N_FOLDS / (1.0 - MLP_HPO_DATA_SPLIT_RATIO)) + N_FOLDS 
        if len(features_all_loaded) < min_samples_for_hpo_then_kfold or len(features_all_loaded) * MLP_HPO_DATA_SPLIT_RATIO < 20:
             print(f"{TermColors.YELLOW}Warning: Not enough data for a robust MLP HPO split and K-Fold (Total: {len(features_all_loaded)}). Skipping MLP HPO.{TermColors.ENDC}")
        else:
            try:
                features_for_kfold, features_hpo_main, labels_for_kfold, labels_hpo_main, original_indices_for_kfold, original_indices_hpo_main = train_test_split(
                    features_all_loaded, labels_all_loaded, original_indices_all_loaded,
                    test_size=MLP_HPO_DATA_SPLIT_RATIO, random_state=SEED + 99, stratify=labels_all_loaded
                )
                print(f"{TermColors.INFO}Data split for MLP: HPO set size: {len(features_hpo_main)}, K-Fold set size: {len(features_for_kfold)}{TermColors.ENDC}")
                run_mlp_hpo(features_hpo_main, labels_hpo_main, original_indices_hpo_main, features_all_loaded.shape[1], NUM_CLASSES)
                del features_hpo_main, labels_hpo_main, original_indices_hpo_main; gc.collect()
            except ValueError as e_hpo_split: 
                print(f"{TermColors.YELLOW}Warning: Stratified split for HPO data failed ({e_hpo_split}). Attempting non-stratified or skipping HPO.{TermColors.ENDC}")
                try:
                    features_for_kfold, features_hpo_main, labels_for_kfold, labels_hpo_main, original_indices_for_kfold, original_indices_hpo_main = train_test_split(
                        features_all_loaded, labels_all_loaded, original_indices_all_loaded,
                        test_size=MLP_HPO_DATA_SPLIT_RATIO, random_state=SEED + 99
                    )
                    print(f"{TermColors.INFO}Data split for MLP (non-stratified): HPO set size: {len(features_hpo_main)}, K-Fold set size: {len(features_for_kfold)}{TermColors.ENDC}")
                    run_mlp_hpo(features_hpo_main, labels_hpo_main, original_indices_hpo_main, features_all_loaded.shape[1], NUM_CLASSES)
                    del features_hpo_main, labels_hpo_main, original_indices_hpo_main; gc.collect()
                except Exception as e_hpo_nonstrat:
                    print(f"{TermColors.RED}Non-stratified HPO split also failed or HPO run error: {e_hpo_nonstrat}. Skipping MLP HPO.{TermColors.ENDC}"); traceback.print_exc()
                    features_for_kfold = features_all_loaded; labels_for_kfold = labels_all_loaded; original_indices_for_kfold = original_indices_all_loaded
            except Exception as e_hpo_run:
                print(f"{TermColors.RED}Error during MLP HPO run: {e_hpo_run}. Continuing with default/current MLP params.{TermColors.ENDC}"); traceback.print_exc()
                features_for_kfold = features_all_loaded; labels_for_kfold = labels_all_loaded; original_indices_for_kfold = original_indices_all_loaded
    elif MLP_DO_HPO and not OPTUNA_AVAILABLE:
        print(f"{TermColors.YELLOW}MLP_DO_HPO is True, but Optuna is not available. Skipping MLP HPO.{TermColors.ENDC}")

    if len(features_for_kfold) < N_FOLDS * 2 :
        print(f"{TermColors.RED}Not enough data remaining for K-Fold CV after HPO split (Samples: {len(features_for_kfold)}, Folds: {N_FOLDS}). Aborting MLP training.{TermColors.ENDC}")
        return final_mlp_stacking_model_path

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_results = defaultdict(list)
    oof_preds_array = np.full((len(original_indices_for_kfold), NUM_CLASSES), np.nan, dtype=np.float32) 
    oof_labels_array = np.full(len(original_indices_for_kfold), -1, dtype=np.int32) 
    mlp_dataloader_num_workers = 0

    # Adaptive learning parameters for MLP K-fold
    ADAPTIVE_MLP_LR_FACTOR = 0.5
    ADAPTIVE_MLP_WD_FACTOR = 1.5 # Increase WD if overfitting
    ADAPTIVE_MLP_MIN_LR = MLP_LEARNING_RATE * 0.01 # Min LR from HPO or default
    ADAPTIVE_MLP_MAX_WD = MLP_WEIGHT_DECAY * 10    # Max WD from HPO or default
    ADAPTIVE_MLP_OVERFIT_THRESHOLD_ACC_DIFF = 0.07 # e.g. if train_acc is 15% > val_acc
    ADAPTIVE_MLP_PATIENCE = 3 # Number of epochs to observe overfitting before acting

    for fold, (train_feature_idx_kfold, val_feature_idx_kfold) in enumerate(skf.split(features_for_kfold, labels_for_kfold)):
        if stop_requested: break
        fold_idx_display = fold + 1
        print(f"\n{TermColors.HEADER}===== Starting MLP Training Fold {fold_idx_display}/{N_FOLDS} ====={TermColors.ENDC}")
        
        train_features, val_features = features_for_kfold[train_feature_idx_kfold], features_for_kfold[val_feature_idx_kfold]
        train_labels, val_labels = labels_for_kfold[train_feature_idx_kfold], labels_for_kfold[val_feature_idx_kfold]
        val_original_ids_this_fold = original_indices_for_kfold[val_feature_idx_kfold]

        mlp_model = optimizer = scheduler = scaler = criterion = swa_mlp_model = ema_mlp_model = None
        gc.collect(); torch.cuda.empty_cache()

        try:
            mlp_model = build_mlp_model(input_size=features_for_kfold.shape[1], num_classes=NUM_CLASSES)
            mlp_model = mlp_model.to(DEVICE)
            if USE_TORCH_COMPILE and hasattr(torch, 'compile') and int(torch.__version__.split('.')[0]) >= 2:
                try: mlp_model = torch.compile(mlp_model, mode='default');
                except Exception as compile_e: print(f"{TermColors.RED}torch.compile() for MLP failed: {compile_e}.{TermColors.ENDC}")

            current_mlp_label_smoothing = 0.0 if MLP_USE_ARCFACE else LABEL_SMOOTHING
            criterion = get_criterion(class_weights=CLASS_WEIGHTS, label_smoothing=current_mlp_label_smoothing)
            # Use HPO-tuned LR and WD as initial values
            optimizer = get_optimizer(mlp_model, lr=MLP_LEARNING_RATE, weight_decay=MLP_WEIGHT_DECAY, optimizer_type=MLP_OPTIMIZER_TYPE, is_mlp_optimizer=True)
            scaler = torch.amp.GradScaler('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda'))
            
            if MLP_USE_SWA:
                swa_mlp_model = AveragedModel(mlp_model, avg_fn=ema_avg_fn if MLP_USE_EMA else None)
                if MLP_USE_EMA and hasattr(swa_mlp_model, 'module'): 
                    for p_swa in swa_mlp_model.module.parameters(): p_swa._is_mlp_ema_param = True
                elif MLP_USE_EMA:
                     for p_swa in swa_mlp_model.parameters(): p_swa._is_mlp_ema_param = True
            if MLP_USE_EMA:
                ema_mlp_model = AveragedModel(mlp_model, avg_fn=ema_avg_fn)
                if hasattr(ema_mlp_model, 'module'):
                    for p_ema in ema_mlp_model.module.parameters(): p_ema._is_mlp_ema_param = True
                else:
                    for p_ema in ema_mlp_model.parameters(): p_ema._is_mlp_ema_param = True
        except Exception as e: print(f"{TermColors.RED}MLP Fold {fold_idx_display} Setup Error: {e}{TermColors.ENDC}"); traceback.print_exc(); continue

        start_mlp_epoch, best_metric, _ = load_checkpoint(fold, mlp_model, optimizer, None, scaler, filename="latest_mlp_checkpoint.pth.tar", is_mlp_checkpoint=True)
        initial_best_val = float('-inf') if CHECKPOINT_MONITOR == 'max' else float('inf')
        if best_metric == initial_best_val: 
             start_mlp_epoch, best_metric, _ = load_checkpoint(fold, mlp_model, optimizer, None, scaler, filename="best_mlp_model.pth.tar", is_mlp_checkpoint=True)

        fold_log_dir = os.path.join(BASE_LOG_DIR, f"mlp_fold_{fold}"); writer = SummaryWriter(log_dir=fold_log_dir)
        fold_stop_requested = False 
        fold_best_val_loss = float('inf'); fold_best_val_acc = 0.0; fold_best_epoch = -1
        epochs_without_improvement = 0
        best_metric_for_early_stopping = best_metric 
        
        adaptive_overfit_counter = 0 # Counter for adaptive adjustments

        train_feature_ds = FeatureDataset(train_features, train_labels, original_indices_for_kfold[train_feature_idx_kfold])
        val_feature_ds = FeatureDataset(val_features, val_labels, val_original_ids_this_fold)

        sampler = None
        if IMBALANCE_STRATEGY == 'WeightedSampler' and CLASS_WEIGHTS is not None:
            labels_list = train_feature_ds.get_labels() 
            if labels_list:
                class_sample_count = np.array([labels_list.count(l) for l in range(NUM_CLASSES)])
                class_sample_count = np.maximum(class_sample_count, 1) 
                weight = 1. / class_sample_count
                samples_weight = torch.from_numpy(np.array([weight[t] for t in labels_list])).double()
                sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        
        train_loader = DataLoader(train_feature_ds, MLP_BATCH_SIZE, sampler=sampler, shuffle=(sampler is None), num_workers=mlp_dataloader_num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_feature_ds, MLP_BATCH_SIZE*2, shuffle=False, num_workers=mlp_dataloader_num_workers, pin_memory=True)
        err_loader = DataLoader(val_feature_ds, ERROR_LOG_BATCH_SIZE, shuffle=False, num_workers=mlp_dataloader_num_workers)  
        
        scheduler = get_scheduler(optimizer, scheduler_type=MLP_SCHEDULER_TYPE, total_epochs=MLP_EPOCHS, plateau_monitor=CHECKPOINT_MONITOR, plateau_mode=CHECKPOINT_MODE, is_mlp_scheduler=True)
        
        if start_mlp_epoch > 0: 
             ckpt_path_sched = os.path.join(BASE_CHECKPOINT_DIR, f"mlp_fold_{fold}", "latest_mlp_checkpoint.pth.tar")
             if scheduler and os.path.isfile(ckpt_path_sched):
                 ckpt_sched = torch.load(ckpt_path_sched, map_location=DEVICE)
                 if 'scheduler' in ckpt_sched and ckpt_sched['scheduler']:
                     try: scheduler.load_state_dict(ckpt_sched['scheduler'])
                     except Exception as e_sch: print(f"{TermColors.YELLOW}MLP Scheduler reload failed: {e_sch}.{TermColors.ENDC}")

        swa_scheduler_mlp = None
        if MLP_USE_SWA and swa_mlp_model:
            swa_scheduler_mlp = SWALR(optimizer, swa_lr=(MLP_LEARNING_RATE * SWA_LR_FACTOR), anneal_epochs=SWA_ANNEAL_EPOCHS, anneal_strategy='cos')
        
        mlp_swa_start_epoch = int(MLP_EPOCHS * MLP_SWA_START_EPOCH_FACTOR)
        last_epoch_completed_for_fold = start_mlp_epoch -1

        for epoch in range(start_mlp_epoch, MLP_EPOCHS):
            if fold_stop_requested or stop_requested: break
            print(f"\n{TermColors.CYAN}--- MLP Fold {fold_idx_display} Epoch {epoch+1}/{MLP_EPOCHS} ---{TermColors.ENDC}")
            
            train_loss, train_acc = train_one_epoch(
                mlp_model, train_loader, criterion, optimizer, scaler, scheduler, 
                epoch, stage_idx=0, stage_epoch=epoch, stage_total_epochs=MLP_EPOCHS, 
                device=DEVICE, writer=writer, num_classes=NUM_CLASSES, ema_model=ema_mlp_model,
                use_sam_flag=False, fold_num=fold_idx_display, is_mlp_training=True
            )
            if train_loss is None: fold_stop_requested = True; break 
            
            val_loss, val_acc, current_oof_preds, current_oof_original_ids = validate_one_epoch(
                mlp_model, val_loader, criterion, DEVICE, epoch, writer, NUM_CLASSES, 
                scheduler=scheduler, swa_model=swa_mlp_model, ema_model=ema_mlp_model, 
                return_preds=True, fold_num=fold_idx_display, is_mlp_validation=True
            )
            if val_loss is None: fold_stop_requested = True; break 

            print(f"MLP F{fold_idx_display} Ep {epoch+1}: Tr L={train_loss:.4f} A={train_acc:.4f} | Val L={val_loss:.4f} A={val_acc:.4f} (Best {CHECKPOINT_MONITOR}: {best_metric:.4f})")

            # --- Adaptive LR / WD for MLP K-fold ---
            is_overfitting_signal = (train_acc > (val_acc + ADAPTIVE_MLP_OVERFIT_THRESHOLD_ACC_DIFF)) and \
                                    (train_acc > 0.60) 
            
            if is_overfitting_signal:
                adaptive_overfit_counter += 1
            elif val_acc > fold_best_val_acc: 
                adaptive_overfit_counter = 0
            
            if adaptive_overfit_counter >= ADAPTIVE_MLP_PATIENCE:
                print(f"{TermColors.YELLOW}MLP Fold {fold_idx_display} Adaptive: Overfitting detected for {adaptive_overfit_counter} epochs.{TermColors.ENDC}")
                current_lr = optimizer.param_groups[0]['lr']
                current_wd = optimizer.param_groups[0]['weight_decay']
                
                new_lr = max(current_lr * ADAPTIVE_MLP_LR_FACTOR, ADAPTIVE_MLP_MIN_LR)
                new_wd = min(current_wd * ADAPTIVE_MLP_WD_FACTOR, ADAPTIVE_MLP_MAX_WD)

                if new_lr < current_lr:
                    optimizer.param_groups[0]['lr'] = new_lr
                    print(f"{TermColors.YELLOW}  Adaptive: Reduced LR to {new_lr:.2e}{TermColors.ENDC}")
                if new_wd > current_wd:
                    optimizer.param_groups[0]['weight_decay'] = new_wd
                    print(f"{TermColors.YELLOW}  Adaptive: Increased WD to {new_wd:.2e}{TermColors.ENDC}")
                
                adaptive_overfit_counter = 0 # Reset counter after adjustment
            # --- End Adaptive LR / WD ---

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
                print(f"{TermColors.WARNING}Early stopping for MLP Fold {fold_idx_display} after {epochs_without_improvement} epochs on {CHECKPOINT_MONITOR}.{TermColors.ENDC}")
                fold_stop_requested = True 
            
            current_metric_for_checkpoint = val_acc if CHECKPOINT_MONITOR == 'val_acc' else val_loss
            is_best = False
            if (CHECKPOINT_MODE == 'max' and current_metric_for_checkpoint > best_metric) or \
               (CHECKPOINT_MODE == 'min' and current_metric_for_checkpoint < best_metric):
                best_metric = current_metric_for_checkpoint; is_best = True
                fold_best_val_loss = val_loss; fold_best_val_acc = val_acc; fold_best_epoch = epoch
                print(f"{TermColors.OKGREEN}MLP F{fold_idx_display} New Best {CHECKPOINT_MONITOR}: {best_metric:.4f}. Saving...{TermColors.ENDC}")
                save_checkpoint(fold, epoch + 1, -1, epoch + 1, mlp_model, optimizer, scheduler, scaler, best_metric, "best_mlp_model.pth.tar", is_mlp_checkpoint=True)
                save_model(fold, mlp_model, "best_mlp_model_state_dict.pth", is_mlp_model=True)
                if MLP_USE_EMA and ema_mlp_model: save_model(fold, ema_mlp_model, "best_ema_mlp_model_state_dict.pth", is_mlp_model=True)
                if MLP_USE_SWA and swa_mlp_model and epoch >= mlp_swa_start_epoch: save_model(fold, swa_mlp_model, "best_swa_mlp_model_state_dict.pth", is_mlp_model=True)

                if current_oof_preds is not None and current_oof_original_ids is not None:
                    for i, original_id_from_val_batch in enumerate(current_oof_original_ids):
                        relative_idx_matches = np.where(original_indices_for_kfold == original_id_from_val_batch)[0]
                        if len(relative_idx_matches) > 0:
                            actual_oof_idx = relative_idx_matches[0]
                            if 0 <= actual_oof_idx < len(oof_preds_array):
                                oof_preds_array[actual_oof_idx] = current_oof_preds[i]
                                oof_labels_array[actual_oof_idx] = labels_for_kfold[actual_oof_idx]
                            else:
                                print(f"{TermColors.YELLOW}Warn: OOF relative index {actual_oof_idx} out of bounds for oof_preds_array (len {len(oof_preds_array)}). ID: {original_id_from_val_batch}{TermColors.ENDC}")
                        else:
                             print(f"{TermColors.YELLOW}Warn: Could not map original ID {original_id_from_val_batch} from validation batch to original_indices_for_kfold for OOF storage.{TermColors.ENDC}")
            
            save_checkpoint(fold, epoch + 1, -1, epoch + 1, mlp_model, optimizer, scheduler, scaler, best_metric, "latest_mlp_checkpoint.pth.tar", is_mlp_checkpoint=True)
            if LOG_MISCLASSIFIED_IMAGES and ((epoch + 1) % 5 == 0 or is_best): 
                log_misclassified(fold, mlp_model, err_loader, criterion, DEVICE, epoch + 1, writer, NUM_CLASSES, is_mlp_logging=True)
            
            last_epoch_completed_for_fold = epoch 
            gc.collect(); torch.cuda.empty_cache()
        
        print(f"MLP Fold {fold_idx_display} finished training. Best Base Val Acc: {fold_best_val_acc:.4f} at Epoch {fold_best_epoch+1 if fold_best_epoch != -1 else 'N/A'}")

        if fold_stop_requested and not stop_requested: 
            print(f"{TermColors.WARNING}MLP Fold {fold_idx_display} epoch loop stopped early at epoch {last_epoch_completed_for_fold + 1}.{TermColors.ENDC}")
            save_checkpoint(fold, last_epoch_completed_for_fold + 1, -1, last_epoch_completed_for_fold + 1, mlp_model, optimizer, scheduler, scaler, best_metric, "interrupted_mlp_checkpoint.pth.tar", is_mlp_checkpoint=True)
        
        if stop_requested: 
            print(f"{TermColors.WARNING}Global stop requested. MLP K-Fold training will terminate after Fold {fold_idx_display}, epoch {last_epoch_completed_for_fold + 1}. Saving final interrupted state.{TermColors.ENDC}")
            save_checkpoint(fold, last_epoch_completed_for_fold + 1, -1, last_epoch_completed_for_fold + 1, mlp_model, optimizer, scheduler, scaler, best_metric, "interrupted_mlp_checkpoint.pth.tar", is_mlp_checkpoint=True)
        else: 
            if MLP_USE_SWA and swa_mlp_model and last_epoch_completed_for_fold >= mlp_swa_start_epoch:
                print(f"{TermColors.CYAN}MLP Fold {fold_idx_display} Updating FINAL SWA BN stats...{TermColors.ENDC}")
                bn_loader_mlp_final = DataLoader(train_feature_ds, MLP_BATCH_SIZE*2, shuffle=True, num_workers=mlp_dataloader_num_workers, pin_memory=True)
                try:
                    swa_mlp_model.to(DEVICE) 
                    torch.optim.swa_utils.update_bn(bn_loader_mlp_final, swa_mlp_model, device=DEVICE)
                    print(f"{TermColors.CYAN}MLP Fold {fold_idx_display} Evaluating FINAL SWA model...{TermColors.ENDC}")
                    swa_final_val_loss, swa_final_val_acc, _, _ = validate_one_epoch(
                        model=swa_mlp_model, dataloader=val_loader, criterion=criterion, device=DEVICE,
                        global_epoch=last_epoch_completed_for_fold + 1, writer=writer, num_classes=NUM_CLASSES,
                        scheduler=None, swa_model=None, ema_model=None, return_preds=False, 
                        fold_num=f"{fold_idx_display}-SWA-Final", is_mlp_validation=True
                    )
                    print(f"{TermColors.SUCCESS}--- MLP Fold {fold_idx_display} FINAL SWA --- Val L={swa_final_val_loss:.4f}, Val Acc={swa_final_val_acc:.4f}{TermColors.ENDC}")
                    fold_results['mlp_swa_acc'].append(swa_final_val_acc); fold_results['mlp_swa_loss'].append(swa_final_val_loss)
                    if writer:
                        writer.add_scalar(f'MLP_Fold_{fold_idx_display}/Loss/val_swa_final', swa_final_val_loss, last_epoch_completed_for_fold + 1)
                        writer.add_scalar(f'MLP_Fold_{fold_idx_display}/Accuracy/val_swa_final', swa_final_val_acc, last_epoch_completed_for_fold + 1)
                    save_model(fold, swa_mlp_model, "final_swa_mlp_model_state_dict.pth", is_mlp_model=True)
                except Exception as e_swa_mlp_final: print(f"{TermColors.RED}MLP FINAL SWA BN/Eval Error: {e_swa_mlp_final}{TermColors.ENDC}"); traceback.print_exc()
                del bn_loader_mlp_final

            if MLP_USE_EMA and ema_mlp_model:
                try:
                    print(f"{TermColors.CYAN}MLP Fold {fold_idx_display} Evaluating FINAL EMA model...{TermColors.ENDC}")
                    ema_mlp_model.to(DEVICE) 
                    ema_final_val_loss, ema_final_val_acc, _, _ = validate_one_epoch(
                        model=ema_mlp_model, dataloader=val_loader, criterion=criterion, device=DEVICE,
                        global_epoch=last_epoch_completed_for_fold + 1, writer=writer, num_classes=NUM_CLASSES,
                        scheduler=None, swa_model=None, ema_model=None, return_preds=False, 
                        fold_num=f"{fold_idx_display}-EMA-Final", is_mlp_validation=True
                    )
                    print(f"{TermColors.SUCCESS}--- MLP Fold {fold_idx_display} FINAL EMA --- Val L={ema_final_val_loss:.4f}, Val Acc={ema_final_val_acc:.4f}{TermColors.ENDC}")
                    fold_results['mlp_ema_acc'].append(ema_final_val_acc); fold_results['mlp_ema_loss'].append(ema_final_val_loss)
                    if writer:
                        writer.add_scalar(f'MLP_Fold_{fold_idx_display}/Loss/val_ema_final', ema_final_val_loss, last_epoch_completed_for_fold + 1)
                        writer.add_scalar(f'MLP_Fold_{fold_idx_display}/Accuracy/val_ema_final', ema_final_val_acc, last_epoch_completed_for_fold + 1)
                    save_model(fold, ema_mlp_model, "final_ema_mlp_model_state_dict.pth", is_mlp_model=True)
                except Exception as e_ema_mlp_final: print(f"{TermColors.RED}MLP FINAL EMA Eval Error: {e_ema_mlp_final}{TermColors.ENDC}"); traceback.print_exc()

        if mlp_model: 
            save_model(fold, mlp_model, "final_mlp_model_state_dict.pth", is_mlp_model=True)
        
        if writer: writer.close()
        fold_results['mlp_best_metric'].append(best_metric) 
        fold_results['mlp_best_val_loss'].append(fold_best_val_loss) 
        fold_results['mlp_best_val_acc'].append(fold_best_val_acc)   
        
        del mlp_model, optimizer, scheduler, scaler, criterion, swa_mlp_model, ema_mlp_model, train_loader, val_loader, err_loader, train_feature_ds, val_feature_ds
        gc.collect(); torch.cuda.empty_cache()
        if stop_requested: break 

    print(f"\n{TermColors.HEADER}===== MLP Cross-Validation Finished ====={TermColors.ENDC}")
    if not stop_requested:
        avg_best_acc = np.mean(fold_results['mlp_best_val_acc']) if fold_results['mlp_best_val_acc'] else 0.0
        std_best_acc = np.std(fold_results['mlp_best_val_acc']) if fold_results['mlp_best_val_acc'] else 0.0
        print(f"Avg MLP Best Base Val Acc: {avg_best_acc:.4f} +/- {std_best_acc:.4f}")
        
        if fold_results['mlp_swa_acc']: 
            avg_swa_acc = np.mean(fold_results['mlp_swa_acc'])
            std_swa_acc = np.std(fold_results['mlp_swa_acc'])
            print(f"Avg MLP Final SWA Val Acc: {avg_swa_acc:.4f} +/- {std_swa_acc:.4f}")
        
        if fold_results['mlp_ema_acc']:
            avg_ema_acc = np.mean(fold_results['mlp_ema_acc'])
            std_ema_acc = np.std(fold_results['mlp_ema_acc'])
            print(f"Avg MLP Final EMA Val Acc: {avg_ema_acc:.4f} +/- {std_ema_acc:.4f}")

        if RUN_STACKING:
            valid_oof_relative_indices = np.where(oof_labels_array != -1)[0] 
            
            if len(valid_oof_relative_indices) < len(original_indices_for_kfold) * 0.5: 
                print(f"{TermColors.YELLOW}Warning: MLP OOF collected for only {len(valid_oof_relative_indices)}/{len(original_indices_for_kfold)} samples. Stacking may be suboptimal.{TermColors.ENDC}")
            
            if len(valid_oof_relative_indices) > 0:
                final_oof_preds = oof_preds_array[valid_oof_relative_indices]
                final_oof_labels = oof_labels_array[valid_oof_relative_indices]
                final_oof_original_ids = original_indices_for_kfold[valid_oof_relative_indices]
                
                if len(final_oof_preds) > 0 and len(final_oof_preds) == len(final_oof_labels):
                    mlp_oof_path = STACKING_OOF_PREDS_PATH.replace(".npz", "_mlp.npz")
                    np.savez_compressed(mlp_oof_path, preds=final_oof_preds, labels=final_oof_labels, original_indices=final_oof_original_ids)
                    print(f"MLP OOF preds saved: {mlp_oof_path}")
                    
                    final_mlp_stacking_model_path = STACKING_META_MODEL_PATH.replace(".joblib", "_mlp.joblib")
                    train_stacking_meta_model(final_oof_preds, final_oof_labels, final_mlp_stacking_model_path)
                else:
                    print(f"{TermColors.RED}Error preparing MLP stacking data (lengths mismatch or empty). Skipping.{TermColors.ENDC}")
                del final_oof_preds, final_oof_labels, final_oof_original_ids
            else:
                print(f"{TermColors.RED}No valid MLP OOF predictions for Stacking. Skipping.{TermColors.ENDC}")
    else:
        print(f"{TermColors.YELLOW}MLP Training interrupted. Stacking/KD skipped.{TermColors.ENDC}")    

    del features_all_loaded, labels_all_loaded, original_indices_all_loaded 
    del features_for_kfold, labels_for_kfold, original_indices_for_kfold 
    del oof_preds_array, oof_labels_array
    gc.collect()
    return final_mlp_stacking_model_path

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
                try: model = torch.compile(model, mode='default'); print(f"{TermColors.GREEN}torch.compile() applied.{TermColors.ENDC}")
                except Exception as compile_e: print(f"{TermColors.RED}torch.compile() failed: {compile_e}.{TermColors.ENDC}")
            criterion = get_criterion(class_weights=CLASS_WEIGHTS)
            optimizer = get_optimizer(model, optimizer_type=OPTIMIZER_TYPE, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, use_sam_flag=USE_SAM, sam_rho=SAM_RHO, sam_adaptive=SAM_ADAPTIVE)
            scaler = torch.amp.GradScaler('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda'))
            if USE_SWA: 
                swa_model = AveragedModel(model, avg_fn=ema_avg_fn if USE_EMA else None)
                if USE_EMA and hasattr(swa_model, '_is_mlp_ema'): swa_model._is_mlp_ema = False # Ensure it uses CombinedModel EMA decay
            if USE_EMA: 
                ema_model = AveragedModel(model, avg_fn=ema_avg_fn)
                if hasattr(ema_model, '_is_mlp_ema'): ema_model._is_mlp_ema = False # Ensure it uses CombinedModel EMA decay

        except Exception as e: print(f"{TermColors.RED}Fold {fold+1} Setup Error: {e}{TermColors.ENDC}"); traceback.print_exc(); continue
        
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
            if loaded_size and stage_idx_loop == start_stg_idx and loaded_size != CURRENT_IMAGE_SIZE: print(f"{TermColors.CRITICAL}Image size mismatch! Ckpt {loaded_size} != Stage {CURRENT_IMAGE_SIZE}. Exit.{TermColors.ENDC}"); fold_stop_requested = True; break
            
            train_transform, val_transform = get_transforms(image_size=CURRENT_IMAGE_SIZE)
            train_loader, val_loader, err_loader = None, None, None; gc.collect()
            try:
                train_ds = PlantDataset(train_df, IMAGE_DIR, train_transform, None, False, CURRENT_IMAGE_SIZE)
                val_ds = PlantDataset(val_df, IMAGE_DIR, val_transform, None, True, CURRENT_IMAGE_SIZE) # include_paths=True for OOF indices
                err_ds = PlantDataset(val_df, IMAGE_DIR, val_transform, None, True, CURRENT_IMAGE_SIZE) # include_paths=True for error logging
                if not train_ds or not val_ds or len(train_ds) == 0 or len(val_ds) == 0 : print(f"{TermColors.RED}Fold {fold+1} Stage {stage_idx_loop+1}: Dataset empty. Skip fold.{TermColors.ENDC}"); fold_stop_requested = True; break
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
                print(f"{TermColors.GREEN}Fold {fold+1} Dataloaders ready (Train: {len(train_loader)}, Val: {len(val_loader)}).{TermColors.ENDC}")
            except Exception as e: print(f"{TermColors.RED}Fold {fold+1} Dataloader Error: {e}{TermColors.ENDC}"); traceback.print_exc(); fold_stop_requested = True; break
            
            current_t_0_val = stage_epochs if SCHEDULER_TYPE == 'CosineWarmRestarts' else T_0
            scheduler = get_scheduler(optimizer, total_epochs=stage_epochs, t_0=current_t_0_val, plateau_monitor=CHECKPOINT_MONITOR, plateau_mode=CHECKPOINT_MODE)
            
            if stage_idx_loop == start_stg_idx and current_stage_start_epoch > 0:
                 ckpt_path_sched = os.path.join(BASE_CHECKPOINT_DIR, f"fold_{fold}", "latest_checkpoint.pth.tar")
                 if scheduler and os.path.isfile(ckpt_path_sched):
                     ckpt_sched = torch.load(ckpt_path_sched, map_location=DEVICE)
                     if 'scheduler' in ckpt_sched and ckpt_sched['scheduler']:
                         try: scheduler.load_state_dict(ckpt_sched['scheduler'])
                         except Exception as e_sch: print(f"{TermColors.YELLOW}Scheduler reload failed: {e_sch}.{TermColors.ENDC}")
            
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
                    current_auto_config.update(train_loss, train_acc, val_loss, val_acc, optimizer, is_sam_active_runtime, current_epoch=global_epoch_counter)
                    if (global_epoch_counter + 1) % 5 == 0 or global_epoch_counter == TOTAL_EPOCHS_PER_FOLD -1: 
                        print(f"\n{TermColors.CYAN}AutoConfig Status (GlobEp {global_epoch_counter+1}):\n{current_auto_config.get_status_report()}{TermColors.ENDC}\n")

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
                    print(f"{TermColors.WARNING}Early stopping for Fold {fold+1} after {epochs_without_improvement} epochs on {CHECKPOINT_MONITOR}.{TermColors.ENDC}")
                    fold_stop_requested = True; break 

                current_metric_for_checkpoint = val_acc if CHECKPOINT_MONITOR == 'val_acc' else val_loss
                is_best = False
                if (CHECKPOINT_MODE == 'max' and current_metric_for_checkpoint > best_metric) or \
                   (CHECKPOINT_MODE == 'min' and current_metric_for_checkpoint < best_metric):
                    best_metric = current_metric_for_checkpoint; is_best = True; fold_best_val_loss = val_loss; fold_best_val_acc = val_acc
                    print(f"{TermColors.OKGREEN}Fold {fold+1} New Best {CHECKPOINT_MONITOR}: {best_metric:.4f}. Saving...{TermColors.ENDC}")
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
            print(f"{TermColors.CYAN}Fold {fold+1} Updating SWA BN stats...{TermColors.ENDC}")
            final_stage_size_swa = PROGRESSIVE_RESIZING_STAGES[-1][1]; final_train_tf_swa, _ = get_transforms(image_size=final_stage_size_swa)
            try:
                final_train_ds_bn_swa = PlantDataset(train_df, IMAGE_DIR, final_train_tf_swa, None, False, final_stage_size_swa)
                if len(final_train_ds_bn_swa) > 0:
                    bn_loader_swa = DataLoader(final_train_ds_bn_swa, BATCH_SIZE*2, shuffle=True, num_workers=NUM_WORKERS)
                    torch.optim.swa_utils.update_bn(bn_loader_swa, swa_model, device=DEVICE); print(f"{TermColors.GREEN}SWA BN updated.{TermColors.ENDC}")
                    _, final_val_tf_swa = get_transforms(image_size=final_stage_size_swa)
                    final_val_ds_eval_swa = PlantDataset(val_df, IMAGE_DIR, final_val_tf_swa, None, False, final_stage_size_swa) # No paths needed for SWA eval
                    if len(final_val_ds_eval_swa) > 0:
                        final_val_loader_swa = DataLoader(final_val_ds_eval_swa, BATCH_SIZE*2, shuffle=False, num_workers=NUM_WORKERS)
                        swa_val_loss, swa_val_acc, _, _ = validate_one_epoch(swa_model, final_val_loader_swa, criterion, DEVICE, global_epoch_counter, writer, NUM_CLASSES, fold_num=f"{fold+1}-SWA")
                        print(f"Fold {fold+1} Final SWA Val L={swa_val_loss:.4f}, A={swa_val_acc:.4f}"); fold_results['swa_acc'].append(swa_val_acc); fold_results['swa_loss'].append(swa_val_loss)
                        save_model(fold, swa_model, "final_swa_model_state_dict.pth")
                    del final_val_ds_eval_swa, final_val_loader_swa
                del final_train_ds_bn_swa, bn_loader_swa
            except Exception as e_swa: print(f"{TermColors.RED}SWA BN/Eval Error: {e_swa}{TermColors.ENDC}"); traceback.print_exc()
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
    
    df_full = None
    try:
        print(f"\n{TermColors.HEADER}--- STEP 1: Load Full Dataset Info ---{TermColors.ENDC}")
        df_full = pd.read_csv(CSV_PATH, sep=',', low_memory=False, on_bad_lines='skip')
        required_source_cols = ['id', 'scientific_name']
        if 'scientificName' in df_full.columns and 'scientific_name' not in df_full.columns: df_full.rename(columns={'scientificName': 'scientific_name'}, inplace=True)
        if not all(col in df_full.columns for col in required_source_cols): print(f"{TermColors.RED}Missing source columns: {[c for c in required_source_cols if c not in df_full.columns]}. Found: {df_full.columns.tolist()}{TermColors.ENDC}"); sys.exit(1)
        df_full = df_full[required_source_cols].dropna().astype({'id': str}); df_full.rename(columns={'scientific_name': 'scientificName'}, inplace=True)
        min_samples = 1; class_counts = df_full['scientificName'].value_counts(); valid_classes = class_counts[class_counts >= min_samples].index
        df_full = df_full[df_full['scientificName'].isin(valid_classes)].reset_index(drop=True)
        if len(df_full) == 0: print(f"{TermColors.RED}Dataframe empty after min_samples filter. Exiting.{TermColors.ENDC}"); sys.exit(1)
        
        label_encoder = LabelEncoder(); df_full['label'] = label_encoder.fit_transform(df_full['scientificName'])
        class_names = list(label_encoder.classes_); NUM_CLASSES = len(class_names)
        if NUM_CLASSES == 0: print(f"{TermColors.RED}Zero classes after encoding. Exiting.{TermColors.ENDC}"); sys.exit(1)
        
        mapping_path = os.path.join(BASE_MODEL_SAVE_DIR, "label_mapping.json"); os.makedirs(BASE_MODEL_SAVE_DIR, exist_ok=True)
        with open(mapping_path, 'w') as f: json.dump(dict(zip(range(NUM_CLASSES), class_names)), f, indent=4)
        
        if DEBUG_MODE: 
            min_debug_samples_per_class = 2 
            stratify_col = df_full['label']
            class_counts_debug = df_full['label'].value_counts()
            valid_labels_for_stratify = class_counts_debug[class_counts_debug >= min_debug_samples_per_class].index
            df_debug_stratify_possible = df_full[df_full['label'].isin(valid_labels_for_stratify)]
            if len(df_debug_stratify_possible) >= min(100, len(df_full)) and len(valid_labels_for_stratify) > 1 :
                 _, df_full = train_test_split(df_debug_stratify_possible, test_size=min(100, len(df_debug_stratify_possible)), random_state=SEED, stratify=df_debug_stratify_possible['label'])
            else: df_full = df_full.sample(n=min(100, len(df_full)), random_state=SEED)
            df_full = df_full.reset_index(drop=True)
        
        label_counts = df_full['label'].value_counts().sort_index(); total_samples_final = len(df_full)
        if total_samples_final == 0: print(f"{TermColors.RED}Dataframe empty before imbalance stats. Exiting.{TermColors.ENDC}"); sys.exit(1)
        freqs = torch.zeros(NUM_CLASSES, dtype=torch.float32); label_counts_reindexed = label_counts.reindex(range(NUM_CLASSES), fill_value=0)
        for i in range(NUM_CLASSES): freqs[i] = label_counts_reindexed.get(i, 0)
        CLASS_FREQUENCIES = freqs.to(DEVICE); CLASS_PRIORS = (CLASS_FREQUENCIES / total_samples_final if total_samples_final > 0 else torch.zeros_like(CLASS_FREQUENCIES)).to(DEVICE)
        try:
            class_weights_array = sk_class_weight.compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=df_full['label'])
            CLASS_WEIGHTS = torch.tensor(class_weights_array, dtype=torch.float32)
        except ValueError as e:
            print(f"{TermColors.RED}Error calculating class weights: {e}.{TermColors.ENDC}")
            if IMBALANCE_STRATEGY in ['WeightedLoss', 'WeightedSampler']: print(f"{TermColors.RED}Exiting due to class weight error with strategy '{IMBALANCE_STRATEGY}'.{TermColors.ENDC}"); sys.exit(1)
            CLASS_WEIGHTS = None
        print(f"{TermColors.GREEN}Data loaded: {len(df_full)} samples, {NUM_CLASSES} classes. Imbalance strategy: {IMBALANCE_STRATEGY}.{TermColors.ENDC}")
    except Exception as e: print(f"{TermColors.RED}Data Load/Prep Error: {e}{TermColors.ENDC}"); traceback.print_exc(); sys.exit(1)

    mlp_stacking_model_path_result = None # Path to the .joblib file for the stacker trained on MLP OOFs
    training_completed_successfully = False # Flag to indicate if MLP training/stacking was successful

    if RUN_MODE == "TRAIN_MLP_ON_FEATURES":
        mlp_oof_predictions_path = STACKING_OOF_PREDS_PATH.replace(".npz", "_mlp.npz")
        potential_mlp_stacking_model_path = STACKING_META_MODEL_PATH.replace(".joblib", "_mlp.joblib")

        # Check if the final stacking model from MLP OOFs already exists
        if os.path.exists(potential_mlp_stacking_model_path) and RUN_STACKING:
            print(f"{TermColors.INFO}MLP stacking model already exists: {potential_mlp_stacking_model_path}. Skipping MLP K-fold training and direct stacking.{TermColors.ENDC}")
            mlp_stacking_model_path_result = potential_mlp_stacking_model_path
            training_completed_successfully = True # Assume previous run was successful
        
        # Else, check if OOF predictions from MLP folds exist, to run only stacking
        elif os.path.exists(mlp_oof_predictions_path) and RUN_STACKING:
            print(f"{TermColors.INFO}MLP OOF predictions found: {mlp_oof_predictions_path}. Attempting to run stacking only.{TermColors.ENDC}")
            try:
                oof_data = np.load(mlp_oof_predictions_path)
                oof_preds = oof_data['preds']
                oof_labels = oof_data['labels']
                
                if len(oof_preds) > 0 and len(oof_preds) == len(oof_labels):
                    train_stacking_meta_model(oof_preds, oof_labels, potential_mlp_stacking_model_path)
                    if os.path.exists(potential_mlp_stacking_model_path):
                        mlp_stacking_model_path_result = potential_mlp_stacking_model_path
                        training_completed_successfully = True
                        print(f"{TermColors.GREEN}Stacking completed using existing MLP OOF predictions.{TermColors.ENDC}")
                    else:
                        print(f"{TermColors.RED}Stacking failed using existing MLP OOF predictions. Will proceed to full MLP training if no other errors.{TermColors.ENDC}")
                        training_completed_successfully = False # Stacking failed, so not successful yet
                else:
                    print(f"{TermColors.YELLOW}MLP OOF predictions file ({mlp_oof_predictions_path}) is invalid or empty. Will proceed to full MLP training.{TermColors.ENDC}")
                    training_completed_successfully = False
            except Exception as e:
                print(f"{TermColors.RED}Error loading or processing existing MLP OOF predictions: {e}. Will proceed to full MLP training.{TermColors.ENDC}")
                traceback.print_exc()
                training_completed_successfully = False
        
        # If neither stacking model nor OOF preds exist (or processing them failed to produce a stacker), run the full MLP training loop
        if not training_completed_successfully:
            print(f"{TermColors.INFO}Starting full MLP K-fold training and stacking process.{TermColors.ENDC}")
            # Ensure features are extracted if not present (train_mlp_on_features_main_loop also checks this)
            if not os.path.exists(FEATURES_NPZ_PATH):
                print(f"{TermColors.YELLOW}Base features file not found: {FEATURES_NPZ_PATH}. Attempting to extract features first.{TermColors.ENDC}")
                extract_all_features(df_full, NUM_CLASSES, label_encoder)
                if not os.path.exists(FEATURES_NPZ_PATH):
                    print(f"{TermColors.RED}Feature extraction failed or did not produce the file: {FEATURES_NPZ_PATH}. Aborting MLP training.{TermColors.ENDC}")
                    sys.exit(1)
            
            mlp_stacking_model_path_result = train_mlp_on_features_main_loop(
                NUM_CLASSES, label_encoder, class_names, CLASS_FREQUENCIES, CLASS_PRIORS, CLASS_WEIGHTS
            )
            # train_mlp_on_features_main_loop returns the path to the stacking model if successful
            training_completed_successfully = not stop_requested and mlp_stacking_model_path_result is not None and os.path.exists(mlp_stacking_model_path_result)
    
    elif RUN_MODE == "FULL_TRAINING":
        print(f"{TermColors.YELLOW}Full training mode's checkpointing for stacking is analogous to MLP mode.{TermColors.ENDC}")
        # This mode would need similar logic to check for its own OOFs and stacker model if it's supposed to resume.
        # For now, it runs from scratch or its own checkpoint logic.
        training_completed_successfully = main_full_training_loop(df_full, NUM_CLASSES, label_encoder, class_names, CLASS_FREQUENCIES, CLASS_PRIORS, CLASS_WEIGHTS)
        if training_completed_successfully and RUN_STACKING and os.path.exists(STACKING_META_MODEL_PATH):
             mlp_stacking_model_path_result = STACKING_META_MODEL_PATH # Assuming full training saves stacker here
        else:
             mlp_stacking_model_path_result = None # Stacking didn't run or failed
             if RUN_STACKING: training_completed_successfully = False


    else:
        print(f"{TermColors.RED}Invalid RUN_MODE: '{RUN_MODE}'.{TermColors.ENDC}"); sys.exit(1)

    # --- Knowledge Distillation using the Stacking Model as Teacher ---
    if RUN_KNOWLEDGE_DISTILLATION and training_completed_successfully and mlp_stacking_model_path_result and os.path.exists(mlp_stacking_model_path_result):
        print(f"\n{TermColors.HEADER}--- FINAL STEP: Distill Stacking Ensemble into Student Model ---{TermColors.ENDC}")
        
        if not os.path.exists(FEATURES_NPZ_PATH):
            print(f"{TermColors.RED}Base features file {FEATURES_NPZ_PATH} not found. Cannot get MLP input size for KD. Skipping KD of stacker.{TermColors.ENDC}")
        else:
            features_data_for_kd = np.load(FEATURES_NPZ_PATH, allow_pickle=True)
            base_mlp_input_size_for_kd = features_data_for_kd['features'].shape[1]
            del features_data_for_kd 
            
            # Determine which MLP models to use based on RUN_MODE
            # If TRAIN_MLP_ON_FEATURES, use mlp_fold_X models.
            # If FULL_TRAINING, this part would need to be adapted if base models for stacking are different.
            # For now, assuming TRAIN_MLP_ON_FEATURES path for KD teacher components.
            
            base_model_dir_prefix = "mlp_" if RUN_MODE == "TRAIN_MLP_ON_FEATURES" else "" # Adjust if FULL_TRAINING has different prefix
            base_model_filename = "best_mlp_model_state_dict.pth" if RUN_MODE == "TRAIN_MLP_ON_FEATURES" else "best_model_state_dict.pth" # Adjust for FULL_TRAINING

            mlp_fold_model_paths_for_kd = [
                os.path.join(BASE_MODEL_SAVE_DIR, f"{base_model_dir_prefix}fold_{f}", base_model_filename) 
                for f in range(N_FOLDS)
            ]
            
            # Verify all necessary MLP model paths exist
            all_base_models_exist = True
            for p_path in mlp_fold_model_paths_for_kd:
                if not os.path.exists(p_path):
                    print(f"{TermColors.RED}  Missing base model for KD teacher: {p_path}{TermColors.ENDC}")
                    all_base_models_exist = False
            
            if not all_base_models_exist:
                print(f"{TermColors.RED}Not all base models for KD teacher found. Skipping distillation of stacking model.{TermColors.ENDC}")
            else:
                stacking_teacher_components_for_kd = {
                    "stacking_model_path": mlp_stacking_model_path_result,
                    "feature_extractor_model_names": MODEL_NAMES, 
                    "feature_extractor_embedding_size": EMBEDDING_SIZE, 
                    "feature_extractor_arcface_m": ARCFACE_M, 
                    "feature_extractor_metric_learning": METRIC_LEARNING_TYPE, 
                    "feature_extractor_checkpoint_path": FEATURE_EXTRACTOR_CHECKPOINT_PATH, 
                    "base_mlp_model_paths": mlp_fold_model_paths_for_kd,
                    "base_mlp_model_input_size": base_mlp_input_size_for_kd,
                    "n_folds_for_stacking": N_FOLDS, 
                    "num_classes": NUM_CLASSES 
                }
                try:
                    train_student_model(
                        student_model_name=KD_STUDENT_MODEL_NAME,
                        student_base_save_path=KD_STUDENT_MODEL_SAVE_PATH, 
                        student_swa_save_path=KD_STUDENT_SWA_MODEL_SAVE_PATH, 
                        student_ema_save_path=KD_STUDENT_EMA_MODEL_SAVE_PATH, 
                        df_full_for_kd=df_full, 
                        image_dir=IMAGE_DIR,
                        num_classes=NUM_CLASSES,
                        stacking_teacher_components=stacking_teacher_components_for_kd
                    )
                except Exception as e_kd_main:
                    print(f"{TermColors.RED}Error during train_student_model (Stacking Teacher): {e_kd_main}{TermColors.ENDC}"); traceback.print_exc()
    elif RUN_KNOWLEDGE_DISTILLATION and training_completed_successfully: # Stacking model path might be missing
         print(f"{TermColors.YELLOW}Knowledge Distillation was enabled, but stacking model path ('{mlp_stacking_model_path_result}') was not available or training was interrupted. Skipping distillation of stacking model.{TermColors.ENDC}")
    elif RUN_KNOWLEDGE_DISTILLATION and not training_completed_successfully:
        print(f"{TermColors.YELLOW}Knowledge Distillation was enabled, but previous training steps did not complete successfully. Skipping.{TermColors.ENDC}")


    print(f"\n{TermColors.OKGREEN}All processes complete for RUN_MODE: {RUN_MODE}.{TermColors.ENDC}")

if __name__ == "__main__":
    main()