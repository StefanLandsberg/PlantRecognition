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
TIMM_AVAILABLE = False
TORCHMETRICS_AVAILABLE = False
ALBUMENTATIONS_AVAILABLE = False
ADAMP_AVAILABLE = False
SAM_AVAILABLE = False
_INITIALIZED = False
LGBM_PRINTED_INFO = False
# OPTUNA_AVAILABLE is not strictly needed in this script if MLP HPO is removed
# However, stacking HPO might use it if you choose to implement it with Optuna later.
# For now, removing Optuna-specific flags unless Stacking HPO needs it.
# Stacking HPO currently uses GridSearchCV.

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    print("FATAL: timm library not found. Required. Install with 'pip install timm'.")
    sys.exit(1)

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

_INITIALIZED = False
def print_library_info():
    global _INITIALIZED, LGBM_PRINTED_INFO
    if _INITIALIZED:
        return
    
    print("INFO: timm library found.")
    if TORCHMETRICS_AVAILABLE: print("INFO: torchmetrics library found.")
    if ALBUMENTATIONS_AVAILABLE: print("INFO: albumentations library found.")
    if ADAMP_AVAILABLE: print("INFO: adamp optimizer found.")
    if SAM_AVAILABLE: print("INFO: sam_optimizer found and available for use.")
    else: print("WARN: sam_optimizer not found, SAM will be disabled even if USE_SAM is True in config.")
    
    if LGBM_AVAILABLE and not LGBM_PRINTED_INFO:
        print("INFO: lightgbm library found and available for stacking.")
        LGBM_PRINTED_INFO = True
    elif not LGBM_AVAILABLE and not LGBM_PRINTED_INFO:
        print(f"{TermColors.WARN}WARN: lightgbm library not found. Stacking with LGBM will not be available. Install with 'pip install lightgbm'.{TermColors.ENDC}")
        LGBM_PRINTED_INFO = True
    _INITIALIZED = True

# --- Terminal Colors ---
colorama.init(autoreset=True)
class TermColors:
    HEADER = '\033[95m'; OKBLUE = '\033[94m'; OKCYAN = '\033[96m'; OKGREEN = '\033[92m'
    WARNING = '\033[93m'; FAIL = '\033[91m'; ENDC = '\033[0m'; BOLD = '\033[1m'
    UNDERLINE = '\033[4m'; INFO = '\033[94m'; DEBUG = '\033[90m'; TRACE = '\033[90m'
    ERROR = '\033[91m'; SUCCESS = '\033[92m'; WARN = '\033[93m'
    CRITICAL = '\033[91m' + '\033[1m'; BLUE = '\033[94m'; CYAN = '\033[96m'
    GREEN = '\033[92m'; YELLOW = '\033[93m'; RED = '\033[91m'; MAGENTA = '\033[95m'; DIM = '\033[2m'

# --- Configuration ---
SEED = 42
DEBUG_MODE = False # Set to True for small dataset and fewer epochs

# --- Path Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Assuming this script is in PlantRecognition/PlantRgognotionV2/scripts/
# Adjust V3_DIR and PROJECT_ROOT if the script is moved elsewhere relative to the project structure.
V3_DIR = os.path.dirname(SCRIPT_DIR) # Should be PlantRecognition/PlantRgognotionV2/
PROJECT_ROOT = os.path.dirname(V3_DIR) # Should be PlantRecognition/

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "plant_images")
CSV_PATH = os.path.join(DATA_DIR, "observations-568783.csv") # Example CSV path

BASE_CHECKPOINT_DIR = os.path.join(V3_DIR, "checkpoints_full_training_pytorch") # Changed suffix
BASE_LOG_DIR = os.path.join(V3_DIR, "logs_full_training_pytorch") # Changed suffix
BASE_MODEL_SAVE_DIR = os.path.join(V3_DIR, "models_full_training_pytorch") # Changed suffix
BASE_ERROR_ANALYSIS_DIR = os.path.join(V3_DIR, "error_analysis_full_training_pytorch") # Changed suffix

# Feature extraction path - still needed if KD uses a feature extractor component
FEATURES_NPZ_PATH = os.path.join(DATA_DIR, "extracted_features_combined_model_v3.npz") # Path if features are extracted by CombinedModel
FEATURE_EXTRACTION_IMAGE_SIZE_CONFIG = (512, 512)
FEATURE_EXTRACTOR_BATCH_SIZE = 64
FEATURE_EXTRACTOR_CHECKPOINT_PATH = None # Optional: Path to a specific checkpoint for the CombinedModel if used as a feature extractor for KD

os.makedirs(BASE_CHECKPOINT_DIR, exist_ok=True); os.makedirs(BASE_LOG_DIR, exist_ok=True)
os.makedirs(BASE_MODEL_SAVE_DIR, exist_ok=True); os.makedirs(BASE_ERROR_ANALYSIS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# --- Training Config (for CombinedModel) ---
BATCH_SIZE = 32 if not DEBUG_MODE else 4
GRADIENT_ACCUMULATION_STEPS = 2 if not DEBUG_MODE else 1
LEARNING_RATE = 1e-4; WEIGHT_DECAY = 1e-5
OPTIMIZER_TYPE = 'AdamP' if ADAMP_AVAILABLE else 'AdamW'

# SAM Configuration (for CombinedModel)
USE_SAM = False 
SAM_RHO = 0.05; SAM_ADAPTIVE = True
USE_AUTO_TRAIN_CONFIG = not USE_SAM # AutoTrainingConfig for CombinedModel

GRADIENT_CLIP_VAL = 1.0
PROGRESSIVE_RESIZING_STAGES = [
    (12 if not DEBUG_MODE else 1, (224, 224)), (10 if not DEBUG_MODE else 1, (384, 384)),
    (8 if not DEBUG_MODE else 1, (448, 448)), (7 if not DEBUG_MODE else 1, (512, 512)),
]
TOTAL_EPOCHS_PER_FOLD = sum(s[0] for s in PROGRESSIVE_RESIZING_STAGES)
CURRENT_IMAGE_SIZE = None
EARLY_STOPPING_PATIENCE = 10 if not DEBUG_MODE else 3

# --- Cross-Validation Config ---
N_FOLDS = 5 if not DEBUG_MODE else 2

# --- Hardware Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 8
MIXED_PRECISION = True if DEVICE.type == 'cuda' else False
USE_TORCH_COMPILE = False # Keep as False unless specifically testing with PyTorch 2.0+ compile

# --- Model Config (for CombinedModel / Feature Extractor) ---
MODEL_NAMES = ["tf_efficientnetv2_l_in21ft1k", "convnext_large_in22ft1k"] 
DROP_PATH_RATE = 0.1; PRETRAINED = True; NUM_CLASSES = -1
EMBEDDING_SIZE = 2048; DROPOUT_RATE = 0.3; GLOBAL_POOLING = 'avg'

# --- Metric Learning Config (for CombinedModel) ---
METRIC_LEARNING_TYPE = 'ArcFace'
ARCFACE_S = 30.0; ARCFACE_M = 0.6

# --- Loss Function & Imbalance Handling ---
LOSS_TYPE = 'CrossEntropy'
LABEL_SMOOTHING = 0.1
FOCAL_ALPHA = 0.25; FOCAL_GAMMA = 2.0
IMBALANCE_STRATEGY = 'WeightedSampler'
LOGIT_ADJUSTMENT_TAU = 1.0
CLASS_FREQUENCIES = None; CLASS_PRIORS = None; CLASS_WEIGHTS = None

# --- Learning Rate Scheduler Config (for CombinedModel) ---
SCHEDULER_TYPE = 'CosineWarmRestarts'
WARMUP_EPOCHS = 3; LR_MAX = LEARNING_RATE; LR_MIN = LEARNING_RATE * 0.01
T_0 = 10; T_MULT = 1
STEP_LR_STEP_SIZE = 5; STEP_LR_GAMMA = 0.1
PLATEAU_FACTOR = 0.2; PLATEAU_PATIENCE = 5; PLATEAU_MIN_LR = 1e-6
PLATEAU_MODE = 'min'; PLATEAU_MONITOR = 'val_loss'

# --- Augmentation Config (for CombinedModel training) ---
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
SAVE_TOP_K = 1 # Not directly used, but good to keep in mind for checkpoint logic

# --- Error Analysis Config ---
ERROR_LOG_BATCH_SIZE = 64
LOG_MISCLASSIFIED_IMAGES = False

# --- Test Time Augmentation (TTA) Config ---
USE_TTA = True
TTA_TRANSFORMS = None

# --- Stacking Config ---
RUN_STACKING = True
STACKING_META_MODEL_PATH = os.path.join(BASE_MODEL_SAVE_DIR, "stacking_meta_model_full_training.joblib") # Suffix change
STACKING_OOF_PREDS_PATH = os.path.join(BASE_MODEL_SAVE_DIR, "oof_predictions_full_training.npz") # Suffix change
STACKING_DO_HPO = True
STACKING_HPO_CV_FOLDS = 3
STACKING_LGBM_PARAM_GRID = {
    'n_estimators': [100, 200, 400], 'learning_rate': [0.02, 0.05, 0.1],
    'num_leaves': [20, 31, 40], 'max_depth': [-1, 10, 15],
    'min_child_samples': [15, 20, 30], 'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9], 'reg_alpha': [0, 0.01, 0.1], 'reg_lambda': [0, 0.01, 0.1],
}

# --- Knowledge Distillation Config ---
RUN_KNOWLEDGE_DISTILLATION = True # Set to False if KD is not desired in this script
KD_STUDENT_MODEL_NAME = "mobilenetv3_small_100"
KD_STUDENT_IMAGE_SIZE = (224, 224)
KD_STUDENT_EMBEDDING_SIZE = 512
KD_STUDENT_DROPOUT = 0.2
KD_EPOCHS = 15 if not DEBUG_MODE else 2
KD_BATCH_SIZE = BATCH_SIZE * 2 # Can be larger for smaller student
KD_LR = 1e-4
KD_ALPHA = 0.5 
KD_TEMPERATURE = 4.0
KD_STUDENT_MODEL_SAVE_PATH = os.path.join(BASE_MODEL_SAVE_DIR, f"distilled_student_{KD_STUDENT_MODEL_NAME}_base.pth")
KD_STUDENT_SWA_MODEL_SAVE_PATH = os.path.join(BASE_MODEL_SAVE_DIR, f"distilled_student_{KD_STUDENT_MODEL_NAME}_swa.pth")
KD_STUDENT_EMA_MODEL_SAVE_PATH = os.path.join(BASE_MODEL_SAVE_DIR, f"distilled_student_{KD_STUDENT_MODEL_NAME}_ema.pth")

KD_STUDENT_USE_SWA = True; KD_STUDENT_SWA_START_EPOCH_FACTOR = 0.75
KD_STUDENT_SWA_LR_FACTOR = 0.05; KD_STUDENT_SWA_ANNEAL_EPOCHS = 5
KD_STUDENT_USE_EMA = True; KD_STUDENT_EMA_DECAY = 0.999

IMAGENET_MEAN = [0.485, 0.456, 0.406]; IMAGENET_STD = [0.229, 0.224, 0.225]

# --- Global Variables ---
stop_requested = False; label_encoder = None; class_names = None

# --- Utility Functions ---
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    print(f"{TermColors.INFO}Seed set to {seed}{TermColors.ENDC}")

def handle_interrupt(signal, frame):
    global stop_requested
    if stop_requested: print(f"\n{TermColors.CRITICAL}Force exiting...{TermColors.ENDC}"); sys.exit(1)
    print(f"\n{TermColors.WARNING}Interrupt received. Finishing current epoch... Press Ctrl+C again to force exit.{TermColors.ENDC}")
    stop_requested = True

def check_keyboard_stop():
    if stop_requested: print(f"{TermColors.WARNING}Stop request detected. Breaking loop...{TermColors.ENDC}")
    return stop_requested

# --- Checkpointing, Saving, Logging ---
def save_checkpoint(fold, global_epoch, stage_idx, stage_epoch, model, optimizer, scheduler, scaler, best_metric, filename="checkpoint.pth.tar"):
    checkpoint_dir = os.path.join(BASE_CHECKPOINT_DIR, f"fold_{fold}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    optimizer_is_sam_instance = hasattr(optimizer, 'base_optimizer') and SAM_AVAILABLE
    should_treat_as_sam_for_saving = optimizer_is_sam_instance and USE_SAM # USE_SAM is for CombinedModel
        
    opt_state_dict = optimizer.base_optimizer.state_dict() if should_treat_as_sam_for_saving else optimizer.state_dict()

    state = {
        'fold': fold, 'global_epoch': global_epoch, 'stage_idx': stage_idx, 'stage_epoch': stage_epoch,
        'image_size': CURRENT_IMAGE_SIZE, 'state_dict': model_state_dict, 'optimizer': opt_state_dict,
        'scheduler': scheduler.state_dict() if scheduler else None, 'scaler': scaler.state_dict() if scaler else None,
        'best_metric': best_metric, 'label_encoder_classes': list(label_encoder.classes_) if label_encoder else None,
        'class_frequencies': CLASS_FREQUENCIES, 'is_sam_optimizer': should_treat_as_sam_for_saving
    }
    try:
        torch.save(state, filepath)
    except Exception as e:
        print(f"{TermColors.RED}Error saving checkpoint {filepath}: {e}{TermColors.ENDC}")

def load_checkpoint(fold, model, optimizer, scheduler, scaler, filename="checkpoint.pth.tar"):
    checkpoint_dir = os.path.join(BASE_CHECKPOINT_DIR, f"fold_{fold}")
    filepath = os.path.join(checkpoint_dir, filename)
    start_global_epoch, start_stage_idx, start_stage_epoch = 0, 0, 0
    loaded_image_size, loaded_class_frequencies, loaded_label_classes = None, None, None
    best_metric = float('-inf') if CHECKPOINT_MODE == 'max' else float('inf')

    if os.path.isfile(filepath):
        print(f"{TermColors.CYAN}Loading Fold {fold} checkpoint '{filename}'...{TermColors.ENDC}")
        try:
            ckpt = torch.load(filepath, map_location=DEVICE)
            start_global_epoch = ckpt.get('global_epoch', 0)
            start_stage_idx = ckpt.get('stage_idx', 0)
            start_stage_epoch = ckpt.get('stage_epoch', 0)
            loaded_image_size = ckpt.get('image_size', None)
            loaded_class_frequencies = ckpt.get('class_frequencies', None)
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
                if not is_compiled and name.startswith('_orig_mod.'): name = name[len('_orig_mod.'):]
                new_state_dict[name] = v
            
            try: model.load_state_dict(new_state_dict, strict=False)
            except RuntimeError as e: print(f"{TermColors.YELLOW}Model Load Warning (strict=False): {e}{TermColors.ENDC}")
            except Exception as e: print(f"{TermColors.RED}Model Load Failed: {e}{TermColors.ENDC}")

            if optimizer and 'optimizer' in ckpt and ckpt['optimizer']:
                current_optimizer_should_be_sam = USE_SAM and SAM_AVAILABLE and hasattr(optimizer, 'base_optimizer')
                opt_to_load_state_into = None
                if current_optimizer_should_be_sam and was_sam_optimizer_in_ckpt: opt_to_load_state_into = optimizer.base_optimizer
                elif not current_optimizer_should_be_sam and not was_sam_optimizer_in_ckpt: opt_to_load_state_into = optimizer
                else:
                    print(f"{TermColors.YELLOW}WARN: Optimizer type mismatch (SAM Ckpt: {was_sam_optimizer_in_ckpt}, SAM Current: {current_optimizer_should_be_sam}).{TermColors.ENDC}")
                    opt_to_load_state_into = optimizer.base_optimizer if current_optimizer_should_be_sam else optimizer
                if opt_to_load_state_into:
                    try: opt_to_load_state_into.load_state_dict(ckpt['optimizer'])
                    except Exception as e: print(f"{TermColors.YELLOW}Optim Load Failed: {e}{TermColors.ENDC}")
            
            if scheduler and 'scheduler' in ckpt and ckpt['scheduler']:
                 try: scheduler.load_state_dict(ckpt['scheduler'])
                 except Exception as e: print(f"{TermColors.YELLOW}Scheduler Load Failed: {e}{TermColors.ENDC}")
            if scaler and 'scaler' in ckpt and ckpt['scaler']:
                try: scaler.load_state_dict(ckpt['scaler'])
                except Exception as e: print(f"{TermColors.YELLOW}Scaler Load Failed: {e}{TermColors.ENDC}")

            print(f"{TermColors.GREEN}Ckpt Fold {fold} loaded. Resume GlobEp {start_global_epoch}. Best {CHECKPOINT_MONITOR}: {best_metric:.4f}{TermColors.ENDC}")
            if loaded_label_classes and label_encoder and list(label_encoder.classes_) != loaded_label_classes:
                print(f"{TermColors.CRITICAL}Label mapping mismatch! Exiting.{TermColors.ENDC}"); sys.exit(1)
        except Exception as e:
            print(f"{TermColors.RED}Error loading checkpoint {filepath}: {e}{TermColors.ENDC}"); traceback.print_exc()
            start_global_epoch, start_stage_idx, start_stage_epoch = 0, 0, 0
            best_metric = float('-inf') if CHECKPOINT_MODE == 'max' else float('inf')
    else:
        print(f"{TermColors.YELLOW}No checkpoint found for Fold {fold} at {filepath}. Starting fresh.{TermColors.ENDC}")
    
    return start_global_epoch, start_stage_idx, start_stage_epoch, best_metric, loaded_label_classes, loaded_image_size, loaded_class_frequencies

def save_model(fold, model, filename="final_model.pth"):
    model_dir = os.path.join(BASE_MODEL_SAVE_DIR, f"fold_{fold}")
    os.makedirs(model_dir, exist_ok=True)
    filepath = os.path.join(model_dir, filename)
    model_to_save = model
    if hasattr(model_to_save, 'module'): model_to_save = model_to_save.module
    if hasattr(model_to_save, '_orig_mod'): model_to_save = model_to_save._orig_mod
    try: torch.save(model_to_save.state_dict(), filepath)
    except Exception as e: print(f"{TermColors.RED}Error saving model {filepath}: {e}{TermColors.ENDC}")

def log_misclassified(fold, model, dataloader, criterion, device, global_epoch, writer, num_classes, max_images=20):
    if not LOG_MISCLASSIFIED_IMAGES: return
    error_dir = os.path.join(BASE_ERROR_ANALYSIS_DIR, f"fold_{fold}")
    os.makedirs(error_dir, exist_ok=True)
    error_log_file = os.path.join(error_dir, f"epoch_{global_epoch}_errors.csv")
    model.eval(); misclassified_count = 0; logged_images = 0
    try:
        with open(error_log_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['image_path', 'true_label', 'predicted_label', 'confidence', 'loss', 'logits_raw', 'logits_adjusted']
            writer_csv = csv.DictWriter(csvfile, fieldnames=fieldnames); writer_csv.writeheader()
            with torch.no_grad():
                for batch_data in tqdm(dataloader, desc=f"Logging Errors Fold {fold}", leave=False):
                    if len(batch_data) != 3: continue # images, labels, paths
                    inputs, labels, paths = batch_data
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    with torch.amp.autocast('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda')):
                        outputs_raw = model(inputs, labels=labels if METRIC_LEARNING_TYPE == 'ArcFace' else None)
                        adj_outputs = outputs_raw
                        if IMBALANCE_STRATEGY == 'LogitAdjust' and CLASS_PRIORS is not None:
                            logit_adj_val = LOGIT_ADJUSTMENT_TAU * torch.log(CLASS_PRIORS + 1e-12)
                            adj_outputs = outputs_raw + logit_adj_val.unsqueeze(0)
                        
                        final_outputs_for_loss_pred = adj_outputs
                        if hasattr(model, 'metric_fc') and isinstance(model.metric_fc, ArcFace): # If model has ArcFace
                             final_outputs_for_loss_pred = model.metric_fc(outputs_raw, labels) # Pass labels to ArcFace
                             if IMBALANCE_STRATEGY == 'LogitAdjust' and CLASS_PRIORS is not None:
                                 final_outputs_for_loss_pred = final_outputs_for_loss_pred + logit_adj_val.unsqueeze(0)
                        
                        loss = criterion(final_outputs_for_loss_pred, labels)
                        preds = torch.argmax(final_outputs_for_loss_pred, dim=1)
                        probs = F.softmax(final_outputs_for_loss_pred, dim=1)

                    misclassified_mask = (preds != labels)
                    misclassified_indices = torch.where(misclassified_mask)[0]
                    for idx in misclassified_indices:
                        misclassified_count += 1; true_label_idx = labels[idx].item(); pred_label_idx = preds[idx].item()
                        confidence = probs[idx, pred_label_idx].item()
                        item_loss = F.cross_entropy(final_outputs_for_loss_pred[idx].unsqueeze(0), labels[idx].unsqueeze(0)).item()
                        identifier = os.path.basename(paths[idx]) if isinstance(paths[idx], str) else str(paths[idx])
                        true_n = class_names[true_label_idx] if class_names and 0 <= true_label_idx < len(class_names) else str(true_label_idx)
                        pred_n = class_names[pred_label_idx] if class_names and 0 <= pred_label_idx < len(class_names) else str(pred_label_idx)
                        writer_csv.writerow({'image_path': identifier, 'true_label': true_n, 'predicted_label': pred_n, 
                                             'confidence': f"{confidence:.4f}", 'loss': f"{item_loss:.4f}", 
                                             'logits_raw': outputs_raw[idx].cpu().numpy().round(2).tolist(), 
                                             'logits_adjusted': adj_outputs[idx].cpu().numpy().round(2).tolist()})
                        if writer and logged_images < max_images:
                            try:
                                mean = torch.tensor(IMAGENET_MEAN).view(3,1,1).to(device); std = torch.tensor(IMAGENET_STD).view(3,1,1).to(device)
                                img_tensor = inputs[idx] * std + mean; img_tensor = torch.clamp(img_tensor, 0, 1)
                                writer.add_image(f"Misclassified/Fold_{fold}/True_{true_n}_Pred_{pred_n}_{identifier}", img_tensor, global_epoch)
                                logged_images += 1
                            except Exception as img_e: print(f"{TermColors.YELLOW}Warn: Failed to log image {identifier}: {img_e}{TermColors.ENDC}")
                    if stop_requested: break
            if stop_requested: return
        print(f"{TermColors.CYAN}Fold {fold} Misclassified images logged ({misclassified_count} errors). CSV: {error_log_file}{TermColors.ENDC}")
    except Exception as e: print(f"{TermColors.RED}Error logging misclassified for fold {fold}: {e}{TermColors.ENDC}"); traceback.print_exc()

# --- Dataset and Transforms ---
class PlantDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None, label_encoder_instance=None, include_paths=False, image_size_override=None):
        self.input_df = dataframe.copy()
        self.image_dir = image_dir
        self.transform = transform
        self.include_paths = include_paths
        self.image_size = image_size_override if image_size_override else (CURRENT_IMAGE_SIZE if CURRENT_IMAGE_SIZE else PROGRESSIVE_RESIZING_STAGES[0][1])
        self.image_data = []
        
        required_cols = ['scientificName', 'id', 'label']
        if not all(col in self.input_df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in self.input_df.columns]
            print(f"{TermColors.RED}PlantDataset input df missing: {missing}. Found: {self.input_df.columns.tolist()}{TermColors.ENDC}")
            self.dataframe = pd.DataFrame(self.image_data); return

        image_file_lookup = defaultdict(list)
        if not os.path.isdir(self.image_dir):
            print(f"{TermColors.RED}Image directory not found: {self.image_dir}{TermColors.ENDC}")
            self.dataframe = pd.DataFrame(self.image_data); return
            
        for root, _, files in os.walk(self.image_dir):
            species_dir_name_from_path = os.path.basename(root)
            prefix_to_strip = species_dir_name_from_path + "_"
            for filename in files:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root, filename)
                    try:
                        if os.path.getsize(full_path) > 0:
                            if filename.startswith(prefix_to_strip):
                                id_part = filename[len(prefix_to_strip):]
                                obs_id_from_file = id_part.split('_')[0]
                                if '.' in obs_id_from_file: obs_id_from_file = obs_id_from_file.split('.')[0]
                                if obs_id_from_file: image_file_lookup[(species_dir_name_from_path, obs_id_from_file)].append(full_path)
                    except OSError: pass
        
        self.input_df['original_index_in_df'] = self.input_df.index
        for _, row in tqdm(self.input_df.iterrows(), total=len(self.input_df), desc="Matching CSV to images", leave=False, disable=DEBUG_MODE):
            try:
                species_name_csv = str(row['scientificName']); obs_id_csv = str(row['id']); label = row['label']
                original_idx = row['original_index_in_df']
                species_dir_name_csv_derived = species_name_csv.replace(' ', '_').replace('/', '_').replace('\\', '_')
                found_files = image_file_lookup.get((species_dir_name_csv_derived, obs_id_csv), [])
                for full_path in found_files:
                    self.image_data.append({'scientificName': species_name_csv, 'label': label, 
                                            'image_path': full_path, 'original_index': original_idx})
            except Exception as e: print(f"{TermColors.RED}Error processing CSV row (ID: {row.get('id', 'N/A')}): {e}{TermColors.ENDC}")
        
        self.dataframe = pd.DataFrame(self.image_data)
        if len(self.dataframe) == 0 and len(self.input_df) > 0: print(f"{TermColors.RED}0 images matched.{TermColors.ENDC}")
        else: print(f"{TermColors.INFO}PlantDataset: Matched {len(self.dataframe)} images from {len(self.input_df)} CSV rows.{TermColors.ENDC}")

    def __len__(self): return len(self.dataframe)
    def get_labels(self):
        if 'label' in self.dataframe.columns: return self.dataframe['label'].tolist()
        else: print(f"{TermColors.RED}'label' missing!{TermColors.ENDC}"); return []

    def __getitem__(self, idx):
        if idx >= len(self.dataframe):
             dummy_img = torch.zeros((3, *self.image_size), dtype=torch.float32); label = -1; original_index = -1
             return (dummy_img, label, "ERROR_IDX_OOB", original_index) if self.include_paths else (dummy_img, label)
        
        row = self.dataframe.iloc[idx]; img_path = row['image_path']; label = row['label']
        original_index = row['original_index']
        try:
            image = Image.open(img_path).convert('RGB'); image = np.array(image)
            if image is None: raise IOError("Image None")
        except Exception as e:
            dummy_img = torch.zeros((3, *self.image_size), dtype=torch.float32)
            err_fn = os.path.basename(img_path) if isinstance(img_path, str) else "UNKNOWN"
            return (dummy_img, label if isinstance(label,int) else -1, f"ERROR_LOAD_{err_fn}", original_index) if self.include_paths else (dummy_img, label if isinstance(label,int) else -1)

        if self.transform:
            try: augmented = self.transform(image=image); image = augmented['image']
            except Exception as e:
                pil_image = Image.fromarray(image)
                fallback_tf = transforms.Compose([transforms.Resize(self.image_size), transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
                try: image = fallback_tf(pil_image)
                except Exception as fb_e:
                    dummy_img = torch.zeros((3, *self.image_size), dtype=torch.float32); err_fn = os.path.basename(img_path) if isinstance(img_path, str) else "UNKNOWN"
                    return (dummy_img, label if isinstance(label,int) else -1, f"ERROR_TF_{err_fn}", original_index) if self.include_paths else (dummy_img, label if isinstance(label,int) else -1)
        
        return (image, label, img_path, original_index) if self.include_paths else (image, label)

def get_transforms(image_size=(224, 224), for_feature_extraction=False): # Simplified augmentations_config
    h, w = int(image_size[0]), int(image_size[1])
    if for_feature_extraction:
        return A.Compose([A.Resize(h,w, interpolation=cv2.INTER_LINEAR), A.Normalize(IMAGENET_MEAN, IMAGENET_STD), ToTensorV2()])

    train_tf_list = [A.Resize(h,w, interpolation=cv2.INTER_LINEAR), A.HorizontalFlip(p=0.5), 
                       A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5), 
                       A.Normalize(IMAGENET_MEAN, IMAGENET_STD), ToTensorV2()]
    if USE_RAND_AUGMENT: train_tf_list.insert(1, A.RandAugment(n=RAND_AUGMENT_N, m=RAND_AUGMENT_M, p=0.5)) # Insert before Normalize
    
    train_transform = A.Compose(train_tf_list)
    val_transform = A.Compose([A.Resize(h,w, interpolation=cv2.INTER_LINEAR), A.Normalize(IMAGENET_MEAN, IMAGENET_STD), ToTensorV2()])
    
    global TTA_TRANSFORMS 
    if USE_TTA: TTA_TRANSFORMS = A.Compose([A.HorizontalFlip(p=1.0), A.Resize(h,w, interpolation=cv2.INTER_LINEAR), A.Normalize(IMAGENET_MEAN, IMAGENET_STD), ToTensorV2()])
    else: TTA_TRANSFORMS = None
    return train_transform, val_transform

# --- Model Architecture ---
class ArcFace(nn.Module): # Copied as is
    def __init__(self, in_features, out_features, s=ARCFACE_S, m=ARCFACE_M, easy_margin=False, ls_eps=0.0):
        super(ArcFace, self).__init__(); self.in_features = in_features; self.out_features = out_features; self.s = s; self.m = m; self.ls_eps = ls_eps
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features)); nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin; self.cos_m = math.cos(m); self.sin_m = math.sin(m); self.th = math.cos(math.pi - m); self.mm = math.sin(math.pi - m) * m
    def forward(self, input_features, label): 
        cosine = F.linear(F.normalize(input_features), F.normalize(self.weight))
        if label is None: return cosine * self.s # For inference or if labels not provided
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).clamp(0, 1); phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin: phi = torch.where(cosine > 0, phi, cosine)
        else: phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=input_features.device); one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0: one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine); output *= self.s
        return output

class CombinedModel(nn.Module): # Copied as is
    def __init__(self, model_names, num_classes, pretrained=True, global_pool='avg', dropout_rate=0.3, embedding_size=512, drop_path_rate=0.1, arcface_s=ARCFACE_S, arcface_m=ARCFACE_M, metric_learning=METRIC_LEARNING_TYPE):
        super().__init__(); self.model_names = model_names; self.num_classes = num_classes; self.embedding_size = embedding_size; self.metric_learning = metric_learning
        self.backbones = nn.ModuleList(); self.total_features = 0; self.feature_dims = {}
        for name in model_names:
            try:
                kwargs = {}; supported_families = ['efficientnet', 'convnext', 'vit', 'swin', 'beit', 'deit']; model_family = name.split('_')[0].lower()
                if any(family in model_family for family in supported_families) and drop_path_rate > 0: kwargs['drop_path_rate'] = drop_path_rate
                backbone = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool=global_pool, **kwargs)
                backbone_features = backbone.num_features
                # Specific feature dim overrides if timm reports them differently with global_pool='avg'
                if 'mobilenetv3_large_100' in name and global_pool == 'avg': backbone_features = 960 
                elif 'efficientnetv2_l' in name and global_pool == 'avg': backbone_features = 1280
                self.feature_dims[name] = backbone_features; self.backbones.append(backbone); self.total_features += backbone_features
            except Exception as e: print(f"{TermColors.RED}Backbone Load Fail {name}: {e}{TermColors.ENDC}"); raise e
        self.embedding_layer = nn.Sequential(nn.Linear(self.total_features, self.embedding_size), nn.BatchNorm1d(self.embedding_size), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(dropout_rate)
        if self.metric_learning == 'ArcFace': self.metric_fc = ArcFace(self.embedding_size, num_classes, s=arcface_s, m=arcface_m)
        else: self.metric_fc = nn.Linear(self.embedding_size, num_classes)
    def forward(self, x, labels=None, return_embedding=False):
        all_features = [backbone(x) for backbone in self.backbones]
        combined_features = torch.cat(all_features, dim=1) if len(all_features) > 1 else all_features[0]
        embedding = self.embedding_layer(combined_features)
        if return_embedding: return embedding
        embedding_dropped = self.dropout(embedding)
        if self.metric_learning == 'ArcFace': output = self.metric_fc(embedding_dropped, labels)
        else: output = self.metric_fc(embedding_dropped)
        return output

def build_model(model_names_list=MODEL_NAMES, num_classes_val=NUM_CLASSES, pretrained_val=PRETRAINED, dropout_val=DROPOUT_RATE, embedding_val=EMBEDDING_SIZE, drop_path_val=DROP_PATH_RATE, pool_val=GLOBAL_POOLING, arc_s_val=ARCFACE_S, arc_m_val=ARCFACE_M, metric_learn_val=METRIC_LEARNING_TYPE):
    return CombinedModel(model_names_list, num_classes_val, pretrained_val, pool_val, dropout_val, embedding_val, drop_path_val, arc_s_val, arc_m_val, metric_learn_val)

# --- Loss Functions ---
class FocalLoss(nn.Module): # Copied as is
    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, reduction='mean'):
        super().__init__(); self.alpha = alpha; self.gamma = gamma; self.reduction = reduction
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none'); pt = torch.exp(-ce_loss); focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss

def get_criterion(loss_type_val=LOSS_TYPE, label_smoothing_val=LABEL_SMOOTHING, class_weights_val=CLASS_WEIGHTS):
    weights = class_weights_val.to(DEVICE) if class_weights_val is not None and IMBALANCE_STRATEGY == 'WeightedLoss' else None
    if loss_type_val == 'FocalLoss': return FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing_val, weight=weights)

# --- Optimizer and Scheduler (Simplified for CombinedModel context) ---
def get_optimizer(model, optimizer_type_val=OPTIMIZER_TYPE, lr_val=LEARNING_RATE, weight_decay_val=WEIGHT_DECAY):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params: raise ValueError("No trainable parameters in model")
    
    if USE_SAM and SAM_AVAILABLE:
        from functools import partial
        base_optimizer_fn_choice = None
        if optimizer_type_val == 'AdamP' and ADAMP_AVAILABLE: base_optimizer_fn_choice = partial(AdamP, lr=lr_val, weight_decay=weight_decay_val, betas=(0.9,0.999), nesterov=True)
        elif optimizer_type_val == 'AdamW': base_optimizer_fn_choice = partial(optim.AdamW, lr=lr_val, weight_decay=weight_decay_val)
        else: 
            print(f"{TermColors.YELLOW}WARN: SAM base optimizer '{optimizer_type_val}' not AdamP/AdamW. Defaulting SAM base to AdamW.{TermColors.ENDC}")
            base_optimizer_fn_choice = partial(optim.AdamW, lr=lr_val, weight_decay=weight_decay_val)
        print(f"{TermColors.BLUE}INFO: Using SAM (rho={SAM_RHO}, adaptive={SAM_ADAPTIVE}). Base: {optimizer_type_val}{TermColors.ENDC}")
        return SAM(params, base_optimizer_fn_choice, rho=SAM_RHO, adaptive=SAM_ADAPTIVE)
    else:
        if optimizer_type_val == 'AdamP' and ADAMP_AVAILABLE: return AdamP(params, lr=lr_val, weight_decay=weight_decay_val, betas=(0.9,0.999), nesterov=True)
        if optimizer_type_val == 'AdamW': return optim.AdamW(params, lr=lr_val, weight_decay=weight_decay_val)
        if optimizer_type_val == 'SGD': return optim.SGD(params, lr=lr_val, weight_decay=weight_decay_val, momentum=0.9, nesterov=True)
        print(f"{TermColors.YELLOW}WARN: Optimizer '{optimizer_type_val}' not AdamP/AdamW/SGD. Defaulting to AdamW.{TermColors.ENDC}")
        return optim.AdamW(params, lr=lr_val, weight_decay=weight_decay_val)

def get_scheduler(optimizer, scheduler_type_val=SCHEDULER_TYPE, total_epochs_val=TOTAL_EPOCHS_PER_FOLD, 
                  warmup_epochs_val=WARMUP_EPOCHS, lr_max_val=LEARNING_RATE, lr_min_val=LR_MIN, 
                  t_0_val=T_0, t_mult_val=T_MULT, step_size_val=STEP_LR_STEP_SIZE, gamma_val=STEP_LR_GAMMA, 
                  plateau_factor_val=PLATEAU_FACTOR, plateau_patience_val=PLATEAU_PATIENCE, 
                  plateau_min_lr_val=PLATEAU_MIN_LR, plateau_mode_val=PLATEAU_MODE, plateau_monitor_val=PLATEAU_MONITOR):
    opt_for_scheduler = optimizer.base_optimizer if hasattr(optimizer, 'base_optimizer') else optimizer
    
    if scheduler_type_val == 'CosineWarmRestarts': main_scheduler = CosineAnnealingWarmRestarts(opt_for_scheduler, T_0=t_0_val, T_mult=t_mult_val, eta_min=lr_min_val)
    elif scheduler_type_val == 'StepLR': main_scheduler = StepLR(opt_for_scheduler, step_size=step_size_val, gamma=gamma_val)
    elif scheduler_type_val == 'ReduceLROnPlateau': main_scheduler = ReduceLROnPlateau(opt_for_scheduler, mode=plateau_mode_val, factor=plateau_factor_val, patience=plateau_patience_val, min_lr=plateau_min_lr_val, verbose=False, monitor=plateau_monitor_val)
    else: return None 
    
    if warmup_epochs_val > 0 and scheduler_type_val != 'ReduceLROnPlateau':
        warmup_scheduler = LinearLR(opt_for_scheduler, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs_val)
        return SequentialLR(opt_for_scheduler, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs_val])
    return main_scheduler

# --- Data Augmentation Helpers --- (Copied as is)
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
    # Simplified: This script only deals with CombinedModel EMA or KD Student EMA
    current_decay = EMA_DECAY # Default for CombinedModel
    if hasattr(averaged_model_parameter, '_is_kd_student_ema_param') and averaged_model_parameter._is_kd_student_ema_param:
        current_decay = KD_STUDENT_EMA_DECAY
    return current_decay * averaged_model_parameter + (1 - current_decay) * model_parameter

# --- Training & Validation Loops (Simplified for CombinedModel context) ---
def train_one_epoch(model, dataloader, criterion, optimizer, scaler, scheduler, global_epoch, stage_idx, stage_epoch, stage_total_epochs, device, writer, num_classes, ema_model, fold_num=0):
    model.train(); running_loss = 0.0; total_samples = 0; all_preds, all_labels = [], []
    is_sam_active_for_optim = hasattr(optimizer, 'base_optimizer') and SAM_AVAILABLE and USE_SAM

    pbar_desc = f"F{fold_num} E{global_epoch+1}/{stage_total_epochs} Tr"
    progress_bar = tqdm(dataloader, desc=pbar_desc, leave=True, bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}')
    if not is_sam_active_for_optim: optimizer.zero_grad() 
    
    for batch_idx, batch_data in enumerate(progress_bar):
        if check_keyboard_stop(): break
        if len(batch_data) != 2: continue # images, labels
        inputs, labels_orig = batch_data
        inputs, labels_orig = inputs.to(device), labels_orig.to(device); batch_size = inputs.size(0)

        use_mixup, use_cutmix = False, False
        r = np.random.rand()
        if mixup_alpha > 0 and cutmix_alpha > 0 and r < AUG_PROBABILITY:
            if np.random.rand() < 0.5: inputs, targets_a, targets_b, lam = mixup_data(inputs, labels_orig, mixup_alpha, device); use_mixup = True
            else: inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels_orig, cutmix_alpha, device); use_cutmix = True
        elif mixup_alpha > 0 and r < AUG_PROBABILITY: inputs, targets_a, targets_b, lam = mixup_data(inputs, labels_orig, mixup_alpha, device); use_mixup = True
        elif cutmix_alpha > 0 and r < AUG_PROBABILITY: inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels_orig, cutmix_alpha, device); use_cutmix = True
        else: lam = 1.0; targets_a, targets_b = labels_orig, labels_orig
        
        labels_for_arcface = targets_a

        with torch.amp.autocast('cuda', enabled=(MIXED_PRECISION and device.type == 'cuda')):
            if is_sam_active_for_optim:
                outputs1 = model(inputs, labels=labels_for_arcface if METRIC_LEARNING_TYPE == 'ArcFace' else None) 
                adj_outputs1 = outputs1
                if IMBALANCE_STRATEGY == 'LogitAdjust' and CLASS_PRIORS is not None:
                    logit_adj = LOGIT_ADJUSTMENT_TAU * torch.log(CLASS_PRIORS + 1e-12)
                    adj_outputs1 = outputs1 + logit_adj.unsqueeze(0)
                loss1 = mixup_criterion(criterion, adj_outputs1, targets_a, targets_b, lam) if use_mixup or use_cutmix else criterion(adj_outputs1, targets_a)
                scaler.scale(loss1 / GRADIENT_ACCUMULATION_STEPS).backward()
                optimizer.first_step(zero_grad=True)

                outputs2 = model(inputs, labels=labels_for_arcface if METRIC_LEARNING_TYPE == 'ArcFace' else None)
                adj_outputs_final = outputs2
                if IMBALANCE_STRATEGY == 'LogitAdjust' and CLASS_PRIORS is not None: adj_outputs_final = outputs2 + logit_adj.unsqueeze(0)
                loss2 = mixup_criterion(criterion, adj_outputs_final, targets_a, targets_b, lam) if use_mixup or use_cutmix else criterion(adj_outputs_final, targets_a)
                loss_final = loss2
                scaler.scale(loss_final / GRADIENT_ACCUMULATION_STEPS).backward()
            else: 
                outputs = model(inputs, labels=labels_for_arcface if METRIC_LEARNING_TYPE == 'ArcFace' else None)
                adj_outputs_final = outputs
                if IMBALANCE_STRATEGY == 'LogitAdjust' and CLASS_PRIORS is not None:
                    logit_adj = LOGIT_ADJUSTMENT_TAU * torch.log(CLASS_PRIORS + 1e-12)
                    adj_outputs_final = outputs + logit_adj.unsqueeze(0)
                loss = mixup_criterion(criterion, adj_outputs_final, targets_a, targets_b, lam) if use_mixup or use_cutmix else criterion(adj_outputs_final, targets_a)
                loss_final = loss
                scaler.scale(loss_final / GRADIENT_ACCUMULATION_STEPS).backward()

        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            if is_sam_active_for_optim: optimizer.second_step(zero_grad=True)
            else:
                if GRADIENT_CLIP_VAL > 0: scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VAL)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            if USE_EMA and ema_model: ema_model.update_parameters(model)
        
        if not torch.isnan(loss_final) and not torch.isinf(loss_final):
            current_step_loss = loss_final.item(); running_loss += current_step_loss * batch_size; total_samples += batch_size
            preds_for_acc = torch.argmax(adj_outputs_final, dim=1)
            all_preds.append(preds_for_acc.detach().cpu()); all_labels.append(labels_orig.detach().cpu())
            if batch_idx % 20 == 0 or batch_idx == len(dataloader) -1:
                current_acc = (torch.cat(all_preds) == torch.cat(all_labels)).sum().item() / len(torch.cat(all_labels)) if all_preds else 0.0
                progress_bar.set_postfix(loss=f"{current_step_loss:.3f}", acc=f"{current_acc:.3f}", lr=f"{optimizer.param_groups[0]['lr']:.1E}")
        else:
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS != 0 and not is_sam_active_for_optim: optimizer.zero_grad()
    
    if stop_requested: return None, None
    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_acc = (torch.cat(all_preds) == torch.cat(all_labels)).sum().item() / total_samples if all_preds and total_samples > 0 else 0.0
    
    swa_start_epoch_val = globals().get('swa_start_epoch_stage', float('inf'))
    if scheduler and not isinstance(scheduler, ReduceLROnPlateau) and not (USE_SWA and stage_epoch >= swa_start_epoch_val):
        scheduler.step()

    if writer:
        writer.add_scalar('Loss/train', epoch_loss, global_epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, global_epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], global_epoch)
    return epoch_loss, epoch_acc

def validate_one_epoch(model, dataloader, criterion, device, global_epoch, writer, num_classes, scheduler=None, swa_model=None, ema_model=None, return_preds=False, fold_num=0):
    model.eval(); 
    if swa_model: swa_model.eval()
    if ema_model: ema_model.eval()
    results = {}; oof_data = {'preds': [], 'indices': []}
    
    models_to_eval = {'base': model}
    if USE_SWA and swa_model: models_to_eval['swa'] = swa_model
    if USE_EMA and ema_model: models_to_eval['ema'] = ema_model

    with torch.no_grad():
        for model_key, current_model in models_to_eval.items():
            current_model.eval(); model_running_loss = 0.0; model_total_samples = 0
            model_all_preds, model_all_labels = [], []
            apply_tta = USE_TTA and TTA_TRANSFORMS is not None
            
            pbar_desc = f"Fold {fold_num} Validate GlobEp {global_epoch+1} ({model_key})"
            progress_bar = tqdm(dataloader, desc=pbar_desc, leave=False)
            for batch_data in progress_bar:
                original_indices_batch = None
                if len(batch_data) == 4 and return_preds: inputs, labels, _, original_indices_batch = batch_data # images, labels, paths, original_indices
                elif len(batch_data) == 2: inputs, labels = batch_data # images, labels
                else: continue
                
                inputs, labels = inputs.to(device), labels.to(device); batch_size = inputs.size(0)
                with torch.amp.autocast('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda')):
                    outputs_orig = current_model(inputs, labels=labels if METRIC_LEARNING_TYPE == 'ArcFace' else None)
                    adj_outputs = outputs_orig
                    if IMBALANCE_STRATEGY == 'LogitAdjust' and CLASS_PRIORS is not None:
                        logit_adj_val = LOGIT_ADJUSTMENT_TAU * torch.log(CLASS_PRIORS + 1e-12)
                        adj_outputs = outputs_orig + logit_adj_val.unsqueeze(0)
                    loss = criterion(adj_outputs, labels)

                tta_adj_outputs = None
                if apply_tta:
                    try:
                        inputs_tta_list = []
                        for i in range(inputs.size(0)):
                             img_np = inputs[i].cpu().permute(1,2,0).numpy() * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
                             img_np = (img_np * 255).astype(np.uint8)
                             augmented = TTA_TRANSFORMS(image=img_np); inputs_tta_list.append(augmented['image'])
                        inputs_tta = torch.stack(inputs_tta_list).to(device)
                        with torch.amp.autocast('cuda', enabled=(MIXED_PRECISION and device.type == 'cuda')):
                            outputs_tta_raw = current_model(inputs_tta, labels=labels if METRIC_LEARNING_TYPE == 'ArcFace' else None)
                            tta_adj_outputs = outputs_tta_raw
                            if IMBALANCE_STRATEGY == 'LogitAdjust' and CLASS_PRIORS is not None: tta_adj_outputs = outputs_tta_raw + logit_adj_val.unsqueeze(0)
                    except Exception as tta_e: print(f"{TermColors.YELLOW}Warn: TTA failed: {tta_e}{TermColors.ENDC}"); tta_adj_outputs = None
                
                final_outputs_for_preds = adj_outputs
                if tta_adj_outputs is not None: final_outputs_for_preds = (adj_outputs + tta_adj_outputs) / 2.0
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    model_running_loss += loss.item() * batch_size; model_total_samples += batch_size
                    preds = torch.argmax(final_outputs_for_preds, dim=1)
                    model_all_preds.append(preds.detach().cpu()); model_all_labels.append(labels.detach().cpu())
                    if return_preds and original_indices_batch is not None and model_key == 'base':
                         oof_data['preds'].append(F.softmax(final_outputs_for_preds, dim=1).detach().cpu())
                         oof_data['indices'].append(original_indices_batch.detach().cpu())
                else: print(f"{TermColors.RED}Warn: NaN/Inf validation loss.{TermColors.ENDC}")
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")
                if stop_requested: return None, None, None, None
            
            epoch_loss = model_running_loss / model_total_samples if model_total_samples > 0 else 0
            epoch_acc = (torch.cat(model_all_preds) == torch.cat(model_all_labels)).sum().item() / model_total_samples if model_all_preds and model_total_samples > 0 else 0.0
            results[model_key] = (epoch_loss, epoch_acc)
            if writer:
                writer.add_scalar(f'Loss/val_{model_key}', epoch_loss, global_epoch)
                writer.add_scalar(f'Accuracy/val_{model_key}', epoch_acc, global_epoch)
    
    base_loss, base_acc = results.get('base', (float('inf'), 0.0))
    oof_preds_concat = torch.cat(oof_data['preds']).numpy() if oof_data['preds'] else None
    oof_indices_concat = torch.cat(oof_data['indices']).numpy() if oof_data['indices'] else None
    
    if scheduler and isinstance(scheduler, ReduceLROnPlateau):
        metric_to_monitor_for_plateau = base_loss if PLATEAU_MONITOR == 'val_loss' else base_acc
        scheduler.step(metric_to_monitor_for_plateau)
    return base_loss, base_acc, oof_preds_concat, oof_indices_concat

# --- Stacking --- (Copied as is, uses global STACKING_... paths)
def train_stacking_meta_model(oof_preds, oof_labels, save_path):
    print(f"{TermColors.CYAN}Training Stacking Meta-Model...{TermColors.ENDC}")
    if oof_preds.ndim == 2 and oof_preds.shape[1] > 1: oof_features = oof_preds
    elif oof_preds.ndim == 2 and oof_preds.shape[1] == 1: oof_features = oof_preds
    elif oof_preds.ndim == 1: oof_features = oof_preds.reshape(-1, 1)
    else: oof_features = np.argmax(oof_preds, axis=1).reshape(-1, 1)
        
    meta_model_name_for_log = "LGBM"; meta_model = None; hpo_best_cv_score = None
    print_accuracy_on = "OOF training data"

    if LGBM_AVAILABLE:
        if STACKING_DO_HPO:
            lgbm_hpo_model = GridSearchCV(LGBMClassifier(random_state=SEED, n_jobs=-1, verbosity=-1), STACKING_LGBM_PARAM_GRID, cv=STACKING_HPO_CV_FOLDS, scoring='accuracy', n_jobs=-1, verbose=0)
            try:
                lgbm_hpo_model.fit(oof_features, oof_labels); meta_model = lgbm_hpo_model.best_estimator_
                hpo_best_cv_score = lgbm_hpo_model.best_score_; meta_model_name_for_log = "LGBM_HPO"
                print_accuracy_on = f"OOF CV (HPO {STACKING_HPO_CV_FOLDS}-fold)"
                print(f"{TermColors.GREEN}  LGBM HPO complete. Best CV score: {hpo_best_cv_score:.4f}{TermColors.ENDC}")
            except Exception as hpo_e: print(f"{TermColors.RED}  Error LGBM HPO: {hpo_e}. Default LGBM.{TermColors.ENDC}"); meta_model = None
        if meta_model is None:
            meta_model_name_for_log = "LGBM_Default"
            meta_model = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=SEED, n_jobs=-1, verbosity=-1)
            try: meta_model.fit(oof_features, oof_labels)
            except Exception as fit_e: print(f"{TermColors.RED}  Error default LGBM: {fit_e}. LogisticReg.{TermColors.ENDC}"); meta_model = None
    if meta_model is None:
        meta_model_name_for_log = "LogisticRegression" 
        meta_model = LogisticRegression(max_iter=1000, random_state=SEED, n_jobs=-1)
        try: meta_model.fit(oof_features, oof_labels)
        except Exception as lr_fit_e: print(f"{TermColors.RED}  Error LogisticReg: {lr_fit_e}. Stacking failed.{TermColors.ENDC}"); return
    try:
        meta_acc = hpo_best_cv_score if hpo_best_cv_score is not None and meta_model_name_for_log == "LGBM_HPO" else accuracy_score(oof_labels, meta_model.predict(oof_features))
        print(f"{TermColors.GREEN}Stacking meta-model ({meta_model_name_for_log}) trained. Acc ({print_accuracy_on}): {meta_acc:.4f}{TermColors.ENDC}")
        joblib.dump({"model": meta_model, "scaler": StandardScaler().fit(oof_features)}, save_path) # Example scaler
        print(f"  Meta-model saved: {save_path}")
    except Exception as e: print(f"{TermColors.RED}Error eval/save stacking meta-model: {e}{TermColors.ENDC}")

# --- Auto Training Configuration --- (Copied as is, applies to CombinedModel)
class AutoTrainingConfig:
    def __init__(self, initial_lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, plateau_patience=5, overfit_patience=3, lr_reduce_factor=0.5, wd_increase_factor=1.5, min_lr_factor=0.01, max_wd_factor=10.0, sam_rho_adjust_factor=1.2, max_sam_rho=0.1):
        self.initial_lr = initial_lr; self.initial_wd = weight_decay; self.current_lr = initial_lr; self.current_wd = weight_decay
        self.plateau_counter = 0; self.overfit_counter = 0; self.no_improvement_counter = 0
        self.plateau_patience = plateau_patience; self.overfit_patience = overfit_patience; self.general_stagnation_patience = plateau_patience + 2
        self.lr_reduce_factor = lr_reduce_factor; self.wd_increase_factor = wd_increase_factor
        self.min_lr = initial_lr * min_lr_factor; self.max_wd = self.initial_wd * max_wd_factor
        self.sam_rho_adjust_factor = sam_rho_adjust_factor; self.max_sam_rho = max_sam_rho
        self.best_val_loss = float('inf'); self.best_val_acc = 0.0; self.epoch_of_best_val_loss = -1; self.epoch_of_best_val_acc = -1
        self.train_losses, self.val_losses, self.train_accs, self.val_accs, self.lr_history, self.wd_history, self.adjustment_log = [],[],[],[],[],[],[]
        self.last_adjustment_epoch = -self.plateau_patience
    def update(self, train_loss, train_acc, val_loss, val_acc, optimizer, scheduler, is_sam_active_runtime, current_epoch): # Added scheduler
        self.train_losses.append(train_loss); self.val_losses.append(val_loss); self.train_accs.append(train_acc); self.val_accs.append(val_acc)
        opt_to_query = optimizer.base_optimizer if is_sam_active_runtime else optimizer
        self.lr_history.append(opt_to_query.param_groups[0]['lr']); self.wd_history.append(opt_to_query.param_groups[0]['weight_decay'])
        val_loss_improved = val_loss < self.best_val_loss; val_acc_improved = val_acc > self.best_val_acc
        made_adjustment_this_epoch = False
        if val_loss_improved: self.best_val_loss = val_loss; self.epoch_of_best_val_loss = current_epoch + 1; self.plateau_counter = 0
        else: self.plateau_counter += 1
        if val_acc_improved: self.best_val_acc = val_acc; self.epoch_of_best_val_acc = current_epoch + 1; self.no_improvement_counter = 0
        else:
            if (CHECKPOINT_MONITOR == 'val_acc' and not val_acc_improved) or \
               (CHECKPOINT_MONITOR != 'val_acc' and not val_loss_improved and not val_acc_improved): self.no_improvement_counter +=1
            elif CHECKPOINT_MONITOR == 'val_loss' and not val_loss_improved and not val_acc_improved: self.no_improvement_counter +=1
        is_overfitting = False
        if len(self.train_losses) >= 2 and len(self.val_losses) >=2: 
            train_loss_decreasing = self.train_losses[-1] < self.train_losses[-2]; val_loss_significantly_increasing = self.val_losses[-1] > (self.val_losses[-2] * 1.03)
            current_gap = self.val_losses[-1] - self.train_losses[-1]; prev_gap = self.val_losses[-2] - self.train_losses[-2]
            gap_widening_condition = current_gap > 0.01 and current_gap > (prev_gap + 0.02)
            min_accuracy_for_overfit_detection = 0.05 
            if self.val_accs[-1] >= min_accuracy_for_overfit_detection and train_loss_decreasing and (val_loss_significantly_increasing or gap_widening_condition):
                self.overfit_counter += 1; is_overfitting = True
            if not is_overfitting and (val_loss_improved or not (current_gap > 0.01 and current_gap > (prev_gap + 0.01))): self.overfit_counter = 0
        if current_epoch - self.last_adjustment_epoch < self.plateau_patience // 2 and self.last_adjustment_epoch >= 0: return False 
        current_optimizer_lr = opt_to_query.param_groups[0]['lr']; current_optimizer_wd = opt_to_query.param_groups[0]['weight_decay']
        if self.overfit_counter >= self.overfit_patience:
            new_lr = max(current_optimizer_lr * self.lr_reduce_factor, self.min_lr); new_wd = min(current_optimizer_wd * self.wd_increase_factor, self.max_wd) 
            if new_lr < current_optimizer_lr or new_wd > current_optimizer_wd:
                self._adjust_optimizer_params(optimizer, new_lr, new_wd, scheduler, is_sam_active_runtime) # Pass scheduler
                self.adjustment_log.append(f"Ep{current_epoch+1}: Overfit! LR->{new_lr:.2e}, WD->{new_wd:.2e}"); made_adjustment_this_epoch = True
            self.overfit_counter = 0; self.plateau_counter = 0; self.no_improvement_counter = 0; self.last_adjustment_epoch = current_epoch
            if is_sam_active_runtime and hasattr(optimizer, 'rho'): self._adjust_sam_parameters(optimizer, is_overfitting)
        elif self.plateau_counter >= self.plateau_patience or self.no_improvement_counter >= self.general_stagnation_patience:
            new_lr = max(current_optimizer_lr * self.lr_reduce_factor, self.min_lr); new_wd = current_optimizer_wd 
            if current_optimizer_lr <= self.min_lr * 2: new_wd = max(current_optimizer_wd * 0.9, self.initial_wd * 0.5) 
            if new_lr < current_optimizer_lr or new_wd < current_optimizer_wd :
                self._adjust_optimizer_params(optimizer, new_lr, new_wd, scheduler, is_sam_active_runtime) # Pass scheduler
                self.adjustment_log.append(f"Ep{current_epoch+1}: Plateau! LR->{new_lr:.2e}, WD->{new_wd:.2e}"); made_adjustment_this_epoch = True
            self.plateau_counter = 0; self.no_improvement_counter = 0; self.last_adjustment_epoch = current_epoch
        return made_adjustment_this_epoch
    def _adjust_optimizer_params(self, optimizer, new_lr, new_wd, scheduler, is_sam_active_runtime): # Added scheduler
        opt_to_adjust = optimizer.base_optimizer if is_sam_active_runtime else optimizer; lr_changed_for_optimizer = False
        for i, param_group in enumerate(opt_to_adjust.param_groups):
            if param_group['lr'] != new_lr: param_group['lr'] = new_lr; lr_changed_for_optimizer = True
            param_group['weight_decay'] = new_wd
        self.current_lr = new_lr; self.current_wd = new_wd
        if lr_changed_for_optimizer and scheduler: # Update scheduler's base_lrs
            target_scheduler = scheduler
            if isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR) and scheduler._schedulers: target_scheduler = scheduler._schedulers[-1]
            if hasattr(target_scheduler, 'base_lrs') and isinstance(target_scheduler.base_lrs, list):
                target_scheduler.base_lrs = [new_lr] * len(target_scheduler.base_lrs)
    def _adjust_sam_parameters(self, sam_optimizer, is_overfitting):
        current_rho = sam_optimizer.rho
        if is_overfitting and current_rho < self.max_sam_rho:
            new_rho = min(current_rho * self.sam_rho_adjust_factor, self.max_sam_rho)
            if new_rho > current_rho: sam_optimizer.rho = new_rho; self.adjustment_log.append(f"SAM rho -> {new_rho:.4f} (overfit)")
    def get_status_report(self):
        status = [f"Best Val Loss: {self.best_val_loss:.4f} (Ep {self.epoch_of_best_val_loss if self.epoch_of_best_val_loss > 0 else 'N/A'})",
                  f"Best Val Acc: {self.best_val_acc:.4f} (Ep {self.epoch_of_best_val_acc if self.epoch_of_best_val_acc > 0 else 'N/A'})"]
        if self.adjustment_log: status.append("Recent adjustments:"); status.extend([f"  - {adj}" for adj in self.adjustment_log[-min(3, len(self.adjustment_log)):]])
        return "\n".join(status)

# --- Knowledge Distillation --- (Copied as is, may need adaptation for CombinedModel teacher)
class DistillationLoss(nn.Module):
    def __init__(self, alpha=KD_ALPHA, temperature=KD_TEMPERATURE, base_criterion=nn.CrossEntropyLoss()):
        super().__init__(); self.alpha = alpha; self.T = temperature; self.base_criterion = base_criterion; self.KLDiv = nn.KLDivLoss(reduction='batchmean')
    def forward(self, student_outputs, teacher_outputs, labels):
        soft_teacher_log_probs = F.log_softmax(teacher_outputs / self.T, dim=1); soft_student_log_probs = F.log_softmax(student_outputs / self.T, dim=1)
        distillation_loss = self.KLDiv(soft_student_log_probs, soft_teacher_log_probs) * (self.alpha * self.T * self.T)
        student_loss = self.base_criterion(student_outputs, labels); total_loss = distillation_loss + (1. - self.alpha) * student_loss
        return total_loss

def train_student_model(student_model_name, student_base_save_path, student_swa_save_path, student_ema_save_path,
                        df_full_for_kd, image_dir_kd, num_classes_kd, stacking_teacher_components):
    print(f"\n{TermColors.HEADER}--- Knowledge Distillation with Stacking Teacher ---{TermColors.ENDC}")
    actual_stacking_model = None; stacking_scaler = None; feature_extractor_for_teacher = None
    base_mlp_models_for_teacher = []; teacher_loaded_successfully = False # This part assumes MLPs

    try: # Load Stacking Teacher (assumes MLP base models for stacker as per original logic)
        print(f"{TermColors.INFO}Loading Stacking Teacher components...{TermColors.ENDC}")
        stacking_package = joblib.load(stacking_teacher_components["stacking_model_path"])
        actual_stacking_model = stacking_package['model']; stacking_scaler = stacking_package.get('scaler') # Scaler might not exist
        if stacking_scaler is None: print(f"{TermColors.YELLOW}  Stacking scaler not found in joblib. Assuming no scaling needed or handled by model.{TermColors.ENDC}")

        # Feature Extractor for teacher (CombinedModel that generated features for MLPs)
        fe_names = stacking_teacher_components["feature_extractor_model_names"]
        fe_embed_size = stacking_teacher_components["feature_extractor_embedding_size"]
        fe_arc_m = stacking_teacher_components["feature_extractor_arcface_m"]
        fe_metric_learning = stacking_teacher_components["feature_extractor_metric_learning"]
        fe_ckpt_path = stacking_teacher_components.get("feature_extractor_checkpoint_path")

        feature_extractor_for_teacher = build_model(model_names_list=fe_names, num_classes_val=num_classes_kd, pretrained_val=PRETRAINED, 
                                                    embedding_val=fe_embed_size, arc_m_val=fe_arc_m, metric_learn_val=fe_metric_learning)
        if fe_ckpt_path and os.path.exists(fe_ckpt_path):
            fe_ckpt = torch.load(fe_ckpt_path, map_location=DEVICE)
            fe_state_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in fe_ckpt['state_dict'].items()}
            feature_extractor_for_teacher.load_state_dict(fe_state_dict, strict=False)
        feature_extractor_for_teacher = feature_extractor_for_teacher.to(DEVICE); feature_extractor_for_teacher.eval()

        # Base MLP Models for teacher
        # THIS IS THE PART THAT IS PROBLEMATIC IF THE TEACHER IS FROM FULL_TRAINING K-FOLD CombinedModels
        # The original script's `train_student_model` expects MLPs here.
        # If `stacking_teacher_components["base_mlp_model_paths"]` point to CombinedModel state_dicts,
        # `build_mlp_model` (which creates SimpleMLP) will fail to load them.
        # For this split, I'm keeping the original logic. User needs to adapt this if teacher is CombinedModel-based.
        base_mlp_input_size = stacking_teacher_components["base_mlp_model_input_size"] # This would be CombinedModel's embedding size
        expected_mlp_folds = stacking_teacher_components["n_folds_for_stacking"]
        
        # Attempt to load base models. If these are CombinedModels, this will likely fail with current build_mlp_model.
        print(f"{TermColors.YELLOW}KD Teacher Warning: Attempting to load base models for teacher. If these are CombinedModels from K-Fold, the current logic (expecting MLPs) might fail or be incorrect.{TermColors.ENDC}")
        from PlantRecognition.PlantRgognotionV2.scripts.PlantRecognitionV3 import build_mlp_model as build_original_mlp_model # Temporary import if needed for SimpleMLP
        
        for mlp_path in stacking_teacher_components["base_mlp_model_paths"]:
            if not os.path.exists(mlp_path): continue
            # This assumes base_mlp_input_size is correct for SimpleMLP
            # And that mlp_path contains a SimpleMLP state_dict
            try:
                # If mlp_path actually points to a CombinedModel state_dict, this will error.
                # We need a way to know if the "base_mlp_model_paths" are for MLPs or CombinedModels.
                # For now, sticking to the original script's assumption of MLPs for the KD teacher's base.
                mlp = build_original_mlp_model(input_size=base_mlp_input_size, num_classes=num_classes_kd) # Using original SimpleMLP builder
                mlp.load_state_dict(torch.load(mlp_path, map_location=DEVICE))
                mlp = mlp.to(DEVICE); mlp.eval()
                base_mlp_models_for_teacher.append(mlp)
            except Exception as e_load_base_kd:
                print(f"{TermColors.RED}  Error loading base model '{mlp_path}' for KD teacher: {e_load_base_kd}. This model might not be an MLP or input size mismatch.{TermColors.ENDC}")

        if len(base_mlp_models_for_teacher) == expected_mlp_folds: teacher_loaded_successfully = True
        else: print(f"{TermColors.RED}  Failed to load all base models for teacher. Expected {expected_mlp_folds}, Got {len(base_mlp_models_for_teacher)}.{TermColors.ENDC}")
    except Exception as e: print(f"{TermColors.RED}Failed to load teacher components: {e}. Skipping KD.{TermColors.ENDC}"); traceback.print_exc()
    if not teacher_loaded_successfully: print(f"{TermColors.RED}Stacking teacher setup failed. Aborting KD.{TermColors.ENDC}"); return

    # Student Model Setup (Image-based CombinedModel)
    try:
        student_model = build_model(model_names_list=[student_model_name], num_classes_val=num_classes_kd, pretrained_val=True,
                                    dropout_val=KD_STUDENT_DROPOUT, embedding_val=KD_STUDENT_EMBEDDING_SIZE, metric_learn_val='None')
        student_model = student_model.to(DEVICE)
    except Exception as e: print(f"{TermColors.RED}Failed to build student: {e}. Skipping KD.{TermColors.ENDC}"); return

    # Dataloaders for Student
    try:
        df_kd_train, df_kd_val = train_test_split(df_full_for_kd, test_size=0.2, random_state=SEED + 42, stratify=df_full_for_kd['label'])
        train_tf_kd, val_tf_kd = get_transforms(image_size=KD_STUDENT_IMAGE_SIZE)
        train_ds_kd = PlantDataset(df_kd_train, image_dir_kd, train_tf_kd, label_encoder, False, KD_STUDENT_IMAGE_SIZE)
        val_ds_kd = PlantDataset(df_kd_val, image_dir_kd, val_tf_kd, label_encoder, False, KD_STUDENT_IMAGE_SIZE)
        if not train_ds_kd or not val_ds_kd or len(train_ds_kd)==0 or len(val_ds_kd)==0: print(f"{TermColors.RED}KD Dataset empty. Skip KD.{TermColors.ENDC}"); return
        kd_sampler = None
        if IMBALANCE_STRATEGY == 'WeightedSampler' and CLASS_WEIGHTS is not None:
            kd_train_labels_list = train_ds_kd.get_labels()
            if kd_train_labels_list:
                kd_class_sample_count = np.array([kd_train_labels_list.count(l) for l in range(num_classes_kd)]); kd_class_sample_count = np.maximum(kd_class_sample_count, 1)
                kd_weight = 1. / kd_class_sample_count; kd_samples_weight = torch.from_numpy(np.array([kd_weight[t] for t in kd_train_labels_list])).double()
                kd_sampler = WeightedRandomSampler(kd_samples_weight, len(kd_samples_weight))
        train_loader_kd = DataLoader(train_ds_kd, KD_BATCH_SIZE, sampler=kd_sampler, shuffle=(kd_sampler is None), num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
        val_loader_kd = DataLoader(val_ds_kd, KD_BATCH_SIZE * 2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    except Exception as e: print(f"{TermColors.RED}Error KD dataloaders: {e}. Skip KD.{TermColors.ENDC}"); return

    student_base_criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING if LOSS_TYPE == 'CrossEntropy' else 0.0)
    criterion_kd = DistillationLoss(alpha=KD_ALPHA, temperature=KD_TEMPERATURE, base_criterion=student_base_criterion)
    optimizer_kd = optim.AdamW(student_model.parameters(), lr=KD_LR, weight_decay=WEIGHT_DECAY)
    scaler_kd = torch.amp.GradScaler('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda'))
    scheduler_kd = CosineAnnealingLR(optimizer_kd, T_max=KD_EPOCHS, eta_min=KD_LR * 0.01)

    swa_student_model = None; swa_student_scheduler = None
    kd_student_swa_start_epoch = int(KD_EPOCHS * KD_STUDENT_SWA_START_EPOCH_FACTOR)
    if KD_STUDENT_USE_SWA:
        swa_student_model = AveragedModel(student_model, avg_fn=ema_avg_fn if KD_STUDENT_USE_EMA else None)
        if KD_STUDENT_USE_EMA: # Tag params for correct EMA decay
            for p_swa_stud in (swa_student_model.module.parameters() if hasattr(swa_student_model, 'module') else swa_student_model.parameters()): p_swa_stud._is_kd_student_ema_param = True
        swa_student_scheduler = SWALR(optimizer_kd, swa_lr=(KD_LR * KD_STUDENT_SWA_LR_FACTOR), anneal_epochs=KD_STUDENT_SWA_ANNEAL_EPOCHS)
    ema_student_model = None
    if KD_STUDENT_USE_EMA:
        ema_student_model = AveragedModel(student_model, avg_fn=ema_avg_fn)
        for p_ema_stud in (ema_student_model.module.parameters() if hasattr(ema_student_model, 'module') else ema_student_model.parameters()): p_ema_stud._is_kd_student_ema_param = True
    
    print(f"{TermColors.CYAN}Starting KD Student Training ({KD_EPOCHS} epochs)...{TermColors.ENDC}")
    best_kd_val_acc = 0.0
    for epoch in range(KD_EPOCHS):
        if stop_requested: break
        student_model.train(); running_loss_kd = 0.0; total_samples_kd = 0
        progress_bar_kd = tqdm(train_loader_kd, desc=f"KD Student Train E{epoch+1}", leave=False)
        for batch_idx, (inputs, labels) in enumerate(progress_bar_kd):
            if stop_requested: break
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE); batch_size = inputs.size(0)
            teacher_outputs_logits = None
            with torch.no_grad(): # Teacher inference
                img_features_for_teacher = feature_extractor_for_teacher(inputs, return_embedding=True)
                mlp_predictions_for_stacker_list = [F.softmax(mlp_model_teacher(img_features_for_teacher), dim=1) for mlp_model_teacher in base_mlp_models_for_teacher]
                stacker_input_features_tensor = torch.cat(mlp_predictions_for_stacker_list, dim=1)
                stacker_input_np = stacker_input_features_tensor.cpu().numpy()
                scaled_stacker_input_np = stacking_scaler.transform(stacker_input_np) if stacking_scaler else stacker_input_np
                teacher_output_probs_np = actual_stacking_model.predict_proba(scaled_stacker_input_np)
                teacher_outputs_logits = torch.log(torch.tensor(teacher_output_probs_np, dtype=torch.float32).to(DEVICE) + 1e-9)
            if teacher_outputs_logits is None: continue
            with torch.amp.autocast('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda')):
                student_outputs_logits = student_model(inputs)
                loss = criterion_kd(student_outputs_logits, teacher_outputs_logits, labels)
            optimizer_kd.zero_grad(); scaler_kd.scale(loss).backward(); scaler_kd.step(optimizer_kd); scaler_kd.update()
            if KD_STUDENT_USE_EMA and ema_student_model: ema_student_model.update_parameters(student_model)
            if not torch.isnan(loss): running_loss_kd += loss.item() * batch_size; total_samples_kd += batch_size
            if batch_idx % 20 == 0 or batch_idx == len(train_loader_kd)-1: progress_bar_kd.set_postfix(loss=f"{loss.item():.4f}")
        if stop_requested: break
        epoch_loss_kd = running_loss_kd / total_samples_kd if total_samples_kd > 0 else float('nan')
        
        # Validation
        student_model.eval(); 
        if swa_student_model: swa_student_model.eval()
        if ema_student_model: ema_student_model.eval()
        val_models_to_eval_kd = {'base_student': student_model}
        if KD_STUDENT_USE_SWA and swa_student_model and epoch >= kd_student_swa_start_epoch : val_models_to_eval_kd['swa_student'] = swa_student_model
        if KD_STUDENT_USE_EMA and ema_student_model: val_models_to_eval_kd['ema_student'] = ema_student_model
        current_best_val_acc_this_epoch = 0.0; best_model_key_this_epoch = 'base_student'

        for model_key, current_eval_model in val_models_to_eval_kd.items():
            current_eval_model.eval(); running_val_loss_kd = 0.0; total_val_samples_kd = 0; all_preds_kd, all_labels_kd = [], []
            with torch.no_grad():
                for inputs_val, labels_val in tqdm(val_loader_kd, desc=f"KD Val E{epoch+1} ({model_key})", leave=False):
                    inputs_val, labels_val = inputs_val.to(DEVICE), labels_val.to(DEVICE)
                    with torch.amp.autocast('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda')): outputs_val = current_eval_model(inputs_val); val_loss = F.cross_entropy(outputs_val, labels_val) 
                    if not torch.isnan(val_loss):
                        running_val_loss_kd += val_loss.item() * inputs_val.size(0); total_val_samples_kd += inputs_val.size(0)
                        all_preds_kd.append(torch.argmax(outputs_val, dim=1).cpu()); all_labels_kd.append(labels_val.cpu())
            epoch_val_loss_kd = running_val_loss_kd / total_val_samples_kd if total_val_samples_kd > 0 else float('nan')
            epoch_val_acc_kd = (torch.cat(all_preds_kd) == torch.cat(all_labels_kd)).sum().item() / total_val_samples_kd if all_preds_kd and total_val_samples_kd > 0 else 0.0
            print(f"  KD Student Val ({model_key}) - Ep {epoch+1}: Loss: {epoch_val_loss_kd:.4f}, Acc: {epoch_val_acc_kd:.4f}")
            if epoch_val_acc_kd > current_best_val_acc_this_epoch: current_best_val_acc_this_epoch = epoch_val_acc_kd; best_model_key_this_epoch = model_key
        
        if current_best_val_acc_this_epoch > best_kd_val_acc:
            best_kd_val_acc = current_best_val_acc_this_epoch
            model_to_save_state = None; save_path_for_best = student_base_save_path
            if best_model_key_this_epoch == 'base_student': model_to_save_state = student_model.state_dict()
            elif best_model_key_this_epoch == 'swa_student' and swa_student_model: model_to_save_state = swa_student_model.module.state_dict(); save_path_for_best = student_swa_save_path
            elif best_model_key_this_epoch == 'ema_student' and ema_student_model: model_to_save_state = ema_student_model.module.state_dict(); save_path_for_best = student_ema_save_path
            if model_to_save_state: torch.save(model_to_save_state, save_path_for_best); print(f"  Saved best student ({best_model_key_this_epoch}) to {save_path_for_best}")
        
        if KD_STUDENT_USE_SWA and swa_student_model and epoch >= kd_student_swa_start_epoch:
            swa_student_model.update_parameters(student_model)
            if swa_student_scheduler: swa_student_scheduler.step()
        elif scheduler_kd: scheduler_kd.step()
        gc.collect(); torch.cuda.empty_cache()
    
    if KD_STUDENT_USE_SWA and swa_student_model and not stop_requested:
        try:
            bn_loader_kd_final = DataLoader(train_ds_kd, KD_BATCH_SIZE*2, shuffle=True, num_workers=NUM_WORKERS)
            swa_student_model.to(DEVICE); torch.optim.swa_utils.update_bn(bn_loader_kd_final, swa_student_model, device=DEVICE)
            torch.save(swa_student_model.module.state_dict(), student_swa_save_path)
            del bn_loader_kd_final
        except Exception as e_swa_final_kd: print(f"{TermColors.RED}KD Student FINAL SWA Error: {e_swa_final_kd}{TermColors.ENDC}")
    if KD_STUDENT_USE_EMA and ema_student_model and not stop_requested:
        try: ema_student_model.to(DEVICE); torch.save(ema_student_model.module.state_dict(), student_ema_save_path)
        except Exception as e_ema_final_kd: print(f"{TermColors.RED}KD Student FINAL EMA Error: {e_ema_final_kd}{TermColors.ENDC}")
    if not os.path.exists(student_base_save_path) and not stop_requested : torch.save(student_model.state_dict(), student_base_save_path)
    print(f"{TermColors.OKGREEN}KD finished. Best student val_acc: {best_kd_val_acc:.4f}{TermColors.ENDC}")
    del student_model, train_loader_kd, val_loader_kd, train_ds_kd, val_ds_kd, actual_stacking_model, stacking_scaler, feature_extractor_for_teacher, base_mlp_models_for_teacher
    if swa_student_model: del swa_student_model
    if ema_student_model: del ema_student_model
    gc.collect(); torch.cuda.empty_cache()

# --- Main Execution Loop (Full Image Training) ---
def main_full_training_loop_entry(df_full_data, num_classes_val, label_encoder_val, class_names_val, class_frequencies_val, class_priors_val, class_weights_val):
    global NUM_CLASSES, label_encoder, class_names, CLASS_FREQUENCIES, CLASS_PRIORS, CLASS_WEIGHTS, CURRENT_IMAGE_SIZE
    NUM_CLASSES = num_classes_val; label_encoder = label_encoder_val; class_names = class_names_val
    CLASS_FREQUENCIES = class_frequencies_val; CLASS_PRIORS = class_priors_val; CLASS_WEIGHTS = class_weights_val
    
    print(f"\n{TermColors.HEADER}--- Main K-Fold Cross-Validation ({N_FOLDS} Folds) ---{TermColors.ENDC}")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_results = defaultdict(list)
    oof_preds_array = np.full((len(df_full_data), NUM_CLASSES), np.nan, dtype=np.float32)
    oof_labels_array = np.full(len(df_full_data), -1, dtype=np.int32)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_full_data, df_full_data['label'])):
        if stop_requested: break
        print(f"\n{TermColors.HEADER}===== Starting Fold {fold+1}/{N_FOLDS} ====={TermColors.ENDC}")
        train_df = df_full_data.iloc[train_idx]; val_df = df_full_data.iloc[val_idx]
        model = optimizer = scheduler = scaler = criterion = swa_model = ema_model = None; gc.collect(); torch.cuda.empty_cache()
        try:
            model = build_model(); model = model.to(DEVICE)
            if USE_TORCH_COMPILE and hasattr(torch, 'compile') and int(torch.__version__.split('.')[0]) >= 2:
                try: model = torch.compile(model, mode='default')
                except Exception as compile_e: print(f"{TermColors.RED}torch.compile() failed: {compile_e}.{TermColors.ENDC}")
            criterion = get_criterion(); optimizer = get_optimizer(model)
            scaler = torch.amp.GradScaler('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda'))
            if USE_SWA: swa_model = AveragedModel(model, avg_fn=ema_avg_fn if USE_EMA else None)
            if USE_EMA: ema_model = AveragedModel(model, avg_fn=ema_avg_fn)
        except Exception as e: print(f"{TermColors.RED}Fold {fold+1} Setup Error: {e}{TermColors.ENDC}"); continue
        
        start_glob_ep, start_stg_idx, start_stg_ep, best_metric, _, loaded_size, _ = load_checkpoint(fold, model, optimizer, None, scaler, "latest_checkpoint.pth.tar")
        if best_metric == (float('-inf') if CHECKPOINT_MODE == 'max' else float('inf')): # If latest was not informative
            start_glob_ep, start_stg_idx, start_stg_ep, best_metric, _, loaded_size, _ = load_checkpoint(fold, model, optimizer, None, scaler, "best_model.pth.tar")

        fold_log_dir = os.path.join(BASE_LOG_DIR, f"fold_{fold}"); writer = SummaryWriter(log_dir=fold_log_dir)
        global_epoch_counter = start_glob_ep; fold_stop_requested = False
        fold_best_val_loss = float('inf'); fold_best_val_acc = 0.0; epochs_without_improvement = 0 
        best_metric_for_early_stopping = best_metric 
        current_auto_config = AutoTrainingConfig() if USE_AUTO_TRAIN_CONFIG else None

        for stage_idx_loop, (stage_epochs, stage_image_size) in enumerate(PROGRESSIVE_RESIZING_STAGES):
            if fold_stop_requested or stop_requested: break
            if stage_idx_loop < start_stg_idx: continue
            current_stage_start_epoch = start_stg_ep if stage_idx_loop == start_stg_idx else 0
            if current_stage_start_epoch >= stage_epochs: continue
            print(f"\n{TermColors.MAGENTA}Fold {fold+1} Stage {stage_idx_loop+1}/{len(PROGRESSIVE_RESIZING_STAGES)}: {stage_epochs} E @ {stage_image_size}{TermColors.ENDC}")
            CURRENT_IMAGE_SIZE = stage_image_size
            if loaded_size and stage_idx_loop == start_stg_idx and loaded_size != CURRENT_IMAGE_SIZE: print(f"{TermColors.CRITICAL}Img size mismatch! Ckpt {loaded_size}!=Stage {CURRENT_IMAGE_SIZE}. Exit.{TermColors.ENDC}"); fold_stop_requested=True; break
            
            train_transform, val_transform = get_transforms(image_size=CURRENT_IMAGE_SIZE)
            train_loader, val_loader, err_loader = None, None, None; gc.collect()
            try:
                train_ds = PlantDataset(train_df, IMAGE_DIR, train_transform, None, False, CURRENT_IMAGE_SIZE)
                val_ds = PlantDataset(val_df, IMAGE_DIR, val_transform, None, True, CURRENT_IMAGE_SIZE) # include_paths for OOF
                err_ds = PlantDataset(val_df, IMAGE_DIR, val_transform, None, True, CURRENT_IMAGE_SIZE) # include_paths for error log
                if not train_ds or not val_ds or len(train_ds)==0 or len(val_ds)==0 : print(f"{TermColors.RED}Dataset empty. Skip fold.{TermColors.ENDC}"); fold_stop_requested=True; break
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
            except Exception as e: print(f"{TermColors.RED}Dataloader Error: {e}{TermColors.ENDC}"); fold_stop_requested=True; break
            
            current_t_0_val = stage_epochs if SCHEDULER_TYPE == 'CosineWarmRestarts' else T_0
            scheduler = get_scheduler(optimizer, total_epochs_val=stage_epochs, t_0_val=current_t_0_val) # Pass correct total_epochs for stage
            if stage_idx_loop == start_stg_idx and current_stage_start_epoch > 0: # Reload scheduler state
                 ckpt_path_sched = os.path.join(BASE_CHECKPOINT_DIR, f"fold_{fold}", "latest_checkpoint.pth.tar")
                 if scheduler and os.path.isfile(ckpt_path_sched):
                     ckpt_sched = torch.load(ckpt_path_sched, map_location=DEVICE)
                     if 'scheduler' in ckpt_sched and ckpt_sched['scheduler']:
                         try: scheduler.load_state_dict(ckpt_sched['scheduler'])
                         except Exception as e_sch: print(f"{TermColors.YELLOW}Scheduler reload failed: {e_sch}.{TermColors.ENDC}")
            
            globals()['swa_start_epoch_stage'] = max(0, int(stage_epochs * SWA_START_EPOCH_GLOBAL_FACTOR)) # SWA start for this stage
            swa_scheduler = SWALR(optimizer, swa_lr=(LEARNING_RATE*SWA_LR_FACTOR), anneal_epochs=SWA_ANNEAL_EPOCHS) if USE_SWA and swa_model else None

            for stage_epoch_loop in range(current_stage_start_epoch, stage_epochs):
                if fold_stop_requested or stop_requested: break
                print(f"\n{TermColors.CYAN}Fold {fold+1} GlobEp {global_epoch_counter+1}/{TOTAL_EPOCHS_PER_FOLD} (Stg {stage_idx_loop+1}: Ep {stage_epoch_loop+1}/{stage_epochs}){TermColors.ENDC}")
                train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, scheduler, global_epoch_counter, stage_idx_loop, stage_epoch_loop, stage_epochs, DEVICE, writer, NUM_CLASSES, ema_model, fold_num=fold+1)
                if train_loss is None: fold_stop_requested = True; break
                val_loss, val_acc, current_oof_preds, current_oof_indices = validate_one_epoch(model, val_loader, criterion, DEVICE, global_epoch_counter, writer, NUM_CLASSES, scheduler, swa_model, ema_model, True, fold+1)
                if val_loss is None: fold_stop_requested = True; break
                
                if current_auto_config:
                    is_sam_active_runtime = hasattr(optimizer, 'base_optimizer') and SAM_AVAILABLE and USE_SAM
                    current_auto_config.update(train_loss, train_acc, val_loss, val_acc, optimizer, scheduler, is_sam_active_runtime, global_epoch_counter) # Pass scheduler
                    if (global_epoch_counter+1)%5==0: print(f"\n{TermColors.CYAN}AutoConfig (GlobEp {global_epoch_counter+1}):\n{current_auto_config.get_status_report()}{TermColors.ENDC}\n")
                print(f"F{fold+1} GEp {global_epoch_counter+1}: Tr L={train_loss:.4f} A={train_acc:.4f} | Val L={val_loss:.4f} A={val_acc:.4f}")

                if USE_SWA and swa_model and stage_epoch_loop >= globals()['swa_start_epoch_stage']:
                    swa_model.update_parameters(model)
                    if swa_scheduler: swa_scheduler.step()
                
                current_metric_for_stopping = val_loss if CHECKPOINT_MONITOR == 'val_loss' else val_acc
                improved_early_stop_metric = (CHECKPOINT_MODE == 'min' and current_metric_for_stopping < best_metric_for_early_stopping) or \
                                            (CHECKPOINT_MODE == 'max' and current_metric_for_stopping > best_metric_for_early_stopping)
                if improved_early_stop_metric: best_metric_for_early_stopping = current_metric_for_stopping; epochs_without_improvement = 0
                else: epochs_without_improvement += 1
                if epochs_without_improvement >= EARLY_STOPPING_PATIENCE: print(f"{TermColors.WARNING}Early stop F{fold+1}.{TermColors.ENDC}"); fold_stop_requested=True; break 

                current_metric_for_checkpoint = val_acc if CHECKPOINT_MONITOR == 'val_acc' else val_loss
                is_best = (CHECKPOINT_MODE == 'max' and current_metric_for_checkpoint > best_metric) or \
                          (CHECKPOINT_MODE == 'min' and current_metric_for_checkpoint < best_metric)
                if is_best:
                    best_metric = current_metric_for_checkpoint; fold_best_val_loss = val_loss; fold_best_val_acc = val_acc
                    print(f"{TermColors.OKGREEN}F{fold+1} New Best {CHECKPOINT_MONITOR}: {best_metric:.4f}. Saving...{TermColors.ENDC}")
                    save_checkpoint(fold, global_epoch_counter+1, stage_idx_loop, stage_epoch_loop+1, model, optimizer, scheduler, scaler, best_metric, "best_model.pth.tar")
                    save_model(fold, model, "best_model_state_dict.pth")
                    if USE_EMA and ema_model: save_model(fold, ema_model, "best_ema_model_state_dict.pth")
                    if USE_SWA and swa_model and stage_epoch_loop >= globals()['swa_start_epoch_stage']: save_model(fold, swa_model, "best_swa_model_state_dict.pth")
                    if current_oof_preds is not None and current_oof_indices is not None:
                        for i, original_idx_val in enumerate(current_oof_indices):
                            if 0 <= original_idx_val < len(oof_preds_array): 
                                oof_preds_array[original_idx_val] = current_oof_preds[i]
                                oof_labels_array[original_idx_val] = df_full_data.loc[original_idx_val, 'label']
                
                save_checkpoint(fold, global_epoch_counter+1, stage_idx_loop, stage_epoch_loop+1, model, optimizer, scheduler, scaler, best_metric, "latest_checkpoint.pth.tar")
                if LOG_MISCLASSIFIED_IMAGES and ((global_epoch_counter+1)%5==0 or is_best): log_misclassified(fold, model, err_loader, criterion, DEVICE, global_epoch_counter+1, writer, NUM_CLASSES)
                global_epoch_counter += 1; gc.collect(); torch.cuda.empty_cache(); start_stg_ep = 0 
            if fold_stop_requested or stop_requested: break
            del train_loader, val_loader, err_loader, train_ds, val_ds, err_ds, train_transform, val_transform, scheduler, swa_scheduler; scheduler=None; gc.collect(); torch.cuda.empty_cache()
        
        if fold_stop_requested or stop_requested: save_checkpoint(fold, global_epoch_counter, stage_idx_loop, stage_epoch_loop if 'stage_epoch_loop' in locals() else 0 , model, optimizer, scheduler, scaler, best_metric, "interrupted_checkpoint.pth.tar"); break
        
        if USE_SWA and swa_model and global_epoch_counter >= int(TOTAL_EPOCHS_PER_FOLD * SWA_START_EPOCH_GLOBAL_FACTOR):
            print(f"{TermColors.CYAN}F{fold+1} SWA BN update...{TermColors.ENDC}")
            final_stage_size_swa = PROGRESSIVE_RESIZING_STAGES[-1][1]; final_train_tf_swa, _ = get_transforms(final_stage_size_swa)
            try:
                final_train_ds_bn_swa = PlantDataset(train_df, IMAGE_DIR, final_train_tf_swa, None, False, final_stage_size_swa)
                if len(final_train_ds_bn_swa) > 0:
                    bn_loader_swa = DataLoader(final_train_ds_bn_swa, BATCH_SIZE*2, shuffle=True, num_workers=NUM_WORKERS)
                    torch.optim.swa_utils.update_bn(bn_loader_swa, swa_model, device=DEVICE)
                    _, final_val_tf_swa = get_transforms(final_stage_size_swa)
                    final_val_ds_eval_swa = PlantDataset(val_df, IMAGE_DIR, final_val_tf_swa, None, False, final_stage_size_swa)
                    if len(final_val_ds_eval_swa) > 0:
                        final_val_loader_swa = DataLoader(final_val_ds_eval_swa, BATCH_SIZE*2, shuffle=False, num_workers=NUM_WORKERS)
                        swa_val_loss, swa_val_acc, _, _ = validate_one_epoch(swa_model, final_val_loader_swa, criterion, DEVICE, global_epoch_counter, writer, NUM_CLASSES, fold_num=f"{fold+1}-SWA")
                        print(f"F{fold+1} Final SWA Val L={swa_val_loss:.4f}, A={swa_val_acc:.4f}"); fold_results['swa_acc'].append(swa_val_acc)
                        save_model(fold, swa_model, "final_swa_model_state_dict.pth")
                    del final_val_ds_eval_swa, final_val_loader_swa
                del final_train_ds_bn_swa, bn_loader_swa
            except Exception as e_swa: print(f"{TermColors.RED}SWA BN/Eval Error: {e_swa}{TermColors.ENDC}")
        save_model(fold, model, "final_model_state_dict.pth")
        if USE_EMA and ema_model: save_model(fold, ema_model, "final_ema_model_state_dict.pth")
        writer.close(); fold_results['best_metric'].append(best_metric); fold_results['best_val_loss'].append(fold_best_val_loss); fold_results['best_val_acc'].append(fold_best_val_acc)
        del model, optimizer, scaler, criterion, swa_model, ema_model, train_df, val_df, writer; gc.collect(); torch.cuda.empty_cache()

    print(f"\n{TermColors.HEADER}===== CV Finished (Full Training) ====={TermColors.ENDC}")
    if not stop_requested:
        avg_best_acc = np.mean(fold_results['best_val_acc']) if fold_results['best_val_acc'] else 0.0
        print(f"Avg Best Val Acc: {avg_best_acc:.4f} +/- {np.std(fold_results['best_val_acc'] if fold_results['best_val_acc'] else [0.0]):.4f}")
        if fold_results['swa_acc']: print(f"Avg Final SWA Val Acc: {np.mean(fold_results['swa_acc']):.4f} +/- {np.std(fold_results['swa_acc']):.4f}")
        if RUN_STACKING:
            valid_oof_indices = np.where(oof_labels_array != -1)[0]
            if len(valid_oof_indices) < len(df_full_data) * 0.5: print(f"{TermColors.YELLOW}Warn: OOF only for {len(valid_oof_indices)}/{len(df_full_data)} samples.{TermColors.ENDC}")
            if len(valid_oof_indices) > 0:
                final_oof_preds = oof_preds_array[valid_oof_indices]; final_oof_labels = oof_labels_array[valid_oof_indices]
                if len(final_oof_preds) > 0 and len(final_oof_preds) == len(final_oof_labels):
                    np.savez_compressed(STACKING_OOF_PREDS_PATH, preds=final_oof_preds, labels=final_oof_labels); print(f"OOF preds saved: {STACKING_OOF_PREDS_PATH}")
                    train_stacking_meta_model(final_oof_preds, final_oof_labels, STACKING_META_MODEL_PATH)
                else: print(f"{TermColors.RED}Error OOF stacking data. Skip.{TermColors.ENDC}")
            else: print(f"{TermColors.RED}No valid OOF for Stacking. Skip.{TermColors.ENDC}")
    else: print(f"{TermColors.YELLOW}Training interrupted. Stacking/KD skipped.{TermColors.ENDC}"); 
    del oof_preds_array, oof_labels_array; gc.collect()
    return not stop_requested

def main():
    global stop_requested, label_encoder, class_names, NUM_CLASSES, CLASS_FREQUENCIES, CLASS_P…