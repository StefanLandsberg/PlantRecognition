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
FEATURES_NPZ_PATH = os.path.join(DATA_DIR, "extracted_features_v3.npz") # Path to save/load features
FEATURE_EXTRACTION_IMAGE_SIZE_CONFIG = (512, 512) # Image size for feature extraction
FEATURE_EXTRACTOR_BATCH_SIZE = 64 # Batch size for extracting features
FEATURE_EXTRACTOR_CHECKPOINT_PATH = None # Optional: Path to a specific checkpoint for the CombinedModel feature extractor

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
MODEL_NAMES = ["tf_efficientnetv2_l_in21ft1k", "convnext_large_in22ft1k"] # Dual model for feature extraction
DROP_PATH_RATE = 0.1; PRETRAINED = True; NUM_CLASSES = -1 # NUM_CLASSES will be updated
EMBEDDING_SIZE = 2048; GLOBAL_POOLING = 'avg' # DROPOUT_RATE for feature extractor is 0 (set in extract_all_features)
ARCFACE_S = 15.0 
ARCFACE_M = 0.25 

# --- MLP Model Config ---
MLP_HIDDEN_DIMS = [1024, 512] # Hidden layer dimensions for the MLP
MLP_DROPOUT_RATE = 0.5
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
MLP_USE_SAM = True
MLP_SAM_RHO = 0.007
MLP_SAM_ADAPTIVE = True

# --- MLP Hyperparameter Optimization (HPO) Config ---
MLP_DO_HPO = True
MLP_HPO_N_TRIALS = 15 if not DEBUG_MODE else 3
MLP_HPO_EPOCHS = 30 if not DEBUG_MODE else 2
MLP_HPO_PATIENCE = 5 if not DEBUG_MODE else 2
MLP_HPO_DATA_SPLIT_RATIO = 0.20
MLP_HPO_INTERNAL_VAL_SPLIT = 0.25
MLP_HPO_STUDY_DB_PATH = os.path.join(BASE_CHECKPOINT_DIR, "mlp_hpo_study.db") # Path for HPO study database
MLP_HPO_LOAD_BEST_ONLY = False  # If True, only load best from DB, don't run new trials unless DB is empty/invalid
                                # Flase, do HPO trails then load best from DB

# --- Loss Function & Imbalance Handling ---
LOSS_TYPE = 'CrossEntropy' # FocalLoss or CrossEntropy
LABEL_SMOOTHING = 0.1 # Applied if not using ArcFace and loss is CrossEntropy
FOCAL_ALPHA = 0.25; FOCAL_GAMMA = 2.0
IMBALANCE_STRATEGY = 'WeightedSampler' # Options: 'None', 'WeightedLoss', 'WeightedSampler', 'LogitAdjust'
LOGIT_ADJUSTMENT_TAU = 1.0
CLASS_FREQUENCIES = None; CLASS_PRIORS = None; CLASS_WEIGHTS = None

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
STACKING_DO_HPO = True
STACKING_HPO_CV_FOLDS = 3 if not DEBUG_MODE else 2
STACKING_LGBM_PARAM_GRID = {
    'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [20, 31, 40], 'max_depth': [-1, 10, 15],
    'min_child_samples': [15, 20, 30], 'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
}

# --- Knowledge Distillation Config ---
RUN_KNOWLEDGE_DISTILLATION = True
KD_STUDENT_MODEL_NAME = "mobilenetv3_small_100" # Student model (image-based)
KD_STUDENT_IMAGE_SIZE = (224, 224)
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
        self.image_size = image_size if image_size else FEATURE_EXTRACTION_IMAGE_SIZE_CONFIG # Fallback
        self.image_data = []
        
        required_cols = ['scientificName', 'id', 'label']
        if not all(col in self.input_df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in self.input_df.columns]
            print(f"{TermColors.RED}PlantDataset input missing: {missing}. Found: {self.input_df.columns.tolist()}{TermColors.ENDC}"); self.dataframe = pd.DataFrame(); return

        image_file_lookup = defaultdict(list)
        if not os.path.isdir(self.image_dir):
            print(f"{TermColors.RED}Image dir not found: {self.image_dir}{TermColors.ENDC}"); self.dataframe = pd.DataFrame(); return
            
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
        
        self.input_df['original_index'] = self.input_df.index
        for _, row in tqdm(self.input_df.iterrows(), total=len(self.input_df), desc="Matching CSV to images", leave=False, disable=DEBUG_MODE):
            try:
                species_name_csv = str(row['scientificName']); obs_id_csv = str(row['id']); label = row['label']
                original_idx_in_df = row['original_index']
                species_dir_name_csv_derived = species_name_csv.replace(' ', '_').replace('/', '_').replace('\\', '_')
                found_files = image_file_lookup.get((species_dir_name_csv_derived, obs_id_csv), [])
                for full_path in found_files:
                    self.image_data.append({'scientificName': species_name_csv, 'label': label, 'image_path': full_path, 'original_index': original_idx_in_df})
            except Exception as e: print(f"{TermColors.RED}Error processing CSV row (ID: {row.get('id', 'N/A')}): {e}{TermColors.ENDC}")
        
        self.dataframe = pd.DataFrame(self.image_data)
        if len(self.dataframe) == 0 and len(self.input_df) > 0: print(f"{TermColors.RED}Found 0 image files. Check paths/IDs.{TermColors.ENDC}")
        else: print(f"{TermColors.INFO}PlantDataset: Matched {len(self.dataframe)} images from {len(self.input_df)} CSV rows.{TermColors.ENDC}")

    def __len__(self): return len(self.dataframe)
    def get_labels(self): return self.dataframe['label'].tolist() if 'label' in self.dataframe.columns else []

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
    def __getitem__(self, idx): return self.features[idx], self.labels[idx], self.original_indices[idx]
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
    def __init__(self, input_size, hidden_dims, num_classes, dropout_rate=MLP_DROPOUT_RATE, use_arcface=MLP_USE_ARCFACE, arcface_s=ARCFACE_S, arcface_m=ARCFACE_M):
        super().__init__(); self.use_arcface = use_arcface
        layers = []; current_dim = input_size
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(current_dim, h_dim), nn.BatchNorm1d(h_dim), nn.ReLU(inplace=True), nn.Dropout(dropout_rate)])
            current_dim = h_dim
        self.hidden_layers = nn.Sequential(*layers)
        if self.use_arcface: self.metric_fc = ArcFace(current_dim, num_classes, s=arcface_s, m=arcface_m)
        else: self.metric_fc = nn.Linear(current_dim, num_classes)

    def forward(self, x, labels=None):
        x = self.hidden_layers(x)
        if self.use_arcface:
            x = self.metric_fc(x, labels) # ArcFace requires labels during training
        else: x = self.metric_fc(x)
        return x

def build_feature_extractor_model(num_classes_fe): # For feature extraction
    return CombinedModel(MODEL_NAMES, num_classes_fe, PRETRAINED, GLOBAL_POOLING, 0.0, EMBEDDING_SIZE, DROP_PATH_RATE, ARCFACE_S, ARCFACE_M, 'None')

def build_mlp_model(input_size, num_classes_mlp): # For MLP training
    return SimpleMLP(input_size, MLP_HIDDEN_DIMS, num_classes_mlp, MLP_DROPOUT_RATE, MLP_USE_ARCFACE, ARCFACE_S, ARCFACE_M)

# --- Loss Functions ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, reduction='mean'):
        super().__init__(); self.alpha = alpha; self.gamma = gamma; self.reduction = reduction
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none'); pt = torch.exp(-ce_loss); focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum() if self.reduction == 'sum' else focal_loss

def get_criterion(class_weights_tensor=None, current_label_smoothing=LABEL_SMOOTHING, is_mlp_criterion=False):
    # For MLP, label smoothing is 0 if MLP_USE_ARCFACE is True.
    actual_label_smoothing = 0.0 if is_mlp_criterion and MLP_USE_ARCFACE else current_label_smoothing
    
    weights = class_weights_tensor.to(DEVICE) if class_weights_tensor is not None and IMBALANCE_STRATEGY == 'WeightedLoss' else None
    if LOSS_TYPE == 'FocalLoss': return FocalLoss()
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
    
    if sched_type == 'CosineWarmRestarts':
        return CosineAnnealingWarmRestarts(opt_for_scheduler, T_0=epochs, T_mult=1, eta_min=lr * 0.01) 
    elif sched_type == 'ReduceLROnPlateau':
        return ReduceLROnPlateau(opt_for_scheduler, mode=CHECKPOINT_MODE, factor=0.2, patience=5, min_lr=lr * 0.001, monitor=CHECKPOINT_MONITOR)
    return None # Default no scheduler if type not matched

# --- Training & Validation Loops (Simplified for MLP) ---
def train_one_epoch_mlp(model, dataloader, criterion, optimizer, scaler, scheduler, global_epoch, fold_num, device, writer, num_classes, ema_model):
    model.train(); running_loss = 0.0; total_samples = 0; all_preds, all_labels = [], []
    is_sam_active = hasattr(optimizer, 'base_optimizer') and SAM_AVAILABLE and MLP_USE_SAM

    pbar_desc = f"MLP F{fold_num} E{global_epoch+1} Tr" if isinstance(fold_num, int) else f"HPO MLP Train Ep {global_epoch+1}"
    progress_bar = tqdm(dataloader, desc=pbar_desc, leave=True, bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}')
    
    for batch_idx, (inputs, labels_orig, _) in enumerate(progress_bar): # (features, labels, original_indices)
        if check_keyboard_stop(): break
        inputs, labels_orig = inputs.to(device), labels_orig.to(device)
        inputs = inputs.float() # Ensure inputs to MLP are float32
        batch_size = inputs.size(0)

        # Determine active outputs for loss (handles LogitAdjust if active)
        def get_active_outputs(raw_outputs):
            if IMBALANCE_STRATEGY == 'LogitAdjust' and CLASS_PRIORS is not None:
                return raw_outputs + (LOGIT_ADJUSTMENT_TAU * torch.log(CLASS_PRIORS + 1e-12)).unsqueeze(0)
            return raw_outputs

        optimizer.zero_grad() # Zero gradients for the base optimizer at the start of the iteration.
                              # SAM's internal steps will also handle zeroing gradients for its base optimizer.
        
        loss_this_batch = None
        outputs_for_metric = None # To store the outputs used for calculating metrics

        with torch.amp.autocast('cuda', enabled=(MIXED_PRECISION and device.type == 'cuda')):
            if is_sam_active:
                # First pass (ascent)
                outputs1_raw = model(inputs, labels=labels_orig if MLP_USE_ARCFACE else None)
                outputs1_active = get_active_outputs(outputs1_raw)
                loss1 = criterion(outputs1_active, labels_orig)
                
                if torch.isnan(loss1) or torch.isinf(loss1):
                    print(f"{TermColors.RED}Warning: NaN/Inf loss detected in SAM step 1 ({pbar_desc}, batch {batch_idx}). Skipping batch update.{TermColors.ENDC}")
                    continue # Skip to the next batch

                scaler.scale(loss1).backward() 
                optimizer.first_step(zero_grad=True) # SAM's first step (ascent), also zeros base_optimizer grads

                # Second pass (descent)
                outputs2_raw = model(inputs, labels=labels_orig if MLP_USE_ARCFACE else None)
                outputs2_active = get_active_outputs(outputs2_raw)
                loss_this_batch = criterion(outputs2_active, labels_orig) # This is L_S(w + e(w))
                outputs_for_metric = outputs2_active
                
                if torch.isnan(loss_this_batch) or torch.isinf(loss_this_batch):
                    print(f"{TermColors.RED}Warning: NaN/Inf loss detected in SAM step 2 ({pbar_desc}, batch {batch_idx}). Skipping batch update.{TermColors.ENDC}")
                    # Model weights were perturbed by first_step. Consider if a reset or different handling is needed if this happens often.
                    continue # Skip to the next batch
                
                scaler.scale(loss_this_batch).backward() # Calculate gradients of L_S(w + e(w))
                scaler.unscale_(optimizer.base_optimizer) # Unscale gradients before clipping for SAM's base optimizer step
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Clip all model parameters
                optimizer.second_step(zero_grad=True) # SAM's second step (descent, calls base_optimizer.step() and base_optimizer.zero_grad())
                scaler.update() # Update scaler's scale factor for next iteration

            else: # Standard MLP training (Non-SAM)
                # optimizer.zero_grad() was called at the start of the loop
                outputs_raw = model(inputs, labels=labels_orig if MLP_USE_ARCFACE else None)
                outputs_active = get_active_outputs(outputs_raw)
                loss_this_batch = criterion(outputs_active, labels_orig)
                outputs_for_metric = outputs_active
                
                if torch.isnan(loss_this_batch) or torch.isinf(loss_this_batch):
                    print(f"{TermColors.RED}Warning: NaN/Inf loss detected ({pbar_desc}, batch {batch_idx}, Non-SAM). Skipping batch update.{TermColors.ENDC}")
                    continue # Skip to the next batch
                
                scaler.scale(loss_this_batch).backward()
                scaler.unscale_(optimizer) # Unscale gradients before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Clip gradients
                scaler.step(optimizer) 
                scaler.update()
        
        # Common post-step operations, only if loss_this_batch was valid and an optimizer step was attempted/made
        if loss_this_batch is not None and not (torch.isnan(loss_this_batch) or torch.isinf(loss_this_batch)):
            if MLP_USE_EMA and ema_model: ema_model.update_parameters(model)
            
            running_loss += loss_this_batch.item() * batch_size
            total_samples += batch_size
            if outputs_for_metric is not None: # Ensure outputs_for_metric was set
                preds = torch.argmax(outputs_for_metric, dim=1)
                all_preds.append(preds.detach().cpu()); all_labels.append(labels_orig.detach().cpu())
            
            if batch_idx % 20 == 0 or batch_idx == len(dataloader) -1 :
                if total_samples > 0 and all_preds: # Ensure lists are not empty
                    current_acc = (torch.cat(all_preds) == torch.cat(all_labels)).sum().item() / total_samples
                    current_lr_val = optimizer.param_groups[0]['lr']
                    if hasattr(optimizer, 'base_optimizer'): # For SAM
                        current_lr_val = optimizer.base_optimizer.param_groups[0]['lr']
                    progress_bar.set_postfix(loss=f"{loss_this_batch.item():.3f}", acc=f"{current_acc:.3f}", lr=f"{current_lr_val:.1E}")
                elif loss_this_batch is not None: # Postfix even if no acc yet, but loss is valid
                     progress_bar.set_postfix(loss=f"{loss_this_batch.item():.3f}")

    if stop_requested: return None, None
    epoch_loss = running_loss / total_samples if total_samples > 0 else float('nan') # Return NaN if no valid samples
    epoch_acc = (torch.cat(all_preds) == torch.cat(all_labels)).sum().item() / total_samples if total_samples > 0 and all_preds else 0.0
    
    if scheduler and not isinstance(scheduler, ReduceLROnPlateau) and not (MLP_USE_SWA and isinstance(fold_num, int) and fold_num >=0 and global_epoch >= int(MLP_EPOCHS * MLP_SWA_START_EPOCH_FACTOR)): # Check fold_num type for SWA part
        scheduler.step()

    if writer and isinstance(fold_num, int) and fold_num >= 0: # "HPO" indicates HPO trial, no detailed logging
        if not np.isnan(epoch_loss): writer.add_scalar(f'MLP/Loss/train', epoch_loss, global_epoch)
        writer.add_scalar(f'MLP/Accuracy/train', epoch_acc, global_epoch)
        current_lr_writer = optimizer.param_groups[0]['lr']
        if hasattr(optimizer, 'base_optimizer'): current_lr_writer = optimizer.base_optimizer.param_groups[0]['lr']
        writer.add_scalar(f'MLP/LearningRate', current_lr_writer, global_epoch)
    return epoch_loss, epoch_acc

def validate_one_epoch_mlp(model, dataloader, criterion, device, global_epoch, writer, num_classes, scheduler=None, swa_model=None, ema_model=None, return_preds=False, fold_num=0):
    model.eval(); 
    if swa_model: swa_model.eval()
    if ema_model: ema_model.eval()
    results = {}; oof_data = {'preds': [], 'indices': []}
    
    models_to_eval = {'base': model}
    if MLP_USE_SWA and swa_model: models_to_eval['swa'] = swa_model
    if MLP_USE_EMA and ema_model: models_to_eval['ema'] = ema_model

    with torch.no_grad():
        for model_key, current_model in models_to_eval.items():
            current_model.eval(); model_running_loss = 0.0; model_total_samples = 0
            model_all_preds, model_all_labels = [], []
            
            pbar_desc = f"MLP F{fold_num} Val E{global_epoch+1} ({model_key})"
            if fold_num == "HPO": pbar_desc = f"HPO MLP Val Ep {global_epoch+1}" # HPO trial
            
            progress_bar = tqdm(dataloader, desc=pbar_desc, leave=False)
            for (inputs, labels, original_indices_batch) in progress_bar: # (features, labels, original_indices)
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.float() # Ensure inputs to MLP are float32
                batch_size = inputs.size(0)
                
                with torch.amp.autocast('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda')):
                    outputs = current_model(inputs, labels=labels if MLP_USE_ARCFACE else None)
                    adj_outputs = outputs
                    if IMBALANCE_STRATEGY == 'LogitAdjust' and CLASS_PRIORS is not None:
                        adj_outputs = outputs + (LOGIT_ADJUSTMENT_TAU * torch.log(CLASS_PRIORS + 1e-12)).unsqueeze(0)
                    loss = criterion(adj_outputs, labels)
                
                if not torch.isnan(loss):
                    model_running_loss += loss.item() * batch_size; model_total_samples += batch_size
                    preds = torch.argmax(adj_outputs, dim=1)
                    model_all_preds.append(preds.detach().cpu()); model_all_labels.append(labels.detach().cpu())
                    if return_preds and model_key == 'base': # OOF from base model predictions
                         oof_data['preds'].append(F.softmax(adj_outputs, dim=1).detach().cpu())
                         oof_data['indices'].append(original_indices_batch.cpu()) # Already on CPU from FeatureDataset
                else: print(f"{TermColors.RED}Warn: NaN/Inf MLP validation loss. Skipping batch.{TermColors.ENDC}")
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")
                if stop_requested: return None, None, None, None
            
            epoch_loss = model_running_loss / model_total_samples if model_total_samples > 0 else 0
            epoch_acc = (torch.cat(model_all_preds) == torch.cat(model_all_labels)).sum().item() / model_total_samples if model_total_samples > 0 else 0.0
            results[model_key] = (epoch_loss, epoch_acc)
            if writer and fold_num != "HPO":
                writer.add_scalar(f'MLP/Loss/val_{model_key}', epoch_loss, global_epoch)
                writer.add_scalar(f'MLP/Accuracy/val_{model_key}', epoch_acc, global_epoch)
    
    base_loss, base_acc = results.get('base', (float('inf'), 0.0))
    oof_preds_concat = torch.cat(oof_data['preds']).numpy() if oof_data['preds'] else None
    oof_indices_concat = torch.cat(oof_data['indices']).numpy() if oof_data['indices'] else None
    
    if scheduler and isinstance(scheduler, ReduceLROnPlateau):
        metric_to_monitor = base_loss if CHECKPOINT_MONITOR == 'val_loss' else base_acc
        scheduler.step(metric_to_monitor)
    return base_loss, base_acc, oof_preds_concat, oof_indices_concat

# --- Stacking ---
def train_stacking_meta_model(oof_preds, oof_labels, save_path):
    print(f"{TermColors.CYAN}Training Stacking Meta-Model (MLP OOFs)...{TermColors.ENDC}")
    if oof_preds.ndim == 1: oof_features = oof_preds.reshape(-1, 1)
    else: oof_features = oof_preds
        
    meta_model = None; hpo_best_cv_score = None; meta_model_name_for_log = "LGBM"

    if LGBM_AVAILABLE:
        if STACKING_DO_HPO:
            print(f"{TermColors.DEBUG}  HPO for LGBM stacker (CV Folds: {STACKING_HPO_CV_FOLDS})...{TermColors.ENDC}")
            lgbm_hpo = GridSearchCV(LGBMClassifier(random_state=SEED, n_jobs=-1, verbosity=-1), STACKING_LGBM_PARAM_GRID,
                                    cv=STACKING_HPO_CV_FOLDS, scoring='accuracy', n_jobs=-1, verbose=0)
            try:
                lgbm_hpo.fit(oof_features, oof_labels); meta_model = lgbm_hpo.best_estimator_
                hpo_best_cv_score = lgbm_hpo.best_score_; meta_model_name_for_log = "LGBM_HPO"
                print(f"{TermColors.GREEN}  LGBM HPO complete. Best CV score: {hpo_best_cv_score:.4f}{TermColors.ENDC}")
            except Exception as e: print(f"{TermColors.RED}  LGBM HPO Error: {e}. Default LGBM.{TermColors.ENDC}"); meta_model = None
        
        if meta_model is None:
            meta_model = LGBMClassifier(random_state=SEED, n_jobs=-1, verbosity=-1); meta_model_name_for_log = "LGBM_Default"
            try: meta_model.fit(oof_features, oof_labels)
            except Exception as e: print(f"{TermColors.RED}  Default LGBM Error: {e}. LogisticReg.{TermColors.ENDC}"); meta_model = None
    
    if meta_model is None:
        meta_model = LogisticRegression(max_iter=1000, random_state=SEED, n_jobs=-1); meta_model_name_for_log = "LogisticRegression"
        try: meta_model.fit(oof_features, oof_labels)
        except Exception as e: print(f"{TermColors.RED}  LogisticReg Error: {e}. Stacking failed.{TermColors.ENDC}"); return

    meta_acc = hpo_best_cv_score if hpo_best_cv_score is not None else accuracy_score(oof_labels, meta_model.predict(oof_features))
    print(f"{TermColors.GREEN}Stacking meta-model ({meta_model_name_for_log}) trained. Accuracy: {meta_acc:.4f}{TermColors.ENDC}")
    joblib.dump({"model": meta_model, "scaler": None}, save_path) # No scaler for MLP OOFs typically
    print(f"  Meta-model saved: {save_path}")

# --- MLP HPO Objective Function & Runner ---
def mlp_hpo_objective(trial, features_hpo_train, labels_hpo_train, features_hpo_val, labels_hpo_val, input_size_mlp, num_classes_mlp):
    global MLP_LEARNING_RATE, MLP_WEIGHT_DECAY, MLP_DROPOUT_RATE, MLP_HIDDEN_DIMS, MLP_OPTIMIZER_TYPE, MLP_SCHEDULER_TYPE # Allow HPO to modify these for the trial
    
    # Store original global values to restore them later
    original_mlp_lr = MLP_LEARNING_RATE
    original_mlp_wd = MLP_WEIGHT_DECAY
    original_mlp_dropout = MLP_DROPOUT_RATE
    original_mlp_hidden_dims = MLP_HIDDEN_DIMS
    original_mlp_optimizer = MLP_OPTIMIZER_TYPE
    original_mlp_scheduler = MLP_SCHEDULER_TYPE

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
        
        # MLP_OPTIMIZER_TYPE = trial.suggest_categorical('mlp_optimizer', ['AdamW', 'AdamP'] if ADAMP_AVAILABLE else ['AdamW'])
        # MLP_SCHEDULER_TYPE = trial.suggest_categorical('mlp_scheduler', ['CosineWarmRestarts', 'ReduceLROnPlateau', 'None'])

        temp_mlp_model = build_mlp_model(input_size_mlp, num_classes_mlp).to(DEVICE)
        temp_optimizer = get_optimizer(temp_mlp_model, is_mlp_optimizer=True)
        temp_scaler = torch.amp.GradScaler('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda'))
        temp_criterion = get_criterion(class_weights_tensor=CLASS_WEIGHTS, current_label_smoothing=LABEL_SMOOTHING, is_mlp_criterion=True)

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
            
            train_loss, train_acc = train_one_epoch_mlp(temp_mlp_model, hpo_train_loader, temp_criterion, temp_optimizer, temp_scaler, temp_scheduler, 
                            epoch, "HPO", DEVICE, None, num_classes_mlp, None) # No EMA/SWA for HPO trials
            if train_loss is None: return -float('inf') # Interrupted

            val_loss, val_acc, _, _ = validate_one_epoch_mlp(temp_mlp_model, hpo_val_loader, temp_criterion, DEVICE, epoch, None, num_classes_mlp, 
                                                scheduler=temp_scheduler, fold_num="HPO") # No EMA/SWA
            if val_loss is None: return -float('inf') # Interrupted

            trial.report(val_acc, epoch)
            if trial.should_prune(): raise optuna.exceptions.TrialPruned()

            if val_acc > best_hpo_trial_val_acc: best_hpo_trial_val_acc = val_acc; epochs_no_improve = 0
            else: epochs_no_improve += 1
            if epochs_no_improve >= MLP_HPO_PATIENCE: break
        
        del temp_mlp_model, temp_optimizer, temp_scaler, temp_criterion, hpo_train_ds, hpo_val_ds, hpo_train_loader, hpo_val_loader, temp_scheduler
        gc.collect(); torch.cuda.empty_cache()
        return best_hpo_trial_val_acc
    finally: # Restore original global MLP parameters
        MLP_LEARNING_RATE = original_mlp_lr
        MLP_WEIGHT_DECAY = original_mlp_wd
        MLP_DROPOUT_RATE = original_mlp_dropout
        MLP_HIDDEN_DIMS = original_mlp_hidden_dims
        MLP_OPTIMIZER_TYPE = original_mlp_optimizer
        MLP_SCHEDULER_TYPE = original_mlp_scheduler


def run_mlp_hpo(features_hpo, labels_hpo, original_indices_hpo, input_size, num_classes_hpo):
    global MLP_HIDDEN_DIMS, MLP_DROPOUT_RATE, MLP_LEARNING_RATE, MLP_WEIGHT_DECAY, MLP_OPTIMIZER_TYPE, MLP_SCHEDULER_TYPE
    if not OPTUNA_AVAILABLE: print(f"{TermColors.YELLOW}Optuna not available. Skipping MLP HPO.{TermColors.ENDC}"); return

    print(f"\n{TermColors.CYAN}--- MLP HPO Process ---{TermColors.ENDC}")
    
    hpo_train_feats, hpo_val_feats, hpo_train_labels, hpo_val_labels = None, None, None, None
    try:
        hpo_train_feats, hpo_val_feats, hpo_train_labels, hpo_val_labels, _, _ = train_test_split(
            features_hpo, labels_hpo, original_indices_hpo, test_size=MLP_HPO_INTERNAL_VAL_SPLIT, 
            random_state=SEED + 100, stratify=labels_hpo)
    except ValueError: 
        hpo_train_feats, hpo_val_feats, hpo_train_labels, hpo_val_labels, _, _ = train_test_split(
            features_hpo, labels_hpo, original_indices_hpo, test_size=MLP_HPO_INTERNAL_VAL_SPLIT, random_state=SEED + 100)
    
    if hpo_train_feats is None or len(hpo_train_feats) == 0: print(f"{TermColors.RED}MLP HPO data split failed. Skip.{TermColors.ENDC}"); return

    study_name = "mlp_plant_recognition_hpo_study"
    storage_name = f"sqlite:///{MLP_HPO_STUDY_DB_PATH}"
    os.makedirs(os.path.dirname(MLP_HPO_STUDY_DB_PATH), exist_ok=True)

    study = optuna.create_study(
        study_name=study_name, storage=storage_name, direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=max(1, MLP_HPO_EPOCHS // 4)),
        load_if_exists=True
    )

    run_new_hpo_trials_this_session = True 
    num_trials_for_this_session = MLP_HPO_N_TRIALS 

    if MLP_HPO_LOAD_BEST_ONLY:
        try:
            completed_trials_in_db = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if completed_trials_in_db:
                print(f"{TermColors.INFO}MLP HPO: Configured to load best from DB only. Found {len(completed_trials_in_db)} completed trials. Will not run new trials.{TermColors.ENDC}")
                run_new_hpo_trials_this_session = False 
            else:
                print(f"{TermColors.YELLOW}MLP HPO: Configured to load best from DB only, but no completed trials found in study '{study_name}'. Will proceed to run {num_trials_for_this_session} HPO trials as fallback.{TermColors.ENDC}")
        except Exception as e_study_check: 
            print(f"{TermColors.YELLOW}MLP HPO: Error checking study for 'load best only' mode: {e_study_check}. Will proceed to run {num_trials_for_this_session} HPO trials as fallback.{TermColors.ENDC}")

    if run_new_hpo_trials_this_session:
        completed_trials_before_this_run = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        if not MLP_HPO_LOAD_BEST_ONLY: # Standard behavior: always run N_TRIALS new ones
             print(f"{TermColors.INFO}MLP HPO: Study '{study_name}' currently has {completed_trials_before_this_run} completed trials.")
        print(f"{TermColors.INFO}MLP HPO: Running {num_trials_for_this_session} new HPO trials for study '{study_name}'.{TermColors.ENDC}")
        
        objective_fn = lambda trial: mlp_hpo_objective(trial, hpo_train_feats, hpo_train_labels, hpo_val_feats, hpo_val_labels, input_size, num_classes_hpo)
        try:
            study.optimize(objective_fn, n_trials=num_trials_for_this_session, callbacks=[lambda study, trial: gc.collect() or torch.cuda.empty_cache()])
        except Exception as e:
            print(f"{TermColors.RED}Optuna study optimization error: {e}{TermColors.ENDC}"); traceback.print_exc()
    if not study.trials:
        print(f"{TermColors.RED}MLP HPO: No trials found in study. Using default MLP params.{TermColors.ENDC}")
        return

    try:
        best_trial = study.best_trial 
        print(f"{TermColors.GREEN}MLP HPO Complete. Best Val Acc from study: {best_trial.value:.4f}{TermColors.ENDC}")
        print(f"  Best HPO Params from study: {best_trial.params}")
        
        MLP_HIDDEN_DIMS = [best_trial.params[f'mlp_h_dim_l{i}'] for i in range(best_trial.params['mlp_n_layers'])]
        MLP_DROPOUT_RATE = best_trial.params['mlp_dropout_rate']
        MLP_LEARNING_RATE = best_trial.params['mlp_lr']
        MLP_WEIGHT_DECAY = best_trial.params['mlp_wd']
        print(f"{TermColors.GREEN}Global MLP parameters updated with HPO results from study '{study_name}'.{TermColors.ENDC}")
    except ValueError: 
        print(f"{TermColors.RED}MLP HPO: No completed trials found in study '{study_name}'. Using default MLP params.{TermColors.ENDC}")
    except Exception as e:
        print(f"{TermColors.RED}Error retrieving best trial from study '{study_name}': {e}{TermColors.ENDC}"); traceback.print_exc()
        print(f"{TermColors.YELLOW}Using default MLP params due to error retrieving best trial.{TermColors.ENDC}")

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
    dataloader = DataLoader(full_ds, FEATURE_EXTRACTOR_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    all_feats, all_lbls, all_orig_idx = [], [], []
    with torch.no_grad():
        for inputs, lbls_batch, _, orig_idx_batch in tqdm(dataloader, desc="Extracting Features"):
            inputs = inputs.to(DEVICE)
            with torch.amp.autocast('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda')):
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

# --- Main MLP Training Loop ---
def train_mlp_on_features_main_loop(num_classes_mlp, label_encoder_mlp, class_names_mlp, class_freq_mlp, class_prior_mlp, class_weight_mlp):
    global NUM_CLASSES, label_encoder, class_names, CLASS_FREQUENCIES, CLASS_PRIORS, CLASS_WEIGHTS
    NUM_CLASSES=num_classes_mlp; label_encoder=label_encoder_mlp; class_names=class_names_mlp
    CLASS_FREQUENCIES=class_freq_mlp; CLASS_PRIORS=class_prior_mlp; CLASS_WEIGHTS=class_weight_mlp

    print(f"\n{TermColors.HEADER}--- Train MLP on Extracted Features ---{TermColors.ENDC}")
    final_stacking_path = None
    if not os.path.exists(FEATURES_NPZ_PATH): print(f"{TermColors.RED}Features not found: {FEATURES_NPZ_PATH}. Run FE.{TermColors.ENDC}"); return final_stacking_path
    
    try:
        data = np.load(FEATURES_NPZ_PATH, allow_pickle=True)
        feats_all, lbls_all, orig_idx_all = data['features'], data['labels'], data['original_indices']
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
    
    if len(feats_kfold) < N_FOLDS * 2: print(f"{TermColors.RED}Not enough data for K-Fold ({len(feats_kfold)}). Abort MLP.{TermColors.ENDC}"); return final_stacking_path

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_results = defaultdict(list)
    oof_preds = np.full((len(orig_idx_kfold), NUM_CLASSES), np.nan, dtype=np.float32) 
    oof_lbls = np.full(len(orig_idx_kfold), -1, dtype=np.int32) 

    for fold, (train_idx, val_idx) in enumerate(skf.split(feats_kfold, lbls_kfold)):
        if stop_requested: break
        print(f"\n{TermColors.HEADER} MLP Fold {fold+1}/{N_FOLDS} {TermColors.ENDC}")
        train_feats, val_feats = feats_kfold[train_idx], feats_kfold[val_idx]
        train_lbls, val_lbls = lbls_kfold[train_idx], lbls_kfold[val_idx]
        val_orig_ids_fold = orig_idx_kfold[val_idx]
        gc.collect(); torch.cuda.empty_cache()

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
        except Exception as e: print(f"{TermColors.RED}MLP Fold {fold+1} Setup Error: {e}{TermColors.ENDC}"); continue

        start_ep, best_met, _ = load_checkpoint(fold, mlp, opt, None, scaler_mlp, "latest_mlp_checkpoint.pth.tar", True)
        if best_met == (float('-inf') if CHECKPOINT_MODE == 'max' else float('inf')): # if latest was not good
            start_ep, best_met, _ = load_checkpoint(fold, mlp, opt, None, scaler_mlp, "best_mlp_model.pth.tar", True)

        writer_mlp = SummaryWriter(log_dir=os.path.join(BASE_LOG_DIR, f"mlp_fold_{fold}"))
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
            ckpt_path_s = os.path.join(BASE_CHECKPOINT_DIR, f"mlp_fold_{fold}", "latest_mlp_checkpoint.pth.tar")
            if os.path.isfile(ckpt_path_s):
                ckpt_s = torch.load(ckpt_path_s, map_location=DEVICE)
                if 'scheduler' in ckpt_s and ckpt_s['scheduler']:
                    try: sched.load_state_dict(ckpt_s['scheduler'])
                    except: print(f"{TermColors.YELLOW}MLP Sched reload failed.{TermColors.ENDC}")

        swa_sched_mlp = SWALR(opt, swa_lr=(MLP_LEARNING_RATE*0.05), anneal_epochs=5) if MLP_USE_SWA and swa_mlp else None
        mlp_swa_start = int(MLP_EPOCHS * MLP_SWA_START_EPOCH_FACTOR)
        last_ep_fold = start_ep -1

        for epoch in range(start_ep, MLP_EPOCHS):
            if fold_stop or stop_requested: break
            print(f"\n{TermColors.CYAN} MLP F{fold+1} Ep {epoch+1}/{MLP_EPOCHS} {TermColors.ENDC}")
            tr_loss, tr_acc = train_one_epoch_mlp(mlp, train_loader_mlp, crit, opt, scaler_mlp, sched, epoch, fold+1, DEVICE, writer_mlp, NUM_CLASSES, ema_mlp)
            if tr_loss is None: fold_stop = True; break
            v_loss, v_acc, oof_p_curr, oof_idx_curr = validate_one_epoch_mlp(mlp, val_loader_mlp, crit, DEVICE, epoch, writer_mlp, NUM_CLASSES, sched, swa_mlp, ema_mlp, True, fold+1)
            if v_loss is None: fold_stop = True; break
            print(f"MLP F{fold+1} Ep {epoch+1}: Tr L={tr_loss:.4f} A={tr_acc:.4f} | Val L={v_loss:.4f} A={v_acc:.4f} (Best {CHECKPOINT_MONITOR}: {best_met:.4f})")

            if MLP_USE_SWA and swa_mlp and epoch >= mlp_swa_start:
                swa_mlp.update_parameters(mlp); 
                if swa_sched_mlp: swa_sched_mlp.step()
            
            curr_met_stop = v_loss if CHECKPOINT_MONITOR == 'val_loss' else v_acc
            improved_early = (CHECKPOINT_MODE == 'min' and curr_met_stop < best_met_early_stop) or \
                             (CHECKPOINT_MODE == 'max' and curr_met_stop > best_met_early_stop)
            if improved_early: best_met_early_stop = curr_met_stop; no_improve_epochs = 0
            else: no_improve_epochs +=1
            if no_improve_epochs >= MLP_EARLY_STOPPING_PATIENCE: fold_stop = True; print(f"{TermColors.WARNING}Early stop MLP F{fold+1}.{TermColors.ENDC}")
            
            curr_met_ckpt = v_acc if CHECKPOINT_MONITOR == 'val_acc' else v_loss
            is_best = (CHECKPOINT_MODE == 'max' and curr_met_ckpt > best_met) or \
                      (CHECKPOINT_MODE == 'min' and curr_met_ckpt < best_met)
            if is_best:
                best_met = curr_met_ckpt; best_val_loss = v_loss; best_val_acc = v_acc; best_ep = epoch
                print(f"{TermColors.OKGREEN}MLP F{fold+1} New Best {CHECKPOINT_MONITOR}: {best_met:.4f}. Save...{TermColors.ENDC}")
                save_checkpoint(fold, epoch+1, mlp, opt, sched, scaler_mlp, best_met, "best_mlp_model.pth.tar", True)
                save_model(fold, mlp, "best_mlp_model_state_dict.pth", True)
                if MLP_USE_EMA and ema_mlp: save_model(fold, ema_mlp, "best_ema_mlp_model_state_dict.pth", True)
                if MLP_USE_SWA and swa_mlp and epoch >= mlp_swa_start: save_model(fold, swa_mlp, "best_swa_mlp_model_state_dict.pth", True)
                if oof_p_curr is not None and oof_idx_curr is not None:
                    for i, orig_id_val in enumerate(oof_idx_curr):
                        match_idx = np.where(orig_idx_kfold == orig_id_val)[0]
                        if len(match_idx) > 0: oof_preds[match_idx[0]] = oof_p_curr[i]; oof_lbls[match_idx[0]] = lbls_kfold[match_idx[0]]
            
            save_checkpoint(fold, epoch+1, mlp, opt, sched, scaler_mlp, best_met, "latest_mlp_checkpoint.pth.tar", True)
            if LOG_MISCLASSIFIED_IMAGES and ((epoch+1)%5==0 or is_best): log_misclassified(fold, mlp, err_loader_mlp, crit, DEVICE, epoch+1, writer_mlp, NUM_CLASSES, True)
            last_ep_fold = epoch; gc.collect(); torch.cuda.empty_cache()
        
        print(f"MLP F{fold+1} done. Best Base Val Acc: {best_val_acc:.4f} at Ep {best_ep+1 if best_ep!=-1 else 'N/A'}")
        if fold_stop and not stop_requested: save_checkpoint(fold, last_ep_fold+1, mlp, opt, sched, scaler_mlp, best_met, "interrupted_mlp_ckpt.pth.tar", True)
        if stop_requested: save_checkpoint(fold, last_ep_fold+1, mlp, opt, sched, scaler_mlp, best_met, "interrupted_mlp_ckpt.pth.tar", True)
        else:
            if MLP_USE_SWA and swa_mlp and last_ep_fold >= mlp_swa_start:
                print(f"{TermColors.CYAN}MLP F{fold+1} Final SWA BN...{TermColors.ENDC}")
                bn_loader_final = DataLoader(train_ds_mlp, MLP_BATCH_SIZE*2, shuffle=True, num_workers=0)
                try:
                    torch.optim.swa_utils.update_bn(bn_loader_final, swa_mlp.to(DEVICE), device=DEVICE)
                    swa_v_loss, swa_v_acc, _, _ = validate_one_epoch_mlp(swa_mlp, val_loader_mlp, crit, DEVICE, last_ep_fold+1, writer_mlp, NUM_CLASSES, fold_num=f"{fold+1}-SWA-Final")
                    print(f"{TermColors.SUCCESS} MLP F{fold+1} FINAL SWA Val L={swa_v_loss:.4f} A={swa_v_acc:.4f}{TermColors.ENDC}")
                    fold_results['mlp_swa_acc'].append(swa_v_acc); save_model(fold, swa_mlp, "final_swa_mlp_model_state_dict.pth", True)
                except Exception as e: print(f"{TermColors.RED}MLP SWA BN/Eval Error: {e}{TermColors.ENDC}")
            if MLP_USE_EMA and ema_mlp:
                try:
                    ema_v_loss, ema_v_acc, _, _ = validate_one_epoch_mlp(ema_mlp.to(DEVICE), val_loader_mlp, crit, DEVICE, last_ep_fold+1, writer_mlp, NUM_CLASSES, fold_num=f"{fold+1}-EMA-Final")
                    print(f"{TermColors.SUCCESS} MLP F{fold+1} FINAL EMA Val L={ema_v_loss:.4f} A={ema_v_acc:.4f}{TermColors.ENDC}")
                    fold_results['mlp_ema_acc'].append(ema_v_acc); save_model(fold, ema_mlp, "final_ema_mlp_model_state_dict.pth", True)
                except Exception as e: print(f"{TermColors.RED}MLP EMA Eval Error: {e}{TermColors.ENDC}")
        if mlp: save_model(fold, mlp, "final_mlp_model_state_dict.pth", True)
        if writer_mlp: writer_mlp.close()
        fold_results['mlp_best_val_acc'].append(best_val_acc)
        del mlp, opt, sched, scaler_mlp, crit, swa_mlp, ema_mlp, train_loader_mlp, val_loader_mlp, err_loader_mlp; gc.collect(); torch.cuda.empty_cache()
        if stop_requested: break

    print(f"\n{TermColors.HEADER} MLP Cross-Validation Finished {TermColors.ENDC}")
    if not stop_requested:
        print(f"Avg MLP Best Base Val Acc: {np.mean(fold_results['mlp_best_val_acc']):.4f} +/- {np.std(fold_results['mlp_best_val_acc']):.4f}")
        if fold_results['mlp_swa_acc']: print(f"Avg MLP Final SWA Val Acc: {np.mean(fold_results['mlp_swa_acc']):.4f} +/- {np.std(fold_results['mlp_swa_acc']):.4f}")
        if fold_results['mlp_ema_acc']: print(f"Avg MLP Final EMA Val Acc: {np.mean(fold_results['mlp_ema_acc']):.4f} +/- {np.std(fold_results['mlp_ema_acc']):.4f}")
        if RUN_STACKING:
            valid_oof_idx = np.where(oof_lbls != -1)[0]
            if len(valid_oof_idx) > 0:
                final_oof_p = oof_preds[valid_oof_idx]; final_oof_l = oof_lbls[valid_oof_idx]; final_oof_orig_id = orig_idx_kfold[valid_oof_idx]
                np.savez_compressed(STACKING_OOF_PREDS_PATH, preds=final_oof_p, labels=final_oof_l, original_indices=final_oof_orig_id)
                print(f"MLP OOF preds saved: {STACKING_OOF_PREDS_PATH}")
                final_stacking_path = STACKING_META_MODEL_PATH
                train_stacking_meta_model(final_oof_p, final_oof_l, final_stacking_path)
            else: print(f"{TermColors.RED}No valid MLP OOFs for Stacking.{TermColors.ENDC}")
    else: print(f"{TermColors.YELLOW}MLP Training interrupted. Stacking/KD skipped.{TermColors.ENDC}")
    del feats_all, lbls_all, orig_idx_all, feats_kfold, lbls_kfold, orig_idx_kfold, oof_preds, oof_lbls; gc.collect()
    return final_stacking_path

# --- Main Execution ---
def main():
    global stop_requested, label_encoder, class_names, NUM_CLASSES, CLASS_FREQUENCIES, CLASS_PRIORS, CLASS_WEIGHTS
    global RUN_STACKING, RUN_KNOWLEDGE_DISTILLATION # These are still relevant for MLP mode

    set_seed(SEED); signal.signal(signal.SIGINT, handle_interrupt); print_library_info()
    print(f"{TermColors.HEADER}===== Plant Recognition MLP Only Mode (PyTorch) ===={TermColors.ENDC}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, Device: {DEVICE}, Debug: {DEBUG_MODE}")
    
    df_full = None
    try: # Data Loading and Preprocessing
        print(f"\n{TermColors.HEADER}--- STEP 1: Load Full Dataset Info ---{TermColors.ENDC}")
        df_full = pd.read_csv(CSV_PATH, sep=',', low_memory=False, on_bad_lines='skip')
        if 'scientificName' in df_full.columns and 'scientific_name' not in df_full.columns: df_full.rename(columns={'scientificName': 'scientific_name'}, inplace=True)
        required_cols = ['id', 'scientific_name']; df_full = df_full[required_cols].dropna().astype({'id': str}); df_full.rename(columns={'scientific_name': 'scientificName'}, inplace=True)
        min_samples = 1; class_counts = df_full['scientificName'].value_counts(); valid_classes = class_counts[class_counts >= min_samples].index
        df_full = df_full[df_full['scientificName'].isin(valid_classes)].reset_index(drop=True)
        if len(df_full) == 0: print(f"{TermColors.RED}Dataframe empty after filter. Exit.{TermColors.ENDC}"); sys.exit(1)
        
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

    # --- MLP Pipeline ---
    mlp_stacking_model_path = None 
    training_successful = False 

    # Check if final MLP stacking model already exists
    if os.path.exists(STACKING_META_MODEL_PATH) and RUN_STACKING:
        print(f"{TermColors.INFO}MLP stacking model exists: {STACKING_META_MODEL_PATH}. Skip MLP K-fold & stacking.{TermColors.ENDC}")
        mlp_stacking_model_path = STACKING_META_MODEL_PATH; training_successful = True
    # Else, check if OOF predictions from MLP folds exist to run only stacking
    elif os.path.exists(STACKING_OOF_PREDS_PATH) and RUN_STACKING:
        print(f"{TermColors.INFO}MLP OOF predictions found: {STACKING_OOF_PREDS_PATH}. Attempt stacking only.{TermColors.ENDC}")
        try:
            oof_data = np.load(STACKING_OOF_PREDS_PATH)
            if len(oof_data['preds']) > 0 and len(oof_data['preds']) == len(oof_data['labels']):
                train_stacking_meta_model(oof_data['preds'], oof_data['labels'], STACKING_META_MODEL_PATH)
                if os.path.exists(STACKING_META_MODEL_PATH):
                    mlp_stacking_model_path = STACKING_META_MODEL_PATH; training_successful = True
                    print(f"{TermColors.GREEN}Stacking from existing MLP OOFs complete.{TermColors.ENDC}")
                else: print(f"{TermColors.RED}Stacking from OOFs failed. Full MLP train.{TermColors.ENDC}")
            else: print(f"{TermColors.YELLOW}MLP OOF file invalid. Full MLP train.{TermColors.ENDC}")
        except Exception as e: print(f"{TermColors.RED}Error with OOFs: {e}. Full MLP train.{TermColors.ENDC}")
    
    if not training_successful: # Run full MLP training and stacking
        print(f"{TermColors.INFO}Starting full MLP K-fold training and stacking.{TermColors.ENDC}")
        if not os.path.exists(FEATURES_NPZ_PATH):
            print(f"{TermColors.YELLOW}Base features not found: {FEATURES_NPZ_PATH}. Extracting features...{TermColors.ENDC}")
            extract_all_features(df_full, NUM_CLASSES, label_encoder)
            if not os.path.exists(FEATURES_NPZ_PATH): print(f"{TermColors.RED}FE failed. Abort MLP.{TermColors.ENDC}"); sys.exit(1)
        
        mlp_stacking_model_path = train_mlp_on_features_main_loop(
            NUM_CLASSES, label_encoder, class_names, CLASS_FREQUENCIES, CLASS_PRIORS, CLASS_WEIGHTS)
        training_successful = not stop_requested and mlp_stacking_model_path is not None and os.path.exists(mlp_stacking_model_path)

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