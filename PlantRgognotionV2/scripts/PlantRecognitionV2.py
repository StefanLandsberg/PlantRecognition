# ... (Existing imports) ...
import os
import sys
import random
import numpy as np
import pandas as pd
import glob
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

# --- Utility Imports ---
import colorama
from PIL import Image
import cv2
import joblib # For saving sklearn models
import optuna # Add Optuna import
from optuna.integration import PyTorchLightningPruningCallback # Example pruner, adjust if not using Lightning


# --- Optional/Advanced Libraries ---
try:
    import timm
    TIMM_AVAILABLE = True
    print("INFO: timm library found.")
except ImportError:
    print("FATAL: timm library not found. Required. Install with 'pip install timm'.")
    sys.exit(1)

try:
    import torchmetrics
    TORCHMETRICS_AVAILABLE = True
    print("INFO: torchmetrics library found.")
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    print("Warning: torchmetrics library not found. Using basic accuracy calculation.")

try:
    import keyboard; KEYBOARD_AVAILABLE = True
except ImportError:
    keyboard = None; KEYBOARD_AVAILABLE = False
    print("Warning: keyboard library not found. Ctrl+C graceful stop disabled.")

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
    print("INFO: Albumentations library found.")
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Warning: Albumentations library not found. Using basic torchvision transforms.")

from torchvision.transforms import RandAugment

# --- Configuration Constants ---
colorama.init(autoreset=True)
class TermColors:
    HEADER = '\033[95m'; OKBLUE = '\033[94m'; OKCYAN = '\033[96m'; OKGREEN = '\033[92m'
    WARNING = '\033[93m'; FAIL = '\033[91m'; ENDC = '\033[0m'; BOLD = '\033[1m'
    UNDERLINE = '\033[4m'; BRIGHT = '\033[1m'; DIM = '\033[2m'; MAGENTA = '\033[95m'
    BLUE = '\033[94m'; CYAN = '\033[96m'; GREEN = '\033[92m'; YELLOW = '\033[93m'; RED = '\033[91m'
    
try:
    from adamp import AdamP; ADAMP_AVAILABLE = True; print("INFO: AdamP optimizer found.")
except ImportError:
    ADAMP_AVAILABLE = False
    print(f"{TermColors.YELLOW}⚠️ AdamP optimizer not found. Install with 'pip install adamp'. Falling back to AdamW.{TermColors.ENDC}")

try:
    from sam_pytorch import SAM; SAM_AVAILABLE = True; print("INFO: SAM optimizer wrapper found.")
except ImportError:
    SAM_AVAILABLE = False
    print(f"{TermColors.YELLOW}⚠️ SAM optimizer not found. Install with 'pip install sam_pytorch'. SAM will be disabled.{TermColors.ENDC}")

# --- Configuration Constants ---
colorama.init(autoreset=True)
class TermColors:
    HEADER = '\033[95m'; OKBLUE = '\033[94m'; OKCYAN = '\033[96m'; OKGREEN = '\033[92m'
    WARNING = '\033[93m'; FAIL = '\033[91m'; ENDC = '\033[0m'; BOLD = '\033[1m'
    UNDERLINE = '\033[4m'; BRIGHT = '\033[1m'; DIM = '\033[2m'; MAGENTA = '\033[95m'
    BLUE = '\033[94m'; CYAN = '\033[96m'; GREEN = '\033[92m'; YELLOW = '\033[93m'; RED = '\033[91m'

# --- General Config ---
SEED = 42
DEBUG_MODE = False

# --- Calculate paths relative to the script file ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level from 'scripts' to 'PlantRgognotionV2'
V2_DIR = os.path.dirname(SCRIPT_DIR)
# Go up one level from 'PlantRgognotionV2' to the project root 'PlantRecognition'
PROJECT_ROOT = os.path.dirname(V2_DIR)

# Define data and output directories based on calculated roots
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "plant_images")
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
BATCH_SIZE = 24; GRADIENT_ACCUMULATION_STEPS = 3
LEARNING_RATE = 8e-5; WEIGHT_DECAY = 2e-5
OPTIMIZER_TYPE = 'AdamP' if ADAMP_AVAILABLE else 'AdamW'
USE_SAM = True if SAM_AVAILABLE else False
SAM_RHO = 0.05; SAM_ADAPTIVE = False
GRADIENT_CLIP_VAL = 1.0
PROGRESSIVE_RESIZING_STAGES = [
    (10 if not DEBUG_MODE else 1, (224, 224)),
    (15 if not DEBUG_MODE else 1, (384, 384)),
    (5 if not DEBUG_MODE else 1, (448, 448)),
]
TOTAL_EPOCHS_PER_FOLD = sum(s[0] for s in PROGRESSIVE_RESIZING_STAGES)
CURRENT_IMAGE_SIZE = None

# --- HPO Config ---
RUN_HPO = True # Set to False to skip HPO and use defaults
HPO_N_TRIALS = 30 if not DEBUG_MODE else 5 # Number of HPO trials to run
HPO_EPOCHS_PER_TRIAL = 3 if not DEBUG_MODE else 1 # Short training for each trial
HPO_IMAGE_SIZE = (224, 224) # Fixed image size for HPO trials
HPO_BATCH_SIZE = BATCH_SIZE * 2 # Can potentially use larger batch for HPO
HPO_ARCHITECTURES_TO_TRY = [ # Architectures Optuna can choose from
    "tf_efficientnet_b0",
    "mobilenetv3_small_100",
    "resnet18",
    "densenet121", # Include one of the original defaults
    "tf_efficientnet_v2_s", # Smaller V2 version
]
HPO_STUDY_NAME = "plant_recognition_hpo"
HPO_STORAGE_DB = f"sqlite:///{BASE_CHECKPOINT_DIR}/hpo_study.db" # Store study results


# --- Cross-Validation Config ---
N_FOLDS = 5 if not DEBUG_MODE else 2

# --- Hardware Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = multiprocessing.cpu_count() // 2
MIXED_PRECISION = True if DEVICE.type == 'cuda' else False
USE_TORCH_COMPILE = True

# --- Model Config (Defaults, will be updated by HPO if RUN_HPO=True) ---
MODEL_NAMES = ["tf_efficientnet_v2_l.in21k_ft_in1k", "densenet121"] # Default teacher backbones
DROP_PATH_RATE = 0.1; PRETRAINED = True; NUM_CLASSES = -1
EMBEDDING_SIZE = 768; DROPOUT_RATE = 0.4; GLOBAL_POOLING = 'avg'

# --- Metric Learning Config (Defaults, potentially tuned by HPO) ---
METRIC_LEARNING_TYPE = 'ArcFace'
ARCFACE_S = 30.0; ARCFACE_M = 0.50; ARCFACE_EASY_MARGIN = False

# --- Loss Function & Imbalance Handling ---
LOSS_TYPE = 'CrossEntropy'
LABEL_SMOOTHING = 0.1
FOCAL_ALPHA = 0.25; FOCAL_GAMMA = 2.0
IMBALANCE_STRATEGY = 'WeightedSampler' # 'WeightedLoss', 'LogitAdjust', 'WeightedSampler', 'None'
LOGIT_ADJUSTMENT_TAU = 1.0
CLASS_FREQUENCIES = None; CLASS_PRIORS = None; CLASS_WEIGHTS = None

# --- Learning Rate Scheduler Config ---
SCHEDULER_TYPE = 'CosineWarmRestarts'
WARMUP_EPOCHS = 3; LR_MAX = LEARNING_RATE; LR_MIN = LEARNING_RATE * 0.01
T_0 = 5; T_MULT = 1
STEP_LR_STEP_SIZE = 5; STEP_LR_GAMMA = 0.1
PLATEAU_FACTOR = 0.1; PLATEAU_PATIENCE = 3; PLATEAU_MIN_LR = 1e-6
PLATEAU_MODE = 'min'; PLATEAU_MONITOR = 'val_loss'

# --- Augmentation Config ---
USE_RAND_AUGMENT = True # Corrected variable name
RAND_AUGMENT_N = 2 # Corrected variable name
RAND_AUGMENT_M = 9 # Corrected variable name
MIXUP_ALPHA = 0.8; CUTMIX_ALPHA = 1.0; AUG_PROBABILITY = 0.5

# --- Averaging Config ---
USE_SWA = True
SWA_START_EPOCH_GLOBAL_FACTOR = 0.75
SWA_LR_FACTOR = 0.05; SWA_ANNEAL_EPOCHS = 5
USE_EMA = True; EMA_DECAY = 0.999

# --- Checkpointing Config ---
CHECKPOINT_MONITOR = 'val_loss'; CHECKPOINT_MODE = 'min'; SAVE_TOP_K = 1

# --- Error Analysis Config ---
ERROR_LOG_BATCH_SIZE = 64

# --- Test Time Augmentation (TTA) Config ---
TTA_TRANSFORMS = None

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
KD_TEACHER_FOLD = 0 # Which fold's best model to use as teacher
KD_EPOCHS = 15 if not DEBUG_MODE else 2
KD_BATCH_SIZE = BATCH_SIZE * 2 # Can often use larger batch for smaller student
KD_LR = 1e-4
KD_ALPHA = 0.5 # Weight for KD loss (KLDiv) vs CrossEntropy loss (1-alpha)
KD_TEMPERATURE = 4.0 # Softening temperature for logits
KD_STUDENT_MODEL_SAVE_PATH = os.path.join(BASE_MODEL_SAVE_DIR, f"kd_student_{KD_STUDENT_MODEL_NAME}.pth")

# --- Global Variables ---
stop_requested = False; label_encoder = None; class_names = None

# --- Utility Functions ---
# ... (set_seed, handle_interrupt, check_keyboard_stop) ...
def set_seed(seed=SEED): # Definition exists
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    print(f"{TermColors.CYAN}ℹ Random seeds set to {seed}{TermColors.ENDC}")

def handle_interrupt(signal, frame): # Definition exists
    global stop_requested
    if not stop_requested: print(f"\n{TermColors.YELLOW}⏸️ Stop requested... Press Ctrl+C again to force exit.{TermColors.ENDC}"); stop_requested = True
    else: print(f"\n{TermColors.RED}❌ Force exiting...{TermColors.ENDC}"); sys.exit(1)

def check_keyboard_stop(): # Ensure this function is defined here
    if KEYBOARD_AVAILABLE and keyboard.is_pressed('ctrl+c'):
        handle_interrupt(None, None)

# ... (save_checkpoint, load_checkpoint, save_model, log_misclassified) ...
def save_checkpoint(fold, global_epoch, stage_idx, stage_epoch, model, optimizer, scheduler, scaler, best_metric, filename="checkpoint.pth.tar"):
    # ... (ensure path includes fold subdir) ...
    checkpoint_dir = os.path.join(BASE_CHECKPOINT_DIR, f"fold_{fold}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    # ... (rest of saving logic) ...
    model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    opt_state_dict = optimizer.base_optimizer.state_dict() if USE_SAM and hasattr(optimizer, 'base_optimizer') else optimizer.state_dict()
    state = { 'fold': fold, 'global_epoch': global_epoch, 'stage_idx': stage_idx, 'stage_epoch': stage_epoch, 'image_size': CURRENT_IMAGE_SIZE,
              'state_dict': model_state_dict, 'optimizer': opt_state_dict, 'scheduler': scheduler.state_dict() if scheduler else None,
              'scaler': scaler.state_dict() if scaler else None, 'best_metric': best_metric,
              'label_encoder_classes': list(label_encoder.classes_) if label_encoder else None, 'class_frequencies': CLASS_FREQUENCIES }
    torch.save(state, filepath)
    print(f"{TermColors.GREEN}✅ Ckpt Fold {fold} saved: {filename} (GlobEp {global_epoch}, Best {CHECKPOINT_MONITOR}: {best_metric:.4f}){TermColors.ENDC}")


def load_checkpoint(fold, model, optimizer, scheduler, scaler, filename="checkpoint.pth.tar"):
    # ... (ensure path includes fold subdir) ...
    checkpoint_dir = os.path.join(BASE_CHECKPOINT_DIR, f"fold_{fold}")
    filepath = os.path.join(checkpoint_dir, filename)
    # ... (rest of loading logic) ...
    start_global_epoch, start_stage_idx, start_stage_epoch = 0, 0, 0
    loaded_image_size, loaded_class_frequencies, loaded_label_classes = None, None, None
    best_metric = float('inf') if CHECKPOINT_MODE == 'min' else float('-inf')
    if os.path.isfile(filepath):
        print(f"{TermColors.CYAN}ℹ Loading Fold {fold} checkpoint '{filename}'...{TermColors.ENDC}")
        ckpt = torch.load(filepath, map_location=DEVICE)
        if ckpt.get('fold', -1) != fold: print(f"{TermColors.YELLOW}Warn: Checkpoint fold ({ckpt.get('fold')}) mismatch!{TermColors.ENDC}")
        start_global_epoch = ckpt.get('global_epoch', 0); start_stage_idx = ckpt.get('stage_idx', 0)
        start_stage_epoch = ckpt.get('stage_epoch', 0); loaded_image_size = ckpt.get('image_size', None)
        loaded_class_frequencies = ckpt.get('class_frequencies', None); best_metric = ckpt.get('best_metric', best_metric)
        loaded_label_classes = ckpt.get('label_encoder_classes', None)
        state_dict = ckpt['state_dict']; new_state_dict = {}
        is_compiled = hasattr(model, '_orig_mod')
        for k, v in state_dict.items(): # Handle compile/SAM/DP prefixes
            name = k
            if name.startswith('module.'): name = name[len('module.'):]
            if is_compiled and not name.startswith('_orig_mod.'): name = '_orig_mod.' + name
            elif not is_compiled and name.startswith('_orig_mod.'): name = name[len('_orig_mod.'):]
            new_state_dict[name] = v
        try:
            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
            if missing: print(f"{TermColors.YELLOW}⚠️ Missing keys: {', '.join(missing)}{TermColors.ENDC}")
            if unexpected: print(f"{TermColors.YELLOW}⚠️ Unexpected keys: {', '.join(unexpected)}{TermColors.ENDC}")
        except Exception as e: print(f"{TermColors.RED}❌ StateDict Load Error: {e}{TermColors.ENDC}")
        opt_to_load = optimizer.base_optimizer if USE_SAM and hasattr(optimizer, 'base_optimizer') else optimizer
        if opt_to_load and 'optimizer' in ckpt:
            try: opt_to_load.load_state_dict(ckpt['optimizer'])
            except Exception as e: print(f"{TermColors.YELLOW}⚠️ Optim Load Failed: {e}{TermColors.ENDC}")
        # Scheduler state loaded per-stage later
        if scaler and 'scaler' in ckpt and ckpt['scaler']:
            try: scaler.load_state_dict(ckpt['scaler'])
            except Exception as e: print(f"{TermColors.YELLOW}⚠️ Scaler Load Failed: {e}{TermColors.ENDC}")
        print(f"{TermColors.GREEN}✅ Ckpt Fold {fold} loaded. Resume GlobEp {start_global_epoch}. Best {CHECKPOINT_MONITOR}: {best_metric:.4f}{TermColors.ENDC}")
    else: print(f"{TermColors.YELLOW}⚠️ No checkpoint found for Fold {fold}. Starting fresh.{TermColors.ENDC}")
    return start_global_epoch, start_stage_idx, start_stage_epoch, best_metric, loaded_label_classes, loaded_image_size, loaded_class_frequencies

def save_model(fold, model, filename="final_model.pth"):
    # ... (ensure path includes fold subdir) ...
    model_dir = os.path.join(BASE_MODEL_SAVE_DIR, f"fold_{fold}")
    os.makedirs(model_dir, exist_ok=True)
    filepath = os.path.join(model_dir, filename)
    # ... (rest of saving logic) ...
    model_to_save = model
    if hasattr(model_to_save, 'module'): model_to_save = model_to_save.module
    if hasattr(model_to_save, '_orig_mod'): model_to_save = model_to_save._orig_mod
    torch.save(model_to_save.state_dict(), filepath)
    print(f"{TermColors.GREEN}✅ Fold {fold} model state_dict saved: {filename}{TermColors.ENDC}")

def log_misclassified(fold, model, dataloader, criterion, device, global_epoch, writer, num_classes, max_images=20):
    # ... (ensure path includes fold subdir) ...
    error_dir = os.path.join(BASE_ERROR_ANALYSIS_DIR, f"fold_{fold}")
    os.makedirs(error_dir, exist_ok=True)
    error_log_file = os.path.join(error_dir, f"epoch_{global_epoch}_errors.csv")
    # ... (rest of logging logic) ...
    model.eval(); misclassified_count = 0
    print(f"{TermColors.CYAN}ℹ Fold {fold} Logging misclassified images for global epoch {global_epoch}...{TermColors.ENDC}")
    with open(error_log_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['image_path', 'true_label', 'predicted_label', 'confidence', 'loss', 'logits_raw', 'logits_adjusted']
        writer_csv = csv.DictWriter(csvfile, fieldnames=fieldnames); writer_csv.writeheader()
        with torch.no_grad():
            for inputs, labels, paths in tqdm(dataloader, desc=f"Fold {fold} Error Analysis", leave=False):
                if misclassified_count >= max_images: break
                inputs, labels = inputs.to(device), labels.to(device)
                with autocast(enabled=MIXED_PRECISION):
                    embeddings = model(inputs)
                    if hasattr(model, 'metric_fc'): logits = model.metric_fc(embeddings, labels)
                    else: raise AttributeError("ArcFace layer not found.")
                    adj_logits = logits
                    if IMBALANCE_STRATEGY == 'LogitAdjust' and CLASS_PRIORS is not None:
                        logit_adj = LOGIT_ADJUSTMENT_TAU * torch.log(CLASS_PRIORS + 1e-12)
                        adj_logits = logits + logit_adj.unsqueeze(0)
                    loss = criterion(adj_logits, labels) # Use adjusted for loss calc if applicable
                probs = F.softmax(adj_logits, dim=1); preds = torch.argmax(probs, dim=1) # Use adjusted for prediction
                confs = torch.max(probs, dim=1)[0]; incorrect = preds != labels; incorrect_idx = torch.where(incorrect)[0]
                for idx in incorrect_idx:
                    if misclassified_count >= max_images: break
                    true_i, pred_i = labels[idx].item(), preds[idx].item(); conf, path = confs[idx].item(), paths[idx]
                    item_loss = criterion(adj_logits[idx].unsqueeze(0), labels[idx].unsqueeze(0)).item()
                    true_n, pred_n = label_encoder.inverse_transform([true_i])[0], label_encoder.inverse_transform([pred_i])[0]
                    writer_csv.writerow({'image_path': path, 'true_label': true_n, 'predicted_label': pred_n, 'confidence': f"{conf:.4f}",
                                         'loss': f"{item_loss:.4f}", 'logits_raw': logits[idx].cpu().numpy().round(2).tolist(),
                                         'logits_adjusted': adj_logits[idx].cpu().numpy().round(2).tolist()})
                    misclassified_count += 1
    print(f"{TermColors.CYAN}ℹ Fold {fold} Misclassified images logged.{TermColors.ENDC}")


# --- Dataset and Transforms ---
# ... (PlantDataset, get_transforms) ...
class PlantDataset(Dataset): # Definition remains largely the same
    def __init__(self, dataframe, image_dir, transform=None, label_encoder=None, include_paths=False, image_size=None):
        self.dataframe = dataframe; self.image_dir = image_dir; self.transform = transform
        self.label_encoder = label_encoder; self.include_paths = include_paths
        self.image_size = image_size if image_size else PROGRESSIVE_RESIZING_STAGES[0][1]
        self.valid_indices = []
        # print(f"{TermColors.CYAN}ℹ Verifying images for size {self.image_size}...{TermColors.ENDC}") # Less verbose
        for idx, row in self.dataframe.iterrows(): # Removed tqdm for less noise during CV
            img_full_path = os.path.join(self.image_dir, row['image_path'])
            if os.path.exists(img_full_path) and os.path.getsize(img_full_path) > 0: # Added size check
                 self.valid_indices.append(idx)
            # else: # Optional: Log missing/empty files
            #     print(f"{TermColors.YELLOW}Warn: Skipping invalid image: {img_full_path}{TermColors.ENDC}")

        self.dataframe = self.dataframe.loc[self.valid_indices].reset_index(drop=True)
        # print(f"{TermColors.GREEN}✅ Found {len(self.dataframe)} valid images.{TermColors.ENDC}") # Less verbose

    def __len__(self): return len(self.dataframe)
    def get_labels(self): return self.dataframe['label'].tolist() # Helper for sampler
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]; img_path = os.path.join(self.image_dir, row['image_path'])
        label_name = row['scientificName']
        try:
            # Use PIL for potentially more robust loading, then convert
            image = Image.open(img_path).convert('RGB')
            image = np.array(image) # Convert PIL to numpy array for albumentations/cv2
            # image = cv2.imread(img_path); image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Original cv2 load
            if image is None: raise IOError("Image loading failed")
        except Exception as e:
            print(f"{TermColors.RED}Err load {img_path}: {e}. Dummy.{TermColors.ENDC}")
            dummy_img = torch.zeros((3, *self.image_size), dtype=torch.float32); label = -1
            return (dummy_img, label, f"ERROR_{img_path}") if self.include_paths else (dummy_img, label)

        label = self.label_encoder.transform([label_name])[0] if self.label_encoder else label_name

        if self.transform:
            try:
                augmented = self.transform(image=image)
                image = augmented['image']
            except Exception as e:
                print(f"{TermColors.RED}Err transform {img_path}: {e}. Fallback.{TermColors.ENDC}")
                try: # Basic fallback using PIL transforms if albumentations fails
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
                    return (dummy_img, label, f"ERROR_TF_{img_path}") if self.include_paths else (dummy_img, label)

        return (image, label, img_path) if self.include_paths else (image, label)

def get_transforms(image_size=PROGRESSIVE_RESIZING_STAGES[0][1], use_albumentations=ALBUMENTATIONS_AVAILABLE):
    global TTA_TRANSFORMS # Moved global declaration here
    # Definition remains the same, just called per stage
    mean=[0.485, 0.456, 0.406]; std=[0.229, 0.224, 0.225]
    print(f"{TermColors.CYAN}ℹ Generating transforms for image size: {image_size}{TermColors.ENDC}")
    if use_albumentations:
        # More robust augmentations, less aggressive CoarseDropout
        train_transform = A.Compose([
            A.RandomResizedCrop(h=image_size[0], w=image_size[1], scale=(0.7, 1.0), ratio=(0.75, 1.33)),
            A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.3), # Vertical flips might be less common for plants
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=cv2.BORDER_REFLECT),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.7),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.MotionBlur(p=0.5),
                A.GaussNoise(p=0.3)
            ], p=0.4),
            # A.CoarseDropout(max_holes=8, max_height=image_size[0]//8, max_width=image_size[1]//8, min_holes=1, fill_value=mean, p=0.3), # Less aggressive cutout
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
        val_transform = A.Compose([
            A.Resize(h=image_size[0], w=image_size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
        # TTA: Simple horizontal flip
        # global TTA_TRANSFORMS; # Removed from here
        TTA_TRANSFORMS = A.Compose([
            A.Resize(h=image_size[0], w=image_size[1]),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    else: # torchvision
        train_tf = [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ]
        if USE_RAND_AUGMENT: # Corrected variable name
             train_tf.append(RandAugment(num_ops=RAND_AUGMENT_N, magnitude=RAND_AUGMENT_M)) # Corrected variable names
        train_tf.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3)) # Less aggressive erasing
        ])
        train_transform = transforms.Compose(train_tf)
        val_transform = transforms.Compose([
            transforms.Resize(int(max(image_size)*1.1)), # Slightly larger resize before crop
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        # global TTA_TRANSFORMS; # Removed from here
        TTA_TRANSFORMS = transforms.Compose([
            transforms.Resize(int(max(image_size)*1.1)),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    return train_transform, val_transform

# --- Model Architecture ---
# ... (ArcFace, CombinedModel, build_model) ...
class ArcFace(nn.Module): # Definition remains the same
    def __init__(self, i, o, s=ARCFACE_S, m=ARCFACE_M, easy=ARCFACE_EASY_MARGIN, ls=0.0):
        super().__init__(); self.i=i; self.o=o; self.s=s; self.m=m; self.ls=ls; self.w=nn.Parameter(torch.FloatTensor(o,i))
        nn.init.xavier_uniform_(self.w); self.easy=easy; self.cos_m=math.cos(m); self.sin_m=math.sin(m); self.th=math.cos(math.pi-m); self.mm=math.sin(math.pi-m)*m
    def forward(self, inp, lbl):
        cos=F.linear(F.normalize(inp), F.normalize(self.w)); sin=torch.sqrt(1.0-torch.pow(cos,2).clamp(0,1))
        phi=cos*self.cos_m-sin*self.sin_m; phi=torch.where(cos>self.th, phi, cos-self.mm) if not self.easy else torch.where(cos>0, phi, cos)
        onehot=torch.zeros(cos.size(), device=inp.device); onehot.scatter_(1, lbl.view(-1,1).long(), 1)
        if self.ls>0: onehot=(1-self.ls)*onehot+self.ls/self.o
        out=(onehot*phi)+((1.0-onehot)*cos); out*=self.s; return out

class CombinedModel(nn.Module):
    def __init__(self, model_names, num_classes, pretrained=True, global_pool='avg', dropout_rate=0.3, embedding_size=512,
                 drop_path_rate=0.1, arcface_s=30.0, arcface_m=0.5, arcface_easy_margin=False):
        super().__init__()
        self.backbones = nn.ModuleList()
        total_features = 0
        if isinstance(model_names, str): # Handle single model name (for HPO)
            model_names = [model_names]

        print(f"{TermColors.CYAN}ℹ Building Combined Model: {', '.join(model_names)}{TermColors.ENDC}")
        for name in model_names:
            print(f"{TermColors.CYAN}  Loading backbone: {name}...{TermColors.ENDC}")
            kwargs = {'drop_path_rate': drop_path_rate} if drop_path_rate > 0 and any(s in name for s in ['convnext', 'efficientnet', 'vit', 'swin']) else {}
            try:
                backbone = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool=global_pool, **kwargs)
                self.backbones.append(backbone)
                total_features += backbone.num_features
                print(f"    {name} loaded. Features: {backbone.num_features}")
            except Exception as e:
                print(f"{TermColors.RED}❌ Backbone Load Fail {name}: {e}{TermColors.ENDC}")
                raise e

        print(f"  Total features: {total_features}, Embedding: {embedding_size}")
        self.embedding = nn.Linear(total_features, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.metric_fc = ArcFace(embedding_size, num_classes, s=arcface_s, m=arcface_m, easy_margin=arcface_easy_margin)

    def forward(self, x):
        features = [bb.forward_features(x) for bb in self.backbones]
        if len(features) == 1:
            combined = features[0]
        else:
            combined = torch.cat(features, dim=1)
        x = self.embedding(combined)
        x = self.bn(x)
        x = self.dropout(x)
        return x # Return embeddings

def build_model(model_names=MODEL_NAMES, num_classes=NUM_CLASSES, pretrained=PRETRAINED, dropout_rate=DROPOUT_RATE,
                embedding_size=EMBEDDING_SIZE, drop_path_rate=DROP_PATH_RATE, global_pool=GLOBAL_POOLING,
                arcface_s=ARCFACE_S, arcface_m=ARCFACE_M, arcface_easy_margin=ARCFACE_EASY_MARGIN):
    # Pass the potentially updated global variables
    return CombinedModel(model_names, num_classes, pretrained, global_pool, dropout_rate, embedding_size,
                         drop_path_rate, arcface_s, arcface_m, arcface_easy_margin)


# --- Loss Functions ---
# ... (FocalLoss, get_criterion) ...
class FocalLoss(nn.Module): # Definition remains the same
    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, reduction='mean'):
        super().__init__(); self.a=alpha; self.g=gamma; self.r=reduction
    def forward(self, i, t):
        ce=F.cross_entropy(i, t, reduction='none'); pt=torch.exp(-ce); loss=self.a*(1-pt)**self.g*ce
        if self.r=='mean': return loss.mean()
        elif self.r=='sum': return loss.sum()
        else: return loss

def get_criterion(loss_type=LOSS_TYPE, label_smoothing=LABEL_SMOOTHING, class_weights=None):
    weights = class_weights.to(DEVICE) if class_weights is not None else None
    if loss_type == 'CrossEntropy':
        print(f"{TermColors.CYAN}ℹ Loss: CrossEntropy (LS: {label_smoothing}, Weights: {'Yes' if weights is not None else 'No'}).{TermColors.ENDC}")
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=weights)
    elif loss_type == 'FocalLoss':
        print(f"{TermColors.CYAN}ℹ Loss: FocalLoss (Alpha: {FOCAL_ALPHA}, Gamma: {FOCAL_GAMMA}).{TermColors.ENDC}")
        if weights is not None: print(f"{TermColors.YELLOW}Warn: Weights ignored by FocalLoss.{TermColors.ENDC}")
        return FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    raise ValueError(f"Unsupported loss: {loss_type}")

# --- Optimizer and Scheduler ---
# ... (get_optimizer, get_scheduler) ...
def get_optimizer(model, optimizer_type=OPTIMIZER_TYPE, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, use_sam=USE_SAM, sam_rho=SAM_RHO, sam_adaptive=SAM_ADAPTIVE):
    # Pass potentially updated global variables
    params = filter(lambda p: p.requires_grad, model.parameters()); base_opt = None
    if optimizer_type == 'AdamP' and ADAMP_AVAILABLE:
        print(f"{TermColors.CYAN}ℹ Optim: AdamP (LR={lr:.1E}, WD={weight_decay:.1E}).{TermColors.ENDC}")
        base_opt = AdamP(params, lr=lr, weight_decay=weight_decay, nesterov=True, betas=(0.9, 0.999))
    elif optimizer_type == 'AdamW' or (optimizer_type == 'AdamP' and not ADAMP_AVAILABLE):
        if optimizer_type == 'AdamP': print(f"{TermColors.YELLOW}Warn: AdamP fallback AdamW.{TermColors.ENDC}")
        print(f"{TermColors.CYAN}ℹ Optim: AdamW (LR={lr:.1E}, WD={weight_decay:.1E}).{TermColors.ENDC}")
        base_opt = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'Adam':
        print(f"{TermColors.CYAN}ℹ Optim: Adam (LR={lr:.1E}, WD={weight_decay:.1E}).{TermColors.ENDC}")
        base_opt = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'SGD':
        print(f"{TermColors.CYAN}ℹ Optim: SGD (LR={lr:.1E}, WD={weight_decay:.1E}).{TermColors.ENDC}")
        base_opt = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)

    if base_opt is None: raise ValueError(f"Optim error: {optimizer_type}")

    if use_sam and SAM_AVAILABLE:
        print(f"{TermColors.CYAN}ℹ Wrap Optim: SAM (rho={sam_rho}, adaptive={sam_adaptive}).{TermColors.ENDC}")
        return SAM(base_optimizer=base_opt, rho=sam_rho, adaptive=sam_adaptive)
    if use_sam and not SAM_AVAILABLE:
        print(f"{TermColors.YELLOW}Warn: SAM unavailable.{TermColors.ENDC}")
    return base_opt

def get_scheduler(optimizer, scheduler_type=SCHEDULER_TYPE, stage_epochs=10, warmup_epochs=WARMUP_EPOCHS, lr_max=LR_MAX, lr_min=LR_MIN,
                  step_size=STEP_LR_STEP_SIZE, gamma=STEP_LR_GAMMA, plateau_factor=PLATEAU_FACTOR, plateau_patience=PLATEAU_PATIENCE,
                  plateau_min_lr=PLATEAU_MIN_LR, plateau_mode=PLATEAU_MODE, plateau_monitor=PLATEAU_MONITOR,
                  t_0=T_0, t_mult=T_MULT): # Added CosineWarmRestarts params
    opt = optimizer.base_optimizer if hasattr(optimizer, 'base_optimizer') else optimizer
    # Use potentially updated global variables for defaults if not passed explicitly
    warmup_epochs = warmup_epochs if warmup_epochs is not None else WARMUP_EPOCHS
    lr_max = lr_max if lr_max is not None else LEARNING_RATE # Use current LR as max
    lr_min = lr_min if lr_min is not None else lr_max * 0.01
    step_size = step_size if step_size is not None else STEP_LR_STEP_SIZE
    gamma = gamma if gamma is not None else STEP_LR_GAMMA
    plateau_factor = plateau_factor if plateau_factor is not None else PLATEAU_FACTOR
    plateau_patience = plateau_patience if plateau_patience is not None else PLATEAU_PATIENCE
    plateau_min_lr = plateau_min_lr if plateau_min_lr is not None else PLATEAU_MIN_LR
    plateau_mode = plateau_mode if plateau_mode is not None else PLATEAU_MODE
    plateau_monitor = plateau_monitor if plateau_monitor is not None else PLATEAU_MONITOR
    t_0 = t_0 if t_0 is not None else T_0
    t_mult = t_mult if t_mult is not None else T_MULT

    if scheduler_type == 'CosineAnnealing': print(f"{TermColors.CYAN}ℹ Sched: CosineAnnealingLR ({stage_epochs} epochs).{TermColors.ENDC}"); return CosineAnnealingLR(opt, T_max=stage_epochs - warmup_epochs, eta_min=lr_min)
    elif scheduler_type == 'StepLR': print(f"{TermColors.CYAN}ℹ Sched: StepLR (step={step_size}, gamma={gamma}).{TermColors.ENDC}"); return StepLR(opt, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'WarmupCosine': print(f"{TermColors.CYAN}ℹ Sched: WarmupCosine ({warmup_epochs} warmup).{TermColors.ENDC}"); warmup = LinearLR(opt, start_factor=1e-6, end_factor=1.0, total_iters=warmup_epochs); cosine = CosineAnnealingLR(opt, T_max=stage_epochs - warmup_epochs, eta_min=lr_min); return SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    elif scheduler_type == 'ReduceLROnPlateau': print(f"{TermColors.CYAN}ℹ Sched: ReduceLROnPlateau ('{plateau_monitor}').{TermColors.ENDC}"); return ReduceLROnPlateau(opt, mode=plateau_mode, factor=plateau_factor, patience=plateau_patience, min_lr=plateau_min_lr, verbose=True)
    elif scheduler_type == 'CosineWarmRestarts': print(f"{TermColors.CYAN}ℹ Sched: CosineAnnealingWarmRestarts (T_0={t_0}, T_mult={t_mult}).{TermColors.ENDC}"); return CosineAnnealingWarmRestarts(opt, T_0=t_0, T_mult=t_mult, eta_min=lr_min)
    print(f"{TermColors.YELLOW}Warn: No scheduler. Constant LR.{TermColors.ENDC}"); return None

# --- Augmentation Helpers ---
# ... (mixup_data, mixup_criterion, rand_bbox, cutmix_data) ...
def mixup_data(x, y, alpha=1.0, device='cuda'): lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0; bs = x.size(0); idx = torch.randperm(bs).to(device); mixed_x = lam * x + (1 - lam) * x[idx, :]; y_a, y_b = y, y[idx]; return mixed_x, y_a, y_b, lam
def mixup_criterion(criterion, pred, y_a, y_b, lam): return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
def rand_bbox(size, lam): W, H = size[2], size[3]; r = np.sqrt(1. - lam); w, h = int(W * r), int(H * r); cx, cy = np.random.randint(W), np.random.randint(H); x1, y1 = np.clip(cx - w // 2, 0, W), np.clip(cy - h // 2, 0, H); x2, y2 = np.clip(cx + w // 2, 0, W), np.clip(cy + h // 2, 0, H); return x1, y1, x2, y2
def cutmix_data(x, y, alpha=1.0, device='cuda'): lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0; bs = x.size(0); idx = torch.randperm(bs).to(device); y_a, y_b = y, y[idx]; x1, y1, x2, y2 = rand_bbox(x.size(), lam); x[:, :, x1:x2, y1:y2] = x[idx, :, x1:x2, y1:y2]; lam = 1 - ((x2 - x1) * (y2 - y1) / (x.size(-1) * x.size(-2))); return x, y_a, y_b, lam

# --- EMA Averaging Function ---
# ... (ema_avg_fn) ...
def ema_avg_fn(averaged_model_parameter, model_parameter, num_averaged):
    return EMA_DECAY * averaged_model_parameter + (1 - EMA_DECAY) * model_parameter

# --- Training and Validation Loops ---
# ... (train_one_epoch, validate_one_epoch) ...
def train_one_epoch(model, dataloader, criterion, optimizer, scaler, scheduler, global_epoch, stage_idx, stage_epoch, stage_total_epochs, device, writer, num_classes, ema_model,
                    mixup_alpha=MIXUP_ALPHA, cutmix_alpha=CUTMIX_ALPHA, aug_probability=AUG_PROBABILITY, grad_clip_val=GRADIENT_CLIP_VAL,
                    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS, use_sam=USE_SAM, fold_num=0): # Added fold_num for HPO logging
    model.train(); running_loss = 0.0; total_samples = 0; all_preds, all_labels = [], []
    is_sam = hasattr(optimizer, 'base_optimizer')
    # Adjust description based on whether it's HPO or main training
    if fold_num == "HPO":
        pbar_desc = f"HPO Trial Train Ep {global_epoch+1}/{stage_total_epochs}"
    else:
        pbar_desc = f"Fold {fold_num} Stage {stage_idx+1} Ep {stage_epoch+1}/{stage_total_epochs} (Glob {global_epoch+1}) Train"

    progress_bar = tqdm(dataloader, desc=pbar_desc, leave=False)
    optimizer.zero_grad()
    for batch_idx, batch_data in enumerate(progress_bar):
        # Handle potential path inclusion in dataloader
        if len(batch_data) == 3: inputs, labels, _ = batch_data # Ignore paths if present
        else: inputs, labels = batch_data

        check_keyboard_stop();
        if stop_requested: break
        inputs, labels = inputs.to(device), labels.to(device); batch_size = inputs.size(0)
        use_mixup, use_cutmix = False, False; r = np.random.rand(1) # Augmentation logic remains same
        if mixup_alpha > 0 and cutmix_alpha > 0 and r < aug_probability:
            if np.random.rand() < 0.5: inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, mixup_alpha, device); use_mixup = True
            else: inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, cutmix_alpha, device); use_cutmix = True
        elif mixup_alpha > 0 and r < aug_probability: inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, mixup_alpha, device); use_mixup = True
        elif cutmix_alpha > 0 and r < aug_probability: inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, cutmix_alpha, device); use_cutmix = True
        else: lam = 1.0; targets_a, targets_b = labels, labels
        with autocast(enabled=MIXED_PRECISION):
            def get_loss(current_outputs): # Loss calculation including logit adjust
                adj_outputs = current_outputs
                if IMBALANCE_STRATEGY == 'LogitAdjust' and CLASS_PRIORS is not None:
                    logit_adj = LOGIT_ADJUSTMENT_TAU * torch.log(CLASS_PRIORS + 1e-12)
                    adj_outputs = current_outputs + logit_adj.unsqueeze(0)
                if use_mixup or use_cutmix: loss = mixup_criterion(criterion, adj_outputs, targets_a, targets_b, lam)
                else: loss = criterion(adj_outputs, labels)
                return loss, adj_outputs
            def get_logits(current_embeddings): # Get logits from ArcFace
                # Handle potential mismatch if model doesn't have metric_fc (e.g., student model)
                if hasattr(model, 'metric_fc'):
                    return model.metric_fc(current_embeddings, targets_a if use_mixup or use_cutmix else labels)
                elif hasattr(model, 'fc'): # Assume standard fc layer if metric_fc not present
                     return model(inputs) # If model directly outputs logits
                else: # Fallback for simple timm models
                    return model(inputs) # Assume model(inputs) gives logits if no specific head found

            if is_sam: # SAM steps remain same
                embeddings = model(inputs); outputs = get_logits(embeddings)
                loss1, adjusted_outputs1 = get_loss(outputs)
                loss1_scaled = loss1 / gradient_accumulation_steps if gradient_accumulation_steps > 1 else loss1
                scaler.scale(loss1_scaled).backward()
                scaler.unscale_(optimizer); optimizer.first_step(zero_grad=True); scaler.update()
                embeddings_perturbed = model(inputs); outputs_perturbed = get_logits(embeddings_perturbed)
                loss_final, outputs_final = get_loss(outputs_perturbed)
            else: # Standard optimizer steps remain same
                embeddings = model(inputs); outputs = get_logits(embeddings)
                loss_final, outputs_final = get_loss(outputs)

        loss_final_scaled = loss_final / gradient_accumulation_steps if gradient_accumulation_steps > 1 else loss_final
        if torch.isnan(loss_final_scaled) or torch.isinf(loss_final_scaled):
             print(f"{TermColors.RED}Warn: NaN/Inf loss @ batch {batch_idx}. Skip.{TermColors.ENDC}")
             if (batch_idx + 1) % gradient_accumulation_steps == 0: optimizer.zero_grad(); continue
        scaler.scale(loss_final_scaled).backward()
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if grad_clip_val > 0: # Gradient clipping remains same
                scaler.unscale_(optimizer)
                params_to_clip = optimizer.base_optimizer.param_groups if is_sam else optimizer.param_groups
                torch.nn.utils.clip_grad_norm_((p for grp in params_to_clip for p in grp['params'] if p.grad is not None), grad_clip_val)
            scaler.step(optimizer.second_step if is_sam else optimizer)
            scaler.update(); optimizer.zero_grad()
            # --- EMA Update ---
            if USE_EMA and ema_model: ema_model.update() # Update EMA model after optimizer step

        current_step_loss = loss_final.item(); running_loss += current_step_loss * batch_size; total_samples += batch_size
        # Ensure outputs_final is used for accuracy calculation
        preds_for_acc = torch.argmax(outputs_final, dim=1)
        all_preds.append(preds_for_acc.detach()); all_labels.append(labels.detach()) # Store predictions (indices) and labels
        progress_bar.set_postfix(loss=f"{current_step_loss:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.1E}")

    if stop_requested: return None, None, None # Return None for EMA metrics too
    epoch_loss = running_loss / total_samples if total_samples > 0 else 0

    # Calculate accuracy from collected predictions and labels
    all_preds_tensor = torch.cat(all_preds); all_labels_tensor = torch.cat(all_labels)
    epoch_acc = (all_preds_tensor == all_labels_tensor).sum().item() / total_samples if total_samples > 0 else 0

    # Only log to writer if it's provided (not during HPO objective usually)
    if writer:
        writer.add_scalar('Loss/train', epoch_loss, global_epoch); writer.add_scalar('Accuracy/train', epoch_acc, global_epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], global_epoch)
        if fold_num != "HPO": # Don't log stage info during HPO
            writer.add_scalar('Stage/Index', stage_idx, global_epoch); writer.add_scalar('Stage/Epoch', stage_epoch, global_epoch)
    return epoch_loss, epoch_acc


def validate_one_epoch(model, dataloader, criterion, device, global_epoch, writer, num_classes, swa_model=None, ema_model=None, return_preds=False, fold_num=0): # Added fold_num
    results = {}
    models_to_eval = {'standard': model}
    if swa_model: models_to_eval['swa'] = swa_model
    if USE_EMA and ema_model: models_to_eval['ema'] = ema_model

    fold_oof_preds = [] # Store predictions for stacking
    fold_oof_labels = [] # Store true labels for stacking

    for model_key, current_model in models_to_eval.items():
        current_model.eval()
        running_loss = 0.0; total_samples = 0; all_preds_raw, all_labels_raw = [], [] # Store raw logits/probs for stacking if needed

        if fold_num == "HPO":
            pbar_desc = f"HPO Trial Validate GlobEp {global_epoch+1} ({model_key})"
        else:
            pbar_desc = f"Fold {fold_num} Validate GlobEp {global_epoch+1} ({model_key})"

        progress_bar = tqdm(dataloader, desc=pbar_desc, leave=False)
        with torch.no_grad():
            for batch_data in progress_bar:
                # Handle potential path inclusion
                if len(batch_data) == 3: inputs, labels, _ = batch_data
                else: inputs, labels = batch_data

                inputs, labels = inputs.to(device), labels.to(device); batch_size = inputs.size(0)
                tta_outputs_list = []
                with autocast(enabled=MIXED_PRECISION):
                    embeddings = current_model(inputs)
                    # Handle potential mismatch if model doesn't have metric_fc (e.g., student model)
                    if hasattr(current_model, 'metric_fc'):
                        outputs = current_model.metric_fc(embeddings, labels)
                    elif hasattr(current_model, 'fc'): # Assume standard fc layer
                         outputs = current_model(inputs) # If model directly outputs logits
                    else: # Fallback
                        outputs = current_model(inputs) # Assume model(inputs) gives logits

                    adj_outputs = outputs
                    if IMBALANCE_STRATEGY == 'LogitAdjust' and CLASS_PRIORS is not None:
                        logit_adj = LOGIT_ADJUSTMENT_TAU * torch.log(CLASS_PRIORS + 1e-12)
                        adj_outputs = outputs + logit_adj.unsqueeze(0)
                    loss = criterion(adj_outputs, labels); tta_outputs_list.append(adj_outputs)
                # --- TTA Logic (remains same) ---
                if TTA_TRANSFORMS and fold_num != "HPO": # Disable TTA during HPO for speed
                    try:
                        if isinstance(TTA_TRANSFORMS, transforms.Compose): inputs_tta = torch.stack([TTA_TRANSFORMS(img) for img in inputs.cpu()]).to(device)
                        else: inputs_tta = None; print(f"{TermColors.YELLOW}Warn: Skip TTA flip (unsupported type).{TermColors.ENDC}")
                        if inputs_tta is not None:
                            with autocast(enabled=MIXED_PRECISION):
                                embed_tta = current_model(inputs_tta)
                                if hasattr(current_model, 'metric_fc'): out_tta = current_model.metric_fc(embed_tta, labels)
                                elif hasattr(current_model, 'fc'): out_tta = current_model(inputs_tta)
                                else: out_tta = current_model(inputs_tta)

                                adj_out_tta = out_tta
                                if IMBALANCE_STRATEGY == 'LogitAdjust' and CLASS_PRIORS is not None: adj_out_tta = out_tta + logit_adj.unsqueeze(0)
                                tta_outputs_list.append(adj_out_tta)
                    except Exception as e: print(f"{TermColors.YELLOW}Warn: TTA failed for {model_key}: {e}.{TermColors.ENDC}")

                final_outputs = torch.stack(tta_outputs_list).mean(dim=0) if len(tta_outputs_list) > 1 else tta_outputs_list[0]
                running_loss += loss.item() * batch_size; total_samples += batch_size

                # Store raw outputs (probabilities) and labels for stacking (only for standard model, not during HPO)
                if model_key == 'standard' and return_preds and fold_num != "HPO":
                    all_preds_raw.append(F.softmax(final_outputs, dim=1).detach().cpu()) # Store probabilities
                    all_labels_raw.append(labels.detach().cpu())

                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / total_samples if total_samples > 0 else 0

        # Calculate accuracy using argmax on final_outputs (even if storing raw for stacking)
        # Need to re-run prediction collection for accuracy if not storing argmax during loop
        temp_preds_acc, temp_labels_acc = [], []
        current_model.eval()
        with torch.no_grad():
             for batch_data_acc in dataloader: # Re-iterate quickly for accuracy calc
                 if len(batch_data_acc) == 3: inputs_acc, labels_acc, _ = batch_data_acc
                 else: inputs_acc, labels_acc = batch_data_acc
                 inputs_acc, labels_acc = inputs_acc.to(device), labels_acc.to(device)
                 with autocast(enabled=MIXED_PRECISION):
                     embeddings = current_model(inputs_acc)
                     if hasattr(current_model, 'metric_fc'): outputs = current_model.metric_fc(embeddings, labels_acc)
                     elif hasattr(current_model, 'fc'): outputs = current_model(inputs_acc)
                     else: outputs = current_model(inputs_acc)
                     adj_outputs = outputs
                     if IMBALANCE_STRATEGY == 'LogitAdjust' and CLASS_PRIORS is not None:
                         logit_adj = LOGIT_ADJUSTMENT_TAU * torch.log(CLASS_PRIORS + 1e-12)
                         adj_outputs = outputs + logit_adj.unsqueeze(0)
                 temp_preds_acc.append(torch.argmax(adj_outputs, dim=1).detach())
                 temp_labels_acc.append(labels_acc.detach())
        all_preds_acc = torch.cat(temp_preds_acc)
        all_labels_acc = torch.cat(temp_labels_acc)
        epoch_acc = (all_preds_acc == all_labels_acc).sum().item() / total_samples if total_samples > 0 else 0


        # Only log to writer if provided (not during HPO objective)
        if writer:
            writer.add_scalar(f'Loss/val_{model_key}', epoch_loss, global_epoch)
            writer.add_scalar(f'Accuracy/val_{model_key}', epoch_acc, global_epoch)
        results[model_key] = {'loss': epoch_loss, 'acc': epoch_acc}
        print(f"  Val ({model_key}): Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}")

        # Store OOF preds from the standard model if requested (and not HPO)
        if model_key == 'standard' and return_preds and fold_num != "HPO":
            fold_oof_preds = torch.cat(all_preds_raw).numpy()
            fold_oof_labels = torch.cat(all_labels_raw).numpy()

    # Return results for the 'standard' model for checkpointing/HPO, plus OOF preds/labels if requested
    if return_preds and fold_num != "HPO":
        return results['standard']['loss'], results['standard']['acc'], fold_oof_preds, fold_oof_labels
    else:
        # For HPO, just return the accuracy of the standard model
        return results['standard']['loss'], results['standard']['acc']


# --- Knowledge Distillation Function ---
# ... (distillation_loss, build_student_model, train_student_model) ...
def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    """Calculates the KD loss."""
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    soft_prob = F.log_softmax(student_logits / temperature, dim=1)
    kd_loss = -torch.sum(soft_targets * soft_prob, dim=1).mean() * (temperature**2)
    # Standard CE loss with labels
    label_loss = F.cross_entropy(student_logits, labels)
    return alpha * kd_loss + (1. - alpha) * label_loss

def build_student_model(model_name, num_classes, pretrained=True, embedding_size=None, dropout_rate=0.2):
    """Builds a simpler student model using timm."""
    print(f"{TermColors.CYAN}ℹ Building Student Model: {model_name}{TermColors.ENDC}")
    try:
        # Create a standard timm model, potentially replacing the classifier
        student_model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        num_features = student_model.num_features
        print(f"  Student base features: {num_features}")

        # Add optional embedding layer + ArcFace or just a simple FC layer
        if embedding_size and embedding_size > 0:
            print(f"  Adding Embedding ({embedding_size}) + ArcFace head to student.")
            student_model.embedding = nn.Linear(num_features, embedding_size)
            student_model.bn = nn.BatchNorm1d(embedding_size)
            student_model.dropout = nn.Dropout(dropout_rate)
            student_model.metric_fc = ArcFace(embedding_size, num_classes, s=ARCFACE_S, m=ARCFACE_M, easy_margin=ARCFACE_EASY_MARGIN) # Use same ArcFace params
            # Modify forward to use the new head
            def forward_student_arcface(self, x):
                x = self.forward_features(x)
                x = self.embedding(x)
                x = self.bn(x)
                x = self.dropout(x)
                # metric_fc is applied outside in the loss calculation
                return x
            student_model.forward = forward_student_arcface.__get__(student_model, student_model.__class__)

        else:
            print(f"  Adding simple Linear head to student.")
            student_model.fc = nn.Linear(num_features, num_classes) # Simple linear classifier
            # Modify forward if necessary (often timm models handle num_classes=0 correctly and expect a final layer)
            # Check if the default forward needs override, usually not if just adding .fc

        return student_model

    except Exception as e:
        print(f"{TermColors.RED}❌ Failed to build student model {model_name}: {e}{TermColors.ENDC}")
        raise e

def train_student_model(teacher_model_path, student_model_name, student_save_path, df_train, df_val, image_dir, label_encoder, num_classes):
    """Trains a student model using knowledge distillation."""
    print(f"\n{TermColors.HEADER}===== Starting Knowledge Distillation =====")
    print(f"Teacher: {teacher_model_path}, Student: {student_model_name}")

    # --- Load Teacher Model ---
    print(f"{TermColors.CYAN}ℹ Loading Teacher model...{TermColors.ENDC}")
    try:
        # Build teacher architecture (CombinedModel) - Use potentially HPO-tuned MODEL_NAMES
        # Need to load the specific architecture used for the teacher fold
        # For simplicity, assume the default MODEL_NAMES were used for the teacher fold (KD_TEACHER_FOLD)
        # A more robust approach would save the exact config with the checkpoint.
        teacher_model = build_model(model_names=MODEL_NAMES, num_classes=num_classes) # Use global MODEL_NAMES here

        # Load state dict - handle potential compile/module prefixes
        ckpt = torch.load(teacher_model_path, map_location=DEVICE)
        state_dict = ckpt['state_dict']; new_state_dict = {}
        is_compiled = False # Assume not compiled when loading for inference/teaching
        for k, v in state_dict.items():
            name = k
            if name.startswith('module.'): name = name[len('module.'):]
            if name.startswith('_orig_mod.'): name = name[len('_orig_mod.'):] # Strip compile prefix if present
            new_state_dict[name] = v
        teacher_model.load_state_dict(new_state_dict, strict=True) # Be strict loading teacher
        teacher_model = teacher_model.to(DEVICE); teacher_model.eval() # Set to eval mode
        print(f"{TermColors.GREEN}✅ Teacher model loaded.{TermColors.ENDC}")
    except Exception as e:
        print(f"{TermColors.RED}❌ Failed to load teacher model: {e}{TermColors.ENDC}"); traceback.print_exc(); return

    # --- Build Student Model ---
    try:
        student_model = build_student_model(student_model_name, num_classes, pretrained=True,
                                            embedding_size=KD_STUDENT_EMBEDDING_SIZE, dropout_rate=KD_STUDENT_DROPOUT)
        student_model = student_model.to(DEVICE)
    except Exception as e:
        print(f"{TermColors.RED}❌ Failed to build student model: {e}{TermColors.ENDC}"); traceback.print_exc(); return

    # --- DataLoaders for Student ---
    print(f"{TermColors.CYAN}ℹ Creating KD dataloaders (Size: {KD_STUDENT_IMAGE_SIZE})...{TermColors.ENDC}")
    try:
        train_tf_kd, val_tf_kd = get_transforms(image_size=KD_STUDENT_IMAGE_SIZE)
        train_ds_kd = PlantDataset(df_train, image_dir, train_tf_kd, label_encoder, False, KD_STUDENT_IMAGE_SIZE)
        val_ds_kd = PlantDataset(df_val, image_dir, val_tf_kd, label_encoder, False, KD_STUDENT_IMAGE_SIZE)
        train_ds_kd.fold = "KD"; val_ds_kd.fold = "KD" # Add identifier

        sampler_kd = None # Optional: Add weighted sampler for KD training too
        # if IMBALANCE_STRATEGY == 'WeightedSampler': ... (add sampler logic if desired)

        train_loader_kd = DataLoader(train_ds_kd, KD_BATCH_SIZE, sampler=sampler_kd, shuffle=(sampler_kd is None), num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
        val_loader_kd = DataLoader(val_ds_kd, KD_BATCH_SIZE*2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        print(f"{TermColors.GREEN}✅ KD Dataloaders ready.{TermColors.ENDC}")
    except Exception as e:
        print(f"{TermColors.RED}❌ KD Dataloader Error: {e}{TermColors.ENDC}"); traceback.print_exc(); return

    # --- KD Training Setup ---
    optimizer_kd = optim.AdamW(student_model.parameters(), lr=KD_LR, weight_decay=WEIGHT_DECAY)
    scheduler_kd = CosineAnnealingLR(optimizer_kd, T_max=KD_EPOCHS, eta_min=KD_LR * 0.01)
    scaler_kd = GradScaler(enabled=MIXED_PRECISION)
    best_val_acc_kd = 0.0
    kd_log_dir = os.path.join(BASE_LOG_DIR, "knowledge_distillation")
    writer_kd = SummaryWriter(log_dir=kd_log_dir)

    # --- KD Training Loop ---
    print(f"{TermColors.CYAN}ℹ Starting KD Training for {KD_EPOCHS} epochs...{TermColors.ENDC}")
    for epoch in range(KD_EPOCHS):
        student_model.train(); teacher_model.eval() # Ensure modes are correct
        running_loss_kd = 0.0; total_samples_kd = 0
        progress_bar_kd = tqdm(train_loader_kd, desc=f"KD Epoch {epoch+1}/{KD_EPOCHS}", leave=False)

        for inputs, labels in progress_bar_kd:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            batch_size = inputs.size(0)
            optimizer_kd.zero_grad()

            with autocast(enabled=MIXED_PRECISION):
                # Get teacher logits (no grad needed)
                with torch.no_grad():
                    teacher_embeddings = teacher_model(inputs)
                    if hasattr(teacher_model, 'metric_fc'): teacher_logits = teacher_model.metric_fc(teacher_embeddings, labels)
                    else: raise AttributeError("Teacher needs metric_fc for KD") # Assuming ArcFace teacher

                # Get student logits/embeddings
                student_outputs = student_model(inputs) # Might be embeddings or logits

                # Get student logits specifically for loss calculation
                if hasattr(student_model, 'metric_fc'): # If student uses ArcFace
                    student_logits = student_model.metric_fc(student_outputs, labels)
                elif hasattr(student_model, 'fc'): # If student has simple FC
                    student_logits = student_outputs # Assumes model(inputs) returned logits
                else: # Fallback
                     student_logits = student_outputs # Assumes model(inputs) returned logits

                loss = distillation_loss(student_logits, teacher_logits, labels, KD_TEMPERATURE, KD_ALPHA)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"{TermColors.RED}Warn: NaN/Inf KD loss. Skip batch.{TermColors.ENDC}"); continue

            scaler_kd.scale(loss).backward()
            scaler_kd.step(optimizer_kd)
            scaler_kd.update()

            running_loss_kd += loss.item() * batch_size
            total_samples_kd += batch_size
            progress_bar_kd.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer_kd.param_groups[0]['lr']:.1E}")

        epoch_loss_kd = running_loss_kd / total_samples_kd if total_samples_kd > 0 else 0
        scheduler_kd.step()

        # --- KD Validation ---
        student_model.eval()
        running_val_loss_kd = 0.0; total_val_samples_kd = 0
        all_val_preds_kd, all_val_labels_kd = [], []
        progress_bar_val_kd = tqdm(val_loader_kd, desc=f"KD Val Epoch {epoch+1}", leave=False)
        criterion_val = nn.CrossEntropyLoss() # Use standard CE for validation accuracy

        with torch.no_grad():
            for inputs_val, labels_val in progress_bar_val_kd:
                inputs_val, labels_val = inputs_val.to(DEVICE), labels_val.to(DEVICE)
                batch_size_val = inputs_val.size(0)
                with autocast(enabled=MIXED_PRECISION):
                    student_outputs_val = student_model(inputs_val)
                    # Get logits for validation loss/accuracy
                    if hasattr(student_model, 'metric_fc'): student_logits_val = student_model.metric_fc(student_outputs_val, labels_val)
                    elif hasattr(student_model, 'fc'): student_logits_val = student_outputs_val
                    else: student_logits_val = student_outputs_val

                    val_loss = criterion_val(student_logits_val, labels_val)

                running_val_loss_kd += val_loss.item() * batch_size_val
                total_val_samples_kd += batch_size_val
                all_val_preds_kd.append(torch.argmax(student_logits_val, dim=1).detach())
                all_val_labels_kd.append(labels_val.detach())

        epoch_val_loss_kd = running_val_loss_kd / total_val_samples_kd if total_val_samples_kd > 0 else 0
        all_val_preds_kd = torch.cat(all_val_preds_kd)
        all_val_labels_kd = torch.cat(all_val_labels_kd)
        epoch_val_acc_kd = (all_val_preds_kd == all_val_labels_kd).sum().item() / total_val_samples_kd if total_val_samples_kd > 0 else 0

        print(f"KD Epoch {epoch+1}: Train Loss={epoch_loss_kd:.4f} | Val Loss={epoch_val_loss_kd:.4f} | Val Acc={epoch_val_acc_kd:.4f}")
        writer_kd.add_scalar('Loss/train_kd', epoch_loss_kd, epoch)
        writer_kd.add_scalar('Loss/val_kd', epoch_val_loss_kd, epoch)
        writer_kd.add_scalar('Accuracy/val_kd', epoch_val_acc_kd, epoch)
        writer_kd.add_scalar('LearningRate/kd', optimizer_kd.param_groups[0]['lr'], epoch)

        if epoch_val_acc_kd > best_val_acc_kd:
            best_val_acc_kd = epoch_val_acc_kd
            print(f"{TermColors.OKGREEN}🏆 New Best KD Val Acc: {best_val_acc_kd:.4f}. Saving student model...{TermColors.ENDC}")
            # Save student model state dict (no fold needed)
            model_to_save = student_model
            if hasattr(model_to_save, 'module'): model_to_save = model_to_save.module
            if hasattr(model_to_save, '_orig_mod'): model_to_save = model_to_save._orig_mod
            torch.save(model_to_save.state_dict(), student_save_path)

        if DEVICE.type == 'cuda': torch.cuda.empty_cache(); gc.collect()

    writer_kd.close()
    print(f"{TermColors.GREEN}✅ Knowledge Distillation finished. Best Val Acc: {best_val_acc_kd:.4f}{TermColors.ENDC}")
    print(f"Student model saved to: {student_save_path}")
    # Clean up KD resources
    del teacher_model, student_model, optimizer_kd, scheduler_kd, scaler_kd, train_loader_kd, val_loader_kd, train_ds_kd, val_ds_kd, writer_kd
    if DEVICE.type == 'cuda': torch.cuda.empty_cache(); gc.collect()


# --- Stacking Ensemble Function ---
# ... (train_stacking_meta_model) ...
def train_stacking_meta_model(oof_preds, oof_labels, save_path):
    """Trains a simple meta-model (Logistic Regression) on OOF predictions."""
    print(f"\n{TermColors.HEADER}===== Starting Stacking Meta-Model Training =====")
    if oof_preds is None or oof_labels is None or len(oof_preds) == 0:
        print(f"{TermColors.RED}❌ No OOF predictions found. Skipping stacking.{TermColors.ENDC}")
        return

    print(f"OOF Predictions shape: {oof_preds.shape}") # Should be (num_val_samples, num_classes)
    print(f"OOF Labels shape: {oof_labels.shape}")   # Should be (num_val_samples,)

    try:
        # Use Logistic Regression as the meta-model
        # Increase max_iter for potentially better convergence on many classes/features
        meta_model = LogisticRegression(solver='liblinear', C=1.0, random_state=SEED, max_iter=500)

        print(f"{TermColors.CYAN}ℹ Training Logistic Regression meta-model...{TermColors.ENDC}")
        meta_model.fit(oof_preds, oof_labels)

        # Evaluate meta-model on the OOF data itself (provides an estimate of performance)
        oof_accuracy = meta_model.score(oof_preds, oof_labels)
        print(f"{TermColors.GREEN}✅ Meta-model trained. OOF Accuracy: {oof_accuracy:.4f}{TermColors.ENDC}")

        # Save the trained meta-model
        joblib.dump(meta_model, save_path)
        print(f"Meta-model saved to: {save_path}")

    except Exception as e:
        print(f"{TermColors.RED}❌ Failed to train or save stacking meta-model: {e}{TermColors.ENDC}")
        traceback.print_exc()

    # Clean up
    del meta_model, oof_preds, oof_labels
    gc.collect()

# --- HPO Objective Function ---
# ... (objective) ...
def objective(trial: optuna.Trial, df_train_hpo, df_val_hpo, image_dir, label_encoder, num_classes):
    """Optuna objective function for HPO."""
    global stop_requested # Allow checking for interrupt during HPO
    if stop_requested: raise optuna.exceptions.TrialPruned("Stop requested")

    # --- Suggest Hyperparameters ---
    hpo_lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    hpo_wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    hpo_dropout = trial.suggest_float("dropout_rate", 0.1, 0.6)
    hpo_arcface_m = trial.suggest_float("arcface_margin", 0.2, 0.8)
    hpo_optimizer_type = trial.suggest_categorical("optimizer", ["AdamW", "AdamP"] if ADAMP_AVAILABLE else ["AdamW"])
    hpo_model_name = trial.suggest_categorical("architecture", HPO_ARCHITECTURES_TO_TRY)
    # Optional: Tune embedding size? (Might require adjusting CombinedModel more)
    # hpo_embedding_size = trial.suggest_categorical("embedding_size", [256, 512, 768])

    print(f"\n{TermColors.CYAN}--- HPO Trial {trial.number} ---")
    print(f"  Params: LR={hpo_lr:.1E}, WD={hpo_wd:.1E}, Dropout={hpo_dropout:.3f}, ArcM={hpo_arcface_m:.3f}, Optim={hpo_optimizer_type}, Arch={hpo_model_name}")

    # --- Setup Trial ---
    try:
        # Build model with suggested architecture (single backbone for HPO)
        # Use a fixed embedding size for simplicity during HPO, or tune it as well
        model_hpo = build_model(model_names=hpo_model_name, num_classes=num_classes, dropout_rate=hpo_dropout,
                                embedding_size=512, arcface_m=hpo_arcface_m) # Fixed embedding size
        model_hpo = model_hpo.to(DEVICE)

        # DataLoaders for HPO trial
        train_tf_hpo, val_tf_hpo = get_transforms(image_size=HPO_IMAGE_SIZE)
        train_ds_hpo = PlantDataset(df_train_hpo, image_dir, train_tf_hpo, label_encoder, False, HPO_IMAGE_SIZE)
        val_ds_hpo = PlantDataset(df_val_hpo, image_dir, val_tf_hpo, label_encoder, False, HPO_IMAGE_SIZE)
        train_ds_hpo.fold = "HPO"; val_ds_hpo.fold = "HPO" # Identifier

        # Use standard sampler for HPO trials for simplicity, or WeightedSampler if desired
        train_loader_hpo = DataLoader(train_ds_hpo, HPO_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
        val_loader_hpo = DataLoader(val_ds_hpo, HPO_BATCH_SIZE*2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        # Optimizer, Criterion, Scaler
        optimizer_hpo = get_optimizer(model_hpo, optimizer_type=hpo_optimizer_type, lr=hpo_lr, weight_decay=hpo_wd, use_sam=False) # Disable SAM for HPO speed
        criterion_hpo = get_criterion(loss_type=LOSS_TYPE, label_smoothing=LABEL_SMOOTHING, class_weights=None) # No weights for HPO simplicity
        scaler_hpo = GradScaler(enabled=MIXED_PRECISION)

    except Exception as e:
        print(f"{TermColors.RED}❌ HPO Trial {trial.number} Setup Error: {e}. Pruning.{TermColors.ENDC}")
        raise optuna.exceptions.TrialPruned(f"Setup failed: {e}")

    # --- Training Loop for HPO Trial ---
    best_trial_acc = 0.0
    for epoch in range(HPO_EPOCHS_PER_TRIAL):
        if stop_requested: raise optuna.exceptions.TrialPruned("Stop requested")
        print(f"  HPO Trial {trial.number} Epoch {epoch+1}/{HPO_EPOCHS_PER_TRIAL}")
        train_loss, train_acc = train_one_epoch(model_hpo, train_loader_hpo, criterion_hpo, optimizer_hpo, scaler_hpo, None, # No scheduler for short HPO
                                                epoch, 0, epoch, HPO_EPOCHS_PER_TRIAL, DEVICE, None, num_classes, None, # No writer, EMA
                                                use_sam=False, fold_num="HPO") # Pass HPO identifier
        if train_loss is None: raise optuna.exceptions.TrialPruned("Stop requested during train")

        val_loss, val_acc = validate_one_epoch(model_hpo, val_loader_hpo, criterion_hpo, DEVICE, epoch, None, num_classes, # No writer
                                               swa_model=None, ema_model=None, return_preds=False, fold_num="HPO") # Pass HPO identifier

        print(f"  HPO Trial {trial.number} Ep {epoch+1}: Train L={train_loss:.4f} A={train_acc:.4f} | Val L={val_loss:.4f} A={val_acc:.4f}")
        best_trial_acc = max(best_trial_acc, val_acc)

        # --- Optuna Pruning (Optional but recommended) ---
        trial.report(val_acc, epoch) # Report intermediate value
        if trial.should_prune():
            print(f"{TermColors.YELLOW}  HPO Trial {trial.number} Pruned at epoch {epoch+1}.{TermColors.ENDC}")
            # Clean up before pruning
            del model_hpo, optimizer_hpo, criterion_hpo, scaler_hpo, train_loader_hpo, val_loader_hpo, train_ds_hpo, val_ds_hpo
            if DEVICE.type == 'cuda': torch.cuda.empty_cache()
            gc.collect()
            raise optuna.exceptions.TrialPruned()

        if DEVICE.type == 'cuda': torch.cuda.empty_cache(); gc.collect()

    # --- Cleanup Trial Resources ---
    del model_hpo, optimizer_hpo, criterion_hpo, scaler_hpo, train_loader_hpo, val_loader_hpo, train_ds_hpo, val_ds_hpo
    if DEVICE.type == 'cuda': torch.cuda.empty_cache()
    gc.collect()

    print(f"--- HPO Trial {trial.number} Finished. Best Val Acc: {best_trial_acc:.4f} ---")
    return best_trial_acc # Return the best validation accuracy achieved during the trial


# --- Main Training Function ---
# ... (main) ...
def main():
    # Make globals modifiable
    global stop_requested, label_encoder, class_names, CLASS_FREQUENCIES, CLASS_PRIORS, CLASS_WEIGHTS, CURRENT_IMAGE_SIZE
    global LEARNING_RATE, WEIGHT_DECAY, DROPOUT_RATE, ARCFACE_M, OPTIMIZER_TYPE, MODEL_NAMES, EMBEDDING_SIZE
    global RUN_STACKING, RUN_KNOWLEDGE_DISTILLATION, NUM_CLASSES # Allow modification if HPO fails or NUM_CLASSES needs update

    set_seed(SEED); signal.signal(signal.SIGINT, handle_interrupt)
    print(f"{TermColors.HEADER}===== Plant Recognition Training (PyTorch - Commercial Grade Features) ===={TermColors.ENDC}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # Initial config print will show defaults
    print(f"Initial Config: Folds={N_FOLDS}, Models={MODEL_NAMES}, Stages={len(PROGRESSIVE_RESIZING_STAGES)}, TotalEpPerFold={TOTAL_EPOCHS_PER_FOLD}, Batch={BATCH_SIZE}x{GRADIENT_ACCUMULATION_STEPS}, LR={LEARNING_RATE}, Optim={OPTIMIZER_TYPE}(SAM:{USE_SAM}), Loss={LOSS_TYPE}(LS:{LABEL_SMOOTHING}), Imbalance={IMBALANCE_STRATEGY}, Sched={SCHEDULER_TYPE}, Device={DEVICE}, MP={MIXED_PRECISION}, Compile={USE_TORCH_COMPILE}, Augs=Mixup/Cutmix/RandAug, SWA={USE_SWA}, EMA={USE_EMA}, TTA=Yes, Stacking={RUN_STACKING}, KD={RUN_KNOWLEDGE_DISTILLATION}, HPO={RUN_HPO}({HPO_N_TRIALS} trials)")
    print(f"------------------------------------------------")

    # --- Data Loading (Load full dataset once) ---
    print(f"\n{TermColors.HEADER}--- STEP 1: Load Full Dataset ---{TermColors.ENDC}")
    try:
        # Change separator from '\t' to ',' based on the error message
        # Remove quoting=csv.QUOTE_NONE as it's likely not needed for comma separation
        print(f"{TermColors.CYAN}ℹ Reading CSV: {CSV_PATH}{TermColors.ENDC}")
        df_full = pd.read_csv(
            CSV_PATH,
            sep=',', # Changed separator to comma
            low_memory=False,
            on_bad_lines='skip' # Keep skipping bad lines for robustness
            # quoting=csv.QUOTE_NONE # Removed this, default quoting is usually fine for CSV
        )
        print(f"{TermColors.GREEN}✅ CSV loaded initially. Shape: {df_full.shape}{TermColors.ENDC}")
        print(f"Columns found: {df_full.columns.tolist()}") # Log columns found

        # --- Data Cleaning and Preprocessing ---
        print(f"{TermColors.CYAN}ℹ Cleaning and preprocessing data...{TermColors.ENDC}")
        # Check expected columns exist before proceeding
        # Use 'scientific_name' as it appears in the error message header list
        required_cols = ['image_url', 'scientific_name']
        if not all(col in df_full.columns for col in required_cols):
            print(f"{TermColors.RED}❌ Error: Missing required columns in CSV after changing separator. Found: {df_full.columns.tolist()}. Required: {required_cols}{TermColors.ENDC}")
            # Add specific checks for common variations if needed
            if 'scientificName' in df_full.columns and 'scientific_name' not in df_full.columns:
                 print(f"{TermColors.YELLOW}⚠️ Found 'scientificName' instead of 'scientific_name'. Renaming.{TermColors.ENDC}")
                 df_full.rename(columns={'scientificName': 'scientific_name'}, inplace=True)
                 required_cols = ['image_url', 'scientific_name'] # Update required cols if renamed
            elif len(df_full.columns) >= 31: # Check if enough columns exist to guess based on position
                 # Example: Guessing based on typical column order (adjust indices if needed)
                 image_url_idx = 9 # Index for 'image_url' in the error message list
                 sci_name_idx = 30 # Index for 'scientific_name' in the error message list
                 if image_url_idx < len(df_full.columns) and sci_name_idx < len(df_full.columns):
                     print(f"{TermColors.YELLOW}⚠️ Attempting to use columns at index {image_url_idx} and {sci_name_idx} as image_url and scientific_name.{TermColors.ENDC}")
                     df_full = df_full.iloc[:, [image_url_idx, sci_name_idx]]
                     df_full.columns = required_cols
                 else:
                     print(f"{TermColors.RED}❌ Cannot reliably guess columns by index.{TermColors.ENDC}")
                     sys.exit(1)
            else:
                 print(f"{TermColors.RED}❌ Cannot proceed without required columns.{TermColors.ENDC}")
                 sys.exit(1)


        # Ensure we use the correct column name 'scientific_name' going forward
        df_full = df_full[required_cols].dropna()
        df_full['image_path'] = df_full['image_url'].apply(os.path.basename)
        # Rename 'scientific_name' to 'scientificName' for consistency with the rest of the script
        df_full.rename(columns={'scientific_name': 'scientificName'}, inplace=True)
        df_full = df_full[['image_path', 'scientificName']] # Use 'scientificName' now

        min_samples = 10
        class_counts = df_full['scientificName'].value_counts()
        valid_classes = class_counts[class_counts >= min_samples].index
        df_full = df_full[df_full['scientificName'].isin(valid_classes)].reset_index(drop=True)
        print(f"Full Filtered DF: {df_full.shape}, Classes: {len(valid_classes)}")
        if DEBUG_MODE: print(f"{TermColors.YELLOW}DEBUG MODE: Small subset.{TermColors.ENDC}"); df_full = df_full.sample(n=min(2000, len(df_full)), random_state=SEED).reset_index(drop=True)
        label_encoder = LabelEncoder(); df_full['label'] = label_encoder.fit_transform(df_full['scientificName'])
        class_names = list(label_encoder.classes_); NUM_CLASSES = len(class_names) # Update global NUM_CLASSES
        print(f"Classes after encoding: {NUM_CLASSES}")
        mapping_path = os.path.join(BASE_MODEL_SAVE_DIR, "label_mapping.json"); # Save mapping once
        os.makedirs(BASE_MODEL_SAVE_DIR, exist_ok=True)
        with open(mapping_path, 'w') as f: json.dump(dict(zip(range(NUM_CLASSES), class_names)), f, indent=4)
        print(f"Label mapping saved.")

        # Calculate imbalance stats on the full dataset (more stable)
        label_counts = df_full['label'].value_counts().sort_index(); total_samples_full = len(df_full)
        freqs = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        for i in range(NUM_CLASSES): freqs[i] = label_counts.get(i, 0)
        CLASS_FREQUENCIES = freqs.to(DEVICE); CLASS_PRIORS = (CLASS_FREQUENCIES / total_samples_full).to(DEVICE)
        class_weights_array = sk_class_weight.compute_class_weight('balanced', classes=np.unique(df_full['label']), y=df_full['label'])
        CLASS_WEIGHTS = torch.tensor(class_weights_array, dtype=torch.float32)
        print(f"Class priors and weights calculated for imbalance handling.")
        if IMBALANCE_STRATEGY == 'LogitAdjust': print(f"  Using Logit Adjustment (tau={LOGIT_ADJUSTMENT_TAU})")
        elif IMBALANCE_STRATEGY == 'WeightedLoss': print(f"  Using Weighted Loss")
        elif IMBALANCE_STRATEGY == 'WeightedSampler': print(f"  Using Weighted Sampler")
        else: print(f"  No specific imbalance strategy selected.")

    except FileNotFoundError:
        # ... (exception handling remains the same) ...
        print(f"{TermColors.RED}❌ Data Error: CSV file not found at {CSV_PATH}{TermColors.ENDC}")
        traceback.print_exc()
        sys.exit(1)
    except pd.errors.EmptyDataError:
         # ... (exception handling remains the same) ...
         print(f"{TermColors.RED}❌ Data Error: CSV file is empty at {CSV_PATH}{TermColors.ENDC}")
         traceback.print_exc()
         sys.exit(1)
    except Exception as e:
        # ... (exception handling remains the same) ...
        print(f"{TermColors.RED}❌ Data Loading/Preprocessing Error: {e}{TermColors.ENDC}")
        traceback.print_exc()
        sys.exit(1)

    # --- HPO Phase ---
    if RUN_HPO:
        print(f"\n{TermColors.HEADER}--- STEP 2: Hyperparameter Optimization ({HPO_N_TRIALS} Trials) ---{TermColors.ENDC}")
        try:
            # Use a subset of data for HPO (e.g., Fold 0)
            skf_hpo = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED) # Use same split strategy
            train_idx_hpo, val_idx_hpo = next(iter(skf_hpo.split(df_full, df_full['label'])))
            df_train_hpo = df_full.iloc[train_idx_hpo]
            df_val_hpo = df_full.iloc[val_idx_hpo]
            print(f"Using Fold 0 for HPO: Train={len(df_train_hpo)}, Val={len(df_val_hpo)}")

            # Create or load Optuna study
            study = optuna.create_study(
                study_name=HPO_STUDY_NAME,
                storage=HPO_STORAGE_DB,
                direction="maximize", # Maximize validation accuracy
                load_if_exists=True,
                pruner=optuna.pruners.MedianPruner() # Example pruner
            )

            # Define the objective function with necessary data
            objective_with_data = lambda trial: objective(trial, df_train_hpo, df_val_hpo, IMAGE_DIR, label_encoder, NUM_CLASSES)

            # Run optimization
            study.optimize(objective_with_data, n_trials=HPO_N_TRIALS, timeout=None) # No timeout

            # --- Apply Best Parameters ---
            best_trial = study.best_trial
            print(f"\n{TermColors.OKGREEN}===== HPO Finished =====")
            print(f"Best Trial Number: {best_trial.number}")
            print(f"Best Value (Val Acc): {best_trial.value:.4f}")
            print("Best Parameters:")
            for key, value in best_trial.params.items():
                print(f"  {key}: {value}")

            # Update global config variables
            LEARNING_RATE = best_trial.params['lr']
            WEIGHT_DECAY = best_trial.params['weight_decay']
            DROPOUT_RATE = best_trial.params['dropout_rate']
            ARCFACE_M = best_trial.params['arcface_margin']
            OPTIMIZER_TYPE = best_trial.params['optimizer']
            # Use the single best architecture found by HPO for the main training
            MODEL_NAMES = [best_trial.params['architecture']] # Use single best model
            # EMBEDDING_SIZE = best_trial.params.get('embedding_size', EMBEDDING_SIZE) # If embedding size was tuned

            print("\nUpdated Configuration for Main Training:")
            print(f"  LEARNING_RATE: {LEARNING_RATE:.1E}")
            print(f"  WEIGHT_DECAY: {WEIGHT_DECAY:.1E}")
            print(f"  DROPOUT_RATE: {DROPOUT_RATE:.3f}")
            print(f"  ARCFACE_M: {ARCFACE_M:.3f}")
            print(f"  OPTIMIZER_TYPE: {OPTIMIZER_TYPE}")
            print(f"  MODEL_NAMES: {MODEL_NAMES}")
            # print(f"  EMBEDDING_SIZE: {EMBEDDING_SIZE}")

            # Clean up HPO dataframes
            del df_train_hpo, df_val_hpo, skf_hpo
            gc.collect()

        except Exception as e:
            print(f"{TermColors.RED}❌ HPO Error: {e}. Proceeding with default parameters.{TermColors.ENDC}")
            traceback.print_exc()
            # Optionally disable subsequent steps if HPO fails critically
            # RUN_STACKING = False
            # RUN_KNOWLEDGE_DISTILLATION = False
    else:
        print(f"{TermColors.YELLOW}⏩ Skipping HPO phase.{TermColors.ENDC}")


    # --- Cross-Validation Loop ---
    print(f"\n{TermColors.HEADER}--- STEP 3: Main K-Fold Cross-Validation ---{TermColors.ENDC}")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_results = defaultdict(list)
    all_oof_preds = [None] * len(df_full) # List to store OOF preds for each sample
    all_oof_labels = [None] * len(df_full) # List to store true labels corresponding to OOF preds
    oof_indices_collected = [] # Keep track of which indices have OOF preds

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_full, df_full['label'])):
        print(f"\n{TermColors.HEADER}===== Starting Fold {fold+1}/{N_FOLDS} ====={TermColors.ENDC}")
        train_df = df_full.iloc[train_idx]; val_df = df_full.iloc[val_idx]
        print(f"Fold {fold+1} - Train: {len(train_df)}, Val: {len(val_df)}")

        # --- Build Model, Loss, Optimizer, Scaler for each fold (using potentially updated HPO params) ---
        print(f"\n{TermColors.BOLD}--- Fold {fold+1}: Setup Training Components ---{TermColors.ENDC}")
        try:
            # Build model using potentially updated MODEL_NAMES, DROPOUT_RATE, ARCFACE_M etc.
            model = build_model(model_names=MODEL_NAMES, num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE, arcface_m=ARCFACE_M)
            model = model.to(DEVICE)
            compiled_model_applied = False # Reset compile flag per fold
            if USE_TORCH_COMPILE and hasattr(torch, 'compile'): # Attempt compile per fold
                try:
                    if int(torch.__version__.split('.')[0]) >= 2:
                        print(f"{TermColors.YELLOW}⏳ Fold {fold+1}: Applying torch.compile()...{TermColors.ENDC}"); model = torch.compile(model, mode='default')
                        compiled_model_applied = True; print(f"{TermColors.GREEN}✅ Fold {fold+1}: torch.compile() applied.{TermColors.ENDC}")
                    else: print(f"{TermColors.YELLOW}⚠️ Fold {fold+1}: torch.compile() needs PyTorch 2.0+. Skip.{TermColors.ENDC}")
                except Exception as ce: print(f"{TermColors.RED}❌ Fold {fold+1}: Compile Error: {ce}. Proceeding without.{TermColors.ENDC}")
            elif USE_TORCH_COMPILE: print(f"{TermColors.YELLOW}⚠️ Fold {fold+1}: torch.compile() unavailable. Skip.{TermColors.ENDC}")

            crit_weights = CLASS_WEIGHTS if IMBALANCE_STRATEGY == 'WeightedLoss' else None
            criterion = get_criterion(loss_type=LOSS_TYPE, label_smoothing=LABEL_SMOOTHING, class_weights=crit_weights)
            # Use potentially updated LEARNING_RATE, WEIGHT_DECAY, OPTIMIZER_TYPE
            optimizer = get_optimizer(model, optimizer_type=OPTIMIZER_TYPE, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, use_sam=USE_SAM)
            scaler = GradScaler(enabled=MIXED_PRECISION); scheduler = None
            # SWA Model (reinitialized per fold)
            swa_model = AveragedModel(model) if USE_SWA else None
            # EMA Model (reinitialized per fold)
            ema_model = AveragedModel(model, avg_fn=ema_avg_fn) if USE_EMA else None

        except Exception as e: print(f"{TermColors.RED}❌ Fold {fold+1} Setup Error: {e}{TermColors.ENDC}"); traceback.print_exc(); continue # Skip fold on error

        # --- Checkpoint Loading for the fold ---
        print(f"\n{TermColors.BOLD}--- Fold {fold+1}: Load Checkpoint ---{TermColors.ENDC}")
        # Pass the potentially updated optimizer to load_checkpoint
        start_glob_ep, start_stg_idx, start_stg_ep, best_metric, loaded_labels, loaded_size, loaded_freqs = load_checkpoint(fold, model, optimizer, None, scaler)

        # --- TensorBoard Writer for the fold ---
        fold_log_dir = os.path.join(BASE_LOG_DIR, f"fold_{fold}")
        writer = SummaryWriter(log_dir=fold_log_dir)

        # --- Fold Training Loop (Progressive Resizing) ---
        print(f"\n{TermColors.BOLD}--- Fold {fold+1}: Start Training ---{TermColors.ENDC}")
        global_epoch_counter = start_glob_ep
        stop_requested = False # Reset stop flag for each fold
        fold_best_val_loss = float('inf')
        fold_best_val_acc = 0.0
        fold_best_oof_preds = None
        fold_best_oof_labels = None

        for stage_idx, (stage_epochs, stage_image_size) in enumerate(PROGRESSIVE_RESIZING_STAGES):
            if stage_idx < start_stg_idx: print(f"{TermColors.YELLOW}⏩ Fold {fold+1} Skip Stage {stage_idx+1}.{TermColors.ENDC}"); continue
            print(f"\n{TermColors.MAGENTA}===== Fold {fold+1} Stage {stage_idx+1}/{len(PROGRESSIVE_RESIZING_STAGES)}: {stage_epochs} Epochs @ {stage_image_size} ====={TermColors.ENDC}")
            CURRENT_IMAGE_SIZE = stage_image_size
            train_transform, val_transform = get_transforms(image_size=CURRENT_IMAGE_SIZE)
            print(f"{TermColors.CYAN}ℹ Fold {fold+1} Creating dataloaders size {CURRENT_IMAGE_SIZE}...{TermColors.ENDC}")
            try: # Create datasets/dataloaders for this fold and stage
                train_ds = PlantDataset(train_df, IMAGE_DIR, train_transform, label_encoder, False, CURRENT_IMAGE_SIZE)
                val_ds = PlantDataset(val_df, IMAGE_DIR, val_transform, label_encoder, False, CURRENT_IMAGE_SIZE)
                err_ds = PlantDataset(val_df.copy(), IMAGE_DIR, val_transform, label_encoder, True, CURRENT_IMAGE_SIZE)
                train_ds.fold = fold+1; val_ds.fold = fold+1; err_ds.fold = fold+1 # Add fold info for logging

                sampler = None
                if IMBALANCE_STRATEGY == 'WeightedSampler':
                    print(f"  Using WeightedRandomSampler for training data.")
                    labels_list = train_ds.get_labels()
                    class_sample_count = np.array([len(np.where(np.array(labels_list) == t)[0]) for t in np.unique(labels_list)])
                    # Handle potential zero counts for classes not in this fold's train set (rare with stratification but possible)
                    class_sample_count = np.maximum(class_sample_count, 1) # Avoid division by zero
                    weight = 1. / class_sample_count
                    samples_weight = np.array([weight[t] for t in labels_list])
                    samples_weight = torch.from_numpy(samples_weight).double()
                    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

                train_loader = DataLoader(train_ds, BATCH_SIZE, sampler=sampler, shuffle=(sampler is None), num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
                val_loader = DataLoader(val_ds, BATCH_SIZE*2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
                err_loader = DataLoader(err_ds, ERROR_LOG_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
                print(f"{TermColors.GREEN}✅ Fold {fold+1} Dataloaders ready.{TermColors.ENDC}")

            except Exception as e: print(f"{TermColors.RED}❌ Fold {fold+1} Dataloader Error: {e}{TermColors.ENDC}"); traceback.print_exc(); break # Stop fold on error

            # Adjust LR based on stage and potentially HPO-tuned base LR
            current_lr = LEARNING_RATE * (0.5 ** stage_idx); print(f"{TermColors.CYAN}ℹ Fold {fold+1} Stage LR: {current_lr:.1E}{TermColors.ENDC}")
            opt_to_adj = optimizer.base_optimizer if hasattr(optimizer, 'base_optimizer') else optimizer
            for pg in opt_to_adj.param_groups: pg['lr'] = current_lr
            # Adjust T_0 for CosineWarmRestarts based on stage epochs (e.g., first cycle is half the stage)
            current_t_0 = max(1, stage_epochs // 2) if SCHEDULER_TYPE == 'CosineWarmRestarts' else T_0
            # Pass potentially updated base LR (current_lr) to scheduler
            scheduler = get_scheduler(optimizer, SCHEDULER_TYPE, stage_epochs, WARMUP_EPOCHS, lr_max=current_lr, lr_min=current_lr*0.01,
                                      t_0=current_t_0, t_mult=T_MULT) # Pass updated LR params

            if stage_idx == start_stg_idx and start_stg_ep > 0: # Load scheduler state if resuming mid-stage
                 ckpt_path = os.path.join(BASE_CHECKPOINT_DIR, f"fold_{fold}", "latest_checkpoint.pth.tar")
                 if scheduler and os.path.isfile(ckpt_path):
                     print(f"{TermColors.CYAN}ℹ Fold {fold+1} Reload latest ckpt for scheduler...{TermColors.ENDC}")
                     ckpt_sched = torch.load(ckpt_path, map_location=DEVICE)
                     if 'scheduler' in ckpt_sched and ckpt_sched['scheduler']:
                         try: scheduler.load_state_dict(ckpt_sched['scheduler'])
                         except Exception as e: print(f"{TermColors.YELLOW}⚠️ Fold {fold+1} Sched reload failed: {e}.{TermColors.ENDC}")

            current_stage_start_epoch = start_stg_ep if stage_idx == start_stg_idx else 0
            swa_start_epoch_global = int(TOTAL_EPOCHS_PER_FOLD * SWA_START_EPOCH_GLOBAL_FACTOR)

            for stage_epoch in range(current_stage_start_epoch, stage_epochs):
                if stop_requested: break
                print(f"\n{TermColors.CYAN}--- Fold {fold+1} GlobEp {global_epoch_counter+1}/{TOTAL_EPOCHS_PER_FOLD} (Stg {stage_idx+1}: Ep {stage_epoch+1}/{stage_epochs}) ---{TermColors.ENDC}")
                train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, scheduler, global_epoch_counter, stage_idx, stage_epoch, stage_epochs, DEVICE, writer, NUM_CLASSES, ema_model, use_sam=USE_SAM, fold_num=fold+1)
                if train_loss is None: stop_requested = True; break

                # --- Validation and OOF Prediction Collection ---
                # Always return preds from validation of the standard model
                val_loss, val_acc, current_oof_preds, current_oof_labels = validate_one_epoch(
                    model, val_loader, criterion, DEVICE, global_epoch_counter, writer, NUM_CLASSES,
                    swa_model=swa_model, ema_model=ema_model, return_preds=True, fold_num=fold+1
                )

                # --- SWA Update ---
                current_swa_scheduler = None
                if USE_SWA and swa_model and global_epoch_counter >= swa_start_epoch_global:
                    swa_model.update() # Update SWA model average (uses internal counter)
                    if 'swa_scheduler' not in locals() or swa_scheduler is None: # Init SWA scheduler on first SWA epoch
                         print(f"{TermColors.CYAN}ℹ Fold {fold+1} Init SWA scheduler...{TermColors.ENDC}")
                         swa_base_opt = optimizer.base_optimizer if hasattr(optimizer, 'base_optimizer') else optimizer
                         # Use potentially updated LEARNING_RATE for SWA LR calculation
                         swa_scheduler = SWALR(swa_base_opt, swa_lr=(LEARNING_RATE * SWA_LR_FACTOR), anneal_epochs=SWA_ANNEAL_EPOCHS, anneal_strategy='cos')
                    current_swa_scheduler = swa_scheduler; current_swa_scheduler.step()
                    print(f"{TermColors.CYAN}ℹ Fold {fold+1} SWA Update. LR: {current_swa_scheduler.get_last_lr()[0]:.1E}{TermColors.ENDC}")

                elif scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau): scheduler.step() # Check specific type
                if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau): scheduler.step(val_loss if PLATEAU_MONITOR == 'val_loss' else val_acc) # Check specific type

                print(f"Fold {fold+1} GlobEp {global_epoch_counter+1}: Train L={train_loss:.4f} A={train_acc:.4f} | Val L={val_loss:.4f} A={val_acc:.4f}")

                # --- Checkpointing and Best OOF Preds Saving ---
                current_metric = val_loss if CHECKPOINT_MONITOR == 'val_loss' else val_acc; is_best = False
                if (CHECKPOINT_MODE == 'min' and current_metric < best_metric) or (CHECKPOINT_MODE == 'max' and current_metric > best_metric):
                    best_metric = current_metric; is_best = True
                    # Store the OOF predictions from the best epoch for this fold
                    fold_best_oof_preds = current_oof_preds
                    fold_best_oof_labels = current_oof_labels
                    fold_best_val_loss = val_loss
                    fold_best_val_acc = val_acc

                save_checkpoint(fold, global_epoch_counter + 1, stage_idx, stage_epoch + 1, model, optimizer, scheduler, scaler, best_metric, "latest_checkpoint.pth.tar")
                if is_best:
                    print(f"{TermColors.OKGREEN}🏆 Fold {fold+1} Best {CHECKPOINT_MONITOR}: {best_metric:.4f}. Save best models & OOF preds.{TermColors.ENDC}")
                    save_checkpoint(fold, global_epoch_counter + 1, stage_idx, stage_epoch + 1, model, optimizer, scheduler, scaler, best_metric, "best_model.pth.tar")
                    save_model(fold, model, "best_model_state_dict.pth")
                    if USE_EMA and ema_model: save_model(fold, ema_model, "best_ema_model_state_dict.pth")

                if (global_epoch_counter + 1) % 5 == 0 or is_best: log_misclassified(fold, model, err_loader, criterion, DEVICE, global_epoch_counter + 1, writer, NUM_CLASSES)
                global_epoch_counter += 1
                if DEVICE.type == 'cuda': torch.cuda.empty_cache(); gc.collect()
            # --- End Stage ---
            if stop_requested: break
            start_stg_ep = 0 # Reset for next stage
            # Clean up stage-specific resources
            del train_loader, val_loader, err_loader, train_ds, val_ds, err_ds, train_transform, val_transform, scheduler
            if 'swa_scheduler' in locals(): del swa_scheduler # Delete if it was created
            if DEVICE.type == 'cuda': torch.cuda.empty_cache(); gc.collect()

        # --- End Fold ---
        if stop_requested: break # Break fold loop if interrupted

        print(f"\n{TermColors.BOLD}--- Fold {fold+1} Finished (Total Global Epochs: {global_epoch_counter}) ---{TermColors.ENDC}")
        fold_results['best_metric'].append(best_metric)
        fold_results['best_val_loss'].append(fold_best_val_loss)
        fold_results['best_val_acc'].append(fold_best_val_acc)

        # Store the best OOF predictions for this fold into the global list
        if fold_best_oof_preds is not None and fold_best_oof_labels is not None:
            print(f"Storing OOF predictions for {len(val_idx)} validation samples in fold {fold+1}.")
            for i, original_idx in enumerate(val_idx):
                all_oof_preds[original_idx] = fold_best_oof_preds[i]
                all_oof_labels[original_idx] = fold_best_oof_labels[i]
            oof_indices_collected.extend(val_idx) # Track which indices we have preds for
        else:
            print(f"{TermColors.YELLOW}Warning: No best OOF predictions found for fold {fold+1}.{TermColors.ENDC}")


        # --- SWA Final Evaluation ---
        if USE_SWA and swa_model and global_epoch_counter >= swa_start_epoch_global:
            print(f"{TermColors.CYAN}ℹ Fold {fold+1} Updating SWA BN stats...{TermColors.ENDC}")
            final_stage_size = PROGRESSIVE_RESIZING_STAGES[-1][1]
            final_train_tf, _ = get_transforms(image_size=final_stage_size)
            # Need to recreate train_ds for BN update loader
            final_train_ds_bn = PlantDataset(train_df, IMAGE_DIR, final_train_tf, label_encoder, False, final_stage_size)
            bn_loader = DataLoader(final_train_ds_bn, BATCH_SIZE*2, shuffle=True, num_workers=NUM_WORKERS)
            try:
                torch.optim.swa_utils.update_bn(bn_loader, swa_model, device=DEVICE); print(f"{TermColors.GREEN}✅ Fold {fold+1} SWA BN updated.{TermColors.ENDC}")
                print(f"{TermColors.CYAN}ℹ Fold {fold+1} Evaluating final SWA model...{TermColors.ENDC}")
                _, final_val_tf = get_transforms(image_size=final_stage_size)
                # Need to recreate val_ds for final eval loader
                final_val_ds_eval = PlantDataset(val_df, IMAGE_DIR, final_val_tf, label_encoder, False, final_stage_size)
                final_val_ds_eval.fold = fold+1 # Add fold info
                final_val_loader = DataLoader(final_val_ds_eval, BATCH_SIZE*2, shuffle=False, num_workers=NUM_WORKERS)
                # Use validate_one_epoch to evaluate SWA model
                swa_val_loss, swa_val_acc = validate_one_epoch(model, final_val_loader, criterion, DEVICE, global_epoch_counter, writer, NUM_CLASSES, swa_model=swa_model, ema_model=None, return_preds=False, fold_num=fold+1) # Evaluate only SWA
                print(f"Fold {fold+1} SWA Model Final Val: Loss={swa_val_loss:.4f} | Acc={swa_val_acc:.4f}")
                writer.add_scalar('Loss/SWA_validation_final', swa_val_loss, fold); writer.add_scalar('Accuracy/SWA_validation_final', swa_val_acc, fold)
                save_model(fold, swa_model, "swa_model_state_dict.pth")
                fold_results['swa_acc'].append(swa_val_acc); fold_results['swa_loss'].append(swa_val_loss)
                del final_train_ds_bn, bn_loader, final_val_ds_eval, final_val_loader # Clean up SWA eval resources
            except Exception as e: print(f"{TermColors.RED}❌ Fold {fold+1} SWA BN/Eval Error: {e}{TermColors.ENDC}"); traceback.print_exc()


        # Save final standard and EMA models
        save_model(fold, model, "final_model_state_dict.pth")
        if USE_EMA and ema_model: save_model(fold, ema_model, "final_ema_model_state_dict.pth")

        writer.close() # Close writer for this fold

        # --- Explicit Memory Cleanup for Fold ---
        print(f"{TermColors.DIM}Cleaning up resources for fold {fold+1}...{TermColors.ENDC}")
        del model, optimizer, scaler, criterion, swa_model, ema_model, train_df, val_df
        if DEVICE.type == 'cuda': torch.cuda.empty_cache()
        gc.collect()
        print(f"{TermColors.DIM}Fold {fold+1} cleanup complete.{TermColors.ENDC}")


    # --- End of Cross-Validation Loop ---
    print(f"\n{TermColors.HEADER}===== Cross-Validation Finished =====")
    if not stop_requested:
        avg_best_metric = np.mean(fold_results['best_metric']) if fold_results['best_metric'] else 0
        std_best_metric = np.std(fold_results['best_metric']) if fold_results['best_metric'] else 0
        avg_best_acc = np.mean(fold_results['best_val_acc']) if fold_results['best_val_acc'] else 0
        std_best_acc = np.std(fold_results['best_val_acc']) if fold_results['best_val_acc'] else 0
        print(f"Average Best Val Acc across {len(fold_results['best_val_acc'])} folds: {avg_best_acc:.4f} +/- {std_best_acc:.4f}")
        print(f"Average Best {CHECKPOINT_MONITOR} across {len(fold_results['best_metric'])} folds: {avg_best_metric:.4f} +/- {std_best_metric:.4f}")
        if fold_results['swa_acc']:
             avg_swa_acc = np.mean(fold_results['swa_acc']); std_swa_acc = np.std(fold_results['swa_acc'])
             print(f"Average Final SWA Accuracy across {len(fold_results['swa_acc'])} folds: {avg_swa_acc:.4f} +/- {std_swa_acc:.4f}")

        # --- Prepare data for Stacking ---
        if RUN_STACKING:
            print(f"\n{TermColors.CYAN}ℹ Preparing data for Stacking...{TermColors.ENDC}")
            collected_indices = sorted(list(set(oof_indices_collected)))
            if len(collected_indices) != len(df_full):
                 print(f"{TermColors.YELLOW}Warning: Only collected OOF predictions for {len(collected_indices)} out of {len(df_full)} samples. Stacking might be suboptimal.{TermColors.ENDC}")
                 # Filter preds/labels to only include collected ones
                 final_oof_preds = np.array([all_oof_preds[i] for i in collected_indices if all_oof_preds[i] is not None])
                 final_oof_labels = np.array([all_oof_labels[i] for i in collected_indices if all_oof_labels[i] is not None])
            else:
                 print(f"{TermColors.GREEN}Collected OOF predictions for all {len(df_full)} samples.{TermColors.ENDC}")
                 final_oof_preds = np.array([p for p in all_oof_preds if p is not None])
                 final_oof_labels = np.array([l for l in all_oof_labels if l is not None])

            if len(final_oof_preds) > 0 and len(final_oof_preds) == len(final_oof_labels):
                 # Save OOF predictions
                 np.savez_compressed(STACKING_OOF_PREDS_PATH, preds=final_oof_preds, labels=final_oof_labels)
                 print(f"OOF predictions saved to {STACKING_OOF_PREDS_PATH}")
                 # Train Stacking Meta-Model
                 train_stacking_meta_model(final_oof_preds, final_oof_labels, STACKING_META_MODEL_PATH)
            else:
                 print(f"{TermColors.RED}Error preparing stacking data. Length mismatch or no data collected. Skipping stacking.{TermColors.ENDC}")
                 final_oof_preds, final_oof_labels = None, None # Ensure cleanup
            del final_oof_preds, final_oof_labels, all_oof_preds, all_oof_labels # Cleanup stacking data
            gc.collect()

    else:
        print(f"{TermColors.YELLOW}Training interrupted. Results may be incomplete. Skipping Stacking and KD.{TermColors.ENDC}")
        RUN_STACKING = False
        RUN_KNOWLEDGE_DISTILLATION = False

    # --- Knowledge Distillation (after CV) ---
    if RUN_KNOWLEDGE_DISTILLATION and not stop_requested:
        print(f"\n{TermColors.HEADER}--- STEP 4: Knowledge Distillation ---{TermColors.ENDC}")
        try:
            # Find the best teacher model path (using fold KD_TEACHER_FOLD)
            teacher_path = os.path.join(BASE_CHECKPOINT_DIR, f"fold_{KD_TEACHER_FOLD}", "best_model.pth.tar")
            if os.path.exists(teacher_path):
                 # Use the full dataset for KD training/validation (or define specific splits if needed)
                 # Re-split full data for KD train/val to avoid using OOF splits directly
                 df_kd_train, df_kd_val = train_test_split(df_full, test_size=0.15, random_state=SEED+1, stratify=df_full['label']) # Simple split for KD
                 train_student_model(teacher_path, KD_STUDENT_MODEL_NAME, KD_STUDENT_MODEL_SAVE_PATH,
                                     df_kd_train, df_kd_val, IMAGE_DIR, label_encoder, NUM_CLASSES)
                 del df_kd_train, df_kd_val # Cleanup KD dataframes
                 gc.collect()
            else:
                 print(f"{TermColors.RED}❌ Teacher model not found at {teacher_path}. Skipping Knowledge Distillation.{TermColors.ENDC}")
        except Exception as e:
            print(f"{TermColors.RED}❌ Error during Knowledge Distillation setup or execution: {e}{TermColors.ENDC}")
            traceback.print_exc()

    print(f"{TermColors.OKGREEN}🎉 All processes complete. Models/logs saved per fold. Stacking/KD models saved if run.{TermColors.ENDC}")


if __name__ == "__main__":
    try:
        # --- Dependency Checks ---
        print(f"{TermColors.CYAN}ℹ Checking dependencies...{TermColors.ENDC}")
        if not TIMM_AVAILABLE:
            print(f"{TermColors.FAIL}❌ FATAL: timm library not found. Required. Install with 'pip install timm'.{TermColors.ENDC}")
            sys.exit(1)
        if OPTIMIZER_TYPE == 'AdamP' and not ADAMP_AVAILABLE:
            print(f"{TermColors.FAIL}❌ FATAL: AdamP optimizer selected but library not found. Install with 'pip install adamp'.{TermColors.ENDC}")
            sys.exit(1)
        if USE_SAM and not SAM_AVAILABLE:
            print(f"{TermColors.FAIL}❌ FATAL: SAM optimizer selected but library not found. Install with 'pip install sam_pytorch'.{TermColors.ENDC}")
            sys.exit(1)
        if not ALBUMENTATIONS_AVAILABLE:
            print(f"{TermColors.YELLOW}⚠️ WARNING: Albumentations not found. Using basic torchvision transforms. Install with 'pip install albumentations'.{TermColors.ENDC}")
        if not TORCHMETRICS_AVAILABLE:
            print(f"{TermColors.YELLOW}⚠️ WARNING: torchmetrics not found. Using basic accuracy calculation. Install with 'pip install torchmetrics'.{TermColors.ENDC}")
        if not KEYBOARD_AVAILABLE:
            print(f"{TermColors.YELLOW}⚠️ WARNING: keyboard library not found. Ctrl+C graceful stop disabled. Install with 'pip install keyboard'.{TermColors.ENDC}")
        if RUN_HPO and 'optuna' not in sys.modules:
             print(f"{TermColors.FAIL}❌ FATAL: Optuna required for HPO but not found. Install with 'pip install optuna'.{TermColors.ENDC}")
             sys.exit(1)
        if RUN_STACKING and 'joblib' not in sys.modules:
             print(f"{TermColors.FAIL}❌ FATAL: Joblib required for Stacking but not found. Install with 'pip install joblib'.{TermColors.ENDC}")
             sys.exit(1)
        print(f"{TermColors.GREEN}✅ Dependencies check passed (or warnings issued).{TermColors.ENDC}")

        # --- Start Main Execution ---
        main()

    except KeyboardInterrupt: # Catch explicit Ctrl+C if signal handler didn't fully catch it
        print(f"\n{TermColors.RED}❌ KeyboardInterrupt detected. Exiting forcefully.{TermColors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{TermColors.FAIL}💥 An unexpected error occurred in the main execution block: {e}{TermColors.ENDC}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Ensure terminal colors are reset on exit, regardless of success or failure
        print(colorama.Style.RESET_ALL)
        print(f"{TermColors.DIM}--- Script execution finished. ---{TermColors.ENDC}")