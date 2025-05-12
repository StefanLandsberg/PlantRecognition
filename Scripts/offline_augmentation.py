import math
import os
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import cv2
from tqdm import tqdm
import json
import random
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import threading


# Defines terminal colours for console output.
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

# Script configuration parameters.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))    # Script directory.
BASE_DIR = os.path.dirname(SCRIPT_DIR)                     # Project root directory.
DATA_DIR = os.path.join(BASE_DIR, "data", "plant_images") # Directory for plant images.
IMAGE_SIZE = (512, 512) # Target image dimensions.
MIN_IMAGES_PER_CLASS = 200  # Minimum images per class after augmentation.
MAX_IMAGES_PER_CLASS = 400  # Maximum images per class after augmentation.
CHECKPOINT_FILE = "augmentation_checkpoint.json" # File to store augmentation progress.

# CPU and GPU optimisation settings.
MAX_CPU_WORKERS = os.cpu_count()  # Number of CPU cores to utilise.
PARALLEL_CLASSES = max(4, os.cpu_count() // 2)  # Number of classes to process in parallel.
BATCH_SIZE = 64  # Batch size for image processing.
USE_GPU = True  # Flag to enable GPU usage.
GPU_BATCH_SIZE = 128  # Batch size for GPU processing. # This constant is not used in the PyTorch version directly
GPU_MEMORY_LIMIT = None  # GPU memory limit; None for no limit. # This constant is not used in PyTorch version

# Configures PyTorch to utilise GPU if available.
def setup_gpu():
    """Initialises GPU for PyTorch operations if USE_GPU is True."""
    if USE_GPU:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"{TermColors.GREEN}Found {num_gpus} GPU(s). Enabling GPU acceleration.{TermColors.ENDC}")

            # Uses only the first GPU if multiple are present (PyTorch default behavior unless specified).
            if num_gpus > 1:
                current_gpu_name = torch.cuda.get_device_name(0)
                print(f"{TermColors.CYAN}Using GPU: {current_gpu_name}{TermColors.ENDC}")
            
            # Performs a small tensor operation to test GPU initialisation.
            try:
                device = torch.device("cuda:0")
                test_tensor = torch.randn(100, 100, device=device)
                test_result = torch.matmul(test_tensor, test_tensor.T)
                torch.sum(test_result).item()  # Force execution.
                print(f"{TermColors.GREEN}GPU test successful.{TermColors.ENDC}")
                return True
            except Exception as e:
                print(f"{TermColors.YELLOW}GPU initialised but test failed: {e}. Will use GPU selectively.{TermColors.ENDC}")
                return False # Fallback to selective GPU usage if test fails
        else:
            print(f"{TermColors.YELLOW}No GPU found. Using CPU only.{TermColors.ENDC}")
            return False
    return False

# Sets USE_GPU_SUCCESSFULLY based on GPU setup outcome.
USE_GPU_SUCCESSFULLY = setup_gpu() if USE_GPU else False
PYTORCH_DEVICE = torch.device("cuda" if USE_GPU_SUCCESSFULLY else "cpu")

def pytorch_preprocess_input(img_hwc_uint8: np.ndarray) -> torch.Tensor:
    """Converts HWC uint8 numpy image to CHW float32 tensor, scaled to [-1, 1]."""
    if not isinstance(img_hwc_uint8, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    
    temp_img_hwc_uint8 = img_hwc_uint8 # Work on a copy if modifications are needed
    if temp_img_hwc_uint8.ndim != 3 or temp_img_hwc_uint8.shape[2] != 3:
        if temp_img_hwc_uint8.ndim == 3 and temp_img_hwc_uint8.shape[2] == 1: # HWC grayscale
            temp_img_hwc_uint8 = np.repeat(temp_img_hwc_uint8, 3, axis=2)
        elif temp_img_hwc_uint8.ndim == 2: # HW grayscale
            temp_img_hwc_uint8 = np.stack([temp_img_hwc_uint8]*3, axis=-1)
        else:
            raise ValueError(f"Input image must be HWC or HW grayscale, got shape {temp_img_hwc_uint8.shape}")

    if temp_img_hwc_uint8.dtype != np.uint8:
        if np.issubdtype(temp_img_hwc_uint8.dtype, np.floating) and temp_img_hwc_uint8.max() > 1.0 and temp_img_hwc_uint8.min() >=0: # Potentially 0-255 float
            temp_img_hwc_uint8 = temp_img_hwc_uint8.astype(np.uint8)
        elif np.issubdtype(temp_img_hwc_uint8.dtype, np.floating) and temp_img_hwc_uint8.max() <=1.0 and temp_img_hwc_uint8.min() >=0: # Potentially 0-1 float
             temp_img_hwc_uint8 = (temp_img_hwc_uint8 * 255.0).astype(np.uint8)
        else:
            raise TypeError(f"Input image must be uint8 or convertible float, got dtype {temp_img_hwc_uint8.dtype}")

    img_chw_float32 = torch.from_numpy(temp_img_hwc_uint8.astype(np.float32).transpose((2, 0, 1)))
    return (img_chw_float32 / 127.5) - 1.0

def load_checkpoint():
    """Loads augmentation progress from the checkpoint file."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                return json.load(f)
        except: # Handles potential errors during file reading or JSON parsing.
            return {"processed_classes": []}
    return {"processed_classes": []}

def save_checkpoint(checkpoint):
    """Saves the current augmentation progress to the checkpoint file."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)

def analyze_class_sizes():
    """Analyses image distribution across classes and calculates augmentation factors."""
    print(f"{TermColors.HEADER}\n{'='*50}")
    print(f"ANALYSING CLASS DISTRIBUTION")
    print(f"{'='*50}{TermColors.ENDC}")
    
    # Retrieves class directories.
    class_dirs = [os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    # Counts original images in each class.
    class_counts = {}
    for class_dir in class_dirs:
        class_name = os.path.basename(class_dir)
        image_files = [f for f in os.listdir(class_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not '_aug' in f]
        class_counts[class_name] = len(image_files)
    
    # Calculates dataset statistics.
    if not class_counts:
        print(f"{TermColors.RED}No classes found in {DATA_DIR}{TermColors.ENDC}")
        return {}
        
    total_classes = len(class_counts)
    total_images = sum(class_counts.values())
    min_images = min(class_counts.values()) if class_counts else 0
    max_images = max(class_counts.values()) if class_counts else 0
    avg_images = total_images / total_classes if total_classes > 0 else 0
    
    print(f"{TermColors.CYAN}Dataset Statistics:{TermColors.ENDC}")
    print(f"  - Total classes: {total_classes}")
    print(f"  - Total original images: {total_images}")
    print(f"  - Images per class range: {min_images} to {max_images}")
    print(f"  - Average images per class: {avg_images:.1f}")
    
    # Calculates custom augmentation factors for each class.
    augmentation_factors = {}
    
    # Categorises classes by image count.
    small_classes = 0  # Classes with fewer than 5 images.
    large_classes = 0  # Classes with more than 30 images.
    
    for class_name, count in class_counts.items():
        if count < 5:
            small_classes += 1
        elif count > 30:
            large_classes += 1
    
    if total_classes > 0:
        print(f"  - Classes with <5 images: {small_classes} ({small_classes/total_classes:.1%})")
        print(f"  - Classes with >30 images: {large_classes} ({large_classes/total_classes:.1%})")
    else:
        print(f"  - Classes with <5 images: 0 (0.0%)")
        print(f"  - Classes with >30 images: 0 (0.0%)")

    # Calculates target augmentation factors to balance the dataset.
    for class_name, count in class_counts.items():
        if count == 0:
            aug_factor = 0.0
        elif count < MIN_IMAGES_PER_CLASS:
            # Calculates factor to reach MIN_IMAGES_PER_CLASS.
            aug_factor = (MIN_IMAGES_PER_CLASS - count) / count if count > 0 else MIN_IMAGES_PER_CLASS
        elif count < MAX_IMAGES_PER_CLASS:
            # Calculates factor to reach MAX_IMAGES_PER_CLASS.
            aug_factor = (MAX_IMAGES_PER_CLASS - count) / count if count > 0 else MAX_IMAGES_PER_CLASS
        else: # count >= MAX_IMAGES_PER_CLASS
            aug_factor = 1.0 # No augmentation needed to increase count.

        augmentation_factors[class_name] = aug_factor
    
    # Prints examples of augmentation strategy.
    print(f"\n{TermColors.CYAN}Augmentation Strategy Examples:{TermColors.ENDC}")
    
    # Sorts classes by image count for display.
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])
    
    # Prints details for the 3 smallest classes.
    print(f"{TermColors.YELLOW}Smallest Classes:{TermColors.ENDC}")
    for class_name, count in sorted_classes[:3]:
        aug_factor = augmentation_factors.get(class_name, 0) 
        # Calculates final count after augmentation.
        if count == 0:
            final_count = 0
        elif count < MIN_IMAGES_PER_CLASS :
            final_count = MIN_IMAGES_PER_CLASS
        elif count < MAX_IMAGES_PER_CLASS:
            final_count = MAX_IMAGES_PER_CLASS
        else: 
            final_count = count 

        # Ensures final count does not exceed MAX_IMAGES_PER_CLASS if it was already above.
        final_count = min(final_count, MAX_IMAGES_PER_CLASS if count < MAX_IMAGES_PER_CLASS else count)

        print(f"  - {class_name}: {count} orig → {final_count:.0f} total (aiming for, factor ~{aug_factor:.1f}x)")
    
    # Prints details for 3 medium-sized classes.
    if len(sorted_classes) > 3:
        mid_idx = len(sorted_classes) // 2
        print(f"{TermColors.GREEN}Medium Classes:{TermColors.ENDC}")
        for class_name, count in sorted_classes[mid_idx:min(mid_idx+3, len(sorted_classes))]:
            aug_factor = augmentation_factors.get(class_name, 0)
            if count == 0:
                final_count = 0
            elif count < MIN_IMAGES_PER_CLASS :
                final_count = MIN_IMAGES_PER_CLASS
            elif count < MAX_IMAGES_PER_CLASS:
                final_count = MAX_IMAGES_PER_CLASS
            else: 
                final_count = count

            final_count = min(final_count, MAX_IMAGES_PER_CLASS if count < MAX_IMAGES_PER_CLASS else count)
            print(f"  - {class_name}: {count} orig → {final_count:.0f} total (aiming for, factor ~{aug_factor:.1f}x)")
    
    # Prints details for the 3 largest classes.
    if len(sorted_classes) > 0:
        print(f"{TermColors.BLUE}Largest Classes:{TermColors.ENDC}")
        for class_name, count in sorted_classes[-3:]:
            aug_factor = augmentation_factors.get(class_name, 0)
            if count == 0:
                final_count = 0
            elif count < MIN_IMAGES_PER_CLASS :
                final_count = MIN_IMAGES_PER_CLASS
            elif count < MAX_IMAGES_PER_CLASS:
                final_count = MAX_IMAGES_PER_CLASS
            else: 
                final_count = count
                
            final_count = min(final_count, MAX_IMAGES_PER_CLASS if count < MAX_IMAGES_PER_CLASS else count)
            print(f"  - {class_name}: {count} orig → {final_count:.0f} total (aiming for, factor ~{aug_factor:.1f}x)")
    
    return augmentation_factors

def apply_occlusion(img, occlusion_type):
    """Applies partial occlusion to an image."""
    h, w = img.shape[:2]
    img_copy = img.copy()
    
    if occlusion_type == "partial_view":
        # Chooses occlusion direction and amount.
        direction = random.choice(["top", "bottom", "left", "right"])
        occlusion_percent = random.uniform(0.1, 0.3)  # Occludes 10-30% of the image.
        
        if direction == "top":
            occlusion_height = int(h * occlusion_percent)
            # Defines natural occlusion colours.
            occlusion_color = random.choice([
                [50, 120, 50],    # Dark green.
                [70, 50, 40],     # Brown.
                [100, 100, 100],  # Grey.
            ])
            img_copy[:occlusion_height, :] = occlusion_color
            
            # Adds noise for realism.
            noise = np.random.randint(-20, 20, (occlusion_height, w, 3), dtype=np.int16) # Use int16 to avoid overflow with uint8
            img_copy[:occlusion_height, :] = np.clip(img_copy[:occlusion_height, :].astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Creates a gradual blend at the occlusion border.
            for i in range(10):
                blend_row = occlusion_height + i
                if blend_row < h:
                    alpha = (10 - i) / 10
                    img_copy[blend_row, :] = (alpha * np.array(occlusion_color) + (1 - alpha) * img[blend_row, :]).astype(np.uint8)
            
        elif direction == "bottom":
            occlusion_height = int(h * occlusion_percent)
            start_row = h - occlusion_height
            
            # Defines natural occlusion colours (ground, litter, soil).
            occlusion_color = random.choice([
                [80, 70, 60],     # Brown soil.
                [120, 110, 70],   # Dry leaves.
                [120, 140, 80],   # Grass.
            ])
            img_copy[start_row:, :] = occlusion_color
            
            # Adds noise for realism.
            noise = np.random.randint(-20, 20, (occlusion_height, w, 3), dtype=np.int16)
            img_copy[start_row:, :] = np.clip(img_copy[start_row:, :].astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Creates a gradual blend at the occlusion border.
            for i in range(10):
                blend_row = start_row - i - 1
                if blend_row >= 0:
                    alpha = (10 - i) / 10
                    img_copy[blend_row, :] = ((1 - alpha) * img[blend_row, :] + alpha * np.array(occlusion_color)).astype(np.uint8)
            
        elif direction == "left":
            occlusion_width = int(w * occlusion_percent)
            
            # Defines natural occlusion colours.
            occlusion_color = random.choice([
                [50, 120, 50],    # Dark green.
                [70, 50, 40],     # Brown.
                [100, 100, 100],  # Grey.
            ])
            img_copy[:, :occlusion_width] = occlusion_color
            
            # Adds noise for realism.
            noise = np.random.randint(-20, 20, (h, occlusion_width, 3), dtype=np.int16)
            img_copy[:, :occlusion_width] = np.clip(img_copy[:, :occlusion_width].astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Creates a gradual blend at the occlusion border.
            for i in range(10):
                blend_col = occlusion_width + i
                if blend_col < w:
                    alpha = (10 - i) / 10
                    img_copy[:, blend_col] = (alpha * np.array(occlusion_color) + (1 - alpha) * img[:, blend_col]).astype(np.uint8)
            
        else:  # "right"
            occlusion_width = int(w * occlusion_percent)
            start_col = w - occlusion_width
            
            # Defines natural occlusion colours.
            occlusion_color = random.choice([
                [50, 120, 50],    # Dark green.
                [70, 50, 40],     # Brown.
                [100, 100, 100],  # Grey.
            ])
            img_copy[:, start_col:] = occlusion_color
            
            # Adds noise for realism.
            noise = np.random.randint(-20, 20, (h, occlusion_width, 3), dtype=np.int16)
            img_copy[:, start_col:] = np.clip(img_copy[:, start_col:].astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Creates a gradual blend at the occlusion border.
            for i in range(10):
                blend_col = start_col - i - 1
                if blend_col >= 0:
                    alpha = (10 - i) / 10
                    img_copy[:, blend_col] = ((1 - alpha) * img[:, blend_col] + alpha * np.array(occlusion_color)).astype(np.uint8)
    
    return img_copy.astype(np.uint8)

def apply_scale_variation(img, scale_type):
    """Applies scale variation effects like macro or distant view."""
    h, w = img.shape[:2]
    img_copy = img.copy()
    
    if scale_type == "macro":
        # Simulates macro photography (close-up).
        
        # Chooses a focus area, biased towards the centre.
        center_bias = random.choice([True, True, False]) 
        
        if center_bias:
            # Focuses on centre with a slight random offset.
            center_x = w // 2 + random.randint(-w//8, w//8)
            center_y = h // 2 + random.randint(-h//8, h//8)
        else:
            # Focuses on a random point.
            center_x = random.randint(w//4, 3*w//4)
            center_y = random.randint(h//4, 3*h//4)
        
        # Determines zoom factor.
        zoom_factor = random.uniform(1.3, 1.8) 
        
        # Calculates crop region.
        new_w = int(w / zoom_factor)
        new_h = int(h / zoom_factor)
        
        # Ensures the crop region is within image bounds.
        x1 = max(0, min(center_x - new_w // 2, w - new_w))
        y1 = max(0, min(center_y - new_h // 2, h - new_h))
        
        # Crops and resizes the image.
        cropped = img_copy[y1:y1+new_h, x1:x1+new_w]
        img_copy = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Adds a slight depth-of-field blur effect (peripheral blur).
        # Creates a mask that is strongest at the edges.
        mask = np.zeros((h, w), dtype=np.float32)
        for y_coord in range(h): 
            for x_coord in range(w): 
                # Calculates distance from centre as 0.0-1.0.
                dist_y = abs(y_coord - h/2) / (h/2) if h > 0 else 0
                dist_x = abs(x_coord - w/2) / (w/2) if w > 0 else 0
                # Uses the maximum of the x and y distances for a smoother effect.
                mask[y_coord, x_coord] = max(dist_y, dist_x) ** 2  # Square for sharper falloff.
        
        # Applies blur based on the mask.
        blurred = cv2.GaussianBlur(img_copy, (21, 21), 0)
        
        # Blends original and blurred images based on the mask.
        mask_3c = np.stack([mask] * 3, axis=2)
        img_copy = img_copy.astype(np.float32) * (1 - mask_3c * 0.7) + blurred.astype(np.float32) * (mask_3c * 0.7)
        img_copy = np.clip(img_copy, 0, 255).astype(np.uint8)
        
    elif scale_type == "distant":
        # Simulates a distant view.
        
        # Resizes down and then up to reduce detail.
        small_size = (max(1, w // 3), max(1,h // 3)) # Ensure dimensions are at least 1
        small_img = cv2.resize(img_copy, small_size, interpolation=cv2.INTER_AREA)
        img_copy = cv2.resize(small_img, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Adds slight blur.
        img_copy = cv2.GaussianBlur(img_copy, (3, 3), 0)
        
        # Adds atmospheric perspective (slight blue/grey tint and reduced contrast).
        atmospheric_color = np.array([180, 200, 230], dtype=np.float32)  # Slight blue-grey.
        
        # Creates a distance gradient (more effect at the top).
        gradient = np.zeros((h, w), dtype=np.float32)
        if h > 0:
            for y_coord in range(h): 
                # More atmospheric effect at top of image.
                gradient[y_coord, :] = 1.0 - y_coord / h
        
        # Applies atmospheric effect.
        atmospheric_strength = random.uniform(0.1, 0.25)
        gradient = gradient * atmospheric_strength
        
        # Converts gradient to 3-channel.
        gradient_3c = np.stack([gradient] * 3, axis=2)
        
        # Blends with atmospheric colour.
        img_copy = img_copy.astype(np.float32) * (1 - gradient_3c) + atmospheric_color * gradient_3c
        
        # Reduces contrast slightly.
        img_copy = (img_copy * 0.9 + np.mean(img_copy, axis=(0,1), keepdims=True) * 0.1) # Keepdims for broadcasting
        img_copy = np.clip(img_copy, 0, 255).astype(np.uint8)
    
    return img_copy.astype(np.uint8)

def apply_plant_age_variation(img, age_type):
    """Applies variations simulating different plant ages."""
    img_copy = img.copy()
    h, w = img.shape[:2]
    
    if age_type == "young":
        # Simulates younger plant appearance (brighter green, less texture).
        # Increases green channel.
        img_copy_float = img_copy.astype(np.float32)
        img_copy_float[:,:,1] *= 1.2
        
        # Slightly reduces red channel for a fresher look.
        img_copy_float[:,:,0] *= 0.9
        img_copy = np.clip(img_copy_float, 0, 255).astype(np.uint8)
        
        # Reduces texture.
        img_copy = cv2.GaussianBlur(img_copy, (3, 3), 0)
        
    elif age_type == "mature":
        # Simulates mature plant appearance (more texture, deeper colours).
        # Enhances texture with sharpening.
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], dtype=np.float32) * 0.5
        # Add identity matrix part for less aggressive sharpening
        sharpened_img = cv2.filter2D(img_copy.astype(np.float32), -1, kernel) 
        img_copy = np.clip(sharpened_img, 0, 255).astype(np.uint8)

        # Adjusts colour channels for deeper colours.
        img_copy_float = img_copy.astype(np.float32)
        img_copy_float[:,:,1] *= 0.9
        img_copy_float[:,:,0] *= 1.05
        img_copy_float[:,:,2] *= 1.05
        img_copy = np.clip(img_copy_float, 0, 255).astype(np.uint8)
        
    elif age_type == "flowering":
        # Simulates flowering plant appearance (vibrant colours, bright spots).
        # Detects potential flower regions (non-green, higher saturation).
        hsv = cv2.cvtColor(img_copy, cv2.COLOR_RGB2HSV)
        
        # Extracts hue and saturation channels.
        hue_channel = hsv[:,:,0] 
        sat_channel = hsv[:,:,1] 
        
        # Creates mask for non-green, high saturation regions.
        non_green_mask = np.logical_or(hue_channel < 40, hue_channel > 80)
        high_sat_mask = sat_channel > 70
        
        # Combines masks to identify potential flower regions.
        flower_mask_single_channel = np.logical_and(non_green_mask, high_sat_mask).astype(np.float32) 
        
        # Enhances detected flower regions.
        if np.mean(flower_mask_single_channel) > 0.01:  # If at least 1% of image contains potential flowers.
            # Dilates mask to include flower edges.
            flower_mask_dilated = cv2.dilate(flower_mask_single_channel.astype(np.uint8), np.ones((5, 5), np.uint8)) 
            flower_mask_blurred = cv2.GaussianBlur(flower_mask_dilated, (9, 9), 0) / 255.0 
            
            # Creates 3-channel mask.
            flower_mask_3c = np.stack([flower_mask_blurred] * 3, axis=2)
            
            # Enhances saturation and brightness in flower regions.
            hsv_float = hsv.astype(np.float32) 
            hsv_float[:,:,1] += flower_mask_blurred * 40  # Increases saturation.
            hsv_float[:,:,2] += flower_mask_blurred * 30  # Increases brightness.
            
            # Clips values to valid range.
            hsv_float[:,:,1] = np.clip(hsv_float[:,:,1], 0, 255)
            hsv_float[:,:,2] = np.clip(hsv_float[:,:,2], 0, 255)
            
            # Converts back to RGB.
            enhanced = cv2.cvtColor(hsv_float.astype(np.uint8), cv2.COLOR_HSV2RGB)
            
            # Blends enhanced flower regions with original image.
            img_copy = (img_copy.astype(np.float32) * (1 - flower_mask_3c) + enhanced.astype(np.float32) * flower_mask_3c).astype(np.uint8)
        else:
            # Adds slight colour spots if no flowers are detected.
            # Chooses a flower colour.
            flower_color_val = random.choice([ 
                [220, 180, 200],  # Pink.
                [200, 150, 150],  # Light red.
                [220, 220, 150],  # Yellow.
                [200, 200, 200],  # White.
                [180, 180, 220],  # Light blue.
            ])
            
            # Creates 3-5 small flower spots.
            num_spots = random.randint(3, 5)
            img_copy_float = img_copy.astype(np.float32)
            for _ in range(num_spots):
                # Sets random position, biased towards top 2/3 of image.
                x_pos = random.randint(0, w-1) 
                y_pos = random.randint(0, int(h*0.7)) 
                
                # Sets random radius.
                radius = random.randint(3, 15)
                
                # Creates circular mask for the spot.
                y_grid, x_grid = np.ogrid[:h, :w]
                dist_from_center = np.sqrt((x_grid - x_pos)**2 + (y_grid - y_pos)**2)
                
                # Creates soft circular mask.
                spot_mask_single_channel = np.clip(1.0 - dist_from_center / (radius + 1e-6), 0, 1.0) ** 2 
                spot_mask_3c = np.stack([spot_mask_single_channel] * 3, axis=2)
                
                # Applies coloured spot.
                img_copy_float = img_copy_float * (1 - spot_mask_3c) + np.array(flower_color_val, dtype=np.float32) * spot_mask_3c
            img_copy = np.clip(img_copy_float, 0, 255).astype(np.uint8)
    
    return img_copy.astype(np.uint8)

def apply_weather_condition(img, condition):
    """Applies weather effects like rain, fog, or bright sun."""
    h, w = img.shape[:2]
    img_copy = img.copy().astype(np.float32)
    
    if condition == "rain":
        # Adds rain streaks.
        num_drops = random.randint(100, 200)
        rain_layer = np.zeros_like(img_copy)
        
        for _ in range(num_drops):
            # Sets random position for rain drop.
            x_pos = random.randint(0, w-1) 
            y_pos = random.randint(0, h-1) 
            
            # Sets random length for rain drop.
            length = random.randint(5, 15)
            
            # Sets rain angle (typically diagonal).
            angle_deg = random.uniform(-30, -5)  # Degrees, Renamed
            
            # Calculates end point of rain drop.
            angle_rad = np.deg2rad(angle_deg)
            x2 = int(x_pos + length * np.cos(angle_rad))
            y2 = int(y_pos + length * np.sin(angle_rad))
            
            # Draws rain drop.
            cv2.line(rain_layer, (x_pos, y_pos), (x2, y2), (200.0, 200.0, 200.0), 1) # Use float color
        
        # Blurs rain layer slightly.
        rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 0)
        
        # Blends rain with original image.
        alpha_blend = 0.7 
        img_copy = cv2.addWeighted(img_copy, alpha_blend, rain_layer, 1-alpha_blend, 0)
        
        # Reduces contrast and adds blue tint for overcast sky effect.
        img_copy = img_copy * 0.9  # Darkens image.
        img_copy[:,:,2] += 10  # Increases blue channel.
        
    elif condition == "fog":
        # Creates fog effect (low contrast, whitish overlay).
        fog_color_val = np.array([220, 220, 220], dtype=np.float32) 
        
        # Creates foggy layer with gradient (more fog at the top).
        fog_layer = np.zeros_like(img_copy)
        if h > 0:
            for y_coord in range(h): 
                # More fog at top.
                fog_intensity = max(0, 0.8 - 0.6 * y_coord / h)
                fog_layer[y_coord, :] = fog_color_val * fog_intensity
        
        # Adds noise to fog for texture.
        noise = np.random.normal(0, 5, fog_layer.shape).astype(np.float32)
        fog_layer += noise
        fog_layer = np.clip(fog_layer, 0, 255)
        
        # Reduces contrast in the original image.
        img_copy = img_copy * 0.8 + np.mean(img_copy, axis=(0,1), keepdims=True) * 0.2
        
        # Blends with fog.
        img_copy = img_copy + fog_layer
        
    elif condition == "bright_sun":
        # Creates bright sunlight effect (high contrast, warm colours).
        # Increases brightness.
        img_copy = img_copy * 1.2
        
        # Increases contrast.
        mean_val = np.mean(img_copy, axis=(0, 1), keepdims=True) 
        img_copy = (img_copy - mean_val) * 1.3 + mean_val
        
        # Adds warm tint (increases red/yellow, decreases blue).
        img_copy[:,:,0] += 15  # More red.
        img_copy[:,:,1] += 10  # More green (creates yellow with red).
        img_copy[:,:,2] -= 10  # Less blue.
        
        # Adds a strong shadow to a part of the image.
        if random.random() < 0.7:  # 70% chance of shadow.
            shadow_direction = random.choice(["top", "left", "right", "bottom"])
            shadow_size = random.uniform(0.2, 0.4)  # Shadow covers 20-40% of image.
            shadow_intensity = random.uniform(0.3, 0.5)  # Shadow darkens by 30-50%.
            
            shadow_mask_single_channel = np.ones((h, w), dtype=np.float32) 
            
            if shadow_direction == "top":
                if h * shadow_size > 0: # Check for non-zero loop range
                    for y_coord in range(int(h * shadow_size)): 
                        gradient = 1.0 - (y_coord / (h * shadow_size))
                        shadow_mask_single_channel[y_coord, :] = 1.0 - (shadow_intensity * gradient)
            elif shadow_direction == "left":
                if w * shadow_size > 0:
                    for x_coord in range(int(w * shadow_size)): 
                        gradient = 1.0 - (x_coord / (w * shadow_size))
                        shadow_mask_single_channel[:, x_coord] = 1.0 - (shadow_intensity * gradient)
            elif shadow_direction == "right":
                if w * shadow_size > 0:
                    for x_coord in range(int(w * shadow_size)): 
                        gradient = 1.0 - (x_coord / (w * shadow_size))
                        shadow_mask_single_channel[:, w-x_coord-1] = 1.0 - (shadow_intensity * gradient)
            elif shadow_direction == "bottom":
                if h * shadow_size > 0:
                    for y_coord in range(int(h * shadow_size)): 
                        gradient = 1.0 - (y_coord / (h * shadow_size))
                        shadow_mask_single_channel[h-y_coord-1, :] = 1.0 - (shadow_intensity * gradient)
            
            # Applies shadow.
            shadow_mask_3c = np.stack([shadow_mask_single_channel] * 3, axis=2)
            img_copy = img_copy * shadow_mask_3c
    
    # Clips values to valid range.
    img_copy = np.clip(img_copy, 0, 255)
    return img_copy.astype(np.uint8)

def create_augmentations(img, class_name=''):
    """Generates a list of augmented images from an input image."""
    # Converts image to numpy array if it is a PyTorch tensor.
    if isinstance(img, torch.Tensor):
        try:
            # Assuming CHW tensor, convert to HWC numpy
            if img.ndim == 4: # Batched CHW
                img_np = img[0].cpu().numpy().transpose((1,2,0))
            elif img.ndim == 3: # Single CHW
                img_np = img.cpu().numpy().transpose((1,2,0))
            else:
                img_np = img.cpu().numpy() # Fallback if not CHW
            
            # If tensor was [-1,1] or [0,1] float, scale back to [0,255] uint8
            if np.issubdtype(img_np.dtype, np.floating):
                if img_np.min() >= -1.01 and img_np.max() <= 1.01: # Likely [-1,1]
                    img_np = ((img_np + 1.0) * 127.5)
                elif img_np.min() >= -0.01 and img_np.max() <= 1.01: # Likely [0,1]
                    img_np = img_np * 255.0
                img_np = np.clip(img_np, 0, 255).astype(np.uint8)

        except Exception as e:
            print(f"Error converting PyTorch tensor to numpy: {e}")
            img_np = np.array(img) # Fallback
            if len(img_np.shape) == 4: # Assuming NCHW if it's a batch
                img_np = img_np[0].transpose((1,2,0)) # to HWC
            elif len(img_np.shape) == 3 and img_np.shape[0] == 3 : # Assuming CHW
                img_np = img_np.transpose((1,2,0)) # to HWC
    else:
        # Assumes image is already a numpy array (HWC, uint8).
        img_np = np.array(img)
    
    # Ensures image has a valid format (HWC, 3 channels, uint8).
    if img_np.ndim != 3 or img_np.shape[2] != 3:
        if img_np.ndim == 3 and img_np.shape[2] == 1: # HWC grayscale
            img_np = np.repeat(img_np, 3, axis=2)
        elif img_np.ndim == 2: # HW grayscale
            img_np = np.stack([img_np] * 3, axis=2)
        else:
            # Returns only the original preprocessed image if format is invalid.
            print(f"Invalid image format for augmentation: shape {img_np.shape}, dtype {img_np.dtype}")
            try:
                return [pytorch_preprocess_input(img_np.astype(np.uint8))]
            except Exception as e_preproc:
                print(f"Preprocessing failed for invalid image: {e_preproc}")
                return [] # Return empty if even preprocessing fails
    
    if img_np.dtype != np.uint8:
        if np.issubdtype(img_np.dtype, np.floating) and img_np.max() <= 1.0 and img_np.min() >=0.0: # Float 0-1
            img_np = (img_np * 255.0).astype(np.uint8)
        elif np.issubdtype(img_np.dtype, np.floating) and img_np.max() <= 255.0 and img_np.min() >=0.0: # Float 0-255
             img_np = img_np.astype(np.uint8)
        else: 
            print(f"Warning: Unexpected image dtype {img_np.dtype}, attempting to convert to uint8.")
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)


    augmented_images_tensors = [] 
    
    # Adds the original preprocessed image.
    try:
        img_orig_tensor = pytorch_preprocess_input(img_np.copy()) 
        augmented_images_tensors.append(img_orig_tensor)
    except Exception as e:
        print(f"Preprocessing original image failed: {e}")

    # Applies seasonal variations.
    seasons = ["summer", "autumn", "winter", "spring", "drought", "overwatered"]
    for season in random.sample(seasons, min(2, len(seasons))): 
        try:
            img_season_np = apply_seasonal_effect(img_np.copy(), season) 
            img_season_tensor = pytorch_preprocess_input(img_season_np) 
            augmented_images_tensors.append(img_season_tensor)
        except Exception as e:
            print(f"Seasonal effect {season} failed: {e}")
    
    # Applies plant-specific transformations.
    plant_transforms = ["leaf_wilt", "leaf_curl", "focus_blur", "growth_stage", "disease_spots"]
    for transform in random.sample(plant_transforms, min(2, len(plant_transforms))):
        try:
            img_transformed_np = apply_plant_specific_transform(img_np.copy(), transform) 
            img_transformed_tensor = pytorch_preprocess_input(img_transformed_np) 
            augmented_images_tensors.append(img_transformed_tensor)
        except Exception as e:
            print(f"Plant transform {transform} failed: {e}")
    
    # Applies lighting condition variations.
    lighting_conditions = ["shadow", "sunflare", "overexposed", "underexposed", "indoor"]
    for light in random.sample(lighting_conditions, min(2, len(lighting_conditions))):
        try:
            img_light_np = apply_lighting_condition(img_np.copy(), light) 
            img_light_tensor = pytorch_preprocess_input(img_light_np) 
            augmented_images_tensors.append(img_light_tensor)
        except Exception as e:
            print(f"Lighting condition {light} failed: {e}")
    
    # Applies occlusion.
    occlusions = ["partial_view"]
    for occlusion in random.sample(occlusions, min(1, len(occlusions))):
        try:
            img_occluded_np = apply_occlusion(img_np.copy(), occlusion) 
            img_occluded_tensor = pytorch_preprocess_input(img_occluded_np) 
            augmented_images_tensors.append(img_occluded_tensor)
        except Exception as e:
            print(f"Occlusion {occlusion} failed: {e}")
    
    # Applies scale variation.
    scales = ["macro", "distant"]
    for scale in random.sample(scales, min(1, len(scales))):
        try:
            img_scale_np = apply_scale_variation(img_np.copy(), scale) 
            img_scale_tensor = pytorch_preprocess_input(img_scale_np) 
            augmented_images_tensors.append(img_scale_tensor)
        except Exception as e:
            print(f"Scale variation {scale} failed: {e}")
    
    # Applies plant age variations.
    ages = ["young", "mature", "flowering"]
    for age in random.sample(ages, min(1, len(ages))):
        try:
            img_age_np = apply_plant_age_variation(img_np.copy(), age) 
            img_age_tensor = pytorch_preprocess_input(img_age_np) 
            augmented_images_tensors.append(img_age_tensor)
        except Exception as e:
            print(f"Age variation {age} failed: {e}")
    
    # Applies weather conditions.
    weather = ["rain", "fog", "bright_sun"]
    for condition in random.sample(weather, min(1, len(weather))):
        try:
            img_weather_np = apply_weather_condition(img_np.copy(), condition) 
            img_weather_tensor = pytorch_preprocess_input(img_weather_np) 
            augmented_images_tensors.append(img_weather_tensor)
        except Exception as e:
            print(f"Weather condition {condition} failed: {e}")
    
    # Applies feature-preserving augmentations.
    try:
        img_features_np = apply_feature_preserving_augmentation(img_np.copy(), class_name) 
        img_features_tensor = pytorch_preprocess_input(img_features_np) 
        augmented_images_tensors.append(img_features_tensor)
    except Exception as e:
        print(f"Feature-preserving augmentation failed: {e}")
        
    # Applies background variation.
    try:
        img_bg_np = apply_background_variation(img_np.copy()) 
        img_bg_tensor = pytorch_preprocess_input(img_bg_np) 
        augmented_images_tensors.append(img_bg_tensor)
    except Exception as e:
        print(f"Background variation failed: {e}")
        
    # Applies part-based augmentation.
    try:
        img_part_np = apply_part_based_augmentation(img_np.copy()) 
        img_part_tensor = pytorch_preprocess_input(img_part_np) 
        augmented_images_tensors.append(img_part_tensor)
    except Exception as e:
        print(f"Part-based augmentation failed: {e}")
    
    pytorch_geometric_transforms = [
        lambda x: TF.hflip(x),
        lambda x: torch.clamp(x + random.uniform(-0.2, 0.2), 0, 1),  # Brightness (additive)
        lambda x: TF.adjust_contrast(x, random.uniform(0.8, 1.2)),   # Contrast (multiplicative)
        lambda x: TF.adjust_saturation(x, random.uniform(0.8, 1.2)), # Saturation (multiplicative)
        lambda x: TF.rotate(x, angle=90.0),
        lambda x: TF.rotate(x, angle=270.0),
        lambda x: T.RandomCrop(size=(x.shape[1], x.shape[2]))(TF.pad(x, padding=(4,4,4,4), padding_mode='reflect'))
    ]

    img_chw_f01_tensor = torch.from_numpy(img_np.transpose((2,0,1)).copy()).float() / 255.0

    for pt_transform_fn in random.sample(pytorch_geometric_transforms, min(4, len(pytorch_geometric_transforms))):
        try:
            transformed_chw_f01_tensor = pt_transform_fn(img_chw_f01_tensor.clone()) # Apply on a clone
            # Convert CHW float [0,1] tensor back to HWC uint8 numpy for preprocessing
            transformed_hwc_uint8_np = (torch.clamp(transformed_chw_f01_tensor, 0, 1) * 255.0).byte().cpu().numpy().transpose((1,2,0))
            
            preprocessed_tensor = pytorch_preprocess_input(transformed_hwc_uint8_np)
            augmented_images_tensors.append(preprocessed_tensor)
        except Exception as e:
            # Get transform name if possible (difficult for lambdas)
            # transform_name = getattr(pt_transform_fn, '__name__', 'lambda')
            print(f"PyTorch transform failed: {e}")
    
    return augmented_images_tensors

def calculate_class_specific_augmentation(class_dir, class_name):
    """Calculates the number of augmentations needed for a specific class."""
    # Gets original image files (excluding already augmented ones).
    image_files = [f for f in os.listdir(class_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not '_aug' in f]
    orig_count = len(image_files)
    
    if orig_count == 0:
        return [], 0, 0
    
    # Calculates how many augmentations are needed to reach MIN_IMAGES_PER_CLASS.
    if orig_count < MIN_IMAGES_PER_CLASS:
        # Calculates total new augmented images needed.
        total_new_images_needed = MIN_IMAGES_PER_CLASS - orig_count
        
        # Calculates augmentations per original image.
        aug_per_image = total_new_images_needed / orig_count if orig_count > 0 else total_new_images_needed # Avoid div by zero
        
        # Sets the target count.
        target_count = MIN_IMAGES_PER_CLASS
    else:
        # No additional augmentations if class already has enough images.
        aug_per_image = 0
        target_count = orig_count
        
        # Indicates if class exceeds MAX_IMAGES_PER_CLASS.
        if orig_count > MAX_IMAGES_PER_CLASS:
            print(f"{TermColors.YELLOW}Class {class_name} has {orig_count} images which exceeds MAX_IMAGES_PER_CLASS ({MAX_IMAGES_PER_CLASS}), but no downsampling will be performed.{TermColors.ENDC}")
    
    # Returns the count of augmentations to create.
    return image_files, target_count, aug_per_image

def apply_feature_preserving_augmentation(img, plant_class):
    """Applies augmentations that aim to preserve key plant features."""
    img_copy = img.copy()
    h, w = img.shape[:2]
    
    # Detects if the plant is likely grass or similar.
    is_grass = any(grass_term in plant_class.lower() for grass_term in 
                  ['grass', 'poaceae', 'carex', 'juncus', 'cyperus', 'bamboo', 
                   'stipa', 'festuca', 'poa'])
    
    # Detects if the plant has small leaves or needles.
    is_small_leaved = any(term in plant_class.lower() for term in 
                         ['fern', 'moss', 'conifer', 'pine', 'juniper', 'cypress',
                          'leaf', 'needle', 'scale']) # Removed 'leaf' as it's too general
    
    # For grass-like plants, preserves vertical structure.
    if is_grass:
        # Applies vertical-preserving distortion.
        stretch_factor = random.uniform(0.9, 1.1)
        new_h = int(h * stretch_factor)
        new_h = max(1, new_h) # Ensure new_h is at least 1
        img_resized = cv2.resize(img_copy, (w, new_h))
        
        # Resizes back to original dimensions by cropping or padding.
        if new_h > h:
            # Crops centre if stretched.
            start_y = (new_h - h) // 2
            img_copy = img_resized[start_y:start_y+h, :]
        elif new_h < h : # Pad if compressed
            # Pads with edge content if compressed.
            pad_top = (h - new_h) // 2
            pad_bottom = h - new_h - pad_top
            img_copy = cv2.copyMakeBorder(img_resized, pad_top, pad_bottom, 0, 0, 
                                        cv2.BORDER_REPLICATE)
        else: # new_h == h
            img_copy = img_resized

        # Enhances edges to emphasise blade margins and venation.
        try:
            gray = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges = cv2.dilate(edges, None, iterations=1) # Added iterations
            edge_mask_single_channel = cv2.GaussianBlur(edges, (3, 3), 0) / 255.0 
            
            # Creates 3-channel edge mask.
            edge_mask_3c = np.stack([edge_mask_single_channel] * 3, axis=2) 
            
            # Subtly enhances edges.
            img_copy_float = img_copy.astype(np.float32) 
            img_copy_float = img_copy_float * (1 + 0.1 * edge_mask_3c)
            img_copy = np.clip(img_copy_float, 0, 255).astype(np.uint8)
        except Exception as e: # Catch specific cv2 errors if possible
            print(f"Grass edge enhancement failed: {e}")
            pass  # Falls back if edge detection fails.
            
    # For small-leaved or needle-like plants, enhances texture.
    elif is_small_leaved:
        # Enhances fine details and texture.
        try:
            # Uses detail enhancement.
            img_copy = cv2.detailEnhance(img_copy, sigma_s=15, sigma_r=0.15)
            
            # Applies slight sharpening.
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
            img_copy = cv2.filter2D(img_copy.astype(np.float32), -1, kernel)
            img_copy = np.clip(img_copy, 0, 255).astype(np.uint8)
        except Exception as e:
            print(f"Small-leaved enhancement failed: {e}")
            pass  # Falls back if enhancement fails.
    
    # For all other plants, preserves colour and enhances key features.
    else:
        # Enhances contrast using CLAHE on L channel of LAB colour space.
        try:
            # Converts to LAB colour space.
            lab = cv2.cvtColor(img_copy, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab) 
            
            # Applies CLAHE to L channel.
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl_enhanced = clahe.apply(l_channel) 
            
            # Merges channels back.
            enhanced_lab = cv2.merge((cl_enhanced, a_channel, b_channel))
            
            # Converts back to RGB.
            img_copy = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
            
            # Adds a small saturation boost.
            hsv = cv2.cvtColor(img_copy, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:,:,1] *= 1.1  # Increases saturation by 10%.
            hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
            img_copy = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        except Exception as e:
            print(f"General plant enhancement failed: {e}")
            pass  # Falls back if enhancement fails.
    
    return img_copy.astype(np.uint8)

def apply_background_variation(img):
    """Applies variations to the image background."""
    h, w = img.shape[:2]
    img_copy = img.copy()
    
    # Chooses a background variation technique.
    variation_type = random.choice(["gradient", "texture", "blur", "natural"])
    
    # Attempts to segment the plant from the background.
    plant_mask_final = None 
    try:
        # Converts to HSV for plant segmentation.
        hsv = cv2.cvtColor(img_copy, cv2.COLOR_RGB2HSV)
        
        # Creates a mask focusing on green/brown plant parts.
        s_channel = hsv[:,:,1] 
        v_channel = hsv[:,:,2] 
        
        # Applies adaptive thresholding on saturation and value channels.
        s_thresh = cv2.adaptiveThreshold(s_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
        v_thresh = cv2.adaptiveThreshold(v_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
        
        # Combines masks.
        plant_mask_initial = cv2.bitwise_and(s_thresh, v_thresh) 
        
        # Cleans up mask with morphological operations.
        kernel = np.ones((5, 5), np.uint8)
        plant_mask_cleaned = cv2.morphologyEx(plant_mask_initial, cv2.MORPH_CLOSE, kernel) 
        plant_mask_cleaned = cv2.morphologyEx(plant_mask_cleaned, cv2.MORPH_OPEN, kernel)
        
        # Dilates mask to include the entire plant.
        plant_mask_dilated = cv2.dilate(plant_mask_cleaned, kernel, iterations=2) 
        
        # Falls back to centre-focused mask if segmentation yields too little foreground.
        if np.mean(plant_mask_dilated) < 10: 
            y_grid, x_grid = np.ogrid[0:h, 0:w] 
            center_y, center_x = h/2, w/2
            dist_from_center = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
            radius = min(h, w) * 0.4  # Assumes centre 40% is plant.
            plant_mask_final = (dist_from_center <= radius).astype(np.uint8) * 255
        else:
            plant_mask_final = plant_mask_dilated

    except Exception as e:
        print(f"Plant segmentation for background variation failed: {e}")
        # Falls back to centre-focused mask if segmentation fails.
        y_grid, x_grid = np.ogrid[0:h, 0:w] 
        center_y, center_x = h/2, w/2
        dist_from_center = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
        radius = min(h, w) * 0.4 
        plant_mask_final = (dist_from_center <= radius).astype(np.uint8) * 255
    
    # Ensures plant_mask_final is 3 channels for blending.
    if len(plant_mask_final.shape) == 2:
        plant_mask_3channel = np.stack([plant_mask_final] * 3, axis=2) / 255.0 
    else: # Already 3 channels (should not happen with current logic but good practice)
        plant_mask_3channel = plant_mask_final / 255.0

    background_new = np.zeros_like(img_copy, dtype=np.float32) 
    # Creates new background based on variation type.
    if variation_type == "gradient":
        # Creates a colour gradient background.
        direction = random.choice(["horizontal", "vertical", "radial"])
        color1 = np.array([random.randint(100, 180) for _ in range(3)], dtype=np.float32) 
        color2 = np.array([random.randint(100, 180) for _ in range(3)], dtype=np.float32)
        
        if direction == "horizontal":
            gradient = np.linspace(0, 1, w).reshape(1, w, 1)
            gradient = np.tile(gradient, (h, 1, 3))
            background_new = color1 * (1 - gradient) + color2 * gradient
        elif direction == "vertical":
            gradient = np.linspace(0, 1, h).reshape(h, 1, 1)
            gradient = np.tile(gradient, (1, w, 3))
            background_new = color1 * (1 - gradient) + color2 * gradient
        else:  # radial
            y_grid, x_grid = np.ogrid[0:h, 0:w] 
            center_y, center_x = h/2, w/2
            max_dist_val = np.sqrt(center_x**2 + center_y**2) + 1e-6 
            dist_from_center = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2) / max_dist_val
            gradient = np.clip(dist_from_center, 0, 1)
            gradient = np.stack([gradient] * 3, axis=2)
            background_new = color1 * (1 - gradient) + color2 * gradient
            
    elif variation_type == "texture":
        # Creates a textured background.
        texture_type = random.choice(["noise", "stripes", "dots"])
        
        if texture_type == "noise":
            # Creates random noise background.
            background_new = np.random.randint(100, 180, (h, w, 3)).astype(np.float32)
            # Smooths noise slightly.
            background_new = cv2.GaussianBlur(background_new, (21, 21), 0)
            
        elif texture_type == "stripes":
            # Creates striped background.
            background_new = np.zeros((h, w, 3), dtype=np.float32)
            stripe_width = random.randint(10, 30)
            color1 = np.array([random.randint(100, 180)] * 3, dtype=np.float32)
            color2 = np.array([random.randint(100, 180)] * 3, dtype=np.float32)
            
            if random.choice([True, False]): # Horizontal or vertical stripes.
                # Horizontal stripes.
                for i in range(0, h, stripe_width*2):
                    background_new[i:min(i+stripe_width,h), :] = color1 # Use min to avoid exceeding h
                    if i+stripe_width < h:
                         background_new[i+stripe_width:min(i+stripe_width*2,h), :] = color2
            else:
                # Vertical stripes.
                for i in range(0, w, stripe_width*2):
                    background_new[:, i:min(i+stripe_width,w)] = color1
                    if i+stripe_width < w:
                        background_new[:, i+stripe_width:min(i+stripe_width*2,w)] = color2
                    
        else:  # dots
            # Creates dotted background.
            background_new = np.ones((h, w, 3), dtype=np.float32) * 160
            dot_radius = random.randint(5, 15)
            num_dots = random.randint(20, 50)
            
            for _ in range(num_dots):
                cx = random.randint(0, w-1)
                cy = random.randint(0, h-1)
                color_val = np.array([random.randint(100, 180)] * 3, dtype=np.float32) 
                
                y_grid, x_grid = np.ogrid[0:h, 0:w] 
                dist_from_center = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
                mask = dist_from_center <= dot_radius
                
                for c_idx in range(3): 
                    background_new[:,:,c_idx][mask] = color_val[c_idx]
                    
    elif variation_type == "blur":
        # Creates blurred version of the original background.
        # Extracts presumed background and blurs it.
        background_only = img_copy.astype(np.float32) * (1 - plant_mask_3channel)
        background_new = cv2.GaussianBlur(background_only, (51, 51), 0)
        
        # Fills in any black spots resulting from masking.
        background_mask_fill = np.all(background_new < 5, axis=2) 
        if np.any(background_mask_fill):
            non_masked_pixels = background_new[~background_mask_fill]
            if non_masked_pixels.size > 0:
                 avg_color = np.mean(non_masked_pixels, axis=0)
                 for c_idx in range(3): 
                    background_new[:,:,c_idx][background_mask_fill] = avg_color[c_idx]
            else: # If all pixels are masked (e.g. very small image or full plant mask)
                background_new[background_mask_fill] = 128 # Fill with grey
                
    else:  # natural
        # Uses a natural gradient like sky or soil.
        bg_type = random.choice(["sky", "soil", "foliage"])
        
        if bg_type == "sky":
            # Creates sky gradient (blue to light blue/white).
            background_new = np.zeros((h, w, 3), dtype=np.float32)
            if h > 0:
                for y_coord in range(h): 
                    factor = y_coord / h
                    background_new[y_coord, :, 0] = 135 + 80 * factor  # R increases.
                    background_new[y_coord, :, 1] = 206 + 30 * factor  # G increases.
                    background_new[y_coord, :, 2] = 235 + 20 * factor  # B increases slightly.
                
        elif bg_type == "soil":
            # Creates soil/ground texture (browns).
            background_new = np.zeros((h, w, 3), dtype=np.float32)
            
            # Sets base brown colour.
            background_new[:,:,0] = random.randint(140, 180)  # R.
            background_new[:,:,1] = random.randint(100, 140)  # G.
            background_new[:,:,2] = random.randint(60, 100)   # B.
            
            # Adds noise for texture.
            noise = np.random.normal(0, 15, (h, w, 3)).astype(np.float32)
            background_new = np.clip(background_new + noise, 0, 255)
            
        else:  # foliage
            # Creates green foliage background.
            background_new = np.zeros((h, w, 3), dtype=np.float32)
            
            # Sets base green colour.
            background_new[:,:,0] = random.randint(60, 120)   # R.
            background_new[:,:,1] = random.randint(120, 180)  # G.
            background_new[:,:,2] = random.randint(60, 100)   # B.
            
            # Adds noise for texture.
            noise = np.random.normal(0, 15, (h, w, 3)).astype(np.float32)
            background_new = np.clip(background_new + noise, 0, 255)
    
    # Softens mask edges for natural blending.
    plant_mask_soft = cv2.GaussianBlur(plant_mask_3channel, (21, 21), 0)
    
    # Blends original image with new background using the plant mask.
    result = img_copy.astype(np.float32) * plant_mask_soft + background_new * (1 - plant_mask_soft)
    
    return result.astype(np.uint8)

def apply_part_based_augmentation(img):
    """Applies augmentations focused on specific plant parts."""
    img_copy = img.copy()
    h, w = img.shape[:2]
    
    # Chooses a random plant part or feature to focus on.
    feature_type = random.choice(["flowers", "leaves", "stem", "texture", "shape"])
    
    # Applies part-specific enhancements.
    if feature_type == "flowers":
        # Enhances flower visibility and colour.
        try:
            # Converts to HSV.
            hsv = cv2.cvtColor(img_copy, cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # Boosts saturation.
            hsv[:,:,1] *= 1.3
            hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
            
            # Boosts brightness slightly.
            hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.1, 0, 255)
            
            # Converts back to RGB.
            img_enhanced_hsv = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB) 
            
            # Applies subtle sharpening to enhance flower details.
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
            # Blend sharpened with original to make it subtle
            sharpened_img = cv2.filter2D(img_enhanced_hsv.astype(np.float32), -1, kernel)
            img_copy = cv2.addWeighted(img_enhanced_hsv.astype(np.float32), 0.7, sharpened_img, 0.3, 0)
            img_copy = np.clip(img_copy, 0, 255).astype(np.uint8)

        except Exception as e:
            print(f"Flower enhancement failed: {e}")
            
    elif feature_type == "leaves":
        # Enhances leaf visibility, edges, and venation.
        try:
            img_copy_float = img_copy.astype(np.float32)
            # Enhances green channel slightly.
            img_copy_float[:,:,1] = np.clip(img_copy_float[:,:,1] * 1.1, 0, 255)
            img_copy_intermediate = img_copy_float.astype(np.uint8) 
            
            # Applies CLAHE to enhance contrast (helps with venation).
            lab = cv2.cvtColor(img_copy_intermediate, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab) 
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl_enhanced = clahe.apply(l_channel) 
            enhanced_lab = cv2.merge((cl_enhanced, a_channel, b_channel))
            img_clahe_enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB) 
            
            # Enhances edges for leaf margins and venation.
            gray = cv2.cvtColor(img_clahe_enhanced, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 30, 100)
            edges = cv2.dilate(edges, None, iterations=1) # Added iterations
            edge_mask_single_channel = cv2.GaussianBlur(edges, (3, 3), 0) / 255.0 
            
            # Creates 3-channel edge mask.
            edge_mask_3c = np.stack([edge_mask_single_channel] * 3, axis=2) 
            
            # Enhances edges.
            img_copy_float = img_clahe_enhanced.astype(np.float32)
            img_copy_float = img_copy_float * (1 + 0.2 * edge_mask_3c)  # 20% edge enhancement.
            img_copy = np.clip(img_copy_float, 0, 255).astype(np.uint8)
        except Exception as e:
            print(f"Leaf enhancement failed: {e}")
            
    elif feature_type == "stem":
        # Enhances stem visibility and structure.
        try:
            # Creates edge-enhanced version highlighting linear structures.
            gray = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)
            
            # Applies vertical Sobel filter (good for stems).
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            sobel_y_abs = np.absolute(sobel_y) 
            sobel_y_norm = np.clip(sobel_y_abs, 0, 255).astype(np.uint8) 
            
            # Enhances Sobel result.
            sobel_y_blurred = cv2.GaussianBlur(sobel_y_norm, (3, 3), 0) 
            
            # Converts to 3 channels for blending.
            sobel_3c = np.stack([sobel_y_blurred] * 3, axis=2) / 255.0
            
            # Enhances contrast in the original image.
            lab = cv2.cvtColor(img_copy, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab) 
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl_enhanced = clahe.apply(l_channel) 
            enhanced_lab = cv2.merge((cl_enhanced, a_channel, b_channel))
            enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
            
            # Blends edge information with enhanced image.
            img_copy_float = enhanced_img.astype(np.float32) 
            img_copy_float = img_copy_float * (1 + 0.3 * sobel_3c)  # 30% stem enhancement.
            img_copy = np.clip(img_copy_float, 0, 255).astype(np.uint8)
        except Exception as e:
            print(f"Stem enhancement failed: {e}")
            
    elif feature_type == "texture":
        # Enhances overall plant texture.
        try:
            # Applies detail enhancement algorithm.
            img_detail_enhanced = cv2.detailEnhance(img_copy, sigma_s=10, sigma_r=0.15) 
            
            # Converts to LAB colour space for contrast enhancement.
            lab = cv2.cvtColor(img_detail_enhanced, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab) 
            
            # Applies CLAHE to L channel.
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl_enhanced = clahe.apply(l_channel) 
            
            # Merges channels back.
            enhanced_lab = cv2.merge((cl_enhanced, a_channel, b_channel))
            
            # Converts back to RGB.
            img_copy = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        except Exception as e:
            print(f"Texture enhancement failed: {e}")
    
    elif feature_type == "shape":
        # Enhances overall plant shape/silhouette.
        try:
            # Creates edge map of plant outline.
            gray = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 30, 100)
            
            # Dilates edges to enhance outline.
            kernel = np.ones((3, 3), np.uint8)
            edges_dilated = cv2.dilate(edges, kernel, iterations=1) 
            
            # Converts to 3 channels.
            edges_3c = np.stack([edges_dilated] * 3, axis=2) / 255.0
            
            # Blends with slight darkening of background.
            hsv = cv2.cvtColor(img_copy, cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # Reduces value (brightness) slightly except at edges.
            hsv[:,:,2] = hsv[:,:,2] * (0.9 + 0.2 * edges_3c[:,:,0]) # Ensure edges_3c is broadcastable
            hsv[:,:,2] = np.clip(hsv[:,:,2], 0, 255)
            
            # Converts back to RGB.
            img_copy = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        except Exception as e:
            print(f"Shape enhancement failed: {e}")
    
    return img_copy.astype(np.uint8)

def apply_seasonal_effect(img, season_type):
    """Applies colour transformations to simulate seasonal effects."""
    img_float = img.astype(np.float32)
    
    if season_type == "summer":
        # Simulates summer: brighter, more saturated, slightly yellow tint.
        img_float = img_float * 1.1  # Brighter.
        img_float[:,:,2] = img_float[:,:,2] * 0.95  # Reduces blue channel.
        img_float[:,:,1] = img_float[:,:,1] * 1.05  # Increases green channel.
        
    elif season_type == "autumn":
        # Simulates autumn: orange/brown tint, lower saturation.
        img_float[:,:,0] = img_float[:,:,0] * 1.15  # Increases red channel.
        img_float[:,:,1] = img_float[:,:,1] * 0.85  # Reduces green channel.
        img_float[:,:,2] = img_float[:,:,2] * 0.7   # Reduces blue channel.
        
    elif season_type == "winter":
        # Simulates winter: cooler, bluer, less saturated.
        img_float = img_float * 0.85  # Darker.
        img_float[:,:,0] = img_float[:,:,0] * 0.8  # Reduces red channel.
        img_float[:,:,2] = img_float[:,:,2] * 1.2  # Increases blue channel.
        
    elif season_type == "spring":
        # Simulates spring: more green, slightly brighter.
        img_float[:,:,1] = img_float[:,:,1] * 1.15  # Increases green channel.
        img_float = img_float * 1.05  # Slightly brighter.
        
    elif season_type == "drought":
        # Simulates drought stress: yellow/brown tones.
        img_float[:,:,0] = img_float[:,:,0] * 1.2   # Increases red.
        img_float[:,:,1] = img_float[:,:,1] * 0.9   # Decreases green.
        img_float[:,:,2] = img_float[:,:,2] * 0.7   # Decreases blue.
        
    elif season_type == "overwatered":
        # Simulates overwatered plants: more blue/green.
        img_float[:,:,0] = img_float[:,:,0] * 0.8   # Decreases red.
        img_float[:,:,1] = img_float[:,:,1] * 0.95  # Slightly decreases green.
        img_float[:,:,2] = img_float[:,:,2] * 1.2   # Increases blue.
        
    # Clips values to valid range and converts back to uint8.
    img_float = np.clip(img_float, 0, 255)
    return img_float.astype(np.uint8)

def apply_lighting_condition(img, light_type):
    """Applies different lighting condition effects to an image."""
    img_float = img.astype(np.float32)
    h, w = img.shape[:2] # Moved h,w definition up
    
    if light_type == "shadow":
        # Creates a shadow effect on part of the image.
        shadow_mask = np.ones_like(img_float)
        
        # Creates a gradient shadow in a random direction.
        direction = random.choice(["top", "left", "right", "bottom"])
        
        if h > 0 and w > 0: # Ensure h and w are positive
            if direction == "top":
                for i in range(h):
                    shadow_mask[i,:,:] = 1.0 - (0.5 * i / h)
            elif direction == "left":
                for i in range(w):
                    shadow_mask[:,i,:] = 1.0 - (0.5 * i / w)
            elif direction == "right":
                for i in range(w):
                    shadow_mask[:,w-1-i,:] = 1.0 - (0.5 * i / w) # Corrected right direction
            else:  # bottom
                for i in range(h):
                    shadow_mask[h-1-i,:,:] = 1.0 - (0.5 * i / h) # Corrected bottom direction
        
        # Applies shadow.
        img_float = img_float * shadow_mask
        
    elif light_type == "sunflare":
        # Adds a sunflare effect (bright spot with halo).
        
        # Sets random position for the flare.
        flare_x = random.randint(0, w-1) if w > 0 else 0
        flare_y = random.randint(0, h-1) if h > 0 else 0
        
        # Creates distance matrix from flare point.
        y_grid, x_grid = np.ogrid[-flare_y:h-flare_y, -flare_x:w-flare_x] 
        dist = np.sqrt(x_grid*x_grid + y_grid*y_grid) 
        
        # Creates flare mask (bright in centre, fading out).
        flare_mask = np.zeros_like(img_float)
        max_dist_flare = min(h, w) / 3 if min(h,w) > 0 else 1 
        flare_intensity_val = (1 - np.clip(dist / (max_dist_flare + 1e-6), 0, 1)) * 100 
        
        # Applies to all channels with yellow-white tint.
        flare_mask[:,:,0] = flare_intensity_val * 1.0  # Red.
        flare_mask[:,:,1] = flare_intensity_val * 1.0  # Green.
        flare_mask[:,:,2] = flare_intensity_val * 0.8  # Blue (slightly less for yellow tint).
        
        # Adds flare to image.
        img_float = img_float + flare_mask
        
    elif light_type == "overexposed":
        # Simulates overexposed parts of the image.
        img_float = img_float * 1.4
        
    elif light_type == "underexposed":
        # Simulates underexposed image (e.g., in shade).
        img_float = img_float * 0.7
        
    elif light_type == "indoor":
        # Simulates indoor lighting (yellowish tint).
        img_float[:,:,0] = img_float[:,:,0] * 1.1  # More red.
        img_float[:,:,1] = img_float[:,:,1] * 1.05  # More green.
        img_float[:,:,2] = img_float[:,:,2] * 0.85  # Less blue.
    
    # Clips values to valid range and converts back to uint8.
    img_float = np.clip(img_float, 0, 255)
    return img_float.astype(np.uint8)

def apply_part_focused_augmentation(img, plant_part="auto"):
    """Applies augmentations focused on automatically or manually specified plant parts."""
    h, w = img.shape[:2]
    
    # Creates a copy to avoid modifying the original.
    result = img.copy()
    
    # Automatically detects plant part if not specified.
    if plant_part == "auto":
        # Converts to HSV for colour segmentation.
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # Defines colour ranges for flower detection.
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])
        
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask_red1, mask_red2)
        
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Combines flower masks.
        current_flower_mask = cv2.bitwise_or(red_mask, yellow_mask) 
        current_flower_mask = cv2.bitwise_or(current_flower_mask, white_mask)
        
        # Defines colour range for leaf detection.
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([90, 255, 255])
        current_leaf_mask = cv2.inRange(hsv, lower_green, upper_green) 
        
        # Defines colour range for fruit detection.
        current_fruit_mask = cv2.bitwise_or(red_mask, yellow_mask) 
        
        # Determines dominant part based on pixel counts.
        flower_pixels = np.sum(current_flower_mask > 0)
        leaf_pixels = np.sum(current_leaf_mask > 0)
        fruit_pixels = np.sum(current_fruit_mask > 0)
        
        # Assigns plant_part based on dominant feature or random choice.
        # Ensure h and w are positive for area calculation
        min_pixels_for_detection = (h * w * 0.05) if h > 0 and w > 0 else 0

        if flower_pixels > leaf_pixels and flower_pixels > fruit_pixels and flower_pixels > min_pixels_for_detection:
            plant_part = "flower"
        elif leaf_pixels > flower_pixels and leaf_pixels > fruit_pixels and leaf_pixels > min_pixels_for_detection:
            plant_part = "leaf"
        elif fruit_pixels > min_pixels_for_detection:
            plant_part = "fruit"
        else:
            # Chooses randomly if no clear part is detected.
            plant_part = random.choice(["flower", "leaf", "stem", "fruit", "whole"])
    
    if plant_part == "flower":
        # Enhances flowers: increases saturation and brightness in flower regions.
        hsv_flower = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32) 
        if 'current_flower_mask' not in locals() or plant_part != "auto": # Recalculate if not auto or not available
            hsv_temp = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
            lower_red1_fl = np.array([0, 100, 100]) 
            upper_red1_fl = np.array([10, 255, 255])
            # ... (rest of flower mask calculation as in 'auto' section) ...
            # This is repetitive, better to structure to reuse mask from 'auto' if possible
            # For now, simplified:
            mask_to_use = current_flower_mask if 'current_flower_mask' in locals() and plant_part == "auto" else cv2.inRange(hsv_temp, lower_red1_fl, upper_red1_fl) # Simplified example
        else:
            mask_to_use = current_flower_mask

        kernel = np.ones((5, 5), np.uint8)
        flower_mask_dilated = cv2.dilate(mask_to_use, kernel, iterations=1) 
        flower_mask_float = flower_mask_dilated.astype(np.float32) / 255.0 
        flower_mask_3d = np.stack([flower_mask_float] * 3, axis=2)
        
        hsv_flower[:,:,1] *= 1.0 + (flower_mask_float * 0.3) 
        hsv_flower[:,:,2] *= 1.0 + (flower_mask_float * 0.2)  
        hsv_flower[:,:,0] = np.clip(hsv_flower[:,:,0], 0, 179)
        hsv_flower[:,:,1:] = np.clip(hsv_flower[:,:,1:], 0, 255)
        enhanced = cv2.cvtColor(hsv_flower.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        background_blur = cv2.GaussianBlur(result, (5, 5), 0) 
        result = (background_blur.astype(np.float32) * (1 - flower_mask_3d) + enhanced.astype(np.float32) * flower_mask_3d).astype(np.uint8)
        
    elif plant_part == "leaf":
        # Similar logic for leaf_mask as for flower_mask
        if 'current_leaf_mask' not in locals() or plant_part != "auto":
            hsv_temp = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
            lower_green_lf = np.array([35, 40, 40]) 
            upper_green_lf = np.array([90, 255, 255]) 
            mask_to_use = cv2.inRange(hsv_temp, lower_green_lf, upper_green_lf)
        else:
            mask_to_use = current_leaf_mask
            
        kernel_lf = np.ones((5, 5), np.uint8) 
        leaf_mask_dilated = cv2.dilate(mask_to_use, kernel_lf, iterations=1) 
        leaf_mask_float = leaf_mask_dilated.astype(np.float32) / 255.0 
        leaf_mask_3d = np.stack([leaf_mask_float] * 3, axis=2)
        
        enhanced_lf = result.astype(np.float32) 
        enhanced_lf[:,:,1] *= 1.0 + (leaf_mask_float * 0.2)
        
        gray_lf = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY) 
        clahe_lf = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)) 
        enhanced_gray_lf = clahe_lf.apply(gray_lf).astype(np.float32)
        
        for c_idx in range(3): 
            blend_factor = leaf_mask_float * 0.5
            enhanced_lf[:,:,c_idx] = enhanced_lf[:,:,c_idx] * (1 - blend_factor) + \
                              enhanced_gray_lf * blend_factor
        
        background_lf_blur = cv2.GaussianBlur(result, (3, 3), 0) 
        result = (background_lf_blur.astype(np.float32) * (1 - leaf_mask_3d) + enhanced_lf * leaf_mask_3d)
        result = np.clip(result, 0, 255).astype(np.uint8)
        
    elif plant_part == "stem":
        gray_st = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY) 
        sobelx_st = cv2.Sobel(gray_st, cv2.CV_64F, 1, 0, ksize=3) 
        sobely_st = cv2.Sobel(gray_st, cv2.CV_64F, 0, 1, ksize=3) 
        magnitude_st = np.sqrt(sobelx_st**2 + sobely_st**2) 
        angle_st = np.arctan2(sobely_st, sobelx_st + 1e-6) * 180 / np.pi # Add epsilon
        
        vertical_mask_st = np.zeros_like(gray_st, dtype=np.float32) 
        vertical_regions_st = np.logical_or( 
            np.logical_and(angle_st > 70, angle_st < 110),
            np.logical_and(angle_st < -70, angle_st > -110)
        )
        vertical_mask_st[vertical_regions_st] = magnitude_st[vertical_regions_st]
        
        if vertical_mask_st.max() > 0:
            vertical_mask_st = vertical_mask_st / vertical_mask_st.max()
        
        kernel_st = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], dtype=np.float32) 
        sharpened_st = cv2.filter2D(result.astype(np.float32), -1, kernel_st) 
        
        vertical_mask_3d_st = np.stack([vertical_mask_st] * 3, axis=2) 
        result_float = result.astype(np.float32) * (1 - vertical_mask_3d_st * 0.7) + sharpened_st * (vertical_mask_3d_st * 0.7) 
        result = np.clip(result_float, 0, 255).astype(np.uint8)
        
    elif plant_part == "fruit":
        if 'current_fruit_mask' not in locals() or plant_part != "auto":
            hsv_temp = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
            lower_red1_fr = np.array([0, 100, 100]) 
            upper_red1_fr = np.array([10, 255, 255])
            lower_red2_fr = np.array([160, 100, 100])
            upper_red2_fr = np.array([179, 255, 255])
            lower_yellow_fr = np.array([15, 100, 100])
            upper_yellow_fr = np.array([35, 255, 255])
            mask_red1_fr = cv2.inRange(hsv_temp, lower_red1_fr, upper_red1_fr)
            mask_red2_fr = cv2.inRange(hsv_temp, lower_red2_fr, upper_red2_fr)
            yellow_mask_fr = cv2.inRange(hsv_temp, lower_yellow_fr, upper_yellow_fr)
            mask_to_use = cv2.bitwise_or(cv2.bitwise_or(mask_red1_fr, mask_red2_fr), yellow_mask_fr)
        else:
            mask_to_use = current_fruit_mask

        kernel_fr = np.ones((5, 5), np.uint8) 
        fruit_mask_dilated = cv2.dilate(mask_to_use, kernel_fr, iterations=1) 
        fruit_mask_float = fruit_mask_dilated.astype(np.float32) / 255.0 
        fruit_mask_3d = np.stack([fruit_mask_float] * 3, axis=2)
        
        hsv_fr = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32) 
        hsv_fr[:,:,1] *= 1.0 + (fruit_mask_float * 0.3) 
        hsv_fr[:,:,2] *= 1.0 + (fruit_mask_float * 0.1)  
        hsv_fr[:,:,0] = np.clip(hsv_fr[:,:,0], 0, 179)
        hsv_fr[:,:,1:] = np.clip(hsv_fr[:,:,1:], 0, 255)
        enhanced_fr = cv2.cvtColor(hsv_fr.astype(np.uint8), cv2.COLOR_HSV2RGB) 
        
        background_fr_blur = cv2.GaussianBlur(result, (5, 5), 0) 
        result = (background_fr_blur.astype(np.float32) * (1 - fruit_mask_3d) + enhanced_fr.astype(np.float32) * fruit_mask_3d).astype(np.uint8)
    
    elif plant_part == "whole":
        kernel_wh = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], dtype=np.float32) 
        sharpened_wh = cv2.filter2D(result.astype(np.float32), -1, kernel_wh)
        sharpened_wh = np.clip(sharpened_wh, 0, 255).astype(np.uint8)
        
        lab_wh = cv2.cvtColor(sharpened_wh, cv2.COLOR_RGB2LAB) 
        l_wh, a_wh, b_wh = cv2.split(lab_wh) 
        clahe_wh = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)) 
        l_clahe_wh = clahe_wh.apply(l_wh) 
        lab_wh_clahe = cv2.merge((l_clahe_wh, a_wh, b_wh)) 
        result = cv2.cvtColor(lab_wh_clahe, cv2.COLOR_LAB2RGB)
    
    return result.astype(np.uint8)

def apply_geometric_transform(img, transform_type):
    """Applies various geometric transformations to an image."""
    h, w = img.shape[:2]
    
    if transform_type == 'rotate':
        # Applies random rotation between -30 and 30 degrees.
        angle = random.uniform(-30, 30)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
    elif transform_type == 'flip':
        # Applies random flip (horizontal, vertical, or both).
        flip_mode = random.choice([-1, 0, 1]) # 0 vertical, 1 horizontal, -1 both
        return cv2.flip(img, flip_mode)
        
    elif transform_type == 'crop':
        # Applies random crop and resize.
        crop_percent = random.uniform(0.8, 0.95)
        crop_w = int(w * crop_percent)
        crop_h = int(h * crop_percent)
        
        # Sets random position for crop.
        x_pos = random.randint(0, w - crop_w) if w > crop_w else 0
        y_pos = random.randint(0, h - crop_h) if h > crop_h else 0
        
        # Performs crop.
        crop_img = img[y_pos:y_pos+crop_h, x_pos:x_pos+crop_w] 
        
        # Resizes back to original dimensions.
        return cv2.resize(crop_img, (w, h), interpolation=cv2.INTER_AREA)
        
    elif transform_type == 'perspective':
        # Applies random perspective transform.
        # Defines the 4 source points.
        pts1 = np.float32([
            [0, 0],
            [w-1, 0], # Use w-1, h-1 for valid pixel coords
            [0, h-1],
            [w-1, h-1]
        ])
        
        # Defines the 4 destination points with random perturbation.
        perturbation = random.uniform(0.05, 0.15)
        max_offset_w = int(w * perturbation)
        max_offset_h = int(h * perturbation)
        
        pts2 = np.float32([
            [random.randint(0, max_offset_w), random.randint(0, max_offset_h)],
            [random.randint(w-1-max_offset_w, w-1), random.randint(0, max_offset_h)],
            [random.randint(0, max_offset_w), random.randint(h-1-max_offset_h, h-1)],
            [random.randint(w-1-max_offset_w, w-1), random.randint(h-1-max_offset_h, h-1)]
        ])
        
        # Gets transformation matrix and applies the perspective transform.
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    return img  # Returns original if no transform applied.

def apply_color_transform(img, transform_type):
    """Applies various colour transformations to an image."""
    img_float = img.astype(np.float32)
    
    if transform_type == 'brightness':
        # Adjusts brightness randomly.
        factor = random.uniform(0.7, 1.3)
        img_float = img_float * factor
        
    elif transform_type == 'contrast':
        # Adjusts contrast randomly.
        factor = random.uniform(0.7, 1.3)
        mean_val = np.mean(img_float, axis=(0, 1), keepdims=True) 
        img_float = (img_float - mean_val) * factor + mean_val
        
    elif transform_type == 'saturation':
        # Adjusts saturation randomly.
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        factor = random.uniform(0.7, 1.3)
        hsv[:, :, 1] = hsv[:, :, 1] * factor
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        img_float = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) # Ensure float output
        
    elif transform_type == 'hue':
        # Shifts hue randomly.
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) # uint8 input for hue
        shift = random.randint(-10, 10) # Hue is 0-179 in OpenCV.
        hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + shift) % 180 # Use int16 for intermediate calc
        img_float = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) # Ensure float output
        
    elif transform_type == 'noise':
        # Adds random noise.
        noise_type = random.choice(['gaussian', 'salt_pepper'])
        
        if noise_type == 'gaussian':
            # Adds Gaussian noise.
            stddev = random.uniform(5, 20)
            noise = np.random.normal(0, stddev, img.shape).astype(np.float32)
            img_float = img_float + noise
            
        else:  # salt_pepper
            # Adds salt and pepper noise.
            s_vs_p = 0.5  # Ratio of salt vs pepper.
            amount = random.uniform(0.01, 0.04)  # Total amount of noise.
            img_float_copy = img_float.copy() # Work on the float copy
            
            # Salt (white) noise.
            num_salt = int(amount * img.size * s_vs_p / 3) # Divide by 3 as img.size is total elements
            if img.shape[0] > 1 and img.shape[1] > 1 and num_salt > 0:
                coords_salt_dim1 = np.random.randint(0, img.shape[0], num_salt) 
                coords_salt_dim2 = np.random.randint(0, img.shape[1], num_salt)
                img_float_copy[coords_salt_dim1, coords_salt_dim2, :] = 255.0
            
            # Pepper (black) noise.
            num_pepper = int(amount * img.size * (1 - s_vs_p) / 3)
            if img.shape[0] > 1 and img.shape[1] > 1 and num_pepper > 0:
                coords_pepper_dim1 = np.random.randint(0, img.shape[0], num_pepper)
                coords_pepper_dim2 = np.random.randint(0, img.shape[1], num_pepper)
                img_float_copy[coords_pepper_dim1, coords_pepper_dim2, :] = 0.0
            img_float = img_float_copy 
    
    # Clips values to valid range and converts back to uint8.
    img_float = np.clip(img_float, 0, 255).astype(np.uint8)
    return img_float

def apply_plant_specific_transform(img, transform_type):
    """Applies transformations specific to plant characteristics."""
    img_copy = img.copy()
    h, w = img.shape[:2]
    
    if transform_type == "leaf_wilt":
        # Simulates wilting leaves with downward distortion.
        rows, cols = img.shape[:2]
        
        # Creates distortion map.
        map_y, map_x = np.mgrid[0:rows, 0:cols].astype(np.float32)
        
        # Applies downward curve, stronger at edges.
        center_col = cols // 2
        if center_col > 0: # Avoid division by zero for 1-pixel wide images
            for i in range(cols):
                dist_factor = abs(i - center_col) / center_col
                map_y[:, i] += dist_factor * 10 * np.sin(np.pi * i / (cols + 1e-6)) # Add epsilon
        
        # Applies distortion.
        img_copy = cv2.remap(img.copy(), map_x, map_y, cv2.INTER_LINEAR)
    
    elif transform_type == "leaf_curl":
        # Simulates leaf curl with wave-like distortion.
        rows, cols = img.shape[:2]
        map_y, map_x = np.mgrid[0:rows, 0:cols].astype(np.float32)
        
        # Applies wave distortion.
        freq = random.uniform(0.02, 0.05)  # Wave frequency.
        amplitude = random.uniform(5, 10)  # Wave amplitude.
        
        for i in range(cols):
            map_y[:, i] += amplitude * np.sin(freq * i)
            
        # Applies distortion.
        img_copy = cv2.remap(img.copy(), map_x, map_y, cv2.INTER_LINEAR)
    
    elif transform_type == "focus_blur":
        # Applies selective focus/blur.
        mask_focus = np.zeros((h, w), dtype=np.float32) 
        center_y, center_x = h // 2, w // 2
        
        # Creates a radial gradient mask.
        y_grid, x_grid = np.ogrid[:h, :w]
        # Ensure h and w are not zero for division
        dist_y_norm = ((y_grid - center_y)/(h if h > 0 else 1)) 
        dist_x_norm = ((x_grid - center_x)/(w if w > 0 else 1))
        mask_focus = 1.0 - np.clip(np.sqrt(dist_y_norm**2 + dist_x_norm**2) * 2.5, 0, 1)
        
        # Blurs the image.
        blurred = cv2.GaussianBlur(img_copy, (15, 15), 0)
        
        # Blends original with blurred version based on mask.
        mask_3c = np.stack([mask_focus] * 3, axis=2)
        img_copy_float = img_copy.astype(np.float32) * mask_3c + blurred.astype(np.float32) * (1 - mask_3c) 
        img_copy = np.clip(img_copy_float, 0, 255).astype(np.uint8)
    
    elif transform_type == "growth_stage":
        # Simulates different plant growth stages.
        stage = random.choice(["young", "mature"])
        
        if stage == "young":
            # Simulates young plant: brighter green, less textured.
            hsv = cv2.cvtColor(img_copy, cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # Shifts hue slightly towards green.
            hsv[:,:,0] = np.clip(hsv[:,:,0] * 0.9 + 30, 0, 179)
            
            # Increases saturation for fresh look.
            hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.2, 0, 255)
            
            # Converts back to RGB.
            img_copy = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            
            # Reduces texture with slight blur.
            img_copy = cv2.GaussianBlur(img_copy, (3, 3), 0)
            
        else:  # mature
            # Simulates mature plant: more texture, deeper colours.
            # Enhances texture with sharpening.
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], dtype=np.float32) * 0.5
            img_copy_float = cv2.filter2D(img_copy.astype(np.float32), -1, kernel) 
            img_copy = np.clip(img_copy_float, 0, 255).astype(np.uint8)
            
            # Creates deeper, slightly less saturated colours.
            hsv = cv2.cvtColor(img_copy, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:,:,1] = np.clip(hsv[:,:,1] * 0.9, 0, 255)  # Reduces saturation slightly.
            img_copy = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    elif transform_type == "disease_spots":
        # Adds simulated disease spots to leaves.
        
        # Converts to HSV to identify green regions (approximating leaves).
        hsv = cv2.cvtColor(img_copy, cv2.COLOR_RGB2HSV)
        
        # Creates mask for green areas.
        lower_green = np.array([30, 30, 30])
        upper_green = np.array([90, 255, 255])
        leaf_mask_single_channel = cv2.inRange(hsv, lower_green, upper_green).astype(np.float32) / 255.0 
        
        # Sets number of disease spots.
        num_spots = random.randint(5, 20)
        img_copy_float = img_copy.astype(np.float32) # Work with float for blending

        for _ in range(num_spots):
            # Attempts to place a spot on a leaf.
            for attempt in range(10):  # Limits attempts.
                # Sets random position.
                x_pos = random.randint(10, w-11) if w > 20 else random.randint(0, max(0, w-1))
                y_pos = random.randint(10, h-11) if h > 20 else random.randint(0, max(0, h-1))
                
                # Checks if point is on a leaf.
                if h > 0 and w > 0 and leaf_mask_single_channel[y_pos, x_pos] > 0.5:
                    # Sets random spot properties.
                    radius = random.randint(2, 8)
                    spot_colour_rgb = random.choice([ 
                        (80, 80, 30),    # Yellow/brown (chlorosis).
                        (50, 50, 50),    # Grey (mildew).
                        (30, 30, 80),    # Dark brown (fungal).
                        (20, 20, 20),    # Black (rot).
                    ])
                    spot_colour_rgb_float = np.array(spot_colour_rgb, dtype=np.float32) 
                    
                    # Creates spot with Gaussian falloff.
                    for dy_offset in range(-radius*2, radius*2 + 1): 
                        for dx_offset in range(-radius*2, radius*2 + 1): 
                            current_y, current_x = y_pos+dy_offset, x_pos+dx_offset 
                            if 0 <= current_y < h and 0 <= current_x < w:
                                # Calculates distance from centre.
                                dist = np.sqrt(dx_offset**2 + dy_offset**2)
                                
                                # Applies Gaussian falloff.
                                if dist <= radius*2:
                                    alpha_blend = np.exp(-(dist**2) / (2 * (radius/1.5)**2 + 1e-6)) 
                                    
                                    # Applies spot colour with alpha blending.
                                    for c_idx in range(3): 
                                        img_copy_float[current_y, current_x, c_idx] = (
                                            img_copy_float[current_y, current_x, c_idx] * (1 - alpha_blend) + 
                                            spot_colour_rgb_float[c_idx] * alpha_blend
                                        )
                    break  # Spot successfully placed.
        img_copy = np.clip(img_copy_float, 0, 255).astype(np.uint8)
    
    return img_copy.astype(np.uint8)

def apply_augmentation(img, intensity='medium'):
    """Applies a series of augmentations to an image with varying intensity."""
    # Defines probability scales based on intensity.
    if intensity == 'low':
        prob_scale = 0.5
    elif intensity == 'high':
        prob_scale = 1.5
    else:  # medium (default)
        prob_scale = 1.0
    
    # Determines if GPU should be used for this image.
    # Ensure img is HWC uint8 numpy array before this point
    if not (isinstance(img, np.ndarray) and img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3):
        # This case should ideally be handled before calling apply_augmentation
        # For safety, convert or raise error
        print(f"Warning: apply_augmentation received unexpected image format: {type(img)}, {getattr(img, 'dtype', 'N/A')}")
        if isinstance(img, torch.Tensor): # Example conversion
            img = (img.cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8) # Assuming CHW float 0-1
        # Add more conversions if needed or raise error
        if not (isinstance(img, np.ndarray) and img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3):
             print("Fallback: Could not convert image to HWC uint8 numpy. Skipping GPU path.")
             should_use_gpu = False
        else:
             should_use_gpu = USE_GPU_SUCCESSFULLY and random.random() < 0.8
    else:
        should_use_gpu = USE_GPU_SUCCESSFULLY and random.random() < 0.8
    
    img_augmented = img.copy() # Work on a copy

    if should_use_gpu and PYTORCH_DEVICE.type == 'cuda':
        # GPU-accelerated augmentation path.
        try:
            # Converts HWC uint8 numpy to CHW float [0,1] tensor on GPU.
            img_chw_f01_gpu = torch.from_numpy(img_augmented.transpose((2,0,1)).copy()).float().to(PYTORCH_DEVICE) / 255.0
            
            transformed_tensor = img_chw_f01_gpu
            
            # Applies random horizontal flip.
            if random.random() < 0.5 * prob_scale:
                transformed_tensor = TF.hflip(transformed_tensor)
            
            # Applies random brightness adjustment (additive).
            if random.random() < 0.5 * prob_scale:
                delta = random.uniform(-0.2, 0.2)
                transformed_tensor = torch.clamp(transformed_tensor + delta, 0, 1)
            
            # Applies random contrast adjustment.
            if random.random() < 0.3 * prob_scale:
                contrast_factor = random.uniform(0.8, 1.2)
                transformed_tensor = TF.adjust_contrast(transformed_tensor, contrast_factor)
            
            # Applies random 90-degree rotation.
            if random.random() < 0.3 * prob_scale:
                k_rot = random.randint(1, 3) 
                transformed_tensor = TF.rotate(transformed_tensor, angle=float(k_rot * 90))
            
            # Converts CHW float [0,1] tensor back to HWC uint8 numpy.
            img_augmented = (torch.clamp(transformed_tensor, 0, 1) * 255.0).byte().cpu().numpy().transpose((1,2,0))
            
        except Exception as e:
            # Falls back to CPU if GPU augmentation fails.
            print(f"GPU augmentation failed: {e}. Falling back to CPU for this image.")
            # img_augmented remains the CPU version (original copy)
            pass 
    
    # Applies CPU-based transformations on img_augmented.
    # Applies random geometric transformations.
    if random.random() < 0.8 * prob_scale:
        transform_type = random.choice(['rotate', 'flip', 'crop', 'perspective'])
        img_augmented = apply_geometric_transform(img_augmented, transform_type)
    
    # Applies random colour transformations.
    if random.random() < 0.7 * prob_scale:
        transform_type = random.choice(['brightness', 'contrast', 'saturation', 'hue', 'noise'])
        img_augmented = apply_color_transform(img_augmented, transform_type)
    
    # Applies seasonal effects.
    if random.random() < 0.3 * prob_scale:
        season_type = random.choice(['summer', 'autumn', 'winter', 'spring'])
        img_augmented = apply_seasonal_effect(img_augmented, season_type)
    
    # Applies one advanced effect.
    if random.random() < 0.5 * prob_scale:
        effect_type = random.choice([
            "lighting", "occlusion", "scale", "age", "weather", 
            "plant_specific", "background" 
        ])
        
        if effect_type == "lighting":
            img_augmented = apply_lighting_condition(img_augmented, random.choice([
                'shadow', 'overexposed', 'underexposed', 'sunflare', 'indoor' # Restored options
            ]))
        elif effect_type == "occlusion":
            img_augmented = apply_occlusion(img_augmented, "partial_view")
        elif effect_type == "scale":
            img_augmented = apply_scale_variation(img_augmented, random.choice(['macro', 'distant']))
        elif effect_type == "age":
            img_augmented = apply_plant_age_variation(img_augmented, random.choice(['young', 'mature', 'flowering']))
        elif effect_type == "weather":
            img_augmented = apply_weather_condition(img_augmented, random.choice(['rain', 'fog', 'bright_sun']))
        elif effect_type == "plant_specific": 
            img_augmented = apply_plant_specific_transform(img_augmented, random.choice(['leaf_wilt', 'leaf_curl', 'disease_spots', 'focus_blur', 'growth_stage'])) # Added more options
        elif effect_type == "background":
            img_augmented = apply_background_variation(img_augmented) 
    
    return img_augmented.astype(np.uint8) # Ensure uint8 output

def main():
    """Main function to orchestrate the offline data augmentation process."""
    try:
        print(f"{TermColors.HEADER}\n{'='*50}")
        print(f"OFFLINE DATA AUGMENTATION (PyTorch Version)")
        print(f"{'='*50}{TermColors.ENDC}")
        
        # Loads checkpoint to resume from a previous run.
        checkpoint = load_checkpoint()
        
        # Analyses dataset and determines augmentation factors.
        print(f"\n{TermColors.CYAN}Analysing dataset...{TermColors.ENDC}")
        aug_factors = analyze_class_sizes()
        
        if not aug_factors:
            print(f"{TermColors.RED}No valid classes found. Check your dataset directory.{TermColors.ENDC}")
            return
            
        # Gets list of classes to process.
        class_dirs = [os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
        classes_to_process = [os.path.basename(d) for d in class_dirs if os.path.basename(d) not in checkpoint["processed_classes"]]
        
        print(f"\n{TermColors.GREEN}✓ Found {len(class_dirs)} total classes.")
        print(f"✓ {len(checkpoint['processed_classes'])} classes already processed.")
        print(f"✓ {len(classes_to_process)} classes to process.{TermColors.ENDC}")
        
        # Prints GPU information.
        if USE_GPU_SUCCESSFULLY:
            print(f"\n{TermColors.GREEN}✓ GPU acceleration enabled and working ({PYTORCH_DEVICE}){TermColors.ENDC}")
        elif USE_GPU: 
            print(f"\n{TermColors.YELLOW}GPU acceleration enabled but may not be optimal - will use selectively ({PYTORCH_DEVICE}){TermColors.ENDC}")
        else: 
            print(f"\n{TermColors.YELLOW}Running in CPU-only mode ({PYTORCH_DEVICE}){TermColors.ENDC}")
        
        # Processes classes in parallel.
        print(f"\n{TermColors.CYAN}Starting augmentation process...{TermColors.ENDC}")
        print(f"Processing {PARALLEL_CLASSES} classes in parallel")
        
        # Uses ThreadPoolExecutor for parallel processing of classes.
        with ThreadPoolExecutor(max_workers=PARALLEL_CLASSES) as executor:
            futures = []
            for class_name_iter in classes_to_process: 
                class_dir_iter = os.path.join(DATA_DIR, class_name_iter) 
                # Submits augmentation task for each class.
                future = executor.submit(
                    augment_class_images, 
                    class_dir_iter, 
                    class_name_iter, 
                    aug_factors # Pass the whole dict, augment_class_images will use it
                )
                futures.append(future)
                
            # Monitors progress of submitted tasks.
            completed_tasks = 0 
            # tqdm can wrap the as_completed iterator for a global progress bar
            for future_item in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Overall Progress", leave=True): 
                completed_tasks += 1
                try:
                    class_name_result, augmented_count_result = future_item.result() 
                    # Updates checkpoint.
                    checkpoint["processed_classes"].append(class_name_result)
                    save_checkpoint(checkpoint)
                    
                except Exception as e:
                    # Retrieve class name from future if possible, though not directly stored
                    print(f"{TermColors.RED}Error processing a class: {e}{TermColors.ENDC}")
                    import traceback
                    traceback.print_exc() # Print full traceback for debugging
                
        print(f"\n{TermColors.GREEN}✓ Augmentation complete! All {len(class_dirs)} classes processed or skipped.{TermColors.ENDC}")
        
    except Exception as e:
        import traceback
        print(f"{TermColors.RED}Error in main function: {e}{TermColors.ENDC}")
        traceback.print_exc()

def augment_class_images(class_dir, class_name, aug_factors_all_classes=None): 
    """Augments images in a class directory to reach MIN_IMAGES_PER_CLASS."""
    # Calculates class-specific augmentation parameters.
    # It recalculates based on current counts.
    image_files, target_count, aug_per_image_ratio = calculate_class_specific_augmentation( 
        class_dir, class_name
    )
    
    if not image_files: # No original images found to process
        # Check if there are ANY images (original or augmented) in the folder
        all_files_in_dir = os.listdir(class_dir)
        any_images_present = any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in all_files_in_dir)

        if not any_images_present:
            # No images of any kind in the directory, attempt to delete it
            try:
                os.rmdir(class_dir)
                print(f"{TermColors.YELLOW}Directory {class_name} contained no images and has been deleted.{TermColors.ENDC}")
            except OSError as e:
                # This typically means the directory is not empty (e.g., contains non-image files or subdirectories)
                print(f"{TermColors.RED}Could not delete directory {class_name}. It contained no images, but was not empty (e.g., non-image files or subdirectories may be present): {e}{TermColors.ENDC}")
        else:
            # No original images to process, but other image files (e.g., existing augmentations) are present.
            print(f"{TermColors.YELLOW}No original images found in {class_name} to process, but other image files exist. Skipping augmentation for this class.{TermColors.ENDC}")
        return class_name, 0 # Return 0 as no new augmentations were made
    
    # Gets existing augmented images.
    existing_aug = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')) and '_aug' in f]
    orig_count = len(image_files)
    total_existing = orig_count + len(existing_aug)
    
    # Skips if minimum target is already met.
    if total_existing >= MIN_IMAGES_PER_CLASS:
        return class_name, 0 # Return 0 augmented images created in this run
        
    # Calculates exact number of new augmented images needed.
    exact_new_needed = target_count - total_existing 
    if exact_new_needed <= 0: 

        return class_name, 0

    # The tqdm total is now exact_new_needed, and unit is "aug"
    with tqdm(total=exact_new_needed, desc=f"Processing {class_name}", leave=True, unit="aug") as pbar:
        # Counter for augmented images created in this run.
        augmented_count_this_run = 0 
        
        # Processes each original image.
        for i, img_file in enumerate(image_files):
            try:
                # Stops if target number of new augmentations is reached.
                # This check is important if earlier images already fulfilled the quota.
                if augmented_count_this_run >= exact_new_needed:
                    break
                    
                img_path = os.path.join(class_dir, img_file)
                
                # Loads the original image.
                original_img_bgr = cv2.imread(img_path) 
                if original_img_bgr is None:
                    # No pbar.update here as it tracks augmentations now
                    continue
                    
                # Converts from BGR to RGB.
                original_img_rgb = cv2.cvtColor(original_img_bgr, cv2.COLOR_BGR2RGB) 
                
                # Resizes to target size.
                original_img_resized = cv2.resize(original_img_rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA) 
                
                # Calculates augmentations for this image for even distribution.
                images_left_to_process = len(image_files) - i 
                if images_left_to_process <= 0: images_left_to_process = 1 
                augs_needed_for_this_image = math.ceil((exact_new_needed - augmented_count_this_run) / images_left_to_process) 
                
                # Creates augmented versions.
                for aug_idx in range(augs_needed_for_this_image):
                    # Stops if target is reached.
                    if augmented_count_this_run >= exact_new_needed:
                        break
                        
                    # Applies augmentation with random intensity.
                    intensity_choice = random.choice(['low', 'medium', 'medium', 'high']) 
                    augmented_img_rgb = apply_augmentation(original_img_resized.copy(), intensity_choice) 
                    
                    # Converts back to BGR for saving.
                    augmented_img_bgr = cv2.cvtColor(augmented_img_rgb, cv2.COLOR_RGB2BGR) 
                    
                    # Generates output filename.
                    base_name = os.path.splitext(img_file)[0]
                    
                    # Find the next available augmentation index for this specific original image
                    current_aug_files_for_base = [f for f in os.listdir(class_dir) if f.startswith(base_name + "_aug")]
                    next_idx = 0
                    if current_aug_files_for_base:
                        indices = []
                        for f_aug in current_aug_files_for_base:
                            try:
                                idx_str = f_aug[len(base_name + "_aug") : f_aug.rfind('.')]
                                indices.append(int(idx_str))
                            except ValueError:
                                continue # Skip if parsing fails
                        if indices:
                            next_idx = max(indices) + 1
                    
                    out_file = f"{base_name}_aug{next_idx}.jpg"
                    out_path = os.path.join(class_dir, out_file)
                    
                    # Saves the augmented image.
                    cv2.imwrite(out_path, augmented_img_bgr)
                    augmented_count_this_run += 1
                    pbar.update(1) # Update progress for each augmentation created
                    
                # Removed pbar.update(1) from here (was updating per original image)
                # Removed pbar.set_postfix as the bar itself now shows aug count vs target
                    
            except Exception as e:
                # Do not update pbar here if an original image fails, as pbar tracks augmentations.
                print(f"{TermColors.RED}Error processing image {img_file} in {class_name}: {e}{TermColors.ENDC}")
    
    final_count = orig_count + len(existing_aug) + augmented_count_this_run 
    
    return class_name, augmented_count_this_run

if __name__ == "__main__":
    main()