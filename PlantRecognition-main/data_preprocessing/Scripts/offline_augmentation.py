import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import cv2
from tqdm import tqdm
import json
import time
import random
import gc
import sys
from scipy import ndimage
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import threading


# Define TermColors for console output
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

# Configuration
# Get the current script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (parent of Scripts)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# Set data directory - use environment variable if set, otherwise use default path
DATA_DIR = os.environ.get('PLANT_DATA_DIR', os.path.join(PROJECT_ROOT, "data", "plant_images"))
# Check if directory exists
if not os.path.exists(DATA_DIR):
    print(f"{TermColors.YELLOW}Warning: Data directory {DATA_DIR} does not exist.{TermColors.ENDC}")
    print(f"{TermColors.YELLOW}You can set the PLANT_DATA_DIR environment variable to specify a different path.{TermColors.ENDC}")

IMAGE_SIZE = (224, 224)
AUGMENTATION_FACTOR = 8  # Increased from 5 to 8 for more variations
MAX_IMAGES_PER_CLASS = 300  # Increased from 200 to 300 for more training data
CHECKPOINT_FILE = "augmentation_checkpoint.json"
# CPU and GPU optimization settings
MAX_CPU_WORKERS = os.cpu_count()  # Use all cores
PARALLEL_CLASSES = max(4, os.cpu_count() // 2)  # Process multiple classes in parallel
BATCH_SIZE = 32  # Increased batch size for better throughput
USE_GPU = True  # Use GPU when available
GPU_BATCH_SIZE = 64  # Larger batch size for GPU processing
GPU_MEMORY_LIMIT = None  # Set to None to use all available GPU memory

# Configure PyTorch to use GPU when available
if USE_GPU:
    try:
        if torch.cuda.is_available():
            print(f"{TermColors.GREEN}✅ Found {torch.cuda.device_count()} GPU(s). Enabling GPU acceleration.{TermColors.ENDC}")
            # Set default device
            DEVICE = torch.device("cuda:0")
            
            # Print GPU info
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"{TermColors.CYAN}ℹ GPU {i}: {gpu_name}{TermColors.ENDC}")
                
            # Set memory strategy
            if GPU_MEMORY_LIMIT:
                torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_LIMIT / 100.0)
                print(f"{TermColors.CYAN}ℹ GPU memory limit set to {GPU_MEMORY_LIMIT}%{TermColors.ENDC}")
        else:
            print(f"{TermColors.YELLOW}⚠️ No GPU found. Using CPU only.{TermColors.ENDC}")
            DEVICE = torch.device("cpu")
            USE_GPU = False
    except Exception as e:
        print(f"{TermColors.RED}❌ Error configuring GPU: {e}. Using CPU only.{TermColors.ENDC}")
        DEVICE = torch.device("cpu")
        USE_GPU = False
else:
    DEVICE = torch.device("cpu")

# Set up normalization transforms - similar to EfficientNetV2 preprocessing
# Mean and std values for ImageNet (used by EfficientNet)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
NORMALIZE = transforms.Normalize(mean=MEAN, std=STD)

def load_checkpoint():
    """Load checkpoint info to resume from previous run"""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                return json.load(f)
        except:
            return {"processed_classes": []}
    return {"processed_classes": []}

def save_checkpoint(checkpoint):
    """Save checkpoint info to resume later"""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)

def apply_seasonal_effect(img, season_type):
    """Apply seasonality effects to plant images
    
    Args:
        img: Input image as numpy array
        season_type: Type of seasonal effect to apply
        
    Returns:
        Augmented image with seasonal effect
    """
    # Convert to numpy if it's a torch tensor
    if isinstance(img, torch.Tensor):
        is_tensor = True
        # If on GPU, move to CPU
        if img.device.type != "cpu":
            img = img.cpu()
        # Convert to numpy
        img_np = img.permute(1, 2, 0).numpy()  # Convert from CxHxW to HxWxC
        img_float = img_np.astype(np.float32)
    else:
        is_tensor = False
        img_float = img.astype(np.float32)
    
    if season_type == "summer":
        # Brighter, more saturated, slightly yellow tint
        img_float = img_float * 1.1  # Brighter
        img_float[:,:,2] = img_float[:,:,2] * 0.95  # Reduce blue channel
        img_float[:,:,1] = img_float[:,:,1] * 1.05  # Increase green channel
        
    elif season_type == "autumn":
        # Orange/brown tint, lower saturation
        img_float[:,:,0] = img_float[:,:,0] * 1.15  # Increase red channel
        img_float[:,:,1] = img_float[:,:,1] * 0.85  # Reduce green channel
        img_float[:,:,2] = img_float[:,:,2] * 0.7   # Reduce blue channel
        
    elif season_type == "winter":
        # Cooler, bluer, less saturated
        img_float = img_float * 0.85  # Darker
        img_float[:,:,0] = img_float[:,:,0] * 0.8  # Reduce red channel
        img_float[:,:,2] = img_float[:,:,2] * 1.2  # Increase blue channel
        
    elif season_type == "spring":
        # More green, slightly brighter
        img_float[:,:,1] = img_float[:,:,1] * 1.15  # Increase green channel
        img_float = img_float * 1.05  # Slightly brighter
        
    elif season_type == "drought":
        # Yellow/brown tones for drought stress
        img_float[:,:,0] = img_float[:,:,0] * 1.2   # Increase red
        img_float[:,:,1] = img_float[:,:,1] * 0.9   # Decrease green
        img_float[:,:,2] = img_float[:,:,2] * 0.7   # Decrease blue
        
    elif season_type == "overwatered":
        # More blue/green for overwatered plants
        img_float[:,:,0] = img_float[:,:,0] * 0.8   # Decrease red
        img_float[:,:,1] = img_float[:,:,1] * 0.95  # Slightly decrease green
        img_float[:,:,2] = img_float[:,:,2] * 1.2   # Increase blue
        
    # Clip to valid range
    img_float = np.clip(img_float, 0, 255)
    
    # Return tensor if input was tensor
    if is_tensor:
        # Convert back to tensor with CxHxW format
        img_tensor = torch.from_numpy(img_float.astype(np.float32)).permute(2, 0, 1)
        return img_tensor
    else:
        return img_float.astype(np.uint8)

def apply_lighting_condition(img, light_type):
    """Apply different lighting conditions typical in plant photography
    
    Args:
        img: Input image as numpy array or torch tensor
        light_type: Type of lighting condition
        
    Returns:
        Augmented image with modified lighting
    """
    # Convert to numpy if it's a torch tensor
    if isinstance(img, torch.Tensor):
        is_tensor = True
        # If on GPU, move to CPU
        if img.device.type != "cpu":
            img = img.cpu()
        # Convert to numpy
        img_np = img.permute(1, 2, 0).numpy()  # Convert from CxHxW to HxWxC
        img_float = img_np.astype(np.float32)
    else:
        is_tensor = False
        img_float = img.astype(np.float32)
    
    h, w = img_float.shape[:2]
    
    if light_type == "shadow":
        # Create shadow effect on part of the image
        shadow_mask = np.ones_like(img_float)
        
        # Create a gradient shadow in a random direction
        direction = random.choice(["top", "left", "right", "bottom"])
        
        if direction == "top":
            for i in range(h):
                shadow_mask[i,:,:] = 1.0 - (0.5 * i / h)
        elif direction == "left":
            for i in range(w):
                shadow_mask[:,i,:] = 1.0 - (0.5 * i / w)
        elif direction == "right":
            for i in range(w):
                shadow_mask[:,i,:] = 1.0 - (0.5 * (w-i) / w)
        else:  # bottom
            for i in range(h):
                shadow_mask[i,:,:] = 1.0 - (0.5 * (h-i) / h)
        
        # Apply shadow
        img_float = img_float * shadow_mask
        
    elif light_type == "sunflare":
        # Add a sunflare effect (bright spot with halo)
        # Random position for the flare
        flare_x = random.randint(0, w-1)
        flare_y = random.randint(0, h-1)
        
        # Create distance matrix from flare point
        y, x = np.ogrid[-flare_y:h-flare_y, -flare_x:w-flare_x]
        dist = np.sqrt(x*x + y*y)
        
        # Create flare mask (bright in center, fading out)
        flare_mask = np.zeros_like(img_float)
        max_dist = min(h, w) / 3
        flare_intensity = (1 - np.clip(dist / max_dist, 0, 1)) * 100
        
        # Apply to all channels with yellow-white tint
        flare_mask[:,:,0] = flare_intensity * 1.0  # Red
        flare_mask[:,:,1] = flare_intensity * 1.0  # Green
        flare_mask[:,:,2] = flare_intensity * 0.8  # Blue (slightly less for yellow tint)
        
        # Add flare to image
        img_float = img_float + flare_mask
        
    elif light_type == "overexposed":
        # Overexposed parts of the image like in bright sunlight
        img_float = img_float * 1.4
        
    elif light_type == "underexposed":
        # Underexposed like in shade
        img_float = img_float * 0.7
        
    elif light_type == "indoor":
        # Indoor lighting (yellowish tint)
        img_float[:,:,0] = img_float[:,:,0] * 1.1  # More red
        img_float[:,:,1] = img_float[:,:,1] * 1.05  # More green
        img_float[:,:,2] = img_float[:,:,2] * 0.85  # Less blue
    
    # Clip to valid range
    img_float = np.clip(img_float, 0, 255)
    
    # Return tensor if input was tensor
    if is_tensor:
        # Convert back to tensor with CxHxW format
        img_tensor = torch.from_numpy(img_float.astype(np.float32)).permute(2, 0, 1)
        return img_tensor
    else:
        return img_float.astype(np.uint8)

def apply_plant_specific_transform(img, transform_type):
    """Apply plant-specific transformations
    
    Args:
        img: Input image as numpy array or torch tensor
        transform_type: Type of plant-specific transformation
        
    Returns:
        Augmented image with plant-specific changes
    """
    # Convert to numpy if it's a torch tensor
    if isinstance(img, torch.Tensor):
        is_tensor = True
        # If on GPU, move to CPU
        if img.device.type != "cpu":
            img = img.cpu()
        # Convert to numpy
        img_np = img.permute(1, 2, 0).numpy()  # Convert from CxHxW to HxWxC
        img_copy = img_np.copy()
    else:
        is_tensor = False
        img_copy = img.copy()
    
    h, w = img_copy.shape[:2]
    
    if transform_type == "leaf_wilt":
        # Simulate wilting leaves by applying a slight downward distortion
        rows, cols = img_copy.shape[:2]
        
        # Create distortion map
        map_y, map_x = np.mgrid[0:rows, 0:cols].astype(np.float32)
        
        # Apply downward curve stronger at the edges
        center_col = cols // 2
        for i in range(cols):
            # Calculate distance from center as a factor for distortion
            dist_factor = abs(i - center_col) / center_col
            # Apply stronger distortion at the edges
            map_y[:, i] += dist_factor * 10 * np.sin(np.pi * i / cols)
        
        # Apply distortion
        img_copy = cv2.remap(img_copy, map_x, map_y, cv2.INTER_LINEAR)
        
    elif transform_type == "leaf_curl":
        # Simulate leaf curl - mild warping effect
        rows, cols = img_copy.shape[:2]
        map_y, map_x = np.mgrid[0:rows, 0:cols].astype(np.float32)
        
        # Create a wave pattern
        for i in range(rows):
            map_x[i,:] += 7 * np.sin(i/10)
        
        img_copy = cv2.remap(img_copy, map_x, map_y, cv2.INTER_LINEAR)
        
    elif transform_type == "focus_blur":
        # Simulate depth-of-field effects common in plant photography
        # Create a radial gradient to simulate focus
        center_x, center_y = random.randint(w//4, 3*w//4), random.randint(h//4, 3*h//4)
        y, x = np.ogrid[-center_y:h-center_y, -center_x:w-center_x]
        mask = x*x + y*y <= min(h,w)**2
        
        # Create blurred version of the image
        blurred_img = cv2.GaussianBlur(img_copy, (15, 15), 0)
        
        # Create distance-based mask for smooth transition
        dist = np.sqrt(x*x + y*y)
        max_dist = min(h, w) / 2
        focus_mask = np.clip(dist / max_dist, 0, 1)
        focus_mask = np.stack([focus_mask] * 3, axis=2)
        
        # Blend original and blurred image
        if is_tensor:
            img_copy = img_np * (1 - focus_mask) + blurred_img * focus_mask
        else:
            img_copy = img * (1 - focus_mask) + blurred_img * focus_mask
        img_copy = img_copy.astype(np.uint8)
        
    elif transform_type == "growth_stage":
        # Simulate different growth stages by adjusting green intensity
        growth_factor = random.uniform(0.8, 1.2)  # 0.8=young, 1.2=mature
        img_copy[:,:,1] = np.clip(img_copy[:,:,1] * growth_factor, 0, 255)
        
    elif transform_type == "disease_spots":
        # Simulate disease spots on leaves
        num_spots = random.randint(3, 15)
        for _ in range(num_spots):
            # Random spot position
            spot_x = random.randint(0, w-1)
            spot_y = random.randint(0, h-1)
            
            # Random spot size
            spot_size = random.randint(3, 10)
            
            # Random spot color (brownish)
            spot_color = (
                random.randint(60, 120),  # B
                random.randint(40, 90),   # G
                random.randint(40, 100)   # R
            )
            
            # Draw the spot
            cv2.circle(img_copy, (spot_x, spot_y), spot_size, spot_color, -1)
    
    # Return tensor if input was tensor
    if is_tensor:
        # Convert back to tensor with CxHxW format
        img_tensor = torch.from_numpy(img_copy.astype(np.float32)).permute(2, 0, 1)
        return img_tensor
    else:
        return img_copy

def create_augmentations(img):
    """Create multiple augmented versions of a plant image with specialized transformations
    
    PyTorch implementation that works with both tensors and numpy arrays
    """
    # Convert input to proper format
    if isinstance(img, torch.Tensor):
        # Already a tensor, make sure it's on CPU for processing
        if img.device.type != "cpu":
            img = img.cpu()
        
        # Make sure it's in right format (C,H,W)
        if img.dim() == 4:  # Has batch dimension
            img = img[0]  # Remove batch dimension
        
        # Convert to numpy for processing
        img_np = img.permute(1, 2, 0).numpy()  # Convert from CxHxW to HxWxC
        img_np = (img_np * 255).astype(np.uint8)  # Scale to 0-255 range
    elif isinstance(img, np.ndarray):
        # Already numpy array
        img_np = img.copy()
    else:
        # Try to convert from PIL or other format
        try:
            img_np = np.array(img)
        except:
            print(f"Error: Unsupported image type {type(img)}")
            # Create empty placeholder
            img_np = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8)
    
    # Ensure we have a valid image
    if len(img_np.shape) != 3 or img_np.shape[2] != 3:
        # Convert grayscale to RGB if needed
        if len(img_np.shape) == 3 and img_np.shape[2] == 1:
            img_np = np.repeat(img_np, 3, axis=2)
        elif len(img_np.shape) == 2:
            img_np = np.stack([img_np] * 3, axis=2)
        else:
            # Invalid image format, return just the original
            img_tensor = torch.tensor(img_np).float() / 255.0
            if len(img_tensor.shape) == 3:
                img_tensor = img_tensor.permute(2, 0, 1)  # HWC to CHW
            return [img_tensor]
    
    augmented_images = []
    
    # Define PyTorch transforms for normalization
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=MEAN, std=STD)
    
    # Original image
    img_tensor = to_tensor(img_np)
    img_norm = normalize(img_tensor)
    augmented_images.append(img_norm)
    
    # 1. PLANT-SPECIFIC TRANSFORMATIONS
    
    # Seasonal variations
    seasons = ["summer", "autumn", "winter", "spring", "drought", "overwatered"]
    for season in random.sample(seasons, 2):
        try:
            img_season = apply_seasonal_effect(img_np.copy(), season)
            img_tensor = to_tensor(img_season)
            img_norm = normalize(img_tensor)
            augmented_images.append(img_norm)
        except Exception as e:
            print(f"Seasonal effect {season} failed: {e}")
    
    # Plant-specific transformations
    plant_transforms = ["leaf_wilt", "leaf_curl", "focus_blur", "growth_stage", "disease_spots"]
    for transform in random.sample(plant_transforms, 2):
        try:
            img_transformed = apply_plant_specific_transform(img_np.copy(), transform)
            img_tensor = to_tensor(img_transformed)
            img_norm = normalize(img_tensor)
            augmented_images.append(img_norm)
        except Exception as e:
            print(f"Plant transform {transform} failed: {e}")
    
    # Lighting conditions
    lighting_conditions = ["shadow", "sunflare", "overexposed", "underexposed", "indoor"]
    for light in random.sample(lighting_conditions, 2):
        try:
            img_light = apply_lighting_condition(img_np.copy(), light)
            img_tensor = to_tensor(img_light)
            img_norm = normalize(img_tensor)
            augmented_images.append(img_norm)
        except Exception as e:
            print(f"Lighting condition {light} failed: {e}")
    
    # 2. SIMPLE TRANSFORMATIONS
    
    # Horizontal flip - guaranteed to work
    try:
        img_flip = np.fliplr(img_np.copy()).copy()  # Add .copy() to ensure contiguous array
        img_tensor = to_tensor(img_flip)
        img_norm = normalize(img_tensor)
        augmented_images.append(img_norm)
    except Exception as e:
        print(f"Flip failed: {e}")
    
    # Rotation - using OpenCV for reliability
    rotation_angles = [5, 10, -5, -10, 15, -15]
    for angle in random.sample(rotation_angles, 2):
        try:
            # Use OpenCV rotation which is more reliable than PyTorch for custom angles
            h, w = img_np.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            img_rot = cv2.warpAffine(img_np.copy(), rotation_matrix, (w, h), 
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            img_tensor = to_tensor(img_rot)
            img_norm = normalize(img_tensor)
            augmented_images.append(img_norm)
        except Exception as e:
            print(f"Rotation {angle} failed: {e}")
    
    # Simple zoom - reliable OpenCV implementation
    zoom_factors = [0.9, 0.85, 1.1, 1.15]
    for factor in random.sample(zoom_factors, 2):
        try:
            h, w = img_np.shape[:2]
            if factor < 1.0:  # Zoom out - crop and resize back up
                # Calculate new dimensions
                new_h, new_w = int(h * factor), int(w * factor)
                # Ensure minimum dimensions
                new_h, new_w = max(new_h, 32), max(new_w, 32)
                # Calculate crop offsets
                y_start = (h - new_h) // 2
                x_start = (w - new_w) // 2
                # Crop center
                cropped = img_np[y_start:y_start+new_h, x_start:x_start+new_w].copy()
                # Resize back up
                img_zoom = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
            else:  # Zoom in - resize down and crop center
                # Calculate intermediate size (larger)
                resize_h, resize_w = int(h * factor), int(w * factor)
                # Resize image larger
                resized = cv2.resize(img_np.copy(), (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
                # Crop center to original size
                y_start = (resize_h - h) // 2
                x_start = (resize_w - w) // 2
                img_zoom = resized[y_start:y_start+h, x_start:x_start+w].copy()
            
            img_tensor = to_tensor(img_zoom)
            img_norm = normalize(img_tensor)
            augmented_images.append(img_norm)
        except Exception as e:
            print(f"Zoom {factor} failed: {e}")
    
    # Brightness variations
    brightness_factors = [0.8, 0.9, 1.1, 1.2]
    for factor in random.sample(brightness_factors, 2):
        try:
            img_bright = img_np.copy() * factor
            img_bright = np.clip(img_bright, 0, 255).astype(np.uint8)
            img_tensor = to_tensor(img_bright)
            img_norm = normalize(img_tensor)
            augmented_images.append(img_norm)
        except Exception as e:
            print(f"Brightness {factor} failed: {e}")
    
    # Simple center crop - more reliable than random crop
    try:
        h, w = img_np.shape[:2]
        crop_size = int(min(h, w) * 0.8)
        y_start = (h - crop_size) // 2
        x_start = (w - crop_size) // 2
        img_crop = img_np[y_start:y_start+crop_size, x_start:x_start+crop_size].copy()
        img_crop = cv2.resize(img_crop, (w, h))
        img_tensor = to_tensor(img_crop)
        img_norm = normalize(img_tensor)
        augmented_images.append(img_norm)
    except Exception as e:
        print(f"Center crop failed: {e}")
    
    # If we have too many augmentations, sample to the desired number
    if len(augmented_images) > AUGMENTATION_FACTOR:
        # Always keep original image (index 0)
        sampled = [augmented_images[0]] + random.sample(augmented_images[1:], AUGMENTATION_FACTOR-1)
        return sampled
    
    return augmented_images

def process_class_with_gpu(class_dir, class_name, thread_id, position, checkpoint):
    """Process a class with GPU acceleration using PyTorch"""
    try:
        # Get all image files
        image_files = [os.path.join(class_dir, f) for f in os.listdir(class_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not '_aug' in f]
        
        # Limit number of images if needed
        if len(image_files) * AUGMENTATION_FACTOR > MAX_IMAGES_PER_CLASS:
            needed_orig_images = MAX_IMAGES_PER_CLASS // AUGMENTATION_FACTOR
            image_files = random.sample(image_files, needed_orig_images)
        
        # Process in GPU-efficient batches
        total_images = len(image_files)
        
        # Register this progress bar
        with lock:
            active_progress_bars[position] = class_name
        
        # Create progress bar for this class
        with tqdm(total=total_images, desc=f"{class_name}", position=position, leave=True) as class_pbar:
            success_count = 0
            
            # Use a two-level processing approach:
            # 1. First thread pool for batching images to the GPU
            # 2. Second thread pool for saving results (CPU-intensive operation)
            
            # Create thread pools
            gpu_executor = ThreadPoolExecutor(max_workers=2)  # Limit GPU workers to avoid overwhelming GPU
            cpu_executor = ThreadPoolExecutor(max_workers=MAX_CPU_WORKERS)  # More workers for CPU tasks
            
            try:
                # Queue for storing augmentation results for CPU processing
                result_queue = []
                
                # Process images in small batches to maintain constant GPU usage
                batch_size = min(4, max(2, total_images // 32))  # Smaller batches for continuous GPU work
                batches = [image_files[i:i+batch_size] for i in range(0, total_images, batch_size)]
                
                # GPU augmentation function - just does the GPU work
                def augment_on_gpu(img_path):
                    try:
                        # Use PyTorch device
                        device = DEVICE if USE_GPU else torch.device("cpu")
                        
                        # Load image
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize(IMAGE_SIZE)
                        
                        # Create augmentations - this is the main GPU-intensive operation
                        augmented_images = create_augmentations(img)
                        
                        # Return data for CPU processing
                        return (img_path, augmented_images, True)
                    except Exception as e:
                        print(f"Error in GPU processing {img_path}: {e}")
                        return (img_path, None, False)
                
                # CPU saving function - runs separately from GPU work
                def save_augmented_images(result_data):
                    img_path, augmented_images, success = result_data
                    if not success:
                        return False
                    
                    try:
                        # Get directory and basename of original image
                        original_dir = os.path.dirname(img_path)
                        basename = os.path.splitext(os.path.basename(img_path))[0]
                        counter = image_files.index(img_path)
                        
                        # Save augmented images in the same directory as the original image
                        for i, augmented_img in enumerate(augmented_images):
                            # Skip the first augmentation (index 0) which is the original image
                            if i == 0:
                                continue
                                
                            # Convert PyTorch tensor to numpy for saving
                            # Move to CPU if on GPU
                            if augmented_img.device.type != "cpu":
                                augmented_img = augmented_img.cpu()
                                
                            # Convert to numpy and denormalize
                            img_np = augmented_img.permute(1, 2, 0).numpy()  # CHW -> HWC
                            
                            # Denormalize using the mean and std
                            img_np = img_np * np.array(STD) + np.array(MEAN)
                            
                            # Scale to 0-255 and convert to uint8
                            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                            
                            # Save image
                            output_path = os.path.join(original_dir, f"{basename}_aug{i}_{counter}.jpg")
                            cv2.imwrite(output_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
                        
                        # Update progress bar
                        nonlocal success_count
                        with lock:
                            success_count += 1
                            class_pbar.update(1)
                        
                        return True
                    except Exception as e:
                        print(f"Error saving augmented images for {img_path}: {e}")
                        return False
                
                # Process all batches
                for batch in batches:
                    # Submit batch for GPU processing
                    gpu_futures = []
                    for img_path in batch:
                        gpu_futures.append(gpu_executor.submit(augment_on_gpu, img_path))
                    
                    # As GPU processing completes, submit results for CPU processing
                    for gpu_future in concurrent.futures.as_completed(gpu_futures):
                        result = gpu_future.result()
                        if result[2]:  # Check success flag
                            # Queue for CPU processing
                            cpu_future = cpu_executor.submit(save_augmented_images, result)
                            result_queue.append(cpu_future)
                
                # Wait for all CPU saving to complete
                for cpu_future in concurrent.futures.as_completed(result_queue):
                    cpu_future.result()  # Just to ensure all saving completes
            
            finally:
                # Clean up executors
                gpu_executor.shutdown()
                cpu_executor.shutdown()
            
            # Clear GPU memory after class is complete
            if USE_GPU:
                try:
                    # Force PyTorch to release GPU memory
                    torch.cuda.empty_cache()
                    # Force more aggressive garbage collection
                    gc.collect()
                except:
                    pass
            
            # Update checkpoint with lock
            with lock:
                if class_name not in checkpoint["processed_classes"]:
                    checkpoint["processed_classes"].append(class_name)
                    save_checkpoint(checkpoint)
                    
                    # Update progress
                    global processed_count
                    processed_count += 1
                    pbar.update(1)
                    
                    # Add to completed classes
                    completed_classes.append(f"{class_name} ({success_count} images)")
                    
                    # Remove from active progress bars
                    if position in active_progress_bars:
                        del active_progress_bars[position]
            
            return success_count
            
    except Exception as e:
        print(f"Error processing {class_name}: {e}")
        import traceback
        traceback.print_exc()
        return 0

def process_classes_parallel(class_dirs, checkpoint):
    """Process multiple classes simultaneously to maximize CPU/GPU utilization
    
    Optimized for maximum GPU utilization with clean display
    """
    # Get unprocessed classes
    unprocessed_classes = []
    for d in class_dirs:
        class_name = os.path.basename(d)
        class_dir = os.path.join(DATA_DIR, class_name)
        if class_name not in checkpoint["processed_classes"]:
            unprocessed_classes.append((class_dir, class_name))
    
    if not unprocessed_classes:
        print(f"{TermColors.GREEN}All classes already processed!{TermColors.ENDC}")
        return
    
    # Calculate how many classes to process in parallel
    num_parallel_classes = min(PARALLEL_CLASSES, len(unprocessed_classes))
    
    # Create thread-safe structures
    global lock, completed_classes, processed_count, active_progress_bars, pbar
    lock = threading.Lock()
    completed_classes = []
    processed_count = 0
    active_progress_bars = {}
    
    # Create a shared event for signaling interruption
    stop_event = threading.Event()
    
    # Clear screen for a clean start
    print("\033[2J\033[H", end="")
    
    # Create space for completed classes and progress bars
    print("\n" * (num_parallel_classes + 3))
    
    # Create overall progress bar at the bottom
    pbar = tqdm(total=len(unprocessed_classes), desc="Overall Progress", position=num_parallel_classes+1)
    
    def process_class_queue(class_queue, thread_id):
        """Process a queue of classes with proper GPU utilization"""
        position = thread_id + 1
        
        for class_dir, class_name in class_queue:
            if stop_event.is_set():
                break
                
            try:
                # Skip if already processed
                with lock:
                    if class_name in checkpoint["processed_classes"]:
                        continue
                
                # Process with GPU - pass checkpoint parameter
                process_class_with_gpu(class_dir, class_name, thread_id, position, checkpoint)
                
                # Force garbage collection
                gc.collect()
                
            except Exception as e:
                print(f"Error in thread {thread_id} for {class_name}: {e}")
                import traceback
                traceback.print_exc()
    
    # Optimize GPU parameters for maximum utilization
    if USE_GPU:
        try:
            # Configure PyTorch for maximum performance
            if torch.cuda.is_available():
                # Set PyTorch to use optimized algorithms
                torch.backends.cudnn.benchmark = True
                
                # Print GPU memory information
                print(f"{TermColors.CYAN}ℹ GPU memory available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB{TermColors.ENDC}")
                
                # Optional: limit memory usage
                if GPU_MEMORY_LIMIT:
                    torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_LIMIT / 100.0)
                    
                # Set environment variables for better performance
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
                
        except Exception as e:
            print(f"Warning: GPU optimization failed: {e}")
    
    # Distribute classes among threads
    thread_class_assignments = [[] for _ in range(num_parallel_classes)]
    for i, class_item in enumerate(unprocessed_classes):
        thread_class_assignments[i % num_parallel_classes].append(class_item)
    
    # Start threads
    threads = []
    try:
        for thread_id, thread_classes in enumerate(thread_class_assignments):
            if thread_classes:
                thread = threading.Thread(
                    target=process_class_queue,
                    args=(thread_classes, thread_id)
                )
                thread.daemon = True
                thread.start()
                threads.append(thread)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
            
    except KeyboardInterrupt:
        stop_event.set()
        print(f"\nInterrupted - saving progress...")
        for thread in threads:
            thread.join(timeout=5)
    
    # Close progress bar
    pbar.close()
    
    # Final summary
    print(f"\n{processed_count}/{len(unprocessed_classes)} classes completed")

def run_augmentation():
    """Main function to run the augmentation process"""
    print(f"{TermColors.HEADER}\n{'='*50}")
    print(f"OFFLINE PLANT IMAGE AUGMENTATION WITH SEASONALITY (PyTorch)")
    print(f"{'='*50}{TermColors.ENDC}")
    
    # Check for CUDA/GPU support
    if torch.cuda.is_available():
        print(f"{TermColors.GREEN}✅ CUDA GPU is available - Using GPU acceleration{TermColors.ENDC}")
        print(f"{TermColors.GREEN}✅ GPU: {torch.cuda.get_device_name(0)}{TermColors.ENDC}")
    else:
        print(f"{TermColors.YELLOW}⚠️ No CUDA GPU available - Using CPU only{TermColors.ENDC}")
    
    print(f"{TermColors.CYAN}ℹ Using {MAX_CPU_WORKERS} CPU worker threads{TermColors.ENDC}")
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    print(f"{TermColors.CYAN}Loaded checkpoint with {len(checkpoint['processed_classes'])} processed classes{TermColors.ENDC}")
    
    # Get class directories
    class_dirs = [os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    print(f"{TermColors.CYAN}Found {len(class_dirs)} plant classes{TermColors.ENDC}")
    print(f"{TermColors.CYAN}Will create up to {AUGMENTATION_FACTOR} variations per image{TermColors.ENDC}")
    print(f"{TermColors.CYAN}Maximum {MAX_IMAGES_PER_CLASS} images per class after augmentation{TermColors.ENDC}")
    print(f"{TermColors.CYAN}Including specialized plant transformations (seasonality, disease, growth stages){TermColors.ENDC}")
    print(f"{TermColors.CYAN}Saving augmented images in the same directories as originals{TermColors.ENDC}")
    
    # Process all classes
    try:
        process_classes_parallel(class_dirs, checkpoint)
        print(f"{TermColors.GREEN}✅ Augmentation completed successfully!{TermColors.ENDC}")
        print(f"{TermColors.GREEN}✅ Augmented images saved alongside original images{TermColors.ENDC}")
        
    except KeyboardInterrupt:
        print(f"{TermColors.YELLOW}⚠️ Process interrupted. Progress saved to checkpoint.{TermColors.ENDC}")
    except Exception as e:
        print(f"{TermColors.RED}❌ Error during augmentation: {e}{TermColors.ENDC}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Initialize PyTorch settings for GPU optimization
    if USE_GPU and torch.cuda.is_available():
        # Enable cuDNN auto-tuner
        torch.backends.cudnn.benchmark = True
        
        # Print CUDA information
        print(f"{TermColors.GREEN}✅ PyTorch CUDA: {torch.version.cuda}{TermColors.ENDC}")
        print(f"{TermColors.GREEN}✅ cuDNN Version: {torch.backends.cudnn.version()}{TermColors.ENDC}")
        
        try:
            # Set memory strategy
            if GPU_MEMORY_LIMIT:
                torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_LIMIT / 100.0)
                print(f"{TermColors.GREEN}✅ GPU memory limit set to {GPU_MEMORY_LIMIT}%{TermColors.ENDC}")
            
            print(f"{TermColors.GREEN}✅ GPU memory configuration successful{TermColors.ENDC}")
        except RuntimeError as e:
            print(f"{TermColors.YELLOW}⚠️ GPU configuration failed: {e}{TermColors.ENDC}")
    
    # Start the augmentation process
    start_time = time.time()
    run_augmentation()
    end_time = time.time()
    
    # Print execution time
    duration = end_time - start_time
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"{TermColors.HEADER}\n{'='*50}")
    print(f"AUGMENTATION COMPLETE")
    print(f"{'='*50}")
    print(f"Total execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s")