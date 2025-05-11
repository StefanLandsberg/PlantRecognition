import math  # Make sure math library is imported
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))    # Script directory
BASE_DIR = os.path.dirname(SCRIPT_DIR)                     # Project root directory
DATA_DIR = os.path.join(BASE_DIR, "data", "plant_images") # Where images are saved
IMAGE_SIZE = (224, 224)
MIN_IMAGES_PER_CLASS = 200  # Increased minimum to ensure all classes have more images
MAX_IMAGES_PER_CLASS = 400  # Increased maximum to allow for more augmentations
CHECKPOINT_FILE = "augmentation_checkpoint.json"
# CPU and GPU optimization settings
MAX_CPU_WORKERS = os.cpu_count()  # Use all cores
PARALLEL_CLASSES = max(4, os.cpu_count() // 2)  # Process multiple classes in parallel
BATCH_SIZE = 64  # Increased batch size for better throughput
USE_GPU = True  # Use GPU when available
GPU_BATCH_SIZE = 128  # Larger batch size for GPU processing
GPU_MEMORY_LIMIT = None  # Set to None to use all available GPU memory

# Configure TensorFlow to use GPU when available
def setup_gpu():
    """Configure GPU for optimal performance"""
    if USE_GPU:
        try:
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                print(f"{TermColors.GREEN}✅ Found {len(physical_devices)} GPU(s). Enabling GPU acceleration.{TermColors.ENDC}")
                
                # Configure memory growth to avoid OOM errors
                for gpu in physical_devices:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Set up mixed precision for better performance
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                
                # Only use the first GPU if multiple are available
                if len(physical_devices) > 1:
                    tf.config.set_visible_devices(physical_devices[0], 'GPU')
                    print(f"{TermColors.CYAN}ℹ Using GPU: {physical_devices[0].name}{TermColors.ENDC}")
                
                # Create and run a small test tensor to initialize GPU
                try:
                    with tf.device('/GPU:0'):
                        test_tensor = tf.random.normal([100, 100])
                        test_result = tf.matmul(test_tensor, tf.transpose(test_tensor))
                        tf.reduce_sum(test_result).numpy()  # Force execution
                    print(f"{TermColors.GREEN}✅ GPU test successful.{TermColors.ENDC}")
                    return True
                except Exception as e:
                    print(f"{TermColors.YELLOW}⚠️ GPU initialized but test failed: {e}. Will use GPU selectively.{TermColors.ENDC}")
                    return False
            else:
                print(f"{TermColors.YELLOW}⚠️ No GPU found. Using CPU only.{TermColors.ENDC}")
                return False
        except Exception as e:
            print(f"{TermColors.RED}❌ Error configuring GPU: {e}. Using CPU only.{TermColors.ENDC}")
            return False
    return False

# Call GPU setup after defining the flags
USE_GPU_SUCCESSFULLY = setup_gpu() if USE_GPU else False

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

def analyze_class_sizes():
    """Analyze the dataset to determine target augmentation factors per class"""
    print(f"{TermColors.HEADER}\n{'='*50}")
    print(f"ANALYZING CLASS DISTRIBUTION")
    print(f"{'='*50}{TermColors.ENDC}")
    
    # Get class directories
    class_dirs = [os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    # Count original images per class
    class_counts = {}
    for class_dir in class_dirs:
        class_name = os.path.basename(class_dir)
        image_files = [f for f in os.listdir(class_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not '_aug' in f]
        class_counts[class_name] = len(image_files)
    
    # Calculate statistics
    if not class_counts:
        print(f"{TermColors.RED}❌ No classes found in {DATA_DIR}{TermColors.ENDC}")
        return {}
        
    total_classes = len(class_counts)
    total_images = sum(class_counts.values())
    min_images = min(class_counts.values())
    max_images = max(class_counts.values())
    avg_images = total_images / total_classes
    
    print(f"{TermColors.CYAN}ℹ Dataset Statistics:{TermColors.ENDC}")
    print(f"  - Total classes: {total_classes}")
    print(f"  - Total original images: {total_images}")
    print(f"  - Images per class range: {min_images} to {max_images}")
    print(f"  - Average images per class: {avg_images:.1f}")
    
    # Calculate custom augmentation factors per class
    augmentation_factors = {}
    
    # Shows class distribution
    small_classes = 0  # Classes with very few images (<5)
    large_classes = 0  # Classes with many images (>30)
    
    for class_name, count in class_counts.items():
        if count < 5:
            small_classes += 1
        elif count > 30:
            large_classes += 1
    
    print(f"  - Classes with <5 images: {small_classes} ({small_classes/total_classes:.1%})")
    print(f"  - Classes with >30 images: {large_classes} ({large_classes/total_classes:.1%})")
    
    # Calculate target augmentation factors to balance the dataset
    for class_name, count in class_counts.items():
        if count == 0:
            augmentation_factors[class_name] = 0
            continue
            
        # Calculate how many augmented images we need for this class
        if count < MIN_IMAGES_PER_CLASS / AUGMENTATION_FACTOR:
            # For very small classes, use maximum augmentation
            aug_factor = AUGMENTATION_FACTOR
        elif count * AUGMENTATION_FACTOR > MAX_IMAGES_PER_CLASS:
            # For very large classes, limit augmentation or sample subset
            # We'll use a subset of original images to avoid excessive data
            images_needed = MAX_IMAGES_PER_CLASS / AUGMENTATION_FACTOR
            aug_factor = min(AUGMENTATION_FACTOR, MAX_IMAGES_PER_CLASS / count)
        else:
            # Normal case - aim for target size but cap at max augmentation factor
            aug_factor = min(AUGMENTATION_FACTOR, TARGET_IMAGES_PER_CLASS / count)
        
        augmentation_factors[class_name] = aug_factor
    
    # Print some examples
    print(f"\n{TermColors.CYAN}ℹ Augmentation Strategy Examples:{TermColors.ENDC}")
    
    # Sort by count for better understanding
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])
    
    # Print smallest 3 classes
    print(f"{TermColors.YELLOW}⚠️ Smallest Classes:{TermColors.ENDC}")
    for class_name, count in sorted_classes[:3]:
        aug_factor = augmentation_factors[class_name]
        final_count = min(count * aug_factor, MAX_IMAGES_PER_CLASS)
        print(f"  - {class_name}: {count} orig → {final_count:.0f} total (×{aug_factor:.1f} augmentation)")
    
    # Print some medium classes
    mid_idx = len(sorted_classes) // 2
    print(f"{TermColors.GREEN}✅ Medium Classes:{TermColors.ENDC}")
    for class_name, count in sorted_classes[mid_idx:mid_idx+3]:
        aug_factor = augmentation_factors[class_name]
        final_count = min(count * aug_factor, MAX_IMAGES_PER_CLASS)
        print(f"  - {class_name}: {count} orig → {final_count:.0f} total (×{aug_factor:.1f} augmentation)")
    
    # Print largest 3 classes
    print(f"{TermColors.BLUE}ℹ Largest Classes:{TermColors.ENDC}")
    for class_name, count in sorted_classes[-3:]:
        aug_factor = augmentation_factors[class_name]
        final_count = min(count * aug_factor, MAX_IMAGES_PER_CLASS)
        print(f"  - {class_name}: {count} orig → {final_count:.0f} total (×{aug_factor:.1f} augmentation)")
    
    return augmentation_factors

def apply_occlusion(img, occlusion_type):
    """Apply occlusion to simulate partially visible plants
    
    Args:
        img: Input image as numpy array
        occlusion_type: Type of occlusion to apply
        
    Returns:
        Augmented image with occlusion
    """
    h, w = img.shape[:2]
    img_copy = img.copy()
    
    if occlusion_type == "partial_view":
        # Choose occlusion direction and amount
        direction = random.choice(["top", "bottom", "left", "right"])
        occlusion_percent = random.uniform(0.1, 0.3)  # Cover 10-30% of the image
        
        if direction == "top":
            occlusion_height = int(h * occlusion_percent)
            # Create natural occlusion (dark green leaves or brown stems)
            occlusion_color = random.choice([
                [50, 120, 50],    # Dark green
                [70, 50, 40],     # Brown
                [100, 100, 100],  # Gray
            ])
            img_copy[:occlusion_height, :] = occlusion_color
            
            # Add some noise for realism
            noise = np.random.randint(-20, 20, (occlusion_height, w, 3))
            img_copy[:occlusion_height, :] = np.clip(img_copy[:occlusion_height, :] + noise, 0, 255)
            
            # Add a gradual blend at the border
            for i in range(10):
                blend_row = occlusion_height + i
                if blend_row < h:
                    alpha = (10 - i) / 10
                    img_copy[blend_row, :] = alpha * np.array(occlusion_color) + (1 - alpha) * img[blend_row, :]
            
        elif direction == "bottom":
            occlusion_height = int(h * occlusion_percent)
            start_row = h - occlusion_height
            
            # Create natural occlusion (ground, leaf litter, or soil)
            occlusion_color = random.choice([
                [80, 70, 60],     # Brown soil
                [120, 110, 70],   # Dry leaves
                [120, 140, 80],   # Grass
            ])
            img_copy[start_row:, :] = occlusion_color
            
            # Add some noise for realism
            noise = np.random.randint(-20, 20, (occlusion_height, w, 3))
            img_copy[start_row:, :] = np.clip(img_copy[start_row:, :] + noise, 0, 255)
            
            # Add a gradual blend at the border
            for i in range(10):
                blend_row = start_row - i - 1
                if blend_row >= 0:
                    alpha = (10 - i) / 10
                    img_copy[blend_row, :] = (1 - alpha) * img[blend_row, :] + alpha * np.array(occlusion_color)
            
        elif direction == "left":
            occlusion_width = int(w * occlusion_percent)
            
            # Create natural occlusion 
            occlusion_color = random.choice([
                [50, 120, 50],    # Dark green
                [70, 50, 40],     # Brown
                [100, 100, 100],  # Gray
            ])
            img_copy[:, :occlusion_width] = occlusion_color
            
            # Add some noise for realism
            noise = np.random.randint(-20, 20, (h, occlusion_width, 3))
            img_copy[:, :occlusion_width] = np.clip(img_copy[:, :occlusion_width] + noise, 0, 255)
            
            # Add a gradual blend at the border
            for i in range(10):
                blend_col = occlusion_width + i
                if blend_col < w:
                    alpha = (10 - i) / 10
                    img_copy[:, blend_col] = alpha * np.array(occlusion_color) + (1 - alpha) * img[:, blend_col]
            
        else:  # right
            occlusion_width = int(w * occlusion_percent)
            start_col = w - occlusion_width
            
            # Create natural occlusion
            occlusion_color = random.choice([
                [50, 120, 50],    # Dark green
                [70, 50, 40],     # Brown
                [100, 100, 100],  # Gray
            ])
            img_copy[:, start_col:] = occlusion_color
            
            # Add some noise for realism
            noise = np.random.randint(-20, 20, (h, occlusion_width, 3))
            img_copy[:, start_col:] = np.clip(img_copy[:, start_col:] + noise, 0, 255)
            
            # Add a gradual blend at the border
            for i in range(10):
                blend_col = start_col - i - 1
                if blend_col >= 0:
                    alpha = (10 - i) / 10
                    img_copy[:, blend_col] = (1 - alpha) * img[:, blend_col] + alpha * np.array(occlusion_color)
    
    return img_copy

def apply_scale_variation(img, scale_type):
    """Apply scale variations to simulate different distances or macro photography
    
    Args:
        img: Input image as numpy array
        scale_type: Type of scale variation to apply
        
    Returns:
        Augmented image with scale variation effect
    """
    h, w = img.shape[:2]
    img_copy = img.copy()
    
    if scale_type == "macro":
        # Simulate macro photography (extreme close-up) by
        # zooming into a part of the image and adding some blur to the periphery
        
        # Choose an area to focus on (typically center biased for plants)
        center_bias = random.choice([True, True, False])  # 2/3 chance of center focus
        
        if center_bias:
            # Focus on center with slight random offset
            center_x = w // 2 + random.randint(-w//8, w//8)
            center_y = h // 2 + random.randint(-h//8, h//8)
        else:
            # Focus on a random point
            center_x = random.randint(w//4, 3*w//4)
            center_y = random.randint(h//4, 3*h//4)
        
        # Determine zoom factor
        zoom_factor = random.uniform(1.3, 1.8)  # 1.3x to 1.8x zoom
        
        # Calculate crop region
        new_w = int(w / zoom_factor)
        new_h = int(h / zoom_factor)
        
        # Ensure the crop region is within bounds
        x1 = max(0, min(center_x - new_w // 2, w - new_w))
        y1 = max(0, min(center_y - new_h // 2, h - new_h))
        
        # Crop and resize
        cropped = img_copy[y1:y1+new_h, x1:x1+new_w]
        img_copy = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Add slight depth-of-field blur effect (peripheral blur)
        # Create a mask that's strongest at the edges
        mask = np.zeros((h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                # Calculate distance from center as 0.0-1.0
                dist_y = abs(y - h/2) / (h/2)
                dist_x = abs(x - w/2) / (w/2)
                # Use the maximum of the x and y distances for a smoother effect
                mask[y, x] = max(dist_y, dist_x) ** 2  # Square for sharper falloff
        
        # Apply blur based on mask
        blurred = cv2.GaussianBlur(img_copy, (21, 21), 0)
        
        # Blend original and blurred based on mask
        mask_3c = np.stack([mask] * 3, axis=2)
        img_copy = img_copy * (1 - mask_3c * 0.7) + blurred * (mask_3c * 0.7)
        img_copy = np.clip(img_copy, 0, 255).astype(np.uint8)
        
    elif scale_type == "distant":
        # Simulate distant view by adding slight blur, reducing detail
        # and adding atmospheric perspective
        
        # Resize down and back up to reduce detail
        small_size = (w // 3, h // 3)
        small_img = cv2.resize(img_copy, small_size, interpolation=cv2.INTER_AREA)
        img_copy = cv2.resize(small_img, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Add slight blur
        img_copy = cv2.GaussianBlur(img_copy, (3, 3), 0)
        
        # Add atmospheric perspective (slight blue/gray tint and reduced contrast)
        atmospheric_color = np.array([180, 200, 230])  # Slight blue-gray
        
        # Create distance gradient (further = more atmospheric effect)
        gradient = np.zeros((h, w), dtype=np.float32)
        for y in range(h):
            # More atmospheric effect at top of image
            gradient[y, :] = 1.0 - y / h
        
        # Apply atmospheric effect
        atmospheric_strength = random.uniform(0.1, 0.25)
        gradient = gradient * atmospheric_strength
        
        # Convert to 3-channel
        gradient_3c = np.stack([gradient] * 3, axis=2)
        
        # Blend with atmospheric color
        img_copy = img_copy * (1 - gradient_3c) + atmospheric_color * gradient_3c
        
        # Reduce contrast slightly
        img_copy = (img_copy * 0.9 + np.mean(img_copy) * 0.1).astype(np.uint8)
    
    return img_copy

def apply_plant_age_variation(img, age_type):
    """Apply variations that simulate different plant growth stages
    
    Args:
        img: Input image as numpy array
        age_type: Type of age variation (young, mature, flowering, etc.)
        
    Returns:
        Augmented image with age variation effect
    """
    img_copy = img.copy()
    h, w = img.shape[:2]
    
    if age_type == "young":
        # Younger plants tend to be brighter green, smaller, less textured
        # Increase green channel
        img_copy[:,:,1] = np.clip(img_copy[:,:,1] * 1.2, 0, 255).astype(np.uint8)
        
        # Slightly reduce red to make it look fresher
        img_copy[:,:,0] = np.clip(img_copy[:,:,0] * 0.9, 0, 255).astype(np.uint8)
        
        # Reduce texture (young plants have less texture)
        img_copy = cv2.GaussianBlur(img_copy, (3, 3), 0)
        
    elif age_type == "mature":
        # Mature plants have more texture, deeper colors
        # Enhance texture with sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * 0.5
        img_copy = cv2.filter2D(img_copy, -1, kernel + np.identity(3) * 0.5)
        
        # Slightly reduce green and increase red/blue for deeper color
        img_copy[:,:,1] = np.clip(img_copy[:,:,1] * 0.9, 0, 255).astype(np.uint8)
        img_copy[:,:,0] = np.clip(img_copy[:,:,0] * 1.05, 0, 255).astype(np.uint8)
        img_copy[:,:,2] = np.clip(img_copy[:,:,2] * 1.05, 0, 255).astype(np.uint8)
        
    elif age_type == "flowering":
        # Flowering plants have more vibrant colors and brighter spots
        # First, detect potential flower regions (non-green, higher saturation)
        hsv = cv2.cvtColor(img_copy, cv2.COLOR_RGB2HSV)
        
        # Detect non-green hues with higher saturation (potential flowers)
        h = hsv[:,:,0]
        s = hsv[:,:,1]
        
        # Create mask for non-green, more saturated regions
        # Green hue is around 60 in the 0-180 OpenCV HSV range
        non_green_mask = np.logical_or(h < 40, h > 80)
        high_sat_mask = s > 70
        
        # Combine masks to get potential flower regions
        flower_mask = np.logical_and(non_green_mask, high_sat_mask).astype(np.float32)
        
        # If potential flowers detected, enhance them
        if np.mean(flower_mask) > 0.01:  # At least 1% of image
            # Dilate mask slightly to include flower edges
            flower_mask = cv2.dilate(flower_mask.astype(np.uint8), np.ones((5, 5), np.uint8))
            flower_mask = cv2.GaussianBlur(flower_mask, (9, 9), 0) / 255.0
            
            # Create 3-channel mask
            flower_mask_3c = np.stack([flower_mask] * 3, axis=2)
            
            # Enhance saturation and brightness in flower regions
            hsv = hsv.astype(np.float32)
            hsv[:,:,1] += flower_mask * 40  # Increase saturation
            hsv[:,:,2] += flower_mask * 30  # Increase brightness
            
            # Clip values
            hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
            hsv[:,:,2] = np.clip(hsv[:,:,2], 0, 255)
            
            # Convert back to RGB
            enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            
            # Use mask to preserve non-flower regions
            img_copy = img_copy * (1 - flower_mask_3c) + enhanced * flower_mask_3c
        else:
            # If no flowers detected, add slight color spots randomly (simulating emerging flowers)
            # Choose flower color
            flower_color = random.choice([
                [220, 180, 200],  # Pink
                [200, 150, 150],  # Light red
                [220, 220, 150],  # Yellow
                [200, 200, 200],  # White
                [180, 180, 220],  # Light blue
            ])
            
            # Create 3-5 small flower spots
            num_spots = random.randint(3, 5)
            for _ in range(num_spots):
                # Random position with bias toward top 2/3 of image
                x = random.randint(0, w-1)
                y = random.randint(0, int(h*0.7))
                
                # Random radius
                radius = random.randint(3, 15)
                
                # Create circular mask for this spot
                y_grid, x_grid = np.ogrid[:h, :w]
                dist_from_center = np.sqrt((x_grid - x)**2 + (y_grid - y)**2)
                
                # Soft circular mask
                spot_mask = np.clip(1.0 - dist_from_center / radius, 0, 1.0) ** 2
                spot_mask_3c = np.stack([spot_mask] * 3, axis=2)
                
                # Apply colored spot
                img_copy = img_copy * (1 - spot_mask_3c) + np.array(flower_color) * spot_mask_3c
    
    return img_copy.astype(np.uint8)

def apply_weather_condition(img, condition):
    """Apply weather condition effects to plant images
    
    Args:
        img: Input image as numpy array
        condition: Weather condition to simulate
        
    Returns:
        Augmented image with weather effect
    """
    h, w = img.shape[:2]
    img_copy = img.copy().astype(np.float32)
    
    if condition == "rain":
        # Add rain streaks
        num_drops = random.randint(100, 200)
        rain_layer = np.zeros_like(img_copy)
        
        for _ in range(num_drops):
            # Random position
            x = random.randint(0, w-1)
            y = random.randint(0, h-1)
            
            # Random length (longer drops in heavier rain)
            length = random.randint(5, 15)
            
            # Rain angle (typically diagonal)
            angle = random.uniform(-30, -5)  # Degrees
            
            # Calculate end point
            angle_rad = np.deg2rad(angle)
            x2 = int(x + length * np.cos(angle_rad))
            y2 = int(y + length * np.sin(angle_rad))
            
            # Draw rain drop
            cv2.line(rain_layer, (x, y), (x2, y2), (200, 200, 200), 1)
        
        # Blur rain layer slightly
        rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 0)
        
        # Blend rain with original
        alpha = 0.7  # Keep 70% of original
        img_copy = cv2.addWeighted(img_copy, alpha, rain_layer, 1-alpha, 0)
        
        # Reduce contrast and add blue tint for overcast sky
        img_copy = img_copy * 0.9  # Darker
        img_copy[:,:,2] += 10  # More blue
        
    elif condition == "fog":
        # Create fog effect (low contrast, whitish overlay)
        fog_color = np.array([220, 220, 220])
        
        # Create foggy layer with gradient (more fog at the top)
        fog_layer = np.zeros_like(img_copy)
        for y in range(h):
            # More fog at top
            fog_intensity = max(0, 0.8 - 0.6 * y / h)
            fog_layer[y, :] = fog_color * fog_intensity
        
        # Add noise to fog for texture
        noise = np.random.normal(0, 5, fog_layer.shape)
        fog_layer += noise
        fog_layer = np.clip(fog_layer, 0, 255)
        
        # Reduce contrast in the original image
        img_copy = img_copy * 0.8 + np.mean(img_copy) * 0.2
        
        # Blend with fog
        img_copy = img_copy + fog_layer
        
    elif condition == "bright_sun":
        # Create bright sunlight effect (high contrast, warm colors)
        # Increase brightness
        img_copy = img_copy * 1.2
        
        # Increase contrast
        mean = np.mean(img_copy, axis=(0, 1), keepdims=True)
        img_copy = (img_copy - mean) * 1.3 + mean
        
        # Add warm tint (increase red/yellow, decrease blue)
        img_copy[:,:,0] += 15  # More red
        img_copy[:,:,1] += 10  # More green (makes yellow with red)
        img_copy[:,:,2] -= 10  # Less blue
        
        # Add strong shadow somewhere in the image
        if random.random() < 0.7:  # 70% chance of shadow
            shadow_direction = random.choice(["top", "left", "right", "bottom"])
            shadow_size = random.uniform(0.2, 0.4)  # Shadow covers 20-40% of the image
            shadow_intensity = random.uniform(0.3, 0.5)  # Shadow darkens by 30-50%
            
            shadow_mask = np.ones((h, w), dtype=np.float32)
            
            if shadow_direction == "top":
                for y in range(int(h * shadow_size)):
                    gradient = 1.0 - (y / (h * shadow_size))
                    shadow_mask[y, :] = 1.0 - (shadow_intensity * gradient)
            elif shadow_direction == "left":
                for x in range(int(w * shadow_size)):
                    gradient = 1.0 - (x / (w * shadow_size))
                    shadow_mask[:, x] = 1.0 - (shadow_intensity * gradient)
            elif shadow_direction == "right":
                for x in range(int(w * shadow_size)):
                    gradient = 1.0 - (x / (w * shadow_size))
                    shadow_mask[:, w-x-1] = 1.0 - (shadow_intensity * gradient)
            elif shadow_direction == "bottom":
                for y in range(int(h * shadow_size)):
                    gradient = 1.0 - (y / (h * shadow_size))
                    shadow_mask[h-y-1, :] = 1.0 - (shadow_intensity * gradient)
            
            # Apply shadow
            shadow_mask_3c = np.stack([shadow_mask] * 3, axis=2)
            img_copy = img_copy * shadow_mask_3c
    
    # Clip to valid range
    img_copy = np.clip(img_copy, 0, 255)
    return img_copy.astype(np.uint8)

def create_augmentations(img, class_name=''):
    """Create multiple augmented versions of a plant image with specialized transformations
    
    Integrated version with all plant-specific augmentations carefully selected for real-world accuracy
    """
    # Convert to numpy array immediately to avoid tensor issues
    if isinstance(img, tf.Tensor):
        try:
            img_np = img.numpy()
            # Remove batch dimension if present
            if len(img_np.shape) == 4:
                img_np = img_np[0]
        except:
            # Direct conversion to numpy if tensor conversion fails
            img_np = np.array(img)
            if len(img_np.shape) == 4:
                img_np = img_np[0]
    else:
        # Already a numpy array
        img_np = np.array(img)
    
    # Ensure we have a valid image
    if len(img_np.shape) != 3 or img_np.shape[2] != 3:
        # Convert grayscale to RGB if needed
        if len(img_np.shape) == 3 and img_np.shape[2] == 1:
            img_np = np.repeat(img_np, 3, axis=2)
        elif len(img_np.shape) == 2:
            img_np = np.stack([img_np] * 3, axis=2)
        else:
            # Invalid image format, return just the original
            return [preprocess_input(img_np)]
    
    augmented_images = []
    
    # Original image
    img_orig = preprocess_input(img_np.copy())
    augmented_images.append(img_orig)
    
    # PLANT-SPECIFIC TRANSFORMATIONS - CAREFULLY SELECTED FOR ACCURACY
    
    # 1. Seasonal variations - very important for plants that change by season
    seasons = ["summer", "autumn", "winter", "spring", "drought", "overwatered"]
    for season in random.sample(seasons, 2):
        try:
            img_season = apply_seasonal_effect(img_np.copy(), season)
            img_season = preprocess_input(img_season)
            augmented_images.append(img_season)
        except Exception as e:
            print(f"Seasonal effect {season} failed: {e}")
    
    # 2. Plant-specific transformations - core plant characteristics
    plant_transforms = ["leaf_wilt", "leaf_curl", "focus_blur", "growth_stage", "disease_spots"]
    for transform in random.sample(plant_transforms, 2):
        try:
            img_transformed = apply_plant_specific_transform(img_np.copy(), transform)
            img_transformed = preprocess_input(img_transformed)
            augmented_images.append(img_transformed)
        except Exception as e:
            print(f"Plant transform {transform} failed: {e}")
    
    # 3. Lighting conditions - plants are photographed in varied lighting
    lighting_conditions = ["shadow", "sunflare", "overexposed", "underexposed", "indoor"]
    for light in random.sample(lighting_conditions, 2):
        try:
            img_light = apply_lighting_condition(img_np.copy(), light)
            img_light = preprocess_input(img_light)
            augmented_images.append(img_light)
        except Exception as e:
            print(f"Lighting condition {light} failed: {e}")
    
    # 4. Occlusion - plants are often partially visible or obscured
    occlusions = ["partial_view"]
    for occlusion in random.sample(occlusions, 1):
        try:
            img_occluded = apply_occlusion(img_np.copy(), occlusion)
            img_occluded = preprocess_input(img_occluded)
            augmented_images.append(img_occluded)
        except Exception as e:
            print(f"Occlusion {occlusion} failed: {e}")
    
    # 5. Scale variation - similar plants can have different sizes in images
    scales = ["macro", "distant"]
    for scale in random.sample(scales, 1):
        try:
            img_scale = apply_scale_variation(img_np.copy(), scale)
            img_scale = preprocess_input(img_scale)
            augmented_images.append(img_scale)
        except Exception as e:
            print(f"Scale variation {scale} failed: {e}")
    
    # 6. Plant age variations - plants look different at different growth stages
    ages = ["young", "mature", "flowering"]
    for age in random.sample(ages, 1):
        try:
            img_age = apply_plant_age_variation(img_np.copy(), age)
            img_age = preprocess_input(img_age)
            augmented_images.append(img_age)
        except Exception as e:
            print(f"Age variation {age} failed: {e}")
    
    # 7. Weather conditions - plants in different weather
    weather = ["rain", "fog", "bright_sun"]
    for condition in random.sample(weather, 1):
        try:
            img_weather = apply_weather_condition(img_np.copy(), condition)
            img_weather = preprocess_input(img_weather)
            augmented_images.append(img_weather)
        except Exception as e:
            print(f"Weather condition {condition} failed: {e}")
    
    # 8. Feature-preserving augmentations (NEW)
    try:
        img_features = apply_feature_preserving_augmentation(img_np.copy(), class_name)
        img_features = preprocess_input(img_features)
        augmented_images.append(img_features)
    except Exception as e:
        print(f"Feature-preserving augmentation failed: {e}")
        
    # 9. Background variation (NEW)
    try:
        img_bg = apply_background_variation(img_np.copy())
        img_bg = preprocess_input(img_bg)
        augmented_images.append(img_bg)
    except Exception as e:
        print(f"Background variation failed: {e}")
        
    # 10. Part-based augmentation (NEW)
    try:
        img_part = apply_part_based_augmentation(img_np.copy())
        img_part = preprocess_input(img_part)
        augmented_images.append(img_part)
    except Exception as e:
        print(f"Part-based augmentation failed: {e}")
    
    # 11. Standard geometric and color transforms (important basics)
    geometric_transforms = [
        tf.image.flip_left_right,  # Horizontal flip
        lambda x: tf.image.random_brightness(x, 0.2),  # Brightness variation
        lambda x: tf.image.random_contrast(x, 0.8, 1.2),  # Contrast variation
        lambda x: tf.image.random_saturation(x, 0.8, 1.2),  # Saturation variation
        lambda x: tf.image.rot90(x, k=1),  # 90-degree rotation
        lambda x: tf.image.rot90(x, k=3),  # 270-degree rotation (k=3 means 3x90 degrees)
        lambda x: tf.image.random_crop(  # Random crop and resize
            tf.pad(x, [[4, 4], [4, 4], [0, 0]], "REFLECT"), 
            size=[tf.shape(x)[0], tf.shape(x)[1], 3]
        )
    ]
    
    # Apply a subset of geometric transforms
    for transform_fn in random.sample(geometric_transforms, 4):
        try:
            img_tensor = tf.convert_to_tensor(img_np)
            img_transformed = transform_fn(img_tensor)
            img_transformed_np = img_transformed.numpy() if hasattr(img_transformed, 'numpy') else np.array(img_transformed)
            img_transformed_np = preprocess_input(img_transformed_np)
            augmented_images.append(img_transformed_np)
        except Exception as e:
            print(f"Transform {transform_fn.__name__} failed: {e}")
    
    return augmented_images

def calculate_class_specific_augmentation(class_dir, class_name, aug_factors=None):
    """
    Calculate how many augmentations to create for a specific class
    based on the number of original images and MIN/MAX limits
    """
    # Get original image files (excluding already augmented ones)
    image_files = [f for f in os.listdir(class_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not '_aug' in f]
    orig_count = len(image_files)
    
    if orig_count == 0:
        return [], 0, 0
    
    # Calculate how many augmentations we need to reach MIN_IMAGES_PER_CLASS
    if orig_count < MIN_IMAGES_PER_CLASS:
        # Calculate how many total augmented images we need
        total_new_images_needed = MIN_IMAGES_PER_CLASS - orig_count
        
        # Calculate how many augmentations per original image (can be fractional)
        aug_per_image = total_new_images_needed / orig_count
        
        # Cap at maximum augmentation factor to prevent over-distortion
        effective_factor = min(aug_per_image, MAX_AUGMENTATIONS_PER_IMAGE)
        
        # The actual target we'll reach
        target_count = min(MIN_IMAGES_PER_CLASS, orig_count + int(total_new_images_needed))
    else:
        # For classes that already have enough images, don't add more
        effective_factor = 0
        target_count = orig_count
        
        # If class exceeds MAX_IMAGES_PER_CLASS, and ENFORCE_EXACT_LIMITS is True, 
        # we would downsample (not implemented in this version)
        if ENFORCE_EXACT_LIMITS and orig_count > MAX_IMAGES_PER_CLASS:
            print(f"{TermColors.YELLOW}⚠️ Class {class_name} has {orig_count} images which exceeds MAX_IMAGES_PER_CLASS ({MAX_IMAGES_PER_CLASS}). Consider manual downsampling.{TermColors.ENDC}")
    
    # Return the exact count of augmentations we'll create
    return image_files, target_count, effective_factor

def apply_feature_preserving_augmentation(img, plant_class):
    """Apply augmentations that preserve key diagnostic plant features
    
    This function applies transformations that maintain taxonomically important 
    features while still creating data variety. Particularly useful for similar-looking
    species like grasses and plants with subtle diagnostic features.
    
    Args:
        img: Input image as numpy array
        plant_class: Name of the plant class (used to determine plant type)
        
    Returns:
        Feature-preserving augmented image
    """
    img_copy = img.copy()
    h, w = img.shape[:2]
    
    # Detect if the plant is likely a grass or similar difficult class
    is_grass = any(grass_term in plant_class.lower() for grass_term in 
                  ['grass', 'poaceae', 'carex', 'juncus', 'cyperus', 'bamboo', 
                   'stipa', 'festuca', 'poa'])
    
    is_small_leaved = any(term in plant_class.lower() for term in 
                         ['fern', 'moss', 'conifer', 'pine', 'juniper', 'cypress',
                          'leaf', 'needle', 'scale'])
    
    # 1. For grass-like plants - preserve vertical structure
    if is_grass:
        # Apply vertical-preserving distortion (avoid horizontal distortion)
        # We only stretch/compress vertically to preserve blade features
        stretch_factor = random.uniform(0.9, 1.1)
        new_h = int(h * stretch_factor)
        img_resized = cv2.resize(img_copy, (w, new_h))
        
        # Ensure we get back to original dimensions
        if new_h > h:
            # If stretched, crop center
            start_y = (new_h - h) // 2
            img_copy = img_resized[start_y:start_y+h, :]
        else:
            # If compressed, pad with image content from edges
            pad_top = (h - new_h) // 2
            pad_bottom = h - new_h - pad_top
            img_copy = cv2.copyMakeBorder(img_resized, pad_top, pad_bottom, 0, 0, 
                                        cv2.BORDER_REPLICATE)
        
        # Enhance edge detection to emphasize blade margins and venation
        try:
            gray = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges = cv2.dilate(edges, None)
            edge_mask = cv2.GaussianBlur(edges, (3, 3), 0) / 255.0
            
            # Create 3-channel edge mask
            edge_mask = np.stack([edge_mask] * 3, axis=2)
            
            # Enhance edges subtly (adding 10% edge enhancement)
            img_copy = img_copy.astype(np.float32)
            img_copy = img_copy * (1 + 0.1 * edge_mask)
            img_copy = np.clip(img_copy, 0, 255).astype(np.uint8)
        except:
            pass  # Fall back to non-enhanced version if edge detection fails
            
    # 2. For small-leaved or needle-like plants - enhance texture
    elif is_small_leaved:
        # Enhance fine details and texture that are diagnostic
        try:
            # Use detail enhancement to bring out the fine structures
            img_copy = cv2.detailEnhance(img_copy, sigma_s=15, sigma_r=0.15)
            
            # Slight sharpening to enhance edges of small structures
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            img_copy = cv2.filter2D(img_copy, -1, kernel)
        except:
            pass  # Fall back to original if enhancement fails
    
    # 3. For all plants - preserve color and enhance key features
    else:
        # Enhance contrast to make features more visible
        # Use CLAHE (Contrast Limited Adaptive Histogram Equalization) on L channel
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(img_copy, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            
            # Merge channels back
            enhanced_lab = cv2.merge((cl, a, b))
            
            # Convert back to RGB
            img_copy = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
            
            # Add a small amount of saturation boost for better feature visibility
            hsv = cv2.cvtColor(img_copy, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:,:,1] *= 1.1  # Increase saturation by 10%
            hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
            img_copy = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        except:
            pass  # Fall back to original if enhancement fails
    
    return img_copy

def apply_background_variation(img):
    """Apply advanced background variation techniques
    
    Plants in the wild are photographed against diverse backgrounds.
    This function varies backgrounds while preserving the plant subject.
    
    Args:
        img: Input image as numpy array
        
    Returns:
        Augmented image with varied background
    """
    h, w = img.shape[:2]
    img_copy = img.copy()
    
    # Choose a background variation technique
    variation_type = random.choice(["gradient", "texture", "blur", "natural"])
    
    # First attempt to segment the plant from the background
    # This is a simplified segmentation approach
    try:
        # Convert to HSV for better plant segmentation
        hsv = cv2.cvtColor(img_copy, cv2.COLOR_RGB2HSV)
        
        # Create a mask focusing on green/brown plant parts
        # This uses saturation and value channels which often separate plants well
        s = hsv[:,:,1]
        v = hsv[:,:,2]
        
        # Adaptive thresholding on saturation and value channels
        s_thresh = cv2.adaptiveThreshold(s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        v_thresh = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        
        # Combine masks
        plant_mask = cv2.bitwise_and(s_thresh, v_thresh)
        
        # Clean up mask with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE, kernel)
        plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_OPEN, kernel)
        
        # Dilate to ensure we get all of the plant
        plant_mask = cv2.dilate(plant_mask, kernel, iterations=2)
        
        # Ensure mask is properly formed
        if np.mean(plant_mask) < 10:  # Too little foreground
            # Fall back to center-focused mask
            y, x = np.ogrid[0:h, 0:w]
            center_y, center_x = h/2, w/2
            dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            radius = min(h, w) * 0.4  # Assume center 40% is plant
            plant_mask = (dist_from_center <= radius).astype(np.uint8) * 255
    except:
        # Fall back to center-focused mask if segmentation fails
        y, x = np.ogrid[0:h, 0:w]
        center_y, center_x = h/2, w/2
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        radius = min(h, w) * 0.4  # Assume center 40% is plant
        plant_mask = (dist_from_center <= radius).astype(np.uint8) * 255
    
    # Ensure plant_mask is 3 channels for blending
    if len(plant_mask.shape) == 2:
        plant_mask = np.stack([plant_mask] * 3, axis=2) / 255.0
    
    # Create new background based on variation type
    if variation_type == "gradient":
        # Create a color gradient background
        direction = random.choice(["horizontal", "vertical", "radial"])
        color1 = np.array([
            random.randint(100, 180),  # R
            random.randint(100, 180),  # G
            random.randint(100, 180),  # B
        ])
        color2 = np.array([
            random.randint(100, 180),  # R
            random.randint(100, 180),  # G
            random.randint(100, 180),  # B
        ])
        
        if direction == "horizontal":
            gradient = np.linspace(0, 1, w).reshape(1, w, 1)
            gradient = np.tile(gradient, (h, 1, 3))
            background = color1 * (1 - gradient) + color2 * gradient
        elif direction == "vertical":
            gradient = np.linspace(0, 1, h).reshape(h, 1, 1)
            gradient = np.tile(gradient, (1, w, 3))
            background = color1 * (1 - gradient) + color2 * gradient
        else:  # radial
            y, x = np.ogrid[0:h, 0:w]
            center_y, center_x = h/2, w/2
            dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            gradient = np.clip(dist_from_center / max_dist, 0, 1)
            gradient = np.stack([gradient] * 3, axis=2)
            background = color1 * (1 - gradient) + color2 * gradient
            
    elif variation_type == "texture":
        # Create a textured background
        texture_type = random.choice(["noise", "stripes", "dots"])
        
        if texture_type == "noise":
            # Random noise background
            background = np.random.randint(100, 180, (h, w, 3)).astype(np.float32)
            # Smooth noise slightly
            background = cv2.GaussianBlur(background, (21, 21), 0)
            
        elif texture_type == "stripes":
            # Striped background
            background = np.zeros((h, w, 3), dtype=np.float32)
            stripe_width = random.randint(10, 30)
            color1 = np.array([random.randint(100, 180)] * 3)
            color2 = np.array([random.randint(100, 180)] * 3)
            
            if random.choice([True, False]):
                # Horizontal stripes
                for i in range(0, h, stripe_width*2):
                    background[i:i+stripe_width, :] = color1
                    background[i+stripe_width:i+stripe_width*2, :] = color2
            else:
                # Vertical stripes
                for i in range(0, w, stripe_width*2):
                    background[:, i:i+stripe_width] = color1
                    background[:, i+stripe_width:i+stripe_width*2] = color2
                    
        else:  # dots
            # Dotted background
            background = np.ones((h, w, 3), dtype=np.float32) * 160
            dot_radius = random.randint(5, 15)
            num_dots = random.randint(20, 50)
            
            for _ in range(num_dots):
                cx = random.randint(0, w-1)
                cy = random.randint(0, h-1)
                color = np.array([random.randint(100, 180)] * 3)
                
                y, x = np.ogrid[0:h, 0:w]
                dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
                mask = dist_from_center <= dot_radius
                
                for c in range(3):
                    background[:,:,c][mask] = color[c]
                    
    elif variation_type == "blur":
        # Blurred version of the original background
        # Extract what we think is the background and blur it
        background_only = img_copy * (1 - plant_mask)
        background = cv2.GaussianBlur(background_only, (51, 51), 0)
        
        # Fill in any black spots
        background_mask = np.all(background < 5, axis=2)
        if np.any(background_mask):
            avg_color = np.mean(background[~background_mask], axis=0)
            for c in range(3):
                background[:,:,c][background_mask] = avg_color[c]
                
    else:  # natural
        # Use a natural gradient like sky or soil
        bg_type = random.choice(["sky", "soil", "foliage"])
        
        if bg_type == "sky":
            # Sky gradient (blue to light blue/white)
            background = np.zeros((h, w, 3), dtype=np.float32)
            for y in range(h):
                # Gradient from top to bottom
                factor = y / h
                background[y, :, 0] = 135 + 80 * factor  # R increases
                background[y, :, 1] = 206 + 30 * factor  # G increases
                background[y, :, 2] = 235 + 20 * factor  # B increases slightly
                
        elif bg_type == "soil":
            # Soil/ground texture (browns)
            background = np.zeros((h, w, 3), dtype=np.float32)
            
            # Base brown color
            background[:,:,0] = random.randint(140, 180)  # R
            background[:,:,1] = random.randint(100, 140)  # G
            background[:,:,2] = random.randint(60, 100)   # B
            
            # Add noise for texture
            noise = np.random.normal(0, 15, (h, w, 3))
            background = np.clip(background + noise, 0, 255)
            
        else:  # foliage
            # Green foliage background
            background = np.zeros((h, w, 3), dtype=np.float32)
            
            # Base green color
            background[:,:,0] = random.randint(60, 120)   # R
            background[:,:,1] = random.randint(120, 180)  # G
            background[:,:,2] = random.randint(60, 100)   # B
            
            # Add noise for texture
            noise = np.random.normal(0, 15, (h, w, 3))
            background = np.clip(background + noise, 0, 255)
    
    # Soften the mask edges for natural blending
    plant_mask_soft = cv2.GaussianBlur(plant_mask, (21, 21), 0)
    
    # Blend original image with new background using the plant mask
    result = img_copy * plant_mask_soft + background * (1 - plant_mask_soft)
    
    return result.astype(np.uint8)

def apply_part_based_augmentation(img, feature_type=None):
    """Apply augmentations focused on specific plant parts that are diagnostic for identification
    
    Plants are often identified by specific parts like flowers, fruits, leaves, or stems.
    This function creates augmentations that highlight and enhance particular plant parts.
    
    Args:
        img: Input image as numpy array
        feature_type: Type of feature to highlight (flowers, leaves, etc.) or None for automatic detection
        
    Returns:
        Augmented image with highlighted plant part
    """
    img_copy = img.copy()
    h, w = img.shape[:2]
    
    # If feature_type not specified, try to detect it or choose randomly
    if feature_type is None:
        # Try basic color-based detection of features
        try:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(img_copy, cv2.COLOR_RGB2HSV)
            
            # Check for flower colors (non-green, higher saturation areas)
            h_channel = hsv[:,:,0]
            s_channel = hsv[:,:,1]
            
            # Count pixels in flower color ranges (approximate)
            flower_hues = cv2.inRange(h_channel, 0, 30) | cv2.inRange(h_channel, 150, 180)  # Red-purple
            flower_hues = flower_hues | cv2.inRange(h_channel, 30, 90)  # Yellow-orange
            high_sat = s_channel > 100
            
            potential_flowers = cv2.bitwise_and(flower_hues, high_sat)
            flower_ratio = np.sum(potential_flowers) / (h * w)
            
            # Check for fruit colors
            fruit_mask = cv2.inRange(hsv, (0, 100, 100), (30, 255, 255))  # Red
            fruit_mask = fruit_mask | cv2.inRange(hsv, (30, 100, 100), (90, 255, 255))  # Yellow-orange
            fruit_ratio = np.sum(fruit_mask) / (h * w)
            
            # Detect leaves (green areas)
            leaf_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
            leaf_ratio = np.sum(leaf_mask) / (h * w)
            
            # Determine dominant feature
            if flower_ratio > 0.1:
                feature_type = "flowers"
            elif fruit_ratio > 0.1:
                feature_type = "fruits"
            elif leaf_ratio > 0.3:
                feature_type = "leaves"
            else:
                # If no clear feature detected, choose randomly
                feature_type = random.choice(["flowers", "leaves", "stem", "texture", "shape"])
        except:
            # Fall back to random selection if detection fails
            feature_type = random.choice(["flowers", "leaves", "stem", "texture", "shape"])
    
    # Apply part-specific enhancements
    if feature_type == "flowers":
        # Enhance flower visibility and color
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(img_copy, cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # Boost saturation
            hsv[:,:,1] *= 1.3
            hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
            
            # Boost brightness slightly
            hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.1, 0, 255)
            
            # Convert back
            img_copy = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            
            # Apply subtle sharpening to enhance flower details
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            img_copy = cv2.filter2D(img_copy, -1, kernel * 0.5 + np.eye(3) * 0.5)
        except:
            pass  # Fall back if enhancement fails
            
    elif feature_type == "leaves":
        # Enhance leaf visibility, edges, and venation
        try:
            # Enhance green channel slightly
            img_copy[:,:,1] = np.clip(img_copy[:,:,1] * 1.1, 0, 255)
            
            # Convert to LAB color space
            lab = cv2.cvtColor(img_copy, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel to enhance contrast (helps with venation)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            
            # Merge channels back
            enhanced_lab = cv2.merge((cl, a, b))
            
            # Convert back to RGB
            img_copy = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
            
            # Edge enhancement for leaf margins and venation
            gray = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 30, 100)
            edges = cv2.dilate(edges, None)
            edge_mask = cv2.GaussianBlur(edges, (3, 3), 0) / 255.0
            
            # Create 3-channel edge mask
            edge_mask = np.stack([edge_mask] * 3, axis=2)
            
            # Enhance edges
            img_copy = img_copy.astype(np.float32)
            img_copy = img_copy * (1 + 0.2 * edge_mask)  # 20% edge enhancement
            img_copy = np.clip(img_copy, 0, 255).astype(np.uint8)
        except:
            pass  # Fall back if enhancement fails
            
    elif feature_type == "fruits":
        # Enhance fruit colors and details
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(img_copy, cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # For fruit detection, focus on red-orange-yellow range
            fruit_mask = cv2.inRange(hsv.astype(np.uint8), (0, 100, 100), (30, 255, 255))  # Red
            fruit_mask = fruit_mask | cv2.inRange(hsv.astype(np.uint8), (30, 100, 100), (90, 255, 255))  # Yellow-orange
            
            # Dilate mask slightly to ensure we get all of the fruit
            kernel = np.ones((5, 5), np.uint8)
            fruit_mask = cv2.dilate(fruit_mask, kernel, iterations=1)
            fruit_mask = cv2.GaussianBlur(fruit_mask, (9, 9), 0)
            
            # Convert mask to float 0-1 range and 3 channels
            fruit_mask = fruit_mask / 255.0
            fruit_mask = np.stack([fruit_mask] * 3, axis=2)
            
            # Boost saturation in fruit areas, leave rest mostly unchanged
            hsv[:,:,1] = hsv[:,:,1] * (1 + 0.3 * fruit_mask[:,:,0])
            hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
            
            # Boost brightness slightly in fruit areas
            hsv[:,:,2] = hsv[:,:,2] * (1 + 0.2 * fruit_mask[:,:,0])
            hsv[:,:,2] = np.clip(hsv[:,:,2], 0, 255)
            
            # Convert back
            img_copy = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            
            # Sharpen fruit details
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(img_copy, -1, kernel * 0.5 + np.eye(3) * 0.5)
            
            # Apply sharpening selectively to fruit areas
            img_copy = img_copy * (1 - fruit_mask) + sharpened * fruit_mask
            img_copy = img_copy.astype(np.uint8)
        except:
            pass  # Fall back if enhancement fails
            
    elif feature_type == "stem":
        # Enhance stem visibility and structure
        try:
            # Create an edge-enhanced version that highlights linear structures
            # Using Sobel operator which is good for detecting vertical edges (stems)
            gray = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)
            
            # Vertical Sobel (good for stems)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            sobel_y = np.absolute(sobel_y)
            sobel_y = np.clip(sobel_y, 0, 255).astype(np.uint8)
            
            # Enhance Sobel result
            sobel_y = cv2.GaussianBlur(sobel_y, (3, 3), 0)
            
            # Convert to 3 channels for blending
            sobel_3c = np.stack([sobel_y] * 3, axis=2) / 255.0
            
            # Slightly enhance contrast in the original image
            lab = cv2.cvtColor(img_copy, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_lab = cv2.merge((cl, a, b))
            enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
            
            # Blend edge information with enhanced image
            img_copy = enhanced_img.astype(np.float32)
            img_copy = img_copy * (1 + 0.3 * sobel_3c)  # 30% stem enhancement
            img_copy = np.clip(img_copy, 0, 255).astype(np.uint8)
        except:
            pass  # Fall back if enhancement fails
            
    elif feature_type == "texture":
        # Enhance overall plant texture (useful for bark, leaf surfaces, etc.)
        try:
            # Detail enhancement algorithm
            img_copy = cv2.detailEnhance(img_copy, sigma_s=10, sigma_r=0.15)
            
            # Convert to LAB color space
            lab = cv2.cvtColor(img_copy, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            
            # Merge channels back
            enhanced_lab = cv2.merge((cl, a, b))
            
            # Convert back to RGB
            img_copy = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        except:
            pass  # Fall back if enhancement fails
    
    elif feature_type == "shape":
        # Enhance overall plant shape/silhouette
        try:
            # Create edge map of plant outline
            gray = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 30, 100)
            
            # Dilate edges to enhance outline
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            # Convert to 3 channels
            edges_3c = np.stack([edges] * 3, axis=2) / 255.0
            
            # Blend with slight darkening of background to make shape pop
            # Convert to HSV
            hsv = cv2.cvtColor(img_copy, cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # Reduce value (brightness) slightly except at edges
            hsv[:,:,2] = hsv[:,:,2] * (0.9 + 0.2 * edges_3c[:,:,0])
            hsv[:,:,2] = np.clip(hsv[:,:,2], 0, 255)
            
            # Convert back
            img_copy = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        except:
            pass  # Fall back if enhancement fails
    
    return img_copy

def apply_seasonal_effect(img, season_type):
    """Apply seasonality effects to plant images
    
    Args:
        img: Input image as numpy array
        season_type: Type of seasonal effect to apply
        
    Returns:
        Augmented image with seasonal effect
    """
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
    return img_float.astype(np.uint8)

def apply_background_variation(img, variation_type):
    """Apply background variations to help model distinguish plants from complex backgrounds
    
    Args:
        img: Input image as numpy array
        variation_type: Type of background variation
        
    Returns:
        Augmented image with modified background
    """
    h, w = img.shape[:2]
    
    if variation_type == "natural_background":
        # Simulate natural backgrounds by preserving foreground and varying background
        
        # Create a foreground mask using color-based segmentation
        # Convert to HSV which is better for plant segmentation
        hsv = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2HSV)
        
        # Plants tend to have higher green or yellow values
        # Create a mask focusing on green/plant parts (not perfect but useful for augmentation)
        green_lower = np.array([25, 40, 40])
        green_upper = np.array([90, 255, 255])
        plant_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Add reddish/purple flowers
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([160, 100, 100])
        red_upper2 = np.array([180, 255, 255])
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        flower_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Combine masks
        plant_mask = cv2.bitwise_or(plant_mask, flower_mask)
        
        # Dilate to improve coverage
        kernel = np.ones((5, 5), np.uint8)
        plant_mask = cv2.dilate(plant_mask, kernel, iterations=2)
        
        # Smooth mask edges
        plant_mask = cv2.GaussianBlur(plant_mask, (9, 9), 0)
        
        # Normalize and invert to get background mask
        plant_mask = plant_mask.astype(float) / 255.0
        bg_mask = 1.0 - plant_mask
        
        # Create new background - could be a texture, gradient or solid color
        bg_type = random.choice(["texture", "gradient", "solid"])
        
        if bg_type == "texture":
            # Create a natural texture (grass, soil, etc.)
            noise = np.random.normal(0, 1, (h, w))
            noise = cv2.GaussianBlur(noise, (21, 21), 0)
            noise = (noise - noise.min()) / (noise.max() - noise.min())  # Normalize to 0-1
            
            # Choose a natural background color
            bg_colors = [
                [139, 173, 95],   # Grass green
                [168, 143, 89],   # Soil brown
                [142, 164, 210],  # Sky blue
                [175, 175, 175],  # Stone grey
                [219, 209, 180],  # Sand
            ]
            bg_color = random.choice(bg_colors)
            
            # Create textured background
            bg = np.zeros_like(img, dtype=float)
            for c in range(3):
                bg[:,:,c] = bg_color[c] * (0.7 + 0.3 * noise)
                
        elif bg_type == "gradient":
            # Create a gradient background
            bg = np.zeros_like(img, dtype=float)
            
            # Choose direction
            direction = random.choice(["vertical", "horizontal", "diagonal"])
            
            # Choose colors for gradient (sky-like, soil-like, or natural tones)
            start_color = np.array([random.randint(100, 200), random.randint(100, 200), random.randint(100, 200)])
            end_color = np.array([random.randint(100, 200), random.randint(100, 200), random.randint(100, 200)])
            
            if direction == "vertical":
                # Vertical gradient
                for y in range(h):
                    ratio = y / h
                    color = start_color * (1 - ratio) + end_color * ratio
                    bg[y, :, :] = np.tile(color, (w, 1))
            elif direction == "horizontal":
                # Horizontal gradient
                for x in range(w):
                    ratio = x / w
                    color = start_color * (1 - ratio) + end_color * ratio
                    bg[:, x, :] = np.tile(color, (h, 1))
            else:
                # Diagonal gradient
                for y in range(h):
                    for x in range(w):
                        ratio = (x + y) / (w + h)
                        color = start_color * (1 - ratio) + end_color * ratio
                        bg[y, x, :] = color
        else:
            # Solid color background
            bg_colors = [
                [139, 173, 95],   # Grass green
                [168, 143, 89],   # Soil brown
                [142, 164, 210],  # Sky blue
                [175, 175, 175],  # Stone grey
                [219, 209, 180],  # Sand
            ]
            bg_color = random.choice(bg_colors)
            bg = np.ones_like(img, dtype=float) * np.array(bg_color)
        
        # Expand masks to 3 channels
        plant_mask_3c = np.stack([plant_mask] * 3, axis=2)
        bg_mask_3c = np.stack([bg_mask] * 3, axis=2)
        
        # Blend original image with new background
        result = img.astype(float) * plant_mask_3c + bg.astype(float) * bg_mask_3c
        
        # Add slight blur at the edges for natural transition
        result = cv2.GaussianBlur(result, (3, 3), 0)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    elif variation_type == "background_blur":
        # Apply selective background blur to simulate depth of field
        
        # Use a simple center mask as a proxy for plant location
        # (In production, a more sophisticated segmentation would be better)
        mask = np.zeros((h, w), dtype=np.float32)
        center_y, center_x = h // 2, w // 2
        radius = min(h, w) // 3
        
        # Create radial gradient mask
        y_grid, x_grid = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((y_grid - center_y)**2 + (x_grid - center_x)**2)
        mask = np.clip(1.0 - dist_from_center / radius, 0, 1)
        
        # Create foreground and background
        foreground = img.copy()
        background = cv2.GaussianBlur(img.copy(), (21, 21), 0)
        
        # Blend based on mask
        mask_3c = np.stack([mask] * 3, axis=2)
        result = foreground * mask_3c + background * (1 - mask_3c)
        
        return result.astype(np.uint8)
    
    elif variation_type == "lighting_variation":
        # Simulate varying lighting conditions across the image
        
        # Create a directional lighting gradient
        gradient = np.zeros((h, w), dtype=np.float32)
        
        # Choose light direction
        light_dir = random.choice(["top", "right", "bottom", "left", "top-right", "top-left"])
        
        if "top" in light_dir:
            for y in range(h):
                gradient[y, :] = 1.0 - y / h
        if "bottom" in light_dir:
            for y in range(h):
                gradient[y, :] = y / h
        if "left" in light_dir:
            for x in range(w):
                gradient[:, x] *= 1.0 - x / w
        if "right" in light_dir:
            for x in range(w):
                gradient[:, x] *= x / w
                
        # Normalize gradient
        if gradient.max() > 0:  # Avoid division by zero
            gradient = gradient / gradient.max()
            
        # Scale the lighting effect (0.7-1.3 range)
        gradient = 0.7 + gradient * 0.6
        
        # Apply gradient to image
        gradient_3c = np.stack([gradient] * 3, axis=2)
        result = img.astype(float) * gradient_3c
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    return img

def apply_lighting_condition(img, light_type):
    """Apply different lighting conditions typical in plant photography
    
    Args:
        img: Input image as numpy array
        light_type: Type of lighting condition
        
    Returns:
        Augmented image with modified lighting
    """
    img_float = img.astype(np.float32)
    
    if light_type == "shadow":
        # Create shadow effect on part of the image
        shadow_mask = np.ones_like(img_float)
        h, w = img.shape[:2]
        
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
        h, w = img.shape[:2]
        
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
    return img_float.astype(np.uint8)

def apply_part_focused_augmentation(img, plant_part="auto"):
    """Apply augmentations that highlight specific plant parts (flowers, leaves, stems, fruits)
    
    Args:
        img: Input image as numpy array
        plant_part: Target plant part to emphasize or "auto" to detect
        
    Returns:
        Augmented image with plant part emphasis
    """
    h, w = img.shape[:2]
    
    # Create a copy to avoid modifying the original
    result = img.copy()
    
    # For auto detection, we'll use color information to guess plant parts
    if plant_part == "auto":
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # Check for flowers using color - flowers are often brighter and more colorful
        # Red/pink/purple flowers
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])
        
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask_red1, mask_red2)
        
        # Yellow/orange flowers
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # White flowers
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Combine all flower masks
        flower_mask = cv2.bitwise_or(red_mask, yellow_mask)
        flower_mask = cv2.bitwise_or(flower_mask, white_mask)
        
        # Leaves are typically green
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([90, 255, 255])
        leaf_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Fruits often have distinctive colors (red, yellow, purple)
        fruit_mask = cv2.bitwise_or(red_mask, yellow_mask)
        
        # Determine which part to focus on based on the amount of each in the image
        flower_pixels = np.sum(flower_mask > 0)
        leaf_pixels = np.sum(leaf_mask > 0)
        fruit_pixels = np.sum(fruit_mask > 0)
        
        # Determine dominant part or fallback to random selection
        if flower_pixels > leaf_pixels and flower_pixels > fruit_pixels and flower_pixels > (h * w * 0.05):
            plant_part = "flower"
        elif leaf_pixels > flower_pixels and leaf_pixels > fruit_pixels and leaf_pixels > (h * w * 0.05):
            plant_part = "leaf"
        elif fruit_pixels > (h * w * 0.05):
            plant_part = "fruit"
        else:
            # If no clear part is detected, choose randomly
            plant_part = random.choice(["flower", "leaf", "stem", "fruit", "whole"])
    
    # Apply part-specific enhancements
    if plant_part == "flower":
        # Enhance flowers: increase saturation and brightness in flower regions
        hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Generate or use flower mask
        if 'flower_mask' not in locals():
            # Define flower colors
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([179, 255, 255])
            lower_yellow = np.array([15, 100, 100])
            upper_yellow = np.array([35, 255, 255])
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            
            hsv_img = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
            mask_red1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
            yellow_mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
            white_mask = cv2.inRange(hsv_img, lower_white, upper_white)
            
            flower_mask = cv2.bitwise_or(cv2.bitwise_or(mask_red1, mask_red2), 
                                         cv2.bitwise_or(yellow_mask, white_mask))
        
        # Dilate mask to include flower edges
        kernel = np.ones((5, 5), np.uint8)
        flower_mask = cv2.dilate(flower_mask, kernel, iterations=1)
        
        # Convert mask to float and normalize
        flower_mask = flower_mask.astype(np.float32) / 255.0
        
        # Create 3-channel mask
        flower_mask_3d = np.stack([flower_mask] * 3, axis=2)
        
        # Enhance saturation and brightness in flower regions
        hsv[:,:,1] *= 1.0 + (flower_mask * 0.3)  # Increase saturation in flower regions
        hsv[:,:,2] *= 1.0 + (flower_mask * 0.2)  # Increase brightness in flower regions
        
        # Convert back to RGB
        hsv[:,:,0] = np.clip(hsv[:,:,0], 0, 179)
        hsv[:,:,1:] = np.clip(hsv[:,:,1:], 0, 255)
        enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # Optionally add focus effect (blur background slightly)
        background = cv2.GaussianBlur(result, (5, 5), 0)
        result = background * (1 - flower_mask_3d) + enhanced * flower_mask_3d
        
    elif plant_part == "leaf":
        # Enhance leaves: increase green channel and contrast in leaf regions
        # Generate or use leaf mask
        if 'leaf_mask' not in locals():
            hsv_img = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
            lower_green = np.array([35, 40, 40])
            upper_green = np.array([90, 255, 255])
            leaf_mask = cv2.inRange(hsv_img, lower_green, upper_green)
        
        # Dilate mask to include leaf edges
        kernel = np.ones((5, 5), np.uint8)
        leaf_mask = cv2.dilate(leaf_mask, kernel, iterations=1)
        
        # Convert mask to float and normalize
        leaf_mask = leaf_mask.astype(np.float32) / 255.0
        
        # Create 3-channel mask
        leaf_mask_3d = np.stack([leaf_mask] * 3, axis=2)
        
        # Enhance green channel and contrast
        enhanced = result.astype(np.float32)
        enhanced[:,:,1] *= 1.0 + (leaf_mask * 0.2)  # Boost green channel
        
        # Apply contrast enhancement to leaf areas
        gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced_gray = clahe.apply(gray)
        
        # Blend enhanced contrast back into leaf areas
        for c in range(3):
            blend_factor = leaf_mask * 0.5
            enhanced[:,:,c] = enhanced[:,:,c] * (1 - blend_factor) + \
                              enhanced_gray.astype(np.float32) * blend_factor
        
        # Apply slight blur to background
        background = cv2.GaussianBlur(result, (3, 3), 0)
        result = background * (1 - leaf_mask_3d) + enhanced * leaf_mask_3d
        result = np.clip(result, 0, 255).astype(np.uint8)
        
    elif plant_part == "stem":
        # Enhance stems: use edge detection and vertical structure enhancement
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        
        # Apply Sobel edge detection with emphasis on vertical edges
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Emphasize vertical edges (stems tend to be vertical)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        angle = np.arctan2(sobely, sobelx) * 180 / np.pi
        
        # Create mask for vertical-ish edges
        vertical_mask = np.zeros_like(gray, dtype=np.float32)
        # Consider angles close to 90 or -90 degrees as vertical
        vertical_regions = np.logical_or(
            np.logical_and(angle > 70, angle < 110),
            np.logical_and(angle < -70, angle > -110)
        )
        vertical_mask[vertical_regions] = magnitude[vertical_regions]
        
        # Normalize mask
        if vertical_mask.max() > 0:
            vertical_mask = vertical_mask / vertical_mask.max()
        
        # Enhance contrast in stem regions
        enhanced = result.astype(np.float32)
        
        # Apply sharpening to the entire image to make stems more visible
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(result, -1, kernel)
        
        # Blend based on vertical mask
        vertical_mask_3d = np.stack([vertical_mask] * 3, axis=2)
        result = result * (1 - vertical_mask_3d * 0.7) + sharpened * (vertical_mask_3d * 0.7)
        result = np.clip(result, 0, 255).astype(np.uint8)
        
    elif plant_part == "fruit":
        # Enhance fruits: increase saturation and contrast in fruit regions
        # Generate or use fruit mask
        if 'fruit_mask' not in locals():
            hsv_img = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
            
            # Red/orange/yellow fruits
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([179, 255, 255])
            lower_yellow = np.array([15, 100, 100])
            upper_yellow = np.array([35, 255, 255])
            
            mask_red1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
            yellow_mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
            
            fruit_mask = cv2.bitwise_or(cv2.bitwise_or(mask_red1, mask_red2), yellow_mask)
        
        # Dilate mask
        kernel = np.ones((5, 5), np.uint8)
        fruit_mask = cv2.dilate(fruit_mask, kernel, iterations=1)
        
        # Convert mask to float and normalize
        fruit_mask = fruit_mask.astype(np.float32) / 255.0
        
        # Create 3-channel mask
        fruit_mask_3d = np.stack([fruit_mask] * 3, axis=2)
        
        # Enhance fruits
        hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Increase saturation and brightness
        hsv[:,:,1] *= 1.0 + (fruit_mask * 0.3)  # Increase saturation
        hsv[:,:,2] *= 1.0 + (fruit_mask * 0.1)  # Slight brightness boost
        
        # Convert back to RGB
        hsv[:,:,0] = np.clip(hsv[:,:,0], 0, 179)
        hsv[:,:,1:] = np.clip(hsv[:,:,1:], 0, 255)
        enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # Add focus effect
        background = cv2.GaussianBlur(result, (5, 5), 0)
        result = background * (1 - fruit_mask_3d) + enhanced * fruit_mask_3d
    
    elif plant_part == "whole":
        # Global enhancement for whole plant
        # Apply general sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(result, -1, kernel)
        
        # Apply contrast enhancement
        lab = cv2.cvtColor(sharpened, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return result.astype(np.uint8)

def apply_geometric_transform(img, transform_type):
    """Apply geometric transformations to the image
    
    Args:
        img: Input image as numpy array
        transform_type: Type of geometric transformation (rotate, flip, crop, perspective)
        
    Returns:
        Transformed image
    """
    h, w = img.shape[:2]
    
    if transform_type == 'rotate':
        # Random rotation between -30 and 30 degrees
        angle = random.uniform(-30, 30)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
    elif transform_type == 'flip':
        # Random flip (horizontal, vertical, or both)
        flip_mode = random.choice([-1, 0, 1])  # -1: both, 0: vertical, 1: horizontal
        return cv2.flip(img, flip_mode)
        
    elif transform_type == 'crop':
        # Random crop and resize
        crop_percent = random.uniform(0.8, 0.95)
        crop_w = int(w * crop_percent)
        crop_h = int(h * crop_percent)
        
        # Random position for crop
        x = random.randint(0, w - crop_w)
        y = random.randint(0, h - crop_h)
        
        # Perform crop
        crop = img[y:y+crop_h, x:x+crop_w]
        
        # Resize back to original dimensions
        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
        
    elif transform_type == 'perspective':
        # Random perspective transform
        # Define the 4 source points
        pts1 = np.float32([
            [0, 0],
            [w, 0],
            [0, h],
            [w, h]
        ])
        
        # Define the 4 destination points with random perturbation
        # We'll keep perturbations modest (5-15% of dimensions) to avoid extreme distortion
        perturbation = random.uniform(0.05, 0.15)
        max_offset_w = int(w * perturbation)
        max_offset_h = int(h * perturbation)
        
        pts2 = np.float32([
            [random.randint(0, max_offset_w), random.randint(0, max_offset_h)],
            [random.randint(w-max_offset_w, w), random.randint(0, max_offset_h)],
            [random.randint(0, max_offset_w), random.randint(h-max_offset_h, h)],
            [random.randint(w-max_offset_w, w), random.randint(h-max_offset_h, h)]
        ])
        
        # Get transformation matrix and apply the perspective transform
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    return img  # Return original if no transform applied

def apply_color_transform(img, transform_type):
    """Apply color transformations to the image
    
    Args:
        img: Input image as numpy array
        transform_type: Type of color transformation
        
    Returns:
        Color-transformed image
    """
    img_float = img.astype(np.float32)
    
    if transform_type == 'brightness':
        # Random brightness adjustment
        factor = random.uniform(0.7, 1.3)
        img_float = img_float * factor
        
    elif transform_type == 'contrast':
        # Random contrast adjustment
        factor = random.uniform(0.7, 1.3)
        mean = np.mean(img_float, axis=(0, 1), keepdims=True)
        img_float = (img_float - mean) * factor + mean
        
    elif transform_type == 'saturation':
        # Convert to HSV and adjust saturation
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        factor = random.uniform(0.7, 1.3)
        hsv[:, :, 1] = hsv[:, :, 1] * factor
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        img_float = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
    elif transform_type == 'hue':
        # Convert to HSV and shift hue
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # Hue is 0-179 in OpenCV
        shift = random.randint(-10, 10)
        hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
        img_float = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
    elif transform_type == 'noise':
        # Add random noise
        noise_type = random.choice(['gaussian', 'salt_pepper'])
        
        if noise_type == 'gaussian':
            # Add gaussian noise
            stddev = random.uniform(5, 20)
            noise = np.random.normal(0, stddev, img.shape).astype(np.float32)
            img_float = img_float + noise
            
        else:  # salt_pepper
            # Add salt and pepper noise
            s_vs_p = 0.5  # ratio of salt vs pepper
            amount = random.uniform(0.01, 0.04)  # total amount of noise
            img_float = img.copy()
            
            # Salt (white) noise
            num_salt = int(amount * img.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape[:-1]]
            img_float[coords[0], coords[1], :] = 255
            
            # Pepper (black) noise
            num_pepper = int(amount * img.size * (1 - s_vs_p))
            coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape[:-1]]
            img_float[coords[0], coords[1], :] = 0
    
    # Clip values to valid range and convert back to uint8
    img_float = np.clip(img_float, 0, 255).astype(np.uint8)
    return img_float

def apply_plant_specific_transform(img, transform_type):
    """Apply plant-specific transformations
    
    Args:
        img: Input image as numpy array
        transform_type: Type of plant-specific transformation
        
    Returns:
        Augmented image with plant-specific changes
    """
    img_copy = img.copy()
    h, w = img.shape[:2]
    
    if transform_type == "leaf_wilt":
        # Simulate wilting leaves by applying a slight downward distortion
        rows, cols = img.shape[:2]
        
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
        img_copy = cv2.remap(img.copy(), map_x, map_y, cv2.INTER_LINEAR)
        
    elif transform_type == "feature_preserving":
        # Feature-preserving augmentation for difficult plant groups
        # This preserves key diagnostic regions while varying less important areas
        
        # Create a mask for the center region (typically contains diagnostic features)
        feature_mask = np.zeros((h, w), dtype=np.float32)
        
        # Define feature region (center with slight bottom bias where flowers/fruits often are)
        center_y, center_x = int(h * 0.55), w // 2
        feature_radius = min(h, w) // 3
        
        # Create a gradient feature mask (1.0 for key features, decreasing outward)
        y_grid, x_grid = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((y_grid - center_y)**2 + (x_grid - center_x)**2)
        
        # Create smooth gradient for feature preservation
        feature_mask = 1.0 - np.clip(dist_from_center / feature_radius, 0, 1)
        feature_mask = feature_mask**2  # Sharper falloff to preserve key features
        
        # Get HSV for more natural modification
        try:
            hsv = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # Apply more aggressive augmentation to non-feature regions
            # Vary saturation and value in background while preserving features
            sat_change = random.uniform(0.7, 1.3)
            val_change = random.uniform(0.8, 1.2)
            
            # Expand mask to 3 channels for HSV manipulation
            mask_3d = np.stack([np.ones_like(feature_mask), feature_mask, feature_mask], axis=2)
            
            # Create transformation matrix (no change for features, more change for background)
            hsv_transform = np.ones_like(hsv)
            hsv_transform[:,:,1] = sat_change  # Saturation change
            hsv_transform[:,:,2] = val_change  # Value change
            
            # Blend based on feature mask - features preserved, background varied
            hsv_mod = hsv * (mask_3d + (1 - mask_3d) * hsv_transform)
            
            # Convert back to RGB
            hsv_mod[:,:,0] = np.clip(hsv_mod[:,:,0], 0, 179)  # Hue range in OpenCV
            hsv_mod[:,:,1:] = np.clip(hsv_mod[:,:,1:], 0, 255)
            img_copy = cv2.cvtColor(hsv_mod.astype(np.uint8), cv2.COLOR_HSV2RGB)
            
        except Exception as e:
            print(f"Feature-preserving augmentation error: {e}. Using simpler method.")
            # Fallback to simpler method if HSV conversion fails
            mask_3d = np.stack([feature_mask] * 3, axis=2)
            background_change = np.ones_like(img) * random.uniform(0.8, 1.2)
            img_copy = img * mask_3d + img * (1 - mask_3d) * background_change
            img_copy = np.clip(img_copy, 0, 255).astype(np.uint8)
    
    return img_copy

def apply_augmentation(img, intensity='medium'):
    """Apply various augmentations to the image with GPU acceleration when possible
    
    Args:
        img: Input image as numpy array
        intensity: Augmentation intensity level (low, medium, high)
        
    Returns:
        Augmented image with various transformations
    """
    # Define probability scales based on intensity
    if intensity == 'low':
        prob_scale = 0.5
    elif intensity == 'high':
        prob_scale = 1.5
    else:  # medium (default)
        prob_scale = 1.0
    
    # Check if we should use GPU for this image
    should_use_gpu = USE_GPU and random.random() < 0.8  # Reduce to 50% chance due to errors
    
    if should_use_gpu:
        # GPU-accelerated augmentation path
        try:
            # Convert to tensor for TensorFlow operations - ensure shape is correct
            img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
            
            # Ensure we have a 3D tensor (HWC format)
            if len(img_tensor.shape) != 3 or img_tensor.shape[2] != 3:
                raise ValueError(f"Invalid image shape: {img_tensor.shape}")
            
            # Apply simple transformations that are less likely to cause tensor shape issues
            transformed_tensor = img_tensor
            
            # Random horizontal flip (safe operation)
            if random.random() < 0.5 * prob_scale:
                transformed_tensor = tf.image.flip_left_right(transformed_tensor)
            
            # Random brightness adjustment (safe operation)
            if random.random() < 0.5 * prob_scale:
                delta = random.uniform(-0.2, 0.2)
                transformed_tensor = tf.image.adjust_brightness(transformed_tensor, delta)
            
            # Random contrast (safe operation, using static inputs)
            if random.random() < 0.3 * prob_scale:
                contrast_factor = random.uniform(0.8, 1.2)
                # Add batch dimension, adjust contrast, then remove batch dimension
                transformed_tensor = tf.expand_dims(transformed_tensor, 0)
                transformed_tensor = tf.image.adjust_contrast(
                    transformed_tensor, contrast_factor)
                transformed_tensor = transformed_tensor[0]
            
            # Random 90-degree rotation (safe operation)
            if random.random() < 0.3 * prob_scale:
                k = random.randint(1, 3)  # 90, 180, or 270 degrees
                transformed_tensor = tf.image.rot90(transformed_tensor, k=k)
            
            # Convert back to numpy and ensure uint8 type
            img_result = tf.clip_by_value(transformed_tensor, 0, 255)
            img_result = tf.cast(img_result, tf.uint8)
            img = img_result.numpy()
            
        except Exception as e:
            # Fall back to CPU silently - we don't need to log every error
            pass
    
    # Apply CPU-based transformations (either as fallback or additional augmentations)
    # Apply random geometric transformations
    if random.random() < 0.8 * prob_scale:
        transform_type = random.choice(['rotate', 'flip', 'crop', 'perspective'])
        img = apply_geometric_transform(img, transform_type)
    
    # Apply random color transformations
    if random.random() < 0.7 * prob_scale:
        transform_type = random.choice(['brightness', 'contrast', 'saturation', 'hue', 'noise'])
        img = apply_color_transform(img, transform_type)
    
    # Apply seasonal effects (less frequently)
    if random.random() < 0.3 * prob_scale:
        season_type = random.choice(['summer', 'autumn', 'winter', 'spring'])
        img = apply_seasonal_effect(img, season_type)
    
    # Apply one advanced effect (plant-specific transformations)
    if random.random() < 0.5 * prob_scale:
        effect_type = random.choice([
            "lighting", "occlusion", "scale", "age", "weather", 
            "plant_specific", "background"
        ])
        
        if effect_type == "lighting":
            img = apply_lighting_condition(img, random.choice([
                'shadow', 'overexposed', 'underexposed'
            ]))
        elif effect_type == "occlusion":
            img = apply_occlusion(img, "partial_view")
        elif effect_type == "scale":
            img = apply_scale_variation(img, random.choice(['macro', 'distant']))
        elif effect_type == "age":
            img = apply_plant_age_variation(img, random.choice(['young', 'mature', 'flowering']))
        elif effect_type == "weather":
            img = apply_weather_condition(img, random.choice(['rain', 'fog', 'bright_sun']))
        elif effect_type == "plant_specific":
            img = apply_plant_specific_transform(img, random.choice(['leaf_wilt', 'feature_preserving']))
        elif effect_type == "background":
            img = apply_background_variation(img, random.choice(['natural_background', 'background_blur']))
    
    return img

def main():
    """Main function to run the augmentation process"""
    try:
        print(f"{TermColors.HEADER}\n{'='*50}")
        print(f"OFFLINE DATA AUGMENTATION")
        print(f"{'='*50}{TermColors.ENDC}")
        
        # Load checkpoint to resume from previous run if available
        checkpoint = load_checkpoint()
        
        # Analyze dataset and determine augmentation factors
        print(f"\n{TermColors.CYAN}ℹ Analyzing dataset...{TermColors.ENDC}")
        aug_factors = analyze_class_sizes()
        
        if not aug_factors:
            print(f"{TermColors.RED}❌ No valid classes found. Check your dataset directory.{TermColors.ENDC}")
            return
            
        # Get list of classes to process
        class_dirs = [os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
        classes_to_process = [os.path.basename(d) for d in class_dirs if os.path.basename(d) not in checkpoint["processed_classes"]]
        
        print(f"\n{TermColors.GREEN}✓ Found {len(class_dirs)} total classes.")
        print(f"✓ {len(checkpoint['processed_classes'])} classes already processed.")
        print(f"✓ {len(classes_to_process)} classes to process.{TermColors.ENDC}")
        
        # Print GPU information
        if USE_GPU_SUCCESSFULLY:
            print(f"\n{TermColors.GREEN}✓ GPU acceleration enabled and working{TermColors.ENDC}")
        elif USE_GPU:
            print(f"\n{TermColors.YELLOW}⚠️ GPU acceleration enabled but may not be optimal - will use selectively{TermColors.ENDC}")
        else:
            print(f"\n{TermColors.YELLOW}⚠️ Running in CPU-only mode{TermColors.ENDC}")
        
        # Process classes in parallel
        print(f"\n{TermColors.CYAN}ℹ Starting augmentation process...{TermColors.ENDC}")
        print(f"ℹ Processing {PARALLEL_CLASSES} classes in parallel")
        
        # Initialize a global lock for thread-safe progress bar updates
        global lock
        lock = threading.Lock()
        
        # Use ThreadPoolExecutor for parallel processing of classes
        with ThreadPoolExecutor(max_workers=PARALLEL_CLASSES) as executor:
            futures = []
            for class_name in classes_to_process:
                class_dir = os.path.join(DATA_DIR, class_name)
                # Start the augmentation for this class
                future = executor.submit(
                    augment_class_images, 
                    class_dir, 
                    class_name, 
                    aug_factors
                )
                futures.append(future)
                
            # Monitor progress
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                completed += 1
                try:
                    class_name, augmented_count = future.result()
                    print(f"{TermColors.GREEN}✓ Completed {class_name}: {augmented_count} augmented images created ({completed}/{len(classes_to_process)}){TermColors.ENDC}")
                    
                    # Update checkpoint
                    checkpoint["processed_classes"].append(class_name)
                    save_checkpoint(checkpoint)
                    
                except Exception as e:
                    print(f"{TermColors.RED}❌ Error processing class: {e}{TermColors.ENDC}")
                
        print(f"\n{TermColors.GREEN}✓ Augmentation complete! All {len(class_dirs)} classes processed.{TermColors.ENDC}")
        
    except Exception as e:
        import traceback
        print(f"{TermColors.RED}❌ Error in main function: {e}{TermColors.ENDC}")
        traceback.print_exc()

def augment_class_images(class_dir, class_name, aug_factors=None):
    """Process all images in a class directory to reach exactly MIN_IMAGES_PER_CLASS"""
    # Calculate class-specific augmentation parameters
    image_files, target_count, aug_factor = calculate_class_specific_augmentation(
        class_dir, class_name, aug_factors
    )
    
    if not image_files:
        print(f"{TermColors.YELLOW}⚠️ No original images found in {class_name}, skipping.{TermColors.ENDC}")
        return class_name, 0
    
    # Get existing augmented images
    existing_aug = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')) and '_aug' in f]
    orig_count = len(image_files)
    total_existing = orig_count + len(existing_aug)
    
    # Skip if we already meet the minimum target
    if total_existing >= MIN_IMAGES_PER_CLASS:
        print(f"{TermColors.GREEN}✓ Class {class_name} already has {total_existing} images (≥{MIN_IMAGES_PER_CLASS}). Skipping.{TermColors.ENDC}")
        return class_name, 0
        
    # Calculate exactly how many new augmented images we need
    exact_new_needed = target_count - orig_count
    
    # Create progress bar for this class
    pbar = tqdm(total=len(image_files), desc=f"Processing {class_name}", leave=True, position=0)
    
    # Counter for augmented images
    augmented_count = 0
    
    # Process each original image
    for i, img_file in enumerate(image_files):
        try:
            # Stop if we've reached our target
            if augmented_count >= exact_new_needed:
                break
                
            img_path = os.path.join(class_dir, img_file)
            
            # Load the original image
            original_img = cv2.imread(img_path)
            if original_img is None:
                pbar.update(1)
                continue
                
            # Convert from BGR to RGB (OpenCV loads as BGR)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            
            # Resize to target size for consistency
            original_img = cv2.resize(original_img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
            
            # Calculate how many augmentations for this image to ensure even distribution
            # Use ceiling division to ensure we make progress toward the target
            images_left = len(image_files) - i
            augs_per_image = math.ceil((exact_new_needed - augmented_count) / images_left)
            augs_per_image = min(augs_per_image, MAX_AUGMENTATIONS_PER_IMAGE)  # Cap at max
            
            # Create augmented versions
            for aug_idx in range(augs_per_image):
                # Stop if we've reached our target
                if augmented_count >= exact_new_needed:
                    break
                    
                # Apply augmentation with random intensity
                intensity = random.choice(['low', 'medium', 'medium', 'high'])
                augmented = apply_augmentation(original_img.copy(), intensity)
                
                # Convert back to BGR for saving
                augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                
                # Generate output filename
                base_name = os.path.splitext(img_file)[0]
                out_file = f"{base_name}_aug{aug_idx}.jpg"
                out_path = os.path.join(class_dir, out_file)
                
                # Save the augmented image
                cv2.imwrite(out_path, augmented)
                augmented_count += 1
                
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({"Augmentations": augmented_count, "Target": exact_new_needed})
                
        except Exception as e:
            pbar.update(1)
            print(f"{TermColors.RED}❌ Error processing image {img_file}: {e}{TermColors.ENDC}")
    
    # Close progress bar
    pbar.close()
    
    final_count = orig_count + augmented_count
    print(f"{TermColors.GREEN}✓ Class {class_name}: Created {augmented_count} augmented images, now has {final_count} total images{TermColors.ENDC}")
    
    return class_name, augmented_count

if __name__ == "__main__":
    main()