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
DATA_DIR = r"C:\Users\stefa\Desktop\New folder\data\plant_images"
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

# Configure TensorFlow to use GPU when available
if USE_GPU:
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            print(f"{TermColors.GREEN}✅ Found {len(physical_devices)} GPU(s). Enabling GPU acceleration.{TermColors.ENDC}")
            # Configure memory growth to avoid OOM errors
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Only use the first GPU if multiple are available
            if len(physical_devices) > 1:
                tf.config.set_visible_devices(physical_devices[0], 'GPU')
                print(f"{TermColors.CYAN}ℹ Using GPU: {physical_devices[0].name}{TermColors.ENDC}")
        else:
            print(f"{TermColors.YELLOW}⚠️ No GPU found. Using CPU only.{TermColors.ENDC}")
    except Exception as e:
        print(f"{TermColors.RED}❌ Error configuring GPU: {e}. Using CPU only.{TermColors.ENDC}")
        USE_GPU = False

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
        
    elif transform_type == "leaf_curl":
        # Simulate leaf curl - mild warping effect
        rows, cols = img.shape[:2]
        map_y, map_x = np.mgrid[0:rows, 0:cols].astype(np.float32)
        
        # Create a wave pattern
        for i in range(rows):
            map_x[i,:] += 7 * np.sin(i/10)
        
        img_copy = cv2.remap(img.copy(), map_x, map_y, cv2.INTER_LINEAR)
        
    elif transform_type == "focus_blur":
        # Simulate depth-of-field effects common in plant photography
        # Create a radial gradient to simulate focus
        center_x, center_y = random.randint(w//4, 3*w//4), random.randint(h//4, 3*h//4)
        y, x = np.ogrid[-center_y:h-center_y, -center_x:w-center_x]
        mask = x*x + y*y <= min(h,w)**2
        
        # Create blurred version of the image
        blurred_img = cv2.GaussianBlur(img.copy(), (15, 15), 0)
        
        # Create distance-based mask for smooth transition
        dist = np.sqrt(x*x + y*y)
        max_dist = min(h, w) / 2
        focus_mask = np.clip(dist / max_dist, 0, 1)
        focus_mask = np.stack([focus_mask] * 3, axis=2)
        
        # Blend original and blurred image
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
    
    return img_copy

def create_augmentations(img):
    """Create multiple augmented versions of a plant image with specialized transformations
    
    New implementation that avoids problematic TensorFlow tensor operations
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
    
    # 1. PLANT-SPECIFIC TRANSFORMATIONS
    
    # Seasonal variations
    seasons = ["summer", "autumn", "winter", "spring", "drought", "overwatered"]
    for season in random.sample(seasons, 2):
        try:
            img_season = apply_seasonal_effect(img_np.copy(), season)
            img_season = preprocess_input(img_season)
            augmented_images.append(img_season)
        except Exception as e:
            print(f"Seasonal effect {season} failed: {e}")
    
    # Plant-specific transformations
    plant_transforms = ["leaf_wilt", "leaf_curl", "focus_blur", "growth_stage", "disease_spots"]
    for transform in random.sample(plant_transforms, 2):
        try:
            img_transformed = apply_plant_specific_transform(img_np.copy(), transform)
            img_transformed = preprocess_input(img_transformed)
            augmented_images.append(img_transformed)
        except Exception as e:
            print(f"Plant transform {transform} failed: {e}")
    
    # Lighting conditions
    lighting_conditions = ["shadow", "sunflare", "overexposed", "underexposed", "indoor"]
    for light in random.sample(lighting_conditions, 2):
        try:
            img_light = apply_lighting_condition(img_np.copy(), light)
            img_light = preprocess_input(img_light)
            augmented_images.append(img_light)
        except Exception as e:
            print(f"Lighting condition {light} failed: {e}")
    
    # 2. SIMPLE TRANSFORMATIONS
    
    # Horizontal flip - guaranteed to work
    try:
        img_flip = np.fliplr(img_np.copy())
        img_flip = preprocess_input(img_flip)
        augmented_images.append(img_flip)
    except Exception as e:
        print(f"Flip failed: {e}")
    
    # Rotation - using OpenCV for reliability
    rotation_angles = [5, 10, -5, -10, 15, -15]
    for angle in random.sample(rotation_angles, 2):
        try:
            # Use OpenCV rotation which is more reliable than TF
            h, w = img_np.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            img_rot = cv2.warpAffine(img_np.copy(), rotation_matrix, (w, h), 
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            img_rot = preprocess_input(img_rot)
            augmented_images.append(img_rot)
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
            
            img_zoom = preprocess_input(img_zoom)
            augmented_images.append(img_zoom)
        except Exception as e:
            print(f"Zoom {factor} failed: {e}")
    
    # Brightness variations
    brightness_factors = [0.8, 0.9, 1.1, 1.2]
    for factor in random.sample(brightness_factors, 2):
        try:
            img_bright = img_np.copy() * factor
            img_bright = np.clip(img_bright, 0, 255)
            img_bright = preprocess_input(img_bright)
            augmented_images.append(img_bright)
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
        img_crop = preprocess_input(img_crop)
        augmented_images.append(img_crop)
    except Exception as e:
        print(f"Center crop failed: {e}")
    
    # If we have too many augmentations, sample to the desired number
    if len(augmented_images) > AUGMENTATION_FACTOR:
        # Always keep original image (index 0)
        sampled = [augmented_images[0]] + random.sample(augmented_images[1:], AUGMENTATION_FACTOR-1)
        return sampled
    
    return augmented_images

def process_class_with_gpu(class_dir, class_name, thread_id, position, checkpoint):
    """Process a class with GPU acceleration"""
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
                        with tf.device('/GPU:0' if USE_GPU else '/CPU:0'):
                            # Load image
                            img = load_img(img_path, target_size=IMAGE_SIZE)
                            img = img_to_array(img)
                            
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
                                
                            # Convert back to uint8 for saving (CPU operation)
                            save_img = ((augmented_img + 1) * 127.5).astype(np.uint8)
                            output_path = os.path.join(original_dir, f"{basename}_aug{i}_{counter}.jpg")
                            cv2.imwrite(output_path, cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR))
                        
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
                    tf.keras.backend.clear_session()
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
                    global processed_count  # Changed from nonlocal to global
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
            # Configure TensorFlow for maximum performance
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                # Set TensorFlow to use maximum GPU memory available
                # This is important for full GPU utilization
                for gpu in gpus:
                    # Disable memory growth to use all available memory
                    tf.config.experimental.set_memory_growth(gpu, False)
                
                # Set environment variables for better GPU performance
                os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
                os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
                os.environ['TF_GPU_THREAD_COUNT'] = '1'
                os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
                
                # Configure TensorFlow for XLA JIT compilation
                # This can significantly improve performance for repeated operations
                tf.config.optimizer.set_jit(True)
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
    print(f"OFFLINE PLANT IMAGE AUGMENTATION WITH SEASONALITY")
    print(f"{'='*50}{TermColors.ENDC}")
    
    # Check for CUDA/GPU support
    if tf.test.is_gpu_available(cuda_only=True):
        print(f"{TermColors.GREEN}✅ CUDA GPU is available - Using GPU acceleration{TermColors.ENDC}")
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
    # Initialize TensorFlow settings for GPU optimization
    if USE_GPU:
        # Allow memory growth to avoid OOM errors
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    
                # Set memory limit to avoid using all GPU memory
                # tf.config.set_logical_device_configuration(
                #     gpus[0],
                #     [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # 4GB limit
                # )
                print(f"{TermColors.GREEN}✅ GPU memory growth enabled{TermColors.ENDC}")
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
    print(f"{'='*50}{TermColors.ENDC}")