import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import random
import time
import json
from pathlib import Path
from enum import Enum

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from scripts.PlantRecognitionV2 import set_seeds, TermColors

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# --- Constants ---
IMAGE_SIZE = (224, 224)
DEFAULT_MAGNITUDE = 10  # 0-30 range by AutoAugment standards
DEFAULT_N_OPS = 2  # Number of operations to apply

# --- AutoAugment Policy Definitions ---
class Operation(Enum):
    # Geometric transforms
    ROTATE = "rotate"
    TRANSLATE_X = "translate_x"
    TRANSLATE_Y = "translate_y"
    SHEAR_X = "shear_x"
    SHEAR_Y = "shear_y"
    FLIP_LR = "flip_lr"
    FLIP_UD = "flip_ud"
    
    # Color transforms
    COLOR = "color"
    CONTRAST = "contrast"
    BRIGHTNESS = "brightness"
    SHARPNESS = "sharpness"
    POSTERIZE = "posterize"
    SOLARIZE = "solarize"
    INVERT = "invert"
    EQUALIZE = "equalize"
    AUTOCONTRAST = "autocontrast"
    
    # Noise & distortion
    GAUSSIAN_NOISE = "gaussian_noise"
    GAUSSIAN_BLUR = "gaussian_blur"
    CUTOUT = "cutout"

# POLICY DEFINITION: Lists of [operation, probability, magnitude] tuples
IMAGENET_POLICY = [
    [[(Operation.POSTERIZE, 0.4, 8), (Operation.ROTATE, 0.6, 9)],
     [(Operation.SOLARIZE, 0.6, 5), (Operation.AUTOCONTRAST, 0.6, 0)]],
    [[(Operation.EQUALIZE, 0.8, 0), (Operation.EQUALIZE, 0.6, 0)],
     [(Operation.POSTERIZE, 0.6, 7), (Operation.POSTERIZE, 0.6, 6)]],
    [[(Operation.EQUALIZE, 0.4, 0), (Operation.SOLARIZE, 0.2, 4)],
     [(Operation.EQUALIZE, 0.4, 0), (Operation.ROTATE, 0.8, 8)]],
    [[(Operation.SOLARIZE, 0.6, 3), (Operation.EQUALIZE, 0.6, 0)],
     [(Operation.POSTERIZE, 0.8, 5), (Operation.EQUALIZE, 1.0, 0)]],
    [[(Operation.ROTATE, 0.2, 3), (Operation.SOLARIZE, 0.6, 8)],
     [(Operation.EQUALIZE, 0.6, 0), (Operation.POSTERIZE, 0.4, 6)]]
]

CIFAR10_POLICY = [
    [[(Operation.INVERT, 0.1, 0), (Operation.CONTRAST, 0.2, 6)],
     [(Operation.ROTATE, 0.7, 2), (Operation.TRANSLATE_X, 0.3, 9)]],
    [[(Operation.SHARPNESS, 0.8, 1), (Operation.SHARPNESS, 0.9, 3)],
     [(Operation.BRIGHTNESS, 0.4, 7), (Operation.COLOR, 0.6, 5)]],
    [[(Operation.EQUALIZE, 0.6, 0), (Operation.EQUALIZE, 0.5, 0)],
     [(Operation.CONTRAST, 0.6, 7), (Operation.SHARPNESS, 0.6, 5)]],
    [[(Operation.BRIGHTNESS, 0.3, 7), (Operation.COLOR, 0.5, 0)],
     [(Operation.SOLARIZE, 0.2, 0), (Operation.INVERT, 0.4, 0)]]
]

# Custom plant policy with operations suitable for plant images
PLANT_POLICY = [
    [[(Operation.ROTATE, 0.7, 10), (Operation.COLOR, 0.5, 5)],
     [(Operation.BRIGHTNESS, 0.5, 5), (Operation.CONTRAST, 0.5, 5)]],
    [[(Operation.BRIGHTNESS, 0.8, 4), (Operation.SHARPNESS, 0.7, 7)],
     [(Operation.TRANSLATE_X, 0.4, 8), (Operation.TRANSLATE_Y, 0.4, 8)]],
    [[(Operation.ROTATE, 0.5, 6), (Operation.EQUALIZE, 0.9, 0)],
     [(Operation.BRIGHTNESS, 0.3, 4), (Operation.SHEAR_X, 0.5, 4)]],
    [[(Operation.SHARPNESS, 0.7, 6), (Operation.AUTOCONTRAST, 0.9, 0)],
     [(Operation.CONTRAST, 0.6, 8), (Operation.BRIGHTNESS, 0.5, 5)]],
    [[(Operation.EQUALIZE, 0.8, 0), (Operation.SHEAR_Y, 0.4, 6)],
     [(Operation.COLOR, 0.5, 7), (Operation.ROTATE, 0.5, 8)]]
]

# Dictionary of available policies
POLICIES = {
    "imagenet": IMAGENET_POLICY,
    "cifar10": CIFAR10_POLICY,
    "plant": PLANT_POLICY
}

class AdvancedAugmentation:
    """
    Implements RandAugment and AutoAugment for plant images.
    
    RandAugment: Randomly selects N operations and applies them with the same magnitude.
    AutoAugment: Uses predefined policies optimized for specific datasets.
    """
    
    def __init__(self, method="randaugment", policy_name="plant", n_ops=DEFAULT_N_OPS, 
                 magnitude=DEFAULT_MAGNITUDE, magnitude_std=0, augmentation_probability=0.5):
        """
        Initialize the augmentation engine.
        
        Args:
            method: "randaugment" or "autoaugment"
            policy_name: Name of the policy to use for AutoAugment
            n_ops: Number of operations to apply (for RandAugment)
            magnitude: Strength of the augmentations (0-30)
            magnitude_std: Standard deviation of magnitude (for randomization)
            augmentation_probability: Probability of applying augmentation to an image
        """
        self.method = method.lower()
        self.policy_name = policy_name
        self.n_ops = n_ops
        self.magnitude = magnitude
        self.magnitude_std = magnitude_std
        self.augmentation_probability = augmentation_probability
        
        # Get policy for AutoAugment
        if self.method == "autoaugment":
            if policy_name not in POLICIES:
                print(f"{TermColors.YELLOW}⚠️ Policy '{policy_name}' not found. Using 'plant' policy.{TermColors.ENDC}")
                self.policy_name = "plant"
            self.policy = POLICIES[self.policy_name]
        
        print(f"{TermColors.GREEN}✅ {self.method.capitalize()} initialized with magnitude={magnitude}.{TermColors.ENDC}")
        
    def _apply_operation(self, image, op_name, magnitude):
        """Apply a single operation to an image"""
        if op_name == Operation.ROTATE:
            max_angle = 30.0
            angle = magnitude * max_angle / 30.0
            angle = tf.random.uniform([], -angle, angle, dtype=tf.float32)
            return tfa.image.rotate(image, angle * np.pi / 180.0)
            
        elif op_name == Operation.TRANSLATE_X:
            max_pixels = 0.3 * IMAGE_SIZE[1]  # 30% of width
            pixels = magnitude * max_pixels / 30.0
            pixels = tf.random.uniform([], -pixels, pixels, dtype=tf.int32)
            return tfa.image.translate(image, [pixels, 0])
            
        elif op_name == Operation.TRANSLATE_Y:
            max_pixels = 0.3 * IMAGE_SIZE[0]  # 30% of height
            pixels = magnitude * max_pixels / 30.0
            pixels = tf.random.uniform([], -pixels, pixels, dtype=tf.int32)
            return tfa.image.translate(image, [0, pixels])
            
        elif op_name == Operation.SHEAR_X:
            max_shear = 0.3  # 30% shear
            level = magnitude * max_shear / 30.0
            level = tf.random.uniform([], -level, level, dtype=tf.float32)
            return tfa.image.shear_x(image, level)
            
        elif op_name == Operation.SHEAR_Y:
            max_shear = 0.3  # 30% shear
            level = magnitude * max_shear / 30.0
            level = tf.random.uniform([], -level, level, dtype=tf.float32)
            return tfa.image.shear_y(image, level)
            
        elif op_name == Operation.FLIP_LR:
            return tf.image.flip_left_right(image)
            
        elif op_name == Operation.FLIP_UD:
            return tf.image.flip_up_down(image)
            
        elif op_name == Operation.COLOR:
            max_factor = 1.8
            factor = 1.0 + (magnitude * (max_factor - 1.0) / 30.0)
            factor = tf.random.uniform([], 1.0/factor, factor, dtype=tf.float32)
            return tf.image.adjust_saturation(image, factor)
            
        elif op_name == Operation.CONTRAST:
            max_factor = 1.8
            factor = 1.0 + (magnitude * (max_factor - 1.0) / 30.0)
            factor = tf.random.uniform([], 1.0/factor, factor, dtype=tf.float32)
            return tf.image.adjust_contrast(image, factor)
            
        elif op_name == Operation.BRIGHTNESS:
            max_factor = 0.3
            factor = magnitude * max_factor / 30.0
            factor = tf.random.uniform([], -factor, factor, dtype=tf.float32)
            return tf.image.adjust_brightness(image, factor)
            
        elif op_name == Operation.SHARPNESS:
            max_factor = 1.8
            factor = 1.0 + (magnitude * (max_factor - 1.0) / 30.0)
            factor = tf.random.uniform([], 1.0/factor, factor, dtype=tf.float32)
            return tfa.image.sharpness(image, factor)
            
        elif op_name == Operation.POSTERIZE:
            max_bit_reduction = 6  # Reduce to as low as 2 bits
            bits = 8 - (magnitude * max_bit_reduction / 30.0)
            bits = tf.maximum(tf.cast(bits, tf.int32), 1)
            return tf.cast(tf.cast(image * 255.0, tf.int32) & (255 - (1 << (8 - bits)) + 1), tf.float32) / 255.0
            
        elif op_name == Operation.SOLARIZE:
            max_threshold = 256
            threshold = max_threshold - (magnitude * max_threshold / 30.0)
            threshold = tf.cast(threshold, tf.int32)
            return tf.where(tf.cast(image * 255.0, tf.int32) < threshold, image, 1.0 - image)
            
        elif op_name == Operation.INVERT:
            return 1.0 - image
            
        elif op_name == Operation.EQUALIZE:
            return tfa.image.equalize(tf.cast(image * 255.0, tf.uint8)) / 255.0
            
        elif op_name == Operation.AUTOCONTRAST:
            return tfa.image.autocontrast(tf.cast(image * 255.0, tf.uint8)) / 255.0
            
        elif op_name == Operation.GAUSSIAN_NOISE:
            max_stddev = 0.2
            stddev = magnitude * max_stddev / 30.0
            noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=stddev)
            return tf.clip_by_value(image + noise, 0.0, 1.0)
            
        elif op_name == Operation.GAUSSIAN_BLUR:
            max_sigma = 3.0
            sigma = magnitude * max_sigma / 30.0
            size = tf.cast(sigma * 4.0 + 0.5, tf.int32) * 2 + 1  # Ensure odd size
            kernel_size = tf.maximum(size, 3)
            return tfa.image.gaussian_filter2d(image, filter_shape=[kernel_size, kernel_size], sigma=[sigma, sigma])
            
        elif op_name == Operation.CUTOUT:
            max_size = 0.5  # 50% of image size
            cut_size = tf.cast(magnitude * max_size * tf.minimum(IMAGE_SIZE[0], IMAGE_SIZE[1]) / 30.0, tf.int32)
            cut_size = tf.maximum(cut_size, 1)
            
            h, w = IMAGE_SIZE
            mask = tf.ones((h, w, tf.shape(image)[-1]))
            
            # Random position
            y = tf.random.uniform([], 0, h - cut_size + 1, dtype=tf.int32)
            x = tf.random.uniform([], 0, w - cut_size + 1, dtype=tf.int32)
            
            # Create mask
            mask_value = tf.zeros((cut_size, cut_size, tf.shape(image)[-1]))
            mask = tf.tensor_scatter_nd_update(
                mask, 
                tf.expand_dims(tf.stack([y, x], axis=0), 0), 
                tf.expand_dims(mask_value, 0)
            )
            
            return image * mask
            
        else:
            print(f"{TermColors.YELLOW}⚠️ Unknown operation: {op_name}. Returning original image.{TermColors.ENDC}")
            return image
    
    def apply_randaugment(self, image):
        """Apply RandAugment to an image"""
        # Get all available operations
        operations = [op for op in Operation]
        # Randomly select n_ops operations
        if isinstance(self.n_ops, int):
            n_to_apply = self.n_ops
        else:
            # If n_ops is a tuple (min, max), randomly select between min and max
            n_to_apply = tf.random.uniform([], self.n_ops[0], self.n_ops[1] + 1, dtype=tf.int32)
            
        selected_ops = random.sample(operations, k=n_to_apply)
        
        # Apply operations sequentially
        for op in selected_ops:
            # Randomize magnitude if std > 0
            if self.magnitude_std > 0:
                mag = tf.random.normal([], mean=self.magnitude, stddev=self.magnitude_std)
                mag = tf.clip_by_value(mag, 0, 30)
            else:
                mag = self.magnitude
                
            # Apply operation
            image = self._apply_operation(image, op, mag)
            
        return image
    
    def apply_autoaugment(self, image):
        """Apply AutoAugment to an image"""
        # Choose a random policy sub-policy
        policy_idx = random.randint(0, len(self.policy) - 1)
        sub_policy_idx = random.randint(0, 1)
        sub_policy = self.policy[policy_idx][sub_policy_idx]
        
        # Apply operations from the sub-policy
        for op, prob, mag in sub_policy:
            if random.random() < prob:
                image = self._apply_operation(image, op, mag)
                
        return image
        
    def augment(self, image):
        """Apply augmentation to an image"""
        # Skip augmentation with 1-p probability
        if random.random() > self.augmentation_probability:
            return image
            
        # Choose the appropriate augmentation method
        if self.method == "randaugment":
            return self.apply_randaugment(image)
        elif self.method == "autoaugment":
            return self.apply_autoaugment(image)
        else:
            print(f"{TermColors.YELLOW}⚠️ Unknown method: {self.method}. Returning original image.{TermColors.ENDC}")
            return image
    
    def augment_batch(self, images):
        """Apply augmentation to a batch of images"""
        return tf.map_fn(self.augment, images)
    
    @tf.function
    def tf_augment(self, image, label):
        """TensorFlow function for use in tf.data pipeline"""
        # Convert any non-float image to float32 in [0, 1]
        if image.dtype != tf.float32:
            image = tf.cast(image, tf.float32)
            if tf.reduce_max(image) > 1.0:
                image = image / 255.0
                
        # Resize image if needed
        if image.shape[0] != IMAGE_SIZE[0] or image.shape[1] != IMAGE_SIZE[1]:
            image = tf.image.resize(image, IMAGE_SIZE)
            
        # Apply augmentation
        if self.method == "randaugment":
            # For RandAugment, apply operations randomly
            if random.random() < self.augmentation_probability:
                operations = [op for op in Operation]
                n_to_apply = self.n_ops if isinstance(self.n_ops, int) else \
                    tf.random.uniform([], self.n_ops[0], self.n_ops[1] + 1, dtype=tf.int32)
                
                # TensorFlow doesn't support random.sample in graph mode
                # Instead, we'll shuffle and take the first n
                indices = tf.range(len(operations))
                indices = tf.random.shuffle(indices)[:n_to_apply]
                
                for i in indices:
                    op = operations[tf.as_int32(i)]
                    
                    # Randomize magnitude if std > 0
                    if self.magnitude_std > 0:
                        mag = tf.random.normal([], mean=self.magnitude, stddev=self.magnitude_std)
                        mag = tf.clip_by_value(mag, 0, 30)
                    else:
                        mag = self.magnitude
                        
                    # Apply operation with 50% chance (simplification for tf.function)
                    if tf.random.uniform([]) < 0.5:
                        image = self._apply_operation(image, op, mag)
        
        elif self.method == "autoaugment" and random.random() < self.augmentation_probability:
            # For AutoAugment, this is harder to implement in tf.function
            # We'll use a simpler approach with fixed operations for demonstration
            policy_idx = tf.random.uniform([], 0, len(self.policy), dtype=tf.int32)
            sub_policy_idx = tf.random.uniform([], 0, 2, dtype=tf.int32)
            
            # Apply first operation with 80% chance (simplified)
            if tf.random.uniform([]) < 0.8:
                op, _, mag = self.policy[policy_idx][sub_policy_idx][0]
                image = self._apply_operation(image, op, mag)
                
            # Apply second operation with 80% chance (simplified)
            if tf.random.uniform([]) < 0.8:
                op, _, mag = self.policy[policy_idx][sub_policy_idx][1]
                image = self._apply_operation(image, op, mag)
                
        return image, label


def create_augmented_dataset(dataset, augmentation, batch_size=32, prefetch=tf.data.AUTOTUNE):
    """Create a dataset with augmentation applied"""
    return dataset.map(
        augmentation.tf_augment, 
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(batch_size).prefetch(prefetch)


def demo_augmentations(image_path, save_dir="augmentation_examples"):
    """
    Demonstrate augmentations on a single image and save results.
    
    Args:
        image_path: Path to the image
        save_dir: Directory to save augmented images
    """
    import matplotlib.pyplot as plt
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    
    # Create augmentation engines
    randaug_low = AdvancedAugmentation(method="randaugment", n_ops=1, magnitude=10)
    randaug_high = AdvancedAugmentation(method="randaugment", n_ops=2, magnitude=20)
    autoaug_plant = AdvancedAugmentation(method="autoaugment", policy_name="plant")
    autoaug_imagenet = AdvancedAugmentation(method="autoaugment", policy_name="imagenet")
    
    # Apply augmentations
    n_examples = 4  # Number of examples per method
    
    # Create figure
    plt.figure(figsize=(15, 12))
    
    # Plot original image
    plt.subplot(5, n_examples, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis("off")
    
    for i in range(1, n_examples):
        plt.subplot(5, n_examples, i + 1)
        plt.imshow(img)
        plt.title("Original")
        plt.axis("off")
    
    # RandAugment (Low)
    set_seeds(42)  # For reproducibility
    for i in range(n_examples):
        plt.subplot(5, n_examples, n_examples + i + 1)
        augmented = randaug_low.augment(img)
        plt.imshow(augmented)
        plt.title(f"RandAug Low {i+1}")
        plt.axis("off")
    
    # RandAugment (High)
    set_seeds(43)
    for i in range(n_examples):
        plt.subplot(5, n_examples, 2 * n_examples + i + 1)
        augmented = randaug_high.augment(img)
        plt.imshow(augmented)
        plt.title(f"RandAug High {i+1}")
        plt.axis("off")
    
    # AutoAugment (Plant)
    set_seeds(44)
    for i in range(n_examples):
        plt.subplot(5, n_examples, 3 * n_examples + i + 1)
        augmented = autoaug_plant.augment(img)
        plt.imshow(augmented)
        plt.title(f"AutoAug Plant {i+1}")
        plt.axis("off")
    
    # AutoAugment (ImageNet)
    set_seeds(45)
    for i in range(n_examples):
        plt.subplot(5, n_examples, 4 * n_examples + i + 1)
        augmented = autoaug_imagenet.augment(img)
        plt.imshow(augmented)
        plt.title(f"AutoAug ImgNet {i+1}")
        plt.axis("off")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "augmentation_comparison.png"), dpi=200)
    print(f"{TermColors.GREEN}✅ Saved augmentation comparison to {save_dir}/augmentation_comparison.png{TermColors.ENDC}")
    
    # Save individual examples
    plt.figure(figsize=(12, 12))
    set_seeds(46)
    
    # Save original
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "original.png"))
    
    # Save 9 examples of RandAugment (medium)
    randaug_med = AdvancedAugmentation(method="randaugment", n_ops=2, magnitude=15)
    for i in range(9):
        plt.clf()
        augmented = randaug_med.augment(img)
        plt.imshow(augmented)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"randaug_example_{i+1}.png"))
    
    # Save 9 examples of AutoAugment (plant)
    for i in range(9):
        plt.clf()
        augmented = autoaug_plant.augment(img)
        plt.imshow(augmented)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"autoaug_example_{i+1}.png"))
    
    print(f"{TermColors.GREEN}✅ Saved individual augmentation examples to {save_dir}/{TermColors.ENDC}")


if __name__ == "__main__":
    print(f"{TermColors.HEADER}\n{'='*50}\nADVANCED AUGMENTATION MODULE\n{'='*50}{TermColors.ENDC}")
    
    if len(sys.argv) > 1:
        # If an image path is provided, run demo
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            print(f"{TermColors.CYAN}ℹ Running augmentation demo on {image_path}...{TermColors.ENDC}")
            demo_augmentations(image_path)
        else:
            print(f"{TermColors.RED}❌ Image not found: {image_path}{TermColors.ENDC}")
    else:
        # Otherwise, show usage info
        print(f"{TermColors.CYAN}Usage: python {os.path.basename(__file__)} <image_path>{TermColors.ENDC}")
        print(f"{TermColors.CYAN}Example: python {os.path.basename(__file__)} plant_image.jpg{TermColors.ENDC}")
        print(f"{TermColors.YELLOW}No image provided. To use this module in your code, import it as follows:{TermColors.ENDC}")
        print("\n```python")
        print("from advanced_augmentation import AdvancedAugmentation")
        print("")
        print("# Create augmentation engine")
        print("augmentation = AdvancedAugmentation(")
        print("    method='randaugment',  # or 'autoaugment'")
        print("    n_ops=2,               # Number of operations to apply in RandAugment")
        print("    magnitude=10,          # Strength of augmentation (0-30)")
        print("    magnitude_std=5,       # Standard deviation for random magnitude")
        print("    augmentation_probability=0.8  # Probability to apply augmentation")
        print(")")
        print("")
        print("# Apply to an image")
        print("augmented_image = augmentation.augment(image)")
        print("")
        print("# Or use in a tf.data pipeline")
        print("dataset = dataset.map(augmentation.tf_augment)")
        print("```")