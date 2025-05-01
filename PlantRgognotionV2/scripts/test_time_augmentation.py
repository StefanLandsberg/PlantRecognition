import os
import sys
import numpy as np
import tensorflow as tf
import json
import time
import gc
import traceback
import warnings
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from scripts.PlantRecognitionV2 import get_preprocessing_function, set_seeds, cleanup_memory, TermColors

# Suppress TensorFlow/System Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

# --- Constants ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
TTA_NUM_AUGMENTATIONS = 10
AUTOTUNE = tf.data.AUTOTUNE

# --- TTA Settings ---
TTA_FLIP = True
TTA_ROTATE = True
TTA_ZOOM = True
TTA_CONTRAST = True
TTA_BRIGHTNESS = True
TTA_SHIFT = True
TTA_BATCH_SIZE = 8  # Lower if memory issues
MAX_ANGLE = 15.0  # Degrees
ZOOM_RANGE = 0.1
CONTRAST_RANGE = 0.1
BRIGHTNESS_RANGE = 0.1
SHIFT_RANGE = 0.1

class TestTimeAugmentation:
    """
    Implements Test-Time Augmentation (TTA) for plant classification models.
    TTA creates multiple augmented versions of each test image and averages predictions.
    """
    
    def __init__(self, peft_model, chunk_models, class_mappings, model_name="DenseNet121", 
                 tta_count=TTA_NUM_AUGMENTATIONS, batch_size=TTA_BATCH_SIZE, 
                 use_all_features=True):
        """
        Initialize the TTA engine.
        
        Args:
            peft_model: The feature extraction model
            chunk_models: List of chunk classifiers
            class_mappings: List of class mappings for each chunk
            model_name: Name of the base model for preprocessing
            tta_count: Number of augmented versions to create
            batch_size: Prediction batch size
            use_all_features: Extract features from all augmentations or just one
        """
        set_seeds()
        self.peft_model = peft_model
        self.chunk_models = chunk_models
        self.class_mappings = class_mappings
        self.model_name = model_name
        self.tta_count = tta_count
        self.batch_size = batch_size
        self.use_all_features = use_all_features
        
        # Create feature extractor model from PEFT model
        try:
            # Find the layer before the final Dense output layer
            feature_layer = peft_model.layers[-2].output  
            if isinstance(peft_model.layers[-2], tf.keras.layers.Dropout):
                feature_layer = peft_model.layers[-3].output  # Handle Dropout case
                
            self.feature_extractor = tf.keras.Model(inputs=peft_model.input, outputs=feature_layer)
            self.feature_extractor.trainable = False
            print(f"{TermColors.GREEN}✅ Feature extractor model created for TTA.{TermColors.ENDC}")
        except Exception as e:
            print(f"{TermColors.RED}❌ Error creating feature extractor model: {e}{TermColors.ENDC}")
            raise
        
        # Get preprocessing function
        self.preprocess_input = get_preprocessing_function(model_name)
        if self.preprocess_input is None:
            raise ValueError(f"Preprocessing function not found for {model_name}")

    def create_augmentations(self, image):
        """Create multiple augmented versions of a single image"""
        augmented_images = []
        
        # Always include original image (preprocessed)
        img = tf.cast(image, tf.float32)
        img = self.preprocess_input(img)
        augmented_images.append(img)
        
        # Generate different augmentations
        for i in range(self.tta_count - 1):
            img = tf.cast(image, tf.float32)  # Start with original
            
            # TTA flipping
            if TTA_FLIP and np.random.random() > 0.5:
                img = tf.image.flip_left_right(img)
                
            # TTA rotation
            if TTA_ROTATE and np.random.random() > 0.5:
                angle = np.random.uniform(-MAX_ANGLE, MAX_ANGLE) * np.pi / 180.0
                img = tfa.image.rotate(img, angle)
                
            # TTA zoom
            if TTA_ZOOM and np.random.random() > 0.5:
                zoom_factor = np.random.uniform(1-ZOOM_RANGE, 1+ZOOM_RANGE)
                orig_height, orig_width = tf.shape(img)[0], tf.shape(img)[1]
                crop_height, crop_width = tf.cast(orig_height/zoom_factor, tf.int32), tf.cast(orig_width/zoom_factor, tf.int32)
                
                # Ensure crop dimensions don't exceed image dimensions
                crop_height = tf.minimum(crop_height, orig_height)
                crop_width = tf.minimum(crop_width, orig_width)
                
                crop_y = tf.random.uniform(shape=[], maxval=orig_height-crop_height+1, dtype=tf.int32)
                crop_x = tf.random.uniform(shape=[], maxval=orig_width-crop_width+1, dtype=tf.int32)
                
                img = tf.image.crop_to_bounding_box(img, crop_y, crop_x, crop_height, crop_width)
                img = tf.image.resize(img, (orig_height, orig_width))
            
            # TTA contrast
            if TTA_CONTRAST and np.random.random() > 0.5:
                contrast_factor = np.random.uniform(1-CONTRAST_RANGE, 1+CONTRAST_RANGE)
                img = tf.image.adjust_contrast(img, contrast_factor)
                
            # TTA brightness
            if TTA_BRIGHTNESS and np.random.random() > 0.5:
                brightness_delta = np.random.uniform(-BRIGHTNESS_RANGE, BRIGHTNESS_RANGE)
                img = tf.image.adjust_brightness(img, brightness_delta)
                
            # TTA translation
            if TTA_SHIFT and np.random.random() > 0.5:
                shift_height = int(IMAGE_SIZE[0] * np.random.uniform(-SHIFT_RANGE, SHIFT_RANGE))
                shift_width = int(IMAGE_SIZE[1] * np.random.uniform(-SHIFT_RANGE, SHIFT_RANGE))
                translations = [shift_height, shift_width, 0]  # [height, width, channels]
                img = tfa.image.translate(img, translations, fill_mode='nearest')
                
            # Apply normalization (model-specific preprocessing)
            img = self.preprocess_input(img)
            augmented_images.append(img)
            
        # Stack all augmentations
        return tf.stack(augmented_images)
    
    def predict_with_tta(self, image_path, top_k=5):
        """
        Predict a single image with TTA.
        
        Args:
            image_path: Path to the image
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with top_k predictions and probabilities
        """
        start_time = time.time()
        
        # Load and resize the image
        try:
            img = tf.io.read_file(image_path)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.resize(img, IMAGE_SIZE)
            
            # Generate augmented versions
            print(f"{TermColors.CYAN}ℹ Creating {self.tta_count} augmented versions for TTA...{TermColors.ENDC}")
            augmented_batch = self.create_augmentations(img)
            
            # Extract features from all augmentations
            if self.use_all_features:
                print(f"{TermColors.CYAN}ℹ Extracting features from all augmentations...{TermColors.ENDC}")
                features_batch = self.feature_extractor.predict(augmented_batch, batch_size=self.batch_size, verbose=0)
            else:
                # Use only the original image for feature extraction (still use TTA for the chunk models)
                print(f"{TermColors.CYAN}ℹ Extracting features from original image only...{TermColors.ENDC}")
                features = self.feature_extractor.predict(tf.expand_dims(augmented_batch[0], 0), verbose=0)
                features_batch = np.repeat(features, self.tta_count, axis=0)
            
            # Get predictions from all chunk models
            all_predictions = []
            all_class_names = []
            
            for chunk_idx, chunk_model in enumerate(self.chunk_models):
                # Predict on all augmented features
                chunk_pred_probs_batch = chunk_model.predict(
                    features_batch, batch_size=self.batch_size, verbose=0
                )
                
                # Average predictions across augmentations
                chunk_pred_probs_avg = np.mean(chunk_pred_probs_batch, axis=0)
                
                # Get top predictions for this chunk
                top_indices = np.argsort(chunk_pred_probs_avg)[::-1][:top_k]
                top_probs = chunk_pred_probs_avg[top_indices]
                
                # Map indices to class names
                mapping = self.class_mappings[chunk_idx]
                top_classes = [mapping.get(str(idx), f"Unknown-{idx}") for idx in top_indices]
                
                # Store predictions
                chunk_preds = [
                    {"class_name": class_name, "probability": float(prob)}
                    for class_name, prob in zip(top_classes, top_probs)
                ]
                all_predictions.append(chunk_preds)
                all_class_names.extend(top_classes)
            
            # Merge predictions from all chunks (taking the highest probability for each class)
            class_to_max_prob = {}
            for chunk_preds in all_predictions:
                for pred in chunk_preds:
                    class_name = pred["class_name"]
                    probability = pred["probability"]
                    if class_name not in class_to_max_prob or probability > class_to_max_prob[class_name]:
                        class_to_max_prob[class_name] = probability
            
            # Sort by probability and get top-k
            final_predictions = [
                {"class_name": class_name, "probability": prob}
                for class_name, prob in sorted(class_to_max_prob.items(), key=lambda x: x[1], reverse=True)[:top_k]
            ]
            
            print(f"{TermColors.GREEN}✅ TTA prediction completed in {time.time() - start_time:.2f}s{TermColors.ENDC}")
            return final_predictions
            
        except Exception as e:
            print(f"{TermColors.RED}❌ Error in TTA prediction: {e}{TermColors.ENDC}")
            traceback.print_exc()
            return [{"error": str(e)}]
    
    def predict_batch_with_tta(self, image_paths, top_k=5):
        """
        Predict a batch of images with TTA.
        
        Args:
            image_paths: List of paths to images
            top_k: Number of top predictions to return
            
        Returns:
            List of dictionaries with top_k predictions and probabilities
        """
        results = []
        for path in tqdm(image_paths, desc="TTA Predictions"):
            result = self.predict_with_tta(path, top_k)
            results.append(result)
            cleanup_memory()  # Free memory after each image
        return results


def load_models(peft_model_path, chunk_model_dir):
    """
    Load all models needed for TTA.
    
    Args:
        peft_model_path: Path to the PEFT model
        chunk_model_dir: Directory containing chunk models
        
    Returns:
        Tuple of (peft_model, chunk_models, class_mappings)
    """
    try:
        # Load PEFT model
        print(f"{TermColors.CYAN}ℹ Loading PEFT model from {peft_model_path}...{TermColors.ENDC}")
        peft_model = tf.keras.models.load_model(peft_model_path, compile=False)
        print(f"{TermColors.GREEN}✅ PEFT model loaded successfully.{TermColors.ENDC}")
        
        # Load all chunk models
        print(f"{TermColors.CYAN}ℹ Looking for chunk models in {chunk_model_dir}...{TermColors.ENDC}")
        chunk_model_paths = []
        mapping_paths = []
        
        # Find chunk models and their mappings
        chunk_models = []
        class_mappings = []
        
        # Search for model files with 'chunk' in the name
        model_files = sorted([f for f in os.listdir(chunk_model_dir) 
                             if os.path.isfile(os.path.join(chunk_model_dir, f)) 
                             and 'chunk' in f and f.endswith('.keras')])
        
        for model_file in model_files:
            chunk_idx = int(model_file.split('_')[1])  # Extract chunk index from filename
            model_path = os.path.join(chunk_model_dir, model_file)
            
            # Find corresponding mapping file
            features_dir = os.path.join(os.path.dirname(os.path.dirname(chunk_model_dir)), 
                                      "data", "features", f"chunk_{chunk_idx}")
            mapping_path = os.path.join(features_dir, "class_mapping.json")
            
            if not os.path.exists(mapping_path):
                print(f"{TermColors.YELLOW}⚠️ Mapping file not found for {model_file}, skipping.{TermColors.ENDC}")
                continue
                
            # Load chunk model
            print(f"{TermColors.CYAN}ℹ Loading chunk model from {model_path}...{TermColors.ENDC}")
            chunk_model = tf.keras.models.load_model(model_path, compile=False)
            
            # Load mapping
            with open(mapping_path, 'r') as f:
                class_mapping = json.load(f)
                
            chunk_models.append(chunk_model)
            class_mappings.append(class_mapping)
            print(f"{TermColors.GREEN}✅ Loaded chunk model {chunk_idx} with {len(class_mapping)} classes.{TermColors.ENDC}")
            
        if not chunk_models:
            raise ValueError("No chunk models found!")
            
        return peft_model, chunk_models, class_mappings
        
    except Exception as e:
        print(f"{TermColors.RED}❌ Error loading models: {e}{TermColors.ENDC}")
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    # Find all required packages for TTA
    try:
        import tensorflow_addons as tfa
    except ImportError:
        print(f"{TermColors.RED}❌ tensorflow-addons package required for TTA. Install with 'pip install tensorflow-addons'{TermColors.ENDC}")
        sys.exit(1)
        
    # Example usage
    print(f"{TermColors.HEADER}\n{'='*50}\nTEST-TIME AUGMENTATION (TTA) MODULE\n{'='*50}{TermColors.ENDC}")
    
    # Check command line arguments
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        model_path = sys.argv[2] if len(sys.argv) > 2 else "checkpoints_v2/DenseNet121_peft_best.keras"
        chunk_model_dir = sys.argv[3] if len(sys.argv) > 3 else "models_v2"
        
        if not os.path.exists(image_path):
            print(f"{TermColors.RED}❌ Image file not found: {image_path}{TermColors.ENDC}")
            sys.exit(1)
            
        # Load models
        peft_model, chunk_models, class_mappings = load_models(model_path, chunk_model_dir)
        if peft_model is None:
            sys.exit(1)
            
        # Create TTA predictor
        tta = TestTimeAugmentation(peft_model, chunk_models, class_mappings)
        
        # Run prediction
        print(f"{TermColors.CYAN}ℹ Running TTA prediction on {image_path}...{TermColors.ENDC}")
        predictions = tta.predict_with_tta(image_path)
        
        # Print results
        print(f"\n{TermColors.HEADER}Top predictions:{TermColors.ENDC}")
        for i, pred in enumerate(predictions):
            print(f"{i+1}. {pred['class_name']}: {pred['probability']:.4f}")
            
    else:
        print(f"{TermColors.YELLOW}⚠️ No image provided for inference.{TermColors.ENDC}")
        print(f"Usage: python {os.path.basename(__file__)} <image_path> [<peft_model_path>] [<chunk_model_dir>]")
        print(f"Example: python {os.path.basename(__file__)} test_image.jpg checkpoints_v2/DenseNet121_peft_best.keras models_v2")