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
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from scripts.PlantRecognitionV2 import get_preprocessing_function, set_seeds, cleanup_memory, TermColors

# Suppress TensorFlow/System Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

# --- Constants ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
TOP_K = 5
ENSEMBLE_DIR = os.path.join(Path(__file__).parent.parent, "models_v2", "ensemble")
os.makedirs(ENSEMBLE_DIR, exist_ok=True)

# --- Ensemble Method Types ---
class EnsembleMethod:
    AVERAGING = "averaging"  # Simple average of probabilities
    WEIGHTED_AVERAGING = "weighted_averaging"  # Weighted average based on validation performance
    STACKING = "stacking"  # Use a meta-model to combine predictions
    VOTING = "voting"  # Hard voting (majority wins)
    BAGGING = "bagging"  # Train same architecture on different data subsets
    BOOSTING = "boosting"  # Train sequentially, focus more on previously misclassified samples

class EnsembleEngine:
    """
    Implements ensemble methods for plant classification models.
    Combines predictions from multiple models to improve accuracy.
    """
    
    def __init__(self, feature_extractor, models_list, model_weights=None, 
                 class_mappings=None, method=EnsembleMethod.AVERAGING, 
                 meta_model=None, model_name="DenseNet121"):
        """
        Initialize the ensemble engine.
        
        Args:
            feature_extractor: The feature extraction model
            models_list: List of classifiers (one per chunk, or multiple per chunk)
            model_weights: Optional weights for each model in weighted averaging
            class_mappings: List of class mappings for each model
            method: Ensemble method to use
            meta_model: Optional meta-model for stacking
            model_name: Name of the base model for preprocessing
        """
        set_seeds()
        self.feature_extractor = feature_extractor
        self.models_list = models_list
        self.model_weights = model_weights
        self.class_mappings = class_mappings
        self.method = method
        self.meta_model = meta_model
        self.model_name = model_name
        
        # Handle default weights (uniform)
        if self.model_weights is None and method == EnsembleMethod.WEIGHTED_AVERAGING:
            self.model_weights = np.ones(len(models_list)) / len(models_list)
                
        # Get preprocessing function
        self.preprocess_input = get_preprocessing_function(model_name)
        if self.preprocess_input is None:
            raise ValueError(f"Preprocessing function not found for {model_name}")
            
        print(f"{TermColors.GREEN}✅ Ensemble engine initialized with {len(models_list)} models using {method} method.{TermColors.ENDC}")
        
    def extract_features(self, image_path):
        """Extract features from a single image"""
        try:
            # Load and preprocess image
            img = tf.io.read_file(image_path)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.resize(img, IMAGE_SIZE)
            img = tf.cast(img, tf.float32)
            img = self.preprocess_input(img)
            img = tf.expand_dims(img, 0)  # Add batch dimension
            
            # Extract features
            features = self.feature_extractor.predict(img, verbose=0)
            return features
        except Exception as e:
            print(f"{TermColors.RED}❌ Error extracting features: {e}{TermColors.ENDC}")
            traceback.print_exc()
            return None
    
    def ensemble_predict(self, image_path, top_k=TOP_K):
        """
        Make a prediction using ensemble of models.
        
        Args:
            image_path: Path to the image
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with top_k predictions and probabilities
        """
        start_time = time.time()
        
        try:
            # Extract features
            features = self.extract_features(image_path)
            if features is None:
                return [{"error": "Feature extraction failed"}]
                
            # Get individual predictions from all models
            all_predictions = []
            all_class_probs = {}  # Maps class_name to list of probabilities from different models
            
            for model_idx, model in enumerate(self.models_list):
                # Get predictions from this model
                pred_probs = model.predict(features, verbose=0)[0]
                
                # Map to class names
                mapping = self.class_mappings[model_idx]
                
                for idx, prob in enumerate(pred_probs):
                    # Get class name from mapping
                    class_name = mapping.get(str(idx), f"Unknown-{idx}")
                    
                    # Store in all_class_probs dictionary
                    if class_name not in all_class_probs:
                        all_class_probs[class_name] = []
                    all_class_probs[class_name].append(prob)
            
            # Apply ensemble method
            final_class_probs = {}
            
            if self.method == EnsembleMethod.AVERAGING:
                # Simple averaging
                for class_name, probs in all_class_probs.items():
                    final_class_probs[class_name] = np.mean(probs)
                    
            elif self.method == EnsembleMethod.WEIGHTED_AVERAGING:
                # Weighted averaging
                for class_name, probs in all_class_probs.items():
                    # Ensure we use weights for the available predictions only
                    valid_indices = [i for i, p in enumerate(probs) if p is not None]
                    if not valid_indices:
                        continue
                        
                    weights = [self.model_weights[i] for i in valid_indices]
                    probs_to_use = [probs[i] for i in valid_indices]
                    
                    # Normalize weights
                    weights_sum = sum(weights)
                    if weights_sum > 0:
                        weights = [w / weights_sum for w in weights]
                        final_class_probs[class_name] = sum(w * p for w, p in zip(weights, probs_to_use))
                    else:
                        final_class_probs[class_name] = np.mean(probs_to_use)
                        
            elif self.method == EnsembleMethod.VOTING:
                # Hard voting (majority)
                class_votes = {}
                
                for model_idx, model in enumerate(self.models_list):
                    pred_probs = model.predict(features, verbose=0)[0]
                    mapping = self.class_mappings[model_idx]
                    
                    # Get the top prediction (highest probability)
                    top_class_idx = np.argmax(pred_probs)
                    top_class = mapping.get(str(top_class_idx), f"Unknown-{top_class_idx}")
                    
                    # Count votes
                    if top_class not in class_votes:
                        class_votes[top_class] = 0
                    class_votes[top_class] += 1
                
                # Convert votes to probabilities
                total_votes = len(self.models_list)
                for class_name, votes in class_votes.items():
                    final_class_probs[class_name] = votes / total_votes
                    
            elif self.method == EnsembleMethod.STACKING and self.meta_model is not None:
                # Stacking: use meta-model to combine predictions
                # First, collect all probabilities into a feature vector
                all_probs = []
                for model in self.models_list:
                    probs = model.predict(features, verbose=0)[0]
                    all_probs.extend(probs)
                
                # Use meta-model to get final predictions
                meta_input = np.array([all_probs])
                meta_probs = self.meta_model.predict(meta_input, verbose=0)[0]
                
                # Map meta-model outputs to class names
                meta_mapping = self.class_mappings[-1]  # Assuming last mapping is for meta-model
                for idx, prob in enumerate(meta_probs):
                    class_name = meta_mapping.get(str(idx), f"Unknown-{idx}")
                    final_class_probs[class_name] = prob
            
            else:
                # Default to averaging if method not implemented
                print(f"{TermColors.YELLOW}⚠️ Ensemble method {self.method} not fully implemented, using averaging.{TermColors.ENDC}")
                for class_name, probs in all_class_probs.items():
                    final_class_probs[class_name] = np.mean(probs)
            
            # Sort and return top-k predictions
            sorted_predictions = sorted(
                [{"class_name": class_name, "probability": float(prob)} 
                 for class_name, prob in final_class_probs.items()],
                key=lambda x: x["probability"],
                reverse=True
            )[:top_k]
            
            print(f"{TermColors.GREEN}✅ Ensemble prediction completed in {time.time() - start_time:.2f}s{TermColors.ENDC}")
            return sorted_predictions
            
        except Exception as e:
            print(f"{TermColors.RED}❌ Error in ensemble prediction: {e}{TermColors.ENDC}")
            traceback.print_exc()
            return [{"error": str(e)}]
    
    def predict_batch(self, image_paths, top_k=TOP_K):
        """
        Predict a batch of images using the ensemble.
        
        Args:
            image_paths: List of paths to images
            top_k: Number of top predictions to return
            
        Returns:
            List of dictionaries with top_k predictions and probabilities
        """
        results = []
        for path in tqdm(image_paths, desc="Ensemble Predictions"):
            result = self.ensemble_predict(path, top_k)
            results.append(result)
            cleanup_memory()  # Free memory after each image
        return results


def create_ensemble_from_models(model_paths, feature_extractor, class_mappings=None, 
                                method=EnsembleMethod.AVERAGING, validation_data=None):
    """
    Create an ensemble from multiple models.
    
    Args:
        model_paths: List of paths to model files
        feature_extractor: Feature extraction model
        class_mappings: Class mappings for each model
        method: Ensemble method to use
        validation_data: Optional validation data for computing model weights
        
    Returns:
        EnsembleEngine object
    """
    try:
        # Load models
        models_list = []
        for path in model_paths:
            print(f"{TermColors.CYAN}ℹ Loading model from {path}...{TermColors.ENDC}")
            model = tf.keras.models.load_model(path, compile=False)
            models_list.append(model)
        
        # Compute weights if using weighted averaging and validation data is provided
        model_weights = None
        if method == EnsembleMethod.WEIGHTED_AVERAGING and validation_data is not None:
            print(f"{TermColors.CYAN}ℹ Computing model weights based on validation data...{TermColors.ENDC}")
            
            # Validation data should be a tuple of (features, labels)
            val_features, val_labels = validation_data
            
            # Calculate accuracy for each model
            accuracies = []
            for model in models_list:
                predictions = model.predict(val_features, verbose=0)
                pred_classes = np.argmax(predictions, axis=1)
                accuracy = accuracy_score(val_labels, pred_classes)
                accuracies.append(accuracy)
                print(f"{TermColors.CYAN}   Model accuracy: {accuracy:.4f}{TermColors.ENDC}")
            
            # Convert accuracies to weights (higher accuracy = higher weight)
            model_weights = np.array(accuracies)
            # Normalize weights to sum to 1
            model_weights = model_weights / model_weights.sum()
            
            print(f"{TermColors.GREEN}✅ Model weights computed: {model_weights}{TermColors.ENDC}")
        
        # Create and return ensemble
        return EnsembleEngine(
            feature_extractor=feature_extractor,
            models_list=models_list,
            model_weights=model_weights,
            class_mappings=class_mappings,
            method=method
        )
    
    except Exception as e:
        print(f"{TermColors.RED}❌ Error creating ensemble: {e}{TermColors.ENDC}")
        traceback.print_exc()
        return None


def build_stacking_meta_model(train_features, train_labels, num_classes, input_shape):
    """
    Build a meta-model for stacking ensemble.
    
    Args:
        train_features: Features from base models' predictions
        train_labels: Ground truth labels
        num_classes: Number of output classes
        input_shape: Shape of the input features
        
    Returns:
        Trained meta-model
    """
    print(f"{TermColors.CYAN}ℹ Building stacking meta-model...{TermColors.ENDC}")
    
    # Simple MLP for stacking
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    print(f"{TermColors.CYAN}ℹ Training stacking meta-model...{TermColors.ENDC}")
    history = model.fit(
        train_features, train_labels,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True
            )
        ],
        verbose=1
    )
    
    # Save meta-model
    meta_model_path = os.path.join(ENSEMBLE_DIR, "stacking_meta_model.keras")
    model.save(meta_model_path)
    print(f"{TermColors.GREEN}✅ Stacking meta-model saved to {meta_model_path}{TermColors.ENDC}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(ENSEMBLE_DIR, "stacking_training_history.png"))
    
    return model


def load_models_for_ensemble(peft_model_path, chunk_model_dir, method=EnsembleMethod.AVERAGING):
    """
    Load all models needed for ensemble.
    
    Args:
        peft_model_path: Path to the PEFT model (feature extractor)
        chunk_model_dir: Directory containing chunk models
        method: Ensemble method to use
        
    Returns:
        Tuple of (feature_extractor, models_list, class_mappings)
    """
    try:
        # Load PEFT model for feature extraction
        print(f"{TermColors.CYAN}ℹ Loading PEFT model from {peft_model_path}...{TermColors.ENDC}")
        peft_model = tf.keras.models.load_model(peft_model_path, compile=False)
        
        # Create feature extractor from PEFT model
        feature_layer = peft_model.layers[-2].output
        if isinstance(peft_model.layers[-2], tf.keras.layers.Dropout):
            feature_layer = peft_model.layers[-3].output  # Handle Dropout case
            
        feature_extractor = tf.keras.Model(inputs=peft_model.input, outputs=feature_layer)
        feature_extractor.trainable = False
        print(f"{TermColors.GREEN}✅ Feature extractor created from PEFT model.{TermColors.ENDC}")
        
        # Find all chunk models
        print(f"{TermColors.CYAN}ℹ Looking for chunk models in {chunk_model_dir}...{TermColors.ENDC}")
        
        # For BAGGING or multiple architectures, we might have different model types
        if method == EnsembleMethod.BAGGING:
            # For bagging, look for models with "bagging" in the name
            model_pattern = "*bagging*.keras"
        else:
            # For other methods, look for regular chunk models
            model_pattern = "*chunk*model*.keras"
            
        # Find all matching model files
        model_files = sorted([f for f in os.listdir(chunk_model_dir) 
                             if os.path.isfile(os.path.join(chunk_model_dir, f)) 
                             and f.endswith('.keras') and ('chunk' in f or 'swa' in f)])
        
        if not model_files:
            raise ValueError(f"No suitable model files found in {chunk_model_dir}")
            
        # Load models and mappings
        models_list = []
        class_mappings = []
        
        for model_file in model_files:
            # For chunk models, extract the chunk index
            if 'chunk' in model_file:
                try:
                    chunk_idx = int(model_file.split('_')[1])
                except (IndexError, ValueError):
                    print(f"{TermColors.YELLOW}⚠️ Could not extract chunk index from {model_file}, skipping.{TermColors.ENDC}")
                    continue
                
                model_path = os.path.join(chunk_model_dir, model_file)
                
                # Find corresponding mapping file
                features_dir = os.path.join(os.path.dirname(os.path.dirname(chunk_model_dir)), 
                                          "data", "features", f"chunk_{chunk_idx}")
                mapping_path = os.path.join(features_dir, "class_mapping.json")
                
                if not os.path.exists(mapping_path):
                    print(f"{TermColors.YELLOW}⚠️ Mapping file not found for {model_file}, skipping.{TermColors.ENDC}")
                    continue
                    
                # Load model and mapping
                print(f"{TermColors.CYAN}ℹ Loading {model_file}...{TermColors.ENDC}")
                model = tf.keras.models.load_model(model_path, compile=False)
                with open(mapping_path, 'r') as f:
                    class_mapping = json.load(f)
                    
                models_list.append(model)
                class_mappings.append(class_mapping)
                print(f"{TermColors.GREEN}✅ Loaded {model_file} with {len(class_mapping)} classes.{TermColors.ENDC}")
        
        if not models_list:
            raise ValueError("No models loaded successfully!")
            
        return feature_extractor, models_list, class_mappings
        
    except Exception as e:
        print(f"{TermColors.RED}❌ Error loading models for ensemble: {e}{TermColors.ENDC}")
        traceback.print_exc()
        return None, None, None


def evaluate_ensemble(ensemble, test_data_dir, test_size=100, random_seed=42):
    """
    Evaluate the ensemble on a test dataset.
    
    Args:
        ensemble: EnsembleEngine object
        test_data_dir: Directory containing test images organized by class
        test_size: Maximum number of test images to use
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with evaluation metrics
    """
    try:
        import random
        random.seed(random_seed)
        
        # Find all image files in test directory (organized by class folders)
        image_paths = []
        true_labels = []
        
        # For each class folder
        for class_name in os.listdir(test_data_dir):
            class_dir = os.path.join(test_data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            # List image files
            img_files = [f for f in os.listdir(class_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            # Take random sample if too many
            if len(img_files) > test_size // len(os.listdir(test_data_dir)):
                img_files = random.sample(img_files, test_size // len(os.listdir(test_data_dir)))
            
            # Add to lists
            for img_file in img_files:
                image_paths.append(os.path.join(class_dir, img_file))
                true_labels.append(class_name)
                
        # Shuffle data
        combined = list(zip(image_paths, true_labels))
        random.shuffle(combined)
        image_paths, true_labels = zip(*combined)
        
        # Limit to test_size
        if len(image_paths) > test_size:
            image_paths = image_paths[:test_size]
            true_labels = true_labels[:test_size]
            
        print(f"{TermColors.CYAN}ℹ Evaluating ensemble on {len(image_paths)} test images...{TermColors.ENDC}")
        
        # Make predictions
        predictions = []
        for path in tqdm(image_paths, desc="Evaluating"):
            pred = ensemble.ensemble_predict(path, top_k=1)
            if pred and "error" not in pred[0]:
                predictions.append(pred[0]["class_name"])
            else:
                # Skip this image if prediction failed
                print(f"{TermColors.YELLOW}⚠️ Prediction failed for {path}, skipping.{TermColors.ENDC}")
                predictions.append(None)
        
        # Filter out Nones
        valid_indices = [i for i, p in enumerate(predictions) if p is not None]
        predictions = [predictions[i] for i in valid_indices]
        true_labels = [true_labels[i] for i in valid_indices]
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions, output_dict=True)
        
        # Print results
        print(f"{TermColors.GREEN}✅ Ensemble evaluation results:{TermColors.ENDC}")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Macro F1-Score: {report['macro avg']['f1-score']:.4f}")
        
        # Create confusion matrix plot
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(true_labels, predictions)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Add class labels if there aren't too many
        if len(set(true_labels)) <= 20:
            classes = sorted(set(true_labels))
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45, ha='right')
            plt.yticks(tick_marks, classes)
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Save plot
        cm_path = os.path.join(ENSEMBLE_DIR, "confusion_matrix.png")
        plt.savefig(cm_path)
        print(f"{TermColors.GREEN}✅ Confusion matrix saved to {cm_path}{TermColors.ENDC}")
        
        # Return metrics
        return {
            "accuracy": accuracy,
            "f1_score": report['macro avg']['f1-score'],
            "precision": report['macro avg']['precision'],
            "recall": report['macro avg']['recall'],
            "report": report
        }
        
    except Exception as e:
        print(f"{TermColors.RED}❌ Error evaluating ensemble: {e}{TermColors.ENDC}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print(f"{TermColors.HEADER}\n{'='*50}\nENSEMBLE METHODS MODULE\n{'='*50}{TermColors.ENDC}")
    
    # Check command line arguments
    if len(sys.argv) > 1:
        # Usage: python ensemble_methods.py <method> <image_path> [<peft_model_path>] [<chunk_model_dir>]
        method_name = sys.argv[1].lower()
        
        # Map method name to enum
        method_map = {
            "averaging": EnsembleMethod.AVERAGING,
            "weighted": EnsembleMethod.WEIGHTED_AVERAGING,
            "stacking": EnsembleMethod.STACKING,
            "voting": EnsembleMethod.VOTING,
            "bagging": EnsembleMethod.BAGGING,
            "boosting": EnsembleMethod.BOOSTING
        }
        
        if method_name not in method_map:
            print(f"{TermColors.RED}❌ Unknown ensemble method: {method_name}{TermColors.ENDC}")
            print(f"Available methods: {', '.join(method_map.keys())}")
            sys.exit(1)
            
        method = method_map[method_name]
        
        if len(sys.argv) > 2:
            image_path = sys.argv[2]
            model_path = sys.argv[3] if len(sys.argv) > 3 else "checkpoints_v2/DenseNet121_peft_best.keras"
            chunk_model_dir = sys.argv[4] if len(sys.argv) > 4 else "models_v2"
            
            if not os.path.exists(image_path):
                print(f"{TermColors.RED}❌ Image file not found: {image_path}{TermColors.ENDC}")
                sys.exit(1)
                
            # Load models
            feature_extractor, models_list, class_mappings = load_models_for_ensemble(
                model_path, chunk_model_dir, method
            )
            
            if feature_extractor is None:
                sys.exit(1)
                
            # Create ensemble
            ensemble = EnsembleEngine(
                feature_extractor=feature_extractor,
                models_list=models_list,
                class_mappings=class_mappings,
                method=method
            )
            
            # Run prediction
            print(f"{TermColors.CYAN}ℹ Running ensemble prediction with {method_name} on {image_path}...{TermColors.ENDC}")
            predictions = ensemble.ensemble_predict(image_path)
            
            # Print results
            print(f"\n{TermColors.HEADER}Top predictions using {method_name}:{TermColors.ENDC}")
            for i, pred in enumerate(predictions):
                print(f"{i+1}. {pred['class_name']}: {pred['probability']:.4f}")
        
        else:
            print(f"{TermColors.YELLOW}⚠️ No image provided for inference.{TermColors.ENDC}")
            print(f"Usage: python {os.path.basename(__file__)} <method> <image_path> [<peft_model_path>] [<chunk_model_dir>]")
            print(f"Example: python {os.path.basename(__file__)} averaging test_image.jpg")
    
    else:
        print(f"{TermColors.YELLOW}⚠️ No ensemble method specified.{TermColors.ENDC}")
        print(f"Usage: python {os.path.basename(__file__)} <method> <image_path> [<peft_model_path>] [<chunk_model_dir>]")
        print(f"Available methods: averaging, weighted, stacking, voting, bagging, boosting")
        print(f"Example: python {os.path.basename(__file__)} averaging test_image.jpg")