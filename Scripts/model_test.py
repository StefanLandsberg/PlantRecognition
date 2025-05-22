import os
import sys
import json
import numpy as np
import requests
import tempfile
import tensorflow as tf
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from termcolor import colored
import os
# Constants
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
IMAGE_SIZE = (224, 224)
CHUNK_TO_TEST = 0  # Default to chunk 0, can be changed via command line arg

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

def load_image_from_url(url):
    """Load an image from a URL and preprocess it."""
    try:
        print(f"{TermColors.CYAN}ℹ Downloading image from URL...{TermColors.ENDC}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name
            
        print(f"{TermColors.GREEN}✅ Image downloaded successfully{TermColors.ENDC}")
        return load_and_preprocess_image(tmp_path), tmp_path
    except Exception as e:
        print(f"{TermColors.RED}❌ Error downloading image: {e}{TermColors.ENDC}")
        return None, None

def load_image_from_file(file_path):
    """Load an image from a local file and preprocess it."""
    try:
        print(f"{TermColors.CYAN}ℹ Loading image from file...{TermColors.ENDC}")
        if not os.path.exists(file_path):
            print(f"{TermColors.RED}❌ File not found: {file_path}{TermColors.ENDC}")
            return None, None
            
        print(f"{TermColors.GREEN}✅ Image loaded successfully{TermColors.ENDC}")
        return load_and_preprocess_image(file_path), file_path
    except Exception as e:
        print(f"{TermColors.RED}❌ Error loading image: {e}{TermColors.ENDC}")
        return None, None

def load_and_preprocess_image(file_path):
    """Load and preprocess an image for the model."""
    # Load image with tensorflow
    img = tf.keras.preprocessing.image.load_img(file_path, target_size=IMAGE_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    # Expand dimensions for batch processing
    img_array = np.expand_dims(img_array, axis=0)
    
    # Use ConvNext preprocessing to match training pipeline
    preprocessed = tf.keras.applications.convnext.preprocess_input(img_array)
    
    print(f"{TermColors.CYAN}ℹ Image preprocessed with ConvNext preprocessing{TermColors.ENDC}")
    
    return preprocessed

def extract_features(img_array):
    """Extract features from the image using both models."""
    print(f"{TermColors.CYAN}ℹ Extracting features from image...{TermColors.ENDC}")
    
    # Load feature extractor models
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        # Create EfficientNetV2L feature extractor
        efficient_base = tf.keras.applications.EfficientNetV2L(
            include_top=False, 
            weights='imagenet',
            input_shape=(*IMAGE_SIZE, 3),
            pooling='avg'
        )
        
        # Create DenseNet121 feature extractor
        dense_base = tf.keras.applications.DenseNet121(
            include_top=False, 
            weights='imagenet',
            input_shape=(*IMAGE_SIZE, 3),
            pooling='avg'
        )
    
    # Extract features from both models
    efficient_features = efficient_base.predict(img_array, verbose=0)
    dense_features = dense_base.predict(img_array, verbose=0)
    
    # Combine features
    combined_features = np.concatenate([efficient_features, dense_features], axis=1)
    
    print(f"{TermColors.GREEN}✅ Features extracted: {combined_features.shape[1]} dimensions{TermColors.ENDC}")
    
    # Clean up to free memory
    del efficient_base, dense_base
    tf.keras.backend.clear_session()
    
    return combined_features

def predict_plant_two_phase(img_array, chunk_idx=0, top_k=5, top_first_phase=30):
    """Two-phase prediction: first get top 30 candidates, then refine to top 5."""
    # Phase 1: Extract features and run initial model for top 30 candidates
    features = extract_features(img_array)
    
    # Load model and metadata
    model_path = os.path.join(MODEL_DIR, f"chunk_{chunk_idx}_model.keras")
    metadata_path = os.path.join(MODEL_DIR, f"chunk_{chunk_idx}_metadata.json")
    
    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        print(f"{TermColors.RED}❌ Model or metadata not found for chunk {chunk_idx}{TermColors.ENDC}")
        return None
    
    # Define custom metrics for model loading
    def sparse_top_k_categorical_accuracy(y_true, y_pred, k=5):
        return tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k)
    
    # Custom objects dictionary with all possible metric names
    custom_objects = {
        'sparse_top_k_categorical_accuracy': sparse_top_k_categorical_accuracy,
        'sparse_top_k_accuracy': sparse_top_k_categorical_accuracy
    }
        
    print(f"{TermColors.CYAN}ℹ Loading model for chunk {chunk_idx}...{TermColors.ENDC}")
    
    try:
        # Load model with custom objects
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Phase 1: Get initial predictions
        print(f"{TermColors.CYAN}ℹ Phase 1: Initial prediction for top {top_first_phase} candidates...{TermColors.ENDC}")
        predictions = model.predict(features, verbose=0)[0]
        
        # Get top candidates from phase 1
        top_indices = np.argsort(predictions)[-top_first_phase:][::-1]
        top_predictions = [(int(idx), float(predictions[idx])) for idx in top_indices]
        
        # Map class indices to names
        class_mapping = metadata.get('class_mapping', {})
        candidate_names = [class_mapping.get(str(idx), f"Unknown-{idx}") for idx, _ in top_predictions]
        
        # Phase 2: Fine-tuned prediction focusing only on top candidates
        print(f"{TermColors.CYAN}ℹ Phase 2: Refined prediction for top {top_k} candidates...{TermColors.ENDC}")
        
        # Create a specialized softmax layer just for the top candidates
        top_indices_list = [idx for idx, _ in top_predictions]
        
        # Re-run softmax just on the top classes for better calibration
        filtered_probs = predictions[top_indices_list]
        rescaled_probs = tf.nn.softmax(filtered_probs / 0.3).numpy()  # Temperature scaling for better calibration
        
        # Sort by recalibrated confidence
        final_predictions = [(candidate_names[i], float(rescaled_probs[i]), top_indices_list[i]) 
                            for i in range(len(top_indices_list))]
        final_predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Free memory
        del model
        tf.keras.backend.clear_session()
        
        return final_predictions[:top_k]
        
    except Exception as e:
        print(f"{TermColors.RED}❌ Error loading or using model: {e}{TermColors.ENDC}")
        print(f"{TermColors.YELLOW}⚠️ Try checking if this chunk's model is corrupt or was saved with different metrics{TermColors.ENDC}")
        
        # Cleanup on error
        tf.keras.backend.clear_session()
        
        return None

def find_similar_plants(features, metadata_paths, top_k=5):
    """Find visually similar plants across all chunks based on feature distance."""
    all_results = []
    
    for metadata_path in metadata_paths:
        # Load class mapping
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            class_mapping = metadata.get('class_mapping', {})
            
        # Find closest matches
        # In a full implementation, this would load feature vectors and compare
        # This is a simplified placeholder
        
    return all_results

def display_prediction_results(predictions, image_path):
    """Display the prediction results with a nice visualization."""
    if not predictions:
        print(f"{TermColors.RED}❌ No predictions available{TermColors.ENDC}")
        return
        
    print(f"\n{TermColors.HEADER}{'='*50}")
    print(f"PLANT RECOGNITION RESULTS")
    print(f"{'='*50}{TermColors.ENDC}")
    
    # Display the top predictions
    for i, (plant_name, confidence, class_idx) in enumerate(predictions):
        confidence_bar = "█" * int(confidence * 20)
        confidence_str = f"[{confidence_bar:<20}] {confidence:.4f}"
        print(f"{i+1}. {TermColors.BOLD}{plant_name}{TermColors.ENDC} - {TermColors.CYAN}{confidence_str}{TermColors.ENDC}")
    
    # Display the image if possible
    try:
        img = Image.open(image_path)
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.title(f"Top prediction: {predictions[0][0]} ({predictions[0][1]:.2%})")
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"{TermColors.YELLOW}⚠️ Could not display image: {e}{TermColors.ENDC}")

def main():
    """Main function to handle command line arguments."""
    chunk_idx = CHUNK_TO_TEST
    top_k = 5
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        # Check if a specific chunk was requested
        if sys.argv[1].isdigit():
            chunk_idx = int(sys.argv[1])
            print(f"{TermColors.CYAN}ℹ Using chunk {chunk_idx} for testing{TermColors.ENDC}")
    
    # Get image input
    while True:
        print(f"\n{TermColors.HEADER}PLANT RECOGNITION TEST - CHUNK {chunk_idx}{TermColors.ENDC}")
        print(f"1. Test with image URL")
        print(f"2. Test with local image file")
        print(f"3. Change chunk (current: {chunk_idx})")
        print(f"4. Exit")
        
        choice = input(f"\nEnter your choice (1-4): ")
        
        if choice == '1':
            url = input("Enter image URL: ")
            img_array, img_path = load_image_from_url(url)
            if img_array is not None:
                predictions = predict_plant_two_phase(img_array, chunk_idx, top_k)
                display_prediction_results(predictions, img_path)
                
        elif choice == '2':
            file_path = input("Enter image file path: ")
            img_array, img_path = load_image_from_file(file_path)
            if img_array is not None:
                predictions = predict_plant_two_phase(img_array, chunk_idx, top_k)
                display_prediction_results(predictions, img_path)
                
        elif choice == '3':
            new_chunk = input(f"Enter chunk number (0-19): ")
            if new_chunk.isdigit() and 0 <= int(new_chunk) <= 19:
                chunk_idx = int(new_chunk)
                print(f"{TermColors.GREEN}✅ Switched to chunk {chunk_idx}{TermColors.ENDC}")
            else:
                print(f"{TermColors.RED}❌ Invalid chunk number{TermColors.ENDC}")
                
        elif choice == '4':
            print(f"{TermColors.GREEN}Goodbye!{TermColors.ENDC}")
            break
            
        else:
            print(f"{TermColors.RED}❌ Invalid choice{TermColors.ENDC}")

if __name__ == "__main__":
    main()