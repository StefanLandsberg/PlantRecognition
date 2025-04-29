import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import efficientnet_v2, densenet
import argparse
import glob

# Define paths
MODEL_PATH = os.path.join('models', 'chunk_0_model.keras')
METADATA_PATH = os.path.join('models', 'chunk_0_metadata.json')

# Define the custom sparse_top_k_accuracy metric function
def sparse_top_k_accuracy(y_true, y_pred, k=10):
    """Calculate top-K accuracy for sparse labels"""
    return tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k)

def load_class_mapping():
    """Load class mapping from metadata file"""
    try:
        import json
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        return metadata.get('class_mapping', {})
    except Exception as e:
        print(f"Error loading class mapping: {e}")
        return {}

def preprocess_image_for_efficientnet(img_path):
    """Preprocess an image for EfficientNetV2L"""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return efficientnet_v2.preprocess_input(img_array)

def preprocess_image_for_densenet(img_path):
    """Preprocess an image for DenseNet121"""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return densenet.preprocess_input(img_array)

def extract_features(img_path):
    """Extract features from both EfficientNetV2L and DenseNet121"""
    # Load feature extractors
    efficient_net = efficientnet_v2.EfficientNetV2L(
        include_top=False, 
        weights='imagenet',
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    dense_net = densenet.DenseNet121(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    
    # Process image for both models
    efficient_net_img = preprocess_image_for_efficientnet(img_path)
    dense_net_img = preprocess_image_for_densenet(img_path)
    
    # Extract features
    efficient_net_features = efficient_net.predict(efficient_net_img)
    dense_net_features = dense_net.predict(dense_net_img)
    
    # Combine features
    combined_features = np.concatenate([efficient_net_features, dense_net_features], axis=1)
    
    print(f"EfficientNetV2L features shape: {efficient_net_features.shape}")
    print(f"DenseNet121 features shape: {dense_net_features.shape}")
    print(f"Combined features shape: {combined_features.shape}")
    
    return combined_features

def predict_plant(img_path, model, class_mapping):
    """Predict plant class from image"""
    # Extract features
    features = extract_features(img_path)
    
    # Get predictions
    predictions = model.predict(features)
    
    # Map predictions to class names
    results = []
    for i in range(len(predictions[0])):
        class_id = str(i)
        class_name = class_mapping.get(class_id, f"Unknown ({i})")
        confidence = float(predictions[0][i])
        results.append({
            'class_id': i,
            'class_name': class_name,
            'confidence': confidence
        })
    
    # Sort by confidence
    results = sorted(results, key=lambda x: x['confidence'], reverse=True)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Test plant recognition model')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--dir', type=str, help='Directory containing images to test')
    args = parser.parse_args()
    
    # Check if at least one argument is provided
    if not args.image and not args.dir:
        parser.error('Please provide either --image or --dir argument')
    
    print("Loading model from", MODEL_PATH)
    # Pass custom_objects to recognize the sparse_top_k_accuracy function
    custom_objects = {'sparse_top_k_accuracy': sparse_top_k_accuracy}
    model = load_model(MODEL_PATH, custom_objects=custom_objects)
    model.summary()
    
    print("Loading class mapping")
    class_mapping = load_class_mapping()
    print(f"Loaded {len(class_mapping)} classes")
    
    # Process single image
    if args.image:
        print(f"\nProcessing image: {args.image}")
        results = predict_plant(args.image, model, class_mapping)
        
        # Display top 5 results
        print("\nTop 5 predictions:")
        for i, result in enumerate(results[:20]):
            confidence_percent = result['confidence'] * 100
            print(f"{i+1}. {result['class_name']}: {confidence_percent:.2f}%")
    
    # Process directory of images
    if args.dir:
        image_files = []
        for ext in ['jpg', 'jpeg', 'png']:
            image_files.extend(glob.glob(os.path.join(args.dir, f'*.{ext}')))
        
        print(f"\nProcessing {len(image_files)} images from {args.dir}")
        
        for img_path in image_files:
            print(f"\nImage: {os.path.basename(img_path)}")
            results = predict_plant(img_path, model, class_mapping)
            
            # Display top 3 results
            print("Top 3 predictions:")
            for i, result in enumerate(results[:3]):
                confidence_percent = result['confidence'] * 100
                print(f"{i+1}. {result['class_name']}: {confidence_percent:.2f}%")

if __name__ == "__main__":
    main()