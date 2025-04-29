import os
import shutil
import json
import tensorflow as tf
import keras.backend as K

# Modify numpy to work with older tensorflowjs versions
import numpy as np
np.object = object  # Fix the deprecated np.object issue
np.bool = bool      # Fix similar issue with np.bool

# Define the custom sparse_top_k_accuracy metric function
def sparse_top_k_accuracy(y_true, y_pred, k=10):
    """Calculate top-K accuracy for sparse labels"""
    return tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k)

# Now import tensorflowjs after fixing numpy issues
import tensorflowjs as tfjs

tf.compat.v1.disable_eager_execution()

# Path configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
WEBAPP_DIR = os.path.join(BASE_DIR, 'webapp')
WEBAPP_MODELS_DIR = os.path.join(WEBAPP_DIR, 'models')
KERAS_MODEL_PATH = os.path.join(MODEL_DIR, 'chunk_0_model.keras')
METADATA_PATH = os.path.join(MODEL_DIR, 'chunk_0_metadata.json')
TFJS_MODEL_DIR = os.path.join(WEBAPP_MODELS_DIR, 'tfjs_model')

def ensure_dir(directory):
    """Make sure the directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def copy_model_files():
    """
    Simply copy the model and metadata files to the webapp directory.
    No conversion needed, just direct file copying.
    """
    print("Setting up model files for the webapp...")
    
    # Ensure the webapp models directory exists
    ensure_dir(WEBAPP_MODELS_DIR)
    
    # Copy the model file
    if os.path.exists(KERAS_MODEL_PATH):
        print(f"Copying model from {KERAS_MODEL_PATH} to {WEBAPP_MODELS_DIR}...")
        shutil.copy(KERAS_MODEL_PATH, os.path.join(WEBAPP_MODELS_DIR, os.path.basename(KERAS_MODEL_PATH)))
    else:
        print(f"Warning: Model file not found at {KERAS_MODEL_PATH}")
    
    # Copy the metadata file
    if os.path.exists(METADATA_PATH):
        print(f"Copying metadata from {METADATA_PATH} to {WEBAPP_MODELS_DIR}...")
        shutil.copy(METADATA_PATH, os.path.join(WEBAPP_MODELS_DIR, os.path.basename(METADATA_PATH)))
    else:
        print(f"Warning: Metadata file not found at {METADATA_PATH}")
    
    print("Model files copied successfully.")
    print("\nNext steps to run the webapp:")
    print("1. Start a web server in the main directory:")
    print("   python -m http.server 8000")
    print("2. Access the webapp at: http://localhost:8000/webapp/")

def convert_model_to_tfjs():
    """
    Convert the Keras model to TensorFlow.js format.
    """
    print(f"Loading model from {KERAS_MODEL_PATH}...")
    if not os.path.exists(KERAS_MODEL_PATH):
        print(f"Error: Model file {KERAS_MODEL_PATH} not found!")
        exit(1)
    
    # Register custom metric when loading the model
    custom_objects = {
        'sparse_top_k_accuracy': sparse_top_k_accuracy
    }
    
    model = tf.keras.models.load_model(KERAS_MODEL_PATH, custom_objects=custom_objects)
    print("Model loaded successfully!")
    
    # Print model summary for verification
    model.summary()
    
    print(f"Converting model to TensorFlow.js format in {TFJS_MODEL_DIR}...")
    ensure_dir(TFJS_MODEL_DIR)
    tfjs.converters.save_keras_model(model, TFJS_MODEL_DIR)
    print("Conversion completed!")

if __name__ == "__main__":
    copy_model_files()
    convert_model_to_tfjs()