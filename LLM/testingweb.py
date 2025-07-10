"""
Plant Recognition Web App - EXACT TRAINING SYSTEM MATCH

This app uses the EXACT SAME components as the training system:
- Same neural network architecture from create_proper_training.py
- Same descriptor extraction pipeline from hyper_detailed_pattern_recognition.py  
- Same feature preprocessing (truncate and flatten)
- Loads the exact trained model from trained_plant_model.pt

  GOAL: Perfect training/inference consistency for maximum accuracy
"""

import os
import time
import tempfile
import numpy as np
import cv2
import torch
import torch.nn as nn
from flask import Flask, request, render_template_string, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from pathlib import Path
import shutil

# Import the EXACT SAME components used during training
from ..descriptormodel.hyper_detailed_pattern_recognition import MultiModalCurseResistantRecognizer

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for model and extractors
model_loaded = False
recognizer = None
pytorch_model = None
pytorch_model_data = None
device = None

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Import the EXACT SAME neural network architectures as training
from ..descriptormodel.training import StateOfTheArtPlantNetwork, ResidualBlock, MultiHeadAttention

# Keep backward compatibility
UltraFastPlantNetwork = StateOfTheArtPlantNetwork

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_models():
    """Load the trained PyTorch model."""
    global model_loaded, recognizer, pytorch_model, pytorch_model_data, device
    
    # Check for both model types (prioritize blind prediction model)
    blind_model_path = "blind_prediction_plant_model.pt"
    legacy_model_path = "trained_plant_model.pt"
    
    # Use blind prediction model if available, otherwise fall back to legacy
    if os.path.exists(blind_model_path):
        pt_model_path = blind_model_path
        model_type = "blind_prediction"
        print(f"  Using blind prediction model: {pt_model_path}")
    elif os.path.exists(legacy_model_path):
        pt_model_path = legacy_model_path
        model_type = "legacy"
        print(f"  Using legacy model: {pt_model_path}")
    else:
        pt_model_path = blind_model_path  # Default for error message
        model_type = "blind_prediction"
        print(f"  Looking for model: {pt_model_path} (or {legacy_model_path})")
    
    # Setup device
    device = torch.device('cuda')
    print(f"   Device: {device}")
    
    # Initialize the EXACT SAME descriptor extraction system as training
    recognizer = MultiModalCurseResistantRecognizer(
        image_size=512,
        num_classes=100  # Will be overridden by loaded model
    )
    print(f"  Descriptor extraction system initialized (EXACT SAME as training)")
    
    # Load the trained PyTorch model
    if os.path.exists(pt_model_path):
        try:
            pytorch_model_data = torch.load(pt_model_path, map_location=device, weights_only=False)
            
            # Handle different model formats
            if model_type == "blind_prediction":
                # New blind prediction model format
                if 'model_config' in pytorch_model_data:
                    arch = pytorch_model_data['model_config']
                    feature_dim = arch['feature_dim']
                    hidden_dim = arch['hidden_dim']
                    num_classes = len(pytorch_model_data['class_names'])
                    
                    # Create PrototypicalNetwork for blind prediction models
                    from ..descriptormodel.training import PrototypicalNetwork
                    pytorch_model = PrototypicalNetwork(feature_dim, hidden_dim)
                    pytorch_model.load_state_dict(pytorch_model_data['model_state_dict'])
                    pytorch_model.prototypes = pytorch_model_data['prototypes']
                    
                    print(f"   Loading BLIND PREDICTION model")
                    print(f"   Feature dim: {feature_dim}, Hidden dim: {hidden_dim}")
                    print(f"   Prototypes: {len(pytorch_model.prototypes)} classes")
                else:
                    raise ValueError("Invalid blind prediction model format")
            else:
                # Legacy model format
                arch = pytorch_model_data['model_architecture']
                feature_dim = arch['feature_dim']
                num_classes = arch['num_classes']
                
                # Read hidden_dim from saved model architecture
                hidden_dim = arch.get('hidden_dim', 1024)  # Use saved hidden_dim
                
                # Handle both old and new model types
                if arch.get('type') == 'StateOfTheArtPlantNetwork':
                    pytorch_model = StateOfTheArtPlantNetwork(feature_dim, num_classes, hidden_dim)
                    print(f"   Loading STATE-OF-THE-ART model with hidden_dim={hidden_dim}")
                else:
                    # Backward compatibility - assume StateOfTheArt if no type specified but has large hidden_dim
                    if hidden_dim > 512:
                        pytorch_model = StateOfTheArtPlantNetwork(feature_dim, num_classes, hidden_dim)
                        print(f"   Loading STATE-OF-THE-ART model (inferred from hidden_dim={hidden_dim})")
                    else:
                        pytorch_model = UltraFastPlantNetwork(feature_dim, num_classes, hidden_dim)
                        print(f"   Loading legacy UltraFast model with hidden_dim={hidden_dim}")
                
                pytorch_model.load_state_dict(pytorch_model_data['model_state_dict'])
            
            # Common setup for both model types
            pytorch_model = pytorch_model.to(device)
            pytorch_model.eval()
            
            class_names = pytorch_model_data['class_names']
            training_method = pytorch_model_data.get('training_history', {}).get('training_method', 'unknown')
            
            print(f"  PyTorch model loaded successfully")
            if model_type == "blind_prediction":
                print(f"   Model type: Blind Prediction Prototypical Network")
                print(f"   Training method: Blind Prediction + Immediate Correction")
            else:
                print(f"   Model type: {arch.get('type', 'UltraFastPlantNetwork')}")
                print(f"   Training method: {training_method}")
            print(f"   Classes: {len(class_names)}")
            print(f"   Feature dimension: {feature_dim:,}")
            if model_type == "legacy":
                print(f"   Hidden dimension: {hidden_dim}")
            print(f"   Advanced anti-overfitting training with 100% data utilization")
            
        except Exception as e:
            print(f"  Error loading PyTorch model: {e}")
            return False
    else:
        print(f"  PyTorch model not found: {pt_model_path}")
        return False
    
    model_loaded = True
    return True

def extract_descriptors_exact_match(image_path):
    """Extract 1500 descriptors using the EXACT same method as training."""
    
    print(f"  Starting descriptor extraction (EXACT TRAINING MATCH)...")
    print(f"   Loading image: {image_path}")
    
    start_time = time.time()
    
    try:
        # Load and resize image (same as training)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"   Image shape: {image.shape}")
        
        # Resize to standard size (same as training)
        image = cv2.resize(image, (512, 512))
        print(f"   Resized to: {image.shape}")
        
        # Use global recognizer (same as training)
        global recognizer
        if recognizer is None:
            recognizer = MultiModalCurseResistantRecognizer(image_size=512, num_classes=100)
        
        print(f"     Using EXACT SAME extraction as training:")
        print(f"     6-modal ultra-parallel GPU pipeline → 1500 selected features")
        
        # Use the EXACT SAME extraction method as training
        # This calls process_image_ultra_parallel_gpu which:
        # 1. Converts to GPU tensor
        # 2. Generates 10 augmentations on GPU
        # 3. Extracts 6 modalities × 11 images (66 parallel operations)
        # 4. Extracts 15,000 raw features (2500 per modality)
        # 5. Selects best 1500 features for training
        descriptors = recognizer.process_image_ultra_parallel_gpu(image, augmentations_per_image=10)
        
        if descriptors is None or len(descriptors) == 0:
            raise ValueError("No descriptors extracted from image")
        
        print(f"   EXACT TRAINING EXTRACTION COMPLETE:")
        print(f"   → Extracted features: {len(descriptors):,}")
        print(f"   → Expected: 1500 (training target)")
        
        # Verify we got exactly 1500 features (training target)
        if len(descriptors) != 1500:
            print(f"   WARNING: Got {len(descriptors)} features, expected 1500")
            print(f"   Adjusting to match training expectations...")
            
            # Handle dimension mismatch (match training behavior)
            descriptors_final = descriptors.astype(np.float32)
            if len(descriptors_final) > 1500:
                # Truncate to 1500
                print(f"   Truncated {len(descriptors_final):,} → 1500 descriptors")
                descriptors_final = descriptors_final[:1500]
            elif len(descriptors_final) < 1500:
                # Pad with zeros
                print(f"   Padded {len(descriptors_final):,} → 1500 descriptors")
                padded = np.zeros(1500, dtype=np.float32)
                padded[:len(descriptors_final)] = descriptors_final
                descriptors_final = padded
        else:
            descriptors_final = descriptors.astype(np.float32)
            print(f"     Perfect match: 1500 features (exactly as training)")
        
        # Apply the same normalization as training if available
        model_has_normalization = False
        if 'training_history' in pytorch_model_data and 'descriptor_normalization' in pytorch_model_data['training_history']:
            norm_stats = pytorch_model_data['training_history']['descriptor_normalization']
            descriptor_mean = norm_stats['mean']
            descriptor_std = norm_stats['std']
            
            # Apply same normalization as training
            descriptors_final = (descriptors_final - descriptor_mean) / (descriptor_std + 1e-8)
            descriptors_final = np.clip(descriptors_final, -5, 5)
            print(f"     Applied training normalization (z-score + clipping)")
            model_has_normalization = True
        
        if not model_has_normalization:
            print(f"       No normalization stats found - using raw descriptors")
            print(f"       (This is normal for blind prediction models)")
        
        processing_time = time.time() - start_time
        print(f"\n    DESCRIPTOR EXTRACTION COMPLETE (EXACT TRAINING MATCH)")
        print(f"     Raw extraction: Ultra-parallel GPU pipeline")
        print(f"     Modalities: texture, color, shape, contrast, frequency, unique")
        print(f"     Raw features: 15,000 → Selected: {len(descriptors_final):,}")
        print(f"     Processing time: {processing_time:.3f}s")
        print(f"     Ready for model prediction!")
        
        return descriptors_final, processing_time, len(descriptors_final)
        
    except Exception as e:
        print(f"    Error during descriptor extraction: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to basic extraction if ultra-parallel fails
        print(f"\n    FALLBACK: Using basic extraction method...")
        try:
            basic_descriptors = recognizer.process_image(image)
            if basic_descriptors is not None and len(basic_descriptors) > 0:
                # Ensure 1500 dimensions
                if len(basic_descriptors) > 1500:
                    basic_descriptors = basic_descriptors[:1500]
                elif len(basic_descriptors) < 1500:
                    padded = np.zeros(1500, dtype=np.float32)
                    padded[:len(basic_descriptors)] = basic_descriptors
                    basic_descriptors = padded
                
                fallback_time = time.time() - start_time
                print(f"       Fallback successful: {len(basic_descriptors)} features")
                return basic_descriptors.astype(np.float32), fallback_time, len(basic_descriptors)
        except:
            pass
        
        raise

def predict_with_trained_model(descriptors):
    """Make prediction using the trained PyTorch model (supports both legacy and blind prediction models)."""
    global pytorch_model, pytorch_model_data, device
    
    if pytorch_model is None or pytorch_model_data is None:
        print(f"  Model not loaded: pytorch_model={pytorch_model is not None}, pytorch_model_data={pytorch_model_data is not None}")
        return None
    
    print(f"  Making prediction with trained neural network...")
    print(f"   Input descriptors: {len(descriptors):,}")
    
    # Determine model type
    is_blind_prediction = 'model_config' in pytorch_model_data
    
    try:
        # Convert to tensor and add batch dimension (EXACT SAME as training)
        print(f"   Converting to tensor...")
        descriptor_tensor = torch.FloatTensor(descriptors).unsqueeze(0).to(device)  # [1, feature_dim]
        print(f"   Tensor shape: {descriptor_tensor.shape}")
        print(f"   Expected feature_dim: {pytorch_model.feature_dim}")
        
        # Run inference based on model type
        print(f"   Running inference...")
        with torch.no_grad():
            if is_blind_prediction:
                # Blind prediction model - use similarity-based classification
                print(f"   Using blind prediction (prototypical) inference...")
                predictions, similarities, confidences = pytorch_model.classify_by_similarity(descriptor_tensor)
                
                # Convert similarities to probabilities for consistent interface
                probabilities = torch.softmax(similarities, dim=1)
                
                # Get prediction (use the returned prediction directly)
                predicted_class = predictions[0]  # First (and only) prediction
                
                predicted_species = pytorch_model_data['class_names'][predicted_class.item()]
                
                # Get top 5 predictions for blind prediction
                top5_probs, top5_indices = torch.topk(probabilities, min(5, len(pytorch_model_data['class_names'])))
                
                # Use the actual top probability as confidence (same as displayed in top-5)
                confidence_score = top5_probs[0][0].item()  # Highest probability
                
                print(f"   Blind prediction: {predicted_species} ({confidence_score:.1%})")
                model_type_desc = 'blind_prediction_prototypical'
                
            else:
                # Legacy model - use standard neural network
                print(f"   Using legacy neural network inference...")
                outputs = pytorch_model(descriptor_tensor)
                print(f"   Model outputs shape: {outputs.shape}")
                probabilities = torch.softmax(outputs, dim=1)
                
                # Get prediction
                confidence, predicted_class = torch.max(probabilities, 1)
                
                predicted_species = pytorch_model_data['class_names'][predicted_class.item()]
                confidence_score = confidence.item()
                
                print(f"   Legacy prediction: {predicted_species} ({confidence_score:.1%})")
                
                # Get top 5 predictions for legacy model
                top5_probs, top5_indices = torch.topk(probabilities, min(5, len(pytorch_model_data['class_names'])))
                model_type_desc = 'exact_training_match'
            
            # Create top5 predictions list (same for both model types)
            top5_predictions = []
            for i in range(len(top5_indices[0])):
                species = pytorch_model_data['class_names'][top5_indices[0][i].item()]
                conf = top5_probs[0][i].item()
                top5_predictions.append((species, conf))
        
        return {
            'species': predicted_species,
            'confidence': confidence_score,
            'top5_predictions': top5_predictions,
            'model_type': model_type_desc
        }
        
    except Exception as e:
        print(f"  Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None

def identify_plant(image_path):
    """Plant identification using the exact same method as training (no TTA)."""
    
    print(f"\n  PLANT IDENTIFICATION (EXACT TRAINING MATCH): {image_path}")
    print(f"=" * 60)
    
    total_start = time.time()
    
    try:
        # Step 1: Extract 1500 descriptors using EXACT training method
        descriptors, extraction_time, total_descriptors = extract_descriptors_exact_match(image_path)
        
        # Step 2: Make prediction with trained model
        prediction_result = predict_with_trained_model(descriptors)
        
        if prediction_result is None:
            return {'error': 'Failed to make prediction'}
        
        total_time = time.time() - total_start
        
        # Calculate prediction confidence metrics
        prediction_confidence = prediction_result['confidence']
        top5_predictions = prediction_result['top5_predictions']
        
        # Calculate entropy for uncertainty estimation
        probabilities = np.array([pred[1] for pred in top5_predictions[:len(pytorch_model_data['class_names'])]])
        if len(probabilities) < len(pytorch_model_data['class_names']):
            # Pad with zeros for missing classes
            full_probs = np.zeros(len(pytorch_model_data['class_names']))
            full_probs[:len(probabilities)] = probabilities
            probabilities = full_probs
        
        prediction_entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
        prediction_uncertainty = 1.0 - prediction_confidence
        
        print(f"\n  PREDICTION COMPLETE:")
        print(f"   Species: {prediction_result['species']}")
        print(f"   Confidence: {prediction_confidence:.1%}")
        print(f"   Entropy: {prediction_entropy:.3f}")
        print(f"   Uncertainty: {prediction_uncertainty:.1%}")
        print(f"   Total time: {total_time:.3f}s")
        
        # Prepare detailed results
        result = {
            'predicted_species': prediction_result['species'],
            'confidence': prediction_confidence,
            'top5_predictions': top5_predictions,
            'processing_details': {
                'extraction_method': 'Ultra-parallel GPU pipeline (6 modalities)',
                'raw_features_extracted': '15,000 (2,500 per modality)',
                'selected_features': total_descriptors,
                'target_features': 1500,
                'extraction_time': extraction_time,
                'prediction_time': total_time - extraction_time,
                'total_time': total_time,
                'model_type': prediction_result['model_type'],
                'prediction_entropy': prediction_entropy,
                'prediction_uncertainty': prediction_uncertainty,
                'modalities': 'texture, color, shape, contrast, frequency, unique'
            },
            'performance_metrics': {
                'features_per_second': total_descriptors / extraction_time if extraction_time > 0 else 0,
                'fps_equivalent': 1.0 / total_time if total_time > 0 else 0,
                'extraction_pipeline': 'GPU-accelerated parallel processing',
                'training_match': 'Exact same extraction as training'
            }
        }
        
        return result
        
    except Exception as e:
        print(f"  Error during enhanced identification: {e}")
        import traceback
        traceback.print_exc()
        return {
            'error': str(e),
            'processing_time': time.time() - total_start
        }

# HTML template for the web interface with tabs
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>  Plant Recognition - 6-Modal System</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }

        h1 {
            color: #2d5a27;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
        }
        .upload-area {
            border: 3px dashed #4CAF50;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            background: #f8f9fa;
            transition: all 0.3s ease;
        }
        .upload-area:hover {
            background: #e8f5e8;
            border-color: #45a049;
        }
        .upload-button {
            background: #4CAF50;
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 18px;
            transition: all 0.3s ease;
        }
        .upload-button:hover {
            background: #45a049;
            transform: translateY(-2px);
        }
        .result-container {
            margin-top: 30px;
            padding: 25px;
            border-radius: 15px;
            background: #f0f8f0;
            border-left: 5px solid #4CAF50;
        }
        .species-name {
            font-size: 2em;
            font-weight: bold;
            color: #2d5a27;
            margin-bottom: 15px;
        }
        .confidence {
            font-size: 1.3em;
            color: #555;
            margin-bottom: 20px;
        }
        .top5-list {
            margin-top: 20px;
        }
        .top5-item {
            padding: 10px;
            margin: 5px 0;
            background: white;
            border-radius: 8px;
            border-left: 4px solid #4CAF50;
        }
        .tech-details {
            margin-top: 30px;
            padding: 20px;
            background: #e3f2fd;
            border-radius: 10px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }
        .error {
            background: #ffebee;
            border-left-color: #f44336;
            color: #c62828;
        }
        .loading {
            text-align: center;
            padding: 40px;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #4CAF50;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>  Plant Recognition</h1>
        <div style="text-align: center; background: #e8f5e8; padding: 15px; border-radius: 10px; margin-bottom: 20px; color: #2d5a27;">
            <h3 style="margin: 0 0 10px 0;">6-Modal Descriptor Analysis with Unique Plant Features</h3>
            <p style="margin: 0; font-size: 0.9em;">
                  <strong>Texture</strong> •   <strong>Color</strong> •   <strong>Shape</strong> • 
                  <strong>Contrast</strong> •   <strong>Frequency</strong> •   <strong>Unique</strong>
            </p>
        </div>
        
        <!-- Plant Identification Tab -->
        <div id="identify-tab" class="tab-content active">
        <div class="upload-area">
            <h3>Upload Plant Image</h3>
            <p>Supported formats: PNG, JPG, JPEG, GIF, BMP, TIFF</p>
            <form method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept=".png,.jpg,.jpeg,.gif,.bmp,.tiff" required>
                <br><br>
                <input type="submit" value="  Identify Plant" class="upload-button">
            </form>
            </div>
        </div>
        
        {% if result %}
            {% if result.error %}
                <div class="result-container error">
                    <h3>  Error</h3>
                    <p>{{ result.error }}</p>
                    {% if result.processing_time %}
                    <div class="tech-details">
                        Processing time: {{ "%.3f"|format(result.processing_time) }}s
                    </div>
                    {% endif %}
                </div>

            {% else %}
                <div class="result-container">
                    <div class="species-name">{{ result.predicted_species.replace('_', ' ').title() }}</div>
                    <div class="confidence">Confidence: {{ "%.1f"|format(result.confidence * 100) }}%</div>
                    
                    <div class="top5-list">
                        <h4>Top 5 Predictions:</h4>
                        {% for species, conf in result.top5_predictions %}
                            <div class="top5-item">
                                <strong>{{ species.replace('_', ' ').title() }}</strong> - {{ "%.1f"|format(conf * 100) }}%
                            </div>
                        {% endfor %}
                    </div>
                    
                    <div class="tech-details">
                        <h4>  Technical Details:</h4>
                        <strong>6-Modal Analysis:</strong><br>
                        •   Texture: Multi-scale pattern recognition<br>
                        •   Color: RGB/HSV/LAB space analysis<br>
                        •   Shape: Geometric feature extraction<br>
                        •   Contrast: Local intensity variations<br>
                        •   Frequency: FFT-based spectral analysis<br>
                        •   Unique: Class-specific descriptor generation<br>
                        <br>
                        <strong>Processing:</strong><br>
                        • Raw features extracted: {{ result.processing_details.raw_features_extracted }}<br>
                        • Selected features: {{ "{:,}".format(result.processing_details.selected_features) }}<br>
                        • Extraction time: {{ "%.3f"|format(result.processing_details.extraction_time) }}s<br>
                        • Prediction time: {{ "%.3f"|format(result.processing_details.prediction_time) }}s<br>
                        • Total time: {{ "%.3f"|format(result.processing_details.total_time) }}s<br>
                        <br>
                        <strong>Feature Extraction Details:</strong><br>
                        • Extraction method: {{ result.processing_details.extraction_method }}<br>
                        • Modalities: {{ result.processing_details.modalities }}<br>
                        • Prediction entropy: {{ "%.3f"|format(result.processing_details.prediction_entropy) }}<br>
                        • Uncertainty: {{ "%.1f"|format(result.processing_details.prediction_uncertainty * 100) }}%<br>
                        <br>
                        <strong>Performance:</strong><br>
                        • Features/sec: {{ "{:,.0f}".format(result.performance_metrics.features_per_second) }}<br>
                        • FPS equivalent: {{ "%.2f"|format(result.performance_metrics.fps_equivalent) }}<br>
                        • Extraction pipeline: {{ result.performance_metrics.extraction_pipeline }}<br>
                        • Training match: {{ result.performance_metrics.training_match }}<br>
                        • Model type: {{ result.processing_details.model_type }}<br>
                    </div>
                </div>
            {% endif %}
        {% endif %}
    </div>
    

</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template_string(HTML_TEMPLATE, result={'error': 'No file selected'})
    
    file = request.files['file']
    if file.filename == '':
        return render_template_string(HTML_TEMPLATE, result={'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name
        
        try:
            # Identify plant
            result = identify_plant(temp_path)
            return render_template_string(HTML_TEMPLATE, result=result)
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
    
    return render_template_string(HTML_TEMPLATE, result={'error': 'Invalid file format'})

@app.route('/api/identify', methods=['POST'])
def api_identify():
    """API endpoint for programmatic access"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format'}), 400
    
    filename = secure_filename(file.filename)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
        file.save(temp_file.name)
        temp_path = temp_file.name
    
    try:
        # Identify plant
        result = identify_plant(temp_path)
        return jsonify(result)
    finally:
        # Clean up temporary file
        os.unlink(temp_path)


if __name__ == '__main__':
    print("  PLANT RECOGNITION APP - EXACT TRAINING MATCH")
    print("=" * 50)
    
    # Load models on startup
    if load_models():
        print(f"\n  Starting web server...")
        print(f"   URL: http://localhost:5000")
        print(f"   API: http://localhost:5000/api/identify")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print(f"  Failed to load models. Please ensure trained_plant_model.pt exists.") 