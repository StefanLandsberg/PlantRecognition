#!/usr/bin/env python3
"""
Plant Recognition Web Server

"""

import os
import time
import cv2
import torch
import numpy as np
from flask import Flask, request, render_template_string, jsonify
from werkzeug.utils import secure_filename
import tempfile
from pathlib import Path

# Import our modules
from hyper_detailed_pattern_recognition import MultiModalCurseResistantRecognizer
from training import DirectClassificationNetwork

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for model and scaler
model = None
model_data = None
scaler = None
recognizer = None
device = None



def load_model_and_scaler():
    """Load the trained model and fitted scaler"""
    global model, model_data, scaler, recognizer, device
    
    model_path = "trained_plant_model.pt"
    
    if not os.path.exists(model_path):
        print(f" Model file not found: {model_path}")
        return False
    
    try:
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f" Device: {device}")
        
        # Load model data
        model_data = torch.load(model_path, map_location=device, weights_only=False)
        print(f" Model data loaded: {model_path}")
        
        # Check if scaler is saved
        if 'feature_scaler' not in model_data:
            print(f" CRITICAL ERROR: No scaler found in model data!")
            print(f"   Available keys: {list(model_data.keys())}")
            print(f"   Model needs to be retrained with scaler saving")
            return False
        
        # Load the fitted scaler
        scaler = model_data['feature_scaler']
        print(f" Fitted scaler loaded: {type(scaler).__name__}")
        
        # Create model
        model_config = model_data['model_config']
        model = DirectClassificationNetwork(
            feature_dim=model_config['feature_dim'],
            num_classes=model_config['num_classes'],
            hidden_dim=model_config.get('hidden_dim', 512)
        ).to(device)
        
        # Load model weights
        model.load_state_dict(model_data['model_state_dict'])
        model.eval()
        print(f" Model loaded: {len(model_data['class_names'])} classes")
        
        # Initialize recognizer
        recognizer = MultiModalCurseResistantRecognizer()
        print(f" Feature recognizer initialized")
        
        print(f" MODEL AND SCALER LOADED SUCCESSFULLY")
        print(f"   Classes: {len(model_data['class_names'])}")
        print(f"   Feature dim: {model_config['feature_dim']}")
        print(f"   Training samples: {model_data.get('training_samples', 'N/A')}")
        print(f"   Final accuracy: {model_data.get('training_history', {}).get('final_accuracy', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f" Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

def extract_features(image_path):
    """Extract features using the same method as training"""
    if recognizer is None:
        raise ValueError("Recognizer not initialized")
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Resize to standard size (same as training)
    image = cv2.resize(image, (512, 512))
    
    # Extract features using EXACT same method as training
    print(f" Extracting features using training method...")
    features = recognizer.process_image_ultra_parallel_gpu(image, augmentations_per_image=10)
    
    if features is None or len(features) == 0:
        raise ValueError("Feature extraction failed")
    
    # Use BOTH dual descriptors like training (features[0] and features[1])
    # This gives us the same 30k->10k->2x5k pipeline as training
    if len(features) < 2:
        raise ValueError(f"Expected 2 dual descriptors, got {len(features)}")
    
    dual_descriptor_A = features[0]  # First 5k descriptor
    dual_descriptor_B = features[1]  # Second 5k descriptor
    
    print(f"   Dual descriptors extracted: 2 × {len(dual_descriptor_A)} features")
    print(f"   Descriptor A range: [{np.min(dual_descriptor_A):.6f}, {np.max(dual_descriptor_A):.6f}]")
    print(f"   Descriptor B range: [{np.min(dual_descriptor_B):.6f}, {np.max(dual_descriptor_B):.6f}]")
    
    return dual_descriptor_A, dual_descriptor_B

def normalize_features(descriptor_A, descriptor_B):
    """Normalize BOTH dual descriptors using the SAME scaler as training"""
    if scaler is None:
        raise ValueError("Scaler not loaded")
    
    print(f" Normalizing features using training scaler...")
    print(f"   Raw features: mean={np.mean(features):.6f}, std={np.std(features):.6f}")
    
    # Apply the SAME normalization as training
    features_2d = features.reshape(1, -1)  # Shape for sklearn
    normalized_features = scaler.transform(features_2d)[0]  # Transform, then extract
    
    print(f"   Normalized features: mean={np.mean(normalized_features):.6f}, std={np.std(normalized_features):.6f}")
    print(f"    Features normalized using training scaler")
    
    return norm_A, norm_B

def predict_plant(norm_A, norm_B):
    """Make ensemble prediction using BOTH dual descriptors like training"""
    if model is None or model_data is None:
        raise ValueError("Model not loaded")
    
    print(f" Making prediction...")
    
    # Convert both descriptors to tensors
    tensor_A = torch.FloatTensor(norm_A).unsqueeze(0).to(device)
    tensor_B = torch.FloatTensor(norm_B).unsqueeze(0).to(device)
    print(f"   Input tensor shapes: A{tensor_A.shape}, B{tensor_B.shape}")
    
    # Make predictions on both descriptors
    with torch.no_grad():
        # Predict with both dual descriptors
        pred_A, prob_A, conf_A = model.predict_with_confidence(tensor_A)
        pred_B, prob_B, conf_B = model.predict_with_confidence(tensor_B)
        
        # Ensemble: average the probabilities
        ensemble_prob = (prob_A + prob_B) / 2.0
        
        # Get final prediction from ensemble
        predicted_class = torch.argmax(ensemble_prob, dim=1).item()
        confidence_score = ensemble_prob[0][predicted_class].item()
        class_probabilities = ensemble_prob[0].cpu().numpy()
        
        print(f"   Descriptor A: class {pred_A[0].item()} ({conf_A[0].item()*100:.1f}%)")
        print(f"   Descriptor B: class {pred_B[0].item()} ({conf_B[0].item()*100:.1f}%)")
        print(f"   Ensemble: class {predicted_class} ({confidence_score*100:.1f}%)")
        
        # Get class name
        predicted_species = model_data['class_names'][predicted_class]
        
        # Get top 5 predictions from ensemble
        top5_indices = np.argsort(class_probabilities)[::-1][:5]
        top5_predictions = []
        for i, class_idx in enumerate(top5_indices):
            species = model_data['class_names'][class_idx]
            prob = class_probabilities[class_idx]
            top5_predictions.append((species, prob))
        
        print(f"    Prediction: {predicted_species} ({confidence_score:.1%})")
        
        return {
            'predicted_species': predicted_species,
            'confidence': confidence_score,  # Use ensemble confidence
            'top5_predictions': top5_predictions,
            'predicted_class_index': predicted_class,
            'calibrated_confidence': confidence_score,
            # Dual descriptor specific metrics for template
            'descriptor_A_class': pred_A[0].item(),
            'descriptor_A_conf': conf_A[0].item(),
            'descriptor_B_class': pred_B[0].item(), 
            'descriptor_B_conf': conf_B[0].item(),
            'extraction_method': '30k→10k→2×5k dual descriptors',
            'descriptor_count': 2,
            'prediction_method': 'Ensemble averaging',
            'pipeline_status': 'Dual descriptor ensemble ✅'
        }

def identify_plant_fixed(image_path):
    """Fixed plant identification with proper scaler handling"""
    print(f"\n FIXED PLANT IDENTIFICATION: {image_path}")
    print(f"=" * 60)
    
    start_time = time.time()
    
    try:
        # Step 1: Extract dual descriptors
        print(f"Step 1: Feature extraction...")
        descriptor_A, descriptor_B = extract_features(image_path)
        
        # Step 2: Normalize both descriptors using training scaler
        print(f"Step 2: Feature normalization...")
        norm_A, norm_B = normalize_features(descriptor_A, descriptor_B)
        
        # Step 3: Make ensemble prediction
        print(f"Step 3: Prediction...")
        result = predict_plant(norm_A, norm_B)
        
        total_time = time.time() - start_time
        result['processing_time'] = total_time
        
        print(f"\n IDENTIFICATION COMPLETE:")
        print(f"   Species: {result['predicted_species']}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   Processing time: {total_time:.3f}s")
        
        return result
        
    except Exception as e:
        error_time = time.time() - start_time
        print(f" Error during identification: {e}")
        import traceback
        traceback.print_exc()
        return {
            'error': str(e),
            'processing_time': error_time
        }

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if model is loaded
        if model is None or scaler is None:
            return render_template_string(HTML_TEMPLATE, result={'error': 'Model or scaler not loaded. Please restart the server.'})
        
        # Check file upload
        if 'file' not in request.files:
            return render_template_string(HTML_TEMPLATE, result={'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return render_template_string(HTML_TEMPLATE, result={'error': 'No file selected'})
        
        if file:
            # Save uploaded file temporarily
            filename = secure_filename(file.filename)
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp_file:
                tmp_file_path = tmp_file.name
                file.save(tmp_file_path)
            
            try:
                # Identify plant using fixed method
                result = identify_plant_fixed(tmp_file_path)
                return render_template_string(HTML_TEMPLATE, result=result)
            finally:
                # Clean up temporary file (with Windows-compatible error handling)
                try:
                    os.unlink(tmp_file_path)
                except PermissionError:
                    # On Windows, sometimes the file is still locked
                    import time
                    time.sleep(0.1)  # Brief delay
                    try:
                        os.unlink(tmp_file_path)
                    except PermissionError:
                        # If still locked, ignore - temp files will be cleaned up eventually
                        pass
    
    return render_template_string(HTML_TEMPLATE, result=None)

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Plant Recognition</title>
    <style>
    body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
            min-height: 100vh;
        }
        .container {
            background: #F2E8CF;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2E7D32;
            text-align: center;
            margin-bottom: 10px;
            font-weight: 1200;
        }
        .subtitle {
            text-align: center;
            color: #4CAF50;
            font-weight: bold;
            margin-bottom: 30px;
            padding: 10px;
            background: #E8F5E8;
            border-radius: 8px;
        }
        .upload-area {
            border: 3px dashed #4CAF50;
            border-radius: 10px;
            padding: 30px;
            text-align: center;
            margin-bottom: 30px;
            transition: all 0.3s ease;
        }
        .upload-area:hover {
            background: #F1E3C0;
        }
        .upload-button {
            background: #6A994E;
            color: white;
            padding: 12px 25px;
            border: none;
            border-radius: 20px;
            cursor: pointer;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        .upload-button:hover {
            background: #45a049;
            transform: translateY(-2px);
        }
        .result-container {
            margin-top: 30px;
            padding: 20px;
            border-radius: 10px;
            background: #F1F8E9;
            border-left: 5px solid #4CAF50;
        }
        .species-name {
            font-size: 1.8em;
            font-weight: bold;
            color: #2E7D32;
            margin-bottom: 10px;
        }
        .confidence {
            font-size: 1.2em;
            color: #666;
            margin-bottom: 15px;
        }
        .top5-list {
            margin-top: 15px;
        }
        .top5-item {
            padding: 8px;
            margin: 3px 0;
            background: white;
            border-radius: 5px;
            border-left: 3px solid #4CAF50;
        }
        .error {
            background: #FFEBEE;
            border-left-color: #F44336;
            color: #C62828;
        }
        .tech-details {
            margin-top: 20px;
            padding: 15px;
            background: #E3F2FD;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }
        #drop-zone {
            transition: background 0.2s, border-color 0.2s;

                        background-image: url('static/drag_image.jpg');
                        background-repeat : no-repeat;
                        padding: 30px;
                        border 2px dashed #386641;
                        border-radius: 10px; 
                        cursor:pointer;
                        background-size: contain;
                        background-position: center;

                        width: 300px;  
                        height: 250px;

                        margin-left: 30%;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Plant Recognition</h1>
        
        <div class="upload-area" id="upload-area">
            <h3>Upload Plant Image</h3>
            <p>Supported formats: JPG, PNG, BMP, TIFF</p>
            <div id="preview" style="text-align:center; margin-bottom:20px;"></div>
            <form method="post" enctype="multipart/form-data" id="upload-form">
                <input type="file" name="file" id="file-input" accept=".jpg,.jpeg,.png,.bmp,.tiff" required style="display:none;">
                <div id="drop-zone">
                    
                    <span id="drop-zone-text">Drag &amp; drop an image here, or <span style="color:#4CAF50; text-decoration:underline; cursor:pointer;" id="browse-link">browse</span></span>
                </div>
                <br>
                <input type="submit" value=" Identify Plant" class="upload-button">
            </form>
        </div>
        
        {% if result %}
            {% if result.error %}
                <div class="result-container error">
                    <h3> Error</h3>
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
                        <h4> Technical Details:</h4>
                        <strong> FIXED PIPELINE:</strong><br>
                        • Feature extraction: Ultra-parallel GPU (2500 features)<br>
                        • Normalization: Using SAME scaler as training<br>
                        • Prediction: Direct classification network<br>
                        • Confidence: Enhanced calibration<br>
                        <br>
                        <strong>Processing:</strong><br>
                        • Descriptor A prediction: Class {{ result.descriptor_A_class }} ({{ "%.1f"|format(result.descriptor_A_conf * 100) }}%)<br>
                        • Descriptor B prediction: Class {{ result.descriptor_B_class }} ({{ "%.1f"|format(result.descriptor_B_conf * 100) }}%)<br>
                        • Ensemble prediction: Class {{ result.predicted_class_index }} ({{ "%.1f"|format(result.confidence * 100) }}%)<br>
                        • Processing time: {{ "%.3f"|format(result.processing_time) }}s<br>
                        • Pipeline: Training-consistent normalization <br>
                        • Scaler: Fitted on {{ "9,135" }} training samples<br>
                    </div>
                </div>
            {% endif %}
        {% endif %}
        

        
        <div id="preview" style="text-align:center; margin-bottom:20px;"></div>
<script>
document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('file-input');
    const previewDiv = document.getElementById('preview');
    const dropZone = document.getElementById('drop-zone');
    const browseLink = document.getElementById('browse-link');
    const dropZoneText = document.getElementById('drop-zone-text');

    fileInput.addEventListener('change', function(e) {
        previewDiv.innerHTML = '';
        const file = e.target.files[0];
        if (file) {
            const img = document.createElement('img');
            img.style.maxWidth = '400px';
            img.style.borderRadius = '10px';
            img.style.boxShadow = '0 4px 16px rgba(0,0,0,0.12)';
            img.src = URL.createObjectURL(file);
            previewDiv.appendChild(img);
        }
    });

    dropZone.addEventListener('click', function() {
        fileInput.click();
    });
    browseLink.addEventListener('click', function(e) {
        e.stopPropagation();
        fileInput.click();
    });

    dropZone.addEventListener('dragover', function(e) {
        e.preventDefault();
        dropZone.style.borderColor = '#388E3C';
    });

    dropZone.addEventListener('dragleave', function(e) {
        e.preventDefault();
        dropZone.style.borderColor = '#4CAF50';
    });

    dropZone.addEventListener('drop', function(e) {
        e.preventDefault();
        dropZone.style.borderColor = '#4CAF50';
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            // Trigger change event for preview
            fileInput.dispatchEvent(new Event('change'));
        }
    });
});
</script>
    </div>
</body>
</html>
"""

if __name__ == '__main__':
    print(" PLANT RECOGNITION WEB SERVER")
    print("=" * 50)
    print(" Loading model and scaler...")
    
    success = load_model_and_scaler()
    if not success:
        print("\n CRITICAL ERROR: Model or scaler loading failed!")
        print("   The model needs to be retrained with scaler saving.")
        print("   Run: python training.py")
        exit(1)
    
    print(f"\n Server ready with fixed scaler handling!")
    print(f"   Model: {len(model_data['class_names'])} classes")
    print(f"   Scaler: Fitted StandardScaler from training")
    print(f"   URL: http://localhost:5000")
    print(f"\n Starting server...")
    
    app.run(debug=True, host='0.0.0.0', port=5000)