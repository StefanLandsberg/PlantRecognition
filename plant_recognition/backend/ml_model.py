import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
import sys
import json

# MLP Head class definition
class MLPHead(torch.nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, num_classes)
        )
    def forward(self, x): 
        return self.net(x)

# Configuration
IMAGE_SIZE = 224
MODEL_PATH = "../models/best_mlp_1000class.pt"
CLASS_NAMES_PATH = "../models/class_names.txt"

# Validate file paths
import os
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}")

class PlantMLPRecognizer:
    def __init__(self, device="cpu"):
        try:
            # Load class names
            with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
                self.class_names = [line.strip() for line in f.readlines()]
            
            if not self.class_names:
                raise ValueError("Class names file is empty")
            
            self.num_classes = len(self.class_names)
            self.device = torch.device(device)
            
            # Load model
            self.model = MLPHead(512, self.num_classes)
            try:
                self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            except Exception as e:
                raise RuntimeError(f"Failed to load model: {e}")
            
            self.model.eval()
            self.model.to(self.device)
            
            # Initialize feature extractor
            try:
                resnet = models.resnet18(weights='IMAGENET1K_V1')
                self.feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1]).to(self.device)
                self.feature_extractor.eval()
            except Exception as e:
                raise RuntimeError(f"Failed to initialize feature extractor: {e}")
            
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor()
            ])
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PlantMLPRecognizer: {e}")

    def extract_features(self, img_path):
        try:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")
            
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                feat = self.feature_extractor(img).squeeze().cpu().numpy()
            
            return feat
        except Exception as e:
            raise RuntimeError(f"Failed to extract features from {img_path}: {e}")

    def predict(self, img_path):
        try:
            features = self.extract_features(img_path)
            feat_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.model(feat_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = probs.argmax()
                
                # Validate prediction index
                if pred_idx >= len(self.class_names):
                    raise ValueError(f"Invalid prediction index: {pred_idx}")
                
                pred_name = self.class_names[pred_idx]
                top5_idx = np.argsort(probs)[::-1][:5]
                top5 = [(self.class_names[i], float(probs[i])) for i in top5_idx]
            
            return {
                "predicted_species": pred_name,
                "confidence": float(probs[pred_idx]),
                "top5_predictions": top5,
                "predicted_class_index": int(pred_idx)
            }
        except Exception as e:
            raise RuntimeError(f"Failed to predict for {img_path}: {e}")

def main():
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Usage: python ml_model.py <image_path>"}))
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    try:
        recognizer = PlantMLPRecognizer()
        result = recognizer.predict(image_path)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()