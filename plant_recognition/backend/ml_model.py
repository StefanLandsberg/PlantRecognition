import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import sys
import json
import torchvision.models as models

class BotanicalHierarchyAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 1. Molecular-level attention (finest details: chlorophyll, chemistry)
        self.molecular_att = nn.Sequential(
            nn.Conv2d(channels, channels//8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//8, 1, 1),
            nn.Sigmoid()
        )
        
        # 2. Cellular-level attention (stomata, trichomes, cell walls)
        self.cellular_att = nn.Sequential(
            nn.Conv2d(channels, channels//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//4, 1, 1),
            nn.Sigmoid()
        )
        
        # 3. Tissue-level attention (vascular bundles, margins, surface textures)
        self.tissue_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(16),  # Focus on 16x16 regions
            nn.Conv2d(channels, channels//6, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//6, 1, 3, padding=1),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
        )
        
        # 4. Organ-level attention (individual leaves, flowers, stems)
        self.organ_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),  # Focus on 8x8 regions
            nn.Conv2d(channels, channels//4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//4, 1, 3, padding=1),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        )
        
        # 5. Structure-level attention (branch patterns, leaf arrangement)
        self.structure_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),  # Focus on 4x4 regions
            nn.Conv2d(channels, channels//6, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//6, 1, 3, padding=1),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )
        
        # 6. Architecture-level attention (overall plant form, growth habit)
        self.architecture_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(2),  # Focus on 2x2 regions
            nn.Conv2d(channels, channels//8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//8, 1, 1),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        # 7. Ecological-level attention (global context, environmental adaptations)
        self.ecological_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//8, channels, 1),
            nn.Sigmoid()
        )
        
        # Advanced hierarchical fusion weights (7 levels)
        self.fusion_weights = nn.Parameter(torch.tensor([0.20, 0.18, 0.16, 0.16, 0.14, 0.10, 0.06]))  # molecular->ecological
    
    def forward(self, x):
        # GPU-optimized parallel computation of all 7 hierarchical attention maps
        with torch.cuda.amp.autocast(enabled=(x.device.type == 'cuda')):
            # Compute all attention maps in parallel for maximum GPU utilization
            molecular = self.molecular_att(x)      # Finest chemical details
            cellular = self.cellular_att(x)        # Cell structures
            tissue = self.tissue_att(x)           # Tissue patterns
            organ = self.organ_att(x)             # Organ features
            structure = self.structure_att(x)      # Structural patterns  
            architecture = self.architecture_att(x) # Plant architecture
            ecological = self.ecological_att(x)    # Global ecological context
        
        # Fast tensor operations for spatial dimension matching
        H, W = x.shape[2], x.shape[3]
        attention_maps = [molecular, cellular, tissue, organ, structure, architecture, ecological]
        
        # Vectorized interpolation for speed
        for i, att_map in enumerate(attention_maps):
            if att_map.shape[2] != H or att_map.shape[3] != W:
                attention_maps[i] = torch.nn.functional.interpolate(
                    att_map, size=(H, W), mode='bilinear', align_corners=False
                )
        
        # Pre-computed softmax weights for faster inference
        weights = torch.softmax(self.fusion_weights, dim=0)
        
        # Optimized tensor fusion using torch.stack for parallel ops
        att_stack = torch.stack(attention_maps, dim=0)  # [7, batch, 1, H, W]
        weights_expanded = weights.view(7, 1, 1, 1, 1)  # Broadcast shape
        hierarchical_att = torch.sum(att_stack * weights_expanded, dim=0)
        
        # Apply comprehensive botanical attention to features
        attended = x * hierarchical_att
        
        return attended, hierarchical_att

class CustomFeatureExtractor(nn.Module):
    def __init__(self, target_features=1024, input_channels=3):
        super().__init__()
        self.target_features = target_features
        
        # Multi-scale feature extraction backbone
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # Progressive feature extraction with BHA at each level
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.bha1 = BotanicalHierarchyAttention(128)
        
        self.block2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.bha2 = BotanicalHierarchyAttention(256)
        
        self.block3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.bha3 = BotanicalHierarchyAttention(512)
        
        self.final_conv = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        # Adaptive feature pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature projection with botanical knowledge
        self.projector = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1024, target_features),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Stem processing
        x = self.stem(x)
        
        # Progressive feature extraction with hierarchical attention
        x = self.block1(x)
        x, att1 = self.bha1(x)
        
        x = self.block2(x)
        x, att2 = self.bha2(x)
        
        x = self.block3(x)
        x, att3 = self.bha3(x)
        
        x = self.final_conv(x)
        
        # Global pooling and projection
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        features = self.projector(x)
        
        return features

class EndToEndPlantClassifier(nn.Module):
    def __init__(self, num_classes, feature_dim=1024):
        super().__init__()
        
        self.feature_extractor = CustomFeatureExtractor(target_features=feature_dim)
        
        self.proven_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 1280),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.proven_head(features)
        return logits
    
    def extract_features(self, x):
        features = self.feature_extractor(x)
        return features
    
    @torch.inference_mode()
    def predict_fast(self, x, top_k=5):
        # Mixed precision for maximum GPU speed
        with torch.cuda.amp.autocast(enabled=(x.device.type == 'cuda')):
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            top_probs, top_indices = torch.topk(probs, min(top_k, logits.size(1)), dim=1)
        return top_indices, top_probs

IMAGE_SIZE = 512
MODEL_PATH = "../models/best_end_to_end_model.pt"
CLASS_NAMES_PATH = "../models/class_names.txt"

import os
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}")

class PlantMLPRecognizer:
    def __init__(self, device="auto"):
        try:
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.device = torch.device(device)
            
            with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
                self.class_names = [line.strip() for line in f.readlines()]
            
            if not self.class_names:
                raise ValueError("Class names file is empty")
            
            self.num_classes = len(self.class_names)
            
            try:
                self.model = EndToEndPlantClassifier(
                    num_classes=self.num_classes, 
                    feature_dim=1024
                )
                self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
                self.model.eval()
                self.model.to(self.device)
                
                # Disable gradients for faster inference
                for param in self.model.parameters():
                    param.requires_grad_(False)
                
                # GPU optimizations for maximum speed
                if self.device.type == 'cuda':
                    # Compile model for faster inference (PyTorch 2.0+)
                    try:
                        self.model = torch.compile(self.model, mode='max-autotune')
                    except:
                        pass  # Fallback if compile not available
                    
                    # Set optimal GPU settings
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.deterministic = False
                
                print(f"✓ Loaded optimized end-to-end model with {self.num_classes} classes on {self.device}")
                
            except Exception as e:
                raise RuntimeError(f"Failed to load end-to-end model: {e}")
            
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PlantMLPRecognizer: {e}")

    def extract_features(self, img_path):
        try:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")
            
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device, non_blocking=True)
            
            with torch.inference_mode():
                features = self.model.feature_extractor(self.model.backbone(img_tensor))
                feat = features.squeeze().cpu().numpy()
            
            return feat
        except Exception as e:
            raise RuntimeError(f"Failed to extract features from {img_path}: {e}")

    def predict(self, img_path):
        try:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")
            
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                with torch.inference_mode():
                    top_indices, top_probs = self.model.predict_fast(img_tensor, top_k=5)
                    
                    top_indices = top_indices.cpu().squeeze().tolist()
                    top_probs = top_probs.cpu().squeeze().tolist()
                    
                    if not isinstance(top_indices, list):
                        top_indices = [top_indices]
                        top_probs = [top_probs]
            
            pred_idx = top_indices[0]
            if pred_idx >= len(self.class_names):
                raise ValueError(f"Invalid prediction index: {pred_idx}")
            
            pred_name = self.class_names[pred_idx]
            confidence = top_probs[0]
            
            top5 = []
            for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
                if idx < len(self.class_names):
                    top5.append((self.class_names[idx], float(prob)))
            
            # Simple output format for LLM
            return {
                "predicted_species": pred_name,
                "confidence": float(confidence),
                "top5_predictions": top5,
                "predicted_class_index": int(pred_idx)
            }
        except Exception as e:
            return {
                "error": str(e),
                "predicted_species": None,
                "confidence": 0.0
            }

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