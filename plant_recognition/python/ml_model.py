#!/usr/bin/env python3
import os
import sys
import json
from pathlib import Path

import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image


# ----------------- Config -----------------
IMAGE_SIZE = 224
HERE = Path(__file__).resolve().parent

def _resolve_existing(*candidates):
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    return None

# 1) Prefer env
env_model = os.getenv("ML_MODEL_PATH")
env_labels = os.getenv("ML_CLASS_NAMES_PATH")

# 2) Common fallbacks
fallback_models = [
    HERE.parent / "models" / "best_mlp_1000class.pt",     # plant_recognition/models
    Path("models/best_mlp_1000class.pt"),                 # repo-root/models
]
fallback_labels = [
    HERE.parent / "models" / "class_names.txt",
    Path("models/class_names.txt"),
]

MODEL_PATH = _resolve_existing(env_model, *fallback_models)
CLASS_NAMES_PATH = _resolve_existing(env_labels, *fallback_labels)

if MODEL_PATH is None:
    raise FileNotFoundError("Model file not found. Set ML_MODEL_PATH or place best_mlp_1000class.pt under plant_recognition/models.")
if CLASS_NAMES_PATH is None:
    raise FileNotFoundError("class_names.txt not found. Set ML_CLASS_NAMES_PATH or place it under plant_recognition/models.")



# ----------------- Model Head -----------------
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
            torch.nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ----------------- Recognizer -----------------
class PlantMLPRecognizer:
    def __init__(self, device=None):
        # device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # class names
        if not CLASS_NAMES_PATH.exists():
            raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}")
        with CLASS_NAMES_PATH.open(encoding="utf-8") as f:
            self.class_names = [line.strip() for line in f if line.strip()]
        if not self.class_names:
            raise ValueError("Class names file is empty")
        self.num_classes = len(self.class_names)

        # feature extractor (ResNet18 penultimate layer)
        try:
            # TorchVision weights API compatibility (new & old)
            try:
                from torchvision.models import ResNet18_Weights
                weights = ResNet18_Weights.IMAGENET1K_V1
                resnet = models.resnet18(weights=weights)
            except Exception:
                # older torchvision fallback
                resnet = models.resnet18(weights="IMAGENET1K_V1")
            self.feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1]).to(self.device)
            self.feature_extractor.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize feature extractor: {e}")

        # head
        self.model = MLPHead(512, self.num_classes).to(self.device)
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        try:
            state = torch.load(MODEL_PATH, map_location=self.device)
            # sanity-check final layer size if state contains weight
            out_w = state.get("net.6.weight")  # last Linear weight
            if out_w is not None and out_w.shape[0] != self.num_classes:
                raise RuntimeError(
                    f"Model head out_features ({out_w.shape[0]}) does not match class_names length ({self.num_classes})."
                )
            self.model.load_state_dict(state, strict=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

        self.model.eval()

        # transforms (ImageNet normalization expected by ResNet)
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # optional perf tweaks
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        self.use_half = self.device.type == "cuda"

    def extract_features(self, img_path: str):
        p = Path(img_path)
        if not p.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")

        img = Image.open(p).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.use_half:
                tensor = tensor.half()
                self.feature_extractor.half()
            feat = self.feature_extractor(tensor)  # (1, 512, 1, 1)
            feat = feat.view(feat.size(0), -1).squeeze(0).float().cpu().numpy()  # (512,)
        return feat

    def predict(self, img_path: str):
        features = self.extract_features(img_path)
        feat_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device).float()

        with torch.no_grad():
            if self.use_half:
                feat_tensor = feat_tensor.half()
                self.model.half()

            logits = self.model(feat_tensor)              # (1, num_classes)
            probs = torch.softmax(logits.float(), dim=1)  # keep in float for safety
            probs_np = probs.cpu().numpy()[0]
            pred_idx = int(probs_np.argmax())

            if pred_idx >= len(self.class_names):
                raise ValueError(f"Invalid prediction index: {pred_idx}")

            pred_name = self.class_names[pred_idx]
            top5_idx = np.argsort(probs_np)[::-1][:5]
            top5 = [(self.class_names[i], float(probs_np[i])) for i in top5_idx]

        return {
            "predicted_species": pred_name,
            "confidence": float(probs_np[pred_idx]),
            "top5_predictions": top5,
            "predicted_class_index": pred_idx,
        }


# ----------------- CLI -----------------
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
