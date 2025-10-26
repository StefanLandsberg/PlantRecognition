import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

warnings.filterwarnings("ignore")

IMAGE_SIZE = 512
MODEL_PATH = "../models/best_end_to_end_model.pt"
CLASS_NAMES_PATH = "../models/class_names.txt"
DEFAULT_MIN_CONFIDENCE = 0.5

_cached_model = None
_cached_class_names = None


def load_class_names():
    global _cached_class_names
    if _cached_class_names is not None:
        return _cached_class_names

    try:
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as fh:
            _cached_class_names = [line.strip() for line in fh if line.strip()]
            return _cached_class_names
    except Exception as exc:
        print(json.dumps({"error": f"Failed to load class names: {exc}"}))
        sys.exit(1)


class BotanicalHierarchyAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.molecular_att = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, 1),
            nn.Sigmoid(),
        )
        self.cellular_att = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid(),
        )
        self.tissue_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(16),
            nn.Conv2d(channels, channels // 6, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 6, 1, 3, padding=1),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=16, mode="bilinear", align_corners=False),
        )
        self.organ_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            nn.Conv2d(channels, channels // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 3, padding=1),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False),
        )
        self.structure_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(channels, channels // 6, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 6, 1, 3, padding=1),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
        )
        self.architecture_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(2),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, 1),
            nn.Sigmoid(),
        )
        self.ecological_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid(),
        )
        self.fusion_weights = nn.Parameter(
            torch.tensor([0.20, 0.18, 0.16, 0.16, 0.14, 0.10, 0.06])
        )

    def forward(self, x):
        maps = [
            self.molecular_att(x),
            self.cellular_att(x),
            self.tissue_att(x),
            self.organ_att(x),
            self.structure_att(x),
            self.architecture_att(x),
            self.ecological_att(x),
        ]
        height, width = x.shape[2], x.shape[3]
        for idx, att in enumerate(maps):
            if att.shape[2] != height or att.shape[3] != width:
                maps[idx] = torch.nn.functional.interpolate(
                    att, size=(height, width), mode="bilinear", align_corners=False
                )
        weights = torch.softmax(self.fusion_weights, dim=0)
        hierarchical_att = sum(weights[i] * maps[i] for i in range(len(maps)))
        attended = x * hierarchical_att
        return attended, hierarchical_att


class CustomFeatureExtractor(nn.Module):
    def __init__(self, target_features=1024, input_channels=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.bha1 = BotanicalHierarchyAttention(128)
        self.block2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.bha2 = BotanicalHierarchyAttention(256)
        self.block3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.bha3 = BotanicalHierarchyAttention(512)
        self.final_conv = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projector = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(1024, target_features), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x, _ = self.bha1(x)
        x = self.block2(x)
        x, _ = self.bha2(x)
        x = self.block3(x)
        x, _ = self.bha3(x)
        x = self.final_conv(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        features = self.projector(x)
        return features


class EndToEndPlantClassifier(nn.Module):
    def __init__(self, num_classes, feature_dim=1024):
        super().__init__()
        self.feature_extractor = CustomFeatureExtractor(target_features=feature_dim)
        self.proven_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.proven_head(features)
        return logits


def load_model(num_classes):
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EndToEndPlantClassifier(num_classes)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        print(json.dumps({"error": "Unsupported checkpoint format"}))
        sys.exit(1)

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device, non_blocking=True)
    if device.type == "cuda":
        model.half()
        torch.backends.cudnn.benchmark = True

    for param in model.parameters():
        param.requires_grad_(False)

    _cached_model = (model, device)
    return _cached_model


def preprocess_image(image_path):
    try:
        with Image.open(image_path) as image:
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            img_array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor.unsqueeze(0)
    except Exception as exc:
        print(json.dumps({"error": f"Failed to preprocess image: {exc}"}))
        sys.exit(1)


def load_config():
    config_path = Path("../models/ml_config.json")
    config = {"min_confidence": DEFAULT_MIN_CONFIDENCE}
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as fh:
                user_cfg = json.load(fh)
                if isinstance(user_cfg, dict):
                    config.update(user_cfg)
        except Exception as exc:
            print(f"Warning: could not parse ml_config.json: {exc}", file=sys.stderr)
    return config


def predict(model, image_tensor, class_names, device):
    config = load_config()
    min_conf = float(config.get("min_confidence", DEFAULT_MIN_CONFIDENCE))

    image_tensor = image_tensor.to(device, non_blocking=True)
    ref_dtype = next(model.parameters()).dtype
    if image_tensor.dtype != ref_dtype:
        image_tensor = image_tensor.to(ref_dtype)

    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        top_conf, top_idx = torch.max(probabilities, dim=1)
        k = min(5, probabilities.shape[1])
        top_vals, top_inds = torch.topk(probabilities, k=k, dim=1)

    confidence = float(top_conf.item())
    class_index = int(top_idx.item())
    predicted_label = (
        class_names[class_index]
        if 0 <= class_index < len(class_names)
        else f"Class_{class_index}"
    )

    is_confident = confidence >= min_conf
    final_label = predicted_label if is_confident else "Unknown species"

    top_k = []
    top_scores = top_vals.squeeze(0).tolist()
    top_indices = [int(i) for i in top_inds.squeeze(0).tolist()]
    for idx, score in zip(top_indices, top_scores):
        label = class_names[idx] if 0 <= idx < len(class_names) else f"Class_{idx}"
        top_k.append({"label": label, "confidence": float(score)})

    return {
        "predicted_species": final_label,
        "confidence": confidence,
        "predicted_class_index": class_index,
        "is_unknown": not is_confident,
        "top_k": top_k,
        "min_confidence": min_conf,
        "pipeline_version": "single-stage-1.0",
        "timestamp": time.time(),
    }


def main():
    start_time = time.time()

    if len(sys.argv) != 2:
        print(json.dumps({"error": "Usage: python ml_model.py <image_path>"}))
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(json.dumps({"error": f"Image file not found: {image_path}"}))
        sys.exit(1)

    try:
        class_names = load_class_names()
        model, device = load_model(len(class_names))
        image_tensor = preprocess_image(image_path)
        result = predict(model, image_tensor, class_names, device)
        result["processing_time"] = f"{time.time() - start_time:.3f}s"
        print(json.dumps(result))
        sys.stdout.flush()
    except Exception as exc:
        print(
            json.dumps(
                {
                    "predicted_species": "Unknown species",
                    "confidence": 0.0,
                    "error": str(exc),
                    "processing_time": f"{time.time() - start_time:.3f}s",
                }
            )
        )
        sys.stdout.flush()


if __name__ == "__main__":
    main()
