import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from pathlib import Path
import numpy as np
import time
import json
import random
from unified_plant_augmentation import UnifiedPlantAugmentationEngine
from plant_feature_extractor import create_feature_extractor

# --- CONFIGURATION ---
DATA_DIR = "../data/plant_images"   # Updated path
IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 100      # Use early stopping!
AUGMENT_COUNT = 30
VAL_SPLIT = 0.2   # % of data for validation
PATIENCE = 8      # Early stopping patience (epochs)
LR = 2e-3

# Validate data directory exists
if not Path(DATA_DIR).exists():
    raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

# Augmentation split: percentage of realistic vs seasonal/edge
AUGMENT_REALISTIC_FRAC = 0.6  # 60% realistic (lighting, angles, basic, user_realistic)
AUGMENT_SEASONAL_FRAC = 0.4   # 40% seasonal/edge (seasonal, plant, weather)

# --- Feature Extraction Dataset ---
class FeatureDataset(Dataset):
    def __init__(self, root_dir, cnn_feature_extractor, augment_count=1):
        self.samples = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = []
        self.cnn_feature_extractor = cnn_feature_extractor
        self.augment_count = augment_count
        
        # Validate root directory
        root_dir = Path(root_dir)
        if not root_dir.exists():
            raise FileNotFoundError(f"Root directory not found: {root_dir}")
        if not root_dir.is_dir():
            raise NotADirectoryError(f"Root path is not a directory: {root_dir}")
        
        # Load samples
        for class_idx, class_dir in enumerate(sorted(root_dir.iterdir())):
            if class_dir.is_dir():
                class_name = class_dir.name
                self.class_to_idx[class_name] = class_idx
                self.idx_to_class.append(class_name)
                for img_file in class_dir.glob("*.*"):
                    if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                        self.samples.append(str(img_file))
                        self.labels.append(class_idx)
        
        if not self.samples:
            raise ValueError(f"No valid image files found in {root_dir}")
        
        self.samples = np.array(self.samples)
        self.labels = np.array(self.labels)
        self.num_classes = len(self.idx_to_class)

        # Augmentation engine
        self.aug_engine = UnifiedPlantAugmentationEngine(image_size=IMAGE_SIZE)
        self.aug_realistic_frac = AUGMENT_REALISTIC_FRAC
        self.aug_seasonal_frac = AUGMENT_SEASONAL_FRAC

        # Realistic methods
        self.realistic_methods = ['realistic']
        # Seasonal & edge methods
        self.seasonal_methods = ['seasonal']

    def __len__(self):
        return len(self.samples) * self.augment_count

    def __getitem__(self, idx):
        try:
            img_idx = idx // self.augment_count
            aug_idx = idx % self.augment_count
            img_path = self.samples[img_idx]
            label = self.labels[img_idx]
            
            # Validate image file exists
            if not Path(img_path).exists():
                raise FileNotFoundError(f"Image file not found: {img_path}")
            
            img = Image.open(img_path).convert('RGB')

            # Decide whether to use realistic or seasonal/edge
            if random.random() < self.aug_realistic_frac:
                method = random.choice(self.realistic_methods)
            else:
                method = random.choice(self.seasonal_methods)

            aug_img = self.aug_engine.generate_single_augmentation(img, method)

            with torch.no_grad():
                feat = self.cnn_feature_extractor(aug_img.unsqueeze(0)).squeeze()
            return feat, label
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            # Return a zero tensor as fallback
            return torch.zeros(512), 0

# --- MLP Head ---
class MLPHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.net(x)

def top5_accuracy(logits, labels):
    top5 = logits.topk(5, dim=1).indices
    correct = top5.eq(labels.unsqueeze(1)).sum().item()
    return correct / labels.size(0)

if __name__ == "__main__":
    # --- Custom Feature Extractor ---
    feature_extractor = create_feature_extractor('cnn', out_features=1024)
    feature_extractor.eval()
    # Keep on CPU for feature extraction

    # --- Dataset & Loader ---
    full_dataset = FeatureDataset(DATA_DIR, cnn_feature_extractor=feature_extractor, augment_count=AUGMENT_COUNT)
    NUM_CLASSES = full_dataset.num_classes

    # Save index-to-class mapping for later inference
    with open("../models/class_names.txt", "w", encoding="utf-8") as f:
        for name in full_dataset.idx_to_class:
            f.write(name + "\n")

    val_len = int(len(full_dataset) * VAL_SPLIT)
    train_len = len(full_dataset) - val_len
    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    # --- Model, Loss, Optimizer, Scheduler ---
    mlp = MLPHead(1024, NUM_CLASSES).cuda()  # Changed from 512 to 1024 to match our feature extractor
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    # --- Early Stopping ---
    best_val_acc = 0
    epochs_since_improve = 0

    print("\n[Training MLP for Few-Shot {} Classes]".format(NUM_CLASSES))
    for epoch in range(EPOCHS):
        mlp.train()
        epoch_start = time.time()
        train_loss, train_correct, train_total, train_top5 = 0, 0, 0, 0

        for feats, labels in train_loader:
            feats = feats.cuda()
            labels = labels.cuda().long()   # <-- Ensure labels are long
            logits = mlp(feats.view(feats.size(0), -1))
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            train_loss += loss.item() * labels.size(0)
            train_correct += (preds == labels).sum().item()
            train_top5 += logits.topk(5, dim=1).indices.eq(labels.unsqueeze(1)).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total if train_total > 0 else 0
        train_loss /= train_total
        train_top5_acc = train_top5 / train_total if train_total > 0 else 0

        # --- Validation ---
        mlp.eval()
        val_loss, val_correct, val_total, val_top5 = 0, 0, 0, 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats = feats.cuda()
                labels = labels.cuda().long()  # <-- Ensure labels are long
                logits = mlp(feats.view(feats.size(0), -1))
                loss = criterion(logits, labels)
                preds = torch.argmax(logits, dim=1)
                val_loss += loss.item() * labels.size(0)
                val_correct += (preds == labels).sum().item()
                val_top5 += logits.topk(5, dim=1).indices.eq(labels.unsqueeze(1)).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0
        val_loss /= val_total
        val_top5_acc = val_top5 / val_total if val_total > 0 else 0

        scheduler.step(val_acc)

        # --- Logging essential info ---
        print(
            f"Epoch {epoch+1:03d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train Top5: {train_top5_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Top5: {val_top5_acc:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.4g} | Time: {time.time()-epoch_start:.2f}s"
        )

        # --- Early Stopping Logic ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_since_improve = 0
            torch.save(mlp.state_dict(), "../models/best_mlp_1000class.pt")
            # Save mapping again in case the dataset changes
            with open("../models/class_names.txt", "w", encoding="utf-8") as f:
                for name in full_dataset.idx_to_class:
                    f.write(name + "\n")
        else:
            epochs_since_improve += 1

        if epochs_since_improve >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}. Best Val Acc: {best_val_acc:.4f}")
            break