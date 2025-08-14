# modeltraining/training.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models
from PIL import Image
from pathlib import Path
import time, random
import numpy as np

from unified_plant_augmentation import UnifiedPlantAugmentationEngine

# ---------- PATHS (file-relative, robust) ----------
HERE = Path(__file__).resolve().parent
DATA_DIR = (HERE.parent / "data_preprocessing" / "data" / "plant_images").resolve()
MODELS_DIR = (HERE.parent / "plant_recognition" / "models").resolve()
MODELS_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_DIR.exists():
    raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

# ---------- CONFIG ----------
IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 100
VAL_SPLIT = 0.2
PATIENCE = 8
LR = 2e-3
FEATURE_DIM = 512  # ResNet18 penultimate layer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- RESNET18 FEATURE EXTRACTOR (matches inference) ----------
def build_resnet_feature_extractor():
    try:
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.IMAGENET1K_V1
        resnet = models.resnet18(weights=weights)
    except Exception:
        resnet = models.resnet18(weights="IMAGENET1K_V1")
    feat = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)
    feat.eval()
    return feat

# ---------- DATASET ----------
class FeatureDataset(Dataset):
    def __init__(self, root_dir, feature_extractor, augment_count=30):
        self.root_dir = Path(root_dir)
        self.feature_extractor = feature_extractor
        self.augment_count = augment_count

        self.samples, self.labels = [], []
        self.class_to_idx = {}
        self.idx_to_class = []

        for class_idx, class_dir in enumerate(sorted(self.root_dir.iterdir())):
            if class_dir.is_dir():
                cls = class_dir.name
                self.class_to_idx[cls] = class_idx
                self.idx_to_class.append(cls)
                for img_file in class_dir.glob("*.*"):
                    if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                        self.samples.append(str(img_file))
                        self.labels.append(class_idx)

        if not self.samples:
            raise ValueError(f"No images found under {root_dir}")

        self.samples = np.array(self.samples)
        self.labels = np.array(self.labels)

        self.aug_engine = UnifiedPlantAugmentationEngine(image_size=IMAGE_SIZE)
        self.realistic_methods = ["realistic"]
        self.seasonal_methods = ["seasonal"]

    def __len__(self):
        return len(self.samples) * self.augment_count

    def __getitem__(self, idx):
        img_idx = idx // self.augment_count
        img_path = self.samples[img_idx]
        label = int(self.labels[img_idx])

        img = Image.open(img_path).convert("RGB")
        method = random.choice(self.realistic_methods if random.random() < 0.6 else self.seasonal_methods)
        tensor = self.aug_engine.generate_single_augmentation(img, method)  # CHW, normalized

        with torch.no_grad():
            t = tensor.unsqueeze(0).to(device)         # (1, 3, H, W)
            feat = self.feature_extractor(t)           # (1, 512, 1, 1)
            feat = feat.view(feat.size(0), -1).squeeze(0).float().cpu()  # (512,)
        return feat, label

# ---------- MLP HEAD (512 -> num_classes) ----------
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
            nn.Linear(128, num_classes),
        )
    def forward(self, x): return self.net(x)

def top5_accuracy(logits, labels):
    top5 = logits.topk(5, dim=1).indices
    correct = top5.eq(labels.unsqueeze(1)).sum().item()
    return correct / labels.size(0)

# ---------- TRAIN ----------
if __name__ == "__main__":
    feature_extractor = build_resnet_feature_extractor()

    full_dataset = FeatureDataset(DATA_DIR, feature_extractor=feature_extractor, augment_count=30)
    NUM_CLASSES = len(full_dataset.idx_to_class)

    # Save class names where inference expects them
    with open(MODELS_DIR / "class_names.txt", "w", encoding="utf-8") as f:
        for name in full_dataset.idx_to_class:
            f.write(name + "\n")

    val_len = int(len(full_dataset) * VAL_SPLIT)
    train_len = len(full_dataset) - val_len
    train_ds, val_ds = random_split(full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    # On Windows: num_workers=0 avoids multiprocessing issues
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

    mlp = MLPHead(FEATURE_DIM, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)  # no verbose=

    best_val_acc = 0.0
    epochs_since_improve = 0

    print(f"\n[Training MLP on {NUM_CLASSES} classes | device={device}]")
    for epoch in range(EPOCHS):
        mlp.train()
        ep_start = time.time()
        train_loss = train_correct = train_total = train_top5 = 0

        for feats, labels in train_loader:
            feats = feats.to(device)
            labels = labels.to(device).long()

            logits = mlp(feats)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            train_loss += loss.item() * labels.size(0)
            train_correct += (preds == labels).sum().item()
            train_top5   += logits.topk(5, dim=1).indices.eq(labels.unsqueeze(1)).sum().item()
            train_total  += labels.size(0)

        train_acc = train_correct / train_total
        train_loss /= train_total
        train_top5_acc = train_top5 / train_total

        # ---- VALIDATION ----
        mlp.eval()
        val_loss = val_correct = val_total = val_top5 = 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats = feats.to(device)
                labels = labels.to(device).long()
                logits = mlp(feats)
                loss = criterion(logits, labels)

                preds = logits.argmax(dim=1)
                val_loss   += loss.item() * labels.size(0)
                val_correct += (preds == labels).sum().item()
                val_top5   += logits.topk(5, dim=1).indices.eq(labels.unsqueeze(1)).sum().item()
                val_total  += labels.size(0)

        val_acc = val_correct / val_total if val_total else 0.0
        val_loss /= val_total if val_total else 1.0
        val_top5_acc = val_top5 / val_total if val_total else 0.0

        scheduler.step(val_acc)

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train Top5: {train_top5_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Top5: {val_top5_acc:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.4g} | Time: {time.time()-ep_start:.1f}s"
        )

        # ---- Early stopping + save best ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_since_improve = 0
            torch.save(mlp.state_dict(), MODELS_DIR / "best_mlp_1000class.pt")
            with open(MODELS_DIR / "class_names.txt", "w", encoding="utf-8") as f:
                for name in full_dataset.idx_to_class:
                    f.write(name + "\n")
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}. Best Val Acc: {best_val_acc:.4f}")
                break
