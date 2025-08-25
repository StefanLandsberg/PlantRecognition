# modeltraining/training.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models
from PIL import Image
from pathlib import Path
import random
from unified_plant_augmentation import UnifiedPlantAugmentationEngine

# Simple, proven configuration
DATA_DIR = "data/plant_images"
IMAGE_SIZE = 512  # Match ml_model.py
BATCH_SIZE = 8  # Lower for 512px images
EPOCHS = 50
LR = 1e-4  # Much higher learning rate
VAL_SPLIT = 0.2
AUGMENT_COUNT = 10

class PlantDataset(Dataset):
    def __init__(self, root_dir, augment_count=10):
        self.samples = []
        self.labels = []
        self.classes = []
        
        root_dir = Path(root_dir)
        for class_idx, class_dir in enumerate(sorted(root_dir.iterdir())):
            if class_dir.is_dir():
                self.classes.append(class_dir.name)
                for img_file in class_dir.glob("*.*"):
                    if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                        self.samples.append(str(img_file))
                        self.labels.append(class_idx)
        
        self.augment_count = augment_count
        self.aug_engine = UnifiedPlantAugmentationEngine(image_size=IMAGE_SIZE, device='cpu')
    
    def __len__(self):
        return len(self.samples) * self.augment_count
    
    def __getitem__(self, idx):
        img_idx = idx // self.augment_count
        aug_idx = idx % self.augment_count
        img_path = self.samples[img_idx]
        label = self.labels[img_idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
            
            if aug_idx == 0:
                # Original image with minimal augmentation
                img_tensor = self.aug_engine.generate_single_augmentation(img, method='realistic')
            else:
                # Apply 60/40 realistic/seasonal augmentation split
                if random.random() < 0.6:
                    aug_method = 'realistic'
                else:
                    aug_method = 'seasonal'
                
                img_tensor = self.aug_engine.generate_single_augmentation(img, method=aug_method)
            
            return img_tensor.cpu(), label
        except Exception:
            # Return properly normalized zero tensor
            zero_tensor = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
            zero_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(zero_tensor)
            return zero_tensor, label

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
        # Extract all 7 hierarchical attention maps
        molecular = self.molecular_att(x)      # Finest chemical details
        cellular = self.cellular_att(x)        # Cell structures
        tissue = self.tissue_att(x)           # Tissue patterns
        organ = self.organ_att(x)             # Organ features
        structure = self.structure_att(x)      # Structural patterns  
        architecture = self.architecture_att(x) # Plant architecture
        ecological = self.ecological_att(x)    # Global ecological context
        
        # Ensure all attention maps match input spatial dimensions
        H, W = x.shape[2], x.shape[3]
        attention_maps = [molecular, cellular, tissue, organ, structure, architecture, ecological]
        
        for i, att_map in enumerate(attention_maps):
            if att_map.shape[2] != H or att_map.shape[3] != W:
                attention_maps[i] = torch.nn.functional.interpolate(att_map, size=(H, W), mode='bilinear', align_corners=False)
        
        # Apply softmax to fusion weights for learnable hierarchy
        weights = torch.softmax(self.fusion_weights, dim=0)
        
        # Combine all 7 attention levels hierarchically
        hierarchical_att = sum(weights[i] * attention_maps[i] for i in range(7))
        
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
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.proven_head(features)
        return logits

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    dataset = PlantDataset(DATA_DIR, augment_count=AUGMENT_COUNT)
    num_classes = len(dataset.classes)
    print(f"Found {len(dataset)} samples, {num_classes} classes")
    
    # Split dataset
    train_size = int((1 - VAL_SPLIT) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Create model
    model = EndToEndPlantClassifier(num_classes, feature_dim=1024).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)

    print("Starting training...")
    best_val_acc = 0
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # Calculate metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch+1:2d} | Train: {train_acc:5.1f}% ({train_loss:.4f}) | Val: {val_acc:5.1f}% ({val_loss:.4f}) | LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), '../plant_recognition/models/best_end_to_end_model.pt')
            
            with open('../plant_recognition/models/class_names.txt', 'w', encoding='utf-8') as f:
                for name in dataset.classes:
                    f.write(name + '\n')
            print(f'  -> New best: {val_acc:.1f}%')
        
        scheduler.step(val_acc)  
    
    print(f'Training complete. Best accuracy: {best_val_acc:.1f}%')

if __name__ == "__main__":
    train()