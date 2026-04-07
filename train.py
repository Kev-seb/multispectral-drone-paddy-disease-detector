import os
import json
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report

# -------------------------
# CONFIG
# -------------------------
DATA_DIR = Path("data/crop_simple")   # ✅ FIXED (IMPORTANT)
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# -------------------------
# TRANSFORMS
# -------------------------
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -------------------------
# DATA LOADER
# -------------------------
def build_dataloaders():
    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
    val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=val_transform)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

    print("\n[✓] Dataset Loaded")
    print("Train:", len(train_ds))
    print("Val:", len(val_ds))
    print("Classes:", train_ds.classes)

    # Save class map
    json.dump(train_ds.class_to_idx,
              open(MODEL_DIR / "class_map.json", "w"), indent=2)

    return train_dl, val_dl, train_ds.classes

# -------------------------
# MODEL
# -------------------------
def build_model(num_classes, device):
    model = models.efficientnet_b0(weights="DEFAULT")

    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )

    return model.to(device)

# -------------------------
# TRAIN LOOP
# -------------------------
def run_epoch(model, loader, criterion, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss, correct, total = 0, 0, 0

    with torch.set_grad_enabled(train):
        for images, labels in tqdm(loader):
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), 100 * correct / total

# -------------------------
# TRAIN
# -------------------------
def train_model(model, train_dl, val_dl, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_acc = 0

    for epoch in range(EPOCHS):
        train_loss, train_acc = run_epoch(model, train_dl, criterion, optimizer, device, True)
        val_loss, val_acc = run_epoch(model, val_dl, criterion, optimizer, device, False)

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pt")
            print("✅ Best model saved")

    return model

# -------------------------
# MAIN
# -------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nUsing device:", device)

    train_dl, val_dl, classes = build_dataloaders()

    model = build_model(len(classes), device)

    model = train_model(model, train_dl, val_dl, device)

    print("\n🎉 Training Complete!")

if __name__ == "__main__":
    main()