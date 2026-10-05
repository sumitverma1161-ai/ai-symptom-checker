"""
train_image_model.py
---------------------
Fine-tunes a MobileNetV2 CNN (transfer learning, PyTorch) on medical/skin
image datasets to predict dermatological and general medical conditions.

Supported dataset layouts (place images in ml_models/data/images/<dataset>/):

  1. Google SCIN (Skin Condition Image Network)
     https://github.com/google-research-datasets/scin
     Folder: ml_models/data/images/scin/
     Structure: scin/<label>/<image>.jpg

  2. The Cancer Imaging Archive (TCIA) — sample sets
     https://www.cancerimagingarchive.net/
     Folder: ml_models/data/images/tcia/
     Structure: tcia/<label>/<image>.jpg  (after conversion from DICOM)

  3. MedImg / CUI Lab
     https://www.cuilab.cn/medimg/
     Folder: ml_models/data/images/medimg/
     Structure: medimg/<label>/<image>.jpg

  4. Any custom folder: ml_models/data/images/<dataset_name>/<label>/<img>

Usage:
    # CPU-only (slow but works anywhere)
    python ml_models/train_image_model.py --epochs 20 --batch_size 16

    # GPU accelerated
    python ml_models/train_image_model.py --epochs 30 --batch_size 32 --device cuda

    # Fine-tune only top layer (fast test)
    python ml_models/train_image_model.py --freeze_base --epochs 10

Outputs:
    ml_models/image_model.pth       — saved model state dict
    ml_models/image_classes.pkl     — ordered class names
    ml_models/image_model_meta.json — training accuracy & class info
"""

import argparse
import json
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from torchvision.models import MobileNet_V2_Weights

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data" / "images"
OUT_DIR = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────
# Image transforms
# ─────────────────────────────────────────────────────────────────
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────
# Build combined ImageFolder dataset from all dataset sub-dirs
# ─────────────────────────────────────────────────────────────────

def build_combined_dataset(data_dir: Path):
    """
    Collect images from all sub-directories (scin/, tcia/, medimg/, …).
    Each sub-directory must follow ImageFolder layout: <label>/<image_file>.
    Returns a torch ImageFolder with merged classes.
    """
    subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(
            f"No image dataset directories found in {data_dir}.\n"
            "Create sub-directories with ImageFolder layout: "
            "<dataset_name>/<label>/<image_file>\n"
            "Supported datasets: scin, tcia, medimg, or any custom folder."
        )

    # Use the first available dataset directory that has class subfolders
    for subdir in subdirs:
        class_dirs = [d for d in subdir.iterdir() if d.is_dir()]
        if class_dirs:
            print(f"[Image dataset] Using: {subdir.name} ({len(class_dirs)} classes)")
            dataset_full = datasets.ImageFolder(root=str(subdir), transform=TRAIN_TRANSFORMS)
            dataset_eval = datasets.ImageFolder(root=str(subdir), transform=EVAL_TRANSFORMS)
            return dataset_full, dataset_eval, dataset_full.classes

    raise FileNotFoundError(
        f"No valid ImageFolder-layout subdirectory found in {data_dir}."
    )


# ─────────────────────────────────────────────────────────────────
# Build MobileNetV2 model
# ─────────────────────────────────────────────────────────────────

def build_model(num_classes: int, freeze_base: bool = False) -> nn.Module:
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

    if freeze_base:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    return model


# ─────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, device, epochs, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.3)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_acc = correct / total

        # ── Validate ───────────────────────────────────────────
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_acc = val_correct / val_total

        scheduler.step()

        print(
            f"Epoch [{epoch:>3}/{epochs}] "
            f"Loss: {running_loss/total:.4f} | "
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    return best_state, best_val_acc


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(DATA_DIR, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Using device: {device}")

    # Load datasets
    dataset_train, dataset_eval, classes = build_combined_dataset(DATA_DIR)
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")

    # Split train/val (85/15)
    n_val = max(1, int(0.15 * len(dataset_train)))
    n_train = len(dataset_train) - n_val
    train_subset, _ = random_split(dataset_train, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    _, val_subset = random_split(dataset_eval, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Build model
    model = build_model(num_classes=num_classes, freeze_base=args.freeze_base).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # Train
    best_state, best_val_acc = train_model(
        model, train_loader, val_loader, device, args.epochs, lr=args.lr
    )

    # Save
    model.load_state_dict(best_state)
    torch.save(best_state, OUT_DIR / "image_model.pth")

    with open(OUT_DIR / "image_classes.pkl", "wb") as f:
        pickle.dump(classes, f)

    meta = {
        "model_type": "MobileNetV2 (fine-tuned, PyTorch)",
        "num_classes": num_classes,
        "classes": classes,
        "input_size": [224, 224],
        "best_val_accuracy": round(float(best_val_acc), 4),
        "epochs_trained": args.epochs,
        "freeze_base": args.freeze_base,
        "datasets": [
            {
                "name": "Google SCIN",
                "url": "https://github.com/google-research-datasets/scin",
                "folder": "ml_models/data/images/scin/"
            },
            {
                "name": "The Cancer Imaging Archive (TCIA)",
                "url": "https://www.cancerimagingarchive.net/",
                "folder": "ml_models/data/images/tcia/"
            },
            {
                "name": "MedImg / CUI Lab",
                "url": "https://www.cuilab.cn/medimg/",
                "folder": "ml_models/data/images/medimg/"
            },
        ],
    }
    with open(OUT_DIR / "image_model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Saved artifacts to {OUT_DIR}/")
    print(f"  image_model.pth")
    print(f"  image_classes.pkl   ({num_classes} classes)")
    print(f"  image_model_meta.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileNetV2 medical image classifier")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--freeze_base", action="store_true", help="Freeze MobileNetV2 base layers")
    args = parser.parse_args()
    main(args)
