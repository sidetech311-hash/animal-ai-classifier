import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# -------------------------------------------------
# Import the model builder you already have (model.py)
# -------------------------------------------------
from model import build_model


def make_sampler(dataset):
    """
    Return a WeightedRandomSampler that balances the classes.
    This prevents the network from biasing toward the majority class.
    """
    # Count how many samples belong to each class
    class_counts = [len([s for s in dataset.samples if s[1] == i])
                    for i in range(len(dataset.classes))]
    # Inverse frequency → weight
    class_weights = [1.0 / c if c > 0 else 0.0 for c in class_counts]
    sample_weights = [class_weights[label] for _, label in dataset.samples]
    return WeightedRandomSampler(sample_weights,
                                 num_samples=len(sample_weights),
                                 replacement=True)


def train(
    data_dir: str,
    model_out: str,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 5e-4,
    device: str = None,
):
    """
    Train a ResNet‑18 (pre‑trained on ImageNet) to classify cats vs. dogs.
    """
    # -------------------------------------------------
    # 0️⃣  Device selection
    # -------------------------------------------------
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Using device: {device}")

    # -------------------------------------------------
    # 1️⃣  Locate train/ and val/ folders
    # -------------------------------------------------
    train_dir = os.path.join(data_dir, "train")
    val_dir   = os.path.join(data_dir, "val")

    if not (os.path.isdir(train_dir) and os.path.isdir(val_dir)):
        raise RuntimeError("Both train/ and val/ directories must exist!")

    # -------------------------------------------------
    # 2️⃣  Transforms (same normalization for train/val)
    # -------------------------------------------------
    common_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225],
            ),
        ]
    )
    # Add a little augmentation for the training set only
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            common_tf,
        ]
    )

    # -------------------------------------------------
    # 3️⃣  Load datasets
    # -------------------------------------------------
    train_set = ImageFolder(train_dir, transform=train_tf)
    val_set   = ImageFolder(val_dir,   transform=common_tf)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=make_sampler(train_set),   # balanced sampling
        num_workers=4,
        pin_memory=True,
        shuffle=False,                    # sampler already shuffles
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    num_classes = len(train_set.classes)
    print(f"🔢  Classes ({num_classes}): {train_set.classes}")

    # -------------------------------------------------
    # 4️⃣  Model – pre‑trained ResNet‑18, fine‑tune the head
    # -------------------------------------------------
    model = build_model(num_classes=num_classes, pretrained=True)
    model = model.to(device)

    # -----------------------------------------------------------------
    # 5️⃣  Loss, optimizer, LR scheduler
    # -----------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Cosine‑Annealing gives a smooth decay of LR across epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # -------------------------------------------------
    # 6️⃣  Training loop
    # -------------------------------------------------
    best_val_loss = float("inf")
    patience = 5                # early‑stop after 5 epochs w/out improvement
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        # ---------- TRAIN ----------
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        epoch_start = time.time()

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total   += imgs.size(0)

        train_loss /= total
        train_acc   = correct / total

        # ---------- VALIDATION ----------
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total   += imgs.size(0)

        val_loss /= val_total
        val_acc   = val_correct / val_total

        scheduler.step()   # update the learning rate

        epoch_time = time.time() - epoch_start
        print(
            f"\n📊 Epoch {epoch}/{epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"time {epoch_time:.1f}s"
        )

        # ---------- CHECKPOINT ----------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            Path(os.path.dirname(model_out)).mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model_state_dict": model.state_dict(),
                 "classes": train_set.classes},
                model_out,
            )
            print(f"✨ Saved NEW BEST model → {model_out}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("🔔 Early stopping triggered – validation hasn’t improved.")
                break

    print("\n✅ Training finished! Best validation loss:", best_val_loss)


# -----------------------------------------------------------------
# 7️⃣  CLI entry point – allows you to run the script directly
# -----------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Root folder containing train/ and val/",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size per step"
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument(
        "--model_out",
        type=str,
        default="./models/model.pth",
        help="Where to save the best checkpoint",
    )
    parser.add_argument(
        "--device", type=str, default=None, help='GPU device, e.g. "cuda"'
    )
    args = parser.parse_args()

    train(
        args.data_dir,
        args.model_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
