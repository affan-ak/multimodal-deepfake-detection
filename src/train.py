"""Training loop with early stopping, logging, and checkpoint saving."""

import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.metrics import roc_auc_score
import numpy as np

from src.config import (
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, DEVICE,
    CHECKPOINTS_DIR, METRICS_DIR, EARLY_STOPPING_PATIENCE,
    MODEL_NAME, NUM_CLASSES, PRETRAINED, RANDOM_SEED
)
from src.model import create_model
from src.dataset import create_dataloaders


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        probs = torch.softmax(outputs, dim=1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    auc = roc_auc_score(all_labels, all_probs)

    return epoch_loss, epoch_acc, auc


def train(faces_dir=None):
    """Full training pipeline. Returns model, history, and split info."""
    set_seed(RANDOM_SEED)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, test_loader, split_info = create_dataloaders(faces_dir)

    # Model
    model = create_model(MODEL_NAME, NUM_CLASSES, PRETRAINED)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {MODEL_NAME} | Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Training loop
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_auc": []}
    best_auc = 0.0
    patience_counter = 0

    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    print("-" * 70)

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - start_time

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        print(
            f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": val_auc,
                "val_acc": val_acc,
                "model_name": MODEL_NAME,
                "config": {
                    "num_classes": NUM_CLASSES,
                    "learning_rate": LEARNING_RATE,
                    "batch_size": split_info.get("train_size", 0),
                    "random_seed": RANDOM_SEED,
                },
            }
            torch.save(checkpoint, CHECKPOINTS_DIR / f"{MODEL_NAME}_best.pth")
            print(f"  → Saved best model (AUC: {val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1} (patience={EARLY_STOPPING_PATIENCE})")
                break

    print("-" * 70)
    print(f"Training complete. Best validation AUC: {best_auc:.4f}")

    # Save training history
    training_manifest = {
        "model_name": MODEL_NAME,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "best_val_auc": best_auc,
        "epochs_trained": len(history["train_loss"]),
        "random_seed": RANDOM_SEED,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": train_loader.batch_size,
        "split_info": split_info,
        "history": history,
    }
    with open(METRICS_DIR / "training_history.json", "w") as f:
        json.dump(training_manifest, f, indent=2)

    return model, history, split_info, test_loader


if __name__ == "__main__":
    train()
