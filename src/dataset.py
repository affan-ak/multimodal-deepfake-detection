"""PyTorch dataset for face crops with train/val/test splitting."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

from src.config import (
    FACES_DIR, BATCH_SIZE, NUM_WORKERS, RANDOM_SEED,
    TRAIN_SPLIT, VAL_SPLIT, FACE_SIZE
)


class DeepfakeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


def get_transforms(split="train"):
    """Get augmentation transforms per split."""
    if split == "train":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomResizedCrop(FACE_SIZE, scale=(0.9, 1.0)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((FACE_SIZE, FACE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])


def collect_samples(faces_dir=None):
    """Collect all face image paths and their labels."""
    if faces_dir is None:
        faces_dir = FACES_DIR

    faces_dir = Path(faces_dir)
    paths = []
    labels = []

    extensions = ("*.jpg", "*.jpeg", "*.png")
    for label_name, label_idx in [("real", 0), ("fake", 1)]:
        label_dir = faces_dir / label_name
        if not label_dir.exists():
            continue
        for ext in extensions:
            for img_path in label_dir.rglob(ext):
                paths.append(str(img_path))
                labels.append(label_idx)

    return paths, labels


def create_dataloaders(faces_dir=None):
    """Create train/val/test dataloaders with stratified splitting."""
    paths, labels = collect_samples(faces_dir)
    paths = np.array(paths)
    labels = np.array(labels)

    print(f"Total samples: {len(paths)}")
    print(f"  Real: {(labels == 0).sum()}")
    print(f"  Fake: {(labels == 1).sum()}")

    # First split: train+val vs test
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths, labels,
        test_size=1 - TRAIN_SPLIT - VAL_SPLIT,
        stratify=labels,
        random_state=RANDOM_SEED
    )

    # Second split: train vs val
    val_ratio = VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=val_ratio,
        stratify=train_val_labels,
        random_state=RANDOM_SEED
    )

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_paths)} ({len(train_paths)/len(paths)*100:.1f}%)")
    print(f"  Val:   {len(val_paths)} ({len(val_paths)/len(paths)*100:.1f}%)")
    print(f"  Test:  {len(test_paths)} ({len(test_paths)/len(paths)*100:.1f}%)")

    train_dataset = DeepfakeDataset(train_paths, train_labels, get_transforms("train"))
    val_dataset = DeepfakeDataset(val_paths, val_labels, get_transforms("val"))
    test_dataset = DeepfakeDataset(test_paths, test_labels, get_transforms("test"))

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    split_info = {
        "train_size": len(train_paths),
        "val_size": len(val_paths),
        "test_size": len(test_paths),
        "train_real": int((train_labels == 0).sum()),
        "train_fake": int((train_labels == 1).sum()),
        "val_real": int((val_labels == 0).sum()),
        "val_fake": int((val_labels == 1).sum()),
        "test_real": int((test_labels == 0).sum()),
        "test_fake": int((test_labels == 1).sum()),
    }

    return train_loader, val_loader, test_loader, split_info
