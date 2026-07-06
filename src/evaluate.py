"""Evaluation module: metrics, ROC curves, confusion matrix, EER."""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score
)
from pathlib import Path

from src.config import DEVICE, CHECKPOINTS_DIR, METRICS_DIR, MODEL_NAME, NUM_CLASSES
from src.model import create_model


def load_best_model(device=None):
    """Load the best checkpoint."""
    if device is None:
        device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    model = create_model(MODEL_NAME, NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(CHECKPOINTS_DIR / f"{MODEL_NAME}_best.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def get_predictions(model, dataloader, device=None):
    """Run inference on a dataloader, return probabilities and labels."""
    if device is None:
        device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    all_probs = []
    all_labels = []
    all_preds = []

    model.eval()
    for images, labels in dataloader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]  # probability of "fake"

        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_preds.extend((probs > 0.5).int().cpu().numpy())

    return np.array(all_probs), np.array(all_labels), np.array(all_preds)


def compute_eer(labels, scores):
    """Compute Equal Error Rate."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]
    return eer, eer_threshold


def plot_roc_curve(labels, probs, save_path):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve — Deepfake Detection", fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(labels, preds, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.colorbar()

    classes = ["Real", "Fake"]
    tick_marks = [0, 1]
    plt.xticks(tick_marks, classes, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     fontsize=16, color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def evaluate(test_loader):
    """Full evaluation on test set. Saves metrics and plots."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_best_model(device)
    print(f"Loaded model from epoch {checkpoint['epoch']} (val AUC: {checkpoint['val_auc']:.4f})")

    probs, labels, preds = get_predictions(model, test_loader, device)

    # Metrics
    auc = roc_auc_score(labels, probs)
    eer, eer_threshold = compute_eer(labels, probs)
    f1 = f1_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)

    report = classification_report(labels, preds, target_names=["Real", "Fake"], output_dict=True)

    results = {
        "test_auc": float(auc),
        "test_eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "test_f1": float(f1),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_accuracy": float((preds == labels).mean()),
        "num_test_samples": int(len(labels)),
        "num_real": int((labels == 0).sum()),
        "num_fake": int((labels == 1).sum()),
        "classification_report": report,
        "model_checkpoint_epoch": int(checkpoint["epoch"]),
        "model_name": MODEL_NAME,
    }

    print("\n" + "=" * 50)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 50)
    print(f"  AUC:       {auc:.4f}")
    print(f"  EER:       {eer:.4f} (threshold: {eer_threshold:.4f})")
    print(f"  F1:        {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  Accuracy:  {(preds == labels).mean():.4f}")
    print(f"  Samples:   {len(labels)} (Real: {(labels==0).sum()}, Fake: {(labels==1).sum()})")
    print("=" * 50)

    # Save
    with open(METRICS_DIR / "evaluation_report.json", "w") as f:
        json.dump(results, f, indent=2)

    plot_roc_curve(labels, probs, METRICS_DIR / "roc_curve.png")
    plot_confusion_matrix(labels, preds, METRICS_DIR / "confusion_matrix.png")

    print(f"\nResults saved to {METRICS_DIR}")
    return results
