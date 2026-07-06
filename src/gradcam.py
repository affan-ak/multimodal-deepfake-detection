"""GradCAM explainability: generate heatmaps showing which regions triggered detection."""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pathlib import Path
import cv2

from src.config import HEATMAPS_DIR, FACE_SIZE, DEVICE, NUM_HEATMAP_SAMPLES


class GradCAM:
    """Gradient-weighted Class Activation Mapping for CNN explanations."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """Generate GradCAM heatmap for a single input."""
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot)

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=(FACE_SIZE, FACE_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam, output


def get_gradcam_target_layer(model):
    """Get the last convolutional layer for GradCAM."""
    # For XceptionNet via timm: the last block before global pooling
    if hasattr(model.backbone, "block12"):
        return model.backbone.block12
    elif hasattr(model.backbone, "conv_head"):
        return model.backbone.conv_head
    # Fallback: find last Conv2d
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    return last_conv


def overlay_heatmap(image_np, heatmap, alpha=0.5):
    """Overlay heatmap on image."""
    colormap = plt.cm.jet(heatmap)[:, :, :3]
    colormap = (colormap * 255).astype(np.uint8)

    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)

    overlay = (alpha * colormap + (1 - alpha) * image_np).astype(np.uint8)
    return overlay


def generate_heatmaps(model, test_loader, num_samples=None, device=None):
    """Generate and save GradCAM heatmaps for sample predictions."""
    if num_samples is None:
        num_samples = NUM_HEATMAP_SAMPLES
    if device is None:
        device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    model.eval()

    target_layer = get_gradcam_target_layer(model)
    gradcam = GradCAM(model, target_layer)

    inv_normalize = transforms.Compose([
        transforms.Normalize(mean=[0, 0, 0], std=[1/0.5, 1/0.5, 1/0.5]),
        transforms.Normalize(mean=[-0.5, -0.5, -0.5], std=[1, 1, 1]),
    ])

    class_names = ["Real", "Fake"]
    sample_count = 0

    for images, labels in test_loader:
        for i in range(images.size(0)):
            if sample_count >= num_samples:
                return

            input_tensor = images[i:i+1].to(device)
            true_label = labels[i].item()

            heatmap, output = gradcam.generate(input_tensor)
            pred_class = output.argmax(dim=1).item()
            pred_prob = torch.softmax(output, dim=1)[0, pred_class].item()

            # Denormalize image for visualization
            img_denorm = inv_normalize(images[i]).permute(1, 2, 0).numpy()
            img_denorm = np.clip(img_denorm, 0, 1)

            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            axes[0].imshow(img_denorm)
            axes[0].set_title(f"Input (True: {class_names[true_label]})", fontsize=11)
            axes[0].axis("off")

            axes[1].imshow(heatmap, cmap="jet")
            axes[1].set_title("GradCAM Heatmap", fontsize=11)
            axes[1].axis("off")

            overlay = overlay_heatmap(img_denorm, heatmap)
            axes[2].imshow(overlay)
            axes[2].set_title(
                f"Overlay (Pred: {class_names[pred_class]}, p={pred_prob:.3f})",
                fontsize=11
            )
            axes[2].axis("off")

            status = "correct" if pred_class == true_label else "incorrect"
            plt.suptitle(
                f"Sample {sample_count+1} — {status.upper()}",
                fontsize=12, fontweight="bold"
            )
            plt.tight_layout()

            filename = f"sample_{sample_count+1:03d}_{class_names[true_label].lower()}_{status}.png"
            plt.savefig(HEATMAPS_DIR / filename, dpi=120, bbox_inches="tight")
            plt.close()

            sample_count += 1

    print(f"Generated {sample_count} GradCAM heatmaps in {HEATMAPS_DIR}")
