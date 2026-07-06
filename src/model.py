"""XceptionNet-based deepfake detector."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class XceptionNetDetector(nn.Module):
    """XceptionNet fine-tuned for binary deepfake classification."""

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "xception",
            pretrained=pretrained,
            num_classes=0,  # remove original head
        )
        feature_dim = self.backbone.num_features  # 2048 for Xception
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def get_features(self, x):
        """Return feature vector before classification head (for GradCAM)."""
        return self.backbone(x)


class EfficientNetDetector(nn.Module):
    """EfficientNet-B4 alternative detector for ensemble potential."""

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=0,
        )
        feature_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


def create_model(model_name="xceptionnet", num_classes=2, pretrained=True):
    """Factory function to create model by name."""
    models = {
        "xceptionnet": XceptionNetDetector,
        "efficientnet_b4": EfficientNetDetector,
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    return models[model_name](num_classes=num_classes, pretrained=pretrained)
