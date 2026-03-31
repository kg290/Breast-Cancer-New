"""
Model 1: MaxViT - Multi-Axis Vision Transformer
CNN + Transformer hybrid for strong local & global feature modeling.
Uses block attention (local) + grid attention (global) across multiple stages.

Anti-overfit strategy:
  - Pretrained backbone (transfer learning)
  - Progressive unfreezing (head first, then full fine-tune)
  - Dropout (0.3) added to classifier head
  - Label smoothing (0.1)
  - Mixup augmentation
  - Gradient clipping
  - Early stopping
  - Warmup + cosine LR schedule
"""

import os
import sys
import torch
import torch.nn as nn
import timm

import config
from dataset import create_dataloaders
from utils import train_model, full_evaluate_and_plot


class MaxViTClassifier(nn.Module):
    """
    Wrapper around timm MaxViT backbone with a custom classifier head.
    Uses AdaptiveAvgPool2d + Dropout-regularized MLP head.
    """

    def __init__(self, backbone, in_features, num_classes, dropout=0.3):
        super().__init__()
        self.backbone = backbone          # timm model (num_classes=0)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(  # Named 'classifier' for freeze/unfreeze detection
            nn.Flatten(),
            nn.LayerNorm(in_features),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.backbone.forward_features(x)  # (B, C, H, W)
        x = self.pool(x)                       # (B, C, 1, 1)
        x = self.classifier(x)                 # (B, num_classes)
        return x


def build_maxvit_model(num_classes=config.NUM_CLASSES, pretrained=True, dropout=config.DROPOUT_RATE):
    """
    Build MaxViT model using timm with a custom dropout-regularized classifier head.
    """
    model_cfg = config.MODELS["maxvit"]
    model_name = model_cfg["name"]

    print(f"\nBuilding model: {model_name}")
    print(f"  Description: {model_cfg['description']}")
    print(f"  Pretrained:  {pretrained}")
    print(f"  Head dropout: {dropout}")

    # Create backbone without classifier head
    backbone = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,  # Remove default classifier
    )

    # Get the feature dimension
    with torch.no_grad():
        dummy = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
        features = backbone.forward_features(dummy)
        in_features = features.shape[1]  # Channel dimension

    print(f"  Feature dim: {in_features}")

    # Wrap with custom classifier
    model = MaxViTClassifier(backbone, in_features, num_classes, dropout)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    return model


def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Data
    print("\n--- Loading Data ---")
    train_loader, val_loader, test_loader = create_dataloaders()

    # Model
    print("\n--- Building MaxViT Model ---")
    model = build_maxvit_model()

    # Train (with all anti-overfit measures)
    model, history = train_model(model, train_loader, val_loader, "MaxViT", device)

    # Evaluate & Plot
    print("\n--- Final Evaluation on Test Set ---")
    metrics = full_evaluate_and_plot(model, test_loader, "MaxViT", history, device)

    print("\n" + "=" * 60)
    print("  MaxViT Training & Evaluation COMPLETE")
    print("=" * 60)
    print(f"  Results saved to: {os.path.join(config.RESULTS_DIR, 'MaxViT')}")
    print(f"  Checkpoint: {os.path.join(config.CHECKPOINT_DIR, 'MaxViT_best.pth')}")

    return model, metrics


if __name__ == "__main__":
    main()
