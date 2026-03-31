"""
Model 3: EfficientViT - Optimized Transformer
Fast inference and low memory usage for breast cancer classification.
Designed for efficient multi-scale attention with linear complexity.

Anti-overfit strategy:
  - Pretrained backbone (transfer learning)
  - Progressive unfreezing (head first, then full fine-tune)
  - Dropout (0.3) in classifier head
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


class EfficientViTClassifier(nn.Module):
    """
    Wrapper around timm EfficientViT backbone with a custom classifier head.
    """

    def __init__(self, backbone, in_features, num_classes, dropout=0.3):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(in_features),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def build_efficientvit_model(num_classes=config.NUM_CLASSES, pretrained=True, dropout=config.DROPOUT_RATE):
    """Build EfficientViT model with custom dropout-regularized classifier head."""
    model_cfg = config.MODELS["efficientvit"]
    model_name = model_cfg["name"]

    print(f"\nBuilding model: {model_name}")
    print(f"  Description: {model_cfg['description']}")
    print(f"  Pretrained:  {pretrained}")
    print(f"  Head dropout: {dropout}")

    backbone = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,
    )

    # Get feature dimension
    with torch.no_grad():
        dummy = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
        features = backbone.forward_features(dummy)
        if features.dim() == 4:
            in_features = features.shape[1]
        else:
            in_features = features.shape[-1]

    print(f"  Feature dim: {in_features}")

    model = EfficientViTClassifier(backbone, in_features, num_classes, dropout)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Data
    print("\n--- Loading Data ---")
    train_loader, val_loader, test_loader = create_dataloaders()

    # Model
    print("\n--- Building EfficientViT Model ---")
    model = build_efficientvit_model()

    # Train
    model, history = train_model(model, train_loader, val_loader, "EfficientViT", device)

    # Evaluate & Plot
    print("\n--- Final Evaluation on Test Set ---")
    metrics = full_evaluate_and_plot(model, test_loader, "EfficientViT", history, device)

    print("\n" + "=" * 60)
    print("  EfficientViT Training & Evaluation COMPLETE")
    print("=" * 60)
    print(f"  Results saved to: {os.path.join(config.RESULTS_DIR, 'EfficientViT')}")
    print(f"  Checkpoint: {os.path.join(config.CHECKPOINT_DIR, 'EfficientViT_best.pth')}")

    return model, metrics


if __name__ == "__main__":
    main()
