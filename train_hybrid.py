"""
Model 4: HybridViT - Multi-Backbone Ensemble Hybrid
Fuses features from MaxViT (global attention) + MobileViTv2 (efficient local)
into a single model with a shared classifier head.

This is NOT a simple voting ensemble — it's a true feature-level fusion model
that learns complementary representations from both architectures.

Architecture:
  MaxViT backbone  ──> pool ──> features_A ──┐
                                              ├──> concat ──> fusion MLP ──> prediction
  MobileViTv2 backbone ──> pool ──> features_B ──┘

Anti-overfit strategy:
  - Both backbones pretrained (transfer learning)
  - Progressive unfreezing (head first, then full fine-tune)
  - Dropout (0.3) in fusion classifier
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


class HybridViTClassifier(nn.Module):
    """
    Feature-level fusion of two ViT backbones with a shared classifier.
    Concatenates pooled features from both backbones, then classifies.
    """

    def __init__(self, backbone_a, backbone_b, feat_dim_a, feat_dim_b,
                 num_classes, dropout=0.3):
        super().__init__()
        self.backbone_a = backbone_a   # MaxViT
        self.backbone_b = backbone_b   # MobileViTv2
        self.pool = nn.AdaptiveAvgPool2d(1)

        fused_dim = feat_dim_a + feat_dim_b

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(fused_dim),
            nn.Dropout(p=dropout),
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # Extract features from both backbones
        feat_a = self.backbone_a.forward_features(x)   # (B, C_a, H, W)
        feat_b = self.backbone_b.forward_features(x)   # (B, C_b, H, W)

        # Pool both to (B, C, 1, 1)
        feat_a = self.pool(feat_a)
        feat_b = self.pool(feat_b)

        # Concatenate along channel dimension
        fused = torch.cat([feat_a, feat_b], dim=1)     # (B, C_a+C_b, 1, 1)

        # Classify
        out = self.classifier(fused)                    # (B, num_classes)
        return out


def _get_feat_dim(backbone, image_size):
    """Probe a backbone to get its output feature dimension."""
    with torch.no_grad():
        dummy = torch.randn(1, 3, image_size, image_size)
        features = backbone.forward_features(dummy)
        if features.dim() == 4:
            return features.shape[1]
        return features.shape[-1]


def build_hybrid_model(num_classes=config.NUM_CLASSES, pretrained=True, dropout=config.DROPOUT_RATE):
    """
    Build HybridViT model: MaxViT + MobileViTv2 feature fusion.
    """
    cfg_a = config.MODELS["maxvit"]
    cfg_b = config.MODELS["mobilevit"]

    print(f"\nBuilding HybridViT model (feature fusion)")
    print(f"  Backbone A: {cfg_a['name']}  ({cfg_a['description']})")
    print(f"  Backbone B: {cfg_b['name']}  ({cfg_b['description']})")
    print(f"  Pretrained: {pretrained}")
    print(f"  Head dropout: {dropout}")

    # Build both backbones (no classifier heads)
    backbone_a = timm.create_model(cfg_a["name"], pretrained=pretrained, num_classes=0)
    backbone_b = timm.create_model(cfg_b["name"], pretrained=pretrained, num_classes=0)

    feat_dim_a = _get_feat_dim(backbone_a, config.IMAGE_SIZE)
    feat_dim_b = _get_feat_dim(backbone_b, config.IMAGE_SIZE)

    print(f"  Feature dim A (MaxViT):    {feat_dim_a}")
    print(f"  Feature dim B (MobileViT): {feat_dim_b}")
    print(f"  Fused dim:                 {feat_dim_a + feat_dim_b}")

    model = HybridViTClassifier(
        backbone_a, backbone_b,
        feat_dim_a, feat_dim_b,
        num_classes, dropout,
    )

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
    print("\n--- Building HybridViT Model ---")
    model = build_hybrid_model()

    # Train
    model, history = train_model(model, train_loader, val_loader, "HybridViT", device)

    # Evaluate & Plot
    print("\n--- Final Evaluation on Test Set ---")
    metrics = full_evaluate_and_plot(model, test_loader, "HybridViT", history, device)

    print("\n" + "=" * 60)
    print("  HybridViT Training & Evaluation COMPLETE")
    print("=" * 60)
    print(f"  Results saved to: {os.path.join(config.RESULTS_DIR, 'HybridViT')}")
    print(f"  Checkpoint: {os.path.join(config.CHECKPOINT_DIR, 'HybridViT_best.pth')}")

    return model, metrics


if __name__ == "__main__":
    main()
