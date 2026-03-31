"""
Configuration file for Breast Cancer Prediction using Vision Transformers.
Shared settings for MaxViT, MobileViT, and EfficientViT models.

Anti-overfit / Anti-underfit strategy:
  - Label smoothing (0.1)
  - Dropout in classifier head (0.3)
  - Gradient clipping (max_norm=1.0)
  - CosineAnnealingWarmRestarts LR schedule with warmup
  - Progressive unfreezing (freeze backbone initially)
  - Mixup augmentation (alpha=0.2)
  - Data augmentation (flips, rotation, color jitter, erasing)
  - Weighted sampling for class imbalance
  - Early stopping with patience
"""

import os

# ============================================================
# Paths
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset Zip")

# Individual dataset paths
DDSM_BENIGN = os.path.join(DATASET_DIR, "DDSM Dataset", "DDSM Dataset", "Benign Masses")
DDSM_MALIGNANT = os.path.join(DATASET_DIR, "DDSM Dataset", "DDSM Dataset", "Malignant Masses")

BUS_UC_BENIGN = os.path.join(DATASET_DIR, "BUS_UC", "BUS_UC", "Benign", "images")
BUS_UC_MALIGNANT = os.path.join(DATASET_DIR, "BUS_UC", "BUS_UC", "Malignant", "images")

BREAKHIS_BASE = os.path.join(DATASET_DIR, "dataset_cancer_v1", "dataset_cancer_v1", "classificacao_binaria")
BREAKHIS_MAGNIFICATIONS = ["40X", "100X", "200X", "400X"]

# Output directories
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

# ============================================================
# Data Configuration
# ============================================================
IMAGE_SIZE = 224          # Input image resolution
NUM_CLASSES = 2           # Binary: Benign (0) vs Malignant (1)
CLASS_NAMES = ["Benign", "Malignant"]

# Train / Validation / Test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

# ============================================================
# Authenticity / Evaluation Strictness
# ============================================================
# Enables stricter dataset curation and reporting for more realistic metrics.
AUTHENTIC_EVAL = True

# DDSM curation controls (applied when AUTHENTIC_EVAL=True)
# Remove generated variants like "... (2).png" that come from the same view.
DDSM_EXCLUDE_AUGMENTED_VARIANTS = True
# Remove DDSM case IDs that appear in both benign and malignant folders.
DDSM_EXCLUDE_AMBIGUOUS_CASES = True

# Report case-level metrics (aggregate predictions per patient/case group).
REPORT_CASE_LEVEL_METRICS = True

# ============================================================
# Training Hyperparameters
# ============================================================
BATCH_SIZE = 16
NUM_EPOCHS = 15  # Increased for better training curves and graph data
LEARNING_RATE = 3e-4       # Peak LR after warmup
WEIGHT_DECAY = 1e-2        # Strong weight decay for AdamW (regularization)
NUM_WORKERS = 4
PIN_MEMORY = True

# Learning rate schedule
WARMUP_EPOCHS = 1          # Linear warmup epochs
LR_MIN = 1e-6              # Minimum LR at end of cosine decay

# Regularization
LABEL_SMOOTHING = 0.1      # Label smoothing for CrossEntropyLoss
DROPOUT_RATE = 0.3         # Dropout in classifier head
GRAD_CLIP_MAX_NORM = 1.0   # Gradient clipping max norm
MIXUP_ALPHA = 0.2          # Mixup augmentation alpha (0 = disabled)

# Progressive unfreezing
FREEZE_BACKBONE_EPOCHS = 1  # Train only head for first N epochs, then unfreeze

# Early stopping
EARLY_STOPPING_PATIENCE = 7
EARLY_STOPPING_MIN_DELTA = 1e-4

# ============================================================
# Data Augmentation
# ============================================================
AUGMENTATION = {
    "horizontal_flip": True,
    "vertical_flip": True,
    "random_rotation": 15,
    "color_jitter_brightness": 0.2,
    "color_jitter_contrast": 0.2,
    "color_jitter_saturation": 0.1,
    "color_jitter_hue": 0.05,
    "random_affine_degrees": 10,
    "random_affine_translate": (0.1, 0.1),
    "random_erasing_prob": 0.15,
}

# ============================================================
# Model-specific settings (timm model names)
# ============================================================
MODELS = {
    "maxvit": {
        "name": "maxvit_tiny_tf_224.in1k",
        "pretrained": True,
        "description": "MaxViT (CNN + Transformer) - Strong local & global feature modeling",
    },
    "mobilevit": {
        "name": "mobilevitv2_100.cvnets_in1k",
        "pretrained": True,
        "description": "MobileViT - Lightweight Hybrid for Efficiency & deployability",
    },
    "efficientvit": {
        "name": "efficientvit_b1.r224_in1k",
        "pretrained": True,
        "description": "EfficientViT - Optimized Transformer for Fast inference, low memory",
    },
}

# Ensure output dirs exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
