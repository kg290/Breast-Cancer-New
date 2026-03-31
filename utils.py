"""
Utility functions for training, evaluation, and visualization.
Shared across all model training scripts.

Includes:
  - Mixup augmentation
  - Warmup + cosine LR scheduler
  - Gradient clipping
  - Progressive backbone unfreezing
  - Label smoothing loss
  - Comprehensive metrics & plots
"""

import os
import time
import copy
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    matthews_corrcoef,
    cohen_kappa_score,
)
from tqdm import tqdm

import config


# ============================================================
# Mixup Augmentation
# ============================================================

def mixup_data(x, y, alpha=0.2):
    """
    Apply mixup augmentation: linearly interpolate between pairs of examples.
    Helps regularize and prevents overconfidence (reduces overfitting).
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for mixup-augmented inputs."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
# Learning Rate Scheduler with Warmup
# ============================================================

class WarmupCosineScheduler:
    """
    Linear warmup for `warmup_epochs`, then cosine annealing to `lr_min`.
    Prevents the model from making drastic updates early when head is random.
    """

    def __init__(self, optimizer, warmup_epochs, total_epochs, lr_max, lr_min=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        lr = self._compute_lr(self.current_epoch)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _compute_lr(self, epoch):
        if epoch <= self.warmup_epochs:
            # Linear warmup from lr_min to lr_max
            return self.lr_min + (self.lr_max - self.lr_min) * (epoch / self.warmup_epochs)
        else:
            # Cosine annealing from lr_max to lr_min
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + math.cos(math.pi * progress))

    def get_lr(self):
        return self._compute_lr(self.current_epoch)


# ============================================================
# Progressive Unfreezing
# ============================================================

def freeze_backbone(model):
    """
    Freeze all parameters except the classifier head.
    This prevents destroying pretrained features during initial head training.

    Detects head params by checking if the top-level module attribute name
    is one of: 'head', 'classifier', 'fc', 'last_linear'.
    """
    # Identify top-level head module names
    head_keywords = {"head", "classifier", "fc", "last_linear"}
    head_module_names = set()
    for attr_name in dir(model):
        if attr_name in head_keywords:
            attr = getattr(model, attr_name, None)
            if isinstance(attr, nn.Module):
                head_module_names.add(attr_name)

    # If no known head found, fall back to the last top-level module
    if not head_module_names:
        children = list(model.named_children())
        if children:
            head_module_names.add(children[-1][0])

    frozen_count = 0
    trainable_count = 0
    for name, param in model.named_parameters():
        # Check if this param belongs to a head module (top-level prefix)
        is_head = any(name.startswith(head_name + ".") or name == head_name
                      for head_name in head_module_names)
        if is_head:
            param.requires_grad = True
            trainable_count += 1
        else:
            param.requires_grad = False
            frozen_count += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Backbone FROZEN: {frozen_count} param groups frozen")
    print(f"  Head modules: {head_module_names}")
    print(f"  Trainable: {trainable:,} / {total:,} parameters")


def unfreeze_all(model):
    """Unfreeze all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  All layers UNFROZEN: {trainable:,} trainable parameters")


# ============================================================
# Early Stopping
# ============================================================

class EarlyStopping:
    """
    Early stopping that monitors val_loss.
    Also tracks overfit gap (train_loss - val_loss) as a warning.
    """

    def __init__(self, patience=7, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_val_loss = None
        self.early_stop = False

    def __call__(self, val_loss, train_loss=None):
        if self.best_val_loss is None:
            self.best_val_loss = val_loss
            return False

        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        # Log overfit warning
        if train_loss is not None and val_loss > 0:
            gap = val_loss - train_loss
            if gap > 0.3:
                print(f"  [WARNING] Overfit gap: val_loss - train_loss = {gap:.4f}")

        return False


# ============================================================
# Training Engine
# ============================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None,
                    use_mixup=True, mixup_alpha=0.2):
    """Train the model for one epoch with optional mixup and gradient clipping."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="  Training", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Mixup augmentation (regularization)
        if use_mixup and mixup_alpha > 0:
            images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)
        else:
            labels_a = labels
            labels_b = labels
            lam = 1.0

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_MAX_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_MAX_NORM)
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        # For accuracy with mixup, compare against primary labels
        correct += (preds == labels_a).sum().item()
        total += labels_a.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate the model on validation/test data (no mixup, no augmentation)."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in tqdm(dataloader, desc="  Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels), np.array(all_probs)


def train_model(model, train_loader, val_loader, model_name, device):
    """
    Full training loop with:
      - Label smoothing loss
      - Warmup + cosine LR schedule
      - Progressive unfreezing (freeze backbone for first N epochs)
      - Mixup augmentation
      - Gradient clipping
      - Early stopping
      - Best checkpoint saving

    Returns:
        model: with best weights loaded
        history: dict with training history
    """
    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"  Epochs: {config.NUM_EPOCHS} | Batch: {config.BATCH_SIZE}")
    print(f"  LR: {config.LEARNING_RATE} | Weight Decay: {config.WEIGHT_DECAY}")
    print(f"  Label Smoothing: {config.LABEL_SMOOTHING} | Mixup: {config.MIXUP_ALPHA}")
    print(f"  Dropout: {config.DROPOUT_RATE} | Grad Clip: {config.GRAD_CLIP_MAX_NORM}")
    print(f"  Warmup: {config.WARMUP_EPOCHS} epochs | Freeze backbone: {config.FREEZE_BACKBONE_EPOCHS} epochs")
    print(f"{'='*60}")

    model = model.to(device)

    # --- Phase 1: Freeze backbone, train only head ---
    print(f"\n--- Phase 1: Training classifier head only (epochs 1-{config.FREEZE_BACKBONE_EPOCHS}) ---")
    freeze_backbone(model)

    # Loss with label smoothing (anti-overfit)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)

    # Optimizer - only trainable params
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # LR Scheduler with warmup
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=config.WARMUP_EPOCHS,
        total_epochs=config.NUM_EPOCHS,
        lr_max=config.LEARNING_RATE,
        lr_min=config.LR_MIN,
    )

    # Mixed precision scaler
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=config.EARLY_STOPPING_MIN_DELTA,
    )

    # History tracking - comprehensive for graphing
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "val_precision": [], "val_recall": [], "val_f1": [], "val_auc": [],
        "val_specificity": [],
        "lr": [],
        "overfit_gap": [],
        "epoch_time": [],
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0
    backbone_unfrozen = False

    start_time = time.time()

    for epoch in range(1, config.NUM_EPOCHS + 1):
        epoch_start = time.time()
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS}")
        print("-" * 40)

        # --- Phase 2: Unfreeze backbone after FREEZE_BACKBONE_EPOCHS ---
        if epoch == config.FREEZE_BACKBONE_EPOCHS + 1 and not backbone_unfrozen:
            print(f"\n--- Phase 2: Unfreezing entire model for fine-tuning ---")
            unfreeze_all(model)
            backbone_unfrozen = True

            # Recreate optimizer with ALL parameters and lower LR for backbone
            # Detect head modules the same way as freeze_backbone
            head_keywords = {"head", "classifier", "fc", "last_linear"}
            head_module_names = set()
            for attr_name in dir(model):
                if attr_name in head_keywords:
                    attr = getattr(model, attr_name, None)
                    if isinstance(attr, nn.Module):
                        head_module_names.add(attr_name)
            if not head_module_names:
                children = list(model.named_children())
                if children:
                    head_module_names.add(children[-1][0])

            backbone_params = []
            head_params = []
            for name, param in model.named_parameters():
                is_head = any(name.startswith(h + ".") or name == h for h in head_module_names)
                if is_head:
                    head_params.append(param)
                else:
                    backbone_params.append(param)

            optimizer = torch.optim.AdamW([
                {"params": backbone_params, "lr": config.LEARNING_RATE * 0.1},  # Lower LR for pretrained layers
                {"params": head_params, "lr": config.LEARNING_RATE},
            ], weight_decay=config.WEIGHT_DECAY)

            # Reset scheduler for remaining epochs
            scheduler = WarmupCosineScheduler(
                optimizer,
                warmup_epochs=1,  # Brief warmup after unfreezing
                total_epochs=config.NUM_EPOCHS - epoch + 1,
                lr_max=config.LEARNING_RATE * 0.1,  # Conservative max for backbone
                lr_min=config.LR_MIN,
            )

        # Train (with mixup after warmup, and only after backbone is unfrozen for better stability)
        use_mixup = backbone_unfrozen and config.MIXUP_ALPHA > 0
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            use_mixup=use_mixup, mixup_alpha=config.MIXUP_ALPHA,
        )

        # Validate (no mixup, no augmentation) - get full predictions for metrics
        val_loss, val_acc, val_preds, val_labels, val_probs = evaluate(model, val_loader, criterion, device)

        # Compute per-epoch validation metrics for detailed logging
        val_prob_pos = val_probs[:, 1]
        val_precision = float(precision_score(val_labels, val_preds, average="binary", zero_division=0))
        val_recall = float(recall_score(val_labels, val_preds, average="binary", zero_division=0))
        val_specificity = float(recall_score(val_labels, val_preds, average="binary", pos_label=0, zero_division=0))
        val_f1 = float(f1_score(val_labels, val_preds, average="binary", zero_division=0))
        try:
            val_auc = float(roc_auc_score(val_labels, val_prob_pos))
        except ValueError:
            val_auc = 0.0

        # LR step
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        overfit_gap = val_loss - train_loss
        epoch_time = time.time() - epoch_start

        # Log all metrics
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)
        history["val_specificity"].append(val_specificity)
        history["val_f1"].append(val_f1)
        history["val_auc"].append(val_auc)
        history["lr"].append(current_lr)
        history["overfit_gap"].append(overfit_gap)
        history["epoch_time"].append(epoch_time)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        print(f"  Val   P={val_precision:.4f} R={val_recall:.4f} F1={val_f1:.4f} AUC={val_auc:.4f} Spec={val_specificity:.4f}")
        print(f"  LR: {current_lr:.2e} | Overfit Gap: {overfit_gap:.4f} | Time: {epoch_time:.1f}s")

        # Save best model (by val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{model_name}_best.pth")
            torch.save({
                "epoch": best_epoch,
                "model_state_dict": best_model_wts,
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_acc": best_val_acc,
            }, ckpt_path)
            print(f"  >> Saved best model (epoch {best_epoch}, val_acc={val_acc:.4f})")

        # Early stopping (only after backbone is unfrozen to give full model a chance)
        if backbone_unfrozen:
            if early_stopping(val_loss, train_loss):
                print(f"\n  Early stopping at epoch {epoch} (no improvement for {config.EARLY_STOPPING_PATIENCE} epochs)")
                break

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed/60:.1f} minutes")
    print(f"Best val_loss: {best_val_loss:.4f} | Best val_acc: {best_val_acc:.4f} at epoch {best_epoch}")

    # Load best weights
    model.load_state_dict(best_model_wts)
    history["training_time_seconds"] = elapsed
    history["best_epoch"] = best_epoch

    return model, history


# ============================================================
# Metrics & Evaluation
# ============================================================

def compute_all_metrics(y_true, y_pred, y_probs):
    """Compute comprehensive classification metrics."""
    y_prob_positive = y_probs[:, 1]

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob_positive))
    except ValueError:
        roc_auc = 0.0

    try:
        avg_precision = float(average_precision_score(y_true, y_prob_positive))
    except ValueError:
        avg_precision = 0.0

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
        "specificity": float(recall_score(y_true, y_pred, average="binary", pos_label=0, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics["true_positives"] = int(tp)
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)

    return metrics


def compute_case_level_metrics(y_true, y_probs, group_ids):
    """
    Aggregate image-level predictions into case-level metrics.

    For each case/group:
      - probability = mean malignant probability across images
      - true label = unique case label (or majority if needed)
      - prediction = probability >= 0.5
    """
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    group_ids = np.asarray(group_ids)

    if len(y_true) != len(group_ids):
        raise ValueError("Case-level evaluation failed: group_ids length != predictions length")

    unique_groups = np.unique(group_ids)
    case_true = []
    case_prob_pos = []
    ambiguous_cases = 0

    for gid in unique_groups:
        mask = group_ids == gid
        labels = y_true[mask]
        probs_pos = y_probs[mask, 1]

        unique_labels = np.unique(labels)
        if len(unique_labels) > 1:
            ambiguous_cases += 1
            # Skip corrupted mixed-label cases in strict reporting.
            continue

        case_true.append(int(unique_labels[0]))
        case_prob_pos.append(float(np.mean(probs_pos)))

    if len(case_true) == 0:
        return {}

    case_true = np.array(case_true, dtype=np.int64)
    case_prob_pos = np.array(case_prob_pos, dtype=np.float32)
    case_pred = (case_prob_pos >= 0.5).astype(np.int64)
    case_probs = np.column_stack([1.0 - case_prob_pos, case_prob_pos])

    metrics = compute_all_metrics(case_true, case_pred, case_probs)
    metrics["n_cases_total"] = int(len(unique_groups))
    metrics["n_cases_evaluated"] = int(len(case_true))
    metrics["n_cases_ambiguous_skipped"] = int(ambiguous_cases)

    return metrics


def wilson_accuracy_interval(num_correct, num_total, z=1.96):
    """Compute Wilson score confidence interval for accuracy."""
    if num_total <= 0:
        return 0.0, 0.0

    p = num_correct / num_total
    z2 = z * z
    denom = 1.0 + z2 / num_total
    center = (p + z2 / (2.0 * num_total)) / denom
    margin = (z / denom) * np.sqrt((p * (1.0 - p) / num_total) + (z2 / (4.0 * num_total * num_total)))
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return float(lower), float(upper)


def print_metrics(metrics, model_name):
    """Print metrics in a formatted table."""
    print(f"\n{'='*60}")
    print(f"  Test Results: {model_name}")
    print(f"{'='*60}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:.<30} {value:.4f}")
        else:
            print(f"  {key:.<30} {value}")
    print(f"{'='*60}")


# ============================================================
# Plotting Functions
# ============================================================

def plot_training_curves(history, model_name, save_dir):
    """Plot comprehensive training curves: loss, accuracy, val metrics, overfit gap, LR."""
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(3, 2, figsize=(18, 18))

    # 1. Loss curves
    axes[0, 0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=3)
    axes[0, 0].plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=3)
    axes[0, 0].set_title(f"{model_name} - Loss Curves")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Accuracy curves
    axes[0, 1].plot(epochs, history["train_acc"], "b-o", label="Train Acc", markersize=3)
    axes[0, 1].plot(epochs, history["val_acc"], "r-o", label="Val Acc", markersize=3)
    axes[0, 1].set_title(f"{model_name} - Accuracy Curves")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Validation Precision / Recall / F1 / Specificity per epoch
    if "val_precision" in history:
        axes[1, 0].plot(epochs, history["val_precision"], "g-o", label="Precision", markersize=3)
        axes[1, 0].plot(epochs, history["val_recall"], "m-o", label="Recall (Sensitivity)", markersize=3)
        axes[1, 0].plot(epochs, history["val_specificity"], "c-o", label="Specificity", markersize=3)
        axes[1, 0].plot(epochs, history["val_f1"], color="orange", marker="o", label="F1 Score", markersize=3)
        axes[1, 0].set_title(f"{model_name} - Validation Metrics per Epoch")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Score")
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)

    # 4. Validation AUC per epoch
    if "val_auc" in history:
        axes[1, 1].plot(epochs, history["val_auc"], color="purple", marker="o", markersize=3, linewidth=2)
        axes[1, 1].set_title(f"{model_name} - Validation AUC per Epoch")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("ROC AUC")
        axes[1, 1].set_ylim([0, 1.05])
        axes[1, 1].grid(True, alpha=0.3)

    # 5. Overfit gap (val_loss - train_loss)
    if "overfit_gap" in history:
        gap = history["overfit_gap"]
    else:
        gap = [v - t for v, t in zip(history["val_loss"], history["train_loss"])]
    axes[2, 0].plot(epochs, gap, "m-o", markersize=3)
    axes[2, 0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[2, 0].fill_between(epochs, gap, 0, alpha=0.15, color="red",
                            where=[g > 0 for g in gap])
    axes[2, 0].fill_between(epochs, gap, 0, alpha=0.15, color="green",
                            where=[g <= 0 for g in gap])
    axes[2, 0].set_title(f"{model_name} - Generalization Gap (Val - Train Loss)")
    axes[2, 0].set_xlabel("Epoch")
    axes[2, 0].set_ylabel("Gap (red=overfit, green=underfit)")
    axes[2, 0].grid(True, alpha=0.3)

    # 6. Learning rate schedule
    axes[2, 1].plot(epochs, history["lr"], "g-o", markersize=3)
    axes[2, 1].set_title(f"{model_name} - Learning Rate Schedule")
    axes[2, 1].set_xlabel("Epoch")
    axes[2, 1].set_ylabel("Learning Rate")
    axes[2, 1].set_yscale("log")
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Additional: Epoch time plot ---
    if "epoch_time" in history and history["epoch_time"]:
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.bar(epochs, history["epoch_time"], color="steelblue", edgecolor="black", linewidth=0.5)
        ax2.set_title(f"{model_name} - Time per Epoch")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Time (seconds)")
        ax2.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{model_name}_epoch_times.png"), dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  Saved training curves plot")


def plot_confusion_matrix(y_true, y_pred, model_name, save_dir):
    """Plot confusion matrix heatmap."""
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES,
                ax=axes[0], cbar=True)
    axes[0].set_title(f"{model_name} - Confusion Matrix (Counts)")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    # Normalized
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES,
                ax=axes[1], cbar=True, vmin=0, vmax=1)
    axes[1].set_title(f"{model_name} - Confusion Matrix (Normalized)")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved confusion matrix plot")


def plot_roc_curve(y_true, y_probs, model_name, save_dir):
    """Plot ROC curve."""
    os.makedirs(save_dir, exist_ok=True)
    y_prob_pos = y_probs[:, 1]
    fpr, tpr, _ = roc_curve(y_true, y_prob_pos)
    auc_score = roc_auc_score(y_true, y_prob_pos)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC Curve (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="blue")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(f"{model_name} - ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_roc_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved ROC curve plot")


def plot_precision_recall_curve(y_true, y_probs, model_name, save_dir):
    """Plot Precision-Recall curve."""
    os.makedirs(save_dir, exist_ok=True)
    y_prob_pos = y_probs[:, 1]
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob_pos)
    ap_score = average_precision_score(y_true, y_prob_pos)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall_vals, precision_vals, "b-", linewidth=2, label=f"PR Curve (AP = {ap_score:.4f})")
    ax.fill_between(recall_vals, precision_vals, alpha=0.1, color="blue")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{model_name} - Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_precision_recall_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved Precision-Recall curve plot")


def plot_prediction_distribution(y_true, y_probs, model_name, save_dir):
    """Plot distribution of prediction probabilities separated by true class."""
    os.makedirs(save_dir, exist_ok=True)
    y_prob_pos = y_probs[:, 1]

    fig, ax = plt.subplots(figsize=(10, 6))
    # Separate by true class for better insight
    benign_probs = y_prob_pos[y_true == 0]
    malignant_probs = y_prob_pos[y_true == 1]

    ax.hist(benign_probs, bins=50, alpha=0.6, color="green", label="True Benign", edgecolor="black", linewidth=0.5)
    ax.hist(malignant_probs, bins=50, alpha=0.6, color="red", label="True Malignant", edgecolor="black", linewidth=0.5)
    ax.axvline(x=0.5, color="black", linestyle="--", linewidth=2, label="Decision Boundary")
    ax.set_xlabel("Predicted Probability (Malignant)")
    ax.set_ylabel("Count")
    ax.set_title(f"{model_name} - Prediction Probability Distribution by True Class")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_prediction_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved prediction distribution plot")


def plot_metrics_bar(metrics, model_name, save_dir):
    """Plot bar chart of key metrics."""
    os.makedirs(save_dir, exist_ok=True)
    keys = ["accuracy", "precision", "recall_sensitivity", "specificity",
            "f1_score", "roc_auc", "matthews_corrcoef", "cohen_kappa"]
    values = [metrics.get(k, 0) for k in keys]
    labels = ["Accuracy", "Precision", "Recall\n(Sensitivity)", "Specificity",
              "F1 Score", "ROC AUC", "MCC", "Cohen\nKappa"]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("viridis", len(keys))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylim([0, 1.15])
    ax.set_ylabel("Score")
    ax.set_title(f"{model_name} - Performance Metrics Summary")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_metrics_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved metrics bar chart")


def plot_class_wise_metrics(y_true, y_pred, model_name, save_dir):
    """Plot per-class precision, recall, F1."""
    os.makedirs(save_dir, exist_ok=True)

    report = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, output_dict=True)

    classes = config.CLASS_NAMES
    precision_vals = [report[c]["precision"] for c in classes]
    recall_vals = [report[c]["recall"] for c in classes]
    f1_vals = [report[c]["f1-score"] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, precision_vals, width, label="Precision", color="#2196F3", edgecolor="black")
    ax.bar(x, recall_vals, width, label="Recall", color="#4CAF50", edgecolor="black")
    ax.bar(x + width, f1_vals, width, label="F1-Score", color="#FF9800", edgecolor="black")

    ax.set_ylabel("Score")
    ax.set_title(f"{model_name} - Class-wise Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim([0, 1.15])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_classwise_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved class-wise metrics plot")


def plot_calibration_curve(y_true, y_probs, model_name, save_dir, n_bins=10):
    """Plot reliability diagram (calibration curve)."""
    os.makedirs(save_dir, exist_ok=True)
    y_prob_pos = y_probs[:, 1]

    # Compute calibration bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_means = []
    bin_true_fractions = []
    bin_counts = []

    for i in range(n_bins):
        mask = (y_prob_pos >= bin_edges[i]) & (y_prob_pos < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_means.append(y_prob_pos[mask].mean())
            bin_true_fractions.append(y_true[mask].mean())
            bin_counts.append(mask.sum())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Calibration curve
    axes[0].plot([0, 1], [0, 1], "r--", linewidth=1, label="Perfectly Calibrated")
    axes[0].plot(bin_means, bin_true_fractions, "b-o", linewidth=2, markersize=6, label=model_name)
    axes[0].set_xlabel("Mean Predicted Probability")
    axes[0].set_ylabel("Fraction of Positives")
    axes[0].set_title(f"{model_name} - Calibration Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Histogram of predictions
    axes[1].hist(y_prob_pos, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    axes[1].set_xlabel("Predicted Probability")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"{model_name} - Prediction Histogram")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_calibration.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved calibration curve plot")


def generate_all_plots(y_true, y_pred, y_probs, history, model_name, save_dir):
    """Generate all plots for a model."""
    print(f"\nGenerating plots for {model_name}...")
    plot_training_curves(history, model_name, save_dir)
    plot_confusion_matrix(y_true, y_pred, model_name, save_dir)
    plot_roc_curve(y_true, y_probs, model_name, save_dir)
    plot_precision_recall_curve(y_true, y_probs, model_name, save_dir)
    plot_prediction_distribution(y_true, y_probs, model_name, save_dir)
    metrics = compute_all_metrics(y_true, y_pred, y_probs)
    plot_metrics_bar(metrics, model_name, save_dir)
    plot_class_wise_metrics(y_true, y_pred, model_name, save_dir)
    plot_calibration_curve(y_true, y_probs, model_name, save_dir)
    return metrics


def save_results(metrics, history, model_name, save_dir):
    """Save metrics and history to JSON."""
    os.makedirs(save_dir, exist_ok=True)

    metrics_path = os.path.join(save_dir, f"{model_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    history_clean = {}
    for k, v in history.items():
        if isinstance(v, list):
            history_clean[k] = [float(x) if isinstance(x, (np.floating, float)) else x for x in v]
        else:
            history_clean[k] = float(v) if isinstance(v, (np.floating, float)) else v

    history_path = os.path.join(save_dir, f"{model_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history_clean, f, indent=2)

    print(f"  Saved results to {save_dir}")


def evaluate_per_source(y_true, y_pred, y_probs, sources, model_name):
    """
    Evaluate and print metrics broken down by source dataset.
    This reveals whether the model genuinely generalizes or is exploiting modality shortcuts.
    """
    unique_sources = sorted(set(sources))
    # Collapse BreakHis magnifications into one group for summary
    source_groups = {}
    for src in unique_sources:
        src_name = str(src)
        group = "BreakHis" if src_name.startswith("BreakHis") else src_name
        if group not in source_groups:
            source_groups[group] = []
        source_groups[group].append(src_name)

    print(f"\n{'='*60}")
    print(f"  Per-Dataset Evaluation: {model_name}")
    print(f"{'='*60}")

    per_source_metrics = {}

    for group_name, group_sources in source_groups.items():
        mask = np.isin(sources, group_sources)
        if mask.sum() == 0:
            continue

        gt = y_true[mask]
        pr = y_pred[mask]
        pb = y_probs[mask]

        n_benign = (gt == 0).sum()
        n_malignant = (gt == 1).sum()
        acc = float(accuracy_score(gt, pr))

        print(f"\n  --- {group_name} ({mask.sum()} images: {n_benign} benign, {n_malignant} malignant) ---")
        print(f"    Accuracy:    {acc:.4f}")

        if len(np.unique(gt)) > 1:
            prec = float(precision_score(gt, pr, average="binary", zero_division=0))
            rec = float(recall_score(gt, pr, average="binary", zero_division=0))
            spec = float(recall_score(gt, pr, average="binary", pos_label=0, zero_division=0))
            f1 = float(f1_score(gt, pr, average="binary", zero_division=0))
            try:
                auc = float(roc_auc_score(gt, pb[:, 1]))
            except ValueError:
                auc = 0.0
            print(f"    Precision:   {prec:.4f}")
            print(f"    Recall:      {rec:.4f}")
            print(f"    Specificity: {spec:.4f}")
            print(f"    F1:          {f1:.4f}")
            print(f"    AUC:         {auc:.4f}")
            per_source_metrics[group_name] = {
                "n_images": int(mask.sum()), "accuracy": acc,
                "precision": prec, "recall": rec, "specificity": spec,
                "f1": f1, "auc": auc,
            }
        else:
            print(f"    (only one class present, skipping detailed metrics)")
            per_source_metrics[group_name] = {
                "n_images": int(mask.sum()), "accuracy": acc,
            }

    print(f"{'='*60}")
    return per_source_metrics


def full_evaluate_and_plot(model, test_loader, model_name, history, device):
    """
    Run full evaluation on test set, compute metrics, generate all plots, save results.
    Also reports per-dataset metrics if source tracking is available.
    """
    save_dir = os.path.join(config.RESULTS_DIR, model_name)
    # Use label smoothing-free loss for clean evaluation
    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc, y_pred, y_true, y_probs = evaluate(model, test_loader, criterion, device)
    print(f"\n  Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    print(f"\n{classification_report(y_true, y_pred, target_names=config.CLASS_NAMES)}")

    metrics = compute_all_metrics(y_true, y_pred, y_probs)
    metrics["test_loss"] = float(test_loss)
    metrics["test_accuracy"] = float(test_acc)

    # Per-dataset breakdown (if sources are tracked)
    if hasattr(test_loader.dataset, "sources") and test_loader.dataset.sources is not None:
        sources = np.array(test_loader.dataset.sources)
        per_source = evaluate_per_source(y_true, y_pred, y_probs, sources, model_name)
        metrics["per_source_metrics"] = per_source

    # Case-level metrics (if group IDs are tracked)
    if (
        getattr(config, "REPORT_CASE_LEVEL_METRICS", False)
        and hasattr(test_loader.dataset, "group_ids")
        and test_loader.dataset.group_ids is not None
    ):
        case_metrics = compute_case_level_metrics(y_true, y_probs, test_loader.dataset.group_ids)
        if case_metrics:
            metrics["case_level_metrics"] = case_metrics

            print(f"\n{'='*60}")
            print(f"  Case-Level Evaluation: {model_name}")
            print(f"{'='*60}")
            print(f"  Cases evaluated: {case_metrics['n_cases_evaluated']} / {case_metrics['n_cases_total']}")
            print(f"  Case accuracy:   {case_metrics['accuracy']:.4f}")
            print(f"  Case F1 score:   {case_metrics['f1_score']:.4f}")
            print(f"  Case ROC AUC:    {case_metrics['roc_auc']:.4f}")

            # In strict mode, promote case-level metrics as the primary headline result.
            if getattr(config, "AUTHENTIC_EVAL", False):
                metric_keys = [
                    "accuracy", "precision", "recall_sensitivity", "specificity",
                    "f1_score", "f1_macro", "f1_weighted", "roc_auc",
                    "average_precision", "matthews_corrcoef", "cohen_kappa",
                    "true_positives", "true_negatives", "false_positives", "false_negatives",
                ]
                metrics["image_level_metrics"] = {
                    k: metrics[k] for k in metric_keys if k in metrics
                }
                metrics["image_level_metrics"]["n_images"] = int(len(y_true))

                for key in metric_keys:
                    if key in case_metrics:
                        metrics[key] = case_metrics[key]

                metrics["test_accuracy"] = float(case_metrics["accuracy"])
                metrics["evaluation_level"] = "case_level"

    if getattr(config, "AUTHENTIC_EVAL", False) and metrics.get("accuracy", 0.0) >= 0.98:
        metrics["authenticity_note"] = (
            "Very high in-domain score under strict protocol. "
            "Use external-dataset validation before making clinical claims."
        )

    if getattr(config, "AUTHENTIC_EVAL", False):
        eval_units = int(metrics["true_positives"] + metrics["true_negatives"] +
                         metrics["false_positives"] + metrics["false_negatives"])
        correct = int(metrics["true_positives"] + metrics["true_negatives"])
        acc_lb, acc_ub = wilson_accuracy_interval(correct, eval_units)

        metrics["accuracy_point_estimate"] = float(correct / eval_units) if eval_units > 0 else 0.0
        metrics["accuracy_ci95_lower"] = acc_lb
        metrics["accuracy_ci95_upper"] = acc_ub
        metrics["accuracy_reporting"] = "wilson_95_lower_bound"

        # Conservative headline accuracy for authentic reporting.
        metrics["accuracy"] = acc_lb
        metrics["test_accuracy"] = acc_lb

    print_metrics(metrics, model_name)

    generate_all_plots(y_true, y_pred, y_probs, history, model_name, save_dir)
    save_results(metrics, history, model_name, save_dir)

    return metrics
