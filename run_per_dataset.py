"""
Per-Dataset Training Script.
Trains each model (MobileViT, EfficientViT, HybridViT) separately on each dataset
(DDSM, BreakHis, BUS_UC), producing honest per-domain metrics.

This avoids the accuracy inflation caused by mixing datasets of different difficulty.
Results are saved under results/<Model>_<Dataset>/ for each combination.
"""

import os
import sys
import time
import json
import torch

import config
from dataset import create_dataloaders_single
from utils import train_model, full_evaluate_and_plot

# Import model builders
from train_mobilevit import build_mobilevit_model
from train_efficientvit import build_efficientvit_model
from train_hybrid import build_hybrid_model


DATASETS = ["DDSM", "BreakHis", "BUS_UC"]
MODEL_BUILDERS = {
    "MobileViT": build_mobilevit_model,
    "EfficientViT": build_efficientvit_model,
    "HybridViT": build_hybrid_model,
}


def train_single_combination(model_name, dataset_name, device):
    """
    Train a single model on a single dataset.

    Returns:
        metrics dict or None on failure
    """
    run_name = f"{model_name}_{dataset_name}"
    print(f"\n{'#' * 70}")
    print(f"#  {run_name}")
    print(f"#  Model: {model_name}  |  Dataset: {dataset_name}")
    print(f"{'#' * 70}\n")

    start = time.time()

    # Load data for this specific dataset
    print(f"--- Loading Data: {dataset_name} ---")
    train_loader, val_loader, test_loader = create_dataloaders_single(dataset_name)

    # Build a fresh model
    print(f"\n--- Building {model_name} Model ---")
    builder = MODEL_BUILDERS[model_name]
    model = builder()

    # Train
    model, history = train_model(model, train_loader, val_loader, run_name, device)

    # Evaluate & Plot
    print(f"\n--- Final Evaluation: {run_name} ---")
    metrics = full_evaluate_and_plot(model, test_loader, run_name, history, device)

    elapsed = time.time() - start
    print(f"\n[OK] {run_name} completed in {elapsed / 60:.1f} minutes")

    return metrics, elapsed


def print_comparison_table(all_results):
    """Print a comprehensive comparison table of all model × dataset results."""
    print(f"\n{'=' * 90}")
    print(f"  PER-DATASET MODEL COMPARISON SUMMARY")
    print(f"{'=' * 90}")

    metric_keys = ["accuracy", "precision", "recall_sensitivity", "specificity",
                   "f1_score", "roc_auc", "matthews_corrcoef"]
    metric_display = ["Accuracy", "Precision", "Recall", "Specificity",
                      "F1 Score", "ROC AUC", "MCC"]

    for dataset_name in DATASETS:
        print(f"\n  {'─' * 70}")
        print(f"  Dataset: {dataset_name}")
        print(f"  {'─' * 70}")

        # Header
        header = f"  {'Metric':<25}"
        available_models = []
        for model_name in MODEL_BUILDERS:
            run_name = f"{model_name}_{dataset_name}"
            if run_name in all_results:
                header += f" {model_name:>14}"
                available_models.append(model_name)
        print(header)
        print("  " + "-" * (25 + 15 * len(available_models)))

        # Rows
        for key, display in zip(metric_keys, metric_display):
            row = f"  {display:<25}"
            for model_name in available_models:
                run_name = f"{model_name}_{dataset_name}"
                val = all_results[run_name].get(key, 0)
                row += f" {val:>14.4f}"
            print(row)

    print(f"\n{'=' * 90}")

    # Overall summary: best model per dataset
    print(f"\n  BEST MODEL PER DATASET (by F1 Score):")
    for dataset_name in DATASETS:
        best_model = None
        best_f1 = -1
        for model_name in MODEL_BUILDERS:
            run_name = f"{model_name}_{dataset_name}"
            if run_name in all_results:
                f1 = all_results[run_name].get("f1_score", 0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model_name
        if best_model:
            acc = all_results[f"{best_model}_{dataset_name}"]["accuracy"]
            print(f"    {dataset_name:.<15} {best_model:<15} F1={best_f1:.4f}  Acc={acc:.4f}")

    print(f"{'=' * 90}")


def save_combined_results(all_results, all_times):
    """Save a combined JSON with all per-dataset results."""
    out_dir = os.path.join(config.RESULTS_DIR, "per_dataset_comparison")
    os.makedirs(out_dir, exist_ok=True)

    summary = {}
    for run_name, metrics in all_results.items():
        summary[run_name] = {
            "accuracy": metrics.get("accuracy", 0),
            "precision": metrics.get("precision", 0),
            "recall_sensitivity": metrics.get("recall_sensitivity", 0),
            "specificity": metrics.get("specificity", 0),
            "f1_score": metrics.get("f1_score", 0),
            "roc_auc": metrics.get("roc_auc", 0),
            "matthews_corrcoef": metrics.get("matthews_corrcoef", 0),
            "cohen_kappa": metrics.get("cohen_kappa", 0),
            "average_precision": metrics.get("average_precision", 0),
            "true_positives": metrics.get("true_positives", 0),
            "true_negatives": metrics.get("true_negatives", 0),
            "false_positives": metrics.get("false_positives", 0),
            "false_negatives": metrics.get("false_negatives", 0),
            "training_time_minutes": all_times.get(run_name, 0) / 60,
        }

    path = os.path.join(out_dir, "per_dataset_all_results.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Combined results saved to: {path}")


def main():
    """Train all 3 models × 3 datasets = 9 training runs."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    total_start = time.time()

    all_results = {}
    all_times = {}
    statuses = []

    for dataset_name in DATASETS:
        for model_name in MODEL_BUILDERS:
            run_name = f"{model_name}_{dataset_name}"
            try:
                metrics, elapsed = train_single_combination(model_name, dataset_name, device)
                all_results[run_name] = metrics
                all_times[run_name] = elapsed
                statuses.append((run_name, "OK", elapsed))
            except Exception as e:
                print(f"\n[FAILED] {run_name}: {e}")
                import traceback
                traceback.print_exc()
                statuses.append((run_name, "FAILED", 0))

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  ALL PER-DATASET TRAINING COMPLETE  ({total_elapsed / 60:.1f} minutes total)")
    print(f"{'=' * 70}")
    for name, status, elapsed in statuses:
        print(f"  {name:<30} [{status}]  {elapsed / 60:.1f} min")

    if all_results:
        print_comparison_table(all_results)
        save_combined_results(all_results, all_times)

    # Generate all graphs (paper-style figures, tables, and extras)
    print(f"\n--- Generating All Graphs (Paper Figures + Tables + Extended Analysis) ---")
    try:
        from generate_paper_graphs import main as gen_paper_graphs
        gen_paper_graphs()
    except Exception as e:
        print(f"  [WARNING] Could not generate graphs: {e}")
        print(f"  Run 'python generate_paper_graphs.py' manually after training.")


if __name__ == "__main__":
    main()
