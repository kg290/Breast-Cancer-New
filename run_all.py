"""
Run all models sequentially: MobileViT, EfficientViT, HybridViT.
Each model trains for NUM_EPOCHS (set in config.py), evaluates on test set,
and saves results/checkpoints.
"""

import os
import sys
import json
import time

import config


def run_model(script_name, model_name):
    """Run a training script and report status."""
    print(f"\n{'#'*70}")
    print(f"#  STARTING: {model_name}")
    print(f"#  Script:   {script_name}")
    print(f"{'#'*70}\n")

    start = time.time()
    exit_code = os.system(f'"{sys.executable}" {script_name}')
    elapsed = time.time() - start

    if exit_code == 0:
        print(f"\n[OK] {model_name} completed in {elapsed/60:.1f} minutes")
    else:
        print(f"\n[FAILED] {model_name} exited with code {exit_code}")

    return exit_code, elapsed


def print_summary():
    """Print a comparison table of all trained models."""
    print(f"\n{'='*70}")
    print(f"  MODEL COMPARISON SUMMARY")
    print(f"{'='*70}")

    models = ["MobileViT", "EfficientViT", "HybridViT"]
    metrics_keys = ["accuracy", "precision", "recall_sensitivity", "specificity",
                    "f1_score", "roc_auc", "matthews_corrcoef"]

    results = {}
    for model_name in models:
        metrics_path = os.path.join(config.RESULTS_DIR, model_name, f"{model_name}_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                results[model_name] = json.load(f)

    if not results:
        print("  No results found.")
        return

    # Header
    header = f"  {'Metric':<25}"
    for name in results:
        header += f" {name:>14}"
    print(header)
    print("  " + "-" * (25 + 15 * len(results)))

    # Rows
    for key in metrics_keys:
        row = f"  {key:<25}"
        for name in results:
            val = results[name].get(key, 0)
            row += f" {val:>14.4f}"
        print(row)

    print(f"{'='*70}")


def main():
    scripts = [
        ("train_mobilevit.py", "MobileViT"),
        ("train_efficientvit.py", "EfficientViT"),
        ("train_hybrid.py", "HybridViT"),
    ]

    total_start = time.time()
    statuses = []

    for script, name in scripts:
        code, elapsed = run_model(script, name)
        statuses.append((name, code, elapsed))

    total_elapsed = time.time() - total_start

    print(f"\n{'='*70}")
    print(f"  ALL TRAINING COMPLETE  ({total_elapsed/60:.1f} minutes total)")
    print(f"{'='*70}")
    for name, code, elapsed in statuses:
        status = "OK" if code == 0 else "FAILED"
        print(f"  {name:<20} [{status}]  {elapsed/60:.1f} min")

    print_summary()


if __name__ == "__main__":
    main()
