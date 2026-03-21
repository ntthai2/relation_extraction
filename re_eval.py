"""Evaluation utilities: metrics computation and artifact generation."""

import json
import csv
import os
import torch
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from config import LABELS


def compute_metrics(predictions, labels, average="macro"):
    """
    Compute F1 score.
    
    Args:
        predictions: List/array of predicted labels
        labels: List/array of true labels
        average: sklearn average type ("macro", "weighted", "micro")
    
    Returns:
        float: F1 score
    """
    return f1_score(labels, predictions, average=average, zero_division=0)


def save_test_artifacts(output_dir, test_labels, test_preds):
    """
    Save evaluation results in multiple formats for easy analysis.
    
    Outputs:
    - test_classification_report.json: Full report dict
    - test_per_class_metrics.csv: Per-class table
    - test_confusion_matrix.txt: Human-readable confusion matrix
    - test_confusion_matrix.csv: CSV confusion matrix
    
    Args:
        output_dir: Directory to save artifacts
        test_labels: List of true labels
        test_preds: List of predicted labels
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate classification report
    report_dict = classification_report(
        test_labels,
        test_preds,
        target_names=LABELS,
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    # Save JSON report
    report_path = os.path.join(output_dir, "test_classification_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    # Save per-class metrics as CSV
    table_path = os.path.join(output_dir, "test_per_class_metrics.csv")
    with open(table_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "precision", "recall", "f1", "support"])
        for label in LABELS:
            row = report_dict[label]
            writer.writerow([
                label,
                f"{row['precision']:.6f}",
                f"{row['recall']:.6f}",
                f"{row['f1-score']:.6f}",
                int(row['support']),
            ])

    # Compute and save confusion matrix
    cm = confusion_matrix(test_labels, test_preds, labels=list(range(len(LABELS))))
    
    cm_txt_path = os.path.join(output_dir, "test_confusion_matrix.txt")
    with open(cm_txt_path, "w", encoding="utf-8") as f:
        f.write("Labels order:\n")
        f.write(", ".join(LABELS) + "\n\n")
        f.write("Confusion matrix (rows=true, cols=pred):\n")
        for row in cm:
            f.write(" ".join(f"{int(v):4d}" for v in row) + "\n")

    cm_csv_path = os.path.join(output_dir, "test_confusion_matrix.csv")
    with open(cm_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + LABELS)
        for idx, label in enumerate(LABELS):
            writer.writerow([label] + cm[idx].tolist())


def aggregate_results(runs):
    """
    Aggregate results from multiple runs (seeds).
    
    Args:
        runs: List of dicts, each with keys:
            - seed
            - best_dev_macro_f1
            - test_macro_f1
            - best_epoch
            - run_dir
    
    Returns:
        dict: Aggregated statistics
    """
    if len(runs) == 1:
        return {
            "n_runs": 1,
            "dev_macro_f1_mean": runs[0]["best_dev_macro_f1"],
            "dev_macro_f1_std": 0.0,
            "test_macro_f1_mean": runs[0]["test_macro_f1"],
            "test_macro_f1_std": 0.0,
        }
    else:
        dev_scores = np.array([r["best_dev_macro_f1"] for r in runs], dtype=np.float64)
        test_scores = np.array([r["test_macro_f1"] for r in runs], dtype=np.float64)
        return {
            "n_runs": len(runs),
            "dev_macro_f1_mean": float(dev_scores.mean()),
            "dev_macro_f1_std": float(dev_scores.std(ddof=1)),
            "test_macro_f1_mean": float(test_scores.mean()),
            "test_macro_f1_std": float(test_scores.std(ddof=1)),
        }
