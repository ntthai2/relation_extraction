"""Experiment runner: orchestrates multi-seed runs with different model/loss configs."""

import json
import random
import os
import torch
import numpy as np
from collections import Counter
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import (
    LABELS, LABEL2ID, NUM_LABELS, MODEL_NAME,
    BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, WARMUP_RATIO, SEED,
    PATIENCE, MIN_DELTA, DEFAULT_OUTPUT_DIR, ENTITY_MARKERS, USE_ENTITY_MARKERS,
    TRAIN_FILE, DEV_FILE, TEST_FILE, MODEL_VARIANTS, LOSS_VARIANTS
)
from dataset import SciERCDataset
from model import SciBERTRelationClassifier, FrozenSciBERTRelationClassifier
from train_core import train_model
from eval import compute_metrics, save_test_artifacts, aggregate_results


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_class_weights(device):
    """Compute class weights from training data for weighted loss."""
    labels = []
    with open(TRAIN_FILE, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            labels.append(LABEL2ID[item["label"]])
    
    counts = Counter(labels)
    total = sum(counts.values())
    weights = [total / (len(LABELS) * counts[i]) for i in range(len(LABELS))]
    return torch.tensor(weights, dtype=torch.float).to(device)


def get_criterion(loss_variant, device):
    """Get loss function based on variant."""
    if loss_variant == "weighted_ce":
        class_weights = compute_class_weights(device)
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_variant == "ce_uniform":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss variant: {loss_variant}")


def run_single_seed(
    seed,
    model_variant,
    loss_variant,
    frozen_bert,
    max_len,
    batch_size,
    num_epochs,
    lr,
    weight_decay,
    warmup_ratio,
    patience,
    min_delta,
    output_dir,
    max_samples_per_split=0,
):
    """
    Run a single experiment with given hyperparams.
    
    Returns:
        dict: Experiment results
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Seed: {seed} | Model: {model_variant} | Loss: {loss_variant} | Frozen: {frozen_bert}")
    print(f"{'='*60}")
    print(f"Device: {device}")

    # Setup tokenizer and add special tokens if needed
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if USE_ENTITY_MARKERS:
        special_tokens = {"additional_special_tokens": ENTITY_MARKERS}
        tokenizer.add_special_tokens(special_tokens)

    # Setup datasets
    train_set = SciERCDataset(
        TRAIN_FILE,
        tokenizer,
        max_len=max_len,
        max_samples=max_samples_per_split,
        use_entity_markers=USE_ENTITY_MARKERS,
    )
    dev_set = SciERCDataset(
        DEV_FILE,
        tokenizer,
        max_len=max_len,
        max_samples=max_samples_per_split,
        use_entity_markers=USE_ENTITY_MARKERS,
    )
    test_set = SciERCDataset(
        TEST_FILE,
        tokenizer,
        max_len=max_len,
        max_samples=max_samples_per_split,
        use_entity_markers=USE_ENTITY_MARKERS,
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    print(f"Train: {len(train_set)} | Dev: {len(dev_set)} | Test: {len(test_set)}")

    # Select model
    pooling_strategy = MODEL_VARIANTS[model_variant]["pooling"]
    if frozen_bert:
        model = FrozenSciBERTRelationClassifier(
            num_labels=NUM_LABELS,
            dropout=0.1,
            pooling=pooling_strategy
        )
    else:
        model = SciBERTRelationClassifier(
            num_labels=NUM_LABELS,
            dropout=0.1,
            pooling=pooling_strategy
        )

    # Resize token embeddings if special tokens were added
    if USE_ENTITY_MARKERS:
        model.bert.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # Get loss function
    criterion = get_criterion(loss_variant, device)

    # Prepare output directory
    run_subdir = os.path.join(
        output_dir,
        f"{model_variant}_{loss_variant}_frozen{int(frozen_bert)}_seed{seed}"
    )
    os.makedirs(run_subdir, exist_ok=True)
    model_save_path = os.path.join(run_subdir, "best_model.pt")

    # Train model
    train_results = train_model(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        device=device,
        num_epochs=num_epochs,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        patience=patience,
        min_delta=min_delta,
        criterion=criterion,
        model_save_path=model_save_path,
    )

    # Save artifacts
    save_test_artifacts(run_subdir, train_results["test_labels"], train_results["test_preds"])

    return {
        "seed": seed,
        "model_variant": model_variant,
        "loss_variant": loss_variant,
        "frozen_bert": frozen_bert,
        "best_epoch": train_results["best_epoch"],
        "best_dev_macro_f1": train_results["best_dev_macro_f1"],
        "test_macro_f1": train_results["test_macro_f1"],
        "run_dir": run_subdir,
    }


def run_experiment_suite(
    model_variants=None,
    loss_variants=None,
    frozen_variants=None,
    seeds=None,
    max_len=256,
    batch_size=32,
    num_epochs=10,
    lr=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    patience=3,
    min_delta=1e-4,
    output_dir=DEFAULT_OUTPUT_DIR,
    run_name="experiment",
    max_samples_per_split=0,
):
    """
    Run a comprehensive experiment suite with different configurations.
    
    Args:
        model_variants: List of model variant keys (e.g., ["e1e2_concat", "cls_only"])
        loss_variants: List of loss variant keys (e.g., ["weighted_ce", "ce_uniform"])
        frozen_variants: List of booleans for frozen BERT experiments
        seeds: List of random seeds
        [other hyperparams...]
        run_name: Experiment name for output folder
    
    Returns:
        dict: Aggregated results across all experiments
    """
    if model_variants is None:
        model_variants = ["e1e2_concat"]
    if loss_variants is None:
        loss_variants = ["weighted_ce"]
    if frozen_variants is None:
        frozen_variants = [False]
    if seeds is None:
        seeds = [42]

    output_root = os.path.join(output_dir, run_name)
    os.makedirs(output_root, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"Experiment Suite: {run_name}")
    print(f"{'#'*60}")
    print(f"Output dir: {output_root}")
    print(f"Model variants: {model_variants}")
    print(f"Loss variants: {loss_variants}")
    print(f"Frozen BERT: {frozen_variants}")
    print(f"Seeds: {seeds}")

    all_runs = []

    for model_var in model_variants:
        for loss_var in loss_variants:
            for frozen in frozen_variants:
                for seed in seeds:
                    result = run_single_seed(
                        seed=seed,
                        model_variant=model_var,
                        loss_variant=loss_var,
                        frozen_bert=frozen,
                        max_len=max_len,
                        batch_size=batch_size,
                        num_epochs=num_epochs,
                        lr=lr,
                        weight_decay=weight_decay,
                        warmup_ratio=warmup_ratio,
                        patience=patience,
                        min_delta=min_delta,
                        output_dir=output_root,
                        max_samples_per_split=max_samples_per_split,
                    )
                    all_runs.append(result)

    # Summary for each (model, loss, frozen) combination
    summary = {}
    for model_var in model_variants:
        for loss_var in loss_variants:
            for frozen in frozen_variants:
                key = f"{model_var}_{loss_var}_frozen{int(frozen)}"
                matching_runs = [
                    r for r in all_runs
                    if r["model_variant"] == model_var
                    and r["loss_variant"] == loss_var
                    and r["frozen_bert"] == frozen
                ]
                summary[key] = {
                    "config": {
                        "model_variant": model_var,
                        "loss_variant": loss_var,
                        "frozen_bert": frozen,
                    },
                    "runs": matching_runs,
                    "aggregate": aggregate_results(matching_runs),
                }

    # Save summary
    summary_path = os.path.join(output_root, "experiment_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'#'*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'#'*60}")
    for key, results in summary.items():
        agg = results["aggregate"]
        print(f"\n{key}:")
        print(f"  Runs: {agg['n_runs']}")
        print(f"  Dev F1:  {agg['dev_macro_f1_mean']:.4f} ± {agg['dev_macro_f1_std']:.4f}")
        print(f"  Test F1: {agg['test_macro_f1_mean']:.4f} ± {agg['test_macro_f1_std']:.4f}")

    print(f"\nSummary saved to: {summary_path}")
    return summary
