"""Experiment runner: orchestrates multi-seed runs with different model/loss configs."""

import json
import random
import os
import torch
import numpy as np
from collections import Counter
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer

from config import (
    LABELS, LABEL2ID, NUM_LABELS, MODEL_NAME, DEBERTA_MODEL_NAME,
    ROBERTA_LARGE_MODEL_NAME, BERT_LARGE_MODEL_NAME, LEVITATED_MARKERS,
    BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, WARMUP_RATIO, SEED,
    PATIENCE, MIN_DELTA, DEFAULT_OUTPUT_DIR, ENTITY_MARKERS, USE_ENTITY_MARKERS,
    TRAIN_FILE, DEV_FILE, TEST_FILE, MODEL_VARIANTS, LOSS_VARIANTS,
    MAX_LEN, DEFAULT_RUN_NAME
)
from re_dataset import (
    SciERCDataset, PLMarkerSciERCDataset, augment_symmetric, undersample_label
)
from re_model import (
    SciBERTRelationClassifier, FrozenSciBERTRelationClassifier,
    DeBERTaRelationClassifier, RoBERTaLargeRelationClassifier, BERTLargeRelationClassifier,
    SpERTRelationClassifier, PLMarkerRelationClassifier, PURELiteRelationClassifier, FocalLoss
)
from re_train_core import train_model, train_spert_model
from re_eval import compute_metrics, save_test_artifacts, aggregate_results


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
    elif loss_variant == "focal":
        class_weights = compute_class_weights(device)
        return FocalLoss(gamma=2.0, weight=class_weights)
    elif loss_variant == "label_smooth":
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        raise ValueError(f"Unknown loss variant: {loss_variant}")


def make_two_group_optimizer(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    head_lr_multiplier: float = 10.0,
) -> AdamW:
    """
    Create optimizer with separate learning rates for encoder and classifier head.
    Encoder gets base_lr; classifier head gets base_lr * head_lr_multiplier.
    """
    return AdamW([
        {"params": model.bert.parameters(), "lr": base_lr, "weight_decay": weight_decay},
        {"params": model.classifier.parameters(), "lr": base_lr * head_lr_multiplier, "weight_decay": 0.0},
    ])


def make_llrd_optimizer(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    decay_factor: float = 0.9,
) -> AdamW:
    """
    Layer-wise learning rate decay. Assigns per-layer decaying LRs.
    Classifier head: base_lr * 2
    Top BERT layer: base_lr
    Each lower layer: multiplied by decay_factor
    Embeddings: lowest LR
    """
    num_layers = model.bert.config.num_hidden_layers
    param_groups = []
    param_groups.append({"params": model.classifier.parameters(),
                          "lr": base_lr * 2, "weight_decay": 0.0})
    for layer_idx in range(num_layers - 1, -1, -1):
        lr = base_lr * (decay_factor ** (num_layers - 1 - layer_idx))
        param_groups.append({"params": model.bert.encoder.layer[layer_idx].parameters(),
                              "lr": lr, "weight_decay": weight_decay})
    param_groups.append({"params": model.bert.embeddings.parameters(),
                          "lr": base_lr * (decay_factor ** num_layers),
                          "weight_decay": weight_decay})
    return AdamW(param_groups)


# ────────────────────────────────────────────────────────────────────────────────
# Model and Dataset Registries
# ────────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "scibert":       SciBERTRelationClassifier,
    "deberta":       DeBERTaRelationClassifier,
    "roberta_large": RoBERTaLargeRelationClassifier,
    "bert_large":    BERTLargeRelationClassifier,
    "spert":         SpERTRelationClassifier,
    "plmarker":      PLMarkerRelationClassifier,
    "pure_lite":     PURELiteRelationClassifier,
}

DATASET_REGISTRY = {
    "scibert":       SciERCDataset,
    "deberta":       SciERCDataset,
    "roberta_large": SciERCDataset,
    "bert_large":    SciERCDataset,
    "spert":         SciERCDataset,
    "plmarker":      PLMarkerSciERCDataset,
    "pure_lite":     SciERCDataset,
}


def get_model_name(model_type: str) -> str:
    """Map model_type CLI string to HuggingFace model name string."""
    return {
        "scibert": MODEL_NAME,
        "deberta": DEBERTA_MODEL_NAME,
        "roberta_large": ROBERTA_LARGE_MODEL_NAME,
        "bert_large": BERT_LARGE_MODEL_NAME,
        "spert": MODEL_NAME,
        "plmarker": MODEL_NAME,
        "pure_lite": MODEL_NAME,
    }[model_type]


def run_single_seed(
    seed: int,
    model_variant: str,
    loss_variant: str,
    frozen_bert: bool,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    patience: int,
    min_delta: float,
    batch_size: int,
    max_len: int,
    max_samples: int,
    output_dir: str,
    model_type: str = "scibert",
    separate_lr: bool = False,
    llrd: bool = False,
    grad_accum_steps: int = 1,
    augment: bool = False,
    undersample_conjunction: bool = False,
    undersample_target: int = 250,
) -> dict:
    """Run a single training seed. Returns result dict with dev/test macro F1."""

    set_seed(seed)

    # Tokenizer
    model_name = get_model_name(model_type)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    special_tokens = {"additional_special_tokens": list(ENTITY_MARKERS)}
    if model_type == "plmarker":
        special_tokens["additional_special_tokens"] += LEVITATED_MARKERS
    tokenizer.add_special_tokens(special_tokens)

    # Dataset
    DatasetClass = DATASET_REGISTRY[model_type]
    train_set = DatasetClass(
        TRAIN_FILE,
        tokenizer,
        max_len=max_len,
        max_samples=max_samples,
        use_entity_markers=USE_ENTITY_MARKERS,
    )
    if augment:
        train_set.samples = augment_symmetric(train_set.samples)
    if undersample_conjunction:
        train_set.samples = undersample_label(
            train_set.samples,
            "CONJUNCTION",
            undersample_target,
            seed=seed,
        )

    dev_set = DatasetClass(
        DEV_FILE,
        tokenizer,
        max_len=max_len,
        max_samples=max_samples,
        use_entity_markers=USE_ENTITY_MARKERS,
    )
    test_set = DatasetClass(
        TEST_FILE,
        tokenizer,
        max_len=max_len,
        max_samples=max_samples,
        use_entity_markers=USE_ENTITY_MARKERS,
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ModelClass = MODEL_REGISTRY[model_type]
    pooling = MODEL_VARIANTS[model_variant]["pooling"]

    if frozen_bert:
        model = FrozenSciBERTRelationClassifier(pooling=pooling)
    else:
        if model_type in {"spert", "plmarker", "pure_lite"}:
            model = ModelClass()
        else:
            model = ModelClass(pooling=pooling)

    model.bert.resize_token_embeddings(len(tokenizer))
    model = model.float().to(device)

    # Optimizer
    if separate_lr and llrd:
        raise ValueError("--separate-lr and --llrd are mutually exclusive.")
    if llrd:
        optimizer = make_llrd_optimizer(model, learning_rate, weight_decay)
    elif separate_lr:
        optimizer = make_two_group_optimizer(model, learning_rate, weight_decay)
    else:
        optimizer = None

    # Loss
    criterion = get_criterion(loss_variant, device)

    # Run directory
    run_dir = os.path.join(
        output_dir,
        f"{model_type}_{model_variant}_{loss_variant}"
        f"_frozen{int(frozen_bert)}"
        f"_llrd{int(llrd)}"
        f"_aug{int(augment)}"
        f"_seed{seed}",
    )
    os.makedirs(run_dir, exist_ok=True)
    model_save_path = os.path.join(run_dir, "best_model.pt")

    # Train
    if model_type == "spert":
        result = train_spert_model(
            model=model,
            train_loader=train_loader,
            dev_loader=dev_loader,
            test_loader=test_loader,
            device=device,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            patience=patience,
            min_delta=min_delta,
            criterion=criterion,
            model_save_path=model_save_path,
            optimizer=optimizer,
            grad_accum_steps=grad_accum_steps,
        )
    else:
        result = train_model(
            model=model,
            train_loader=train_loader,
            dev_loader=dev_loader,
            test_loader=test_loader,
            device=device,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            patience=patience,
            min_delta=min_delta,
            criterion=criterion,
            model_save_path=model_save_path,
            optimizer=optimizer,
            grad_accum_steps=grad_accum_steps,
        )

    save_test_artifacts(run_dir, result["test_labels"], result["test_preds"])
    return {
        "seed": seed,
        "model_variant": model_variant,
        "loss_variant": loss_variant,
        "frozen_bert": frozen_bert,
        "best_epoch": result["best_epoch"],
        "best_dev_macro_f1": result["best_dev_macro_f1"],
        "test_macro_f1": result["test_macro_f1"],
        "run_dir": run_dir,
    }


def run_experiment_suite(
    model_variants: list[str],
    loss_variants: list[str],
    seeds: list[int],
    frozen_bert: bool = False,
    num_epochs: int = EPOCHS,
    learning_rate: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    warmup_ratio: float = WARMUP_RATIO,
    patience: int = PATIENCE,
    min_delta: float = MIN_DELTA,
    batch_size: int = BATCH_SIZE,
    max_len: int = MAX_LEN,
    max_samples: int = 0,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    run_name: str = DEFAULT_RUN_NAME,
    model_type: str = "scibert",
    separate_lr: bool = False,
    llrd: bool = False,
    grad_accum_steps: int = 1,
    augment: bool = False,
    undersample_conjunction: bool = False,
    undersample_target: int = 250,
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
    frozen_variants = [frozen_bert]

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
                        model_type=model_type,
                        separate_lr=separate_lr,
                        llrd=llrd,
                        grad_accum_steps=grad_accum_steps,
                        augment=augment,
                        undersample_conjunction=undersample_conjunction,
                        undersample_target=undersample_target,
                        num_epochs=num_epochs,
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        warmup_ratio=warmup_ratio,
                        patience=patience,
                        min_delta=min_delta,
                        batch_size=batch_size,
                        max_len=max_len,
                        max_samples=max_samples,
                        output_dir=output_root,
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
