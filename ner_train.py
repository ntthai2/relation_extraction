"""Training script for the NER model."""

import argparse
import os
import random

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from config import ENTITY_MARKERS, HIDDEN_SIZE, MODEL_NAME
from ner_dataset import (
    NERDataset,
    NER_ID2LABEL,
    NER_LABEL2ID,
    NER_LABELS,
    NUM_NER_LABELS,
)
from ner_model import SciBERTNER


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_token_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    model.eval()
    all_gold: list[int] = []
    all_pred: list[int] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = logits.argmax(dim=-1)

            valid_mask = labels != -100
            gold = labels[valid_mask].detach().cpu().tolist()
            pred = preds[valid_mask].detach().cpu().tolist()

            all_gold.extend(gold)
            all_pred.extend(pred)

    return all_gold, all_pred


def token_f1_excluding_o(gold: list[int], pred: list[int]) -> float:
    non_o_mask = [g != NER_LABEL2ID["O"] for g in gold]
    gold_non_o = [g for g, keep in zip(gold, non_o_mask) if keep]
    pred_non_o = [p for p, keep in zip(pred, non_o_mask) if keep]

    if not gold_non_o:
        return 0.0

    return float(
        f1_score(
            gold_non_o,
            pred_non_o,
            average="macro",
            labels=[1, 2, 3, 4, 5, 6],
            zero_division=0,
        )
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits.view(-1, NUM_NER_LABELS), labels.view(-1))
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


def print_per_class_report(gold: list[int], pred: list[int]) -> None:
    labels = [1, 2, 3, 4, 5, 6]
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold,
        pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    print("\nPer-class token metrics (test):")
    print(f"{'Label':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    for i, label_id in enumerate(labels):
        print(
            f"{NER_ID2LABEL[label_id]:<12} "
            f"{precision[i]:>10.4f} "
            f"{recall[i]:>10.4f} "
            f"{f1[i]:>10.4f}"
        )


def train_ner(
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 2e-5,
    max_len: int = 256,
    patience: int = 5,
    seeds: list[int] = [42],
    output_dir: str = "runs/ner",
    smoke_test: bool = False,
) -> None:
    if HIDDEN_SIZE != 768:
        print(f"Warning: HIDDEN_SIZE from config is {HIDDEN_SIZE} (expected SciBERT size 768).")

    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ENTITY_MARKERS})

    dataset_max_samples = 128 if smoke_test else 0
    effective_epochs = 1 if smoke_test else epochs

    train_ds = NERDataset("scierc/train.json", tokenizer, max_len=max_len, max_samples=dataset_max_samples)
    dev_ds = NERDataset("scierc/dev.json", tokenizer, max_len=max_len, max_samples=dataset_max_samples)
    test_ds = NERDataset("scierc/test.json", tokenizer, max_len=max_len, max_samples=dataset_max_samples)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Train/Dev/Test samples: {len(train_ds)}/{len(dev_ds)}/{len(test_ds)}")

    for seed in seeds:
        set_seed(seed)
        print(f"\n=== Seed {seed} ===")

        if len(seeds) == 1:
            seed_output_dir = output_dir
        else:
            seed_output_dir = os.path.join(output_dir, f"seed_{seed}")
            os.makedirs(seed_output_dir, exist_ok=True)

        best_ckpt_path = os.path.join(seed_output_dir, "best_ner_model.pt")

        model = SciBERTNER(num_labels=NUM_NER_LABELS)
        model.bert.resize_token_embeddings(len(tokenizer))
        model.to(device)

        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        total_steps = max(1, len(train_loader) * effective_epochs)
        warmup_steps = int(total_steps * 0.1)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        best_dev_f1 = -1.0
        epochs_no_improve = 0

        for epoch in range(1, effective_epochs + 1):
            train_loss = run_epoch(model, train_loader, optimizer, scheduler, criterion, device)
            dev_gold, dev_pred = collect_token_predictions(model, dev_loader, device)
            dev_f1 = token_f1_excluding_o(dev_gold, dev_pred)

            print(f"Epoch {epoch:02d}/{effective_epochs} | train_loss={train_loss:.4f} | dev_token_f1={dev_f1:.4f}")

            if dev_f1 > best_dev_f1 + 1e-4:
                best_dev_f1 = dev_f1
                epochs_no_improve = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "seed": seed,
                        "best_dev_f1": best_dev_f1,
                        "labels": NER_LABELS,
                    },
                    best_ckpt_path,
                )
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience}).")
                break

        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        test_gold, test_pred = collect_token_predictions(model, test_loader, device)
        test_f1 = token_f1_excluding_o(test_gold, test_pred)
        print(f"Best dev token F1 (excl O): {best_dev_f1:.4f}")
        print(f"Test token F1 (excl O): {test_f1:.4f}")
        print_per_class_report(test_gold, test_pred)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SciBERT NER with BIO tagging on SciERC.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-len", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds, e.g. '13,42,87'")
    parser.add_argument("--output-dir", default="runs/ner", help="Output directory")
    parser.add_argument("--smoke-test", action="store_true", help="Run tiny quick check (128 samples, 1 epoch)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    train_ner(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_len=args.max_len,
        patience=args.patience,
        seeds=seeds,
        output_dir=args.output_dir,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
