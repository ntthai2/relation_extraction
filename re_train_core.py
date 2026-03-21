"""Core training and evaluation loop."""

import torch
from torch import nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from re_eval import compute_metrics, save_test_artifacts


def train_epoch(model, loader, optimizer, scheduler, criterion, device, grad_accum_steps=1):
    """
    Train for one epoch with optional gradient accumulation.
    
    Args:
        grad_accum_steps: Number of batches to accumulate gradients over (default 1, no accumulation)
    
    Returns:
        float: Average loss over the epoch
    """
    model.train()
    total_loss = 0

    for step, batch in enumerate(loader):
        optimizer.zero_grad()
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            e1_pos=batch["e1_pos"].to(device),
            e2_pos=batch["e2_pos"].to(device),
        )
        loss = criterion(logits, batch["label"].to(device))
        loss = loss / grad_accum_steps  # Scale loss for accumulation
        loss.backward()
        
        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * grad_accum_steps  # Store unscaled loss

    return total_loss / len(loader)


def eval_epoch(model, loader, criterion, device):
    """
    Evaluate on a dataset.
    
    Returns:
        tuple: (avg_loss, macro_f1, predictions, labels)
    """
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                e1_pos=batch["e1_pos"].to(device),
                e2_pos=batch["e2_pos"].to(device),
            )
            loss = criterion(logits, batch["label"].to(device))
            total_loss += loss.item()
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch["label"].tolist())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    avg_loss = total_loss / len(loader)

    return avg_loss, macro_f1, all_preds, all_labels


def train_model(
    model,
    train_loader,
    dev_loader,
    test_loader,
    device,
    num_epochs,
    learning_rate,
    weight_decay,
    warmup_ratio,
    patience,
    min_delta,
    criterion,
    model_save_path,
    optimizer=None,
    grad_accum_steps=1,
):
    """
    Train a model with early stopping, returning final metrics.
    
    Args:
        model: PyTorch model
        train_loader, dev_loader, test_loader: DataLoaders
        device: torch.device
        num_epochs: Max training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay for AdamW
        warmup_ratio: Warmup ratio for scheduler
        patience: Early stopping patience
        min_delta: Min improvement to reset patience
        criterion: Loss function
        model_save_path: Where to save best checkpoint
        optimizer: Optional custom optimizer. If None, AdamW is created.
        grad_accum_steps: Gradient accumulation steps (default 1, no accumulation)
    
    Returns:
        dict: Training metrics (best_epoch, dev_f1, test_f1, etc.)
    """
    if optimizer is None:
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_ratio * total_steps),
        num_training_steps=total_steps
    )

    best_macro_f1 = -1.0
    best_epoch = 0
    no_improve = 0

    print("\n── Training ──────────────────────────────")
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device, grad_accum_steps)
        dev_loss, dev_f1, _, _ = eval_epoch(model, dev_loader, criterion, device)

        improved = dev_f1 > (best_macro_f1 + min_delta)
        marker = " ← best" if improved else ""
        print(f"Epoch {epoch:02d} | train_loss: {train_loss:.4f} | dev_loss: {dev_loss:.4f} | dev_macro_f1: {dev_f1:.4f}{marker}")

        if improved:
            best_macro_f1 = dev_f1
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience}).")
                break

    # Test with best checkpoint
    print(f"\n── Test Evaluation ──────────────────────")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    _, test_f1, test_preds, test_labels = eval_epoch(model, test_loader, criterion, device)
    print(f"Test Macro F1: {test_f1:.4f}")

    return {
        "best_epoch": best_epoch,
        "best_dev_macro_f1": float(best_macro_f1),
        "test_macro_f1": float(test_f1),
        "test_preds": test_preds,
        "test_labels": test_labels,
    }



# ────────────────────────────────────────────────────────────────────────────────
# SpERT Training (with extra forward arguments)
# ────────────────────────────────────────────────────────────────────────────────


def train_spert_epoch(model, loader, optimizer, scheduler, criterion, device, grad_accum_steps=1):
    """
    Train SpERT for one epoch (handles e1_word_len, e2_word_len forward args).
    """
    model.train()
    total_loss = 0

    for step, batch in enumerate(loader):
        optimizer.zero_grad()
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            e1_pos=batch["e1_pos"].to(device),
            e2_pos=batch["e2_pos"].to(device),
            e1_word_len=batch["e1_word_len"].to(device),
            e2_word_len=batch["e2_word_len"].to(device),
        )
        loss = criterion(logits, batch["label"].to(device))
        loss = loss / grad_accum_steps
        loss.backward()
        
        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * grad_accum_steps

    return total_loss / len(loader)


def eval_spert_epoch(model, loader, criterion, device):
    """
    Evaluate SpERT (handles e1_word_len, e2_word_len forward args).
    """
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                e1_pos=batch["e1_pos"].to(device),
                e2_pos=batch["e2_pos"].to(device),
                e1_word_len=batch["e1_word_len"].to(device),
                e2_word_len=batch["e2_word_len"].to(device),
            )
            loss = criterion(logits, batch["label"].to(device))
            total_loss += loss.item()
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch["label"].tolist())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    avg_loss = total_loss / len(loader)

    return avg_loss, macro_f1, all_preds, all_labels


def train_spert_model(
    model,
    train_loader,
    dev_loader,
    test_loader,
    device,
    num_epochs,
    learning_rate,
    weight_decay,
    warmup_ratio,
    patience,
    min_delta,
    criterion,
    model_save_path,
    optimizer=None,
    grad_accum_steps=1,
):
    """
    Train SpERT with early stopping, returning final metrics.
    Identical to train_model() but calls SpERT-specific train/eval functions.
    """
    if optimizer is None:
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_ratio * total_steps),
        num_training_steps=total_steps
    )

    best_macro_f1 = -1.0
    best_epoch = 0
    no_improve = 0

    print("\n── Training (SpERT) ──────────────────────")
    for epoch in range(1, num_epochs + 1):
        train_loss = train_spert_epoch(model, train_loader, optimizer, scheduler, criterion, device, grad_accum_steps)
        dev_loss, dev_f1, _, _ = eval_spert_epoch(model, dev_loader, criterion, device)

        improved = dev_f1 > (best_macro_f1 + min_delta)
        marker = " ← best" if improved else ""
        print(f"Epoch {epoch:02d} | train_loss: {train_loss:.4f} | dev_loss: {dev_loss:.4f} | dev_macro_f1: {dev_f1:.4f}{marker}")

        if improved:
            best_macro_f1 = dev_f1
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience}).")
                break

    # Test with best checkpoint
    print(f"\n── Test Evaluation ──────────────────────")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    _, test_f1, test_preds, test_labels = eval_spert_epoch(model, test_loader, criterion, device)
    print(f"Test Macro F1: {test_f1:.4f}")

    return {
        "best_epoch": best_epoch,
        "best_dev_macro_f1": float(best_macro_f1),
        "test_macro_f1": float(test_f1),
        "test_preds": test_preds,
        "test_labels": test_labels,
    }
