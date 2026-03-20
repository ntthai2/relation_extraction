
# ────────────────────────────────────────────────────────────────────────────────
# Complete run_single_seed Implementation (for replace in run_exp.py)
# ────────────────────────────────────────────────────────────────────────────────

def run_single_seed(
    seed,
    model_type="scibert",
    model_variant="e1e2_concat",
    loss_variant="weighted_ce",
    frozen_bert=False,
    max_len=256,
    batch_size=32,
    num_epochs=10,
    lr=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    patience=3,
    min_delta=1e-4,
    output_dir="runs",
    max_samples_per_split=0,
    separate_lr=False,
    llrd=False,
    grad_accum_steps=1,
    augment=False,
    undersample_conjunction=False,
    undersample_target=250,
):
    """
    Run a single experiment with given hyperparams.
    Supports all new model types, optimizers, and data augmentation.
    
    Returns:
        dict: Experiment results
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Seed: {seed} | Model: {model_type} | Type: {model_variant} | Loss: {loss_variant}")
    print(f"LR: {lr} | SepLR: {separate_lr} | LLRD: {llrd} | GradAccum: {grad_accum_steps}")
    print(f"{'='*60}")
    print(f"Device: {device}")

    # Determine model name for tokenizer
    model_name_for_tokenizer = MODEL_NAME
    if model_type == "deberta":
        from config import DEBERTA_MODEL_NAME
        model_name_for_tokenizer = DEBERTA_MODEL_NAME
    elif model_type == "roberta_large":
        from config import ROBERTA_LARGE_MODEL_NAME
        model_name_for_tokenizer = ROBERTA_LARGE_MODEL_NAME
    elif model_type == "bert_large":
        from config import BERT_LARGE_MODEL_NAME
        model_name_for_tokenizer = BERT_LARGE_MODEL_NAME

    # Setup tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(model_name_for_tokenizer, use_fast=True)
    special_tokens = {"additional_special_tokens": ENTITY_MARKERS}
    
    # Add levitated markers for PL-Marker
    if model_type == "plmarker":
        special_tokens = {"additional_special_tokens": ENTITY_MARKERS + LEVITATED_MARKERS}
    
    tokenizer.add_special_tokens(special_tokens)

    # Get dataset class from registry
    dataset_class = DATASET_REGISTRY.get(model_type, SciERCDataset)

    # Setup datasets
    train_set = dataset_class(
        TRAIN_FILE,
        tokenizer,
        max_len=max_len,
        max_samples=max_samples_per_split,
        use_entity_markers=USE_ENTITY_MARKERS,
    )
    
    # Apply data augmentation (training only)
    if augment:
        train_set.samples = augment_symmetric(train_set.samples)
    
    # Apply undersampling (training only)
    if undersample_conjunction:
        train_set.samples = undersample_label(
            train_set.samples,
            "CONJUNCTION",
            undersample_target,
            seed=seed
        )

    dev_set = dataset_class(
        DEV_FILE,
        tokenizer,
        max_len=max_len,
        max_samples=max_samples_per_split,
        use_entity_markers=USE_ENTITY_MARKERS,
    )
    
    test_set = dataset_class(
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

    # Get model class from registry
    model_class = MODEL_REGISTRY.get(model_type, SciBERTRelationClassifier)

    # Select model
    pooling_strategy = MODEL_VARIANTS[model_variant]["pooling"]
    
    if model_type in ["scibert", "deberta", "roberta_large", "bert_large", "pure_lite", "plmarker"]:
        # Standard models that support pooling strategies
        if frozen_bert and model_type == "scibert":
            model = FrozenSciBERTRelationClassifier(
                num_labels=NUM_LABELS,
                dropout=0.1,
                pooling=pooling_strategy
            )
        else:
            # Note: frozen_bert only applies to scibert for now
            model = model_class(
                num_labels=NUM_LABELS,
                dropout=0.1,
                pooling=pooling_strategy if hasattr(model_class, '__init__') and 'pooling' in model_class.__init__.__code__.co_varnames else None,
            ) if model_type != "spert" else model_class(num_labels=NUM_LABELS, dropout=0.1)
    elif model_type == "spert":
        model = model_class(num_labels=NUM_LABELS, dropout=0.1)
    else:
        model = model_class(num_labels=NUM_LABELS, dropout=0.1)

    # Resize token embeddings if special tokens were added
    model.bert.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # Get loss function
    criterion = get_criterion(loss_variant, device)

    # Create optimizer
    optimizer = None
    if separate_lr:
        optimizer = make_two_group_optimizer(model, lr, weight_decay, head_lr_multiplier=10.0)
    elif llrd:
        optimizer = make_llrd_optimizer(model, lr, weight_decay, decay_factor=0.9)
    # else: None (train_model will create default AdamW)

    # Prepare output directory
    run_subdir = os.path.join(
        output_dir,
        f"{model_type}_{model_variant}_{loss_variant}_frozen{int(frozen_bert)}_"
        f"seplr{int(separate_lr)}_llrd{int(llrd)}_aug{int(augment)}_seed{seed}"
    )
    os.makedirs(run_subdir, exist_ok=True)
    model_save_path = os.path.join(run_subdir, "best_model.pt")

    # Train model
    if model_type == "spert":
        from train_core import train_spert_model
        train_results = train_spert_model(
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
            optimizer=optimizer,
            grad_accum_steps=grad_accum_steps,
        )
    else:
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
            optimizer=optimizer,
            grad_accum_steps=grad_accum_steps,
        )

    # Save artifacts
    save_test_artifacts(run_subdir, train_results["test_labels"], train_results["test_preds"])

    return {
        "seed": seed,
        "model_type": model_type,
        "model_variant": model_variant,
        "loss_variant": loss_variant,
        "frozen_bert": frozen_bert,
        "separate_lr": separate_lr,
        "llrd": llrd,
        "augment": augment,
        "best_epoch": train_results["best_epoch"],
        "best_dev_macro_f1": train_results["best_dev_macro_f1"],
        "test_macro_f1": train_results["test_macro_f1"],
        "run_dir": run_subdir,
    }
