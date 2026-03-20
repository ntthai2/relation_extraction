# Relation Extraction

A modular pipeline for fine-tuning transformer models on the SciERC dataset for scientific
relation classification. Supports multiple encoder backbones, RE-specific architectures,
pooling strategies, loss functions, and training tricks.

## Quick Start

```bash
# Test setup (1 epoch, 128 samples)
python main.py --smoke-test

# Run main model (single seed)
python main.py --model-type scibert --seeds 42

# Full 5-seed run with best known config
python main.py \
  --model-type scibert \
  --model-variants e1e2_concat \
  --loss-variants ce_uniform \
  --seeds 13,21,42,87,100 \
  --epochs 100 \
  --patience 10
```

For all options: `python main.py --help`  
For experiment guides and results: see [EXPERIMENTS.md](EXPERIMENTS.md)  
For architecture explanations: see [EXPLANATION.md](EXPLANATION.md)

---

## Project Structure

| File | Responsibility | Lines |
|---|---|---|
| `config.py` | All constants and hyperparameters | ~150 |
| `dataset.py` | SciERC loading, word→subword alignment, augmentation | ~280 |
| `model.py` | All model architectures + FocalLoss | ~600 |
| `train_core.py` | Training loop, early stopping, gradient accumulation | ~240 |
| `eval.py` | Metrics, reporting, artifact generation | ~104 |
| `run_exp.py` | Experiment orchestration, registries, optimizers | ~380 |
| `main.py` | CLI entry point only | ~220 |

### Data

- `scierc/train.txt` — 3,219 training samples
- `scierc/dev.txt` — 455 dev samples
- `scierc/test.txt` — 974 test samples

Format: one JSON per line — `{"text": "... [[ e1 ]] ... << e2 >> ...", "label": "...", "metadata": [e1_start, e1_end, e2_start, e2_end]}`

---

## Supported Models

### Encoder Backbones (`--model-type`)

| Model type | Checkpoint | Hidden size | Notes |
|---|---|---|---|
| `scibert` | `allenai/scibert_scivocab_uncased` | 768 | **Best overall — use this** |
| `deberta` | `microsoft/deberta-v3-base` | 768 | Unstable on small data |
| `roberta_large` | `roberta-large` | 1024 | Slightly below SciBERT |
| `bert_large` | `bert-large-uncased` | 1024 | Slightly below SciBERT |

### RE-specific Architectures (`--model-type`)

| Model type | Description | Notes |
|---|---|---|
| `spert` | Span width embeddings + between-entity context | Marginally below SciBERT |
| `plmarker` | Levitated markers at true span boundaries | Underperforms on small data |
| `pure_lite` | Two-pass encoding with continuous pseudo-types | Experimental |

### Pooling Strategies (`--model-variants`)

| Variant | Description | Recommendation |
|---|---|---|
| `e1e2_concat` | Concatenate [E1] and [E2] representations | **Best — always use this** |
| `cls_only` | [CLS] token only | Baseline |
| `mean_pool` | Average all tokens | Ablation |
| `e1_only` | [E1] token only | Ablation |

---

## Loss Functions (`--loss-variants`)

| Loss | Description | Recommendation |
|---|---|---|
| `ce_uniform` | Standard cross-entropy | **Best — use this** |
| `weighted_ce` | Class-weighted cross-entropy | Slightly below ce_uniform |
| `focal` | Focal loss (down-weights easy examples) | Hurts performance |
| `label_smooth` | Cross-entropy with label smoothing 0.1 | Neutral |

---

## Training Options

| Flag | Type | Default | Purpose |
|---|---|---|---|
| `--model-type` | str | `scibert` | Encoder / architecture |
| `--model-variants` | str | `e1e2_concat` | Pooling strategy |
| `--loss-variants` | str | `weighted_ce` | Loss function |
| `--seeds` | str | `42` | Comma-separated seeds |
| `--epochs` | int | `10` | Max training epochs |
| `--patience` | int | `3` | Early stopping patience |
| `--min-delta` | float | `1e-4` | Min F1 improvement to continue |
| `--lr` | float | `2e-5` | Learning rate |
| `--batch-size` | int | `32` | Batch size |
| `--separate-lr` | flag | False | Separate LR for encoder/head |
| `--llrd` | flag | False | Layer-wise LR decay |
| `--grad-accum-steps` | int | `1` | Gradient accumulation steps |
| `--augment` | flag | False | Symmetric relation augmentation |
| `--undersample-conjunction` | flag | False | Reduce CONJUNCTION dominance |
| `--frozen-bert` | flag | False | Freeze encoder weights |
| `--smoke-test` | flag | False | 1 epoch, 128 samples debug run |
| `--run-name` | str | `scibert_scierc` | Output directory name |

---

## Best Known Results

**Best config: `scibert + e1e2_concat + ce_uniform`**  
Test macro F1: **0.8243 ± 0.0052** (5 seeds: 13, 21, 42, 87, 100)

Per-class breakdown (best seed):

| Relation | F1 | Precision | Recall |
|---|---|---|---|
| USED-FOR | 0.94 | 0.95 | 0.94 |
| CONJUNCTION | 0.95 | 0.92 | 0.98 |
| EVALUATE-FOR | 0.87 | 0.87 | 0.88 |
| HYPONYM-OF | 0.90 | 0.88 | 0.91 |
| COMPARE | 0.86 | 0.89 | 0.84 |
| FEATURE-OF | 0.65 | 0.63 | 0.68 |
| PART-OF | 0.64 | 0.70 | 0.59 |

Full experiment results across all phases in [EXPERIMENTS.md](EXPERIMENTS.md).

---

## Output Structure

```
runs/
└── <run_name>/
    ├── experiment_summary.json                        ← Aggregated results (mean ± std)
    └── <model>_<variant>_<loss>_frozen0_seed<N>/
        ├── best_model.pt
        ├── test_classification_report.json
        ├── test_per_class_metrics.csv
        ├── test_confusion_matrix.csv
        └── test_confusion_matrix.txt
```
