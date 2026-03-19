# Relation Extraction

A modular pipeline for fine-tuning SciBERT on the SciERC dataset with multiple baselines, ablations, and evaluation frameworks for rigorous experimental analysis.

## Quick Start

```bash
# Test setup (1 epoch, 128 samples)
python main.py --smoke-test

# Run main model (single seed)
python main.py --seeds 42

# For detailed experiment guides, see EXPERIMENTS.md
```

For all options: `python main.py --help`

---

## Project Structure

### Core Modules (1,010 lines of clean, modular code)

- **config.py** (107 lines) - Centralized configuration & constants
- **dataset.py** (79 lines) - Dataset loading with entity marker handling
- **model.py** (151 lines) - Model architectures (4 pooling strategies)
- **train_core.py** (109 lines) - Training loop with early stopping
- **eval.py** (104 lines) - Metrics, reporting, artifact generation
- **run_exp.py** (265 lines) - Experiment orchestration & grid search
- **main.py** (196 lines) - CLI entry point

### Data

- `scierc/train.txt` - 3,219 training samples
- `scierc/dev.txt` - 455 dev samples
- `scierc/test.txt` - 974 test samples

### Reference

- `train.py` - Original monolithic version (kept for reference)
- `best_model.pt` - Pre-trained checkpoint
- `train.log` - Training log from initial run

---

## Model Variants

Choose which pooling strategy to use:

| Variant | Description | When to use |
|---------|-------------|------------|
| `e1e2_concat` | Concatenate [E1] & [E2] representations | **Main model** ✓ |
| `cls_only` | Use [CLS] token only | Baseline |
| `mean_pool` | Average all tokens | Ablation |
| `e1_only` | Use [E1] token only | Ablation |

---

## Loss Functions

| Loss | Description | When to use |
|------|-------------|------------|
| `weighted_ce` | Class-weighted cross-entropy | **Main model** ✓ (handles imbalance) |
| `ce_uniform` | Standard cross-entropy | Ablation (test if weighting matters) |

---

## Training Options

| Flag | Purpose | Example |
|------|---------|---------|
| `--seeds` | Random seeds (comma-separated) | `--seeds 13,21,42,87,100` |
| `--model-variants` | Pooling strategies | `--model-variants e1e2_concat,cls_only` |
| `--loss-variants` | Loss functions | `--loss-variants weighted_ce,ce_uniform` |
| `--frozen-bert` | Freeze encoder (baseline) | `--frozen-bert` |
| `--epochs` | Training epochs | `--epochs 15` |
| `--lr` | Learning rate | `--lr 3e-5` |
| `--batch-size` | Batch size | `--batch-size 16` |
| `--patience` | Early stopping patience | `--patience 5` |
| `--smoke-test` | Debug mode (1 epoch, 128 samples) | `--smoke-test` |



See [EXPERIMENTS.md](EXPERIMENTS.md) for detailed experiment guides, output structure, and analysis instructions.

---

## Baseline Results (Initial Training)

- **Dev macro F1:** 0.8898
- **Test macro F1:** 0.8313
- **Best epoch:** 8 / 10

Class-wise performance:
| Relation | F1 | Precision | Recall |
|----------|-----|-----------|--------|
| USED-FOR | 0.94 | 0.95 | 0.94 |
| CONJUNCTION | 0.95 | 0.92 | 0.98 |
| EVALUATE-FOR | 0.87 | 0.87 | 0.88 |
| HYPONYM-OF | 0.90 | 0.88 | 0.91 |
| PART-OF | 0.64 | 0.70 | 0.59 |
| FEATURE-OF | 0.65 | 0.63 | 0.68 |
| COMPARE | 0.86 | 0.89 | 0.84 |

**Key insight:** Minority relations (PART-OF, FEATURE-OF) are weak. Test if ablations improve them.

## Latest Comprehensive Ablation (Mar 18, 2026)

From the 40-run suite (`4 pooling × 2 losses × 5 seeds`):

- Best config: `e1e2_concat + ce_uniform`
- Test macro-F1: **0.8243 ± 0.0052**
- Runner-up: `e1e2_concat + weighted_ce` with `0.8157 ± 0.0155`

Details and full ranking are documented in [EXPERIMENTS.md](EXPERIMENTS.md).



---

## Key Files to Know

- `config.py` - Change defaults here (labels, model, hyperparams)
- `main.py` - See what CLI options are available here
- `run_exp.py` - Orchestration logic, where experiments are coordinated
- `runs/*/experiment_summary.json` - Your results (check this!)

## Extending the Code

All details in [EXPERIMENTS.md](EXPERIMENTS.md#extending-the-code) including:
- Adding new pooling strategies
- Adding new loss functions
- Adding new metrics
- Configuration changes
