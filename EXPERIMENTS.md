# Experiments Guide

Detailed guides for running experiments, interpreting results, and extending the codebase.

## Experiment Examples

### 1. Compare Pooling Strategies

Which token representation works best?

```bash
python main.py \
  --model-variants e1e2_concat,cls_only,mean_pool,e1_only \
  --seeds 42 \
  --run-name pooling_ablation
```

**Expected result:** `e1e2_concat` should outperform others because it leverages entity information.

**Analysis:**
- Best performance → E1/E2 pooling captures entity context
- CLS performance → Shows if entity markers matter
- Mean pool performance → Tests if averaging across sequence helps

### 2. Test Class Weighting

Does class weighting help with imbalanced data?

```bash
python main.py \
  --loss-variants weighted_ce,ce_uniform \
  --seeds 42 \
  --run-name loss_ablation
```

**Expected result:** `weighted_ce` should outperform `ce_uniform` on minority relations.

**Analysis:**
- Check `test_per_class_metrics.csv` for PART-OF and FEATURE-OF F1
- Class weighting should help minority classes more than majority
- Justifies using weighted loss in main model

### 3. Frozen Encoder Baseline

Do we need to fine-tune BERT, or just train the classifier?

```bash
python main.py \
  --frozen-bert \
  --seeds 42 \
  --run-name frozen_encoder_baseline
```

**Expected result:** Fine-tuning (default) should beat frozen encoder significantly.

**Analysis:**
- If gap is small → Maybe full fine-tuning isn't needed
- If gap is large → Justifies the computational cost of fine-tuning

### 4. Multi-Seed Main Result (Recommended for Thesis)

Run 5 seeds for robust statistics:

```bash
python main.py \
  --seeds 13,21,42,87,100 \
  --epochs 10 \
  --patience 3 \
  --run-name main_result_multiseed
```

**Time:** ~2-3 hours on GPU  
**Output:** Results with mean ± std across 5 runs

**Include in thesis:**
- Mean test F1 ± std (shows reproducibility)
- Best/worst seed variation (shows robustness)
- Confidence interval implies ~95% CI using ±2×std

### 5. Comprehensive Ablation Suite

All model/loss combinations × multiple seeds:

```bash
python main.py \
  --model-variants e1e2_concat,cls_only,mean_pool,e1_only \
  --loss-variants weighted_ce,ce_uniform \
  --seeds 13,21,42,87,100 \
  --run-name comprehensive_ablation
```

**Time:** ~10-15 hours on GPU  
**Output:** 40 experiments with full analysis

**Generates:**
- 4 pooling × 2 losses × 5 seeds = 40 runs
- Structured output: `runs/comprehensive_ablation/`
- `experiment_summary.json` aggregates all results

### Comprehensive Ablation Results (Mar 18, 2026)

Source: `runs/comprehensive_ablation/experiment_summary.json`

| Rank | Model Variant | Loss Variant | Test Macro-F1 (mean ± std) | Dev Macro-F1 (mean ± std) |
|------|---------------|--------------|------------------------------|----------------------------|
| 1 | e1e2_concat | ce_uniform | **0.8243 ± 0.0052** | 0.8882 ± 0.0086 |
| 2 | e1e2_concat | weighted_ce | 0.8157 ± 0.0155 | 0.8929 ± 0.0069 |
| 3 | e1_only | weighted_ce | 0.8132 ± 0.0051 | 0.8714 ± 0.0034 |
| 4 | e1_only | ce_uniform | 0.8078 ± 0.0114 | 0.8732 ± 0.0152 |
| 5 | cls_only | weighted_ce | 0.7918 ± 0.0062 | 0.8437 ± 0.0137 |
| 6 | mean_pool | weighted_ce | 0.7879 ± 0.0041 | 0.8524 ± 0.0099 |
| 7 | cls_only | ce_uniform | 0.7836 ± 0.0095 | 0.8504 ± 0.0114 |
| 8 | mean_pool | ce_uniform | 0.7801 ± 0.0220 | 0.8527 ± 0.0094 |

**Main conclusions:**
- Best overall setup: `e1e2_concat + ce_uniform`.
- `e1e2_concat` is strongest under both losses.
- `weighted_ce` helps 3/4 pooling variants (`cls_only`, `mean_pool`, `e1_only`), but hurts `e1e2_concat`.
- `mean_pool + ce_uniform` is unstable across seeds (highest std).

**Recommended default after this ablation:**
- `--model-variants e1e2_concat`
- `--loss-variants ce_uniform`
- `--seeds 13,21,42,87,100`

### What to Try Beyond SciBERT

All suggestions below can reuse the same training/eval pipeline with minimal code changes (swap model checkpoint and hidden size if needed).

| Model | Why try it | Priority |
|------|------------|----------|
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | Strong biomedical/scientific terminology coverage; often strong transfer in scientific IE tasks | High |
| `allenai/specter2_base` | Scientific-document-pretrained encoder from AI2; may improve scientific relation semantics | High |
| `microsoft/deberta-v3-base` | Strong general encoder baseline; tests whether domain-specific pretraining is actually necessary | High |
| `roberta-base` | Robust general baseline with different tokenization dynamics | Medium |
| `sentence-transformers/all-mpnet-base-v2` | Strong sentence-level semantics; useful if relation signal is distributed across context | Medium |

**Practical experiment order:**
1. Keep best current setup fixed (`e1e2_concat + ce_uniform`).
2. Swap encoder only and run 3 seeds first (`13,42,87`) for screening.
3. Promote top 2 encoders to full 5-seed evaluation.

### Data Improvements Beyond Existing SciERC Splits

Even if you keep SciERC as the core benchmark, you can improve data quality and effective supervision:

1. Hard-example mining
- After each run, collect high-confidence mistakes from confusion pairs (for example, `PART-OF` vs `FEATURE-OF`).
- Upweight or oversample these hard subsets in later epochs.

2. Controlled augmentation (entity-safe)
- Use augmentation that preserves entity spans and labels: synonym substitution outside entity markers, light paraphrasing, or back-translation.
- Keep marker indices valid and reject noisy augmentations with low semantic similarity.

3. Distant supervision from scientific abstracts
- Build weakly labeled pairs using pattern rules (for example, "X is part of Y", "X feature of Y").
- Pre-train classifier head on weak labels, then fine-tune on gold SciERC.

4. Curriculum by label difficulty
- Start training with high-support/easier relations, then gradually mix in low-support labels.
- Helps stabilize minority-class learning.

5. Threshold calibration and class-wise bias correction
- Calibrate decision behavior on dev set (temperature scaling or class-wise bias terms).
- Useful when optimizing macro-F1 over imbalanced labels.

6. Cross-dataset transfer (if thesis scope allows)
- Pre-fine-tune relation encoder on another scientific RE dataset, then adapt to SciERC.
- Evaluate with same SciERC test protocol for fair comparison.

---

## Training Options Reference

| Flag | Type | Default | Example |
|------|------|---------|---------|
| `--seeds` | str | "42" | `--seeds 13,21,42,87,100` |
| `--model-variants` | str | "e1e2_concat" | `--model-variants e1e2_concat,cls_only` |
| `--loss-variants` | str | "weighted_ce" | `--loss-variants weighted_ce,ce_uniform` |
| `--frozen-bert` | flag | False | `--frozen-bert` |
| `--epochs` | int | 10 | `--epochs 15` |
| `--batch-size` | int | 32 | `--batch-size 16` |
| `--lr` | float | 2e-5 | `--lr 3e-5` |
| `--weight-decay` | float | 0.01 | `--weight-decay 0.001` |
| `--patience` | int | 3 | `--patience 5` |
| `--min-delta` | float | 1e-4 | `--min-delta 1e-3` |
| `--max-len` | int | 256 | `--max-len 512` |
| `--warmup-ratio` | float | 0.1 | `--warmup-ratio 0.2` |
| `--max-samples` | int | 0 (full) | `--max-samples 500` |
| `--smoke-test` | flag | False | `--smoke-test` |
| `--run-name` | str | "scibert_scierc" | `--run-name my_experiment` |
| `--output-dir` | str | "runs" | `--output-dir /tmp/runs` |

---

## Output Structure

Results are organized by experiment:

```
runs/
└── <experiment_name>/
    ├── experiment_summary.json              ← Main results (mean ± std)
    ├── <variant>_<loss>_frozen0_seed13/
    │   ├── best_model.pt                   ← Checkpoint
    │   ├── test_classification_report.json ← Full sklearn report
    │   ├── test_per_class_metrics.csv      ← For spreadsheets
    │   └── test_confusion_matrix.csv       ← Error analysis
    ├── <variant>_<loss>_frozen0_seed21/
    │   └── ...
    └── ...
```

### experiment_summary.json

Aggregates all runs in an experiment:

```json
{
  "e1e2_concat_weighted_ce_frozen0": {
    "config": {
      "model_variant": "e1e2_concat",
      "loss_variant": "weighted_ce",
      "frozen_bert": false
    },
    "runs": [
      {
        "seed": 42,
        "best_epoch": 8,
        "best_dev_macro_f1": 0.8898,
        "test_macro_f1": 0.8313,
        "run_dir": "runs/exp/e1e2_concat_weighted_ce_frozen0_seed42"
      }
    ],
    "aggregate": {
      "n_runs": 5,
      "dev_macro_f1_mean": 0.8840,
      "dev_macro_f1_std": 0.0034,
      "test_macro_f1_mean": 0.8313,
      "test_macro_f1_std": 0.0052
    }
  }
}
```

### test_per_class_metrics.csv

Per-relation evaluation:

```csv
label,precision,recall,f1,support
USED-FOR,0.945227,0.938066,0.941615,533
CONJUNCTION,0.923664,0.983606,0.952826,123
EVALUATE-FOR,0.869565,0.879120,0.874320,91
HYPONYM-OF,0.884058,0.910448,0.897196,67
PART-OF,0.698113,0.587302,0.637884,63
FEATURE-OF,0.625000,0.677966,0.650364,59
COMPARE,0.888889,0.842105,0.864865,38
```

Use this to identify weak relations!

### test_confusion_matrix.csv

Raw confusion values for error analysis:

```csv
true\pred,USED-FOR,CONJUNCTION,EVALUATE-FOR,HYPONYM-OF,PART-OF,FEATURE-OF,COMPARE
USED-FOR,500,5,3,10,5,5,5
CONJUNCTION,2,121,0,0,0,0,0
...
```

Diagonal = correct predictions. Off-diagonal = confusions.

---

## Understanding Results

### After Running an Experiment

1. **Check `experiment_summary.json`**
   ```bash
   cat runs/<exp_name>/experiment_summary.json | python -m json.tool
   ```
   Look at `aggregate` section for mean ± std performance.

2. **Analyze per-class performance**
   ```bash
   cat runs/<exp_name>/*/test_per_class_metrics.csv
   ```
   Which relations are hard? Is the model balanced?

3. **Study confusions**
   ```bash
   cat runs/<exp_name>/*/test_confusion_matrix.csv
   ```
   What does the model confuse? 

### Example Analysis

```
Baseline (CLS-only): Test F1 = 0.75
Proposed (E1E2):     Test F1 = 0.83

Improvement: +8 points on PART-OF (0.45 → 0.64)
             +5 points on FEATURE-OF (0.60 → 0.65)
             +2 points on HYPONYM-OF (0.88 → 0.90)

Conclusion: Entity-specific pooling helps minority relations most.
```

### Interpreting Multi-Seed Statistics

```
Test F1: 0.8313 ± 0.0052

Interpretation:
- Point estimate: 0.8313
- 95% CI: [0.8258, 0.8368] (approximately ±2×std)
- Std error: 0.0052/√5 = 0.00233
- Very stable across seeds → Robust method
```

---

## Configuration & Hyperparameters

### In config.py

Centralized settings:

```python
# Model & Dataset
LABELS = ["USED-FOR", "CONJUNCTION", ...]
MODEL_NAME = "allenai/scibert_scivocab_uncased"
MAX_LEN = 256
HIDDEN_SIZE = 768

# Training Defaults
BATCH_SIZE = 32
EPOCHS = 10
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

# Early Stopping
PATIENCE = 3
MIN_DELTA = 1e-4
```

### Override via CLI

```bash
# Longer training with more patience
python main.py --epochs 15 --patience 5

# Aggressive learning
python main.py --lr 5e-5 --batch-size 8

# Conservative training
python main.py --epochs 5 --patience 1 --lr 1e-5
```

### Tuning Guide

| Problem | Tuning |
|---------|--------|
| Model plateaus early | ↑ patience, ↓ min_delta |
| Loss explodes | ↓ lr, ↑ batch_size |
| CUDA OOM | ↓ batch_size, ↓ max_len |
| Weak minority classes | ↑ epochs, keep weighted_ce |
| High variance across seeds | ↑ batch_size, ↑ warmup_ratio |

---

## Extending the Code

### Add a New Pooling Strategy

**Step 1:** Edit `model.py` → `SciBERTRelationClassifier.forward()` method

```python
elif self.pooling == "my_strategy":
    # Your pooling logic here
    my_repr = seq_out[torch.arange(B), my_pos]  # [B, 768]
    my_repr = self.dropout_layer(my_repr)
    logits = self.classifier(my_repr)
```

**Step 2:** Add to `config.py` → `MODEL_VARIANTS`

```python
MODEL_VARIANTS = {
    ...
    "my_strategy": {
        "name": "My Custom Pooling",
        "pooling": "my_strategy",
        "description": "What this does"
    },
}
```

**Step 3:** Run

```bash
python main.py --model-variants my_strategy --seeds 42
```

### Add a New Loss Function

**Step 1:** Edit `run_exp.py` → `get_criterion()` function

```python
elif loss_variant == "my_loss":
    return MyLossFunction()
```

**Step 2:** Add to `config.py` → `LOSS_VARIANTS`

```python
LOSS_VARIANTS = {
    ...
    "my_loss": {
        "name": "My Custom Loss",
        "description": "Why this is useful"
    },
}
```

**Step 3:** Run

```bash
python main.py --loss-variants my_loss --seeds 42
```

### Add a New Evaluation Metric

**Step 1:** Edit `eval.py` → `save_test_artifacts()` function

```python
# Compute your metric
my_metric_value = compute_my_metric(test_labels, test_preds)

# Save to JSON or CSV
# e.g., in classification_report
report_dict["my_metric"] = my_metric_value
```

**Step 2:** Run normally, metrics auto-included in output

```bash
python main.py --seeds 42
```

---

## Troubleshooting

### ModuleNotFoundError

**Problem:**
```
ModuleNotFoundError: No module named 'sklearn'
```

**Solution:**
```bash
conda activate ntt_det
```

### CUDA Out of Memory

**Problem:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```bash
# Reduce batch size
python main.py --batch-size 16

# Reduce sequence length
python main.py --max-len 128

# Test with small data first
python main.py --smoke-test
```

### Training Not Improving

**Problem:**
```
Epoch 10 | train_loss: 1.94 | dev_loss: 1.95 | dev_macro_f1: 0.10
```

**Solutions:**
```bash
# More epochs + patience
python main.py --epochs 20 --patience 5

# Higher learning rate
python main.py --lr 5e-5

# More warmup
python main.py --warmup-ratio 0.3
```

### Reproducibility Issues

**Problem:**
```
Results differ between runs with same seed
```

**Solutions:**
```bash
# Verify CUDA sync
export CUDA_LAUNCH_BLOCKING=1

# Check all seeds show same trend
python main.py --seeds 42,42,42
```

### Slow Training

**Problem:**
```
1 epoch takes 15 minutes
```

**Solutions:**
```bash
# Check GPU usage
nvidia-smi

# Single seed first
python main.py --seeds 42

# Check if DataLoader has bottleneck
python main.py --num-workers 4
```

---

## Bachelor Thesis Recommendation

### Phase 1: Justify Design Choices (2-3 hours)

Run ablations to show your choices are justified:

```bash
python main.py \
  --model-variants e1e2_concat,cls_only,mean_pool \
  --loss-variants weighted_ce,ce_uniform \
  --seeds 42 \
  --run-name thesis_phase1_ablation
```

**Include in thesis:**
1. Pooling comparison table (e1e2 wins → justify concatenation)
2. Loss comparison table (weighted wins → justify class weighting)
3. Frozen encoder baseline (fine-tuning wins → justify computation cost)

### Phase 2: Robust Main Result (10-15 hours)

Multi-seed experiment showing reproducibility:

```bash
python main.py \
  --seeds 13,21,42,87,100 \
  --epochs 10 \
  --patience 3 \
  --run-name thesis_phase2_main_multiseed
```

**Include in thesis:**
1. Results table: Test F1 mean ± std
2. Best/worst seed variation
3. Per-class breakdown (shows balanced performance)
4. Confusion matrix analysis (where model struggles)

### Phase 3: Analysis (Writing time)

Explain results using generated artifacts:

```
# View key metrics
cat runs/thesis_phase2_main_multiseed/experiment_summary.json

# Per-class performance
head -20 runs/thesis_phase2_main_multiseed/*/test_per_class_metrics.csv

# Confusion patterns
cat runs/thesis_phase2_main_multiseed/*/test_confusion_matrix.csv
```

### Total Time: 12-18 hours GPU time

Produces publication-ready results! 📊
