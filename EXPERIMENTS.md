# Experiments Guide

Guides for running experiments, interpreting results, and extending the codebase.

---

## Completed Experiments & Results

### Phase 0 — Initial Training (Single Seed)

```bash
python main.py --seeds 42
```

- Dev macro F1: 0.8898 — Test macro F1: **0.8313** — Best epoch: 8/10

---

### Phase 1 — Pooling & Loss Ablation (Mar 18, 2026)

40 runs: 4 pooling × 2 losses × 5 seeds

```bash
python main.py \
  --model-variants e1e2_concat,cls_only,mean_pool,e1_only \
  --loss-variants weighted_ce,ce_uniform \
  --seeds 13,21,42,87,100 \
  --run-name comprehensive_ablation
```

| Rank | Pooling | Loss | Test F1 | Std | Dev F1 | Std |
|---|---|---|---|---|---|---|
| 1 | e1e2_concat | ce_uniform | **0.8243** | 0.0052 | 0.8882 | 0.0086 |
| 2 | e1e2_concat | weighted_ce | 0.8157 | 0.0155 | 0.8929 | 0.0069 |
| 3 | e1_only | weighted_ce | 0.8132 | 0.0051 | 0.8714 | 0.0034 |
| 4 | e1_only | ce_uniform | 0.8078 | 0.0114 | 0.8732 | 0.0152 |
| 5 | cls_only | weighted_ce | 0.7918 | 0.0062 | 0.8437 | 0.0137 |
| 6 | mean_pool | weighted_ce | 0.7879 | 0.0041 | 0.8524 | 0.0099 |
| 7 | cls_only | ce_uniform | 0.7836 | 0.0095 | 0.8504 | 0.0114 |
| 8 | mean_pool | ce_uniform | 0.7801 | 0.0220 | 0.8527 | 0.0094 |

**Conclusions:**
- `e1e2_concat` is the best pooling strategy under both loss functions
- `ce_uniform` beats `weighted_ce` for `e1e2_concat` — class weighting hurts the best config
- `mean_pool + ce_uniform` has the highest variance (0.0220) — most unstable

**Fixed from here on:** `e1e2_concat + ce_uniform + seeds 13,21,42,87,100`

---

### Phase 2 — Loss Variants on SciBERT (Mar 2026)

```bash
python main.py \
  --model-type scibert \
  --model-variants e1e2_concat \
  --loss-variants focal,label_smooth \
  --seeds 13,21,42,87,100 \
  --epochs 100 --patience 10 \
  --run-name phase1_loss
```

| Loss | Test F1 | Std |
|---|---|---|
| **ce_uniform (baseline)** | **0.8243** | 0.0052 |
| label_smooth | 0.8188 | 0.0116 |
| weighted_ce | 0.8157 | 0.0155 |
| focal | 0.8083 | 0.0091 |

**Conclusion:** `ce_uniform` remains best. Focal loss actively hurts — down-weighting easy USED-FOR examples removes genuinely useful gradient signal.

---

### Phase 3 — Backbone Comparison (Mar 2026)

All runs: `e1e2_concat + ce_uniform + epochs 100 + patience 10`

```bash
python main.py --model-type deberta --model-variants e1e2_concat --loss-variants ce_uniform \
  --seeds 13,21,42,87,100 --epochs 100 --patience 10 --run-name phase2_deberta

python main.py --model-type roberta_large --model-variants e1e2_concat --loss-variants ce_uniform \
  --seeds 13,21,42,87,100 --epochs 100 --patience 10 --run-name phase2_roberta_large

python main.py --model-type bert_large --model-variants e1e2_concat --loss-variants ce_uniform \
  --seeds 13,21,42,87,100 --epochs 100 --patience 10 --run-name phase2_bert_large
```

| Model | Test F1 | Std | vs SciBERT |
|---|---|---|---|
| **SciBERT** | **0.8243** | 0.0052 | — |
| BERT-large | 0.8159 | 0.0101 | -0.0084 |
| RoBERTa-large | 0.8135 | 0.0123 | -0.0108 |
| DeBERTa-v3 | 0.7399 | 0.0696 | -0.0844 |

**DeBERTa per-seed breakdown:**
```
seed 1: 0.6728   seed 2: 0.7162   seed 3: 0.8155   seed 4: 0.8125   seed 5: 0.6822
```

**Conclusions:**
- SciBERT wins — domain pretraining beats model size on small data (~3200 samples)
- BERT-large and RoBERTa-large match SciBERT within noise but don't beat it
- DeBERTa shows high variance (two seeds converged well, three failed) — LR sensitivity on small datasets

---

### Phase 4 — Training Tricks on SciBERT (Mar 2026)

All runs: `scibert + e1e2_concat + ce_uniform + epochs 100 + patience 10`

```bash
python main.py --model-type scibert --model-variants e1e2_concat --loss-variants ce_uniform \
  --separate-lr --seeds 13,21,42,87,100 --epochs 100 --patience 10 --run-name phase3_separate_lr

python main.py --model-type scibert --model-variants e1e2_concat --loss-variants ce_uniform \
  --llrd --seeds 13,21,42,87,100 --epochs 100 --patience 10 --run-name phase3_llrd

python main.py --model-type scibert --model-variants e1e2_concat --loss-variants ce_uniform \
  --augment --seeds 13,21,42,87,100 --epochs 100 --patience 10 --run-name phase3_augment

python main.py --model-type scibert --model-variants e1e2_concat --loss-variants ce_uniform \
  --undersample-conjunction --seeds 13,21,42,87,100 --epochs 100 --patience 10 --run-name phase3_undersample
```

| Config | Test F1 | Std | vs Baseline |
|---|---|---|---|
| **Baseline** | **0.8243** | 0.0052 | — |
| + augment | 0.8241 | 0.0098 | -0.0002 |
| + separate_lr | 0.8215 | 0.0065 | -0.0028 |
| + llrd | 0.8210 | 0.0041 | -0.0033 |
| + undersample | 0.8127 | 0.0062 | -0.0116 |

**Conclusions:**
- No trick improved the baseline meaningfully
- Undersample CONJUNCTION actively hurts — removing majority-class signal is counterproductive
- LLRD produces the lowest variance (0.0041) — use it if stability matters more than mean F1

---

### Phase 5 — RE-specific Architectures (Mar 2026)

All runs: `e1e2_concat + ce_uniform + epochs 100 + patience 10`

```bash
python main.py --model-type spert --model-variants e1e2_concat --loss-variants ce_uniform \
  --seeds 13,21,42,87,100 --epochs 100 --patience 10 --run-name phase4_spert

python main.py --model-type plmarker --model-variants e1e2_concat --loss-variants ce_uniform \
  --seeds 13,21,42,87,100 --epochs 100 --patience 10 --run-name phase4_plmarker
```

| Model | Test F1 | Std | vs Baseline |
|---|---|---|---|
| **SciBERT baseline** | **0.8243** | 0.0052 | — |
| SpERT | 0.8179 | 0.0053 | -0.0064 |
| PLMarker | 0.7786 | 0.0112 | -0.0457 |

**PURE-Lite:** skipped — implementation bug (dtype mismatch in embedding injection) and pattern of results made it unlikely to reverse the trend.

**Conclusions:**
- SpERT is close to baseline — span width and context don't add much when entities are already marked
- PLMarker underperforms significantly — levitated marker injection needs more data to learn new positional semantics
- Architectural complexity doesn't help on ~3200 training samples

---

## Overall Summary

| Model | Test F1 | Std |
|---|---|---|
| **SciBERT + ce_uniform** | **0.8243** | 0.0052 |
| SciBERT + label_smooth | 0.8188 | 0.0116 |
| SciBERT + weighted_ce | 0.8157 | 0.0155 |
| SciBERT + separate_lr | 0.8215 | 0.0065 |
| SciBERT + llrd | 0.8210 | 0.0041 |
| SciBERT + augment | 0.8241 | 0.0098 |
| SciBERT + undersample | 0.8127 | 0.0062 |
| SciBERT + focal | 0.8083 | 0.0091 |
| BERT-large | 0.8159 | 0.0101 |
| RoBERTa-large | 0.8135 | 0.0123 |
| DeBERTa-v3 | 0.7399 | 0.0696 |
| SpERT | 0.8179 | 0.0053 |
| PLMarker | 0.7786 | 0.0112 |

**Final recommendation:** `SciBERT + e1e2_concat + ce_uniform`, trained with `--epochs 100 --patience 10`.

---

## Running New Experiments

### Recommended settings for any new run

```bash
python main.py \
  --model-type scibert \
  --model-variants e1e2_concat \
  --loss-variants ce_uniform \
  --seeds 13,21,42,87,100 \
  --epochs 100 \
  --patience 10 \
  --run-name <descriptive_name>
```

### CUDA OOM with large models

Halve batch size and double accumulation steps — effective batch size stays the same:

```bash
--batch-size 16 --grad-accum-steps 2
```

### Screening a new backbone (3 seeds first)

```bash
python main.py \
  --model-type <new_model> \
  --model-variants e1e2_concat \
  --loss-variants ce_uniform \
  --seeds 13,42,87 \
  --epochs 100 --patience 10 \
  --run-name screen_<new_model>
```

Promote to 5 seeds only if 3-seed mean F1 > 0.82.

---

## Interpreting Results

### Check summary

```bash
cat runs/<exp_name>/experiment_summary.json | python -m json.tool
```

### Check per-class performance

```bash
grep -r "PART-OF\|FEATURE-OF" runs/<exp_name>/*/test_per_class_metrics.csv
```

### Multi-seed statistics

```
Test F1: 0.8243 ± 0.0052
 → 95% CI ≈ [0.8192, 0.8295]  (±2×std)
 → Std error = 0.0052 / √5 = 0.0023
```

### Tuning guide

| Problem | Fix |
|---|---|
| Still improving at epoch limit | Increase `--epochs`, keep `--patience 10` |
| High variance across seeds | Increase `--batch-size`, increase `--warmup-ratio` |
| New backbone not converging | Try `--llrd` or `--separate-lr` |
| CUDA OOM | `--batch-size 16 --grad-accum-steps 2` |
| Weak minority classes | Check per-class CSV — model-level fixes unlikely to help on this data size |

---

## Training Options Reference

| Flag | Type | Default | Purpose |
|---|---|---|---|
| `--model-type` | str | `scibert` | Model architecture |
| `--model-variants` | str | `e1e2_concat` | Pooling strategy |
| `--loss-variants` | str | `weighted_ce` | Loss function |
| `--seeds` | str | `42` | Comma-separated random seeds |
| `--epochs` | int | `10` | Max training epochs |
| `--patience` | int | `3` | Early stopping patience |
| `--min-delta` | float | `1e-4` | Min F1 improvement threshold |
| `--lr` | float | `2e-5` | Learning rate |
| `--batch-size` | int | `32` | Training batch size |
| `--weight-decay` | float | `0.01` | AdamW weight decay |
| `--warmup-ratio` | float | `0.1` | Warmup steps fraction |
| `--max-len` | int | `256` | Max tokenizer sequence length |
| `--separate-lr` | flag | False | Separate LR for encoder vs head |
| `--llrd` | flag | False | Layer-wise LR decay |
| `--grad-accum-steps` | int | `1` | Gradient accumulation steps |
| `--augment` | flag | False | Symmetric relation augmentation |
| `--undersample-conjunction` | flag | False | Undersample CONJUNCTION class |
| `--undersample-target` | int | `250` | CONJUNCTION target count |
| `--frozen-bert` | flag | False | Freeze encoder |
| `--smoke-test` | flag | False | Debug run (1 epoch, 128 samples) |
| `--run-name` | str | `scibert_scierc` | Output folder name |
| `--output-dir` | str | `runs` | Root output directory |
| `--log-file` | str | `` | Optional log file path |
| `--max-samples` | int | `0` | Cap samples per split (0 = full) |

---

## Extending the Code

### Add a new pooling strategy

1. Add branch in `SciBERTRelationClassifier.forward()` in `model.py`
2. Update `classifier_input_dim` logic in `__init__` if dimension changes
3. Add entry to `MODEL_VARIANTS` in `config.py`

### Add a new loss function

1. Add branch in `get_criterion()` in `run_exp.py`
2. Add entry to `LOSS_VARIANTS` in `config.py`

### Add a new backbone

1. Add model name constant to `config.py`
2. Add new class to `model.py` using `self.bert = AutoModel.from_pretrained(...)` — keep `.bert` attribute name
3. Add to `MODEL_REGISTRY` and `DATASET_REGISTRY` in `run_exp.py`
4. Add to `get_model_name()` in `run_exp.py`
5. Add `--model-type` choice in `main.py`

### Add a new metric

1. Compute in `save_test_artifacts()` in `eval.py`
2. Add to the JSON/CSV output — auto-included in all future runs

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | `conda activate ntt_det` |
| CUDA OOM | `--batch-size 16 --grad-accum-steps 2` or `--max-len 128` |
| Dev F1 stuck at 0.09–0.15 for 10 epochs | Model needs more epochs — use `--epochs 100 --patience 10` |
| High variance across seeds | Normal for DeBERTa on small data — try `--llrd` |
| Results differ with same seed | `export CUDA_LAUNCH_BLOCKING=1` |
| `No such file or directory` on glob | Use double quotes or no quotes: `cat runs/exp/*/file.json` |
