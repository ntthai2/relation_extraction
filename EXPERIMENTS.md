# Experiments Guide

Guides for running experiments, interpreting results, and extending the codebase.

---

## Completed Experiments & Results

### Phase 0 — Initial Training (Single Seed)

```bash
python re_main.py --seeds 42
```

- Dev macro F1: 0.8898 — Test macro F1: **0.8313** — Best epoch: 8/10

---

### Phase 1 — Pooling & Loss Ablation (Mar 18, 2026)

40 runs: 4 pooling × 2 losses × 5 seeds

```bash
python re_main.py \
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
python re_main.py \
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
python re_main.py --model-type deberta --model-variants e1e2_concat --loss-variants ce_uniform \
  --seeds 13,21,42,87,100 --epochs 100 --patience 10 --run-name phase2_deberta

python re_main.py --model-type roberta_large --model-variants e1e2_concat --loss-variants ce_uniform \
  --seeds 13,21,42,87,100 --epochs 100 --patience 10 --run-name phase2_roberta_large

python re_main.py --model-type bert_large --model-variants e1e2_concat --loss-variants ce_uniform \
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
python re_main.py --model-type scibert --model-variants e1e2_concat --loss-variants ce_uniform \
  --separate-lr --seeds 13,21,42,87,100 --epochs 100 --patience 10 --run-name phase3_separate_lr

python re_main.py --model-type scibert --model-variants e1e2_concat --loss-variants ce_uniform \
  --llrd --seeds 13,21,42,87,100 --epochs 100 --patience 10 --run-name phase3_llrd

python re_main.py --model-type scibert --model-variants e1e2_concat --loss-variants ce_uniform \
  --augment --seeds 13,21,42,87,100 --epochs 100 --patience 10 --run-name phase3_augment

python re_main.py --model-type scibert --model-variants e1e2_concat --loss-variants ce_uniform \
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
python re_main.py --model-type spert --model-variants e1e2_concat --loss-variants ce_uniform \
  --seeds 13,21,42,87,100 --epochs 100 --patience 10 --run-name phase4_spert

python re_main.py --model-type plmarker --model-variants e1e2_concat --loss-variants ce_uniform \
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

## RE Overall Summary

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

## KG Pipeline — NER Training

```bash
python ner_train.py \
  --epochs 100 --lr 2e-5 --patience 10 \
  --seeds 13,42,87 \
  --output-dir runs/ner
```

| Seed | Best Dev F1 | Test F1 |
|---|---|---|
| 13 | 0.7799 | 0.7989 |
| 42 | 0.7985 | 0.7956 |
| 87 | 0.7881 | 0.7922 |
| **Mean** | **0.7888** | **0.7956 ± 0.003** |

Per-class test F1 (seed 42, best checkpoint):

| Label | F1 | Precision | Recall |
|---|---|---|---|
| B-METHOD | 0.780 | 0.779 | 0.780 |
| I-METHOD | 0.796 | 0.811 | 0.782 |
| I-DATASET | 0.730 | 0.714 | 0.747 |
| I-TASK | 0.700 | 0.700 | 0.699 |
| B-TASK | 0.702 | 0.683 | 0.722 |
| B-DATASET | 0.645 | 0.652 | 0.639 |

Best checkpoint: `runs/ner/seed_42/best_ner_model.pt` → copied to `best_ner_model.pt`

---

## KG Pipeline — Data Collection

Abstracts fetched from arXiv using `fetch_arxiv.py` with date-range queries targeting
November–December of each year for temporal consistency.

```bash
python fetch_arxiv.py --category cs.CL --date-from 20241101 --date-to 20241231 --target 3000 --output json/abstracts_cscl_2024.json
python fetch_arxiv.py --category cs.CL --date-from 20251101 --date-to 20251231 --target 3000 --output json/abstracts_cscl_2025.json
python fetch_arxiv.py --category cs.LG --date-from 20251101 --date-to 20251231 --target 3000 --output json/abstracts_cslg.json
python fetch_arxiv.py --category cs.CV --date-from 20251101 --date-to 20251231 --target 3000 --output json/abstracts_cscv.json
```

| Source | Period | Papers fetched | Triples extracted |
|---|---|---|---|
| cs.CL | Nov–Dec 2024 | 2,972 (exhausted) | 119,819 |
| cs.CL | Nov–Dec 2025 | 2,911 (exhausted) | 134,680 |
| cs.LG | Nov–Dec 2025 | 7,078 (exhausted) | 279,090 |
| cs.CV | Nov–Dec 2025 | 6,162 (exhausted) | 285,793 |
| **Total** | | **19,123** | **819,382** |

Note: "exhausted" means the API returned all available papers for that window — the
target of 3000 was not the binding constraint for cs.LG and cs.CV.

Triples extracted via:
```bash
python -c "
from inference import ScienceIEPipeline, extract_from_file
pipeline = ScienceIEPipeline()
extract_from_file(pipeline, 'json/abstracts_cscl_2024.json', 'json/triples_cscl_2024.json')
extract_from_file(pipeline, 'json/abstracts_cscl_2025.json', 'json/triples_cscl_2025.json')
extract_from_file(pipeline, 'json/abstracts_cslg.json', 'json/triples_cslg.json')
extract_from_file(pipeline, 'json/abstracts_cscv.json', 'json/triples_cscv.json')
"
```

---

## KG Pipeline — Analyses

### Graph Statistics

```bash
python build_kg.py --triples json/triples_cscl_2024.json --output-viz html/kg_cscl_2024.html --output-stats json/kg_stats_cscl_2024.json
python build_kg.py --triples json/triples_cscl_2025.json --output-viz html/kg_cscl_2025.html --output-stats json/kg_stats_cscl_2025.json
python build_kg.py --triples json/triples_cslg.json --output-viz html/kg_cslg.html --output-stats json/kg_stats_cslg.json
python build_kg.py --triples json/triples_cscv.json --output-viz html/kg_cscv.html --output-stats json/kg_stats_cscv.json
```

| Graph | Nodes | Unique edges | Node types (METHOD/TASK/DATASET) |
|---|---|---|---|
| cs.CL 2024 | 26,227 | 111,204 | 12,720 / 8,180 / 5,327 |
| cs.CL 2025 | 29,615 | 125,918 | 15,020 / 9,222 / 5,816 |
| cs.LG 2025 | 68,388 | 263,009 | 40,395 / 20,558 / 8,569 |
| cs.CV 2025 | 67,272 | 269,355 | 35,750 / 21,054 / 10,468 |

---

### Trend Analysis: cs.CL 2024 vs 2025

```bash
python analyze_kg.py --analysis trend \
  --triples-a json/triples_cscl_2024.json \
  --triples-b json/triples_cscl_2025.json \
  --label-a "cs.CL 2024" --label-b "cs.CL 2025" --top-n 30
```

**Relation distribution shift (normalized per paper):**

| Relation | 2024 | 2025 | Growth |
|---|---|---|---|
| USED-FOR | 20.42 | 22.55 | +10.4% |
| CONJUNCTION | 7.25 | 9.21 | +27.1% |
| COMPARE | 2.10 | 2.83 | +34.8% |
| EVALUATE-FOR | 4.26 | 4.79 | +12.4% |

All relations grew — the field produces structurally denser knowledge in 2025.
COMPARE grew fastest (+34.8%), reflecting proliferation of competing reasoning models
and RAG variants being benchmarked against each other.

**Top growing entities (2024→2025):**

| Entity | Degree 2024→2025 | Growth |
|---|---|---|
| gpt-5 | 2→261 | +13,180% |
| grpo | 6→147 | +2,393% |
| reasoning models | 6→144 | +2,342% |
| reasoning traces | 6→108 | +1,732% |
| rlvr (new in 2025) | — | degree 98 |

**New entities exclusive to 2025:** gemini-2.5-pro, gpt-4.1, qwen3-8b, aime25, deepseek-r1, large reasoning models

**Disappeared entities (in 2024, absent 2025):** codellama, llama-2-7b, o1-preview, gpt-4v — all superseded by newer models

**Key structural shift:** `in-context learning → USED-FOR → large language models` was the 3rd most frequent USED-FOR pair in 2024 (weight=30). It does not appear in the 2025 top 10. Reinforcement learning replaced it as the dominant paradigm.

---

### Taxonomy: LLM Family Tree

```bash
python analyze_kg.py --analysis taxonomy \
  --triples json/triples_cscl_2025.json \
  --root "large language models" --min-weight 2
```

```
large language models (incoming HYPONYM-OF weight: 339)
├── gpt-4          (weight=9)
├── gpt-4o         (weight=8)
├── llama          (weight=7)
├── gpt-4.1        (weight=5)
├── gemini         (weight=4)
├── gemini 2.5 pro (weight=4)
├── chatgpt        (weight=3)
├── claude         (weight=3)
├── claude 3.5 sonnet (weight=3)
├── gpt-5          (weight=3)
├── deepseek-r1    (weight=3)
├── gemma          (weight=2)
├── deepseek       (weight=2)
├── deepseek-v3    (weight=2)
└── gpt-5-mini     (weight=2)
```

Other taxonomy roots: `language models` (weight=47), `natural language processing tasks` (35),
`prompting strategies` (29), `transformer-based models` (25).

---

### Method-Task Coverage

```bash
python analyze_kg.py --analysis method-task \
  --triples json/triples_cscl_2025.json \
  --query "large language models" --top-n 20

python analyze_kg.py --analysis method-task \
  --triples json/triples_cscl_2025.json \
  --query "reasoning" --top-n 20
```

**Large language models** (1,327 unique papers — 46% of cs.CL 2025 corpus):

Top USED-FOR tasks: reasoning (weight=86, papers=68), generation (20), multi-step reasoning (20), code generation (17), retrieval (17), healthcare (14), mathematical reasoning (12)

Top related methods: gpt-4 (16), gpt-4o (13), llama (13), gpt-4.1 (10), gemini (8), deepseek-r1 (8), vision-language models (8)

**Named entity recognition** (13 unique papers — 0.4% of corpus):

The contrast between 1,327 and 13 papers quantifies the field's shift from traditional
structured prediction to LLM-centric research.

---

### Dataset Discovery

```bash
python analyze_kg.py --analysis dataset-discovery \
  --triples json/triples_cscl_2025.json --top-n 30
```

Top datasets by paper count in cs.CL 2025:

| Dataset | Papers | Primary use |
|---|---|---|
| english | 101 | Cross-lingual comparison baseline |
| low-resource languages | 52 | Transfer learning target |
| benchmarks (general) | 36 | Reasoning evaluation |
| social media | 29 | Sentiment, misinformation |
| gsm8k | 15 | Mathematical reasoning |
| arabic | 14 | Multilingual NLP |
| spanish | 14 | Multilingual NLP |
| french | 14 | Multilingual NLP |

English as top dataset (101 papers) reflects its dual role as the primary language
and comparison baseline. Arabic, Hindi, Bangla, Spanish, French all appearing confirms
multilingual NLP is a major active subfield.

---

### Gap Analysis

```bash
python analyze_kg.py --analysis gap \
  --triples json/triples_cscl_2025.json \
  --min-cooccurrence 5 --top-n 20
```

Top genuine gaps after normalization (entity pairs that co-occur but have no extracted relation):

| Entity A | Entity B | Co-occurrences | Interpretation |
|---|---|---|---|
| gsm8k | large language models | 11 | Evaluation relation implied, not stated |
| chinese | large language models | 10 | Capability discussed without formal relation |
| lora | low-rank adaptation | 6 | Same method, two names — residual normalization gap |

The prevalence of implicit relations (gaps 1–2) confirms that evaluation-relation
coverage would improve significantly with full-text extraction.

---

### Cross-Domain Comparison

```bash
python analyze_kg.py --analysis cross-domain \
  --triples-list json/triples_cscl_2025.json json/triples_cslg.json json/triples_cscv.json \
  --labels "cs.CL 2025" "cs.LG 2025" "cs.CV 2025" --top-n 20
```

**LLM centrality (normalized degree):**

| Entity | cs.CL | cs.LG | cs.CV |
|---|---|---|---|
| large language models | **2.805** | 0.737 | 0.238 |
| vision-language models | 0.145 | 0.077 | **0.411** |
| reinforcement learning | 0.229 | **0.230** | 0.130 |
| reasoning | **0.373** | 0.112 | 0.188 |

LLMs are ~12× more central in cs.CL than cs.CV. In cs.CV, vision-language models are
the dominant hub — invisible without cross-domain comparison.

**Domain-exclusive entities (top by degree):**

- cs.CL: Turkish, Thai, Polish, mt5, multilingual llms — language diversity
- cs.LG: deep reinforcement learning, dynamical systems, symbolic regression, q-learning — mathematical ML
- cs.CV: sam3, identity preservation, 3d data, camera pose estimation — spatial understanding

**Universal pattern:** `large language models → USED-FOR → reasoning` is the top USED-FOR
pair in all three domains. Reasoning is the cross-domain obsession of late 2025 AI research.

**Relation distributions are structurally consistent:**

| Relation | cs.CL | cs.LG | cs.CV |
|---|---|---|---|
| USED-FOR | 50.2% | 52.7% | 54.7% |
| CONJUNCTION | 20.5% | 18.8% | 17.5% |
| EVALUATE-FOR | 10.6% | 8.9% | 10.1% |

The way scientific communities express knowledge is structurally universal across subfields.

---

## Running New Experiments

### RE — recommended settings

```bash
python re_main.py \
  --model-type scibert \
  --model-variants e1e2_concat \
  --loss-variants ce_uniform \
  --seeds 13,21,42,87,100 \
  --epochs 100 \
  --patience 10 \
  --run-name <descriptive_name>
```

### CUDA OOM with large models

```bash
--batch-size 16 --grad-accum-steps 2
```

### Screening a new backbone (3 seeds first)

```bash
python re_main.py \
  --model-type <new_model> \
  --model-variants e1e2_concat \
  --loss-variants ce_uniform \
  --seeds 13,42,87 \
  --epochs 100 --patience 10 \
  --run-name screen_<new_model>
```

Promote to 5 seeds only if 3-seed mean F1 > 0.82.

### KG pipeline on a new domain

```bash
# Fetch
python fetch_arxiv.py --category <cat> --date-from <YYYYMMDD> --date-to <YYYYMMDD> --target 3000 --output json/abstracts_<name>.json

# Extract
python -c "
from inference import ScienceIEPipeline, extract_from_file
pipeline = ScienceIEPipeline()
extract_from_file(pipeline, 'json/abstracts_<name>.json', 'json/triples_<name>.json')
"

# Build and analyze
python build_kg.py --triples json/triples_<name>.json --output-viz html/kg_<name>.html --output-stats json/kg_stats_<name>.json
python analyze_kg.py --analysis method-task --triples json/triples_<name>.json --query "<entity>"
```

---

## Interpreting Results

### Check RE summary

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

1. Add branch in `SciBERTRelationClassifier.forward()` in `re_model.py`
2. Update `classifier_input_dim` logic in `__init__` if dimension changes
3. Add entry to `MODEL_VARIANTS` in `config.py`

### Add a new loss function

1. Add branch in `get_criterion()` in `re_run_exp.py`
2. Add entry to `LOSS_VARIANTS` in `config.py`

### Add a new backbone

1. Add model name constant to `config.py`
2. Add new class to `re_model.py` using `self.bert = AutoModel.from_pretrained(...)` — keep `.bert` attribute name
3. Add to `MODEL_REGISTRY` and `DATASET_REGISTRY` in `re_run_exp.py`
4. Add to `get_model_name()` in `re_run_exp.py`
5. Add `--model-type` choice in `re_main.py`

### Add a new metric

1. Compute in `save_test_artifacts()` in `re_eval.py`
2. Add to the JSON/CSV output — auto-included in all future runs

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | `conda activate ntt_det` |
| CUDA OOM | `--batch-size 16 --grad-accum-steps 2` or `--max-len 128` |
| Dev F1 stuck at 0.09–0.15 for 10 epochs | Use `--epochs 100 --patience 10` |
| High variance across seeds | Normal for DeBERTa on small data — try `--llrd` |
| Results differ with same seed | `export CUDA_LAUNCH_BLOCKING=1` |
| `No such file or directory` on glob | Use no quotes: `cat runs/exp/*/file.json` |
| arXiv API returns wrong date range | Use `--date-from`/`--date-to` flags, not `--start-year` |
| KG node count unexpectedly high | Check normalization — may need new entries in ACRONYM_MAP |
