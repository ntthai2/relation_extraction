# Scientific Knowledge Graph Pipeline

An end-to-end pipeline for extracting scientific entities and relations from NLP papers
and constructing a knowledge graph. Built on SciBERT fine-tuned on SciERC.

---

## What This Does

```
arXiv abstracts
    → NER (SciBERT token classifier)     detect Task / Method / Dataset entities
    → RE  (SciBERT relation classifier)  classify relations between entity pairs
    → Knowledge Graph (NetworkX + pyvis) interactive visualization and analysis
```

---

## Quick Start

### Run full pipeline on arXiv papers

```bash
# 1. Fetch abstracts
python fetch_arxiv.py --target 500 --output abstracts.json

# 2. Extract triples
python -c "
from inference import ScienceIEPipeline, extract_from_file
pipeline = ScienceIEPipeline()
extract_from_file(pipeline, 'abstracts.json', 'triples.json')
"

# 3. Build knowledge graph
python build_kg.py --triples triples.json
```

### Run on a single abstract

```python
from inference import ScienceIEPipeline
pipeline = ScienceIEPipeline()
triples = pipeline.extract("Your abstract text here...")
for t in triples:
    print(f"{t['subject']} ({t['subject_type']}) --{t['relation']}--> {t['object']} ({t['object_type']})")
```

### Train or retrain models

```bash
# NER
python ner_train.py --epochs 100 --lr 2e-5 --patience 10 --seeds 13,42,87

# RE (smoke test)
python re_main.py --smoke-test

# RE (full best config)
python re_main.py \
  --model-type scibert \
  --model-variants e1e2_concat \
  --loss-variants ce_uniform \
  --seeds 13,21,42,87,100 \
  --epochs 100 \
  --patience 10
```

---

## Project Structure

### Pipeline files

| File | Responsibility |
|---|---|
| `fetch_arxiv.py` | Fetch cs.CL abstracts from arXiv API |
| `inference.py` | End-to-end NER → RE → triples on raw text |
| `build_kg.py` | Build NetworkX KG, analyze, export pyvis HTML |

### NER files

| File | Responsibility |
|---|---|
| `ner_dataset.py` | BIO tagging dataset from SciERC JSON |
| `ner_model.py` | SciBERTNER token classification model |
| `ner_train.py` | NER training script |

### RE files

| File | Responsibility |
|---|---|
| `config.py` | All constants, label mappings, hyperparameters |
| `re_dataset.py` | SciERC loading, word→subword alignment, augmentation |
| `re_model.py` | RE model architectures + FocalLoss |
| `re_train_core.py` | Training loop, early stopping, gradient accumulation |
| `re_eval.py` | Metrics, reporting, artifact generation |
| `re_run_exp.py` | Experiment orchestration, registries, optimizers |
| `re_main.py` | RE CLI entry point |

### Data

```
scierc/
├── train.json / train.txt   — 3,219 training samples
├── dev.json   / dev.txt     — 455 dev samples
└── test.json  / test.txt    — 974 test samples
```

- `.json` — full SciERC format with NER, relations, coreference, sentences
- `.txt` — relation-only format: `{"text": "... [[ e1 ]] ... << e2 >> ...", "label": "...", "metadata": [e1_start, e1_end, e2_start, e2_end]}`

### Checkpoints

| File | Description |
|---|---|
| `best_re_model.pt` | Best RE model (SciBERT + e1e2_concat + ce_uniform, seed 42) |
| `best_ner_model.pt` | Best NER model (SciBERT token classifier, seed 42) |

---

## Model Performance

### RE — Relation Extraction

**Best config:** `scibert + e1e2_concat + ce_uniform`  
**Test macro F1: 0.8243 ± 0.0052** (5 seeds)

| Relation | F1 | Precision | Recall |
|---|---|---|---|
| CONJUNCTION | 0.95 | 0.92 | 0.98 |
| USED-FOR | 0.94 | 0.95 | 0.94 |
| HYPONYM-OF | 0.90 | 0.88 | 0.91 |
| EVALUATE-FOR | 0.87 | 0.87 | 0.88 |
| COMPARE | 0.86 | 0.89 | 0.84 |
| FEATURE-OF | 0.65 | 0.63 | 0.68 |
| PART-OF | 0.64 | 0.70 | 0.59 |

### NER — Named Entity Recognition

**Best config:** SciBERT token classifier, 3 entity types  
**Test token F1: 0.7956 ± 0.003** (3 seeds)

| Entity type | F1 | Precision | Recall |
|---|---|---|---|
| METHOD | 0.780 | 0.779 | 0.780 |
| DATASET | 0.716 | 0.675 | 0.726 |
| TASK | 0.698 | 0.681 | 0.717 |

---

## Knowledge Graph (500 cs.CL papers, 2024–2025)

| Stat | Value |
|---|---|
| Nodes | 6,290 |
| Unique edges | 22,917 |
| Papers | 500 |
| Triples extracted | 23,556 |

**Node type distribution:** METHOD 3,161 / TASK 1,951 / DATASET 1,178

**Relation distribution:**

| Relation | Count |
|---|---|
| USED-FOR | 11,873 |
| CONJUNCTION | 4,907 |
| EVALUATE-FOR | 2,391 |
| COMPARE | 1,434 |
| HYPONYM-OF | 1,240 |
| FEATURE-OF | 1,096 |
| PART-OF | 527 |

**Top entities by degree:** large language models (1,440), reasoning (159), inference (125),
reinforcement learning (125), generation (107)

**Outputs:** `triples.json`, `kg_stats.json`, `kg_visualization.html`

---

## RE Training Options

| Flag | Type | Default | Purpose |
|---|---|---|---|
| `--model-type` | str | `scibert` | Encoder / architecture |
| `--model-variants` | str | `e1e2_concat` | Pooling strategy |
| `--loss-variants` | str | `weighted_ce` | Loss function |
| `--seeds` | str | `42` | Comma-separated seeds |
| `--epochs` | int | `10` | Max training epochs |
| `--patience` | int | `3` | Early stopping patience |
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

For full RE experiment results: see [EXPERIMENTS.md](EXPERIMENTS.md)  
For architecture explanations: see [EXPLANATION.md](EXPLANATION.md)
