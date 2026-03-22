# Concepts & Architecture Explanation

Reference for understanding the design decisions in this codebase.
For experimental results and analysis outputs: see [EXPERIMENTS.md](EXPERIMENTS.md).

---

## Overview

This project has two components built on top of each other:

1. **Relation Extraction (RE)** — given a sentence with two pre-marked entities, classify the semantic relation between them. Trained on SciERC.
2. **Knowledge Graph Pipeline** — given raw arXiv abstracts, detect entities (NER), classify relations between all entity pairs (RE), and build a queryable knowledge graph with six analysis modes.

---

## Part 1 — Relation Extraction

### The Task

**Input:**
```
[E1] convolutional network [/E1] is used for [E2] image segmentation [/E2]
```
**Output:** `USED-FOR`

The RE model does not find entities — they are pre-marked. It only classifies the relation.

### Relation Types (SciERC)

| Label | Example |
|---|---|
| USED-FOR | "BERT is used for NER" |
| CONJUNCTION | "parsing and tagging" |
| EVALUATE-FOR | "F1 score for NER" |
| HYPONYM-OF | "LSTM is a type of RNN" |
| PART-OF | "attention is part of the transformer" |
| FEATURE-OF | "hidden size of BERT" |
| COMPARE | "BERT vs RoBERTa" |

Class distribution is heavily skewed — USED-FOR makes up 52% of training data.
PART-OF and FEATURE-OF are the hardest classes (F1 ~0.64–0.65).

---

### Entity Markers

Special tokens `[E1]`, `[/E1]`, `[E2]`, `[/E2]` are inserted around entities before
tokenization. This tells the model exactly which spans to focus on.

Raw text in data: `[[ convolutional network ]] is used for << image segmentation >>`
After preprocessing: `[E1] convolutional network [/E1] is used for [E2] image segmentation [/E2]`

These are added as additional special tokens to the tokenizer vocabulary so the model
treats them as atomic units rather than splitting them into subwords.

---

### Pooling Strategies

A transformer outputs one vector per token. For classification, we need a single
fixed-size vector. The pooling strategy decides which token(s) to use.

| Strategy | What it uses | Intuition |
|---|---|---|
| `e1e2_concat` | [E1] and [E2] start token vectors, concatenated | Directly encodes both entities — **best for RE** |
| `e1_only` | [E1] start token vector | Partial entity info |
| `cls_only` | [CLS] sentence summary vector | Ignores entity positions |
| `mean_pool` | Average of all token vectors | Dilutes entity signal with context |

`e1e2_concat` wins because relation extraction is fundamentally about two entities.
Using both entity representations directly captures more signal than any single-vector
approach. Empirically, it outperformed all other strategies by 1–4 F1 points across
all experiments.

---

### Classifier Head

All models use the same two-layer MLP on top of the pooled representation:

```
pooled_repr → Linear(input_dim, 768) → ReLU → Dropout → Linear(768, 7) → logits
```

Input dim is `768 * 2 = 1536` for `e1e2_concat`, `768` for all others.
For large models (RoBERTa-large, BERT-large), replace 768 with 1024.

---

### Loss Functions

| Loss | Behaviour | Finding |
|---|---|---|
| `ce_uniform` | Penalizes all classes equally | **Best — counterintuitively** |
| `weighted_ce` | Higher penalty for rare classes | Slightly hurts `e1e2_concat` |
| `focal` | Down-weights easy examples | Actively hurts — removes useful USED-FOR signal |
| `label_smooth` | Soft targets (0.1 smoothing) | Neutral |

`ce_uniform` wins despite class imbalance because USED-FOR dominates training data but
is also the easiest class to learn. Suppressing its gradient (via weighting or focal
loss) removes information rather than redirecting it.

---

### Word-to-Subword Alignment

The `metadata` field in the dataset provides entity boundaries as word-level indices.
BERT uses subword tokenization, so "classifications" might become `["class", "##ification", "##s"]`.

`word_span_to_token_span()` in `re_dataset.py` converts word-level indices to subword
token indices using the fast tokenizer's `word_ids()` output. Required for SpERT (span
widths) and PL-Marker (marker injection at true boundaries).

---

### RE-specific Architectures

**SpERT** adds span width embeddings and a context vector (mean of tokens between
`[E1]` and `[E2]`). Adds complexity without gain — SpERT F1: 0.8179 vs baseline 0.8243.

**PL-Marker** injects `[M]`/`[/M]` tokens at true word-level span boundaries from
`metadata`. Underperformed (F1: 0.7786) — requires more training data than ~3200 samples.

**PURE-Lite** performs two forward passes: Pass 1 extracts entity representations and
projects them to pseudo-type vectors. Pass 2 adds these to token embeddings and
re-encodes. Has a known dtype mismatch bug. Not evaluated.

---

### Optimizers

**Standard (default):** `AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)`

**Separate LR (`--separate-lr`):** encoder at `lr`, classifier head at `lr * 5`.
The classifier is randomly initialized and tolerates a higher LR.

**LLRD (`--llrd`):** per-layer decaying LR. Top transformer layer gets `lr`, each layer
below multiplied by 0.9. Produces the lowest variance of all optimizer variants (std 0.0041).

---

### Early Stopping

Training stops when dev macro F1 does not improve by more than `min_delta` for
`patience` consecutive epochs. Best checkpoint saved and used for test evaluation.

`--epochs 100 --patience 10` is the recommended setting: SciBERT converges around
epoch 15–20. The 100-epoch ceiling is never reached in practice.

---

### RE Key Findings

1. **Domain pretraining beats model size.** SciBERT (domain-adapted, 768-dim) outperforms BERT-large and RoBERTa-large (general, 1024-dim) on SciERC.
2. **Architectural complexity does not help on small data.** SpERT and PL-Marker both underperform. ~3200 training samples is insufficient for the additional inductive biases these models introduce.
3. **`e1e2_concat + ce_uniform` is a strong, stable baseline.** No training trick or architecture change improved it meaningfully across 13 experiment configurations.
4. **Minority class performance is a data problem, not a model problem.** PART-OF and FEATURE-OF remain weak (F1 ~0.64–0.65) regardless of loss, architecture, or optimizer.

---

## Part 2 — Knowledge Graph Pipeline

### NER — Named Entity Recognition

`ner_model.py` implements a SciBERT token classifier trained on SciERC NER annotations.
It produces BIO tags over input words:

```
O, B-TASK, I-TASK, B-METHOD, I-METHOD, B-DATASET, I-DATASET
```

**Entity type mapping from SciERC's 6 types to 3:**

| SciERC type | Maps to | Reason |
|---|---|---|
| Task | TASK | Keep |
| Method | METHOD | Keep |
| Material | DATASET | Material includes datasets |
| Metric | skip | Less useful for KG |
| OtherScientificTerm | skip | Too noisy |
| Generic | skip | Pronouns/placeholders only |

**Nested span resolution:** SciERC contains nested spans. Inner span is kept, outer
discarded. Inner spans are more specific and produce more reusable KG nodes.

**Subword alignment:** only the first subword of each word contributes to loss and
predictions. Continuation subwords receive label `-100` (ignored by cross-entropy).

**NER results:** test token F1 ~0.796 ± 0.003 (3 seeds). Weakest class: B-DATASET
(0.623) — the Material→Dataset mapping is imperfect since Material in SciERC includes
non-dataset resources like WordNet and Wikipedia.

---

### Inference Pipeline (`inference.py`)

End-to-end: raw text → structured triples.

1. Sentence splitting via `nltk.sent_tokenize`
2. NER on each sentence → entity spans with types
3. All ordered pairs `(a, b)` and `(b, a)` enumerated per sentence, including same-type pairs (needed for CONJUNCTION and COMPARE)
4. Each pair formatted with `[E1]`/`[/E1]` and `[E2]`/`[/E2]` markers
5. Batched RE inference with softmax confidence scores
6. Predictions below threshold (default 0.5) discarded — the RE model has no NULL class so thresholding is the only noise filter
7. Entity normalization: lowercase, strip punctuation, remove leading determiners, expand common acronyms via ACRONYM_MAP, strip parenthetical abbreviations via regex

Output per triple: `subject`, `subject_type`, `relation`, `object`, `object_type`,
`confidence`, `source_sentence`, `paper_id`.

---

### Knowledge Graph Construction (`build_kg.py`)

- **Graph type:** `nx.MultiDiGraph` — supports multiple relation types between the same node pair
- **Nodes:** unique normalized entity strings, attributes: `type`, `count`
- **Edges:** directed subject→object, attributes: `relation`, `weight` (frequency), `confidence_mean`, `papers`
- **Deduplication:** same subject→object→relation triples merged by incrementing weight; different relations between same pair kept as separate edges
- **Self-loops filtered:** subject == object pairs discarded

**Visualization:** pyvis interactive HTML. Top nodes by degree, edges filtered by
minimum weight. Node color encodes type (blue=TASK, orange=METHOD, green=DATASET).
Edge color encodes relation type.

---

### Analyses (`analyze_kg.py`)

Six analysis modes accessible via `--analysis` flag:

| Mode | What it does |
|---|---|
| `gap` | Entity pairs that co-occur in same papers but have no extracted relation |
| `trend` | Compare two triple sets — growing/declining entities, new/disappeared entities, relation shifts |
| `method-task` | Given a query entity, show all methods/tasks/datasets it connects to |
| `taxonomy` | Build HYPONYM-OF tree from a root entity |
| `dataset-discovery` | Rank datasets by paper count with their associated tasks |
| `cross-domain` | Compare graphs across multiple domains with normalized metrics |

All outputs saved to `json/`. See [EXPERIMENTS.md](EXPERIMENTS.md#kg-pipeline--analyses) for results.

---

### Entity Normalization Design

Normalization is critical for KG quality — without it, "LLMs", "llm", and
"large language models (LLMs)" become three separate nodes. The normalize function
applies in order:

1. Lowercase and strip whitespace
2. Strip leading/trailing punctuation
3. Remove leading determiners (the, a, an)
4. Strip trailing parenthetical abbreviations via regex: `"method (abbrev)"` → `"method"`
5. Look up in ACRONYM_MAP for explicit mappings (40+ entries covering LLM variants,
   RAG variants, model families, common abbreviations)

Generic noise entities (model, data, text, human, etc.) are filtered entirely from
the triple stream before graph construction.

---

### Key Findings from Full Dataset (19,123 papers)

1. **Universal pattern:** `large language models → USED-FOR → reasoning` is the top USED-FOR pair in cs.CL, cs.LG, and cs.CV simultaneously. Reasoning is the cross-domain obsession of late 2025 AI research.
2. **Temporal shift:** In cs.CL, reinforcement learning replaced in-context learning as the dominant paradigm applied to LLMs between 2024 and 2025. COMPARE relations grew +34.8% normalized.
3. **Domain identity:** LLMs are 12× more central in cs.CL (norm degree 2.805) than cs.CV (0.238). In cs.CV, vision-language models (0.411) are the dominant hub.
4. **Relation structure is universal:** USED-FOR accounts for 50–55% of all relations in all three domains, CONJUNCTION for 17–20%. The way scientific communities express knowledge is structurally consistent across subfields.
5. **Abstract-level limitation:** Gap analysis shows that evaluation relations (benchmark-model pairs) are frequently implied rather than stated, suggesting full-text extraction would significantly increase EVALUATE-FOR coverage.
