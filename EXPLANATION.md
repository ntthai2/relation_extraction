# Concepts & Architecture Explanation

Reference for understanding the design decisions in this codebase. For experimental
results, see [EXPERIMENTS.md](EXPERIMENTS.md).

---

## The Task: Relation Extraction

Given a sentence with two marked scientific entities, classify the semantic relation
between them.

**Input:**
```
[E1] convolutional network [/E1] is used for [E2] image segmentation [/E2]
```

**Output:** `USED-FOR`

The model does not find entities — they are pre-marked. It only classifies the relation.

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

## Entity Markers

Special tokens `[E1]`, `[/E1]`, `[E2]`, `[/E2]` are inserted around entities before
tokenization. This tells the model exactly which spans to focus on.

Raw text in data: `[[ convolutional network ]] is used for << image segmentation >>`  
After preprocessing: `[E1] convolutional network [/E1] is used for [E2] image segmentation [/E2]`

These are added as additional special tokens to the tokenizer vocabulary so the model
treats them as atomic units rather than splitting them into subwords.

---

## Pooling Strategies

A transformer outputs one vector per token. For classification, we need a single
fixed-size vector. The pooling strategy decides which token(s) to use.

| Strategy | What it uses | Intuition |
|---|---|---|
| `e1e2_concat` | [E1] and [E2] start token vectors, concatenated | Directly encodes both entities — **best for RE** |
| `e1_only` | [E1] start token vector | Partial entity info |
| `cls_only` | [CLS] sentence summary vector | Ignores entity positions |
| `mean_pool` | Average of all token vectors | Dilutes entity signal with context |

**Why `e1e2_concat` wins:** relation extraction is fundamentally about two entities.
Using both entity representations directly captures more signal than any single-vector
approach. Empirically, `e1e2_concat` outperformed all other strategies by 1–4 F1 points
across all experiments (see [EXPERIMENTS.md](EXPERIMENTS.md)).

---

## Classifier Head

All models use the same two-layer MLP on top of the pooled representation:

```
pooled_repr → Linear(input_dim, 768) → ReLU → Dropout → Linear(768, 7) → logits
```

Input dim is `768 * 2 = 1536` for `e1e2_concat`, `768` for all others.
For large models (RoBERTa-large, BERT-large), replace 768 with 1024.

---

## Loss Functions

| Loss | Behaviour | Finding |
|---|---|---|
| `ce_uniform` | Penalizes all classes equally | **Best — counterintuitively** |
| `weighted_ce` | Higher penalty for rare classes | Slightly hurts `e1e2_concat` |
| `focal` | Down-weights easy examples | Actively hurts — removes useful USED-FOR signal |
| `label_smooth` | Soft targets (0.1 smoothing) | Neutral |

**Why `ce_uniform` wins despite class imbalance:** USED-FOR dominates training data but
is also the easiest class to learn. The signal from easy correct predictions still
contributes useful gradient updates. Suppressing it (via weighting or focal loss) removes
information rather than redirecting it.

---

## Word-to-Subword Alignment

The `metadata` field in the dataset provides entity boundaries as word-level indices
(e.g., `[7, 7, 9, 10]`). BERT uses subword tokenization, so "classifications" might
become `["class", "##ification", "##s"]` — three tokens for one word.

`word_span_to_token_span()` in `dataset.py` converts word-level indices to subword
token indices using the fast tokenizer's `word_ids()` output. This is required for
SpERT (span widths) and PL-Marker (marker injection at true boundaries).

---

## SpERT

SpERT augments the standard entity marker approach with:
1. **Span width embeddings** — learned embedding for entity length (in words)
2. **Context vector** — mean of tokens strictly between [E1] and [E2]

Classifier input: `e1_repr || e2_repr || e1_width_emb || e2_width_emb || context`

Empirically, this adds complexity without gain on SciERC — the existing markers already
capture enough boundary information (SpERT test F1: 0.8179 vs baseline 0.8243).

---

## PL-Marker

Instead of placing markers at the original `[[`/`>>` boundaries in the text, PL-Marker
injects `[M]`/`[/M]` tokens at the true word-level span boundaries derived from `metadata`.
This gives the model more precise positional information about where each entity starts and ends.

Underperformed on SciERC (test F1: 0.7786) — the additional positional precision requires
more training data than ~3200 samples to learn from effectively.

---

## PURE-Lite

PURE-Lite performs two forward passes through the encoder:
1. **Pass 1:** standard encoding — extract entity representations and project them to
   pseudo-type vectors (continuous, not discrete)
2. **Pass 2:** add pseudo-type vectors to token embeddings at entity positions, re-encode

This is inspired by PURE (Princeton IE), which uses gold entity type labels for typed
markers. PURE-Lite replaces discrete types with continuous projections since SciERC
does not provide entity types in this dataset format.

Implementation has a known bug (dtype mismatch in embedding injection). Not yet evaluated.

---

## Optimizers

### Standard (default)
`AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)` — single LR for all parameters.

### Separate LR (`--separate-lr`)
Two param groups: encoder at `lr`, classifier head at `lr * 5`.
Rationale: the classifier is randomly initialized and can tolerate a higher LR; the
pretrained encoder needs a gentler update.

### LLRD (`--llrd`)
Per-layer decaying LR: top transformer layer gets `lr`, each layer below is multiplied
by a decay factor (default 0.9), embeddings get the lowest LR.
Rationale: lower layers encode general linguistic knowledge that should change slowly;
upper layers encode task-specific features that can adapt more aggressively.
Produces the lowest variance results of all optimizer variants (std 0.0041).

---

## Early Stopping

Training stops when dev macro F1 does not improve by more than `min_delta` for
`patience` consecutive epochs. The best checkpoint is saved and used for test evaluation.

**Why `--epochs 100 --patience 10` for new models:**
Large general-purpose models like DeBERTa start from non-domain-specific representations
and need more epochs to adapt. SciBERT converges around epoch 15–20 with these settings.
The 100-epoch ceiling is never reached in practice — it just prevents artificial truncation.

---

## Key Findings Summary

1. **Domain pretraining beats model size.** SciBERT (domain-adapted, 768-dim) outperforms
   BERT-large and RoBERTa-large (general, 1024-dim) on SciERC.

2. **Architectural complexity doesn't help on small data.** SpERT and PL-Marker both
   underperform the baseline. ~3200 training samples is insufficient to learn the
   additional inductive biases these models introduce.

3. **`e1e2_concat + ce_uniform` is a strong, stable baseline.** No training trick or
   architecture change improved it meaningfully across 13 experiment configurations.

4. **Minority class performance is a data problem, not a model problem.** PART-OF and
   FEATURE-OF remain weak (F1 ~0.64–0.65) regardless of loss function, architecture,
   or optimizer. More annotated data or cross-dataset transfer is needed to address this.
