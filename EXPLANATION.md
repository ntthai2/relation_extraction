# Concepts & Architecture Explanation

Reference for understanding the design decisions in this codebase. For experimental
results, see [EXPERIMENTS.md](EXPERIMENTS.md).

---

## Overview

This project has two components built on top of each other:

1. **Relation Extraction (RE)** — given a sentence with two pre-marked entities, classify the semantic relation between them. Trained on SciERC.
2. **Knowledge Graph Pipeline** — given raw arXiv abstracts, detect entities (NER), classify relations between all entity pairs (RE), and build a queryable knowledge graph.

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
PART-OF and FEATURE-OF are the hardest classes (F1 ~0.64-0.65).

---

### Entity Markers

Special tokens [E1], [/E1], [E2], [/E2] are inserted around entities before
tokenization. This tells the model exactly which spans to focus on.

Raw text in data: [[ convolutional network ]] is used for << image segmentation >>
After preprocessing: [E1] convolutional network [/E1] is used for [E2] image segmentation [/E2]

These are added as additional special tokens to the tokenizer vocabulary so the model
treats them as atomic units rather than splitting them into subwords.

---

### Pooling Strategies

A transformer outputs one vector per token. For classification, we need a single
fixed-size vector. The pooling strategy decides which token(s) to use.

| Strategy | What it uses | Intuition |
|---|---|---|
| e1e2_concat | [E1] and [E2] start token vectors, concatenated | Directly encodes both entities — best for RE |
| e1_only | [E1] start token vector | Partial entity info |
| cls_only | [CLS] sentence summary vector | Ignores entity positions |
| mean_pool | Average of all token vectors | Dilutes entity signal with context |

Why e1e2_concat wins: relation extraction is fundamentally about two entities.
Using both entity representations directly captures more signal than any single-vector
approach. Empirically, e1e2_concat outperformed all other strategies by 1-4 F1 points
across all experiments (see EXPERIMENTS.md).

---

### Classifier Head

All models use the same two-layer MLP on top of the pooled representation:

  pooled_repr -> Linear(input_dim, 768) -> ReLU -> Dropout -> Linear(768, 7) -> logits

Input dim is 768 * 2 = 1536 for e1e2_concat, 768 for all others.
For large models (RoBERTa-large, BERT-large), replace 768 with 1024.

---

### Loss Functions

| Loss | Behaviour | Finding |
|---|---|---|
| ce_uniform | Penalizes all classes equally | Best - counterintuitively |
| weighted_ce | Higher penalty for rare classes | Slightly hurts e1e2_concat |
| focal | Down-weights easy examples | Actively hurts - removes useful USED-FOR signal |
| label_smooth | Soft targets (0.1 smoothing) | Neutral |

Why ce_uniform wins despite class imbalance: USED-FOR dominates training data but
is also the easiest class to learn. The signal from easy correct predictions still
contributes useful gradient updates. Suppressing it (via weighting or focal loss) removes
information rather than redirecting it.

---

### Word-to-Subword Alignment

The metadata field in the dataset provides entity boundaries as word-level indices
(e.g., [7, 7, 9, 10]). BERT uses subword tokenization, so "classifications" might
become ["class", "##ification", "##s"] - three tokens for one word.

word_span_to_token_span() in re_dataset.py converts word-level indices to subword
token indices using the fast tokenizer's word_ids() output. This is required for
SpERT (span widths) and PL-Marker (marker injection at true boundaries).

---

### RE-specific Architectures

SpERT augments the standard entity marker approach with span width embeddings and a
context vector (mean of tokens between [E1] and [E2]). Empirically adds complexity
without gain — SpERT test F1: 0.8179 vs baseline 0.8243.

PL-Marker injects [M]/[/M] tokens at true word-level span boundaries derived from
metadata. Underperformed on SciERC (test F1: 0.7786) — requires more training data
than ~3200 samples to learn the new positional semantics effectively.

PURE-Lite performs two forward passes: Pass 1 extracts entity representations and
projects them to pseudo-type vectors. Pass 2 adds these to token embeddings and
re-encodes. Has a known bug (dtype mismatch). Not yet evaluated.

---

### Optimizers

Standard (default): AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

Separate LR (--separate-lr): two param groups - encoder at lr, classifier head at
lr * 5. The classifier is randomly initialized and tolerates a higher LR.

LLRD (--llrd): per-layer decaying LR. Top transformer layer gets lr, each layer below
is multiplied by a decay factor (default 0.9). Produces the lowest variance of all
optimizer variants (std 0.0041).

---

### Early Stopping

Training stops when dev macro F1 does not improve by more than min_delta for patience
consecutive epochs. Best checkpoint saved and used for test evaluation.

Why --epochs 100 --patience 10: SciBERT converges around epoch 15-20. The 100-epoch
ceiling is never reached in practice - it just prevents artificial truncation.
Large general-purpose models like DeBERTa may need more epochs to adapt from their
non-domain-specific initialization.

---

### RE Key Findings

1. Domain pretraining beats model size. SciBERT (domain-adapted, 768-dim) outperforms
   BERT-large and RoBERTa-large (general, 1024-dim) on SciERC.
2. Architectural complexity does not help on small data. SpERT and PL-Marker both
   underperform. ~3200 training samples is insufficient to learn the additional inductive
   biases these models introduce.
3. e1e2_concat + ce_uniform is a strong, stable baseline. No training trick or
   architecture change improved it meaningfully across 13 experiment configurations.
4. Minority class performance is a data problem, not a model problem. PART-OF and
   FEATURE-OF remain weak (F1 ~0.64-0.65) regardless of loss function, architecture,
   or optimizer. More annotated data or cross-dataset transfer is needed.

---

## Part 2 — Knowledge Graph Pipeline

### NER — Named Entity Recognition

To run the RE model on raw text, entities must first be detected automatically.
ner_model.py implements a SciBERT token classifier trained on SciERC NER annotations.

BIO tagging: each word gets one of 7 labels:
  O, B-TASK, I-TASK, B-METHOD, I-METHOD, B-DATASET, I-DATASET

Entity type mapping from SciERC 6 types to 3:
  Task                -> TASK    (keep)
  Method              -> METHOD  (keep)
  Material            -> DATASET (rename - Material includes datasets)
  Metric              -> skip
  OtherScientificTerm -> skip
  Generic             -> skip

Metric, OtherScientificTerm, and Generic are dropped because they add noise or are
less useful for KG construction. Generic entities are pronouns/placeholders that only
exist to support coreference resolution (which this pipeline skips).

Nested span resolution: SciERC contains nested entity spans. The inner span is kept,
the outer is discarded. Inner spans tend to be more specific and produce more reusable
KG nodes.

Subword alignment: only the first subword of each word contributes to loss and
predictions. Continuation subwords receive label -100 (ignored by cross-entropy).

NER results: test token F1 ~0.796 +/- 0.003 (3 seeds). Weakest class: B-DATASET
(0.623) - the Material->Dataset mapping is imperfect since Material in SciERC includes
non-dataset resources like WordNet and Wikipedia.

---

### Inference Pipeline (inference.py)

End-to-end pipeline: raw text -> triples.

1. Sentence splitting - nltk.sent_tokenize
2. NER - SciBERTNER on each sentence, BIO -> entity spans
3. Pair enumeration - all ordered pairs (a, b) and (b, a) within each sentence,
   including same-type pairs (needed for CONJUNCTION and COMPARE)
4. RE input formatting - insert [E1]/[/E1] and [E2]/[/E2] around entity spans
5. RE inference - batched forward pass, softmax confidence scores
6. Confidence filtering - discard predictions below threshold (default 0.5).
   The RE model has no NULL class so confidence thresholding is the only noise filter.
7. Entity normalization - lowercase, strip punctuation, remove leading determiners,
   expand common acronyms (LLMs -> large language models, etc.)

Output per triple: subject, subject_type, relation, object, object_type, confidence,
source_sentence, paper_id.

---

### Knowledge Graph Construction (build_kg.py)

Graph type: nx.MultiDiGraph - supports multiple relation types between the same node pair.
Nodes: unique normalized entity strings, attributes: type, count.
Edges: directed subject->object, attributes: relation, weight (frequency),
       confidence_mean, papers (list of source paper IDs).
Deduplication: same subject->object->relation triples are merged by incrementing weight.
               Different relations between the same pair remain as separate edges.
Self-loops filtered: subject == object pairs are discarded.

Visualization: pyvis interactive HTML, top 300 nodes by degree, edges with weight >= 2.
Node color encodes type (blue=TASK, orange=METHOD, green=DATASET).
Edge color encodes relation type.

---

### KG Findings (500 cs.CL papers, 2024-2025)

1. Large language models is the dominant hub - degree 1,440, more than 9x the next node.
   Every NLP subfield connects through it.
2. Top research pattern - LLMs -> USED-FOR -> reasoning (weight=10). Reasoning is both
   the top application and the top evaluation target.
3. Training paradigm shift - reinforcement learning <-> CONJUNCTION <-> supervised
   fine-tuning appears 5 times each direction, reflecting the post-SFT RLHF/RLVR trend.
4. RAG has matured - RAG variants are being compared against each other and against base
   LLMs, signaling the technology has moved from novelty to benchmark.
5. LLM taxonomy forming - GPT-4o, Gemini, LLaMA all HYPONYM-OF large language models.
6. Multilingual NLP active - Chinese <-> English is the top CONJUNCTION pair (weight=6).

Known quality issues:
- "language models" and "large language models" remain separate nodes
- "human" misclassified as METHOD - model picks up "human evaluation" patterns
- Generic terms (data, text, models) appear as entities due to NER imprecision
- Abstract-only extraction misses relations that only appear in methodology/results sections
