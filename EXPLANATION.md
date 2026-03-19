# Comprehensive Ablation Summary

Date of run: 2026-03-18  
Summary created: 2026-03-19

## 0. Theory Primer (For Non-NLP Readers)

### What task are we solving?

This project does **relation extraction**.

- Input: a sentence with two scientific entities (for example, "neural network" and "image classification").
- Output: the semantic relation between those two entities (for example, `USED-FOR`, `PART-OF`, `FEATURE-OF`).

So the model is not just finding entities; it is classifying how they are related.

### What are E1 and E2?

`E1` and `E2` mean the two target entities in a sentence.

- `E1` = first entity
- `E2` = second entity

In preprocessing, we insert special markers into the sentence so the model knows exactly which spans matter.

Example:

`[E1] convolutional network [/E1] is used for [E2] image segmentation [/E2]`

Here, the relation label is likely `USED-FOR`.

### Why pooling strategies matter

A transformer outputs a vector for each token. We must convert token-level outputs into one feature vector for classification.

- `e1e2_concat`: take the vector at `E1` start token and the vector at `E2` start token, then concatenate.
- `e1_only`: use only the `E1` vector.
- `cls_only`: use the special `[CLS]` sentence vector.
- `mean_pool`: average all token vectors.

Intuition: relation extraction is centered on **two entities**, so methods that explicitly use both entity vectors often perform better.

#### What is the CLS token?
In transformer models (like BERT), the input sentence is prepended with a special token called `[CLS]` (for "classification").
- The model produces a vector for `[CLS]` that summarizes the whole sentence.
- Using `cls_only` means the classifier only looks at this summary vector, ignoring entity-specific information.

#### What is mean pooling?
Mean pooling takes the average of all token vectors in the sentence.
- This approach treats every token equally, including both entities and context.
- It can dilute the signal from the entities, but sometimes helps when context is important.

#### How is mean pool different from CLS?
- `[CLS]` is a learned summary vector, trained to capture sentence-level meaning.
- Mean pool is a simple average, not learned, and can be influenced by irrelevant tokens.
- In relation extraction, entity-specific pooling (like `e1e2_concat`) usually outperforms both.

### What is the loss function here?
Loss is the training objective minimized by gradient descent.

- `ce_uniform`: standard cross-entropy, all classes weighted equally.
- `weighted_ce`: cross-entropy with larger weights for rare classes.

Why weighted loss can help: SciERC is imbalanced, so minority relations may be ignored unless they receive larger penalty when misclassified.

### How do the two losses differ?

- `ce_uniform` (uniform cross-entropy):
  - All relation classes are treated equally.
  - The model is penalized the same amount for mistakes on common and rare classes.
  - Can lead to poor performance on minority relations.
- `weighted_ce` (weighted cross-entropy):
  - Assigns higher penalty to mistakes on rare classes.
  - Helps the model pay more attention to minority relations.
  - Often improves macro-F1, but can sometimes hurt overall accuracy if weights are too aggressive.

### What is tokenization?

Tokenization is splitting a sentence into smaller pieces (tokens) for the model.
- Transformers use subword tokenization, so words like "classification" may become "class", "##ification".
- Entity markers (`[E1]`, `[E2]`) are added to help the model locate entities.

### What are entity markers?

Special tokens like `[E1]`, `[/E1]`, `[E2]`, `[/E2]` are inserted around the entities of interest.
- This tells the model which parts of the sentence are the target entities.

### What is F1 score?

F1 score is a measure of model accuracy that balances precision and recall.
- Precision: How many predicted relations are correct?
- Recall: How many true relations did the model find?
- F1: Harmonic mean of precision and recall.

### What is macro-F1?

Macro-F1 averages F1 scores across all classes, treating each class equally.
- This is important for imbalanced datasets, so rare relations are not ignored.

### What is standard deviation in results?

Standard deviation shows how much the results vary across different random seeds.
- Low std means the method is stable and reproducible.

## 1. Goal

We ran a comprehensive ablation to compare:
- Pooling strategies: `e1e2_concat`, `cls_only`, `mean_pool`, `e1_only`
- Loss functions: `weighted_ce`, `ce_uniform`
- Random seeds: `13, 21, 42, 87, 100`

The objective was to identify the most robust configuration for SciERC relation extraction.

## 2. Command Used

```bash
python main.py \
  --model-variants e1e2_concat,cls_only,mean_pool,e1_only \
  --loss-variants weighted_ce,ce_uniform \
  --seeds 13,21,42,87,100 \
  --run-name comprehensive_ablation
```

## 3. Experiment Scale

- Total configurations: 4 model variants × 2 loss variants = 8
- Seeds per configuration: 5
- Total runs: 40
- Output directory: `runs/comprehensive_ablation/`
- Aggregated summary: `runs/comprehensive_ablation/experiment_summary.json`

## 4. Main Results (Test Macro-F1)

| Rank | Model Variant | Loss Variant | Test Macro-F1 (mean ± std) | Dev Macro-F1 (mean ± std) |
|---|---|---|---|---|
| 1 | e1e2_concat | ce_uniform | **0.8243 ± 0.0052** | 0.8882 ± 0.0086 |
| 2 | e1e2_concat | weighted_ce | 0.8157 ± 0.0155 | 0.8929 ± 0.0069 |
| 3 | e1_only | weighted_ce | 0.8132 ± 0.0051 | 0.8714 ± 0.0034 |
| 4 | e1_only | ce_uniform | 0.8078 ± 0.0114 | 0.8732 ± 0.0152 |
| 5 | cls_only | weighted_ce | 0.7918 ± 0.0062 | 0.8437 ± 0.0137 |
| 6 | mean_pool | weighted_ce | 0.7879 ± 0.0041 | 0.8524 ± 0.0099 |
| 7 | cls_only | ce_uniform | 0.7836 ± 0.0095 | 0.8504 ± 0.0114 |
| 8 | mean_pool | ce_uniform | 0.7801 ± 0.0220 | 0.8527 ± 0.0094 |

## 5. What We Learned

1. Best overall configuration is `e1e2_concat + ce_uniform`.
2. `e1e2_concat` is the strongest pooling strategy regardless of loss choice.
3. `weighted_ce` helps 3/4 pooling strategies (`cls_only`, `mean_pool`, `e1_only`) but hurts `e1e2_concat`.
4. `mean_pool + ce_uniform` is unstable across seeds (largest test std = 0.0220).

## 6. Final Decision After This Ablation

Use the following as the new default experiment setting:
- model variant: `e1e2_concat`
- loss variant: `ce_uniform`
- seeds: `13, 21, 42, 87, 100`

## 7. Artifacts Generated

Each run folder contains:
- `best_model.pt`
- `test_classification_report.json`
- `test_per_class_metrics.csv`
- `test_confusion_matrix.csv`
- `test_confusion_matrix.txt`

Example run folders:
- `runs/comprehensive_ablation/e1e2_concat_ce_uniform_frozen0_seed13/`
- `runs/comprehensive_ablation/e1e2_concat_ce_uniform_frozen0_seed42/`

## 8. Recommended Next Steps

1. Encoder swap experiments (keep best current setup fixed):
   - `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
   - `allenai/specter2_base`
   - `microsoft/deberta-v3-base`
2. Run 3-seed screening for new encoders first (`13, 42, 87`), then full 5-seed on top 2.
3. Improve data effectiveness with hard-example mining and entity-safe augmentation while keeping SciERC as the benchmark.
