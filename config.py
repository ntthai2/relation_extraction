"""Centralized configuration for SciBERT relation extraction."""

# ── Dataset & Labels ──────────────────────────────────────────────────────────

LABELS = [
    "USED-FOR", "CONJUNCTION", "EVALUATE-FOR", "HYPONYM-OF",
    "PART-OF", "FEATURE-OF", "COMPARE"
]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
NUM_LABELS = len(LABELS)

DATA_DIR = "scierc"
TRAIN_FILE = "scierc/train.txt"
DEV_FILE = "scierc/dev.txt"
TEST_FILE = "scierc/test.txt"

# ── Model ─────────────────────────────────────────────────────────────────────

MODEL_NAME = "allenai/scibert_scivocab_uncased"
DEBERTA_MODEL_NAME = "microsoft/deberta-v3-base"
ROBERTA_LARGE_MODEL_NAME = "roberta-large"
BERT_LARGE_MODEL_NAME = "bert-large-uncased"
MAX_LEN = 256
HIDDEN_SIZE = 768

# ── Training ──────────────────────────────────────────────────────────────────

BATCH_SIZE = 32
EPOCHS = 10
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42

# ── Early Stopping ────────────────────────────────────────────────────────────

PATIENCE = 3
MIN_DELTA = 1e-4

# ── Output ────────────────────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR = "runs"
DEFAULT_LOG_PATH = ""  # If empty, logs auto-saved to runs/<run_name>/train.log
DEFAULT_RUN_NAME = "scibert_scierc"

# ── Model Variants ───────────────────────────────────────────────────────────

MODEL_VARIANTS = {
    "e1e2_concat": {
        "name": "Entity Start Token Concatenation (main)",
        "pooling": "e1e2_concat",
        "description": "Concatenate [E1] and [E2] representations"
    },
    "cls_only": {
        "name": "CLS Token Only (baseline)",
        "pooling": "cls",
        "description": "Use [CLS] token representation only"
    },
    "mean_pool": {
        "name": "Mean Pooling (ablation)",
        "pooling": "mean",
        "description": "Average all token representations"
    },
    "e1_only": {
        "name": "E1 Only (ablation)",
        "pooling": "e1_only",
        "description": "Use only [E1] token representation"
    },
}

# ── Loss Variants ─────────────────────────────────────────────────────────────

LOSS_VARIANTS = {
    "weighted_ce": {
        "name": "Weighted CrossEntropy",
        "description": "Class-weighted cross-entropy for imbalance"
    },
    "ce_uniform": {
        "name": "Uniform CrossEntropy",
        "description": "Standard cross-entropy without class weights"
    },
    "focal": {
        "name": "Focal Loss",
        "description": "Down-weights well-classified examples; focuses on hard/minority classes"
    },
    "label_smooth": {
        "name": "Label Smoothing",
        "description": "Cross-entropy with label smoothing (0.1)"
    },
}

# ── Special Tokens ────────────────────────────────────────────────────────────

ENTITY_MARKERS = ["[E1]", "[/E1]", "[E2]", "[/E2]"]
LEVITATED_MARKERS = ["[M]", "[/M]"]
USE_ENTITY_MARKERS = True

# ── SpERT Hyperparameters ──────────────────────────────────────────────────────

SPERT_WIDTH_EMBEDDING_DIM = 25
SPERT_MAX_SPAN_WIDTH = 30

# ── PURE-Lite Hyperparameters ──────────────────────────────────────────────────

PURE_LITE_TYPE_DIM = 64

# ── Data Augmentation ──────────────────────────────────────────────────────────

SYMMETRIC_RELATIONS = {"CONJUNCTION", "COMPARE"}
UNDERSAMPLE_CONJUNCTION_TARGET = 250
