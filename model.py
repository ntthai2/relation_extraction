"""Model architectures for relation extraction baselines and ablations."""

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel
from config import (
    MODEL_NAME, DEBERTA_MODEL_NAME, ROBERTA_LARGE_MODEL_NAME, BERT_LARGE_MODEL_NAME,
    HIDDEN_SIZE, NUM_LABELS, SPERT_WIDTH_EMBEDDING_DIM, SPERT_MAX_SPAN_WIDTH, PURE_LITE_TYPE_DIM
)


class SciBERTRelationClassifier(nn.Module):
    """
    SciBERT encoder with configurable pooling strategy + classifier.
    
    Pooling strategies:
    - e1e2_concat: concatenate [E1] and [E2] representations
    - cls: use [CLS] token only
    - mean: average all token representations
    - e1_only: use [E1] token only
    """

    def __init__(self, num_labels=NUM_LABELS, dropout=0.1, pooling="e1e2_concat"):
        """
        Args:
            num_labels: Number of output classes
            dropout: Dropout rate for classification head
            pooling: Pooling strategy ("e1e2_concat", "cls", "mean", "e1_only")
        """
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        self.pooling = pooling
        self.dropout_layer = nn.Dropout(dropout)

        # Compute classifier input size based on pooling strategy
        if pooling == "e1e2_concat":
            classifier_input_dim = HIDDEN_SIZE * 2
        else:
            classifier_input_dim = HIDDEN_SIZE

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(HIDDEN_SIZE, num_labels)
        )

    def forward(self, input_ids, attention_mask, e1_pos, e2_pos):
        """
        Args:
            input_ids: [B, seq_len]
            attention_mask: [B, seq_len]
            e1_pos: [B] positions of [E1] tokens
            e2_pos: [B] positions of [E2] tokens
        
        Returns:
            logits: [B, num_labels]
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = outputs.last_hidden_state  # [B, seq_len, 768]

        if self.pooling == "e1e2_concat":
            # Gather [E1] and [E2] token representations and concatenate
            B = seq_out.size(0)
            e1_repr = seq_out[torch.arange(B), e1_pos]  # [B, 768]
            e2_repr = seq_out[torch.arange(B), e2_pos]  # [B, 768]
            combined = torch.cat([e1_repr, e2_repr], dim=-1)  # [B, 1536]
            combined = self.dropout_layer(combined)
            logits = self.classifier(combined)

        elif self.pooling == "cls":
            # Use [CLS] token only
            cls_repr = seq_out[:, 0, :]  # [B, 768]
            cls_repr = self.dropout_layer(cls_repr)
            logits = self.classifier(cls_repr)

        elif self.pooling == "mean":
            # Mean pooling across sequence
            mean_repr = seq_out.mean(dim=1)  # [B, 768]
            mean_repr = self.dropout_layer(mean_repr)
            logits = self.classifier(mean_repr)

        elif self.pooling == "e1_only":
            # Use [E1] token only
            B = seq_out.size(0)
            e1_repr = seq_out[torch.arange(B), e1_pos]  # [B, 768]
            e1_repr = self.dropout_layer(e1_repr)
            logits = self.classifier(e1_repr)

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        return logits


class FrozenSciBERTRelationClassifier(nn.Module):
    """
    SciBERT encoder (frozen) + trainable classification head.
    Baseline to test if fine-tuning is necessary.
    """

    def __init__(self, num_labels=NUM_LABELS, dropout=0.1, pooling="e1e2_concat"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        self.pooling = pooling
        self.dropout_layer = nn.Dropout(dropout)

        if pooling == "e1e2_concat":
            classifier_input_dim = HIDDEN_SIZE * 2
        else:
            classifier_input_dim = HIDDEN_SIZE

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(HIDDEN_SIZE, num_labels)
        )

    def forward(self, input_ids, attention_mask, e1_pos, e2_pos):
        """Same forward pass as SciBERTRelationClassifier, but with frozen BERT."""
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            seq_out = outputs.last_hidden_state

        if self.pooling == "e1e2_concat":
            B = seq_out.size(0)
            e1_repr = seq_out[torch.arange(B), e1_pos]
            e2_repr = seq_out[torch.arange(B), e2_pos]
            combined = torch.cat([e1_repr, e2_repr], dim=-1)
            combined = self.dropout_layer(combined)
            logits = self.classifier(combined)

        elif self.pooling == "cls":
            cls_repr = seq_out[:, 0, :]
            cls_repr = self.dropout_layer(cls_repr)
            logits = self.classifier(cls_repr)

        elif self.pooling == "mean":
            mean_repr = seq_out.mean(dim=1)
            mean_repr = self.dropout_layer(mean_repr)
            logits = self.classifier(mean_repr)

        elif self.pooling == "e1_only":
            B = seq_out.size(0)
            e1_repr = seq_out[torch.arange(B), e1_pos]
            e1_repr = self.dropout_layer(e1_repr)
            logits = self.classifier(e1_repr)

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        return logits


# ────────────────────────────────────────────────────────────────────────────────
# New Models
# ────────────────────────────────────────────────────────────────────────────────


class DeBERTaRelationClassifier(nn.Module):
    """
    DeBERTa-v3-base encoder with configurable pooling + classifier head.
    Drop-in replacement for SciBERTRelationClassifier.
    """
    def __init__(self, num_labels=NUM_LABELS, dropout=0.1, pooling="e1e2_concat"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(DEBERTA_MODEL_NAME)
        self.pooling = pooling
        self.dropout_layer = nn.Dropout(dropout)
        
        hidden_size = self.bert.config.hidden_size
        if pooling == "e1e2_concat":
            classifier_input_dim = hidden_size * 2
        else:
            classifier_input_dim = hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels)
        )

    def forward(self, input_ids, attention_mask, e1_pos, e2_pos):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = outputs.last_hidden_state

        if self.pooling == "e1e2_concat":
            B = seq_out.size(0)
            e1_repr = seq_out[torch.arange(B), e1_pos]
            e2_repr = seq_out[torch.arange(B), e2_pos]
            combined = torch.cat([e1_repr, e2_repr], dim=-1)
            combined = self.dropout_layer(combined)
            logits = self.classifier(combined)
        elif self.pooling == "cls":
            cls_repr = seq_out[:, 0, :]
            cls_repr = self.dropout_layer(cls_repr)
            logits = self.classifier(cls_repr)
        elif self.pooling == "mean":
            mean_repr = seq_out.mean(dim=1)
            mean_repr = self.dropout_layer(mean_repr)
            logits = self.classifier(mean_repr)
        elif self.pooling == "e1_only":
            B = seq_out.size(0)
            e1_repr = seq_out[torch.arange(B), e1_pos]
            e1_repr = self.dropout_layer(e1_repr)
            logits = self.classifier(e1_repr)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        return logits


class RoBERTaLargeRelationClassifier(nn.Module):
    """
    RoBERTa-large encoder (hidden_size=1024) with configurable pooling + classifier head.
    """
    def __init__(self, num_labels=NUM_LABELS, dropout=0.1, pooling="e1e2_concat"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(ROBERTA_LARGE_MODEL_NAME)
        self.pooling = pooling
        self.dropout_layer = nn.Dropout(dropout)
        
        hidden_size = self.bert.config.hidden_size
        if pooling == "e1e2_concat":
            classifier_input_dim = hidden_size * 2
        else:
            classifier_input_dim = hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels)
        )

    def forward(self, input_ids, attention_mask, e1_pos, e2_pos):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = outputs.last_hidden_state

        if self.pooling == "e1e2_concat":
            B = seq_out.size(0)
            e1_repr = seq_out[torch.arange(B), e1_pos]
            e2_repr = seq_out[torch.arange(B), e2_pos]
            combined = torch.cat([e1_repr, e2_repr], dim=-1)
            combined = self.dropout_layer(combined)
            logits = self.classifier(combined)
        elif self.pooling == "cls":
            cls_repr = seq_out[:, 0, :]
            cls_repr = self.dropout_layer(cls_repr)
            logits = self.classifier(cls_repr)
        elif self.pooling == "mean":
            mean_repr = seq_out.mean(dim=1)
            mean_repr = self.dropout_layer(mean_repr)
            logits = self.classifier(mean_repr)
        elif self.pooling == "e1_only":
            B = seq_out.size(0)
            e1_repr = seq_out[torch.arange(B), e1_pos]
            e1_repr = self.dropout_layer(e1_repr)
            logits = self.classifier(e1_repr)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        return logits


class BERTLargeRelationClassifier(nn.Module):
    """
    BERT-large-uncased encoder (hidden_size=1024) with configurable pooling + classifier head.
    """
    def __init__(self, num_labels=NUM_LABELS, dropout=0.1, pooling="e1e2_concat"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(BERT_LARGE_MODEL_NAME)
        self.pooling = pooling
        self.dropout_layer = nn.Dropout(dropout)
        
        hidden_size = self.bert.config.hidden_size
        if pooling == "e1e2_concat":
            classifier_input_dim = hidden_size * 2
        else:
            classifier_input_dim = hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels)
        )

    def forward(self, input_ids, attention_mask, e1_pos, e2_pos):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = outputs.last_hidden_state

        if self.pooling == "e1e2_concat":
            B = seq_out.size(0)
            e1_repr = seq_out[torch.arange(B), e1_pos]
            e2_repr = seq_out[torch.arange(B), e2_pos]
            combined = torch.cat([e1_repr, e2_repr], dim=-1)
            combined = self.dropout_layer(combined)
            logits = self.classifier(combined)
        elif self.pooling == "cls":
            cls_repr = seq_out[:, 0, :]
            cls_repr = self.dropout_layer(cls_repr)
            logits = self.classifier(cls_repr)
        elif self.pooling == "mean":
            mean_repr = seq_out.mean(dim=1)
            mean_repr = self.dropout_layer(mean_repr)
            logits = self.classifier(mean_repr)
        elif self.pooling == "e1_only":
            B = seq_out.size(0)
            e1_repr = seq_out[torch.arange(B), e1_pos]
            e1_repr = self.dropout_layer(e1_repr)
            logits = self.classifier(e1_repr)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        return logits


class SpERTRelationClassifier(nn.Module):
    """
    SpERT-inspired model using entity span representations + between-entity context.
    Requires e1_word_len, e2_word_len from dataset.

    ⚠️ Has 2 extra forward args. Cannot use train_core.py directly.
    """
    def __init__(self, num_labels=NUM_LABELS, dropout=0.1,
                 width_embedding_dim=SPERT_WIDTH_EMBEDDING_DIM,
                 max_span_width=SPERT_MAX_SPAN_WIDTH,
                 model_name=MODEL_NAME):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        self.width_embedding = nn.Embedding(max_span_width + 1, width_embedding_dim)
        self.max_span_width = max_span_width
        self.dropout_layer = nn.Dropout(dropout)
        
        classifier_input_dim = hidden_size * 2 + width_embedding_dim * 2 + hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels)
        )

    def forward(self, input_ids, attention_mask, e1_pos, e2_pos, e1_word_len, e2_word_len):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = outputs.last_hidden_state
        B = seq_out.size(0)
        
        # Entity representations
        e1_repr = seq_out[torch.arange(B), e1_pos]
        e2_repr = seq_out[torch.arange(B), e2_pos]
        
        # Width embeddings
        e1_width_idx = torch.clamp(e1_word_len, max=self.max_span_width)
        e2_width_idx = torch.clamp(e2_word_len, max=self.max_span_width)
        e1_width_repr = self.width_embedding(e1_width_idx)
        e2_width_repr = self.width_embedding(e2_width_idx)
        
        # Context: mean of tokens between e1 and e2
        context_repr_list = []
        for b in range(B):
            e1_p = e1_pos[b].item()
            e2_p = e2_pos[b].item()
            if e2_p > e1_p + 1:
                context = seq_out[b, e1_p+1:e2_p, :].mean(dim=0)
            else:
                context = torch.zeros(seq_out.size(-1), device=seq_out.device)
            context_repr_list.append(context)
        context_repr = torch.stack(context_repr_list, dim=0)
        
        # Concatenate all
        combined = torch.cat([e1_repr, e2_repr, e1_width_repr, e2_width_repr, context_repr], dim=-1)
        combined = self.dropout_layer(combined)
        logits = self.classifier(combined)
        
        return logits


class PLMarkerRelationClassifier(nn.Module):
    """
    PL-Marker-inspired classifier using levitated markers at real span boundaries.
    Uses [M] markers in dataset instead of [E1]/[E2].
    """
    def __init__(self, num_labels=NUM_LABELS, dropout=0.1, model_name=MODEL_NAME):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout_layer = nn.Dropout(dropout)

        classifier_input_dim = hidden_size * 2  # Always e1e2_concat with [M] markers
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels)
        )

    def forward(self, input_ids, attention_mask, e1_pos, e2_pos):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = outputs.last_hidden_state
        
        B = seq_out.size(0)
        e1_repr = seq_out[torch.arange(B), e1_pos]
        e2_repr = seq_out[torch.arange(B), e2_pos]
        combined = torch.cat([e1_repr, e2_repr], dim=-1)
        combined = self.dropout_layer(combined)
        logits = self.classifier(combined)
        
        return logits


class PURELiteRelationClassifier(nn.Module):
    """
    PURE-Lite: two-pass encoding without gold entity types.
    Uses continuous pseudo-type vectors instead of discrete types.
    """
    def __init__(self, num_labels=NUM_LABELS, dropout=0.1,
                 type_dim=PURE_LITE_TYPE_DIM, model_name=MODEL_NAME):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout_layer = nn.Dropout(dropout)
        
        # Type projection MLPs
        self.e1_type_proj = nn.Linear(hidden_size, type_dim)
        self.e2_type_proj = nn.Linear(hidden_size, type_dim)
        
        classifier_input_dim = hidden_size * 2
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels)
        )

    def forward(self, input_ids, attention_mask, e1_pos, e2_pos):
        # Pass 1
        outputs1 = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_out1 = outputs1.last_hidden_state
        B = seq_out1.size(0)
        
        e1_repr_pass1 = seq_out1[torch.arange(B), e1_pos]
        e2_repr_pass1 = seq_out1[torch.arange(B), e2_pos]
        
        e1_type_vec = self.e1_type_proj(e1_repr_pass1)
        e2_type_vec = self.e2_type_proj(e2_repr_pass1)
        
        # Pass 2: inject type vectors into input embeddings
        inputs_embeds = self.bert.embeddings.word_embeddings(input_ids).clone()
        for b in range(B):
            inputs_embeds[b, e1_pos[b]] = inputs_embeds[b, e1_pos[b]] + e1_type_vec[b]
            inputs_embeds[b, e2_pos[b]] = inputs_embeds[b, e2_pos[b]] + e2_type_vec[b]
        
        outputs2 = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        seq_out2 = outputs2.last_hidden_state
        
        e1_repr_pass2 = seq_out2[torch.arange(B), e1_pos]
        e2_repr_pass2 = seq_out2[torch.arange(B), e2_pos]
        
        combined = torch.cat([e1_repr_pass2, e2_repr_pass2], dim=-1)
        combined = self.dropout_layer(combined)
        logits = self.classifier(combined)
        
        return logits


# ────────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ────────────────────────────────────────────────────────────────────────────────


class FocalLoss(nn.Module):
    """Down-weights well-classified examples; focuses training on hard/minority classes."""
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

