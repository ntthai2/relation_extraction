"""Model architectures for relation extraction baselines and ablations."""

import torch
from torch import nn
from transformers import AutoModel
from config import MODEL_NAME, HIDDEN_SIZE, NUM_LABELS


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
