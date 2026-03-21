"""NER model: SciBERT encoder + token classification head."""

import torch
from torch import nn
from transformers import AutoModel

from config import HIDDEN_SIZE, MODEL_NAME
from ner_dataset import NUM_NER_LABELS


class SciBERTNER(nn.Module):
    def __init__(self, num_labels: int = NUM_NER_LABELS, dropout: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(HIDDEN_SIZE, num_labels)

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        """
        Args:
            input_ids:      [B, seq_len]
            attention_mask: [B, seq_len]
        Returns:
            logits: [B, seq_len, num_labels]  — raw logits, no softmax
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = outputs.last_hidden_state
        seq_out = self.dropout(seq_out)
        logits = self.classifier(seq_out)
        return logits
