"""Dataset loading and preprocessing for SciBERT relation extraction."""

import json
import torch
from torch.utils.data import Dataset
from config import LABEL2ID, MAX_LEN


class SciERCDataset(Dataset):
    """
    Reads .txt files where each line is a JSON with:
      - "text"  : sentence with [[ head ]] and << tail >> markers
      - "label" : relation type
    
    Strategy: replace [[ ]] and << >> with special tokens
      [[ → [E1]   ]] → [/E1]
      << → [E2]   >> → [/E2]
    Then feed to SciBERT, take entity token representations, pool, and classify.
    """

    def __init__(self, filepath, tokenizer, max_len=MAX_LEN, max_samples=0, use_entity_markers=True):
        """
        Args:
            filepath: Path to the JSON Lines file
            tokenizer: HuggingFace tokenizer
            max_len: Maximum sequence length
            max_samples: If > 0, cap the number of samples loaded
            use_entity_markers: If False, don't replace [[ ]] and << >>
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_entity_markers = use_entity_markers
        self.samples = []
        
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                self.samples.append({
                    "text": item["text"],
                    "label": LABEL2ID[item["label"]]
                })
                if max_samples > 0 and len(self.samples) >= max_samples:
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]["text"]
        label = self.samples[idx]["label"]

        # Replace visual markers with special tokens (if enabled)
        if self.use_entity_markers:
            text = text.replace("[[", "[E1]").replace("]]", "[/E1]")
            text = text.replace("<<", "[E2]").replace(">>", "[/E2]")

        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # Find positions of [E1] and [E2] start tokens if markers are used
        e1_pos = 0  # default to [CLS]
        e2_pos = 0  # default to [CLS]
        
        if self.use_entity_markers:
            e1_id = self.tokenizer.convert_tokens_to_ids("[E1]")
            e2_id = self.tokenizer.convert_tokens_to_ids("[E2]")

            e1_matches = (input_ids == e1_id).nonzero(as_tuple=True)[0]
            e2_matches = (input_ids == e2_id).nonzero(as_tuple=True)[0]

            # Fallback to [CLS] if marker not found (truncation edge case)
            e1_pos = e1_matches[0].item() if len(e1_matches) > 0 else 0
            e2_pos = e2_matches[0].item() if len(e2_matches) > 0 else 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "e1_pos": torch.tensor(e1_pos, dtype=torch.long),
            "e2_pos": torch.tensor(e2_pos, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long)
        }
