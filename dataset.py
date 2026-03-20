"""Dataset loading and preprocessing for SciBERT relation extraction."""

import json
import torch
import random
from torch.utils.data import Dataset
from config import LABEL2ID, MAX_LEN, SYMMETRIC_RELATIONS, UNDERSAMPLE_CONJUNCTION_TARGET


def word_span_to_token_span(
    word_ids: list[int | None],
    word_start: int,
    word_end: int,
) -> tuple[int, int]:
    """
    Convert inclusive word-level span [word_start, word_end] to the first and last
    subword token indices (inclusive) using word_ids() from a HuggingFace fast tokenizer.

    Args:
        word_ids: List of word indices from tokenizer.word_ids(batch_index=0)
        word_start: Inclusive word start index (0-based)
        word_end: Inclusive word end index (0-based)

    Returns:
        (token_start, token_end) indices, or (0, 0) as fallback if span not found (truncation edge case).
    """
    token_start, token_end = None, None
    for tok_idx, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid == word_start and token_start is None:
            token_start = tok_idx
        if wid == word_end:
            token_end = tok_idx
    if token_start is None or token_end is None:
        return 0, 0
    return token_start, token_end


class SciERCDataset(Dataset):
    """
    Reads .txt files where each line is a JSON with:
      - "text"  : sentence with [[ head ]] and << tail >> markers
      - "label" : relation type
      - "metadata" : [e1_start, e1_end, e2_start, e2_end] word-level indices
    
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
                    "label": LABEL2ID[item["label"]],
                    "label_str": item["label"],
                    "e1_start": item["metadata"][0],
                    "e1_end": item["metadata"][1],
                    "e2_start": item["metadata"][2],
                    "e2_end": item["metadata"][3],
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

        # Compute word-to-subword alignment for entity spans
        word_ids = enc.word_ids(batch_index=0)
        e1_tok_start, e1_tok_end = word_span_to_token_span(
            word_ids, self.samples[idx]["e1_start"], self.samples[idx]["e1_end"]
        )
        e2_tok_start, e2_tok_end = word_span_to_token_span(
            word_ids, self.samples[idx]["e2_start"], self.samples[idx]["e2_end"]
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "e1_pos": torch.tensor(e1_pos, dtype=torch.long),
            "e2_pos": torch.tensor(e2_pos, dtype=torch.long),
            "e1_tok_start": torch.tensor(e1_tok_start, dtype=torch.long),
            "e1_tok_end": torch.tensor(e1_tok_end, dtype=torch.long),
            "e2_tok_start": torch.tensor(e2_tok_start, dtype=torch.long),
            "e2_tok_end": torch.tensor(e2_tok_end, dtype=torch.long),
            "e1_word_len": torch.tensor(
                self.samples[idx]["e1_end"] - self.samples[idx]["e1_start"] + 1,
                dtype=torch.long
            ),
            "e2_word_len": torch.tensor(
                self.samples[idx]["e2_end"] - self.samples[idx]["e2_start"] + 1,
                dtype=torch.long
            ),
            "label": torch.tensor(label, dtype=torch.long)
        }


class PLMarkerSciERCDataset(SciERCDataset):
    """
    PL-Marker variant using levitated markers [M] and [/M] at real span boundaries
    from metadata, instead of [E1]/[E2]/[/E1]/[/E2] markers.
    
    The dataset injects [M] before the first word and [/M] after the last word
    of each entity span (from metadata), using word-level boundaries.
    """
    
    def __init__(self, filepath, tokenizer, max_len=MAX_LEN, max_samples=0, use_entity_markers=True):
        super().__init__(filepath, tokenizer, max_len, max_samples, use_entity_markers=False)
        # Always use False for parent, since we handle markers differently

    def __getitem__(self, idx):
        text = self.samples[idx]["text"]
        label = self.samples[idx]["label"]
        
        # Remove original visual markers [[]], <<>>
        text = text.replace("[[", "").replace("]]", "")
        text = text.replace("<<", "").replace(">>", "")
        
        # Split into words
        words = text.split()
        
        # Insert [M] markers at real span boundaries (word-level)
        e1_start = self.samples[idx]["e1_start"]
        e1_end = self.samples[idx]["e1_end"]
        e2_start = self.samples[idx]["e2_start"]
        e2_end = self.samples[idx]["e2_end"]
        
        # Build marked text: insert [M] before first word, [/M] after last word of each span
        marked_words = words.copy()
        
        # Insert in reverse order to maintain indices
        if e2_start <= e2_end:
            marked_words.insert(e2_end + 1, "[/M]")
            marked_words.insert(e2_start, "[M]")
        if e1_start <= e1_end:
            marked_words.insert(e1_end + 1, "[/M]")
            marked_words.insert(e1_start, "[M]")
        
        text_with_markers = " ".join(marked_words)
        
        # Tokenize
        enc = self.tokenizer(
            text_with_markers,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        
        # Find two [M] positions for e1 and e2
        m_id = self.tokenizer.convert_tokens_to_ids("[M]")
        m_matches = (input_ids == m_id).nonzero(as_tuple=True)[0]
        
        # First [M] is e1, second [M] is e2
        e1_pos = m_matches[0].item() if len(m_matches) > 0 else 0
        e2_pos = m_matches[1].item() if len(m_matches) > 1 else 0
        
        # Compute word-to-subword alignment for entity spans (from word-level metadata)
        word_ids = enc.word_ids(batch_index=0)
        e1_tok_start, e1_tok_end = word_span_to_token_span(
            word_ids, self.samples[idx]["e1_start"], self.samples[idx]["e1_end"]
        )
        e2_tok_start, e2_tok_end = word_span_to_token_span(
            word_ids, self.samples[idx]["e2_start"], self.samples[idx]["e2_end"]
        )
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "e1_pos": torch.tensor(e1_pos, dtype=torch.long),
            "e2_pos": torch.tensor(e2_pos, dtype=torch.long),
            "e1_tok_start": torch.tensor(e1_tok_start, dtype=torch.long),
            "e1_tok_end": torch.tensor(e1_tok_end, dtype=torch.long),
            "e2_tok_start": torch.tensor(e2_tok_start, dtype=torch.long),
            "e2_tok_end": torch.tensor(e2_tok_end, dtype=torch.long),
            "e1_word_len": torch.tensor(
                self.samples[idx]["e1_end"] - self.samples[idx]["e1_start"] + 1,
                dtype=torch.long
            ),
            "e2_word_len": torch.tensor(
                self.samples[idx]["e2_end"] - self.samples[idx]["e2_start"] + 1,
                dtype=torch.long
            ),
            "label": torch.tensor(label, dtype=torch.long)
        }



# ────────────────────────────────────────────────────────────────────────────────
# Data Augmentation Functions
# ────────────────────────────────────────────────────────────────────────────────


def augment_symmetric(samples: list[dict]) -> list[dict]:
    """
    For CONJUNCTION and COMPARE (symmetric relations), double training data by swapping entity order.
    
    Args:
        samples: List of sample dicts from SciERCDataset
    
    Returns:
        Augmented samples list with swapped entity pairs for symmetric relations
    """
    augmented = []
    for s in samples:
        augmented.append(s)
        if s["label_str"] in SYMMETRIC_RELATIONS:
            # Swap entity markers in text
            new_text = (s["text"]
                .replace("[[", "TEMP1").replace("]]", "TEMP2")
                .replace("<<", "[[").replace(">>", "]]")
                .replace("TEMP1", "<<").replace("TEMP2", ">>"))
            
            # Swap metadata
            new_metadata = [s["e2_start"], s["e2_end"], s["e1_start"], s["e1_end"]]
            
            augmented.append({
                "text":      new_text,
                "label":     s["label"],
                "label_str": s["label_str"],
                "e1_start":  new_metadata[0],
                "e1_end":    new_metadata[1],
                "e2_start":  new_metadata[2],
                "e2_end":    new_metadata[3],
            })
    return augmented


def undersample_label(
    samples: list[dict],
    label_str: str,
    target_count: int,
    seed: int = 42,
) -> list[dict]:
    """
    Randomly subsample instances of a target label to reduce its dominance.
    Apply to training split only — never dev or test.
    
    Args:
        samples: List of sample dicts
        label_str: Label name to undersample (e.g., "CONJUNCTION")
        target_count: Target sample count for that label
        seed: Random seed for reproducibility
    
    Returns:
        List with undersampled label
    """
    rng = random.Random(seed)
    keep = [s for s in samples if s["label_str"] != label_str]
    majority = [s for s in samples if s["label_str"] == label_str]
    keep += rng.sample(majority, min(target_count, len(majority)))
    return keep
