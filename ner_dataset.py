"""NER dataset for scientific entity extraction using BIO tagging."""

import json

import torch
from torch.utils.data import Dataset


NER_LABELS = [
    "O",
    "B-TASK",
    "I-TASK",
    "B-METHOD",
    "I-METHOD",
    "B-DATASET",
    "I-DATASET",
]
NER_LABEL2ID = {label: i for i, label in enumerate(NER_LABELS)}
NER_ID2LABEL = {i: label for i, label in enumerate(NER_LABELS)}
NUM_NER_LABELS = len(NER_LABELS)

SCIERC_TO_KG_TYPE = {
    "Task": "TASK",
    "Method": "METHOD",
    "Material": "DATASET",
    "Metric": None,
    "OtherScientificTerm": None,
    "Generic": None,
}


def resolve_nested(entities: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """
    Given list of (start, end, type) tuples (sentence-level indices),
    resolve nested spans by keeping inner spans and discarding outer spans.
    Sort by span length ascending, then greedily keep non-overlapping spans
    starting from shortest.
    """
    if not entities:
        return []

    sorted_entities = sorted(entities, key=lambda x: (x[1] - x[0], x[0], x[1], x[2]))
    kept: list[tuple[int, int, str]] = []

    for start, end, entity_type in sorted_entities:
        # Discard exact duplicate spans/types.
        if any(start == ks and end == ke and entity_type == kt for ks, ke, kt in kept):
            continue

        # Keep partial overlaps, but discard outer spans that contain a kept span.
        is_outer = any(start <= ks and ke <= end for ks, ke, _ in kept)
        if is_outer:
            continue

        kept.append((start, end, entity_type))

    return sorted(kept, key=lambda x: (x[0], x[1], x[2]))


class NERDataset(Dataset):
    """SciERC JSONL dataset converted to token-level BIO labels."""

    def __init__(
        self,
        filepath: str,
        tokenizer,
        max_len: int = 256,
        max_samples: int = 0,
    ):
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("NERDataset requires a fast tokenizer (use_fast=True).")

        self.samples: list[dict[str, torch.Tensor]] = []

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                doc = json.loads(line)
                sentences = doc["sentences"]
                ner_by_sentence = doc["ner"]

                sentence_start = 0
                for sentence_index, words in enumerate(sentences):
                    sentence_end = sentence_start + len(words) - 1

                    entities_sentence_level: list[tuple[int, int, str]] = []
                    for doc_start, doc_end, entity_type in ner_by_sentence[sentence_index]:
                        if not (sentence_start <= doc_start <= sentence_end):
                            continue

                        mapped_type = SCIERC_TO_KG_TYPE.get(entity_type)
                        if mapped_type is None:
                            continue

                        sent_start = doc_start - sentence_start
                        sent_end = doc_end - sentence_start
                        if sent_start < 0 or sent_end >= len(words) or sent_start > sent_end:
                            continue

                        entities_sentence_level.append((sent_start, sent_end, mapped_type))

                    kept_entities = resolve_nested(entities_sentence_level)

                    tags = ["O"] * len(words)
                    for start, end, entity_type in kept_entities:
                        tags[start] = f"B-{entity_type}"
                        for i in range(start + 1, end + 1):
                            tags[i] = f"I-{entity_type}"

                    enc = tokenizer(
                        words,
                        is_split_into_words=True,
                        max_length=max_len,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )
                    word_ids = enc.word_ids(batch_index=0)

                    labels = []
                    prev_word_id = None
                    for word_id in word_ids:
                        if word_id is None:
                            labels.append(-100)
                        elif word_id != prev_word_id:
                            labels.append(NER_LABEL2ID[tags[word_id]])
                        else:
                            labels.append(-100)
                        prev_word_id = word_id

                    self.samples.append(
                        {
                            "input_ids": enc["input_ids"].squeeze(0),
                            "attention_mask": enc["attention_mask"].squeeze(0),
                            "labels": torch.tensor(labels, dtype=torch.long),
                        }
                    )

                    if max_samples > 0 and len(self.samples) >= max_samples:
                        return

                    sentence_start += len(words)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]
