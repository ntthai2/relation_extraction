"""
End-to-end inference pipeline: raw text -> NER -> entity pairs -> RE -> triples.

Usage:
    from inference import ScienceIEPipeline
    pipeline = ScienceIEPipeline()
    triples = pipeline.extract(text)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import nltk
import torch
from transformers import AutoTokenizer

from config import ENTITY_MARKERS, HIDDEN_SIZE, ID2LABEL, LABEL2ID, MODEL_NAME
from ner_dataset import NER_ID2LABEL, NER_LABEL2ID, NER_LABELS, NUM_NER_LABELS
from ner_model import SciBERTNER
from re_model import SciBERTRelationClassifier

# Safe default for sentence tokenization resources.
nltk.download("punkt_tab", quiet=True)


def normalize_entity(text: str) -> str:
    """
    Basic normalization:
    1. Replace newlines and multiple whitespace with a single space
    2. Strip leading and trailing punctuation: . , ; : ! ? ) ] }
    3. Strip leading punctuation: ( [ {
    4. Lowercase
    5. Strip leading/trailing whitespace
    6. Remove leading determiners: "the ", "a ", "an "
    7. Collapse multiple spaces to single space
    """
    normalized = " ".join(text.split())
    normalized = normalized.strip(".,;:!?)]}")
    normalized = normalized.lstrip("([{")
    normalized = normalized.strip().lower()
    for det in ("the ", "a ", "an "):
        if normalized.startswith(det):
            normalized = normalized[len(det) :]
            break
    return " ".join(normalized.split())


def decode_bio(word_labels: list[str], words: list[str]) -> list[dict[str, Any]]:
    """
    Convert a list of BIO labels (one per word) to entity span dicts.
    Handles B- and I- prefixes correctly.
    Ignores O labels.
    Returns list of {"start": int, "end": int, "type": str, "text": str}
    """
    entities: list[dict[str, Any]] = []
    start_idx: int | None = None
    current_type: str | None = None

    def close_entity(end_idx: int) -> None:
        nonlocal start_idx, current_type
        if start_idx is None or current_type is None:
            return
        entities.append(
            {
                "start": start_idx,
                "end": end_idx,
                "type": current_type,
                "text": " ".join(words[start_idx : end_idx + 1]),
            }
        )
        start_idx = None
        current_type = None

    for i, label in enumerate(word_labels):
        if label == "O":
            close_entity(i - 1)
            continue

        if "-" not in label:
            close_entity(i - 1)
            continue

        prefix, ent_type = label.split("-", 1)

        if prefix == "B":
            close_entity(i - 1)
            start_idx = i
            current_type = ent_type
            continue

        if prefix == "I":
            # If malformed BIO sequence starts with I-*, start a new span.
            if start_idx is None or current_type != ent_type:
                close_entity(i - 1)
                start_idx = i
                current_type = ent_type
            continue

        close_entity(i - 1)

    close_entity(len(word_labels) - 1)
    return entities


def format_re_input(
    words: list[str],
    e1_start: int,
    e1_end: int,
    e2_start: int,
    e2_end: int,
) -> str:
    """
    Insert [E1]/[/E1] around entity_a and [E2]/[/E2] around entity_b.
    Returns the marked sentence as a string.
    Handles the case where e1 comes after e2 in the sentence correctly
    by inserting markers from right to left to preserve indices.
    """
    marked_words = list(words)
    insertions = [
        (e1_start, "[E1]"),
        (e1_end + 1, "[/E1]"),
        (e2_start, "[E2]"),
        (e2_end + 1, "[/E2]"),
    ]

    # Sort descending by position so earlier indices are unaffected.
    for pos, token in sorted(insertions, key=lambda x: x[0], reverse=True):
        safe_pos = min(max(pos, 0), len(marked_words))
        marked_words.insert(safe_pos, token)

    return " ".join(marked_words)


class ScienceIEPipeline:
    def __init__(
        self,
        ner_checkpoint: str = "runs/ner/seed_42/best_ner_model.pt",
        re_checkpoint: str = "best_re_model.pt",
        device: str = "auto",  # "auto", "cuda", "cpu"
        confidence_threshold: float = 0.5,
        max_len: int = 256,
        batch_size: int = 32,  # for RE batching
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ENTITY_MARKERS})

        self.ner_model = SciBERTNER(num_labels=NUM_NER_LABELS)
        self.ner_model.bert.resize_token_embeddings(len(self.tokenizer))
        ner_ckpt = torch.load(ner_checkpoint, map_location=self.device)
        self.ner_model.load_state_dict(ner_ckpt["model_state_dict"])
        self.ner_model.to(self.device)
        self.ner_model.eval()

        self.re_model = SciBERTRelationClassifier(num_labels=len(LABEL2ID), pooling="e1e2_concat")
        self.re_model.bert.resize_token_embeddings(len(self.tokenizer))
        re_ckpt = torch.load(re_checkpoint, map_location=self.device)
        self.re_model.load_state_dict(re_ckpt)
        self.re_model.to(self.device)
        self.re_model.eval()

        self.confidence_threshold = confidence_threshold
        self.max_len = max_len
        self.batch_size = batch_size

        # Explicitly reference imported label maps to enforce non-hardcoded usage.
        self._ner_label_space = (NER_LABELS, NER_LABEL2ID)
        self._re_label_space = (LABEL2ID, ID2LABEL)
        self._hidden_size = HIDDEN_SIZE

    def extract(self, text: str) -> list[dict[str, Any]]:
        if not text or not text.strip():
            return []

        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
            sentences = nltk.sent_tokenize(text)

        sentence_records: list[dict[str, Any]] = []

        with torch.no_grad():
            for sentence_idx, sentence in enumerate(sentences):
                words = sentence.split()
                if not words:
                    continue

                enc = self.tokenizer(
                    words,
                    is_split_into_words=True,
                    max_length=self.max_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)

                logits = self.ner_model(input_ids=input_ids, attention_mask=attention_mask)
                preds = logits.argmax(dim=-1).squeeze(0).detach().cpu().tolist()

                word_ids = enc.word_ids(batch_index=0)
                word_labels = ["O"] * len(words)
                seen_words: set[int] = set()

                for tok_idx, word_id in enumerate(word_ids):
                    if word_id is None or word_id in seen_words:
                        continue
                    if word_id >= len(words):
                        continue
                    label_id = int(preds[tok_idx])
                    word_labels[word_id] = NER_ID2LABEL.get(label_id, "O")
                    seen_words.add(word_id)

                entities = decode_bio(word_labels, words)
                for entity in entities:
                    entity["sentence_idx"] = sentence_idx

                sentence_records.append(
                    {
                        "sentence_idx": sentence_idx,
                        "sentence_text": sentence,
                        "words": words,
                        "entities": entities,
                    }
                )

        pair_examples: list[dict[str, Any]] = []
        for rec in sentence_records:
            entities = rec["entities"]
            words = rec["words"]
            for i, entity_a in enumerate(entities):
                for j, entity_b in enumerate(entities):
                    if i == j:
                        continue
                    if entity_a["start"] == entity_b["start"] and entity_a["end"] == entity_b["end"]:
                        continue

                    marked_text = format_re_input(
                        words=words,
                        e1_start=entity_a["start"],
                        e1_end=entity_a["end"],
                        e2_start=entity_b["start"],
                        e2_end=entity_b["end"],
                    )
                    pair_examples.append(
                        {
                            "marked_text": marked_text,
                            "entity_a": entity_a,
                            "entity_b": entity_b,
                            "source_sentence": rec["sentence_text"],
                        }
                    )

        if not pair_examples:
            return []

        triples: list[dict[str, Any]] = []

        e1_id = self.tokenizer.convert_tokens_to_ids("[E1]")
        e2_id = self.tokenizer.convert_tokens_to_ids("[E2]")

        with torch.no_grad():
            for start in range(0, len(pair_examples), self.batch_size):
                batch = pair_examples[start : start + self.batch_size]
                batch_texts = [item["marked_text"] for item in batch]

                enc = self.tokenizer(
                    batch_texts,
                    max_length=self.max_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )

                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)

                e1_pos_list: list[int] = []
                e2_pos_list: list[int] = []

                input_ids_cpu = enc["input_ids"]
                for i in range(input_ids_cpu.size(0)):
                    ids = input_ids_cpu[i]
                    e1_matches = (ids == e1_id).nonzero(as_tuple=True)[0]
                    e2_matches = (ids == e2_id).nonzero(as_tuple=True)[0]
                    e1_pos_list.append(int(e1_matches[0].item()) if len(e1_matches) > 0 else 0)
                    e2_pos_list.append(int(e2_matches[0].item()) if len(e2_matches) > 0 else 0)

                e1_pos = torch.tensor(e1_pos_list, dtype=torch.long, device=self.device)
                e2_pos = torch.tensor(e2_pos_list, dtype=torch.long, device=self.device)

                logits = self.re_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    e1_pos=e1_pos,
                    e2_pos=e2_pos,
                )

                probs = torch.softmax(logits, dim=-1)
                confs, pred_ids = probs.max(dim=-1)

                for i, item in enumerate(batch):
                    confidence = float(confs[i].item())
                    if confidence < self.confidence_threshold:
                        continue

                    pred_id = int(pred_ids[i].item())
                    relation = ID2LABEL[pred_id]

                    entity_a = item["entity_a"]
                    entity_b = item["entity_b"]

                    triples.append(
                        {
                            "subject": normalize_entity(entity_a["text"]),
                            "subject_type": entity_a["type"],
                            "relation": relation,
                            "object": normalize_entity(entity_b["text"]),
                            "object_type": entity_b["type"],
                            "confidence": confidence,
                            "source_sentence": item["source_sentence"],
                        }
                    )

        triples.sort(key=lambda x: x["confidence"], reverse=True)
        return triples


def extract_from_file(
    pipeline: ScienceIEPipeline,
    input_path: str,
    output_path: str,
) -> list[dict[str, Any]]:
    """
    Run pipeline on a list of abstracts from a JSON file.
    Each triple in output includes "paper_id" from the source document.
    Writes results to output_path as JSON.
    Returns full list of triples.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    all_triples: list[dict[str, Any]] = []
    for doc in docs:
        text = doc.get("text", "")
        paper_id = doc.get("paper_id", "")
        triples = pipeline.extract(text)
        for triple in triples:
            triple_with_id = dict(triple)
            triple_with_id["paper_id"] = paper_id
            all_triples.append(triple_with_id)

    output_parent = Path(output_path).parent
    output_parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_triples, f, ensure_ascii=False, indent=2)

    return all_triples


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference pipeline: NER -> RE -> triples")
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    parser.add_argument(
        "--ner-checkpoint",
        default="runs/ner/seed_42/best_ner_model.pt",
        help="Path to NER checkpoint",
    )
    parser.add_argument(
        "--re-checkpoint",
        default="best_re_model.pt",
        help="Path to RE checkpoint",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Filter out predictions below this confidence",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="RE inference batch size",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Inference device",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    pipeline = ScienceIEPipeline(
        ner_checkpoint=args.ner_checkpoint,
        re_checkpoint=args.re_checkpoint,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        batch_size=args.batch_size,
    )
    extract_from_file(
        pipeline=pipeline,
        input_path=args.input,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
