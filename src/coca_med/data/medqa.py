"""Loader and normalizer for GBaker/MedQA-USMLE-4-options."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from coca_med.data.schema import MedicalQAExample, stable_example_id


MEDQA_DATASET_ID = "GBaker/MedQA-USMLE-4-options"
MEDQA_OPTION_ORDER = ("A", "B", "C", "D")


def normalize_medqa_row(
    row: dict[str, Any],
    *,
    split: str = "train",
    index: int | None = None,
) -> MedicalQAExample:
    """Normalize one MedQA row to the project schema."""

    raw_options = row.get("options") or {}
    choices = {
        label: str(raw_options[label])
        for label in MEDQA_OPTION_ORDER
        if label in raw_options and raw_options[label] is not None
    }
    gold_label = str(row.get("answer_idx") or "").strip().upper() or None
    gold_answer = choices.get(gold_label or "", row.get("answer"))
    metadata = {
        "meta_info": row.get("meta_info"),
        "metamap_phrases": list(row.get("metamap_phrases") or []),
        "source_answer": row.get("answer"),
    }

    question = str(row.get("question") or "").strip()
    return MedicalQAExample(
        id=stable_example_id("medqa", split, index, question),
        dataset="medqa",
        question=question,
        context="",
        choices=choices,
        gold_label=gold_label,
        gold_answer=str(gold_answer).strip() if gold_answer is not None else None,
        metadata=metadata,
    )


def iter_medqa_examples(
    split: str = "train",
    *,
    limit: int | None = None,
    dataset_id: str = MEDQA_DATASET_ID,
) -> Iterable[MedicalQAExample]:
    from datasets import Dataset, load_dataset

    dataset: Dataset = load_dataset(dataset_id, split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    for index, row in enumerate(dataset):
        yield normalize_medqa_row(row, split=split, index=index)


def load_medqa_examples(
    split: str = "train",
    *,
    limit: int | None = None,
    dataset_id: str = MEDQA_DATASET_ID,
) -> list[MedicalQAExample]:
    return list(iter_medqa_examples(split=split, limit=limit, dataset_id=dataset_id))
