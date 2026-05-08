"""Loader and normalizer for qiaojin/PubMedQA."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from coca_med.data.schema import MedicalQAExample, stable_example_id


PUBMEDQA_DATASET_ID = "qiaojin/PubMedQA"
PUBMEDQA_CHOICES = {"yes": "yes", "no": "no", "maybe": "maybe"}
PUBMEDQA_VALID_LABELS = frozenset(PUBMEDQA_CHOICES)


def _list_from_context(context: dict[str, Any], key: str) -> list[str]:
    values = context.get(key, []) if isinstance(context, dict) else []
    if values is None:
        return []
    if isinstance(values, list):
        return [str(value) for value in values]
    return [str(values)]


def flatten_pubmed_context(context: dict[str, Any] | None, *, max_chars: int | None = None) -> str:
    """Turn PubMedQA context sections into plain text for prompting."""

    context = context or {}
    contexts = _list_from_context(context, "contexts")
    labels = _list_from_context(context, "labels")

    sections: list[str] = []
    for idx, text in enumerate(contexts):
        label = labels[idx] if idx < len(labels) and labels[idx] else f"Context {idx + 1}"
        sections.append(f"{label}: {text}")

    joined = "\n".join(sections).strip()
    if max_chars is not None and len(joined) > max_chars:
        return joined[: max_chars - 3].rstrip() + "..."
    return joined


def normalize_pubmedqa_row(
    row: dict[str, Any],
    *,
    config: str,
    split: str = "train",
    index: int | None = None,
    max_context_chars: int | None = None,
) -> MedicalQAExample:
    """Normalize one PubMedQA row to the project schema."""

    raw_label = row.get("final_decision")
    gold_label = str(raw_label).strip().lower() if raw_label is not None else None
    if gold_label not in PUBMEDQA_VALID_LABELS:
        gold_label = None

    question = str(row.get("question") or "").strip()
    context = row.get("context") or {}
    metadata = {
        "pubid": row.get("pubid"),
        "config": config,
        "long_answer": row.get("long_answer"),
        "context_labels": _list_from_context(context, "labels"),
        "meshes": _list_from_context(context, "meshes"),
        "reasoning_required_pred": _list_from_context(context, "reasoning_required_pred"),
        "reasoning_free_pred": _list_from_context(context, "reasoning_free_pred"),
    }

    return MedicalQAExample(
        id=stable_example_id(f"pubmedqa/{config}", split, index, question),
        dataset=f"pubmedqa/{config}",
        question=question,
        context=flatten_pubmed_context(context, max_chars=max_context_chars),
        choices=dict(PUBMEDQA_CHOICES),
        gold_label=gold_label,
        gold_answer=gold_label,
        metadata=metadata,
    )


def iter_pubmedqa_examples(
    config: str = "pqa_labeled",
    *,
    split: str = "train",
    limit: int | None = None,
    dataset_id: str = PUBMEDQA_DATASET_ID,
    max_context_chars: int | None = None,
    include_unlabeled: bool = True,
) -> Iterable[MedicalQAExample]:
    from datasets import Dataset, load_dataset

    dataset: Dataset = load_dataset(dataset_id, config, split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    for index, row in enumerate(dataset):
        example = normalize_pubmedqa_row(
            row,
            config=config,
            split=split,
            index=index,
            max_context_chars=max_context_chars,
        )
        if example.gold_label is None and not include_unlabeled:
            continue
        yield example


def load_pubmedqa_examples(
    config: str = "pqa_labeled",
    *,
    split: str = "train",
    limit: int | None = None,
    dataset_id: str = PUBMEDQA_DATASET_ID,
    max_context_chars: int | None = None,
    include_unlabeled: bool = True,
) -> list[MedicalQAExample]:
    return list(
        iter_pubmedqa_examples(
            config=config,
            split=split,
            limit=limit,
            dataset_id=dataset_id,
            max_context_chars=max_context_chars,
            include_unlabeled=include_unlabeled,
        )
    )
