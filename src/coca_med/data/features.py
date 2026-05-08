"""Feature engineering for normalized medical QA examples."""

from __future__ import annotations

import math
import re
from dataclasses import replace
from typing import Any

from coca_med.data.schema import MedicalQAExample


WORD_RE = re.compile(r"\b\w+\b")


def word_count(text: str | None) -> int:
    return len(WORD_RE.findall(text or ""))


def char_count(text: str | None) -> int:
    return len(text or "")


def _metadata_sequence_len(metadata: dict[str, Any], key: str) -> int:
    value = metadata.get(key)
    if isinstance(value, list):
        return len(value)
    return 0


def estimate_difficulty(features: dict[str, float | int | bool | str]) -> float:
    """A transparent heuristic used for analysis and curriculum experiments."""

    question_words = float(features.get("question_words", 0))
    context_words = float(features.get("context_words", 0))
    num_choices = float(features.get("num_choices", 0))
    biomedical_terms = float(
        features.get("num_metamap_phrases", 0) or features.get("num_mesh_terms", 0) or 0
    )
    return question_words + math.sqrt(context_words) + 2.0 * num_choices + 0.5 * biomedical_terms


def build_features(example: MedicalQAExample) -> dict[str, float | int | bool | str]:
    metadata = example.metadata
    features: dict[str, float | int | bool | str] = {
        "dataset": example.dataset,
        "question_chars": char_count(example.question),
        "question_words": word_count(example.question),
        "context_chars": char_count(example.context),
        "context_words": word_count(example.context),
        "num_choices": len(example.choices),
        "has_context": bool(example.context),
        "has_gold_label": example.gold_label is not None,
        "answer_label": example.gold_label or "",
        "num_metamap_phrases": _metadata_sequence_len(metadata, "metamap_phrases"),
        "num_mesh_terms": _metadata_sequence_len(metadata, "meshes"),
        "has_long_answer": bool(metadata.get("long_answer")),
    }
    features["difficulty_proxy"] = estimate_difficulty(features)
    return features


def add_engineered_features(example: MedicalQAExample) -> MedicalQAExample:
    merged = dict(example.features)
    merged.update(build_features(example))
    return replace(example, features=merged)


def add_features_to_examples(examples: list[MedicalQAExample]) -> list[MedicalQAExample]:
    return [add_engineered_features(example) for example in examples]
