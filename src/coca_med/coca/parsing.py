"""Parsing confidence-first model outputs and medical answers."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass

from coca_med.data.schema import ChoiceMap, MedicalQAExample


CONFIDENCE_RE = re.compile(
    r"<confidence>\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*</confidence>",
    flags=re.IGNORECASE | re.DOTALL,
)
FINAL_ANSWER_RE = re.compile(
    r"(?:final\s+(?:answer|decision)|answer|decision)\s*(?:is|:)?\s*([A-D]|yes|no|maybe)\b",
    flags=re.IGNORECASE,
)


@dataclass(slots=True)
class ParsedCompletion:
    raw_text: str
    confidence: float | None
    answer_text: str
    valid_confidence: bool


def parse_confidence_completion(text: str) -> ParsedCompletion:
    match = CONFIDENCE_RE.search(text or "")
    if not match:
        return ParsedCompletion(
            raw_text=text or "",
            confidence=None,
            answer_text=(text or "").strip(),
            valid_confidence=False,
        )

    try:
        confidence = float(match.group(1))
    except ValueError:
        confidence = None

    valid = confidence is not None and 0.0 <= confidence <= 1.0
    answer_text = ((text or "")[: match.start()] + (text or "")[match.end() :]).strip()
    return ParsedCompletion(
        raw_text=text or "",
        confidence=confidence if valid else None,
        answer_text=answer_text,
        valid_confidence=valid,
    )


def normalize_text(text: str | None) -> str:
    text = (text or "").lower().strip()
    translator = str.maketrans("", "", string.punctuation)
    return " ".join(text.translate(translator).split())


def extract_medqa_prediction(answer_text: str, choices: ChoiceMap) -> str | None:
    text = answer_text or ""
    final_match = FINAL_ANSWER_RE.search(text)
    if final_match:
        candidate = final_match.group(1).upper()
        if candidate in choices:
            return candidate

    letter_match = re.search(r"(?:^|\b|\()([A-D])(?:\)|\.|:|\b)", text, flags=re.IGNORECASE)
    if letter_match:
        candidate = letter_match.group(1).upper()
        if candidate in choices:
            return candidate

    normalized_answer = normalize_text(text)
    for label, option_text in choices.items():
        normalized_option = normalize_text(option_text)
        if normalized_option and normalized_option in normalized_answer:
            return label
    return None


def extract_pubmedqa_prediction(answer_text: str) -> str | None:
    text = answer_text or ""
    final_match = FINAL_ANSWER_RE.search(text)
    if final_match:
        candidate = final_match.group(1).lower()
        if candidate in {"yes", "no", "maybe"}:
            return candidate

    normalized = normalize_text(text)
    for label in ("yes", "no", "maybe"):
        if re.search(rf"\b{label}\b", normalized):
            return label
    return None


def extract_prediction(example: MedicalQAExample, completion_text: str) -> str | None:
    parsed = parse_confidence_completion(completion_text)
    if example.dataset.startswith("medqa"):
        return extract_medqa_prediction(parsed.answer_text, example.choices)
    if example.dataset.startswith("pubmedqa"):
        return extract_pubmedqa_prediction(parsed.answer_text)
    return parsed.answer_text.strip() or None


def is_answer_correct(example: MedicalQAExample, completion_text: str) -> bool:
    if example.gold_label is None:
        return False
    prediction = extract_prediction(example, completion_text)
    if prediction is None:
        return False
    return normalize_text(prediction) == normalize_text(example.gold_label)
