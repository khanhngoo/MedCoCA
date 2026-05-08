"""Metrics for confidence-first QA."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class EvaluationMetrics:
    accuracy: float
    auroc: float | None
    ece: float
    brier: float
    confidence_success_rate: float
    token_to_confidence: float | None
    count: int

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "accuracy": self.accuracy,
            "auroc": self.auroc,
            "ece": self.ece,
            "brier": self.brier,
            "confidence_success_rate": self.confidence_success_rate,
            "token_to_confidence": self.token_to_confidence,
            "count": self.count,
        }


def expected_calibration_error(
    confidences: list[float],
    correctness: list[int | bool],
    *,
    bins: int = 10,
) -> float:
    if not confidences:
        return 0.0
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correctness, dtype=np.float64)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        if upper == 1.0:
            mask = (conf >= lower) & (conf <= upper)
        else:
            mask = (conf >= lower) & (conf < upper)
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(float(corr[mask].mean()) - float(conf[mask].mean()))
    return ece


def brier_score(confidences: list[float], correctness: list[int | bool]) -> float:
    if not confidences:
        return 0.0
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correctness, dtype=np.float64)
    return float(np.mean((conf - corr) ** 2))


def safe_auroc(confidences: list[float], correctness: list[int | bool]) -> float | None:
    labels = np.asarray(correctness, dtype=np.int32)
    scores = np.asarray(confidences, dtype=np.float64)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return None

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    positive_rank_sum = float(ranks[labels == 1].sum())
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def aggregate_metrics(
    *,
    confidences: list[float | None],
    correctness: list[int | bool],
    token_to_confidence: list[int | None] | None = None,
    bins: int = 10,
) -> EvaluationMetrics:
    valid_pairs = [
        (confidence, int(correct))
        for confidence, correct in zip(confidences, correctness, strict=True)
        if confidence is not None
    ]
    valid_confidences = [float(confidence) for confidence, _ in valid_pairs]
    valid_correctness = [correct for _, correct in valid_pairs]
    accuracy = float(np.mean(correctness)) if correctness else 0.0
    success_rate = len(valid_pairs) / max(1, len(confidences))

    valid_ttc = []
    if token_to_confidence is not None:
        valid_ttc = [value for value in token_to_confidence if value is not None]

    return EvaluationMetrics(
        accuracy=accuracy,
        auroc=safe_auroc(valid_confidences, valid_correctness) if valid_pairs else None,
        ece=expected_calibration_error(valid_confidences, valid_correctness, bins=bins),
        brier=brier_score(valid_confidences, valid_correctness),
        confidence_success_rate=success_rate,
        token_to_confidence=float(np.mean(valid_ttc)) if valid_ttc else None,
        count=len(correctness),
    )
