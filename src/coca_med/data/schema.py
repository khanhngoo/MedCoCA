"""Shared schema for medical QA examples."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ChoiceMap = dict[str, str]


@dataclass(slots=True)
class MedicalQAExample:
    """A normalized QA example used by prompts, rewards, and evaluation."""

    id: str
    dataset: str
    question: str
    choices: ChoiceMap
    gold_label: str | None = None
    gold_answer: str | None = None
    context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    features: dict[str, float | int | bool | str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "dataset": self.dataset,
            "question": self.question,
            "context": self.context,
            "choices": self.choices,
            "gold_label": self.gold_label,
            "gold_answer": self.gold_answer,
            "metadata": self.metadata,
            "features": self.features,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MedicalQAExample":
        return cls(
            id=str(payload["id"]),
            dataset=str(payload["dataset"]),
            question=str(payload["question"]),
            context=str(payload.get("context") or ""),
            choices={str(k): str(v) for k, v in (payload.get("choices") or {}).items()},
            gold_label=payload.get("gold_label"),
            gold_answer=payload.get("gold_answer"),
            metadata=dict(payload.get("metadata") or {}),
            features=dict(payload.get("features") or {}),
        )


def stable_example_id(dataset: str, split: str, index: int | None, fallback: str) -> str:
    """Create readable IDs that are deterministic when the source index is known."""

    if index is not None:
        return f"{dataset}:{split}:{index}"
    return f"{dataset}:{split}:{abs(hash(fallback))}"
