"""JSONL helpers for normalized examples."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from coca_med.data.schema import MedicalQAExample


def write_examples_jsonl(examples: Iterable[MedicalQAExample], path: str | Path) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_examples_jsonl(path: str | Path) -> list[MedicalQAExample]:
    examples: list[MedicalQAExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                examples.append(MedicalQAExample.from_dict(json.loads(line)))
    return examples
