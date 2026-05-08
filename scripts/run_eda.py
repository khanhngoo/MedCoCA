#!/usr/bin/env python
"""Export lightweight EDA summaries for MedQA and PubMedQA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset


def describe_numeric(frame: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    existing = [column for column in columns if column in frame.columns]
    if not existing:
        return {}
    return json.loads(frame[existing].describe().to_json())


def medqa_summary() -> dict[str, Any]:
    ds = load_dataset("GBaker/MedQA-USMLE-4-options")
    frames = []
    for split, split_ds in ds.items():
        frame = split_ds.to_pandas()
        frame["split"] = split
        frame["question_words"] = frame["question"].str.split().str.len()
        frame["num_metamap_phrases"] = frame["metamap_phrases"].apply(
            lambda value: len(value) if isinstance(value, list) else 0
        )
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    return {
        "split_sizes": {split: len(split_ds) for split, split_ds in ds.items()},
        "answer_distribution": data.groupby(["split", "answer_idx"]).size().to_dict(),
        "meta_info_top": data["meta_info"].value_counts(dropna=False).head(25).to_dict(),
        "lengths": describe_numeric(data, ["question_words", "num_metamap_phrases"]),
    }


def pubmedqa_summary() -> dict[str, Any]:
    configs = ["pqa_artificial", "pqa_labeled", "pqa_unlabeled"]
    frames = []
    sizes = {}
    for config in configs:
        split_ds = load_dataset("qiaojin/PubMedQA", config, split="train")
        sizes[config] = len(split_ds)
        frame = split_ds.to_pandas()
        frame["config"] = config
        frame["question_words"] = frame["question"].str.split().str.len()
        frame["context_count"] = frame["context"].apply(
            lambda context: len(context.get("contexts", [])) if isinstance(context, dict) else 0
        )
        frame["context_words"] = frame["context"].apply(
            lambda context: sum(
                len(text.split()) for text in context.get("contexts", [])
            )
            if isinstance(context, dict)
            else 0
        )
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    return {
        "config_sizes": sizes,
        "decision_distribution": data.groupby(["config", "final_decision"]).size().to_dict(),
        "missing_decisions": data["final_decision"].isna().groupby(data["config"]).sum().to_dict(),
        "lengths": describe_numeric(data, ["question_words", "context_count", "context_words"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/eda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {"medqa": medqa_summary(), "pubmedqa": pubmedqa_summary()}
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
