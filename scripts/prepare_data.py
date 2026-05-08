#!/usr/bin/env python
"""Prepare normalized medical QA examples for CoCA training and evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from coca_med.data.features import add_features_to_examples
from coca_med.data.io import write_examples_jsonl
from coca_med.data.medqa import load_medqa_examples
from coca_med.data.pubmedqa import load_pubmedqa_examples


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets.yaml")
    parser.add_argument("--output-dir", default="artifacts/data")
    parser.add_argument("--limit-per-split", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    preprocessing = config.get("preprocessing", {})
    max_context_chars = preprocessing.get("max_context_chars")
    add_features = bool(preprocessing.get("add_engineered_features", True))
    datasets_config = config["datasets"]

    train_examples = []
    eval_examples = []

    medqa_cfg = datasets_config["medqa"]
    if medqa_cfg.get("include_in_training", True):
        train_examples.extend(
            load_medqa_examples(
                split=medqa_cfg.get("train_split", "train"),
                limit=args.limit_per_split,
                dataset_id=medqa_cfg.get("dataset_id", "GBaker/MedQA-USMLE-4-options"),
            )
        )
    eval_examples.extend(
        load_medqa_examples(
            split=medqa_cfg.get("eval_split", "test"),
            limit=args.limit_per_split,
            dataset_id=medqa_cfg.get("dataset_id", "GBaker/MedQA-USMLE-4-options"),
        )
    )

    pubmed_cfg = datasets_config["pubmedqa"]
    for pubmed_config in pubmed_cfg.get("train_configs", []):
        train_examples.extend(
            load_pubmedqa_examples(
                config=pubmed_config,
                split=pubmed_cfg.get("split", "train"),
                limit=args.limit_per_split,
                dataset_id=pubmed_cfg.get("dataset_id", "qiaojin/PubMedQA"),
                max_context_chars=max_context_chars,
                include_unlabeled=bool(pubmed_cfg.get("include_unlabeled_for_training", False)),
            )
        )

    for pubmed_config in pubmed_cfg.get("eval_configs", []):
        eval_examples.extend(
            load_pubmedqa_examples(
                config=pubmed_config,
                split=pubmed_cfg.get("split", "train"),
                limit=args.limit_per_split,
                dataset_id=pubmed_cfg.get("dataset_id", "qiaojin/PubMedQA"),
                max_context_chars=max_context_chars,
                include_unlabeled=False,
            )
        )

    if add_features:
        train_examples = add_features_to_examples(train_examples)
        eval_examples = add_features_to_examples(eval_examples)

    output_dir = Path(args.output_dir)
    train_count = write_examples_jsonl(train_examples, output_dir / "train.jsonl")
    eval_count = write_examples_jsonl(eval_examples, output_dir / "eval.jsonl")
    print(f"Wrote {train_count} train examples and {eval_count} eval examples to {output_dir}")


if __name__ == "__main__":
    main()
