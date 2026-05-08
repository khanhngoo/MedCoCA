#!/usr/bin/env python
"""Evaluate a confidence-first CoCA model on prepared medical QA examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from coca_med.coca.parsing import is_answer_correct, parse_confidence_completion
from coca_med.coca.prompts import build_chat_messages, build_plain_prompt
from coca_med.data.io import read_examples_jsonl
from coca_med.data.schema import MedicalQAExample
from coca_med.eval.metrics import aggregate_metrics


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def torch_dtype(name: str | None) -> torch.dtype | None:
    if name is None:
        return None
    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(str(name).lower())


def render_prompt(tokenizer: Any, example: MedicalQAExample) -> str:
    messages = build_chat_messages(example, include_step_by_step=True)
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return build_plain_prompt(example, include_step_by_step=True) + "\n\nASSISTANT:"


def load_eval_model(config: dict[str, Any]):
    model_cfg = config["model"]
    model_path = Path(model_cfg["name_or_path"])
    name_or_path = str(model_path if model_path.exists() else model_cfg.get("fallback_base_model"))
    dtype = torch_dtype(model_cfg.get("torch_dtype"))
    quantization_config = None
    if model_cfg.get("load_in_4bit", False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype or torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        name_or_path,
        torch_dtype=dtype,
        quantization_config=quantization_config,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def token_to_confidence(text: str, tokenizer: Any) -> int | None:
    lower = text.lower()
    end_idx = lower.find("</confidence>")
    if end_idx < 0:
        return None
    confidence_prefix = text[: end_idx + len("</confidence>")]
    return len(tokenizer(confidence_prefix, add_special_tokens=False)["input_ids"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    examples = read_examples_jsonl(config["data"]["prepared_eval_path"])
    if args.limit is not None:
        examples = examples[: args.limit]

    model, tokenizer = load_eval_model(config)
    gen_cfg = config.get("generation", {})
    metric_cfg = config.get("metrics", {})
    predictions_path = Path(metric_cfg.get("predictions_path", "artifacts/eval/predictions.jsonl"))
    metrics_path = Path(metric_cfg.get("output_path", "artifacts/eval/metrics.json"))
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    confidences: list[float | None] = []
    correctness: list[int] = []
    ttc_values: list[int | None] = []

    with predictions_path.open("w", encoding="utf-8") as handle:
        for example in tqdm(examples):
            prompt = render_prompt(tokenizer, example)
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=int(config["data"].get("max_prompt_length", 2048)),
            ).to(model.device)
            with torch.no_grad():
                generation_kwargs = {
                    "max_new_tokens": int(config["data"].get("max_completion_length", 512)),
                    "do_sample": bool(gen_cfg.get("do_sample", False)),
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                }
                if generation_kwargs["do_sample"]:
                    generation_kwargs["temperature"] = float(gen_cfg.get("temperature", 1.0))
                generated = model.generate(
                    **encoded,
                    **generation_kwargs,
                )
            completion = tokenizer.decode(
                generated[0][encoded["input_ids"].shape[-1] :],
                skip_special_tokens=True,
            ).strip()
            parsed = parse_confidence_completion(completion)
            correct = int(is_answer_correct(example, completion))
            confidences.append(parsed.confidence)
            correctness.append(correct)
            ttc_values.append(token_to_confidence(completion, tokenizer))
            handle.write(
                json.dumps(
                    {
                        "id": example.id,
                        "dataset": example.dataset,
                        "gold_label": example.gold_label,
                        "completion": completion,
                        "confidence": parsed.confidence,
                        "valid_confidence": parsed.valid_confidence,
                        "correct": correct,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    metrics = aggregate_metrics(
        confidences=confidences,
        correctness=correctness,
        token_to_confidence=ttc_values,
        bins=int(metric_cfg.get("ece_bins", 10)),
    )
    metrics_path.write_text(json.dumps(metrics.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps(metrics.to_dict(), indent=2))


if __name__ == "__main__":
    main()
