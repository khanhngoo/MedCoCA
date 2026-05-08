#!/usr/bin/env python
"""Multi-GPU CoCA training with grouped rollouts and segmented GRPO loss."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

import torch
import yaml
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

from coca_med.coca.loss import segmented_grpo_loss
from coca_med.coca.prompts import build_chat_messages, build_plain_prompt
from coca_med.coca.rewards import compute_coca_group_rewards, rewards_to_advantages
from coca_med.coca.segments import build_segment_mask
from coca_med.data.io import read_examples_jsonl
from coca_med.data.schema import MedicalQAExample


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


def render_prompt(tokenizer: Any, example: MedicalQAExample, *, include_step_by_step: bool) -> str:
    messages = build_chat_messages(example, include_step_by_step=include_step_by_step)
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return build_plain_prompt(example, include_step_by_step=include_step_by_step) + "\n\nASSISTANT:"


def load_model_and_tokenizer(config: dict[str, Any], accelerator: Accelerator):
    model_cfg = config["model"]
    dtype = torch_dtype(model_cfg.get("torch_dtype"))
    quantization_config = None
    if model_cfg.get("load_in_4bit", False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype or torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name_or_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name_or_path"],
        torch_dtype=dtype,
        quantization_config=quantization_config,
        attn_implementation=model_cfg.get("attn_implementation"),
        trust_remote_code=True,
        device_map={"": accelerator.local_process_index} if quantization_config else None,
    )
    model.config.use_cache = bool(config.get("training", {}).get("use_cache", False))

    lora_cfg = config.get("lora", {})
    if lora_cfg.get("enabled", False):
        if quantization_config is not None:
            model = prepare_model_for_kbit_training(model)
        model = get_peft_model(
            model,
            LoraConfig(
                r=int(lora_cfg.get("r", 16)),
                lora_alpha=int(lora_cfg.get("alpha", 32)),
                lora_dropout=float(lora_cfg.get("dropout", 0.05)),
                target_modules=list(lora_cfg.get("target_modules", [])),
                task_type="CAUSAL_LM",
            ),
        )

    return model, tokenizer


def completion_logprobs(model, full_ids: torch.Tensor, prompt_length: int) -> torch.Tensor:
    inputs = full_ids[:-1].unsqueeze(0)
    labels = full_ids[1:].unsqueeze(0)
    logits = model(input_ids=inputs).logits
    logprobs = torch.log_softmax(logits, dim=-1)
    gathered = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1)).squeeze(0).squeeze(-1)
    return gathered[prompt_length - 1 :]


def masks_for_completion(
    completion_text: str,
    tokenizer: Any,
    sequence_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = build_segment_mask(completion_text, tokenizer)
    confidence = torch.zeros(sequence_length, dtype=torch.bool, device=device)
    answer = torch.zeros(sequence_length, dtype=torch.bool, device=device)
    for idx in mask.confidence_token_indices:
        if idx < sequence_length:
            confidence[idx] = True
    for idx in mask.answer_token_indices:
        if idx < sequence_length:
            answer[idx] = True
    if not answer.any():
        answer[:] = ~confidence
    return confidence, answer


def train_one_group(
    *,
    model,
    tokenizer,
    optimizer,
    accelerator: Accelerator,
    example: MedicalQAExample,
    config: dict[str, Any],
) -> dict[str, float]:
    data_cfg = config["data"]
    coca_cfg = config["coca"]
    train_cfg = config["training"]

    prompt = render_prompt(
        tokenizer,
        example,
        include_step_by_step=bool(coca_cfg.get("include_step_by_step", True)),
    )
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=int(data_cfg.get("max_prompt_length", 2048)),
    ).to(accelerator.device)
    prompt_length = encoded["input_ids"].shape[-1]

    with torch.no_grad():
        generated = accelerator.unwrap_model(model).generate(
            **encoded,
            max_new_tokens=int(data_cfg.get("max_completion_length", 512)),
            num_return_sequences=int(coca_cfg.get("group_size", 8)),
            do_sample=bool(train_cfg.get("do_sample", True)),
            temperature=float(train_cfg.get("temperature", 1.0)),
            top_p=float(train_cfg.get("top_p", 1.0)),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    completions = [
        tokenizer.decode(row[prompt_length:], skip_special_tokens=True).strip()
        for row in generated
    ]
    rewards = compute_coca_group_rewards(
        example,
        completions,
        invalid_confidence_reward=float(coca_cfg.get("invalid_confidence_reward", -1.0)),
    )
    confidence_advantages, answer_advantages = rewards_to_advantages(rewards)

    old_logprobs = []
    with torch.no_grad():
        for row in generated:
            old_logprobs.append(completion_logprobs(model, row, prompt_length).detach())

    total_loss = torch.zeros((), device=accelerator.device)
    for row, completion, old_lp, conf_adv, ans_adv in zip(
        generated,
        completions,
        old_logprobs,
        confidence_advantages,
        answer_advantages,
        strict=True,
    ):
        new_lp = completion_logprobs(model, row, prompt_length)
        length = min(new_lp.shape[0], old_lp.shape[0])
        new_lp = new_lp[:length].unsqueeze(0)
        old_lp = old_lp[:length].unsqueeze(0)
        confidence_mask, answer_mask = masks_for_completion(
            completion,
            tokenizer,
            length,
            accelerator.device,
        )
        loss = segmented_grpo_loss(
            new_lp,
            old_lp,
            confidence_mask.unsqueeze(0),
            answer_mask.unsqueeze(0),
            torch.tensor([conf_adv], dtype=torch.float32, device=accelerator.device),
            torch.tensor([ans_adv], dtype=torch.float32, device=accelerator.device),
            clip_range=float(coca_cfg.get("clip_range", 0.2)),
        )
        total_loss = total_loss + loss / max(1, len(completions))

    accelerator.backward(total_loss)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    return {
        "loss": float(total_loss.detach().cpu()),
        "answer_reward": float(sum(r.answer_reward for r in rewards) / max(1, len(rewards))),
        "confidence_reward": float(sum(r.confidence_reward for r in rewards) / max(1, len(rewards))),
        "gesr": float(rewards[0].gesr if rewards else 0.0),
        "valid_confidence_rate": float(
            sum(1 for reward in rewards if reward.valid_confidence) / max(1, len(rewards))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_qwen25_7b_lora.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config["training"]
    set_seed(int(train_cfg.get("seed", 42)))
    accelerator = Accelerator(
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 1))
    )

    examples = read_examples_jsonl(config["data"]["prepared_train_path"])
    if not examples:
        raise ValueError("No training examples found. Run scripts/prepare_data.py first.")

    model, tokenizer = load_model_and_tokenizer(config, accelerator)
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-6)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    model.train()

    output_dir = Path(train_cfg["output_dir"])
    max_steps = int(train_cfg.get("max_steps", 1000))
    iterator = itertools.cycle(examples)
    progress = tqdm(range(max_steps), disable=not accelerator.is_local_main_process)

    for step in progress:
        metrics = train_one_group(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            accelerator=accelerator,
            example=next(iterator),
            config=config,
        )
        if step % int(train_cfg.get("logging_steps", 10)) == 0:
            progress.set_postfix(metrics)
        if accelerator.is_main_process and (step + 1) % int(train_cfg.get("save_steps", 100)) == 0:
            save_dir = output_dir / f"step-{step + 1}"
            accelerator.unwrap_model(model).save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
