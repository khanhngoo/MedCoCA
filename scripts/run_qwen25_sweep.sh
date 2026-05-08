#!/usr/bin/env bash
set -euo pipefail

python scripts/prepare_data.py --config configs/datasets.yaml --output-dir artifacts/data

for size in 1_5b 3b 7b; do
  accelerate launch scripts/train_coca_grpo.py --config "configs/train_qwen25_${size}_lora.yaml"
  accelerate launch scripts/evaluate_coca.py --config "configs/eval_qwen25_${size}.yaml"
done
