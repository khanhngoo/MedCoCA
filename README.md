# CoCA Medical QA

This project adapts **CoCA: Co-optimized Confidence and Answers** to medical
question-answering datasets from Hugging Face:

- `GBaker/MedQA-USMLE-4-options`
- `qiaojin/PubMedQA` with `pqa_artificial`, `pqa_labeled`, and `pqa_unlabeled`

CoCA is a confidence-first training setup. The model must produce confidence
before the answer:

```text
<confidence>0.73</confidence> answer text here
```

During grouped rollouts, answer correctness produces the answer reward, while
the group empirical success rate (GESR) becomes the dynamic target for the
confidence reward.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For multi-GPU training, configure Accelerate once:

```bash
accelerate config
```

## Notebook Workflow

Start with the notebooks:

```bash
jupyter lab notebooks
```

- `notebooks/01_eda_medqa.ipynb`
- `notebooks/02_eda_pubmedqa.ipynb`
- `notebooks/03_feature_engineering.ipynb`

The notebook logic is also available as scripts for repeatable runs.

## Data Preparation

```bash
python scripts/run_eda.py --output-dir artifacts/eda
python scripts/prepare_data.py --config configs/datasets.yaml --output-dir artifacts/data
```

Prepared examples use a shared `MedicalQAExample` schema with dataset name,
question, optional context, choices, gold labels, metadata, and engineered
features.

## Training

The paper evaluates the Qwen2.5 Instruct family at 1.5B, 3B, and 7B. This
project includes LoRA/QLoRA configs for all three sizes:

```bash
accelerate launch scripts/train_coca_grpo.py --config configs/train_qwen25_1_5b_lora.yaml
accelerate launch scripts/train_coca_grpo.py --config configs/train_qwen25_3b_lora.yaml
accelerate launch scripts/train_coca_grpo.py --config configs/train_qwen25_7b_lora.yaml
```

The training script implements grouped rollouts, CoCA rewards, and a segmented
GRPO loss utility that applies confidence advantages to confidence tokens and
answer advantages to answer tokens.

To run the full Qwen2.5 model sweep:

```bash
bash scripts/run_qwen25_sweep.sh
```

## Evaluation

```bash
accelerate launch scripts/evaluate_coca.py --config configs/eval_qwen25_1_5b.yaml
accelerate launch scripts/evaluate_coca.py --config configs/eval_qwen25_3b.yaml
accelerate launch scripts/evaluate_coca.py --config configs/eval_qwen25_7b.yaml
```

Metrics include:

- accuracy
- AUROC
- expected calibration error (ECE)
- Brier score
- confidence-generation success rate
- token-to-confidence (TTC)

## Tests

```bash
pytest
```

Tests cover dataset normalization, confidence parsing, answer correctness,
CoCA rewards, segment masks, and calibration metrics.
