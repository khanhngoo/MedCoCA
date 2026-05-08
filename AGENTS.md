# AGENTS.md - CoCA Medical QA Project Guide

This document provides essential context for AI agents and developers working on the MedCoCA project.

## Project Overview

**MedCoCA** implements "Confidence Before Answering: A Paradigm Shift for Efficient LLM Uncertainty Estimation" adapted for medical question-answering datasets.

**Core Innovation:** Models output confidence *before* generating answers:
```
<confidence>0.73</confidence> The answer is B
```

This enables early decision-making (routing, refusal) and better calibration compared to traditional answer-first approaches.

## Architecture

### 1. Data Layer (`src/coca_med/data/`)

**Schema:** `MedicalQAExample` - unified format for both datasets
- `id`, `dataset`, `question`, `context`, `choices`
- `gold_label` (A/B/C/D or yes/no/maybe)
- `gold_answer` (full text)
- `metadata` (dataset-specific fields)
- `features` (engineered features)

**Loaders:**
- `medqa.py`: GBaker/MedQA-USMLE-4-options → multiple choice A-D
- `pubmedqa.py`: qiaojin/PubMedQA (pqa_artificial, pqa_labeled, pqa_unlabeled) → yes/no/maybe

**Key Design:** Lazy loading via generators; HuggingFace datasets imported inside functions to avoid import-time errors.

### 2. Feature Engineering (`src/coca_med/data/features.py`)

Features computed for every example:
- Text stats: `question_words`, `question_chars`, `context_words`
- Structure: `num_choices`, `num_metamap_phrases`, `num_mesh_terms`
- Flags: `has_context`, `has_long_answer`, `has_gold_label`
- **Difficulty proxy:** `question_words + sqrt(context_words) + 2*num_choices + 0.5*biomedical_terms`

### 3. CoCA Core (`src/coca_med/coca/`)

**Prompts (`prompts.py`):**
- System prompt enforces `<confidence>X</confidence> answer` format
- User prompt includes question, context, choices, and answer instruction
- Supports chat templates (Qwen) or plain text fallback

**Parsing (`parsing.py`):**
- `parse_confidence_completion()`: Extracts confidence value, validates 0-1 range
- `extract_medqa_prediction()`: Handles "final answer is B", letter matching, option text matching
- `extract_pubmedqa_prediction()`: Handles "yes/no/maybe" extraction
- `is_answer_correct()`: Dataset-aware correctness check

**Rewards (`rewards.py`):**
```python
answer_reward = 1 if correct else 0
GESR = mean(answer_rewards across group)  # dynamic target
confidence_reward = -(confidence - GESR)²  # Brier-style
```
- Advantages: group-normalized (mean=0, std=1) for stable training

**Segments (`segments.py`):**
- Build token-level masks: which tokens belong to confidence vs answer segments
- Uses tokenizer offset mapping to align with generation positions

**Loss (`loss.py`):**
- Segmented GRPO: separate policy gradients for confidence and answer tokens
- Clipped importance ratios (PPO-style) with per-segment advantages

### 4. Training Pipeline (`scripts/train_coca_grpo.py`)

**Flow per training step:**
1. Render prompt for one example
2. Generate `group_size` completions (default: 8) from current policy
3. Compute answer correctness for each completion
4. Compute GESR (group empirical success rate)
5. Compute confidence rewards via Brier penalty
6. Calculate advantages (group-normalized)
7. Build segment masks for each completion
8. Compute segmented GRPO loss
9. Backprop and update

**Model Support:**
- Qwen2.5-Instruct family: 1.5B, 3B, 7B
- LoRA/QLoRA for memory-efficient training
- Multi-GPU via Accelerate

**Configs:**
- `configs/train_qwen25_1_5b_lora.yaml`
- `configs/train_qwen25_3b_lora.yaml`
- `configs/train_qwen25_7b_lora.yaml`

### 5. Evaluation (`src/coca_med/eval/`, `scripts/evaluate_coca.py`)

**Metrics:**
- `accuracy`: Proportion correct
- `auroc`: Area under ROC (confidence vs correctness discrimination)
- `ece`: Expected Calibration Error (binned calibration)
- `brier`: Mean squared error between confidence and binary correctness
- `confidence_success_rate`: % of valid confidence extractions
- `token_to_confidence`: How many tokens until confidence tag (latency proxy)

## Key Design Decisions

### 1. Dataset Normalization
**Decision:** Convert both MedQA and PubMedQA to common `MedicalQAExample` format.
**Rationale:** Single training loop handles both; feature engineering and metrics work uniformly.

### 2. Dynamic Confidence Targets (GESR)
**Decision:** Confidence target is computed from rollout group success, not frozen labels.
**Rationale:** As the policy improves, confidence targets automatically adjust; prevents overfitting to static difficulty patterns.

### 3. Segmented Rewards
**Decision:** Separate advantages for confidence and answer tokens.
**Rationale:** Prevents reward hacking where model sacrifices answer quality to improve confidence calibration.

### 4. Invalid Confidence Handling
**Decision:** Invalid/missing confidence gets reward = -1.0 (configurable).
**Rationale:** Forces model to learn valid format; avoids silent failures.

### 5. Lightweight Dependencies
**Decision:** Heavy ML packages (torch, datasets) imported lazily inside functions where possible.
**Rationale:** Allows running tests and lightweight scripts without full ML environment.

## File Structure

```
MedCoCA/
├── configs/                    # YAML configs for datasets, training, eval
│   ├── datasets.yaml          # MedQA + PubMedQA dataset mixing
│   ├── train_qwen25_*_lora.yaml  # Model-specific training configs
│   └── eval_qwen25_*.yaml     # Model-specific eval configs
├── notebooks/                 # EDA and prototyping
│   ├── 01_eda_medqa.ipynb
│   ├── 02_eda_pubmedqa.ipynb
│   └── 03_feature_engineering.ipynb
├── scripts/                   # Executable pipelines
│   ├── run_eda.py            # Export EDA summaries
│   ├── prepare_data.py       # Normalize and save JSONL
│   ├── train_coca_grpo.py    # Main training script
│   ├── evaluate_coca.py      # Evaluation script
│   └── run_qwen25_sweep.sh   # Run all three model sizes
├── src/coca_med/             # Main package
│   ├── data/                 # Dataset loading, schema, features
│   ├── coca/                 # CoCA logic: prompts, parsing, rewards, loss
│   └── eval/                 # Metrics
├── tests/                    # Unit tests
├── pyproject.toml            # Dependencies and package config
├── README.md                 # User-facing documentation
└── AGENTS.md                 # This file
```

## Common Tasks

### Add a new dataset
1. Create `src/coca_med/data/newdataset.py` with `normalize_*_row()` function
2. Return `MedicalQAExample` with appropriate `choices`, `gold_label`, `gold_answer`
3. Add to `prepare_data.py` dataset loading logic
4. Update `configs/datasets.yaml`

### Modify the confidence format
1. Edit regex in `src/coca_med/coca/parsing.py` (`CONFIDENCE_RE`)
2. Update `src/coca_med/coca/prompts.py` (`SYSTEM_PROMPT`)
3. Update `src/coca_med/coca/segments.py` (`CONFIDENCE_TAG_RE`)
4. Run tests: `pytest tests/test_coca_core.py`

### Add a new metric
1. Add function to `src/coca_med/eval/metrics.py`
2. Include in `aggregate_metrics()` return value
3. Update `EvaluationMetrics` dataclass
4. Tests in `tests/test_metrics.py`

### Change model size
```bash
# 1.5B
accelerate launch scripts/train_coca_grpo.py --config configs/train_qwen25_1_5b_lora.yaml

# 3B
accelerate launch scripts/train_coca_grpo.py --config configs/train_qwen25_3b_lora.yaml

# 7B (default)
accelerate launch scripts/train_coca_grpo.py --config configs/train_qwen25_7b_lora.yaml
```

### Use a different base model
Edit config YAML:
```yaml
model:
  name_or_path: your-org/your-model
  torch_dtype: bfloat16
  load_in_4bit: true
```

Ensure the model supports chat templates or adjust `prompts.py` to use plain format.

## Testing

Run all tests:
```bash
pytest
```

Test categories:
- `test_data_normalization.py`: MedQA/PubMedQA → MedicalQAExample conversion
- `test_coca_core.py`: Parsing, rewards, segments, loss
- `test_metrics.py`: ECE, Brier, AUROC calculations

Heavy dependencies (torch) are lazily imported; tests that need them skip gracefully if unavailable.

## Training Tips

**Before training:**
1. Run EDA: `python scripts/run_eda.py`
2. Prepare data: `python scripts/prepare_data.py`
3. Verify with small limit: `python scripts/prepare_data.py --limit-per-split 100`

**Memory issues:**
- Reduce `group_size` in config (default: 8)
- Increase `gradient_accumulation_steps`, reduce `train_batch_size`
- Enable `load_in_4bit: true` (already on by default)
- Reduce LoRA rank `r` (default: 16)

**Monitoring:**
- Training logs: `loss`, `answer_reward`, `confidence_reward`, `gesr`, `valid_confidence_rate`
- Evaluation: Check `ece` and `brier` for calibration, `auroc` for discrimination

**Typical training time:**
- 1.5B: ~2-4 hours on single A100 for 1000 steps
- 3B: ~4-6 hours
- 7B: ~8-12 hours

## Paper Alignment

| Paper Element | Implementation |
|---------------|----------------|
| Models | Qwen2.5-1.5B/3B/7B-Instruct |
| Group size | 8 (default, paper uses G) |
| Confidence reward | `-(confidence - GESR)²` |
| Answer reward | Binary correctness |
| Loss | Segmented GRPO (Equation 10-11) |
| ECE bins | 10 (paper default) |
| Training | GRPO with dynamic targets |

Differences from paper (practical adaptations):
- LoRA instead of full fine-tuning (memory efficiency)
- Medical QA instead of math/code/factual benchmarks
- HuggingFace TRL-style implementation instead of MindSpeed-RL

## Questions?

- Check `README.md` for setup and quickstart
- Check `notebooks/` for interactive exploration
- Check `tests/` for usage examples
- Original paper: "Confidence Before Answering: A Paradigm Shift for Efficient LLM Uncertainty Estimation" (arXiv:2603.05881)
