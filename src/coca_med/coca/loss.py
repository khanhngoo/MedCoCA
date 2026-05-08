"""Segmented GRPO loss used by CoCA training."""

from __future__ import annotations

import torch


def segmented_grpo_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    confidence_mask: torch.Tensor,
    answer_mask: torch.Tensor,
    confidence_advantages: torch.Tensor,
    answer_advantages: torch.Tensor,
    *,
    clip_range: float = 0.2,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute CoCA's segmented clipped policy-gradient objective.

    Shapes:
        new_logprobs, old_logprobs, confidence_mask, answer_mask: [batch, seq]
        confidence_advantages, answer_advantages: [batch]

    The returned value is a loss to minimize. It maximizes the sum of the
    confidence-segment and answer-segment clipped objectives from the paper.
    """

    ratio = torch.exp(new_logprobs - old_logprobs)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)

    confidence_advantages = confidence_advantages.unsqueeze(-1)
    answer_advantages = answer_advantages.unsqueeze(-1)

    confidence_objective = torch.minimum(
        ratio * confidence_advantages,
        clipped_ratio * confidence_advantages,
    )
    answer_objective = torch.minimum(
        ratio * answer_advantages,
        clipped_ratio * answer_advantages,
    )

    mask = confidence_mask.float() + answer_mask.float()
    if attention_mask is not None:
        mask = mask * attention_mask.float()

    objective = (
        confidence_objective * confidence_mask.float()
        + answer_objective * answer_mask.float()
    )
    denom = mask.sum().clamp_min(1.0)
    return -(objective.sum() / denom)
