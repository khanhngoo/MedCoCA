"""CoCA reward computation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coca_med.coca.parsing import is_answer_correct, parse_confidence_completion
from coca_med.data.schema import MedicalQAExample


@dataclass(slots=True)
class CoCAReward:
    completion: str
    answer_reward: float
    confidence_reward: float
    gesr: float
    confidence: float | None
    valid_confidence: bool

    @property
    def combined_reward(self) -> float:
        return self.answer_reward + self.confidence_reward


def normalize_advantages(values: list[float], eps: float = 1e-6) -> list[float]:
    if not values:
        return []
    array = np.asarray(values, dtype=np.float32)
    std = float(array.std())
    if std < eps:
        return [0.0 for _ in values]
    normalized = (array - float(array.mean())) / (std + eps)
    return normalized.astype(float).tolist()


def compute_coca_group_rewards(
    example: MedicalQAExample,
    completions: list[str],
    *,
    invalid_confidence_reward: float = -1.0,
) -> list[CoCAReward]:
    answer_rewards = [1.0 if is_answer_correct(example, completion) else 0.0 for completion in completions]
    gesr = float(np.mean(answer_rewards)) if answer_rewards else 0.0

    rewards: list[CoCAReward] = []
    for completion, answer_reward in zip(completions, answer_rewards, strict=True):
        parsed = parse_confidence_completion(completion)
        if parsed.valid_confidence and parsed.confidence is not None:
            confidence_reward = -float((parsed.confidence - gesr) ** 2)
            confidence = parsed.confidence
        else:
            confidence_reward = float(invalid_confidence_reward)
            confidence = None
        rewards.append(
            CoCAReward(
                completion=completion,
                answer_reward=answer_reward,
                confidence_reward=confidence_reward,
                gesr=gesr,
                confidence=confidence,
                valid_confidence=parsed.valid_confidence,
            )
        )
    return rewards


def rewards_to_advantages(rewards: list[CoCAReward]) -> tuple[list[float], list[float]]:
    confidence_advantages = normalize_advantages([reward.confidence_reward for reward in rewards])
    answer_advantages = normalize_advantages([reward.answer_reward for reward in rewards])
    return confidence_advantages, answer_advantages
