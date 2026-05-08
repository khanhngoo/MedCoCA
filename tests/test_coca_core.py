from coca_med.coca.parsing import (
    extract_medqa_prediction,
    extract_pubmedqa_prediction,
    is_answer_correct,
    parse_confidence_completion,
)
from coca_med.coca.rewards import compute_coca_group_rewards, rewards_to_advantages
from coca_med.coca.segments import build_segment_mask_from_offsets, find_confidence_char_span
from coca_med.data.schema import MedicalQAExample


def medqa_example() -> MedicalQAExample:
    return MedicalQAExample(
        id="x",
        dataset="medqa",
        question="Question?",
        choices={"A": "Alpha", "B": "Beta", "C": "Gamma", "D": "Delta"},
        gold_label="B",
        gold_answer="Beta",
    )


def test_parse_confidence_completion() -> None:
    parsed = parse_confidence_completion("<confidence>0.42</confidence> final answer is B")

    assert parsed.valid_confidence
    assert parsed.confidence == 0.42
    assert "final answer" in parsed.answer_text


def test_answer_extractors() -> None:
    assert extract_medqa_prediction("After reasoning, final answer is B.", medqa_example().choices) == "B"
    assert extract_pubmedqa_prediction("The final decision is maybe.") == "maybe"


def test_answer_correctness() -> None:
    assert is_answer_correct(medqa_example(), "<confidence>0.9</confidence> final answer is B")
    assert not is_answer_correct(medqa_example(), "<confidence>0.9</confidence> final answer is A")


def test_coca_group_rewards_and_advantages() -> None:
    rewards = compute_coca_group_rewards(
        medqa_example(),
        [
            "<confidence>0.5</confidence> final answer is B",
            "<confidence>0.5</confidence> final answer is A",
        ],
    )
    confidence_advantages, answer_advantages = rewards_to_advantages(rewards)

    assert rewards[0].gesr == 0.5
    assert rewards[0].answer_reward == 1.0
    assert rewards[1].answer_reward == 0.0
    assert len(confidence_advantages) == 2
    assert len(answer_advantages) == 2
    assert answer_advantages[0] > answer_advantages[1]


def test_segment_mask_from_offsets() -> None:
    text = "<confidence>0.5</confidence> answer"
    span = find_confidence_char_span(text)
    mask = build_segment_mask_from_offsets(
        [(0, 12), (12, 15), (15, 28), (29, 35)],
        confidence_span=span,
        text_length=len(text),
    )

    assert mask.confidence_token_indices == [0, 1, 2]
    assert mask.answer_token_indices == [3]


def test_segmented_grpo_loss_prefers_advantaged_tokens() -> None:
    import pytest

    torch = pytest.importorskip("torch")
    from coca_med.coca.loss import segmented_grpo_loss

    new_logprobs = torch.log(torch.tensor([[0.6, 0.7]]))
    old_logprobs = torch.log(torch.tensor([[0.5, 0.5]]))
    confidence_mask = torch.tensor([[True, False]])
    answer_mask = torch.tensor([[False, True]])
    loss = segmented_grpo_loss(
        new_logprobs,
        old_logprobs,
        confidence_mask,
        answer_mask,
        torch.tensor([1.0]),
        torch.tensor([1.0]),
    )

    assert loss.item() < 0
