"""Confidence-first prompt construction."""

from __future__ import annotations

from coca_med.data.schema import MedicalQAExample


SYSTEM_PROMPT = (
    "You are a careful medical question-answering assistant. "
    "You must first state your confidence as a number between 0 and 1 enclosed "
    "in <confidence> </confidence> tags, then provide the answer. "
    "Use this exact format: <confidence> confidence level here </confidence> answer here"
)


def format_choices(example: MedicalQAExample) -> str:
    return "\n".join(f"{label}. {text}" for label, text in example.choices.items())


def expected_answer_instruction(example: MedicalQAExample) -> str:
    if example.dataset.startswith("medqa"):
        return "Put the final answer as one option letter: A, B, C, or D."
    if example.dataset.startswith("pubmedqa"):
        return "Put the final decision as one of: yes, no, maybe."
    return "Put the final answer clearly at the end."


def build_user_prompt(example: MedicalQAExample, *, include_step_by_step: bool = True) -> str:
    parts: list[str] = []
    if example.context:
        parts.append(f"Context:\n{example.context}")
    parts.append(f"Question:\n{example.question}")
    if example.choices:
        parts.append(f"Choices:\n{format_choices(example)}")

    instruction = expected_answer_instruction(example)
    if include_step_by_step:
        instruction = f"Please reason step by step. {instruction}"
    parts.append(instruction)
    return "\n\n".join(parts)


def build_chat_messages(
    example: MedicalQAExample,
    *,
    include_step_by_step: bool = True,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(example, include_step_by_step=include_step_by_step)},
    ]


def build_plain_prompt(example: MedicalQAExample, *, include_step_by_step: bool = True) -> str:
    messages = build_chat_messages(example, include_step_by_step=include_step_by_step)
    return "\n\n".join(f"{message['role'].upper()}: {message['content']}" for message in messages)
