"""Token segmentation for confidence and answer spans."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


CONFIDENCE_TAG_RE = re.compile(
    r"<confidence>\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*</confidence>",
    flags=re.IGNORECASE | re.DOTALL,
)


@dataclass(slots=True)
class SegmentMask:
    confidence_token_indices: list[int]
    answer_token_indices: list[int]
    confidence_char_span: tuple[int, int] | None
    answer_char_span: tuple[int, int]


def find_confidence_char_span(text: str) -> tuple[int, int] | None:
    match = CONFIDENCE_TAG_RE.search(text or "")
    if not match:
        return None
    return match.span()


def build_segment_mask_from_offsets(
    offsets: list[tuple[int, int]],
    *,
    confidence_span: tuple[int, int] | None,
    text_length: int,
) -> SegmentMask:
    confidence_indices: list[int] = []
    answer_indices: list[int] = []
    answer_start = confidence_span[1] if confidence_span is not None else 0
    answer_span = (answer_start, text_length)

    for token_idx, (start, end) in enumerate(offsets):
        if start == end:
            continue
        if confidence_span is not None and start < confidence_span[1] and end > confidence_span[0]:
            confidence_indices.append(token_idx)
        elif end > answer_span[0]:
            answer_indices.append(token_idx)

    return SegmentMask(
        confidence_token_indices=confidence_indices,
        answer_token_indices=answer_indices,
        confidence_char_span=confidence_span,
        answer_char_span=answer_span,
    )


def build_segment_mask(text: str, tokenizer: Any) -> SegmentMask:
    """Build token indices for confidence and answer segments.

    Requires a fast tokenizer with offset mapping support. The training script
    uses this on completion-only text so indices align with completion tokens.
    """

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    return build_segment_mask_from_offsets(
        offsets,
        confidence_span=find_confidence_char_span(text),
        text_length=len(text or ""),
    )
