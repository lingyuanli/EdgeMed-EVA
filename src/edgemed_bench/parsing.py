"""Deterministic output parsing for official Med-CMR response formats."""

from __future__ import annotations

import re

ANSWER_MARKER = re.compile(r"(?im)^\s*answer\s*[:\-]\s*[\(\[]?([A-E])[\)\]]?\s*[\.!]?\s*$")
LETTER_LINE = re.compile(r"(?im)^\s*[\(\[]?([A-E])[\)\]]?\s*[\.!]?\s*$")
OPEN_REASONING = re.compile(r"(?is)\breasoning\s*:\s*(.*?)(?=\n\s*answer\s*:)")
OPEN_ANSWER = re.compile(r"(?is)\banswer\s*:\s*(.+?)\s*$")


def parse_mcq(text: str) -> tuple[str | None, str]:
    stripped = text.strip()
    if len(stripped) == 1 and stripped.upper() in "ABCDE":
        return stripped.upper(), "exact_letter"
    marker = ANSWER_MARKER.search(stripped)
    if marker:
        return marker.group(1).upper(), "answer_marker"
    line = LETTER_LINE.search(stripped)
    if line:
        return line.group(1).upper(), "standalone_line"
    return None, "invalid"


def parse_open(text: str) -> tuple[str | None, str | None, str]:
    reasoning_match = OPEN_REASONING.search(text)
    answer_match = OPEN_ANSWER.search(text)
    if reasoning_match and answer_match:
        return reasoning_match.group(1).strip(), answer_match.group(1).strip(), "strict"
    if answer_match:
        return None, answer_match.group(1).strip(), "answer_only"
    return None, None, "invalid"
