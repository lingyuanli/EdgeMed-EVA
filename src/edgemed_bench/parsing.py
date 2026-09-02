"""Deterministic output parsing for official Med-CMR response formats."""

from __future__ import annotations

import json
import re

ANSWER_MARKER = re.compile(r"(?im)^\s*answer\s*[:\-]\s*[\(\[]?([A-E])[\)\]]?\s*[\.!]?\s*$")
LEADING_OPTION = re.compile(r"(?i)^\s*[\(\[]?([A-E])[\)\].:\-]\s*")
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
    leading = LEADING_OPTION.search(stripped)
    if leading:
        return leading.group(1).upper(), "leading_option"
    line = LETTER_LINE.search(stripped)
    if line:
        return line.group(1).upper(), "standalone_line"
    return None, "invalid"


def parse_structured_mcq(
    text: str,
) -> tuple[str | None, str | None, list[str] | None, str]:
    """Parse the strict B1 JSON surface without repairing or guessing fields."""

    try:
        value = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError):
        return None, None, None, "invalid_structured_json"
    if not isinstance(value, dict) or set(value) != {"observation", "hypotheses", "answer"}:
        return None, None, None, "invalid_structured_schema"
    observation = value["observation"]
    hypotheses = value["hypotheses"]
    answer = value["answer"]
    if not isinstance(observation, str) or not observation.strip():
        return None, None, None, "invalid_structured_schema"
    if (
        not isinstance(hypotheses, list)
        or not 1 <= len(hypotheses) <= 3
        or any(not isinstance(item, str) or item not in "ABCDE" for item in hypotheses)
        or len(set(hypotheses)) != len(hypotheses)
    ):
        return None, None, None, "invalid_structured_schema"
    if not isinstance(answer, str) or answer not in "ABCDE" or answer not in hypotheses:
        return None, None, None, "invalid_structured_schema"
    return answer, observation.strip(), hypotheses, "structured_json"


def parse_evidence_answer_mcq(text: str) -> tuple[str | None, str | None, str]:
    """Parse the minimal B1-v2 evidence/answer JSON without repair."""

    try:
        value = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError):
        return None, None, "invalid_evidence_answer_json"
    if not isinstance(value, dict) or set(value) != {"observation", "answer"}:
        return None, None, "invalid_evidence_answer_schema"
    observation = value["observation"]
    answer = value["answer"]
    if not isinstance(observation, str) or not observation.strip():
        return None, None, "invalid_evidence_answer_schema"
    if not isinstance(answer, str) or answer not in "ABCDE":
        return None, None, "invalid_evidence_answer_schema"
    return answer, observation.strip(), "evidence_answer_json"


def parse_open(text: str) -> tuple[str | None, str | None, str]:
    reasoning_match = OPEN_REASONING.search(text)
    answer_match = OPEN_ANSWER.search(text)
    if reasoning_match and answer_match:
        return reasoning_match.group(1).strip(), answer_match.group(1).strip(), "strict"
    if answer_match:
        return None, answer_match.group(1).strip(), "answer_only"
    return None, None, "invalid"


def parse_open_answer_only(text: str) -> tuple[str | None, str | None, str]:
    """Parse the external answer-only variant without changing direct-open behavior."""
    reasoning, answer, status = parse_open(text)
    if answer is not None:
        return reasoning, answer, status
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) == 1 and len(lines[0].split()) <= 20:
        return None, lines[0], "bare_answer"
    return None, None, "invalid"
