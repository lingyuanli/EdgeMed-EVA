"""Prompts transcribed from the Med-CMR arXiv supplement."""

from __future__ import annotations

import hashlib


def mcq_prompt(question: str, options: dict[str, str]) -> str:
    option_lines = "\n".join(f"{letter}) {options[letter]}" for letter in "ABCDE")
    return (
        "Please carefully observe this medical image and answer the following question:\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{option_lines}\n\n"
        "Answer only with the option letter (A–E)."
    )


def open_prompt(question: str) -> str:
    return (
        "Please carefully observe this medical image and answer the following question:\n\n"
        f"Question: {question}\n\n"
        "Think step by step, integrating both visual features and medical knowledge to "
        "reach your conclusion. Then provide the final answer to the question in one short "
        "sentence or a single medical term.\n\n"
        "Output format (Must follow strictly):\n"
        "Reasoning: <visual and diagnostic reasoning process>\n"
        "Answer: <final answer to the question>"
    )


def prompt_hash(kind: str) -> str:
    if kind == "mcq":
        text = mcq_prompt("{question}", {letter: f"{{option_{letter}}}" for letter in "ABCDE"})
    elif kind == "open":
        text = open_prompt("{question}")
    else:
        raise ValueError(f"Unknown prompt kind: {kind}")
    return hashlib.sha256(text.encode()).hexdigest()
