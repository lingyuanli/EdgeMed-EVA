"""Prompts transcribed from the Med-CMR arXiv supplement."""

from __future__ import annotations

import hashlib


MCQ_PROMPT_VARIANTS = ("direct", "structured_evidence")


def mcq_prompt(
    question: str,
    options: dict[str, str],
    variant: str = "direct",
) -> str:
    option_lines = "\n".join(f"{letter}) {options[letter]}" for letter in "ABCDE")
    stem = (
        "Please carefully observe this medical image and answer the following question:\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{option_lines}\n\n"
    )
    if variant == "direct":
        return stem + "Answer only with the option letter (A–E)."
    if variant == "structured_evidence":
        return stem + (
            "Return exactly one compact JSON object with no markdown or extra text:\n"
            '{"observation":"one concise image-grounded finding",'
            '"hypotheses":["A","B"],"answer":"A"}\n'
            "The observation must state only a visible finding relevant to the question. "
            "Hypotheses must contain 1–3 distinct option letters (A–E), including the final "
            "answer. Do not provide hidden chain-of-thought; provide only this short evidence summary."
        )
    raise ValueError(f"Unknown MCQ prompt variant: {variant}")


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


def prompt_hash(kind: str, variant: str = "direct") -> str:
    if kind == "mcq":
        text = mcq_prompt(
            "{question}",
            {letter: f"{{option_{letter}}}" for letter in "ABCDE"},
            variant=variant,
        )
    elif kind == "open":
        if variant != "direct":
            raise ValueError("Open prompts currently support only the direct variant")
        text = open_prompt("{question}")
    else:
        raise ValueError(f"Unknown prompt kind: {kind}")
    return hashlib.sha256(text.encode()).hexdigest()
