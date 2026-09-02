"""Prompts transcribed from the Med-CMR arXiv supplement."""

from __future__ import annotations

import hashlib


MCQ_PROMPT_VARIANTS = (
    "direct",
    "structured_evidence",
    "evidence_answer_v2",
    "semantic_option",
)
OPEN_PROMPT_VARIANTS = ("direct", "answer_only")


def mcq_prompt(
    question: str,
    options: dict[str, str],
    variant: str = "direct",
) -> str:
    option_letters = "".join(sorted(options))
    if option_letters not in {"ABCD", "ABCDE"}:
        raise ValueError(f"MCQ options must be A-D or A-E, got: {option_letters}")
    option_lines = "\n".join(f"{letter}) {options[letter]}" for letter in option_letters)
    stem = (
        "Please carefully observe this medical image and answer the following question:\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{option_lines}\n\n"
    )
    if variant == "direct":
        return stem + f"Answer only with the option letter (A–{option_letters[-1]})."
    if variant == "semantic_option":
        return stem + (
            "Return exactly one line in this format:\n"
            "Answer: <complete text of the single best option>\n"
            "Copy the selected option text exactly. Do not output an option letter, reasoning, "
            "explanation, markdown, or any other text."
        )
    if variant == "structured_evidence":
        return stem + (
            "Return exactly one compact JSON object with no markdown or extra text:\n"
            '{"observation":"one concise image-grounded finding",'
            '"hypotheses":["A","B"],"answer":"A"}\n'
            "The observation must state only a visible finding relevant to the question. "
            f"Hypotheses must contain 1–3 distinct option letters (A–{option_letters[-1]}), including the final "
            "answer. Do not provide hidden chain-of-thought; provide only this short evidence summary."
        )
    if variant == "evidence_answer_v2":
        return stem + (
            "Return exactly one compact JSON object with no markdown or extra text:\n"
            '{"observation":"visible finding in 20 words or fewer","answer":"A"}\n'
            "Use exactly these two keys. The observation must be one concise, image-grounded "
            f"visible finding of no more than 20 words. The answer must be one option letter A–{option_letters[-1]}. "
            "Do not add explanations, hypotheses, reasoning, citations, or additional keys."
        )
    raise ValueError(f"Unknown MCQ prompt variant: {variant}")


def open_prompt(question: str, variant: str = "direct") -> str:
    stem = (
        "Please carefully observe this medical image and answer the following question:\n\n"
        f"Question: {question}\n\n"
    )
    if variant == "answer_only":
        return stem + (
            "Return exactly one line in this format:\n"
            "Answer: <one short sentence or a single medical term>\n"
            "Do not include reasoning, explanation, markdown, or any other text."
        )
    if variant != "direct":
        raise ValueError(f"Unknown open prompt variant: {variant}")
    return stem + (
        "Think step by step, integrating both visual features and medical knowledge to "
        "reach your conclusion. Then provide the final answer to the question in one short "
        "sentence or a single medical term.\n\n"
        "Output format (Must follow strictly):\n"
        "Reasoning: <visual and diagnostic reasoning process>\n"
        "Answer: <final answer to the question>"
    )


def prompt_hash(kind: str, variant: str = "direct", option_letters: str = "ABCDE") -> str:
    if kind == "mcq":
        text = mcq_prompt(
            "{question}",
            {letter: f"{{option_{letter}}}" for letter in option_letters},
            variant=variant,
        )
    elif kind == "open":
        text = open_prompt("{question}", variant=variant)
    else:
        raise ValueError(f"Unknown prompt kind: {kind}")
    return hashlib.sha256(text.encode()).hexdigest()
