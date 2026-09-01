from edgemed_bench.parsing import parse_mcq, parse_open, parse_structured_mcq
from edgemed_bench.prompts import mcq_prompt, open_prompt, prompt_hash


def test_mcq_prompt_contains_only_inference_fields() -> None:
    prompt = mcq_prompt("Question?", {letter: f"Option {letter}" for letter in "ABCDE"})
    assert "Question: Question?" in prompt
    assert "A) Option A" in prompt
    assert "Answer only with the option letter (A–E)." in prompt
    assert "ground truth" not in prompt.lower()
    assert "visual_description" not in prompt


def test_open_prompt_contract() -> None:
    prompt = open_prompt("Question?")
    assert "Reasoning:" in prompt
    assert "Answer:" in prompt
    assert "one short sentence or a single medical term" in prompt


def test_direct_prompt_hash_remains_frozen() -> None:
    assert prompt_hash("mcq") == "d8245cf6e33e209b9819935c40bea5a9a47efcd5c8469842e87afb2d18a9160a"


def test_structured_mcq_prompt_is_answer_blind() -> None:
    prompt = mcq_prompt(
        "Question?",
        {letter: f"Option {letter}" for letter in "ABCDE"},
        variant="structured_evidence",
    )
    assert '"observation"' in prompt
    assert '"hypotheses"' in prompt
    assert '"answer"' in prompt
    assert "ground truth" not in prompt.lower()
    assert "visual_description" not in prompt
    assert prompt_hash("mcq", "structured_evidence") != prompt_hash("mcq")


def test_mcq_parser_is_conservative() -> None:
    assert parse_mcq("A") == ("A", "exact_letter")
    assert parse_mcq("Answer: (c)") == ("C", "answer_marker")
    assert parse_mcq("C) C) Leading option with explanation") == ("C", "leading_option")
    assert parse_mcq("Reasoning here\nD.") == ("D", "standalone_line")
    assert parse_mcq("A differential includes B and C") == (None, "invalid")


def test_structured_mcq_parser_accepts_only_bound_schema() -> None:
    text = '{"observation":"Focal opacity","hypotheses":["B","D"],"answer":"D"}'
    assert parse_structured_mcq(text) == (
        "D",
        "Focal opacity",
        ["B", "D"],
        "structured_json",
    )
    assert parse_structured_mcq("D") == (None, None, None, "invalid_structured_json")
    assert parse_structured_mcq(
        '{"observation":"Focal opacity","hypotheses":["B"],"answer":"D"}'
    ) == (None, None, None, "invalid_structured_schema")
    assert parse_structured_mcq(
        '{"observation":"Focal opacity","hypotheses":["B"],"answer":"B","extra":1}'
    ) == (None, None, None, "invalid_structured_schema")


def test_open_parser() -> None:
    reasoning, answer, status = parse_open("Reasoning: visible finding\nAnswer: diagnosis")
    assert (reasoning, answer, status) == ("visible finding", "diagnosis", "strict")
    assert parse_open("No schema") == (None, None, "invalid")
