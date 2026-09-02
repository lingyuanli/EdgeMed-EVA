import copy
from typing import Any


class Trajectory:
    """Complete message trajectory of an agent run."""

    def __init__(
        self,
        question: str,
        messages: list[dict],
        total_steps: int,
        final_answer: str,
        finish_reason: str,
    ) -> None:
        self.question = question
        self.messages = copy.deepcopy(messages)
        self.total_steps = total_steps
        self.final_answer = final_answer
        self.finish_reason = finish_reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "messages": copy.deepcopy(self.messages),
            "total_steps": self.total_steps,
            "final_answer": self.final_answer,
            "finish_reason": self.finish_reason,
        }
