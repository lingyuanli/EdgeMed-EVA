"""Evidence-gated medical multimodal Agent controller."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .io import reject_reference_fields
from .medical_agent_tools import MedicalToolExecutor, TOOL_SCHEMAS


SYSTEM_CONTRACT = """You are an evidence-grounded medical multimodal benchmark agent.
Use only the supplied clinical context and visual tool artifacts. A tool extracts pixels; it
does not provide a diagnosis. Distinguish visible observation from clinical interpretation,
cite evidence identifiers in the final structured output, and never invent missing evidence.
Call at most one enabled visual tool per decision turn. When ready, return no tool call; a
separate finalizer will then produce the answer schema."""


class AgentBackend(Protocol):
    def decide(self, messages: list[dict[str, Any]], tools: dict[str, Any]) -> dict[str, Any]: ...

    def finalize(self, messages: list[dict[str, Any]], output_schema: dict[str, Any]) -> dict[str, Any]: ...


@dataclass
class AgentRunResult:
    prediction: dict[str, Any]
    trajectory: dict[str, Any]
    tool_traces: list[dict[str, Any]]


FINAL_SCHEMA = {
    "required": [
        "sample_id",
        "hypotheses",
        "evidence",
        "answer",
        "answer_evidence_ids",
        "confidence",
        "insufficient_evidence",
        "tool_trace_ids",
    ]
}


def run_medical_agent(
    sample: dict[str, Any],
    backend: AgentBackend,
    data_root: Path,
    artifact_dir: Path,
    allowed_tools: tuple[str, ...] = tuple(TOOL_SCHEMAS),
    max_steps: int = 4,
) -> AgentRunResult:
    """Run one sample without accepting any reference-bearing inference row."""
    reject_reference_fields([sample])
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    executor = MedicalToolExecutor(
        sample=sample,
        data_root=data_root,
        artifact_dir=artifact_dir,
        allowed_tools=allowed_tools,
        max_calls=max_steps,
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_CONTRACT},
        {
            "role": "user",
            "content": {
                "sample_id": sample["sample_id"],
                "question_type": sample["question_type"],
                "question": sample["question"],
                "options": sample.get("options"),
                "clinical_context": sample.get("clinical_context", ""),
                "media": sample["media"],
            },
        },
    ]
    finish_reason = "max_steps"
    for _ in range(max_steps):
        turn = backend.decide(messages, {name: TOOL_SCHEMAS[name] for name in allowed_tools})
        if not isinstance(turn, dict):
            raise TypeError("Agent backend decision must be an object")
        tool_call = turn.get("tool_call")
        assistant_message = {"role": "assistant", "content": turn.get("content", "")}
        if isinstance(turn.get("_model_call"), dict):
            assistant_message["model_call"] = turn["_model_call"]
        if tool_call is not None:
            assistant_message["tool_call"] = tool_call
        messages.append(assistant_message)
        if tool_call is None:
            finish_reason = "evidence_sufficient"
            break
        if not isinstance(tool_call, dict) or not isinstance(tool_call.get("arguments"), dict):
            raise ValueError("tool_call must contain name and object arguments")
        result, trace = executor.execute(str(tool_call.get("name")), tool_call["arguments"])
        messages.append(
            {
                "role": "tool",
                "tool_call_id": trace["trace_id"],
                "name": trace["tool_name"],
                "content": result,
            }
        )

    successful = [trace for trace in executor.traces if trace["status"] == "completed"]
    if not successful:
        raise RuntimeError("Finalization blocked: no successful visual evidence acquisition")
    final = backend.finalize(messages, FINAL_SCHEMA)
    if not isinstance(final, dict):
        raise TypeError("Agent finalizer must return an object")
    final_model_call = final.pop("_model_call", None)
    final_message = {"role": "assistant", "content": final, "phase": "final"}
    if isinstance(final_model_call, dict):
        final_message["model_call"] = final_model_call
    messages.append(final_message)
    prediction = {
        "schema_version": "edgemed-medical-agent-prediction/v1",
        "sample_id": sample["sample_id"],
        "status": "completed",
        "parsed_answer": final.get("answer"),
        "agent_output": final,
        "tool_trace_ids": [trace["trace_id"] for trace in executor.traces],
        "finish_reason": finish_reason,
    }
    trajectory = {
        "schema_version": "edgemed-medical-agent-trajectory/v1",
        "sample_id": sample["sample_id"],
        "messages": messages,
        "tool_trace_ids": prediction["tool_trace_ids"],
        "finish_reason": finish_reason,
        "decision_calls": sum(message["role"] == "assistant" and message.get("phase") != "final" for message in messages),
        "finalizer_calls": 1,
    }
    return AgentRunResult(prediction=prediction, trajectory=trajectory, tool_traces=executor.traces)
