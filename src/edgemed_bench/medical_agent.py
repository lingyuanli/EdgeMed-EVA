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


def canonicalize_final_evidence(final: dict[str, Any]) -> list[dict[str, Any]]:
    """Apply trace-independent structural invariants and return an audit log."""
    normalizations: list[dict[str, Any]] = []
    evidence = final.get("evidence")
    if not isinstance(evidence, list):
        return normalizations
    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            continue
        if (
            item.get("acquisition") in {"inspect_overview", "temporal_skim"}
            and item.get("region_xyxy_1000") is not None
        ):
            normalizations.append(
                {
                    "rule": "non_region_acquisition_requires_null_region",
                    "evidence_index": index,
                    "before": item["region_xyxy_1000"],
                    "after": None,
                }
            )
            item["region_xyxy_1000"] = None
    return normalizations


FINAL_SCHEMA = {
    "sample_id": "exact sample id from CASE",
    "hypotheses": [
        {"id": "H1", "label": "short hypothesis", "status": "supported|refuted|uncertain"}
    ],
    "evidence": [
        {
            "evidence_id": "E1",
            "media_id": "exact media id from CASE",
            "view_or_time": "view name or timestamp",
            "region_xyxy_1000": [0, 0, 1000, 1000],
            "observation": "visible pixels only",
            "supports": ["H1"],
            "contradicts": [],
            "acquisition": "inspect_overview|temporal_skim|region_inspect",
            "confidence": 0.5,
        }
    ],
    "answer": "one visible MCQ option letter, or null for an open question",
    "answer_text": "concise answer text",
    "answer_evidence_ids": ["E1"],
    "confidence": 0.5,
    "insufficient_evidence": False,
    "tool_trace_ids": ["exact completed trace id from TOOL_RESULT"],
}


def run_medical_agent(
    sample: dict[str, Any],
    backend: AgentBackend,
    data_root: Path,
    artifact_dir: Path,
    allowed_tools: tuple[str, ...] = tuple(TOOL_SCHEMAS),
    max_steps: int = 4,
    initial_visual_policy: str = "none",
) -> AgentRunResult:
    """Run one sample without accepting any reference-bearing inference row."""
    reject_reference_fields([sample])
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if initial_visual_policy not in {
        "none",
        "overview",
        "overview_then_region",
        "overview_then_localize",
    }:
        raise ValueError(f"Unknown initial_visual_policy: {initial_visual_policy}")
    if initial_visual_policy in {
        "overview", "overview_then_region", "overview_then_localize"
    } and "inspect_overview" not in allowed_tools:
        raise ValueError("initial overview policy requires inspect_overview in allowed_tools")
    if (
        initial_visual_policy in {"overview_then_region", "overview_then_localize"}
        and "region_inspect" not in allowed_tools
    ):
        raise ValueError("targeted region policy requires region_inspect in allowed_tools")
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
    if initial_visual_policy in {
        "overview",
        "overview_then_region",
        "overview_then_localize",
    }:
        initial_call = {"name": "inspect_overview", "arguments": {"sample_count": 1}}
        messages.append(
            {
                "role": "assistant",
                "content": "Acquire a low-resolution overview before planning targeted evidence.",
                "tool_call": initial_call,
                "policy_intervention": "initial_overview_required",
            }
        )
        if initial_visual_policy == "overview_then_region":
            messages.append(
                {
                    "role": "user",
                    "content": {
                        "controller_requirement": (
                            "After inspecting the overview, call region_inspect exactly once. "
                            "Choose a question-relevant box with normalized area in [0.01,0.64]; "
                            "do not use the full frame and give a concrete target."
                        )
                    },
                    "policy_intervention": "targeted_region_required",
                }
            )
        initial_result, initial_trace = executor.execute(
            initial_call["name"], initial_call["arguments"]
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": initial_trace["trace_id"],
                "name": initial_trace["tool_name"],
                "content": initial_result,
            }
        )
    finish_reason = "max_steps"
    if initial_visual_policy == "overview_then_localize":
        localize = getattr(backend, "localize", None)
        if not callable(localize):
            raise TypeError("overview_then_localize requires backend.localize")
        turn = localize(messages, {name: TOOL_SCHEMAS[name] for name in allowed_tools})
        if not isinstance(turn, dict):
            raise TypeError("Agent backend localization must be an object")
        tool_call = turn.get("tool_call")
        if not isinstance(tool_call, dict) or tool_call.get("name") != "region_inspect":
            raise RuntimeError("Dedicated localizer did not select region_inspect")
        if not isinstance(tool_call.get("arguments"), dict):
            raise ValueError("Dedicated localizer returned invalid tool arguments")
        localization_message = {
            "role": "assistant",
            "content": turn.get("content", ""),
            "phase": "localize",
            "tool_call": tool_call,
            "policy_intervention": "dedicated_region_localizer",
        }
        if isinstance(turn.get("_model_call"), dict):
            localization_message["model_call"] = turn["_model_call"]
        messages.append(localization_message)
        result, trace = executor.execute("region_inspect", tool_call["arguments"])
        messages.append(
            {
                "role": "tool",
                "tool_call_id": trace["trace_id"],
                "name": trace["tool_name"],
                "content": result,
            }
        )
        if trace["status"] != "completed":
            raise RuntimeError("Dedicated localizer produced a failed region tool call")
        finish_reason = "forced_localization_completed"

    for _ in range(0 if initial_visual_policy == "overview_then_localize" else max_steps):
        turn = backend.decide(messages, {name: TOOL_SCHEMAS[name] for name in allowed_tools})
        if not isinstance(turn, dict):
            raise TypeError("Agent backend decision must be an object")
        tool_call = turn.get("tool_call")
        assistant_message = {
            "role": "assistant", "content": turn.get("content", ""), "phase": "decision"
        }
        if isinstance(turn.get("_model_call"), dict):
            assistant_message["model_call"] = turn["_model_call"]
        if tool_call is not None:
            assistant_message["tool_call"] = tool_call
        messages.append(assistant_message)
        completed_region = any(
            trace["status"] == "completed" and trace["tool_name"] == "region_inspect"
            for trace in executor.traces
        )
        if initial_visual_policy == "overview_then_region" and not completed_region:
            if not isinstance(tool_call, dict) or tool_call.get("name") != "region_inspect":
                raise RuntimeError(
                    "Forced targeted-region policy was not followed by the backend"
                )
        if tool_call is None:
            has_visual_evidence = any(
                trace["status"] == "completed" for trace in executor.traces
            )
            if has_visual_evidence:
                finish_reason = "evidence_sufficient"
                break
            if "inspect_overview" not in allowed_tools:
                break
            tool_call = {"name": "inspect_overview", "arguments": {"sample_count": 1}}
            messages.append(
                {
                    "role": "assistant",
                    "content": "Visual evidence is required before finalization.",
                    "tool_call": tool_call,
                    "policy_intervention": "first_visual_acquisition_required",
                }
            )
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
    policy_normalizations = canonicalize_final_evidence(final)
    final_message = {"role": "assistant", "content": final, "phase": "final"}
    if isinstance(final_model_call, dict):
        final_message["model_call"] = final_model_call
    if policy_normalizations:
        final_message["policy_normalizations"] = policy_normalizations
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
    if policy_normalizations:
        prediction["policy_normalizations"] = policy_normalizations
    trajectory = {
        "schema_version": "edgemed-medical-agent-trajectory/v1",
        "sample_id": sample["sample_id"],
        "messages": messages,
        "tool_trace_ids": prediction["tool_trace_ids"],
        "finish_reason": finish_reason,
        "decision_calls": sum(
            message.get("phase") == "decision" for message in messages
        ),
        "localizer_calls": sum(
            message.get("phase") == "localize" for message in messages
        ),
        "policy_interventions": sum(
            isinstance(message.get("policy_intervention"), str) for message in messages
        ),
        "finalizer_calls": 1,
    }
    return AgentRunResult(prediction=prediction, trajectory=trajectory, tool_traces=executor.traces)
