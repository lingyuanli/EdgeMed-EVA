"""Local Qwen3.5 multimodal backend for the evidence-gated Agent protocol."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from PIL import Image

from .io import sha256_file
from .run import resize_to_pixel_budget


def parse_json_object(text: str) -> dict[str, Any]:
    """Extract exactly one JSON object without guessing or schema repair."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].strip().casefold() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    decoder = json.JSONDecoder()
    try:
        value, end = decoder.raw_decode(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError("Model output is not exactly one JSON object") from exc
    if isinstance(value, dict) and not stripped[end:].strip():
        return value
    raise ValueError("Model output is not exactly one JSON object")


def validate_model_source(model_path: Path, source_manifest_path: Path) -> dict[str, Any]:
    source = json.loads(source_manifest_path.read_text())
    model = source.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("weight_shards"), dict):
        raise ValueError("Model source manifest has no weight_shards")
    verified = {}
    for relative_path, record in model["weight_shards"].items():
        path = model_path / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_size = path.stat().st_size
        actual_sha256 = sha256_file(path)
        if actual_size != record.get("bytes") or actual_sha256 != record.get("sha256"):
            raise ValueError(f"Model shard differs from source manifest: {relative_path}")
        verified[relative_path] = {"bytes": actual_size, "sha256": actual_sha256}
    return {
        "repo_id": model.get("repo_id"),
        "revision": model.get("revision"),
        "weight_shards": verified,
        "source_manifest_sha256": sha256_file(source_manifest_path),
    }


class Qwen35MedicalAgentBackend:
    def __init__(
        self,
        model_path: Path,
        model_source_manifest: Path,
        *,
        decision_max_new_tokens: int = 192,
        final_max_new_tokens: int = 512,
        max_image_pixels: int = 786_432,
        verify_weights: bool = True,
    ) -> None:
        import accelerate
        import bitsandbytes
        import torch
        import transformers
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the Qwen Agent backend")
        if torch.cuda.get_device_capability(0) != (7, 0):
            raise RuntimeError(f"Expected V100 SM70, got {torch.cuda.get_device_capability(0)}")
        self.model_path = model_path.resolve()
        self.model_source_manifest = model_source_manifest.resolve()
        self.decision_max_new_tokens = decision_max_new_tokens
        self.final_max_new_tokens = final_max_new_tokens
        self.max_image_pixels = max_image_pixels
        self._torch = torch
        source = (
            validate_model_source(self.model_path, self.model_source_manifest)
            if verify_weights
            else {"source_manifest_sha256": sha256_file(self.model_source_manifest)}
        )
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        load_started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            local_files_only=True,
            quantization_config=quantization,
            dtype=torch.float16,
            device_map={"": 0},
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.receipt = {
            "backend": "qwen35-medical-agent/v1",
            "model_path": str(self.model_path),
            "model_source": source,
            "load_seconds": time.perf_counter() - load_started,
            "generation": {
                "do_sample": False,
                "thinking_mode": False,
                "decision_max_new_tokens": decision_max_new_tokens,
                "final_max_new_tokens": final_max_new_tokens,
                "max_image_pixels": max_image_pixels,
                "dtype": "float16",
                "quantization": "nf4-double-quant",
                "attention": "eager",
            },
            "environment": {
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "transformers": transformers.__version__,
                "accelerate": accelerate.__version__,
                "bitsandbytes": bitsandbytes.__version__,
                "gpu": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
            },
        }

    @staticmethod
    def _history(messages: list[dict[str, Any]]) -> tuple[str, list[Path]]:
        lines = []
        artifacts: list[Path] = []
        for message in messages:
            role = message.get("role")
            if role == "system":
                continue
            if role == "user":
                lines.append("CASE=" + json.dumps(message.get("content"), ensure_ascii=False, sort_keys=True))
            elif role == "assistant":
                record = {"content": message.get("content", "")}
                if message.get("tool_call") is not None:
                    record["tool_call"] = message["tool_call"]
                lines.append("ASSISTANT_DECISION=" + json.dumps(record, ensure_ascii=False, sort_keys=True))
            elif role == "tool":
                content = message.get("content")
                record = {
                    "trace_id": message.get("tool_call_id"),
                    "name": message.get("name"),
                    "content": content,
                }
                lines.append("TOOL_RESULT=" + json.dumps(record, ensure_ascii=False, sort_keys=True))
                if isinstance(content, dict) and content.get("status") == "completed":
                    path = Path(str(content.get("artifact_path", ""))).resolve()
                    if path.is_file() and path not in artifacts:
                        artifacts.append(path)
        return "\n".join(lines), artifacts

    def _generate(
        self, messages: list[dict[str, Any]], instruction: str, max_new_tokens: int, phase: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        history, artifact_paths = self._history(messages)
        opened: list[Image.Image] = []
        try:
            content: list[dict[str, Any]] = []
            for path in artifact_paths:
                with Image.open(path) as source:
                    image = resize_to_pixel_budget(source.convert("RGB"), self.max_image_pixels)
                opened.append(image)
                content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": instruction + "\n\n" + history})
            model_messages = [{"role": "user", "content": content}]
            inputs = self.processor.apply_chat_template(
                model_messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
            input_tokens = int(inputs["input_ids"].shape[1])
            started = time.perf_counter()
            with self._torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            new_tokens = generated[:, input_tokens:]
            raw_output = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
            parsed = parse_json_object(raw_output)
            call = {
                "phase": phase,
                "raw_output": raw_output,
                "input_tokens": input_tokens,
                "output_tokens": int(new_tokens.shape[1]),
                "latency_seconds": time.perf_counter() - started,
                "artifact_sha256": [sha256_file(path) for path in artifact_paths],
                "do_sample": False,
                "thinking_mode": False,
            }
            return parsed, call
        finally:
            for image in opened:
                image.close()

    def decide(self, messages: list[dict[str, Any]], tools: dict[str, Any]) -> dict[str, Any]:
        instruction = """Choose the next evidence action for this medical benchmark case.
Return exactly one JSON object and no Markdown. Use one of these forms:
{"content":"brief evidence gap","tool_call":{"name":"enabled tool","arguments":{}}}
{"content":"evidence is sufficient","tool_call":null}
The first decision must acquire visual evidence. Never call a tool outside ENABLED_TOOLS.
For single-image media, do not call temporal_skim. A repeated request is invalid.
ENABLED_TOOLS=""" + json.dumps(tools, ensure_ascii=False, sort_keys=True)
        parsed, call = self._generate(
            messages, instruction, self.decision_max_new_tokens, "decision"
        )
        if "tool_call" not in parsed:
            raise ValueError("Decision output is missing tool_call")
        parsed["_model_call"] = call
        return parsed

    def finalize(
        self, messages: list[dict[str, Any]], output_schema: dict[str, Any]
    ) -> dict[str, Any]:
        instruction = """Produce the final evidence-grounded answer. Return exactly one JSON object
and no Markdown. Required fields are sample_id, hypotheses, evidence, answer,
answer_evidence_ids, confidence, insufficient_evidence, and tool_trace_ids. For MCQ,
answer must be one visible option letter. Every evidence item needs evidence_id, media_id,
observation, acquisition, confidence, supports, contradicts, and region_xyxy_1000 (null is
allowed for overview evidence). Cite only completed tool trace IDs. Do not treat tool metadata
as a diagnosis. OUTPUT_SCHEMA=""" + json.dumps(output_schema, sort_keys=True)
        parsed, call = self._generate(messages, instruction, self.final_max_new_tokens, "final")
        parsed["_model_call"] = call
        return parsed
