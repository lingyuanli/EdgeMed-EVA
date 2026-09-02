"""Qwen3.5-4B direct Med-CMR inference runner with exact-resume JSONL outputs."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from .io import append_jsonl, read_jsonl, reject_reference_fields, sha256_file, write_json
from .parsing import (
    parse_evidence_answer_mcq,
    parse_mcq,
    parse_open,
    parse_open_answer_only,
    parse_structured_mcq,
)
from .prompts import (
    MCQ_PROMPT_VARIANTS,
    OPEN_PROMPT_VARIANTS,
    mcq_prompt,
    open_prompt,
    prompt_hash,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def select_rows(
    rows: list[dict[str, Any]],
    limit: int | None,
    sample_id_file: Path | None,
) -> list[dict[str, Any]]:
    if sample_id_file is not None:
        requested = [line.strip() for line in sample_id_file.read_text().splitlines() if line.strip()]
        by_id = {row["sample_id"]: row for row in rows}
        missing = [sample_id for sample_id in requested if sample_id not in by_id]
        if missing:
            raise KeyError(f"Unknown sample ids: {missing[:10]}")
        rows = [by_id[sample_id] for sample_id in requested]
    if limit is not None:
        rows = rows[:limit]
    return rows


def resize_to_pixel_budget(image: Image.Image, max_pixels: int | None) -> Image.Image:
    if max_pixels is None or image.width * image.height <= max_pixels:
        return image
    scale = (max_pixels / (image.width * image.height)) ** 0.5
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    resized = image.copy()
    resized.thumbnail(size, Image.Resampling.LANCZOS)
    return resized


def build_prompt(row: dict[str, Any], kind: str, prompt_variant: str = "direct") -> str:
    if kind == "mcq":
        return mcq_prompt(row["question"], row["options"], variant=prompt_variant)
    return open_prompt(row["question"], variant=prompt_variant)


def completed_ids(
    predictions_path: Path,
    selected_ids: set[str],
    contract_sha256: str,
) -> set[str]:
    if not predictions_path.exists():
        return set()
    completed: set[str] = set()
    for row in read_jsonl(predictions_path):
        sample_id = row.get("sample_id")
        if sample_id not in selected_ids:
            raise RuntimeError(f"Existing prediction is outside the selected contract: {sample_id}")
        if sample_id in completed:
            raise RuntimeError(f"Duplicate existing prediction: {sample_id}")
        if row.get("status") != "completed":
            raise RuntimeError(f"Existing prediction is not completed: {sample_id}")
        if row.get("contract_sha256") != contract_sha256:
            raise RuntimeError(f"Existing prediction contract mismatch: {sample_id}")
        completed.add(sample_id)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("mcq", "open"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-source-manifest", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--adapter-source-manifest", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-id-file", type=Path)
    parser.add_argument(
        "--prompt-variant",
        choices=tuple(dict.fromkeys(MCQ_PROMPT_VARIANTS + OPEN_PROMPT_VARIANTS)),
        default="direct",
    )
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-image-pixels", type=int)
    parser.add_argument("--sync-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (args.adapter_path is None) != (args.adapter_source_manifest is None):
        raise ValueError("adapter-path and adapter-source-manifest must be provided together")
    if args.adapter_path is not None:
        adapter_source = json.loads(args.adapter_source_manifest.read_text())
        if adapter_source.get("status") != "completed" or not adapter_source.get("adapter_hashes"):
            raise ValueError("Adapter source manifest is not a completed hash-bound training run")
        expected_adapter_path = (args.adapter_source_manifest.parent / "adapter").resolve()
        if args.adapter_path.resolve() != expected_adapter_path:
            raise ValueError("Adapter path is not bound to its source training run")
        for relative_path, expected_sha256 in adapter_source["adapter_hashes"].items():
            artifact_path = (args.adapter_source_manifest.parent / relative_path).resolve()
            if not artifact_path.is_file() or sha256_file(artifact_path) != expected_sha256:
                raise ValueError(f"Adapter artifact missing or changed: {relative_path}")
    if args.max_image_pixels is not None and args.max_image_pixels <= 0:
        raise ValueError("max-image-pixels must be positive")

    import accelerate
    import bitsandbytes
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability(0) != (7, 0):
        raise RuntimeError(f"Expected V100 SM70, got {torch.cuda.get_device_capability(0)}")

    rows = read_jsonl(args.manifest)
    reject_reference_fields(rows)
    if any(row.get("kind") != args.kind for row in rows):
        raise ValueError("Manifest kind mismatch")
    if args.kind == "mcq" and args.prompt_variant not in MCQ_PROMPT_VARIANTS:
        raise ValueError("Unsupported MCQ prompt variant")
    if args.kind == "open" and args.prompt_variant not in OPEN_PROMPT_VARIANTS:
        raise ValueError("Unsupported open prompt variant")
    rows = select_rows(rows, args.limit, args.sample_id_file)
    if not rows:
        raise ValueError("No rows selected")
    option_letters = "ABCDE"
    if args.kind == "mcq":
        option_schemas = {"".join(sorted(row["options"])) for row in rows}
        if len(option_schemas) != 1:
            raise ValueError(f"Mixed MCQ option schemas: {sorted(option_schemas)}")
        option_letters = next(iter(option_schemas))
        if option_letters not in {"ABCD", "ABCDE"}:
            raise ValueError(f"Unsupported MCQ option schema: {option_letters}")

    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.jsonl"
    events_path = run_dir / "events.jsonl"
    run_manifest_path = run_dir / "run_manifest.json"
    if predictions_path.exists() and not args.resume:
        raise FileExistsError(f"Predictions already exist; use --resume: {predictions_path}")

    max_new_tokens = args.max_new_tokens or (64 if args.kind == "mcq" else 512)
    selected_ids = [row["sample_id"] for row in rows]
    selected_ids_sha256 = __import__("hashlib").sha256(
        "\n".join(selected_ids).encode()
    ).hexdigest()
    contract = {
        "kind": args.kind,
        "manifest_sha256": sha256_file(args.manifest),
        "model_path": str(args.model_path.resolve()),
        "model_source_manifest_sha256": sha256_file(args.model_source_manifest),
        "prompt_sha256": prompt_hash(args.kind, args.prompt_variant, option_letters),
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "thinking_mode": False,
        "dtype": "float16",
        "quantization": "nf4-double-quant",
        "attention": "eager",
        "selected_count": len(selected_ids),
        "selected_sample_ids_sha256": selected_ids_sha256,
    }
    # Preserve the historical B0 contract byte-for-byte. Only non-default variants
    # add a new field, so an exact B0 resume remains possible.
    if args.prompt_variant != "direct":
        contract["prompt_variant"] = args.prompt_variant
    if option_letters != "ABCDE":
        contract["option_letters"] = option_letters
    if args.max_image_pixels is not None:
        contract["max_image_pixels"] = args.max_image_pixels
        contract["image_resize"] = "aspect-preserving-lanczos"
    if args.adapter_path is not None:
        contract["adapter_path"] = str(args.adapter_path.resolve())
        contract["adapter_source_manifest_sha256"] = sha256_file(args.adapter_source_manifest)
    contract_sha = __import__("hashlib").sha256(
        json.dumps(contract, sort_keys=True).encode()
    ).hexdigest()
    if run_manifest_path.exists():
        existing = json.loads(run_manifest_path.read_text())
        if existing.get("contract_sha256") != contract_sha:
            raise RuntimeError("Resume contract differs from existing run")
    done = completed_ids(predictions_path, set(selected_ids), contract_sha)

    previous_manifest = json.loads(run_manifest_path.read_text()) if run_manifest_path.exists() else {}
    manifest = {
        "schema_version": "edgemed-run/v1",
        "run_id": run_dir.name,
        "status": "running",
        "started_at": previous_manifest.get("started_at", utc_now()),
        "resume_count": int(previous_manifest.get("resume_count", 0)) + int(bool(previous_manifest)),
        "code_commit": git_commit(),
        "contract": contract,
        "contract_sha256": contract_sha,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "accelerate": accelerate.__version__,
            "bitsandbytes": bitsandbytes.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "hardware_bf16": torch.cuda.is_bf16_supported(including_emulation=False),
        },
    }
    write_json(run_manifest_path, manifest)

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        local_files_only=True,
        quantization_config=quantization,
        dtype=torch.float16,
        device_map={"": 0},
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    if args.adapter_path is not None:
        import peft
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=False)
        manifest["environment"]["peft"] = peft.__version__
    model.eval()
    model_load_seconds = time.perf_counter() - load_started
    torch.cuda.reset_peak_memory_stats()

    pending = [row for row in rows if row["sample_id"] not in done]
    with events_path.open("a", encoding="utf-8") as events:
        append_jsonl(
            events,
            {
                "event": "run_started" if not done else "run_resumed",
                "time": utc_now(),
                "selected": len(rows),
                "already_completed": len(done),
                "pending": len(pending),
                "contract_sha256": contract_sha,
            },
            sync=True,
        )

    output_handle = predictions_path.open("a", encoding="utf-8")
    started = time.perf_counter()
    completed_this_process = 0
    try:
        for position, row in enumerate(pending, 1):
            sample_started = time.perf_counter()
            image_path = (args.data_root / row["image_path"]).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            if sha256_file(image_path) != row["image_sha256"]:
                raise ValueError(f"Image hash mismatch: {row['sample_id']}")

            with Image.open(image_path) as source:
                image = source.convert("RGB")
            image = resize_to_pixel_budget(image, args.max_image_pixels)
            prompt = build_prompt(row, args.kind, args.prompt_variant)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
            input_tokens = int(inputs["input_ids"].shape[1])
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            new_tokens = generated[:, input_tokens:]
            raw_output = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
            output_tokens = int(new_tokens.shape[1])
            if args.kind == "mcq" and args.prompt_variant == "evidence_answer_v2":
                parsed_answer, observation, parse_status = parse_evidence_answer_mcq(raw_output)
                parsed = {
                    "parsed_answer": parsed_answer,
                    "parsed_observation": observation,
                }
            elif args.kind == "mcq" and args.prompt_variant == "structured_evidence":
                parsed_answer, observation, hypotheses, parse_status = parse_structured_mcq(
                    raw_output
                )
                parsed = {
                    "parsed_answer": parsed_answer,
                    "parsed_observation": observation,
                    "parsed_hypotheses": hypotheses,
                }
            elif args.kind == "mcq":
                parsed_answer, parse_status = parse_mcq(raw_output)
                parsed: dict[str, Any] = {"parsed_answer": parsed_answer}
            else:
                if args.prompt_variant == "answer_only":
                    reasoning, parsed_answer, parse_status = parse_open_answer_only(raw_output)
                else:
                    reasoning, parsed_answer, parse_status = parse_open(raw_output)
                parsed = {"parsed_reasoning": reasoning, "parsed_answer": parsed_answer}

            result = {
                "schema_version": "edgemed-prediction/v1",
                "sample_id": row["sample_id"],
                "status": "completed",
                "task": row["task"],
                "raw_output": raw_output,
                "parse_status": parse_status,
                **parsed,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_seconds": time.perf_counter() - sample_started,
                "image_sha256": row["image_sha256"],
                "processed_image_size": [image.width, image.height],
                "prompt_sha256": contract["prompt_sha256"],
                "contract_sha256": contract_sha,
            }
            completed_this_process += 1
            append_jsonl(
                output_handle,
                result,
                sync=(completed_this_process % args.sync_every == 0),
            )
            if position == 1 or position % 10 == 0 or position == len(pending):
                elapsed = time.perf_counter() - started
                print(
                    f"PROGRESS completed={position}/{len(pending)} "
                    f"seconds={elapsed:.2f} last={row['sample_id']}",
                    flush=True,
                )
    except BaseException as error:
        manifest.update(
            {
                "status": "failed",
                "finished_at": utc_now(),
                "error_type": type(error).__name__,
                "completed_this_process": completed_this_process,
            }
        )
        write_json(run_manifest_path, manifest)
        with events_path.open("a", encoding="utf-8") as events:
            append_jsonl(
                events,
                {
                    "event": "run_failed",
                    "time": utc_now(),
                    "error_type": type(error).__name__,
                    "completed_this_process": completed_this_process,
                },
                sync=True,
            )
        raise
    finally:
        output_handle.flush()
        os.fsync(output_handle.fileno())
        output_handle.close()

    total_seconds = time.perf_counter() - started
    manifest.update(
        {
            "status": "completed",
            "finished_at": utc_now(),
            "model_load_seconds": model_load_seconds,
            "inference_seconds": total_seconds,
            "completed_total": len(done) + completed_this_process,
            "max_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "predictions_sha256": sha256_file(predictions_path),
        }
    )
    write_json(run_manifest_path, manifest)
    with events_path.open("a", encoding="utf-8") as events:
        append_jsonl(
            events,
            {
                "event": "run_completed",
                "time": utc_now(),
                "completed_total": manifest["completed_total"],
                "inference_seconds": manifest["inference_seconds"],
                "predictions_sha256": manifest["predictions_sha256"],
                "contract_sha256": contract_sha,
            },
            sync=True,
        )


if __name__ == "__main__":
    main()
