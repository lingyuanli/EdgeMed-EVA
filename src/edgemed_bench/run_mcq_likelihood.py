"""Answer-isolated MCQ inference by normalized option-text likelihood."""

from __future__ import annotations

import argparse
import hashlib
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
from .qwen_agent_backend import validate_model_source
from .run import resize_to_pixel_budget, select_rows

PROMPT_TEMPLATE = (
    "Please carefully observe this medical image and answer the following question "
    "with the most likely short answer.\n\nQuestion: {question}\n\nAnswer:"
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


def mean_target_logprob(logits: Any, input_ids: Any, prefix_length: int) -> tuple[float, int]:
    """Return mean autoregressive log probability after an exact prefix."""
    import torch

    if input_ids.ndim != 2 or logits.ndim != 3 or input_ids.shape[0] != 1:
        raise ValueError("Expected one batched token sequence and its logits")
    if prefix_length < 1 or prefix_length >= input_ids.shape[1]:
        raise ValueError("Candidate completion must contain at least one token")
    target_ids = input_ids[:, prefix_length:]
    predicting_logits = logits[:, prefix_length - 1 : -1, :]
    if predicting_logits.shape[1] != target_ids.shape[1]:
        raise ValueError("Target/logit alignment failed")
    token_logprobs = torch.log_softmax(predicting_logits.float(), dim=-1).gather(
        -1, target_ids.unsqueeze(-1)
    )
    return float(token_logprobs.mean().item()), int(target_ids.shape[1])


def completed_ids(path: Path, selected: set[str], contract_sha256: str) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    for row in read_jsonl(path):
        sample_id = str(row.get("sample_id"))
        if sample_id not in selected or sample_id in completed:
            raise RuntimeError(f"Prediction is duplicate or outside contract: {sample_id}")
        if row.get("status") != "completed" or row.get("contract_sha256") != contract_sha256:
            raise RuntimeError(f"Existing prediction contract mismatch: {sample_id}")
        completed.add(sample_id)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-source-manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-id-file", type=Path)
    parser.add_argument("--max-image-pixels", type=int)
    parser.add_argument("--sync-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
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

    rows = select_rows(read_jsonl(args.manifest), args.limit, args.sample_id_file)
    reject_reference_fields(rows)
    if not rows or any(row.get("kind") != "mcq" for row in rows):
        raise ValueError("A non-empty MCQ manifest is required")
    option_schemas = {tuple(sorted(row["options"])) for row in rows}
    if len(option_schemas) != 1 or next(iter(option_schemas)) not in {
        tuple("ABCD"), tuple("ABCDE")
    }:
        raise ValueError(f"Unsupported or mixed option schema: {option_schemas}")
    selected_ids = [str(row["sample_id"]) for row in rows]
    contract = {
        "schema_version": "edgemed-mcq-option-likelihood-contract/v1",
        "manifest_sha256": sha256_file(args.manifest),
        "model_source_manifest_sha256": sha256_file(args.model_source_manifest),
        "prompt_sha256": hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest(),
        "scoring": "mean-assistant-option-text-logprob-including-eom",
        "options_visible_in_user_prompt": False,
        "do_sample": False,
        "thinking_mode": False,
        "dtype": "float16",
        "quantization": "nf4-double-quant",
        "attention": "eager",
        "selected_count": len(rows),
        "selected_ids_sha256": hashlib.sha256("\n".join(selected_ids).encode()).hexdigest(),
    }
    if args.max_image_pixels is not None:
        contract["max_image_pixels"] = args.max_image_pixels
        contract["image_resize"] = "aspect-preserving-lanczos"
    contract_sha = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.jsonl"
    manifest_path = run_dir / "run_manifest.json"
    events_path = run_dir / "events.jsonl"
    if predictions_path.exists() and not args.resume:
        raise FileExistsError(f"Predictions already exist; use --resume: {predictions_path}")
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("contract_sha256") != contract_sha:
            raise RuntimeError("Resume contract differs from existing run")
    done = completed_ids(predictions_path, set(selected_ids), contract_sha)
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    source_receipt = validate_model_source(args.model_path, args.model_source_manifest)
    manifest = {
        "schema_version": "edgemed-mcq-option-likelihood-run/v1",
        "run_id": run_dir.name,
        "status": "running",
        "started_at": previous.get("started_at", utc_now()),
        "resume_count": int(previous.get("resume_count", 0)) + int(bool(previous)),
        "code_commit": git_commit(),
        "contract": contract,
        "contract_sha256": contract_sha,
        "model_source": source_receipt,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "accelerate": accelerate.__version__,
            "bitsandbytes": bitsandbytes.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    write_json(manifest_path, manifest)

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
    model.eval()
    load_seconds = time.perf_counter() - load_started
    torch.cuda.reset_peak_memory_stats()
    pending = [row for row in rows if str(row["sample_id"]) not in done]
    with events_path.open("a", encoding="utf-8") as events:
        append_jsonl(
            events,
            {
                "event": "run_resumed" if done else "run_started",
                "time": utc_now(),
                "selected": len(rows),
                "already_completed": len(done),
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
            if not image_path.is_file() or sha256_file(image_path) != row["image_sha256"]:
                raise ValueError(f"Missing or changed image: {row['sample_id']}")
            with Image.open(image_path) as source:
                image = resize_to_pixel_budget(source.convert("RGB"), args.max_image_pixels)
            user_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": PROMPT_TEMPLATE.format(question=row["question"])},
                    ],
                }
            ]
            prefix = processor.apply_chat_template(
                user_messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=True,
                return_tensors="pt",
            )
            prefix_ids = prefix["input_ids"]
            option_scores = {}
            for letter in sorted(row["options"]):
                candidate = str(row["options"][letter]).strip()
                full = processor.apply_chat_template(
                    user_messages + [{"role": "assistant", "content": candidate}],
                    tokenize=True,
                    add_generation_prompt=False,
                    enable_thinking=False,
                    return_dict=True,
                    return_tensors="pt",
                )
                full_ids = full["input_ids"]
                prefix_length = int(prefix_ids.shape[1])
                if full_ids.shape[1] <= prefix_length or not torch.equal(
                    full_ids[:, :prefix_length], prefix_ids
                ):
                    raise RuntimeError(f"Assistant completion prefix mismatch: {row['sample_id']}")
                full = {key: value.to("cuda") for key, value in full.items()}
                with torch.inference_mode():
                    logits = model(**full).logits
                mean_logprob, token_count = mean_target_logprob(
                    logits, full["input_ids"], prefix_length
                )
                option_scores[letter] = {
                    "mean_logprob": mean_logprob,
                    "token_count": token_count,
                    "option_text_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
                }
            ranked = sorted(
                option_scores,
                key=lambda letter: (-option_scores[letter]["mean_logprob"], letter),
            )
            result = {
                "schema_version": "edgemed-mcq-option-likelihood-prediction/v1",
                "sample_id": row["sample_id"],
                "status": "completed",
                "task": row["task"],
                "parsed_answer": ranked[0],
                "parse_status": "option_likelihood_rank",
                "option_scores": option_scores,
                "score_margin": option_scores[ranked[0]]["mean_logprob"]
                - option_scores[ranked[1]]["mean_logprob"],
                "latency_seconds": time.perf_counter() - sample_started,
                "image_sha256": row["image_sha256"],
                "processed_image_size": [image.width, image.height],
                "contract_sha256": contract_sha,
            }
            completed_this_process += 1
            append_jsonl(
                output_handle,
                result,
                sync=(completed_this_process % args.sync_every == 0),
            )
            if position == 1 or position % 10 == 0 or position == len(pending):
                print(f"PROGRESS completed={position}/{len(pending)} sample={row['sample_id']}", flush=True)
    except BaseException as error:
        manifest.update(
            {
                "status": "failed",
                "finished_at": utc_now(),
                "error_type": type(error).__name__,
                "completed_this_process": completed_this_process,
            }
        )
        write_json(manifest_path, manifest)
        raise
    finally:
        output_handle.flush()
        os.fsync(output_handle.fileno())
        output_handle.close()

    manifest.update(
        {
            "status": "completed",
            "finished_at": utc_now(),
            "model_load_seconds": load_seconds,
            "inference_seconds": time.perf_counter() - started,
            "completed_total": len(done) + completed_this_process,
            "max_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "predictions_sha256": sha256_file(predictions_path),
        }
    )
    write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
