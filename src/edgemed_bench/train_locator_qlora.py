"""Single-V100 QLoRA for question-conditioned medical region localization."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

from PIL import Image

from .io import append_jsonl, read_jsonl, reject_reference_fields, sha256_file, write_json
from .medical_agent_tools import TOOL_SCHEMAS
from .qwen_agent_backend import LOCALIZATION_CONTRACT, validate_model_source
from .run import resize_to_pixel_budget
from .train_qlora import (
    LANGUAGE_LORA_PATTERN,
    assistant_loss_labels,
    git_commit,
    require_finite_gradient_norm,
    utc_now,
)


def localization_instruction() -> str:
    return LOCALIZATION_CONTRACT + "\nREGION_TOOL=" + json.dumps(
        TOOL_SCHEMAS["region_inspect"], ensure_ascii=False, sort_keys=True
    )


def localization_prompt(row: dict[str, Any]) -> str:
    case = {
        "sample_id": row["sample_id"],
        "question_type": "localization",
        "question": row["question"],
        "options": None,
        "clinical_context": "",
        "media": [
            {
                "media_id": "image-0",
                "kind": "image",
                "path": row["image_path"],
                "sha256": row["image_sha256"],
                "modality": "unknown",
                "view": "unknown",
                "timepoint": "unknown",
            }
        ],
    }
    return localization_instruction() + "\n\nCASE=" + json.dumps(
        case, ensure_ascii=False, sort_keys=True
    )


def localization_target(target: dict[str, Any]) -> str:
    tool_call = target.get("tool_call")
    if not isinstance(tool_call, dict) or tool_call.get("name") != "region_inspect":
        raise ValueError("Locator target must contain region_inspect")
    arguments = tool_call.get("arguments")
    if not isinstance(arguments, dict) or arguments.get("media_id") != "image-0":
        raise ValueError("Locator target must bind media_id=image-0")
    box = arguments.get("region_xyxy_1000")
    if (
        not isinstance(box, list)
        or len(box) != 4
        or any(not isinstance(value, int) for value in box)
    ):
        raise ValueError("Locator target must contain four integer coordinates")
    x1, y1, x2, y2 = box
    area = (x2 - x1) * (y2 - y1) / 1_000_000
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        raise ValueError("Locator target coordinates are out of bounds")
    if not 0.01 <= area <= 0.64:
        raise ValueError("Locator target area is outside the frozen interval")
    target_label = str(target.get("target_label") or arguments.get("target") or "region")
    return json.dumps(
        {"content": f"inspect {target_label}", "tool_call": tool_call},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def encode_locator_example(
    processor: Any,
    row: dict[str, Any],
    target: dict[str, Any],
    image: Image.Image,
) -> dict[str, Any]:
    user = {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": localization_prompt(row)},
        ],
    }
    prefix = processor.apply_chat_template(
        [user],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    )
    full = processor.apply_chat_template(
        [
            user,
            {
                "role": "assistant",
                "content": [{"type": "text", "text": localization_target(target)}],
            },
        ],
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    )
    prefix_ids = prefix["input_ids"][0].tolist()
    full_ids = full["input_ids"][0].tolist()
    labels = assistant_loss_labels(prefix_ids, full_ids)
    import torch

    full["labels"] = torch.tensor([labels], dtype=torch.long)
    return full


def load_training_rows(
    manifest_path: Path, targets_path: Path
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    rows = read_jsonl(manifest_path)
    reject_reference_fields(rows)
    targets_list = read_jsonl(targets_path)
    targets = {str(row["sample_id"]): row for row in targets_list}
    if len(targets) != len(targets_list):
        raise ValueError("Duplicate locator target sample ids")
    if {str(row["sample_id"]) for row in rows} != set(targets):
        raise ValueError("Locator manifest and target sample ids differ")
    if any(row.get("kind") != "localization" for row in rows):
        raise ValueError("Locator QLoRA accepts localization surfaces only")
    paired = [(row, targets[str(row["sample_id"])]) for row in rows]
    for _, target in paired:
        localization_target(target)
    return paired


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-source-manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--max-image-pixels", type=int, default=786_432)
    parser.add_argument("--grad-scaler-init-scale", type=float, default=1.0)
    args = parser.parse_args()

    import accelerate
    import bitsandbytes
    import peft
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    if args.max_steps <= 0 or args.gradient_accumulation <= 0:
        raise ValueError("max-steps and gradient-accumulation must be positive")
    if args.max_image_pixels <= 0:
        raise ValueError("max-image-pixels must be positive")
    if not math.isfinite(args.grad_scaler_init_scale) or args.grad_scaler_init_scale <= 0:
        raise ValueError("grad-scaler-init-scale must be finite and positive")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (7, 0):
        raise RuntimeError("This frozen locator route requires one V100 SM70 GPU")

    paired = load_training_rows(args.manifest, args.targets)
    model_source = validate_model_source(args.model_path, args.model_source_manifest)
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"Training run dir already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    events_path = run_dir / "events.jsonl"
    contract = {
        "schema_version": "edgemed-locator-qlora-contract/v1",
        "manifest_sha256": sha256_file(args.manifest),
        "targets_sha256": sha256_file(args.targets),
        "model_source_manifest_sha256": sha256_file(args.model_source_manifest),
        "localization_contract_sha256": __import__("hashlib").sha256(
            localization_instruction().encode()
        ).hexdigest(),
        "objective": "assistant-region-tool-json-tokens-only",
        "max_steps": args.max_steps,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": 0.05,
        "lora_target_regex": LANGUAGE_LORA_PATTERN,
        "vision_encoder_trainable": False,
        "projector_trainable": False,
        "base_quantization": "nf4-double-quant",
        "compute_dtype": "float16",
        "micro_batch": 1,
        "max_image_pixels": args.max_image_pixels,
        "image_resize": "aspect-preserving-lanczos",
        "grad_scaler_init_scale": args.grad_scaler_init_scale,
        "seed": args.seed,
    }
    contract_sha = __import__("hashlib").sha256(
        json.dumps(contract, sort_keys=True).encode()
    ).hexdigest()
    manifest = {
        "schema_version": "edgemed-training-run/v1",
        "status": "running",
        "started_at": utc_now(),
        "code_commit": git_commit(),
        "contract": contract,
        "contract_sha256": contract_sha,
        "model_source": model_source,
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "accelerate": accelerate.__version__,
            "bitsandbytes": bitsandbytes.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    write_json(run_dir / "run_manifest.json", manifest)

    order = list(range(len(paired)))
    random.Random(args.seed).shuffle(order)
    required_examples = args.max_steps * args.gradient_accumulation
    order = [order[index % len(order)] for index in range(required_examples)]
    torch.manual_seed(args.seed)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
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
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=LANGUAGE_LORA_PATTERN,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.config.use_cache = False
    model.train()
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    if trainable <= 0:
        raise RuntimeError("No trainable LoRA parameters were created")
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=args.grad_scaler_init_scale)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    losses: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    with events_path.open("w", encoding="utf-8") as events:
        append_jsonl(
            events,
            {
                "event": "training_started",
                "time": utc_now(),
                "contract_sha256": contract_sha,
                "trainable_parameters": trainable,
                "total_parameters": total,
            },
            sync=True,
        )
        for example_position, row_index in enumerate(order, 1):
            row, target = paired[row_index]
            image_path = (args.data_root / row["image_path"]).resolve()
            if not image_path.is_file() or sha256_file(image_path) != row["image_sha256"]:
                raise ValueError(f"Missing or changed image: {row['sample_id']}")
            with Image.open(image_path) as source:
                image = resize_to_pixel_budget(source.convert("RGB"), args.max_image_pixels)
            batch = encode_locator_example(processor, row, target, image)
            batch = {key: value.to("cuda") for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch, use_cache=False).loss
                scaled_loss = loss / args.gradient_accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at example {example_position}: {loss}"
                )
            scaler.scale(scaled_loss).backward()
            losses.append(float(loss.detach()))
            if example_position % args.gradient_accumulation == 0:
                step = example_position // args.gradient_accumulation
                scale_before = float(scaler.get_scale())
                scaler.unscale_(optimizer)
                grad_norm = require_finite_gradient_norm(
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0, error_if_nonfinite=True
                    ),
                    step,
                )
                scaler.step(optimizer)
                scaler.update()
                scale_after = float(scaler.get_scale())
                if scale_after < scale_before:
                    raise FloatingPointError(
                        f"GradScaler skipped optimizer step {step}: "
                        f"scale {scale_before} -> {scale_after}"
                    )
                optimizer.zero_grad(set_to_none=True)
                append_jsonl(
                    events,
                    {
                        "event": "optimizer_step",
                        "time": utc_now(),
                        "step": step,
                        "loss": losses[-1],
                        "grad_norm": grad_norm,
                        "grad_scale_before": scale_before,
                        "grad_scale_after": scale_after,
                        "optimizer_step_applied": True,
                        "peak_cuda_mib": torch.cuda.max_memory_allocated() / 1024**2,
                    },
                    sync=True,
                )

    adapter_dir = run_dir / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    processor.save_pretrained(run_dir / "processor")
    adapter_hashes = {
        path.relative_to(run_dir).as_posix(): sha256_file(path)
        for path in sorted(adapter_dir.rglob("*"))
        if path.is_file()
    }
    manifest.update(
        {
            "status": "completed",
            "completed_at": utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "optimizer_steps": args.max_steps,
            "examples_seen": len(order),
            "loss": {
                "first": losses[0],
                "last": losses[-1],
                "mean": sum(losses) / len(losses),
                "finite": all(math.isfinite(value) for value in losses),
            },
            "parameters": {"trainable": trainable, "total": total},
            "peak_cuda_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "adapter_hashes": adapter_hashes,
        }
    )
    write_json(run_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
