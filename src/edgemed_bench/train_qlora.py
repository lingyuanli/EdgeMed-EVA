"""Single-V100, answer-only QLoRA smoke/training loop for admitted external MCQs."""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from .io import append_jsonl, read_jsonl, reject_reference_fields, sha256_file, write_json
from .prompts import mcq_prompt, prompt_hash
from .run import resize_to_pixel_budget

LANGUAGE_LORA_PATTERN = (
    r".*model\.language_model\.layers\.\d+\."
    r"(?:linear_attn\.(?:out_proj|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a)"
    r"|self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
    r"|mlp\.(?:gate_proj|up_proj|down_proj))"
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


def assistant_loss_labels(prefix_ids: list[int], full_ids: list[int]) -> list[int]:
    if len(full_ids) <= len(prefix_ids) or full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("Assistant target is not an exact continuation of the prompt tokens")
    return [-100] * len(prefix_ids) + full_ids[len(prefix_ids) :]


def encode_example(processor: Any, row: dict[str, Any], image: Image.Image) -> dict[str, Any]:
    prompt = mcq_prompt(row["question"], row["options"], variant="direct")
    user = {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
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
        [user, {"role": "assistant", "content": [{"type": "text", "text": row["answer"]}]}],
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-source-manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--max-image-pixels", type=int, default=786432)
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
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (7, 0):
        raise RuntimeError("This frozen training route requires one V100 SM70 GPU")

    rows = read_jsonl(args.manifest)
    reject_reference_fields(rows)
    references = {row["sample_id"]: str(row["answer"]).upper() for row in read_jsonl(args.references)}
    if {row["sample_id"] for row in rows} != set(references):
        raise ValueError("Manifest/reference sample IDs differ")
    if any(row.get("kind") != "mcq" for row in rows):
        raise ValueError("T1a QLoRA accepts MCQ inference surfaces only")
    option_schemas = {"".join(sorted(row["options"])) for row in rows}
    if len(option_schemas) != 1 or next(iter(option_schemas)) not in {"ABCD", "ABCDE"}:
        raise ValueError(f"Unsupported or mixed option schemas: {sorted(option_schemas)}")
    option_letters = next(iter(option_schemas))
    for row in rows:
        row["answer"] = references[row["sample_id"]]
        if row["answer"] not in row["options"]:
            raise ValueError(f"Invalid answer for {row['sample_id']}")

    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"Training run dir already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    events_path = run_dir / "events.jsonl"
    contract = {
        "schema_version": "edgemed-t1a-qlora-contract/v1",
        "manifest_sha256": sha256_file(args.manifest),
        "references_sha256": sha256_file(args.references),
        "model_source_manifest_sha256": sha256_file(args.model_source_manifest),
        "prompt_sha256": prompt_hash("mcq", "direct", option_letters),
        "option_letters": option_letters,
        "objective": "assistant-answer-tokens-only",
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
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "accelerate": accelerate.__version__,
            "bitsandbytes": bitsandbytes.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    write_json(run_dir / "run_manifest.json", manifest)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    required_examples = args.max_steps * args.gradient_accumulation
    if required_examples > len(order):
        order = [order[index % len(order)] for index in range(required_examples)]
    else:
        order = order[:required_examples]

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
    scaler = torch.amp.GradScaler("cuda")
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
            row = rows[row_index]
            image_path = (args.data_root / row["image_path"]).resolve()
            if not image_path.is_file() or sha256_file(image_path) != row["image_sha256"]:
                raise ValueError(f"Missing or changed image: {row['sample_id']}")
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            image = resize_to_pixel_budget(image, args.max_image_pixels)
            batch = encode_example(processor, row, image)
            batch = {key: value.to("cuda") for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch, use_cache=False).loss
                scaled_loss = loss / args.gradient_accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at example {example_position}: {loss}")
            scaler.scale(scaled_loss).backward()
            losses.append(float(loss.detach()))
            if example_position % args.gradient_accumulation == 0:
                scaler.unscale_(optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step = example_position // args.gradient_accumulation
                append_jsonl(
                    events,
                    {
                        "event": "optimizer_step",
                        "time": utc_now(),
                        "step": step,
                        "loss": losses[-1],
                        "grad_norm": grad_norm,
                        "peak_cuda_mib": torch.cuda.max_memory_allocated() / 1024**2,
                    },
                    sync=True,
                )

    adapter_dir = run_dir / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    processor.save_pretrained(run_dir / "processor")
    adapter_files = {
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
            "adapter_hashes": adapter_files,
        }
    )
    write_json(run_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
