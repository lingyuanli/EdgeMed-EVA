#!/usr/bin/env python3
"""Verify the V100 CUDA stack and optionally run Qwen3.5-4B 4-bit vision inference."""

from __future__ import annotations

import argparse
import time

import torch


def cuda_smoke() -> None:
    import bitsandbytes as bnb

    assert torch.cuda.is_available(), "CUDA is not available"
    capability = torch.cuda.get_device_capability(0)
    arch_list = torch.cuda.get_arch_list()
    assert capability == (7, 0), f"Expected V100 SM70, got {capability}"
    assert "sm_70" in arch_list, f"PyTorch wheel lacks sm_70: {arch_list}"

    x = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
    y = x @ x
    torch.cuda.synchronize()
    assert torch.isfinite(y).all()

    layer = bnb.nn.Linear4bit(
        128,
        64,
        bias=False,
        compute_dtype=torch.float16,
        quant_type="nf4",
        quant_storage=torch.uint8,
    ).cuda()
    qout = layer(torch.randn((2, 128), device="cuda", dtype=torch.float16))
    torch.cuda.synchronize()
    assert torch.isfinite(qout).all()

    hardware_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"capability={capability}")
    print(f"arch_list={arch_list}")
    print(f"hardware_bf16={hardware_bf16}")
    print("cuda_fp16_nf4_smoke=PASS")


def model_smoke(model_path: str) -> None:
    from PIL import Image, ImageDraw
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
    )

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    started = time.time()
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        local_files_only=True,
        quantization_config=quantization,
        dtype=torch.float16,
        device_map={"": 0},
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    print(f"model_load_seconds={time.time() - started:.2f}")
    print(f"model_footprint_mib={model.get_memory_footprint() / 1024**2:.2f}")

    image = Image.new("RGB", (512, 512), "black")
    draw = ImageDraw.Draw(image)
    draw.ellipse((135, 120, 375, 390), outline="white", width=8)
    draw.rectangle((235, 210, 280, 300), fill="white")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        "Describe the most visible geometric structures in this synthetic "
                        "image in one short sentence. This is a software smoke test, not a "
                        "medical image."
                    ),
                },
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {key: value.to("cuda") for key, value in inputs.items()}

    started = time.time()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=48, do_sample=False)
    new_tokens = generated[:, inputs["input_ids"].shape[1] :]
    output = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    torch.cuda.synchronize()

    print(f"generation_seconds={time.time() - started:.2f}")
    print(f"max_allocated_mib={torch.cuda.max_memory_allocated() / 1024**2:.2f}")
    print(f"output={output}")
    print("qwen35_v100_multimodal_smoke=PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-only", action="store_true")
    parser.add_argument("--model-path", default="/home/ubuntu/models/Qwen3.5-4B")
    args = parser.parse_args()

    cuda_smoke()
    if not args.cuda_only:
        model_smoke(args.model_path)


if __name__ == "__main__":
    main()
