#!/usr/bin/env python3
"""Download a Hugging Face snapshot sequentially without printing credentials."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def load_token(env_file: Path | None) -> None:
    if env_file is None:
        return
    for raw in env_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        if key in {
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "HUGGINGFACE_HUB_TOKEN",
        }:
            os.environ["HF_TOKEN"] = value.strip().strip("'\"")
            return
    raise RuntimeError(f"No recognized Hugging Face token variable found in {env_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args()

    load_token(args.env_file)

    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        revision=args.revision,
        max_workers=args.max_workers,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"snapshot_ready={path}")


if __name__ == "__main__":
    main()
