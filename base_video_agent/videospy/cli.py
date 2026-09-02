import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.videospy import build_config, prompts_config

from .agent import VideoSpyAgent
from .utils import append_commit_id


def build_agent_config(args: argparse.Namespace) -> dict:
    config_path = getattr(args, "config_path", None)
    prompt_path = getattr(args, "prompt_path", None)
    return build_config(
        args,
        config_paths=[Path(config_path)] if config_path else [],
        prompt_paths=[Path(prompt_path)] if prompt_path else [],
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="videospy")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--user_query", required=True)
    parser.add_argument("--subtitle_path", default=None)
    parser.add_argument("--output_dir", default="./output/")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--config",
        dest="config_path",
        default=None,
        help="Agent YAML config. Defaults to config/videospy/general.yaml.",
    )
    parser.add_argument(
        "--prompt",
        dest="prompt_path",
        default=None,
        help="Prompt YAML config. Defaults to config/videospy/prompts.yaml.",
    )

    parser.add_argument("--model_name", default=None)
    parser.add_argument("--api_base", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--api_version", default=None)
    parser.add_argument("--reasoning_effort", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    video_id = Path(args.video_path).stem
    run_name = append_commit_id(
        f"{video_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = output_dir / "videospy" / "demo" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = build_agent_config(args)
    agent = VideoSpyAgent(
        config=config,
        video_path=args.video_path,
        subtitle_path=args.subtitle_path,
        output_dir=str(run_dir),
        tools=config["tools"]["enabled"],
        verbose=args.verbose,
    )
    trajectory = agent.run(args.user_query).to_dict()
    prediction = trajectory.get("final_answer", "")
    print(f"Question: {args.user_query}")
    print(f"Prediction: {prediction}")

    (run_dir / "prediction.json").write_text(
        json.dumps({"prediction": prediction}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "trajectory.json").write_text(
        json.dumps(trajectory, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
