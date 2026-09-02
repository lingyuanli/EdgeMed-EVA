import argparse
import copy
import json
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from videospy.utils import append_commit_id

MODES = ("smoke", "lite", "full")
AGENTS = ("videoseek", "videospy")
VIDEOSPY_CONFIG_DIR = Path(__file__).with_name("config")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--agent", choices=AGENTS, default="videoseek")
    parser.add_argument("--mode", choices=MODES, default="smoke")
    parser.add_argument(
        "--config",
        dest="config_path",
        default=None,
        help="Agent YAML config override.",
    )
    parser.add_argument(
        "--prompt",
        dest="prompt_path",
        default=None,
        help="VideoSpy prompt YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Parent directory for new benchmark runs (runs land under <output-dir>/<agent>/<dataset>/).",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Existing run directory to resume, or an explicit directory for a new run.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print agent step logs.")

    parser.add_argument("--model-name", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-version", default=None)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)


def build_agent_config(args: argparse.Namespace) -> dict:
    if getattr(args, "agent", "videoseek") == "videospy":
        from config.videospy import build_config

        default_config_path = VIDEOSPY_CONFIG_DIR / "videospy_general.yml"
        config_paths = [default_config_path] if default_config_path.is_file() else []
        prompt_paths = [VIDEOSPY_CONFIG_DIR / "videospy_prompt.yml"]
        if getattr(args, "config_path", None):
            config_paths.append(Path(args.config_path))
        if getattr(args, "prompt_path", None):
            prompt_paths.append(Path(args.prompt_path))
        return build_config(
            args,
            config_paths=config_paths,
            prompt_paths=prompt_paths,
        )

    if getattr(args, "prompt_path", None):
        raise ValueError("--prompt is only supported with --agent videospy.")
    from config.videoseek import (
        general_config,
        init_config,
        load_config,
        prompts_config,
    )

    config_path = getattr(args, "config_path", None)
    base_config = (
        load_config(Path(config_path).expanduser()) if config_path else general_config
    )
    config = copy.deepcopy(base_config)
    config.update(copy.deepcopy(prompts_config))
    return init_config(config, args)


def get_agent_class(agent_name: str):
    if agent_name == "videospy":
        from videospy.agent import VideoSpyAgent

        return VideoSpyAgent
    from videoseek.agent import VideoSeekAgent

    return VideoSeekAgent


def get_agent_tools(config: dict, agent_name: str) -> list[str]:
    if agent_name == "videospy":
        return config["tools"]["enabled"]
    return config["tools"]


def get_agent_model_name(config: dict, agent_name: str) -> str:
    if agent_name == "videospy":
        return str(config["agent"]["model"].get("model_name", ""))
    return str(config.get("model_name", ""))


def create_run_dir(
    dataset_name: str,
    mode: str,
    output_dir: str,
    run_dir: str | None,
    agent_name: str = "videoseek",
    model_name: str = "",
) -> Path:
    if run_dir is not None:
        path = Path(run_dir).expanduser().resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_segment = "".join(
            c if c.isalnum() or c in ("-", "_", ".") else "_"
            for c in str(model_name)
        )
        base_run_name = (
            f"{mode}_{model_segment}_{timestamp}"
            if model_segment
            else f"{mode}_{timestamp}"
        )
        run_name = (
            append_commit_id(base_run_name)
            if agent_name == "videospy"
            else base_run_name
        )
        path = (
            Path(output_dir).expanduser().resolve()
            / agent_name
            / dataset_name
            / run_name
        )
    path.mkdir(parents=True, exist_ok=True)
    (path / "trajectories").mkdir(exist_ok=True)
    return path


def safe_config(config: dict) -> dict:
    def is_sensitive(key: str) -> bool:
        lowered = key.lower()
        return (
            lowered in {"key", "token", "api_token", "access_token"}
            or lowered.endswith("_key")
            or "secret" in lowered
        )

    def redact(value):
        if isinstance(value, dict):
            return {
                key: redact(item)
                for key, item in value.items()
                if not is_sensitive(key)
            }
        if isinstance(value, list):
            return [redact(item) for item in value]
        return value

    return redact(config)


def ensure_run_manifest(run_dir: Path, manifest: dict) -> None:
    path = run_dir / "run.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable_manifest = manifest
        if "agent" not in existing and manifest.get("agent") == "videoseek":
            comparable_manifest = copy.deepcopy(manifest)
            comparable_manifest.pop("agent")
            comparable_manifest.get("agent_config", {}).pop("max_tokens", None)
        if existing != comparable_manifest:
            raise ValueError(
                f"Run configuration does not match existing manifest: {path}"
            )
        return
    write_json(path, manifest)


def load_latest_records(path: Path) -> dict[str, dict]:
    latest = {}
    if not path.exists():
        return latest
    lines = path.read_text(encoding="utf-8").splitlines()
    nonempty_lines = [line for line in lines if line.strip()]
    for line_number, line in enumerate(nonempty_lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if line_number == len(nonempty_lines) - 1:
                repaired = "\n".join(nonempty_lines[:-1])
                path.write_text(f"{repaired}\n" if repaired else "", encoding="utf-8")
                continue
            raise
        latest[str(record["sample_key"])] = record
    return latest


def append_record(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()


def has_task_metrics(record: dict) -> bool:
    return all(
        key in record for key in ("agent_rounds", "token_usage", "elapsed_seconds")
    )


def is_completed_record(record: dict) -> bool:
    return record.get("status") == "success" and has_task_metrics(record)


def attach_task_metrics(
    record: dict,
    agent,
    trajectory: dict | None,
    elapsed_seconds: float,
) -> None:
    fallback_rounds = 0
    if trajectory is not None:
        fallback_rounds = trajectory.get(
            "total_steps",
            len(trajectory.get("steps", [])),
        )
    empty_token_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "llm_calls": 0,
        "calls_without_usage": 0,
    }
    record["agent_rounds"] = getattr(agent, "last_run_rounds", fallback_rounds)
    token_usage = empty_token_usage.copy()
    token_usage.update(getattr(agent, "last_run_token_usage", {}) or {})
    record["token_usage"] = token_usage
    record["elapsed_seconds"] = round(elapsed_seconds, 3)


def aggregate_task_metrics(
    selected_keys: list[str], latest_records: dict[str, dict]
) -> dict:
    records = [
        latest_records[str(key)]
        for key in selected_keys
        if has_task_metrics(latest_records.get(str(key), {}))
    ]
    measured_tasks = len(records)
    token_fields = (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "llm_calls",
        "calls_without_usage",
    )
    token_totals = {
        field: sum(record["token_usage"].get(field, 0) for record in records)
        for field in token_fields
    }
    total_rounds = sum(record["agent_rounds"] for record in records)
    total_elapsed = round(sum(record["elapsed_seconds"] for record in records), 3)
    return {
        "measured_tasks": measured_tasks,
        "agent_rounds": {
            "total": total_rounds,
            "average": total_rounds / measured_tasks if measured_tasks else 0.0,
        },
        "token_usage": token_totals,
        "elapsed_seconds": {
            "total": total_elapsed,
            "average": total_elapsed / measured_tasks if measured_tasks else 0.0,
        },
    }


class _TeeStream:
    def __init__(self, console, log_file):
        self.console = console
        self.log_file = log_file

    def write(self, text):
        self.console.write(text)
        self.log_file.write(text)
        return len(text)

    def flush(self):
        self.console.flush()
        self.log_file.flush()

    def isatty(self):
        return self.console.isatty()

    def fileno(self):
        return self.console.fileno()

    @property
    def encoding(self):
        return self.console.encoding


@contextmanager
def tee_output(path: Path):
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with path.open("a", encoding="utf-8", buffering=1) as log_file:
        sys.stdout = _TeeStream(original_stdout, log_file)
        sys.stderr = _TeeStream(original_stderr, log_file)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def write_json(path: Path, value) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def write_trajectory(run_dir: Path, sample_key: str, trajectory: dict) -> str:
    filename = "".join(
        character if character.isalnum() or character in ("-", "_") else "_"
        for character in str(sample_key)
    )
    relative_path = Path("trajectories") / f"{filename}.json"
    write_json(run_dir / relative_path, trajectory)
    return str(relative_path)
