import argparse
import copy
import os
from pathlib import Path
from typing import Iterable

import yaml


MODEL_KEYS = {
    "model_name",
    "api_base",
    "api_key",
    "api_version",
    "max_tokens",
    "reasoning_effort",
    "seed",
    "temperature",
    "max_retries",
}
CONFIG_DIR = Path(__file__).resolve().parent


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return config


def merge_configs(base: dict, *overrides: dict) -> dict:
    merged = copy.deepcopy(base)
    for override in overrides:
        for key, value in override.items():
            if isinstance(merged.get(key), dict) and isinstance(value, dict):
                merged[key] = merge_configs(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
    return merged


def _general_config_path() -> Path:
    local_path = CONFIG_DIR / "general.yaml"
    return local_path if local_path.is_file() else CONFIG_DIR / "general.example.yaml"


def _resolve_model_environment(model_config: dict) -> None:
    for key in ("api_base", "api_key", "api_version"):
        value = model_config.get(key)
        if isinstance(value, str):
            model_config[key] = os.getenv(value) or value


def init_config(config: dict, args: argparse.Namespace) -> dict:
    config = copy.deepcopy(config)
    agent_config = config["agent"]
    agent_model = agent_config["model"]

    for key in MODEL_KEYS:
        value = getattr(args, key, None)
        if value is not None:
            agent_model[key] = value
    if getattr(args, "max_steps", None) is not None:
        agent_config["max_steps"] = args.max_steps

    _resolve_model_environment(agent_model)
    for tool_name in config["tools"]["enabled"]:
        tool_config = config["tools"][tool_name]
        if "model" in tool_config:
            _resolve_model_environment(tool_config["model"])
        for nested_config in tool_config.values():
            if isinstance(nested_config, dict) and "model" in nested_config:
                _resolve_model_environment(nested_config["model"])
    return config


def build_config(
    args: argparse.Namespace,
    config_paths: Iterable[Path] = (),
    prompt_paths: Iterable[Path] = (),
) -> dict:
    config_overrides = [load_config(Path(path).expanduser()) for path in config_paths]
    prompt_overrides = [load_config(Path(path).expanduser()) for path in prompt_paths]
    config = merge_configs(general_config, *config_overrides)
    prompts = merge_configs(prompts_config, *prompt_overrides)
    return init_config(merge_configs(config, prompts), args)


general_config = load_config(_general_config_path())
prompts_config = load_config(CONFIG_DIR / "prompts.yaml")

__all__ = [
    "build_config",
    "general_config",
    "init_config",
    "load_config",
    "merge_configs",
    "prompts_config",
]
