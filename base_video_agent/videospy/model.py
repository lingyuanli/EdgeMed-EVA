import copy
import json
import random
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from litellm import completion

from .metrics import record_llm_response


_active_model_calls: ContextVar[list[dict] | None] = ContextVar(
    "videospy_model_calls",
    default=None,
)


@contextmanager
def collect_model_calls():
    model_calls = []
    token = _active_model_calls.set(model_calls)
    try:
        yield model_calls
    finally:
        _active_model_calls.reset(token)


class ModelClient:
    """Configured wrapper around a chat-completions model."""

    def __init__(self, config: dict) -> None:
        self.config = dict(config)

    def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        return_json: bool = False,
        extra_body: dict | None = None,
    ) -> dict:
        delay = 1.0
        max_retries = self.config.get("max_retries", 8)
        model_input = {
            "model": self.config["model_name"],
            "messages": copy.deepcopy(messages),
        }
        request_extra_body = dict(extra_body or {})
        enable_thinking = self.config.get("enable_thinking")
        if enable_thinking is not None:
            request_extra_body.setdefault("enable_thinking", enable_thinking)

        for attempt in range(max_retries + 1):
            try:
                response = completion(
                    model=self.config["model_name"],
                    messages=messages,
                    api_base=self.config.get("api_base"),
                    api_key=self.config.get("api_key"),
                    api_version=self.config.get("api_version"),
                    max_completion_tokens=self.config.get("max_tokens", 32768),
                    seed=self.config.get("seed", 42),
                    temperature=self.config.get("temperature", 1.0),
                    reasoning_effort=self.config.get("reasoning_effort", "medium"),
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format={"type": "json_object"} if return_json else None,
                    timeout=900,
                    drop_params=True,
                    extra_body=request_extra_body or None,
                )
                record_llm_response(response)
                message = self._message_to_dict(response.choices[0].message)
                model_calls = _active_model_calls.get()
                if model_calls is not None:
                    model_calls.append(
                        {
                            "input": model_input,
                            "output": copy.deepcopy(message),
                        }
                    )
                return message
            except Exception as error:
                if not self._is_retryable(error) or attempt == max_retries:
                    raise
                delay *= 2 * (1 + random.random())
                print(f"Retrying in {delay:.1f} seconds: {error}")
                time.sleep(delay)

        raise RuntimeError("Model request failed")

    @staticmethod
    def _message_to_dict(message: Any) -> dict:
        if hasattr(message, "model_dump"):
            return message.model_dump(exclude_none=True)
        if hasattr(message, "json"):
            return json.loads(message.json())
        if isinstance(message, dict):
            return dict(message)
        raise TypeError(f"Unsupported model message type: {type(message).__name__}")

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        message = str(error).lower()
        return any(
            marker in message
            for marker in (
                "rate limit",
                "timed out",
                "too many requests",
                "forbidden for url",
                "the maximum usage",
                "server had an error",
                "has no attribute 'upper'",
                "internal",
            )
        )
