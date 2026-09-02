import json
import traceback
from abc import ABC, abstractmethod

from decord import VideoReader

from .core import Trajectory
from .metrics import track_agent_usage
from .model import ModelClient, collect_model_calls
from .tools import DEFAULT_TOOL_REGISTRY
from .utils import (
    convert_to_free_form_text_representation,
    load_subtitles,
    parse_video_timestamp,
)


class VisualEvidenceRequiredError(RuntimeError):
    pass


class BaseAgent(ABC):
    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.final_answer = None
        self.question = None
        self.last_run_rounds = 0
        self.last_run_token_usage = {}

    def reset(self) -> None:
        self.messages = self.construct_initial_messages()
        self.final_answer = None
        self.question = None
        self.last_run_rounds = 0

    @abstractmethod
    def construct_initial_messages(self) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def run(self, question: str) -> Trajectory:
        raise NotImplementedError


class VideoSpyAgent(BaseAgent):
    def __init__(
        self,
        config: dict,
        video_path: str,
        subtitle_path: str | None,
        output_dir: str,
        tools: list[str],
        verbose: bool = False,
        video_reader=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.video_path = video_path
        self.vr = video_reader if video_reader is not None else VideoReader(video_path)
        self.tool_registry = DEFAULT_TOOL_REGISTRY
        self.tools = self.tool_registry.resolve_tools(tools)
        self.output_dir = output_dir
        self.verbose = verbose

        self.duration = round(len(self.vr) / self.vr.get_avg_fps(), 2)
        self.subtitles = load_subtitles(subtitle_path)
        self.model_client = ModelClient(config["agent"]["model"])
        self.max_steps = config["agent"]["max_steps"]
        self._successful_tool_requests: set[tuple] = set()
        self.messages = self.construct_initial_messages()

    def reset(self) -> None:
        super().reset()
        self._successful_tool_requests.clear()

    def construct_initial_messages(self) -> list[dict]:
        system_prompt = self.config["SYSTEM_PROMPT"].format(
            overview_num_frames=self.config["tools"]["overview"]["num_frames"],
            clip_skim_num_frames=self.config["tools"]["clip_skim"]["num_frames"],
        )
        return [{"role": "system", "content": system_prompt}]

    def _tool_request_key(
        self, function_name: str, parameters: dict
    ) -> tuple | None:
        if function_name == "overview":
            return (function_name,)
        if function_name == "clip_skim":
            return (
                function_name,
                parse_video_timestamp(parameters["start_time"]),
                parse_video_timestamp(parameters["end_time"]),
            )
        if function_name == "frame_inspect":
            query = " ".join(str(parameters["query"]).split()).casefold()
            return (
                function_name,
                parse_video_timestamp(parameters["timestamp"]),
                query,
            )
        return None

    def _execute_tool_call(self, tool_call: dict) -> tuple[dict, bool]:
        function = tool_call.get("function", {})
        function_name = function.get("name", "")
        model_calls = []
        succeeded = False

        try:
            parameters = json.loads(function.get("arguments", "{}"))
            if not isinstance(parameters, dict):
                raise ValueError("Tool arguments must be a JSON object")
            if not self.tool_registry.has_tool(function_name):
                raise ValueError(f"Invalid function name: {function_name}")

            request_key = self._tool_request_key(function_name, parameters)
            if request_key in self._successful_tool_requests:
                outcome = (
                    f"Duplicate `{function_name}` inspection skipped because it "
                    "already succeeded. Inspect a different interval or detail, "
                    "or answer from the existing evidence."
                )
            else:
                execution_parameters = dict(parameters)
                execution_parameters.update(
                    {"vr": self.vr, "subtitles": self.subtitles}
                )
                with collect_model_calls() as model_calls:
                    outcome = self.tool_registry.get_function(function_name)(
                        config=self.config["tools"][function_name],
                        parameters=execution_parameters,
                    )
                if outcome is None:
                    raise ValueError(f"Tool returns None results: {function_name}")
                succeeded = True
                if request_key is not None:
                    self._successful_tool_requests.add(request_key)
        except Exception as error:
            outcome = (
                f"Tool `{function_name}` execution failed.\n"
                f"Exception: {type(error).__name__}: {error}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )

        if self.verbose:
            print("OBSERVATION")
            print(outcome)
            print("--------------------------------")

        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call.get("id", ""),
            "name": function_name,
            "content": str(outcome),
        }
        if model_calls:
            tool_message["model_call"] = model_calls[0]
        return tool_message, succeeded

    def _print_assistant_message(self, label: str, message: dict) -> None:
        print(label)
        reasoning_content = message.get("reasoning_content")
        if reasoning_content:
            print("## == REASONING CONTENT ==")
            print(reasoning_content)
        print("## == CONTENT ==")
        print(message.get("content") or "")
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            print("## == TOOL CALLS ==")
            print(json.dumps(tool_calls, ensure_ascii=False, indent=2))
        print("--------------------------------")

    def _messages_for_model(self) -> list[dict]:
        return [
            {key: value for key, value in message.items() if key != "model_call"}
            for message in self.messages
        ]

    def _trajectory(self, total_steps: int, finish_reason: str) -> Trajectory:
        return Trajectory(
            question=self.question or "",
            messages=self.messages,
            total_steps=total_steps,
            final_answer=self.final_answer or "",
            finish_reason=finish_reason,
        )

    def _answer(self, total_steps: int, finish_reason: str) -> Trajectory:
        limit_reached = finish_reason == "reach_max_steps"
        prefix = (
            "The visual tool-call limit has been reached. "
            if limit_reached
            else "The evidence-gathering phase is complete. "
        )
        self.messages.append(
            {
                "role": "user",
                "content": (
                    prefix
                    + "Review the complete conversation, including the question, "
                    "subtitles, and visual tool observations. Provide the final "
                    "answer based only on that evidence. Follow the final-answer "
                    "format in the system prompt and any explicit output requirement "
                    "in the question exactly. Return only the requested answer, "
                    "without reasoning, headings, code fences, or extra text."
                ),
            }
        )
        assistant_message = self.model_client.chat(
            self._messages_for_model(),
            tools=[],
        )
        assistant_message.setdefault("role", "assistant")
        self.messages.append(assistant_message)
        total_steps += 1
        self.last_run_rounds = total_steps
        if self.verbose:
            self._print_assistant_message("[FINAL ANSWER] ASSISTANT", assistant_message)
        self.final_answer = (assistant_message.get("content") or "").strip()
        return self._trajectory(total_steps, finish_reason)

    @track_agent_usage
    def run(self, question: str) -> Trajectory:
        self.reset()
        self.question = question
        subtitles = convert_to_free_form_text_representation(
            self.subtitles, content_type="subtitle"
        )
        self.messages.append(
            {
                "role": "user",
                "content": (
                    f"Video Duration: {self.duration:.01f}s\n\n"
                    f"Video Subtitles:\n{subtitles}\n\n"
                    f"Question:\n{question}"
                ),
            }
        )

        if self.verbose:
            video_id = self.video_path.split("/")[-1].split(".")[0]
            print("--------------------------------")
            print(f"Video ID: {video_id} ({self.duration:.01f}s)")
            print("--------------------------------")
            print("Question:")
            print(question)
            print("--------------------------------")

        if not self.subtitles and not self.tools:
            raise VisualEvidenceRequiredError(
                "Visual evidence is required because no subtitles are available, "
                "but no visual tools are configured."
            )

        total_steps = 0
        has_visual_evidence = False
        for step in range(self.max_steps):
            visual_evidence_required = not self.subtitles and not has_visual_evidence
            assistant_message = self.model_client.chat(
                self._messages_for_model(),
                tools=self.tools,
                tool_choice="required" if visual_evidence_required else "auto",
            )
            assistant_message.setdefault("role", "assistant")
            self.messages.append(assistant_message)
            total_steps += 1
            self.last_run_rounds = total_steps

            tool_calls = assistant_message.get("tool_calls") or []
            if self.verbose:
                self._print_assistant_message(
                    f"[STEP {step + 1} / {self.max_steps}] ASSISTANT",
                    assistant_message,
                )

            if not tool_calls:
                if visual_evidence_required:
                    raise VisualEvidenceRequiredError(
                        "The model returned no structured visual tool call before "
                        "any visual evidence was collected."
                    )
                return self._answer(total_steps, "stop")

            for tool_call in tool_calls:
                tool_message, succeeded = self._execute_tool_call(tool_call)
                self.messages.append(tool_message)
                has_visual_evidence = has_visual_evidence or succeeded

        if not self.subtitles and not has_visual_evidence:
            raise VisualEvidenceRequiredError(
                "The visual tool-call limit was reached without collecting any "
                "successful visual evidence."
            )
        return self._answer(total_steps, "reach_max_steps")
