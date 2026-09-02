"""Deterministic, diagnosis-free visual tools for medical images and frame sequences."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .io import sha256_file


TOOL_SCHEMAS = {
    "inspect_overview": {
        "description": "Create a uniformly sampled overview across medical views or time.",
        "required": [],
    },
    "temporal_skim": {
        "description": "Uniformly sample one medical frame sequence inside a time interval.",
        "required": ["media_id", "start_time", "end_time"],
    },
    "region_inspect": {
        "description": "Crop one image or one selected frame at native resolution.",
        "required": ["media_id", "region_xyxy_1000", "target"],
    },
}


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _uniform_indices(length: int, count: int) -> list[int]:
    if length <= 0:
        return []
    count = max(1, min(count, length))
    if count == 1:
        return [0]
    return sorted({round(index * (length - 1) / (count - 1)) for index in range(count)})


def _media_frames(media: dict[str, Any], data_root: Path) -> list[dict[str, Any]]:
    kind = media.get("kind", "image")
    if kind == "image":
        raw_frames = [{
            "path": media["path"],
            "timestamp": media.get("timestamp", 0.0),
            "sha256": media.get("sha256"),
        }]
    elif kind == "image_sequence":
        raw_frames = media.get("frames")
        if not isinstance(raw_frames, list) or not raw_frames:
            raise ValueError(f"image_sequence has no frames: {media.get('media_id')}")
    else:
        raise ValueError(f"Unsupported media kind: {kind}")
    frames = []
    for index, frame in enumerate(raw_frames):
        path = (data_root / str(frame["path"])).resolve()
        try:
            path.relative_to(data_root.resolve())
        except ValueError as exc:
            raise ValueError(f"Media path escapes data root: {path}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        frames.append(
            {
                "frame_index": index,
                "path": path,
                "timestamp": float(frame.get("timestamp", index)),
                "expected_sha256": frame.get("sha256"),
            }
        )
    return frames


def _contact_sheet(frames: list[dict[str, Any]], output: Path) -> None:
    opened = []
    try:
        for frame in frames:
            image = Image.open(frame["path"]).convert("RGB")
            image.thumbnail((384, 384), Image.Resampling.LANCZOS)
            opened.append((frame, image.copy()))
        width = max(image.width for _, image in opened)
        height = max(image.height for _, image in opened) + 28
        sheet = Image.new("RGB", (width * len(opened), height), "white")
        draw = ImageDraw.Draw(sheet)
        for index, (frame, image) in enumerate(opened):
            x = index * width + (width - image.width) // 2
            sheet.paste(image, (x, 0))
            draw.text((index * width + 4, height - 22), f"t={frame['timestamp']:.3f}s", fill="black")
        output.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output, format="PNG", optimize=False)
    finally:
        for _, image in opened:
            image.close()


class MedicalToolExecutor:
    """Execute only an explicit per-run tool allowlist and preserve every trace."""

    def __init__(
        self,
        sample: dict[str, Any],
        data_root: Path,
        artifact_dir: Path,
        allowed_tools: tuple[str, ...] = tuple(TOOL_SCHEMAS),
        max_calls: int = 4,
    ) -> None:
        self.sample = sample
        self.data_root = data_root.resolve()
        self.artifact_dir = artifact_dir.resolve()
        self.allowed_tools = frozenset(allowed_tools)
        unknown = self.allowed_tools - TOOL_SCHEMAS.keys()
        if unknown:
            raise ValueError(f"Unknown allowed tools: {sorted(unknown)}")
        self.max_calls = max_calls
        self.traces: list[dict[str, Any]] = []
        self._request_hashes: set[str] = set()
        media = sample.get("media")
        if not isinstance(media, list) or not media:
            raise ValueError("Agent sample requires non-empty media")
        self._media = {str(item["media_id"]): item for item in media}
        if len(self._media) != len(media):
            raise ValueError("Duplicate media_id")
        self._frame_cache: dict[str, list[dict[str, Any]]] = {}

    def _select_frames(self, media_id: str) -> list[dict[str, Any]]:
        if media_id not in self._media:
            raise ValueError(f"Unknown media_id: {media_id}")
        if media_id not in self._frame_cache:
            self._frame_cache[media_id] = _media_frames(self._media[media_id], self.data_root)
        return self._frame_cache[media_id]

    @staticmethod
    def _bind_selected(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for frame in frames:
            actual = frame.setdefault("sha256", sha256_file(frame["path"]))
            expected = frame.get("expected_sha256")
            if expected is not None and actual != expected:
                raise ValueError(f"Media hash mismatch: {frame['path']}")
        return frames

    def execute(self, name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        started = time.perf_counter()
        request = {"name": name, "arguments": arguments}
        request_hash = _canonical_hash(request)
        trace_id = f"{self.sample['sample_id']}:T{len(self.traces) + 1}"
        trace: dict[str, Any] = {
            "schema_version": "edgemed-medical-tool-trace/v1",
            "trace_id": trace_id,
            "sample_id": self.sample["sample_id"],
            "tool_name": name,
            "request": arguments,
            "request_sha256": request_hash,
            "status": "failed",
        }
        try:
            if name not in self.allowed_tools:
                raise PermissionError(f"Tool is not enabled for this run: {name}")
            if len(self.traces) >= self.max_calls:
                raise RuntimeError(f"Tool budget exceeded: {self.max_calls}")
            if request_hash in self._request_hashes:
                raise ValueError("Duplicate tool call")
            required = TOOL_SCHEMAS[name]["required"]
            missing = [field for field in required if field not in arguments]
            if missing:
                raise ValueError(f"Missing tool arguments: {missing}")

            if name == "inspect_overview":
                result, selected = self._overview(arguments)
            elif name == "temporal_skim":
                result, selected = self._temporal_skim(arguments)
            else:
                result, selected = self._region_inspect(arguments)
            trace.update(
                {
                    "status": "completed",
                    "input_media_sha256": [frame["sha256"] for frame in selected],
                    "output_artifact": result["artifact_path"],
                    "output_sha256": result["artifact_sha256"],
                    "selected_frames": result["selected_frames"],
                }
            )
            self._request_hashes.add(request_hash)
            return result, trace
        except Exception as exc:
            trace["error"] = f"{type(exc).__name__}: {exc}"
            return {"trace_id": trace_id, "status": "failed", "error": trace["error"]}, trace
        finally:
            trace["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
            self.traces.append(trace)

    def _write_sheet(self, frames: list[dict[str, Any]], request: dict[str, Any]) -> dict[str, Any]:
        provisional = self.artifact_dir / f"{_canonical_hash(request)}.png"
        _contact_sheet(frames, provisional)
        digest = sha256_file(provisional)
        final_path = self.artifact_dir / f"{digest}.png"
        if final_path != provisional:
            if final_path.exists():
                provisional.unlink()
            else:
                provisional.rename(final_path)
        return {
            "status": "completed",
            "artifact_path": str(final_path),
            "artifact_sha256": digest,
            "selected_frames": [
                {
                    "media_id": frame["media_id"],
                    "frame_index": frame["frame_index"],
                    "timestamp": frame["timestamp"],
                    "source_sha256": frame["sha256"],
                }
                for frame in frames
            ],
            "semantic_observation": None,
        }

    def _overview(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        count = int(arguments.get("sample_count", 6))
        if count < 1 or count > 16:
            raise ValueError("sample_count must be in [1,16]")
        pool: list[dict[str, Any]] = []
        for media_id in self._media:
            frames = self._select_frames(media_id)
            for index in _uniform_indices(len(frames), min(count, len(frames))):
                pool.append({**frames[index], "media_id": media_id})
        selected = self._bind_selected([pool[index] for index in _uniform_indices(len(pool), count)])
        return self._write_sheet(selected, {"tool": "inspect_overview", **arguments}), selected

    def _temporal_skim(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        media_id = str(arguments["media_id"])
        if self._media[media_id].get("kind", "image") != "image_sequence":
            raise ValueError("temporal_skim requires image_sequence media")
        start = float(arguments["start_time"])
        end = float(arguments["end_time"])
        count = int(arguments.get("sample_count", 6))
        if start < 0 or end <= start or count < 1 or count > 16:
            raise ValueError("Invalid temporal interval or sample_count")
        candidates = [frame for frame in self._select_frames(media_id) if start <= frame["timestamp"] <= end]
        if not candidates:
            raise ValueError("No frames fall inside requested interval")
        selected = self._bind_selected([
            {**candidates[index], "media_id": media_id}
            for index in _uniform_indices(len(candidates), count)
        ])
        return self._write_sheet(selected, {"tool": "temporal_skim", **arguments}), selected

    def _region_inspect(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        media_id = str(arguments["media_id"])
        target = str(arguments["target"]).strip()
        if not target:
            raise ValueError("target must be non-empty")
        box = arguments["region_xyxy_1000"]
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("region_xyxy_1000 must contain four values")
        x1, y1, x2, y2 = [max(0, min(1000, int(value))) for value in box]
        if x2 - x1 < 10 or y2 - y1 < 10:
            raise ValueError("Crop is empty or smaller than 1% of an axis")
        frames = self._select_frames(media_id)
        timestamp = arguments.get("timestamp")
        if len(frames) > 1 and timestamp is None:
            raise ValueError("timestamp is required for image_sequence region inspection")
        frame = min(frames, key=lambda item: abs(item["timestamp"] - float(timestamp or 0.0)))
        with Image.open(frame["path"]) as image:
            image = image.convert("RGB")
            pixel_box = (
                x1 * image.width // 1000,
                y1 * image.height // 1000,
                max(1, x2 * image.width // 1000),
                max(1, y2 * image.height // 1000),
            )
            cropped = image.crop(pixel_box)
            cropped.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            provisional = self.artifact_dir / f"{_canonical_hash({'tool': 'region_inspect', **arguments})}.png"
            provisional.parent.mkdir(parents=True, exist_ok=True)
            cropped.save(provisional, format="PNG", optimize=False)
        digest = sha256_file(provisional)
        final_path = self.artifact_dir / f"{digest}.png"
        if final_path != provisional:
            if final_path.exists():
                provisional.unlink()
            else:
                provisional.rename(final_path)
        selected = self._bind_selected([{**frame, "media_id": media_id}])
        result = {
            "status": "completed",
            "artifact_path": str(final_path),
            "artifact_sha256": digest,
            "selected_frames": [
                {
                    "media_id": media_id,
                    "frame_index": frame["frame_index"],
                    "timestamp": frame["timestamp"],
                    "source_sha256": frame["sha256"],
                }
            ],
            "region_xyxy_1000": [x1, y1, x2, y2],
            "target": target,
            "semantic_observation": None,
        }
        return result, selected
