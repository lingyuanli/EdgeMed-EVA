import base64
from io import BytesIO

import numpy as np
from PIL import Image

from videospy.model import ModelClient
from videospy.utils import (
    convert_to_free_form_text_representation,
    parse_video_timestamp,
    select_subtitles_in_range,
    validate_time_range,
)


clip_skim_tool = {
    "type": "function",
    "function": {
        "name": "clip_skim",
        "description": (
            "Browse a selected video interval by uniformly sampling frames. Use it "
            "to understand what happens in the interval and locate moments relevant "
            "to the query. Use frame_inspect when a specific frame needs closer "
            "examination."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A concise question or purpose for browsing the selected "
                        "video clip."
                    ),
                },
                "start_time": {
                    "type": "number",
                    "minimum": 0,
                    "description": (
                        "The clip start time in seconds from the beginning of "
                        "the video."
                    ),
                },
                "end_time": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": (
                        "The clip end time in seconds from the beginning of the "
                        "video. It must be greater than start_time. A value past "
                        "the video end is clipped."
                    ),
                },
            },
            "required": ["query", "start_time", "end_time"],
            "additionalProperties": False,
        },
    },
}


def _sample_frames(video_reader, start_time, end_time, num_frames):
    fps = video_reader.get_avg_fps()
    start_frame = min(max(int(start_time * fps), 0), len(video_reader) - 1)
    end_frame = min(max(int(end_time * fps), start_frame), len(video_reader) - 1)
    sample_count = min(end_frame - start_frame + 1, num_frames)
    frame_indices = np.linspace(start_frame, end_frame, sample_count).astype(int)
    timestamps = np.array(
        [round(index / fps, 1) for index in frame_indices],
        dtype=np.float32,
    )
    frames = video_reader.get_batch(frame_indices).asnumpy()
    return frames, timestamps


def _resize_frames(frames):
    _, height, width, _ = frames.shape
    scale = 256 / min(height, width)
    target_height = max(1, int(round(height * scale)))
    target_width = max(1, int(round(width * scale)))
    return [
        np.array(
            Image.fromarray(frame).resize(
                (target_width, target_height), Image.BICUBIC
            )
        )
        for frame in frames
    ]


def _frame_content(frames, timestamps):
    content = []
    for frame, timestamp in zip(frames, timestamps):
        image_buffer = BytesIO()
        Image.fromarray(frame).save(image_buffer, format="jpeg")
        image_data = base64.b64encode(image_buffer.getvalue()).decode("utf-8")
        content.extend(
            [
                {"type": "text", "text": f"{timestamp:.1f}s"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ]
        )
    return content


def execute_clip_skim(config: dict, parameters: dict) -> str:
    num_frames = config["num_frames"]
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}.")

    query = parameters["query"]
    start_time = parse_video_timestamp(parameters["start_time"])
    requested_end_time = parse_video_timestamp(parameters["end_time"])
    video_reader = parameters["vr"]
    video_duration = len(video_reader) / video_reader.get_avg_fps()
    end_time = min(requested_end_time, video_duration)
    validate_time_range(start_time, end_time, video_duration)
    subtitles = convert_to_free_form_text_representation(
        select_subtitles_in_range(parameters["subtitles"], start_time, end_time),
        content_type="subtitle",
    )

    frames, timestamps = _sample_frames(
        video_reader,
        start_time,
        end_time,
        num_frames,
    )
    frames = _resize_frames(frames)
    content = [
        {
            "type": "text",
            "text": f"Video clip ({start_time:.1f}s - {end_time:.1f}s):\n",
        }
    ]
    # Image.fromarray(frames[0]).show()
    content.extend(_frame_content(frames, timestamps))
    content.append(
        {
            "type": "text",
            "text": (
                f"Video Subtitles:\n{subtitles}\n\nQuestion:\n{query}\n\n"
                "Treat the question only as an inspection goal, not as evidence. "
                "Do not accept any claim or suggested interpretation in the question "
                "unless the frames directly support it. For each displayed timestamp, "
                "give one concise bullet that first states visible appearance or "
                "motion and then, only if supported, an interpretation. "
                "Use only the displayed timestamps and never infer unseen events. "
                "State uncertainty or plausible alternatives when evidence is "
                "ambiguous. Do not answer the question directly. End by naming only "
                "candidate timestamps that should be checked with frame_inspect."
            ),
        }
    )

    response = ModelClient(config["model"]).chat(
        messages=[{"role": "user", "content": content}]
    )
    if end_time < requested_end_time:
        interval_status = (
            f"Interval adjusted to video bounds: requested {start_time:.1f}s - "
            f"{requested_end_time:.1f}s, inspected {start_time:.1f}s - "
            f"{end_time:.1f}s.\n"
        )
    else:
        interval_status = ""
    return (
        "Inspection mode: clip_skim\n"
        f"{interval_status}Evidence status: coarse clip inspection.\n"
        f"{response['content']}"
    )
