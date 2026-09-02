import base64
from io import BytesIO

import numpy as np
from PIL import Image

from videospy.model import ModelClient
from videospy.utils import convert_to_free_form_text_representation


def _encode_image(frame: np.ndarray) -> dict:
    image_buffer = BytesIO()
    Image.fromarray(frame).save(image_buffer, format="jpeg")
    image_data = base64.b64encode(image_buffer.getvalue()).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
    }


overview_tool = {
    "type": "function",
    "function": {
        "name": "overview",
        "description": "To get a timestamped video summary for the entire video.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
}


def execute_overview(config: dict, parameters: dict) -> str:
    video_reader = parameters["vr"]
    duration = round(len(video_reader) / video_reader.get_avg_fps(), 1)
    subtitles = convert_to_free_form_text_representation(
        parameters["subtitles"], content_type="subtitle"
    )

    total_frames = len(video_reader)
    target_num_frames = int(np.ceil(config["num_frames"] / 8.0) * 8)
    num_frames = (
        min(target_num_frames, total_frames // 8 * 8)
        if total_frames >= 8
        else total_frames
    )

    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    timestamps = np.array(
        [round(index / video_reader.get_avg_fps(), 1) for index in frame_indices],
        dtype=np.float32,
    )
    timestamp_values = ", ".join(f"{timestamp:.1f}s" for timestamp in timestamps)
    frames = video_reader.get_batch(frame_indices).asnumpy()

    _, height, width, _ = frames.shape
    scale = 256 / min(height, width)
    target_height = max(1, int(round(height * scale)))
    target_width = max(1, int(round(width * scale)))
    frames = np.stack(
        [
            np.array(
                Image.fromarray(frame).resize(
                    (target_width, target_height), Image.BICUBIC
                )
            )
            for frame in frames
        ],
        axis=0,
    )

    content = [
        {
            "type": "text",
            "text": (
                f"The video segment is located at 0.0s - {duration:.1f}s:\n"
                "The video frames are uniformly sampled.\n"
            ),
        }
    ]
    # Image.fromarray(frames[0]).show()
    if num_frames >= 8:
        _, height, width, channels = frames.shape
        timestamp_grids = timestamps.reshape(-1, 2, 4)
        frame_grids = frames.reshape(-1, 2, 4, height, width, channels)
        frame_grids = frame_grids.transpose(0, 1, 3, 2, 4, 5).reshape(
            -1, 2 * height, 4 * width, channels
        )
        for frame, timestamp_grid in zip(frame_grids, timestamp_grids):
            row_1 = ", ".join(f"{value:.1f}s" for value in timestamp_grid[0])
            row_2 = ", ".join(f"{value:.1f}s" for value in timestamp_grid[1])
            content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            f"[{timestamp_grid[0, 0]:.1f}s - "
                            f"{timestamp_grid[-1, -1]:.1f}s]:\n"
                            f"Timestamp Matrix:\n[[{row_1}],\n[{row_2}]]\n"
                        ),
                    },
                    _encode_image(frame),
                ]
            )
    else:
        for frame, timestamp in zip(frames, timestamps):
            content.extend(
                [
                    {"type": "text", "text": f"{timestamp:.1f}s"},
                    _encode_image(frame),
                ]
            )

    content.append(
        {
            "type": "text",
            "text": (
                f"Video Subtitles:\n{subtitles}\n\n"
                "Please generate descriptions for each frame in the video. The "
                "descriptions should be concise and detailed (~50 words each).\n"
                "Return exactly one line for every sampled frame, in chronological "
                "order, using the format TIMESTAMP: FRAME_DESCRIPTION. Ensure every "
                "timestamp exactly matches one of the provided values: "
                f"[{timestamp_values}].\n"
                "Use each timestamp exactly once. Do not include headings, bullets, "
                "code fences, or any other text."
            ),
        }
    )

    response = ModelClient(config["model"]).chat(
        messages=[{"role": "user", "content": content}]
    )
    return response["content"].strip()
