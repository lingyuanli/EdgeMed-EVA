import base64
from io import BytesIO

from PIL import Image

from videospy.model import ModelClient
from videospy.utils import (
    convert_to_free_form_text_representation,
    parse_video_timestamp,
    select_subtitles_in_range,
)


frame_inspect_tool = {
    "type": "function",
    "function": {
        "name": "frame_inspect",
        "description": (
            "Examine one video frame at a specific timestamp. Use query to state "
            "which visible detail should be checked more closely."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A concise question describing the detail to check in the "
                        "selected frame."
                    ),
                },
                "timestamp": {
                    "type": "number",
                    "minimum": 0,
                    "description": (
                        "The frame time in seconds from the beginning of the video."
                    ),
                },
            },
            "required": ["query", "timestamp"],
            "additionalProperties": False,
        },
    },
}


def execute_frame_inspect(config: dict, parameters: dict) -> str:
    query = parameters["query"]
    timestamp = parse_video_timestamp(parameters["timestamp"])
    video_reader = parameters["vr"]
    fps = video_reader.get_avg_fps()
    video_duration = len(video_reader) / fps
    if timestamp < 0 or timestamp > video_duration:
        raise ValueError(
            f"timestamp must be within the video duration "
            f"(0.0s - {video_duration:.1f}s), got {timestamp}."
        )

    frame_index = min(int(timestamp * fps + 0.5), len(video_reader) - 1)
    actual_timestamp = frame_index / fps
    frame = video_reader.get_batch([frame_index]).asnumpy()[0]
    subtitles = convert_to_free_form_text_representation(
        select_subtitles_in_range(
            parameters["subtitles"], actual_timestamp, actual_timestamp
        ),
        content_type="subtitle",
    )
    # Image.fromarray(frame).show()
    image_buffer = BytesIO()
    Image.fromarray(frame).save(image_buffer, format="jpeg")
    image_data = base64.b64encode(image_buffer.getvalue()).decode("utf-8")
    content = [
        {
            "type": "text",
            "text": (
                f"Requested timestamp: {timestamp:.3f}s\n"
                f"Actual frame timestamp: {actual_timestamp:.3f}s\n"
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
        },
        {
            "type": "text",
            "text": (
                f"Video Subtitles:\n{subtitles}\n\nQuestion:\n{query}\n\n"
                "Treat the question only as an inspection goal, not as evidence. "
                "Do not accept any claim or suggested interpretation in the question "
                "unless this frame directly supports it. First state the directly "
                "visible details, then give a brief interpretation only if supported. "
                "Do not infer unseen actions, "
                "temporal order, or events outside the frame. For visible text, "
                "transcribe exact characters, numbers, units, and currency symbols. "
                "For counts, enumerate distinct visible objects by position. State "
                "plausible alternatives when ambiguous and keep the response "
                "concise. If the requested detail is not clearly visible, "
                "explicitly report 'Insufficient visual evidence.'"
            ),
        },
    ]

    response = ModelClient(config["model"]).chat(
        messages=[{"role": "user", "content": content}]
    )
    return (
        f"Frame inspection at {actual_timestamp:.3f}s "
        f"(requested {timestamp:.3f}s):\n{response['content']}"
    )
