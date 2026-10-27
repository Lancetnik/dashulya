from io import BytesIO
from typing import Literal

import av


def animation_bytes(
    *, format: Literal["mp4", "gif"] = "mp4", width: int = 32, frames: int = 30
) -> bytes:
    """Encode a local grayscale animation; no Telegram or model API is involved."""
    buffer = BytesIO()
    with av.open(buffer, "w", format=format) as writer:
        stream = writer.add_stream("mpeg4" if format == "mp4" else "gif", rate=10)
        stream.width = width
        stream.height = 32
        stream.pix_fmt = "yuv420p" if format == "mp4" else "rgb8"
        for index in range(frames):
            frame = av.VideoFrame(width, 32, "rgb24")
            plane = frame.planes[0]
            plane.update(bytes([min(index * 8, 240)]) * plane.buffer_size)
            for packet in stream.encode(frame):
                writer.mux(packet)
        for packet in stream.encode(None):
            writer.mux(packet)
    return buffer.getvalue()
