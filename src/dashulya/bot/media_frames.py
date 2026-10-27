"""Bounded in-memory extraction of an animation's first frame."""

from fractions import Fraction
from io import BytesIO

import av

MAX_INPUT_PIXELS = 4096 * 4096
MAX_EDGE = 768


class FrameDecodeError(ValueError):
    """The downloaded media cannot produce a usable first frame."""


class FrameBudgetError(FrameDecodeError):
    """The first frame exceeds the remaining image byte budget."""


def _jpeg(frame: av.VideoFrame) -> bytes:
    scale = min(1, MAX_EDGE / max(frame.width, frame.height))
    width = max(2, int(frame.width * scale) // 2 * 2)
    height = max(2, int(frame.height * scale) // 2 * 2)
    codec = av.CodecContext.create("mjpeg", "w")
    assert isinstance(codec, av.VideoCodecContext)
    codec.width, codec.height = width, height
    codec.pix_fmt = "yuvj420p"
    codec.time_base = Fraction(1, 25)
    codec.thread_count = 1
    image = frame.reformat(width=width, height=height, format="yuvj420p")
    image.pts = None
    return b"".join(
        bytes(packet) for packet in [*codec.encode(image), *codec.encode(None)]
    )


def _first_frame(container: av.container.InputContainer, *, max_bytes: int) -> bytes:
    streams = container.streams.video
    if not streams:
        raise FrameDecodeError
    stream = streams[0]
    stream.codec_context.thread_count = 1
    if stream.codec_context.width * stream.codec_context.height > MAX_INPUT_PIXELS:
        raise FrameDecodeError
    frame = next(container.decode(stream), None)
    if frame is None or frame.width * frame.height > MAX_INPUT_PIXELS:
        raise FrameDecodeError
    encoded = _jpeg(frame)
    if len(encoded) > max_bytes:
        raise FrameBudgetError
    return encoded


def first_frame(data: bytes, *, max_bytes: int) -> bytes:
    """Return only the first decoded frame as a size-bounded JPEG.

    The caller offloads this CPU work from the event loop and supplies the shared
    native-image byte budget. Later frames are not decoded.
    Input is a local buffer; external protocols are disabled.
    """
    if not (
        data.startswith((b"GIF87a", b"GIF89a", b"\x1aE\xdf\xa3")) or data[4:8] == b"ftyp"
    ):
        raise FrameDecodeError
    try:
        with av.open(
            BytesIO(data), mode="r", options={"protocol_whitelist": "pipe"}
        ) as container:
            return _first_frame(container, max_bytes=max_bytes)
    except av.error.FFmpegError:
        raise FrameDecodeError from None
