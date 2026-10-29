from collections.abc import Iterator
from contextlib import nullcontext
from io import BytesIO
from unittest.mock import Mock, patch

import av
import pytest

from dashulya.bot.media_frames import FrameBudgetError, FrameDecodeError, first_frame
from tests.media_fixtures import animation_bytes


class TestMediaFrame:
    def test_mp4_and_gif_return_only_the_first_frame(
        self, subtests: pytest.Subtests
    ) -> None:
        for format in ("mp4", "gif"):
            with subtests.test(format=format):
                image = first_frame(animation_bytes(format=format), max_bytes=100_000)
                assert image.startswith(b"\xff\xd8\xff")
                with av.open(BytesIO(image), mode="r") as reader:
                    frame = next(reader.decode(video=0)).reformat(format="gray")
                    assert bytes(frame.planes[0])[0] < 5

    def test_decoder_validates_the_first_frame_without_requesting_a_second(
        self, subtests: pytest.Subtests
    ) -> None:
        frame = av.VideoFrame(32, 32, "rgb24")
        frame.planes[0].update(bytes(frame.planes[0].buffer_size))

        def frames(
            decoded: av.VideoFrame | Mock | None,
        ) -> Iterator[av.VideoFrame | Mock]:
            if decoded is not None:
                yield decoded
                pytest.fail("A second frame must not be requested")

        for decoded in (frame, None, Mock(width=4097, height=4096)):
            with subtests.test(decoded=decoded):
                stream = Mock(codec_context=Mock(width=32, height=32))
                container = Mock(streams=Mock(video=[stream]))
                container.decode.return_value = frames(decoded)
                with patch(
                    "dashulya.bot.media_frames.av.open",
                    return_value=nullcontext(container),
                ):
                    if decoded is frame:
                        assert first_frame(b"GIF89a", max_bytes=100_000)
                    else:
                        with pytest.raises(FrameDecodeError):
                            first_frame(b"GIF89a", max_bytes=100_000)

    def test_large_frame_is_resized_before_model_input(self) -> None:
        image = first_frame(animation_bytes(width=1024, frames=1), max_bytes=100_000)
        with av.open(BytesIO(image), mode="r") as reader:
            frame = next(reader.decode(video=0))
            assert frame.width == 768
            assert frame.height < 32

    def test_encoded_frame_obeys_the_remaining_image_budget(self) -> None:
        clip = animation_bytes()
        first = first_frame(clip, max_bytes=100_000)
        assert first_frame(clip, max_bytes=len(first)) == first
        with pytest.raises(FrameBudgetError):
            first_frame(clip, max_bytes=len(first) - 1)

    def test_invalid_or_truncated_animation_is_not_passed_to_model(
        self, subtests: pytest.Subtests
    ) -> None:
        for data in (b"not media", b"GIF89a broken", b"\x00\x00\x00\x20ftypbroken"):
            with subtests.test(data=data), pytest.raises(FrameDecodeError):
                first_frame(data, max_bytes=100_000)
