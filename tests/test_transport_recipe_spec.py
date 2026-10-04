"""CUDA-ordering contracts for the hardware-codec recipe."""

from __future__ import annotations

import pytest

import pixtreme as px


@pytest.mark.req("REQ-PIX-015")
@pytest.mark.req("REQ-PIX-105")
def test_from_array_passes_the_current_consumer_stream_to_dlpack() -> None:
    """DLPack array import passes the current CUDA consumer stream to the producer before using its GPU pixels."""
    import cupy as cp

    class RecordingProducer:
        def __init__(self, array: cp.ndarray) -> None:
            self.array = array
            self.streams: list[int | None] = []

        def __dlpack_device__(self) -> tuple[int, int]:
            return self.array.__dlpack_device__()

        def __dlpack__(self, *, stream: int | None = None) -> object:
            self.streams.append(stream)
            return self.array.__dlpack__(stream=stream)

    source = cp.zeros((2, 2, 3), dtype=cp.float32)
    producer = RecordingProducer(source)
    consumer = cp.cuda.Stream(non_blocking=True)

    with consumer:
        frame = px.io.from_array(producer, colorspace="sRGB", gamma="sRGB", channels="RGB")

    assert producer.streams == [consumer.ptr]
    assert frame.data.data.ptr == source.data.ptr
