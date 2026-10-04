"""Inspect the RLE packet structure required by its EXR gate fixture."""

from __future__ import annotations

from pixtreme._io.formats.exr.container import _ExrContainer


def inspect_rle_chunks(container: _ExrContainer) -> tuple[int, int, int]:
    descriptors = tuple(chunk.rle for chunk in container.chunks)
    if any(descriptor is None for descriptor in descriptors):
        raise AssertionError("the RLE gate fixture has a chunk without an RLE descriptor")
    compressed = tuple(descriptor for descriptor in descriptors if descriptor is not None and not descriptor.raw_stored)
    packet_count = sum(len(descriptor.packets) for descriptor in compressed)
    if not compressed or not packet_count:
        raise AssertionError("the RLE gate fixture contains no compressed RLE packet")
    return len(descriptors), len(compressed), packet_count
