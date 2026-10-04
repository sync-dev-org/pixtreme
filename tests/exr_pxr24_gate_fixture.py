"""Inspect the PXR24 byte planes required by its EXR gate fixture."""

from __future__ import annotations

from pixtreme._io.formats.exr.container import _ExrContainer


def inspect_pxr24_chunks(container: _ExrContainer) -> tuple[int, int, int]:
    descriptors = tuple(chunk.pxr24 for chunk in container.chunks)
    if any(descriptor is None for descriptor in descriptors):
        raise AssertionError("the PXR24 gate fixture has a chunk without a PXR24 descriptor")
    compressed = tuple(descriptor for descriptor in descriptors if descriptor is not None and not descriptor.raw_stored)
    plane_count = sum(len(descriptor.planes) for descriptor in compressed)
    if not compressed or not plane_count:
        raise AssertionError("the PXR24 gate fixture contains no compressed PXR24 plane")
    return len(descriptors), len(compressed), plane_count
