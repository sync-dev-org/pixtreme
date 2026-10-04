"""Inspect the B44 and B44A blocks required by their EXR gate fixtures."""

from __future__ import annotations

from pixtreme._io.formats.exr.container import _ExrContainer


def inspect_b44_chunks(container: _ExrContainer, compression: str) -> tuple[int, int, int, int]:
    descriptors = tuple(chunk.b44 for chunk in container.chunks)
    if any(descriptor is None for descriptor in descriptors):
        raise AssertionError(f"the {compression.upper()} gate fixture has a chunk without a B44 descriptor")
    compressed = tuple(descriptor for descriptor in descriptors if descriptor is not None and not descriptor.raw_stored)
    dense_blocks = sum(block.stored_size == 14 for descriptor in compressed for block in descriptor.blocks)
    flat_blocks = sum(block.stored_size == 3 for descriptor in compressed for block in descriptor.blocks)
    if not compressed or not dense_blocks:
        raise AssertionError(f"the {compression.upper()} gate fixture contains no dense B44 block")
    if compression == "b44a" and not flat_blocks:
        raise AssertionError("the B44A gate fixture contains no flat block")
    return len(descriptors), len(compressed), dense_blocks, flat_blocks
