"""Generate an RLE EXR GPU comparison sheet."""

from __future__ import annotations

from exr_visual_comparison import CompressionVisual, VisualInspection, main
from openexr_dev_oracle import OpenEXR

from pixtreme._io.formats.exr.container import _ExrContainer


def _inspect_rle(container: _ExrContainer) -> VisualInspection:
    compressed = tuple(chunk.rle for chunk in container.chunks if chunk.rle is not None and not chunk.rle.raw_stored)
    if not compressed:
        raise AssertionError("RLE visual fixture contains no compressed chunk")
    return VisualInspection(compressed_chunks=len(compressed))


if __name__ == "__main__":
    main((CompressionVisual("rle", OpenEXR.RLE_COMPRESSION, "float32", 65536.0, False, _inspect_rle),))
