"""Generate a PXR24 EXR GPU comparison sheet."""

from __future__ import annotations

from exr_visual_comparison import CompressionVisual, VisualInspection, main
from openexr_dev_oracle import OpenEXR

from pixtreme._io.formats.exr.container import _ExrContainer


def _inspect_pxr24(container: _ExrContainer) -> VisualInspection:
    compressed = tuple(
        chunk.pxr24 for chunk in container.chunks if chunk.pxr24 is not None and not chunk.pxr24.raw_stored
    )
    if not compressed:
        raise AssertionError("PXR24 visual fixture contains no compressed chunk")
    return VisualInspection(compressed_chunks=len(compressed))


if __name__ == "__main__":
    main((CompressionVisual("pxr24", OpenEXR.PXR24_COMPRESSION, "float32", 4096.0, False, _inspect_pxr24),))
