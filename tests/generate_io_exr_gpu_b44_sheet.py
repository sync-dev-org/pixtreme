"""Generate B44 and B44A EXR GPU comparison sheets."""

from __future__ import annotations

from exr_visual_comparison import CompressionVisual, VisualInspection, main
from openexr_dev_oracle import OpenEXR

from pixtreme._io.formats.exr.container import _ExrContainer


def _inspect_b44(container: _ExrContainer) -> VisualInspection:
    compressed = tuple(chunk.b44 for chunk in container.chunks if chunk.b44 is not None and not chunk.b44.raw_stored)
    if not compressed:
        raise AssertionError("B44 visual fixture contains no compressed chunk")
    dense_blocks = sum(block.stored_size == 14 for descriptor in compressed for block in descriptor.blocks)
    flat_blocks = sum(block.stored_size == 3 for descriptor in compressed for block in descriptor.blocks)
    plinear_sections = sum(
        section.perceptually_linear for descriptor in compressed for section in descriptor.channel_sections
    )
    if not dense_blocks or not plinear_sections:
        raise AssertionError("B44 visual fixture must contain dense and pLinear blocks")
    if container.compression == "b44a" and not flat_blocks:
        raise AssertionError("B44A visual fixture must contain a flat block")
    return VisualInspection(len(compressed), dense_blocks, flat_blocks, plinear_sections)


if __name__ == "__main__":
    main(
        (
            CompressionVisual("b44", OpenEXR.B44_COMPRESSION, "float16", 16.0, True, _inspect_b44),
            CompressionVisual("b44a", OpenEXR.B44A_COMPRESSION, "float16", 16.0, True, _inspect_b44),
        )
    )
