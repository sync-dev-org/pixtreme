"""Internal path-blur CUDA source structure contracts."""

from __future__ import annotations

import cupy as cp
import pytest

import pixtreme as px


@pytest.mark.req("REQ-PIX-011")
@pytest.mark.req("REQ-PIX-101")
@pytest.mark.parametrize("operation", ("directional", "vector"))
def test_path_blur_rgb_optimized_gather_is_bit_exact_with_generic_path(operation: str) -> None:
    """The optimized RGB path blur produces the same pixel bits as per channel gathering."""
    generator = cp.random.default_rng(20260817)
    rgb = generator.random((23, 31, 3), dtype=cp.float32)
    rgba = cp.concatenate((rgb, cp.zeros((23, 31, 1), dtype=cp.float32)), axis=2)
    rgb_frame = px.io.from_array(rgb, colorspace="sRGB", gamma="linear", channels="RGB")
    rgba_frame = px.io.from_array(rgba, colorspace="sRGB", gamma="linear", channels=("R", "G", "B", "A"))

    if operation == "directional":
        rgb_output = px.filter.directional_blur(rgb_frame, angle=30.0, length=128.0, border="wrap")
        generic_output = px.filter.directional_blur(rgba_frame, angle=30.0, length=128.0, border="wrap")
    else:
        vector_data = cp.empty((23, 31, 2), dtype=cp.float32)
        vector_data[..., 0] = cp.float32(128.0)
        vector_data[..., 1] = cp.float32(0.0)
        vector_frame = px.io.from_array(
            vector_data,
            colorspace="sRGB",
            gamma="linear",
            channels=("X", "Y"),
        )
        rgb_output = px.filter.vector_blur(rgb_frame, vector=vector_frame, border="wrap")
        generic_output = px.filter.vector_blur(rgba_frame, vector=vector_frame, border="wrap")

    assert cp.array_equal(rgb_output.data, generic_output.data[..., :3])
