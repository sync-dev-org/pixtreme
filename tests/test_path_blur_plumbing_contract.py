"""Internal path-blur CUDA source structure contracts."""

from __future__ import annotations

import re

import cupy as cp
import pytest

import pixtreme as px
from pixtreme._filter import directional_radial, vector


def _normalized_call_arguments(source: str, function_name: str) -> tuple[str, ...]:
    match = re.search(rf"{function_name}\s*\((.*?)\)\s*;", source, flags=re.DOTALL)
    assert match is not None
    return tuple(" ".join(argument.split()) for argument in match.group(1).split(","))


def test_path_blur_rgb_tile_loader_halo_and_barrier_characterization() -> None:
    """REQ-TEST-001 and REQ-TEST-003 characterization: freezes the current normalized loader arguments and barrier
    order. Correctness is not independently established because the current CUDA source is the oracle rather than a
    runtime synchronization reference. Remove these pins when runtime coverage supersedes them, or specify and update
    them only for an intentional loader/synchronization contract change."""
    helper_name = "pixtreme_path_load_rgb_tile"
    assert directional_radial._PATH_BLUR_PREAMBLE.count(f"{helper_name}(") == 1
    # The force-inline requirement binds to this helper's own declaration; the return
    # type is left free so unrelated signature evolution stays outside the contract.
    assert re.search(
        rf"__device__\s+__forceinline__\s+\S+\s+{helper_name}\(",
        directional_radial._PATH_BLUR_PREAMBLE,
    )

    directional_source = directional_radial._PATH_BLUR_KERNEL_SOURCE.removeprefix(
        directional_radial._PATH_BLUR_PREAMBLE
    )
    vector_source = vector._VECTOR_BLUR_KERNEL_SOURCE.removeprefix(directional_radial._PATH_BLUR_PREAMBLE)
    for source in (directional_source, vector_source):
        assert source.count(f"{helper_name}(") == 1
        call_offset = source.index(f"{helper_name}(")
        barrier_offset = source.index("__syncthreads();", call_offset)
        assert call_offset < barrier_offset

    common_prefix = ("source", "tile", "width", "height", "block_x", "block_y")
    common_suffix = ("border", "border_value")
    assert _normalized_call_arguments(directional_source, helper_name) == (
        *common_prefix,
        "halo_x",
        "halo_y",
        *common_suffix,
    )
    assert _normalized_call_arguments(vector_source, helper_name) == (
        *common_prefix,
        "halo",
        "halo",
        *common_suffix,
    )


def test_path_blur_global_gather_checks_each_footprint_once() -> None:
    """REQ-TEST-001: the RGB global gather checks an in-bounds 4 x 4 footprint once per sample."""
    preamble = directional_radial._PATH_BLUR_PREAMBLE
    assert "pixtreme_path_rgb_footprint_in_bounds" in preamble
    assert "pixtreme_path_bicubic_rgb_precomputed" in preamble
    assert "#ifdef PIXTREME_PATH_SHARED" in preamble
    assert "pixtreme_vector_blur_rgb_global_path" in vector._VECTOR_BLUR_KERNEL_SOURCE
    assert "sample_count >= 65" in vector._VECTOR_BLUR_KERNEL_SOURCE


@pytest.mark.parametrize("operation", ("directional", "vector"))
def test_path_blur_rgb_optimized_gather_is_bit_exact_with_generic_path(operation: str) -> None:
    """REQ-TEST-001 and REQ-TEST-003: optimized RGB path accumulation is bit-exact with per-channel gathering."""
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
