"""Internal path-blur CUDA source structure contracts."""

from __future__ import annotations

import re

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
