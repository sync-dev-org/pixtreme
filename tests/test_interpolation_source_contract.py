"""Contracts for the shared CUDA point-interpolation substrate."""

from __future__ import annotations

import ast
import inspect
import math

import cupy as cp
import numpy as np
import pytest

from pixtreme._core.interpolation import (
    _POINT_INTERPOLATION_DEVICE_SOURCE,
    _POINT_INTERPOLATION_SPECS,
    _POINT_INTERPOLATION_TOKENS,
    _specialized_point_weight_source,
)


def _reference_weight(interpolation: str, distance: float) -> float:
    """Independent fp64 piecewise oracle for the public point-filter vocabulary."""
    x = abs(distance)
    if interpolation == "bilinear":
        return max(0.0, 1.0 - x)
    if interpolation == "bicubic":
        a = -0.5
        if x < 1.0:
            return (a + 2.0) * x**3 - (a + 3.0) * x**2 + 1.0
        if x < 2.0:
            return a * x**3 - 5.0 * a * x**2 + 8.0 * a * x - 4.0 * a
        return 0.0
    if interpolation in {"b-spline", "mitchell"}:
        b, c = (1.0, 0.0) if interpolation == "b-spline" else (1.0 / 3.0, 1.0 / 3.0)
        if x < 1.0:
            return ((12.0 - 9.0 * b - 6.0 * c) * x**3 + (-18.0 + 12.0 * b + 6.0 * c) * x**2 + (6.0 - 2.0 * b)) / 6.0
        if x < 2.0:
            return (
                (-b - 6.0 * c) * x**3 + (6.0 * b + 30.0 * c) * x**2 + (-12.0 * b - 48.0 * c) * x + (8.0 * b + 24.0 * c)
            ) / 6.0
        return 0.0
    lobes = int(interpolation.removeprefix("lanczos"))
    if x == 0.0:
        return 1.0
    if x >= lobes:
        return 0.0
    pi_x = math.pi * x
    return lobes * math.sin(pi_x) * math.sin(pi_x / lobes) / (pi_x * pi_x)


def test_point_interpolation_mapping_preserves_canonical_runtime_indices_and_coefficients() -> None:
    """REQ-TEST-001: canonical point tokens retain the existing kernel argument indices and coefficients."""
    assert _POINT_INTERPOLATION_TOKENS == (
        "nearest",
        "bilinear",
        "bicubic",
        "b-spline",
        "mitchell",
        "lanczos2",
        "lanczos3",
        "lanczos4",
    )
    assert {
        token: (spec.index, spec.family, spec.b, spec.c, spec.lobes)
        for token, spec in _POINT_INTERPOLATION_SPECS.items()
    } == {
        "nearest": (0, "nearest", None, None, None),
        "bilinear": (1, "linear", None, None, None),
        "bicubic": (2, "keys", None, None, None),
        "b-spline": (3, "mitchell", "1.0f", "0.0f", None),
        "mitchell": (4, "mitchell", "1.0f / 3.0f", "1.0f / 3.0f", None),
        "lanczos2": (5, "lanczos", None, None, 2),
        "lanczos3": (6, "lanczos", None, None, 3),
        "lanczos4": (7, "lanczos", None, None, 4),
    }


@pytest.mark.parametrize("interpolation", _POINT_INTERPOLATION_TOKENS[1:])
def test_specialized_point_weight_source_matches_independent_piecewise_boundary_oracle(interpolation: str) -> None:
    """REQ-TEST-001: generated inline weights match an independent oracle at every piecewise support boundary."""
    distances = np.asarray((-4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0), dtype=np.float32)
    source = (
        _specialized_point_weight_source(interpolation)
        + r"""
extern "C" __global__ void evaluate_weights(
    const float* __restrict__ distances,
    float* __restrict__ output,
    const int count
) {
    const int index = (int)(blockDim.x * blockIdx.x + threadIdx.x);
    if (index < count) {
        output[index] = pixtreme_weight(distances[index]);
    }
}
"""
    )
    device_distances = cp.asarray(distances)
    output = cp.empty_like(device_distances)
    kernel = cp.RawKernel(source, "evaluate_weights")

    kernel((1,), (32,), (device_distances, output, np.int32(distances.size)))

    expected = np.asarray(
        [_reference_weight(interpolation, float(distance)) for distance in distances], dtype=np.float32
    )
    # 3e-7 covers CUDA sinf at integral Lanczos zeros and fp32 coefficient rounding.
    np.testing.assert_allclose(output.get(), expected, rtol=3e-7, atol=3e-7)


def test_runtime_selected_consumers_embed_the_canonical_device_fragment_once() -> None:
    """REQ-TEST-003 structure contract: runtime-selected kernels embed exactly one shared point-weight preamble."""
    import pixtreme._composite.merge as composite_merge
    import pixtreme._transform.resize as resize
    import pixtreme._transform.warp_affine as warp_affine

    for source in (
        composite_merge._COMPOSITE_KERNEL_SOURCE,
        resize._RESIZE_KERNEL_SOURCE,
        warp_affine._WARP_AFFINE_KERNEL_SOURCE,
    ):
        assert source.count(_POINT_INTERPOLATION_DEVICE_SOURCE) == 1
        assert source.count("__device__ float pixtreme_keys_weight(") == 1
        assert source.count("__device__ float pixtreme_mitchell_weight(") == 1
        assert source.count("__device__ float pixtreme_lanczos_weight(") == 1
        assert source.count("__device__ float pixtreme_point_weight(") == 1


def test_python_consumer_kernel_dataflow_binds_shared_interpolation_sources() -> None:
    """REQ-TEST-003 structure contract: runtime and specialized consumer builders bind their returned source dataflow
    to the canonical interpolation substrate."""
    import pixtreme._composite.merge as composite_merge
    import pixtreme._transform.resize as resize
    import pixtreme._transform.warp_affine as warp_affine
    from pixtreme._io.wire import sampling

    for module in (composite_merge, resize, warp_affine):
        source = inspect.getsource(module)
        tree = ast.parse(source)
        kernel_assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id.endswith("_KERNEL_SOURCE") for target in node.targets)
        )
        shared_loads = [
            node
            for node in ast.walk(kernel_assignment.value)
            if isinstance(node, ast.Name)
            and node.id == "_POINT_INTERPOLATION_DEVICE_SOURCE"
            and isinstance(node.ctx, ast.Load)
        ]
        assert len(shared_loads) == 1

    sampling_tree = ast.parse(inspect.getsource(sampling))
    functions = {
        node.name: node for node in sampling_tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    subsampled_builder = functions["_subsampled_kernel_source"]
    source_assignment = next(
        node
        for node in subsampled_builder.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "source" for target in node.targets)
    )
    specialized_calls = [
        node
        for node in ast.walk(source_assignment.value)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_specialized_point_weight_source"
    ]
    assert len(specialized_calls) == 1
    assert any(
        isinstance(node, ast.Return) and isinstance(node.value, ast.Name) and node.value.id == "source"
        for node in subsampled_builder.body
    )

    to_weight_builder = functions["_to_weight_function"]
    specialized_returns = [
        node
        for node in ast.walk(to_weight_builder)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "_specialized_point_weight_source"
    ]
    assert len(specialized_returns) == 1


def test_wire_upsampling_embeds_one_branch_free_specialized_fragment() -> None:
    """REQ-TEST-003 structure contract: wire upsampling emits one inline specialized weight without runtime dispatch."""
    from pixtreme._io.wire import sampling

    for interpolation in _POINT_INTERPOLATION_TOKENS:
        source = sampling._subsampled_kernel_source("yuv420p", 8, interpolation, "left")
        expected = _specialized_point_weight_source(interpolation)
        if expected:
            assert source.count(expected) == 1
            assert source.count("__device__ float pixtreme_weight(") == 1
        else:
            assert "__device__ float pixtreme_weight(" not in source
        assert "pixtreme_point_weight" not in source


@pytest.mark.parametrize("interpolation", ("bilinear", "bicubic"))
def test_wire_downsampling_keeps_geometry_owned_distance_scale_in_specialized_fragment(interpolation: str) -> None:
    """REQ-TEST-003 structure contract: wire downsampling selects the shared equation while retaining its 0.5 scale."""
    from pixtreme._io.wire import sampling

    source = sampling._to_subsampled_kernel_source("yuv420p", 8, interpolation, "left")
    expected = _specialized_point_weight_source(interpolation, distance_scale="0.5f")

    assert source.count(expected) == 1
    assert source.count("__device__ float pixtreme_weight(") == 1
    assert "pixtreme_point_weight" not in source


def test_consumer_token_subsets_are_derived_without_renumbering_runtime_arguments() -> None:
    """REQ-TEST-003 structure contract: consumer subsets preserve canonical runtime indices and area stays index eight."""
    import pixtreme._composite.merge as composite_merge
    import pixtreme._transform.resize as resize
    import pixtreme._transform.warp_affine as warp_affine
    from pixtreme._io.wire import sampling

    point_and_area = (*_POINT_INTERPOLATION_TOKENS, "area")
    assert composite_merge._COMPOSITE_INTERPOLATION_TOKENS == _POINT_INTERPOLATION_TOKENS
    assert resize._INTERPOLATION_TOKENS == point_and_area
    assert warp_affine._INTERPOLATION_TOKENS == point_and_area
    assert sampling._INTERPOLATION_TOKENS == _POINT_INTERPOLATION_TOKENS
    assert sampling._TO_INTERPOLATION_TOKENS == (
        _POINT_INTERPOLATION_TOKENS[0],
        _POINT_INTERPOLATION_TOKENS[1],
        _POINT_INTERPOLATION_TOKENS[2],
        "area",
    )
