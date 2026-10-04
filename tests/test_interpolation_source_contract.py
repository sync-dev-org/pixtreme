"""Contracts for the shared CUDA point-interpolation substrate."""

from __future__ import annotations

import math

import cupy as cp
import numpy as np
import pytest

from pixtreme._core.interpolation import (
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


@pytest.mark.req("REQ-PIX-009")
@pytest.mark.parametrize("interpolation", _POINT_INTERPOLATION_TOKENS[1:])
def test_specialized_point_weight_source_matches_independent_piecewise_boundary_oracle(interpolation: str) -> None:
    """Wire interpolation weights match independent equations at each filter-support boundary."""
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
