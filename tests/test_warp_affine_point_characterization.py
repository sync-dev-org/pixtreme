"""Issue #56 acceptance 2–3: preserve every bit of point-warp results."""

from __future__ import annotations

import numpy as np
import pytest
from warp_affine_point_reference import BORDERS, POINT_INTERPOLATIONS, point_reference

import pixtreme as px


def _rotation() -> np.ndarray:
    angle = np.deg2rad(5.0)
    linear = 1.01 * np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    center = np.asarray([26.0, 19.0])
    return np.column_stack((linear, center - linear @ center))


# Shapes are (height, width). Fractional phases exercise all taps; canvases
# cross block boundaries and leave partial blocks on both axes.
GEOMETRIES = (
    pytest.param((35, 51), (35, 51), [[1, 0, 0], [0, 1, 0]], False, id="identity"),
    pytest.param((19, 27), (35, 49), [[1.4, 0, 0.31], [0, 1.7, -0.47]], False, id="enlarge"),
    pytest.param((39, 53), (37, 51), _rotation(), False, id="rotate"),
    pytest.param((97, 131), (19, 23), [[0.04, 0, 0.31], [0, 0.06, -0.47]], False, id="strong-shrink"),
    pytest.param((41, 59), (33, 47), [[1, 1.75, -2.31], [-0.25, 1, 0.47]], False, id="shear"),
    pytest.param((37, 51), (19, 23), [[1.3, -0.21, 0.31], [0.15, 0.9, -0.47]], True, id="inverse"),
    pytest.param((1, 1), (19, 21), [[1.2, 0.1, 0.31], [-0.2, 1.3, -0.47]], False, id="one-pixel"),
    pytest.param((1, 5), (19, 21), [[1, 0, 0.31], [0, 1, -0.47]], False, id="one-row"),
    pytest.param((5, 1), (19, 21), [[1, 0, 0.31], [0, 1, -0.47]], True, id="one-column"),
    pytest.param((3, 2), (19, 21), [[0.7, 0.1, 0.31], [-0.2, 1.3, -0.47]], False, id="below-support"),
    pytest.param((73, 89), (19, 21), [[1, 0, -8.31], [0, 1, -6.47]], True, id="period-seam"),
    pytest.param((17, 29), (1, 19), [[1, 0.2, -3.31], [0.1, 1, -0.47]], True, id="one-output-row"),
    pytest.param((17, 29), (19, 1), [[1, 0.2, -3.31], [0.1, 1, -0.47]], True, id="one-output-column"),
    pytest.param((19, 23), (19, 21), [[1, 0, 1e18], [0, 1, -1e18]], True, id="far-border"),
    pytest.param((19, 23), (19, 21), [[1, 0, -8.01], [0, 1, -7.99]], True, id="constant-cutoff"),
)


@pytest.mark.req("REQ-PIX-010")
@pytest.mark.parametrize("interpolation", POINT_INTERPOLATIONS)
@pytest.mark.parametrize("border", BORDERS)
@pytest.mark.parametrize("channels", (1, 2, 3, 4))
@pytest.mark.parametrize("input_shape,output_shape,matrix,inverse", GEOMETRIES)
def test_point_warp_preserves_bits_across_geometry_borders_and_channels(
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
    matrix: object,
    inverse: bool,
    channels: int,
    border: str,
    interpolation: str,
) -> None:
    """画像の幾何変換は補間、境界、channel 数、画像寸法によらず変更前と bit 一致する。

    Characterization: issue #56 acceptance 2–3 freezes the original point
    kernel. Unsigned integer views compare signed zero and all mantissa bits;
    no numerical tolerance or current production source defines the oracle.
    """
    import cupy as cp

    values = np.random.default_rng(5603).uniform(-2.0, 3.0, (*input_shape, channels)).astype(np.float32)
    values.flat[0] = -0.0
    frame = px.io.from_array(
        cp.asarray(values), colorspace="sRGB", gamma="linear", channels=tuple(f"signal_{i}" for i in range(channels))
    )
    declared = np.asarray(matrix, dtype=np.float32)
    height, width = output_shape
    border_value = -0.375
    expected = point_reference(
        frame.data,
        declared,
        inverse=inverse,
        width=width,
        height=height,
        interpolation=interpolation,
        border=border,
        border_value=border_value,
    )
    actual = px.transform.warp_affine(
        frame,
        declared,
        inverse=inverse,
        width=width,
        height=height,
        interpolation=interpolation,
        border=border,
        border_value=border_value if border == "constant" else None,
    )
    np.testing.assert_array_equal(cp.asnumpy(actual.data).view(np.uint32), cp.asnumpy(expected).view(np.uint32))
