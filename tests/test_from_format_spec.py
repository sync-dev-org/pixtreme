"""Specification tests for the from_<format> family."""

from __future__ import annotations

import inspect
import math
import re
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import pixtreme as px

INTERPOLATIONS = (
    "nearest",
    "bilinear",
    "bicubic",
    "b-spline",
    "mitchell",
    "lanczos2",
    "lanczos3",
    "lanczos4",
)
SITING_OFFSETS = {
    "left": (0.0, 0.5),
    "center": (0.5, 0.5),
    "topleft": (0.0, 0.0),
}
COLORSPACES = ("sRGB", "Rec.709", "Rec.2020", "ACES2065-1", "ACEScg", "S-Gamut3", "S-Gamut3.Cine")
GAMMAS = ("linear", "srgb", "rec709", "bt1886", "pq", "hlg", "s-log3", "logc4", "cineon", "2.2", "2.4", "2.6")
MATRICES = ("bt601", "bt709", "bt2020", "native")
FROM_FORMAT_CASES = (
    ("from_uyvy422", [128, 16, 128, 235], np.uint8, {"width": 2, "height": 1}),
    ("from_v210", np.zeros(32, dtype=np.uint32), np.uint32, {"width": 2, "height": 1}),
    ("from_nv12", [16, 235, 16, 235, 128, 128], np.uint8, {"width": 2, "height": 2}),
    (
        "from_p010",
        np.asarray([64, 940, 64, 940, 512, 512], dtype=np.uint16) << 6,
        np.uint16,
        {"width": 2, "height": 2},
    ),
    ("from_yuv420p", [16, 235, 16, 235, 128, 128], np.uint8, {"width": 2, "height": 2}),
    ("from_yuv422p", [16, 235, 128, 128], np.uint8, {"width": 2, "height": 1}),
    ("from_yuv444p", [64, 512, 960], np.uint16, {"width": 1, "height": 1}),
    ("from_yuva444p", [256, 512, 768, 1023], np.uint16, {"width": 1, "height": 1}),
)


def _device(values: Any, *, dtype: Any) -> Any:
    import cupy as cp

    return cp.asarray(np.asarray(values, dtype=dtype))


def _actionable(error: pytest.ExceptionInfo[ValueError]) -> None:
    message = str(error.value)
    assert "why=" in message
    assert "; what=" in message
    assert "; how=" in message


def _range_reference(codes: np.ndarray, *, bit_depth: int, range: str, alpha: bool = False) -> np.ndarray:
    """Independent H.273 code-position oracle for Y, Cb, Cr, and optional full-scale A."""
    values = codes.astype(np.float32)
    maximum = np.float32((1 << bit_depth) - 1)
    if range == "full":
        return values / maximum

    scale = np.float32(1 << (bit_depth - 8))
    lower = np.float32(16.0) * scale
    extents = np.asarray((219.0, 224.0, 224.0), dtype=np.float32) * scale
    color = (values[..., :3] - lower) / extents
    if not alpha:
        return color
    return np.concatenate((color, values[..., 3:4] / maximum), axis=2)


def _weight(interpolation: str, distance: float) -> float:
    """Independent fp64 form of the resize-family point kernels."""
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
    return math.sin(math.pi * x) * math.sin(math.pi * x / lobes) / ((math.pi * x) ** 2 / lobes)


def _axis_plan(coordinate: float, extent: int, interpolation: str) -> tuple[tuple[int, float], ...]:
    if interpolation == "nearest":
        index = min(max(math.floor(coordinate + 0.5), 0), extent - 1)
        return ((index, 1.0),)
    base = math.floor(coordinate)
    if interpolation == "bilinear":
        start, count = base, 2
    elif interpolation.startswith("lanczos"):
        lobes = int(interpolation.removeprefix("lanczos"))
        start, count = base - (lobes - 1), 2 * lobes
    else:
        start, count = base - 1, 4
    raw = [
        (min(max(index, 0), extent - 1), _weight(interpolation, coordinate - index))
        for index in range(start, start + count)
    ]
    total = sum(weight for _index, weight in raw)
    return tuple((index, weight / total) for index, weight in raw)


def _upsample_reference(
    plane: np.ndarray,
    *,
    height: int,
    width: int,
    interpolation: str,
    subsample_x: int,
    subsample_y: int,
    offset: tuple[float, float],
) -> np.ndarray:
    """Independent separable evaluation of the sheet's chroma sample coordinates."""
    result = np.empty((height, width), dtype=np.float64)
    for y in range(height):
        vertical = (
            _axis_plan((y - offset[1]) / subsample_y, plane.shape[0], interpolation) if subsample_y > 1 else ((y, 1.0),)
        )
        for x in range(width):
            horizontal = _axis_plan((x - offset[0]) / subsample_x, plane.shape[1], interpolation)
            result[y, x] = sum(
                float(plane[source_y, source_x]) * weight_y * weight_x
                for source_y, weight_y in vertical
                for source_x, weight_x in horizontal
            )
    return result.astype(np.float32)


def _expected_frame(
    y: np.ndarray,
    cb: np.ndarray,
    cr: np.ndarray,
    *,
    interpolation: str,
    offset: tuple[float, float],
    subsample_x: int,
    subsample_y: int,
) -> np.ndarray:
    return np.stack(
        (
            y.astype(np.float32),
            _upsample_reference(
                cb,
                height=y.shape[0],
                width=y.shape[1],
                interpolation=interpolation,
                subsample_x=subsample_x,
                subsample_y=subsample_y,
                offset=offset,
            ),
            _upsample_reference(
                cr,
                height=y.shape[0],
                width=y.shape[1],
                interpolation=interpolation,
                subsample_x=subsample_x,
                subsample_y=subsample_y,
                offset=offset,
            ),
        ),
        axis=2,
    )


def _pack_v210_row(y: list[int], cb: list[int], cr: list[int], *, width: int) -> np.ndarray:
    """Independent word packer used only to build from_v210 fixtures."""
    words = np.zeros(((width + 47) // 48) * 32, dtype=np.uint32)
    for group_start in range(0, width, 6):
        group = group_start // 6
        ys = [y[min(group_start + offset, width - 1)] for offset in range(6)]
        chroma_start = group_start // 2
        cbs = [cb[min(chroma_start + offset, len(cb) - 1)] for offset in range(3)]
        crs = [cr[min(chroma_start + offset, len(cr) - 1)] for offset in range(3)]
        base = group * 4
        words[base] = cbs[0] | (ys[0] << 10) | (crs[0] << 20)
        words[base + 1] = ys[1] | (cbs[1] << 10) | (ys[2] << 20)
        words[base + 2] = crs[1] | (ys[3] << 10) | (cbs[2] << 20)
        words[base + 3] = ys[4] | (crs[2] << 10) | (ys[5] << 20)
    return words


@pytest.mark.parametrize("ndi_shape", (False, True))
def test_from_uyvy422_accepts_flat_and_ndi_shapes_with_the_same_pixel_layout(ndi_shape: bool) -> None:
    """v1-format-boundary acceptance 24, 28, and 32: UYVY shape forms decode U0 Y0 V0 Y1 with half-up nearest ties."""
    height, width = 2, 4
    packed = np.asarray(
        [
            [[10, 20], [30, 40], [50, 60], [70, 80]],
            [[90, 100], [110, 120], [130, 140], [150, 160]],
        ],
        dtype=np.uint8,
    )
    source = packed if ndi_shape else packed.reshape(-1)
    expected_codes = _expected_frame(
        packed[..., 1],
        packed[:, ::2, 0],
        packed[:, 1::2, 0],
        interpolation="nearest",
        offset=(0.0, 0.0),
        subsample_x=2,
        subsample_y=1,
    )

    result = px.io.from_uyvy422(
        _device(source, dtype=np.uint8),
        width=width,
        height=height,
        range="full",
        interpolation="nearest",
    )

    np.testing.assert_allclose(
        px.io.to_array(
            result,
        ).get(),
        _range_reference(expected_codes, bit_depth=8, range="full"),
        rtol=0.0,
        atol=2e-7,
    )


def test_from_v210_decodes_words_and_ignores_the_zero_filled_128_byte_row_padding() -> None:
    """v1-format-boundary acceptance 1, 5, 27, 29, and 33: v210 words and row alignment match a hand-built fixture."""
    width, height = 7, 1
    y = [100, 101, 102, 103, 104, 105, 106]
    cb = [200, 201, 202, 203]
    cr = [300, 301, 302, 303]
    source = _pack_v210_row(y, cb, cr, width=width)
    assert source.shape == (32,)
    assert np.count_nonzero(source[8:]) == 0
    expected_codes = _expected_frame(
        np.asarray(y, dtype=np.float32)[None, :],
        np.asarray(cb, dtype=np.float32)[None, :],
        np.asarray(cr, dtype=np.float32)[None, :],
        interpolation="nearest",
        offset=(0.0, 0.0),
        subsample_x=2,
        subsample_y=1,
    )

    result = px.io.from_v210(
        _device(source, dtype=np.uint32),
        width=width,
        height=height,
        range="full",
        interpolation="nearest",
    )

    np.testing.assert_allclose(
        px.io.to_array(
            result,
        ).get(),
        _range_reference(expected_codes, bit_depth=10, range="full"),
        rtol=0.0,
        atol=2e-7,
    )


@pytest.mark.parametrize(
    ("name", "bit_depth", "dtype", "container_shift"),
    (("from_nv12", 8, np.uint8, 0), ("from_p010", 10, np.uint16, 6)),
)
def test_semiplanar_420_formats_decode_y_then_interleaved_cbcr(
    name: str,
    bit_depth: int,
    dtype: type[np.generic],
    container_shift: int,
) -> None:
    """v1-format-boundary acceptance 1, 5, 27, 30, and 33: NV12/P010 share Y + interleaved UV layout, with P010 MSB alignment."""
    height, width = 4, 4
    y = np.arange(height * width, dtype=np.uint16).reshape(height, width) + 64
    cb = np.asarray([[128, 256], [512, 768]], dtype=np.uint16)
    cr = np.asarray([[900, 700], [500, 300]], dtype=np.uint16)
    uv = np.stack((cb, cr), axis=2).reshape(-1)
    maximum = (1 << bit_depth) - 1
    codes = np.concatenate((y.reshape(-1), uv)) & maximum
    source = (codes << container_shift).astype(dtype)
    expected_codes = _expected_frame(
        y & maximum,
        cb & maximum,
        cr & maximum,
        interpolation="nearest",
        offset=SITING_OFFSETS["topleft"],
        subsample_x=2,
        subsample_y=2,
    )

    result = getattr(px.io, name)(
        _device(source, dtype=dtype),
        width=width,
        height=height,
        range="full",
        siting="topleft",
        interpolation="nearest",
    )

    np.testing.assert_allclose(
        px.io.to_array(
            result,
        ).get(),
        _range_reference(expected_codes, bit_depth=bit_depth, range="full"),
        rtol=0.0,
        atol=2e-7,
    )


@pytest.mark.parametrize(
    ("name", "bit_depth", "height", "width", "plane_count", "subsample_x", "subsample_y"),
    (
        ("from_yuv420p", 8, 2, 2, 3, 2, 2),
        ("from_yuv420p", 10, 2, 2, 3, 2, 2),
        ("from_yuv422p", 8, 1, 2, 3, 2, 1),
        ("from_yuv422p", 10, 1, 2, 3, 2, 1),
        ("from_yuv422p", 12, 1, 2, 3, 2, 1),
        ("from_yuv444p", 10, 1, 1, 3, 1, 1),
        ("from_yuv444p", 12, 1, 1, 3, 1, 1),
        ("from_yuva444p", 12, 1, 1, 4, 1, 1),
    ),
)
def test_planar_bit_depths_use_lower_aligned_codes_and_plane_order(
    name: str,
    bit_depth: int,
    height: int,
    width: int,
    plane_count: int,
    subsample_x: int,
    subsample_y: int,
) -> None:
    """v1-format-boundary acceptance 3, 5, 12, 15, 27, 32, and 33: planar bit depths, lower alignment, plane order, and alpha scale are fixed."""
    dtype = np.uint8 if bit_depth == 8 else np.uint16
    maximum = (1 << bit_depth) - 1
    y = np.arange(height * width, dtype=np.uint16).reshape(height, width) + maximum // 5
    chroma_shape = (height // subsample_y, width // subsample_x)
    cb = np.full(chroma_shape, maximum // 3, dtype=np.uint16)
    cr = np.full(chroma_shape, maximum * 2 // 3, dtype=np.uint16)
    planes = [y.reshape(-1), cb.reshape(-1), cr.reshape(-1)]
    if plane_count == 4:
        alpha = np.full((height, width), maximum * 3 // 4, dtype=np.uint16)
        planes.append(alpha.reshape(-1))
    high_junk = 0 if bit_depth == 8 else 0xF000
    source = (np.concatenate(planes) | high_junk).astype(dtype)
    kwargs: dict[str, object] = {"width": width, "height": height, "bit_depth": bit_depth, "range": "legal"}
    if subsample_x > 1:
        kwargs["interpolation"] = "nearest"
    if subsample_y > 1:
        kwargs["siting"] = "topleft"

    result = getattr(px.io, name)(_device(source, dtype=dtype), **kwargs)

    color_codes = _expected_frame(
        y & maximum,
        cb & maximum,
        cr & maximum,
        interpolation="nearest",
        offset=SITING_OFFSETS["topleft"],
        subsample_x=subsample_x,
        subsample_y=subsample_y,
    )
    if plane_count == 4:
        expected_codes = np.concatenate((color_codes, (alpha & maximum)[..., None]), axis=2)
    else:
        expected_codes = color_codes
    np.testing.assert_allclose(
        px.io.to_array(
            result,
        ).get(),
        _range_reference(expected_codes, bit_depth=bit_depth, range="legal", alpha=plane_count == 4),
        rtol=0.0,
        atol=2e-7,
    )
    assert result.channels == (("Y", "Cb", "Cr", "A") if plane_count == 4 else ("Y", "Cb", "Cr"))


def test_from_yuv422p_replaces_the_old_name_and_preserves_ten_bit_range_headroom() -> None:
    """v1-format-boundary acceptance 4, 12, 16, and 37: renamed 10-bit planar decoding keeps below/above-legal code positions unclipped."""
    y = np.asarray([0, 1023], dtype=np.uint16)
    cb = np.asarray([64], dtype=np.uint16)
    cr = np.asarray([960], dtype=np.uint16)
    source = np.concatenate((y, cb, cr))
    expected_codes = np.asarray([[[0, 64, 960], [1023, 64, 960]]], dtype=np.uint16)

    result = px.io.from_yuv422p(
        _device(source, dtype=np.uint16),
        width=2,
        height=1,
        bit_depth=10,
        interpolation="nearest",
    )
    actual = px.io.to_array(
        result,
    ).get()

    np.testing.assert_allclose(
        actual,
        _range_reference(expected_codes, bit_depth=10, range="legal"),
        rtol=0.0,
        atol=2e-7,
    )
    assert actual[0, 0, 0] < 0.0
    assert actual[0, 1, 0] > 1.0
    assert not hasattr(px, "from_yuv422p10le")


@pytest.mark.parametrize("siting", tuple(SITING_OFFSETS))
@pytest.mark.parametrize("interpolation", INTERPOLATIONS)
def test_from_yuv420p_filter_and_siting_match_the_independent_coordinate_oracle(
    siting: str,
    interpolation: str,
) -> None:
    """v1-format-boundary acceptance 17, 20-22, 24-26, and 35: every 420 filter/siting pair follows the sheet coordinates."""
    height, width = 6, 8
    y = np.arange(height * width, dtype=np.uint8).reshape(height, width) + 32
    cb = np.asarray([[16, 64, 128, 240], [32, 96, 160, 224], [48, 112, 176, 208]], dtype=np.uint8)
    cr = np.flip(cb, axis=(0, 1)).copy()
    source = np.concatenate((y.reshape(-1), cb.reshape(-1), cr.reshape(-1)))
    expected_codes = _expected_frame(
        y,
        cb,
        cr,
        interpolation=interpolation,
        offset=SITING_OFFSETS[siting],
        subsample_x=2,
        subsample_y=2,
    )

    result = px.io.from_yuv420p(
        _device(source, dtype=np.uint8),
        width=width,
        height=height,
        range="full",
        siting=siting,
        interpolation=interpolation,
    )

    # 3e-6 covers fp32 CUDA weight evaluation versus the independent fp64
    # oracle, while remaining below one 8-bit code step by three orders.
    np.testing.assert_allclose(
        px.io.to_array(
            result,
        ).get(),
        _range_reference(expected_codes, bit_depth=8, range="full"),
        rtol=0.0,
        atol=3e-6,
    )


@pytest.mark.parametrize("interpolation", INTERPOLATIONS)
def test_from_yuv422p_uses_the_same_horizontal_filter_family_without_vertical_filtering(interpolation: str) -> None:
    """v1-format-boundary acceptance 19, 21, 22, 24-26, and 35: 422 is horizontally co-sited and vertically full."""
    height, width = 2, 6
    y = np.arange(height * width, dtype=np.uint8).reshape(height, width) + 32
    cb = np.asarray([[16, 96, 240], [240, 96, 16]], dtype=np.uint8)
    cr = np.asarray([[240, 128, 32], [32, 128, 240]], dtype=np.uint8)
    source = np.concatenate((y.reshape(-1), cb.reshape(-1), cr.reshape(-1)))
    expected_codes = _expected_frame(
        y,
        cb,
        cr,
        interpolation=interpolation,
        offset=(0.0, 0.0),
        subsample_x=2,
        subsample_y=1,
    )

    result = px.io.from_yuv422p(
        _device(source, dtype=np.uint8),
        width=width,
        height=height,
        bit_depth=8,
        range="full",
        interpolation=interpolation,
    )

    np.testing.assert_allclose(
        px.io.to_array(
            result,
        ).get(),
        _range_reference(expected_codes, bit_depth=8, range="full"),
        rtol=0.0,
        atol=3e-6,
    )


@pytest.mark.parametrize(
    ("name", "bit_depth", "dtype", "shift"),
    (("from_nv12", 8, np.uint8, 0), ("from_p010", 10, np.uint16, 6)),
)
def test_semiplanar_formats_share_center_sited_lanczos_mapping(
    name: str,
    bit_depth: int,
    dtype: type[np.generic],
    shift: int,
) -> None:
    """v1-format-boundary acceptance 18, 21, 22, 26, and 35: NV12/P010 use the same siting/filter coordinates as planar 420."""
    height, width = 6, 8
    maximum = (1 << bit_depth) - 1
    y = np.arange(height * width, dtype=np.uint16).reshape(height, width) * 3 & maximum
    cb = np.asarray([[10, 200, 500, 900], [100, 300, 600, 800], [200, 400, 700, 1000]], dtype=np.uint16) & maximum
    cr = np.flip(cb, axis=1).copy()
    uv = np.stack((cb, cr), axis=2).reshape(-1)
    source = (np.concatenate((y.reshape(-1), uv)) << shift).astype(dtype)
    expected_codes = _expected_frame(
        y,
        cb,
        cr,
        interpolation="lanczos3",
        offset=SITING_OFFSETS["center"],
        subsample_x=2,
        subsample_y=2,
    )

    result = getattr(px.io, name)(
        _device(source, dtype=dtype),
        width=width,
        height=height,
        range="full",
        siting="center",
        interpolation="lanczos3",
    )

    np.testing.assert_allclose(
        px.io.to_array(
            result,
        ).get(),
        _range_reference(expected_codes, bit_depth=bit_depth, range="full"),
        rtol=0.0,
        atol=3e-6,
    )


def test_siting_tokens_have_numerically_distinct_impulse_centroids() -> None:
    """v1-format-boundary acceptance 17, 18, and 37: an interior chroma impulse distinguishes all three sample phases by centroid."""
    height = width = 12
    y = np.zeros((height, width), dtype=np.uint8)
    cb = np.zeros((height // 2, width // 2), dtype=np.uint8)
    cr = np.zeros_like(cb)
    cb[2, 2] = 255
    source = np.concatenate((y.reshape(-1), cb.reshape(-1), cr.reshape(-1)))
    centroids: dict[str, tuple[float, float]] = {}

    for siting, (offset_x, offset_y) in SITING_OFFSETS.items():
        result = px.io.from_yuv420p(
            _device(source, dtype=np.uint8),
            width=width,
            height=height,
            range="full",
            siting=siting,
            interpolation="bilinear",
        )
        plane = px.io.to_array(
            result,
        ).get()[..., 1]
        yy, xx = np.indices(plane.shape)
        centroids[siting] = (float((plane * xx).sum() / plane.sum()), float((plane * yy).sum() / plane.sum()))
        np.testing.assert_allclose(centroids[siting], (4.0 + offset_x, 4.0 + offset_y), rtol=0.0, atol=1e-6)

    assert len(set(centroids.values())) == 3


@pytest.mark.parametrize(
    ("name", "source", "dtype", "kwargs", "channels"),
    (
        ("from_uyvy422", [128, 16, 128, 235], np.uint8, {"width": 2, "height": 1}, ("Y", "Cb", "Cr")),
        (
            "from_v210",
            np.zeros(32, dtype=np.uint32),
            np.uint32,
            {"width": 2, "height": 1},
            ("Y", "Cb", "Cr"),
        ),
        ("from_nv12", [16, 235, 16, 235, 128, 128], np.uint8, {"width": 2, "height": 2}, ("Y", "Cb", "Cr")),
        (
            "from_p010",
            np.asarray([64, 940, 64, 940, 512, 512], dtype=np.uint16) << 6,
            np.uint16,
            {"width": 2, "height": 2},
            ("Y", "Cb", "Cr"),
        ),
        (
            "from_yuv420p",
            [16, 235, 16, 235, 128, 128],
            np.uint8,
            {"width": 2, "height": 2},
            ("Y", "Cb", "Cr"),
        ),
        ("from_yuv422p", [16, 235, 128, 128], np.uint8, {"width": 2, "height": 1}, ("Y", "Cb", "Cr")),
        (
            "from_yuv444p",
            [64, 512, 960],
            np.uint16,
            {"width": 1, "height": 1},
            ("Y", "Cb", "Cr"),
        ),
        (
            "from_yuva444p",
            [256, 512, 768, 1023],
            np.uint16,
            {"width": 1, "height": 1},
            ("Y", "Cb", "Cr", "A"),
        ),
    ),
)
def test_all_from_formats_return_fixed_contiguous_fp32_placeholder_frames(
    name: str,
    source: Any,
    dtype: type[np.generic],
    kwargs: dict[str, int],
    channels: tuple[str, ...],
) -> None:
    """v1-format-boundary acceptance 1 and 32: all eight entries return fixed-metadata contiguous fp32 Frames."""
    result = getattr(px.io, name)(_device(source, dtype=dtype), **kwargs)

    assert isinstance(result, px.core.Frame)
    assert result.dtype == np.dtype(np.float32)
    assert result.data.flags.c_contiguous
    assert result.shape == (kwargs["height"], kwargs["width"], len(channels))
    assert (result.colorspace, result.gamma, result.channels) == ("Rec.709", "rec709", channels)


@pytest.mark.parametrize("name", [case[0] for case in FROM_FORMAT_CASES])
def test_from_format_signatures_expose_keyword_only_metadata_overrides(name: str) -> None:
    """v1-from-format-metadata acceptance 1: all eight signatures expose optional keyword-only metadata claims."""
    parameters = inspect.signature(getattr(px.io, name)).parameters

    for axis in ("colorspace", "gamma"):
        parameter = parameters[axis]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.annotation == "str | None"
        assert parameter.default is None


@pytest.mark.parametrize(
    ("colorspace", "gamma", "expected"),
    (
        (None, None, ("Rec.709", "rec709")),
        ("ACEScg", None, ("ACEScg", "rec709")),
        (None, "linear", ("Rec.709", "linear")),
        ("ACEScg", "linear", ("ACEScg", "linear")),
    ),
)
@pytest.mark.parametrize(
    ("name", "source", "dtype", "kwargs"),
    FROM_FORMAT_CASES,
    ids=[case[0] for case in FROM_FORMAT_CASES],
)
def test_from_format_metadata_overrides_resolve_independently_over_placeholders(
    name: str,
    source: Any,
    dtype: type[np.generic],
    kwargs: dict[str, int],
    colorspace: str | None,
    gamma: str | None,
    expected: tuple[str, str],
) -> None:
    """v1-from-format-metadata acceptance 1-2: None preserves each placeholder and explicit claims win per axis."""
    result = getattr(px.io, name)(
        _device(source, dtype=dtype),
        **kwargs,
        colorspace=colorspace,
        gamma=gamma,
    )

    assert (result.colorspace, result.gamma) == expected


@pytest.mark.parametrize(
    ("axis", "accepted"),
    (("colorspace", COLORSPACES), ("gamma", GAMMAS), ("matrix", MATRICES)),
)
@pytest.mark.parametrize(
    ("name", "source", "dtype", "kwargs"),
    FROM_FORMAT_CASES,
    ids=[case[0] for case in FROM_FORMAT_CASES],
)
def test_from_format_metadata_tokens_match_frame_assignment_domains(
    name: str,
    source: Any,
    dtype: type[np.generic],
    kwargs: dict[str, int],
    axis: str,
    accepted: tuple[str, ...],
) -> None:
    """v1-from-format-metadata acceptance 3; v1-color-semantics acceptance 5 and 27.

    Format metadata uses the Frame token domains and rejects values outside them.
    """
    function = getattr(px.io, name)
    device_source = _device(source, dtype=dtype)

    for token in accepted:
        result = function(device_source, **kwargs, **{axis: token})
        assert getattr(result, axis) == token

    with pytest.raises(ValueError) as error:
        function(device_source, **kwargs, **{axis: f"unknown-{axis}"})
    _actionable(error)
    assert repr(accepted) in str(error.value)


@pytest.mark.parametrize(
    ("name", "source", "dtype", "kwargs"),
    FROM_FORMAT_CASES,
    ids=[case[0] for case in FROM_FORMAT_CASES],
)
def test_from_format_metadata_overrides_do_not_change_pixels(
    name: str,
    source: Any,
    dtype: type[np.generic],
    kwargs: dict[str, int],
) -> None:
    """v1-from-format-metadata acceptance 4: metadata claims leave decoded pixels bit-identical."""
    import cupy as cp

    device_source = _device(source, dtype=dtype)
    default = getattr(px.io, name)(device_source, **kwargs)
    explicit = getattr(px.io, name)(device_source, **kwargs, colorspace="ACEScg", gamma="linear")

    assert cp.array_equal(
        px.io.to_array(
            default,
        ),
        px.io.to_array(
            explicit,
        ),
    )


@pytest.mark.parametrize(
    ("name", "source", "dtype", "kwargs"),
    FROM_FORMAT_CASES,
    ids=[case[0] for case in FROM_FORMAT_CASES],
)
def test_from_format_frames_still_allow_post_construction_metadata_correction(
    name: str,
    source: Any,
    dtype: type[np.generic],
    kwargs: dict[str, int],
) -> None:
    """v1-from-format-metadata acceptance 5: validated attribute assignment remains a pixel-neutral correction path."""
    import cupy as cp

    result = getattr(px.io, name)(_device(source, dtype=dtype), **kwargs)
    pixels = px.io.to_array(
        result,
    ).copy()

    result.colorspace = "S-Gamut3"
    result.gamma = "s-log3"

    assert (result.colorspace, result.gamma) == ("S-Gamut3", "s-log3")
    assert cp.array_equal(
        px.io.to_array(
            result,
        ),
        pixels,
    )


@pytest.mark.parametrize(
    ("name", "source", "dtype", "kwargs"),
    (
        ("from_uyvy422", [128, 16, 128, 235], np.uint8, {"width": 2, "height": 1}),
        ("from_v210", np.zeros(32), np.uint32, {"width": 2, "height": 1}),
        ("from_nv12", [16, 235, 16, 235, 128, 128], np.uint8, {"width": 2, "height": 2}),
        ("from_p010", np.zeros(6), np.uint16, {"width": 2, "height": 2}),
        ("from_yuv420p", [16, 235, 16, 235, 128, 128], np.uint8, {"width": 2, "height": 2}),
        ("from_yuv422p", [16, 235, 128, 128], np.uint8, {"width": 2, "height": 1}),
        ("from_yuv444p", [64, 512, 960], np.uint16, {"width": 1, "height": 1}),
        ("from_yuva444p", [64, 512, 960, 1023], np.uint16, {"width": 1, "height": 1}),
    ),
)
def test_all_from_formats_reject_unknown_range_tokens_actionably(
    name: str,
    source: Any,
    dtype: type[np.generic],
    kwargs: dict[str, int],
) -> None:
    """v1-format-boundary acceptance 11: range is a case-sensitive three-element fail-fast token axis."""
    with pytest.raises(ValueError) as error:
        getattr(px.io, name)(_device(source, dtype=dtype), range="FULL", **kwargs)
    _actionable(error)
    assert "('legal', 'full')" in str(error.value)


@pytest.mark.parametrize("name", ("from_yuv420p", "from_nv12", "from_p010"))
def test_420_formats_reject_unknown_siting_and_interpolation_tokens_actionably(name: str) -> None:
    """v1-format-boundary acceptance 11, 17, 18, and 21: 420 token axes enumerate accepted recovery values."""
    dtype = np.uint16 if name == "from_p010" else np.uint8
    source = _device(np.zeros(6), dtype=dtype)

    with pytest.raises(ValueError) as siting_error:
        getattr(px.io, name)(source, width=2, height=2, siting="top-left")
    _actionable(siting_error)
    assert tuple(SITING_OFFSETS) == ("left", "center", "topleft")
    assert repr(tuple(SITING_OFFSETS)) in str(siting_error.value)

    with pytest.raises(ValueError) as interpolation_error:
        getattr(px.io, name)(source, width=2, height=2, interpolation="area")
    _actionable(interpolation_error)
    assert repr(INTERPOLATIONS) in str(interpolation_error.value)


@pytest.mark.parametrize(
    ("name", "valid_depths"),
    (
        ("from_yuv420p", (8, 10)),
        ("from_yuv422p", (8, 10, 12)),
        ("from_yuv444p", (10, 12)),
        ("from_yuva444p", (12,)),
    ),
)
def test_planar_formats_reject_bit_depths_outside_their_closed_domains(
    name: str, valid_depths: tuple[int, ...]
) -> None:
    """v1-format-boundary acceptance 3 and 11: every planar bit_depth domain is closed and actionable."""
    source = _device(np.zeros(6), dtype=np.uint8)
    kwargs = {"width": 2, "height": 2} if name == "from_yuv420p" else {"width": 2, "height": 1}

    with pytest.raises(ValueError) as error:
        getattr(px.io, name)(source, bit_depth=9, **kwargs)
    _actionable(error)
    assert repr(valid_depths) in str(error.value)


@pytest.mark.parametrize(
    ("name", "source", "dtype", "kwargs"),
    (
        ("from_uyvy422", [128, 16, 128, 235], np.uint8, {"width": 2, "height": 1}),
        ("from_v210", np.zeros(32), np.uint32, {"width": 2, "height": 1}),
        ("from_nv12", np.zeros(6), np.uint8, {"width": 2, "height": 2}),
        ("from_p010", np.zeros(6), np.uint16, {"width": 2, "height": 2}),
        ("from_yuv420p", np.zeros(6), np.uint8, {"width": 2, "height": 2}),
        ("from_yuv422p", np.zeros(4), np.uint8, {"width": 2, "height": 1}),
        ("from_yuv444p", np.zeros(3), np.uint16, {"width": 1, "height": 1}),
        ("from_yuva444p", np.zeros(4), np.uint16, {"width": 1, "height": 1}),
    ),
)
def test_from_formats_reject_non_cupy_wrong_dtype_size_and_noncontiguous_buffers(
    name: str,
    source: Any,
    dtype: type[np.generic],
    kwargs: dict[str, int],
) -> None:
    """v1-format-boundary acceptance 5, 27-30, and 37: buffer type, dtype, exact layout size, and contiguity fail fast."""
    import cupy as cp

    function: Callable[..., object] = getattr(px.io, name)
    valid = _device(source, dtype=dtype)
    with pytest.raises(ValueError) as type_error:
        function(np.asarray(source, dtype=dtype), **kwargs)
    _actionable(type_error)

    wrong_dtype = cp.asarray(valid, dtype=cp.uint16 if valid.dtype == cp.uint8 else cp.uint8)
    with pytest.raises(ValueError) as dtype_error:
        function(wrong_dtype, **kwargs)
    _actionable(dtype_error)

    with pytest.raises(ValueError) as size_error:
        function(valid[:-1], **kwargs)
    _actionable(size_error)

    doubled = cp.concatenate((valid, valid))
    noncontiguous = doubled[::2]
    assert noncontiguous.size == valid.size and not noncontiguous.flags.c_contiguous
    with pytest.raises(ValueError) as contiguous_error:
        function(noncontiguous, **kwargs)
    _actionable(contiguous_error)


@pytest.mark.parametrize(
    ("name", "source", "kwargs", "bad_width", "bad_height"),
    (
        ("from_uyvy422", [0, 0, 0, 0], {"width": 2, "height": 1}, 1, 0),
        ("from_nv12", np.zeros(6), {"width": 2, "height": 2}, 1, 1),
        ("from_p010", np.zeros(6), {"width": 2, "height": 2}, 1, 1),
        ("from_yuv420p", np.zeros(6), {"width": 2, "height": 2}, 1, 1),
        ("from_yuv422p", np.zeros(4), {"width": 2, "height": 1}, 1, 0),
    ),
)
def test_subsampled_even_dimension_constraints_fail_fast(
    name: str,
    source: Any,
    kwargs: dict[str, int],
    bad_width: int,
    bad_height: int,
) -> None:
    """v1-format-boundary acceptance 10 and 37: 420 requires even width/height and 422 requires even width."""
    dtype = np.uint16 if name == "from_p010" else np.uint8
    valid = _device(source, dtype=dtype)
    with pytest.raises(ValueError) as width_error:
        getattr(px.io, name)(valid, width=bad_width, height=kwargs["height"])
    _actionable(width_error)
    with pytest.raises(ValueError) as height_error:
        getattr(px.io, name)(valid, width=kwargs["width"], height=bad_height)
    _actionable(height_error)


def test_v210_accepts_odd_width_but_requires_the_aligned_row_word_count() -> None:
    """v1-format-boundary acceptance 10, 29, and 37: v210 has no even-width restriction and validates 128-byte row storage."""
    source = _device(np.zeros(32), dtype=np.uint32)
    result = px.io.from_v210(source, width=1, height=1, interpolation="nearest")
    assert result.shape == (1, 1, 3)

    with pytest.raises(ValueError) as error:
        px.io.from_v210(source[:-1], width=1, height=1)
    _actionable(error)
    assert "32" in str(error.value)


def test_from_format_kernel_entries_are_declared_and_each_public_call_launches_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1-format-boundary acceptance 25: factories bind declared entries and public paths launch one fused pass."""
    import pixtreme._io.wire.uyvy422 as subsampled_module
    import pixtreme._io.wire.yuv444p as planar_module

    subsampled_kernel = subsampled_module._from_kernel(8, "nearest", "topleft")
    subsampled_source = subsampled_module._subsampled_kernel_source("uyvy422", 8, "nearest", "topleft")
    planar_kernel = planar_module._from_kernel(10)
    planar_source = planar_module._planar_444_kernel_source(10, alpha=False)

    for kernel, source in ((subsampled_kernel, subsampled_source), (planar_kernel, planar_source)):
        declaration = rf'extern\s+"C"\s+__global__\s+void\s+{re.escape(kernel.name)}\s*\('
        assert re.search(declaration, source)

    class CountingKernel:
        def __init__(self, kernel: Any) -> None:
            self.kernel = kernel
            self.launches = 0

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            self.launches += 1
            return self.kernel(*args, **kwargs)

    counted_subsampled = CountingKernel(subsampled_kernel)
    counted_planar = CountingKernel(planar_kernel)
    monkeypatch.setattr(subsampled_module, "_from_kernel", lambda *_args: counted_subsampled)
    monkeypatch.setattr(planar_module, "_from_kernel", lambda *_args: counted_planar)

    subsampled = px.io.from_uyvy422(
        _device([128, 16, 128, 235], dtype=np.uint8),
        width=2,
        height=1,
        interpolation="nearest",
    )
    planar = px.io.from_yuv444p(
        _device([64, 512, 960], dtype=np.uint16),
        width=1,
        height=1,
        bit_depth=10,
    )

    assert counted_subsampled.launches == 1
    assert counted_planar.launches == 1
    np.testing.assert_allclose(subsampled.data.get(), [[[0.0, 0.5, 0.5], [1.0, 0.5, 0.5]]], rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(planar.data.get(), [[[0.0, 0.5, 1.0]]], rtol=0.0, atol=1e-6)
