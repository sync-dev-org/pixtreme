"""Specification tests for the io.to_<format> family."""

from __future__ import annotations

import inspect
import math
import re
from typing import Any

import numpy as np
import pytest

import pixtreme as px

TO_INTERPOLATIONS = ("nearest", "bilinear", "bicubic", "area")
SITING_OFFSETS = {
    "left": (0.0, 0.5),
    "center": (0.5, 0.5),
    "topleft": (0.0, 0.0),
}


def _frame(values: np.ndarray, *, channels: tuple[str, ...] = ("Y", "Cb", "Cr")) -> px.core.Frame:
    import cupy as cp

    return px.core.Frame(
        data=cp.asarray(np.ascontiguousarray(values)),
        colorspace="Rec.709",
        gamma="rec709",
        channels=channels,
    )


def _actionable(error: pytest.ExceptionInfo[ValueError]) -> None:
    message = str(error.value)
    assert "why=" in message
    assert "; what=" in message
    assert "; how=" in message


def _values(height: int, width: int, *, alpha: bool = False) -> np.ndarray:
    yy, xx = np.indices((height, width), dtype=np.float32)
    y = np.mod(xx * 17.0 + yy * 29.0 + 11.0, 251.0) / np.float32(255.0)
    cb = np.mod(xx * 43.0 + yy * 13.0 + 37.0, 251.0) / np.float32(255.0)
    cr = np.mod(xx * 7.0 + yy * 53.0 + 83.0, 251.0) / np.float32(255.0)
    planes = [y, cb, cr]
    if alpha:
        planes.append(np.mod(xx * 31.0 + yy * 19.0 + 101.0, 251.0) / np.float32(255.0))
    return np.stack(planes, axis=2).astype(np.float32)


def _point_weight(interpolation: str, distance: float) -> float:
    """Independent fp64 scaled-kernel oracle from the feature sheet."""
    x = abs(distance) / 2.0
    if interpolation == "bilinear":
        return max(0.0, 1.0 - x)
    if interpolation == "bicubic":
        a = -0.5
        if x < 1.0:
            return (a + 2.0) * x**3 - (a + 3.0) * x**2 + 1.0
        if x < 2.0:
            return a * x**3 - 5.0 * a * x**2 + 8.0 * a * x - 4.0 * a
        return 0.0
    raise AssertionError(f"unexpected point interpolation {interpolation!r}")


def _axis_plan(center: float, extent: int, interpolation: str) -> tuple[tuple[int, float], ...]:
    if interpolation == "nearest":
        index = min(max(math.floor(center + 0.5), 0), extent - 1)
        return ((index, 1.0),)

    base = math.floor(center)
    if interpolation == "area":
        raw = []
        left, right = center - 1.0, center + 1.0
        for index in range(base - 2, base + 4):
            overlap = max(0.0, min(index + 0.5, right) - max(index - 0.5, left))
            if overlap > 0.0:
                raw.append((min(max(index, 0), extent - 1), overlap))
    else:
        support = 2 if interpolation == "bilinear" else 4
        raw = [
            (min(max(index, 0), extent - 1), _point_weight(interpolation, index - center))
            for index in range(base - support + 1, base + support + 1)
        ]
    total = sum(weight for _index, weight in raw)
    return tuple((index, weight / total) for index, weight in raw if weight != 0.0)


def _downsample_reference(
    plane: np.ndarray,
    *,
    subsample_x: int,
    subsample_y: int,
    offset: tuple[float, float],
    interpolation: str,
) -> np.ndarray:
    """Independent separable coverage/scaled-point evaluation at the specified chroma phase."""
    output_height = (plane.shape[0] + subsample_y - 1) // subsample_y
    output_width = (plane.shape[1] + subsample_x - 1) // subsample_x
    result = np.empty((output_height, output_width), dtype=np.float64)
    for output_y in range(output_height):
        center_y = output_y * subsample_y + offset[1]
        vertical = _axis_plan(center_y, plane.shape[0], interpolation) if subsample_y > 1 else ((output_y, 1.0),)
        for output_x in range(output_width):
            center_x = output_x * subsample_x + offset[0]
            horizontal = _axis_plan(center_x, plane.shape[1], interpolation)
            result[output_y, output_x] = sum(
                float(plane[source_y, source_x]) * weight_y * weight_x
                for source_y, weight_y in vertical
                for source_x, weight_x in horizontal
            )
    return result


def _quantize_reference(values: np.ndarray, *, bit_depth: int, range: str, component: str) -> np.ndarray:
    """Independent H.273 inverse-map, half-away rounding, and container-only clip oracle."""
    maximum = float((1 << bit_depth) - 1)
    if range == "full" or component == "A":
        mapped = values.astype(np.float64) * maximum
    else:
        code_scale = float(1 << (bit_depth - 8))
        extent = (219.0 if component == "Y" else 224.0) * code_scale
        mapped = values.astype(np.float64) * extent + 16.0 * code_scale
    rounded = np.where(mapped >= 0.0, np.floor(mapped + 0.5), np.ceil(mapped - 0.5))
    return np.clip(rounded, 0.0, maximum).astype(np.uint32)


def _subsampled_codes(
    values: np.ndarray,
    *,
    bit_depth: int,
    range: str,
    subsample_y: int,
    offset: tuple[float, float],
    interpolation: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = _quantize_reference(values[..., 0], bit_depth=bit_depth, range=range, component="Y")
    cb_values = _downsample_reference(
        values[..., 1],
        subsample_x=2,
        subsample_y=subsample_y,
        offset=offset,
        interpolation=interpolation,
    )
    cr_values = _downsample_reference(
        values[..., 2],
        subsample_x=2,
        subsample_y=subsample_y,
        offset=offset,
        interpolation=interpolation,
    )
    cb = _quantize_reference(cb_values, bit_depth=bit_depth, range=range, component="Cb")
    cr = _quantize_reference(cr_values, bit_depth=bit_depth, range=range, component="Cr")
    return y, cb, cr


def _pack_v210(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    """Independent v210 word layout with replicated group tails and zero row padding."""
    height, width = y.shape
    row_words = ((width + 47) // 48) * 32
    result = np.zeros((height, row_words), dtype=np.uint32)
    for row in range(height):
        for group_start in range(0, width, 6):
            group = group_start // 6
            ys = [int(y[row, min(group_start + offset, width - 1)]) for offset in range(6)]
            chroma_start = group_start // 2
            cbs = [int(cb[row, min(chroma_start + offset, cb.shape[1] - 1)]) for offset in range(3)]
            crs = [int(cr[row, min(chroma_start + offset, cr.shape[1] - 1)]) for offset in range(3)]
            base = group * 4
            result[row, base] = cbs[0] | (ys[0] << 10) | (crs[0] << 20)
            result[row, base + 1] = ys[1] | (cbs[1] << 10) | (ys[2] << 20)
            result[row, base + 2] = crs[1] | (ys[3] << 10) | (cbs[2] << 20)
            result[row, base + 3] = ys[4] | (crs[2] << 10) | (ys[5] << 20)
    return result.reshape(-1)


def _reference_output(
    name: str,
    values: np.ndarray,
    *,
    bit_depth: int,
    range: str,
    siting: str = "topleft",
    interpolation: str = "area",
) -> np.ndarray:
    if name in {"yuv444p", "yuva444p"}:
        planes = [
            _quantize_reference(values[..., 0], bit_depth=bit_depth, range=range, component="Y"),
            _quantize_reference(values[..., 1], bit_depth=bit_depth, range=range, component="Cb"),
            _quantize_reference(values[..., 2], bit_depth=bit_depth, range=range, component="Cr"),
        ]
        if name == "yuva444p":
            planes.append(_quantize_reference(values[..., 3], bit_depth=bit_depth, range=range, component="A"))
        return np.concatenate([plane.reshape(-1) for plane in planes]).astype(np.uint16)

    subsample_y = 2 if name in {"nv12", "p010", "yuv420p"} else 1
    offset = SITING_OFFSETS[siting] if subsample_y == 2 else (0.0, 0.0)
    y, cb, cr = _subsampled_codes(
        values,
        bit_depth=bit_depth,
        range=range,
        subsample_y=subsample_y,
        offset=offset,
        interpolation=interpolation,
    )
    if name == "uyvy422":
        packed = np.empty((values.shape[0], values.shape[1] * 2), dtype=np.uint8)
        packed[:, 0::4] = cb
        packed[:, 1::4] = y[:, 0::2]
        packed[:, 2::4] = cr
        packed[:, 3::4] = y[:, 1::2]
        return packed.reshape(-1)
    if name == "v210":
        return _pack_v210(y, cb, cr)
    if name in {"nv12", "p010"}:
        uv = np.stack((cb, cr), axis=2).reshape(-1)
        result = np.concatenate((y.reshape(-1), uv))
        if name == "p010":
            return (result << 6).astype(np.uint16)
        return result.astype(np.uint8)
    dtype = np.uint8 if bit_depth == 8 else np.uint16
    return np.concatenate((y.reshape(-1), cb.reshape(-1), cr.reshape(-1))).astype(dtype)


def test_to_format_functions_have_exact_keyword_only_signatures() -> None:
    """v1-public-namespace acceptance 10: every io exit has one static format signature."""
    expected = {
        "to_uyvy422": (("frame", "range", "interpolation"), {"range": "legal", "interpolation": "area"}),
        "to_v210": (("frame", "range", "interpolation"), {"range": "legal", "interpolation": "area"}),
        "to_nv12": (
            ("frame", "range", "siting", "interpolation"),
            {"range": "legal", "siting": "left", "interpolation": "area"},
        ),
        "to_p010": (
            ("frame", "range", "siting", "interpolation"),
            {"range": "legal", "siting": "left", "interpolation": "area"},
        ),
        "to_yuv420p": (
            ("frame", "bit_depth", "range", "siting", "interpolation"),
            {"bit_depth": 8, "range": "legal", "siting": "left", "interpolation": "area"},
        ),
        "to_yuv422p": (
            ("frame", "bit_depth", "range", "interpolation"),
            {"bit_depth": 8, "range": "legal", "interpolation": "area"},
        ),
        "to_yuv444p": (("frame", "bit_depth", "range"), {"bit_depth": 10, "range": "legal"}),
        "to_yuva444p": (("frame", "bit_depth", "range"), {"bit_depth": 12, "range": "legal"}),
    }
    for name, (parameter_names, defaults) in expected.items():
        signature = inspect.signature(getattr(px.io, name))
        assert tuple(signature.parameters) == parameter_names
        for parameter_name in parameter_names[1:]:
            parameter = signature.parameters[parameter_name]
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
            assert parameter.default == defaults[parameter_name]


@pytest.mark.parametrize(
    ("name", "height", "width", "kwargs"),
    (
        ("to_uyvy422", 2, 6, {"range": "full", "interpolation": "bicubic"}),
        ("to_v210", 2, 7, {"range": "full", "interpolation": "area"}),
        ("to_nv12", 4, 6, {"range": "full", "siting": "center", "interpolation": "area"}),
        ("to_p010", 4, 6, {"range": "legal", "siting": "topleft", "interpolation": "bilinear"}),
        (
            "to_yuv420p",
            4,
            6,
            {"bit_depth": 10, "range": "full", "siting": "left", "interpolation": "bicubic"},
        ),
        ("to_yuv422p", 3, 6, {"bit_depth": 12, "range": "legal", "interpolation": "area"}),
        ("to_yuv444p", 2, 3, {"bit_depth": 12, "range": "full"}),
        ("to_yuva444p", 2, 3, {"bit_depth": 12, "range": "legal"}),
    ),
)
def test_all_to_formats_match_independent_numpy_layout_and_range_reference(
    name: str,
    height: int,
    width: int,
    kwargs: dict[str, object],
) -> None:
    """v1-format-boundary acceptance 12-15 and 27-35: every packing layout is bit-exact to an independent oracle."""
    alpha = name == "to_yuva444p"
    values = _values(height, width, alpha=alpha)
    channels = ("Y", "Cb", "Cr", "A") if alpha else ("Y", "Cb", "Cr")
    actual = getattr(px.io, name)(_frame(values, channels=channels), **kwargs).get()
    expected = _reference_output(
        name.removeprefix("to_"),
        values,
        bit_depth=int(kwargs.get("bit_depth", 10 if name in {"to_v210", "to_p010"} else 8)),
        range=str(kwargs["range"]),
        siting=str(kwargs.get("siting", "topleft")),
        interpolation=str(kwargs.get("interpolation", "area")),
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("siting", tuple(SITING_OFFSETS))
@pytest.mark.parametrize("interpolation", TO_INTERPOLATIONS)
def test_to_yuv420p_filter_and_siting_match_the_independent_scaled_kernel_oracle(
    siting: str,
    interpolation: str,
) -> None:
    """v1-format-boundary acceptance 17, 18, and 21-24, 35: every down-filter/siting pair follows the sheet."""
    values = _values(8, 10)
    actual = px.io.to_yuv420p(
        _frame(values),
        range="full",
        siting=siting,
        interpolation=interpolation,
    ).get()
    expected = _reference_output(
        "yuv420p",
        values,
        bit_depth=8,
        range="full",
        siting=siting,
        interpolation=interpolation,
    )
    difference = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    assert int(difference.max()) <= 1
    # The only allowance is an fp32/fp64 near-tie; this fixture caps it at two percent of codes.
    assert float(np.count_nonzero(difference) / difference.size) <= 0.02


def test_area_coverage_and_nearest_half_up_choose_the_sheet_phase_samples() -> None:
    """v1-format-boundary acceptance 21-24: area coverage and nearest half-up have distinct center-sited results."""
    values = np.zeros((2, 4, 3), dtype=np.float32)
    values[..., 0] = 0.0
    values[..., 1] = np.asarray((0.0, 0.25, 0.75, 1.0), dtype=np.float32)
    values[..., 2] = values[..., 1]
    frame = _frame(values)

    nearest = px.io.to_yuv420p(frame, range="full", siting="center", interpolation="nearest").get()
    area = px.io.to_yuv420p(frame, range="full", siting="center", interpolation="area").get()
    chroma_start = values.shape[0] * values.shape[1]

    np.testing.assert_array_equal(nearest[chroma_start : chroma_start + 2], np.asarray((64, 255), dtype=np.uint8))
    np.testing.assert_array_equal(area[chroma_start : chroma_start + 2], np.asarray((32, 223), dtype=np.uint8))


def test_legal_headroom_container_clip_half_away_rounding_and_alpha_full_scale() -> None:
    """v1-format-boundary acceptance 12-15 and 37: legal headroom survives, physical overflow clips, ties round away."""
    y_codes = np.asarray((-5.0, 4.0, 64.5, 940.5, 1019.0, 1030.0))
    cb_codes = np.asarray((-5.0, 4.0, 64.5, 512.5, 1019.0, 1030.0))
    values = np.stack(
        (
            (y_codes - 64.0) / 876.0,
            (cb_codes - 64.0) / 896.0,
            (cb_codes - 64.0) / 896.0,
        ),
        axis=1,
    )[None, ...].astype(np.float32)
    output = px.io.to_yuv444p(_frame(values), bit_depth=10, range="legal").get().reshape(3, -1)
    expected_y = np.asarray((0, 4, 65, 941, 1019, 1023), dtype=np.uint16)
    expected_c = np.asarray((0, 4, 65, 513, 1019, 1023), dtype=np.uint16)
    np.testing.assert_array_equal(output[0], expected_y)
    np.testing.assert_array_equal(output[1], expected_c)
    np.testing.assert_array_equal(output[2], expected_c)

    alpha_values = np.zeros((1, 4, 4), dtype=np.float32)
    alpha_values[..., :3] = 0.5
    alpha_values[..., 3] = np.asarray((0.0, 0.5, 1.0, 2.0), dtype=np.float32)
    alpha = px.io.to_yuva444p(_frame(alpha_values, channels=("Y", "Cb", "Cr", "A")), range="legal").get()
    np.testing.assert_array_equal(alpha[-4:], np.asarray((0, 2048, 4095, 4095), dtype=np.uint16))


@pytest.mark.parametrize(
    ("name", "channels"),
    (
        ("to_uyvy422", ("R", "G", "B")),
        ("to_v210", ("R", "G", "B")),
        ("to_nv12", ("R", "G", "B")),
        ("to_p010", ("R", "G", "B")),
        ("to_yuv420p", ("R", "G", "B")),
        ("to_yuv422p", ("R", "G", "B")),
        ("to_yuv444p", ("R", "G", "B")),
        ("to_yuva444p", ("Y", "Cb", "Cr")),
    ),
)
def test_to_formats_reject_wrong_channels_and_non_fp32_frames_actionably(
    name: str,
    channels: tuple[str, ...],
) -> None:
    """v1-format-boundary acceptance 6 and v1-color-semantics acceptance 34: fail with directional guidance."""
    channel_values = np.zeros((2, 2, len(channels)), dtype=np.float32)
    with pytest.raises(ValueError) as channels_error:
        getattr(px.io, name)(_frame(channel_values, channels=channels))
    _actionable(channels_error)
    assert "px.color.rgb_to_ycbcr" in str(channels_error.value)

    valid_channels = ("Y", "Cb", "Cr", "A") if name == "to_yuva444p" else ("Y", "Cb", "Cr")
    dtype_values = np.zeros((2, 2, len(valid_channels)), dtype=np.float16)
    with pytest.raises(ValueError) as dtype_error:
        getattr(px.io, name)(_frame(dtype_values, channels=valid_channels))
    _actionable(dtype_error)
    assert "float32" in str(dtype_error.value)


@pytest.mark.parametrize("name", ("to_nv12", "to_p010", "to_yuv420p"))
def test_420_to_formats_reject_unknown_tokens_and_odd_dimensions(name: str) -> None:
    """v1-format-boundary acceptance 10, 11, 21, and 37: closed tokens and even 420 dimensions fail fast."""
    valid = _frame(np.zeros((2, 2, 3), dtype=np.float32))
    for axis, value in (("range", "FULL"), ("siting", "top-left"), ("interpolation", "lanczos3")):
        with pytest.raises(ValueError) as token_error:
            getattr(px.io, name)(valid, **{axis: value})
        _actionable(token_error)

    for shape in ((2, 3, 3), (3, 2, 3)):
        with pytest.raises(ValueError) as dimension_error:
            getattr(px.io, name)(_frame(np.zeros(shape, dtype=np.float32)))
        _actionable(dimension_error)


@pytest.mark.parametrize(
    ("name", "valid_depths"),
    (
        ("to_yuv420p", (8, 10)),
        ("to_yuv422p", (8, 10, 12)),
        ("to_yuv444p", (10, 12)),
        ("to_yuva444p", (12,)),
    ),
)
def test_planar_to_formats_reject_bit_depths_outside_their_closed_domains(
    name: str,
    valid_depths: tuple[int, ...],
) -> None:
    """v1-format-boundary acceptance 3 and 11: planar to bit_depth domains enumerate recovery values."""
    channels = ("Y", "Cb", "Cr", "A") if name == "to_yuva444p" else ("Y", "Cb", "Cr")
    frame = _frame(np.zeros((2, 2, len(channels)), dtype=np.float32), channels=channels)
    with pytest.raises(ValueError) as error:
        getattr(px.io, name)(frame, bit_depth=9)
    _actionable(error)
    assert repr(valid_depths) in str(error.value)


@pytest.mark.parametrize("name", ("to_uyvy422", "to_yuv422p"))
def test_non_v210_422_to_formats_require_even_width(name: str) -> None:
    """v1-format-boundary acceptance 10 and 37: ordinary 422 exits reject odd width while v210 accepts it."""
    with pytest.raises(ValueError) as error:
        getattr(px.io, name)(_frame(np.zeros((2, 3, 3), dtype=np.float32)))
    _actionable(error)

    result = px.io.to_v210(
        _frame(_values(1, 1)),
    )
    assert result.shape == (32,)


def test_to_outputs_are_private_c_contiguous_one_dimensional_arrays_with_fixed_dtypes() -> None:
    """v1-format-boundary acceptance 5 and 27-31: every exit owns a fresh flat container of the specified dtype."""
    frame = _frame(_values(4, 4))
    expected = {
        "to_uyvy422": (np.dtype(np.uint8), 32),
        "to_v210": (np.dtype(np.uint32), 128),
        "to_nv12": (np.dtype(np.uint8), 24),
        "to_p010": (np.dtype(np.uint16), 24),
        "to_yuv420p": (np.dtype(np.uint8), 24),
        "to_yuv422p": (np.dtype(np.uint8), 32),
        "to_yuv444p": (np.dtype(np.uint16), 48),
    }
    for name, (dtype, size) in expected.items():
        first = getattr(px.io, name)(frame)
        second = getattr(px.io, name)(frame)
        assert first.ndim == 1 and first.flags.c_contiguous
        assert np.dtype(first.dtype) == dtype and first.size == size
        assert first.data.ptr != second.data.ptr

    alpha = px.io.to_yuva444p(
        _frame(_values(4, 4, alpha=True), channels=("Y", "Cb", "Cr", "A")),
    )
    assert alpha.ndim == 1 and alpha.flags.c_contiguous
    assert np.dtype(alpha.dtype) == np.dtype(np.uint16) and alpha.size == 64


def test_v210_zero_fills_every_padding_word_it_owns() -> None:
    """v1-format-boundary acceptance 29, 33, and 37: aligned v210 row padding is deterministically zero."""
    width, height = 7, 2
    output = px.io.to_v210(_frame(_values(height, width)), range="full", interpolation="area").get().reshape(height, 32)
    assert np.count_nonzero(output[:, 8:]) == 0


@pytest.mark.parametrize("interpolation", TO_INTERPOLATIONS)
def test_constant_chroma_round_trips_every_downsampling_filter(interpolation: str) -> None:
    """v1-format-boundary acceptance 36: 420 constant chroma is unchanged through every to/from filter pair."""
    import cupy as cp

    source_codes = np.concatenate(
        (
            np.arange(24, dtype=np.uint8),
            np.full(6, 64, dtype=np.uint8),
            np.full(6, 192, dtype=np.uint8),
        )
    )
    source = px.io.from_yuv420p(
        cp.asarray(source_codes),
        width=6,
        height=4,
        range="full",
        siting="center",
        interpolation="nearest",
    )
    encoded = px.io.to_yuv420p(source, range="full", siting="center", interpolation=interpolation)
    restored = px.io.from_yuv420p(
        encoded,
        width=6,
        height=4,
        range="full",
        siting="center",
        interpolation="nearest",
    )
    np.testing.assert_array_equal(encoded.get(), source_codes)
    np.testing.assert_array_equal(
        px.io.to_array(
            restored,
        ).get(),
        px.io.to_array(
            source,
        ).get(),
    )


def test_subsampled_nearest_round_trip_preserves_the_originating_buffer() -> None:
    """v1-format-boundary acceptance 36: a subsampled-origin Frame returns the same codes through nearest."""
    source = np.concatenate(
        (
            np.arange(24, dtype=np.uint8),
            np.asarray((10, 30, 50, 70, 90, 110), dtype=np.uint8),
            np.asarray((210, 190, 170, 150, 130, 110), dtype=np.uint8),
        )
    )
    frame = px.io.from_yuv420p(
        __import__("cupy").asarray(source),
        width=6,
        height=4,
        range="full",
        siting="topleft",
        interpolation="nearest",
    )
    encoded = px.io.to_yuv420p(frame, range="full", siting="topleft", interpolation="nearest")
    np.testing.assert_array_equal(encoded.get(), source)


@pytest.mark.parametrize(("name", "bit_depth"), (("yuv444p", 10), ("yuv444p", 12), ("yuva444p", 12)))
def test_planar_444_code_origin_frames_round_trip_bit_exactly(name: str, bit_depth: int) -> None:
    """v1-format-boundary acceptance 15 and 36: code-origin 444 Frames survive to/from range quantization bit exactly."""
    channel_count = 4 if name == "yuva444p" else 3
    maximum = (1 << bit_depth) - 1
    source = np.arange(1, channel_count * 6 + 1, dtype=np.uint16) * 37 & maximum
    import cupy as cp

    frame = getattr(px.io, f"from_{name}")(
        cp.asarray(source),
        width=3,
        height=2,
        bit_depth=bit_depth,
        range="legal",
    )
    encoded = getattr(px.io, f"to_{name}")(frame, bit_depth=bit_depth, range="legal")
    restored = getattr(px.io, f"from_{name}")(
        encoded,
        width=3,
        height=2,
        bit_depth=bit_depth,
        range="legal",
    )
    np.testing.assert_array_equal(encoded.get(), source)
    np.testing.assert_array_equal(
        px.io.to_array(
            restored,
        ).get(),
        px.io.to_array(
            frame,
        ).get(),
    )


def test_to_format_kernel_entries_are_declared_and_each_public_call_launches_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1-format-boundary acceptance 25: factories bind declared entries and public paths launch one fused pass."""
    import pixtreme._io.wire.uyvy422 as subsampled_module
    import pixtreme._io.wire.yuv444p as planar_module

    subsampled_kernel = subsampled_module._to_kernel(8, "nearest", "topleft")
    subsampled_source = subsampled_module._to_subsampled_kernel_source("uyvy422", 8, "nearest", "topleft")
    planar_kernel = planar_module._to_kernel()
    planar_source = planar_module._to_planar_444_kernel_source(alpha=False)

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
    monkeypatch.setattr(subsampled_module, "_to_kernel", lambda *_args: counted_subsampled)
    monkeypatch.setattr(planar_module, "_to_kernel", lambda *_args: counted_planar)

    subsampled = px.io.to_uyvy422(
        _frame(np.asarray([[[0.0, 0.5, 0.5], [1.0, 0.5, 0.5]]], dtype=np.float32)),
        interpolation="nearest",
    )
    planar = px.io.to_yuv444p(
        _frame(np.asarray([[[0.0, 0.5, 1.0]]], dtype=np.float32)),
        bit_depth=10,
    )

    assert counted_subsampled.launches == 1
    assert counted_planar.launches == 1
    np.testing.assert_array_equal(subsampled.get(), np.asarray([128, 16, 128, 235], dtype=np.uint8))
    np.testing.assert_array_equal(planar.get(), np.asarray([64, 512, 960], dtype=np.uint16))


def test_frame_exits_are_functions_not_methods_or_root_exports() -> None:
    """v1-public-namespace acceptance 1 and 10: Frame exits live only in io."""
    assert len(px.__all__) == 14
    for name in (
        "to_uyvy422",
        "to_v210",
        "to_nv12",
        "to_p010",
        "to_yuv420p",
        "to_yuv422p",
        "to_yuv444p",
        "to_yuva444p",
    ):
        assert hasattr(px.io, name)
        assert not hasattr(px.core.Frame, name)
        assert name not in px.__all__
