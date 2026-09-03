"""Specification and independent numerical-oracle tests for histogram equalization."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import pixtreme as px


def _frame(
    values: Any,
    *,
    dtype: np.dtype[Any] | type[np.generic] = np.float32,
    colorspace: str = "sRGB",
    gamma: str = "linear",
    channels: tuple[str, ...] | None = None,
    matrix: str | None = None,
) -> px.core.Frame:
    import cupy as cp

    data = np.asarray(values, dtype=dtype)
    resolved_channels = channels or tuple(f"channel-{index}" for index in range(data.shape[2]))
    return px.io.from_array(
        cp.asarray(data),
        colorspace=colorspace,
        gamma=gamma,
        channels=resolved_channels,
        matrix=matrix,
    )


def _assert_actionable(error: pytest.ExceptionInfo[ValueError], *required: str) -> None:
    message = str(error.value)
    assert "why=" in message
    assert "; what=" in message
    assert "; how=" in message
    for fragment in required:
        assert fragment in message


def _bin_indices(values: np.ndarray, *, domain: tuple[float, float], bins: int) -> np.ndarray:
    """Host binning derived directly from v1-histogram acceptance 9."""
    lo, hi = domain
    clamped = np.clip(values.astype(np.float64), lo, hi)
    scaled = (clamped - lo) / (hi - lo) * bins
    return np.minimum(np.floor(scaled), bins - 1).astype(np.int64)


def _equalize_reference(values: np.ndarray, *, domain: tuple[float, float], bins: int) -> np.ndarray:
    """Host direct empirical CDF derived from v1-histogram acceptance 9-11."""
    indices = _bin_indices(values, domain=domain, bins=bins)
    height, width, channel_count = values.shape
    output = np.empty(values.shape, dtype=np.float64)
    for channel in range(channel_count):
        counts = np.bincount(indices[..., channel].reshape(-1), minlength=bins)
        cdf = np.cumsum(counts, dtype=np.float64) / (height * width)
        output[..., channel] = cdf[indices[..., channel]]
    return output.astype(np.float32)


def _mirror_index(index: int, extent: int) -> int:
    if extent <= 1:
        return 0
    period = 2 * extent - 2
    reflected = index % period
    return reflected if reflected < extent else period - reflected


def _waterfill_reference(counts: np.ndarray, *, cap: float) -> np.ndarray:
    """Solve v1-histogram acceptance 13 analytically by ordered saturation thresholds."""
    raw = counts.astype(np.float64)
    target = float(np.sum(raw))
    clipped = np.minimum(raw, cap)
    current = float(np.sum(clipped))
    if current >= target:
        return clipped

    thresholds = np.sort(cap - raw[raw < cap])
    level = 0.0
    active = len(thresholds)
    offset = 0
    while offset < len(thresholds):
        next_level = float(thresholds[offset])
        growth = (next_level - level) * active
        if current + growth >= target:
            level += (target - current) / active
            return np.minimum(raw + level, cap)
        current += growth
        level = next_level
        saturated = 0
        while offset + saturated < len(thresholds) and thresholds[offset + saturated] == next_level:
            saturated += 1
        active -= saturated
        offset += saturated

    raise AssertionError("water-fill feasibility follows from clip_limit >= 1")


def _clahe_reference(
    values: np.ndarray,
    *,
    clip_limit: float,
    tiles_y: int,
    tiles_x: int,
    domain: tuple[float, float],
    bins: int,
) -> np.ndarray:
    """Independent host pipeline derived only from v1-histogram acceptance 9 and 12-16."""
    height, width, channel_count = values.shape
    tile_height = (height + tiles_y - 1) // tiles_y
    tile_width = (width + tiles_x - 1) // tiles_x
    padded_height = tiles_y * tile_height
    padded_width = tiles_x * tile_width
    y_indices = np.asarray([_mirror_index(index, height) for index in range(padded_height)])
    x_indices = np.asarray([_mirror_index(index, width) for index in range(padded_width)])
    padded = values[y_indices[:, None], x_indices[None, :], :]
    padded_bins = _bin_indices(padded, domain=domain, bins=bins)

    tile_pixels = tile_height * tile_width
    cap = clip_limit * tile_pixels / bins
    luts = np.empty((tiles_y, tiles_x, channel_count, bins), dtype=np.float64)
    for tile_y in range(tiles_y):
        for tile_x in range(tiles_x):
            tile = padded_bins[
                tile_y * tile_height : (tile_y + 1) * tile_height,
                tile_x * tile_width : (tile_x + 1) * tile_width,
            ]
            for channel in range(channel_count):
                counts = np.bincount(tile[..., channel].reshape(-1), minlength=bins)
                filled = _waterfill_reference(counts, cap=cap)
                cdf = np.cumsum(filled, dtype=np.float64) / tile_pixels
                cdf[-1] = 1.0
                luts[tile_y, tile_x, channel] = cdf

    source_bins = _bin_indices(values, domain=domain, bins=bins)
    output = np.empty(values.shape, dtype=np.float64)
    for y in range(height):
        v = np.clip((y + 0.5) / tile_height - 0.5, 0.0, tiles_y - 1)
        y0 = int(np.floor(v))
        y1 = min(y0 + 1, tiles_y - 1)
        fraction_y = v - y0
        for x in range(width):
            u = np.clip((x + 0.5) / tile_width - 0.5, 0.0, tiles_x - 1)
            x0 = int(np.floor(u))
            x1 = min(x0 + 1, tiles_x - 1)
            fraction_x = u - x0
            for channel in range(channel_count):
                index = source_bins[y, x, channel]
                top = (1.0 - fraction_x) * luts[y0, x0, channel, index] + fraction_x * luts[y0, x1, channel, index]
                bottom = (1.0 - fraction_x) * luts[y1, x0, channel, index] + fraction_x * luts[y1, x1, channel, index]
                output[y, x, channel] = (1.0 - fraction_y) * top + fraction_y * bottom
    return output.astype(np.float32)


def test_histogram_operations_have_exact_signatures_and_single_canonical_paths() -> None:
    """v1-public-namespace acceptance 7: two exact APIs exist only once under px.color."""
    expected = {
        "equalize_histogram": (("frame", "domain", "bins"), ((0.0, 1.0), 1024)),
        "clahe": (
            ("frame", "clip_limit", "tiles_y", "tiles_x", "domain", "bins"),
            (2.0, 8, 8, (0.0, 1.0), 1024),
        ),
    }
    for name, (parameter_names, defaults) in expected.items():
        function = getattr(px.color, name)
        signature = inspect.signature(function)
        assert tuple(signature.parameters) == parameter_names
        assert signature.parameters["frame"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        keyword_names = parameter_names[1:]
        assert tuple(signature.parameters[name].default for name in keyword_names) == defaults
        assert all(signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY for name in keyword_names)
        assert px.color.__all__.count(name) == 1
        assert not hasattr(px.filter, name)
        assert not hasattr(px, name)
        assert not hasattr(px.core.Frame, name)


def test_equalize_histogram_matches_direct_per_channel_cdf_at_domain_and_bin_boundaries() -> None:
    """v1-histogram acceptance 9-11 and 18: clamp, boundary bins, empty bins, and direct CDF match NumPy."""
    values = np.asarray(
        [
            [[-0.5, 0.24], [0.0, 0.24], [0.249999, 0.24], [0.25, 0.24], [0.5, 0.24]],
            [[0.749999, 0.24], [0.75, 0.24], [0.999999, 0.24], [1.0, 0.24], [1.5, 0.24]],
        ],
        dtype=np.float32,
    )
    source = _frame(values, channels=("custom", "R"))
    expected = _equalize_reference(values, domain=(0.0, 1.0), bins=4)

    actual = px.io.to_array(
        px.color.equalize_histogram(source, domain=(0.0, 1.0), bins=4),
    ).get()

    np.testing.assert_array_equal(actual, expected)
    assert np.min(actual) >= 0.0
    assert np.max(actual) == 1.0
    np.testing.assert_array_equal(actual[..., 1], 1.0)


@pytest.mark.parametrize(
    ("shape", "tiles_y", "tiles_x", "bins", "clip_limit"),
    (((5, 7, 3), 2, 3, 8, 2.0), ((1, 7, 2), 1, 3, 16, 1.75)),
)
def test_clahe_matches_independent_mirror_waterfill_and_tile_center_oracle(
    shape: tuple[int, int, int],
    tiles_y: int,
    tiles_x: int,
    bins: int,
    clip_limit: float,
) -> None:
    """v1-histogram acceptance 9 and 12-18: the full deterministic CLAHE pipeline matches a host oracle."""
    values = np.random.default_rng(20260803).uniform(-0.4, 1.4, size=shape).astype(np.float32)
    expected = _clahe_reference(
        values,
        clip_limit=clip_limit,
        tiles_y=tiles_y,
        tiles_x=tiles_x,
        domain=(-0.25, 1.25),
        bins=bins,
    )

    actual = px.io.to_array(
        px.color.clahe(
            _frame(values),
            clip_limit=clip_limit,
            tiles_y=tiles_y,
            tiles_x=tiles_x,
            domain=(-0.25, 1.25),
            bins=bins,
        ),
    ).get()

    # LUT storage and lookup are fp32 on device; the independent oracle accumulates in host fp64.
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=3e-6)
    assert np.min(actual) >= 0.0
    assert np.max(actual) <= 1.0


def test_clahe_clip_limit_one_is_the_exact_uniform_histogram_mapping() -> None:
    """v1-histogram acceptance 13-15: clip_limit=1 gives g[i]=N/B and one-over-B CDF increments."""
    values = np.asarray(
        [[[0.0], [0.1], [0.3], [0.55], [0.8]], [[1.0], [1.2], [-0.2], [0.45], [0.7]]],
        dtype=np.float32,
    )
    indices = _bin_indices(values, domain=(0.0, 1.0), bins=4)
    expected = ((indices + 1) / 4.0).astype(np.float32)

    source = _frame(values)
    actual = px.io.to_array(
        px.color.clahe(source, clip_limit=1.0, tiles_y=2, tiles_x=2, bins=4),
    ).get()

    np.testing.assert_array_equal(actual, expected)
    assert source.shape == actual.shape


def test_clahe_waterfill_saturates_the_cap_without_scan_order_remainder() -> None:
    """v1-histogram acceptance 13-15: one tile has the unique cap-respecting fractional histogram."""
    values = np.asarray([[[0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.3], [0.6]]], dtype=np.float32)
    expected_lut = np.asarray((0.375, 0.625, 0.875, 1.0), dtype=np.float32)
    indices = _bin_indices(values, domain=(0.0, 1.0), bins=4)

    actual = px.io.to_array(
        px.color.clahe(
            _frame(values),
            clip_limit=1.5,
            tiles_y=1,
            tiles_x=1,
            domain=(0.0, 1.0),
            bins=4,
        ),
    ).get()

    np.testing.assert_array_equal(actual, expected_lut[indices])


@pytest.mark.parametrize("name", ("equalize_histogram", "clahe"))
def test_histogram_operations_preserve_metadata_shape_storage_and_input(name: str) -> None:
    """v1-histogram acceptance 3; v1-red-tokens acceptance 68: both operations retain ARRI metadata."""
    values = np.random.default_rng(4903).uniform(-0.2, 1.2, size=(8, 9, 4)).astype(np.float32)
    source = _frame(
        values,
        colorspace="ACEScg",
        gamma="ARRI-LogC4",
        channels=("A", "custom", "R", "Z"),
        matrix="native",
    )
    source_before = px.io.to_array(source, copy=True).get()
    kwargs = {"tiles_y": 2, "tiles_x": 3, "bins": 16} if name == "clahe" else {"bins": 16}

    result = getattr(px.color, name)(source, **kwargs)

    assert result.shape == source.shape
    assert result.dtype == np.dtype(np.float32)
    assert result.data.flags.c_contiguous
    assert result.data.data.ptr != source.data.data.ptr
    assert (result.colorspace, result.gamma, result.channels, result.matrix) == (
        source.colorspace,
        source.gamma,
        source.channels,
        source.matrix,
    )
    np.testing.assert_array_equal(
        px.io.to_array(
            source,
        ).get(),
        source_before,
    )


@pytest.mark.parametrize("name", ("equalize_histogram", "clahe"))
def test_histogram_operations_reject_non_frames_and_non_float32_with_conversion_guidance(name: str) -> None:
    """v1-histogram acceptance 4: Frame and fp32 checks are actionable and precede processing."""
    import cupy as cp

    function = getattr(px.color, name)
    with pytest.raises(ValueError) as non_frame_error:
        function(cp.zeros((8, 8, 1), dtype=cp.float32))
    _assert_actionable(non_frame_error, "Frame", "px.io.from_array")

    conversion_paths = {
        np.float16: ("px.values.cast_dtype",),
        np.uint8: ("px.values.recode_dtype", "px.values.dequantize"),
        np.uint16: ("px.values.recode_dtype", "px.values.dequantize"),
    }
    for dtype, paths in conversion_paths.items():
        with pytest.raises(ValueError) as dtype_error:
            function(_frame(np.zeros((8, 8, 1)), dtype=dtype))
        _assert_actionable(dtype_error, "float32", *paths)


@pytest.mark.parametrize("name", ("equalize_histogram", "clahe"))
@pytest.mark.parametrize(
    "domain",
    (
        [0.0, 1.0],
        (0.0,),
        (0.0, 1.0, 2.0),
        (False, 1.0),
        (0.0, object()),
        (0.0, np.inf),
        (np.nan, 1.0),
        (1.0, 1.0),
        (2.0, 1.0),
    ),
)
def test_histogram_operations_reject_invalid_domains(name: str, domain: object) -> None:
    """v1-histogram acceptance 5: domain rejects structural, type, endpoint-finiteness, and ordering violations."""
    kwargs = {"tiles_y": 1, "tiles_x": 1} if name == "clahe" else {}
    with pytest.raises(ValueError) as error:
        getattr(px.color, name)(_frame(np.zeros((2, 3, 1))), domain=domain, **kwargs)
    _assert_actionable(error, "domain", "(minimum, maximum)")


def test_equalize_histogram_accepts_large_opposing_domain_with_finite_float64_width() -> None:
    """v1-histogram acceptance 5 and 9: an fp64-finite opposing-sign domain preserves the declared bin formula."""
    values = np.asarray([[[-1e38], [1e38]]], dtype=np.float32)

    actual = px.io.to_array(
        px.color.equalize_histogram(_frame(values), domain=(-1e38, 1e38), bins=4),
    ).get()

    np.testing.assert_array_equal(actual, np.asarray([[[0.5], [1.0]]], dtype=np.float32))


@pytest.mark.parametrize("name", ("equalize_histogram", "clahe"))
def test_histogram_operations_reject_domain_with_nonfinite_float64_width(name: str) -> None:
    """v1-histogram acceptance 5 and 9: a domain whose fp64 width overflows is rejected before binning."""
    kwargs = {"tiles_y": 1, "tiles_x": 1} if name == "clahe" else {}

    with pytest.raises(ValueError) as error:
        getattr(px.color, name)(_frame(np.zeros((2, 3, 1))), domain=(-1e308, 1e308), **kwargs)

    _assert_actionable(error, "domain", "maximum - minimum", "finite")


@pytest.mark.parametrize("name", ("equalize_histogram", "clahe"))
@pytest.mark.parametrize("domain", ((-(10**400), 1.0), (0.0, 10**400)))
def test_histogram_operations_reject_domain_elements_that_cannot_convert_to_float64(
    name: str,
    domain: tuple[int | float, int | float],
) -> None:
    """v1-histogram acceptance 5: domain elements must convert to finite fp64 without leaking OverflowError."""
    kwargs = {"tiles_y": 1, "tiles_x": 1} if name == "clahe" else {}

    with pytest.raises(ValueError) as error:
        getattr(px.color, name)(_frame(np.zeros((2, 3, 1))), domain=domain, **kwargs)

    _assert_actionable(error, "domain", "float64", "finite")


@pytest.mark.parametrize("name", ("equalize_histogram", "clahe"))
@pytest.mark.parametrize("bins", (True, np.int64(8), 2.0, 1, 65537))
def test_histogram_operations_reject_invalid_bin_counts(name: str, bins: object) -> None:
    """v1-histogram acceptance 6: bins is a built-in int in the inclusive 2..65536 range."""
    kwargs = {"tiles_y": 1, "tiles_x": 1} if name == "clahe" else {}
    with pytest.raises(ValueError) as error:
        getattr(px.color, name)(_frame(np.zeros((2, 3, 1))), bins=bins, **kwargs)
    _assert_actionable(error, "bins", "2", "65536")


@pytest.mark.parametrize("name", ("equalize_histogram", "clahe"))
@pytest.mark.parametrize("bins", (2, 65536))
def test_histogram_operations_accept_both_bin_count_boundaries(name: str, bins: int) -> None:
    """v1-histogram acceptance 6 and 17: both bin bounds work even when bins exceeds tile pixels."""
    kwargs = {"tiles_y": 1, "tiles_x": 1} if name == "clahe" else {}
    result = getattr(px.color, name)(_frame(np.asarray([[[0.0]], [[1.0]]], dtype=np.float32)), bins=bins, **kwargs)
    assert result.shape == (2, 1, 1)


@pytest.mark.parametrize("clip_limit", (True, "2", np.nan, np.inf, -1.0, 0.999999))
def test_clahe_rejects_invalid_clip_limits(clip_limit: object) -> None:
    """v1-histogram acceptance 7: clip_limit is a finite non-bool real at least one."""
    with pytest.raises(ValueError) as error:
        px.color.clahe(_frame(np.zeros((2, 3, 1))), clip_limit=clip_limit, tiles_y=1, tiles_x=1)
    _assert_actionable(error, "clip_limit", "1.0")


def test_clahe_rejects_clip_limit_that_cannot_convert_to_float64() -> None:
    """v1-histogram acceptance 7: clip_limit conversion overflow becomes an actionable ValueError."""
    with pytest.raises(ValueError) as error:
        px.color.clahe(_frame(np.zeros((2, 3, 1))), clip_limit=10**400, tiles_y=1, tiles_x=1)

    _assert_actionable(error, "clip_limit", "float64", "finite")


@pytest.mark.parametrize(
    ("argument", "value", "dimension"),
    (
        ("tiles_y", True, "height=2"),
        ("tiles_y", np.int64(1), "height=2"),
        ("tiles_y", 1.0, "height=2"),
        ("tiles_y", 0, "height=2"),
        ("tiles_y", 3, "height=2"),
        ("tiles_x", True, "width=3"),
        ("tiles_x", np.int64(1), "width=3"),
        ("tiles_x", 1.0, "width=3"),
        ("tiles_x", 0, "width=3"),
        ("tiles_x", 4, "width=3"),
    ),
)
def test_clahe_rejects_invalid_tile_counts_with_axis_specific_guidance(
    argument: str,
    value: object,
    dimension: str,
) -> None:
    """v1-histogram acceptance 8: each tile axis is a bounded positive built-in int with axis-specific errors."""
    kwargs = {"tiles_y": 1, "tiles_x": 1, argument: value}
    with pytest.raises(ValueError) as error:
        px.color.clahe(_frame(np.zeros((2, 3, 1))), **kwargs)
    _assert_actionable(error, argument, "built-in int", dimension)


def test_clahe_rejects_legacy_tiles_tuple_keyword() -> None:
    """v1-histogram acceptance 2 and 8: the legacy tiles tuple is absent from the canonical API."""
    with pytest.raises(TypeError, match="unexpected keyword argument 'tiles'"):
        px.color.clahe(_frame(np.zeros((2, 3, 1))), tiles=(1, 1))


def test_clahe_default_tile_counts_match_explicit_eight_by_eight_bitwise() -> None:
    """v1-histogram acceptance 2 and 18: default tile counts are exactly the explicit 8-by-8 behavior."""
    values = np.random.default_rng(818).uniform(-0.2, 1.2, size=(9, 10, 2)).astype(np.float32)
    source = _frame(values)

    implicit = px.io.to_array(
        px.color.clahe(source),
    ).get()
    explicit = px.io.to_array(
        px.color.clahe(source, tiles_y=8, tiles_x=8),
    ).get()

    np.testing.assert_array_equal(implicit, explicit)


def test_clahe_is_bitwise_deterministic_for_repeated_calls() -> None:
    """v1-histogram acceptance 18: count, water-fill, CDF, and interpolation are repeatable."""
    values = np.random.default_rng(1818).uniform(-0.5, 1.5, size=(11, 13, 3)).astype(np.float32)
    source = _frame(values)
    kwargs = {"clip_limit": 1.7, "tiles_y": 4, "tiles_x": 5, "domain": (-0.25, 1.25), "bins": 31}

    first = px.io.to_array(
        px.color.clahe(source, **kwargs),
    ).get()
    second = px.io.to_array(
        px.color.clahe(source, **kwargs),
    ).get()

    np.testing.assert_array_equal(first, second)


def test_histogram_docstrings_are_self_contained_and_vocabulary_adds_no_token_axes() -> None:
    """v1-histogram acceptance 19-20: docs expose every numeric contract without adding named tokens."""
    equalize_docstring = inspect.getdoc(px.color.equalize_histogram) or ""
    for required in (
        "equalize_histogram(frame, *, domain=(0.0, 1.0)",
        "bins=1024) -> Frame",
        "domain",
        "clamp",
        "floor",
        "direct empirical CDF",
        "per channel",
        "float32",
        "metadata",
        "input remains unchanged",
        "px.values.cast_dtype",
        "px.values.recode_dtype",
        "px.values.dequantize",
    ):
        assert required in equalize_docstring

    clahe_docstring = inspect.getdoc(px.color.clahe) or ""
    for required in (
        "clahe(frame, *, clip_limit=2.0, tiles_y=8, tiles_x=8",
        "domain=(0.0, 1.0), bins=1024) -> Frame",
        "domain",
        "clamp",
        "floor",
        "cap = clip_limit * tile pixels / bins",
        "water-fill",
        "bottom/right",
        "mirror",
        "tile-center",
        "bilinear",
        "bins exceeds",
        "per channel",
        "float32",
        "metadata",
        "input remains unchanged",
        "px.values.cast_dtype",
        "px.values.recode_dtype",
        "px.values.dequantize",
    ):
        assert required in clahe_docstring

    vocabulary = (Path(__file__).resolve().parents[1] / "docs_site" / "tokens.md").read_text(encoding="utf-8")
    assert all(f"\n## {name}\n" not in vocabulary for name in ("domain", "bins", "clip_limit", "tiles_y", "tiles_x"))
