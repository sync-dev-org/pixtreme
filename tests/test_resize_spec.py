"""Specification and numerical-property tests for resize."""

from __future__ import annotations

import inspect
import math
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
    "area",
)


def _frame(
    values: Any,
    *,
    colorspace: str = "sRGB",
    gamma: str = "linear",
    channels: str | list[str] = "RGB",
) -> px.core.Frame:
    import cupy as cp

    return px.io.from_array(
        cp.asarray(np.asarray(values, dtype=np.float32)),
        colorspace=colorspace,
        gamma=gamma,
        channels=channels,
    )


def _assert_actionable(error: pytest.ExceptionInfo[ValueError]) -> None:
    message = str(error.value)
    assert "why=" in message
    assert "; what=" in message
    assert "; how=" in message


def _keys_weight(distance: float) -> float:
    x = abs(distance)
    a = -0.5
    if x < 1.0:
        return (a + 2.0) * x**3 - (a + 3.0) * x**2 + 1.0
    if x < 2.0:
        return a * x**3 - 5.0 * a * x**2 + 8.0 * a * x - 4.0 * a
    return 0.0


def _mitchell_weight(distance: float, *, b: float, c: float) -> float:
    x = abs(distance)
    if x < 1.0:
        return ((12.0 - 9.0 * b - 6.0 * c) * x**3 + (-18.0 + 12.0 * b + 6.0 * c) * x**2 + (6.0 - 2.0 * b)) / 6.0
    if x < 2.0:
        return (
            (-b - 6.0 * c) * x**3 + (6.0 * b + 30.0 * c) * x**2 + (-12.0 * b - 48.0 * c) * x + (8.0 * b + 24.0 * c)
        ) / 6.0
    return 0.0


def _lanczos_weight(distance: float, *, lobes: int) -> float:
    x = abs(distance)
    if x == 0.0:
        return 1.0
    if x >= lobes:
        return 0.0
    return float(np.sinc(x) * np.sinc(x / lobes))


def _axis_samples(token: str, coordinate: float) -> tuple[tuple[int, ...], np.ndarray]:
    base = math.floor(coordinate)
    if token == "bilinear":
        indices = (base, base + 1)
        weights = np.asarray((1.0 - (coordinate - base), coordinate - base), dtype=np.float64)
    elif token in {"bicubic", "b-spline", "mitchell"}:
        indices = tuple(range(base - 1, base + 3))
        if token == "bicubic":
            weights = np.asarray([_keys_weight(coordinate - index) for index in indices], dtype=np.float64)
        else:
            b, c = (1.0, 0.0) if token == "b-spline" else (1.0 / 3.0, 1.0 / 3.0)
            weights = np.asarray(
                [_mitchell_weight(coordinate - index, b=b, c=c) for index in indices], dtype=np.float64
            )
    else:
        lobes = int(token.removeprefix("lanczos"))
        indices = tuple(range(base - lobes + 1, base + lobes + 1))
        weights = np.asarray([_lanczos_weight(coordinate - index, lobes=lobes) for index in indices], dtype=np.float64)
        weights /= weights.sum()
    return indices, weights


def _area_reference(source: np.ndarray, *, width: int, height: int) -> np.ndarray:
    input_height, input_width, channels = source.shape
    output = np.empty((height, width, channels), dtype=np.float64)
    for output_y in range(height):
        top = output_y * input_height / height
        bottom = (output_y + 1) * input_height / height
        for output_x in range(width):
            left = output_x * input_width / width
            right = (output_x + 1) * input_width / width
            total = np.zeros(channels, dtype=np.float64)
            weight_sum = 0.0
            for source_y in range(math.floor(top), math.ceil(bottom)):
                weight_y = max(0.0, min(bottom, source_y + 1.0) - max(top, float(source_y)))
                clamped_y = min(max(source_y, 0), input_height - 1)
                for source_x in range(math.floor(left), math.ceil(right)):
                    weight_x = max(0.0, min(right, source_x + 1.0) - max(left, float(source_x)))
                    clamped_x = min(max(source_x, 0), input_width - 1)
                    weight = weight_x * weight_y
                    total += source[clamped_y, clamped_x].astype(np.float64) * weight
                    weight_sum += weight
            output[output_y, output_x] = total / weight_sum
    return output.astype(np.float32)


def _resize_reference(source: np.ndarray, *, width: int, height: int, interpolation: str) -> np.ndarray:
    if interpolation == "area":
        return _area_reference(source, width=width, height=height)

    input_height, input_width, channels = source.shape
    output = np.empty((height, width, channels), dtype=np.float64)
    for output_y in range(height):
        source_y = (output_y + 0.5) * input_height / height - 0.5
        for output_x in range(width):
            source_x = (output_x + 0.5) * input_width / width - 0.5
            if interpolation == "nearest":
                nearest_y = min(max(math.floor(source_y + 0.5), 0), input_height - 1)
                nearest_x = min(max(math.floor(source_x + 0.5), 0), input_width - 1)
                output[output_y, output_x] = source[nearest_y, nearest_x]
                continue

            y_indices, y_weights = _axis_samples(interpolation, source_y)
            x_indices, x_weights = _axis_samples(interpolation, source_x)
            value = np.zeros(channels, dtype=np.float64)
            for y_index, weight_y in zip(y_indices, y_weights, strict=True):
                clamped_y = min(max(y_index, 0), input_height - 1)
                for x_index, weight_x in zip(x_indices, x_weights, strict=True):
                    clamped_x = min(max(x_index, 0), input_width - 1)
                    value += source[clamped_y, clamped_x].astype(np.float64) * weight_y * weight_x
            output[output_y, output_x] = value
    return output.astype(np.float32)


def test_resize_public_signature_and_frame_only_entry_are_actionable() -> None:
    """v1-resize acceptance 1: resize is public, keyword-sized, Frame-only, and returns a Frame."""
    import cupy as cp

    signature = inspect.signature(px.transform.resize)
    assert tuple(signature.parameters) == ("frame", "width", "height", "factor", "interpolation")
    assert signature.parameters["frame"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in ("width", "height", "factor", "interpolation"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters[name].default is None
    assert "resize" in px.transform.__all__

    with pytest.raises(ValueError) as error:
        px.transform.resize(cp.zeros((2, 2, 1), dtype=cp.float32), width=1, height=1)
    _assert_actionable(error)


@pytest.mark.parametrize(
    "kwargs",
    (
        {},
        {"width": 2},
        {"height": 2},
        {"width": 2, "height": 2, "factor": 1.0},
        {"width": 2, "factor": 1.0},
        {"height": 2, "factor": 1.0},
    ),
)
def test_resize_rejects_ambiguous_or_incomplete_size_modes(kwargs: dict[str, Any]) -> None:
    """v1-resize acceptance 2: exactly width+height or factor alone is required with an actionable error."""
    source = _frame(np.zeros((2, 3, 1), dtype=np.float32), channels=["signal"])

    with pytest.raises(ValueError) as error:
        px.transform.resize(source, **kwargs)
    _assert_actionable(error)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"width": 0, "height": 2},
        {"width": 2, "height": -1},
        {"width": 2.0, "height": 2},
        {"width": True, "height": 2},
        {"factor": 0.0},
        {"factor": -0.5},
        {"factor": "2"},
        {"factor": True},
        {"factor": 0.1},
    ),
)
def test_resize_rejects_invalid_dimensions_and_factors(kwargs: dict[str, Any]) -> None:
    """v1-resize acceptance 4: dimensions are positive ints and factor is positive with nonempty output."""
    source = _frame(np.zeros((2, 3, 1), dtype=np.float32), channels=["signal"])

    with pytest.raises(ValueError) as error:
        px.transform.resize(source, **kwargs)
    _assert_actionable(error)


def test_resize_factor_uses_half_up_rounding_and_accepts_real_scalars() -> None:
    """v1-resize acceptance 3 and 4: factor uses floor(dim*factor+0.5), including the 1080 regression."""
    source = _frame(np.zeros((1080, 3, 1), dtype=np.float32), channels=["signal"])

    result = px.transform.resize(source, factor=np.float32(2.0 / 3.0), interpolation="nearest")

    assert result.shape == (720, 2, 1)


def test_resize_accepts_all_tokens_and_unknown_token_lists_the_vocabulary() -> None:
    """v1-resize acceptance 5: all nine exact tokens work and unknown vocabulary fails with the accepted set."""
    source = _frame(np.arange(12, dtype=np.float32).reshape(3, 4, 1), channels=["signal"])

    for interpolation in INTERPOLATIONS:
        assert px.transform.resize(source, width=2, height=2, interpolation=interpolation).shape == (2, 2, 1)

    with pytest.raises(ValueError) as error:
        px.transform.resize(source, width=2, height=2, interpolation="linear")
    _assert_actionable(error)
    for interpolation in INTERPOLATIONS:
        assert interpolation in str(error.value)


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    ((3, 3, "area"), (7, 6, "lanczos4"), (7, 3, "area"), (5, 4, "lanczos4")),
)
def test_resize_auto_default_is_size_driven(width: int, height: int, expected: str) -> None:
    """v1-resize acceptance 6: any shrinking axis selects area, otherwise auto selects lanczos4."""
    values = np.linspace(-0.25, 1.25, 4 * 5 * 2, dtype=np.float32).reshape(4, 5, 2)
    source = _frame(values, channels=["left", "right"])

    automatic = px.transform.resize(source, width=width, height=height)
    explicit = px.transform.resize(source, width=width, height=height, interpolation=expected)

    np.testing.assert_array_equal(
        px.io.to_array(
            automatic,
        ).get(),
        px.io.to_array(
            explicit,
        ).get(),
    )


@pytest.mark.parametrize("interpolation", INTERPOLATIONS)
@pytest.mark.parametrize(("width", "height"), ((3, 2), (7, 6)))
def test_resize_kernels_match_an_independent_centered_numpy_oracle(
    interpolation: str,
    width: int,
    height: int,
) -> None:
    """v1-resize acceptance 7, 8, 10, and 13-15: centered fixed-support kernels match NumPy."""
    rng = np.random.default_rng(20260716)
    values = rng.uniform(-0.25, 1.25, size=(4, 5, 3)).astype(np.float32)
    source = _frame(values, channels=["temperature", "mask", "depth"])
    expected = _resize_reference(values, width=width, height=height, interpolation=interpolation)

    result = px.transform.resize(source, width=width, height=height, interpolation=interpolation)

    # 3e-5 covers float32 coordinate arithmetic, up to 64 weighted products for
    # lanczos4, and GPU sinf rounding while remaining below 0.003% of unit scale.
    np.testing.assert_allclose(
        px.io.to_array(
            result,
        ).get(),
        expected,
        rtol=3e-5,
        atol=3e-5,
    )


def test_resize_known_nearest_bilinear_and_area_solutions_fix_center_and_edge_rules() -> None:
    """v1-resize acceptance 7, 8, and 13: hand-computed small images fix center mapping and replicate edges."""
    horizontal = _frame(np.asarray([[[0.0], [10.0]]], dtype=np.float32), channels=["signal"])
    bilinear = px.transform.resize(horizontal, width=4, height=1, interpolation="bilinear")
    nearest = px.transform.resize(horizontal, width=4, height=1, interpolation="nearest")
    np.testing.assert_array_equal(
        px.io.to_array(
            bilinear,
        ).get()[0, :, 0],
        np.asarray([0.0, 2.5, 7.5, 10.0]),
    )
    np.testing.assert_array_equal(
        px.io.to_array(
            nearest,
        ).get()[0, :, 0],
        np.asarray([0.0, 0.0, 10.0, 10.0]),
    )

    grid = np.arange(16, dtype=np.float32).reshape(4, 4, 1)
    area = px.transform.resize(_frame(grid, channels=["signal"]), width=2, height=2, interpolation="area")
    np.testing.assert_array_equal(
        px.io.to_array(
            area,
        ).get()[..., 0],
        np.asarray([[2.5, 4.5], [10.5, 12.5]], dtype=np.float32),
    )


def test_resize_is_per_channel_label_independent_and_preserves_metadata() -> None:
    """v1-resize acceptance 9 and 16: arbitrary channels are independent and all metadata survives."""
    values = np.asarray(
        [
            [[0.0, 100.0, -5.0], [1.0, 200.0, -4.0]],
            [[2.0, 300.0, -3.0], [3.0, 400.0, -2.0]],
        ],
        dtype=np.float32,
    )
    source = _frame(
        values,
        colorspace="ACEScg",
        gamma="logc4",
        channels=["temperature", "confidence", "depth"],
    )
    expected = _resize_reference(values, width=5, height=3, interpolation="bilinear")

    result = px.transform.resize(source, width=5, height=3, interpolation="bilinear")

    # One float32 ulp grows to about 3.1e-5 at the 400-valued channel; the
    # relative term covers that representation scale without relaxing small values.
    np.testing.assert_allclose(
        px.io.to_array(
            result,
        ).get(),
        expected,
        rtol=1e-7,
        atol=2e-6,
    )
    assert (result.width, result.height) == (5, 3)
    assert (result.colorspace, result.gamma, result.channels) == (
        "ACEScg",
        "logc4",
        ("temperature", "confidence", "depth"),
    )


def test_resize_keeps_scene_values_and_filter_overshoot_unclamped() -> None:
    """v1-resize acceptance 10: input excursions and cubic undershoot pass through without clamping."""
    excursions = _frame(np.asarray([[[-0.5], [1.5]]], dtype=np.float32), channels=["signal"])
    linear = px.io.to_array(
        px.transform.resize(excursions, width=4, height=1, interpolation="bilinear"),
    ).get()
    assert float(linear.min()) == pytest.approx(-0.5)
    assert float(linear.max()) == pytest.approx(1.5)

    impulse_values = np.asarray([[[0.0], [1.0], [0.0], [0.0]]], dtype=np.float32)
    impulse = _frame(impulse_values, channels=["signal"])
    cubic = px.io.to_array(
        px.transform.resize(impulse, width=12, height=1, interpolation="bicubic"),
    ).get()
    expected = _resize_reference(impulse_values, width=12, height=1, interpolation="bicubic")
    assert float(expected.min()) < 0.0
    np.testing.assert_allclose(cubic, expected, rtol=0.0, atol=2e-6)
    assert float(cubic.min()) < 0.0


def test_resize_calculates_float32_and_same_size_kernel_classes_are_distinct() -> None:
    """v1-resize acceptance 11 and 12: output is fp32; interpolating kernels are identity, approximating ones smooth."""
    rng = np.random.default_rng(11)
    values = rng.uniform(-0.2, 1.2, size=(5, 6, 2)).astype(np.float32)
    source = _frame(values, channels=["first", "second"])

    for interpolation in ("nearest", "bilinear", "bicubic", "lanczos2", "lanczos3", "lanczos4"):
        result = px.transform.resize(source, width=6, height=5, interpolation=interpolation)
        assert result.dtype == np.dtype(np.float32)
        np.testing.assert_allclose(
            px.io.to_array(
                result,
            ).get(),
            values,
            rtol=0.0,
            atol=2e-6,
        )

    for interpolation in ("b-spline", "mitchell"):
        result = px.transform.resize(source, width=6, height=5, interpolation=interpolation)
        assert result.dtype == np.dtype(np.float32)
        assert not np.allclose(
            px.io.to_array(
                result,
            ).get(),
            values,
            rtol=0.0,
            atol=2e-6,
        )


def test_resize_always_returns_a_new_frame_and_private_allocation() -> None:
    """v1-resize acceptance 16 and 17: even an identity-sized call returns a new Frame and allocation."""
    values = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    source = _frame(values, colorspace="Rec.2020", gamma="pq", channels="BGR")

    result = px.transform.resize(source, width=3, height=2, interpolation="nearest")

    assert isinstance(result, px.core.Frame)
    assert result is not source
    assert result.data.data.ptr != source.data.data.ptr
    assert (result.colorspace, result.gamma, result.channels) == ("Rec.2020", "pq", ("B", "G", "R"))
    np.testing.assert_array_equal(
        px.io.to_array(
            result,
        ).get(),
        values,
    )


def test_vocabulary_defines_resize_tokens_subsets_geometry_and_area(vocabulary_markdown: str) -> None:
    """v1-resize acceptance 18: interpolation vocabulary is shared and defines the complete resize contract."""
    markdown = vocabulary_markdown
    section = markdown.split("## interpolation\n", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]

    for required in (
        "from_yuv420p",
        "chroma siting",
        "resize",
        *INTERPOLATIONS,
        "floor(dim × factor + 0.5)",
        "replicate",
        "pixel center",
        "scale-aware AA",
        "source 領域",
    ):
        assert required in section


def test_resize_docstring_is_a_self_contained_llm_readable_contract() -> None:
    """v1-resize acceptance 19: the docstring states every non-obvious call and numeric rule."""
    docstring = inspect.getdoc(px.transform.resize)
    assert docstring is not None
    for required in (
        "width and height",
        "factor",
        "floor(dim * factor + 0.5)",
        "area",
        "lanczos4",
        "does not clamp",
        "src = (dst + 0.5) * (input / output) - 0.5",
    ):
        assert required in docstring
