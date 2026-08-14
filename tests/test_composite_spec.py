"""Specification, contract, and property tests for ``merge``."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import pixtreme as px

BLENDS = (
    "normal",
    "lighten",
    "add",
    "screen",
    "darken",
    "multiply",
    "difference",
    "overlay",
    "hardlight",
    "softlight",
)
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


def _frame(
    values: np.ndarray,
    *,
    channels: tuple[str, ...] = ("R", "G", "B"),
    colorspace: str = "ACEScg",
    gamma: str = "linear",
) -> px.core.Frame:
    import cupy as cp

    return px.io.from_array(
        cp.asarray(values),
        colorspace=colorspace,
        gamma=gamma,
        channels=channels,
    )


def _host(frame: px.core.Frame) -> np.ndarray:
    return px.io.to_array(
        frame,
    ).get()


def _assert_actionable(error: pytest.ExceptionInfo[ValueError]) -> None:
    message = str(error.value)
    assert "why=" in message
    assert "; what=" in message
    assert "; how=" in message


def _weight(interpolation: str, distance: float) -> float:
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
    return float(np.sinc(x) * np.sinc(x / lobes))


def _axis_plan(coordinate: float, extent: int, interpolation: str) -> tuple[tuple[int | None, np.float32], ...]:
    source_coordinate = coordinate - 0.5
    if interpolation == "nearest":
        index = math.floor(source_coordinate + 0.5)
        return ((index if 0 <= index < extent else None, np.float32(1.0)),)

    base = math.floor(source_coordinate)
    if interpolation == "bilinear":
        start, sample_count = base, 2
    elif interpolation.startswith("lanczos"):
        lobes = int(interpolation.removeprefix("lanczos"))
        start, sample_count = base - (lobes - 1), 2 * lobes
    else:
        start, sample_count = base - 1, 4
    raw = [_weight(interpolation, source_coordinate - index) for index in range(start, start + sample_count)]
    weight_sum = sum(raw)
    return tuple(
        (
            index if 0 <= index < extent else None,
            np.float32(weight / weight_sum if weight_sum != 0.0 else 0.0),
        )
        for index, weight in zip(range(start, start + sample_count), raw, strict=True)
    )


def _blend_reference(background: np.float32, source: np.float32, blend: str) -> np.float32:
    cb = np.float32(background)
    cs = np.float32(source)
    one = np.float32(1.0)
    two = np.float32(2.0)
    if blend == "normal":
        return cs
    if blend == "lighten":
        return np.maximum(cb, cs)
    if blend == "add":
        return np.float32(cb + cs)
    if blend == "screen":
        return np.float32(one - (one - cb) * (one - cs))
    if blend == "darken":
        return np.minimum(cb, cs)
    if blend == "multiply":
        return np.float32(cb * cs)
    if blend == "difference":
        return np.abs(np.float32(cb - cs))
    if blend == "overlay":
        return np.float32(two * cb * cs) if cb <= np.float32(0.5) else np.float32(one - two * (one - cb) * (one - cs))
    if blend == "hardlight":
        return np.float32(two * cb * cs) if cs <= np.float32(0.5) else np.float32(one - two * (one - cb) * (one - cs))
    if blend == "softlight":
        if cs <= np.float32(0.5):
            return np.float32(cb - (one - two * cs) * cb * (one - cb))
        d = (
            np.float32(((np.float32(16.0) * cb - np.float32(12.0)) * cb + np.float32(4.0)) * cb)
            if cb <= np.float32(0.25)
            else np.sqrt(cb, dtype=np.float32)
        )
        return np.float32(cb + (two * cs - one) * (d - cb))
    raise AssertionError(blend)


def _sample(
    foreground: np.ndarray,
    *,
    channel_index: int | None,
    alpha_index: int | None,
    x_plan: tuple[tuple[int | None, np.float32], ...],
    y_plan: tuple[tuple[int | None, np.float32], ...],
    associate: bool,
) -> np.float32:
    value = np.float32(0.0)
    for y_index, weight_y in y_plan:
        for x_index, weight_x in x_plan:
            if y_index is None or x_index is None:
                continue
            sample = (
                np.float32(1.0) if channel_index is None else np.float32(foreground[y_index, x_index, channel_index])
            )
            if associate and channel_index is not None and alpha_index is not None:
                sample = np.float32(sample * np.float32(foreground[y_index, x_index, alpha_index]))
            value = np.float32(value + np.float32(sample * weight_x * weight_y))
    return value


def _composite_reference(
    background: np.ndarray,
    foreground: np.ndarray,
    *,
    background_channels: tuple[str, ...],
    foreground_channels: tuple[str, ...],
    blend: str = "normal",
    opacity: float = 1.0,
    mask: np.ndarray | None = None,
    alpha: str = "premultiplied",
    position: tuple[float, float] | None = None,
    scale: float | tuple[float, float] = 1.0,
    rotation: float = 0.0,
    interpolation: str = "bilinear",
) -> np.ndarray:
    output = background.copy()
    background_alpha_index = background_channels.index("A") if "A" in background_channels else None
    foreground_alpha_index = foreground_channels.index("A") if "A" in foreground_channels else None
    background_color_indices = tuple(index for index, label in enumerate(background_channels) if label != "A")
    foreground_color_indices = {
        background_index: foreground_channels.index(background_channels[background_index])
        for background_index in background_color_indices
    }
    scale_x, scale_y = (float(scale), float(scale)) if isinstance(scale, (int, float)) else scale
    position_x, position_y = (background.shape[1] / 2.0, background.shape[0] / 2.0) if position is None else position
    anchor_x = foreground.shape[1] / 2.0
    anchor_y = foreground.shape[0] / 2.0
    radians = math.radians(rotation)
    cosine = math.cos(radians)
    sine = math.sin(radians)

    for y in range(background.shape[0]):
        for x in range(background.shape[1]):
            dx = x + 0.5 - position_x
            dy = y + 0.5 - position_y
            source_x = anchor_x + (cosine * dx - sine * dy) / scale_x
            source_y = anchor_y + (sine * dx + cosine * dy) / scale_y
            x_plan = _axis_plan(source_x, foreground.shape[1], interpolation)
            y_plan = _axis_plan(source_y, foreground.shape[0], interpolation)
            source_alpha = _sample(
                foreground,
                channel_index=foreground_alpha_index,
                alpha_index=foreground_alpha_index,
                x_plan=x_plan,
                y_plan=y_plan,
                associate=False,
            )
            alpha_b = (
                np.float32(1.0)
                if background_alpha_index is None
                else np.float32(background[y, x, background_alpha_index])
            )
            mask_value = np.float32(1.0 if mask is None else mask[y, x, 0])
            effective_alpha = np.float32(source_alpha * mask_value * np.float32(opacity))
            alpha_out = np.float32(effective_alpha + alpha_b * (np.float32(1.0) - effective_alpha))

            for background_index in background_color_indices:
                source_premultiplied = _sample(
                    foreground,
                    channel_index=foreground_color_indices[background_index],
                    alpha_index=foreground_alpha_index,
                    x_plan=x_plan,
                    y_plan=y_plan,
                    associate=alpha == "straight",
                )
                source_color = (
                    np.float32(source_premultiplied / source_alpha)
                    if source_alpha != np.float32(0.0)
                    else np.float32(0.0)
                )
                stored_background = np.float32(background[y, x, background_index])
                background_color = (
                    np.float32(stored_background / alpha_b)
                    if alpha == "premultiplied" and background_alpha_index is not None and alpha_b != np.float32(0.0)
                    else (
                        np.float32(0.0)
                        if alpha == "premultiplied"
                        and background_alpha_index is not None
                        and alpha_b == np.float32(0.0)
                        else stored_background
                    )
                )
                blend_value = _blend_reference(background_color, source_color, blend)
                composite_source = np.float32((np.float32(1.0) - alpha_b) * source_color + alpha_b * blend_value)
                premultiplied_out = np.float32(
                    effective_alpha * composite_source
                    + alpha_b * (np.float32(1.0) - effective_alpha) * background_color
                )
                output[y, x, background_index] = (
                    np.float32(premultiplied_out / alpha_out)
                    if alpha == "straight" and background_alpha_index is not None and alpha_out != np.float32(0.0)
                    else (
                        np.float32(0.0)
                        if alpha == "straight" and background_alpha_index is not None and alpha_out == np.float32(0.0)
                        else premultiplied_out
                    )
                )
            if background_alpha_index is not None:
                output[y, x, background_alpha_index] = alpha_out
    return output


def test_composite_public_signature_metadata_and_private_storage() -> None:
    """v1-derivative-filters acceptance 17: merge stays in the expanded 68-point surface."""
    signature = inspect.signature(px.composite.merge)
    assert tuple(signature.parameters) == (
        "background",
        "foreground",
        "blend",
        "opacity",
        "mask",
        "alpha",
        "position",
        "scale",
        "rotation",
        "interpolation",
        "adapt",
    )
    assert signature.parameters["background"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert signature.parameters["foreground"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in tuple(signature.parameters)[2:]:
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert {name: signature.parameters[name].default for name in tuple(signature.parameters)[2:]} == {
        "blend": "normal",
        "opacity": 1.0,
        "mask": None,
        "alpha": "premultiplied",
        "position": None,
        "scale": 1.0,
        "rotation": 0.0,
        "interpolation": "bilinear",
        "adapt": False,
    }
    assert "merge" in px.composite.__all__
    assert len(px.composite.__all__) == 1

    background = _frame(
        np.zeros((3, 5, 4), dtype=np.float32),
        channels=("B", "A", "R", "G"),
        colorspace="Rec.2020",
        gamma="pq",
    )
    foreground = _frame(
        np.zeros((2, 4, 4), dtype=np.float32),
        channels=("R", "G", "B", "A"),
        colorspace="Rec.2020",
        gamma="pq",
    )
    result = px.composite.merge(background, foreground)

    assert isinstance(result, px.core.Frame)
    assert result.shape == background.shape
    assert result.channels == background.channels
    assert result.colorspace == background.colorspace
    assert result.gamma == background.gamma
    assert result.data.data.ptr != background.data.data.ptr
    assert result.data.data.ptr != foreground.data.data.ptr


@pytest.mark.parametrize(
    ("background_factory", "foreground_factory"),
    (
        (lambda: object(), lambda: _frame(np.zeros((1, 1, 1), dtype=np.float32), channels=("matte",))),
        (lambda: _frame(np.zeros((1, 1, 1), dtype=np.float32), channels=("matte",)), lambda: object()),
    ),
)
def test_composite_rejects_non_frame_inputs_actionably(
    background_factory: Callable[[], object],
    foreground_factory: Callable[[], object],
) -> None:
    """v1-composite acceptance 1: both positional image inputs are Frame-only boundaries.

    Inputs are built lazily inside the test so that GPU-less collection never initializes CUDA (I-60).
    """
    background = background_factory()
    foreground = foreground_factory()
    with pytest.raises(ValueError) as error:
        px.composite.merge(background, foreground)
    _assert_actionable(error)
    assert "Frame" in str(error.value)


@pytest.mark.parametrize("dtype", (np.float16, np.uint8, np.uint16))
def test_composite_rejects_non_float32_images_for_both_adapt_modes(dtype: Any) -> None:
    """v1-composite acceptance 3 and 6: dtype is never an implicit or adapt-enabled conversion."""
    background = _frame(np.zeros((2, 2, 1), dtype=np.float32), channels=("matte",))
    foreground = _frame(np.zeros((2, 2, 1), dtype=dtype), channels=("matte",))

    for adapt in (False, True):
        with pytest.raises(ValueError) as error:
            px.composite.merge(background, foreground, adapt=adapt)
        _assert_actionable(error)
        assert str(np.dtype(dtype)) in str(error.value)
        assert "float32" in str(error.value)


def test_composite_label_mapping_is_order_independent_and_default_mismatches_name_both_values() -> None:
    """v1-composite acceptance 4-5: labels map colors while metadata mismatches fail with both values."""
    background = _frame(
        np.asarray([[[0.1, 0.2, 0.3]]], dtype=np.float32),
        channels=("B", "G", "R"),
        colorspace="ACEScg",
        gamma="linear",
    )
    foreground = _frame(
        np.asarray([[[1.0, 2.0, 3.0]]], dtype=np.float32),
        channels=("R", "G", "B"),
        colorspace="ACEScg",
        gamma="linear",
    )

    result = px.composite.merge(background, foreground, interpolation="nearest")
    np.testing.assert_array_equal(_host(result), np.asarray([[[3.0, 2.0, 1.0]]], dtype=np.float32))

    mismatches = (
        (
            _frame(np.zeros((1, 1, 3), dtype=np.float32), channels=("R", "G", "Z")),
            ("channels", "B", "Z"),
        ),
        (
            _frame(
                np.zeros((1, 1, 3), dtype=np.float32),
                channels=("R", "G", "B"),
                colorspace="sRGB",
                gamma="linear",
            ),
            ("colorspace", "ACEScg", "sRGB"),
        ),
        (
            _frame(
                np.zeros((1, 1, 3), dtype=np.float32),
                channels=("R", "G", "B"),
                colorspace="ACEScg",
                gamma="srgb",
            ),
            ("gamma", "linear", "srgb"),
        ),
    )
    for mismatched, required in mismatches:
        with pytest.raises(ValueError) as error:
            px.composite.merge(background, mismatched)
        _assert_actionable(error)
        assert all(value in str(error.value) for value in required)


def test_composite_adapt_matches_public_channel_and_color_transform_composition() -> None:
    """v1-composite acceptance 6: adapt follows YCbCr-to-RGB then color-transform public semantics."""
    background_values = np.asarray([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]], dtype=np.float32)
    foreground_values = np.asarray([[[0.4, 0.3, 0.7, 0.25], [0.8, 0.1, 0.2, 0.75]]], dtype=np.float32)
    background = _frame(background_values, colorspace="ACEScg", gamma="linear")
    foreground = _frame(
        foreground_values,
        channels=("Y", "Cb", "Cr", "A"),
        colorspace="Rec.709",
        gamma="rec709",
    )
    rgb = px.color.ycbcr_to_rgb(foreground)
    prepared = px.color.rgb_to_rgb(rgb, output_colorspace="ACEScg", output_gamma="linear")
    expected = _composite_reference(
        background_values,
        _host(prepared),
        background_channels=background.channels,
        foreground_channels=prepared.channels,
        alpha="straight",
        interpolation="nearest",
    )

    result = px.composite.merge(background, foreground, alpha="straight", interpolation="nearest", adapt=True)

    np.testing.assert_allclose(_host(result), expected, rtol=4e-5, atol=4e-5)


def test_composite_adapt_missing_ycbcr_matrix_cause_is_actionable() -> None:
    """REQ-API-012 / v1-composite acceptance 6: missing YCbCr provenance is actionable at its raise site."""
    values = np.zeros((1, 1, 3), dtype=np.float32)
    background = _frame(values, channels=("Y", "Cb", "Cr"), colorspace="ACEScg")
    foreground = _frame(values, channels=("R", "G", "B"), colorspace="ACEScg")

    with pytest.raises(ValueError) as error:
        px.composite.merge(background, foreground, adapt=True)
    cause = error.value.__cause__
    assert cause is not None
    message = str(cause)
    assert "why=" in message
    assert "; what=" in message
    assert "; how=" in message


def test_composite_adapt_unassociates_and_reassociates_premultiplied_color() -> None:
    """v1-composite acceptance 6 and 14: premultiplied adapt transforms straight color, then restores association."""
    background = _frame(np.zeros((1, 2, 3), dtype=np.float32), colorspace="ACEScg", gamma="linear")
    straight_values = np.asarray([[[0.4, 0.3, 0.7, 0.25], [0.8, 0.1, 0.2, 0.75]]], dtype=np.float32)
    premultiplied_values = straight_values.copy()
    premultiplied_values[..., :3] *= premultiplied_values[..., 3:4]
    straight = _frame(
        straight_values,
        channels=("Y", "Cb", "Cr", "A"),
        colorspace="Rec.709",
        gamma="rec709",
    )
    premultiplied = _frame(
        premultiplied_values,
        channels=("Y", "Cb", "Cr", "A"),
        colorspace="Rec.709",
        gamma="rec709",
    )

    straight_result = px.composite.merge(
        background,
        straight,
        alpha="straight",
        interpolation="nearest",
        adapt=True,
    )
    premultiplied_result = px.composite.merge(
        background,
        premultiplied,
        alpha="premultiplied",
        interpolation="nearest",
        adapt=True,
    )

    np.testing.assert_allclose(_host(premultiplied_result), _host(straight_result), rtol=4e-5, atol=4e-5)


@pytest.mark.parametrize(
    ("foreground_channels", "background_channels"),
    (
        (("Y",), ("R", "G", "B")),
        (("R", "G", "B", "Z"), ("Y", "Cb", "Cr")),
        (("foo", "bar", "baz"), ("R", "G", "B")),
    ),
)
def test_composite_adapt_rejects_channel_pairs_without_a_deterministic_conversion(
    foreground_channels: tuple[str, ...],
    background_channels: tuple[str, ...],
) -> None:
    """v1-composite acceptance 6: adapt does not invent channels or promote arbitrary mattes."""
    background = _frame(
        np.zeros((1, 1, len(background_channels)), dtype=np.float32),
        channels=background_channels,
    )
    foreground = _frame(
        np.zeros((1, 1, len(foreground_channels)), dtype=np.float32),
        channels=foreground_channels,
    )

    with pytest.raises(ValueError) as error:
        px.composite.merge(background, foreground, adapt=True)
    _assert_actionable(error)
    assert "channels" in str(error.value)


@pytest.mark.parametrize(
    ("kwargs", "axis"),
    (
        ({"position": (1.0,)}, "position"),
        ({"position": (1.0, float("nan"))}, "position"),
        ({"scale": 0.0}, "scale"),
        ({"scale": (1.0, -1.0)}, "scale"),
        ({"scale": float("inf")}, "scale"),
        ({"rotation": float("nan")}, "rotation"),
        ({"interpolation": "area"}, "interpolation"),
        ({"interpolation": "Bilinear"}, "interpolation"),
        ({"alpha": "Premultiplied"}, "alpha"),
        ({"blend": "over"}, "blend"),
        ({"blend": "Normal"}, "blend"),
        ({"opacity": True}, "opacity"),
        ({"opacity": -0.1}, "opacity"),
        ({"opacity": 1.1}, "opacity"),
        ({"adapt": 1}, "adapt"),
    ),
)
def test_composite_control_axes_fail_fast_actionably(kwargs: dict[str, object], axis: str) -> None:
    """v1-composite acceptance 7-9, 12-13, and 15: every control axis is finite, closed, and case-sensitive."""
    frame = _frame(np.zeros((2, 2, 1), dtype=np.float32), channels=("matte",))

    with pytest.raises(ValueError) as error:
        px.composite.merge(frame, frame, **kwargs)
    _assert_actionable(error)
    assert axis in str(error.value)


def test_composite_non_frame_guidance_names_the_public_merge_path_for_both_inputs() -> None:
    """REQ-API-012: both merge Frame validators guide callers to the public composite path."""
    frame = _frame(np.zeros((1, 1, 1), dtype=np.float32), channels=("matte",))

    for background, foreground in ((None, None), (frame, None)):
        with pytest.raises(ValueError) as error:
            px.composite.merge(background, foreground)  # type: ignore[arg-type]
        _assert_actionable(error)
        assert "px.composite.merge" in str(error.value)


def test_composite_opacity_float_overflow_is_translated_actionably() -> None:
    """REQ-API-012: finite-Real conversion overflow becomes the opacity boundary's actionable ValueError."""
    frame = _frame(np.zeros((1, 1, 1), dtype=np.float32), channels=("matte",))

    with pytest.raises(ValueError) as error:
        px.composite.merge(frame, frame, opacity=10**1000)

    _assert_actionable(error)
    assert "opacity" in str(error.value)
    assert isinstance(error.value.__cause__, OverflowError)


def test_composite_nearest_inverse_mapping_uses_center_anchor_scale_rotation_and_position() -> None:
    """v1-composite acceptance 7-8: nearest placement follows the specified inverse transform at pixel centers."""
    background_values = np.zeros((5, 6, 1), dtype=np.float32)
    foreground_values = np.arange(1, 13, dtype=np.float32).reshape(3, 4, 1)
    background = _frame(background_values, channels=("matte",))
    foreground = _frame(foreground_values, channels=("matte",))
    kwargs = {
        "position": (2.25, 3.0),
        "scale": (1.5, 0.75),
        "rotation": 90.0,
        "interpolation": "nearest",
    }
    expected = _composite_reference(
        background_values,
        foreground_values,
        background_channels=background.channels,
        foreground_channels=foreground.channels,
        **kwargs,
    )

    result = px.composite.merge(background, foreground, **kwargs)

    np.testing.assert_array_equal(_host(result), expected)


@pytest.mark.parametrize("interpolation", INTERPOLATIONS)
def test_composite_interpolation_matches_independent_transparent_edge_oracle(interpolation: str) -> None:
    """v1-composite acceptance 9-10: all eight kernels use transparent zero edges without cutoff renormalization."""
    background_values = np.linspace(-0.2, 0.3, 5 * 6 * 2, dtype=np.float32).reshape(5, 6, 2)
    foreground_values = np.asarray(
        [
            [[0.2, 0.3], [0.6, 0.7], [1.0, 0.4]],
            [[-0.3, 0.8], [1.4, 0.2], [0.5, 1.1]],
            [[0.9, 0.6], [0.1, 0.5], [1.8, 0.9]],
        ],
        dtype=np.float32,
    )
    background = _frame(background_values, channels=("matte", "A"))
    foreground = _frame(foreground_values, channels=("matte", "A"))
    kwargs = {
        "position": (1.1, 2.35),
        "scale": (1.25, 0.8),
        "rotation": 17.0,
        "interpolation": interpolation,
        "alpha": "straight",
    }
    expected = _composite_reference(
        background_values,
        foreground_values,
        background_channels=background.channels,
        foreground_channels=foreground.channels,
        **kwargs,
    )

    result = px.composite.merge(background, foreground, **kwargs)

    np.testing.assert_allclose(_host(result), expected, rtol=8e-5, atol=8e-5)


def test_composite_mask_is_untransformed_unclamped_and_opacity_zero_is_bit_exact() -> None:
    """v1-composite acceptance 11-12: mask multiplies source alpha in background coordinates without clamp."""
    background_values = np.asarray(
        [[[0.2, 0.4], [0.6, 0.8]], [[1.0, 0.5], [-0.2, 0.25]]],
        dtype=np.float32,
    )
    foreground_values = np.asarray(
        [[[1.5, 0.5], [-0.5, 0.5]], [[0.75, 0.5], [2.0, 0.5]]],
        dtype=np.float32,
    )
    mask_values = np.asarray([[[-0.5], [0.25]], [[1.5], [2.0]]], dtype=np.float32)
    background = _frame(background_values, channels=("matte", "A"))
    foreground = _frame(foreground_values, channels=("matte", "A"))
    mask = _frame(mask_values, channels=("custom-mask",), colorspace="sRGB", gamma="pq")
    expected = _composite_reference(
        background_values,
        foreground_values,
        background_channels=background.channels,
        foreground_channels=foreground.channels,
        mask=mask_values,
        opacity=0.75,
        alpha="straight",
        interpolation="nearest",
    )

    result = px.composite.merge(
        background,
        foreground,
        mask=mask,
        opacity=0.75,
        alpha="straight",
        interpolation="nearest",
    )
    identity = px.composite.merge(background, foreground, opacity=0.0)

    np.testing.assert_allclose(_host(result), expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(_host(identity), background_values)
    assert identity.data.data.ptr != background.data.data.ptr


@pytest.mark.parametrize(
    "mask_factory",
    (
        lambda: object(),
        lambda: _frame(np.zeros((1, 2, 1), dtype=np.float32), channels=("mask",)),
        lambda: _frame(np.zeros((2, 2, 2), dtype=np.float32), channels=("left", "right")),
        lambda: _frame(np.zeros((2, 2, 1), dtype=np.float16), channels=("mask",)),
    ),
)
def test_composite_mask_structure_and_dtype_fail_fast(mask_factory: Callable[[], object]) -> None:
    """v1-composite acceptance 3 and 11: mask is a same-geometry, one-channel float32 Frame.

    Masks are built lazily inside the test so that GPU-less collection never initializes CUDA (I-60).
    """
    mask = mask_factory()
    background = _frame(np.zeros((2, 2, 1), dtype=np.float32), channels=("matte",))
    foreground = _frame(np.zeros((2, 2, 1), dtype=np.float32), channels=("matte",))

    with pytest.raises(ValueError) as error:
        px.composite.merge(background, foreground, mask=mask)
    _assert_actionable(error)
    assert "mask" in str(error.value)


def test_composite_straight_and_premultiplied_inputs_are_equivalent_after_transform() -> None:
    """v1-composite acceptance 13-14: equivalent alpha encodings converge after associated interpolation."""
    background_straight = np.asarray(
        [
            [[0.2, 0.4, 0.6, 0.25], [0.8, 0.3, 0.1, 0.75], [0.1, 0.9, 0.5, 0.5]],
            [[0.7, 0.2, 0.4, 0.6], [0.3, 0.5, 0.9, 0.2], [0.6, 0.4, 0.2, 0.8]],
        ],
        dtype=np.float32,
    )
    foreground_straight = np.asarray(
        [
            [[1.0, 0.2, 0.4, 0.2], [0.4, 1.2, 0.1, 0.8]],
            [[0.3, 0.5, 1.4, 0.6], [1.5, -0.2, 0.7, 0.4]],
        ],
        dtype=np.float32,
    )
    background_premultiplied = background_straight.copy()
    background_premultiplied[..., :3] *= background_premultiplied[..., 3:4]
    foreground_premultiplied = foreground_straight.copy()
    foreground_premultiplied[..., :3] *= foreground_premultiplied[..., 3:4]
    straight_result = px.composite.merge(
        _frame(background_straight, channels=("R", "G", "B", "A")),
        _frame(foreground_straight, channels=("R", "G", "B", "A")),
        alpha="straight",
        position=(1.2, 0.85),
        scale=(1.1, 0.9),
        rotation=-13.0,
        interpolation="bilinear",
    )
    premultiplied_result = px.composite.merge(
        _frame(background_premultiplied, channels=("R", "G", "B", "A")),
        _frame(foreground_premultiplied, channels=("R", "G", "B", "A")),
        alpha="premultiplied",
        position=(1.2, 0.85),
        scale=(1.1, 0.9),
        rotation=-13.0,
        interpolation="bilinear",
    )
    premultiplied_as_straight = _host(premultiplied_result)
    alpha_out = premultiplied_as_straight[..., 3:4]
    premultiplied_as_straight[..., :3] = np.divide(
        premultiplied_as_straight[..., :3],
        alpha_out,
        out=np.zeros_like(premultiplied_as_straight[..., :3]),
        where=alpha_out != 0.0,
    )

    np.testing.assert_allclose(premultiplied_as_straight, _host(straight_result), rtol=5e-5, atol=5e-5)


def test_composite_without_foreground_alpha_samples_implicit_coverage() -> None:
    """v1-composite acceptance 10 and 14: an A-less foreground has interpolated one-inside, zero-outside coverage."""
    background_values = np.zeros((3, 4, 1), dtype=np.float32)
    foreground_values = np.ones((2, 2, 1), dtype=np.float32)
    background = _frame(background_values, channels=("matte",))
    foreground = _frame(foreground_values, channels=("matte",))
    kwargs = {"position": (0.5, 1.25), "interpolation": "bilinear"}
    expected = _composite_reference(
        background_values,
        foreground_values,
        background_channels=background.channels,
        foreground_channels=foreground.channels,
        **kwargs,
    )

    result = px.composite.merge(background, foreground, **kwargs)

    np.testing.assert_allclose(_host(result), expected, rtol=1e-6, atol=1e-6)
    assert np.any((_host(result) > 0.0) & (_host(result) < 1.0))


@pytest.mark.parametrize("blend", BLENDS)
def test_composite_blends_and_source_over_match_independent_fp32_equations(blend: str) -> None:
    """v1-composite acceptance 15-17: all ten W3C-derived blends feed the specified unclamped source-over equations."""
    background_values = np.asarray([[[-0.3, 1.4, 0.65]]], dtype=np.float32)
    foreground_values = np.asarray([[[1.7, -0.2, 0.4]]], dtype=np.float32)
    mask_values = np.asarray([[[1.25]]], dtype=np.float32)
    background = _frame(background_values, channels=("low", "high", "A"))
    foreground = _frame(foreground_values, channels=("low", "high", "A"))
    mask = _frame(mask_values, channels=("mask",))
    expected = _composite_reference(
        background_values,
        foreground_values,
        background_channels=background.channels,
        foreground_channels=foreground.channels,
        blend=blend,
        opacity=0.75,
        mask=mask_values,
        alpha="straight",
        interpolation="nearest",
    )

    result = px.composite.merge(
        background,
        foreground,
        blend=blend,
        opacity=0.75,
        mask=mask,
        alpha="straight",
        interpolation="nearest",
    )

    np.testing.assert_allclose(_host(result), expected, rtol=3e-6, atol=3e-6)


def test_composite_premultiplied_alpha_zero_defines_unassociated_color_as_zero() -> None:
    """v1-composite acceptance 13 and 17: zero alpha never exposes arbitrary premultiplied stored color."""
    background = _frame(
        np.asarray([[[9.0, -4.0, 0.0]]], dtype=np.float32),
        channels=("R", "G", "A"),
    )
    foreground = _frame(
        np.asarray([[[-7.0, 12.0, 1.0]]], dtype=np.float32),
        channels=("R", "G", "A"),
    )

    result = px.composite.merge(background, foreground, interpolation="nearest")

    np.testing.assert_array_equal(_host(result), np.asarray([[[-7.0, 12.0, 1.0]]], dtype=np.float32))


def test_composite_docstring_is_self_contained_for_geometry_alpha_adapt_and_ownership() -> None:
    """v1-composite acceptance 1-17: the public docstring carries the non-obvious call contract."""
    docstring = inspect.getdoc(px.composite.merge) or ""
    for required in (
        "background",
        "foreground",
        "position",
        "scale",
        "rotation",
        "interpolation",
        "mask",
        "opacity",
        "straight",
        "premultiplied",
        "adapt",
        "float32",
        "new",
        "storage",
        "clamp",
    ):
        assert required in docstring


def test_composite_implementation_uses_a_gpu_raw_kernel() -> None:
    """v1-composite acceptance 1 and 20: structural contract keeps per-pixel transform and blend on the GPU."""
    import pixtreme._composite.merge as composite_module

    source = inspect.getsource(composite_module)
    factory: Callable[[], object] = composite_module._composite_kernel
    assert "cp.RawKernel" in inspect.getsource(factory)
    assert "pixtreme_composite_images" in source
