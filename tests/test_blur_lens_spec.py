"""Specification and numerical-property tests for flat-aperture lens blur."""

from __future__ import annotations

import inspect
import math
from numbers import Integral
from typing import Any

import numpy as np
import pytest

import pixtreme as px

BORDERS = ("mirror", "replicate", "wrap", "constant")
SUBSAMPLES = 16


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


def _coverage_reference(*, radius: float, blades: int | None, rotation: float) -> np.ndarray:
    """Build the specified 16x16 center-subsampled aperture independently in NumPy."""
    bound = math.ceil(radius + 0.5)
    coverage = np.zeros((2 * bound + 1, 2 * bound + 1), dtype=np.float32)
    subpixel = (np.arange(SUBSAMPLES, dtype=np.float64) + 0.5) / SUBSAMPLES - 0.5
    local_x, local_y = np.meshgrid(subpixel, subpixel)

    vertices: np.ndarray | None = None
    if blades is not None:
        angles = np.deg2rad(rotation) + np.arange(blades, dtype=np.float64) * (2.0 * np.pi / blades)
        # Array y grows down, so positive mathematical rotation is visually counterclockwise.
        vertices = np.stack((radius * np.cos(angles), -radius * np.sin(angles)), axis=1)

    for offset_y in range(-bound, bound + 1):
        for offset_x in range(-bound, bound + 1):
            samples_x = offset_x + local_x
            samples_y = offset_y + local_y
            if vertices is None:
                inside = samples_x * samples_x + samples_y * samples_y <= radius * radius
            else:
                inside = np.ones(local_x.shape, dtype=np.bool_)
                for vertex_index in range(len(vertices)):
                    start = vertices[vertex_index]
                    end = vertices[(vertex_index + 1) % len(vertices)]
                    edge_x, edge_y = end - start
                    cross = edge_x * (samples_y - start[1]) - edge_y * (samples_x - start[0])
                    inside &= cross <= 0.0
            coverage[offset_y + bound, offset_x + bound] = np.float32(np.count_nonzero(inside) / 256.0)
    return coverage


def _lens_reference(
    source: np.ndarray,
    *,
    radius: float,
    blades: int | None,
    rotation: float,
    border: str,
    border_value: float,
) -> np.ndarray:
    """Apply the specified aperture as direct convolution using NumPy padding."""
    if radius == 0.0:
        return source.copy()
    coverage = _coverage_reference(radius=radius, blades=blades, rotation=rotation)
    coverage_sum = np.sum(coverage, dtype=np.float32)
    if coverage_sum == 0.0:
        return source.copy()
    weights = coverage / coverage_sum
    bound = coverage.shape[0] // 2
    pad_width = ((bound, bound), (bound, bound), (0, 0))
    if border == "mirror":
        padded = np.pad(source, pad_width, mode="reflect")
    elif border == "replicate":
        padded = np.pad(source, pad_width, mode="edge")
    elif border == "wrap":
        padded = np.pad(source, pad_width, mode="wrap")
    else:
        padded = np.pad(source, pad_width, mode="constant", constant_values=border_value)

    output = np.empty_like(source, dtype=np.float32)
    flipped = weights[::-1, ::-1, np.newaxis]
    for y in range(source.shape[0]):
        for x in range(source.shape[1]):
            window = padded[y : y + coverage.shape[0], x : x + coverage.shape[1]]
            output[y, x] = np.sum(window * flipped, axis=(0, 1), dtype=np.float32)
    return output


def test_blur_lens_public_signature_frame_entry_and_return_contract() -> None:
    """v1-blur-lens acceptance 1: expose the exact Frame-only public signature and return a Frame."""
    import cupy as cp

    signature = inspect.signature(px.filter.lens_blur)
    assert tuple(signature.parameters) == ("frame", "radius", "blades", "rotation", "border", "border_value")
    assert signature.parameters["frame"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for parameter in ("radius", "blades", "rotation", "border", "border_value"):
        assert signature.parameters[parameter].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["radius"].default is inspect.Parameter.empty
    assert signature.parameters["blades"].default is None
    assert signature.parameters["rotation"].default is None
    assert signature.parameters["border"].default == "mirror"
    assert signature.parameters["border_value"].default is None
    assert "lens_blur" in px.filter.__all__

    source = _frame(np.zeros((2, 3, 1)), channels=["signal"])
    assert isinstance(px.filter.lens_blur(source, radius=1.0), px.core.Frame)
    with pytest.raises(ValueError) as error:
        px.filter.lens_blur(cp.zeros((2, 3, 1), dtype=cp.float32), radius=1.0)
    _assert_actionable(error)


@pytest.mark.parametrize("radius", (-1, -0.25, True, "1", float("nan"), float("inf"), float("-inf")))
def test_blur_lens_radius_is_a_nonnegative_finite_non_bool_real(radius: object) -> None:
    """v1-blur-lens acceptance 2: radius rejects negative, non-real, bool, and non-finite values."""
    source = _frame(np.zeros((2, 3, 1)), channels=["signal"])
    with pytest.raises(ValueError) as error:
        px.filter.lens_blur(source, radius=radius)  # type: ignore[arg-type]
    _assert_actionable(error)


@pytest.mark.parametrize("radius", (0, 0.25, 2, np.float32(1.5)))
def test_blur_lens_radius_accepts_nonnegative_reals_without_an_artificial_upper_bound(radius: float) -> None:
    """v1-blur-lens acceptance 2: finite nonnegative real radii are accepted without an API cap."""
    source = _frame(np.zeros((2, 3, 1)), channels=["signal"])
    assert px.filter.lens_blur(source, radius=radius).shape == source.shape


@pytest.mark.parametrize("blades", (-1, 0, 1, 2, True, 3.0, "3"))
def test_blur_lens_blades_is_none_or_an_integer_of_at_least_three(blades: object) -> None:
    """v1-blur-lens acceptance 3: blades rejects values outside None or non-bool integers >= 3."""
    source = _frame(np.zeros((2, 3, 1)), channels=["signal"])
    with pytest.raises(ValueError) as error:
        px.filter.lens_blur(source, radius=1.0, blades=blades)  # type: ignore[arg-type]
    _assert_actionable(error)


@pytest.mark.parametrize("blades", (None, 3, 6, np.int32(5)))
def test_blur_lens_blades_accepts_none_and_integral_values(blades: Integral | None) -> None:
    """v1-blur-lens acceptance 3: None selects a circle and integer blade counts select polygons."""
    source = _frame(np.zeros((2, 3, 1)), channels=["signal"])
    assert px.filter.lens_blur(source, radius=1.0, blades=blades).shape == source.shape


def test_blur_lens_rotation_requires_blades_and_defaults_to_zero_for_polygons() -> None:
    """v1-blur-lens acceptance 4 and 6: rotation is polygon-only and omitted means the +x zero-degree vertex."""
    source = _frame(np.arange(35, dtype=np.float32).reshape(5, 7, 1), channels=["signal"])
    implicit = px.io.to_array(
        px.filter.lens_blur(source, radius=2.0, blades=3),
    ).get()
    explicit = px.io.to_array(
        px.filter.lens_blur(source, radius=2.0, blades=3, rotation=0.0),
    ).get()
    np.testing.assert_array_equal(implicit, explicit)

    for rotation in (0, True, -45.0):
        with pytest.raises(ValueError) as error:
            px.filter.lens_blur(source, radius=1.0, rotation=rotation)
        _assert_actionable(error)


@pytest.mark.parametrize("rotation", (True, "0", float("nan"), float("inf"), float("-inf")))
def test_blur_lens_polygon_rotation_is_a_finite_non_bool_real(rotation: object) -> None:
    """v1-blur-lens acceptance 4: polygon rotation rejects bool, non-real, and non-finite values."""
    source = _frame(np.zeros((2, 3, 1)), channels=["signal"])
    with pytest.raises(ValueError) as error:
        px.filter.lens_blur(source, radius=1.0, blades=5, rotation=rotation)  # type: ignore[arg-type]
    _assert_actionable(error)


def test_blur_lens_border_axis_defaults_and_fails_fast_with_all_tokens() -> None:
    """v1-blur-lens acceptance 5: border is the exact four-token axis defaulting to mirror."""
    source = _frame(np.arange(12, dtype=np.float32).reshape(3, 4, 1), channels=["signal"])
    default = px.io.to_array(
        px.filter.lens_blur(source, radius=1.5),
    ).get()
    mirror = px.io.to_array(
        px.filter.lens_blur(source, radius=1.5, border="mirror"),
    ).get()
    np.testing.assert_array_equal(default, mirror)
    for border in BORDERS:
        border_kwargs = {"border_value": -0.25} if border == "constant" else {}
        assert px.filter.lens_blur(source, radius=1.5, border=border, **border_kwargs).shape == source.shape

    with pytest.raises(ValueError) as error:
        px.filter.lens_blur(source, radius=1.5, border="reflect")
    _assert_actionable(error)
    for token in BORDERS:
        assert token in str(error.value)


def test_blur_lens_border_value_contract_is_finite_real_symmetric_and_bool_excluding() -> None:
    """v1-blur-lens acceptance 5: constant requires border_value and every other border forbids it."""
    source = _frame(np.zeros((2, 3, 1)), channels=["signal"])
    for border_value in (-2, 1.5, np.float32(0.25)):
        assert (
            px.filter.lens_blur(source, radius=1.0, border="constant", border_value=border_value).shape == source.shape
        )
    for border_value in (None, True, "0", float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError) as error:
            px.filter.lens_blur(source, radius=1.0, border="constant", border_value=border_value)  # type: ignore[arg-type]
        _assert_actionable(error)
    for border in BORDERS[:3]:
        with pytest.raises(ValueError) as error:
            px.filter.lens_blur(source, radius=1.0, border=border, border_value=-0.25)
        _assert_actionable(error)


def test_radius_one_circle_matches_the_hand_counted_3x3_coverage_kernel() -> None:
    """v1-blur-lens acceptance 7-8: radius-one circle uses the hand-counted 16x16 coverage weights."""
    values = np.zeros((7, 7, 1), dtype=np.float32)
    values[3, 3, 0] = 1.0
    result = px.filter.lens_blur(_frame(values, channels=["signal"]), radius=1.0, border="constant", border_value=0.0)

    counts = np.asarray(((21, 118, 21), (118, 256, 118), (21, 118, 21)), dtype=np.float32)
    expected = np.zeros_like(values)
    expected[2:5, 2:5, 0] = counts / np.float32(812.0)
    np.testing.assert_allclose(
        px.io.to_array(
            result,
        ).get(),
        expected,
        rtol=1e-6,
        atol=1e-7,
    )


@pytest.mark.parametrize(("blades", "rotation"), ((None, 0.0), (3, 0.0), (5, 27.5), (6, -40.0)))
@pytest.mark.parametrize("border", BORDERS)
def test_blur_lens_matches_independent_numpy_coverage_and_direct_convolution_oracle(
    blades: int | None,
    rotation: float,
    border: str,
) -> None:
    """v1-blur-lens acceptance 6-8 and 10: circle/polygon fp32 gather matches an independent NumPy oracle."""
    rng = np.random.default_rng(20260717)
    values = rng.uniform(-0.8, 1.8, size=(4, 5, 3)).astype(np.float32)
    border_value = -0.65
    expected = _lens_reference(
        values,
        radius=1.75,
        blades=blades,
        rotation=rotation,
        border=border,
        border_value=border_value,
    )
    border_kwargs = {"border_value": border_value} if border == "constant" else {}
    result = px.filter.lens_blur(
        _frame(values, channels=["temperature", "mask", "depth"]),
        radius=1.75,
        blades=blades,
        rotation=None if blades is None else rotation,
        border=border,
        **border_kwargs,
    )

    # Both paths use the specified fp32 weights; 2e-6 covers accumulation-order differences.
    np.testing.assert_allclose(
        px.io.to_array(
            result,
        ).get(),
        expected,
        rtol=2e-6,
        atol=2e-6,
    )
    assert result.dtype == np.dtype(np.float32)


@pytest.mark.parametrize(("radius", "blades"), ((0.0, None), (0.0, 5), (0.01, None), (0.01, 3)))
def test_zero_and_zero_coverage_apertures_are_exact_identities_in_private_storage(
    radius: float,
    blades: int | None,
) -> None:
    """v1-blur-lens acceptance 9 and 11; v1-red-tokens acceptance 68: identities retain ARRI metadata."""
    values = np.linspace(-0.75, 1.75, 30, dtype=np.float32).reshape(3, 5, 2)
    source = _frame(values, colorspace="ACEScg", gamma="ARRI-LogC4", channels=["depth", "confidence"])

    result = px.filter.lens_blur(source, radius=radius, blades=blades)

    assert result is not source
    assert result.data.data.ptr != source.data.data.ptr
    np.testing.assert_array_equal(
        px.io.to_array(
            result,
        ).get(),
        values,
    )
    assert (result.width, result.height) == (source.width, source.height)
    assert (result.colorspace, result.gamma, result.channels) == ("ACEScg", "ARRI-LogC4", ("depth", "confidence"))


def test_blur_lens_preserves_metadata_channels_and_unclamped_scene_values_in_private_storage() -> None:
    """v1-blur-lens acceptance 10-11; v1-red-tokens acceptance 68: renamed ARRI metadata survives."""
    values = np.asarray(
        [
            [[-0.5, 1.5], [-0.5, 1.5], [-0.5, 1.5]],
            [[-0.5, 1.5], [-0.5, 1.5], [-0.5, 1.5]],
        ],
        dtype=np.float32,
    )
    source = _frame(values, colorspace="ACEScg", gamma="ARRI-LogC4", channels=["depth", "confidence"])

    result = px.filter.lens_blur(source, radius=1.25, blades=5, rotation=18.0, border="wrap")

    assert result.data.data.ptr != source.data.data.ptr
    assert result.shape == source.shape
    assert (result.colorspace, result.gamma, result.channels) == ("ACEScg", "ARRI-LogC4", ("depth", "confidence"))
    assert float(result.data.min()) < 0.0
    assert float(result.data.max()) > 1.0
    np.testing.assert_allclose(
        px.io.to_array(
            result,
        ).get()[..., 0],
        -0.5,
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        px.io.to_array(
            result,
        ).get()[..., 1],
        1.5,
        rtol=0.0,
        atol=2e-7,
    )


def test_blur_lens_docstring_is_a_self_contained_llm_readable_contract() -> None:
    """v1-blur-lens acceptance 12: docstring states optics, aperture, border, identity, and cost contracts."""
    docstring = inspect.getdoc(px.filter.lens_blur)
    assert docstring is not None
    for required in (
        "flat uniform aperture",
        "scene-linear",
        "above 1.0",
        "bokeh",
        "circumradius",
        "aperture area",
        "blades",
        "rotation",
        "0 degrees",
        "+x",
        "visually counterclockwise",
        "partial coverage",
        "16 x 16",
        "mirror",
        "replicate",
        "wrap",
        "constant",
        "border_value",
        "does not clamp",
        "radius = 0",
        "exact identity",
        "radius squared",
    ):
        assert required in docstring
