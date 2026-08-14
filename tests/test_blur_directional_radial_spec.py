"""Specification and numerical-property tests for directional and radial path blurs."""

from __future__ import annotations

import inspect
import math
from typing import Any, Literal

import numpy as np
import pytest

import pixtreme as px

BORDERS = ("mirror", "replicate", "wrap", "constant")
PathKind = Literal["directional", "zoom", "spin"]


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


def _border_index(index: int, extent: int, border: str) -> int | None:
    if border == "constant" and not 0 <= index < extent:
        return None
    if extent <= 1:
        return 0
    if border == "replicate":
        return min(max(index, 0), extent - 1)
    if border == "wrap":
        return index % extent
    period = 2 * extent - 2
    reflected = index % period
    return reflected if reflected < extent else period - reflected


def _keys_weight(distance: float) -> float:
    x = abs(distance)
    if x < 1.0:
        return 1.5 * x**3 - 2.5 * x**2 + 1.0
    if x < 2.0:
        return -0.5 * x**3 + 2.5 * x**2 - 4.0 * x + 2.0
    return 0.0


def _bicubic_sample(
    source: np.ndarray,
    *,
    x: float,
    y: float,
    border: str,
    border_value: float = 0.0,
) -> np.ndarray:
    base_x = math.floor(x)
    base_y = math.floor(y)
    value = np.zeros(source.shape[2], dtype=np.float64)
    for sample_y in range(base_y - 1, base_y + 3):
        weight_y = _keys_weight(y - sample_y)
        source_y = _border_index(sample_y, source.shape[0], border)
        for sample_x in range(base_x - 1, base_x + 3):
            weight_x = _keys_weight(x - sample_x)
            source_x = _border_index(sample_x, source.shape[1], border)
            sample_value = (
                np.full(source.shape[2], border_value, dtype=np.float64)
                if source_x is None or source_y is None
                else source[source_y, source_x].astype(np.float64)
            )
            value += sample_value * weight_y * weight_x
    return value


def _path_reference(
    source: np.ndarray,
    *,
    kind: PathKind,
    magnitude: float,
    border: str,
    border_value: float = 0.0,
    direction: float = 0.0,
    center: tuple[float, float] | None = None,
) -> np.ndarray:
    height, width, _ = source.shape
    center_x, center_y = center if center is not None else ((width - 1) / 2.0, (height - 1) / 2.0)
    output = np.empty(source.shape, dtype=np.float64)
    direction_radians = math.radians(direction)
    magnitude_radians = math.radians(magnitude)

    for y in range(height):
        for x in range(width):
            offset_x = x - center_x
            offset_y = y - center_y
            radius = math.hypot(offset_x, offset_y)
            path_length = (
                magnitude if kind == "directional" else radius * (magnitude if kind == "zoom" else magnitude_radians)
            )
            sample_count = max(2, math.ceil(path_length) + 1)
            total = np.zeros(source.shape[2], dtype=np.float64)
            for sample in range(sample_count):
                unit = sample / (sample_count - 1)
                symmetric = unit - 0.5
                if kind == "directional":
                    distance = symmetric * magnitude
                    sample_x = x + distance * math.cos(direction_radians)
                    sample_y = y - distance * math.sin(direction_radians)
                elif kind == "zoom":
                    scale = 1.0 + symmetric * magnitude
                    sample_x = center_x + offset_x * scale
                    sample_y = center_y + offset_y * scale
                else:
                    angle = symmetric * magnitude_radians
                    sample_x = center_x + offset_x * math.cos(angle) + offset_y * math.sin(angle)
                    sample_y = center_y - offset_x * math.sin(angle) + offset_y * math.cos(angle)
                total += _bicubic_sample(
                    source,
                    x=sample_x,
                    y=sample_y,
                    border=border,
                    border_value=border_value,
                )
            output[y, x] = total / sample_count
    return output.astype(np.float32)


def test_blur_directional_radial_public_signatures_and_frame_only_entries_are_actionable() -> None:
    """v1-blur-directional-radial acceptance 1 + v1-blur-vector acceptance 13: constant signatures."""
    import cupy as cp

    expected = {
        "directional_blur": ("frame", "angle", "length", "border", "border_value"),
        "zoom_blur": ("frame", "amount", "center", "border", "border_value"),
        "spin_blur": ("frame", "angle", "center", "border", "border_value"),
    }
    calls = {
        "directional_blur": {"angle": 0.0, "length": 1.0},
        "zoom_blur": {"amount": 0.25},
        "spin_blur": {"angle": 10.0},
    }
    array = cp.zeros((2, 2, 1), dtype=cp.float32)
    for name, parameters in expected.items():
        function = getattr(px.filter, name)
        signature = inspect.signature(function)
        assert tuple(signature.parameters) == parameters
        assert signature.parameters["frame"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        for parameter in parameters[1:]:
            assert signature.parameters[parameter].kind is inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters["border"].default == "mirror"
        assert signature.parameters["border_value"].default is None
        assert name in px.filter.__all__
        with pytest.raises(ValueError) as error:
            function(array, **calls[name])
        _assert_actionable(error)


@pytest.mark.parametrize(
    ("name", "parameter", "base"),
    (
        ("directional_blur", "length", {"angle": 0.0, "length": 1.0}),
        ("zoom_blur", "amount", {"amount": 0.25}),
        ("spin_blur", "angle", {"angle": 10.0}),
    ),
)
@pytest.mark.parametrize("value", (0, -0.5, True, "1", float("nan"), float("inf")))
def test_blur_directional_radial_reject_invalid_positive_magnitudes(
    name: str,
    parameter: str,
    base: dict[str, object],
    value: object,
) -> None:
    """v1-blur-directional-radial acceptance 2-4: path magnitudes are finite positive real values."""
    source = _frame(np.zeros((2, 2, 1), dtype=np.float32), channels=["signal"])
    kwargs = {**base, parameter: value}

    with pytest.raises(ValueError) as error:
        getattr(px.filter, name)(source, **kwargs)
    _assert_actionable(error)


@pytest.mark.parametrize("angle", (0, -450.5, 720, np.float32(17.25)))
def test_directional_accepts_any_finite_real_angle(angle: object) -> None:
    """v1-blur-directional-radial acceptance 2 and 7: directional angle is a periodic signed real degree value."""
    source = _frame(np.arange(9, dtype=np.float32).reshape(3, 3, 1), channels=["signal"])

    assert px.filter.directional_blur(source, angle=angle, length=1.0).shape == source.shape


@pytest.mark.parametrize("angle", (True, "0", float("nan"), float("inf")))
def test_directional_rejects_nonfinite_or_nonreal_angles(angle: object) -> None:
    """v1-blur-directional-radial acceptance 2: unusable directional angle values fail actionably."""
    source = _frame(np.zeros((2, 2, 1), dtype=np.float32), channels=["signal"])

    with pytest.raises(ValueError) as error:
        px.filter.directional_blur(source, angle=angle, length=1.0)
    _assert_actionable(error)


@pytest.mark.parametrize("name", ("zoom_blur", "spin_blur"))
@pytest.mark.parametrize("center", ((), (1.0,), (1.0, 2.0, 3.0), ("x", 1.0), (True, 1.0), (float("nan"), 1.0)))
def test_radial_blurs_reject_invalid_centers_actionably(name: str, center: object) -> None:
    """v1-blur-directional-radial acceptance 5: center must be None or a finite two-real pair."""
    source = _frame(np.zeros((2, 2, 1), dtype=np.float32), channels=["signal"])
    kwargs = {"amount": 0.25} if name == "zoom_blur" else {"angle": 10.0}

    with pytest.raises(ValueError) as error:
        getattr(px.filter, name)(source, center=center, **kwargs)
    _assert_actionable(error)


@pytest.mark.parametrize("name,kwargs", (("zoom_blur", {"amount": 0.4}), ("spin_blur", {"angle": 25.0})))
def test_radial_default_center_is_geometric_and_off_image_centers_are_accepted(
    name: str,
    kwargs: dict[str, float],
) -> None:
    """v1-blur-directional-radial acceptance 5: center defaults geometrically and may lie outside the image."""
    values = np.arange(20, dtype=np.float32).reshape(4, 5, 1)
    source = _frame(values, channels=["signal"])
    function = getattr(px.filter, name)

    automatic = px.io.to_array(
        function(source, **kwargs),
    ).get()
    explicit = px.io.to_array(
        function(source, center=(2.0, 1.5), **kwargs),
    ).get()
    outside = function(source, center=(-8.5, 13.25), **kwargs)

    np.testing.assert_array_equal(automatic, explicit)
    assert outside.shape == source.shape


@pytest.mark.parametrize(
    ("name", "kwargs"),
    (
        ("directional_blur", {"angle": 31.0, "length": 2.5}),
        ("zoom_blur", {"amount": 0.65, "center": (1.2, 0.7)}),
        ("spin_blur", {"angle": 38.0, "center": (1.2, 0.7)}),
    ),
)
@pytest.mark.parametrize("border", BORDERS)
def test_blur_directional_radial_matches_independent_numpy_oracles(
    name: str,
    kwargs: dict[str, object],
    border: str,
) -> None:
    """v1-blur-directional-radial acceptance 7-16 and 18 + v1-blur-vector acceptance 12-13: border oracle."""
    rng = np.random.default_rng(20260717)
    values = rng.uniform(-0.7, 1.7, size=(4, 5, 3)).astype(np.float32)
    source = _frame(values, channels=["temperature", "mask", "depth"])
    border_value = -0.45
    if name == "directional_blur":
        expected = _path_reference(
            values,
            kind="directional",
            magnitude=float(kwargs["length"]),
            direction=float(kwargs["angle"]),
            border=border,
            border_value=border_value,
        )
    elif name == "zoom_blur":
        expected = _path_reference(
            values,
            kind="zoom",
            magnitude=float(kwargs["amount"]),
            center=kwargs["center"],  # type: ignore[arg-type]
            border=border,
            border_value=border_value,
        )
    else:
        expected = _path_reference(
            values,
            kind="spin",
            magnitude=float(kwargs["angle"]),
            center=kwargs["center"],  # type: ignore[arg-type]
            border=border,
            border_value=border_value,
        )

    border_kwargs = {"border_value": border_value} if border == "constant" else {}
    result = getattr(px.filter, name)(source, border=border, **border_kwargs, **kwargs)

    # The reference evaluates geometry and weights in float64 before fp32 output;
    # 2e-4 covers the specified GPU fp32 path while remaining below 0.02% of unit scale.
    np.testing.assert_allclose(
        px.io.to_array(
            result,
        ).get(),
        expected,
        rtol=2e-4,
        atol=2e-4,
    )
    assert result.dtype == np.dtype(np.float32)


def test_directional_known_axis_solutions_and_half_turn_periodicity() -> None:
    """v1-blur-directional-radial acceptance 7, 8, 11, and 13-14: axis samples and 180-degree period hold."""
    y, x = np.mgrid[:5, :5]
    values = (x**2 + 10 * y**2).astype(np.float32)[..., np.newaxis]
    source = _frame(values, channels=["signal"])

    horizontal = px.io.to_array(
        px.filter.directional_blur(source, angle=0.0, length=2.0, border="replicate"),
    ).get()
    vertical = px.io.to_array(
        px.filter.directional_blur(source, angle=90.0, length=2.0, border="replicate"),
    ).get()
    first = px.io.to_array(
        px.filter.directional_blur(source, angle=17.0, length=3.25),
    ).get()
    half_turn = px.io.to_array(
        px.filter.directional_blur(source, angle=197.0, length=3.25),
    ).get()
    many_turns = px.io.to_array(
        px.filter.directional_blur(source, angle=17.0 + 360.0 * 1_000_000, length=3.25),
    ).get()

    assert float(horizontal[2, 2, 0]) == pytest.approx((41.0 + 44.0 + 49.0) / 3.0, abs=2e-5)
    assert float(vertical[2, 2, 0]) == pytest.approx((14.0 + 44.0 + 94.0) / 3.0, abs=2e-5)
    np.testing.assert_allclose(first, half_turn, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(first, many_turns, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("border", BORDERS)
def test_paths_and_bicubic_support_may_cross_multiple_image_periods(border: str) -> None:
    """v1-blur-directional-radial acceptance 18 + v1-blur-vector acceptance 12-13: long-path border math."""
    values = np.arange(6, dtype=np.float32).reshape(2, 3, 1)
    source = _frame(values, channels=["signal"])
    border_value = 1.75
    expected = _path_reference(
        values,
        kind="directional",
        magnitude=12.5,
        direction=23.0,
        border=border,
        border_value=border_value,
    )

    border_kwargs = {"border_value": border_value} if border == "constant" else {}
    result = px.filter.directional_blur(source, angle=23.0, length=12.5, border=border, **border_kwargs)

    np.testing.assert_allclose(
        px.io.to_array(
            result,
        ).get(),
        expected,
        rtol=2e-4,
        atol=2e-4,
    )


def test_zero_length_pixels_are_exactly_interpolating_and_allocated_privately() -> None:
    """v1-blur-directional-radial acceptance 12-13 and 17: degenerate paths are exact in new storage."""
    values = np.linspace(-0.5, 1.5, 25, dtype=np.float32).reshape(5, 5, 1)
    source = _frame(values, channels=["signal"])
    directional = px.filter.directional_blur(source, angle=33.0, length=np.finfo(np.float32).tiny)
    zoom = px.filter.zoom_blur(source, amount=2.0)
    spin = px.filter.spin_blur(source, angle=180.0)

    np.testing.assert_array_equal(
        px.io.to_array(
            directional,
        ).get(),
        values,
    )
    assert (
        float(
            px.io.to_array(
                zoom,
            ).get()[2, 2, 0]
        )
        == values[2, 2, 0]
    )
    assert (
        float(
            px.io.to_array(
                spin,
            ).get()[2, 2, 0]
        )
        == values[2, 2, 0]
    )
    for result in (directional, zoom, spin):
        assert result.data.data.ptr != source.data.data.ptr


def test_blur_directional_radial_is_label_independent_unclamped_and_preserves_metadata_privately() -> None:
    """v1-blur-directional-radial acceptance 15-17: scene values and arbitrary metadata survive independently."""
    values = np.asarray(
        [
            [[-0.5, 1.5], [-0.5, 1.5]],
            [[-0.5, 1.5], [-0.5, 1.5]],
        ],
        dtype=np.float32,
    )
    source = _frame(values, colorspace="ACEScg", gamma="logc4", channels=["depth", "confidence"])

    result = px.filter.spin_blur(source, angle=70.0, center=(-2.0, 0.5), border="wrap")

    assert isinstance(result, px.core.Frame)
    assert result is not source
    assert result.data.data.ptr != source.data.data.ptr
    assert result.shape == source.shape
    assert (result.colorspace, result.gamma, result.channels) == ("ACEScg", "logc4", ("depth", "confidence"))
    assert float(result.data.min()) < 0.0
    assert float(result.data.max()) > 1.0


@pytest.mark.parametrize(
    ("name", "kwargs"),
    (
        ("directional_blur", {"angle": 15.0, "length": 2.0}),
        ("zoom_blur", {"amount": 0.5}),
        ("spin_blur", {"angle": 25.0}),
    ),
)
def test_blur_directional_radial_border_axis_accepts_exact_tokens_and_lists_them_on_error(
    name: str,
    kwargs: dict[str, float],
) -> None:
    """v1-blur-directional-radial acceptance 6 + v1-blur-vector acceptance 13: border has four exact tokens."""
    source = _frame(np.arange(12, dtype=np.float32).reshape(3, 4, 1), channels=["signal"])
    function = getattr(px.filter, name)
    default = px.io.to_array(
        function(source, **kwargs),
    ).get()
    mirror = px.io.to_array(
        function(source, border="mirror", **kwargs),
    ).get()
    np.testing.assert_array_equal(default, mirror)
    for border in BORDERS:
        border_kwargs = {"border_value": -0.25} if border == "constant" else {}
        assert function(source, border=border, **border_kwargs, **kwargs).shape == source.shape

    with pytest.raises(ValueError) as error:
        function(source, border="reflect", **kwargs)
    _assert_actionable(error)
    for token in BORDERS:
        assert token in str(error.value)


def test_blur_directional_radial_docstrings_are_self_contained_llm_readable_contracts() -> None:
    """v1-blur-directional-radial acceptance 19 + v1-blur-vector acceptance 17: constant border docstrings."""
    for name in ("directional_blur", "zoom_blur", "spin_blur"):
        docstring = inspect.getdoc(getattr(px.filter, name))
        assert docstring is not None
        for required in (
            "max(2, ceil(path length) + 1)",
            "bicubic",
            "Keys a = -0.5",
            "mirror",
            "replicate",
            "wrap",
            "constant",
            "border_value",
            "does not clamp",
        ):
            assert required in docstring

    directional_docstring = inspect.getdoc(px.filter.directional_blur)
    assert directional_docstring is not None
    for required in (
        "p + t",
        "[-length / 2, +length / 2]",
        "degree",
        "0 degrees is +x",
        "counterclockwise",
    ):
        assert required in directional_docstring

    for name in ("zoom_blur", "spin_blur"):
        docstring = inspect.getdoc(getattr(px.filter, name))
        assert docstring is not None
        for required in ("geometric center", "outside the image", "center distance", "cost"):
            assert required in docstring

    zoom_docstring = inspect.getdoc(px.filter.zoom_blur)
    assert zoom_docstring is not None
    assert "center + (p - center) * s" in zoom_docstring

    spin_docstring = inspect.getdoc(px.filter.spin_blur)
    assert spin_docstring is not None
    for required in ("circular arc", "degree", "0 degrees is +x", "counterclockwise"):
        assert required in spin_docstring
