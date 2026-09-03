"""Specification and numerical-property tests for per-pixel vector blur."""

from __future__ import annotations

import inspect
import math
from typing import Any

import numpy as np
import pytest

import pixtreme as px

BORDERS = ("mirror", "replicate", "wrap", "constant")
SHUTTERS = ("centered", "forward", "backward")
EXISTING_CALLS = (
    ("gaussian_blur", {"sigma": 0.7}),
    ("box_blur", {"size": 3}),
    ("median_blur", {"size": 3}),
    ("bilateral_blur", {"sigma_space": 0.7, "sigma_value": 0.4}),
    ("convolve_box", {"size": (3, 5), "normalize": True}),
    ("directional_blur", {"angle": 15.0, "length": 2.0}),
    ("zoom_blur", {"amount": 0.5}),
    ("spin_blur", {"angle": 25.0}),
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


def _blur_operation(name: str) -> Any:
    module = px.filter if name == "convolve_box" else px.filter
    return getattr(module, name)


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
    border_value: float,
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


def _vector_reference(
    source: np.ndarray,
    vector: np.ndarray,
    *,
    shutter: str,
    border: str,
    border_value: float,
) -> np.ndarray:
    starts = {"centered": -0.5, "forward": 0.0, "backward": -1.0}
    output = np.empty(source.shape, dtype=np.float64)
    for y in range(source.shape[0]):
        for x in range(source.shape[1]):
            vector_x = float(vector[y, x, 0])
            vector_y = float(vector[y, x, 1])
            sample_count = max(2, math.ceil(math.hypot(vector_x, vector_y)) + 1)
            total = np.zeros(source.shape[2], dtype=np.float64)
            for sample in range(sample_count):
                unit = sample / (sample_count - 1)
                t = starts[shutter] + unit
                total += _bicubic_sample(
                    source,
                    x=x + t * vector_x,
                    y=y + t * vector_y,
                    border=border,
                    border_value=border_value,
                )
            output[y, x] = total / sample_count
    return output.astype(np.float32)


def test_blur_vector_public_signature_frame_entry_and_return_contract() -> None:
    """v1-blur-vector acceptance 1: the exact public signature is Frame-only and returns Frame."""
    import cupy as cp

    signature = inspect.signature(px.filter.vector_blur)
    assert tuple(signature.parameters) == ("frame", "vector", "shutter", "border", "border_value")
    assert signature.parameters["frame"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for parameter in ("vector", "shutter", "border", "border_value"):
        assert signature.parameters[parameter].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["shutter"].default == "centered"
    assert signature.parameters["border"].default == "mirror"
    assert signature.parameters["border_value"].default is None
    assert "vector_blur" in px.filter.__all__

    vector = _frame(np.zeros((2, 3, 2)), channels=["x", "y"])
    result = px.filter.vector_blur(_frame(np.zeros((2, 3, 1)), channels=["signal"]), vector=vector)
    assert isinstance(result, px.core.Frame)
    with pytest.raises(ValueError) as error:
        px.filter.vector_blur(cp.zeros((2, 3, 1), dtype=cp.float32), vector=vector)
    _assert_actionable(error)


def test_blur_vector_requires_a_two_channel_spatially_matching_frame_field() -> None:
    """v1-blur-vector acceptance 2: vector is a matching two-channel Frame with actionable failures."""
    import cupy as cp

    source = _frame(np.zeros((3, 4, 1)), channels=["signal"])
    invalid_vectors = (
        cp.zeros((3, 4, 2), dtype=cp.float32),
        _frame(np.zeros((3, 4, 1)), channels=["x"]),
        _frame(np.zeros((3, 4, 3)), channels=["x", "y", "z"]),
        _frame(np.zeros((2, 4, 2)), channels=["x", "y"]),
        _frame(np.zeros((3, 5, 2)), channels=["x", "y"]),
    )
    for vector in invalid_vectors:
        with pytest.raises(ValueError) as error:
            px.filter.vector_blur(source, vector=vector)
        _assert_actionable(error)


def test_blur_vector_component_positions_use_x_right_and_y_down_independent_of_labels() -> None:
    """v1-blur-vector acceptance 3: channel positions are +x right and +y down in pixel coordinates."""
    y, x = np.mgrid[:4, :4]
    values = (x + 10 * y).astype(np.float32)[..., np.newaxis]
    source = _frame(values, channels=["signal"])
    x_field = np.zeros((4, 4, 2), dtype=np.float32)
    x_field[..., 0] = 2.0
    y_field = np.zeros((4, 4, 2), dtype=np.float32)
    y_field[..., 1] = 2.0

    horizontal = px.io.to_array(
        px.filter.vector_blur(
            source,
            vector=_frame(x_field, channels=["vertical_label", "horizontal_label"]),
            shutter="forward",
            border="replicate",
        ),
    ).get()
    vertical = px.io.to_array(
        px.filter.vector_blur(
            source,
            vector=_frame(y_field, channels=["horizontal_label", "vertical_label"]),
            shutter="forward",
            border="replicate",
        ),
    ).get()

    assert float(horizontal[1, 1, 0]) == 12.0
    assert float(vertical[1, 1, 0]) == 21.0


def test_blur_vector_shutter_axis_defaults_and_fails_fast_with_all_tokens() -> None:
    """v1-blur-vector acceptance 4; v1-token-vocabulary acceptance 3: shutter is a normalized three-token axis."""
    source = _frame(np.arange(12, dtype=np.float32).reshape(3, 4, 1), channels=["signal"])
    vector = _frame(np.ones((3, 4, 2), dtype=np.float32), channels=["x", "y"])
    default = px.io.to_array(
        px.filter.vector_blur(source, vector=vector),
    ).get()
    centered = px.io.to_array(
        px.filter.vector_blur(source, vector=vector, shutter="centered"),
    ).get()
    np.testing.assert_array_equal(default, centered)
    for shutter in SHUTTERS:
        assert px.filter.vector_blur(source, vector=vector, shutter=shutter).shape == source.shape

    with pytest.raises(ValueError) as error:
        px.filter.vector_blur(source, vector=vector, shutter="symmetric")
    _assert_actionable(error)
    for token in SHUTTERS:
        assert token in str(error.value)


def test_blur_vector_border_axis_defaults_and_fails_fast_with_all_tokens() -> None:
    """v1-blur-vector acceptance 5; v1-token-vocabulary acceptance 3: border is a normalized four-token axis."""
    source = _frame(np.arange(12, dtype=np.float32).reshape(3, 4, 1), channels=["signal"])
    vector = _frame(np.ones((3, 4, 2), dtype=np.float32), channels=["x", "y"])
    default = px.io.to_array(
        px.filter.vector_blur(source, vector=vector),
    ).get()
    mirror = px.io.to_array(
        px.filter.vector_blur(source, vector=vector, border="mirror"),
    ).get()
    np.testing.assert_array_equal(default, mirror)
    for border in BORDERS:
        border_kwargs = {"border_value": 0.25} if border == "constant" else {}
        assert px.filter.vector_blur(source, vector=vector, border=border, **border_kwargs).shape == source.shape

    with pytest.raises(ValueError) as error:
        px.filter.vector_blur(source, vector=vector, border="reflect")
    _assert_actionable(error)
    for token in BORDERS:
        assert token in str(error.value)


def test_blur_vector_border_value_contract_is_finite_real_symmetric_and_bool_excluding() -> None:
    """v1-blur-vector acceptance 6: constant requires a finite non-bool real and other borders forbid it."""
    source = _frame(np.zeros((2, 3, 1)), channels=["signal"])
    vector = _frame(np.zeros((2, 3, 2)), channels=["x", "y"])
    for border_value in (-2, 1.5, np.float32(0.25)):
        assert (
            px.filter.vector_blur(source, vector=vector, border="constant", border_value=border_value).shape
            == source.shape
        )
    for border_value in (None, True, "0", float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError) as error:
            px.filter.vector_blur(source, vector=vector, border="constant", border_value=border_value)
        _assert_actionable(error)
    for border in BORDERS[:3]:
        with pytest.raises(ValueError) as error:
            px.filter.vector_blur(source, vector=vector, border=border, border_value=-0.25)
        _assert_actionable(error)


@pytest.mark.parametrize(("name", "kwargs"), EXISTING_CALLS)
def test_existing_blurs_share_the_symmetric_border_value_fail_fast_contract(
    name: str,
    kwargs: dict[str, object],
) -> None:
    """v1-blur-vector acceptance 13: all existing blurs require/forbid border_value symmetrically."""
    source = _frame(np.zeros((3, 4, 1)), channels=["signal"])
    function = _blur_operation(name)
    with pytest.raises(ValueError) as missing:
        function(source, border="constant", **kwargs)
    _assert_actionable(missing)
    with pytest.raises(ValueError) as excess:
        function(source, border="mirror", border_value=0.0, **kwargs)
    _assert_actionable(excess)
    for invalid in (True, float("nan"), float("inf")):
        with pytest.raises(ValueError) as invalid_value:
            function(source, border="constant", border_value=invalid, **kwargs)
        _assert_actionable(invalid_value)


@pytest.mark.parametrize("shutter", SHUTTERS)
@pytest.mark.parametrize("border", BORDERS)
def test_blur_vector_matches_independent_numpy_gather_bicubic_oracle(shutter: str, border: str) -> None:
    """v1-blur-vector acceptance 3 and 7-12; v1-red-tokens acceptance 68: vector metadata is inert."""
    rng = np.random.default_rng(20260717)
    values = rng.uniform(-0.8, 1.8, size=(4, 5, 3)).astype(np.float32)
    y, x = np.mgrid[:4, :5]
    vector_values = np.stack(((x - 1.7) * 0.9 + 0.35, (y - 1.2) * -0.8 + 0.2), axis=2).astype(np.float32)
    border_value = -0.65
    expected = _vector_reference(
        values,
        vector_values,
        shutter=shutter,
        border=border,
        border_value=border_value,
    )
    source = _frame(values, channels=["temperature", "mask", "depth"])
    vector = _frame(
        vector_values,
        colorspace="ACEScg",
        gamma="ARRI-LogC4",
        channels=["not_x", "not_y"],
    )
    border_kwargs = {"border_value": border_value} if border == "constant" else {}

    result = px.filter.vector_blur(source, vector=vector, shutter=shutter, border=border, **border_kwargs)

    # The oracle evaluates geometry and weights in float64; 3e-4 covers the specified
    # GPU fp32 path while staying below 0.03% of unit scale.
    np.testing.assert_allclose(
        px.io.to_array(
            result,
        ).get(),
        expected,
        rtol=3e-4,
        atol=3e-4,
    )
    assert result.dtype == np.dtype(np.float32)


def test_blur_vector_zero_field_is_exact_identity_in_private_storage() -> None:
    """v1-blur-vector acceptance 9-10 and 15: zero vectors interpolate exactly into new storage."""
    values = np.linspace(-0.75, 1.75, 30, dtype=np.float32).reshape(3, 5, 2)
    source = _frame(values, channels=["depth", "confidence"])
    vector = _frame(np.zeros((3, 5, 2)), channels=["x", "y"])

    result = px.filter.vector_blur(source, vector=vector)

    assert result.data.data.ptr != source.data.data.ptr
    np.testing.assert_array_equal(
        px.io.to_array(
            result,
        ).get(),
        values,
    )


def test_uniform_centered_vector_matches_directional_blur_known_solution() -> None:
    """v1-blur-vector acceptance 7-10: a uniform integer horizontal field matches directional blur."""
    rng = np.random.default_rng(17)
    values = rng.uniform(-0.4, 1.4, size=(5, 6, 2)).astype(np.float32)
    source = _frame(values, channels=["first", "second"])
    vector_values = np.zeros((5, 6, 2), dtype=np.float32)
    vector_values[..., 0] = 4.0
    vector = _frame(vector_values, channels=["x", "y"])

    actual = px.io.to_array(
        px.filter.vector_blur(source, vector=vector, border="wrap"),
    ).get()
    expected = px.io.to_array(
        px.filter.directional_blur(source, angle=0.0, length=4.0, border="wrap"),
    ).get()

    np.testing.assert_array_equal(actual, expected)


def test_blur_vector_preserves_source_metadata_and_ignores_vector_metadata() -> None:
    """v1-blur-vector acceptance 15; v1-red-tokens acceptance 68: source ARRI metadata survives."""
    values = np.linspace(-0.5, 1.5, 24, dtype=np.float32).reshape(3, 4, 2)
    vector_values = np.full((3, 4, 2), (0.75, -0.25), dtype=np.float32)
    source = _frame(values, colorspace="ACEScg", gamma="ARRI-LogC4", channels=["depth", "confidence"])
    first_vector = _frame(vector_values, colorspace="sRGB", gamma="sRGB", channels=["x", "y"])
    second_vector = _frame(vector_values, colorspace="Rec.2020", gamma="PQ", channels=["vertical", "horizontal"])

    first = px.filter.vector_blur(source, vector=first_vector)
    second = px.filter.vector_blur(source, vector=second_vector)

    assert first.data.data.ptr != source.data.data.ptr
    assert first.shape == source.shape
    assert (first.colorspace, first.gamma, first.channels) == ("ACEScg", "ARRI-LogC4", ("depth", "confidence"))
    np.testing.assert_array_equal(
        px.io.to_array(
            first,
        ).get(),
        px.io.to_array(
            second,
        ).get(),
    )


def test_blur_vector_docstring_is_a_self_contained_llm_readable_contract() -> None:
    """v1-blur-vector acceptance 11 and 16: docstring states the complete gather and cost contract."""
    docstring = inspect.getdoc(px.filter.vector_blur)
    assert docstring is not None
    for required in (
        "v(p)",
        "straight line",
        "motion boundary",
        "scatter",
        "channel 0 = x",
        "channel 1 = y",
        "+x is right",
        "+y is down",
        "pixels",
        "centered",
        "[-1/2, +1/2]",
        "forward",
        "[0, 1]",
        "backward",
        "[-1, 0]",
        "max(2, ceil(|v(p)|) + 1)",
        "bicubic",
        "Keys a = -0.5",
        "mirror",
        "replicate",
        "wrap",
        "constant",
        "border_value",
        "does not clamp",
        "finite",
        "undefined",
        "cost",
        "|v|",
    ):
        assert required in docstring
