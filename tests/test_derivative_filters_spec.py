"""Specification, numerical-oracle, and documentation tests for derivative filters."""

from __future__ import annotations

import inspect
import math
from typing import Any

import numpy as np
import pytest

import pixtreme as px

BORDERS = ("mirror", "replicate", "wrap", "constant")
DIRECTIONS = ("x", "y", "magnitude")

_SOBEL_X = np.asarray(((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0)))
_SOBEL_Y = _SOBEL_X.T
_LAPLACIAN = np.asarray(((0.0, 1.0, 0.0), (1.0, -4.0, 1.0), (0.0, 1.0, 0.0)))


def _frame(
    values: Any,
    *,
    colorspace: str = "sRGB",
    gamma: str = "linear",
    channels: str | tuple[str, ...] | list[str] = "RGB",
    dtype: str = "float32",
) -> px.core.Frame:
    import cupy as cp

    return px.io.from_array(
        cp.asarray(np.asarray(values, dtype=np.dtype(dtype))),
        colorspace=colorspace,
        gamma=gamma,
        channels=channels,
    )


def _assert_actionable(error: pytest.ExceptionInfo[ValueError]) -> None:
    message = str(error.value)
    assert "why=" in message
    assert "; what=" in message
    assert "; how=" in message


def _border_index(index: int, extent: int, border: str) -> int:
    if extent <= 1:
        return 0
    if border == "replicate":
        return min(max(index, 0), extent - 1)
    if border == "wrap":
        return index % extent
    period = 2 * extent - 2
    reflected = index % period
    return reflected if reflected < extent else period - reflected


def _sample(
    source: np.ndarray,
    *,
    x: int,
    y: int,
    channel: int,
    border: str,
    border_value: float,
) -> float:
    height, width, _ = source.shape
    if border == "constant" and not (0 <= x < width and 0 <= y < height):
        return border_value
    return float(source[_border_index(y, height, border), _border_index(x, width, border), channel])


def _convolve_reference(
    source: np.ndarray,
    kernel: np.ndarray,
    *,
    border: str,
    border_value: float,
) -> np.ndarray:
    """Independent scalar NumPy oracle derived from v1-derivative-filters acceptance 8 and 11."""
    output = np.empty_like(source, dtype=np.float32)
    height, width, channel_count = source.shape
    radius_y = kernel.shape[0] // 2
    radius_x = kernel.shape[1] // 2
    for y in range(height):
        for x in range(width):
            for channel in range(channel_count):
                total = 0.0
                for kernel_y, row in enumerate(kernel):
                    for kernel_x, coefficient in enumerate(row):
                        total += float(coefficient) * _sample(
                            source,
                            x=x + kernel_x - radius_x,
                            y=y + kernel_y - radius_y,
                            channel=channel,
                            border=border,
                            border_value=border_value,
                        )
                output[y, x, channel] = np.float32(total)
    return output


def _gaussian_reference(
    source: np.ndarray,
    *,
    sigma: float,
    border: str,
    border_value: float,
) -> np.ndarray:
    """Host float64 Gaussian reference independent of pixtreme's blur implementation."""
    radius = math.ceil(3.0 * sigma)
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    weights = np.exp(-0.5 * coordinates * coordinates / (sigma * sigma))
    weights /= weights.sum()
    horizontal = np.empty_like(source, dtype=np.float64)
    output = np.empty_like(source, dtype=np.float64)
    height, width, channel_count = source.shape
    for y in range(height):
        for x in range(width):
            for channel in range(channel_count):
                horizontal[y, x, channel] = sum(
                    float(weight)
                    * _sample(
                        source,
                        x=x + offset,
                        y=y,
                        channel=channel,
                        border=border,
                        border_value=border_value,
                    )
                    for offset, weight in zip(range(-radius, radius + 1), weights, strict=True)
                )
    for y in range(height):
        for x in range(width):
            for channel in range(channel_count):
                output[y, x, channel] = sum(
                    float(weight)
                    * _sample(
                        horizontal,
                        x=x,
                        y=y + offset,
                        channel=channel,
                        border=border,
                        border_value=border_value,
                    )
                    for offset, weight in zip(range(-radius, radius + 1), weights, strict=True)
                )
    return output.astype(np.float32)


def test_derivative_public_signatures_and_frame_only_entries_are_exact() -> None:
    """v1-derivative-filters acceptance 1, 6, 10, and 13: three exact Frame APIs are public."""
    import cupy as cp

    expected = {
        "sobel": (
            ("frame", inspect.Parameter.empty),
            ("direction", "magnitude"),
            ("border", "mirror"),
            ("border_value", None),
        ),
        "laplacian": (("frame", inspect.Parameter.empty), ("border", "mirror"), ("border_value", None)),
        "difference_of_gaussians": (
            ("frame", inspect.Parameter.empty),
            ("sigma1", inspect.Parameter.empty),
            ("sigma2", inspect.Parameter.empty),
            ("border", "mirror"),
            ("border_value", None),
        ),
    }
    source = _frame(np.zeros((2, 3, 1), dtype=np.float32), channels=["signal"])
    calls = {
        "sobel": {},
        "laplacian": {},
        "difference_of_gaussians": {"sigma1": 1.0, "sigma2": 2.0},
    }
    for name, parameters in expected.items():
        function = getattr(px.filter, name)
        signature = inspect.signature(function)
        assert tuple((parameter.name, parameter.default) for parameter in signature.parameters.values()) == parameters
        assert signature.parameters["frame"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert all(
            parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in tuple(signature.parameters.values())[1:]
        )
        assert name in px.filter.__all__
        assert isinstance(function(source, **calls[name]), px.core.Frame)
        with pytest.raises(ValueError) as error:
            function(cp.zeros((2, 3, 1), dtype=cp.float32), **calls[name])
        _assert_actionable(error)


@pytest.mark.parametrize("border", BORDERS)
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_sobel_matches_independent_small_image_oracle_for_every_token(direction: str, border: str) -> None:
    """v1-derivative-filters acceptance 4 and 7-8: every direction and border matches a hand-derived oracle."""
    values = np.asarray(
        [
            [[-0.5, 0.2], [0.0, 1.1], [1.5, -0.3], [2.0, 0.7]],
            [[0.4, -1.0], [0.8, 0.5], [1.2, 1.8], [1.7, -0.6]],
            [[-0.2, 0.9], [0.3, -0.4], [0.9, 1.3], [1.4, 2.1]],
        ],
        dtype=np.float32,
    )
    border_value = -0.75
    expected_x = _convolve_reference(values, _SOBEL_X, border=border, border_value=border_value)
    expected_y = _convolve_reference(values, _SOBEL_Y, border=border, border_value=border_value)
    expected = {
        "x": expected_x,
        "y": expected_y,
        "magnitude": np.sqrt(expected_x * expected_x + expected_y * expected_y),
    }[direction]
    kwargs = {"border_value": border_value} if border == "constant" else {}

    result = px.filter.sobel(_frame(values, channels=["A", "custom"]), direction=direction, border=border, **kwargs)

    np.testing.assert_allclose(
        px.io.to_array(
            result,
        ).get(),
        expected,
        rtol=2e-6,
        atol=2e-6,
    )


def test_sobel_standard_scale_and_magnitude_composition_are_fixed() -> None:
    """v1-derivative-filters acceptance 8-9: unit ramp response is 8 and magnitude composes x/y."""
    ramp = np.broadcast_to(np.arange(5, dtype=np.float32)[None, :, None], (5, 5, 1)).copy()
    source = _frame(ramp, channels=["signal"])
    horizontal = px.io.to_array(
        px.filter.sobel(source, direction="x"),
    ).get()
    vertical = px.io.to_array(
        px.filter.sobel(source, direction="y"),
    ).get()
    magnitude = px.io.to_array(
        px.filter.sobel(source),
    ).get()

    assert float(horizontal[2, 2, 0]) == 8.0
    assert float(vertical[2, 2, 0]) == 0.0
    np.testing.assert_allclose(magnitude, np.sqrt(horizontal * horizontal + vertical * vertical), rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("border", BORDERS)
def test_laplacian_matches_independent_oracle_and_uniform_neutrality(border: str) -> None:
    """v1-derivative-filters acceptance 4 and 10-12: fixed Laplacian and uniform neutrality cover all borders."""
    values = np.asarray(
        [[[-0.5], [0.2], [1.5]], [[0.7], [2.0], [-0.4]], [[1.2], [0.1], [0.8]]],
        dtype=np.float32,
    )
    border_value = 0.35
    expected = _convolve_reference(values, _LAPLACIAN, border=border, border_value=border_value)
    kwargs = {"border_value": border_value} if border == "constant" else {}
    actual = px.io.to_array(
        px.filter.laplacian(_frame(values, channels=["signal"]), border=border, **kwargs),
    ).get()
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)

    uniform_value = -0.25
    uniform = _frame(np.full((3, 4, 2), uniform_value, dtype=np.float32), channels=["A", "mask"])
    uniform_kwargs = {"border_value": uniform_value} if border == "constant" else {}
    np.testing.assert_array_equal(
        px.io.to_array(
            px.filter.laplacian(uniform, border=border, **uniform_kwargs),
        ).get(),
        0.0,
    )


@pytest.mark.parametrize("border", BORDERS)
@pytest.mark.parametrize(("sigma1", "sigma2"), ((0.7, 1.2), (1.2, 0.7), (0.9, 0.9)))
def test_difference_of_gaussians_matches_independent_host_reference(border: str, sigma1: float, sigma2: float) -> None:
    """v1-derivative-filters acceptance 4 and 13-15: DoG preserves order and equality for every border."""
    values = np.asarray(
        [[[-0.5], [0.0], [1.2], [1.8]], [[0.3], [1.5], [-0.2], [0.7]], [[1.1], [0.4], [2.0], [-0.7]]],
        dtype=np.float32,
    )
    border_value = -0.4
    expected = _gaussian_reference(
        values, sigma=sigma1, border=border, border_value=border_value
    ) - _gaussian_reference(values, sigma=sigma2, border=border, border_value=border_value)
    kwargs = {"border_value": border_value} if border == "constant" else {}
    actual = px.io.to_array(
        px.filter.difference_of_gaussians(
            _frame(values, channels=["signal"]), sigma1=sigma1, sigma2=sigma2, border=border, **kwargs
        ),
    ).get()

    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=5e-5)
    if sigma1 == sigma2:
        np.testing.assert_array_equal(actual, 0.0)


def test_difference_of_gaussians_equals_the_public_blur_composition() -> None:
    """v1-derivative-filters acceptance 14: the public Gaussian composition is the numerical contract."""
    rng = np.random.default_rng(20260730)
    source = _frame(rng.uniform(-1.0, 2.0, size=(4, 5, 2)).astype(np.float32), channels=["A", "Z"])
    expected = (
        px.filter.gaussian_blur(source, sigma=0.8, border="wrap").data
        - px.filter.gaussian_blur(source, sigma=1.7, border="wrap").data
    )
    actual = px.filter.difference_of_gaussians(source, sigma1=0.8, sigma2=1.7, border="wrap")
    np.testing.assert_array_equal(
        px.io.to_array(
            actual,
        ).get(),
        expected.get(),
    )


@pytest.mark.parametrize("direction", ("horizontal", "diagonal", "length", None, 1))
def test_sobel_rejects_unknown_direction_actionably(direction: object) -> None:
    """v1-derivative-filters acceptance 5 and 7; v1-token-vocabulary acceptance 7: direction stays closed."""
    source = _frame(np.zeros((2, 2, 1), dtype=np.float32), channels=["signal"])
    with pytest.raises(ValueError) as error:
        px.filter.sobel(source, direction=direction)  # type: ignore[arg-type]
    _assert_actionable(error)
    for token in DIRECTIONS:
        assert token in str(error.value)


@pytest.mark.parametrize("parameter", ("sigma1", "sigma2"))
@pytest.mark.parametrize("value", (True, "1", 0.0, -1.0, float("inf"), float("-inf"), float("nan")))
def test_difference_of_gaussians_rejects_invalid_sigmas_actionably(parameter: str, value: object) -> None:
    """v1-derivative-filters acceptance 5 and 15: both sigmas require positive finite real values."""
    source = _frame(np.zeros((2, 2, 1), dtype=np.float32), channels=["signal"])
    kwargs: dict[str, object] = {"sigma1": 1.0, "sigma2": 2.0}
    kwargs[parameter] = value
    with pytest.raises(ValueError) as error:
        px.filter.difference_of_gaussians(source, **kwargs)
    _assert_actionable(error)
    assert parameter in str(error.value)


@pytest.mark.parametrize("name", ("sobel", "laplacian", "difference_of_gaussians"))
@pytest.mark.parametrize("dtype", ("float16", "uint8", "uint16"))
def test_derivative_filters_reject_non_fp32_with_conversion_guidance(name: str, dtype: str) -> None:
    """v1-derivative-filters acceptance 2: every derivative operation requires fp32 with a cast path."""
    source = _frame(np.ones((2, 2, 1)), channels=["signal"], dtype=dtype)
    kwargs = {"sigma1": 1.0, "sigma2": 2.0} if name == "difference_of_gaussians" else {}
    with pytest.raises(ValueError) as error:
        getattr(px.filter, name)(source, **kwargs)
    _assert_actionable(error)
    assert "float32" in str(error.value)
    assert any(token in str(error.value) for token in ("cast_dtype", "recode_dtype", "dequantize"))


@pytest.mark.parametrize("name", ("sobel", "laplacian", "difference_of_gaussians"))
def test_derivative_filters_share_the_border_error_contract(name: str) -> None:
    """v1-derivative-filters acceptance 4-5: four borders and constant-only finite values fail fast."""
    source = _frame(np.zeros((2, 2, 1), dtype=np.float32), channels=["signal"])
    base = {"sigma1": 1.0, "sigma2": 2.0} if name == "difference_of_gaussians" else {}
    function = getattr(px.filter, name)
    for border in BORDERS:
        kwargs = {"border_value": -0.5} if border == "constant" else {}
        assert function(source, border=border, **base, **kwargs).shape == source.shape
    for border, border_value in (("reflect", None), ("constant", None), ("constant", float("nan")), ("mirror", 0.0)):
        with pytest.raises(ValueError) as error:
            function(source, border=border, border_value=border_value, **base)
        _assert_actionable(error)


def test_derivative_filters_preserve_metadata_channels_scene_values_and_input() -> None:
    """v1-derivative-filters acceptance 1-3; v1-red-tokens acceptance 68: ARRI metadata survives filters."""
    values = np.asarray(
        [[[-1.0, 2.0], [0.5, -0.5], [3.0, 1.0]], [[2.0, -1.0], [-2.0, 4.0], [1.5, 0.0]]],
        dtype=np.float32,
    )
    source = _frame(values, colorspace="ACEScg", gamma="ARRI-LogC4", channels=["A", "application-mask"])
    relabeled = _frame(values, colorspace="ACEScg", gamma="ARRI-LogC4", channels=["Z", "Y"])
    before = px.io.to_array(source, copy=True).get()
    operations = (
        lambda frame: px.filter.sobel(frame, direction="x", border="wrap"),
        lambda frame: px.filter.laplacian(frame, border="wrap"),
        lambda frame: px.filter.difference_of_gaussians(frame, sigma1=0.5, sigma2=1.0, border="wrap"),
    )
    for operation in operations:
        result = operation(source)
        relabeled_result = operation(relabeled)
        assert result.shape == source.shape
        assert result.data.data.ptr != source.data.data.ptr
        assert (result.colorspace, result.gamma, result.channels) == (
            "ACEScg",
            "ARRI-LogC4",
            ("A", "application-mask"),
        )
        np.testing.assert_array_equal(
            px.io.to_array(
                result,
            ).get(),
            px.io.to_array(
                relabeled_result,
            ).get(),
        )
        assert float(result.data.min()) < 0.0
        assert float(result.data.max()) > 1.0
    np.testing.assert_array_equal(
        px.io.to_array(
            source,
        ).get(),
        before,
    )


def test_derivative_docstrings_are_self_contained_operational_contracts() -> None:
    """v1-derivative-filters acceptance 1-15: public docstrings expose kernels, tokens, and value contracts."""
    docstrings = {
        name: inspect.getdoc(getattr(px.filter, name)) or ""
        for name in ("sobel", "laplacian", "difference_of_gaussians")
    }
    for docstring in docstrings.values():
        for required in (
            "mirror",
            "replicate",
            "wrap",
            "constant",
            "border_value",
            "float32",
            "all channels",
            "does not clamp",
        ):
            assert required in docstring
    for required in ("x", "y", "magnitude", "sqrt(x**2 + y**2)", "[-1, 0, 1]", "[1, 2, 1]"):
        assert required in docstrings["sobel"]
    for required in ("[0, 1, 0]", "[1, -4, 1]", "LoG"):
        assert required in docstrings["laplacian"]
    for required in ("gaussian_blur", "sigma1", "sigma2", "ceil(3 * sigma)", "equal", "order"):
        assert required in docstrings["difference_of_gaussians"]


def test_derivative_vocabulary_defines_sobel_direction_and_shared_border_default(vocabulary_markdown: str) -> None:
    """v1-derivative-filters acceptance 16: vocabulary fixes direction semantics and mirror border inheritance."""
    direction_section = vocabulary_markdown.split("## sobel direction\n", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
    border_section = vocabulary_markdown.split("## border\n", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
    for token in DIRECTIONS:
        assert f"`{token}`" in direction_section
    for required in (
        "horizontal direction",
        "vertical edges",
        "vertical direction",
        "horizontal edges",
        "sqrt",
        "default",
    ):
        assert required in direction_section
    for name in ("sobel", "laplacian", "difference_of_gaussians"):
        assert name in border_section
    assert "mirror" in border_section
    assert "default" in border_section
