"""Specification, independent-oracle, and documentation tests for ``px.feature``."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import pixtreme as px
from pixtreme._feature.features import _DIRECT_MATCH_OPERATION_LIMIT, _window_sums

BORDERS = ("mirror", "replicate", "wrap", "constant")
METHODS = ("sqdiff", "sqdiff_normed", "ccorr", "ccorr_normed", "ccoeff", "ccoeff_normed")


def _frame(
    values: Any,
    *,
    dtype: np.dtype[Any] | type[np.generic] = np.float32,
    colorspace: str = "ACEScg",
    gamma: str = "linear",
    channels: str | tuple[str, ...] | list[str] = "RGB",
    matrix: str | None = None,
) -> px.core.Frame:
    import cupy as cp

    return px.io.from_array(
        cp.asarray(np.asarray(values, dtype=dtype)),
        colorspace=colorspace,
        gamma=gamma,
        channels=channels,
        matrix=matrix,
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


def _sobel_at(
    source: np.ndarray,
    *,
    x: int,
    y: int,
    channel: int,
    border: str,
    border_value: float,
) -> tuple[float, float]:
    """Evaluate the fixed Sobel pair in host scalars without a pixtreme operation."""
    samples = np.empty((3, 3), dtype=np.float64)
    for kernel_y in range(3):
        for kernel_x in range(3):
            samples[kernel_y, kernel_x] = _sample(
                source,
                x=x + kernel_x - 1,
                y=y + kernel_y - 1,
                channel=channel,
                border=border,
                border_value=border_value,
            )
    derivative_x = (
        -samples[0, 0] + samples[0, 2] - 2.0 * samples[1, 0] + 2.0 * samples[1, 2] - samples[2, 0] + samples[2, 2]
    )
    derivative_y = (
        -samples[0, 0] - 2.0 * samples[0, 1] - samples[0, 2] + samples[2, 0] + 2.0 * samples[2, 1] + samples[2, 2]
    )
    return derivative_x, derivative_y


def _harris_reference(
    source: np.ndarray,
    *,
    block_size: int,
    k: float,
    border: str,
    border_value: float,
) -> np.ndarray:
    """Evaluate v1-analysis-pair acceptance 9-13 on the host in scalar loops."""
    height, width, channel_count = source.shape
    radius = block_size // 2
    output = np.empty((height, width), dtype=np.float32)
    for output_y in range(height):
        for output_x in range(width):
            aggregate_a = 0.0
            aggregate_b = 0.0
            aggregate_d = 0.0
            for offset_y in range(-radius, radius + 1):
                for offset_x in range(-radius, radius + 1):
                    tensor_a = 0.0
                    tensor_b = 0.0
                    tensor_d = 0.0
                    for channel in range(channel_count):
                        derivative_x, derivative_y = _sobel_at(
                            source,
                            x=output_x + offset_x,
                            y=output_y + offset_y,
                            channel=channel,
                            border=border,
                            border_value=border_value,
                        )
                        tensor_a += derivative_x * derivative_x
                        tensor_b += derivative_x * derivative_y
                        tensor_d += derivative_y * derivative_y
                    aggregate_a += tensor_a
                    aggregate_b += tensor_b
                    aggregate_d += tensor_d
            response = (
                aggregate_a * aggregate_d
                - aggregate_b * aggregate_b
                - k * (aggregate_a + aggregate_d) * (aggregate_a + aggregate_d)
            )
            output[output_y, output_x] = np.float32(response)
    return output


def _match_reference(source: np.ndarray, template: np.ndarray, *, method: str) -> np.ndarray:
    """Evaluate all six template metrics in host NumPy without a pixtreme operation."""
    height, width, _ = source.shape
    template_height, template_width, _ = template.shape
    output = np.empty((height - template_height + 1, width - template_width + 1), dtype=np.float32)
    template64 = template.astype(np.float64)
    template_mean = template64.mean(axis=(0, 1), keepdims=True)
    centered_template = template64 - template_mean
    template_energy = float(np.sum(template64 * template64))
    centered_template_energy = float(np.sum(centered_template * centered_template))
    for y in range(output.shape[0]):
        for x in range(output.shape[1]):
            window = source[y : y + template_height, x : x + template_width, :].astype(np.float64)
            if method.startswith("ccoeff"):
                centered_window = window - window.mean(axis=(0, 1), keepdims=True)
                numerator = float(np.sum(centered_window * centered_template))
                denominator_squared = float(np.sum(centered_window * centered_window)) * centered_template_energy
            elif method.startswith("ccorr"):
                numerator = float(np.sum(window * template64))
                denominator_squared = float(np.sum(window * window)) * template_energy
            else:
                difference = window - template64
                numerator = float(np.sum(difference * difference))
                denominator_squared = float(np.sum(window * window)) * template_energy
            if not method.endswith("_normed"):
                output[y, x] = np.float32(numerator)
            elif denominator_squared > 0.0:
                output[y, x] = np.float32(numerator / np.sqrt(denominator_squared))
            elif method == "sqdiff_normed" and numerator > 0.0:
                output[y, x] = np.float32(np.inf)
            else:
                output[y, x] = np.float32(0.0)
    return output


def test_analysis_public_signatures_paths_and_array_contract_are_exact() -> None:
    """v1-analysis-pair acceptance 1, 2, 4-6, and 23: both ops have one exact path and raw-array output."""
    import cupy as cp

    corner_signature = inspect.signature(px.feature.corner_harris)
    assert tuple(corner_signature.parameters) == ("frame", "block_size", "k", "border", "border_value")
    assert corner_signature.parameters["frame"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert corner_signature.parameters["block_size"].default == 3
    assert corner_signature.parameters["k"].default == 0.04
    assert corner_signature.parameters["border"].default == "mirror"
    assert corner_signature.parameters["border_value"].default is None
    assert all(
        corner_signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        for name in ("block_size", "k", "border", "border_value")
    )

    match_signature = inspect.signature(px.feature.match_template)
    assert tuple(match_signature.parameters) == ("frame", "template", "method")
    assert match_signature.parameters["frame"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert match_signature.parameters["template"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert match_signature.parameters["method"].kind is inspect.Parameter.KEYWORD_ONLY
    assert match_signature.parameters["method"].default == "ccoeff_normed"

    source = _frame(np.arange(60, dtype=np.float32).reshape(4, 5, 3) / np.float32(10.0))
    template = _frame(source.data[:2, :3, :].get())
    corner = px.feature.corner_harris(source)
    matched = px.feature.match_template(source, template)
    for result, shape in ((corner, (4, 5)), (matched, (3, 3))):
        assert isinstance(result, cp.ndarray)
        assert result.shape == shape
        assert result.ndim == 2
        assert result.dtype == cp.float32
        assert result.flags.c_contiguous
        assert not isinstance(result, px.core.Frame)
        for metadata_name in ("colorspace", "gamma", "channels", "matrix"):
            assert not hasattr(result, metadata_name)

    assert px.feature.__all__.count("corner_harris") == 1
    assert px.feature.__all__.count("match_template") == 1
    assert not hasattr(px, "corner_harris")
    assert not hasattr(px, "match_template")
    assert not hasattr(px.filter, "corner_harris")
    assert not hasattr(px.filter, "match_template")
    assert not hasattr(px.core.Frame, "corner_harris")
    assert not hasattr(px.core.Frame, "match_template")


@pytest.mark.parametrize("name", ("corner_harris", "match_template"))
def test_analysis_rejects_non_frame_and_non_float32_inputs_actionably(name: str) -> None:
    """v1-analysis-pair acceptance 3: both entries require float32 Frame inputs with conversion guidance."""
    import cupy as cp

    function = getattr(px.feature, name)
    for value in (cp.zeros((2, 2, 1), dtype=cp.float32), np.zeros((2, 2, 1), dtype=np.float32), object()):
        arguments = (value,) if name == "corner_harris" else (value, _frame(np.zeros((1, 1, 1)), channels=["Y"]))
        with pytest.raises(ValueError) as error:
            function(*arguments)
        _assert_actionable(error)

    for dtype in (np.float16, np.uint8, np.uint16):
        source = _frame(np.ones((2, 2, 1)), dtype=dtype, channels=["Y"])
        arguments = (source,) if name == "corner_harris" else (source, source)
        with pytest.raises(ValueError) as error:
            function(*arguments)
        _assert_actionable(error)
        assert "float32" in str(error.value)
        assert any(token in str(error.value) for token in ("cast_dtype", "recode_dtype", "dequantize"))


@pytest.mark.parametrize("border", BORDERS)
def test_corner_harris_matches_independent_color_tensor_oracle_for_every_border(border: str) -> None:
    """v1-analysis-pair acceptance 5 and 8-13: all borders match the host color-tensor Harris formula."""
    values = np.asarray(
        [
            [[-0.5, 0.2, 1.4], [0.0, 1.1, -0.3], [1.5, -0.3, 0.7], [2.0, 0.7, 1.2]],
            [[0.4, -1.0, 0.6], [0.8, 0.5, 1.9], [1.2, 1.8, -0.6], [1.7, -0.6, 0.1]],
            [[-0.2, 0.9, 1.1], [0.3, -0.4, 0.8], [0.9, 1.3, -0.7], [1.4, 2.1, 0.2]],
        ],
        dtype=np.float32,
    )
    border_value = -0.75
    expected = _harris_reference(values, block_size=3, k=0.04, border=border, border_value=border_value)
    kwargs = {"border_value": border_value} if border == "constant" else {}
    actual = px.feature.corner_harris(_frame(values, channels=("A", "custom", "Z")), border=border, **kwargs).get()
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-4)


@pytest.mark.parametrize(("shape", "block_size"), (((1, 4, 2), 1), ((4, 1, 2), 5), ((2, 3, 2), 7)))
def test_corner_harris_supports_one_pixel_axes_and_windows_larger_than_the_image(
    shape: tuple[int, int, int], block_size: int
) -> None:
    """v1-analysis-pair acceptance 11 and 13-14: centered odd windows remain defined beyond every image edge."""
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / np.float32(3.0) - np.float32(0.5)
    expected = _harris_reference(values, block_size=block_size, k=0.07, border="mirror", border_value=0.0)
    actual = px.feature.corner_harris(
        _frame(values, channels=[f"c{index}" for index in range(shape[2])]),
        block_size=block_size,
        k=0.07,
    ).get()
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=5e-4)


def test_corner_harris_combines_channels_before_response_and_preserves_scene_scale() -> None:
    """v1-analysis-pair acceptance 7-8 and 10-12: orthogonal channels form one unclamped color tensor."""
    coordinates_y, coordinates_x = np.indices((7, 7), dtype=np.float32)
    values = np.stack((coordinates_x, coordinates_y), axis=2)
    source = _frame(values, channels=("A", "application-gradient"))
    relabeled = _frame(values, channels=("Z", "custom"))

    combined = px.feature.corner_harris(source, block_size=1, k=0.04, border="replicate")
    relabeled_result = px.feature.corner_harris(relabeled, block_size=1, k=0.04, border="replicate")
    channel_x = px.feature.corner_harris(_frame(values[..., :1], channels=["x"]), block_size=1, border="replicate")
    channel_y = px.feature.corner_harris(_frame(values[..., 1:], channels=["y"]), block_size=1, border="replicate")

    np.testing.assert_array_equal(combined.get(), relabeled_result.get())
    assert float(combined[3, 3]) > 1.0
    assert float(channel_x[3, 3]) < 0.0
    assert float(channel_y[3, 3]) < 0.0
    assert float(combined[3, 3]) != pytest.approx(float(channel_x[3, 3] + channel_y[3, 3]))


@pytest.mark.parametrize("block_size", (True, 1.0, "3", 0, -1, 2, 4))
def test_corner_harris_rejects_invalid_block_sizes_actionably(block_size: object) -> None:
    """v1-analysis-pair acceptance 14: block_size is a positive odd built-in int."""
    with pytest.raises(ValueError) as error:
        px.feature.corner_harris(_frame(np.zeros((2, 2, 1)), channels=["Y"]), block_size=block_size)  # type: ignore[arg-type]
    _assert_actionable(error)
    assert "odd" in str(error.value)


@pytest.mark.parametrize(
    "k",
    (True, "0.04", object(), 0.0, -0.1, 0.25, 1.0, float("nan"), float("inf"), float("-inf")),
)
def test_corner_harris_rejects_invalid_k_actionably(k: object) -> None:
    """v1-analysis-pair acceptance 15: k must convert to a finite real strictly between zero and one quarter."""
    with pytest.raises(ValueError) as error:
        px.feature.corner_harris(_frame(np.zeros((2, 2, 1)), channels=["Y"]), k=k)  # type: ignore[arg-type]
    _assert_actionable(error)
    assert "0.25" in str(error.value)


def test_corner_harris_accepts_real_k_boundaries_inside_the_open_interval() -> None:
    """v1-analysis-pair acceptance 15: representable values immediately inside both k bounds are accepted."""
    source = _frame(np.zeros((1, 1, 1)), channels=["Y"])
    for k in (np.nextafter(0.0, 1.0), np.nextafter(0.25, 0.0), np.float32(0.04)):
        assert px.feature.corner_harris(source, k=k).shape == (1, 1)


def test_corner_harris_uses_the_shared_border_error_contract() -> None:
    """v1-analysis-pair acceptance 13 and 16: border tokens and constant-only finite values fail fast."""
    source = _frame(np.zeros((2, 2, 1)), channels=["Y"])
    for border in BORDERS:
        kwargs = {"border_value": -0.5} if border == "constant" else {}
        assert px.feature.corner_harris(source, border=border, **kwargs).shape == (2, 2)
    for border, border_value in (
        ("reflect", None),
        ("constant", None),
        ("constant", True),
        ("constant", float("nan")),
        ("constant", float("inf")),
        ("mirror", 0.0),
    ):
        with pytest.raises(ValueError) as error:
            px.feature.corner_harris(source, border=border, border_value=border_value)  # type: ignore[arg-type]
        _assert_actionable(error)


@pytest.mark.parametrize("method", METHODS)
def test_match_template_matches_independent_valid_map_oracle_for_every_method(method: str) -> None:
    """v1-analysis-pair acceptance 6, 8, and 19-22: every metric matches a host valid-window oracle."""
    values = np.asarray(
        [
            [[-0.5, 0.2], [0.0, 1.1], [1.5, -0.3], [2.0, 0.7]],
            [[0.4, -1.0], [0.8, 0.5], [1.2, 1.8], [1.7, -0.6]],
            [[-0.2, 0.9], [0.3, -0.4], [0.9, 1.3], [1.4, 2.1]],
        ],
        dtype=np.float32,
    )
    template_values = np.asarray(
        [[[-0.4, 0.1], [0.2, 1.0]], [[0.5, -0.8], [0.7, 0.4]]],
        dtype=np.float32,
    )
    expected = _match_reference(values, template_values, method=method)
    actual = px.feature.match_template(
        _frame(values, channels=("A", "custom")),
        _frame(template_values, channels=("A", "custom")),
        method=method,
    ).get()
    np.testing.assert_allclose(actual, expected, rtol=4e-5, atol=4e-5)


def test_match_template_valid_coordinates_equal_shapes_and_one_by_one_are_exact() -> None:
    """v1-analysis-pair acceptance 6, 18, and 22: output indices are template top-left positions without padding."""
    values = np.arange(30, dtype=np.float32).reshape(3, 5, 2) - np.float32(7.0)
    template_values = values[1:, 2:, :].copy()
    response = px.feature.match_template(
        _frame(values, channels=("Z", "A")),
        _frame(template_values, channels=("Z", "A")),
        method="sqdiff",
    ).get()
    assert response.shape == (2, 3)
    assert np.unravel_index(int(np.argmin(response)), response.shape) == (1, 2)
    assert response[1, 2] == 0.0

    equal_response = px.feature.match_template(
        _frame(template_values, channels=("Z", "A")),
        _frame(template_values, channels=("Z", "A")),
        method="sqdiff",
    ).get()
    np.testing.assert_array_equal(equal_response, np.zeros((1, 1), dtype=np.float32))

    one_pixel = values[2:3, 4:5, :]
    one_pixel_response = px.feature.match_template(
        _frame(values, channels=("Z", "A")),
        _frame(one_pixel, channels=("Z", "A")),
        method="ccorr",
    ).get()
    np.testing.assert_allclose(one_pixel_response, _match_reference(values, one_pixel, method="ccorr"))


def test_match_template_zero_denominators_follow_the_exact_method_rules() -> None:
    """v1-analysis-pair acceptance 20: zero energy and zero variance never introduce NaN or epsilon."""
    zeros = _frame(np.zeros((3, 4, 2), dtype=np.float32), channels=("A", "custom"))
    zero_template = _frame(np.zeros((2, 2, 2), dtype=np.float32), channels=("A", "custom"))
    ones = _frame(np.ones((3, 4, 2), dtype=np.float32), channels=("A", "custom"))

    np.testing.assert_array_equal(px.feature.match_template(zeros, zero_template, method="sqdiff_normed").get(), 0.0)
    sqdiff_nonzero = px.feature.match_template(ones, zero_template, method="sqdiff_normed").get()
    assert np.isposinf(sqdiff_nonzero).all()
    for method in ("ccorr_normed", "ccoeff_normed"):
        np.testing.assert_array_equal(px.feature.match_template(ones, zero_template, method=method).get(), 0.0)
        assert not np.isnan(px.feature.match_template(zeros, zero_template, method=method).get()).any()


def test_window_sums_accumulator_dtype_is_explicit_and_preserves_integer_exactness() -> None:
    """REQ-TEST-001: keep float32 and explicitly requested integer accumulator policies exact."""
    import cupy as cp

    exact_value = 2**24 + 1
    source = cp.asarray([[[exact_value], [1]], [[2], [3]]], dtype=cp.int64)
    integer = _window_sums(source, height=2, width=2, dtype=cp.int64)
    floating = _window_sums(source, height=2, width=2, dtype=cp.float32)

    assert integer.dtype == cp.int64
    assert int(integer[0, 0, 0]) == exact_value + 6
    assert floating.dtype == cp.float32
    host_source = source.get()
    host_integral = np.cumsum(np.cumsum(host_source, axis=0, dtype=np.float32), axis=1, dtype=np.float32)
    assert float(floating[0, 0, 0]) == float(host_integral[-1, -1, 0])


def test_change_window_sums_uses_int32_through_signed_accumulator_limit_and_int64_above() -> None:
    """REQ-TEST-001: constant-window change counts use int32 through INT32_MAX and int64 above."""
    import cupy as cp

    from pixtreme._feature.features import _change_window_sums

    source = cp.zeros((1, 1, 1), dtype=cp.bool_)
    int32_max = int(np.iinfo(np.int32).max)

    at_limit = _change_window_sums(source, height=1, width=int32_max)
    above_limit = _change_window_sums(source, height=1, width=int32_max + 1)

    assert at_limit.size == above_limit.size == 0
    assert at_limit.dtype == cp.int32
    assert above_limit.dtype == cp.int64


@pytest.mark.parametrize(("height", "width"), ((1, 1), (1, 3), (3, 1), (2, 3), (4, 4)))
def test_match_template_constant_window_mask_matches_exact_host_property(height: int, width: int) -> None:
    """REQ-TEST-003: an independent host equality oracle fixes exact multi-channel constant-window detection."""
    import cupy as cp

    from pixtreme._feature.features import _constant_window_mask

    generator = np.random.default_rng(20260831 + height * 10 + width)
    source = generator.integers(-2, 3, size=(6, 7, 4), dtype=np.int16).astype(np.float32)
    source[1:5, 2:6, :] = np.asarray([0.125, -0.5, 1.25, 2.0], dtype=np.float32)
    expected = np.empty((source.shape[0] - height + 1, source.shape[1] - width + 1), dtype=np.bool_)
    for y in range(expected.shape[0]):
        for x in range(expected.shape[1]):
            window = source[y : y + height, x : x + width, :]
            expected[y, x] = np.all(window == window[0, 0, :])

    actual = _constant_window_mask(cp.asarray(source), height=height, width=width).get()

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("method", ("ccorr_normed", "ccoeff", "ccoeff_normed"))
@pytest.mark.parametrize("channel_count", (1, 2, 3, 4, 7, 16, 33))
def test_match_template_fft_fused_response_preserves_compositional_bits_characterization(
    method: str, channel_count: int
) -> None:
    """characterization: REQ-TEST-003 freezes the existing fp32 FFT post-processing bits across kernel fusion.

    The public formula is independently covered above, but its accepted optimization must also retain the current
    CuPy operation ordering exactly. Retire this snapshot-relative oracle if that explicit bit-identity requirement is
    replaced by a numeric tolerance contract.
    """
    import cupy as cp

    from pixtreme._feature import features

    generator = np.random.default_rng(20260817 + channel_count)
    correlation = cp.asarray([[2.0, -1.5], [0.75, 4.0]], dtype=cp.float32)
    window_sums = cp.asarray(generator.uniform(-2.0, 2.0, size=(2, 2, channel_count)), dtype=cp.float32)
    squared_window_sums = cp.asarray(generator.uniform(4.0, 8.0, size=(2, 2, channel_count)), dtype=cp.float32)
    template_sums = cp.asarray(generator.uniform(-1.0, 1.0, size=channel_count), dtype=cp.float32)
    template_energy = cp.asarray(np.float32(10.0 * channel_count), dtype=cp.float32)
    zero_variance = cp.asarray([[False, True], [False, False]], dtype=cp.bool_)
    spatial_count = np.float32(8.0)

    source_energy = cp.sum(squared_window_sums, axis=2, dtype=cp.float32)
    if method == "ccorr_normed":
        numerator = correlation
        denominator_squared = source_energy * template_energy
        expected = features._normalized_response(numerator, denominator_squared, sqdiff=False)
    else:
        numerator = correlation - cp.sum(
            window_sums * template_sums[None, None, :] / spatial_count,
            axis=2,
            dtype=cp.float32,
        )
        if method == "ccoeff":
            expected = cp.where(zero_variance, np.float32(0.0), numerator)
        else:
            centered_source_energy = source_energy - cp.sum(
                window_sums * window_sums / spatial_count,
                axis=2,
                dtype=cp.float32,
            )
            centered_template_energy = template_energy - cp.sum(
                template_sums * template_sums / spatial_count,
                dtype=cp.float32,
            )
            denominator_squared = cp.where(
                zero_variance,
                np.float32(0.0),
                centered_source_energy * centered_template_energy,
            )
            expected = features._normalized_response(numerator, denominator_squared, sqdiff=False)

    actual = features._fused_match_template_fft_response(
        correlation,
        window_sums,
        squared_window_sums,
        template_sums,
        template_energy,
        zero_variance,
        spatial_count=spatial_count,
        method=method,
    )

    np.testing.assert_array_equal(actual.get().view(np.uint32), expected.get().view(np.uint32))


@pytest.mark.parametrize(
    ("frame_shape", "template_shape", "uses_fft"),
    (
        pytest.param((50, 50, 2), (20, 20, 2), False, id="direct"),
        pytest.param((100, 100, 2), (30, 30, 2), True, id="fft"),
    ),
)
def test_match_template_nonzero_constant_ccoeff_normed_has_zero_variance(
    frame_shape: tuple[int, int, int],
    template_shape: tuple[int, int, int],
    *,
    uses_fft: bool,
) -> None:
    """v1-analysis-pair acceptance 20: both operation-limit paths return zero for exact constant variance."""
    output_height = frame_shape[0] - template_shape[0] + 1
    output_width = frame_shape[1] - template_shape[1] + 1
    operation_count = output_height * output_width * template_shape[0] * template_shape[1] * frame_shape[2]
    assert (operation_count > _DIRECT_MATCH_OPERATION_LIMIT) is uses_fft

    constant = np.float32(0.1)
    values = np.full(frame_shape, constant, dtype=np.float32)
    template_values = np.full(template_shape, constant, dtype=np.float32)
    response = px.feature.match_template(
        _frame(values, channels=("A", "custom")),
        _frame(template_values, channels=("A", "custom")),
        method="ccoeff_normed",
    ).get()

    np.testing.assert_array_equal(response, np.zeros((output_height, output_width), dtype=np.float32))


def test_match_template_large_ccoeff_normed_path_matches_the_independent_oracle() -> None:
    """v1-analysis-pair acceptance 19-20: the large-input path preserves per-channel normalized coefficients."""
    generator = np.random.default_rng(20260803)
    values = generator.uniform(-1.0, 2.0, size=(96, 97, 2)).astype(np.float32)
    template_values = generator.uniform(-0.8, 1.4, size=(29, 31, 2)).astype(np.float32)
    expected = _match_reference(values, template_values, method="ccoeff_normed")
    actual = px.feature.match_template(
        _frame(values, channels=("A", "custom")),
        _frame(template_values, channels=("A", "custom")),
        method="ccoeff_normed",
    ).get()
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)


def test_match_template_score_direction_offset_scale_and_channel_means_are_fixed() -> None:
    """v1-analysis-pair acceptance 7-8 and 19-22: score direction and per-channel centering are observable."""
    pattern = np.asarray(
        [[[0.0, 2.0], [1.0, 4.0]], [[2.0, 8.0], [4.0, 16.0]]],
        dtype=np.float32,
    )
    values = np.zeros((2, 6, 2), dtype=np.float32)
    values[:, 0:2, :] = pattern + np.asarray([5.0, -3.0], dtype=np.float32)
    values[:, 2:4, :] = pattern * np.float32(2.0)
    values[:, 4:6, :] = pattern
    frame = _frame(values, channels=("A", "custom"))
    template = _frame(pattern, channels=("A", "custom"))

    sqdiff = px.feature.match_template(frame, template, method="sqdiff").get()
    ccorr = px.feature.match_template(frame, template, method="ccorr").get()
    ccoeff = px.feature.match_template(frame, template, method="ccoeff").get()
    assert int(np.argmin(sqdiff)) == 4
    assert int(np.argmax(ccorr)) == 2
    assert int(np.argmax(ccoeff)) == 2
    assert ccoeff[0, 0] == pytest.approx(ccoeff[0, 4])
    assert sqdiff[0, 0] > 1.0
    assert not np.array_equal(ccoeff, ccorr)


@pytest.mark.parametrize("method", ("SQDIFF", "ccoeff-NORMED", "tm_ccorr", 0, None, object()))
def test_match_template_rejects_unknown_methods_actionably(method: object) -> None:
    """v1-analysis-pair acceptance 21: method is a case-sensitive six-token axis, never an integer constant."""
    source = _frame(np.zeros((2, 2, 1)), channels=["Y"])
    with pytest.raises(ValueError) as error:
        px.feature.match_template(source, source, method=method)  # type: ignore[arg-type]
    _assert_actionable(error)
    for token in METHODS:
        assert token in str(error.value)


@pytest.mark.parametrize(
    ("attribute", "frame_kwargs", "template_kwargs"),
    (
        ("channel count", {"channels": ("R", "G")}, {"channels": ("R",)}),
        ("channels", {"channels": ("R", "G")}, {"channels": ("G", "R")}),
        ("colorspace", {"colorspace": "ACEScg"}, {"colorspace": "sRGB"}),
        ("gamma", {"gamma": "linear"}, {"gamma": "srgb"}),
        ("matrix", {"matrix": None}, {"matrix": "bt709"}),
    ),
)
def test_match_template_rejects_metadata_and_channel_mismatches_actionably(
    attribute: str, frame_kwargs: dict[str, object], template_kwargs: dict[str, object]
) -> None:
    """v1-analysis-pair acceptance 17: both Frames must agree in dtype, channels, and all metadata."""
    frame_channels = frame_kwargs.pop("channels", ("R", "G"))
    template_channels = template_kwargs.pop("channels", ("R", "G"))
    frame_values = np.ones((3, 3, len(frame_channels)), dtype=np.float32)
    template_values = np.ones((2, 2, len(template_channels)), dtype=np.float32)
    frame = _frame(frame_values, channels=frame_channels, **frame_kwargs)
    template = _frame(template_values, channels=template_channels, **template_kwargs)
    with pytest.raises(ValueError) as error:
        px.feature.match_template(frame, template)
    _assert_actionable(error)
    assert attribute in str(error.value)


@pytest.mark.parametrize(
    ("frame_shape", "template_shape"),
    (
        ((2, 3, 1), (3, 2, 1)),
        ((3, 2, 1), (2, 3, 1)),
    ),
)
def test_match_template_rejects_oversized_valid_geometry_actionably(
    frame_shape: tuple[int, int, int], template_shape: tuple[int, int, int]
) -> None:
    """v1-analysis-pair acceptance 18: template height and width must fit inside the frame."""
    frame = _frame(np.zeros(frame_shape, dtype=np.float32), channels=["Y"])
    template = _frame(np.zeros(template_shape, dtype=np.float32), channels=["Y"])
    with pytest.raises(ValueError) as error:
        px.feature.match_template(frame, template)
    _assert_actionable(error)
    assert str(frame.shape) in str(error.value)
    assert str(template.shape) in str(error.value)
    assert "valid" in str(error.value)


def test_analysis_preserves_both_inputs_and_returns_private_storage_every_time() -> None:
    """v1-analysis-pair acceptance 4 and 7: data, metadata, and storage are immutable and never shared."""
    values = np.arange(60, dtype=np.float32).reshape(4, 5, 3) / np.float32(7.0) - np.float32(2.0)
    template_values = values[1:3, 2:5, :].copy()
    source = _frame(values, colorspace="ACEScg", gamma="logc4", channels=("A", "custom", "Z"), matrix="native")
    template = _frame(
        template_values,
        colorspace="ACEScg",
        gamma="logc4",
        channels=("A", "custom", "Z"),
        matrix="native",
    )
    source_before = source.data.copy()
    template_before = template.data.copy()
    source_metadata = (source.colorspace, source.gamma, source.channels, source.matrix)
    template_metadata = (template.colorspace, template.gamma, template.channels, template.matrix)

    first_corner = px.feature.corner_harris(source, border="wrap")
    second_corner = px.feature.corner_harris(source, border="wrap")
    first_match = px.feature.match_template(source, template, method="ccoeff_normed")
    second_match = px.feature.match_template(source, template, method="ccoeff_normed")

    assert first_corner.data.ptr not in (source.data.data.ptr, template.data.data.ptr, second_corner.data.ptr)
    assert first_match.data.ptr not in (source.data.data.ptr, template.data.data.ptr, second_match.data.ptr)
    np.testing.assert_array_equal(source.data.get(), source_before.get())
    np.testing.assert_array_equal(template.data.get(), template_before.get())
    assert (source.colorspace, source.gamma, source.channels, source.matrix) == source_metadata
    assert (template.colorspace, template.gamma, template.channels, template.matrix) == template_metadata


def test_analysis_docstrings_are_self_contained_operational_contracts() -> None:
    """v1-analysis-pair acceptance 23: both public docstrings explain formula, array boundary, and reconstruction."""
    corner_docstring = inspect.getdoc(px.feature.corner_harris) or ""
    match_docstring = inspect.getdoc(px.feature.match_template) or ""
    for docstring in (corner_docstring, match_docstring):
        for required in (
            "float32 Frame",
            "all channels",
            "(y, x)",
            "2D",
            "C-contiguous",
            "cupy.ndarray",
            "new storage",
            "does not mutate",
            "does not clamp",
            "cast_dtype",
            "recode_dtype",
            "dequantize",
            "from_array",
            "channel dimension",
            "metadata",
        ):
            assert required in docstring
    for required in ("Sobel", "tensor", "box sum", "A * D - B**2", "mirror", "replicate", "wrap", "constant"):
        assert required in corner_docstring
    for required in (*METHODS, "valid", "smaller", "larger", "+inf", "zero"):
        assert required in match_docstring


def test_analysis_vocabulary_defines_methods_and_harris_border(vocabulary_markdown: str) -> None:
    """v1-analysis-pair acceptance 24: vocabulary fixes all metric tokens, score direction, and Harris border."""
    method_section = vocabulary_markdown.split("## template matching method\n", maxsplit=1)[1].split(
        "\n## ", maxsplit=1
    )[0]
    border_section = vocabulary_markdown.split("## border\n", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
    for token in METHODS:
        assert f"`{token}`" in method_section
    for required in ("ccoeff_normed", "default", "Lower", "Higher", "mean", "sqrt"):
        assert required in method_section
    for required in ("px.feature.corner_harris", "mirror", "gradient stage", "aggregation window"):
        assert required in border_section


def test_analysis_requirements_define_modules_and_array_response_boundary() -> None:
    """v1-analysis-pair acceptance 27 / v1-public-namespace acceptance 1 and 8: canon retains feature responses."""
    requirements_path = Path(__file__).resolve().parents[1] / "docs" / "requirements.md"
    if not requirements_path.is_file():
        pytest.skip("repo-only documentation contract: docs/requirements.md is absent from this distribution")
    requirements = requirements_path.read_text(encoding="utf-8")
    architecture = requirements.split("**REQ-ARCH-008:", maxsplit=1)[1].split("\n\n", maxsplit=1)[0]
    modules = requirements.split("**REQ-API-009:", maxsplit=1)[1].split("**REQ-API-010:", maxsplit=1)[0]
    boundaries = requirements.split("**REQ-API-010:", maxsplit=1)[1].split("**REQ-API-011:", maxsplit=1)[0]
    assert "13 module" in architecture
    assert "`px.io.from_array`" in architecture
    assert "13 module" in modules
    assert "`feature`" in modules
    assert "`metrics`" in modules
    assert "画像ではない測定配列" in boundaries
    for path in ("px.feature.corner_harris", "px.feature.match_template"):
        assert path in boundaries
    assert "Frame → device 配列" in boundaries
    assert "px.io.to_array" in boundaries
