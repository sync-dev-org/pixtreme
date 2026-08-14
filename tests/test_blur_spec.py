"""Specification and numerical-property tests for blur filters."""

from __future__ import annotations

import inspect
import math
from typing import Any

import numpy as np
import pytest

import pixtreme as px

BORDERS = ("mirror", "replicate", "wrap", "constant")
MEDIAN_MAX_SIZE = 7


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


def _pad(
    source: np.ndarray,
    *,
    radius_y: int,
    radius_x: int,
    border: str,
    border_value: float = 0.0,
) -> np.ndarray:
    padding = ((radius_y, radius_y), (radius_x, radius_x), (0, 0))
    if border == "constant":
        return np.pad(source, padding, mode="constant", constant_values=border_value)
    mode = {"mirror": "reflect", "replicate": "edge", "wrap": "wrap"}[border]
    return np.pad(source, padding, mode=mode)


def _box_reference(
    source: np.ndarray,
    *,
    size: int | tuple[int, int],
    normalize: bool,
    border: str,
    border_value: float = 0.0,
) -> np.ndarray:
    height, width = (size, size) if isinstance(size, int) else size
    radius_y = height // 2
    radius_x = width // 2
    padded = _pad(source, radius_y=radius_y, radius_x=radius_x, border=border, border_value=border_value)
    output = np.empty(source.shape, dtype=np.float64)
    for y in range(source.shape[0]):
        for x in range(source.shape[1]):
            value = padded[y : y + height, x : x + width].astype(np.float64).sum(axis=(0, 1))
            output[y, x] = value / (height * width) if normalize else value
    return output.astype(np.float32)


def _median_reference(source: np.ndarray, *, size: int, border: str, border_value: float = 0.0) -> np.ndarray:
    radius = size // 2
    padded = _pad(source, radius_y=radius, radius_x=radius, border=border, border_value=border_value)
    output = np.empty(source.shape, dtype=np.float32)
    for y in range(source.shape[0]):
        for x in range(source.shape[1]):
            output[y, x] = np.median(padded[y : y + size, x : x + size], axis=(0, 1))
    return output


def _gaussian_reference(source: np.ndarray, *, sigma: float, border: str, border_value: float = 0.0) -> np.ndarray:
    radius = math.ceil(3.0 * sigma)
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    axis = np.exp(-(coordinates**2) / (2.0 * sigma**2))
    kernel = np.outer(axis, axis)
    kernel /= kernel.sum()
    padded = _pad(source, radius_y=radius, radius_x=radius, border=border, border_value=border_value)
    output = np.empty(source.shape, dtype=np.float64)
    for y in range(source.shape[0]):
        for x in range(source.shape[1]):
            window = padded[y : y + kernel.shape[0], x : x + kernel.shape[1]].astype(np.float64)
            output[y, x] = np.sum(window * kernel[..., np.newaxis], axis=(0, 1))
    return output.astype(np.float32)


def _bilateral_reference(
    source: np.ndarray,
    *,
    sigma_space: float,
    sigma_value: float,
    border: str,
    border_value: float = 0.0,
) -> np.ndarray:
    radius = math.ceil(3.0 * sigma_space)
    padded = _pad(source, radius_y=radius, radius_x=radius, border=border, border_value=border_value)
    output = np.empty(source.shape, dtype=np.float64)
    for y in range(source.shape[0]):
        for x in range(source.shape[1]):
            center = source[y, x].astype(np.float64)
            weighted = np.zeros(source.shape[2], dtype=np.float64)
            weight_sum = 0.0
            for offset_y in range(-radius, radius + 1):
                for offset_x in range(-radius, radius + 1):
                    neighbor = padded[y + offset_y + radius, x + offset_x + radius].astype(np.float64)
                    distance_squared = float(offset_y * offset_y + offset_x * offset_x)
                    difference = neighbor - center
                    value_distance_squared = float(np.dot(difference, difference))
                    weight = math.exp(-distance_squared / (2.0 * sigma_space**2)) * math.exp(
                        -value_distance_squared / (2.0 * sigma_value**2)
                    )
                    weighted += neighbor * weight
                    weight_sum += weight
            output[y, x] = weighted / weight_sum
    return output.astype(np.float32)


def test_blur_public_signatures_and_frame_only_entries_are_actionable() -> None:
    """v1-blur acceptance 1 + v1-blur-vector acceptance 13: signatures expose constant border values."""
    import cupy as cp

    expected = {
        "gaussian_blur": ("frame", "sigma", "border", "border_value"),
        "box_blur": ("frame", "size", "border", "border_value"),
        "median_blur": ("frame", "size", "border", "border_value"),
        "bilateral_blur": ("frame", "sigma_space", "sigma_value", "border", "border_value"),
        "convolve_box": ("frame", "size", "normalize", "border", "border_value"),
    }
    calls = {
        "gaussian_blur": {"sigma": 1.0},
        "box_blur": {"size": 3},
        "median_blur": {"size": 3},
        "bilateral_blur": {"sigma_space": 1.0, "sigma_value": 0.25},
        "convolve_box": {"size": 3, "normalize": True},
    }
    array = cp.zeros((2, 2, 1), dtype=cp.float32)
    for name, parameters in expected.items():
        function = _blur_operation(name)
        signature = inspect.signature(function)
        assert tuple(signature.parameters) == parameters
        assert signature.parameters["frame"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        for parameter in parameters[1:]:
            assert signature.parameters[parameter].kind is inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters["border"].default == "mirror"
        assert signature.parameters["border_value"].default is None
        module = px.filter if name == "convolve_box" else px.filter
        assert name in module.__all__
        with pytest.raises(ValueError) as error:
            function(array, **calls[name])
        _assert_actionable(error)


@pytest.mark.parametrize(
    ("name", "parameter"),
    (("gaussian_blur", "sigma"), ("bilateral_blur", "sigma_space"), ("bilateral_blur", "sigma_value")),
)
@pytest.mark.parametrize("value", (0, -0.5, True, "1", float("nan"), float("inf")))
def test_blur_rejects_nonpositive_or_nonfinite_real_sigmas(name: str, parameter: str, value: object) -> None:
    """v1-blur acceptance 2: all sigma parameters are finite positive real numbers with actionable errors."""
    source = _frame(np.zeros((2, 2, 1), dtype=np.float32), channels=["signal"])
    kwargs: dict[str, object] = {"sigma": 1.0} if name == "gaussian_blur" else {"sigma_space": 1.0, "sigma_value": 1.0}
    kwargs[parameter] = value

    with pytest.raises(ValueError) as error:
        _blur_operation(name)(source, **kwargs)
    _assert_actionable(error)


@pytest.mark.parametrize("name", ("box_blur", "median_blur"))
@pytest.mark.parametrize("size", (0, -1, 2, 2.0, True))
def test_square_blurs_reject_invalid_odd_sizes(name: str, size: object) -> None:
    """v1-blur acceptance 3: square box and median sizes are positive odd built-in integers."""
    source = _frame(np.zeros((2, 2, 1), dtype=np.float32), channels=["signal"])

    with pytest.raises(ValueError) as error:
        _blur_operation(name)(source, size=size)
    _assert_actionable(error)


def test_median_accepts_seven_and_rejects_the_first_larger_odd_size() -> None:
    """v1-blur acceptance 3 and provisional 3: median's measured GPU implementation limit is seven."""
    source = _frame(np.arange(9, dtype=np.float32).reshape(3, 3, 1), channels=["signal"])
    assert px.filter.median_blur(source, size=MEDIAN_MAX_SIZE).shape == source.shape

    with pytest.raises(ValueError) as error:
        px.filter.median_blur(source, size=MEDIAN_MAX_SIZE + 2)
    _assert_actionable(error)
    assert str(MEDIAN_MAX_SIZE) in str(error.value)


@pytest.mark.parametrize(
    "size",
    (0, -1, 2, 2.0, True, (), (3,), (3, 3, 3), (0, 3), (3, 0), (2, 3), (3, 2), [3, 3]),
)
def test_convolve_box_rejects_invalid_scalar_or_pair_sizes(size: object) -> None:
    """v1-blur acceptance 4: box convolution accepts a positive odd int or a two-int height-width pair."""
    source = _frame(np.zeros((2, 2, 1), dtype=np.float32), channels=["signal"])

    with pytest.raises(ValueError) as error:
        px.filter.convolve_box(source, size=size, normalize=True)
    _assert_actionable(error)


def test_convolve_box_requires_an_explicit_boolean_normalize() -> None:
    """v1-blur acceptance 5: normalize has no default and rejects non-bool values actionably."""
    signature = inspect.signature(px.filter.convolve_box)
    assert signature.parameters["normalize"].default is inspect.Parameter.empty
    source = _frame(np.zeros((2, 2, 1), dtype=np.float32), channels=["signal"])

    with pytest.raises(TypeError):
        px.filter.convolve_box(source, size=3)
    for value in (0, 1, None, "true"):
        with pytest.raises(ValueError) as error:
            px.filter.convolve_box(source, size=3, normalize=value)
        _assert_actionable(error)


@pytest.mark.parametrize(
    ("name", "kwargs"),
    (
        ("gaussian_blur", {"sigma": 0.6}),
        ("box_blur", {"size": 3}),
        ("median_blur", {"size": 3}),
        ("bilateral_blur", {"sigma_space": 0.6, "sigma_value": 0.4}),
        ("convolve_box", {"size": (3, 5), "normalize": True}),
    ),
)
def test_blur_border_axis_accepts_exact_tokens_and_lists_them_on_error(name: str, kwargs: dict[str, object]) -> None:
    """v1-blur acceptance 6 + v1-blur-vector acceptance 13: filters accept four exact border tokens."""
    source = _frame(np.arange(12, dtype=np.float32).reshape(3, 4, 1), channels=["signal"])
    function = _blur_operation(name)
    default = function(source, **kwargs)
    mirror = function(source, border="mirror", **kwargs)
    np.testing.assert_array_equal(
        px.io.to_array(
            default,
        ).get(),
        px.io.to_array(
            mirror,
        ).get(),
    )
    for border in BORDERS:
        border_kwargs = {"border_value": -0.25} if border == "constant" else {}
        assert function(source, border=border, **border_kwargs, **kwargs).shape == source.shape

    with pytest.raises(ValueError) as error:
        function(source, border="reflect", **kwargs)
    _assert_actionable(error)
    for token in BORDERS:
        assert token in str(error.value)


def test_box_known_corner_solutions_fix_all_border_definitions() -> None:
    """v1-blur acceptance 7-10 and 19 + v1-blur-vector acceptance 12-13: hand-computed borders."""
    values = np.arange(9, dtype=np.float32).reshape(3, 3, 1)
    source = _frame(values, channels=["signal"])
    expected_corner = {"mirror": 24.0, "replicate": 12.0, "wrap": 36.0, "constant": 58.0}

    for border, expected in expected_corner.items():
        border_kwargs = {"border_value": 10.0} if border == "constant" else {}
        result = px.filter.convolve_box(source, size=3, normalize=False, border=border, **border_kwargs)
        assert (
            float(
                px.io.to_array(
                    result,
                ).get()[0, 0, 0]
            )
            == expected
        )
        assert result.shape == source.shape


def test_wrap_uses_modulo_when_the_kernel_is_larger_than_the_image() -> None:
    """v1-blur acceptance 9 and 10: wrap remains periodic for a kernel wider and taller than the image."""
    values = np.arange(6, dtype=np.float32).reshape(2, 3, 1)
    source = _frame(values, channels=["signal"])
    expected = _box_reference(values, size=(7, 9), normalize=False, border="wrap")

    result = px.filter.convolve_box(source, size=(7, 9), normalize=False, border="wrap")

    np.testing.assert_array_equal(
        px.io.to_array(
            result,
        ).get(),
        expected,
    )


@pytest.mark.parametrize("border", BORDERS)
@pytest.mark.parametrize(
    ("name", "kwargs"),
    (
        ("gaussian_blur", {"sigma": 0.7}),
        ("box_blur", {"size": 5}),
        ("median_blur", {"size": 5}),
        ("bilateral_blur", {"sigma_space": 0.7, "sigma_value": 0.35}),
        ("convolve_box", {"size": (3, 5), "normalize": False}),
    ),
)
def test_blur_kernels_match_independent_numpy_oracles(name: str, kwargs: dict[str, object], border: str) -> None:
    """v1-blur acceptance 7-13 and 16-19 + v1-blur-vector acceptance 12-13: NumPy border oracle."""
    rng = np.random.default_rng(20260716)
    values = rng.uniform(-0.4, 1.4, size=(3, 4, 3)).astype(np.float32)
    source = _frame(values, channels=["temperature", "mask", "depth"])
    border_value = -0.35
    if name == "gaussian_blur":
        expected = _gaussian_reference(values, sigma=float(kwargs["sigma"]), border=border, border_value=border_value)
    elif name == "box_blur":
        expected = _box_reference(
            values,
            size=int(kwargs["size"]),
            normalize=True,
            border=border,
            border_value=border_value,
        )
    elif name == "median_blur":
        expected = _median_reference(values, size=int(kwargs["size"]), border=border, border_value=border_value)
    elif name == "bilateral_blur":
        expected = _bilateral_reference(
            values,
            sigma_space=float(kwargs["sigma_space"]),
            sigma_value=float(kwargs["sigma_value"]),
            border=border,
            border_value=border_value,
        )
    else:
        expected = _box_reference(
            values,
            size=kwargs["size"],
            normalize=bool(kwargs["normalize"]),
            border=border,
            border_value=border_value,
        )

    border_kwargs = {"border_value": border_value} if border == "constant" else {}
    result = _blur_operation(name)(source, border=border, **border_kwargs, **kwargs)

    # Direct kernels accumulate at most 63 fp32 samples in these fixtures;
    # 4e-5 covers expf and sum-order rounding while staying below 0.004% of unit scale.
    np.testing.assert_allclose(
        px.io.to_array(
            result,
        ).get(),
        expected,
        rtol=4e-5,
        atol=4e-5,
    )
    assert result.dtype == np.dtype(np.float32)


def test_blur_is_label_independent_unclamped_and_preserves_metadata_privately() -> None:
    """v1-blur acceptance 11-14: labels do not steer math, scene excursions and metadata survive privately."""
    values = np.asarray(
        [
            [[-0.5, 1.5], [-0.5, 1.5]],
            [[-0.5, 1.5], [-0.5, 1.5]],
        ],
        dtype=np.float32,
    )
    source = _frame(values, colorspace="ACEScg", gamma="logc4", channels=["depth", "confidence"])

    result = px.filter.box_blur(source, size=3, border="wrap")

    assert isinstance(result, px.core.Frame)
    assert result is not source
    assert result.data.data.ptr != source.data.data.ptr
    assert result.shape == source.shape
    assert (result.colorspace, result.gamma, result.channels) == ("ACEScg", "logc4", ("depth", "confidence"))
    np.testing.assert_array_equal(
        px.io.to_array(
            result,
        ).get(),
        values,
    )
    assert float(result.data.min()) < 0.0
    assert float(result.data.max()) > 1.0


@pytest.mark.parametrize(
    ("name", "kwargs"),
    (
        ("box_blur", {"size": 1}),
        ("median_blur", {"size": 1}),
        ("convolve_box", {"size": 1, "normalize": True}),
        ("convolve_box", {"size": 1, "normalize": False}),
    ),
)
def test_size_one_is_an_identity_with_private_storage(name: str, kwargs: dict[str, object]) -> None:
    """v1-blur acceptance 15: every size-one box or median path is an identity in a new allocation."""
    values = np.linspace(-0.5, 1.5, 18, dtype=np.float32).reshape(2, 3, 3)
    source = _frame(values)

    result = _blur_operation(name)(source, **kwargs)

    assert result.data.data.ptr != source.data.data.ptr
    np.testing.assert_array_equal(
        px.io.to_array(
            result,
        ).get(),
        values,
    )


def test_bilateral_value_distance_couples_all_channels() -> None:
    """v1-blur acceptance 11 and 17: one Euclidean all-channel distance supplies the weight for every channel."""
    values = np.asarray([[[0.0, 0.0], [0.1, 2.0], [0.2, 0.0]]], dtype=np.float32)
    source = _frame(values, channels=["first", "second"])
    kwargs = {"sigma_space": 0.6, "sigma_value": 0.5, "border": "replicate"}
    expected = _bilateral_reference(values, **kwargs)
    uncoupled = np.concatenate(
        [_bilateral_reference(values[..., index : index + 1], **kwargs) for index in range(values.shape[2])],
        axis=2,
    )

    result = px.io.to_array(
        px.filter.bilateral_blur(source, **kwargs),
    ).get()

    np.testing.assert_allclose(result, expected, rtol=2e-5, atol=2e-5)
    assert not np.allclose(result, uncoupled, rtol=1e-4, atol=1e-4)


def test_vocabulary_defines_border_tokens_defaults_and_cross_library_correspondence(
    vocabulary_markdown: str,
) -> None:
    """v1-blur acceptance 20 + v1-blur-vector acceptance 14: border vocabulary fixes all four tokens."""
    markdown = vocabulary_markdown
    section = markdown.split("## border\n", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]

    for required in (
        *BORDERS,
        "default",
        "np.pad",
        "scipy.ndimage",
        "cv2",
        "reflect",
        "REFLECT_101",
        "edge",
        "REPLICATE",
        "period",
        "name",
        "different behavior",
        "border_value",
    ):
        assert required in section


def test_blur_docstrings_are_self_contained_llm_readable_contracts() -> None:
    """v1-blur acceptance 21 + v1-blur-vector acceptance 17: docstrings expose constant border values."""
    for name in ("gaussian_blur", "box_blur", "median_blur", "bilateral_blur", "convolve_box"):
        docstring = inspect.getdoc(_blur_operation(name))
        assert docstring is not None
        for required in ("mirror", "replicate", "wrap", "constant", "border_value", "does not clamp"):
            assert required in docstring

    gaussian_docstring = inspect.getdoc(px.filter.gaussian_blur)
    assert gaussian_docstring is not None
    assert "radius = ceil(3 * sigma)" in gaussian_docstring

    bilateral_docstring = inspect.getdoc(px.filter.bilateral_blur)
    assert bilateral_docstring is not None
    for required in ("radius = ceil(3 * sigma_space)", "sigma_value", "working-value scale", "does not normalize"):
        assert required in bilateral_docstring

    convolve_docstring = inspect.getdoc(px.filter.convolve_box)
    assert convolve_docstring is not None
    assert "normalize" in convolve_docstring
    assert "must be passed explicitly" in convolve_docstring
