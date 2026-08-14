"""Specification tests for the Gaussian unsharp-mask operation."""

from __future__ import annotations

import inspect
import math
from typing import Any

import numpy as np
import pytest

import pixtreme as px

BORDERS = ("mirror", "replicate", "wrap", "constant")


def _frame(
    values: Any,
    *,
    dtype: np.dtype[Any] | type[np.generic] = np.float32,
    colorspace: str = "sRGB",
    gamma: str = "linear",
    channels: str | tuple[str, ...] = "RGB",
) -> px.core.Frame:
    import cupy as cp

    return px.io.from_array(
        cp.asarray(np.asarray(values, dtype=dtype)),
        colorspace=colorspace,
        gamma=gamma,
        channels=channels,
    )


def _assert_actionable(error: pytest.ExceptionInfo[ValueError]) -> None:
    message = str(error.value)
    assert "why=" in message
    assert "; what=" in message
    assert "; how=" in message


def _gaussian_fixture(
    source: np.ndarray,
    *,
    sigma: float,
    border: str,
    border_value: float = 0.0,
) -> np.ndarray:
    """Evaluate the sheet formula directly in host float64, independent of the GPU implementation."""
    radius = math.ceil(3.0 * sigma)
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    axis_weights = np.exp(-(coordinates**2) / (2.0 * sigma**2))
    axis_weights /= axis_weights.sum()
    kernel = np.outer(axis_weights, axis_weights)
    padding = ((radius, radius), (radius, radius), (0, 0))
    if border == "constant":
        padded = np.pad(source, padding, mode="constant", constant_values=border_value)
    else:
        mode = {"mirror": "reflect", "replicate": "edge", "wrap": "wrap"}[border]
        padded = np.pad(source, padding, mode=mode)
    output = np.empty(source.shape, dtype=np.float64)
    for y in range(source.shape[0]):
        for x in range(source.shape[1]):
            window = padded[y : y + kernel.shape[0], x : x + kernel.shape[1]].astype(np.float64)
            output[y, x] = np.sum(window * kernel[..., np.newaxis], axis=(0, 1))
    return output


def _unsharp_fixture(
    source: np.ndarray,
    *,
    sigma: float,
    amount: float,
    border: str,
    border_value: float = 0.0,
) -> np.ndarray:
    blurred = _gaussian_fixture(source, sigma=sigma, border=border, border_value=border_value)
    source64 = source.astype(np.float64)
    return (source64 + amount * (source64 - blurred)).astype(np.float32)


def test_unsharp_mask_public_signature_and_frame_only_entry_are_exact() -> None:
    """v1-unsharp-mask acceptance 1: sigma and amount are required keyword-only parameters."""
    import cupy as cp

    signature = inspect.signature(px.filter.unsharp_mask)
    assert tuple(signature.parameters) == ("frame", "sigma", "amount", "border", "border_value")
    assert signature.parameters["frame"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in ("sigma", "amount", "border", "border_value"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["sigma"].default is inspect.Parameter.empty
    assert signature.parameters["amount"].default is inspect.Parameter.empty
    assert signature.parameters["border"].default == "mirror"
    assert signature.parameters["border_value"].default is None

    with pytest.raises(ValueError) as error:
        px.filter.unsharp_mask(cp.zeros((2, 2, 1), dtype=cp.float32), sigma=1.0, amount=1.0)
    _assert_actionable(error)


@pytest.mark.parametrize("border", BORDERS)
@pytest.mark.parametrize("amount", (1.25, -0.5))
def test_unsharp_mask_matches_hand_derived_gaussian_fixture_for_every_border(border: str, amount: float) -> None:
    """v1-unsharp-mask acceptance 2, 4, and 6: the formula and all borders match an independent fixture."""
    values = np.asarray(
        [
            [[-0.25, 0.1], [0.25, 0.4], [0.75, 1.2], [1.25, -0.1]],
            [[0.0, 0.8], [0.5, -0.3], [1.0, 0.6], [1.5, 0.2]],
            [[0.2, 1.4], [0.4, 0.0], [0.8, -0.5], [1.1, 0.9]],
        ],
        dtype=np.float32,
    )
    sigma = 0.7
    border_value = -0.35
    source = _frame(values, channels=("depth", "confidence"))
    expected = _unsharp_fixture(
        values,
        sigma=sigma,
        amount=amount,
        border=border,
        border_value=border_value,
    )
    border_kwargs = {"border_value": border_value} if border == "constant" else {}

    result = px.filter.unsharp_mask(source, sigma=sigma, amount=amount, border=border, **border_kwargs)

    np.testing.assert_allclose(
        px.io.to_array(
            result,
        ).get(),
        expected,
        rtol=5e-5,
        atol=5e-5,
    )


def test_zero_amount_is_a_private_bit_exact_identity() -> None:
    """v1-unsharp-mask acceptance 3 and 8: amount zero copies every fp32 bit without aliasing input."""
    values = np.asarray([[[-0.0], [0.25], [-1.5]], [[2.0], [1.0], [0.0]]], dtype=np.float32)
    source = _frame(values, channels=("signal",))

    result = px.filter.unsharp_mask(source, sigma=1.25, amount=-0.0, border="wrap")

    actual = px.io.to_array(
        result,
    ).get()
    np.testing.assert_array_equal(actual.view(np.uint32), values.view(np.uint32))
    assert result is not source
    assert result.data.data.ptr != source.data.data.ptr


@pytest.mark.parametrize(
    ("parameter", "value"),
    (
        ("sigma", True),
        ("sigma", "1"),
        ("sigma", 0.0),
        ("sigma", -1.0),
        ("sigma", float("inf")),
        ("sigma", float("nan")),
        ("amount", True),
        ("amount", "1"),
        ("amount", float("inf")),
        ("amount", float("-inf")),
        ("amount", float("nan")),
    ),
)
def test_unsharp_mask_rejects_invalid_sigma_and_amount_actionably(parameter: str, value: object) -> None:
    """v1-unsharp-mask acceptance 4: sigma is positive finite and amount is finite with three-part errors."""
    source = _frame(np.zeros((2, 2, 1), dtype=np.float32), channels=("signal",))
    kwargs: dict[str, object] = {"sigma": 1.0, "amount": 1.0}
    kwargs[parameter] = value

    with pytest.raises(ValueError) as error:
        px.filter.unsharp_mask(source, **kwargs)

    _assert_actionable(error)
    assert parameter in str(error.value)


def test_unsharp_mask_preserves_halo_excursions_without_clipping() -> None:
    """v1-unsharp-mask acceptance 5: overshoot, undershoot, and scene values are never clipped."""
    values = np.zeros((5, 5, 1), dtype=np.float32)
    values[2, 2, 0] = 1.5
    source = _frame(values, channels=("signal",))

    result = px.io.to_array(
        px.filter.unsharp_mask(source, sigma=0.7, amount=2.0, border="replicate"),
    ).get()

    assert float(result.max()) > 1.5
    assert float(result.min()) < 0.0


@pytest.mark.parametrize(
    ("border", "border_value"),
    (("reflect", None), ("Mirror", None), ("constant", None), ("constant", float("inf")), ("mirror", 0.0)),
)
def test_unsharp_mask_uses_the_shared_border_error_contract(border: str, border_value: object) -> None:
    """v1-unsharp-mask acceptance 6: border tokens and border_value follow the blur contract exactly."""
    source = _frame(np.zeros((2, 2, 1), dtype=np.float32), channels=("signal",))

    with pytest.raises(ValueError) as error:
        px.filter.unsharp_mask(source, sigma=1.0, amount=1.0, border=border, border_value=border_value)

    _assert_actionable(error)


@pytest.mark.parametrize(
    ("dtype", "guidance"),
    ((np.float16, "cast_dtype"), (np.uint8, "dequantize"), (np.uint16, "dequantize")),
)
def test_unsharp_mask_rejects_non_fp32_frames_with_conversion_guidance(
    dtype: np.dtype[Any] | type[np.generic],
    guidance: str,
) -> None:
    """v1-unsharp-mask acceptance 7: non-fp32 storage fails with an appropriate conversion path."""
    source = _frame(np.zeros((2, 2, 1)), dtype=dtype, channels=("signal",))

    with pytest.raises(ValueError) as error:
        px.filter.unsharp_mask(source, sigma=1.0, amount=1.0)

    _assert_actionable(error)
    assert "float32" in str(error.value)
    assert guidance in str(error.value)


def test_unsharp_mask_is_channel_label_independent_and_preserves_metadata_and_input() -> None:
    """v1-unsharp-mask acceptance 7-8: all channels including A share one formula and metadata/input survive."""
    values = np.linspace(-0.5, 1.5, 48, dtype=np.float32).reshape(3, 4, 4)
    source = _frame(values, colorspace="ACEScg", gamma="logc4", channels=("R", "G", "B", "A"))
    relabeled = _frame(values, colorspace="ACEScg", gamma="logc4", channels=("Z", "Y", "Cb", "Cr"))
    source_before = px.io.to_array(
        source,
    ).get()

    result = px.filter.unsharp_mask(source, sigma=0.8, amount=1.1)
    relabeled_result = px.filter.unsharp_mask(relabeled, sigma=0.8, amount=1.1)

    np.testing.assert_array_equal(
        px.io.to_array(
            source,
        ).get(),
        source_before,
    )
    np.testing.assert_array_equal(
        px.io.to_array(
            result,
        ).get(),
        px.io.to_array(
            relabeled_result,
        ).get(),
    )
    assert result.shape == source.shape
    assert result.dtype == np.dtype(np.float32)
    assert (result.colorspace, result.gamma, result.channels) == ("ACEScg", "logc4", ("R", "G", "B", "A"))


def test_unsharp_mask_docstring_is_a_self_contained_contract() -> None:
    """v1-unsharp-mask acceptance 1-8: the public docstring exposes the operational contract."""
    docstring = inspect.getdoc(px.filter.unsharp_mask)
    assert docstring is not None
    for required in (
        "input + amount * (input - G(input))",
        "radius = ceil(3 * sigma)",
        "mirror",
        "replicate",
        "wrap",
        "constant",
        "border_value",
        "negative",
        "does not clamp",
        "all channels",
        "float32",
        "cast_dtype",
    ):
        assert required in docstring


def test_border_vocabulary_lists_unsharp_mask_as_an_accepting_api(vocabulary_markdown: str) -> None:
    """v1-unsharp-mask acceptance 9: border vocabulary lists the new API without adding a token."""
    section = vocabulary_markdown.split("## border\n", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]

    assert "unsharp_mask" in section
    for token in BORDERS:
        assert token in section
