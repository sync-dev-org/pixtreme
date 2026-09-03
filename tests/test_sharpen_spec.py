"""Specification tests for the fixed-Laplacian sharpen operation."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest

import pixtreme as px

BORDERS = ("mirror", "replicate", "wrap", "constant")
_LAPLACIAN = np.asarray(((0.0, 1.0, 0.0), (1.0, -4.0, 1.0), (0.0, 1.0, 0.0)))


def _frame(
    values: Any,
    *,
    dtype: np.dtype[Any] | type[np.generic] = np.float32,
    colorspace: str = "sRGB",
    gamma: str = "linear",
    channels: str | tuple[str, ...] = "RGB",
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


def _laplacian_reference(source: np.ndarray, *, border: str, border_value: float) -> np.ndarray:
    """Evaluate the fixed Laplacian in host NumPy without using pixtreme filters."""
    output = np.empty_like(source, dtype=np.float32)
    height, width, channel_count = source.shape
    for y in range(height):
        for x in range(width):
            for channel in range(channel_count):
                total = 0.0
                for kernel_y, row in enumerate(_LAPLACIAN):
                    for kernel_x, coefficient in enumerate(row):
                        total += float(coefficient) * _sample(
                            source,
                            x=x + kernel_x - 1,
                            y=y + kernel_y - 1,
                            channel=channel,
                            border=border,
                            border_value=border_value,
                        )
                output[y, x, channel] = np.float32(total)
    return output


def _sharpen_reference(source: np.ndarray, *, amount: float, border: str, border_value: float) -> np.ndarray:
    laplacian = _laplacian_reference(source, border=border, border_value=border_value)
    return (source.astype(np.float64) - amount * laplacian.astype(np.float64)).astype(np.float32)


def test_sharpen_public_signature_and_single_canonical_path_are_exact() -> None:
    """v1-sharpen acceptance 1 and 9: amount is required keyword-only on the sole public path."""
    import cupy as cp

    signature = inspect.signature(px.filter.sharpen)
    assert tuple(signature.parameters) == ("frame", "amount", "border", "border_value")
    assert signature.parameters["frame"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in ("amount", "border", "border_value"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["amount"].default is inspect.Parameter.empty
    assert signature.parameters["border"].default == "mirror"
    assert signature.parameters["border_value"].default is None
    assert px.filter.__all__.count("sharpen") == 1
    assert not hasattr(px, "sharpen")
    assert not hasattr(px.core.Frame, "sharpen")

    with pytest.raises(ValueError) as error:
        px.filter.sharpen(cp.zeros((2, 2, 1), dtype=cp.float32), amount=1.0)
    _assert_actionable(error)


@pytest.mark.parametrize("border", BORDERS)
@pytest.mark.parametrize("amount", (1.25, -0.5))
def test_sharpen_matches_independent_laplacian_oracle_for_every_border(border: str, amount: float) -> None:
    """v1-sharpen acceptance 2, 3, 6, and 8: every channel and border follows the fixed formula."""
    values = np.asarray(
        [
            [[-0.25, 0.1], [0.25, 0.4], [0.75, 1.2], [1.25, -0.1]],
            [[0.0, 0.8], [0.5, -0.3], [1.0, 0.6], [1.5, 0.2]],
            [[0.2, 1.4], [0.4, 0.0], [0.8, -0.5], [1.1, 0.9]],
        ],
        dtype=np.float32,
    )
    border_value = -0.35
    expected = _sharpen_reference(values, amount=amount, border=border, border_value=border_value)
    kwargs = {"border_value": border_value} if border == "constant" else {}

    result = px.filter.sharpen(
        _frame(values, channels=("depth", "confidence")),
        amount=amount,
        border=border,
        **kwargs,
    )

    # Two fp32 GPU arithmetic stages may round once more than the host float64 composition.
    np.testing.assert_allclose(
        px.io.to_array(
            result,
        ).get(),
        expected,
        rtol=2e-6,
        atol=2e-6,
    )


def test_sharpen_one_pixel_extent_and_constant_scene_border_values_match_the_oracle() -> None:
    """v1-sharpen acceptance 3 and 8: one-pixel axes and unbounded constant values retain border semantics."""
    values = np.asarray([[[-0.5], [0.25], [1.5]]], dtype=np.float32)
    source = _frame(values, channels=("signal",))

    for border_value in (-1.25, 1.75):
        expected = _sharpen_reference(values, amount=0.75, border="constant", border_value=border_value)
        result = px.filter.sharpen(source, amount=0.75, border="constant", border_value=border_value)
        np.testing.assert_allclose(
            px.io.to_array(
                result,
            ).get(),
            expected,
            rtol=2e-6,
            atol=2e-6,
        )


@pytest.mark.parametrize("amount", (0.0, -0.0))
def test_sharpen_zero_amount_is_a_validated_private_bit_exact_identity(amount: float) -> None:
    """v1-sharpen acceptance 4 and 5: signed zero copies every fp32 bit into private storage."""
    values = np.asarray([[[-0.0], [0.25], [-1.5]], [[2.0], [1.0], [0.0]]], dtype=np.float32)
    source = _frame(values, channels=("signal",))

    result = px.filter.sharpen(source, amount=amount, border="wrap")

    actual = px.io.to_array(
        result,
    ).get()
    np.testing.assert_array_equal(actual.view(np.uint32), values.view(np.uint32))
    assert result is not source
    assert result.data.data.ptr != source.data.data.ptr
    assert result.data.flags.c_contiguous


@pytest.mark.parametrize(
    ("border", "border_value"),
    (("unknown", None), ("constant", None), ("constant", float("nan")), ("mirror", 0.0)),
)
def test_sharpen_zero_amount_does_not_skip_border_validation(border: str, border_value: object) -> None:
    """v1-sharpen acceptance 4 and 8: the identity path still validates border and border_value."""
    source = _frame(np.zeros((2, 2, 1), dtype=np.float32), channels=("signal",))

    with pytest.raises(ValueError) as error:
        px.filter.sharpen(source, amount=0.0, border=border, border_value=border_value)

    _assert_actionable(error)


@pytest.mark.parametrize("amount", (True, "1", 1 + 0j, float("nan"), float("inf"), float("-inf")))
def test_sharpen_rejects_invalid_amount_actionably(amount: object) -> None:
    """v1-sharpen acceptance 2: bool, non-real, and non-finite amount values fail fast."""
    source = _frame(np.zeros((2, 2, 1), dtype=np.float32), channels=("signal",))

    with pytest.raises(ValueError) as error:
        px.filter.sharpen(source, amount=amount)

    _assert_actionable(error)
    assert "amount" in str(error.value)


def test_sharpen_preserves_halo_excursions_shape_contiguity_and_input_storage() -> None:
    """v1-sharpen acceptance 5: halos remain unclipped in a new C-contiguous fp32 allocation."""
    values = np.zeros((5, 5, 1), dtype=np.float32)
    values[2, 2, 0] = 1.5
    source = _frame(values, channels=("signal",))
    source_before = px.io.to_array(
        source,
    ).get()

    result = px.filter.sharpen(source, amount=2.0, border="replicate")
    actual = px.io.to_array(
        result,
    ).get()

    assert float(actual.max()) > 1.5
    assert float(actual.min()) < 0.0
    assert result.shape == source.shape
    assert result.dtype == np.dtype(np.float32)
    assert result.data.flags.c_contiguous
    assert result.data.data.ptr != source.data.data.ptr
    np.testing.assert_array_equal(
        px.io.to_array(
            source,
        ).get(),
        source_before,
    )


def test_sharpen_is_channel_label_independent_and_preserves_all_metadata() -> None:
    """v1-sharpen acceptance 6; v1-red-tokens acceptance 68: labels do not affect renamed ARRI metadata."""
    values = np.linspace(-0.5, 1.5, 48, dtype=np.float32).reshape(3, 4, 4)
    source = _frame(
        values,
        colorspace="ACEScg",
        gamma="ARRI-LogC4",
        channels=("R", "G", "B", "A"),
        matrix="native",
    )
    relabeled = _frame(
        values,
        colorspace="ACEScg",
        gamma="ARRI-LogC4",
        channels=("Z", "Y", "Cb", "Cr"),
        matrix="native",
    )

    result = px.filter.sharpen(source, amount=1.1)
    relabeled_result = px.filter.sharpen(relabeled, amount=1.1)

    np.testing.assert_array_equal(
        px.io.to_array(
            result,
        ).get(),
        px.io.to_array(
            relabeled_result,
        ).get(),
    )
    assert (result.colorspace, result.gamma, result.channels, result.matrix) == (
        "ACEScg",
        "ARRI-LogC4",
        ("R", "G", "B", "A"),
        "native",
    )


@pytest.mark.parametrize(
    ("dtype", "guidance"),
    (
        (np.float16, ("cast_dtype",)),
        (np.uint8, ("recode_dtype", "dequantize")),
        (np.uint16, ("recode_dtype", "dequantize")),
    ),
)
def test_sharpen_rejects_non_fp32_frames_with_conversion_guidance(
    dtype: np.dtype[Any] | type[np.generic],
    guidance: tuple[str, ...],
) -> None:
    """v1-sharpen acceptance 7: every accepted non-fp32 storage dtype gets an actionable cast path."""
    source = _frame(np.zeros((2, 2, 1)), dtype=dtype, channels=("signal",))

    with pytest.raises(ValueError) as error:
        px.filter.sharpen(source, amount=1.0)

    _assert_actionable(error)
    message = str(error.value)
    assert "float32" in message
    assert all(name in message for name in guidance)


@pytest.mark.parametrize(
    ("border", "border_value"),
    (
        ("reflect", None),
        ("reflect", None),
        ("constant", None),
        ("constant", True),
        ("constant", float("inf")),
        ("mirror", 0.0),
    ),
)
def test_sharpen_uses_the_shared_border_error_contract(border: str, border_value: object) -> None:
    """v1-sharpen acceptance 8: the four border tokens and conditional value follow the shared contract."""
    source = _frame(np.zeros((2, 2, 1), dtype=np.float32), channels=("signal",))

    with pytest.raises(ValueError) as error:
        px.filter.sharpen(source, amount=1.0, border=border, border_value=border_value)

    _assert_actionable(error)


def test_sharpen_docstring_is_a_self_contained_contract() -> None:
    """v1-sharpen acceptance 9: the public docstring exposes the complete operational contract."""
    docstring = inspect.getdoc(px.filter.sharpen)
    assert docstring is not None
    for required in (
        "input - amount * laplacian(input)",
        "fixed non-normalized 3x3 Laplacian",
        "mirror",
        "replicate",
        "wrap",
        "constant",
        "border_value",
        "float32",
        "all channels",
        "does not clamp",
        "metadata",
        "input remains unchanged",
        "cast_dtype",
        "recode_dtype",
        "dequantize",
    ):
        assert required in docstring


def test_border_vocabulary_lists_sharpen_as_an_accepting_api(vocabulary_markdown: str) -> None:
    """v1-sharpen acceptance 10: border vocabulary adds the API without adding a token."""
    section = vocabulary_markdown.split("## border\n", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]

    assert "px.filter.sharpen" in section
    assert "default to `mirror`" in section
    for token in BORDERS:
        assert f"`{token}`" in section
