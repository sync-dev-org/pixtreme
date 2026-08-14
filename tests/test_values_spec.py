"""Specification tests for pixel-value quantization and range conversion."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import pixtreme as px

BIT_DEPTHS = (8, 10, 12, 14, 16)


def _assert_actionable(error: pytest.ExceptionInfo[ValueError]) -> None:
    message = str(error.value)
    assert "why=" in message
    assert "; what=" in message
    assert "; how=" in message


def _frame(
    values: Any,
    *,
    colorspace: str = "sRGB",
    gamma: str = "linear",
    channels: str | list[str] | None = None,
    dtype: Any = np.float32,
) -> px.core.Frame:
    import cupy as cp

    array = np.asarray(values, dtype=dtype)
    if array.ndim == 1:
        array = array.reshape(1, 1, -1)
    labels = channels if channels is not None else [f"channel-{index}" for index in range(array.shape[2])]
    return px.io.from_array(cp.asarray(array), colorspace=colorspace, gamma=gamma, channels=labels)


@pytest.mark.parametrize("shape", ((0, 2, 3), (2, 0, 3)), ids=("zero-height", "zero-width"))
def test_frame_construction_rejects_empty_spatial_input_before_value_operations(
    shape: tuple[int, int, int],
) -> None:
    """REQ-ARCH-002: empty spatial input is rejected before any values operation can receive a Frame."""
    import cupy as cp

    with pytest.raises(ValueError) as error:
        px.io.from_array(
            cp.empty(shape, dtype=cp.float32),
            colorspace="ACEScg",
            gamma="linear",
            channels="RGB",
        )
    _assert_actionable(error)
    assert repr(shape) in str(error.value)
    assert "at least 1" in str(error.value)


def test_quantize_values_clips_and_rounds_half_away_from_zero_at_ties() -> None:
    """v1-quantize-values acceptance 1 and 5: unsigned full-scale ties round upward after clipping."""
    values = np.asarray(
        [-1.0, 0.0, 0.5 / 255.0, 1.5 / 255.0, 0.5, 254.5 / 255.0, 1.0, 2.0],
        dtype=np.float32,
    )
    source = _frame(values)

    result = px.values.quantize(source, bit_depth=8)

    assert result.dtype == np.dtype(np.uint8)
    np.testing.assert_array_equal(
        px.io.to_array(
            result,
        )
        .get()
        .reshape(-1),
        np.asarray([0, 0, 1, 2, 128, 255, 255, 255], dtype=np.uint8),
    )


@pytest.mark.parametrize("bit_depth", BIT_DEPTHS)
def test_quantize_values_derives_the_container_and_full_scale(bit_depth: int) -> None:
    """v1-quantize-values acceptance 1 and 3: every accepted depth uses maximum code 2^B-1."""
    maximum = (1 << bit_depth) - 1
    container = np.uint8 if bit_depth == 8 else np.uint16
    source = _frame([0.0, 0.5, 1.0])

    result = px.values.quantize(source, bit_depth=bit_depth)

    assert result.dtype == np.dtype(container)
    np.testing.assert_array_equal(
        px.io.to_array(
            result,
        )
        .get()
        .reshape(-1),
        np.asarray([0, (maximum + 1) // 2, maximum], dtype=container),
    )


@pytest.mark.parametrize("bit_depth", BIT_DEPTHS)
def test_dequantize_values_divides_by_the_declared_maximum_without_rounding(bit_depth: int) -> None:
    """v1-quantize-values acceptance 2 and 3: integer codes divide by 2^B-1 into fp32."""
    maximum = (1 << bit_depth) - 1
    container = np.uint8 if bit_depth == 8 else np.uint16
    codes = np.asarray([0, maximum // 2, maximum], dtype=container)
    source = _frame(codes, dtype=container)

    result = px.values.dequantize(source, bit_depth=bit_depth)

    assert result.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(
        px.io.to_array(
            result,
        )
        .get()
        .reshape(-1),
        codes.astype(np.float32) * np.float32(1.0 / maximum),
    )


def test_dequantize_values_preserves_codes_above_the_declared_maximum() -> None:
    """v1-quantize-values acceptance 2: container-valid overshoot remains above 1.0."""
    source = _frame([0, 1023, 2046, 65535], dtype=np.uint16)

    result = px.values.dequantize(source, bit_depth=10)

    np.testing.assert_array_equal(
        px.io.to_array(
            result,
        )
        .get()
        .reshape(-1),
        np.asarray([0, 1023, 2046, 65535], dtype=np.float32) * np.float32(1.0 / 1023.0),
    )


@pytest.mark.parametrize("invalid", (0, 7, 9, 11, 13, 15, 17, 18, "8", True, 10.0))
@pytest.mark.parametrize("operation", ("quantize", "dequantize"))
def test_value_quantization_rejects_bit_depths_outside_the_closed_set(operation: str, invalid: object) -> None:
    """REQ-API-012 / v1-quantize-values acceptance 3: invalid bit depths fail actionably."""
    dtype = np.float32 if operation == "quantize" else np.uint8
    source = _frame([0, 0, 0], dtype=dtype)

    with pytest.raises(ValueError, match=r"\(8, 10, 12, 14, 16\)") as error:
        getattr(px.values, operation)(source, bit_depth=invalid)
    _assert_actionable(error)


@pytest.mark.parametrize(
    ("dtype", "routes"),
    (
        (np.float16, ("cast_dtype",)),
        (np.uint8, ("recode_dtype", "dequantize")),
        (np.uint16, ("recode_dtype", "dequantize")),
    ),
)
def test_quantize_values_rejects_non_float32_with_a_conversion_route(
    dtype: type[np.generic],
    routes: tuple[str, ...],
) -> None:
    """REQ-API-012 / v1-recode-dtype acceptance 9: fp32 errors retain an actionable conversion route."""
    source = _frame([0, 0, 0], dtype=dtype)

    with pytest.raises(ValueError) as error:
        px.values.quantize(source, bit_depth=8)
    _assert_actionable(error)
    message = str(error.value)
    assert "float32" in message
    positions = tuple(message.index(route) for route in routes)
    assert positions == tuple(sorted(positions))


@pytest.mark.parametrize(
    ("bit_depth", "dtype"),
    ((8, np.float32), (8, np.float16), (8, np.uint16), (10, np.float32), (10, np.float16), (10, np.uint8)),
)
def test_dequantize_values_requires_the_bit_depth_container(
    bit_depth: int,
    dtype: type[np.generic],
) -> None:
    """REQ-API-012 / v1-quantize-values acceptance 2: container mismatches fail actionably."""
    source = _frame([0, 0, 0], dtype=dtype)
    expected = "uint8" if bit_depth == 8 else "uint16"

    with pytest.raises(ValueError, match=expected) as error:
        px.values.dequantize(source, bit_depth=bit_depth)
    _assert_actionable(error)


def test_value_quantization_returns_new_frames_and_preserves_input_and_metadata() -> None:
    """v1-quantize-values acceptance 4: both directions allocate and preserve all metadata axes."""
    float_source = _frame(
        [0.0, 0.5, 1.0],
        colorspace="ACEScg",
        gamma="linear",
        channels="BGR",
    )
    float_original = (
        px.io.to_array(
            float_source,
        )
        .get()
        .copy()
    )
    quantized = px.values.quantize(float_source, bit_depth=10)
    integer_original = (
        px.io.to_array(
            quantized,
        )
        .get()
        .copy()
    )
    restored = px.values.dequantize(quantized, bit_depth=10)

    assert quantized is not float_source
    assert restored is not quantized
    assert quantized.data.data.ptr != float_source.data.data.ptr
    assert restored.data.data.ptr != quantized.data.data.ptr
    assert (quantized.colorspace, quantized.gamma, quantized.channels) == ("ACEScg", "linear", ("B", "G", "R"))
    assert (restored.colorspace, restored.gamma, restored.channels) == ("ACEScg", "linear", ("B", "G", "R"))
    np.testing.assert_array_equal(
        px.io.to_array(
            float_source,
        ).get(),
        float_original,
    )
    np.testing.assert_array_equal(
        px.io.to_array(
            quantized,
        ).get(),
        integer_original,
    )


def test_uint16_all_codes_round_trip_through_fp32_quantization() -> None:
    """v1-quantize-values acceptance 5: every 16-bit code survives code-to-fp32-to-code."""
    codes = np.arange(1 << 16, dtype=np.uint16).reshape(256, 256, 1)
    source = _frame(codes, channels="Y", dtype=np.uint16)

    restored = px.values.quantize(px.values.dequantize(source, bit_depth=16), bit_depth=16)

    np.testing.assert_array_equal(
        px.io.to_array(
            restored,
        ).get(),
        codes,
    )


@pytest.mark.parametrize(
    ("bit_depth", "denominator", "y_min", "y_max", "c_min", "c_max"),
    (
        (8, 255.0, 16.0, 235.0, 16.0, 240.0),
        (10, 1023.0, 64.0, 940.0, 64.0, 960.0),
        (12, 4095.0, 256.0, 3760.0, 256.0, 3840.0),
        (14, 16383.0, 1024.0, 15040.0, 1024.0, 15360.0),
        (16, 65535.0, 4096.0, 60160.0, 4096.0, 61440.0),
    ),
)
def test_range_functions_use_the_h273_luma_and_chroma_positions(
    bit_depth: int,
    denominator: float,
    y_min: float,
    y_max: float,
    c_min: float,
    c_max: float,
) -> None:
    """v1-quantize-values acceptance 6 and 7: direction-named functions preserve the former H.273 oracle."""
    legal_positions = np.asarray(
        [
            [[y_min / denominator, c_min / denominator, c_max / denominator]],
            [[y_max / denominator, c_max / denominator, c_min / denominator]],
        ],
        dtype=np.float32,
    )
    full_positions = np.asarray([[[0.0, 0.0, 1.0]], [[1.0, 1.0, 0.0]]], dtype=np.float32)

    full = px.values.legal_to_full(_frame(legal_positions, channels="YCbCr"), bit_depth=bit_depth)
    legal = px.values.full_to_legal(_frame(full_positions, channels="YCbCr"), bit_depth=bit_depth)

    np.testing.assert_allclose(
        px.io.to_array(
            full,
        ).get(),
        full_positions,
        rtol=0.0,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        px.io.to_array(
            legal,
        ).get(),
        legal_positions,
        rtol=0.0,
        atol=2e-7,
    )


def test_range_functions_apply_luma_positions_to_rgb_and_round_trip_overshoot() -> None:
    """v1-quantize-values acceptance 7 and 8: RGB uses luma scale and linear overshoot round-trips."""
    rgb = _frame([16.0 / 255.0, 125.5 / 255.0, 235.0 / 255.0], channels="RGB")
    np.testing.assert_allclose(
        px.io.to_array(
            px.values.legal_to_full(rgb, bit_depth=8),
        ).get(),
        np.asarray([[[0.0, 0.5, 1.0]]], dtype=np.float32),
        rtol=0.0,
        atol=2e-7,
    )

    values = np.asarray([[[-0.25, 1.25, 0.5], [1.5, -0.5, 2.0]]], dtype=np.float32)
    source = _frame(values, channels="YCbCr")
    legal = px.values.full_to_legal(source, bit_depth=14)
    restored = px.values.legal_to_full(legal, bit_depth=14)

    assert float(legal.data[0, 0, 0]) < 0.0
    assert float(legal.data[0, 0, 1]) > 1.0
    assert float(legal.data[0, 1, 2]) > 1.0
    np.testing.assert_allclose(
        px.io.to_array(
            restored,
        ).get(),
        values,
        rtol=0.0,
        atol=5e-7,
    )


@pytest.mark.parametrize("operation", (px.values.legal_to_full, px.values.full_to_legal))
def test_range_functions_default_to_eight_bit_positions(operation: Callable[..., px.core.Frame]) -> None:
    """v1-subpackage-reorg acceptance 4: omitting bit_depth is bit-identical to an explicit value of eight."""
    source = _frame([0.0, 0.5, 1.0], channels="RGB")

    default = operation(source)
    explicit = operation(source, bit_depth=8)

    np.testing.assert_array_equal(
        px.io.to_array(
            default,
        ).get(),
        px.io.to_array(
            explicit,
        ).get(),
    )


@pytest.mark.parametrize("operation", ("legal_to_full", "full_to_legal"))
def test_range_functions_allocate_preserve_metadata_and_reject_undefined_channels(operation: str) -> None:
    """REQ-API-012 / v1-quantize-values acceptance 7: undefined channel ranges fail actionably."""
    source = _frame(
        [0.1, 0.2, 0.3],
        colorspace="ACEScg",
        gamma="linear",
        channels="BGR",
    )
    original = (
        px.io.to_array(
            source,
        )
        .get()
        .copy()
    )
    result = getattr(px.values, operation)(source, bit_depth=10)

    assert result is not source
    assert result.data.data.ptr != source.data.data.ptr
    assert (result.colorspace, result.gamma, result.channels) == ("ACEScg", "linear", ("B", "G", "R"))
    np.testing.assert_array_equal(
        px.io.to_array(
            source,
        ).get(),
        original,
    )

    invalid = _frame([0.0, 0.0, 0.0], channels=["R", "temperature", "B"])
    with pytest.raises(ValueError, match="channel") as error:
        getattr(px.values, operation)(invalid, bit_depth=10)
    _assert_actionable(error)


@pytest.mark.parametrize(
    ("dtype", "routes"),
    (
        (np.float16, ("cast_dtype",)),
        (np.uint8, ("recode_dtype", "dequantize")),
        (np.uint16, ("recode_dtype", "dequantize")),
    ),
)
@pytest.mark.parametrize("operation", ("legal_to_full", "full_to_legal"))
def test_range_functions_reject_non_float32_with_the_dtype_specific_route(
    operation: str,
    dtype: type[np.generic],
    routes: tuple[str, ...],
) -> None:
    """v1-recode-dtype acceptance 9: fp32 errors prioritize recoding and retain bit-grid guidance."""
    source = _frame([0, 0, 0], dtype=dtype)

    with pytest.raises(ValueError) as error:
        getattr(px.values, operation)(source, bit_depth=8)
    positions = tuple(str(error.value).index(route) for route in routes)
    assert positions == tuple(sorted(positions))


@pytest.mark.parametrize("invalid", (0, 7, 9, 11, 13, 15, 17, 18, "8", True, 10.0))
@pytest.mark.parametrize("operation", ("legal_to_full", "full_to_legal"))
def test_range_functions_reject_bit_depths_outside_the_closed_set(operation: str, invalid: object) -> None:
    """v1-quantize-values acceptance 8: both range directions share the five-value bit-depth domain."""
    source = _frame([0.0, 0.5, 1.0], channels="RGB")

    with pytest.raises(ValueError, match=r"expected one of \(8, 10, 12, 14, 16\)"):
        getattr(px.values, operation)(source, bit_depth=invalid)


@pytest.mark.parametrize(
    "operation",
    (
        pytest.param(lambda: px.values.quantize(object(), bit_depth=8), id="quantize"),
        pytest.param(lambda: px.values.dequantize(object(), bit_depth=8), id="dequantize"),
        pytest.param(lambda: px.values.legal_to_full(object(), bit_depth=8), id="legal-to-full"),
        pytest.param(lambda: px.values.full_to_legal(object(), bit_depth=8), id="full-to-legal"),
    ),
)
def test_value_operations_reject_non_frames(operation: Callable[[], object]) -> None:
    """REQ-API-012 / v1-quantize-values acceptance 1, 2, and 6: non-Frames fail actionably."""
    with pytest.raises(ValueError, match="Frame") as error:
        operation()
    _assert_actionable(error)
