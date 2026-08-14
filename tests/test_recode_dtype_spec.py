"""Specification tests for the meaning-preserving recode_dtype operation."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest

import pixtreme as px

DTYPES = ("float32", "float16", "uint8", "uint16", "uint32")


def _frame(
    values: Any,
    *,
    dtype: str,
    colorspace: str = "ACEScg",
    gamma: str = "linear",
    channels: str | list[str] | None = None,
) -> px.core.Frame:
    import cupy as cp

    array = np.asarray(values, dtype=dtype)
    if array.ndim == 1:
        array = array.reshape(1, 1, -1)
    labels = channels if channels is not None else [f"channel-{index}" for index in range(array.shape[2])]
    return px.io.from_array(
        cp.asarray(array),
        colorspace=colorspace,
        gamma=gamma,
        channels=labels,
    )


def _host(frame: px.core.Frame) -> np.ndarray:
    return px.io.to_array(
        frame,
    ).get()


def test_recode_dtype_is_public_with_the_exact_dtype_signature() -> None:
    """v1-recode-dtype acceptance 1 and 11: the public operation has one required keyword-only dtype claim."""
    operation = getattr(px.values, "recode_dtype")
    signature = inspect.signature(operation)

    assert "recode_dtype" in px.values.__all__
    assert tuple(signature.parameters) == ("frame", "dtype")
    assert signature.parameters["frame"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert signature.parameters["dtype"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["dtype"].default is inspect.Parameter.empty


@pytest.mark.parametrize("invalid", ("fp32", "float64", "int32", "UINT8", None, 1, True, ("float32",)))
def test_recode_dtype_rejects_values_outside_the_five_token_closed_set(invalid: object) -> None:
    """v1-exr-runtime-independence acceptance 9: dtype accepts exactly five case-sensitive string tokens."""
    with pytest.raises(ValueError, match=r"float32.*float16.*uint8.*uint16.*uint32"):
        getattr(px.values, "recode_dtype")(_frame([0.0, 0.5, 1.0], dtype="float32"), dtype=invalid)


def test_recode_dtype_rejects_non_frames() -> None:
    """v1-recode-dtype acceptance 1: the operation requires a Frame input."""
    with pytest.raises(ValueError, match="Frame"):
        getattr(px.values, "recode_dtype")(object(), dtype="float32")


@pytest.mark.parametrize(
    ("source_dtype", "target_dtype", "maximum", "codes"),
    (
        ("uint8", "float32", 255, [0, 1, 127, 128, 254, 255]),
        ("uint8", "float16", 255, [0, 1, 127, 128, 254, 255]),
        ("uint16", "float32", 65535, [0, 1, 32767, 32768, 65534, 65535]),
        ("uint16", "float16", 65535, [0, 1, 32767, 32768, 65504, 65535]),
        ("uint32", "float32", 4294967295, [0, 1, 16777216, 16777217, 2147483648, 4294967295]),
        ("uint32", "float16", 4294967295, [0, 1, 16777216, 2147483648, 4292870144, 4294967295]),
    ),
)
def test_uint_to_float_maps_the_complete_container_code_range(
    source_dtype: str,
    target_dtype: str,
    maximum: int,
    codes: list[int],
) -> None:
    """v1-exr-runtime-independence acceptance 9: uint codes divide by their full container maximum."""
    source_codes = np.asarray(codes, dtype=source_dtype)
    expected = (source_codes.astype(np.float32) * np.float32(1.0 / maximum)).astype(target_dtype)

    result = getattr(px.values, "recode_dtype")(_frame(source_codes, dtype=source_dtype), dtype=target_dtype)

    assert result.dtype == np.dtype(target_dtype)
    np.testing.assert_array_equal(_host(result).reshape(-1), expected)


@pytest.mark.parametrize(("source_dtype", "bit_depth"), (("uint8", 8), ("uint16", 16)))
def test_uint_to_float32_is_bit_identical_to_dequantize_values(source_dtype: str, bit_depth: int) -> None:
    """v1-recode-dtype acceptance 2: the everyday uint-to-fp32 lane equals full-container dequantization."""
    maximum = (1 << bit_depth) - 1
    codes = np.linspace(0, maximum, num=257, dtype=np.uint16).astype(source_dtype)
    source = _frame(codes, dtype=source_dtype, channels=[f"code-{index}" for index in range(codes.size)])

    result = getattr(px.values, "recode_dtype")(source, dtype="float32")
    expected = px.values.dequantize(source, bit_depth=bit_depth)

    np.testing.assert_array_equal(_host(result), _host(expected))


@pytest.mark.parametrize(("target_dtype", "bit_depth"), (("uint8", 8), ("uint16", 16)))
def test_float32_to_uint_clips_scales_and_rounds_half_away_from_zero(
    target_dtype: str,
    bit_depth: int,
) -> None:
    """v1-recode-dtype acceptance 3: float conversion uses full scale and upward half ties after clipping."""
    maximum = (1 << bit_depth) - 1
    values = np.asarray(
        [-1.0, 0.0, 0.5 / maximum, 1.5 / maximum, 0.5, (maximum - 0.5) / maximum, 1.0, 2.0],
        dtype=np.float32,
    )
    expected = np.asarray([0, 0, 1, 2, (maximum + 1) // 2, maximum, maximum, maximum], dtype=target_dtype)

    result = getattr(px.values, "recode_dtype")(_frame(values, dtype="float32"), dtype=target_dtype)

    assert result.dtype == np.dtype(target_dtype)
    np.testing.assert_array_equal(_host(result).reshape(-1), expected)


@pytest.mark.parametrize(("target_dtype", "bit_depth"), (("uint8", 8), ("uint16", 16)))
def test_float32_to_uint_is_bit_identical_to_quantize_values(target_dtype: str, bit_depth: int) -> None:
    """v1-recode-dtype acceptance 3: the everyday fp32-to-uint lane equals full-container quantization."""
    values = np.linspace(-0.25, 1.25, num=1025, dtype=np.float32)
    source = _frame(values, dtype="float32", channels=[f"value-{index}" for index in range(values.size)])

    result = getattr(px.values, "recode_dtype")(source, dtype=target_dtype)
    expected = px.values.quantize(source, bit_depth=bit_depth)

    np.testing.assert_array_equal(_host(result), _host(expected))


def test_float32_to_uint32_clips_scales_and_rounds_half_away_from_zero() -> None:
    """v1-exr-runtime-independence acceptance 9: float to uint32 uses the 4294967295 full-scale grid."""
    maximum = 4294967295
    values = np.asarray([-1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 2.0], dtype=np.float32)
    expected = np.floor(np.clip(values.astype(np.float64), 0.0, 1.0) * maximum + 0.5).astype(np.uint32)

    result = px.values.recode_dtype(_frame(values, dtype="float32"), dtype="uint32")

    assert result.dtype == np.dtype(np.uint32)
    np.testing.assert_array_equal(_host(result).reshape(-1), expected)


@pytest.mark.parametrize(
    ("source_dtype", "target_dtype", "codes"),
    (
        ("uint8", "uint32", [0, 1, 127, 128, 254, 255]),
        ("uint16", "uint32", [0, 1, 32767, 32768, 65534, 65535]),
        ("uint32", "uint8", [0, 1, 8421504, 2147483648, 4286545791, 4294967295]),
        ("uint32", "uint16", [0, 1, 32768, 2147483648, 4294901759, 4294967295]),
    ),
)
def test_uint32_integer_pairs_match_exact_full_scale_host_arithmetic(
    source_dtype: str, target_dtype: str, codes: list[int]
) -> None:
    """v1-exr-runtime-independence acceptance 9: integer recodes use exact full-scale half-up arithmetic."""
    source = np.asarray(codes, dtype=source_dtype)
    source_maximum = int(np.iinfo(source_dtype).max)
    target_maximum = int(np.iinfo(target_dtype).max)
    expected = np.asarray(
        [(int(code) * target_maximum + source_maximum // 2) // source_maximum for code in source],
        dtype=target_dtype,
    )

    result = px.values.recode_dtype(_frame(source, dtype=source_dtype), dtype=target_dtype)

    np.testing.assert_array_equal(_host(result).reshape(-1), expected)


@pytest.mark.parametrize(
    ("target_dtype", "expected"), (("uint8", [0, 0, 128, 255, 255]), ("uint16", [0, 0, 32768, 65535, 65535]))
)
def test_float16_to_uint_preserves_normalized_meaning(target_dtype: str, expected: list[int]) -> None:
    """v1-recode-dtype acceptance 3: float16 values use the same normalized clip-and-scale meaning."""
    source = _frame([-1.0, 0.0, 0.5, 1.0, 2.0], dtype="float16")

    result = getattr(px.values, "recode_dtype")(source, dtype=target_dtype)

    np.testing.assert_array_equal(_host(result).reshape(-1), np.asarray(expected, dtype=target_dtype))


@pytest.mark.parametrize("source_dtype", ("float32", "float16"))
@pytest.mark.parametrize("target_dtype", ("float32", "float16"))
def test_float_to_float_is_bit_identical_to_literal_cast(source_dtype: str, target_dtype: str) -> None:
    """v1-recode-dtype acceptance 4: float pairs apply literal casting with no scale or clipping."""
    source = _frame([-2.0, -0.0, 0.1, 1.0, 7.75], dtype=source_dtype)

    result = getattr(px.values, "recode_dtype")(source, dtype=target_dtype)
    expected = px.values.cast_dtype(source, dtype=target_dtype)

    np.testing.assert_array_equal(_host(result), _host(expected))


def test_uint8_to_uint16_is_exactly_code_times_257_and_round_trips() -> None:
    """v1-recode-dtype acceptance 5: all uint8 codes widen exactly and survive the two-way recode."""
    codes = np.arange(256, dtype=np.uint8)
    source = _frame(codes, dtype="uint8", channels=[f"code-{index}" for index in range(codes.size)])

    widened = getattr(px.values, "recode_dtype")(source, dtype="uint16")
    restored = getattr(px.values, "recode_dtype")(widened, dtype="uint8")

    np.testing.assert_array_equal(_host(widened).reshape(-1), codes.astype(np.uint16) * np.uint16(257))
    np.testing.assert_array_equal(_host(restored), _host(source))


def test_uint16_to_uint8_uses_divide_by_257_equivalent_rounding() -> None:
    """v1-recode-dtype acceptance 5: narrowing follows normalized rescale with half-away rounding."""
    codes = np.asarray([0, 1, 128, 129, 257, 385, 386, 65406, 65407, 65535], dtype=np.uint16)
    expected = np.asarray([0, 0, 0, 1, 1, 1, 2, 254, 255, 255], dtype=np.uint8)

    result = getattr(px.values, "recode_dtype")(_frame(codes, dtype="uint16"), dtype="uint8")

    np.testing.assert_array_equal(_host(result).reshape(-1), expected)


@pytest.mark.parametrize("source_dtype", DTYPES)
@pytest.mark.parametrize("target_dtype", DTYPES)
def test_every_dtype_pair_returns_private_storage_and_preserves_metadata(
    source_dtype: str,
    target_dtype: str,
) -> None:
    """v1-recode-dtype acceptance 6; v1-exr-runtime-independence acceptance 9.

    All 25 pairs allocate private storage and preserve metadata and input.
    """
    values = [0, 1, 2] if source_dtype.startswith("uint") else [-0.25, 0.5, 1.25]
    source = _frame(
        values,
        dtype=source_dtype,
        colorspace="Rec.2020",
        gamma="pq",
        channels=["signal-0", "signal-1", "signal-2"],
    )
    original = _host(source).copy()

    result = getattr(px.values, "recode_dtype")(source, dtype=target_dtype)

    assert result is not source
    assert result.data.data.ptr != source.data.data.ptr
    assert (result.colorspace, result.gamma, result.channels) == (
        source.colorspace,
        source.gamma,
        source.channels,
    )
    np.testing.assert_array_equal(_host(source), original)
