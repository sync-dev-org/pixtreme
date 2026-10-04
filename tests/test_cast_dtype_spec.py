"""Specification tests for the literal-storage cast_dtype operation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import pixtreme as px


def _assert_actionable(error: pytest.ExceptionInfo[ValueError]) -> None:
    message = str(error.value)
    assert "why=" in message
    assert "; what=" in message
    assert "; how=" in message


def _frame(values: Any, *, dtype: str) -> px.core.Frame:
    import cupy as cp

    array = np.asarray(values, dtype=dtype).reshape(1, 1, -1)
    labels = [f"channel-{index}" for index in range(array.shape[2])]
    return px.io.from_array(cp.asarray(array), colorspace="ACEScg", gamma="linear", channels=labels)


@pytest.mark.req("REQ-PIX-008")
@pytest.mark.parametrize("source_dtype", ("float32", "float16", "uint8", "uint16", "uint32"))
@pytest.mark.parametrize("target_dtype", ("float32", "float16", "uint8", "uint16", "uint32"))
def test_cast_dtype_matches_literal_astype_for_every_frame_dtype_pair(
    source_dtype: str,
    target_dtype: str,
) -> None:
    """Casting between every supported pair of Frame dtypes follows literal CuPy astype numeric semantics."""
    values = [0, 1, 2, 7] if source_dtype.startswith("uint") else [0.0, 1.0, 2.0, 7.75]
    source = _frame(values, dtype=source_dtype)
    expected = np.asarray(values, dtype=source_dtype).astype(target_dtype)

    result = px.values.cast_dtype(source, dtype=target_dtype)

    assert result.dtype == np.dtype(target_dtype)
    np.testing.assert_array_equal(
        px.io.to_array(
            result,
        )
        .get()
        .reshape(-1),
        expected,
    )


@pytest.mark.req("REQ-PIX-002")
@pytest.mark.req("REQ-PIX-008")
@pytest.mark.parametrize("dtype", ("float32", "float16", "uint8", "uint16", "uint32"))
def test_cast_dtype_always_allocates_and_preserves_metadata(dtype: str) -> None:
    """Casting a Frame to its existing dtype still allocates separate storage and preserves its color metadata."""
    source = _frame([0, 1, 2], dtype=dtype)

    result = px.values.cast_dtype(source, dtype=dtype)

    assert result is not source
    assert result.data.data.ptr != source.data.data.ptr
    assert (result.colorspace, result.gamma, result.channels) == (
        source.colorspace,
        source.gamma,
        source.channels,
    )


@pytest.mark.req("REQ-PIX-008")
@pytest.mark.req("REQ-PIX-017")
@pytest.mark.parametrize("invalid", ("fp32", "float64", "int32", "uint64", "unknown"))
def test_cast_dtype_rejects_unknown_tokens(invalid: str) -> None:
    """Casting a Frame rejects unsupported dtype names with an error that lists valid choices."""
    with pytest.raises(ValueError, match="dtype") as error:
        px.values.cast_dtype(_frame([0, 1, 2], dtype="float32"), dtype=invalid)
    _assert_actionable(error)
