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


@pytest.mark.parametrize("source_dtype", ("float32", "float16", "uint8", "uint16", "uint32"))
@pytest.mark.parametrize("target_dtype", ("float32", "float16", "uint8", "uint16", "uint32"))
def test_cast_dtype_matches_literal_astype_for_every_frame_dtype_pair(
    source_dtype: str,
    target_dtype: str,
) -> None:
    """v1-exr-runtime-independence acceptance 8: all 25 cast pairs use literal CuPy astype semantics."""
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


@pytest.mark.parametrize("dtype", ("float32", "float16", "uint8", "uint16", "uint32"))
def test_cast_dtype_always_allocates_and_preserves_metadata(dtype: str) -> None:
    """v1-exr-runtime-independence acceptance 8: same-dtype cast allocates and preserves metadata."""
    source = _frame([0, 1, 2], dtype=dtype)

    result = px.values.cast_dtype(source, dtype=dtype)

    assert result is not source
    assert result.data.data.ptr != source.data.data.ptr
    assert (result.colorspace, result.gamma, result.channels) == (
        source.colorspace,
        source.gamma,
        source.channels,
    )


@pytest.mark.parametrize("invalid", ("fp32", "float64", "int32", "uint64", "unknown"))
def test_cast_dtype_rejects_unknown_tokens(invalid: str) -> None:
    """REQ-API-012 / v1-io acceptance 19; v1-token-vocabulary acceptance 7: invalid dtype tokens fail actionably."""
    with pytest.raises(ValueError, match="dtype") as error:
        px.values.cast_dtype(_frame([0, 1, 2], dtype="float32"), dtype=invalid)
    _assert_actionable(error)
