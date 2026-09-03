"""Characterization tests for the recode_dtype RawKernel trial."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import pixtreme as px
import pixtreme._values.cast as cast_module

_DTYPES = ("float32", "float16", "uint8", "uint16", "uint32")
_CONTAINER_MAXIMA = {"uint8": 255, "uint16": 65535, "uint32": 4294967295}


def _frame(values: Any, *, dtype: str) -> px.core.Frame:
    import cupy as cp

    array = cp.asarray(np.asarray(values, dtype=dtype).reshape(1, -1, 1))
    return px.io.from_array(array, colorspace="ACEScg", gamma="linear", channels=("Y",))


def _legacy_recode_dtype_data(frame: px.core.Frame, *, dtype: str) -> Any:
    """Retain the pre-trial CuPy composition as the comparison oracle."""
    import cupy as cp

    source_dtype = frame.dtype.name
    if source_dtype == dtype or (source_dtype.startswith("float") and dtype.startswith("float")):
        return frame.data.astype(dtype)

    if source_dtype.startswith("uint") and dtype.startswith("uint"):
        source_maximum = _CONTAINER_MAXIMA[source_dtype]
        target_maximum = _CONTAINER_MAXIMA[dtype]
        source = frame.data.astype(cp.uint64)
        scaled = (source * np.uint64(target_maximum) + np.uint64(source_maximum // 2)) // np.uint64(source_maximum)
        return scaled.astype(dtype)

    if source_dtype.startswith("uint"):
        source_maximum = _CONTAINER_MAXIMA[source_dtype]
        return (frame.data.astype(cp.float32) * np.float32(1.0 / source_maximum)).astype(dtype)

    normalized = frame.data if source_dtype == "float32" else frame.data.astype(cp.float32)
    if dtype == "uint32":
        maximum = np.float64(_CONTAINER_MAXIMA[dtype])
        return cp.floor(cp.clip(normalized.astype(cp.float64), 0.0, 1.0) * maximum + np.float64(0.5)).astype(cp.uint32)
    return px.values.quantize(
        px.io.from_array(normalized, colorspace=frame.colorspace, gamma=frame.gamma, channels=frame.channels),
        bit_depth=8 if dtype == "uint8" else 16,
    ).data


def _trial_values(source_dtype: str, target_dtype: str) -> np.ndarray:
    generator = np.random.default_rng(20260819)
    if source_dtype.startswith("uint"):
        maximum = _CONTAINER_MAXIMA[source_dtype]
        boundaries = np.asarray(
            [
                0,
                1,
                maximum // 2,
                maximum // 2 + 1,
                maximum - 1,
                maximum,
            ],
            dtype=source_dtype,
        )
        random_values = generator.integers(0, maximum + 1, size=2048, dtype=np.uint64).astype(source_dtype)
        return np.concatenate((boundaries, random_values))

    boundaries = [
        -np.inf,
        -1.0,
        -0.0,
        0.0,
        0.5,
        1.0,
        np.nextafter(np.float32(1.0), np.float32(2.0)),
        2.0,
        np.inf,
        np.nan,
    ]
    if target_dtype.startswith("uint"):
        maximum = _CONTAINER_MAXIMA[target_dtype]
        for numerator in (0.5, 1.5, maximum - 0.5):
            boundary = np.float32(numerator / maximum)
            boundaries.extend(
                (
                    np.nextafter(boundary, np.float32(-np.inf)),
                    boundary,
                    np.nextafter(boundary, np.float32(np.inf)),
                )
            )
    random_values = generator.uniform(-0.25, 1.25, size=2048).astype(source_dtype)
    return np.concatenate((np.asarray(boundaries, dtype=source_dtype), random_values))


@pytest.mark.parametrize("source_dtype", _DTYPES)
@pytest.mark.parametrize("target_dtype", _DTYPES)
def test_rawkernel_trial_matches_legacy_boundary_and_random_bits_characterization(
    source_dtype: str,
    target_dtype: str,
) -> None:
    """characterization: issue #1 freezes the pre-trial recode_dtype bits for RawKernel comparison.

    The CuPy composition is retained above because the optimization trial must preserve its output exactly;
    this characterization can be replaced by specification or removed when issue #1 closes the trial.
    """
    import cupy as cp

    source = _frame(_trial_values(source_dtype, target_dtype), dtype=source_dtype)
    expected = _legacy_recode_dtype_data(source, dtype=target_dtype)
    candidate = getattr(cast_module, "_recode_dtype_rawkernel")(source, dtype=target_dtype)

    assert candidate.dtype == np.dtype(target_dtype)
    cp.testing.assert_array_equal(candidate.data, expected)
