"""Specification tests for automatic dtype conversion at image write boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import cupy as cp
import numpy as np
import pytest

import pixtreme as px

_DTYPES = (np.uint8, np.uint16, np.uint32, np.float16, np.float32)
_NATIVE_DTYPES = {
    "PNG": frozenset(("uint8", "uint16")),
    "JPEG": frozenset(("uint8",)),
    "TIFF": frozenset(("uint8", "uint16")),
    "JPEG2000": frozenset(("uint8", "uint16")),
    "WEBP": frozenset(("uint8",)),
    "BMP": frozenset(("uint8",)),
    "PNM": frozenset(("uint8", "uint16")),
    "EXR": frozenset(("float16", "float32", "uint32")),
}
_DEFAULT_DTYPES = {
    "PNG": "uint8",
    "JPEG": "uint8",
    "TIFF": "uint8",
    "JPEG2000": "uint8",
    "WEBP": "uint8",
    "BMP": "uint8",
    "PNM": "uint8",
    "EXR": "float16",
}


def _source_values(dtype: type[np.generic]) -> np.ndarray:
    if np.issubdtype(dtype, np.floating):
        samples = np.asarray(
            (-0.25, 0.0, 0.5 / 255.0, 1.5 / 255.0, 0.25, 0.5, 254.5 / 255.0, 1.0, 1.25),
            dtype=dtype,
        )
    else:
        maximum = np.iinfo(dtype).max
        samples = np.asarray((0, 1, maximum // 4, maximum // 2, maximum - 1, maximum), dtype=dtype)
    return np.resize(samples, 16 * 16 * 3).reshape(16, 16, 3)


def _host_recode(values: np.ndarray, target_dtype: str) -> np.ndarray:
    source_dtype = values.dtype
    if source_dtype.name == target_dtype:
        return values.copy()
    if np.issubdtype(source_dtype, np.floating) and target_dtype.startswith("float"):
        return values.astype(target_dtype)
    if np.issubdtype(source_dtype, np.integer) and target_dtype.startswith("uint"):
        source_maximum = int(np.iinfo(source_dtype).max)
        target_maximum = int(np.iinfo(target_dtype).max)
        numerator = values.astype(np.uint64) * np.uint64(target_maximum) + np.uint64(source_maximum // 2)
        return (numerator // np.uint64(source_maximum)).astype(target_dtype)
    if np.issubdtype(source_dtype, np.integer):
        normalized = values.astype(np.float32) * np.float32(1.0 / int(np.iinfo(source_dtype).max))
    else:
        normalized = values.astype(np.float32)
    if target_dtype.startswith("float"):
        return normalized.astype(target_dtype)
    if target_dtype == "uint32":
        target_maximum = np.float64(np.iinfo(target_dtype).max)
        scaled = np.clip(normalized.astype(np.float64), 0.0, 1.0) * target_maximum
        return np.floor(scaled + 0.5).astype(target_dtype)
    target_maximum = np.float32(np.iinfo(target_dtype).max)
    scaled = np.clip(normalized, np.float32(0.0), np.float32(1.0)) * target_maximum
    return np.floor(scaled + np.float32(0.5)).astype(target_dtype)


@pytest.mark.parametrize(
    ("route", "format_name", "format_token", "suffix", "kwargs"),
    (
        ("encode", "PNG", "png", ".png", {}),
        ("write", "PNG", None, ".png", {}),
        ("encode", "JPEG", "jpeg", ".jpg", {"quality": 100}),
        ("write", "JPEG", None, ".jpg", {"quality": 100}),
        ("encode", "TIFF", "tiff", ".tiff", {"compression": "lzw"}),
        ("write", "TIFF", None, ".tiff", {"compression": "lzw"}),
        ("encode", "JPEG2000", "jpeg2000", ".jp2", {"lossless": True}),
        ("write", "JPEG2000", None, ".jp2", {"lossless": True}),
        ("encode", "WEBP", "webp", ".webp", {"lossless": True}),
        ("write", "WEBP", None, ".webp", {"lossless": True}),
        ("encode", "BMP", "bmp", ".bmp", {}),
        ("write", "BMP", None, ".bmp", {}),
        ("encode", "PNM", "pnm", ".pnm", {}),
        ("write", "PNM", None, ".pnm", {}),
        ("write", "EXR", None, ".exr", {}),
    ),
)
@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda value: np.dtype(value).name)
def test_write_boundaries_accept_every_frame_dtype_and_preserve_the_input(
    tmp_path: Path,
    route: Literal["encode", "write"],
    format_name: str,
    format_token: str | None,
    suffix: str,
    kwargs: dict[str, Any],
    dtype: type[np.generic],
) -> None:
    """v1-write-dtype-convert acceptance 1-5 and 7; v1-bytes-boundary acceptance 6;
    v1-exr-runtime-independence acceptance 44: every boundary recodes independently without mutation.
    """
    values = _source_values(dtype)
    frame = px.io.from_array(cp.asarray(values), colorspace="ACEScg", gamma="linear", channels="RGB")
    before = frame.data.copy()
    before_metadata = (frame.colorspace, frame.gamma, frame.channels, frame.matrix)
    if format_name == "EXR":
        expected_dtype = "uint32" if frame.dtype.name == "uint32" else "float16"
    else:
        expected_dtype = (
            frame.dtype.name if frame.dtype.name in _NATIVE_DTYPES[format_name] else _DEFAULT_DTYPES[format_name]
        )
    expected = _host_recode(values, expected_dtype)

    if route == "encode":
        assert format_token is not None
        payload = px.io.encode_image(frame, format=format_token, **kwargs)
        decoded = px.io.decode_image(payload, unchanged=True, colorspace="ACEScg", gamma="linear")
    else:
        path = tmp_path / f"{format_name.lower()}-{frame.dtype.name}{suffix}"
        assert px.io.write_image(path, frame, **kwargs) is None
        decoded = px.io.read_image(path, unchanged=True, colorspace="ACEScg", gamma="linear")

    assert (decoded.dtype.name, decoded.shape, decoded.channels) == (expected_dtype, frame.shape, frame.channels)
    if format_name != "JPEG":
        np.testing.assert_array_equal(
            px.io.to_array(
                decoded,
            ).get(),
            expected,
        )
    assert frame.dtype == np.dtype(dtype)
    cp.testing.assert_array_equal(frame.data, before)
    assert (frame.colorspace, frame.gamma, frame.channels, frame.matrix) == before_metadata


@pytest.mark.parametrize(
    ("format_token", "suffix", "dtype", "kwargs"),
    (
        ("png", None, np.uint16, {}),
        ("jpeg", None, np.uint8, {"quality": 100}),
        ("tiff", None, np.uint16, {"compression": "lzw"}),
        ("jpeg2000", None, np.uint16, {"lossless": True}),
        ("webp", None, np.uint8, {"lossless": True}),
        ("bmp", None, np.uint8, {}),
        ("pnm", None, np.uint16, {}),
        (None, ".exr", np.float16, {}),
        (None, ".tga", np.uint8, {}),
    ),
)
def test_native_dtype_selection_does_not_call_recode_dtype(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    format_token: str | None,
    suffix: str | None,
    dtype: type[np.generic],
    kwargs: dict[str, Any],
) -> None:
    """v1-write-dtype-convert acceptance 2: every native format input bypasses numeric dtype conversion."""
    values = _source_values(dtype)
    frame = px.io.from_array(cp.asarray(values), colorspace="ACEScg", gamma="linear", channels="RGB")

    def fail_recode(*args: object, **kwargs: object) -> px.core.Frame:
        raise AssertionError("native dtype must not be recoded")

    monkeypatch.setattr("pixtreme._values.cast.recode_dtype", fail_recode)

    if format_token is not None:
        assert isinstance(px.io.encode_image(frame, format=format_token, **kwargs), bytes)
    else:
        assert suffix is not None
        assert px.io.write_image(tmp_path / f"native{suffix}", frame, **kwargs) is None


def test_write_image_preserves_the_unwritable_output_path_error(tmp_path: Path) -> None:
    """v1-write-dtype-convert acceptance 5: an unwritable output path stays an actionable RuntimeError."""
    frame = px.io.from_array(
        cp.zeros((1, 1, 3), dtype=cp.uint8),
        colorspace="sRGB",
        gamma="sRGB",
        channels="RGB",
    )
    output = tmp_path / "missing-parent" / "output.png"

    with pytest.raises(RuntimeError, match=r"why=.*what=.*how=") as error:
        px.io.write_image(output, frame)

    assert isinstance(error.value.__cause__, OSError)
    assert not output.exists()
    assert not output.parent.exists()
