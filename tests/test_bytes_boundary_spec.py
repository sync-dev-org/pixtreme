"""Specification tests for encoded-bytes image boundaries."""

from __future__ import annotations

import inspect
import io
import re
from pathlib import Path

import cupy as cp
import numpy as np
import pytest
from PIL import Image

import pixtreme as px

_ACTIONABLE = r"why=.*what=.*how="
_FORMAT_NAMES = ("JPEG 2000", "JPEG", "PNG", "TIFF", "WebP", "BMP", "PNM", "EXR", "TGA", "HDR", "DPX")
_FORMAT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(" + "|".join(re.escape(name) for name in _FORMAT_NAMES) + r")(?![A-Za-z0-9])"
)
_VALUES = np.array(
    [
        [[0, 17, 255], [31, 127, 223], [5, 99, 201]],
        [[255, 33, 1], [64, 128, 192], [240, 160, 80]],
    ],
    dtype=np.uint8,
)


def _frame(values: np.ndarray = _VALUES) -> px.core.Frame:
    return px.io.from_array(cp.asarray(values), colorspace="sRGB", gamma="srgb", channels="RGB")


def _save_fixture(tmp_path: Path, format_token: str, values: np.ndarray = _VALUES) -> Path:
    suffix = {"jpeg": ".jpg", "png": ".png", "tiff": ".tiff"}[format_token]
    path = tmp_path / f"fixture{suffix}"
    Image.fromarray(values).save(path)
    return path


def _assert_frames_equal(actual: px.core.Frame, expected: px.core.Frame) -> None:
    assert (actual.colorspace, actual.gamma, actual.channels, actual.dtype) == (
        expected.colorspace,
        expected.gamma,
        expected.channels,
        expected.dtype,
    )
    cp.testing.assert_array_equal(actual.data, expected.data)


def _documented_formats(text: str) -> set[str]:
    return {match.group(1) for match in _FORMAT_PATTERN.finditer(text)}


def _paragraph_containing(docstring: str, fragment: str) -> str:
    matches = [paragraph for paragraph in docstring.split("\n\n") if fragment in paragraph]
    assert len(matches) == 1
    return matches[0]


def _sentence_containing(docstring: str, fragment: str) -> str:
    normalized = " ".join(docstring.split())
    matches = [sentence for sentence in normalized.split(". ") if fragment in sentence]
    assert len(matches) == 1
    return matches[0]


def test_bytes_boundary_public_signatures_are_exact() -> None:
    """v1-bytes-boundary acceptance 1, 5, and 14: both public APIs expose the fixed keyword grammar."""
    decode = inspect.signature(px.io.decode_image)
    assert tuple(decode.parameters) == ("data", "channels", "unchanged", "colorspace", "gamma")
    assert decode.parameters["data"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name, default in (("channels", None), ("unchanged", False), ("colorspace", None), ("gamma", None)):
        assert decode.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert decode.parameters[name].default is default

    encode = inspect.signature(px.io.encode_image)
    assert tuple(encode.parameters) == (
        "frame",
        "format",
        "quality",
        "compression",
        "compression_level",
        "lossless",
    )
    assert encode.parameters["frame"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert encode.parameters["format"].kind is inspect.Parameter.KEYWORD_ONLY
    assert encode.parameters["format"].default is inspect.Parameter.empty
    for name in ("quality", "compression", "compression_level", "lossless"):
        assert encode.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert encode.parameters[name].default is None


@pytest.mark.parametrize("format_token", ("jpeg", "png", "tiff"))
def test_decode_image_matches_the_file_boundary(tmp_path: Path, format_token: str) -> None:
    """v1-bytes-boundary acceptance 2: sniffed bytes and file decoding return identical pixels and metadata."""
    path = _save_fixture(tmp_path, format_token)

    decoded = px.io.decode_image(path.read_bytes())
    from_file = px.io.read_image(path)

    _assert_frames_equal(decoded, from_file)


@pytest.mark.parametrize(
    "payload",
    (
        b"RIFF\x0c\x00\x00\x00WEBPVP8 \x00\x00\x00\x00",
        b"not an encoded image",
        b"",
    ),
)
def test_decode_image_rejects_recognizable_unsupported_and_unknown_bytes(payload: bytes) -> None:
    """v1-bytes-boundary acceptance 3: unsupported and unidentifiable bytes fail with actionable context."""
    with pytest.raises(ValueError, match=_ACTIONABLE):
        px.io.decode_image(payload)


def test_read_and_decode_docstrings_state_the_file_only_and_bytes_boundaries() -> None:
    """REQ-API-005 and v1-bytes-boundary acceptance 1-4: both image inputs enumerate their format boundary."""
    read_docstring = inspect.getdoc(px.io.read_image) or ""
    decode_docstring = inspect.getdoc(px.io.decode_image) or ""
    raster_formats = {"JPEG", "PNG", "TIFF", "JPEG 2000", "WebP", "BMP", "PNM"}
    file_only_formats = {"EXR", "TGA", "HDR", "DPX"}

    assert _documented_formats(_paragraph_containing(read_docstring, "``path`` selects")) == (
        raster_formats | file_only_formats
    )
    assert _documented_formats(_paragraph_containing(decode_docstring, "``data`` accepts")) == (
        raster_formats | file_only_formats
    )
    assert _documented_formats(_sentence_containing(read_docstring, "file-only")) == file_only_formats
    assert _documented_formats(_sentence_containing(decode_docstring, "file-only")) == file_only_formats
    assert set(re.findall(r":class:`([^`]+)`", read_docstring)) == {
        "FileNotFoundError",
        "ValueError",
        "RuntimeError",
    }
    assert set(re.findall(r":class:`([^`]+)`", decode_docstring)) == {"ValueError", "RuntimeError"}


def test_decode_image_rejects_malformed_headers_with_value_error() -> None:
    """REQ-API-005 and v1-bytes-boundary acceptance 2: a recognized but malformed header is invalid input, not a codec failure."""
    truncated_png = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR"
    with pytest.raises(ValueError):
        px.io.decode_image(truncated_png)


def test_decode_image_metadata_overrides_and_native_depth_match_read_image(tmp_path: Path) -> None:
    """v1-bytes-boundary acceptance 4: raster defaults, per-call claims, and unchanged depth mirror file I/O."""
    values = np.array([[0, 1, 32768, 65535]], dtype=np.uint16)
    path = tmp_path / "gray16.png"
    Image.fromarray(values).save(path)
    payload = path.read_bytes()

    decoded = px.io.decode_image(payload, unchanged=True, colorspace="ACEScg", gamma="linear")
    from_file = px.io.read_image(path, unchanged=True, colorspace="ACEScg", gamma="linear")

    _assert_frames_equal(decoded, from_file)
    assert decoded.dtype == np.dtype(np.uint16)
    assert (decoded.colorspace, decoded.gamma) == ("ACEScg", "linear")


@pytest.mark.parametrize("format_token", ("jpg", "JPEG", "jpeg2k", None, 1))
def test_encode_image_requires_a_supported_format_token(format_token: object) -> None:
    """v1-io-formats acceptance 3: format remains required and its extended typed token set fails fast."""
    frame = _frame()
    with pytest.raises(ValueError, match=_ACTIONABLE):
        px.io.encode_image(frame, format=format_token)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        px.io.encode_image(frame)  # type: ignore[call-arg]


@pytest.mark.parametrize("quality", (0, 101, True, 90.0))
def test_jpeg_quality_is_typed_bounded_and_format_specific(quality: object) -> None:
    """v1-bytes-boundary acceptance 7: JPEG quality alone accepts exact integers from 1 through 100."""
    frame = _frame()
    with pytest.raises(ValueError, match=_ACTIONABLE):
        px.io.encode_image(frame, format="jpeg", quality=quality)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=_ACTIONABLE):
        px.io.encode_image(frame, format="png", quality=90)

    assert isinstance(px.io.encode_image(frame, format="jpeg", quality=90), bytes)


@pytest.mark.parametrize("compression", ("zip", "LZW", 1))
def test_tiff_compression_tokens_are_closed_and_format_specific(compression: object) -> None:
    """v1-bytes-boundary acceptance 8: TIFF compression accepts only the case-sensitive none/lzw tokens."""
    frame = _frame()
    with pytest.raises(ValueError, match=_ACTIONABLE):
        px.io.encode_image(frame, format="tiff", compression=compression)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=_ACTIONABLE):
        px.io.encode_image(frame, format="png", compression="lzw")


@pytest.mark.parametrize("compression", ("none", "lzw"))
def test_tiff_compression_round_trip_is_bit_exact(compression: str) -> None:
    """v1-bytes-boundary acceptance 8: both TIFF compression tokens preserve pixels across the bytes boundary."""
    frame = _frame()
    decoded = px.io.decode_image(px.io.encode_image(frame, format="tiff", compression=compression), unchanged=True)
    _assert_frames_equal(decoded, frame)


@pytest.mark.parametrize("compression_level", range(10))
def test_png_compression_levels_round_trip_bit_exactly(compression_level: int) -> None:
    """v1-bytes-boundary acceptance 9: every PNG zlib level 0 through 9 is lossless."""
    frame = _frame()
    decoded = px.io.decode_image(
        px.io.encode_image(frame, format="png", compression_level=compression_level),
        unchanged=True,
    )
    _assert_frames_equal(decoded, frame)


@pytest.mark.parametrize("compression_level", (-1, 10, True, 1.0, "1"))
def test_png_compression_level_is_typed_bounded_and_format_specific(compression_level: object) -> None:
    """v1-bytes-boundary acceptance 9: PNG compression levels fail fast outside the exact integer domain."""
    frame = _frame()
    with pytest.raises(ValueError, match=_ACTIONABLE):
        px.io.encode_image(frame, format="png", compression_level=compression_level)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=_ACTIONABLE):
        px.io.encode_image(frame, format="tiff", compression_level=1)


@pytest.mark.parametrize(
    ("format_token", "suffix", "kwargs"),
    (
        ("jpeg", ".jpg", {"quality": 93}),
        ("png", ".png", {"compression_level": 4}),
        ("tiff", ".tiff", {"compression": "lzw"}),
    ),
)
def test_bytes_and_file_encode_round_trips_match(
    tmp_path: Path,
    format_token: str,
    suffix: str,
    kwargs: dict[str, object],
) -> None:
    """v1-bytes-boundary acceptance 10-11: bytes and file output share pixels, metadata, and encode parameters."""
    frame = _frame()
    path = tmp_path / f"round-trip{suffix}"

    payload = px.io.encode_image(frame, format=format_token, **kwargs)  # type: ignore[arg-type]
    px.io.write_image(path, frame, **kwargs)  # type: ignore[arg-type]

    assert isinstance(payload, bytes)
    _assert_frames_equal(px.io.decode_image(payload, unchanged=True), px.io.read_image(path, unchanged=True))


def test_write_image_new_encode_parameters_share_fail_fast_validation(tmp_path: Path) -> None:
    """v1-bytes-boundary acceptance 11: file output enforces the same parameter/format matrix as bytes output."""
    frame = _frame()
    invalid = (
        (tmp_path / "quality.png", {"quality": 90}),
        (tmp_path / "compression.jpg", {"compression": "lzw"}),
        (tmp_path / "compression.tiff", {"compression": "zip"}),
        (tmp_path / "level.tiff", {"compression_level": 4}),
        (tmp_path / "level.png", {"compression_level": 10}),
    )
    for path, kwargs in invalid:
        with pytest.raises(ValueError, match=_ACTIONABLE):
            px.io.write_image(path, frame, **kwargs)  # type: ignore[arg-type]


def test_requirements_marks_the_encoded_bytes_boundary_as_implemented() -> None:
    """v1-bytes-boundary acceptance 13: REQ-API-010 records the two implemented bytes APIs."""
    requirements_path = Path(__file__).resolve().parents[1] / "docs" / "requirements.md"
    if not requirements_path.is_file():
        pytest.skip("repo-only documentation contract: docs/requirements.md is absent from this distribution")
    requirements = requirements_path.read_text(encoding="utf-8")
    assert "| bytes (encoded) | `px.io.decode_image` | `px.io.encode_image` |" in requirements
    assert "`px.io.decode_image` (名前予約)" not in requirements
    assert "`px.io.encode_image` (名前予約)" not in requirements


@pytest.mark.parametrize(("compression", "expected_tag"), (("none", 1), ("lzw", 5)))
def test_tiff_compression_token_controls_bytes_output_encoding(compression: str, expected_tag: int) -> None:
    """v1-bytes-boundary acceptance 8: TIFF compression tokens select the container scheme in the emitted bytes.

    Independent Pillow oracle over TIFF tag 259 (Compression); fails if the option
    is silently ignored, which lossless round-trip assertions cannot detect.
    """
    payload = px.io.encode_image(_frame(), format="tiff", compression=compression)
    with Image.open(io.BytesIO(payload)) as image:
        assert image.tag_v2[259] == expected_tag


@pytest.mark.parametrize(("compression", "expected_tag"), (("none", 1), ("lzw", 5)))
def test_tiff_compression_token_controls_file_output_encoding(
    tmp_path: Path, compression: str, expected_tag: int
) -> None:
    """v1-bytes-boundary acceptance 11: write_image applies the same TIFF compression scheme as encode_image."""
    path = tmp_path / "compression.tiff"
    px.io.write_image(path, _frame(), compression=compression)
    with Image.open(path) as image:
        assert image.tag_v2[259] == expected_tag


def test_png_compression_level_controls_bytes_output_size() -> None:
    """v1-bytes-boundary acceptance 9: zlib levels materially change the emitted payload on compressible content.

    Level 0 stores uncompressed deflate blocks, so a constant image must shrink at
    level 9; fails if compression_level is silently ignored.
    """
    frame = _frame(np.zeros((64, 64, 3), dtype=np.uint8))
    fastest = px.io.encode_image(frame, format="png", compression_level=0)
    smallest = px.io.encode_image(frame, format="png", compression_level=9)
    assert len(smallest) < len(fastest)


def test_png_compression_level_controls_file_output_size(tmp_path: Path) -> None:
    """v1-bytes-boundary acceptance 11: write_image applies the same PNG compression levels as encode_image."""
    frame = _frame(np.zeros((64, 64, 3), dtype=np.uint8))
    fastest_path = tmp_path / "level0.png"
    smallest_path = tmp_path / "level9.png"
    px.io.write_image(fastest_path, frame, compression_level=0)
    px.io.write_image(smallest_path, frame, compression_level=9)
    assert smallest_path.stat().st_size < fastest_path.stat().st_size
