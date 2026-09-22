"""Specification tests for raster EXIF orientation at file, bytes, and header boundaries."""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
import warnings
from pathlib import Path

import cupy as cp
import numpy as np
import pytest
from generate_io_fixtures import (
    encode_lossless_oriented_webp,
    encode_oriented_raster,
    encode_plain_raster,
    exif_payload,
    orientation_pattern,
    png_with_exif,
)

import pixtreme as px

ROOT = Path(__file__).resolve().parents[1]
_SUFFIXES = {"JPEG": ".jpg", "PNG": ".png", "TIFF": ".tiff", "WEBP": ".webp"}
_TARGET_FORMATS = tuple(_SUFFIXES)
_ACTIONABLE = r"^why=.*; what=.*; how=.*"


def _oriented(values: np.ndarray, orientation: int) -> np.ndarray:
    operations = {
        1: lambda value: value,
        2: lambda value: value[:, ::-1],
        3: lambda value: value[::-1, ::-1],
        4: lambda value: value[::-1],
        5: lambda value: value.transpose(1, 0, 2),
        6: lambda value: value[::-1].transpose(1, 0, 2),
        7: lambda value: value[::-1, ::-1].transpose(1, 0, 2),
        8: lambda value: value[:, ::-1].transpose(1, 0, 2),
    }
    return operations[orientation](values)


def _frame_values(frame: px.core.Frame) -> np.ndarray:
    return cp.asnumpy(frame.data)


def _assert_same_frame(actual: px.core.Frame, expected: px.core.Frame) -> None:
    assert (actual.shape, actual.dtype, actual.colorspace, actual.gamma, actual.channels, actual.matrix) == (
        expected.shape,
        expected.dtype,
        expected.colorspace,
        expected.gamma,
        expected.channels,
        expected.matrix,
    )
    cp.testing.assert_array_equal(actual.data, expected.data)


def test_orientation_public_signatures_and_exact_bool_validation_are_fixed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v1-io-orientation acceptance 1: both decode boundaries add the same fail-fast exact-bool keyword."""
    read_signature = inspect.signature(px.io.read_image)
    decode_signature = inspect.signature(px.io.decode_image)

    assert tuple(read_signature.parameters)[-1] == "apply_exif_orientation"
    assert tuple(decode_signature.parameters)[-1] == "apply_exif_orientation"
    for signature in (read_signature, decode_signature):
        parameter = signature.parameters["apply_exif_orientation"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is True

    payload = encode_plain_raster("PNG")
    path = tmp_path / "valid.png"
    path.write_bytes(payload)

    def forbidden_decode(*args: object, **kwargs: object) -> None:
        raise AssertionError("pixel decode was reached")

    monkeypatch.setattr("pixtreme._io.frontend._decode_raster_frame", forbidden_decode)
    for invalid in (0, 1, None, "true"):
        with pytest.raises(ValueError, match=_ACTIONABLE):
            px.io.read_image(path, apply_exif_orientation=invalid)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match=_ACTIONABLE):
            px.io.decode_image(payload, apply_exif_orientation=invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize("format_name", _TARGET_FORMATS)
@pytest.mark.parametrize("orientation", range(1, 9))
def test_target_formats_apply_the_normative_orientation_at_file_bytes_and_header_boundaries(
    tmp_path: Path, format_name: str, orientation: int
) -> None:
    """v1-io-orientation acceptance 2-5 and 9: hand-defined mappings govern every target boundary."""
    payload = encode_oriented_raster(format_name, orientation)
    path = tmp_path / f"orientation-{orientation}{_SUFFIXES[format_name]}"
    path.write_bytes(payload)

    stored = px.io.decode_image(payload, unchanged=True, apply_exif_orientation=False)
    from_file_stored = px.io.read_image(path, unchanged=True, apply_exif_orientation=False)
    expected = _oriented(_frame_values(stored), orientation)
    from_bytes_default = px.io.decode_image(payload, unchanged=True)
    from_bytes_true = px.io.decode_image(payload, unchanged=True, apply_exif_orientation=True)
    from_file_default = px.io.read_image(path, unchanged=True)
    from_file_true = px.io.read_image(path, unchanged=True, apply_exif_orientation=True)
    header = px.io.read_header(path)

    _assert_same_frame(from_file_stored, stored)
    for actual in (from_bytes_default, from_bytes_true, from_file_default, from_file_true):
        np.testing.assert_array_equal(_frame_values(actual), expected)
        assert (actual.dtype, actual.colorspace, actual.gamma, actual.channels, actual.matrix) == (
            stored.dtype,
            stored.colorspace,
            stored.gamma,
            stored.channels,
            stored.matrix,
        )
    stored_height, stored_width = _frame_values(stored).shape[:2]
    expected_width, expected_height = (
        (stored_height, stored_width) if orientation >= 5 else (stored_width, stored_height)
    )
    assert (header.width, header.height, header.orientation) == (expected_width, expected_height, orientation)


@pytest.mark.parametrize("format_name", _TARGET_FORMATS)
def test_orientation_preserves_channel_selection_and_file_bytes_equivalence(tmp_path: Path, format_name: str) -> None:
    """v1-io-orientation acceptance 4 and 9: selection labels and samples survive orientation unchanged."""
    payload = encode_oriented_raster(format_name, 6)
    path = tmp_path / f"selected{_SUFFIXES[format_name]}"
    path.write_bytes(payload)

    stored = px.io.decode_image(payload, channels="BR", unchanged=True, apply_exif_orientation=False)
    from_file_stored = px.io.read_image(path, channels="BR", unchanged=True, apply_exif_orientation=False)
    oriented_bytes = px.io.decode_image(payload, channels="BR", unchanged=True, apply_exif_orientation=True)
    oriented_file = px.io.read_image(path, channels="BR", unchanged=True, apply_exif_orientation=True)

    assert stored.channels == ("B", "R")
    _assert_same_frame(from_file_stored, stored)
    _assert_same_frame(oriented_file, oriented_bytes)
    np.testing.assert_array_equal(_frame_values(oriented_bytes), _oriented(_frame_values(stored), 6))


def test_lossless_webp_exif_is_decoded_at_file_and_bytes_boundaries(tmp_path: Path) -> None:
    """v1-io-orientation acceptance 2-5: VP8L plus EXIF follows the same mapping as lossy WebP."""
    payload = encode_lossless_oriented_webp(6)
    path = tmp_path / "lossless-oriented.webp"
    path.write_bytes(payload)

    stored = px.io.decode_image(payload, unchanged=True, apply_exif_orientation=False)
    expected = _oriented(_frame_values(stored), 6)
    from_bytes = px.io.decode_image(payload, unchanged=True)
    from_file = px.io.read_image(path, unchanged=True)
    header = px.io.read_header(path)

    np.testing.assert_array_equal(_frame_values(from_bytes), expected)
    _assert_same_frame(from_file, from_bytes)
    assert (header.width, header.height, header.orientation) == (18, 24, 6)


def test_webp_payload_read_failure_after_header_parse_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v1-io-icc acceptance 20; v1-io-orientation acceptance 2 and REQ-API-012: a WebP file read failure after the header is parsed
    surfaces as the public three-part RuntimeError with the original OSError chained as its cause."""
    path = tmp_path / "oriented.webp"
    path.write_bytes(encode_oriented_raster("WEBP", 6))
    read_calls: list[Path] = []
    original_read_bytes = Path.read_bytes

    def failing_read_bytes(self: Path) -> bytes:
        if self != path:
            return original_read_bytes(self)
        read_calls.append(self)
        raise PermissionError("simulated pixel-read failure")

    monkeypatch.setattr(Path, "read_bytes", failing_read_bytes)

    with pytest.raises(RuntimeError, match=_ACTIONABLE) as raised:
        px.io.read_image(path)

    assert isinstance(raised.value.__cause__, PermissionError)
    assert "simulated pixel-read failure" in str(raised.value)
    assert read_calls == [path]


def test_image_header_orientation_defaults_to_one_for_direct_construction(tmp_path: Path) -> None:
    """v1-io-orientation acceptance 5: the frozen public model gives direct construction a default of one."""
    path = tmp_path / "plain.png"
    path.write_bytes(encode_plain_raster("PNG"))
    values = px.io.read_header(path).model_dump()
    values.pop("orientation", None)

    header = px.io.ImageHeader(**values)

    assert header.orientation == 1
    assert set(px.io.ImageHeader.model_fields) == {"format", "width", "height", "parts", "color", "orientation"}


def test_oriented_headers_for_all_target_formats_are_codec_lazy_and_gpu_free(tmp_path: Path) -> None:
    """v1-io-orientation acceptance 5 and 6: orientation-aware header probing remains CPU-only."""
    paths: list[Path] = []
    for format_name in _TARGET_FORMATS:
        path = tmp_path / f"oriented{_SUFFIXES[format_name]}"
        path.write_bytes(encode_oriented_raster(format_name, 6))
        paths.append(path)
    script = """
import sys
import pixtreme as px
for value in sys.argv[1:]:
    header = px.io.read_header(value)
    assert (header.width, header.height, header.orientation) == (18, 24, 6)
assert "nvidia.nvimgcodec" not in sys.modules
assert "OpenEXR" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script, *(str(path) for path in paths)],
        cwd=ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


_INVALID_EXIF = (
    ("zero", exif_payload(((274, 3, 1, 0),))),
    ("above-eight", exif_payload(((274, 3, 1, 9),))),
    ("truncated", b"II*\x00\x08\x00\x00\x00\x01\x00"),
    ("byte-order", exif_payload((), byte_order=b"ZZ")),
    ("type", exif_payload(((274, 4, 1, 6),))),
    ("count", exif_payload(((274, 3, 2, 6),))),
    ("duplicate", exif_payload(((274, 3, 1, 6), (274, 3, 1, 6)))),
    ("conflict", exif_payload(((274, 3, 1, 2), (274, 3, 1, 6)))),
)


@pytest.mark.parametrize(
    ("case", "metadata"), _INVALID_EXIF, ids=lambda value: value if isinstance(value, str) else None
)
@pytest.mark.parametrize("boundary", ("header", "file", "bytes"))
def test_invalid_orientation_metadata_warns_and_falls_back_at_every_boundary(
    tmp_path: Path, case: str, metadata: bytes, boundary: str
) -> None:
    """v1-io-orientation acceptance 3 and 7: invalid optional metadata warns and becomes identity everywhere."""
    payload = png_with_exif(metadata)
    path = tmp_path / f"invalid-{case}.png"
    path.write_bytes(payload)

    with pytest.warns(UserWarning, match="EXIF orientation"):
        if boundary == "header":
            assert px.io.read_header(path).orientation == 1
        elif boundary == "file":
            frame = px.io.read_image(path, unchanged=True, apply_exif_orientation=False)
            np.testing.assert_array_equal(_frame_values(frame), orientation_pattern())
        else:
            frame = px.io.decode_image(payload, unchanged=True, apply_exif_orientation=False)
            np.testing.assert_array_equal(_frame_values(frame), orientation_pattern())


@pytest.mark.parametrize("format_name", _TARGET_FORMATS)
def test_missing_orientation_is_silent_identity_for_every_target_format(tmp_path: Path, format_name: str) -> None:
    """v1-io-orientation acceptance 3, 5, and 7: an absent tag is silent and resolves to one."""
    payload = encode_plain_raster(format_name)
    path = tmp_path / f"plain{_SUFFIXES[format_name]}"
    path.write_bytes(payload)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        header = px.io.read_header(path)
        stored = px.io.decode_image(payload, unchanged=True, apply_exif_orientation=False)
        oriented = px.io.read_image(path, unchanged=True, apply_exif_orientation=True)

    assert header.orientation == 1
    _assert_same_frame(oriented, stored)
    assert [item for item in caught if "EXIF orientation" in str(item.message)] == []


def test_secondary_ifd_orientation_does_not_apply_to_the_primary_image(tmp_path: Path) -> None:
    """v1-io-orientation acceptance 6: thumbnail and secondary-IFD orientation cannot affect the primary image."""
    metadata = exif_payload((), secondary_entries=((274, 3, 1, 6),))
    payload = png_with_exif(metadata)
    path = tmp_path / "secondary.png"
    path.write_bytes(payload)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        header = px.io.read_header(path)
        frame = px.io.read_image(path, unchanged=True)

    assert (header.width, header.height, header.orientation) == (24, 18, 1)
    np.testing.assert_array_equal(_frame_values(frame), orientation_pattern())
    assert [item for item in caught if "EXIF orientation" in str(item.message)] == []


def test_non_target_file_and_bytes_formats_ignore_the_orientation_switch(tmp_path: Path) -> None:
    """v1-io-orientation acceptance 8 and 9: non-target formats accept the switch without changing results."""
    values = np.arange(4 * 8 * 3, dtype=np.uint8).reshape(4, 8, 3)
    frame = px.io.from_array(cp.asarray(values), colorspace="sRGB", gamma="sRGB", channels="RGB")
    file_cases = (
        ("image.jp2", {"lossless": True}),
        ("image.bmp", {}),
        ("image.pnm", {}),
        ("image.exr", {}),
        ("image.tga", {}),
        ("image.hdr", {}),
        ("image.dpx", {}),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for name, options in file_cases:
            path = tmp_path / name
            px.io.write_image(path, frame, **options)
            reference = px.io.read_image(path)
            _assert_same_frame(px.io.read_image(path, apply_exif_orientation=True), reference)
            _assert_same_frame(px.io.read_image(path, apply_exif_orientation=False), reference)
            assert px.io.read_header(path).orientation == 1

        for format_name, options in (("jpeg2000", {"lossless": True}), ("bmp", {}), ("pnm", {})):
            payload = px.io.encode_image(frame, format=format_name, **options)
            reference = px.io.decode_image(payload)
            _assert_same_frame(px.io.decode_image(payload, apply_exif_orientation=True), reference)
            _assert_same_frame(px.io.decode_image(payload, apply_exif_orientation=False), reference)

    assert [item for item in caught if "EXIF orientation" in str(item.message)] == []
