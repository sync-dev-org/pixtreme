"""Specification tests for the public uint32 OpenEXR dtype boundary."""

from __future__ import annotations

import struct
from pathlib import Path

import cupy as cp
import numpy as np
import pytest

import pixtreme as px

_COMPRESSIONS = ("none", "rle", "zip", "zips", "piz", "pxr24", "b44", "b44a", "dwaa", "dwab")
_PIXEL_TYPES = {np.dtype(np.uint32): 0, np.dtype(np.float16): 1, np.dtype(np.float32): 2}


def _attribute(name: str, attribute_type: str, payload: bytes) -> bytes:
    return name.encode() + b"\x00" + attribute_type.encode() + b"\x00" + struct.pack("<I", len(payload)) + payload


def _none_exr(channels: dict[str, np.ndarray]) -> bytes:
    """Build an uncompressed scanline EXR directly from the version-2 wire grammar."""
    names = tuple(sorted(channels))
    shapes = {np.asarray(channels[name]).shape for name in names}
    assert len(shapes) == 1
    height, width = shapes.pop()
    channel_payload = bytearray()
    for name in names:
        source = np.asarray(channels[name])
        channel_payload.extend(name.encode() + b"\x00")
        channel_payload.extend(struct.pack("<iB3xii", _PIXEL_TYPES[source.dtype], 0, 1, 1))
    channel_payload.append(0)
    data_window = (0, 0, width - 1, height - 1)
    header = (
        struct.pack("<II", 20000630, 2)
        + b"".join(
            (
                _attribute("channels", "chlist", bytes(channel_payload)),
                _attribute("compression", "compression", b"\x00"),
                _attribute("dataWindow", "box2i", struct.pack("<iiii", *data_window)),
                _attribute("displayWindow", "box2i", struct.pack("<iiii", *data_window)),
                _attribute("lineOrder", "lineOrder", b"\x00"),
                _attribute("pixelAspectRatio", "float", struct.pack("<f", 1.0)),
                _attribute("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0)),
                _attribute("screenWindowWidth", "float", struct.pack("<f", 1.0)),
            )
        )
        + b"\x00"
    )
    chunks: list[bytes] = []
    for y in range(height):
        row = b"".join(
            np.asarray(channels[name])[y].astype(channels[name].dtype.newbyteorder("<"), copy=False).tobytes()
            for name in names
        )
        chunks.append(struct.pack("<ii", y, len(row)) + row)
    first_chunk = len(header) + height * 8
    offsets: list[int] = []
    cursor = first_chunk
    for chunk in chunks:
        offsets.append(cursor)
        cursor += len(chunk)
    return header + b"".join(struct.pack("<Q", offset) for offset in offsets) + b"".join(chunks)


def _recode_oracle(values: np.ndarray, target_dtype: str) -> np.ndarray:
    source = np.asarray(values)
    target = np.dtype(target_dtype)
    if source.dtype == target or (np.issubdtype(source.dtype, np.floating) and np.issubdtype(target, np.floating)):
        return source.astype(target)
    if np.issubdtype(source.dtype, np.integer) and np.issubdtype(target, np.integer):
        source_maximum = int(np.iinfo(source.dtype).max)
        target_maximum = int(np.iinfo(target).max)
        flat = [(int(value) * target_maximum + source_maximum // 2) // source_maximum for value in source.reshape(-1)]
        return np.asarray(flat, dtype=target).reshape(source.shape)
    if np.issubdtype(source.dtype, np.integer):
        return (source.astype(np.float32) * np.float32(1.0 / int(np.iinfo(source.dtype).max))).astype(target)
    maximum = int(np.iinfo(target).max)
    return np.floor(np.clip(source.astype(np.float64), 0.0, 1.0) * maximum + 0.5).astype(target)


def test_hand_built_uint_and_mixed_wire_reads_follow_the_public_dtype_contract(tmp_path: Path) -> None:
    """v1-exr-runtime-independence acceptance 10-12, 44, and 46:
    hand-built UINT wire fixes the supported lossy default-read and native unchanged semantics.
    """
    uint_values = np.asarray([[0, 1], [16777217, 4294967295]], dtype=np.uint32)
    uint_path = tmp_path / "uint.exr"
    uint_path.write_bytes(_none_exr({label: uint_values for label in "RGB"}))

    default = px.io.read_image(uint_path)
    unchanged = px.io.read_image(uint_path, unchanged=True)

    expected_uint = np.repeat(uint_values[..., None], 3, axis=2)
    assert (default.dtype, unchanged.dtype) == (np.dtype(np.float32), np.dtype(np.uint32))
    np.testing.assert_array_equal(
        px.io.to_array(
            default,
        ).get(),
        expected_uint.astype(np.float32),
    )
    np.testing.assert_array_equal(
        px.io.to_array(
            unchanged,
        ).get(),
        expected_uint,
    )

    mixed_path = tmp_path / "mixed.exr"
    mixed_path.write_bytes(
        _none_exr(
            {
                "R": uint_values,
                "G": np.asarray([[0.0, 0.5], [1.0, 2.0]], dtype=np.float16),
                "B": np.asarray([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
            }
        )
    )
    mixed = px.io.read_image(mixed_path)
    assert mixed.dtype == np.dtype(np.float32)
    with pytest.raises(ValueError, match=r"why=.*mixed.*what=.*how=.*unchanged=False"):
        px.io.read_image(mixed_path, unchanged=True)


@pytest.mark.parametrize("suffix", ("png", "tga", "hdr", "dpx"))
def test_explicit_write_dtype_is_rejected_outside_exr(tmp_path: Path, suffix: str) -> None:
    """v1-exr-runtime-independence acceptance 2: non-EXR dtype claims fail with format and recovery guidance."""
    frame = px.io.from_array(cp.ones((1, 1, 3), dtype=cp.float32), colorspace="sRGB", gamma="linear", channels="RGB")

    with pytest.raises(ValueError, match=rf"why=.*dtype.*what=.*{suffix.upper()}.*how=.*EXR"):
        px.io.write_image(tmp_path / f"image.{suffix}", frame, dtype="float16")


@pytest.mark.parametrize("dtype", ("uint16", "fp32", "FLOAT", "", 32, True))
def test_exr_write_dtype_rejects_values_outside_the_three_token_set(tmp_path: Path, dtype: object) -> None:
    """v1-exr-runtime-independence acceptance 1: EXR dtype is an exact three-token closed set."""
    frame = px.io.from_array(cp.ones((1, 1, 3), dtype=cp.float32), colorspace="ACEScg", gamma="linear", channels="RGB")

    with pytest.raises(ValueError, match=r"why=.*dtype.*what=.*how=.*float16.*float32.*uint32"):
        px.io.write_image(tmp_path / "invalid.exr", frame, dtype=dtype)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("input_dtype", "dtype", "storage_dtype"),
    (
        ("float32", None, "float16"),
        ("uint8", None, "float16"),
        ("uint32", None, "uint32"),
        ("float16", "float32", "float32"),
        ("float32", "uint32", "uint32"),
        ("uint32", "float16", "float16"),
    ),
)
def test_exr_write_dtype_resolution_matches_independent_full_scale_oracle(
    tmp_path: Path, input_dtype: str, dtype: str | None, storage_dtype: str
) -> None:
    """v1-exr-runtime-independence acceptance 13-14: default and explicit write dtype use full-scale recoding."""
    if input_dtype.startswith("uint"):
        maximum = int(np.iinfo(input_dtype).max)
        values = np.asarray([0, maximum // 4, maximum // 2, maximum], dtype=input_dtype)
    else:
        values = np.asarray([0.0, 0.25, 0.5, 1.0], dtype=input_dtype)
    image = np.resize(values, (2, 2, 3))
    frame = px.io.from_array(cp.asarray(image), colorspace="ACEScg", gamma="linear", channels="RGB")
    before = frame.data.copy()
    path = tmp_path / f"{input_dtype}-{storage_dtype}.exr"

    px.io.write_image(path, frame, compression="none", dtype=dtype)

    header = px.io.read_header(path)
    restored = px.io.read_image(path, unchanged=True)
    expected = _recode_oracle(image, storage_dtype)
    assert set(header.parts[0].channels.values()) == {storage_dtype}
    assert restored.dtype == np.dtype(storage_dtype)
    np.testing.assert_array_equal(
        px.io.to_array(
            restored,
        ).get(),
        expected,
    )
    assert frame.dtype == np.dtype(input_dtype)
    cp.testing.assert_array_equal(frame.data, before)


@pytest.mark.parametrize("compression", _COMPRESSIONS)
def test_native_uint32_write_is_bit_exact_for_every_compression(tmp_path: Path, compression: str) -> None:
    """v1-exr-runtime-independence acceptance 10 and 15: all UINT codecs preserve bits and promote literally."""
    values = np.asarray(
        [[[0, 1, 16777217], [4294967295, 2147483648, 305419896]]],
        dtype=np.uint32,
    )
    frame = px.io.from_array(cp.asarray(values), colorspace="ACEScg", gamma="linear", channels="RGB")
    before = frame.data.copy()
    path = tmp_path / f"uint32-{compression}.exr"

    px.io.write_image(path, frame, compression=compression)

    header = px.io.read_header(path)
    restored = px.io.read_image(path, unchanged=True)
    promoted = px.io.read_image(path)
    assert set(header.parts[0].channels.values()) == {"uint32"}
    assert restored.dtype == np.dtype(np.uint32)
    assert promoted.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(
        px.io.to_array(
            restored,
        ).get(),
        values,
    )
    np.testing.assert_array_equal(
        px.io.to_array(
            promoted,
        ).get(),
        values.astype(np.float32),
    )
    cp.testing.assert_array_equal(frame.data, before)
