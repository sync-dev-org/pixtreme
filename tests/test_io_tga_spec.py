"""Specification tests for the Truevision TGA file boundary."""

from __future__ import annotations

import os
import struct
import subprocess
import sys
from pathlib import Path

import cupy as cp
import numpy as np
import pytest

import pixtreme as px

ROOT = Path(__file__).resolve().parents[1]
_ACTIONABLE = r"why=.*what=.*how="


def _header(
    *,
    width: int = 3,
    height: int = 2,
    image_type: int = 2,
    pixel_depth: int = 24,
    descriptor: int | None = None,
    color_map_type: int = 0,
    color_map_origin: int = 0,
    color_map_length: int = 0,
    color_map_depth: int = 0,
    image_id: bytes = b"",
) -> bytes:
    if descriptor is None:
        descriptor = (8 if pixel_depth == 32 else 0) | 0x20
    return (
        struct.pack(
            "<BBBHHBHHHHBB",
            len(image_id),
            color_map_type,
            image_type,
            color_map_origin,
            color_map_length,
            color_map_depth,
            0,
            0,
            width,
            height,
            pixel_depth,
            descriptor,
        )
        + image_id
    )


def _file_order_pixels(rgb: np.ndarray, *, top_origin: bool) -> list[bytes]:
    rows = rgb if top_origin else rgb[::-1]
    if rgb.shape[2] == 3:
        file_order = rows[..., [2, 1, 0]]
    else:
        file_order = rows[..., [2, 1, 0, 3]]
    return [pixel.tobytes() for pixel in file_order.reshape(-1, file_order.shape[2])]


def _uncompressed_fixture(rgb: np.ndarray, *, top_origin: bool) -> bytes:
    descriptor = (8 if rgb.shape[2] == 4 else 0) | (0x20 if top_origin else 0)
    return _header(
        width=rgb.shape[1],
        height=rgb.shape[0],
        image_type=2,
        pixel_depth=rgb.shape[2] * 8,
        descriptor=descriptor,
    ) + b"".join(_file_order_pixels(rgb, top_origin=top_origin))


def _hand_built_rle_fixture(rgb: np.ndarray, *, top_origin: bool) -> bytes:
    """Build the fixed A,A,B / C,D,D corpus from literal TGA packet rules."""
    rows = rgb if top_origin else rgb[::-1]
    packets = bytearray()
    file_order = [2, 1, 0] if rgb.shape[2] == 3 else [2, 1, 0, 3]
    for row in rows:
        file_pixels = [pixel[file_order].tobytes() for pixel in row]
        if np.array_equal(row[0], row[1]):
            packets.extend(b"\x81" + file_pixels[0])
            packets.extend(b"\x00" + file_pixels[2])
        else:
            packets.extend(b"\x00" + file_pixels[0])
            packets.extend(b"\x81" + file_pixels[1])
    descriptor = (8 if rgb.shape[2] == 4 else 0) | (0x20 if top_origin else 0)
    return _header(
        width=3,
        height=2,
        image_type=10,
        pixel_depth=rgb.shape[2] * 8,
        descriptor=descriptor,
    ) + bytes(packets)


def _decode_rle(payload: bytes, *, pixel_size: int, pixel_count: int) -> tuple[bytes, tuple[tuple[bool, int], ...]]:
    """Independent host decoder for TGA packet bytes, kinds, and lengths."""
    output = bytearray()
    packets: list[tuple[bool, int]] = []
    offset = 0
    while len(output) < pixel_count * pixel_size:
        packet_header = payload[offset]
        offset += 1
        count = (packet_header & 0x7F) + 1
        is_run = bool(packet_header & 0x80)
        packets.append((is_run, count))
        if is_run:
            pixel = payload[offset : offset + pixel_size]
            offset += pixel_size
            output.extend(pixel * count)
        else:
            size = count * pixel_size
            output.extend(payload[offset : offset + size])
            offset += size
    assert len(output) == pixel_count * pixel_size
    return bytes(output), tuple(packets)


def _uint8_oracle(values: np.ndarray) -> np.ndarray:
    if values.dtype == np.dtype(np.uint8):
        return values.copy()
    if np.issubdtype(values.dtype, np.integer):
        maximum = int(np.iinfo(values.dtype).max)
        scaled = [(int(value) * 255 + maximum // 2) // maximum for value in values.reshape(-1)]
        return np.asarray(scaled, dtype=np.uint8).reshape(values.shape)
    normalized = values.astype(np.float32)
    return np.floor(np.clip(normalized, 0.0, 1.0) * np.float32(255.0) + np.float32(0.5)).astype(np.uint8)


@pytest.mark.parametrize("image_type", (2, 10), ids=("uncompressed", "rle"))
@pytest.mark.parametrize("pixel_depth", (24, 32), ids=("rgb", "rgba"))
@pytest.mark.parametrize("top_origin", (False, True), ids=("bottom-left", "top-left"))
def test_tga_read_supports_true_color_rle_and_both_vertical_origins(
    tmp_path: Path, image_type: int, pixel_depth: int, top_origin: bool
) -> None:
    """v1-tga acceptance 1 and 3: supported storage variants return upright normalized RGB(A)."""
    channel_count = pixel_depth // 8
    a = np.array([10, 20, 30, 130], dtype=np.uint8)[:channel_count]
    b = np.array([40, 50, 60, 140], dtype=np.uint8)[:channel_count]
    c = np.array([70, 80, 90, 150], dtype=np.uint8)[:channel_count]
    d = np.array([100, 110, 120, 160], dtype=np.uint8)[:channel_count]
    rgb = np.array([[a, a, b], [c, d, d]], dtype=np.uint8)
    payload = (
        _uncompressed_fixture(rgb, top_origin=top_origin)
        if image_type == 2
        else _hand_built_rle_fixture(rgb, top_origin=top_origin)
    )
    path = tmp_path / "input.TGA"
    path.write_bytes(payload)

    actual = px.io.read_image(path)

    assert (actual.dtype, actual.channels, actual.colorspace, actual.gamma) == (
        np.dtype(np.float32),
        ("R", "G", "B", "A")[:channel_count],
        "sRGB",
        "srgb",
    )
    assert actual.data.flags.c_contiguous
    np.testing.assert_array_equal(
        px.io.to_array(
            actual,
        ).get(),
        rgb.astype(np.float32) / np.float32(255.0),
    )


def test_tga_read_unchanged_selects_channels_and_overrides_metadata(tmp_path: Path) -> None:
    """v1-tga acceptance 3 and 4: native uint8, label order, and metadata claims match raster reads."""
    rgba = np.array([[[1, 2, 3, 4], [250, 128, 64, 32]]], dtype=np.uint8)
    path = tmp_path / "rgba.tga"
    path.write_bytes(_uncompressed_fixture(rgba, top_origin=True))

    actual = px.io.read_image(
        path,
        unchanged=True,
        channels="ABR",
        colorspace="Rec.2020",
        gamma="pq",
    )

    assert (actual.dtype, actual.channels, actual.colorspace, actual.gamma) == (
        np.dtype(np.uint8),
        ("A", "B", "R"),
        "Rec.2020",
        "pq",
    )
    np.testing.assert_array_equal(
        px.io.to_array(
            actual,
        ).get(),
        rgba[..., [3, 2, 0]],
    )


def test_tga_read_resolves_every_repeated_output_channel_position(tmp_path: Path) -> None:
    """v1-tga acceptance 3: channel selection remains label-driven beyond four output positions."""
    rgba = np.array([[[1, 2, 3, 4]]], dtype=np.uint8)
    path = tmp_path / "repeated-alpha.tga"
    path.write_bytes(_uncompressed_fixture(rgba, top_origin=True))

    actual = px.io.read_image(path, unchanged=True, channels=("R", "G", "B", "A", "A"))

    assert actual.channels == ("R", "G", "B", "A", "A")
    np.testing.assert_array_equal(
        px.io.to_array(
            actual,
        ).get(),
        rgba[..., [0, 1, 2, 3, 3]],
    )


@pytest.mark.parametrize(
    ("payload", "observed"),
    (
        (_header(image_type=1, pixel_depth=8, color_map_type=1), "image_type=1"),
        (_header(image_type=9, pixel_depth=8, color_map_type=1), "image_type=9"),
        (_header(image_type=3, pixel_depth=8), "image_type=3"),
        (_header(image_type=11, pixel_depth=8), "image_type=11"),
        (_header(image_type=2, pixel_depth=16), "pixel_depth=16"),
        (_header(image_type=2, descriptor=0x10), "right-to-left"),
        (_header(image_type=2, descriptor=0x40), "reserved"),
        (_header(image_type=2, pixel_depth=24, descriptor=1), "attribute_bits=1"),
        (_header(image_type=2, pixel_depth=32, descriptor=0), "attribute_bits=0"),
        (_header(image_type=2, color_map_type=1), "color_map_type=1"),
        (_header(image_type=2, color_map_length=1), "color_map_length=1"),
    ),
    ids=(
        "colormap",
        "colormap-rle",
        "grayscale",
        "grayscale-rle",
        "depth-16",
        "right-origin",
        "reserved-bit",
        "rgb-attribute",
        "rgba-attribute",
        "color-map-present",
        "color-map-fields",
    ),
)
def test_tga_read_rejects_out_of_scope_configurations_before_gpu_transfer(
    tmp_path: Path, payload: bytes, observed: str
) -> None:
    """v1-tga acceptance 2: unsupported header configurations fail fast with actionable ValueError."""
    path = tmp_path / "unsupported.tga"
    path.write_bytes(payload)

    with pytest.raises(ValueError, match=_ACTIONABLE) as error:
        px.io.read_image(path)

    assert observed in str(error.value)


@pytest.mark.parametrize(
    ("channels", "values"),
    (
        (
            "BGR",
            np.array(
                [
                    [[30, 20, 10], [60, 50, 40], [90, 80, 70]],
                    [[30, 20, 10], [30, 20, 10], [120, 110, 100]],
                ],
                dtype=np.uint8,
            ),
        ),
        (
            ("A", "B", "G", "R"),
            np.array([[[4, 3, 2, 1], [8, 7, 6, 5], [12, 11, 10, 9]]], dtype=np.uint8),
        ),
    ),
    ids=("rgb", "rgba"),
)
def test_tga_write_is_rle_top_left_and_independently_decodable(
    tmp_path: Path,
    channels: str | tuple[str, ...],
    values: np.ndarray,
) -> None:
    """v1-tga acceptance 6 and 7: writer fixes type, depth, origin, packet bounds, and swizzle."""
    frame = px.io.from_array(cp.asarray(values), colorspace="sRGB", gamma="srgb", channels=channels)
    path = tmp_path / "output.tga"

    assert px.io.write_image(path, frame) is None

    payload = path.read_bytes()
    fields = struct.unpack("<BBBHHBHHHHBB", payload[:18])
    channel_count = values.shape[2]
    assert fields[2] == 10
    assert fields[9:12] == (values.shape[0], channel_count * 8, 0x28 if channel_count == 4 else 0x20)
    assert payload[-26:] == struct.pack("<II", 0, 0) + b"TRUEVISION-XFILE.\x00"
    decoded, packets = _decode_rle(
        payload[18:-26], pixel_size=channel_count, pixel_count=values.shape[0] * values.shape[1]
    )
    packet_lengths = tuple(count for _, count in packets)
    assert all(1 <= count <= 128 for count in packet_lengths)
    for pixel_offset, count in zip(np.cumsum((0, *packet_lengths[:-1])), packet_lengths, strict=True):
        assert count <= values.shape[1] - (int(pixel_offset) % values.shape[1])
    file_pixels = np.frombuffer(decoded, dtype=np.uint8).reshape(values.shape)
    input_labels = tuple(channels)
    expected = values[..., [input_labels.index(label) for label in ("R", "G", "B")]]
    if channel_count == 4:
        expected = np.concatenate((expected, values[..., [input_labels.index("A")]]), axis=2)
        file_pixels = file_pixels[..., [2, 1, 0, 3]]
    else:
        file_pixels = file_pixels[..., [2, 1, 0]]
    np.testing.assert_array_equal(file_pixels, expected)


def test_tga_write_packets_never_cross_scanlines(tmp_path: Path) -> None:
    """v1-tga acceptance 7: an equal-color row boundary still produces separate packets."""
    values = np.full((2, 3, 3), 17, dtype=np.uint8)
    frame = px.io.from_array(cp.asarray(values), colorspace="sRGB", gamma="srgb", channels="RGB")
    path = tmp_path / "rows.tga"

    px.io.write_image(path, frame)

    payload = path.read_bytes()[18:-26]
    _, packets = _decode_rle(payload, pixel_size=3, pixel_count=6)
    packet_lengths = tuple(count for _, count in packets)
    assert packet_lengths == (3, 3)


@pytest.mark.parametrize("packet_kind", ("raw", "run"))
@pytest.mark.parametrize("pixel_count", (128, 129))
def test_tga_read_decodes_rle_packet_count_boundaries(tmp_path: Path, packet_kind: str, pixel_count: int) -> None:
    """v1-tga acceptance 1 and 13: hand-built raw and run packets decode at the 128-pixel limit."""
    if packet_kind == "run":
        rgb = np.repeat(np.array([[[11, 22, 33]]], dtype=np.uint8), pixel_count, axis=1)
    else:
        values = np.arange(pixel_count, dtype=np.uint8)
        rgb = np.stack((values, values ^ np.uint8(0x55), values ^ np.uint8(0xAA)), axis=1)[None, ...]
    file_pixels = rgb[0][..., [2, 1, 0]]
    packets = bytearray()
    for start in range(0, pixel_count, 128):
        count = min(128, pixel_count - start)
        if packet_kind == "run":
            packets.append(0x80 | (count - 1))
            packets.extend(file_pixels[start].tobytes())
        else:
            packets.append(count - 1)
            packets.extend(file_pixels[start : start + count].tobytes())
    path = tmp_path / f"{packet_kind}-{pixel_count}.tga"
    path.write_bytes(_header(width=pixel_count, height=1, image_type=10) + bytes(packets))

    actual = px.io.read_image(path, unchanged=True)

    np.testing.assert_array_equal(
        px.io.to_array(
            actual,
        ).get(),
        rgb,
    )


@pytest.mark.parametrize(
    ("packet_kind", "pixel_count", "expected_packets"),
    (
        ("raw", 128, ((False, 128),)),
        ("raw", 129, ((False, 128), (False, 1))),
        ("run", 128, ((True, 128),)),
        ("run", 129, ((True, 128), (True, 1))),
    ),
)
def test_tga_write_splits_raw_and_run_packets_at_128_pixels(
    tmp_path: Path,
    packet_kind: str,
    pixel_count: int,
    expected_packets: tuple[tuple[bool, int], ...],
) -> None:
    """v1-tga acceptance 7 and 13: writer preserves packet kind and splits counts above 128."""
    if packet_kind == "run":
        values = np.repeat(np.array([[[17, 34, 51]]], dtype=np.uint8), pixel_count, axis=1)
    else:
        codes = np.arange(pixel_count, dtype=np.uint8)
        values = np.stack((codes, codes ^ np.uint8(0x55), codes ^ np.uint8(0xAA)), axis=1)[None, ...]
    frame = px.io.from_array(cp.asarray(values), colorspace="sRGB", gamma="srgb", channels="RGB")
    path = tmp_path / f"writer-{packet_kind}-{pixel_count}.tga"

    px.io.write_image(path, frame)

    decoded, packets = _decode_rle(path.read_bytes()[18:-26], pixel_size=3, pixel_count=pixel_count)
    assert packets == expected_packets
    actual_bgr = np.frombuffer(decoded, dtype=np.uint8).reshape(1, pixel_count, 3)
    np.testing.assert_array_equal(actual_bgr[..., [2, 1, 0]], values)


@pytest.mark.parametrize(
    ("width", "packet_data", "observed"),
    (
        (1, b"", "next packet header"),
        (2, b"\x01\x03\x02\x01", "raw packet has truncated pixel values"),
        (2, b"\x81\x03\x02", "run packet has a truncated pixel value"),
        (1, b"\x81\x03\x02\x01", "exceeds the declared image dimensions"),
    ),
    ids=("missing-header", "truncated-raw", "truncated-run", "pixel-count-overflow"),
)
def test_tga_read_rejects_malformed_rle_packets_with_actionable_runtime_error(
    tmp_path: Path, width: int, packet_data: bytes, observed: str
) -> None:
    """v1-tga acceptance 13: malformed or truncated RLE packets keep the public corruption contract."""
    path = tmp_path / "corrupt-rle.tga"
    path.write_bytes(_header(width=width, height=1, image_type=10) + packet_data)

    with pytest.raises(RuntimeError, match=_ACTIONABLE) as error:
        px.io.read_image(path)

    assert observed in str(error.value)


def test_tga_write_read_round_trip_is_exact_on_the_255_grid(tmp_path: Path) -> None:
    """v1-tga acceptance 8: native codes and their normalized float32 grid round-trip bit exactly."""
    codes = np.array(
        [[[0, 1, 2, 3], [127, 128, 254, 255]], [[255, 0, 128, 64], [9, 8, 7, 6]]],
        dtype=np.uint8,
    )
    frame = px.io.from_array(cp.asarray(codes), colorspace="sRGB", gamma="srgb", channels="RGBA")
    path = tmp_path / "roundtrip.tga"

    px.io.write_image(path, frame)
    native = px.io.read_image(path, unchanged=True)
    normalized = px.io.read_image(path)

    np.testing.assert_array_equal(
        px.io.to_array(
            native,
        ).get(),
        codes,
    )
    np.testing.assert_array_equal(
        px.io.to_array(
            normalized,
        ).get(),
        codes.astype(np.float32) / np.float32(255.0),
    )


@pytest.mark.parametrize(
    ("dtype", "samples"),
    (
        (np.uint8, [0, 1, 127, 128, 254, 255]),
        (np.uint16, [0, 128, 129, 257, 385, 386, 65406, 65535]),
        (np.uint32, [0, 16843009, 2147483648, 4278124286, 4294967295]),
        (np.float16, [-1.0, 0.0, 0.5, 1.0, 2.0]),
        (np.float32, [-1.0, 0.0, 0.5 / 255.0, 1.5 / 255.0, 0.5, 254.5 / 255.0, 1.0, 2.0]),
    ),
    ids=("uint8", "uint16", "uint32", "float16", "float32"),
)
def test_tga_write_converts_every_frame_dtype_to_uint8_with_independent_oracle(
    tmp_path: Path, dtype: type[np.generic], samples: list[float]
) -> None:
    """v1-write-dtype-convert acceptance 6; v1-exr-runtime-independence acceptance 9.

    All five dtypes share the independently derived uint8 recode contract.
    """
    one_channel = np.asarray(samples, dtype=dtype).reshape(1, -1, 1)
    values = np.repeat(one_channel, 3, axis=2)
    frame = px.io.from_array(cp.asarray(values), colorspace="sRGB", gamma="srgb", channels="RGB")
    before = frame.data.copy()
    path = tmp_path / f"{np.dtype(dtype).name}.tga"

    px.io.write_image(path, frame)

    payload = path.read_bytes()
    decoded, _ = _decode_rle(payload[18:-26], pixel_size=3, pixel_count=values.shape[1])
    actual_bgr = np.frombuffer(decoded, dtype=np.uint8).reshape(values.shape)
    np.testing.assert_array_equal(actual_bgr[..., [2, 1, 0]], _uint8_oracle(values))
    assert frame.dtype == np.dtype(dtype)
    cp.testing.assert_array_equal(frame.data, before)


def test_tga_write_rejects_non_rgb_layout(tmp_path: Path) -> None:
    """v1-write-dtype-convert acceptance 5; v1-tga acceptance 6.

    TGA output retains the channel-layout error contract while accepting every dtype.
    """
    y_frame = px.io.from_array(cp.zeros((1, 1, 1), dtype=cp.uint8), colorspace="sRGB", gamma="srgb", channels="Y")

    with pytest.raises(ValueError, match=_ACTIONABLE):
        px.io.write_image(tmp_path / "gray.tga", y_frame)
    assert not (tmp_path / "gray.tga").exists()


def test_tga_read_header_is_gpu_free_and_preserves_the_public_model(tmp_path: Path) -> None:
    """v1-tga acceptance 9: the TGA header parser is pure CPU and keeps ImageHeader fields unchanged."""
    path = tmp_path / "header.tga"
    path.write_bytes(_header(width=7, height=5, pixel_depth=32))
    script = """
import sys
import pixtreme as px
h = px.io.read_header(sys.argv[1])
assert (h.format, h.width, h.height) == ("TGA", 7, 5)
assert h.parts[0].channels == {"R": "uint8", "G": "uint8", "B": "uint8", "A": "uint8"}
assert set(px.io.ImageHeader.model_fields) == {"format", "width", "height", "parts", "color"}
assert "nvidia.nvimgcodec" not in sys.modules
assert "OpenEXR" not in sys.modules
"""

    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        cwd=ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


def test_tga_remains_outside_bytes_boundaries(tmp_path: Path) -> None:
    """v1-tga acceptance 10: TGA is file-only and adds no bytes token or signature path."""
    payload = _header(width=1, height=1) + b"\x03\x02\x01"
    frame = px.io.from_array(cp.zeros((1, 1, 3), dtype=cp.uint8), colorspace="sRGB", gamma="srgb", channels="RGB")

    with pytest.raises(ValueError, match=_ACTIONABLE):
        px.io.decode_image(payload)
    with pytest.raises(ValueError, match=_ACTIONABLE):
        px.io.encode_image(frame, format="tga")
    with pytest.raises(ValueError, match=_ACTIONABLE):
        px.io.write_image(tmp_path / "options.tga", frame, quality=90)
