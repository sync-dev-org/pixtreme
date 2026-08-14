"""Specification tests for tiled, multipart, deep, and sampled EXR reads."""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np
import pytest

import pixtreme as px
import pixtreme._io.formats.exr.container as exr_container


def _attribute(name: str, attribute_type: str, payload: bytes) -> bytes:
    return name.encode() + b"\x00" + attribute_type.encode() + b"\x00" + struct.pack("<I", len(payload)) + payload


def _channel_list(channels: tuple[tuple[str, int, tuple[int, int]], ...]) -> bytes:
    payload = bytearray()
    for name, pixel_type, sampling in channels:
        payload.extend(name.encode() + b"\x00")
        payload.extend(struct.pack("<iB3xii", pixel_type, 0, *sampling))
    payload.append(0)
    return bytes(payload)


def _part_attributes(
    *,
    data_window: tuple[int, int, int, int],
    channels: tuple[tuple[str, int, tuple[int, int]], ...],
    compression: int = 0,
    line_order: int = 0,
    name: str | None = None,
    image_type: str | None = None,
    chunk_count: int | None = None,
    tiles: tuple[int, int, int, int] | None = None,
    deep: bool = False,
) -> bytes:
    attributes = [
        _attribute("channels", "chlist", _channel_list(channels)),
        _attribute("compression", "compression", bytes((compression,))),
        _attribute("dataWindow", "box2i", struct.pack("<iiii", *data_window)),
        _attribute("displayWindow", "box2i", struct.pack("<iiii", *data_window)),
        _attribute("lineOrder", "lineOrder", bytes((line_order,))),
        _attribute("pixelAspectRatio", "float", struct.pack("<f", 1.0)),
        _attribute("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0)),
        _attribute("screenWindowWidth", "float", struct.pack("<f", 1.0)),
    ]
    if name is not None:
        attributes.append(_attribute("name", "string", name.encode()))
    if image_type is not None:
        attributes.append(_attribute("type", "string", image_type.encode()))
    if chunk_count is not None:
        attributes.append(_attribute("chunkCount", "int", struct.pack("<i", chunk_count)))
    if tiles is not None:
        x_size, y_size, level_mode, rounding_mode = tiles
        attributes.append(
            _attribute("tiles", "tiledesc", struct.pack("<IIB", x_size, y_size, level_mode | 16 * rounding_mode))
        )
    if deep:
        attributes.extend(
            (
                _attribute("version", "int", struct.pack("<i", 1)),
                _attribute("maxSamplesPerPixel", "int", struct.pack("<i", 1)),
            )
        )
    return b"".join(attributes) + b"\x00"


def _level_size(size: int, level: int, rounding_mode: int) -> int:
    divisor = 1 << level
    return max(1, math.ceil(size / divisor) if rounding_mode else size // divisor)


def _level_count(size: int, rounding_mode: int) -> int:
    count = 1
    while _level_size(size, count - 1, rounding_mode) > 1:
        count += 1
    return count


def _tile_identities(
    *,
    width: int,
    height: int,
    tile_size: tuple[int, int],
    level_mode: int,
    rounding_mode: int,
) -> tuple[tuple[int, int, int, int], ...]:
    x_levels = _level_count(width, rounding_mode)
    y_levels = _level_count(height, rounding_mode)
    if level_mode == 0:
        levels = ((0, 0),)
    elif level_mode == 1:
        levels = tuple((level, level) for level in range(max(x_levels, y_levels)))
    else:
        levels = tuple((level_x, level_y) for level_y in range(y_levels) for level_x in range(x_levels))
    identities: list[tuple[int, int, int, int]] = []
    for level_x, level_y in levels:
        level_width = _level_size(width, level_x, rounding_mode)
        level_height = _level_size(height, level_y, rounding_mode)
        columns = math.ceil(level_width / tile_size[0])
        rows = math.ceil(level_height / tile_size[1])
        identities.extend((tile_x, tile_y, level_x, level_y) for tile_y in range(rows) for tile_x in range(columns))
    return tuple(identities)


def _build_tiled_read_fixture(*, compression: int, level_mode: int, line_order: int) -> tuple[bytes, np.ndarray]:
    width, height = (5, 3)
    tile_width, tile_height = (2, 2)
    rounding_mode = 1
    identities = _tile_identities(
        width=width,
        height=height,
        tile_size=(tile_width, tile_height),
        level_mode=level_mode,
        rounding_mode=rounding_mode,
    )
    attributes = _part_attributes(
        data_window=(-2, 7, 2, 9),
        channels=(("R", 1, (1, 1)),),
        compression=compression,
        line_order=line_order,
        tiles=(tile_width, tile_height, level_mode, rounding_mode),
    )
    header = struct.pack("<II", 20000630, 2 | 0x200) + attributes
    expected = np.empty((height, width, 1), dtype=np.float16)
    chunks: dict[tuple[int, int, int, int], bytes] = {}
    for tile_x, tile_y, level_x, level_y in identities:
        level_width = _level_size(width, level_x, rounding_mode)
        level_height = _level_size(height, level_y, rounding_mode)
        stored_width = min(tile_width, level_width - tile_x * tile_width)
        stored_height = min(tile_height, level_height - tile_y * tile_height)
        if (level_x, level_y) == (0, 0):
            values = np.asarray(
                [
                    100 * (tile_y * tile_height + row) + tile_x * tile_width + column
                    for row in range(stored_height)
                    for column in range(stored_width)
                ],
                dtype="<f2",
            ).reshape(stored_height, stored_width)
            expected[
                tile_y * tile_height : tile_y * tile_height + stored_height,
                tile_x * tile_width : tile_x * tile_width + stored_width,
                0,
            ] = values
        else:
            values = np.full((stored_height, stored_width), -99.0, dtype="<f2")
        payload = values.tobytes()
        chunks[(tile_x, tile_y, level_x, level_y)] = (
            struct.pack("<iiiii", tile_x, tile_y, level_x, level_y, len(payload)) + payload
        )
    cursor = len(header) + len(identities) * 8
    offsets: dict[tuple[int, int, int, int], int] = {}
    chunk_blob = bytearray()
    for identity in reversed(identities):
        offsets[identity] = cursor
        chunk_blob.extend(chunks[identity])
        cursor += len(chunks[identity])
    table = b"".join(struct.pack("<Q", offsets[identity]) for identity in identities)
    return header + table + bytes(chunk_blob), expected


def _multipart_file(
    part_headers: tuple[bytes, ...],
    part_chunks: tuple[tuple[bytes, ...], ...],
    *,
    non_image: bool = False,
) -> bytes:
    flags = 0x1000 | (0x800 if non_image else 0)
    header = struct.pack("<II", 20000630, 2 | flags) + b"".join(part_headers) + b"\x00"
    physical: list[tuple[int, int, bytes]] = [
        (part_index, chunk_index, chunk)
        for part_index, chunks in enumerate(part_chunks)
        for chunk_index, chunk in enumerate(chunks)
    ]
    cursor = len(header) + len(physical) * 8
    offsets: dict[tuple[int, int], int] = {}
    chunk_blob = bytearray()
    for part_index, chunk_index, chunk in reversed(physical):
        offsets[(part_index, chunk_index)] = cursor
        chunk_blob.extend(chunk)
        cursor += len(chunk)
    table = b"".join(
        struct.pack("<Q", offsets[(part_index, chunk_index)])
        for part_index, chunks in enumerate(part_chunks)
        for chunk_index in range(len(chunks))
    )
    return header + table + bytes(chunk_blob)


def _scanline_chunk(part_index: int, y: int, values: tuple[float, ...]) -> bytes:
    payload = np.asarray(values, dtype="<f2").tobytes()
    return struct.pack("<iii", part_index, y, len(payload)) + payload


@pytest.mark.parametrize("compression", range(10))
@pytest.mark.parametrize("level_mode", (0, 1, 2))
@pytest.mark.parametrize("line_order", (0, 1, 2))
def test_tiled_level_zero_materializes_all_compressions_without_table_or_line_order_dependence(
    tmp_path: Path,
    compression: int,
    level_mode: int,
    line_order: int,
) -> None:
    """v1-exr-runtime-independence acceptance 16, 18, and 19: level zero owns the public HWC output."""
    payload, expected = _build_tiled_read_fixture(
        compression=compression,
        level_mode=level_mode,
        line_order=line_order,
    )
    path = tmp_path / f"tiled-{compression}-{level_mode}-{line_order}.exr"
    path.write_bytes(payload)

    actual = px.io.read_image(path, channels="R", unchanged=True)

    assert actual.data.dtype.name == "float16"
    np.testing.assert_array_equal(actual.data.get(), expected)


def test_multipart_channels_decode_from_their_own_part_and_preserve_requested_order(tmp_path: Path) -> None:
    """v1-exr-runtime-independence acceptance 22 and 23: each selected channel uses its owning part."""
    window = (-1, 2, 0, 3)
    headers = (
        _part_attributes(
            data_window=window,
            channels=(("R", 1, (1, 1)),),
            name="beauty",
            image_type="scanlineimage",
            chunk_count=2,
        ),
        _part_attributes(
            data_window=window,
            channels=(("G", 1, (1, 1)),),
            compression=2,
            line_order=2,
            name="fill",
            image_type="scanlineimage",
            chunk_count=2,
        ),
    )
    payload = _multipart_file(
        headers,
        (
            (_scanline_chunk(0, 2, (1.0, 2.0)), _scanline_chunk(0, 3, (3.0, 4.0))),
            (_scanline_chunk(1, 2, (10.0, 20.0)), _scanline_chunk(1, 3, (30.0, 40.0))),
        ),
    )
    path = tmp_path / "multipart-flat.exr"
    path.write_bytes(payload)

    actual = px.io.read_image(path, channels=("fill.G", "R"), unchanged=True)

    expected = np.asarray([[[10.0, 1.0], [20.0, 2.0]], [[30.0, 3.0], [40.0, 4.0]]], dtype=np.float16)
    np.testing.assert_array_equal(actual.data.get(), expected)
    assert actual.channels == ("fill.G", "R")


def test_default_selection_ignores_deep_duplicates_and_explicit_deep_selection_is_actionable(tmp_path: Path) -> None:
    """v1-exr-runtime-independence acceptance 24 and 26: deep affects only explicit deep reads."""
    window = (0, 0, 0, 0)
    flat_header = _part_attributes(
        data_window=window,
        channels=(("B", 1, (1, 1)), ("G", 1, (1, 1)), ("R", 1, (1, 1))),
        name="flat",
        image_type="scanlineimage",
        chunk_count=1,
    )
    deep_header = _part_attributes(
        data_window=window,
        channels=(("B", 1, (1, 1)), ("G", 1, (1, 1)), ("R", 1, (1, 1))),
        name="deep",
        image_type="deepscanline",
        chunk_count=1,
        deep=True,
    )
    flat_payload = np.asarray((3.0, 2.0, 1.0), dtype="<f2").tobytes()
    flat_chunk = struct.pack("<iii", 0, 0, len(flat_payload)) + flat_payload
    deep_chunk = struct.pack("<iiQQQ", 1, 0, 0, 0, 0)
    path = tmp_path / "mixed-deep-flat.exr"
    path.write_bytes(_multipart_file((flat_header, deep_header), ((flat_chunk,), (deep_chunk,)), non_image=True))

    actual = px.io.read_image(path, unchanged=True)

    np.testing.assert_array_equal(actual.data.get(), np.asarray([[[1.0, 2.0, 3.0]]], dtype=np.float16))
    with pytest.raises(ValueError) as caught:
        px.io.read_image(path, channels=("deep.R",), unchanged=True)
    message = str(caught.value)
    assert "deep" in message and "R" in message and "unsupported" in message
    assert "why=" in message and "what=" in message and "how=" in message


def _build_subsampled_fixture() -> tuple[bytes, dict[str, np.ndarray]]:
    window = (-3, 2, 2, 3)
    channels = (("A", 1, (2, 2)), ("B", 1, (2, 2)), ("C", 1, (1, 1)))
    header = struct.pack("<II", 20000630, 2) + _part_attributes(data_window=window, channels=channels)
    rows = (
        np.asarray((1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0), dtype="<f2"),
        np.asarray((106.0, 107.0, 108.0, 109.0, 110.0, 111.0), dtype="<f2"),
    )
    table_start = len(header)
    chunk_start = table_start + 16
    chunks = []
    offsets = []
    for y, row in zip((2, 3), rows, strict=True):
        payload = row.tobytes()
        offsets.append(chunk_start)
        chunk = struct.pack("<ii", y, len(payload)) + payload
        chunks.append(chunk)
        chunk_start += len(chunk)
    table = b"".join(struct.pack("<Q", offset) for offset in offsets)
    expected = {
        "A": np.asarray([[[1.0], [2.0], [3.0]]], dtype=np.float16),
        "B": np.asarray([[[10.0], [20.0], [30.0]]], dtype=np.float16),
        "C": np.arange(100.0, 112.0, dtype=np.float16).reshape(2, 6, 1),
    }
    return header + table + b"".join(chunks), expected


def _rle_transform(raw: bytes) -> bytes:
    reordered = bytearray(raw[::2] + raw[1::2])
    predicted = bytearray(reordered)
    for index in range(len(reordered) - 1, 0, -1):
        predicted[index] = (reordered[index] - reordered[index - 1] + 128) & 0xFF
    return bytes(predicted)


def _rle_packets(transformed: bytes) -> bytes:
    encoded = bytearray()
    cursor = 0
    while cursor < len(transformed):
        run_end = cursor + 1
        while run_end < len(transformed) and transformed[run_end] == transformed[cursor] and run_end - cursor < 128:
            run_end += 1
        if run_end - cursor >= 3:
            encoded.extend((run_end - cursor - 1, transformed[cursor]))
            cursor = run_end
            continue
        literal_start = cursor
        cursor = run_end
        while cursor < len(transformed) and cursor - literal_start < 128:
            candidate_end = cursor + 1
            while (
                candidate_end < len(transformed)
                and transformed[candidate_end] == transformed[cursor]
                and candidate_end - cursor < 128
            ):
                candidate_end += 1
            if candidate_end - cursor >= 3:
                break
            cursor = candidate_end
        literal = transformed[literal_start:cursor]
        encoded.append((-len(literal)) & 0xFF)
        encoded.extend(literal)
    return bytes(encoded)


def _build_rle_subsampled_fixture() -> tuple[bytes, dict[str, np.ndarray]]:
    window = (-3, 2, 2, 3)
    channels = (("A", 1, (2, 2)), ("B", 1, (2, 2)), ("C", 1, (1, 1)))
    header = struct.pack("<II", 20000630, 2) + _part_attributes(
        data_window=window,
        channels=channels,
        compression=1,
    )
    rows = (
        np.asarray((0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0), dtype="<f2"),
        np.asarray((3.0, 3.0, 3.0, 3.0, 3.0, 3.0), dtype="<f2"),
    )
    table_start = len(header)
    chunk_start = table_start + 16
    chunks = []
    offsets = []
    for y, row in zip((2, 3), rows, strict=True):
        raw = row.tobytes()
        payload = _rle_packets(_rle_transform(raw))
        assert len(payload) < len(raw)
        offsets.append(chunk_start)
        chunk = struct.pack("<ii", y, len(payload)) + payload
        chunks.append(chunk)
        chunk_start += len(chunk)
    table = b"".join(struct.pack("<Q", offset) for offset in offsets)
    expected = {
        "A": np.zeros((1, 3, 1), dtype=np.float16),
        "B": np.ones((1, 3, 1), dtype=np.float16),
        "C": np.full((2, 6, 1), 3.0, dtype=np.float16),
    }
    return header + table + b"".join(chunks), expected


def test_subsampled_channels_materialize_only_stored_lattice_samples(tmp_path: Path) -> None:
    """v1-exr-runtime-independence acceptance 27 and 28: saved samples retain their real HWC shape."""
    payload, expected = _build_subsampled_fixture()
    path = tmp_path / "subsampled.exr"
    path.write_bytes(payload)

    single = px.io.read_image(path, channels="A", unchanged=True)
    stacked = px.io.read_image(path, channels=("B", "A"), unchanged=True)

    np.testing.assert_array_equal(single.data.get(), expected["A"])
    np.testing.assert_array_equal(stacked.data.get(), np.concatenate((expected["B"], expected["A"]), axis=2))
    with pytest.raises(ValueError) as caught:
        px.io.read_image(path, channels=("A", "C"), unchanged=True)
    message = str(caught.value)
    assert "A" in message and "C" in message and "(1, 3)" in message and "(2, 6)" in message
    assert "why=" in message and "what=" in message and "how=" in message


def test_compressed_subsampled_channels_materialize_without_openexr_fallback(tmp_path: Path) -> None:
    """v1-exr-runtime-independence acceptance 4, 27, and 28: compressed samples retain their real lattice."""
    payload, expected = _build_rle_subsampled_fixture()
    path = tmp_path / "subsampled-rle.exr"
    path.write_bytes(payload)

    container = exr_container._parse_exr_container(path)
    assert all(not chunk.raw_stored for chunk in container.parts[0].chunks)

    single = px.io.read_image(path, channels="A", unchanged=True)
    stacked = px.io.read_image(path, channels=("B", "A"), unchanged=True)

    assert single.data.shape == (1, 3, 1)
    np.testing.assert_array_equal(single.data.get(), expected["A"])
    np.testing.assert_array_equal(stacked.data.get(), np.concatenate((expected["B"], expected["A"]), axis=2))
