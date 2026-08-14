"""Specification tests for the runtime-independent EXR container model."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

import pytest

import pixtreme as px
from pixtreme._io.formats.exr.container import _parse_exr_container


def _attribute(name: str, attribute_type: str, payload: bytes) -> bytes:
    return name.encode() + b"\x00" + attribute_type.encode() + b"\x00" + struct.pack("<I", len(payload)) + payload


def _channel_list(
    channels: tuple[tuple[str, int, tuple[int, int]], ...] = (("R", 1, (1, 1)),),
) -> bytes:
    payload = bytearray()
    for name, pixel_type, sampling in channels:
        payload.extend(name.encode() + b"\x00")
        payload.extend(struct.pack("<iB3xii", pixel_type, 0, *sampling))
    payload.append(0)
    return bytes(payload)


def _part_attributes(
    *,
    data_window: tuple[int, int, int, int],
    channels: tuple[tuple[str, int, tuple[int, int]], ...] = (("R", 1, (1, 1)),),
    compression: int = 0,
    line_order: int = 0,
    name: str | None = None,
    image_type: str | None = None,
    chunk_count: int | None = None,
    tiles: tuple[int, int, int, int] | None = None,
    part_version: int | None = None,
    max_samples_per_pixel: int | None = None,
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
    if part_version is not None:
        attributes.append(_attribute("version", "int", struct.pack("<i", part_version)))
    if max_samples_per_pixel is not None:
        attributes.append(_attribute("maxSamplesPerPixel", "int", struct.pack("<i", max_samples_per_pixel)))
    if tiles is not None:
        x_size, y_size, level_mode, rounding_mode = tiles
        attributes.append(
            _attribute("tiles", "tiledesc", struct.pack("<IIB", x_size, y_size, level_mode | 16 * rounding_mode))
        )
    return b"".join(attributes) + b"\x00"


def _container_header(version_flags: int, *parts: bytes) -> bytes:
    multipart_terminator = b"\x00" if version_flags & 0x1000 else b""
    return struct.pack("<II", 20000630, 2 | version_flags) + b"".join(parts) + multipart_terminator


def _assert_actionable_identity(error: BaseException, *identity: str) -> None:
    message = str(error)
    assert message.startswith("why=")
    assert "; what=" in message
    assert "; how=" in message
    for fragment in identity:
        assert fragment in message


@dataclass(frozen=True)
class _MultipartFixture:
    payload: bytes
    part_offsets: tuple[tuple[int, ...], ...]


def _build_multipart_sampling_deep_fixture() -> _MultipartFixture:
    flat_window = (-3, 2, 2, 3)
    deep_window = (10, 20, 10, 20)
    flat_header = _part_attributes(
        data_window=flat_window,
        channels=(("R", 1, (2, 2)),),
        line_order=1,
        name="flat",
        image_type="scanlineimage",
        chunk_count=2,
    )
    deep_header = _part_attributes(
        data_window=deep_window,
        channels=(("Z", 2, (1, 1)),),
        compression=2,
        name="deep",
        image_type="deepscanline",
        chunk_count=1,
        part_version=1,
        max_samples_per_pixel=1,
    )
    header = _container_header(0x800 | 0x1000, flat_header, deep_header)

    stored_samples = struct.pack("<3H", 0x3C00, 0x4000, 0x4200)
    flat_payloads = {2: stored_samples, 3: b""}
    flat_chunks = {y: struct.pack("<iii", 0, y, len(flat_payloads[y])) + flat_payloads[y] for y in (2, 3)}
    deep_chunk = struct.pack("<iiQQQ", 1, 20, 0, 0, 0)
    physical_chunks = (("deep", deep_chunk), ("flat-3", flat_chunks[3]), ("flat-2", flat_chunks[2]))
    cursor = len(header) + 3 * 8
    offsets: dict[str, int] = {}
    chunk_blob = bytearray()
    for key, chunk in physical_chunks:
        offsets[key] = cursor
        chunk_blob.extend(chunk)
        cursor += len(chunk)
    part_offsets = ((offsets["flat-2"], offsets["flat-3"]), (offsets["deep"],))
    tables = b"".join(struct.pack("<Q", offset) for table in part_offsets for offset in table)
    return _MultipartFixture(payload=header + tables + bytes(chunk_blob), part_offsets=part_offsets)


def _level_size(size: int, level: int, rounding_mode: int) -> int:
    divisor = 1 << level
    return max(1, math.ceil(size / divisor) if rounding_mode else size // divisor)


def _level_count(size: int, rounding_mode: int) -> int:
    count = 1
    while _level_size(size, count - 1, rounding_mode) > 1:
        count += 1
    return count


@dataclass(frozen=True)
class _TiledFixture:
    payload: bytes
    table_start: int
    identities: tuple[tuple[int, int, int, int], ...]
    offsets: tuple[int, ...]


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


def _build_tiled_fixture(
    *,
    level_mode: int,
    rounding_mode: int,
    reverse_physical: bool = True,
    compression: int = 0,
) -> _TiledFixture:
    width, height = (5, 3)
    tile_width, tile_height = (2, 2)
    identities = _tile_identities(
        width=width,
        height=height,
        tile_size=(tile_width, tile_height),
        level_mode=level_mode,
        rounding_mode=rounding_mode,
    )
    attributes = _part_attributes(
        data_window=(-2, 7, 2, 9),
        compression=compression,
        tiles=(tile_width, tile_height, level_mode, rounding_mode),
    )
    header = struct.pack("<II", 20000630, 2 | 0x200) + attributes
    chunks: dict[tuple[int, int, int, int], bytes] = {}
    for tile_x, tile_y, level_x, level_y in identities:
        level_width = _level_size(width, level_x, rounding_mode)
        level_height = _level_size(height, level_y, rounding_mode)
        stored_width = min(tile_width, level_width - tile_x * tile_width)
        stored_height = min(tile_height, level_height - tile_y * tile_height)
        packed_size = stored_width * stored_height * 2
        chunks[(tile_x, tile_y, level_x, level_y)] = (
            struct.pack("<iiiii", tile_x, tile_y, level_x, level_y, packed_size)
            + bytes((level_x + level_y + 1,)) * packed_size
        )
    physical_identities = tuple(reversed(identities)) if reverse_physical else identities
    cursor = len(header) + len(identities) * 8
    offsets_by_identity: dict[tuple[int, int, int, int], int] = {}
    chunk_blob = bytearray()
    for identity in physical_identities:
        offsets_by_identity[identity] = cursor
        chunk_blob.extend(chunks[identity])
        cursor += len(chunks[identity])
    offsets = tuple(offsets_by_identity[identity] for identity in identities)
    table = b"".join(struct.pack("<Q", offset) for offset in offsets)
    return _TiledFixture(
        payload=header + table + bytes(chunk_blob), table_start=len(header), identities=identities, offsets=offsets
    )


def test_multipart_parts_own_chunks_deep_state_and_sampling_geometry(tmp_path: Path) -> None:
    """v1-exr-runtime-independence acceptance 22, 25, and 29: part and lattice ownership is explicit."""
    fixture = _build_multipart_sampling_deep_fixture()
    path = tmp_path / "multipart-sampling-deep.exr"
    path.write_bytes(fixture.payload)

    container = _parse_exr_container(path)
    header = px.io.read_header(path)

    flat, deep = container.parts
    assert (flat.name, flat.compression, flat.line_order, flat.data_window, flat.deep) == (
        "flat",
        "none",
        1,
        (-3, 2, 2, 3),
        False,
    )
    assert (deep.name, deep.compression, deep.line_order, deep.data_window, deep.deep) == (
        "deep",
        "zips",
        0,
        (10, 20, 10, 20),
        True,
    )
    assert (flat.offset_table, deep.offset_table) == fixture.part_offsets
    assert tuple((chunk.part_index, chunk.y) for chunk in flat.chunks) == ((0, 2), (0, 3))
    assert tuple((chunk.part_index, chunk.y) for chunk in deep.chunks) == ((1, 20),)
    assert tuple(container.data[chunk.payload_start : chunk.payload_end] for chunk in flat.chunks) == (
        struct.pack("<3H", 0x3C00, 0x4000, 0x4200),
        b"",
    )
    sampling = flat.channels[0].sampling
    assert sampling is not None
    assert tuple(sampling.x_coordinates) == (-2, 0, 2)
    assert tuple(sampling.y_coordinates) == (2,)
    assert sampling.shape == (1, 3)
    assert tuple((part.name, part.deep) for part in header.parts) == (("flat", False), ("deep", True))


@pytest.mark.parametrize(
    ("missing", "name", "image_type", "chunk_count"),
    (
        ("name", None, "scanlineimage", 1),
        ("type", "flat", None, 1),
        ("chunkCount", "flat", "scanlineimage", None),
    ),
)
@pytest.mark.parametrize("version_flags", (0x1000, 0x800))
def test_multipart_or_non_image_required_attributes_are_not_defaulted(
    tmp_path: Path,
    missing: str,
    name: str | None,
    image_type: str | None,
    chunk_count: int | None,
    version_flags: int,
) -> None:
    """v1-exr-runtime-independence acceptance 22, 25, and 42: extended layout is explicit."""
    first = _part_attributes(
        data_window=(0, 0, 0, 0),
        name=name,
        image_type=image_type,
        chunk_count=chunk_count,
    )
    parts = (first,)
    if version_flags & 0x1000:
        parts += (
            _part_attributes(
                data_window=(0, 0, 0, 0),
                name="right",
                image_type="scanlineimage",
                chunk_count=1,
            ),
        )
    path = tmp_path / f"extended-layout-{version_flags}-missing-{missing}.exr"
    path.write_bytes(_container_header(version_flags, *parts))

    with pytest.raises(RuntimeError) as caught:
        px.io.read_header(path)

    _assert_actionable_identity(caught.value, "part=0", f"attribute={missing!r}")


@pytest.mark.parametrize(
    ("version_flags", "image_type", "tiles"),
    (
        (0x1000, "volume", None),
        (0x800, "scanlineimage", None),
        (0, "deepscanline", None),
        (0x1000, "deepscanline", None),
        (0x200, "scanlineimage", (1, 1, 0, 0)),
    ),
)
def test_part_type_and_version_flags_must_describe_the_same_layout(
    tmp_path: Path,
    version_flags: int,
    image_type: str,
    tiles: tuple[int, int, int, int] | None,
) -> None:
    """v1-exr-runtime-independence acceptance 25 and 42: flags and part type cannot contradict."""
    first = _part_attributes(
        data_window=(0, 0, 0, 0),
        name="left",
        image_type=image_type,
        chunk_count=1,
        tiles=tiles,
        part_version=1,
        max_samples_per_pixel=1,
    )
    parts = (first,)
    if version_flags & 0x1000:
        parts += (
            _part_attributes(
                data_window=(0, 0, 0, 0),
                name="right",
                image_type="scanlineimage",
                chunk_count=1,
            ),
        )
    path = tmp_path / f"flag-type-{version_flags}-{image_type}.exr"
    path.write_bytes(_container_header(version_flags, *parts))

    with pytest.raises(RuntimeError) as caught:
        px.io.read_header(path)

    _assert_actionable_identity(caught.value, "part=0", f"type={image_type!r}")


@pytest.mark.parametrize(
    ("version_flags", "image_type", "part_version", "max_samples_per_pixel"),
    ((0, "scanlineimage", None, None), (0x800, "deepscanline", 1, 1)),
)
def test_non_tiled_parts_must_not_carry_a_tiles_attribute(
    tmp_path: Path,
    version_flags: int,
    image_type: str,
    part_version: int | None,
    max_samples_per_pixel: int | None,
) -> None:
    """v1-exr-runtime-independence acceptance 22 and 42: a stray tiles attribute contradicts non-tiled part types."""
    first = _part_attributes(
        data_window=(0, 0, 3, 3),
        name="left",
        image_type=image_type,
        chunk_count=4 if image_type == "deepscanline" else None,
        tiles=(2, 2, 0, 0),
        part_version=part_version,
        max_samples_per_pixel=max_samples_per_pixel,
    )
    path = tmp_path / f"stray-tiles-{image_type}.exr"
    path.write_bytes(_container_header(version_flags, first))

    with pytest.raises(RuntimeError) as caught:
        px.io.read_header(path)

    _assert_actionable_identity(caught.value, "part=0", f"type={image_type!r}", "attribute='tiles'")


@pytest.mark.parametrize(
    ("version_flags", "image_type"),
    ((0x200, "tiledimage"), (0x1000, "tiledimage"), (0x800, "deeptile")),
)
def test_tiled_part_requires_a_tile_description(
    tmp_path: Path,
    version_flags: int,
    image_type: str,
) -> None:
    """v1-exr-runtime-independence acceptance 17 and 42: tiled part layout requires tiles."""
    first = _part_attributes(
        data_window=(0, 0, 0, 0),
        name="tiles",
        image_type=image_type,
        chunk_count=1,
        part_version=1 if image_type == "deeptile" else None,
        max_samples_per_pixel=1 if image_type == "deeptile" else None,
    )
    parts = (first,)
    if version_flags & 0x1000:
        parts += (
            _part_attributes(
                data_window=(0, 0, 0, 0),
                name="flat",
                image_type="scanlineimage",
                chunk_count=1,
            ),
        )
    path = tmp_path / f"missing-tiles-{image_type}.exr"
    path.write_bytes(_container_header(version_flags, *parts))

    with pytest.raises(RuntimeError) as caught:
        px.io.read_header(path)

    _assert_actionable_identity(caught.value, "part=0", "attribute='tiles'")


@pytest.mark.parametrize("missing", ("version", "maxSamplesPerPixel"))
def test_deep_part_requires_its_deep_header_attributes(tmp_path: Path, missing: str) -> None:
    """v1-exr-runtime-independence acceptance 25 and 42: deep headers carry their mandatory metadata."""
    part = _part_attributes(
        data_window=(0, 0, 0, 0),
        name="deep",
        image_type="deepscanline",
        chunk_count=1,
        part_version=None if missing == "version" else 1,
        max_samples_per_pixel=None if missing == "maxSamplesPerPixel" else 1,
    )
    path = tmp_path / f"deep-missing-{missing}.exr"
    path.write_bytes(_container_header(0x800, part))

    with pytest.raises(RuntimeError) as caught:
        px.io.read_header(path)

    _assert_actionable_identity(caught.value, "part=0", f"attribute={missing!r}")


@pytest.mark.parametrize(
    ("level_mode", "rounding_mode", "expected_levels", "expected_count"),
    (
        (0, 0, ((0, 0, 5, 3, 3, 2),), 6),
        (
            1,
            1,
            ((0, 0, 5, 3, 3, 2), (1, 1, 3, 2, 2, 1), (2, 2, 2, 1, 1, 1), (3, 3, 1, 1, 1, 1)),
            10,
        ),
        (
            2,
            0,
            (
                (0, 0, 5, 3, 3, 2),
                (1, 0, 2, 3, 1, 2),
                (2, 0, 1, 3, 1, 2),
                (0, 1, 5, 1, 3, 1),
                (1, 1, 2, 1, 1, 1),
                (2, 1, 1, 1, 1, 1),
            ),
            15,
        ),
    ),
)
def test_tiled_level_geometry_derives_every_grid_and_offset_entry(
    tmp_path: Path,
    level_mode: int,
    rounding_mode: int,
    expected_levels: tuple[tuple[int, int, int, int, int, int], ...],
    expected_count: int,
) -> None:
    """v1-exr-runtime-independence acceptance 17: tiledesc alone determines every level grid and table entry."""
    fixture = _build_tiled_fixture(level_mode=level_mode, rounding_mode=rounding_mode)
    path = tmp_path / f"levels-{level_mode}-{rounding_mode}.exr"
    path.write_bytes(fixture.payload)

    part = _parse_exr_container(path).parts[0]

    assert part.tile_description is not None
    assert (part.tile_description.x_size, part.tile_description.y_size) == (2, 2)
    assert (
        tuple(
            (level.level_x, level.level_y, level.width, level.height, level.tile_columns, level.tile_rows)
            for level in part.levels
        )
        == expected_levels
    )
    assert part.expected_chunk_count == expected_count == len(part.offset_table) == len(part.chunks)
    assert (
        tuple((chunk.tile_x, chunk.tile_y, chunk.level_x, chunk.level_y) for chunk in part.chunks) == fixture.identities
    )


@pytest.mark.parametrize(
    ("corruption", "expected_context"),
    (
        ("duplicate-offset", "previous_owner="),
        ("level-identity", "observed_level="),
        ("truncated-payload", "file_size="),
    ),
)
def test_tiled_level_corruption_reports_part_level_tile_and_action(
    tmp_path: Path,
    corruption: str,
    expected_context: str,
) -> None:
    """v1-exr-runtime-independence acceptance 21: tile identity and truncation corruption is actionable."""
    fixture = _build_tiled_fixture(level_mode=1, rounding_mode=1, reverse_physical=False)
    payload = bytearray(fixture.payload)
    final_offset = fixture.offsets[-1]
    if corruption == "duplicate-offset":
        struct.pack_into("<Q", payload, fixture.table_start + 8, fixture.offsets[0])
    elif corruption == "level-identity":
        struct.pack_into("<ii", payload, final_offset + 8, 0, 0)
    else:
        payload.pop()
    path = tmp_path / f"{corruption}.exr"
    path.write_bytes(payload)

    with pytest.raises(RuntimeError) as caught:
        px.io.read_header(path)

    _assert_actionable_identity(caught.value, "part=0", expected_context)


def test_nonselected_tiled_level_offsets_are_validated_eagerly(tmp_path: Path) -> None:
    """v1-exr-runtime-independence acceptance 20 and 21: nonzero levels get bounds and span validation."""
    fixture = _build_tiled_fixture(level_mode=1, rounding_mode=1)
    payload = bytearray(fixture.payload)
    struct.pack_into("<Q", payload, fixture.table_start + (len(fixture.offsets) - 1) * 8, len(payload) - 2)
    path = tmp_path / "nonselected-level-out-of-bounds.exr"
    path.write_bytes(payload)

    with pytest.raises(RuntimeError) as caught:
        px.io.read_header(path)

    _assert_actionable_identity(caught.value, "part=0", "level=(3, 3)", "tile=(0, 0)")


def test_tiled_offset_table_truncation_is_rejected_before_chunk_parsing(tmp_path: Path) -> None:
    """v1-exr-runtime-independence acceptance 20 and 42: every declared table entry is complete."""
    fixture = _build_tiled_fixture(level_mode=1, rounding_mode=1)
    table_end = fixture.table_start + len(fixture.offsets) * 8
    path = tmp_path / "truncated-offset-table.exr"
    path.write_bytes(fixture.payload[: table_end - 1])

    with pytest.raises(RuntimeError) as caught:
        px.io.read_header(path)

    _assert_actionable_identity(caught.value, f"tables={fixture.table_start}:{table_end}")


def test_tiled_chunk_spans_must_not_intersect(tmp_path: Path) -> None:
    """v1-exr-runtime-independence acceptance 20 and 42: chunk file spans are disjoint."""
    fixture = _build_tiled_fixture(level_mode=1, rounding_mode=1, reverse_physical=False, compression=2)
    payload = bytearray(fixture.payload)
    first_offset, second_offset = fixture.offsets[:2]
    packed_size = struct.unpack_from("<i", payload, first_offset + 16)[0]
    assert first_offset + 20 + packed_size == second_offset
    struct.pack_into("<i", payload, first_offset + 16, packed_size + 1)
    path = tmp_path / "intersecting-chunk-spans.exr"
    path.write_bytes(payload)

    with pytest.raises(RuntimeError) as caught:
        px.io.read_header(path)

    _assert_actionable_identity(caught.value, "previous_part=0", "current_part=0", "previous_span=", "current_span=")
