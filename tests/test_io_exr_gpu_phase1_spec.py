"""Specification tests for the Phase 1 GPU OpenEXR read boundary."""

from __future__ import annotations

import os
import struct
import subprocess
import sys
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import exr_test_harness as exr_harness
import numpy as np
import pytest

import pixtreme as px
import pixtreme._io.formats.exr.codec_none as exr_none
import pixtreme._io.formats.exr.packing as exr_packing
import pixtreme._io.formats.exr.selection as io
import pixtreme._io.header as io_header

ROOT = Path(__file__).resolve().parents[1]
_COMPRESSION_CODES = {"none": 0, "zips": 2, "zip": 3}
_PIXEL_TYPES = {np.dtype(np.float16): 1, np.dtype(np.float32): 2}


@dataclass(frozen=True)
class _ExrFixture:
    payload: bytes
    channels: Mapping[str, np.ndarray]
    table_offset: int
    chunk_offsets: tuple[int, ...]
    payload_offsets: tuple[int, ...]
    packed_sizes: tuple[int, ...]


def _attribute(name: str, attribute_type: str, payload: bytes) -> bytes:
    return name.encode() + b"\x00" + attribute_type.encode() + b"\x00" + struct.pack("<I", len(payload)) + payload


def _channel_list(channels: Mapping[str, np.ndarray], sampling: tuple[int, int]) -> bytes:
    payload = bytearray()
    for name in sorted(channels):
        pixel_type = _PIXEL_TYPES[np.dtype(channels[name].dtype)]
        payload.extend(name.encode() + b"\x00")
        payload.extend(struct.pack("<iB3xii", pixel_type, 0, *sampling))
    payload.append(0)
    return bytes(payload)


def _zip_transform(raw: bytes) -> bytes:
    """Independent host implementation of the OpenEXR ZIP byte transforms."""
    reordered = bytearray(raw[::2] + raw[1::2])
    predicted = bytearray(reordered)
    for index in range(len(reordered) - 1, 0, -1):
        predicted[index] = (reordered[index] - reordered[index - 1] + 128) & 0xFF
    return bytes(predicted)


def _restore_predictor_scalar_oracle(transformed: np.ndarray) -> np.ndarray:
    """Restore predictor bytes with the OpenEXR scalar recurrence, independent of production helpers."""
    values = np.asarray(transformed, dtype=np.uint8)
    restored = np.empty_like(values)
    if values.size:
        restored[0] = values[0]
    for index in range(1, values.size):
        restored[index] = (int(restored[index - 1]) + int(values[index]) - 128) & 0xFF
    return restored


def _chunk_bytes(channels: Mapping[str, np.ndarray], first_row: int, row_count: int) -> bytes:
    payload = bytearray()
    for row in range(first_row, first_row + row_count):
        for name in sorted(channels):
            dtype = np.dtype(channels[name].dtype).newbyteorder("<")
            payload.extend(np.asarray(channels[name][row], dtype=dtype).tobytes())
    return bytes(payload)


def _build_exr(
    channels: Mapping[str, np.ndarray],
    *,
    compression: str,
    storage: str | Sequence[str] = "raw",
    origin: tuple[int, int] = (0, 0),
    line_order: int = 0,
    sampling: tuple[int, int] = (1, 1),
    reverse_physical_chunks: bool = False,
) -> _ExrFixture:
    """Build a minimal scanline EXR without using the OpenEXR codec."""
    shapes = {array.shape for array in channels.values()}
    assert len(shapes) == 1
    height, width = shapes.pop()
    x_min, y_min = origin
    data_window = (x_min, y_min, x_min + width - 1, y_min + height - 1)
    attributes = (
        _attribute("channels", "chlist", _channel_list(channels, sampling)),
        _attribute("compression", "compression", bytes((_COMPRESSION_CODES[compression],))),
        _attribute("dataWindow", "box2i", struct.pack("<iiii", *data_window)),
        _attribute("displayWindow", "box2i", struct.pack("<iiii", *data_window)),
        _attribute("lineOrder", "lineOrder", bytes((line_order,))),
        _attribute("pixelAspectRatio", "float", struct.pack("<f", 1.0)),
        _attribute("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0)),
        _attribute("screenWindowWidth", "float", struct.pack("<f", 1.0)),
    )
    header = struct.pack("<II", 20000630, 2) + b"".join(attributes) + b"\x00"
    lines_per_chunk = 16 if compression == "zip" else 1
    row_starts = tuple(range(0, height, lines_per_chunk))
    storage_kinds = (storage,) * len(row_starts) if isinstance(storage, str) else tuple(storage)
    assert len(storage_kinds) == len(row_starts)

    logical_chunks: list[tuple[int, bytes]] = []
    for row_start, storage_kind in zip(row_starts, storage_kinds, strict=True):
        row_count = min(lines_per_chunk, height - row_start)
        raw = _chunk_bytes(channels, row_start, row_count)
        if compression != "none" and storage_kind == "deflate":
            packed = zlib.compress(_zip_transform(raw))
            assert len(packed) < len(raw), "fixture must exercise the Deflate branch"
        else:
            packed = raw
        logical_chunks.append((y_min + row_start, struct.pack("<ii", y_min + row_start, len(packed)) + packed))

    physical_chunks = list(reversed(logical_chunks)) if reverse_physical_chunks else logical_chunks
    table_offset = len(header)
    cursor = table_offset + 8 * len(logical_chunks)
    offsets_by_y: dict[int, int] = {}
    physical_blob = bytearray()
    payload_offsets_by_y: dict[int, int] = {}
    packed_sizes_by_y: dict[int, int] = {}
    for y, chunk in physical_chunks:
        offsets_by_y[y] = cursor
        payload_offsets_by_y[y] = cursor + 8
        packed_sizes_by_y[y] = len(chunk) - 8
        physical_blob.extend(chunk)
        cursor += len(chunk)
    logical_y = tuple(y_min + row_start for row_start in row_starts)
    offset_table = b"".join(struct.pack("<Q", offsets_by_y[y]) for y in logical_y)
    return _ExrFixture(
        payload=header + offset_table + bytes(physical_blob),
        channels=channels,
        table_offset=table_offset,
        chunk_offsets=tuple(offsets_by_y[y] for y in logical_y),
        payload_offsets=tuple(payload_offsets_by_y[y] for y in logical_y),
        packed_sizes=tuple(packed_sizes_by_y[y] for y in logical_y),
    )


def _write_fixture(path: Path, fixture: _ExrFixture) -> None:
    path.write_bytes(fixture.payload)


def _compressible_rgb(height: int, width: int, dtype: type[np.generic] = np.float32) -> dict[str, np.ndarray]:
    y, x = np.mgrid[:height, :width]
    return {
        "R": np.asarray((x % 4) * 0.25, dtype=dtype),
        "G": np.asarray((y % 3) * -0.5, dtype=dtype),
        "B": np.asarray(1.25 + (x + y) % 2, dtype=dtype),
    }


def test_host_predictor_restore_matches_hand_calculated_fixture() -> None:
    """v1-exr-gpu-phase1 acceptance 27: the host predictor oracle is fixed by hand-calculated bytes."""
    transformed = np.asarray((10, 20, 250, 1), dtype=np.uint8)

    restored = exr_packing._restore_predictor_host(transformed)

    np.testing.assert_array_equal(restored, np.asarray((10, 158, 24, 153), dtype=np.uint8))


def test_gpu_segmented_predictor_restore_matches_independent_host_oracle() -> None:
    """v1-exr-gpu-phase1 acceptance 11 and 27: segmented GPU restore matches the fixed host oracle."""
    import cupy as cp

    first = np.resize(np.asarray((10, 20, 250, 1), dtype=np.uint8), 5003)
    second = np.resize(np.asarray((4, 128, 255, 0, 7), dtype=np.uint8), 9317)
    raw = np.asarray((3, 1, 4, 1, 5, 9), dtype=np.uint8)
    host = np.concatenate((first, second, raw))
    offsets = np.asarray((0, first.size, first.size + second.size), dtype=np.int64)
    sizes = np.asarray((first.size, second.size, raw.size), dtype=np.int64)
    compressed = np.asarray((1, 1, 0), dtype=np.uint8)
    expected_adler = np.asarray((zlib.adler32(first), zlib.adler32(second), 0), dtype=np.uint32)
    device = cp.asarray(host)

    exr_packing._restore_exr_gpu_chunks(device, offsets, sizes, compressed, expected_adler)

    expected = np.concatenate((_restore_predictor_scalar_oracle(first), _restore_predictor_scalar_oracle(second), raw))
    np.testing.assert_array_equal(device.get(), expected)


def test_gpu_restore_reports_adler_mismatch_chunk_and_accepts_valid_checksums() -> None:
    """v1-exr-gpu-phase1 acceptance 11 and 27: the GPU Adler-32 status flags exactly the mismatched chunk."""
    import cupy as cp

    first = np.resize(np.asarray((10, 20, 250, 1), dtype=np.uint8), 5003)
    second = np.resize(np.asarray((4, 128, 255, 0, 7), dtype=np.uint8), 9317)
    host = np.concatenate((first, second))
    offsets = np.asarray((0, first.size), dtype=np.int64)
    sizes = np.asarray((first.size, second.size), dtype=np.int64)
    compressed = np.asarray((1, 1), dtype=np.uint8)
    valid_adler = np.asarray((zlib.adler32(first), zlib.adler32(second)), dtype=np.uint32)

    valid_failures = exr_packing._restore_exr_gpu_chunks(cp.asarray(host), offsets, sizes, compressed, valid_adler)

    assert valid_failures.size == 0

    corrupted_adler = valid_adler.copy()
    corrupted_adler[1] ^= np.uint32(1)

    failed = exr_packing._restore_exr_gpu_chunks(cp.asarray(host), offsets, sizes, compressed, corrupted_adler)

    np.testing.assert_array_equal(failed, np.asarray((1,), dtype=np.int64))


def test_container_parser_describes_reordered_partial_zip_chunks_without_codec_import(tmp_path: Path) -> None:
    """v1-exr-gpu-phase1 acceptance 2, 6, 7, and 9: one parser validates the complete GPU candidate."""
    fixture = _build_exr(
        _compressible_rgb(17, 32),
        compression="zip",
        storage=("deflate", "raw"),
        origin=(-4, 9),
        line_order=2,
        reverse_physical_chunks=True,
    )
    path = tmp_path / "descriptor.exr"
    _write_fixture(path, fixture)
    codec_modules_before = {name for name in sys.modules if name == "OpenEXR" or name.startswith("nvidia.nvcomp")}

    descriptor = io_header._parse_exr(path)

    assert descriptor.gpu_eligible is True
    assert descriptor.compression == "zip"
    assert descriptor.line_order == 2
    assert descriptor.data_window == (-4, 9, 27, 25)
    assert descriptor.lines_per_chunk == 16
    assert descriptor.expected_chunk_count == 2
    assert descriptor.offset_table == fixture.chunk_offsets
    assert tuple((chunk.y, chunk.row_start, chunk.row_count) for chunk in descriptor.chunks) == (
        (9, 0, 16),
        (25, 16, 1),
    )
    assert tuple(chunk.expected_size for chunk in descriptor.chunks) == (16 * 32 * 3 * 4, 32 * 3 * 4)
    assert descriptor.offset_table[0] > descriptor.offset_table[1]
    channels = descriptor.parts[0].channels
    assert tuple((channel.name, channel.dtype, channel.x_sampling, channel.y_sampling) for channel in channels) == (
        ("B", "float32", 1, 1),
        ("G", "float32", 1, 1),
        ("R", "float32", 1, 1),
    )
    assert all(
        attribute.payload_start < attribute.payload_end <= len(fixture.payload)
        for attribute in descriptor.parts[0].attributes.values()
    )
    assert {
        name for name in sys.modules if name == "OpenEXR" or name.startswith("nvidia.nvcomp")
    } == codec_modules_before


@pytest.mark.parametrize("corruption", ("duplicate-offset", "duplicate-y", "truncated-payload"))
def test_container_corruption_fails_before_pixel_decode(tmp_path: Path, corruption: str) -> None:
    """v1-exr-gpu-phase1 acceptance 7 and 19: malformed candidate chunk topology is rejected eagerly."""
    fixture = _build_exr(_compressible_rgb(2, 16), compression="zips", storage=("deflate", "raw"))
    payload = bytearray(fixture.payload)
    if corruption == "duplicate-offset":
        struct.pack_into("<Q", payload, fixture.table_offset + 8, fixture.chunk_offsets[0])
    elif corruption == "duplicate-y":
        struct.pack_into("<i", payload, fixture.chunk_offsets[1], 0)
    else:
        payload.pop()
    path = tmp_path / f"{corruption}.exr"
    path.write_bytes(payload)

    with pytest.raises(RuntimeError, match=r"why=.*what=.*how="):
        px.io.read_header(path)


def test_unknown_version_flag_is_rejected_before_gpu_or_cpu_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v1-exr-gpu-phase1 acceptance 6 and 19: unknown v2 flags are malformed, not fallback candidates."""
    fixture = _build_exr(_compressible_rgb(1, 16), compression="none")
    payload = bytearray(fixture.payload)
    struct.pack_into("<I", payload, 4, 2 | 0x2000)
    path = tmp_path / "unknown-version-flag.exr"
    path.write_bytes(payload)
    backend_calls: list[str] = []

    def forbid_backend(*args: object, **kwargs: object) -> object:
        backend_calls.append("called")
        raise AssertionError("unknown EXR version flags must fail before pixel backend selection")

    monkeypatch.setattr(io, "_read_exr_gpu", forbid_backend)
    monkeypatch.setattr(io, "_read_exr_custom_cpu", forbid_backend)
    monkeypatch.setattr(io, "_read_exr_native", forbid_backend)

    with pytest.raises(RuntimeError, match=r"why=.*flag.*what=.*0x00002000.*how="):
        px.io.read_image(path)

    assert backend_calls == []


@pytest.mark.parametrize(
    ("compression", "height", "storage"),
    (("zips", 2, ("deflate", "raw")), ("zip", 17, ("deflate", "raw"))),
)
def test_public_gpu_read_handles_none_deflate_raw_and_partial_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compression: str,
    height: int,
    storage: str | Sequence[str],
) -> None:
    """v1-exr-gpu-phase1 acceptance 10, 11, 12, and 28: public reads use every Phase 1 GPU chunk path."""
    channels = _compressible_rgb(height, 32)
    fixture = _build_exr(
        channels,
        compression=compression,
        storage=storage,
        origin=(-7, 11) if compression == "zip" else (0, 0),
        line_order={"zips": 2, "zip": 0}[compression],
        reverse_physical_chunks=True,
    )
    path = tmp_path / f"{compression}.exr"
    _write_fixture(path, fixture)
    original = io._read_exr_gpu
    calls = 0

    def spy(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setitem(io._EXR_ROUTING, (compression, "read"), "gpu")
    monkeypatch.setattr(io, "_read_exr_gpu", spy)

    result = px.io.read_image(path, unchanged=True)

    expected = np.stack([channels[name] for name in "RGB"], axis=2)
    assert calls == 1
    assert result.channels == ("R", "G", "B")
    assert result.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(
        px.io.to_array(
            result,
        ).get(),
        expected,
    )


@pytest.mark.parametrize(
    ("compression", "height", "storage"),
    (("zips", 2, ("deflate", "raw")), ("zip", 17, ("deflate", "raw"))),
)
def test_public_custom_cpu_read_handles_none_deflate_raw_and_partial_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compression: str,
    height: int,
    storage: str | Sequence[str],
) -> None:
    """v1-exr-gpu-phase1 acceptance 5, 11, and 13: ZIP/ZIPS CPU inflate reuses the GPU byte restore."""
    channels = _compressible_rgb(height, 32)
    fixture = _build_exr(
        channels,
        compression=compression,
        storage=storage,
        origin=(-7, 11) if compression == "zip" else (0, 0),
        line_order={"zips": 2, "zip": 0}[compression],
        reverse_physical_chunks=True,
    )
    path = tmp_path / f"custom-cpu-{compression}.exr"
    _write_fixture(path, fixture)
    original = io._read_exr_custom_cpu
    calls = 0
    restore_original = exr_packing._restore_exr_gpu_chunks
    restore_calls = 0

    def spy(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    def restore_spy(*args: object, **kwargs: object) -> object:
        nonlocal restore_calls
        restore_calls += 1
        return restore_original(*args, **kwargs)

    monkeypatch.setitem(io._EXR_ROUTING, (compression, "read"), "custom_cpu")
    monkeypatch.setattr(io, "_read_exr_custom_cpu", spy)
    monkeypatch.setattr("pixtreme._io.formats.exr.codec_zip._restore_exr_gpu_chunks", restore_spy)

    result = px.io.read_image(path, unchanged=True)

    expected = np.stack([channels[name] for name in "RGB"], axis=2)
    assert calls == 1
    assert restore_calls == (1 if compression in ("zip", "zips") else 0)
    assert result.channels == ("R", "G", "B")
    assert result.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(
        px.io.to_array(
            result,
        ).get(),
        expected,
    )


def test_gpu_read_promotes_mixed_types_and_preserves_dotted_selection_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v1-exr-gpu-phase1 acceptance 12 and 27: dtype and dotted routing match independent source arrays."""
    height, width = 2, 32
    channels = {
        "B": np.full((height, width), -0.5, dtype=np.float16),
        "G": np.full((height, width), 2.0, dtype=np.float32),
        "R": np.full((height, width), 0.25, dtype=np.float16),
        "diffuse.R": np.asarray(np.mgrid[:height, :width][1] / 8.0, dtype=np.float32),
    }
    fixture = _build_exr(channels, compression="zips", storage=("deflate", "raw"))
    path = tmp_path / "mixed.exr"
    _write_fixture(path, fixture)
    monkeypatch.setitem(io._EXR_ROUTING, ("zips", "read"), "gpu")

    default = px.io.read_image(path)
    selected = px.io.read_image(path, channels=["diffuse.R", "B"])

    assert default.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(
        px.io.to_array(
            default,
        ).get(),
        np.stack([channels[name] for name in "RGB"], axis=2).astype(np.float32),
    )
    assert selected.channels == ("diffuse.R", "B")
    np.testing.assert_array_equal(
        px.io.to_array(
            selected,
        ).get(),
        np.stack([channels["diffuse.R"], channels["B"]], axis=2).astype(np.float32),
    )
    with pytest.raises(ValueError, match=r"why=.*mixed.*what=.*how=.*unchanged=False"):
        px.io.read_image(path, channels=["diffuse.R", "B"], unchanged=True)


def test_gpu_deflate_read_writes_uniform_half_directly_to_float16_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v1-exr-gpu-phase1 acceptance 11 and 12: compressed uniform HALF lands in final fp16 storage."""
    channels = _compressible_rgb(2, 64, np.float16)
    fixture = _build_exr(channels, compression="zips", storage=("deflate", "deflate"))
    path = tmp_path / "half-deflate.exr"
    _write_fixture(path, fixture)
    monkeypatch.setitem(io._EXR_ROUTING, ("zips", "read"), "gpu")

    result = px.io.read_image(path, unchanged=True)

    assert result.dtype == np.dtype(np.float16)
    np.testing.assert_array_equal(
        px.io.to_array(
            result,
        ).get(),
        np.stack([channels[name] for name in "RGB"], axis=2),
    )


@pytest.mark.parametrize("compression_name", ("NO_COMPRESSION", "ZIPS_COMPRESSION", "ZIP_COMPRESSION"))
def test_openexr_reference_files_use_gpu_and_preserve_scene_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, compression_name: str
) -> None:
    """v1-exr-gpu-phase1 acceptance 5 and 13: reference EXRs match independent channel arrays sample-for-sample."""
    from openexr_dev_oracle import OpenEXR

    values = np.array(
        [
            [-1.0, 0.0, 1.25, np.inf] * 8,
            [np.nan, -0.25, 4.0, -np.inf] * 8,
        ],
        dtype=np.float32,
    )
    channels = {"R": values, "G": values + np.float32(0.5), "B": values * np.float32(2.0)}
    path = tmp_path / f"reference-{compression_name}.exr"
    OpenEXR.File({"compression": getattr(OpenEXR, compression_name)}, dict(channels)).write(str(path))
    compression = {
        "NO_COMPRESSION": "none",
        "ZIP_COMPRESSION": "zip",
        "ZIPS_COMPRESSION": "zips",
    }[compression_name]
    route = io._EXR_ROUTING[(compression, "read")]
    original = {
        "native": io._read_exr_native,
        "custom_cpu": io._read_exr_custom_cpu,
    }[route]
    calls = 0

    def spy(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(io, f"_read_exr_{route}", spy)

    result = px.io.read_image(path, unchanged=True)

    assert calls == 1
    np.testing.assert_array_equal(
        px.io.to_array(
            result,
        ).get(),
        np.stack([channels[name] for name in "RGB"], axis=2),
    )


def test_piz_reference_file_uses_the_source_fixed_gpu_route(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """v1-exr-runtime-independence acceptance 36 and 48: PIZ public read uses the source-fixed GPU route."""
    from openexr_dev_oracle import OpenEXR

    channels = _compressible_rgb(3, 8)
    path = tmp_path / "piz.exr"
    OpenEXR.File({"compression": OpenEXR.PIZ_COMPRESSION}, dict(channels)).write(str(path))
    original = io._read_exr_gpu
    gpu_calls = 0

    def gpu_spy(*args: object, **kwargs: object) -> object:
        nonlocal gpu_calls
        gpu_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(io, "_read_exr_gpu", gpu_spy)

    result = px.io.read_image(path, unchanged=True)

    assert gpu_calls == 1
    np.testing.assert_array_equal(
        px.io.to_array(
            result,
        ).get(),
        np.stack([channels[name] for name in "RGB"], axis=2),
    )


@pytest.mark.parametrize("corruption", ("zlib-header", "preset-dictionary", "inflate-size", "adler"))
def test_zip_integrity_failure_is_actionable_and_never_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    """v1-exr-gpu-phase1 acceptance 19, 20, and 28: wrapper corruption fails fast outside CPU fallback."""
    channels = _compressible_rgb(1, 64)
    fixture = _build_exr(channels, compression="zips", storage="deflate")
    payload = bytearray(fixture.payload)
    payload_start = fixture.payload_offsets[0]
    if corruption == "zlib-header":
        payload[payload_start] ^= 0x01
    elif corruption == "preset-dictionary":
        cmf = payload[payload_start]
        payload[payload_start + 1] = next(flg for flg in range(0x20, 0x40) if ((cmf << 8) | flg) % 31 == 0)
    elif corruption == "inflate-size":
        wrong_size_stream = zlib.compress(_zip_transform(_chunk_bytes(channels, 0, 1)[:-4]))
        chunk_offset = fixture.chunk_offsets[0]
        payload = payload[: chunk_offset + 4] + struct.pack("<i", len(wrong_size_stream)) + wrong_size_stream
    else:
        payload[payload_start + fixture.packed_sizes[0] - 1] ^= 0x01
    path = tmp_path / f"bad-{corruption}.exr"
    path.write_bytes(payload)
    monkeypatch.setitem(io._EXR_ROUTING, ("zips", "read"), "gpu")

    with pytest.raises(RuntimeError, match=r"why=.*what=.*how=") as error:
        px.io.read_image(path)

    expected_word = {
        "zlib-header": "zlib",
        "preset-dictionary": "dictionary",
        "inflate-size": "sizes",
        "adler": "adler",
    }[corruption]
    assert expected_word in str(error.value).lower()


def test_zlib_trailing_byte_before_adler_is_actionable_and_never_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v1-exr-gpu-phase1 acceptance 19: Deflate must end exactly at the Adler-32 trailer boundary."""
    channels = _compressible_rgb(1, 64)
    fixture = _build_exr(channels, compression="zips", storage="deflate")
    payload = bytearray(fixture.payload)
    payload_start = fixture.payload_offsets[0]
    trailer_start = payload_start + fixture.packed_sizes[0] - 4
    payload[trailer_start:trailer_start] = b"X"
    struct.pack_into("<i", payload, fixture.chunk_offsets[0] + 4, fixture.packed_sizes[0] + 1)
    path = tmp_path / "bad-zlib-trailing-byte.exr"
    path.write_bytes(payload)
    monkeypatch.setitem(io._EXR_ROUTING, ("zips", "read"), "gpu")

    with pytest.raises(RuntimeError, match=r"why=.*trailing.*what=.*how="):
        px.io.read_image(path)


def test_unexpected_nvcomp_failure_keeps_cause_and_never_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v1-exr-gpu-phase1 acceptance 20: unexpected nvCOMP failure is a fail-fast backend defect."""
    fixture = _build_exr(_compressible_rgb(1, 64), compression="zips", storage="deflate")
    path = tmp_path / "nvcomp-failure.exr"
    _write_fixture(path, fixture)

    def fail_nvcomp(*args: object, **kwargs: object) -> object:
        raise RuntimeError("synthetic nvCOMP failure")

    monkeypatch.setitem(io._EXR_ROUTING, ("zips", "read"), "gpu")
    monkeypatch.setattr(io, "_decode_deflate_chunks", fail_nvcomp)

    with pytest.raises(RuntimeError, match=r"why=.*nvCOMP.*what=.*how=") as error:
        px.io.read_image(path)

    assert isinstance(error.value.__cause__, RuntimeError)
    assert "synthetic nvCOMP failure" in str(error.value.__cause__)


def test_exr_header_probe_and_import_remain_nvcomp_codec_and_gpu_free(tmp_path: Path) -> None:
    """v1-exr-gpu-phase1 acceptance 8 and 21: import and header parsing remain GPU-less and codec-lazy."""
    fixture = _build_exr(_compressible_rgb(1, 8), compression="none")
    path = tmp_path / "header.exr"
    _write_fixture(path, fixture)
    script = """
import importlib.abc
import sys
class BlockCodecs(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "OpenEXR" or fullname.startswith("nvidia.nvcomp"):
            raise ModuleNotFoundError(fullname)
        return None
sys.meta_path.insert(0, BlockCodecs())
import pixtreme as px
header = px.io.read_header(sys.argv[1])
assert (header.format, header.width, header.height) == ("EXR", 8, 1)
assert "OpenEXR" not in sys.modules
assert "nvidia.nvcomp" not in sys.modules
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


def test_gpu_read_lane_has_one_image_staging_transfer_and_no_host_frame_synthesis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1-exr-gpu-phase1 acceptance 10: GPU ZIP read stages encoded bytes once and returns the device image."""
    channels = _compressible_rgb(17, 32)
    fixture = _build_exr(channels, compression="zip", storage="deflate")
    path = tmp_path / "gpu-transfer.zip.exr"
    _write_fixture(path, fixture)
    container = io_header._parse_exr(path)
    prepared = io._prepare_exr_read_chunks(container)
    transfers = exr_harness._record_cupy_transfers(monkeypatch)

    actual = io._read_exr_gpu(container, tuple(channels), output_dtype="float32")

    exr_harness._assert_cupy_transfer_budget(
        transfers, direction="h2d", max_count=14, max_total_nbytes=347, max_shape_elements=167
    )
    exr_harness._assert_cupy_transfer_budget(
        transfers, direction="d2h", max_count=1, max_total_nbytes=8, max_shape_elements=2
    )

    staging = [
        transfer
        for transfer in transfers
        if transfer.direction == "h2d"
        and transfer.nbytes == prepared.host_staging.nbytes
        and transfer.shape == prepared.host_staging.shape
        and transfer.dtype == prepared.host_staging.dtype.name
    ]
    assert len(staging) == 1
    assert actual.shape == (17, 32, 3)
    assert actual.dtype == np.dtype(np.float32)
    assert not [transfer for transfer in transfers if transfer.direction == "d2h" and transfer.shape == actual.shape]


def test_none_read_lane_unpacks_directly_from_container_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1-exr-runtime-independence acceptance 37 and 48: NONE stages the file once and returns a device image."""
    channels = _compressible_rgb(3, 8)
    fixture = _build_exr(channels, compression="none")
    path = tmp_path / "none-transfer.exr"
    _write_fixture(path, fixture)
    container = io_header._parse_exr(path)
    transfers = exr_harness._record_cupy_transfers(monkeypatch)

    actual = exr_none._read_exr_none(container, tuple(container.parts[0].channels), output_dtype="float32")

    exr_harness._assert_cupy_transfer_budget(
        transfers, direction="h2d", max_count=6, max_total_nbytes=736, max_shape_elements=649
    )
    exr_harness._assert_cupy_transfer_budget(
        transfers, direction="d2h", max_count=0, max_total_nbytes=0, max_shape_elements=0
    )

    file_transfers = [
        transfer
        for transfer in transfers
        if transfer.direction == "h2d"
        and transfer.nbytes == len(container.data)
        and transfer.shape == (len(container.data),)
        and transfer.dtype == "uint8"
    ]
    assert len(file_transfers) == 1
    assert actual.shape == (3, 8, 3)
    assert actual.dtype == np.dtype(np.float32)
    assert not [transfer for transfer in transfers if transfer.direction == "d2h"]


@pytest.mark.parametrize("compression", ("none", "zip", "zips"))
@pytest.mark.parametrize("dtype", (np.uint8, np.uint16, np.float16, np.float32))
def test_gpu_writer_cross_backend_round_trips_every_storage_dtype(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compression: str,
    dtype: type[np.generic],
) -> None:
    """v1-exr-gpu-phase1 acceptance 5, 14, 17, and 18; v1-exr-runtime-independence acceptance 5:
    OpenEXR reads every GPU-written storage class with unique labels and ACES header metadata.
    """
    import cupy as cp
    from openexr_dev_oracle import OpenEXR

    height, width = 17, 32
    if np.issubdtype(dtype, np.integer):
        maximum = np.iinfo(dtype).max
        values = np.resize(
            np.asarray((0, 1, maximum // 4, maximum // 2, maximum - 1, maximum), dtype=dtype), height * width * 4
        )
    else:
        values = np.resize(
            np.asarray((-1.0, -0.0, 0.25, 1.0, 2.5, np.inf, -np.inf, np.nan), dtype=dtype),
            height * width * 4,
        )
    values = values.reshape(height, width, 4)
    labels = ("diffuse.R", "G", "B", "A")
    frame = px.io.from_array(cp.asarray(values), colorspace="ACES2065-1", gamma="linear", channels=labels)
    path = tmp_path / f"gpu-{compression}-{np.dtype(dtype).name}.exr"
    gpu_calls = 0
    original = io._write_exr_gpu

    def gpu_spy(*args: object, **kwargs: object) -> object:
        nonlocal gpu_calls
        gpu_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setitem(io._EXR_ROUTING, (compression, "write"), "gpu")
    monkeypatch.setattr(io, "_write_exr_gpu", gpu_spy)

    output_dtype = "float16" if np.dtype(dtype) == np.dtype(np.float16) else "float32"
    px.io.write_image(path, frame, compression=compression, dtype=output_dtype)

    reference = OpenEXR.File(str(path), separate_channels=True)
    decoded = np.stack([np.asarray(reference.channels()[label].pixels) for label in labels], axis=2)
    expected = (
        values.astype(np.float32) * np.float32(1.0 / np.iinfo(dtype).max)
        if np.issubdtype(dtype, np.integer)
        else values
    )
    assert gpu_calls == 1
    assert (
        reference.header()["compression"]
        == {
            "none": OpenEXR.NO_COMPRESSION,
            "zip": OpenEXR.ZIP_COMPRESSION,
            "zips": OpenEXR.ZIPS_COMPRESSION,
        }[compression]
    )
    assert reference.header()["acesImageContainerFlag"] == 1
    np.testing.assert_array_equal(decoded, expected)


def test_gpu_writer_batches_zips_chunks_beyond_cuda_grid_y_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v1-exr-gpu-phase1 acceptance 1, 14, and 18: tall ZIPS output remains OpenEXR-readable."""
    import cupy as cp
    from openexr_dev_oracle import OpenEXR

    height = 65_536
    expected = np.linspace(-1.0, 2.0, height, dtype=np.float32).reshape(height, 1)
    frame = px.io.from_array(
        cp.asarray(expected[..., np.newaxis]),
        colorspace="ACEScg",
        gamma="linear",
        channels=("Y",),
    )
    path = tmp_path / "tall-zips.exr"
    monkeypatch.setitem(io._EXR_ROUTING, ("zips", "write"), "gpu")

    px.io.write_image(path, frame, compression="zips", dtype="float32")

    reference = OpenEXR.File(str(path), separate_channels=True)
    decoded = np.asarray(reference.channels()["Y"].pixels)
    assert reference.header()["compression"] == OpenEXR.ZIPS_COMPRESSION
    np.testing.assert_array_equal(decoded, expected)


@pytest.mark.parametrize(("compression", "height"), (("zips", 65_536), ("zip", 1_048_561)))
def test_gpu_writer_chunk_launch_ranges_cover_cuda_grid_y_boundary(compression: str, height: int) -> None:
    """v1-exr-gpu-phase1 acceptance 14 and 16: ZIPS and ZIP chunk counts are grid-Y independent."""
    lines_per_chunk = 1 if compression == "zips" else 16
    chunk_count = (height + lines_per_chunk - 1) // lines_per_chunk

    assert chunk_count == 65_536
    assert exr_packing._chunk_launch_ranges(chunk_count) == ((0, 65_535), (65_535, 65_536))


@pytest.mark.parametrize(("compression", "height"), (("zip", 17), ("zips", 3)))
def test_gpu_writer_emits_independent_zlib_streams_and_raw_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compression: str,
    height: int,
) -> None:
    """v1-exr-gpu-phase1 acceptance 15, 16, and 27: host zlib and hand transforms verify both chunk branches."""
    import cupy as cp

    width = 64
    compressible = _compressible_rgb(height, width)
    compressible_values = np.stack([compressible[name] for name in "RGB"], axis=2)
    compressible_frame = px.io.from_array(
        cp.asarray(compressible_values), colorspace="ACEScg", gamma="linear", channels="RGB"
    )
    compressible_path = tmp_path / f"compressed-{compression}.exr"
    monkeypatch.setitem(io._EXR_ROUTING, (compression, "write"), "gpu")

    px.io.write_image(compressible_path, compressible_frame, compression=compression, dtype="float32")

    compressed_container = io_header._parse_exr(compressible_path)
    assert any(not chunk.raw_stored for chunk in compressed_container.chunks)
    for chunk in compressed_container.chunks:
        raw = _chunk_bytes(compressible, chunk.row_start, chunk.row_count)
        payload = compressed_container.data[chunk.payload_start : chunk.payload_end]
        if chunk.raw_stored:
            assert payload == raw
        else:
            assert zlib.decompress(payload) == _zip_transform(raw)

    generator = np.random.default_rng(20260808)
    incompressible = generator.integers(0, 1 << 32, size=(height, width, 3), dtype=np.uint32).view(np.float32)
    incompressible_frame = px.io.from_array(
        cp.asarray(incompressible), colorspace="ACEScg", gamma="linear", channels="RGB"
    )
    incompressible_path = tmp_path / f"raw-{compression}.exr"

    px.io.write_image(incompressible_path, incompressible_frame, compression=compression, dtype="float32")

    raw_container = io_header._parse_exr(incompressible_path)
    assert all(chunk.raw_stored for chunk in raw_container.chunks)
    for chunk in raw_container.chunks:
        expected = _chunk_bytes(
            {label: incompressible[..., index] for index, label in enumerate("RGB")},
            chunk.row_start,
            chunk.row_count,
        )
        assert raw_container.data[chunk.payload_start : chunk.payload_end] == expected


def test_gpu_write_lane_has_no_openexr_codec_or_full_frame_host_synthesis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1-exr-gpu-phase1 acceptance 14 and 16: ZIPS writes one final byte payload and no host image."""
    import cupy as cp

    frame = px.io.from_array(
        cp.zeros((3, 64, 3), dtype=cp.float32), colorspace="ACEScg", gamma="linear", channels="RGB"
    )
    path = tmp_path / "zips-transfer.exr"
    transfers = exr_harness._record_cupy_transfers(monkeypatch)

    px.io.write_image(path, frame, compression="zips")

    container = io_header._parse_exr(path)
    final_payload_bytes = sum(chunk.packed_size for chunk in container.chunks)
    exr_harness._assert_cupy_transfer_budget(
        transfers, direction="h2d", max_count=12, max_total_nbytes=255, max_shape_elements=3
    )
    exr_harness._assert_cupy_transfer_budget(
        transfers,
        direction="d2h",
        max_count=1,
        max_total_nbytes=final_payload_bytes,
        max_shape_elements=final_payload_bytes,
    )
    final_transfers = [
        transfer
        for transfer in transfers
        if transfer.direction == "d2h"
        and transfer.has_output_buffer
        and transfer.dtype == "uint8"
        and transfer.nbytes == final_payload_bytes
    ]
    assert len(final_transfers) == 1
    assert not [
        transfer
        for transfer in transfers
        if transfer.direction == "d2h"
        and transfer.shape == frame.data.shape
        and transfer.dtype == frame.data.dtype.name
    ]


def test_gpu_deflate_write_reuses_the_device_stream_codec() -> None:
    """v1-exr-gpu-phase1 acceptance 15 and 16: repeated Deflate encoding reuses one device-stream codec."""
    import cupy as cp

    transformed = cp.zeros(512, dtype=cp.uint8)
    exr_packing._nvcomp_deflate_codec.cache_clear()

    exr_packing._encode_deflate_chunks(transformed, ((0, 512),))
    exr_packing._encode_deflate_chunks(transformed, ((0, 512),))

    cache = exr_packing._nvcomp_deflate_codec.cache_info()
    assert (cache.misses, cache.hits, cache.currsize) == (1, 1, 1)


def test_none_write_interleaves_chunk_headers_before_one_pinned_transfer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1-exr-gpu-phase1 acceptance 14 and 16: NONE transfers one header-interleaved device payload."""
    import cupy as cp

    frame = px.io.from_array(
        cp.arange(3 * 8 * 3, dtype=cp.float32).reshape(3, 8, 3),
        colorspace="ACEScg",
        gamma="linear",
        channels="RGB",
    )
    path = tmp_path / "none-write-transfer.exr"
    transfers = exr_harness._record_cupy_transfers(monkeypatch)

    px.io.write_image(path, frame, compression="none")

    container = io_header._parse_exr(path)
    expected_payload_bytes = sum(8 + chunk.packed_size for chunk in container.chunks)
    exr_harness._assert_cupy_transfer_budget(
        transfers, direction="h2d", max_count=1, max_total_nbytes=12, max_shape_elements=3
    )
    exr_harness._assert_cupy_transfer_budget(
        transfers,
        direction="d2h",
        max_count=1,
        max_total_nbytes=expected_payload_bytes,
        max_shape_elements=expected_payload_bytes,
    )
    final_transfers = [
        transfer
        for transfer in transfers
        if transfer.direction == "d2h"
        and transfer.has_output_buffer
        and transfer.nbytes == expected_payload_bytes
        and transfer.shape == (expected_payload_bytes,)
        and transfer.dtype == "uint8"
    ]
    assert len(final_transfers) == 1


def test_unexpected_nvcomp_write_failure_keeps_cause_and_never_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v1-exr-gpu-phase1 acceptance 20 and 28: a GPU write defect fails fast instead of entering CPU encoding."""
    import cupy as cp

    frame = px.io.from_array(
        cp.zeros((2, 64, 3), dtype=cp.float32), colorspace="ACEScg", gamma="linear", channels="RGB"
    )
    path = tmp_path / "nvcomp-write-failure.exr"

    def fail_nvcomp(*args: object, **kwargs: object) -> object:
        raise RuntimeError("synthetic nvCOMP write failure")

    monkeypatch.setitem(io._EXR_ROUTING, ("zips", "write"), "gpu")
    monkeypatch.setattr(io, "_encode_deflate_chunks", fail_nvcomp)

    with pytest.raises(RuntimeError, match=r"why=.*nvCOMP.*what=.*how=") as error:
        px.io.write_image(path, frame, compression="zips")

    assert isinstance(error.value.__cause__, RuntimeError)
    assert "synthetic nvCOMP write failure" in str(error.value.__cause__)
    assert not path.exists()


def test_unexpected_nvcomp_write_value_error_is_actionable_and_keeps_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v1-exr-gpu-phase1 acceptance 20: backend ValueError is a caused actionable GPU defect."""
    import cupy as cp

    frame = px.io.from_array(
        cp.zeros((2, 64, 3), dtype=cp.float32), colorspace="ACEScg", gamma="linear", channels="RGB"
    )
    path = tmp_path / "nvcomp-write-value-error.exr"

    def fail_nvcomp(*args: object, **kwargs: object) -> object:
        raise ValueError("synthetic nvCOMP ValueError")

    monkeypatch.setitem(io._EXR_ROUTING, ("zips", "write"), "gpu")
    monkeypatch.setattr(io, "_encode_deflate_chunks", fail_nvcomp)

    with pytest.raises(RuntimeError, match=r"why=.*nvCOMP.*what=.*how=") as error:
        px.io.write_image(path, frame, compression="zips")

    assert type(error.value.__cause__) is ValueError
    assert "synthetic nvCOMP ValueError" in str(error.value.__cause__)
    assert not path.exists()


def test_phase1_auto_selection_is_source_fixed_from_performance_gate() -> None:
    """v1-exr-gpu-phase1 acceptance 1, 3-5, and 23: measured selection is fixed without public controls."""
    phase1_selection = {key: backend for key, backend in io._EXR_ROUTING.items() if key[0] in {"none", "zip", "zips"}}

    assert phase1_selection == {
        ("none", "read"): "native",
        ("none", "write"): "gpu",
        ("zip", "read"): "custom_cpu",
        ("zip", "write"): "gpu",
        ("zips", "read"): "custom_cpu",
        ("zips", "write"): "gpu",
    }
