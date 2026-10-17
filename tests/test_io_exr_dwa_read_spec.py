"""Specification tests for the DWAA/DWAB OpenEXR DWA read lanes."""

from __future__ import annotations

import struct
from pathlib import Path

import cupy as cp
import numpy as np
import pytest

import pixtreme as px
import pixtreme._io.formats.exr.codec_dwa as exr_dwa
import pixtreme._io.formats.exr.container as exr_container
import pixtreme._io.formats.exr.selection as io


def _pack_bits(fields: list[tuple[int, int]]) -> tuple[bytes, int]:
    value = 0
    bit_count = 0
    for field, width in fields:
        assert 0 <= field < 1 << width
        value = (value << width) | field
        bit_count += width
    padding = (-bit_count) % 8
    value <<= padding
    return value.to_bytes((bit_count + padding) // 8, "big"), bit_count


def _repeat_huffman_stream(*, repeat_count: int = 2) -> bytes:
    # Code lengths (1, 2, 2) produce canonical codes 1, 00, 01 for
    # symbols 1, 2, and repeat-pseudo-symbol 3. The stream emits 2,
    # repeats it twice, then emits 1.
    table, _ = _pack_bits([(1, 6), (2, 6), (2, 6)])
    data, data_bit_count = _pack_bits([(0b00, 2), (0b01, 2), (repeat_count, 8), (0b1, 1)])
    return struct.pack("<IIIII", 1, 3, len(table), data_bit_count, 0) + table + data


@pytest.mark.req("SP-PIX-007")
def test_dwa_huffman_decode_expands_the_generic_repeat_symbol_and_consumes_every_bit() -> None:
    """DWA Huffman reads expand repeated symbols and consume exactly the encoded bit stream."""
    stream = _repeat_huffman_stream()
    table = exr_dwa._parse_dwa_huffman_table(stream)

    decoded = exr_dwa._decode_dwa_huffman_host(stream, table, expected_count=4)

    np.testing.assert_array_equal(decoded, np.asarray((2, 2, 2, 1), dtype=np.uint16))


@pytest.mark.req("SP-PIX-007")
@pytest.mark.req("SP-PIX-017")
def test_dwa_huffman_decode_rejects_a_zero_length_repeat() -> None:
    """DWA Huffman reads reject a zero-length repeat with a useful error."""
    stream = _repeat_huffman_stream(repeat_count=0)
    table = exr_dwa._parse_dwa_huffman_table(stream)

    with pytest.raises(RuntimeError, match=r"why=.*repeat.*what=.*how="):
        exr_dwa._decode_dwa_huffman_host(stream, table, expected_count=2)


@pytest.mark.req("SP-PIX-007")
@pytest.mark.req("SP-PIX-018")
@pytest.mark.req("SP-PIX-101")
def test_dwa_gpu_huffman_decode_parallel_segments_match_host_across_a_repeat_boundary() -> None:
    """Parallel GPU DWA Huffman decoding matches host output when a repeated symbol crosses a segment boundary."""
    table_bits, _ = _pack_bits([(1, 6), (2, 6), (2, 6)])
    fields = [(0b00, 2), *([(0b01, 2), (255, 8)] * 4), (0b01, 2), (3, 8), (0b01, 2), (7, 8), (0b1, 1)]
    data, data_bit_count = _pack_bits(fields)
    stream = struct.pack("<IIIII", 1, 3, len(table_bits), data_bit_count, 0) + table_bits + data
    table = exr_dwa._parse_dwa_huffman_table(stream)
    expected = np.concatenate((np.full(1031, 2, dtype=np.uint16), np.asarray((1,), dtype=np.uint16)))

    actual = exr_dwa._decode_dwa_huffman_gpu(
        cp.asarray(np.frombuffer(stream, dtype=np.uint8)),
        data_offsets=(table.data_span.start,),
        tables=(table,),
        output_counts=(expected.size,),
        record_labels=(17,),
    )

    np.testing.assert_array_equal(actual.get(), expected)


def _write_dwa_with_late_huffman_repeat(path: Path, *, compression: str, tail_literal: bool, repeat_count: int) -> None:
    from openexr_dev_oracle import OpenEXR

    # A single scanline chunk contains 260 constant 8x8 Y blocks. Keep the
    # independent writer's DC stream and encode its 260 EOBs explicitly so a
    # parallel decode boundary lands on a repeat close to the end of the stream.
    OpenEXR.File(
        {
            "compression": {"dwaa": OpenEXR.DWAA_COMPRESSION, "dwab": OpenEXR.DWAB_COMPRESSION}[compression],
            "dwaCompressionLevel": 45.0,
        },
        {"Y": np.full((32, 520), np.float16(0.25), dtype=np.float16)},
    ).write(str(path))
    container = exr_container._parse_exr_container(path)
    (chunk,) = container.chunks
    descriptor = chunk.dwa
    assert descriptor is not None and descriptor.leader is not None
    assert descriptor.leader.ac_element_count == 260

    # EOB 0xff00 and repeat 0xff01 have one-bit codes 0 and 1.
    packed_table, _ = _pack_bits([(1, 6), (1, 6)])
    fields = [(0, 1)] * 256 + [(1, 1), (repeat_count, 8)]
    if tail_literal:
        fields.append((0, 1))
    encoded, bit_count = _pack_bits(fields)
    stream = struct.pack("<IIIII", 0xFF00, 0xFF01, len(packed_table), bit_count, 0) + packed_table + encoded
    payload = bytearray(container.data)
    struct.pack_into("<Q", payload, chunk.payload_start + 24, len(stream))
    size_delta = len(stream) - descriptor.ac_span.size
    struct.pack_into("<i", payload, chunk.payload_start - 4, chunk.packed_size + size_delta)
    payload[descriptor.ac_span.start : descriptor.ac_span.end] = stream
    path.write_bytes(payload)


@pytest.mark.req("SP-PIX-007")
@pytest.mark.parametrize("compression", ("dwaa", "dwab"))
@pytest.mark.parametrize("tail_literal", (True, False), ids=("literal-after-repeat", "repeat-at-end"))
def test_dwa_read_matches_openexr_when_a_late_repeat_reaches_the_stream_end(
    tmp_path: Path, compression: str, tail_literal: bool
) -> None:
    """DWA images remain readable when a repeat near the end follows many literal coefficients."""
    from openexr_dev_oracle import OpenEXR

    path = tmp_path / f"late-repeat-{compression}.exr"
    _write_dwa_with_late_huffman_repeat(
        path, compression=compression, tail_literal=tail_literal, repeat_count=3 if tail_literal else 4
    )
    expected = OpenEXR.File(str(path), separate_channels=True).channels()["Y"].pixels

    actual = px.io.read_image(path, channels=("Y",), unchanged=True)

    # The tolerance is one HALF ulp near 1 plus accumulated fp32 inverse-DCT rounding.
    np.testing.assert_allclose(actual.data.get()[..., 0], expected, rtol=2e-3, atol=2e-3)


@pytest.mark.req("SP-PIX-007")
@pytest.mark.req("SP-PIX-017")
@pytest.mark.parametrize("compression", ("dwaa", "dwab"))
@pytest.mark.parametrize("repeat_count", (3, 5), ids=("too-few-coefficients", "too-many-coefficients"))
def test_dwa_read_rejects_a_late_repeat_that_disagrees_with_the_declared_coefficient_count(
    tmp_path: Path, compression: str, repeat_count: int
) -> None:
    """DWA reads reject a final repeat that leaves missing coefficients or adds excess coefficients."""
    path = tmp_path / f"invalid-late-repeat-{compression}.exr"
    _write_dwa_with_late_huffman_repeat(path, compression=compression, tail_literal=False, repeat_count=repeat_count)

    with pytest.raises(RuntimeError, match=r"why=.*DWA.*what=.*how="):
        px.io.read_image(path, channels=("Y",), unchanged=True)


def _write_reference_dwa(path: Path, compression: str) -> dict[str, np.ndarray]:
    from openexr_dev_oracle import OpenEXR

    height = 33 if compression == "dwaa" else 257
    width = 17
    y, x = np.mgrid[:height, :width]
    ramp = np.float32(0.125) + x.astype(np.float32) / np.float32(24.0) + y.astype(np.float32) / height
    channels = {
        "beauty.R": ramp.astype(np.float16),
        "beauty.G": (ramp * np.float32(0.625)).astype(np.float16),
        "beauty.B": (ramp * np.float32(-0.25)).astype(np.float16),
        "beauty.A": np.where((x + y) % 5, np.float16(1.0), np.float16(0.25)).astype(np.float16),
        "depth.Z": (ramp * np.float32(17.0) - np.float32(3.0)).astype(np.float32),
        "luma.Y": np.full((height, width), np.float16(0.375), dtype=np.float16),
    }
    compression_token = {
        "dwaa": OpenEXR.DWAA_COMPRESSION,
        "dwab": OpenEXR.DWAB_COMPRESSION,
    }[compression]
    origin = (-3, 7)
    maximum = (origin[0] + width - 1, origin[1] + height - 1)
    header = {
        "compression": compression_token,
        "dwaCompressionLevel": 45.0,
        "dataWindow": (np.asarray(origin, dtype=np.int32), np.asarray(maximum, dtype=np.int32)),
        "displayWindow": (np.asarray(origin, dtype=np.int32), np.asarray(maximum, dtype=np.int32)),
        "lineOrder": OpenEXR.DECREASING_Y,
    }
    OpenEXR.File(header, channels).write(str(path))
    reference = OpenEXR.File(str(path), separate_channels=True)
    return {name: np.asarray(reference.channels()[name].pixels) for name in channels}


@pytest.mark.req("SP-PIX-007")
@pytest.mark.parametrize("compression", ("dwaa", "dwab"))
@pytest.mark.parametrize("dtype", ("float16", "float32"))
@pytest.mark.parametrize("shape", ((4, 6), (8, 12), (16, 16), (64, 64), (128, 128), (256, 256), (512, 512)))
def test_written_dwa_rgb_reads_match_openexr_across_sizes_and_float_storage(
    tmp_path: Path,
    compression: str,
    dtype: str,
    shape: tuple[int, int],
) -> None:
    """DWAA/DWAB RGB reads match OpenEXR across image sizes with float16 and float32 storage."""
    from openexr_dev_oracle import OpenEXR

    y, x = np.indices(shape, dtype=np.float32)
    values = np.stack([((x + 9 * y + offset) % 256) / 255 for offset in (0, 41, 101)], axis=-1)
    source = px.io.from_array(cp.asarray(values), colorspace="ACEScg", gamma="linear", channels="RGB")
    path = tmp_path / f"rgb-{compression}-{dtype}.exr"
    px.io.write_image(path, source, compression=compression, dtype=dtype)
    reference = OpenEXR.File(str(path), separate_channels=True).channels()
    expected = np.stack([reference[channel].pixels for channel in ("R", "G", "B")], axis=-1)

    actual = px.io.read_image(path, unchanged=True)

    assert actual.channels == ("R", "G", "B")
    assert actual.data.dtype == np.dtype(dtype)
    assert actual.data.flags.c_contiguous
    # One HALF ulp near 1 plus accumulated fp32 inverse-DCT/CSC rounding, as in the reference read tests.
    np.testing.assert_allclose(actual.data.get(), expected, rtol=2e-3, atol=2e-3)


def _eager_dwa_reconstruct_lossy_blocks(
    spatial: cp.ndarray,
    transfer_flags: np.ndarray,
    csc_triplets: np.ndarray,
) -> cp.ndarray:
    """Test-side rendition of the pre-fusion read color, inverse-transfer, and HALF-selection path."""
    if csc_triplets.size:
        triplets = cp.asarray(csc_triplets)
        y_plane = spatial[triplets[:, 0]].copy()
        cb_plane = spatial[triplets[:, 1]].copy()
        cr_plane = spatial[triplets[:, 2]].copy()
        spatial[triplets[:, 0]] = y_plane + cp.float32(1.5747) * cr_plane
        spatial[triplets[:, 1]] = y_plane - cp.float32(0.1873) * cb_plane - cp.float32(0.4682) * cr_plane
        spatial[triplets[:, 2]] = y_plane + cp.float32(1.8556) * cb_plane
    nonlinear = spatial.astype(cp.float16)
    nonlinear_float = nonlinear.astype(cp.float32)
    magnitude = cp.abs(nonlinear_float)
    linear = cp.where(
        magnitude <= cp.float32(1.0),
        cp.power(magnitude, cp.float32(2.2)),
        cp.exp(cp.float32(2.2) * (magnitude - cp.float32(1.0))),
    )
    linear = cp.copysign(linear, nonlinear_float)
    linear = cp.where(cp.isfinite(nonlinear_float), linear, cp.float32(0.0))
    reconstructed = cp.where(
        cp.asarray(transfer_flags)[:, None, None],
        linear.astype(cp.float16),
        nonlinear,
    )
    return cp.ascontiguousarray(reconstructed).view(cp.uint16).reshape(-1)


@pytest.mark.req("SP-PIX-007")
@pytest.mark.req("SP-PIX-018")
@pytest.mark.req("SP-PIX-101")
@pytest.mark.parametrize("compression", ("dwaa", "dwab"))
def test_dwa_transfer_fusion_preserves_decode_output_characterization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compression: str,
) -> None:
    """characterization: DWA reads retain the recorded output bits for partial, compressed, and raw chunks."""
    fused_helper = exr_dwa._reconstruct_dwa_lossy_blocks_gpu
    path = tmp_path / f"reference-{compression}.exr"
    _write_reference_dwa(path, compression)
    container = exr_container._parse_exr_container(path)
    assert any(not chunk.raw_stored for chunk in container.chunks)
    assert any(chunk.raw_stored for chunk in container.chunks)
    assert container.chunks[-1].row_count < container.lines_per_chunk

    selected = ("depth.Z", "beauty.A", "beauty.B", "beauty.R", "beauty.G", "luma.Y")
    candidate = px.io.read_image(path, channels=selected, colorspace="Rec.709", gamma="linear")
    candidate_half = px.io.read_image(
        path,
        channels=("beauty.R", "beauty.G", "beauty.B"),
        unchanged=True,
    )
    monkeypatch.setattr(exr_dwa, "_reconstruct_dwa_lossy_blocks_gpu", _eager_dwa_reconstruct_lossy_blocks)
    eager = px.io.read_image(path, channels=selected, colorspace="Rec.709", gamma="linear")
    eager_half = px.io.read_image(
        path,
        channels=("beauty.R", "beauty.G", "beauty.B"),
        unchanged=True,
    )

    assert fused_helper is not _eager_dwa_reconstruct_lossy_blocks
    assert (candidate.channels, candidate.data.dtype) == (eager.channels, eager.data.dtype)
    assert (candidate_half.channels, candidate_half.data.dtype) == (eager_half.channels, eager_half.data.dtype)
    np.testing.assert_array_equal(candidate.data.get().view(np.uint32), eager.data.get().view(np.uint32))
    np.testing.assert_array_equal(candidate_half.data.get().view(np.uint16), eager_half.data.get().view(np.uint16))


@pytest.mark.req("SP-PIX-007")
@pytest.mark.parametrize("compression", ("dwaa", "dwab"))
@pytest.mark.parametrize("backend", ("custom_cpu", "gpu"))
def test_reference_dwa_read_lanes_cover_lossy_lossless_partial_and_selected_channels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compression: str,
    backend: str,
) -> None:
    """GPU and custom CPU DWA reads recover reference lossy, lossless, partial, and selected channel samples."""
    path = tmp_path / f"reference-{compression}.exr"
    reference = _write_reference_dwa(path, compression)
    container = exr_container._parse_exr_container(path)
    assert exr_container._container_gpu_eligible(container) is True
    assert any(not chunk.raw_stored for chunk in container.chunks)
    assert any(chunk.raw_stored for chunk in container.chunks)
    assert container.data_window[:2] == (-3, 7)

    calls = 0
    selected_backend = io._read_exr_custom_cpu if backend == "custom_cpu" else io._read_exr_gpu

    def spy(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return selected_backend(*args, **kwargs)

    monkeypatch.setitem(io._EXR_ROUTING, (compression, "read"), backend)
    if backend == "gpu":
        monkeypatch.setattr(
            exr_dwa,
            "_read_exr_dwa_custom_cpu",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("GPU DWA decode delegated to the CPU lane")),
        )
    monkeypatch.setattr(io, f"_read_exr_{backend}", spy)
    selected = ("depth.Z", "beauty.A", "beauty.B", "beauty.R", "beauty.G", "luma.Y")

    frame = px.io.read_image(path, channels=selected, colorspace="Rec.709", gamma="linear")
    actual = frame.data.get()

    assert calls == 1
    assert frame.channels == selected
    assert frame.data.flags.c_contiguous
    assert frame.data.dtype == np.dtype(np.float32)
    assert (frame.colorspace, frame.gamma) == ("Rec.709", "linear")
    np.testing.assert_array_equal(actual[..., 0], reference["depth.Z"].astype(np.float32))
    np.testing.assert_array_equal(actual[..., 1], reference["beauty.A"].astype(np.float32))
    # The tolerance is one HALF ulp near 1 plus accumulated fp32 inverse-DCT/CSC rounding.
    for output_index, name in enumerate(selected[2:], start=2):
        np.testing.assert_allclose(actual[..., output_index], reference[name].astype(np.float32), rtol=2e-3, atol=2e-3)


@pytest.mark.req("SP-PIX-007")
@pytest.mark.req("SP-PIX-017")
@pytest.mark.parametrize("backend", ("custom_cpu", "gpu"))
def test_dwa_read_lanes_fail_fast_on_a_corrupt_zlib_substream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    """DWA reads report corrupt zlib substreams with their cause instead of switching decoders."""
    path = tmp_path / f"corrupt-dwaa-{backend}.exr"
    _write_reference_dwa(path, "dwaa")
    container = exr_container._parse_exr_container(path)
    compressed = next(chunk for chunk in container.chunks if not chunk.raw_stored)
    descriptor = compressed.dwa
    assert descriptor is not None and descriptor.leader is not None
    assert descriptor.leader.unknown_compressed_size > 0
    corrupted = bytearray(path.read_bytes())
    corrupted[descriptor.unknown_span.end - 1] ^= 1
    path.write_bytes(corrupted)

    monkeypatch.setitem(io._EXR_ROUTING, ("dwaa", "read"), backend)

    with pytest.raises(RuntimeError, match=r"why=.*DWA.*what=.*how="):
        px.io.read_image(path, channels=("depth.Z",))


@pytest.mark.req("SP-PIX-007")
@pytest.mark.req("SP-PIX-008")
def test_custom_cpu_dwa_read_preserves_uniform_half_with_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unchanged custom CPU DWA read returns selected uniform HALF channels as float16 pixels."""
    path = tmp_path / "reference-dwaa-half.exr"
    reference = _write_reference_dwa(path, "dwaa")
    monkeypatch.setitem(io._EXR_ROUTING, ("dwaa", "read"), "custom_cpu")

    frame = px.io.read_image(path, channels=("beauty.A", "beauty.R"), unchanged=True)
    actual = frame.data.get()

    assert frame.data.dtype == np.dtype(np.float16)
    np.testing.assert_array_equal(actual[..., 0], reference["beauty.A"])
    np.testing.assert_allclose(actual[..., 1], reference["beauty.R"], rtol=2e-3, atol=2e-3)
