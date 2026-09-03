"""Specification tests for the Phase 2 OpenEXR DWA read lanes."""

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


def test_dwa_huffman_decode_expands_the_generic_repeat_symbol_and_consumes_every_bit() -> None:
    """v1-exr-gpu-phase2 acceptance 9: generic repeat decode preserves symbols and the exact stream end."""
    stream = _repeat_huffman_stream()
    table = exr_container._parse_dwa_huffman_table(stream)

    decoded = exr_dwa._decode_dwa_huffman_host(stream, table, expected_count=4)

    np.testing.assert_array_equal(decoded, np.asarray((2, 2, 2, 1), dtype=np.uint16))


def test_dwa_huffman_decode_rejects_a_zero_length_repeat() -> None:
    """v1-exr-gpu-phase2 acceptance 9 and 24: an invalid generic repeat fails with actionable context."""
    stream = _repeat_huffman_stream(repeat_count=0)
    table = exr_container._parse_dwa_huffman_table(stream)

    with pytest.raises(RuntimeError, match=r"why=.*repeat.*what=.*how="):
        exr_dwa._decode_dwa_huffman_host(stream, table, expected_count=2)


def test_dwa_gpu_huffman_decode_parallel_segments_match_host_across_a_repeat_boundary() -> None:
    """v1-exr-gpu-phase2 acceptance 9 and 27: segmented GPU decode preserves a boundary repeat exactly."""
    table_bits, _ = _pack_bits([(1, 6), (2, 6), (2, 6)])
    fields = [(0b00, 2), *([(0b01, 2), (255, 8)] * 4), (0b01, 2), (3, 8), (0b01, 2), (7, 8), (0b1, 1)]
    data, data_bit_count = _pack_bits(fields)
    stream = struct.pack("<IIIII", 1, 3, len(table_bits), data_bit_count, 0) + table_bits + data
    table = exr_container._parse_dwa_huffman_table(stream)
    expected = np.concatenate((np.full(1031, 2, dtype=np.uint16), np.asarray((1,), dtype=np.uint16)))

    actual = exr_dwa._decode_dwa_huffman_gpu(
        cp.asarray(np.frombuffer(stream, dtype=np.uint8)),
        data_offsets=(table.data_span.start,),
        tables=(table,),
        output_counts=(expected.size,),
        record_labels=(17,),
    )

    np.testing.assert_array_equal(actual.get(), expected)


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


@pytest.mark.parametrize("compression", ("dwaa", "dwab"))
def test_dwa_transfer_fusion_preserves_decode_output_characterization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compression: str,
) -> None:
    """characterization: issue #1 RawKernel trial acceptance 2 and 5 freezes eager DWA decode bits for partial,
    compressed, and raw chunks until the transfer/color contract changes; OpenEXR remains the independent oracle.
    """
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


@pytest.mark.parametrize("compression", ("dwaa", "dwab"))
@pytest.mark.parametrize("backend", ("custom_cpu", "gpu"))
def test_reference_dwa_read_lanes_cover_lossy_lossless_partial_and_selected_channels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compression: str,
    backend: str,
) -> None:
    """v1-exr-gpu-phase2 acceptance 12-16: both forced read lanes reconstruct the reference DWA channel classes."""
    path = tmp_path / f"reference-{compression}.exr"
    reference = _write_reference_dwa(path, compression)
    container = exr_container._parse_exr_container(path)
    assert container.dwa_eligible is True
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


@pytest.mark.parametrize("backend", ("custom_cpu", "gpu"))
def test_dwa_read_lanes_fail_fast_on_a_corrupt_zlib_substream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    """v1-exr-gpu-phase2 acceptance 24-25: eligible corruption is actionable and never hidden by fallback."""
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


def test_custom_cpu_dwa_read_preserves_uniform_half_with_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v1-exr-gpu-phase2 acceptance 14 and 16: the custom CPU lane lands selected HALF once as fp16."""
    path = tmp_path / "reference-dwaa-half.exr"
    reference = _write_reference_dwa(path, "dwaa")
    monkeypatch.setitem(io._EXR_ROUTING, ("dwaa", "read"), "custom_cpu")

    frame = px.io.read_image(path, channels=("beauty.A", "beauty.R"), unchanged=True)
    actual = frame.data.get()

    assert frame.data.dtype == np.dtype(np.float16)
    np.testing.assert_array_equal(actual[..., 0], reference["beauty.A"])
    np.testing.assert_allclose(actual[..., 1], reference["beauty.R"], rtol=2e-3, atol=2e-3)
