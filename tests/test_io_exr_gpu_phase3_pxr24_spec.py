"""Specification tests for the Phase 3 OpenEXR PXR24 GPU and custom-CPU lanes."""

from __future__ import annotations

import importlib
import struct
import zlib
from dataclasses import replace
from pathlib import Path

import exr_test_harness as exr_harness
import numpy as np
import pytest

import pixtreme as px
import pixtreme._io.formats.exr.codec_pxr24 as exr_pxr24
import pixtreme._io.formats.exr.container as exr_container
import pixtreme._io.formats.exr.selection as io


def _float24_encode_bits(bits: np.ndarray) -> np.ndarray:
    """Independent NumPy implementation of the OpenEXR PXR24 FLOAT conversion."""
    source = np.asarray(bits, dtype=np.uint32)
    sign = source & np.uint32(0x80000000)
    exponent = source & np.uint32(0x7F800000)
    mantissa = source & np.uint32(0x007FFFFF)
    magnitude = exponent | mantissa
    truncated = magnitude >> np.uint32(8)
    rounded = (magnitude + (mantissa & np.uint32(0x80))) >> np.uint32(8)
    finite = exponent != np.uint32(0x7F800000)
    became_infinity = (rounded & np.uint32(0x7FFFFF)) == np.uint32(0x7F8000)
    encoded = np.where(finite & became_infinity, truncated, rounded).astype(np.uint32)
    nan = (exponent == np.uint32(0x7F800000)) & (mantissa != 0)
    nan_payload_missing = nan & ((truncated & np.uint32(0x7FFF)) == 0)
    encoded = np.where(nan, truncated | nan_payload_missing.astype(np.uint32), encoded)
    return (sign >> np.uint32(8)) | encoded


def _encode_rows(bits: np.ndarray, pixel_type: int) -> np.ndarray:
    """Independent row-reset modular-delta and MSB-first plane oracle."""
    plane_count = {0: 4, 1: 2, 2: 3}[pixel_type]
    mask = np.uint32((1 << (plane_count * 8)) - 1)
    values = _float24_encode_bits(bits) if pixel_type == 2 else np.asarray(bits, dtype=np.uint32) & mask
    previous = np.zeros_like(values)
    previous[:, 1:] = values[:, :-1]
    difference = (values - previous) & mask
    return np.stack(
        [
            ((difference >> np.uint32(8 * (plane_count - plane - 1))) & np.uint32(0xFF)).astype(np.uint8)
            for plane in range(plane_count)
        ],
        axis=1,
    )


def _decode_rows(planes: np.ndarray, pixel_type: int) -> np.ndarray:
    """Independent plane gather and segmented cumulative-sum oracle."""
    plane_count = {0: 4, 1: 2, 2: 3}[pixel_type]
    mask = np.uint64((1 << (plane_count * 8)) - 1)
    difference = np.zeros((planes.shape[0], planes.shape[2]), dtype=np.uint64)
    for plane in range(plane_count):
        difference |= planes[:, plane].astype(np.uint64) << np.uint64(8 * (plane_count - plane - 1))
    values = (np.cumsum(difference, axis=1, dtype=np.uint64) & mask).astype(np.uint32)
    return values << np.uint32(8) if pixel_type == 2 else values


def _attribute(name: str, attribute_type: str, payload: bytes) -> bytes:
    return name.encode() + b"\x00" + attribute_type.encode() + b"\x00" + struct.pack("<I", len(payload)) + payload


def _single_channel_header(*, width: int, height: int, pixel_type: int) -> bytes:
    channel_payload = b"Y\x00" + struct.pack("<iB3xii", pixel_type, 0, 1, 1) + b"\x00"
    data_window = struct.pack("<iiii", 0, 0, width - 1, height - 1)
    attributes = (
        _attribute("channels", "chlist", channel_payload),
        _attribute("compression", "compression", b"\x05"),
        _attribute("dataWindow", "box2i", data_window),
        _attribute("displayWindow", "box2i", data_window),
        _attribute("lineOrder", "lineOrder", b"\x00"),
        _attribute("pixelAspectRatio", "float", struct.pack("<f", 1.0)),
        _attribute("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0)),
        _attribute("screenWindowWidth", "float", struct.pack("<f", 1.0)),
    )
    return struct.pack("<II", 20000630, 2) + b"".join(attributes) + b"\x00"


def _mixed_channel_header(*, width: int, height: int) -> bytes:
    channel_payload = (
        b"".join(
            name.encode() + b"\x00" + struct.pack("<iB3xii", pixel_type, 0, 1, 1)
            for name, pixel_type in (("U", 0), ("H", 1), ("F", 2))
        )
        + b"\x00"
    )
    data_window = struct.pack("<iiii", 0, 0, width - 1, height - 1)
    attributes = (
        _attribute("channels", "chlist", channel_payload),
        _attribute("compression", "compression", b"\x05"),
        _attribute("dataWindow", "box2i", data_window),
        _attribute("displayWindow", "box2i", data_window),
        _attribute("lineOrder", "lineOrder", b"\x00"),
        _attribute("pixelAspectRatio", "float", struct.pack("<f", 1.0)),
        _attribute("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0)),
        _attribute("screenWindowWidth", "float", struct.pack("<f", 1.0)),
    )
    return struct.pack("<II", 20000630, 2) + b"".join(attributes) + b"\x00"


def _build_single_channel_pxr24(
    *,
    width: int,
    height: int,
    pixel_type: int,
    chunks: tuple[tuple[int, bytes], ...],
) -> bytes:
    header = _single_channel_header(width=width, height=height, pixel_type=pixel_type)
    first_chunk = len(header) + len(chunks) * 8
    offsets: list[int] = []
    records: list[bytes] = []
    cursor = first_chunk
    for y, payload in chunks:
        record = struct.pack("<ii", y, len(payload)) + payload
        offsets.append(cursor)
        records.append(record)
        cursor += len(record)
    return header + b"".join(struct.pack("<Q", offset) for offset in offsets) + b"".join(records)


def _build_mixed_channel_pxr24(
    uint_bits: np.ndarray,
    half_bits: np.ndarray,
    float_bits: np.ndarray,
) -> tuple[bytes, bytes]:
    height, width = uint_bits.shape
    assert half_bits.shape == (height, width)
    assert float_bits.shape == (height, width)
    materialized = b"".join(
        _encode_rows(bits[row : row + 1], pixel_type).tobytes()
        for row in range(height)
        for pixel_type, bits in ((0, uint_bits), (1, half_bits), (2, float_bits))
    )
    payload = zlib.compress(materialized)
    assert len(payload) < width * height * (4 + 2 + 4)
    header = _mixed_channel_header(width=width, height=height)
    chunk_offset = len(header) + 8
    chunk = struct.pack("<ii", 0, len(payload)) + payload
    return header + struct.pack("<Q", chunk_offset) + chunk, materialized


@pytest.mark.parametrize(
    ("pixel_type", "bits"),
    (
        (
            0,
            np.asarray(
                (
                    (0x00000000, 0xFFFFFFFF, 0x00000001, 0x80000000, 0x7FFFFFFF),
                    (0xFFFFFFFF, 0x00000000, 0xFFFFFFFE, 0x00000002, 0xA5A5A5A5),
                ),
                dtype=np.uint32,
            ),
        ),
        (
            1,
            np.asarray(
                (
                    (0x0000, 0xFFFF, 0x0001, 0x8000, 0x7FFF),
                    (0xFFFF, 0x0000, 0xFFFE, 0x0002, 0xA5A5),
                ),
                dtype=np.uint32,
            ),
        ),
        (
            2,
            np.asarray(
                (
                    (0x00000000, 0x80000000, 0x3F800000, 0x3F800080, 0x7F7FFFFF),
                    (0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7F800001, 0xFFC00001),
                ),
                dtype=np.uint32,
            ),
        ),
    ),
    ids=("uint", "half", "float24-special"),
)
def test_pxr24_gpu_primitive_matches_independent_plane_and_float24_oracle(
    pixel_type: int,
    bits: np.ndarray,
) -> None:
    """v1-exr-gpu-phase3 acceptance 16, 17, and 20: GPU primitives match independent bit equations."""
    cp = importlib.import_module("cupy")
    expected_planes = _encode_rows(bits, pixel_type)

    actual_planes = exr_pxr24._encode_pxr24_rows_gpu(cp.asarray(bits), pixel_type).get()
    actual_bits = exr_harness._decode_pxr24_rows_gpu(cp.asarray(actual_planes), pixel_type).get()

    np.testing.assert_array_equal(actual_planes, expected_planes)
    np.testing.assert_array_equal(actual_bits, _decode_rows(expected_planes, pixel_type))
    if pixel_type == 2:
        assert np.all((actual_bits & np.uint32(0xFF)) == 0)
        assert actual_bits[0, 4] == np.uint32(0x7F7FFF00)
        assert actual_bits[1, 3] == np.uint32(0x7F800100)


def test_pxr24_float24_rounding_carries_into_exponent_at_halfway_boundary() -> None:
    """v1-exr-gpu-phase3 acceptance 17 and 20: FLOAT24 halfway carry matches an independent bit oracle."""
    cp = importlib.import_module("cupy")
    bits = np.asarray(((0x3FFFFF80, 0xBFFFFF80),), dtype=np.uint32)
    expected_encoded = np.asarray(((0x00400000, 0x00C00000),), dtype=np.uint32)
    expected_planes = np.asarray((((0x40, 0x80), (0x00, 0x00), (0x00, 0x00)),), dtype=np.uint8)
    expected_decoded = np.asarray(((0x40000000, 0xC0000000),), dtype=np.uint32)

    oracle_encoded = _float24_encode_bits(bits)
    oracle_planes = _encode_rows(bits, 2)
    oracle_decoded = _decode_rows(oracle_planes, 2)
    actual_planes = exr_pxr24._encode_pxr24_rows_gpu(cp.asarray(bits), 2).get()
    actual_decoded = exr_harness._decode_pxr24_rows_gpu(cp.asarray(actual_planes), 2).get()

    np.testing.assert_array_equal(oracle_encoded, expected_encoded)
    np.testing.assert_array_equal(oracle_planes, expected_planes)
    np.testing.assert_array_equal(oracle_decoded, expected_decoded)
    np.testing.assert_array_equal(actual_planes, oracle_planes)
    np.testing.assert_array_equal(actual_decoded, oracle_decoded)


@pytest.mark.parametrize("pixel_type", (0, 1, 2), ids=("uint", "half", "float24"))
def test_pxr24_gpu_primitive_resets_wide_rows_without_cross_row_predictor(pixel_type: int) -> None:
    """v1-exr-gpu-phase3 acceptance 16 and 20: wide row-channel segments reset independently across blocks."""
    cp = importlib.import_module("cupy")
    random = np.random.default_rng(0x24C0DEC)
    bits = random.integers(0, 1 << 32, size=(3, 257), dtype=np.uint32)
    if pixel_type == 1:
        bits &= np.uint32(0xFFFF)
    expected_planes = _encode_rows(bits, pixel_type)

    actual_planes = exr_pxr24._encode_pxr24_rows_gpu(cp.asarray(bits), pixel_type).get()
    actual_bits = exr_harness._decode_pxr24_rows_gpu(cp.asarray(actual_planes), pixel_type).get()

    np.testing.assert_array_equal(actual_planes, expected_planes)
    np.testing.assert_array_equal(actual_bits, _decode_rows(expected_planes, pixel_type))


@pytest.mark.parametrize("backend", ("custom_cpu", "gpu"))
def test_forced_pxr24_read_bypasses_plane_decode_for_raw_stored_chunk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    """v1-exr-gpu-phase3 acceptance 5, 18, and 22: raw equality bypasses the PXR24 plane stream."""
    bits = np.asarray((0x3F800055, 0xBF0000AA, 0x7F800000, 0x7FC00001), dtype=np.uint32)
    path = tmp_path / f"pxr24-raw-{backend}.exr"
    path.write_bytes(
        _build_single_channel_pxr24(
            width=bits.size,
            height=1,
            pixel_type=2,
            chunks=((0, bits.astype("<u4").tobytes()),),
        )
    )
    container = exr_container._parse_exr_container(path)
    assert container.chunks[0].raw_stored is True
    assert container.chunks[0].phase3 is not None
    assert container.chunks[0].phase3.planes == ()
    monkeypatch.setitem(io._EXR_ROUTING, ("pxr24", "read"), backend)

    frame = px.io.read_image(path, channels=["Y"], unchanged=True, colorspace="ACEScg", gamma="linear")

    np.testing.assert_array_equal(frame.data.get().reshape(-1).view(np.uint32), bits)


@pytest.mark.parametrize(
    ("payload", "why"),
    (
        (zlib.compress(bytes(11)), "materialized size"),
        (zlib.compress(bytes(12))[:-1] + bytes((zlib.compress(bytes(12))[-1] ^ 1,)), "invalid zlib stream"),
    ),
    ids=("inflate-size", "adler"),
)
def test_pxr24_descriptor_rejects_inflate_size_and_adler_mismatch(
    tmp_path: Path,
    payload: bytes,
    why: str,
) -> None:
    """v1-exr-gpu-phase3 acceptance 18 and 22: wrapper, Adler, and plane-size corruption is actionable."""
    assert len(payload) < 16
    path = tmp_path / "invalid-pxr24.exr"
    path.write_bytes(_build_single_channel_pxr24(width=4, height=1, pixel_type=2, chunks=((0, payload),)))

    with pytest.raises(RuntimeError) as exc_info:
        exr_container._parse_exr_container(path)

    message = str(exc_info.value)
    assert "why=" in message
    assert why in message
    assert "what=chunk_y=0" in message
    assert "how=" in message


def test_pxr24_materializer_rejects_plane_row_channel_ownership_mismatch(tmp_path: Path) -> None:
    """v1-exr-gpu-phase3 acceptance 18 and 22: plane ownership is revalidated before scatter."""
    payload = zlib.compress(bytes(12))
    assert len(payload) < 16
    path = tmp_path / "pxr24-plane-ownership.exr"
    path.write_bytes(_build_single_channel_pxr24(width=4, height=1, pixel_type=2, chunks=((0, payload),)))
    container = exr_container._parse_exr_container(path)
    chunk = container.chunks[0]
    descriptor = chunk.phase3
    assert descriptor is not None
    bad_first_plane = replace(descriptor.planes[0], plane_index=1)
    bad_descriptor = replace(descriptor, planes=(bad_first_plane, *descriptor.planes[1:]))
    bad_container = replace(container, chunks=(replace(chunk, phase3=bad_descriptor),))
    prepared = exr_pxr24._prepare_exr_pxr24_read_chunks(bad_container, materialize_host=False)

    with pytest.raises(RuntimeError, match="does not own its complete byte-plane span"):
        exr_pxr24._pxr24_channel_row_records(bad_container, prepared, bad_container.parts[0].channels[0])


@pytest.mark.parametrize("backend", ("custom_cpu", "gpu"))
def test_forced_pxr24_read_materializes_selected_half_and_float_from_mixed_type_planes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    """v1-exr-gpu-phase3 acceptance 16, 18, and 20: mixed planes decode by row and selected channel."""
    height = 16
    uint_bits = np.tile(
        np.asarray((0x01020304, 0x10203040, 0x89ABCDEF, 0x00000001, 0xFFFFFFFF), dtype=np.uint32),
        (height, 1),
    )
    half_values = np.tile(np.asarray((0.5, -3.0, 7.25, 1.0, 65504.0), dtype=np.float16), (height, 1))
    half_bits = half_values.view(np.uint16).astype(np.uint32)
    float_bits = np.tile(
        np.asarray((0x3F800055, 0xBF0000AA, 0x40000080, 0x3FFFFF80, 0x00800080), dtype=np.uint32),
        (height, 1),
    )
    file_bytes, expected_materialized = _build_mixed_channel_pxr24(uint_bits, half_bits, float_bits)
    path = tmp_path / f"mixed-type-pxr24-{backend}.exr"
    path.write_bytes(file_bytes)

    container = exr_container._parse_exr_container(path)
    assert container.phase3_eligible is True
    assert tuple(channel.pixel_type for channel in container.parts[0].channels) == (0, 1, 2)
    assert container.chunks[0].raw_stored is False
    prepared = exr_pxr24._prepare_exr_pxr24_read_chunks(container, materialize_host=True)
    np.testing.assert_array_equal(prepared.host_materialized, np.frombuffer(expected_materialized, dtype=np.uint8))
    monkeypatch.setitem(io._EXR_ROUTING, ("pxr24", "read"), backend)

    frame = px.io.read_image(path, channels=["H", "F"], unchanged=False, colorspace="ACEScg", gamma="linear")

    expected_half_bits = _decode_rows(_encode_rows(half_bits, 1), 1).astype(np.uint16)
    expected_float_bits = _decode_rows(_encode_rows(float_bits, 2), 2)
    expected = np.stack(
        (expected_half_bits.view(np.float16).astype(np.float32), expected_float_bits.view(np.float32)),
        axis=2,
    )
    assert frame.channels == ("H", "F")
    np.testing.assert_array_equal(frame.data.get().view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("backend", ("custom_cpu", "gpu"))
def test_openexr_pxr24_write_round_trips_through_both_forced_read_lanes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    """v1-exr-gpu-phase3 acceptance 5, 18, 20, and 21: OpenEXR output matches both internal read lanes."""
    from openexr_dev_oracle import OpenEXR

    height, width = 33, 3
    half = np.zeros((height, width), dtype=np.float16)
    single_bits = np.full((height, width), np.uint32(0x3F800055), dtype=np.uint32)
    half[-1] = np.asarray((0.5, -3.0, 7.25), dtype=np.float16)
    single_bits[-1] = np.asarray((0xBF0000AA, 0x7F7FFFFF, 0x7F800001), dtype=np.uint32)
    single = single_bits.view(np.float32)
    path = tmp_path / f"openexr-pxr24-{backend}.exr"
    OpenEXR.File({"compression": OpenEXR.PXR24_COMPRESSION}, {"H": half, "F": single}).write(str(path))
    container = exr_container._parse_exr_container(path)
    assert container.phase3_eligible is True
    assert tuple(chunk.raw_stored for chunk in container.chunks) == (False, False, True)
    monkeypatch.setitem(io._EXR_ROUTING, ("pxr24", "read"), backend)

    frame = px.io.read_image(path, channels=["H", "F"], unchanged=False, colorspace="ACEScg", gamma="linear")

    expected_single_bits = single_bits.copy()
    expected_single_bits[:32] = _float24_encode_bits(single_bits[:32]) << np.uint32(8)
    expected = np.stack((half.astype(np.float32), expected_single_bits.view(np.float32)), axis=2)
    assert frame.channels == ("H", "F")
    assert frame.data.flags.c_contiguous
    np.testing.assert_array_equal(frame.data.get().view(np.uint32), expected.view(np.uint32))
    unchanged_half = px.io.read_image(path, channels=["H"], unchanged=True, colorspace="ACEScg", gamma="linear")
    assert unchanged_half.data.dtype.name == "float16"
    np.testing.assert_array_equal(unchanged_half.data.get().view(np.uint16), half[..., np.newaxis].view(np.uint16))


@pytest.mark.parametrize("dtype", (np.float16, np.float32))
def test_gpu_pxr24_write_round_trips_in_openexr_with_partial_raw_chunk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dtype: type[np.generic],
) -> None:
    """v1-exr-gpu-phase3 acceptance 9, 17, 19, and 21: GPU output cross-decodes with mixed chunk storage."""
    import cupy as cp
    from openexr_dev_oracle import OpenEXR

    height, width = 33, 3
    values = np.zeros((height, width), dtype=dtype)
    values[:32] = dtype(1.00013)
    values[-1] = np.asarray((-3.125, 0.33337, 11.75), dtype=dtype)
    frame = px.io.from_array(cp.asarray(values[..., np.newaxis]), colorspace="ACEScg", gamma="linear", channels=("Y",))
    path = tmp_path / f"gpu-pxr24-{np.dtype(dtype).name}.exr"
    calls = 0
    original = io._write_exr_gpu

    def gpu_spy(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setitem(io._EXR_ROUTING, ("pxr24", "write"), "gpu")
    monkeypatch.setattr(io, "_write_exr_gpu", gpu_spy)

    px.io.write_image(path, frame, compression="pxr24", dtype=np.dtype(dtype).name)

    container = exr_container._parse_exr_container(path)
    assert calls == 1
    assert tuple(chunk.raw_stored for chunk in container.chunks) == (False, False, True)
    reference = OpenEXR.File(str(path), separate_channels=True)
    decoded = np.asarray(reference.channels()["Y"].pixels)
    assert reference.header()["compression"] == OpenEXR.PXR24_COMPRESSION
    if dtype == np.float16:
        np.testing.assert_array_equal(decoded.view(np.uint16), values.view(np.uint16))
    else:
        expected_bits = values.view(np.uint32).copy()
        expected_bits[:32] = _float24_encode_bits(expected_bits[:32]) << np.uint32(8)
        np.testing.assert_array_equal(decoded.view(np.uint32), expected_bits)


def test_pxr24_lanes_keep_batch_and_image_level_transfer_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1-exr-gpu-phase3 acceptance 8, 9, 18, and 19: each PXR24 read lane stages one prepared batch."""
    from openexr_dev_oracle import OpenEXR

    values = np.linspace(-3.0, 5.0, 33 * 17, dtype=np.float32).reshape(33, 17)
    path = tmp_path / "pxr24-transfer.exr"
    OpenEXR.File({"compression": OpenEXR.PXR24_COMPRESSION}, {"Y": values}).write(str(path))
    container = exr_container._parse_exr_container(path)
    selected = container.parts[0].channels
    gpu_prepared = exr_pxr24._prepare_exr_pxr24_read_chunks(container, materialize_host=False)
    cpu_prepared = exr_pxr24._prepare_exr_pxr24_read_chunks(container, materialize_host=True)
    transfers = exr_harness._record_cupy_transfers(monkeypatch)

    gpu_actual = exr_pxr24._read_exr_pxr24_gpu(container, selected, output_dtype="float32")

    exr_harness._assert_cupy_transfer_budget(
        transfers, direction="h2d", max_count=8, max_total_nbytes=1_559, max_shape_elements=440
    )
    exr_harness._assert_cupy_transfer_budget(
        transfers, direction="d2h", max_count=0, max_total_nbytes=0, max_shape_elements=0
    )

    gpu_batches = [
        transfer
        for transfer in transfers
        if transfer.direction == "h2d"
        and transfer.nbytes == gpu_prepared.host_staging.nbytes
        and transfer.shape == gpu_prepared.host_staging.shape
        and transfer.dtype == gpu_prepared.host_staging.dtype.name
    ]
    assert len(gpu_batches) == 1
    assert gpu_actual.shape == (33, 17, 1)
    assert gpu_actual.dtype == np.dtype(np.float32)

    transfers.clear()
    cpu_actual = exr_pxr24._read_exr_pxr24_custom_cpu(container, selected, output_dtype="float32")
    exr_harness._assert_cupy_transfer_budget(
        transfers, direction="h2d", max_count=8, max_total_nbytes=2_802, max_shape_elements=1_683
    )
    exr_harness._assert_cupy_transfer_budget(
        transfers, direction="d2h", max_count=0, max_total_nbytes=0, max_shape_elements=0
    )
    cpu_batches = [
        transfer
        for transfer in transfers
        if transfer.direction == "h2d"
        and transfer.nbytes == cpu_prepared.host_staging.nbytes
        and transfer.shape == cpu_prepared.host_staging.shape
        and transfer.dtype == cpu_prepared.host_staging.dtype.name
    ]
    assert len(cpu_batches) == 1
    assert cpu_actual.shape == gpu_actual.shape
    assert cpu_actual.dtype == gpu_actual.dtype
