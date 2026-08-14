"""Specification tests for the Phase 3 OpenEXR RLE GPU and custom-CPU lanes."""

from __future__ import annotations

import importlib
import struct
from pathlib import Path

import exr_test_harness as exr_harness
import numpy as np
import pytest

import pixtreme as px
import pixtreme._io.formats.exr.codec_rle as exr_rle
import pixtreme._io.formats.exr.container as exr_container
import pixtreme._io.formats.exr.selection as io


def _attribute(name: str, attribute_type: str, payload: bytes) -> bytes:
    return name.encode() + b"\x00" + attribute_type.encode() + b"\x00" + struct.pack("<I", len(payload)) + payload


def _single_channel_header(*, width: int, height: int, pixel_type: int) -> bytes:
    channel_payload = b"Y\x00" + struct.pack("<iB3xii", pixel_type, 0, 1, 1) + b"\x00"
    data_window = struct.pack("<iiii", 0, 0, width - 1, height - 1)
    attributes = (
        _attribute("channels", "chlist", channel_payload),
        _attribute("compression", "compression", b"\x01"),
        _attribute("dataWindow", "box2i", data_window),
        _attribute("displayWindow", "box2i", data_window),
        _attribute("lineOrder", "lineOrder", b"\x00"),
        _attribute("pixelAspectRatio", "float", struct.pack("<f", 1.0)),
        _attribute("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0)),
        _attribute("screenWindowWidth", "float", struct.pack("<f", 1.0)),
    )
    return struct.pack("<II", 20000630, 2) + b"".join(attributes) + b"\x00"


def _build_single_channel_rle(*, width: int, pixel_type: int, payloads: tuple[bytes, ...]) -> bytes:
    header = _single_channel_header(width=width, height=len(payloads), pixel_type=pixel_type)
    first_chunk = len(header) + len(payloads) * 8
    offsets: list[int] = []
    chunks: list[bytes] = []
    cursor = first_chunk
    for y, payload in enumerate(payloads):
        chunk = struct.pack("<ii", y, len(payload)) + payload
        offsets.append(cursor)
        chunks.append(chunk)
        cursor += len(chunk)
    return header + b"".join(struct.pack("<Q", offset) for offset in offsets) + b"".join(chunks)


def _restore_rle_transform(transformed: bytes) -> bytes:
    """Independent host oracle for predictor restore followed by even/odd interleave."""
    predicted = np.frombuffer(transformed, dtype=np.uint8).astype(np.int64)
    grouped = np.bitwise_and(np.cumsum(predicted) - np.arange(predicted.size) * 128, 255).astype(np.uint8)
    raw = np.empty_like(grouped)
    half = (grouped.size + 1) // 2
    raw[::2] = grouped[:half]
    raw[1::2] = grouped[half:]
    return raw.tobytes()


def _decode_packets(payload: bytes, *, expected_size: int) -> bytes:
    """Independent host packet oracle accepting the complete signed-header grammar."""
    decoded = bytearray()
    offset = 0
    while offset < len(payload):
        header = struct.unpack_from("b", payload, offset)[0]
        offset += 1
        if header < 0:
            size = -header
            decoded.extend(payload[offset : offset + size])
            offset += size
        else:
            size = header + 1
            decoded.extend(payload[offset : offset + 1] * size)
            offset += 1
    assert offset == len(payload)
    assert len(decoded) == expected_size
    return bytes(decoded)


@pytest.mark.parametrize("backend", ("custom_cpu", "gpu"))
def test_forced_rle_read_accepts_full_signed_header_grammar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    """v1-exr-gpu-phase3 acceptance 5, 11, 12, and 14: forced lanes accept every valid packet boundary."""
    literal_128 = bytes((index * 37 + 11) & 0xFF for index in range(128))
    transformed = literal_128 + b"\x33" + b"\xa5" + b"\x7c" * 128 + b"\x19\xe2"
    payload = b"\x80" + literal_128 + b"\xff\x33" + b"\x00\xa5" + b"\x7f\x7c" + b"\xfe\x19\xe2"
    assert _decode_packets(payload, expected_size=260) == transformed
    expected_raw = _restore_rle_transform(transformed)
    path = tmp_path / f"noncanonical-rle-{backend}.exr"
    path.write_bytes(_build_single_channel_rle(width=65, pixel_type=2, payloads=(payload,)))

    container = exr_container._parse_exr_container(path)
    assert container.phase3_eligible is True
    assert container.chunks[0].raw_stored is False
    monkeypatch.setitem(io._EXR_ROUTING, ("rle", "read"), backend)

    frame = px.io.read_image(path, channels=["Y"], unchanged=True, colorspace="ACEScg", gamma="linear")

    actual_bits = frame.data.get().reshape(-1).view(np.uint32)
    expected_bits = np.frombuffer(expected_raw, dtype="<u4")
    np.testing.assert_array_equal(actual_bits, expected_bits)


@pytest.mark.parametrize("backend", ("custom_cpu", "gpu"))
@pytest.mark.parametrize(
    ("payload", "produced_size", "why"),
    (
        (b"\x0e\xa5", 15, "does not produce"),
        (b"\x10\xa5", 17, "expands beyond"),
    ),
    ids=("underflow", "overflow"),
)
def test_forced_rle_read_rejects_complete_packet_stream_with_wrong_materialized_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    payload: bytes,
    produced_size: int,
    why: str,
) -> None:
    """v1-exr-runtime-independence acceptance 43 and 45: RLE size mismatch fails before materialization."""
    expected_size = 16
    assert len(payload) < expected_size
    path = tmp_path / f"rle-{backend}-{produced_size}.exr"
    path.write_bytes(_build_single_channel_rle(width=4, pixel_type=2, payloads=(payload,)))
    monkeypatch.setitem(io._EXR_ROUTING, ("rle", "read"), backend)
    materialize = "_read_exr_gpu" if backend == "gpu" else "_read_exr_custom_cpu"
    monkeypatch.setattr(
        io,
        materialize,
        lambda *_args, **_kwargs: pytest.fail("an invalid RLE descriptor must fail before payload materialization"),
    )

    with pytest.raises(RuntimeError) as exc_info:
        px.io.read_image(path, channels=["Y"], unchanged=True, colorspace="ACEScg", gamma="linear")

    message = str(exc_info.value)
    assert "why=" in message
    assert why in message
    assert "what=chunk_y=0, " in message
    assert f"expected={expected_size}" in message
    assert f"{produced_size}" in message
    assert "how=" in message


def test_rle_gpu_packet_encoder_emits_only_canonical_boundaries() -> None:
    """v1-exr-gpu-phase3 acceptance 8 and 13: GPU packet scan emits canonical literal and run packets."""
    cp = importlib.import_module("cupy")
    literal = bytes(range(127))
    first = literal + b"\xc8" * 3 + b"\xc9" * 128 + b"\xca" * 2
    second = b"\x07" * 128
    transformed = cp.asarray(np.frombuffer(first + second, dtype=np.uint8))

    encoded, offsets, sizes = exr_rle._encode_rle_packets_gpu(
        transformed,
        (0, len(first)),
        (len(first), len(second)),
    )
    host = encoded.get().tobytes()

    expected_first = b"\x81" + literal + b"\x02\xc8" + b"\x7f\xc9" + b"\xfe\xca\xca"
    expected_second = b"\x7f\x07"
    assert offsets == (0, len(expected_first))
    assert sizes == (len(expected_first), len(expected_second))
    assert host == expected_first + expected_second
    assert _decode_packets(host[offsets[0] : offsets[0] + sizes[0]], expected_size=len(first)) == first
    assert _decode_packets(host[offsets[1] : offsets[1] + sizes[1]], expected_size=len(second)) == second


@pytest.mark.parametrize(
    ("run_length", "expected_packets", "expected_literal_classes"),
    (
        (128, (b"\x7f\x5a",), (False,)),
        (129, (b"\x7f\x5a", b"\xff\x5a"), (False, True)),
        (130, (b"\x7f\x5a", b"\xfe\x5a\x5a"), (False, True)),
        (256, (b"\x7f\x5a", b"\x7f\x5a"), (False, False)),
        (257, (b"\x7f\x5a", b"\x7f\x5a", b"\xff\x5a"), (False, False, True)),
        (258, (b"\x7f\x5a", b"\x7f\x5a", b"\xfe\x5a\x5a"), (False, False, True)),
    ),
)
def test_rle_gpu_packet_encoder_splits_long_run_short_tails_canonically(
    run_length: int,
    expected_packets: tuple[bytes, ...],
    expected_literal_classes: tuple[bool, ...],
) -> None:
    """v1-exr-gpu-phase3 acceptance 13: long runs leave one- and two-byte tails as canonical literals."""
    cp = importlib.import_module("cupy")
    transformed_bytes = b"\x5a" * run_length

    encoded, offsets, sizes = exr_rle._encode_rle_packets_gpu(
        cp.asarray(np.frombuffer(transformed_bytes, dtype=np.uint8)),
        (0,),
        (run_length,),
    )
    host = encoded.get().tobytes()
    packets = exr_container._parse_phase3_rle_packets(host, payload_start=0, chunk_y=0, expected_size=run_length)
    packet_bytes = tuple(host[packet.input_span.start : packet.input_span.end] for packet in packets)

    assert offsets == (0,)
    assert sizes == (sum(map(len, expected_packets)),)
    assert packet_bytes == expected_packets
    assert tuple(packet.literal for packet in packets) == expected_literal_classes
    assert _decode_packets(host, expected_size=run_length) == transformed_bytes


@pytest.mark.parametrize("backend", ("custom_cpu", "gpu"))
def test_openexr_rle_write_round_trips_through_both_forced_read_lanes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    """v1-exr-gpu-phase3 acceptance 5, 9, and 15: OpenEXR output is bit-exact in both internal read lanes."""
    from openexr_dev_oracle import OpenEXR

    height, width = 5, 96
    random = np.random.default_rng(0x51A7)
    half = np.empty((height, width), dtype=np.float16)
    single = np.empty((height, width), dtype=np.float32)
    half[0::2] = np.float16(0.375)
    single[0::2] = np.float32(-0.625)
    half[1::2] = random.uniform(-2.0, 3.0, size=half[1::2].shape).astype(np.float16)
    single[1::2] = random.uniform(-4.0, 5.0, size=single[1::2].shape).astype(np.float32)
    path = tmp_path / f"openexr-rle-{backend}.exr"
    OpenEXR.File({"compression": OpenEXR.RLE_COMPRESSION}, {"H": half, "F": single}).write(str(path))
    reference_file = OpenEXR.File(str(path), separate_channels=True)
    reference = np.stack(
        [np.asarray(reference_file.channels()[name].pixels).astype(np.float32) for name in ("H", "F")], axis=2
    )
    container = exr_container._parse_exr_container(path)
    assert container.phase3_eligible is True
    assert any(not chunk.raw_stored for chunk in container.chunks)
    assert any(chunk.raw_stored for chunk in container.chunks)
    monkeypatch.setitem(io._EXR_ROUTING, ("rle", "read"), backend)

    frame = px.io.read_image(
        path,
        channels=["H", "F"],
        unchanged=False,
        colorspace="ACEScg",
        gamma="linear",
    )

    assert frame.channels == ("H", "F")
    assert frame.data.flags.c_contiguous
    np.testing.assert_array_equal(frame.data.get(), reference)


@pytest.mark.parametrize("dtype", (np.float16, np.float32))
def test_gpu_rle_write_round_trips_in_openexr_with_compressed_and_raw_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dtype: type[np.generic],
) -> None:
    """v1-exr-gpu-phase3 acceptance 9, 13, and 15: GPU write returns only final mixed payloads and is bit-exact."""
    import cupy as cp
    from openexr_dev_oracle import OpenEXR

    height, width = 4, 257
    random = np.random.default_rng(0xC0DEC)
    values = np.empty((height, width), dtype=dtype)
    values[0] = dtype(0.0)
    values[1] = random.uniform(-8.0, 8.0, size=width).astype(dtype)
    values[2] = dtype(0.0)
    values[3] = random.uniform(-16.0, 16.0, size=width).astype(dtype)
    frame = px.io.from_array(
        cp.asarray(values[..., np.newaxis]),
        colorspace="ACEScg",
        gamma="linear",
        channels=("Y",),
    )
    path = tmp_path / f"gpu-rle-{np.dtype(dtype).name}.exr"
    calls = 0
    original = io._write_exr_gpu

    def gpu_spy(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setitem(io._EXR_ROUTING, ("rle", "write"), "gpu")
    monkeypatch.setattr(io, "_write_exr_gpu", gpu_spy)

    px.io.write_image(path, frame, compression="rle", dtype=np.dtype(dtype).name)

    container = exr_container._parse_exr_container(path)
    assert calls == 1
    assert tuple(chunk.raw_stored for chunk in container.chunks) == (False, True, False, True)
    reference = OpenEXR.File(str(path), separate_channels=True)
    decoded = np.asarray(reference.channels()["Y"].pixels)
    assert reference.header()["compression"] == OpenEXR.RLE_COMPRESSION
    np.testing.assert_array_equal(decoded, values)


def test_rle_lanes_keep_image_level_transfer_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1-exr-gpu-phase3 acceptance 8 and 9: each RLE read lane crosses H2D once with one image batch."""
    from openexr_dev_oracle import OpenEXR

    values = np.resize(np.asarray((0.0, 0.0, 1.0, -2.0, 3.5), dtype=np.float32), 4 * 257).reshape(4, 257)
    path = tmp_path / "rle-transfer.exr"
    OpenEXR.File({"compression": OpenEXR.RLE_COMPRESSION}, {"Y": values}).write(str(path))
    container = exr_container._parse_exr_container(path)
    selected = container.parts[0].channels
    prepared = exr_rle._prepare_exr_rle_read_chunks(container)
    transfers = exr_harness._record_cupy_transfers(monkeypatch)

    gpu_actual = exr_rle._read_exr_rle_gpu(container, selected, output_dtype="float32")

    exr_harness._assert_cupy_transfer_budget(
        transfers, direction="h2d", max_count=19, max_total_nbytes=4_464, max_shape_elements=4_072
    )
    exr_harness._assert_cupy_transfer_budget(
        transfers, direction="d2h", max_count=0, max_total_nbytes=0, max_shape_elements=0
    )

    gpu_batches = [
        transfer
        for transfer in transfers
        if transfer.direction == "h2d"
        and transfer.nbytes == prepared.host_staging.nbytes
        and transfer.shape == prepared.host_staging.shape
        and transfer.dtype == prepared.host_staging.dtype.name
    ]
    assert len(gpu_batches) == 1
    assert gpu_actual.shape == (4, 257, 1)
    assert gpu_actual.dtype == np.dtype(np.float32)

    transfers.clear()
    cpu_actual = exr_rle._read_exr_rle_custom_cpu(container, selected, output_dtype="float32")
    decoded_bytes = int(prepared.decoded_sizes.sum())
    exr_harness._assert_cupy_transfer_budget(
        transfers, direction="h2d", max_count=14, max_total_nbytes=4_372, max_shape_elements=4_112
    )
    exr_harness._assert_cupy_transfer_budget(
        transfers, direction="d2h", max_count=0, max_total_nbytes=0, max_shape_elements=0
    )
    cpu_batches = [
        transfer
        for transfer in transfers
        if transfer.direction == "h2d"
        and transfer.nbytes == decoded_bytes
        and transfer.shape == (decoded_bytes,)
        and transfer.dtype == "uint8"
    ]
    assert len(cpu_batches) == 1
    assert cpu_actual.shape == gpu_actual.shape
    assert cpu_actual.dtype == gpu_actual.dtype
