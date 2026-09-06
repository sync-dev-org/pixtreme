"""Specification tests for the DPX file boundary."""

from __future__ import annotations

import os
import struct
import subprocess
import sys
import warnings
from pathlib import Path

import cupy as cp
import numpy as np
import pytest

import pixtreme as px

ROOT = Path(__file__).resolve().parents[1]
_ACTIONABLE = r"why=.*what=.*how="
_DATA_OFFSET = 2048


def _pack_samples(samples: np.ndarray, *, bit_depth: int, byte_order: str) -> bytes:
    """Pack one scanline from the ST 268-1 Method A bit diagrams, independently."""
    values = np.asarray(samples, dtype=np.uint16).reshape(-1)
    endian = "big" if byte_order == ">" else "little"
    if bit_depth == 8:
        return values.astype(np.uint8).tobytes()
    if bit_depth == 10:
        packed = bytearray()
        for offset in range(0, values.size, 3):
            group = np.zeros(3, dtype=np.uint32)
            source = values[offset : offset + 3]
            group[: source.size] = source
            word = (int(group[0]) << 22) | (int(group[1]) << 12) | (int(group[2]) << 2)
            packed.extend(word.to_bytes(4, endian))
        return bytes(packed)
    shift = 4 if bit_depth == 12 else 0
    return b"".join((int(value) << shift).to_bytes(2, endian) for value in values)


def _dpx_fixture(
    codes: np.ndarray,
    *,
    bit_depth: int,
    byte_order: str = ">",
    transfer: int = 1,
    descriptor: int | None = None,
    packing: int | None = None,
    encoding: int = 0,
    orientation: int = 0,
    elements: int = 1,
    signed: int = 0,
    element_offset: int = _DATA_OFFSET,
    file_offset: int | None = None,
    generic_header_length: int = 1664,
    industry_header_length: int = 384,
    user_header_length: int = 0,
    encryption_key: int = 0xFFFFFFFF,
    file_size_delta: int = 0,
    eol_padding: int = 0,
) -> bytes:
    """Build a minimal single-element DPX without using production helpers."""
    source = np.asarray(codes, dtype=np.uint16)
    height, width, channels = source.shape
    if descriptor is None:
        descriptor = {3: 50, 4: 51}[channels]
    if packing is None:
        packing = 1 if bit_depth in (10, 12) else 0
    rows = [_pack_samples(row, bit_depth=bit_depth, byte_order=byte_order) for row in source]
    payload = b"".join(row + bytes(eol_padding) for row in rows)
    file_size = element_offset + len(payload)
    header = bytearray(b"\xff" * element_offset)
    ascii_ranges = [
        (36, 660),
        *((820 + element * 72, 852 + element * 72) for element in range(8)),
        (1432, 1620),
        (1664, 1712),
        (1732, 1864),
    ]
    for start, end in ascii_ranges:
        if start < len(header):
            clipped_end = min(end, len(header))
            header[start:clipped_end] = bytes(clipped_end - start)
    header[:4] = b"SDPX" if byte_order == ">" else b"XPDS"
    header[8:16] = b"V2.0\0\0\0\0"
    struct.pack_into(f"{byte_order}I", header, 4, element_offset if file_offset is None else file_offset)
    struct.pack_into(f"{byte_order}I", header, 16, file_size + file_size_delta)
    struct.pack_into(f"{byte_order}I", header, 20, 1)
    struct.pack_into(f"{byte_order}I", header, 24, generic_header_length)
    struct.pack_into(f"{byte_order}I", header, 28, industry_header_length)
    struct.pack_into(f"{byte_order}I", header, 32, user_header_length)
    struct.pack_into(f"{byte_order}I", header, 660, encryption_key)
    struct.pack_into(f"{byte_order}H", header, 768, orientation)
    struct.pack_into(f"{byte_order}H", header, 770, elements)
    struct.pack_into(f"{byte_order}I", header, 772, width)
    struct.pack_into(f"{byte_order}I", header, 776, height)
    struct.pack_into(f"{byte_order}I", header, 780, signed)
    struct.pack_into(f"{byte_order}I", header, 784, 0)
    struct.pack_into(f"{byte_order}I", header, 792, (1 << min(bit_depth, 16)) - 1)
    header[800] = descriptor
    header[801] = transfer
    header[802] = 6
    header[803] = bit_depth
    struct.pack_into(f"{byte_order}H", header, 804, packing)
    struct.pack_into(f"{byte_order}H", header, 806, encoding)
    struct.pack_into(f"{byte_order}I", header, 808, element_offset)
    struct.pack_into(f"{byte_order}I", header, 812, eol_padding)
    struct.pack_into(f"{byte_order}I", header, 816, 0)
    if len(header) > 1931:
        header[1931] = 0
    return bytes(header) + payload


def _unpack_written_dpx(payload: bytes) -> tuple[np.ndarray, dict[str, int]]:
    """Decode writer bytes from the normative bit layout without production code."""
    assert payload[:4] == b"SDPX"
    width, height = struct.unpack_from(">II", payload, 772)
    descriptor, transfer, bit_depth = payload[800], payload[801], payload[803]
    packing, encoding = struct.unpack_from(">HH", payload, 804)
    data_offset, eol_padding = struct.unpack_from(">II", payload, 808)
    channels = {50: 3, 51: 4}[descriptor]
    sample_count = width * channels
    row_bytes = (
        sample_count if bit_depth == 8 else ((sample_count + 2) // 3 * 4 if bit_depth == 10 else sample_count * 2)
    )
    output = np.empty((height, width, channels), dtype=np.uint16)
    offset = data_offset
    for row_index in range(height):
        row = payload[offset : offset + row_bytes]
        if bit_depth == 8:
            samples = np.frombuffer(row, dtype=np.uint8).astype(np.uint16)
        elif bit_depth == 10:
            samples_list: list[int] = []
            for word_offset in range(0, len(row), 4):
                word = int.from_bytes(row[word_offset : word_offset + 4], "big")
                samples_list.extend(((word >> 22) & 0x3FF, (word >> 12) & 0x3FF, (word >> 2) & 0x3FF))
            samples = np.asarray(samples_list[:sample_count], dtype=np.uint16)
        else:
            samples = np.frombuffer(row, dtype=">u2").astype(np.uint16)
            if bit_depth == 12:
                samples >>= 4
        output[row_index] = samples.reshape(width, channels)
        offset += row_bytes + eol_padding
    assert offset == len(payload)
    return output, {
        "width": width,
        "height": height,
        "descriptor": descriptor,
        "transfer": transfer,
        "bit_depth": bit_depth,
        "packing": packing,
        "encoding": encoding,
        "data_offset": data_offset,
    }


def _assert_written_header_is_st268(payload: bytes, *, width: int, height: int) -> None:
    """Check ST 268-1 header fields by normative byte offsets, independently of production code."""
    undefined = b"\xff"
    assert payload[:4] == b"SDPX"
    assert payload[8:16] == b"V2.0\0\0\0\0"
    assert struct.unpack_from(">I", payload, 4)[0] == 2048
    assert struct.unpack_from(">I", payload, 16)[0] == len(payload)
    assert struct.unpack_from(">IIII", payload, 20) == (1, 1664, 384, 0)
    assert payload[36:660] == bytes(624)
    assert struct.unpack_from(">I", payload, 660)[0] == 0xFFFFFFFF
    assert payload[664:768] == undefined * 104

    for element in range(8):
        element_start = 780 + element * 72
        if element > 0:
            assert payload[element_start : element_start + 40] == undefined * 40
        assert payload[element_start + 40 : element_start + 72] == bytes(32)
    assert payload[1356:1408] == undefined * 52

    assert struct.unpack_from(">II", payload, 1408) == (0, 0)
    assert payload[1416:1424] == undefined * 8
    assert struct.unpack_from(">II", payload, 1424) == (width, height)
    assert payload[1432:1620] == bytes(188)
    assert payload[1620:1664] == undefined * 44

    assert payload[1664:1712] == bytes(48)
    assert payload[1712:1732] == undefined * 20
    assert payload[1732:1864] == bytes(132)
    assert payload[1864:1931] == undefined * 67
    assert payload[1931] == 0
    assert payload[1932:2048] == undefined * 116


def _codes(bit_depth: int, channels: int) -> np.ndarray:
    maximum = (1 << bit_depth) - 1
    values = np.array((0, 1, maximum // 4, maximum // 2, maximum - 1, maximum), dtype=np.uint16)
    return np.resize(values, (2, 3, channels))


@pytest.mark.parametrize("bit_depth", (8, 10, 12, 16))
@pytest.mark.parametrize("byte_order", (">", "<"), ids=("big", "little"))
@pytest.mark.parametrize("channels", (3, 4), ids=("rgb", "rgba"))
def test_dpx_read_decodes_both_endians_depths_and_descriptors(
    tmp_path: Path, bit_depth: int, byte_order: str, channels: int
) -> None:
    """v1-dpx acceptance 1, 3, and 13: independent filled fixtures decode on the exact quantization grid."""
    codes = _codes(bit_depth, channels)
    path = tmp_path / "fixture.DPX"
    path.write_bytes(_dpx_fixture(codes, bit_depth=bit_depth, byte_order=byte_order, eol_padding=4))

    actual = px.io.read_image(path)

    assert (actual.dtype, actual.channels, actual.colorspace, actual.gamma) == (
        np.dtype(np.float32),
        tuple("RGBA"[:channels]),
        "Rec.709",
        "Cineon",
    )
    assert actual.data.flags.c_contiguous
    expected = codes.astype(np.float32) / np.float32((1 << bit_depth) - 1)
    np.testing.assert_array_equal(
        px.io.to_array(
            actual,
        ).get(),
        expected,
    )


@pytest.mark.parametrize("bit_depth", (8, 10, 12, 16))
def test_dpx_read_unchanged_returns_native_integer_codes_and_selects_labels(tmp_path: Path, bit_depth: int) -> None:
    """v1-dpx acceptance 2 and 3: unchanged and duplicate label selection share the GPU unpack path."""
    codes = _codes(bit_depth, 4)
    path = tmp_path / "unchanged.dpx"
    path.write_bytes(_dpx_fixture(codes, bit_depth=bit_depth, byte_order="<"))

    actual = px.io.read_image(path, unchanged=True, channels=("B", "R", "R"), colorspace="ACEScg", gamma="linear")

    assert actual.dtype == np.dtype(np.uint8 if bit_depth == 8 else np.uint16)
    assert (actual.channels, actual.colorspace, actual.gamma) == (("B", "R", "R"), "ACEScg", "linear")
    np.testing.assert_array_equal(
        px.io.to_array(
            actual,
        ).get(),
        codes[..., (2, 0, 0)].astype(actual.dtype),
    )


def test_dpx_read_resets_partial_ten_bit_groups_at_each_scanline(tmp_path: Path) -> None:
    """v1-dpx acceptance 1 and 13: RGBA rows with a partial final word never bleed into the next scanline."""
    codes = np.array([[[1, 2, 3, 4]], [[1020, 1021, 1022, 1023]]], dtype=np.uint16)
    path = tmp_path / "partial-word.dpx"
    path.write_bytes(_dpx_fixture(codes, bit_depth=10, byte_order="<"))

    actual = px.io.read_image(path, unchanged=True)

    np.testing.assert_array_equal(
        px.io.to_array(
            actual,
        ).get(),
        codes,
    )


def test_dpx_read_transfers_only_flat_packed_uint8_bytes_before_gpu_unpack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v1-dpx acceptance 2: H2D sees one packed byte buffer and no host decoded integer or float image."""
    codes = _codes(10, 4)
    path = tmp_path / "transfer.dpx"
    path.write_bytes(_dpx_fixture(codes, bit_depth=10, byte_order="<", eol_padding=4))
    original_asarray = cp.asarray
    host_inputs: list[np.ndarray] = []

    def capture_asarray(value: object, *args: object, **kwargs: object) -> cp.ndarray:
        if isinstance(value, np.ndarray):
            host_inputs.append(value)
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr("pixtreme._io.formats.dpx.cp.asarray", capture_asarray)

    px.io.read_image(path)

    row_bytes = ((codes.shape[1] * codes.shape[2] + 2) // 3) * 4
    assert len(host_inputs) == 1
    assert (host_inputs[0].dtype, host_inputs[0].shape) == (
        np.dtype(np.uint8),
        (codes.shape[0] * (row_bytes + 4),),
    )


@pytest.mark.parametrize(
    ("transfer", "bit_depth", "gamma", "mappable"),
    (
        (1, 10, "Cineon", True),
        (3, 10, "Cineon", True),
        (13, 10, "Cineon", True),
        (2, 12, "linear", True),
        (4, 8, "Rec.709", True),
        (5, 8, "Rec.709", True),
        (6, 8, "Rec.709", True),
        (7, 8, "Rec.709", True),
        (8, 8, "Rec.709", True),
        (9, 8, "Rec.709", True),
        (10, 8, "Rec.709", True),
        (255, 10, "Cineon", False),
        (255, 8, "Rec.709", False),
        (255, 12, "linear", False),
        (255, 16, "linear", False),
    ),
)
def test_dpx_transfer_mapping_and_depth_fallback(
    tmp_path: Path, transfer: int, bit_depth: int, gamma: str, mappable: bool
) -> None:
    """v1-dpx acceptance 4: transfer codes map explicitly and unknown values use depth defaults with warning."""
    path = tmp_path / "transfer.dpx"
    path.write_bytes(_dpx_fixture(_codes(bit_depth, 3), bit_depth=bit_depth, transfer=transfer))

    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        header = px.io.read_header(path)
        frame = px.io.read_image(path)

    assert (header.color.colorspace, header.color.gamma, header.color.mappable) == ("Rec.709", gamma, mappable)
    assert frame.gamma == gamma
    assert bool(observed) is (not mappable)


@pytest.mark.parametrize(
    ("overrides", "needle"),
    (
        ({"orientation": 1}, "orientation"),
        ({"elements": 2}, "element"),
        ({"signed": 1}, "unsigned"),
        ({"descriptor": 6}, "descriptor"),
        ({"encoding": 1}, "RLE"),
        ({"bit_depth": 7}, "bit depth"),
        ({"bit_depth": 10, "packing": 0}, "packing"),
        ({"bit_depth": 12, "packing": 2}, "packing"),
        ({"bit_depth": 8, "packing": 1}, "packing"),
    ),
)
def test_dpx_read_rejects_every_out_of_scope_header_before_decode(
    tmp_path: Path, overrides: dict[str, int], needle: str
) -> None:
    """v1-dpx acceptance 1: unsupported structural variants are actionable closed sets."""
    bit_depth = overrides.get("bit_depth", 10)
    options = {"bit_depth": bit_depth, **overrides}
    fixture = _dpx_fixture(_codes(bit_depth if bit_depth in (8, 10, 12, 16) else 8, 3), **options)
    path = tmp_path / "invalid.dpx"
    path.write_bytes(fixture)

    with pytest.raises(ValueError, match=_ACTIONABLE) as error:
        px.io.read_image(path)

    assert needle.lower() in str(error.value).lower()


def test_dpx_read_rejects_encryption_before_h2d(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """v1-dpx acceptance 1 and 13: encrypted storage is rejected actionably before any host-to-device copy."""
    path = tmp_path / "encrypted.dpx"
    path.write_bytes(_dpx_fixture(_codes(10, 3), bit_depth=10, encryption_key=0x12345678))

    def fail_h2d(*args: object, **kwargs: object) -> cp.ndarray:
        raise AssertionError("malformed DPX input must not reach H2D")

    monkeypatch.setattr("pixtreme._io.formats.dpx.cp.asarray", fail_h2d)

    with pytest.raises(ValueError, match=_ACTIONABLE) as error:
        px.io.read_image(path)

    assert "encrypt" in str(error.value).lower()


@pytest.mark.parametrize(
    ("overrides", "needle"),
    (
        ({"element_offset": 820}, "header"),
        ({"generic_header_length": 1660}, "generic"),
        ({"file_offset": _DATA_OFFSET + 4}, "offset"),
        ({"file_size_delta": 1}, "file size"),
    ),
)
def test_dpx_read_rejects_inconsistent_file_structure_before_h2d(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, int],
    needle: str,
) -> None:
    """v1-dpx acceptance 1 and 13: file/header/element extents are consistent before packed-byte H2D."""
    path = tmp_path / "inconsistent.dpx"
    path.write_bytes(_dpx_fixture(_codes(10, 3), bit_depth=10, **overrides))

    def fail_h2d(*args: object, **kwargs: object) -> cp.ndarray:
        raise AssertionError("malformed DPX input must not reach H2D")

    monkeypatch.setattr("pixtreme._io.formats.dpx.cp.asarray", fail_h2d)

    with pytest.raises(RuntimeError, match=_ACTIONABLE) as error:
        px.io.read_image(path)

    assert needle in str(error.value).lower()


def test_dpx_read_rejects_truncated_payload_actionably(tmp_path: Path) -> None:
    """v1-dpx acceptance 1 and 13: declared scanline extent cannot exceed the file boundary."""
    path = tmp_path / "truncated.dpx"
    path.write_bytes(_dpx_fixture(_codes(10, 3), bit_depth=10)[:-1])

    with pytest.raises(RuntimeError, match=_ACTIONABLE) as error:
        px.io.read_image(path)

    assert "truncat" in str(error.value).lower()


@pytest.mark.parametrize("bit_depth", (8, 10, 12, 16))
@pytest.mark.parametrize("channels", (3, 4), ids=("rgb", "rgba"))
def test_dpx_write_emits_big_endian_filled_bytes_and_round_trips_on_grid(
    tmp_path: Path, bit_depth: int, channels: int
) -> None:
    """v1-dpx acceptance 6, 7, and 13: exact SDPX bytes and read-after-write follow an independent oracle."""
    values = np.resize(np.array((-0.1, 0.0, 0.1, 0.5, 0.9, 1.0, 1.1), dtype=np.float32), (2, 3, channels))
    labels = tuple("RGBA"[:channels])
    frame = px.io.from_array(cp.asarray(values[..., ::-1]), colorspace="Rec.709", gamma="Cineon", channels=labels[::-1])
    path = tmp_path / "written.dpx"

    assert px.io.write_image(path, frame, bit_depth=bit_depth) is None

    payload = path.read_bytes()
    _assert_written_header_is_st268(payload, width=3, height=2)
    codes, fields = _unpack_written_dpx(payload)
    maximum = (1 << bit_depth) - 1
    expected_codes = np.floor(
        np.clip(values, np.float32(0.0), np.float32(1.0)) * np.float32(maximum) + np.float32(0.5)
    ).astype(np.uint16)
    np.testing.assert_array_equal(codes, expected_codes)
    assert fields == {
        "width": 3,
        "height": 2,
        "descriptor": 50 if channels == 3 else 51,
        "transfer": 1,
        "bit_depth": bit_depth,
        "packing": 1 if bit_depth in (10, 12) else 0,
        "encoding": 0,
        "data_offset": _DATA_OFFSET,
    }
    restored = px.io.read_image(path)
    np.testing.assert_array_equal(
        px.io.to_array(
            restored,
        ).get(),
        expected_codes.astype(np.float32) / np.float32(maximum),
    )


@pytest.mark.parametrize("dtype", (np.uint8, np.uint16, np.uint32, np.float16, np.float32))
def test_dpx_write_accepts_every_storage_dtype_via_float32_recode(tmp_path: Path, dtype: type[np.generic]) -> None:
    """v1-exr-runtime-independence acceptance 9: all five storage dtypes use full-scale recode semantics."""
    if np.issubdtype(dtype, np.integer):
        maximum = np.iinfo(dtype).max
        values = np.array((0, maximum // 2, maximum), dtype=dtype)
        expected_float = values.astype(np.float32) / np.float32(maximum)
    else:
        values = np.array((0.0, 0.5, 1.0), dtype=dtype)
        expected_float = values.astype(np.float32)
    image = np.resize(values, (1, 2, 3))
    frame = px.io.from_array(cp.asarray(image), colorspace="Rec.709", gamma="linear", channels="RGB")
    before = frame.data.copy()
    path = tmp_path / f"{np.dtype(dtype).name}.dpx"

    px.io.write_image(path, frame, bit_depth=10)

    codes, _ = _unpack_written_dpx(path.read_bytes())
    expected = np.resize(expected_float, (1, 2, 3))
    expected_codes = np.floor(
        np.clip(expected, np.float32(0.0), np.float32(1.0)) * np.float32(1023.0) + np.float32(0.5)
    ).astype(np.uint16)
    np.testing.assert_array_equal(codes, expected_codes)
    assert frame.dtype == np.dtype(dtype)
    cp.testing.assert_array_equal(frame.data, before)


def test_dpx_write_float32_native_bypasses_recode_dtype(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """v1-dpx acceptance 6: native float32 reaches the quantization kernel without numeric recoding."""
    frame = px.io.from_array(cp.ones((1, 1, 3), dtype=cp.float32), colorspace="Rec.709", gamma="linear", channels="RGB")

    def fail_recode(*args: object, **kwargs: object) -> px.core.Frame:
        raise AssertionError("native float32 must not be recoded")

    monkeypatch.setattr("pixtreme._values.cast.recode_dtype", fail_recode)

    assert px.io.write_image(tmp_path / "native.dpx", frame) is None


@pytest.mark.parametrize(
    ("gamma", "transfer"),
    (
        ("Cineon", 1),
        ("linear", 2),
        ("S-Log", 3),
        ("S-Log2", 3),
        ("S-Log3", 3),
        ("ARRI-LogC3", 3),
        ("ARRI-LogC4", 3),
        ("Blackmagic-Film-Gen-5", 3),
        ("DaVinci-Intermediate", 3),
        ("RED-Log3G10", 3),
        ("REDlogFilm", 1),
        ("Canon-Log", 3),
        ("Canon-Log-2", 3),
        ("Canon-Log-3", 3),
        ("V-Log", 3),
        ("D-Log", 3),
        ("F-Log", 3),
        ("F-Log2", 3),
        ("N-Log", 3),
        ("L-Log", 3),
        ("Apple-Log", 3),
        ("Samsung-Log", 3),
        ("ACEScc", 3),
        ("ACEScct", 3),
        ("Rec.709", 6),
        ("sRGB", 6),
        ("BT.1886", 6),
        ("PQ", 6),
        ("HLG", 6),
        ("Gamma-2.2", 6),
        ("Gamma-2.4", 6),
        ("Gamma-2.5", 6),
        ("Gamma-2.6", 6),
    ),
)
def test_dpx_write_maps_frame_gamma_to_transfer_characteristic(tmp_path: Path, gamma: str, transfer: int) -> None:
    """v1-dpx acceptance 8; v1-sony-tokens acceptance 12; v1-arri-tokens acceptance 27;
    v1-red-tokens acceptance 70; v1-canon-tokens acceptance 91; v1-panasonic-tokens acceptance 110;
    v1-standard-tokens acceptance 133; v1-vendor-a-tokens acceptance 159; v1-vendor-b-tokens acceptance 186.

    The header records the closed gamma mapping, including the vendor B camera-log transfers as logarithmic code 3.
    """
    frame = px.io.from_array(cp.ones((1, 1, 3), dtype=cp.float32), colorspace="Rec.709", gamma=gamma, channels="RGB")
    path = tmp_path / "transfer.dpx"

    px.io.write_image(path, frame)

    _, fields = _unpack_written_dpx(path.read_bytes())
    assert fields["transfer"] == transfer


@pytest.mark.parametrize("bit_depth", (0, 9, 32))
def test_dpx_write_rejects_unknown_bit_depths(tmp_path: Path, bit_depth: int) -> None:
    """v1-dpx acceptance 6 and 8: bit_depth is an actionable closed set."""
    frame = px.io.from_array(cp.ones((1, 1, 3), dtype=cp.float32), colorspace="Rec.709", gamma="linear", channels="RGB")
    path = tmp_path / "invalid.dpx"

    with pytest.raises(ValueError, match=_ACTIONABLE) as error:
        px.io.write_image(path, frame, bit_depth=bit_depth)

    assert "bit_depth" in str(error.value)
    assert not path.exists()


@pytest.mark.parametrize(("shape", "channels"), (((1, 1, 2), "RG"), ((1, 1, 3), ("R", "R", "B"))))
def test_dpx_write_rejects_non_rgb_rgba_or_duplicate_layouts(
    tmp_path: Path, shape: tuple[int, int, int], channels: str | tuple[str, ...]
) -> None:
    """v1-dpx acceptance 6 and 8: writer layout is unique RGB or RGBA only."""
    frame = px.io.from_array(cp.ones(shape, dtype=cp.float32), colorspace="Rec.709", gamma="linear", channels=channels)

    with pytest.raises(ValueError, match=_ACTIONABLE):
        px.io.write_image(tmp_path / "invalid.dpx", frame)


def test_bit_depth_is_rejected_for_non_dpx_writes(tmp_path: Path) -> None:
    """v1-dpx acceptance 8: explicit bit_depth never silently disappears on another container."""
    frame = px.io.from_array(cp.ones((1, 1, 3), dtype=cp.uint8), colorspace="sRGB", gamma="sRGB", channels="RGB")

    with pytest.raises(ValueError, match=_ACTIONABLE) as error:
        px.io.write_image(tmp_path / "invalid.png", frame, bit_depth=10)

    assert "bit_depth" in str(error.value)


def test_dpx_read_header_is_gpu_free_and_preserves_the_public_model(tmp_path: Path) -> None:
    """v1-dpx acceptance 9: pure CPU probing reports native codes and mapped metadata without model changes."""
    path = tmp_path / "header.dpx"
    path.write_bytes(_dpx_fixture(_codes(12, 4), bit_depth=12, byte_order="<", transfer=2))
    script = """
import sys
import pixtreme as px
h = px.io.read_header(sys.argv[1])
assert (h.format, h.width, h.height) == ("DPX", 3, 2)
assert h.parts[0].channels == {"R": "uint16", "G": "uint16", "B": "uint16", "A": "uint16"}
assert (h.color.colorspace, h.color.gamma, h.color.mappable) == ("Rec.709", "linear", True)
assert h.color.raw["bit_depth"] == 12
assert h.color.raw["byte_order"] == "little"
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


def test_dpx_remains_outside_bytes_boundaries(tmp_path: Path) -> None:
    """v1-dpx acceptance 10: DPX stays file-only and accepts no bytes format token."""
    payload = _dpx_fixture(_codes(10, 3), bit_depth=10)
    frame = px.io.from_array(cp.ones((1, 1, 3), dtype=cp.float32), colorspace="Rec.709", gamma="linear", channels="RGB")

    with pytest.raises(ValueError, match=_ACTIONABLE):
        px.io.decode_image(payload)
    with pytest.raises(ValueError, match=_ACTIONABLE):
        px.io.encode_image(frame, format="dpx")
