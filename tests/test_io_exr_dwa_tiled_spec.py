"""OpenEXR oracle coverage for compressed DWA tiles and mixed AC encodings."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

import pixtreme as px
import pixtreme._io.formats.exr.selection as exr_selection


def _write_tiled_dwa(path: Path, compression: str) -> np.ndarray:
    from openexr_dev_oracle import OpenEXR

    y, x = np.indices((35, 37), dtype=np.float32)
    ramp = np.float32(0.125) + x / np.float32(48) + y / np.float32(64)
    channels = {name: (ramp * scale).astype(np.float16) for name, scale in zip("RGB", (1, 0.625, -0.25), strict=True)}
    tiles = OpenEXR.TileDescription()
    tiles.xSize = tiles.ySize = 32
    OpenEXR.File(
        {
            "type": OpenEXR.tiledimage,
            "tiles": tiles,
            "compression": getattr(OpenEXR, f"{compression.upper()}_COMPRESSION"),
        },
        channels,
    ).write(str(path))
    reference = OpenEXR.File(str(path), separate_channels=True).channels()
    return np.stack([reference[name].pixels for name in "RGB"], axis=-1)


@pytest.mark.req("SP-PIX-007")
@pytest.mark.parametrize("compression", ("dwaa", "dwab"))
@pytest.mark.parametrize("backend", (pytest.param("gpu", marks=pytest.mark.req("SP-PIX-018")), "custom_cpu"))
def test_compressed_dwa_tiles_read_like_openexr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, compression: str, backend: str
) -> None:
    """Compressed DWA tiles match the OpenEXR oracle, including edge tiles, while the GPU lane keeps pixels on device."""
    from exr_test_harness import _record_cupy_transfers

    path = tmp_path / f"tiled-{compression}.exr"
    expected = _write_tiled_dwa(path, compression)
    monkeypatch.setitem(exr_selection._EXR_ROUTING, (compression, "read"), backend)
    transfers = _record_cupy_transfers(monkeypatch)

    actual = px.io.read_image(path, unchanged=True)

    if backend == "gpu":
        for transfer in transfers:
            # The inverse-DCT basis is the only floating point host input; coefficients and pixels stay on device.
            if transfer.dtype in ("float16", "float32", "float64", "uint16"):
                assert (transfer.direction, transfer.dtype, transfer.shape) == ("h2d", "float32", (8, 8))
    assert actual.channels == ("R", "G", "B")
    assert actual.data.dtype == np.dtype("float16")
    # One HALF ulp near 1 plus fp32 inverse-DCT/CSC rounding, as in the existing DWA oracle tests.
    np.testing.assert_allclose(actual.data.get(), expected, rtol=2e-3, atol=2e-3)


def _header_end(payload: bytes) -> int:
    cursor = 8
    while payload[cursor]:
        cursor = payload.index(b"\0", cursor) + 1
        cursor = payload.index(b"\0", cursor) + 1
        size = struct.unpack_from("<I", payload, cursor)[0]
        cursor += 4 + size
    return cursor + 1


def _first_tile_payload(path: Path) -> bytes:
    data = path.read_bytes()
    offset = struct.unpack_from("<Q", data, _header_end(data))[0]
    size = struct.unpack_from("<i", data, offset + 16)[0]
    return data[offset + 20 : offset + 20 + size]


@pytest.mark.req("SP-PIX-007")
@pytest.mark.parametrize("backend", ("gpu", "custom_cpu"))
def test_dwa_scanline_chunks_can_mix_huffman_and_deflate_ac(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    """Scanline reads preserve each chunk's coefficients when Huffman AC and DEFLATE AC share a file."""
    from openexr_dev_oracle import OpenEXR
    from test_io_exr_container_decode_spec import _part_attributes

    tiled = tmp_path / "source-tiled.exr"
    _write_tiled_dwa(tiled, "dwaa")
    y, x = np.indices((32, 32), dtype=np.float32)
    ramp = 0.5 + x / 40 + y / 80
    scanline = tmp_path / "source-scanline.exr"
    OpenEXR.File(
        {"compression": OpenEXR.DWAA_COMPRESSION},
        {name: (ramp * scale).astype(np.float16) for name, scale in zip("RGB", (1, 0.5, 0.25), strict=True)},
    ).write(str(scanline))
    data = scanline.read_bytes()
    offset = struct.unpack_from("<Q", data, _header_end(data))[0]
    size = struct.unpack_from("<i", data, offset + 4)[0]
    payloads = (data[offset + 8 : offset + 8 + size], _first_tile_payload(tiled))
    assert tuple(struct.unpack_from("<Q", payload, 80)[0] for payload in payloads) == (0, 1)
    header = struct.pack("<II", 20000630, 2) + _part_attributes(
        data_window=(0, 0, 31, 63),
        channels=tuple((name, 1, (1, 1)) for name in "BGR"),
        compression=8,
    )
    cursor = len(header) + 16
    chunks = [struct.pack("<ii", index * 32, len(payload)) + payload for index, payload in enumerate(payloads)]
    path = tmp_path / "mixed-ac-scanline.exr"
    path.write_bytes(header + struct.pack("<QQ", cursor, cursor + len(chunks[0])) + b"".join(chunks))
    reference = OpenEXR.File(str(path), separate_channels=True).channels()
    expected = np.stack([reference[name].pixels for name in "RGB"], axis=-1)
    monkeypatch.setitem(exr_selection._EXR_ROUTING, ("dwaa", "read"), backend)

    actual = px.io.read_image(path, unchanged=True)

    # The existing DWA oracle tolerance covers HALF and inverse-DCT/CSC rounding.
    np.testing.assert_allclose(actual.data.get(), expected, rtol=2e-3, atol=2e-3)


@pytest.mark.req("SP-PIX-017")
@pytest.mark.parametrize("backend", ("gpu", "custom_cpu"))
def test_dwa_deflate_ac_rejects_a_corrupt_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    """A damaged DEFLATE AC stream reports its integrity failure instead of returning damaged pixels."""
    path = tmp_path / "corrupt-ac.exr"
    _write_tiled_dwa(path, "dwaa")
    data = bytearray(path.read_bytes())
    tile = struct.unpack_from("<Q", data, _header_end(data))[0]
    start = tile + 20
    unknown_size, _, ac_size = struct.unpack_from("<QQQ", data, start + 8)
    rules_size = struct.unpack_from("<H", data, start + 88)[0]
    data[start + 88 + rules_size + unknown_size + ac_size - 1] ^= 1
    path.write_bytes(data)
    monkeypatch.setitem(exr_selection._EXR_ROUTING, ("dwaa", "read"), backend)

    with pytest.raises(RuntimeError, match=r"why=.*(?:Adler|zlib|checksum).*how=") as caught:
        px.io.read_image(path, unchanged=True)
    assert "AC" in str(caught.value)


@pytest.mark.req("SP-PIX-017")
@pytest.mark.parametrize("version", (0, 1))
def test_unsupported_legacy_dwa_reports_a_user_action(tmp_path: Path, version: int) -> None:
    """Unsupported legacy DWA versions explain which encoding to re-export instead of exposing a codec guard."""
    from openexr_dev_oracle import OpenEXR
    from test_io_exr_container_decode_spec import _part_attributes

    tiled = tmp_path / "source-tiled.exr"
    _write_tiled_dwa(tiled, "dwaa")
    payload = bytearray(_first_tile_payload(tiled))
    rules_size = struct.unpack_from("<H", payload, 88)[0]
    del payload[88 : 88 + rules_size]
    struct.pack_into("<Q", payload, 0, version)
    header = struct.pack("<II", 20000630, 2) + _part_attributes(
        data_window=(0, 0, 31, 31),
        channels=tuple((name, 1, (1, 1)) for name in "BGR"),
        compression=8,
    )
    path = tmp_path / f"legacy-dwa-v{version}.exr"
    path.write_bytes(header + struct.pack("<Qii", len(header) + 8, 0, len(payload)) + payload)
    assert OpenEXR.File(str(path), separate_channels=True).channels()["R"].pixels.shape == (32, 32)

    with pytest.raises(ValueError) as caught:
        px.io.read_image(path, unchanged=True)

    message = str(caught.value)
    assert "why=" in message and "what=" in message and "how=" in message
    assert "version" in message and str(version) in message
    assert "re-encode" in message and "ZIP" in message
