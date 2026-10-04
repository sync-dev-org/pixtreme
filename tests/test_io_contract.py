"""Contract tests for public image I/O boundaries."""

from __future__ import annotations

import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import pixtreme as px

ROOT = Path(__file__).resolve().parents[1]


def _assert_actionable(error: BaseException) -> None:
    message = str(error)
    assert "why=" in message
    assert "; what=" in message
    assert "; how=" in message


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + chunk_type + payload + b"\x00\x00\x00\x00"


def _exr_attribute(name: str, attribute_type: str, payload: bytes) -> bytes:
    return name.encode() + b"\x00" + attribute_type.encode() + b"\x00" + struct.pack("<I", len(payload)) + payload


def _exr_header(*attributes: bytes) -> bytes:
    return struct.pack("<II", 20000630, 2) + b"".join(attributes) + b"\x00"


@pytest.mark.req("REQ-PIX-007")
def test_image_header_is_a_frozen_minimal_pydantic_model(tmp_path: Path) -> None:
    """Image header inspection exposes a fixed immutable model with orientation and storage fields."""
    path = tmp_path / "sample.png"
    Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8), mode="RGB").save(path)

    header = px.io.read_header(path)

    assert isinstance(header, px.io.ImageHeader)
    assert set(px.io.ImageHeader.model_fields) == {"format", "width", "height", "parts", "color", "orientation"}
    assert px.io.ImageHeader.model_config["frozen"] is True
    assert (header.format, header.width, header.height) == ("PNG", 3, 2)
    assert header.parts[0].name == ""
    assert header.parts[0].channels == {"R": "uint8", "G": "uint8", "B": "uint8"}
    with pytest.raises(Exception):
        header.width = 4  # type: ignore[misc]


@pytest.mark.req("REQ-PIX-007")
def test_read_header_uses_no_gpu_codec_or_cuda_visible_device(tmp_path: Path) -> None:
    """Reading an image header succeeds without decoding pixels or opening a CUDA device."""
    path = tmp_path / "sample.png"
    Image.fromarray(np.zeros((2, 3), dtype=np.uint16)).save(path)
    script = """
import sys
import pixtreme as px
h = px.io.read_header(sys.argv[1])
assert (h.format, h.width, h.height) == ("PNG", 3, 2)
assert "nvidia.nvimgcodec" not in sys.modules
assert "OpenEXR" not in sys.modules
"""
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}

    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.req("REQ-PIX-007")
def test_pixtreme_import_is_lazy_with_both_io_dependencies_blocked() -> None:
    """Importing pixtreme succeeds even when optional image I/O backends cannot be imported."""
    script = """
import importlib.abc
import sys
class BlockIO(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "OpenEXR" or fullname == "nvidia.nvimgcodec":
            raise ModuleNotFoundError(fullname)
        return None
sys.meta_path.insert(0, BlockIO())
import pixtreme
assert "OpenEXR" not in sys.modules
assert "nvidia.nvimgcodec" not in sys.modules
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.req("REQ-PIX-007")
@pytest.mark.req("REQ-PIX-017")
@pytest.mark.parametrize("suffix", (".gif", ".heic", ".cin", ""))
def test_read_header_rejects_unsupported_extensions_with_actionable_errors(tmp_path: Path, suffix: str) -> None:
    """Header inspection rejects unsupported filename extensions and explains the accepted image formats."""
    path = tmp_path / f"image{suffix}"
    path.write_bytes(b"not an image")

    with pytest.raises(ValueError, match=r"why=.*what=.*how="):
        px.io.read_header(path)


@pytest.mark.req("REQ-PIX-007")
@pytest.mark.req("REQ-PIX-017")
@pytest.mark.parametrize(
    ("suffix", "payload", "observed"),
    (
        (".png", b"", "requested=8 bytes, received=0 bytes"),
        (".png", b"not png!", "signature=b'not png!'"),
        (
            ".png",
            b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", b"\x00" * 12),
            "payload_length=12",
        ),
        (
            ".png",
            b"\x89PNG\r\n\x1a\n"
            + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 0, 2, 8, 2, 0, 0, 0))
            + _png_chunk(b"IEND", b""),
            "color_type=2, width=0, height=2",
        ),
        (".jpg", b"XX", "signature=b'XX'"),
        (".jpg", b"\xff\xd8\xff\xe0\x00\x01", "marker=0xe0, length=1"),
        (".tiff", b"ZZ", "byte_order=b'ZZ'"),
        (".tiff", b"II" + struct.pack("<HI", 41, 8), "magic=41"),
        (".tiff", b"II" + struct.pack("<HIH", 42, 8, 0), "width=0, height=0"),
        (
            ".exr",
            _exr_header(
                _exr_attribute("channels", "chlist", b"R\x00" + b"\x00" * 15),
                _exr_attribute("dataWindow", "box2i", struct.pack("<iiii", 0, 0, 0, 0)),
            ),
            "channel='R', entry_bytes=15",
        ),
        (
            ".exr",
            _exr_header(
                _exr_attribute("channels", "chlist", b"R\x00" + struct.pack("<i", 3) + b"\x00" * 12),
                _exr_attribute("dataWindow", "box2i", struct.pack("<iiii", 0, 0, 0, 0)),
            ),
            "channel='R', pixel_type=3",
        ),
        (".exr", struct.pack("<II", 0, 2), "magic=0"),
        (".exr", _exr_header(_exr_attribute("foo", "string", b"")), "attributes=('foo',)"),
        (".exr", _exr_header(), "parts=0, dimensions=None"),
    ),
    ids=(
        "truncated-field",
        "png-signature",
        "png-ihdr",
        "png-header-values",
        "jpeg-signature",
        "jpeg-marker-length",
        "tiff-byte-order",
        "tiff-magic",
        "tiff-dimensions",
        "exr-channel-list",
        "exr-pixel-type",
        "exr-signature",
        "exr-required-attributes",
        "exr-image-parts",
    ),
)
def test_read_header_corruption_causes_are_actionable(
    tmp_path: Path,
    suffix: str,
    payload: bytes,
    observed: str,
) -> None:
    """Each supported image header parser reports the corrupt input, its cause, and how to recover."""
    path = tmp_path / f"corrupt{suffix}"
    path.write_bytes(payload)

    with pytest.raises(RuntimeError) as error:
        px.io.read_header(path)

    cause = error.value.__cause__
    assert isinstance(cause, ValueError)
    _assert_actionable(cause)
    assert observed in str(cause)
