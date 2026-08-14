"""Contract tests for public image I/O boundaries."""

from __future__ import annotations

import ast
import os
import struct
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
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


def test_image_header_is_a_frozen_minimal_pydantic_model(tmp_path: Path) -> None:
    """v1-io acceptance 17: ImageHeader exposes the fixed minimal header inspection shape."""
    path = tmp_path / "sample.png"
    Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8), mode="RGB").save(path)

    header = px.io.read_header(path)

    assert isinstance(header, px.io.ImageHeader)
    assert set(px.io.ImageHeader.model_fields) == {"format", "width", "height", "parts", "color"}
    assert px.io.ImageHeader.model_config["frozen"] is True
    assert (header.format, header.width, header.height) == ("PNG", 3, 2)
    assert header.parts[0].name == ""
    assert header.parts[0].channels == {"R": "uint8", "G": "uint8", "B": "uint8"}
    with pytest.raises(Exception):
        header.width = 4  # type: ignore[misc]


def test_read_header_uses_no_gpu_codec_or_cuda_visible_device(tmp_path: Path) -> None:
    """v1-io acceptance 18: header probing succeeds without pixel decode or a visible GPU."""
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


def test_pixtreme_import_is_lazy_with_both_io_dependencies_blocked() -> None:
    """v1-io acceptance 23: importing pixtreme does not import or require either I/O backend."""
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


def test_source_and_dependency_metadata_have_no_direct_opencv_boundary() -> None:
    """v1-io acceptance 24: pixtreme neither imports cv2 nor declares an OpenCV Python dependency."""
    imports: list[tuple[Path, int, str]] = []
    for path in (ROOT / "src" / "pixtreme").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        importlib_aliases = {"importlib"}
        import_module_aliases: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "importlib":
                        importlib_aliases.add(alias.asname or alias.name)
                    if alias.name.split(".", 1)[0] == "cv2":
                        imports.append((path, node.lineno, alias.name))
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".", 1)[0] == "cv2":
                    imports.append((path, node.lineno, node.module or "cv2"))
                if node.module == "importlib":
                    import_module_aliases.update(
                        alias.asname or alias.name for alias in node.names if alias.name == "import_module"
                    )
            elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
                requested = node.args[0].value
                if not isinstance(requested, str) or requested.split(".", 1)[0] != "cv2":
                    continue
                direct = isinstance(node.func, ast.Name) and node.func.id in import_module_aliases
                qualified = (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "import_module"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in importlib_aliases
                )
                if direct or qualified:
                    imports.append((path, node.lineno, requested))

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    dependency_strings = list(project.get("dependencies", ()))
    for group in project.get("optional-dependencies", {}).values():
        dependency_strings.extend(group)
    dependency_names = {canonicalize_name(Requirement(value).name) for value in dependency_strings}

    assert imports == []
    assert not {name for name in dependency_names if name.startswith("opencv-")}


@pytest.mark.parametrize("suffix", (".gif", ".heic", ".cin", ""))
def test_read_header_rejects_unsupported_extensions_with_actionable_errors(tmp_path: Path, suffix: str) -> None:
    """v1-io acceptance 2 / v1-io-formats acceptance 2: unsupported headers remain actionable."""
    path = tmp_path / f"image{suffix}"
    path.write_bytes(b"not an image")

    with pytest.raises(ValueError, match=r"why=.*what=.*how="):
        px.io.read_header(path)


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
    """REQ-API-012: every reachable container parser failure names why, observed input, and recovery."""
    path = tmp_path / f"corrupt{suffix}"
    path.write_bytes(payload)

    with pytest.raises(RuntimeError) as error:
        px.io.read_header(path)

    cause = error.value.__cause__
    assert isinstance(cause, ValueError)
    _assert_actionable(cause)
    assert observed in str(cause)
