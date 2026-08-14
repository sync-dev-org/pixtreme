"""Specification tests for the final runtime-independent EXR routing boundary."""

from __future__ import annotations

import ast
import inspect
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import pixtreme._io.formats.exr.selection as selection

_COMPRESSIONS = ("none", "rle", "zip", "zips", "piz", "pxr24", "b44", "b44a", "dwaa", "dwab")
_EXPECTED_ROUTING = {
    ("none", "read"): "native",
    ("none", "write"): "gpu",
    ("rle", "read"): "gpu",
    ("rle", "write"): "gpu",
    ("zip", "read"): "custom_cpu",
    ("zip", "write"): "gpu",
    ("zips", "read"): "custom_cpu",
    ("zips", "write"): "gpu",
    ("piz", "read"): "gpu",
    ("piz", "write"): "gpu",
    ("pxr24", "read"): "custom_cpu",
    ("pxr24", "write"): "gpu",
    ("b44", "read"): "gpu",
    ("b44", "write"): "gpu",
    ("b44a", "read"): "gpu",
    ("b44a", "write"): "gpu",
    ("dwaa", "read"): "gpu",
    ("dwaa", "write"): "gpu",
    ("dwab", "read"): "gpu",
    ("dwab", "write"): "gpu",
}


def test_source_fixed_routing_table_drives_every_read_and_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v1-exr-runtime-independence acceptance 36 and 48: spies fix the complete source routing table."""
    assert selection._EXR_ROUTING == _EXPECTED_ROUTING

    read_calls: list[tuple[str, str]] = []

    def read_native(container: object, _selected: object, *, output_dtype: str) -> str:
        read_calls.append(("native", container.compression))  # type: ignore[attr-defined]
        return output_dtype

    def read_gpu(container: object, _selected: object, *, output_dtype: str) -> str:
        read_calls.append(("gpu", container.compression))  # type: ignore[attr-defined]
        return output_dtype

    def read_custom_cpu(container: object, _selected: object, *, output_dtype: str) -> str:
        read_calls.append(("custom_cpu", container.compression))  # type: ignore[attr-defined]
        return output_dtype

    monkeypatch.setattr(selection, "_read_exr_native", read_native)
    monkeypatch.setattr(selection, "_read_exr_gpu", read_gpu)
    monkeypatch.setattr(selection, "_read_exr_custom_cpu", read_custom_cpu)
    for compression in _COMPRESSIONS:
        container = SimpleNamespace(compression=compression)
        assert selection._decode_exr_view(container, ("Y",), output_dtype="float32") == "float32"

    assert read_calls == [(_EXPECTED_ROUTING[(compression, "read")], compression) for compression in _COMPRESSIONS]

    write_calls: list[tuple[str, str]] = []

    def write_backend(
        _path: Path,
        _frame: object,
        *,
        compression: str,
        dwa_level: float | None,
        backend: str,
    ) -> None:
        assert dwa_level is None
        write_calls.append((backend, compression))

    monkeypatch.setattr(selection, "_write_exr_with_backend", write_backend)
    for compression in _COMPRESSIONS:
        selection._write_exr(tmp_path / f"{compression}.exr", object(), compression=compression, dwa_level=None)  # type: ignore[arg-type]

    assert write_calls == [("gpu", compression) for compression in _COMPRESSIONS]


def test_none_has_one_internal_read_lane_and_label() -> None:
    """v1-exr-runtime-independence acceptance 37 and 48: NONE has one observable internal read lane."""
    assert selection._EXR_ROUTING[("none", "read")] == "native"
    assert "_read_exr_none" not in inspect.getsource(selection._read_exr_gpu)
    assert "_read_exr_none" not in inspect.getsource(selection._read_exr_custom_cpu)
    assert inspect.getsource(selection._read_exr_native).count("_read_exr_none") == 1


def test_production_source_has_no_openexr_runtime_backend_or_wrapper() -> None:
    """v1-exr-runtime-independence acceptance 38, 41, and 48: source AST excludes the OpenEXR runtime graph."""
    source_root = Path(__file__).resolve().parents[1] / "src" / "pixtreme"
    imported_at: list[str] = []
    wrapper_names: list[str] = []
    forbidden_wrappers = {"_read_exr_cpu_pixels", "_write_exr_cpu"}
    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(alias.name == "OpenEXR" for alias in node.names):
                imported_at.append(f"{path.relative_to(source_root)}:{node.lineno}")
            if isinstance(node, ast.ImportFrom) and node.module == "OpenEXR":
                imported_at.append(f"{path.relative_to(source_root)}:{node.lineno}")
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in forbidden_wrappers:
                wrapper_names.append(node.name)

    selection_source = inspect.getsource(selection)
    assert imported_at == []
    assert wrapper_names == []
    assert "_EXR_AUTO_SELECTION" not in selection_source
    assert '"openexr"' not in selection_source
    assert "'openexr'" not in selection_source


def test_package_metadata_keeps_openexr_out_of_runtime_dependencies() -> None:
    """v1-exr-runtime-independence acceptance 39: OpenEXR is a dev-group dependency only."""
    import tomllib

    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as stream:
        pyproject = tomllib.load(stream)

    runtime_dependencies = pyproject["project"]["dependencies"]
    dev_dependencies = pyproject["dependency-groups"]["dev"]
    assert not any(requirement.startswith("openexr") for requirement in runtime_dependencies)
    assert any(requirement.startswith("openexr") for requirement in dev_dependencies)


def test_openexr_test_usage_is_isolated_behind_the_named_dev_oracle() -> None:
    """v1-exr-runtime-independence acceptance 47 and 48: direct OpenEXR imports live only in the dev oracle."""
    tests_root = Path(__file__).resolve().parent
    direct_imports: list[str] = []
    for path in tests_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(alias.name == "OpenEXR" for alias in node.names):
                direct_imports.append(str(path.relative_to(tests_root)))
            if isinstance(node, ast.ImportFrom) and node.module == "OpenEXR":
                direct_imports.append(str(path.relative_to(tests_root)))

    oracle_source = (tests_root / "openexr_dev_oracle.py").read_text(encoding="utf-8")
    assert direct_imports == []
    assert 'import_module("OpenEXR")' in oracle_source


def test_all_exr_codecs_run_when_openexr_import_is_blocked(tmp_path: Path) -> None:
    """v1-exr-runtime-independence acceptance 40, 41, 43, and 48: runtime subprocess never reaches OpenEXR."""
    script = r"""
import importlib.abc
from pathlib import Path
import sys

class RejectOpenEXR(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "OpenEXR" or fullname.startswith("OpenEXR."):
            raise ModuleNotFoundError("OpenEXR is intentionally unavailable")
        return None

sys.meta_path.insert(0, RejectOpenEXR())

import cupy as cp
import pixtreme as px

root = Path(sys.argv[1])
data = cp.asarray([0, 1, 16777217, 4294967295], dtype=cp.uint32).reshape(2, 2, 1)
frame = px.io.from_array(data, colorspace="ACEScg", gamma="linear", channels=("U",))
for compression in ("none", "rle", "zip", "zips", "piz", "pxr24", "b44", "b44a", "dwaa", "dwab"):
    output = root / f"{compression}.exr"
    px.io.write_image(output, frame, compression=compression)
    restored = px.io.read_image(output, channels=("U",), unchanged=True)
    cp.testing.assert_array_equal(restored.data, data)
assert "OpenEXR" not in sys.modules
"""
    env = os.environ.copy()
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
