"""Acceptance tests for the optional ``pixtreme.infer`` and ``pixtreme.transport`` companion namespaces."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import pixtreme as px

EXPECTED_ROOT_ALL = (
    "core",
    "io",
    "color",
    "filter",
    "transform",
    "draw",
    "generate",
    "morphology",
    "metrics",
    "feature",
    "values",
    "channel",
    "composite",
    "__version__",
)


def _run_python(script: str, *, companion_root: Path | None = None) -> subprocess.CompletedProcess[str]:
    source_root = Path(__file__).resolve().parents[1] / "src"
    python_path = [str(source_root)]
    if companion_root is not None:
        python_path.append(str(companion_root))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        cwd=source_root.parent,
        env=environment,
        text=True,
        timeout=10,
    )


def _write_infer_companion(tmp_path: Path, source: str) -> Path:
    companion_root = tmp_path / "companion"
    infer_root = companion_root / "pixtreme" / "infer"
    infer_root.mkdir(parents=True)
    (infer_root / "__init__.py").write_text(source, encoding="utf-8")
    return companion_root


def _assert_process_succeeded(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@pytest.mark.parametrize(
    ("name", "distribution"),
    (("infer", "pixtreme-infer"), ("transport", "pixtreme-transport")),
)
def test_missing_companion_reports_its_install_guidance(name: str, distribution: str) -> None:
    """v1-infer-namespace acceptance 1: each missing companion reports its matching install command."""
    result = _run_python(
        f"""
from pathlib import Path

import pixtreme as px

px.__path__[:] = [str(Path(px.__file__).resolve().parent)]
try:
    getattr(px, {name!r})
except ImportError as error:
    assert {f"pip install {distribution}"!r} in str(error)
    assert isinstance(error.__cause__, ModuleNotFoundError)
    assert error.__cause__.name == {f"pixtreme.{name}"!r}
else:
    raise AssertionError({f"px.{name} unexpectedly resolved without its companion"!r})
"""
    )

    _assert_process_succeeded(result)


def test_unknown_root_attribute_remains_attribute_error() -> None:
    """v1-infer-namespace acceptance 2: names outside the companion list retain the root AttributeError contract."""
    with pytest.raises(AttributeError, match="module 'pixtreme' has no attribute 'nonexistent'"):
        px.nonexistent


def test_split_companion_loads_without_changing_the_root_surface(tmp_path: Path) -> None:
    """v1-infer-namespace acceptance 3, 4, and 6: split infer loads while the exact root surface stays fixed."""
    companion_root = _write_infer_companion(tmp_path, 'MARKER = "split-companion"\n')
    result = _run_python(
        f"""
import pixtreme as px

before_all = px.__all__
before_public = {{name for name in vars(px) if not name.startswith("_") or name == "__version__"}}
infer = px.infer

assert infer.__name__ == "pixtreme.infer"
assert infer.MARKER == "split-companion"
assert {str(companion_root / "pixtreme")!r} in px.__path__
assert before_all == {EXPECTED_ROOT_ALL!r}
assert px.__all__ == before_all
assert {{name for name in vars(px) if not name.startswith("_") or name == "__version__"}} == before_public
assert before_public == set(before_all)
assert "infer" not in vars(px)
""",
        companion_root=companion_root,
    )

    _assert_process_succeeded(result)


def test_infer_internal_module_not_found_error_propagates_unchanged(tmp_path: Path) -> None:
    """v1-infer-namespace acceptance 5: an infer dependency failure is not rewritten as install guidance."""
    companion_root = _write_infer_companion(tmp_path, "import deliberately_missing_infer_dependency\n")
    result = _run_python(
        """
import pixtreme as px

try:
    px.infer
except ModuleNotFoundError as error:
    assert error.name == "deliberately_missing_infer_dependency"
    assert "pip install pixtreme-infer" not in str(error)
else:
    raise AssertionError("the companion's missing dependency did not fail")
""",
        companion_root=companion_root,
    )

    _assert_process_succeeded(result)
