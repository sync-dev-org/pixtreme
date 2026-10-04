"""Specification tests for the final runtime-independent EXR routing boundary."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.req("REQ-PIX-007")
def test_all_exr_codecs_run_when_openexr_import_is_blocked(tmp_path: Path) -> None:
    """Every supported EXR codec reads and writes when importing OpenEXR is blocked."""
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
