"""Current source and documentation contracts needed by acceptance traceability."""

from __future__ import annotations

import ast
import inspect
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _dotted_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = _dotted_name(node.value)
        return None if owner is None else f"{owner}.{node.attr}"
    return None


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def test_import_keeps_the_cuda_primary_context_inactive() -> None:
    """v1-frame-core acceptance 2: importing pixtreme allocates no CUDA resource.

    The subprocess asks the CUDA driver for device 0's primary-context state
    before and after import.  The probe initializes the driver but never retains
    or creates a context, so an import-time device allocation activates the state
    and is observed without relying on CuPy's allocator as the oracle.
    """
    script = r"""
import ctypes

driver = ctypes.CDLL("libcuda.so.1")
assert driver.cuInit(0) == 0
device = ctypes.c_int()
assert driver.cuDeviceGet(ctypes.byref(device), 0) == 0

def primary_context_active():
    flags = ctypes.c_uint()
    active = ctypes.c_int()
    result = driver.cuDevicePrimaryCtxGetState(device, ctypes.byref(flags), ctypes.byref(active))
    assert result == 0
    return active.value

assert primary_context_active() == 0
import pixtreme
assert primary_context_active() == 0
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def test_current_public_names_follow_the_documented_naming_rules_and_reservations() -> None:
    """v1-public-namespace acceptance 11: current docs and leaves retain the naming grammar."""
    import pixtreme as px

    requirements = (ROOT / "docs" / "requirements.md").read_text(encoding="utf-8")
    feature = (ROOT / "docs" / "features" / "v1-public-namespace.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    rules = (
        "葉名は短縮せず機構まで含む自己記述名",
        "`X_to_Y` は値が変わる色表現変換に予約",
        "module + leaf の二階建て",
    )
    for rule in rules:
        assert rule in requirements
    assert "## 命名規則" in feature
    assert "The package root exposes 13 modules" in readme

    modules = tuple(name for name in px.__all__ if inspect.ismodule(getattr(px, name)))
    assert len(modules) == 13
    public_functions = {
        (module_name, leaf)
        for module_name in modules
        for leaf in getattr(px, module_name).__all__
        if inspect.isfunction(getattr(getattr(px, module_name), leaf))
    }
    assert {(module_name, leaf) for module_name, leaf in public_functions if module_name == "core"} == {
        ("core", "channels")
    }
    public_operations = {(module_name, leaf) for module_name, leaf in public_functions if module_name != "core"}
    leaves = {leaf for _, leaf in public_functions}
    assert leaves.isdisjoint(
        {
            "invert",
            "premult",
            "unpremult",
            "swap_rb",
            "histogram",
            "depth_merge",
        }
    )
    assert {leaf for leaf in leaves if "_to_" in leaf} == {
        "full_to_legal",
        "gamma_to_linear",
        "hsv_to_rgb",
        "legal_to_full",
        "linear_to_gamma",
        "rgb_to_grayscale",
        "rgb_to_hsv",
        "rgb_to_rgb",
        "rgb_to_ycbcr",
        "ycbcr_to_rgb",
        "ycbcr_to_ycbcr",
    }
    public_surface_section = feature.split("## module 公開面\n", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
    assert "`core.channels`" in public_surface_section
    for module_name, leaf in public_operations:
        assert f"`{module_name}`" in public_surface_section
        assert f"`{leaf}`" in public_surface_section


def test_tga_read_source_has_one_flat_payload_transfer_and_one_gpu_pass() -> None:
    """v1-tga acceptance 5: source fixes the intended one-pass TGA read structure.

    This is an intentional source-structure contract: CPU work may parse and
    expand RLE into flat bytes, but it must not build an HWC NumPy image.  The
    image payload crosses H2D once through ``np.frombuffer`` and one RawKernel
    launch writes the final Frame storage.
    """
    source_path = ROOT / "src" / "pixtreme" / "_io" / "formats" / "tga.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    function = _function(tree, "_read_tga_frame")
    calls = tuple(node for node in ast.walk(function) if isinstance(node, ast.Call))

    payload_transfers = tuple(
        call
        for call in calls
        if _dotted_name(call.func) == "cp.asarray"
        and call.args
        and isinstance(call.args[0], ast.Call)
        and _dotted_name(call.args[0].func) == "np.frombuffer"
    )
    kernel_launches = tuple(
        call for call in calls if isinstance(call.func, ast.Call) and _dotted_name(call.func.func) == "_tga_read_kernel"
    )
    forbidden_host_image_builders = tuple(
        call for call in calls if _dotted_name(call.func) in {"np.array", "np.asarray", "np.empty", "np.zeros"}
    )

    assert len(payload_transfers) == 1
    assert len(kernel_launches) == 1
    assert forbidden_host_image_builders == ()

    kernel_launch = kernel_launches[0]
    assert len(kernel_launch.args) == 3
    kernel_arguments = kernel_launch.args[2]
    assert isinstance(kernel_arguments, ast.Tuple)
    kernel_output = kernel_arguments.elts[1]
    assert isinstance(kernel_output, ast.Name)

    returns = tuple(node for node in function.body if isinstance(node, ast.Return))
    assert len(returns) == 1
    frame_call = returns[0].value
    assert isinstance(frame_call, ast.Call) and _dotted_name(frame_call.func) == "Frame"
    frame_data_keywords = tuple(keyword for keyword in frame_call.keywords if keyword.arg == "data")
    assert len(frame_data_keywords) == 1
    frame_data = frame_data_keywords[0].value
    assert isinstance(frame_data, ast.Name) and frame_data.id == kernel_output.id

    kernel_end = (kernel_launch.end_lineno or kernel_launch.lineno, kernel_launch.end_col_offset or 0)
    post_kernel_output_uses = tuple(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Name) and node.id == kernel_output.id and (node.lineno, node.col_offset) > kernel_end
    )
    assert len(post_kernel_output_uses) == 1 and post_kernel_output_uses[0] is frame_data
