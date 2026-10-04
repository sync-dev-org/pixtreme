"""Run development scripts against the current package without changing tracked fixtures."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# These repeat performance gates and would measure the host rather than check a generated artifact.
PERFORMANCE_GATES = {
    Path("tests/run_exr_dwa_gate.py"),
    Path("tests/run_exr_rle_gate_repeat.py"),
    Path("tests/run_exr_pxr24_gate_repeat.py"),
    Path("tests/run_exr_b44_gate_repeat.py"),
    Path("tests/run_exr_piz_gate_repeat.py"),
}
# The public mirror has its own behavior tests in test_public_mirror_spec.py.
DEDICATED_TEST = {Path("tools/mirror_public.py")}
EXCLUDED = PERFORMANCE_GATES | DEDICATED_TEST

SCRIPT_PATHS = tuple(
    sorted(
        path.relative_to(ROOT)
        for directory, pattern in (("tools", "*.py"), ("tests", "generate_*.py"), ("tests", "run_*.py"))
        for path in (ROOT / directory).glob(pattern)
        if path.relative_to(ROOT) not in EXCLUDED
    )
)

IMPORT_ONLY = {Path("tools/aces_ocio.py"), Path("tests/generate_io_fixtures.py")}
EXR_INPUT_SCRIPTS = {
    Path("tests/generate_io_exr_gpu_rle_sheet.py"),
    Path("tests/generate_io_exr_gpu_pxr24_sheet.py"),
    Path("tests/generate_io_exr_gpu_b44_sheet.py"),
    Path("tests/generate_io_exr_gpu_piz_sheet.py"),
    Path("tools/generate_exr_gpu_none_zip_visual.py"),
    Path("tools/generate_exr_gpu_dwa_visual.py"),
}

# Directory arguments have different spellings, so record their public CLI and generated filenames here.
DIRECTORY_OUTPUTS: dict[Path, tuple[str | None, tuple[str, ...]]] = {
    Path("tools/generate_chroma_siting_visual.py"): (
        "--output-dir",
        ("chroma-siting-decode.png", "chroma-siting-encode.png", "chroma-siting-visual.json"),
    ),
    Path("tools/generate_draw_text_supersample_visual.py"): (
        "--output-dir",
        tuple(
            f"draw-text-supersample-size-{size:02d}-{style}.png"
            for size in (12, 32, 64)
            for style in ("body", "single-outline", "multi-outline")
        )
        + ("draw-text-supersample-metrics.json",),
    ),
    Path("tools/generate_exr_gpu_none_zip_visual.py"): (
        "--output-dir",
        tuple(
            f"exr-gpu-{compression}-{suffix}"
            for compression in ("none", "zip", "zips")
            for suffix in ("cpu-reference.exr", "gpu-write.exr")
        )
        + tuple(f"exr-gpu-{compression}.png" for compression in ("none", "zip", "zips"))
        + ("exr-gpu-none-zip-metrics.json",),
    ),
    Path("tools/generate_exr_gpu_dwa_visual.py"): (
        "--output-dir",
        tuple(
            f"exr-gpu-{compression}-level-{level}-{suffix}"
            for compression in ("dwaa", "dwab")
            for level in ("10", "45", "100")
            for suffix in ("openexr-reference.exr", "hybrid-write.exr")
        )
        + tuple(
            f"exr-gpu-{compression}-level-{level}.png"
            for compression in ("dwaa", "dwab")
            for level in ("10", "45", "100")
        )
        + ("exr-gpu-dwa-metrics.json",),
    ),
    Path("tools/generate_lut_shaper_visual.py"): (
        "--output-dir",
        ("lut-shaper-17-log.3dl", "lut-shaper-visual.png", "lut-shaper-visual-dark-crop.png", "lut-shaper-visual.json"),
    ),
    Path("tools/generate_lut_visual.py"): ("--output-dir", ("lut-visual-look.cube", "lut-before-after.png")),
    Path("tools/generate_p216_visual.py"): (
        "--output-dir",
        (
            "p216-from-filters.png",
            "p216-to-filters.png",
            "p216-phase-zoom.png",
            "p216-source-pattern.png",
            "p216-visual.json",
        ),
    ),
    Path("tests/generate_arri_tokens_sheet.py"): (None, ("logc3-ei800-curve.png", "arri-wide-gamut-comparison.png")),
    Path("tests/generate_blackmagic_tokens_sheet.py"): (
        None,
        ("blackmagic-transfer-curves.png", "blackmagic-gamut-comparison.png"),
    ),
    Path("tests/generate_canon_tokens_sheet.py"): (
        "--output-dir",
        ("canon-transfer-curves.png", "canon-gamut-conversion.png"),
    ),
    Path("tests/generate_draw_text_user_font_fixtures.py"): (
        "--output",
        ("noto-user-variable.otf", "noto-user-static.otf", "noto-user-collection.ttc"),
    ),
    Path("tests/generate_icc_tokens_sheet.py"): ("--output-dir", ("icc-transfer-curves.png", "icc-rgb-to-rgb.png")),
    Path("tests/generate_io_exr_gpu_rle_sheet.py"): (
        "--output-dir",
        (
            "exr-gpu-rle-openexr-reference.exr",
            "exr-gpu-rle-gpu-write.exr",
            "exr-gpu-rle.png",
            "exr-gpu-comparison-metrics.json",
        ),
    ),
    Path("tests/generate_io_exr_gpu_pxr24_sheet.py"): (
        "--output-dir",
        (
            "exr-gpu-pxr24-openexr-reference.exr",
            "exr-gpu-pxr24-gpu-write.exr",
            "exr-gpu-pxr24.png",
            "exr-gpu-comparison-metrics.json",
        ),
    ),
    Path("tests/generate_io_exr_gpu_b44_sheet.py"): (
        "--output-dir",
        tuple(
            f"exr-gpu-{compression}-{suffix}"
            for compression in ("b44", "b44a")
            for suffix in ("openexr-reference.exr", "gpu-write.exr")
        )
        + ("exr-gpu-b44.png", "exr-gpu-b44a.png", "exr-gpu-comparison-metrics.json"),
    ),
    Path("tests/generate_io_exr_gpu_piz_sheet.py"): (
        "--output-dir",
        (
            "exr-gpu-piz-openexr-reference.exr",
            "exr-gpu-piz-gpu-write.exr",
            "exr-gpu-piz.png",
            "exr-gpu-piz-metrics.json",
        ),
    ),
    Path("tests/generate_log_negative_extension_sheet.py"): (
        None,
        ("slog3-negative-extension.png", "logc4-negative-extension.png"),
    ),
    Path("tests/generate_lut_shaper_oracle.py"): (
        None,
        tuple(
            f"{edge}-{curve}/{name}"
            for edge in (17, 33)
            for curve in ("log", "power", "weak")
            for name in ("metadata.json", "source.3dl", "input.f32", "output.f32")
        ),
    ),
    Path("tests/generate_panasonic_tokens_sheet.py"): (
        "--output-dir",
        ("panasonic-vlog-transfer.png", "panasonic-v-gamut-conversion.png"),
    ),
    Path("tests/generate_red_tokens_sheet.py"): (
        "--output-dir",
        ("red-transfer-curves.png", "red-gamut-conversions.png"),
    ),
    Path("tests/generate_sony_tokens_sheet.py"): (
        None,
        ("slog-signed-curve.png", "slog2-signed-curve.png", "sgamut-equivalence.png"),
    ),
    Path("tests/generate_standard_tokens_sheet.py"): (
        "--output-dir",
        ("standard-transfer-curves.png", "standard-gamut-conversions.png"),
    ),
    Path("tests/generate_vendor_a_tokens_sheet.py"): (None, ("vendor-a-transfers.png", "vendor-a-gamuts.png")),
    Path("tests/generate_vendor_b_tokens_sheet.py"): (None, ("vendor-b-transfers.png", "vendor-b-composites.png")),
}

FILE_SUFFIXES = {
    Path("tools/bake_aces13_analytic_oracle.py"): ".npz",
    Path("tools/bake_aces20_analytic_oracle.py"): ".npz",
    Path("tools/bake_aces20_tables.py"): ".py",
}


def _write_source_exr(path: Path) -> None:
    """Create a deterministic temporary RGB EXR for comparison generators."""
    import numpy as np
    import OpenEXR

    height, width = 540, 960
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    red = np.ascontiguousarray(np.broadcast_to(x, (height, width)))
    green = np.ascontiguousarray(np.broadcast_to(y, (height, width)))
    blue = np.ascontiguousarray((red + green) * np.float32(0.5))
    OpenEXR.File(
        {"type": OpenEXR.scanlineimage, "compression": OpenEXR.ZIP_COMPRESSION},
        {"R": red, "G": green, "B": blue},
    ).write(str(path))


@pytest.mark.parametrize("script_path", SCRIPT_PATHS, ids=str)
def test_development_script_runs_and_writes_expected_files(script_path: Path, tmp_path: Path) -> None:
    """Each development script runs with the current package and writes its expected output in a temporary directory."""
    script = ROOT / script_path
    if script_path in IMPORT_ONLY:
        command = [sys.executable, "-c", "import runpy, sys; runpy.run_path(sys.argv[1])", str(script)]
        expected: tuple[Path, ...] = ()
    elif script_path in DIRECTORY_OUTPUTS:
        flag, filenames = DIRECTORY_OUTPUTS[script_path]
        output = tmp_path / "generated"
        command = [sys.executable, str(script), *((flag,) if flag else ()), str(output)]
        if script_path in EXR_INPUT_SCRIPTS:
            source_exr = tmp_path / "source.exr"
            _write_source_exr(source_exr)
            command.extend(("--source-exr", str(source_exr)))
        expected = tuple(output / filename for filename in filenames)
    else:
        output = tmp_path / f"generated{FILE_SUFFIXES.get(script_path, '.png')}"
        command = [sys.executable, str(script), str(output)]
        expected = (output,)

    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False, timeout=120)
    assert result.returncode == 0, f"{script_path}: {result.stdout}\n{result.stderr}"
    for path in expected:
        assert path.is_file() and path.stat().st_size > 0, f"{script_path} did not write {path}"
