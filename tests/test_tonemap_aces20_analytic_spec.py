"""Acceptance tests for the analytic ACES 2.0 tonemap path."""

from __future__ import annotations

import inspect
import re
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import cupy as cp
import numpy as np
import pytest
from repository_contracts import require_repo_file

import pixtreme as px

ROOT = Path(__file__).resolve().parents[1]
ORACLE_PATH = ROOT / "tests" / "data" / "tonemap_aces20_analytic_oracle.npz"
TABLE_PATH = ROOT / "src" / "pixtreme" / "_color" / "aces20_tables.py"

_ANALYTIC_COMBINATIONS = (
    ("ACES-1.3", "Rec.709", "BT.1886"),
    ("ACES-1.3", "sRGB", "sRGB"),
    ("ACES-2.0", "Rec.709", "BT.1886"),
    ("ACES-2.0", "sRGB", "sRGB"),
)
_BT2408_COMBINATIONS = (
    ("BT.2408", "Rec.2020", "HLG"),
    ("BT.2408", "Rec.2020", "PQ"),
)
_SUPPLIED_COMBINATIONS = (*_ANALYTIC_COMBINATIONS, *_BT2408_COMBINATIONS)
_EXPECTED_BOUNDARY_GROUPS = (
    "ap1_limit_",
    "hue_wrap_",
    "reach_interval_",
    "cusp_interval_",
    "lower_hull_",
    "upper_hull_",
    "compression_threshold_",
)


def _frame(
    values: Any,
    *,
    colorspace: str = "ACES2065-1",
    gamma: str = "linear",
    channels: str | list[str] = "RGB",
    dtype: np.dtype[Any] | type[np.generic] = np.float32,
) -> px.core.Frame:
    channel_count = len(px.core.channels(channels))
    data = cp.asarray(values, dtype=dtype).reshape(-1, 1, channel_count)
    return px.io.from_array(data, colorspace=colorspace, gamma=gamma, channels=channels)


def test_public_signature_tokens_and_exact_six_row_supply_table() -> None:
    """v1-view-transform-lut-removal acceptance 2 and 4: signature and the exact six-row grammar are fixed."""
    import pixtreme._color.transform as implementation

    signature = inspect.signature(px.color.rgb_to_rgb)
    assert tuple(signature.parameters) == (
        "frame",
        "input_colorspace",
        "input_gamma",
        "output_colorspace",
        "output_gamma",
        "tonemap",
    )
    assert signature.parameters["tonemap"].default is None
    assert implementation._BT2408_COMBINATIONS == _BT2408_COMBINATIONS
    assert implementation._ANALYTIC_COMBINATIONS == _ANALYTIC_COMBINATIONS
    assert implementation._SUPPORTED_COMBINATIONS == _SUPPLIED_COMBINATIONS

    source = _frame((0.18, 0.18, 0.18))
    for tonemap, output_colorspace, output_gamma in _SUPPLIED_COMBINATIONS:
        result = px.color.rgb_to_rgb(
            source,
            output_colorspace=output_colorspace,
            output_gamma=output_gamma,
            tonemap=tonemap,
        )
        assert (result.colorspace, result.gamma, result.matrix) == (output_colorspace, output_gamma, None)


@pytest.mark.parametrize(
    ("tonemap", "output_colorspace", "output_gamma"),
    (
        ("ACES-2.0", "sRGB", "linear"),
        ("ACES-2.0", None, None),
        ("ACES-2.0", "Rec.709", "sRGB"),
        ("ACES-2.0", "Rec.2020", "PQ"),
    ),
)
def test_table_external_forms_fail_before_pixel_processing_with_the_complete_six_row_recipe(
    tonemap: str, output_colorspace: str | None, output_gamma: str | None
) -> None:
    """v1-view-transform-lut-removal acceptance 1: invalid forms fail with why, what, and the six-row how."""
    with pytest.raises(ValueError) as error:
        px.color.rgb_to_rgb(
            _frame((0.18, 0.18, 0.18)),
            output_colorspace=output_colorspace,
            output_gamma=output_gamma,
            tonemap=tonemap,
        )
    message = str(error.value)
    assert all(part in message for part in ("why=", "what=", "how="))
    how = message.split("how=", maxsplit=1)[1]
    for combination in _SUPPLIED_COMBINATIONS:
        assert repr(combination) in how


@pytest.mark.parametrize(
    ("output_colorspace", "output_gamma", "fixture_key"),
    (
        ("Rec.709", "BT.1886", "output_rec709_bt1886"),
        ("sRGB", "sRGB", "output_srgb_srgb"),
    ),
)
def test_analytic_output_matches_the_raw_ocio_cpu_oracle_corpus(
    output_colorspace: str, output_gamma: str, fixture_key: str
) -> None:
    """v1-tonemap-aces20-analytic acceptance 4-7: raw OCIO float32 corpus matches within 2e-4."""
    with np.load(ORACLE_PATH, allow_pickle=False) as fixture:
        source_values = np.asarray(fixture["input"], dtype=np.float32)
        expected = np.asarray(fixture[fixture_key], dtype=np.float32)
        names = tuple(str(name) for name in fixture["names"])
        assert str(fixture["config_name"].item()) == "studio-config-v4.0.0_aces-v2.0_ocio-v2.5"
        assert str(fixture["view"].item()) == "ACES 2.0 - SDR 100 nits (Rec.709)"
        assert str(fixture["ocio_version"].item()) == "2.5.2"
        assert int(fixture["seed"].item()) == 20260805
        assert int(fixture["stratified_count"].item()) >= 8192
        assert float(fixture["tolerance"].item()) == pytest.approx(2e-4)
        assert source_values.shape[0] == len(names) >= 8192 + 17 + 7 * 3
        assert source_values.shape[1] == 3
        assert np.isfinite(source_values).all()
        assert np.isfinite(expected).all()
        for prefix in _EXPECTED_BOUNDARY_GROUPS:
            assert tuple(name for name in names if name.startswith(prefix)) == tuple(
                f"{prefix}{position}" for position in ("below", "at", "above")
            )

    result = px.color.rgb_to_rgb(
        _frame(source_values),
        output_colorspace=output_colorspace,
        output_gamma=output_gamma,
        tonemap="ACES-2.0",
    )
    actual = result.data.get().reshape(-1, 3)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-4)


def test_analytic_input_claims_override_metadata_and_equivalent_ap1_ap0_light_converges() -> None:
    """v1-tonemap-aces20-analytic acceptance 2 and 4: claims win and equivalent AP1/AP0 light converges."""
    ap1_to_ap0 = np.asarray(
        (
            (0.69545224, 0.1406787, 0.16386907),
            (0.04479456, 0.8596711, 0.09553432),
            (-0.00552588, 0.00402521, 1.0015007),
        ),
        dtype=np.float32,
    )
    ap1 = np.asarray((0.4, 0.18, 0.1), dtype=np.float32)
    source_ap1 = _frame(ap1, colorspace="sRGB", gamma="sRGB")
    source_ap0 = _frame(ap1_to_ap0 @ ap1)

    from_claim = px.color.rgb_to_rgb(
        source_ap1,
        input_colorspace="ACEScg",
        input_gamma="linear",
        output_colorspace="sRGB",
        output_gamma="sRGB",
        tonemap="ACES-2.0",
    )
    from_ap0 = px.color.rgb_to_rgb(
        source_ap0,
        output_colorspace="sRGB",
        output_gamma="sRGB",
        tonemap="ACES-2.0",
    )
    np.testing.assert_allclose(from_claim.data.get(), from_ap0.data.get(), rtol=0.0, atol=2e-4)
    assert (source_ap1.colorspace, source_ap1.gamma) == ("sRGB", "sRGB")


def test_analytic_accepts_every_input_axis_token() -> None:
    """v1-tonemap-aces20-analytic acceptance 2 and 4: every existing input colorspace and gamma token is accepted."""
    from pixtreme._core.frame import _COLORSPACE_TOKENS, _GAMMA_TOKENS

    source = _frame((0.0, 0.0, 0.0), colorspace="ACEScg", gamma="linear")
    for input_colorspace in _COLORSPACE_TOKENS:
        result = px.color.rgb_to_rgb(
            source,
            input_colorspace=input_colorspace,
            output_colorspace="sRGB",
            output_gamma="sRGB",
            tonemap="ACES-2.0",
        )
        assert np.isfinite(result.data.get()).all()
    for input_gamma in _GAMMA_TOKENS:
        result = px.color.rgb_to_rgb(
            source,
            input_gamma=input_gamma,
            output_colorspace="sRGB",
            output_gamma="sRGB",
            tonemap="ACES-2.0",
        )
        assert np.isfinite(result.data.get()).all()


def test_algorithm_tables_are_exact_363_record_float32_source_constants() -> None:
    """v1-tonemap-aces20-analytic acceptance 10-11 and issue #1: tables use global read-only memory."""
    import pixtreme._color.aces20_tables as tables

    assert tables._ACES20_TABLE_RECORDS == 363
    assert tables._ACES20_TABLE_FLOAT_COUNT == 1815
    assert tables._ACES20_TABLE_BYTES == 7260
    assert re.fullmatch(r"[0-9a-f]{64}", tables._ACES20_TABLE_SHA256)
    assert tables._ACES20_TABLE_OCIO_VERSION == "2.5.2"
    assert tables._ACES20_TABLE_CONFIG == "studio-config-v4.0.0_aces-v2.0_ocio-v2.5"
    assert tables._ACES20_TABLE_VIEW == "ACES 2.0 - SDR 100 nits (Rec.709)"
    source = tables._ACES20_TABLE_CUDA_SOURCE
    assert "__device__ const float aces20_reach_m[363]" in source
    assert "__device__ const float aces20_gamut_hues[363]" in source
    assert "__device__ const float aces20_gamut_cusp[1089]" in source
    assert source.count("__device__ const float") == 3


def test_table_tool_recreates_the_checked_in_source_byte_for_byte(tmp_path: Path) -> None:
    """v1-tonemap-aces20-analytic acceptance 10-11 and 20-21; GitHub #29: the table tool is deterministic."""
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    tool_path = require_repo_file("tools/bake_aces20_tables.py")
    command = [sys.executable, str(tool_path)]
    subprocess.run([*command, str(first)], cwd=ROOT, check=True, capture_output=True, text=True, timeout=120)
    subprocess.run([*command, str(second)], cwd=ROOT, check=True, capture_output=True, text=True, timeout=120)
    assert first.read_bytes() == second.read_bytes() == TABLE_PATH.read_bytes()


def test_analytic_runtime_is_one_fused_pass_with_constant_tables_and_no_lut_or_ocio_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1-tonemap-aces20-analytic acceptance 8-12: analytic routing is one pass and LUT/OCIO/file independent."""
    import pixtreme._color.aces20_analytic as implementation

    calls = 0
    real_kernel_factory = implementation._aces20_transform_kernel

    def counted_kernel_factory(input_gamma: str, output_gamma: str) -> cp.RawKernel:
        nonlocal calls
        calls += 1
        return real_kernel_factory(input_gamma, output_gamma)

    monkeypatch.setattr(implementation, "_aces20_transform_kernel", counted_kernel_factory)
    result = px.color.rgb_to_rgb(
        _frame((0.4, 0.18, 0.1)),
        output_colorspace="sRGB",
        output_gamma="sRGB",
        tonemap="ACES-2.0",
    )
    assert np.isfinite(result.data.get()).all()
    assert calls == 1

    apply_source = inspect.getsource(implementation._apply_aces20_data)
    module_source = inspect.getsource(implementation)
    assert apply_source.count("_aces20_transform_kernel(input_gamma, output_gamma)(") == 1
    assert "Frame(" not in apply_source
    for forbidden in ("PyOpenColorIO", "importlib.resources", "np.load", "_load_lut", "shaper", "tetrahedral"):
        assert forbidden not in module_source
    assert "cp.clip" not in module_source


def test_frame_contract_and_reference_internal_range_are_preserved() -> None:
    """v1-tonemap-aces20-analytic acceptance 14 and 16: labels, metadata, storage, and internal range are fixed."""
    source = _frame((9.0, 4096.0, 0.75, -2048.0, 0.1), channels=["Z", "R", "A", "G", "B"])
    before = source.data.copy()
    result = px.color.rgb_to_rgb(
        source,
        output_colorspace="sRGB",
        output_gamma="sRGB",
        tonemap="ACES-2.0",
    )
    actual = result.data.get()[0, 0]
    assert result.channels == source.channels
    assert (result.colorspace, result.gamma, result.matrix) == ("sRGB", "sRGB", None)
    assert actual[0].view(np.uint32) == np.float32(9.0).view(np.uint32)
    assert actual[2].view(np.uint32) == np.float32(0.75).view(np.uint32)
    assert np.all((actual[[1, 3, 4]] >= 0.0) & (actual[[1, 3, 4]] <= 1.0))
    assert result.data.data.ptr != source.data.data.ptr
    cp.testing.assert_array_equal(source.data, before)


@pytest.mark.parametrize("channels", ("YCbCr", "RG", ["R", "G", "A"]))
def test_analytic_rejects_missing_rgb_before_rendering(channels: str | list[str]) -> None:
    """v1-tonemap-aces20-analytic acceptance 15: missing RGB labels fail before the analytic pass."""
    with pytest.raises(ValueError, match="R, G, and B"):
        px.color.rgb_to_rgb(
            _frame(np.zeros(len(px.core.channels(channels)), dtype=np.float32), channels=channels),
            output_colorspace="sRGB",
            output_gamma="sRGB",
            tonemap="ACES-2.0",
        )


@pytest.mark.parametrize(
    ("dtype", "routes"),
    (
        (np.float16, ("cast_dtype",)),
        (np.uint8, ("recode_dtype", "dequantize")),
        (np.uint16, ("recode_dtype", "dequantize")),
    ),
)
def test_analytic_rejects_non_float32_with_dtype_specific_guidance(
    dtype: type[np.generic], routes: tuple[str, ...]
) -> None:
    """v1-tonemap-aces20-analytic acceptance 15: every non-fp32 storage fails with the existing conversion recipe."""
    with pytest.raises(ValueError) as error:
        px.color.rgb_to_rgb(
            _frame((0, 0, 0), dtype=dtype),
            output_colorspace="sRGB",
            output_gamma="sRGB",
            tonemap="ACES-2.0",
        )
    message = str(error.value)
    assert "float32" in message
    assert tuple(message.index(route) for route in routes) == tuple(sorted(message.index(route) for route in routes))


def test_remaining_routes_retain_their_exact_float32_bits_and_metadata() -> None:
    """v1-view-transform-lut-removal acceptance 2: every retained route remains bit-identical."""
    from pixtreme._color.aces13_analytic import _apply_aces13_data
    from pixtreme._color.transform import _bt2408_gain, _compose_matrix, _transform_data

    source = _frame((0.4, 0.18, 0.1, 1.2, -0.1, 0.4))
    none = px.color.rgb_to_rgb(source)
    cp.testing.assert_array_equal(none.data.view(cp.uint32), source.data.view(cp.uint32))

    aces13 = px.color.rgb_to_rgb(source, output_colorspace="sRGB", output_gamma="sRGB", tonemap="ACES-1.3")
    expected_aces13 = _apply_aces13_data(
        source.data,
        source.channels,
        input_gamma="linear",
        output_gamma="sRGB",
        matrix=_compose_matrix("ACES2065-1", "ACES2065-1"),
    )
    cp.testing.assert_array_equal(aces13.data.view(cp.uint32), expected_aces13.view(cp.uint32))

    bt2408 = px.color.rgb_to_rgb(source, output_colorspace="Rec.2020", output_gamma="PQ", tonemap="BT.2408")
    expected_bt2408 = _transform_data(
        source.data,
        source.channels,
        input_gamma="linear",
        output_gamma="PQ",
        matrix=_compose_matrix("ACES2065-1", "Rec.2020"),
        gain=_bt2408_gain("PQ"),
    )
    cp.testing.assert_array_equal(bt2408.data.view(cp.uint32), expected_bt2408.view(cp.uint32))

    assert (none.colorspace, none.gamma, none.matrix) == ("ACES2065-1", "linear", None)
    assert (aces13.colorspace, aces13.gamma, aces13.matrix) == ("sRGB", "sRGB", None)
    assert (bt2408.colorspace, bt2408.gamma, bt2408.matrix) == ("Rec.2020", "PQ", None)


def test_oracle_tool_recreates_the_committed_fixture_byte_for_byte(tmp_path: Path) -> None:
    """v1-tonemap-aces20-analytic acceptance 6-7 and 20-21; GitHub #29: the OCIO oracle is deterministic."""
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    tool_path = require_repo_file("tools/bake_aces20_analytic_oracle.py")
    command = [sys.executable, str(tool_path)]
    subprocess.run([*command, str(first)], cwd=ROOT, check=True, capture_output=True, text=True, timeout=120)
    subprocess.run([*command, str(second)], cwd=ROOT, check=True, capture_output=True, text=True, timeout=120)
    assert first.read_bytes() == second.read_bytes() == ORACLE_PATH.read_bytes()


def test_docs_docstring_registry_and_visual_generator_expose_the_six_row_boundary() -> None:
    """v1-view-transform-lut-removal acceptance 4 and 8; GitHub #29: public contracts expose six rows."""
    requirements_path = require_repo_file("docs/requirements.md")
    requirements = requirements_path.read_text(encoding="utf-8")
    vocabulary = (ROOT / "docs_site" / "tokens.md").read_text(encoding="utf-8")
    docstring = inspect.getdoc(px.color.rgb_to_rgb)
    visual_source = (ROOT / "tests" / "generate_tonemap_aces20_analytic_sheet.py").read_text(encoding="utf-8")
    performance_source = (ROOT / "tests" / "test_performance_spec.py").read_text(encoding="utf-8")
    assert docstring is not None
    normalized_docstring = " ".join(docstring.split())

    for text in (requirements, vocabulary, docstring):
        for token in ("ACES-1.3", "ACES-2.0", "BT.2408"):
            assert token in text
        for required in ("analytic", "clip"):
            assert required in text
    for required in ("algorithm table", "363", "7,260", "OCIO", "post-clip"):
        assert required in vocabulary
    assert "Both ``output_colorspace`` and ``output_gamma`` must be supplied explicitly" in normalized_docstring
    assert "``Rec.709`` / ``BT.1886`` and ``sRGB`` / ``sRGB``" in normalized_docstring
    assert "Plain ``ACES-2.0`` is not supplied" not in normalized_docstring
    supply_table = vocabulary.split("## tonemap combinations", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
    for tonemap, output_colorspace, output_gamma in _SUPPLIED_COMBINATIONS:
        assert f"| `{tonemap}` | `{output_colorspace}` | `{output_gamma}` |" in supply_table
    for required in (
        "ACES-1.3",
        "ACES-2.0",
        "BT.2408",
        "hue",
        "cusp",
        "gamut",
        "highlight",
    ):
        assert required in visual_source
    for required in (
        "color-aces13-analytic-srgb",
        "color-aces20-analytic-srgb",
        "color-bt2408-rec2020-pq",
    ):
        assert required in performance_source


@pytest.mark.performance
def test_analytic_fhd_median_is_within_the_absolute_limit() -> None:
    """v1-view-transform-lut-removal acceptance 2: the unchanged ACES 2.0 path remains within 1.5 ms.

    Warmup is time-based (at least 0.5 seconds, matching the registry harness) so that measurement
    starts only after the GPU has ramped from idle to boost clocks. A fixed iteration count finishes
    inside the ramp and reports idle-clock timings when preceding suite cases leave the GPU idle (I-58).
    """
    generator = cp.random.default_rng(20260717)
    values = generator.random((1080, 1920, 3), dtype=cp.float32)
    source = px.io.from_array(values, colorspace="ACES2065-1", gamma="linear", channels="RGB")

    def measure(tonemap: str) -> float:
        warmup_started_at = perf_counter()
        while perf_counter() - warmup_started_at < 0.5:
            px.color.rgb_to_rgb(source, output_colorspace="sRGB", output_gamma="sRGB", tonemap=tonemap)
            cp.cuda.Stream.null.synchronize()
        samples: list[float] = []
        for _ in range(31):
            start = cp.cuda.Event()
            end = cp.cuda.Event()
            start.record()
            px.color.rgb_to_rgb(source, output_colorspace="sRGB", output_gamma="sRGB", tonemap=tonemap)
            end.record()
            end.synchronize()
            samples.append(float(cp.cuda.get_elapsed_time(start, end)))
        return float(np.median(np.asarray(samples, dtype=np.float64)))

    analytic_ms = measure("ACES-2.0")
    assert analytic_ms <= 1.5, analytic_ms
