"""Acceptance tests for the analytic ACES 1.3 tonemap path."""

from __future__ import annotations

import hashlib
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

import pixtreme as px

ROOT = Path(__file__).resolve().parents[1]
ORACLE_PATH = ROOT / "tests" / "data" / "tonemap_aces13_analytic_oracle.npz"

_ANALYTIC_COMBINATIONS = (
    ("aces-1.3", "Rec.709", "bt1886"),
    ("aces-1.3", "sRGB", "srgb"),
    ("aces-2.0", "Rec.709", "bt1886"),
    ("aces-2.0", "sRGB", "srgb"),
)
_LUT_COMBINATIONS = (
    ("aces-1.3-lut", "Rec.709", "bt1886"),
    ("aces-1.3-lut", "sRGB", "srgb"),
    ("aces-2.0-lut", "Rec.709", "bt1886"),
    ("aces-2.0-lut", "sRGB", "srgb"),
)
_BT2408_COMBINATIONS = (
    ("bt2408", "Rec.2020", "hlg"),
    ("bt2408", "Rec.2020", "pq"),
)
_SUPPLIED_COMBINATIONS = (*_ANALYTIC_COMBINATIONS, *_LUT_COMBINATIONS, *_BT2408_COMBINATIONS)
_ASSET_SHA256 = {
    "view_transform_aces-1.3_rec709_bt1886.npz": "071e62774058b724e5e19ff696dc034d91327e109da539c3701b786187ddbccf",
    "view_transform_aces-1.3_srgb_srgb.npz": "40030a2299be1527fbd5745796525ebae430ef44cb57e67767b298355f171088",
    "view_transform_aces-2.0_rec709_bt1886.npz": "2c1c230fcf24f32e06ffcc082586e090c2bf650b40fb8157bb44362d6da77165",
    "view_transform_aces-2.0_srgb_srgb.npz": "a6b3ae6440ec11486d67d0c1ad9795913f69b2eb080d9c115c84ed3ba8a91a47",
}
_BOUNDARY_POSITIONS = ("below", "at", "above")
_EXPECTED_ORACLE_BOUNDARY_NAMES = (
    *(f"curve0_{index}_{position}" for index in range(5) for position in _BOUNDARY_POSITIONS),
    *(f"curve1_{index}_{position}" for index in range(15) for position in _BOUNDARY_POSITIONS),
)


def _actionable_slots(message: str) -> tuple[str, str, str]:
    match = re.fullmatch(r"why=(.+?);\s*what=(.+?);\s*how=(.+)", message)
    assert match is not None
    return match.groups()


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


def test_public_signature_tokens_and_exact_ten_row_supply_table() -> None:
    """v1-tonemap-aces20-analytic acceptance 1-3: signature and the exact ten-row grammar are fixed."""
    import pixtreme._color.transform as implementation
    import pixtreme._color.view_transform as _view_transform

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
    assert _view_transform._ANALYTIC_COMBINATIONS == _ANALYTIC_COMBINATIONS
    assert _view_transform._LUT_COMBINATIONS == _LUT_COMBINATIONS
    assert _view_transform._PUBLIC_TO_INTERNAL_LUT == {
        "aces-1.3-lut": "aces-1.3",
        "aces-2.0-lut": "aces-2.0",
    }

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
        ("aces-2.0", "Rec.2020", "pq"),
        ("ACES-1.3", "sRGB", "srgb"),
        ("aces-1.3", None, None),
        ("aces-1.3-lut", "Rec.2020", "pq"),
        ("bt2408", "sRGB", "srgb"),
    ),
)
def test_table_external_forms_fail_before_pixel_processing_with_the_complete_ten_row_recipe(
    tonemap: str, output_colorspace: str | None, output_gamma: str | None
) -> None:
    """v1-tonemap-aces20-analytic acceptance 3: invalid forms fail with the complete why/what/how recipe."""
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
    assert "('aces-2.0', 'sRGB', 'srgb')" in how


@pytest.mark.parametrize(
    ("output_colorspace", "output_gamma", "fixture_key"),
    (
        ("Rec.709", "bt1886", "output_rec709_bt1886"),
        ("sRGB", "srgb", "output_srgb_srgb"),
    ),
)
def test_analytic_output_matches_the_raw_ocio_cpu_oracle_corpus(
    output_colorspace: str, output_gamma: str, fixture_key: str
) -> None:
    """v1-tonemap-aces13-analytic acceptance 4-7: raw OCIO float32 corpus matches within 5e-5."""
    with np.load(ORACLE_PATH, allow_pickle=False) as fixture:
        source_values = np.asarray(fixture["input"], dtype=np.float32)
        expected = np.asarray(fixture[fixture_key], dtype=np.float32)
        names = tuple(str(name) for name in fixture["names"])
        assert str(fixture["config_name"].item()) == "studio-config-v2.2.0_aces-v1.3_ocio-v2.4"
        assert str(fixture["view"].item()) == "ACES 1.0 - SDR Video"
        assert int(fixture["seed"].item()) == 20260805
        assert float(fixture["tolerance"].item()) == pytest.approx(5e-5)
        boundary_names = tuple(name for name in names if name.startswith(("curve0_", "curve1_")))
        assert boundary_names == _EXPECTED_ORACLE_BOUNDARY_NAMES
        assert int(fixture["deterministic_patch_count"].item()) == 17 + len(_EXPECTED_ORACLE_BOUNDARY_NAMES)
        assert source_values.shape == (17 + len(_EXPECTED_ORACLE_BOUNDARY_NAMES) + 4096, 3)
        assert np.min(source_values) == np.float32(-0.5)
        assert np.max(source_values) == np.float32(16.0)
        assert np.isfinite(source_values).all()
        assert np.isfinite(expected).all()

    source = _frame(source_values)
    result = px.color.rgb_to_rgb(
        source,
        output_colorspace=output_colorspace,
        output_gamma=output_gamma,
        tonemap="aces-1.3",
    )
    actual = result.data.get().reshape(-1, 3)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=5e-5)
    assert np.max(actual) > 1.0


def test_analytic_input_claims_override_metadata_and_equivalent_ap1_ap0_light_converges() -> None:
    """v1-tonemap-aces13-analytic acceptance 2 and 4: claims win and equivalent scene light converges."""
    ap1_to_ap0 = np.asarray(
        (
            (0.69545224, 0.1406787, 0.16386907),
            (0.04479456, 0.8596711, 0.09553432),
            (-0.00552588, 0.00402521, 1.0015007),
        ),
        dtype=np.float32,
    )
    ap1 = np.asarray((0.4, 0.18, 0.1), dtype=np.float32)
    ap0 = ap1_to_ap0 @ ap1
    source_ap1 = _frame(ap1, colorspace="sRGB", gamma="srgb")
    source_ap0 = _frame(ap0)

    from_claim = px.color.rgb_to_rgb(
        source_ap1,
        input_colorspace="ACEScg",
        input_gamma="linear",
        output_colorspace="sRGB",
        output_gamma="srgb",
        tonemap="aces-1.3",
    )
    from_ap0 = px.color.rgb_to_rgb(
        source_ap0,
        output_colorspace="sRGB",
        output_gamma="srgb",
        tonemap="aces-1.3",
    )
    np.testing.assert_allclose(from_claim.data.get(), from_ap0.data.get(), rtol=0.0, atol=5e-5)
    assert (source_ap1.colorspace, source_ap1.gamma) == ("sRGB", "srgb")


def test_analytic_accepts_every_input_axis_token() -> None:
    """v1-tonemap-aces13-analytic acceptance 2 and 4: every input colorspace and gamma token is accepted."""
    from pixtreme._core.frame import _COLORSPACE_TOKENS, _GAMMA_TOKENS

    source = _frame((0.0, 0.0, 0.0), colorspace="ACEScg", gamma="linear")
    for input_colorspace in _COLORSPACE_TOKENS:
        result = px.color.rgb_to_rgb(
            source,
            input_colorspace=input_colorspace,
            output_colorspace="sRGB",
            output_gamma="srgb",
            tonemap="aces-1.3",
        )
        assert np.isfinite(result.data.get()).all()
    for input_gamma in _GAMMA_TOKENS:
        result = px.color.rgb_to_rgb(
            source,
            input_gamma=input_gamma,
            output_colorspace="sRGB",
            output_gamma="srgb",
            tonemap="aces-1.3",
        )
        assert np.isfinite(result.data.get()).all()


def test_analytic_is_label_driven_preserves_auxiliary_bits_and_returns_private_storage() -> None:
    """v1-tonemap-aces13-analytic acceptance 14-15 and 17: labels, metadata, storage, and range are preserved."""
    source = _frame(
        (9.0, 0.4, 0.75, 0.18, 0.1),
        channels=["Z", "R", "A", "G", "B"],
    )
    before = source.data.copy()
    result = px.color.rgb_to_rgb(
        source,
        output_colorspace="sRGB",
        output_gamma="srgb",
        tonemap="aces-1.3",
    )
    actual = result.data.get()[0, 0]
    assert result.channels == source.channels
    assert (result.colorspace, result.gamma, result.matrix) == ("sRGB", "srgb", None)
    assert actual[0].view(np.uint32) == np.float32(9.0).view(np.uint32)
    assert actual[2].view(np.uint32) == np.float32(0.75).view(np.uint32)
    assert result.data.data.ptr != source.data.data.ptr
    cp.testing.assert_array_equal(source.data, before)


@pytest.mark.parametrize("channels", ("YCbCr", "RG", ["R", "G", "A"]))
def test_analytic_rejects_missing_rgb_before_rendering(channels: str | list[str]) -> None:
    """v1-tonemap-aces13-analytic acceptance 15: missing RGB labels fail before the analytic pass."""
    with pytest.raises(ValueError, match="R, G, and B"):
        px.color.rgb_to_rgb(
            _frame(np.zeros(len(px.core.channels(channels)), dtype=np.float32), channels=channels),
            output_colorspace="sRGB",
            output_gamma="srgb",
            tonemap="aces-1.3",
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
    """v1-tonemap-aces13-analytic acceptance 16: non-fp32 data fails with the existing conversion recipe."""
    with pytest.raises(ValueError) as error:
        px.color.rgb_to_rgb(
            _frame((0, 0, 0), dtype=dtype),
            output_colorspace="sRGB",
            output_gamma="srgb",
            tonemap="aces-1.3",
        )
    message = str(error.value)
    assert "float32" in message
    assert tuple(message.index(route) for route in routes) == tuple(sorted(message.index(route) for route in routes))


def test_analytic_rejects_non_frame_with_three_part_guidance() -> None:
    """v1-tonemap-aces13-analytic acceptance 16: non-Frame input has complete actionable guidance."""
    with pytest.raises(ValueError) as error:
        px.color.rgb_to_rgb(  # type: ignore[arg-type]
            cp.zeros((1, 1, 3), dtype=cp.float32),
            output_colorspace="sRGB",
            output_gamma="srgb",
            tonemap="aces-1.3",
        )
    _, what, how = _actionable_slots(str(error.value))
    assert "ndarray" in what
    assert "pixtreme.core.Frame" in how


@pytest.mark.parametrize(
    ("parameter", "invalid_value"),
    (
        ("input_colorspace", "acescg"),
        ("output_colorspace", "rec709"),
        ("input_gamma", "Linear"),
        ("output_gamma", "sRGB"),
    ),
)
def test_analytic_rejects_every_unknown_axis_token_with_three_part_guidance(parameter: str, invalid_value: str) -> None:
    """v1-tonemap-aces13-analytic acceptance 16: every named-axis error has why, what, and how."""
    from pixtreme._core.frame import _COLORSPACE_TOKENS, _GAMMA_TOKENS

    accepted = _COLORSPACE_TOKENS if parameter.endswith("colorspace") else _GAMMA_TOKENS
    arguments: dict[str, str] = {
        "output_colorspace": "sRGB",
        "output_gamma": "srgb",
        "tonemap": "aces-1.3",
        parameter: invalid_value,
    }
    with pytest.raises(ValueError) as error:
        px.color.rgb_to_rgb(
            _frame((0.0, 0.0, 0.0)),
            **arguments,  # type: ignore[arg-type]
        )
    why, what, how = _actionable_slots(str(error.value))
    assert parameter in why
    assert repr(invalid_value) in what
    assert repr(accepted) in how


@pytest.mark.parametrize(
    ("public_token", "internal_version", "output_colorspace", "output_gamma"),
    (
        ("aces-1.3-lut", "aces-1.3", "Rec.709", "bt1886"),
        ("aces-1.3-lut", "aces-1.3", "sRGB", "srgb"),
        ("aces-2.0-lut", "aces-2.0", "Rec.709", "bt1886"),
        ("aces-2.0-lut", "aces-2.0", "sRGB", "srgb"),
    ),
)
def test_lut_public_tokens_are_bit_identical_to_the_unchanged_internal_versions(
    public_token: str, internal_version: str, output_colorspace: str, output_gamma: str
) -> None:
    """v1-tonemap-aces13-analytic acceptance 10-13: public LUT names map to unchanged internal identities."""
    from pixtreme._color.view_transform import _apply_lut_data, _load_lut_identity

    source = _frame((0.4, 0.18, 0.1, 1.2, -0.1, 0.4))
    public = px.color.rgb_to_rgb(
        source,
        output_colorspace=output_colorspace,
        output_gamma=output_gamma,
        tonemap=public_token,
    )
    shaper, shaper_domain, lut = _load_lut_identity(internal_version, output_colorspace, output_gamma)
    internal_data = _apply_lut_data(
        source.data,
        source.channels,
        shaper=shaper,
        shaper_domain=shaper_domain,
        lut=lut,
        output_gamma=output_gamma,
    )
    cp.testing.assert_array_equal(public.data.view(cp.uint32), internal_data.view(cp.uint32))
    assert (public.colorspace, public.gamma, public.channels, public.matrix) == (
        output_colorspace,
        output_gamma,
        source.channels,
        None,
    )


def test_lut_assets_retain_their_names_metadata_and_exact_bytes() -> None:
    """v1-tonemap-aces13-analytic acceptance 10-12: the four packaged LUT archives remain byte-identical."""
    data_dir = ROOT / "src" / "pixtreme" / "data"
    for filename, expected_sha256 in _ASSET_SHA256.items():
        path = data_dir / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256
        with np.load(path, allow_pickle=False) as archive:
            expected_version = "aces-1.3" if "aces-1.3" in filename else "aces-2.0"
            assert str(archive["version"].item()) == expected_version


def test_analytic_runtime_is_one_fused_pass_with_no_lut_or_ocio_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    """v1-tonemap-aces13-analytic acceptance 8-9 and 13: analytic routing is one pass and data-free."""
    import pixtreme._color.aces13_analytic as implementation
    import pixtreme._color.view_transform as _view_transform

    calls = 0
    real_kernel_factory = implementation._aces13_transform_kernel

    def counted_kernel_factory(input_gamma: str, output_gamma: str) -> cp.RawKernel:
        nonlocal calls
        calls += 1
        return real_kernel_factory(input_gamma, output_gamma)

    def forbidden_lut(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"analytic path touched LUT args={args!r}, kwargs={kwargs!r}")

    monkeypatch.setattr(implementation, "_aces13_transform_kernel", counted_kernel_factory)
    monkeypatch.setattr(_view_transform, "_load_lut", forbidden_lut)
    result = px.color.rgb_to_rgb(
        _frame((0.4, 0.18, 0.1)),
        output_colorspace="sRGB",
        output_gamma="srgb",
        tonemap="aces-1.3",
    )
    assert np.isfinite(result.data.get()).all()
    assert calls == 1

    apply_source = inspect.getsource(implementation._apply_aces13_data)
    module_source = inspect.getsource(implementation)
    assert apply_source.count("_aces13_transform_kernel(input_gamma, output_gamma)(") == 1
    assert "Frame(" not in apply_source
    for forbidden in ("PyOpenColorIO", "importlib.resources", "np.load", "_load_lut", "shaper", "tetrahedral"):
        assert forbidden not in module_source


def test_oracle_tool_recreates_the_committed_fixture_byte_for_byte(tmp_path: Path) -> None:
    """v1-tonemap-aces13-analytic acceptance 20-21: the external OCIO oracle is byte-deterministic."""
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    tool_path = ROOT / "tools" / "bake_aces13_analytic_oracle.py"
    if not tool_path.is_file():
        pytest.skip("repo-only tooling contract: tools/bake_aces13_analytic_oracle.py is absent from this distribution")
    command = [sys.executable, str(tool_path)]
    subprocess.run([*command, str(first)], cwd=ROOT, check=True, capture_output=True, text=True, timeout=120)
    subprocess.run([*command, str(second)], cwd=ROOT, check=True, capture_output=True, text=True, timeout=120)
    assert first.read_bytes() == second.read_bytes() == ORACLE_PATH.read_bytes()


def test_docs_docstring_registry_and_visual_generator_expose_the_new_supply_boundary() -> None:
    """v1-tonemap-aces20-analytic acceptance 18-19 and 22-23: public texts and tooling use the ten rows."""
    requirements_path = ROOT / "docs" / "requirements.md"
    if not requirements_path.is_file():
        pytest.skip("repo-only documentation contract: docs/requirements.md is absent from this distribution")
    requirements = requirements_path.read_text(encoding="utf-8")
    vocabulary = (ROOT / "docs_site" / "tokens.md").read_text(encoding="utf-8")
    docstring = inspect.getdoc(px.color.rgb_to_rgb)
    visual_source = (ROOT / "tests" / "generate_tonemap_aces13_analytic_sheet.py").read_text(encoding="utf-8")
    performance_source = (ROOT / "tests" / "test_performance_spec.py").read_text(encoding="utf-8")
    assert docstring is not None
    normalized_docstring = " ".join(docstring.split())

    for text in (requirements, vocabulary, docstring):
        for token in ("aces-1.3", "aces-1.3-lut", "aces-2.0", "aces-2.0-lut", "bt2408"):
            assert token in text
        for required in ("analytic", "LUT", "clip"):
            assert required in text
    assert "Both ``output_colorspace`` and ``output_gamma`` must be supplied explicitly" in normalized_docstring
    assert "``Rec.709`` / ``bt1886`` and ``sRGB`` / ``srgb``" in normalized_docstring
    assert "Plain ``aces-2.0`` is not supplied" not in normalized_docstring
    supply_table = vocabulary.split("## tonemap combinations", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
    for tonemap, output_colorspace, output_gamma in _SUPPLIED_COMBINATIONS:
        assert f"| `{tonemap}` | `{output_colorspace}` | `{output_gamma}` |" in supply_table
    assert supply_table.count("| `aces-2.0` |") == 2
    for required in ("absolute", "difference", "aces-1.3-lut", "gamut", "highlight"):
        assert required in visual_source
    for required in (
        "color-aces13-analytic-srgb",
        "color-aces20-analytic-srgb",
        "color-aces20-lut-srgb",
        "color-bt2408-rec2020-pq",
    ):
        assert required in performance_source


@pytest.mark.performance
def test_analytic_fhd_median_is_within_lut_ratio_and_absolute_limits() -> None:
    """v1-tonemap-aces13-analytic acceptance 24: same-run FHD median is <=1.5x LUT and <=0.20 ms.

    Warmup is time-based (at least 0.5 seconds, matching the registry harness) so that measurement
    starts only after the GPU has ramped from idle to boost clocks. A fixed iteration count finishes
    inside the ramp and reports idle-clock timings when preceding suite cases leave the GPU idle (I-58).
    """
    values = cp.linspace(np.float32(-0.25), np.float32(4.0), 1920 * 1080 * 3, dtype=cp.float32).reshape(1080, 1920, 3)
    source = px.io.from_array(values, colorspace="ACES2065-1", gamma="linear", channels="RGB")

    def measure(tonemap: str) -> float:
        warmup_started_at = perf_counter()
        while perf_counter() - warmup_started_at < 0.5:
            px.color.rgb_to_rgb(source, output_colorspace="sRGB", output_gamma="srgb", tonemap=tonemap)
            cp.cuda.Stream.null.synchronize()
        samples: list[float] = []
        for _ in range(31):
            start = cp.cuda.Event()
            end = cp.cuda.Event()
            start.record()
            px.color.rgb_to_rgb(source, output_colorspace="sRGB", output_gamma="srgb", tonemap=tonemap)
            end.record()
            end.synchronize()
            samples.append(float(cp.cuda.get_elapsed_time(start, end)))
        return float(np.median(np.asarray(samples, dtype=np.float64)))

    analytic_ms = measure("aces-1.3")
    lut_ms = measure("aces-1.3-lut")
    assert analytic_ms <= lut_ms * 1.5, (analytic_ms, lut_ms)
    assert analytic_ms <= 0.20, (analytic_ms, lut_ms)
