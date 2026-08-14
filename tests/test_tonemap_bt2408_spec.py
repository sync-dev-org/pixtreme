"""Specification, contract, and independent-oracle tests for BT.2408 direct mapping."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import pixtreme as px

ROOT = Path(__file__).resolve().parents[1]

_COLORSPACE_DEFINITIONS = {
    "sRGB": (((0.640, 0.330), (0.300, 0.600), (0.150, 0.060)), (0.3127, 0.3290)),
    "Rec.709": (((0.640, 0.330), (0.300, 0.600), (0.150, 0.060)), (0.3127, 0.3290)),
    "Rec.2020": (((0.708, 0.292), (0.170, 0.797), (0.131, 0.046)), (0.3127, 0.3290)),
    "ACEScg": (((0.713, 0.293), (0.165, 0.830), (0.128, 0.044)), (0.32168, 0.33767)),
}
_BRADFORD = np.asarray(
    (
        (0.8951, 0.2664, -0.1614),
        (-0.7502, 1.7135, 0.0367),
        (0.0389, -0.0685, 1.0296),
    ),
    dtype=np.float64,
)
_ACES_COMBINATIONS = (
    ("aces-1.3", "Rec.709", "bt1886"),
    ("aces-1.3", "sRGB", "srgb"),
    ("aces-2.0", "Rec.709", "bt1886"),
    ("aces-2.0", "sRGB", "srgb"),
    ("aces-1.3-lut", "Rec.709", "bt1886"),
    ("aces-1.3-lut", "sRGB", "srgb"),
    ("aces-2.0-lut", "Rec.709", "bt1886"),
    ("aces-2.0-lut", "sRGB", "srgb"),
)
_BT2408_COMBINATIONS = (("bt2408", "Rec.2020", "hlg"), ("bt2408", "Rec.2020", "pq"))


def _frame(
    values: Any,
    *,
    colorspace: str = "Rec.709",
    gamma: str = "linear",
    channels: str | list[str] = "RGB",
    dtype: Any = np.float32,
) -> px.core.Frame:
    import cupy as cp

    array = np.asarray(values, dtype=dtype)
    if array.ndim == 1:
        array = array.reshape(1, 1, -1)
    elif array.ndim == 2:
        array = array.reshape(1, *array.shape)
    return px.io.from_array(cp.asarray(array), colorspace=colorspace, gamma=gamma, channels=channels)


def _xy_to_xyz(xy: tuple[float, float]) -> np.ndarray:
    x, y = xy
    return np.asarray((x / y, 1.0, (1.0 - x - y) / y), dtype=np.float64)


def _rgb_to_xyz(colorspace: str) -> np.ndarray:
    primaries, white = _COLORSPACE_DEFINITIONS[colorspace]
    unscaled = np.asarray(
        (
            tuple(x / y for x, y in primaries),
            (1.0, 1.0, 1.0),
            tuple((1.0 - x - y) / y for x, y in primaries),
        ),
        dtype=np.float64,
    )
    return unscaled @ np.diag(np.linalg.solve(unscaled, _xy_to_xyz(white)))


def _matrix_to_rec2020(input_colorspace: str) -> np.ndarray:
    _, input_white = _COLORSPACE_DEFINITIONS[input_colorspace]
    _, output_white = _COLORSPACE_DEFINITIONS["Rec.2020"]
    source_cones = _BRADFORD @ _xy_to_xyz(input_white)
    output_cones = _BRADFORD @ _xy_to_xyz(output_white)
    adaptation = np.linalg.inv(_BRADFORD) @ np.diag(output_cones / source_cones) @ _BRADFORD
    return np.linalg.inv(_rgb_to_xyz("Rec.2020")) @ adaptation @ _rgb_to_xyz(input_colorspace)


def _decode_srgb(values: np.ndarray) -> np.ndarray:
    return np.where(values <= 0.04045, values / 12.92, np.power((values + 0.055) / 1.055, 2.4))


def _bt2408_gain(output_gamma: str) -> np.float64:
    if output_gamma == "pq":
        return np.float64(203) / np.float64(10000)
    a = np.float64(0.17883277)
    b = np.float64(1) - np.float64(4) * a
    c = np.float64(0.5) - a * np.log(np.float64(4) * a)
    return (np.exp((np.float64(0.75) - c) / a) + b) / np.float64(12)


def _encode_hlg(values: np.ndarray) -> np.ndarray:
    a = np.float64(0.17883277)
    b = np.float64(1) - np.float64(4) * a
    c = np.float64(0.5) - a * np.log(np.float64(4) * a)
    magnitude = np.abs(values)
    encoded = np.empty_like(magnitude)
    low = magnitude <= np.float64(1) / np.float64(12)
    encoded[low] = np.sqrt(np.float64(3) * magnitude[low])
    encoded[~low] = a * np.log(np.float64(12) * magnitude[~low] - b) + c
    return np.copysign(encoded, values)


def _encode_pq(values: np.ndarray) -> np.ndarray:
    m1 = np.float64(2610) / np.float64(16384)
    m2 = np.float64(2523) / np.float64(32)
    c1 = np.float64(3424) / np.float64(4096)
    c2 = np.float64(2413) / np.float64(128)
    c3 = np.float64(2392) / np.float64(128)
    magnitude = np.abs(values)
    powered = np.power(magnitude, m1)
    encoded = np.power((c1 + c2 * powered) / (np.float64(1) + c3 * powered), m2)
    return np.copysign(encoded, values)


def _oracle(
    values: np.ndarray,
    *,
    input_colorspace: str,
    input_gamma: str,
    output_gamma: str,
) -> np.ndarray:
    decoded = _decode_srgb(values) if input_gamma == "srgb" else values
    linear_rec2020 = decoded @ _matrix_to_rec2020(input_colorspace).T
    scaled = linear_rec2020 * _bt2408_gain(output_gamma)
    return _encode_hlg(scaled) if output_gamma == "hlg" else _encode_pq(scaled)


def test_bt2408_coexists_with_the_fixed_signature_and_exact_ten_combination_table() -> None:
    """v1-tonemap-aces20-analytic acceptance 1-3: signature is fixed and the runtime supplies ten exits."""
    import pixtreme._color.transform as implementation
    import pixtreme._color.view_transform as view_implementation

    assert tuple(inspect.signature(px.color.rgb_to_rgb).parameters) == (
        "frame",
        "input_colorspace",
        "input_gamma",
        "output_colorspace",
        "output_gamma",
        "tonemap",
    )
    assert (*view_implementation._SUPPORTED_COMBINATIONS, *implementation._BT2408_COMBINATIONS) == (
        *_ACES_COMBINATIONS,
        *_BT2408_COMBINATIONS,
    )


@pytest.mark.parametrize(("tonemap", "output_colorspace", "output_gamma"), (*_ACES_COMBINATIONS, *_BT2408_COMBINATIONS))
def test_every_tonemap_combination_is_operational(tonemap: str, output_colorspace: str, output_gamma: str) -> None:
    """v1-tonemap-aces20-analytic acceptance 2: eight ACES and two direct-mapping rows all execute."""
    result = px.color.rgb_to_rgb(
        _frame([0.18, 0.18, 0.18]),
        output_colorspace=output_colorspace,
        output_gamma=output_gamma,
        tonemap=tonemap,
    )
    assert (result.colorspace, result.gamma) == (output_colorspace, output_gamma)
    assert np.isfinite(
        px.io.to_array(
            result,
        ).get()
    ).all()


@pytest.mark.parametrize(
    ("tonemap", "output_colorspace", "output_gamma"),
    (
        ("bt2408", None, None),
        ("bt2408", "Rec.2020", None),
        ("bt2408", None, "pq"),
        ("bt2408", "Rec.709", "pq"),
        ("bt2408", "Rec.2020", "linear"),
        ("BT2408", "Rec.2020", "pq"),
        ("unknown", "Rec.2020", "pq"),
    ),
)
def test_bt2408_missing_unknown_and_table_external_combinations_fail_with_actionable_guidance(
    tonemap: str, output_colorspace: str | None, output_gamma: str | None
) -> None:
    """v1-tonemap-bt2408 acceptance 2-3: all table-external forms fail-fast with why/what/how."""
    with pytest.raises(ValueError) as error:
        px.color.rgb_to_rgb(
            _frame([0.18, 0.18, 0.18]),
            output_colorspace=output_colorspace,
            output_gamma=output_gamma,
            tonemap=tonemap,
        )
    message = str(error.value)
    assert all(part in message for part in ("why=", "what=", "how=", "bt2408", "Rec.2020", "hlg", "pq"))


@pytest.mark.parametrize(("output_gamma", "expected"), (("hlg", 0.75), ("pq", None)))
def test_bt2408_places_rec709_linear_reference_white_from_independent_published_equations(
    output_gamma: str, expected: float | None
) -> None:
    """v1-tonemap-bt2408 acceptance 4-5 and 14: reference white follows independent HLG/PQ equations."""
    result = px.color.rgb_to_rgb(
        _frame([1.0, 1.0, 1.0]),
        output_colorspace="Rec.2020",
        output_gamma=output_gamma,
        tonemap="bt2408",
    )
    oracle = (
        np.full(3, expected, dtype=np.float64)
        if expected is not None
        else _encode_pq(np.full(3, np.float64(203) / np.float64(10000)))
    )
    np.testing.assert_allclose(
        px.io.to_array(
            result,
        ).get()[0, 0],
        oracle,
        rtol=0.0,
        atol=5e-6,
    )


@pytest.mark.parametrize("output_gamma", ("hlg", "pq"))
def test_bt2408_matches_independent_decode_matrix_gain_encode_oracle_for_signed_scene_values(
    output_gamma: str,
) -> None:
    """v1-tonemap-bt2408 acceptance 6-7, 12, and 14: ordered technical composition is signed and unclipped."""
    values = np.asarray(
        (
            (-0.04, 0.18, 0.90),
            (0.25, 0.50, 1.50),
            (1.25, 0.40, 0.10),
        ),
        dtype=np.float64,
    )
    source = _frame(values.astype(np.float32), colorspace="ACEScg", gamma="srgb")
    result = px.color.rgb_to_rgb(
        source,
        output_colorspace="Rec.2020",
        output_gamma=output_gamma,
        tonemap="bt2408",
    )
    expected = _oracle(values, input_colorspace="ACEScg", input_gamma="srgb", output_gamma=output_gamma)

    # 1e-4 covers one fp32 3x3 FMA and device exp/log/pow against host float64,
    # including ST 2084's steep near-black exponent, without masking a gain/order error.
    output = px.io.to_array(
        result,
    ).get()[0]
    np.testing.assert_allclose(output, expected, rtol=0.0, atol=1e-4)
    white_signal = 0.75 if output_gamma == "hlg" else float(_encode_pq(np.asarray((203 / 10000,)))[0])
    assert output.min() < 0.0
    assert output[1].max() > white_signal


@pytest.mark.parametrize("output_gamma", ("hlg", "pq"))
def test_bt2408_input_claims_override_metadata_and_equivalent_rec2020_linear_inputs_converge(
    output_gamma: str,
) -> None:
    """v1-tonemap-bt2408 acceptance 7 and 9: claims win, input is immutable, and equivalent light converges."""
    encoded = np.asarray((0.2, 0.5, 0.8), dtype=np.float64)
    source = _frame(encoded, colorspace="ACEScg", gamma="linear")
    original = (
        px.io.to_array(
            source,
        )
        .get()
        .copy()
    )
    claimed = px.color.rgb_to_rgb(
        source,
        input_colorspace="Rec.709",
        input_gamma="srgb",
        output_colorspace="Rec.2020",
        output_gamma=output_gamma,
        tonemap="bt2408",
    )
    rec2020_linear = _decode_srgb(encoded) @ _matrix_to_rec2020("Rec.709").T
    reference = px.color.rgb_to_rgb(
        _frame(rec2020_linear, colorspace="Rec.2020", gamma="linear"),
        output_colorspace="Rec.2020",
        output_gamma=output_gamma,
        tonemap="bt2408",
    )

    np.testing.assert_array_equal(
        px.io.to_array(
            source,
        ).get(),
        original,
    )
    assert (source.colorspace, source.gamma) == ("ACEScg", "linear")
    np.testing.assert_allclose(
        px.io.to_array(
            claimed,
        ).get(),
        px.io.to_array(
            reference,
        ).get(),
        rtol=0.0,
        atol=1e-4,
    )


@pytest.mark.parametrize(
    "input_colorspace", ("sRGB", "Rec.709", "Rec.2020", "ACES2065-1", "ACEScg", "S-Gamut3", "S-Gamut3.Cine")
)
def test_bt2408_accepts_every_input_colorspace_token(input_colorspace: str) -> None:
    """v1-tonemap-bt2408 acceptance 2 and 7: every existing input colorspace token remains legal."""
    result = px.color.rgb_to_rgb(
        _frame([0.18, 0.18, 0.18]),
        input_colorspace=input_colorspace,
        output_colorspace="Rec.2020",
        output_gamma="pq",
        tonemap="bt2408",
    )
    assert np.isfinite(
        px.io.to_array(
            result,
        ).get()
    ).all()


@pytest.mark.parametrize(
    "input_gamma",
    ("linear", "srgb", "rec709", "bt1886", "pq", "hlg", "s-log3", "logc4", "cineon", "2.2", "2.4", "2.6"),
)
def test_bt2408_accepts_every_input_gamma_token(input_gamma: str) -> None:
    """v1-tonemap-bt2408 acceptance 2 and 7: every existing input transfer token remains legal."""
    result = px.color.rgb_to_rgb(
        _frame([0.18, 0.18, 0.18]),
        input_gamma=input_gamma,
        output_colorspace="Rec.2020",
        output_gamma="hlg",
        tonemap="bt2408",
    )
    assert np.isfinite(
        px.io.to_array(
            result,
        ).get()
    ).all()


def test_bt2408_is_label_driven_preserves_auxiliary_bits_and_returns_private_metadata() -> None:
    """v1-tonemap-bt2408 acceptance 9-10: RGB labels transform while auxiliaries and input storage stay exact."""
    values = np.asarray((9.0, 0.7, 1.0, 0.5, 0.25), dtype=np.float32)
    source = _frame(values, channels=["Z", "A", "B", "R", "G"])
    original = (
        px.io.to_array(
            source,
        )
        .get()
        .copy()
    )
    result = px.color.rgb_to_rgb(
        source,
        output_colorspace="Rec.2020",
        output_gamma="pq",
        tonemap="bt2408",
    )
    output = px.io.to_array(
        result,
    ).get()[0, 0]

    assert (result.colorspace, result.gamma, result.channels, result.matrix) == (
        "Rec.2020",
        "pq",
        ("Z", "A", "B", "R", "G"),
        None,
    )
    assert output[0] == values[0]
    assert output[1] == values[1]
    np.testing.assert_array_equal(
        px.io.to_array(
            source,
        ).get(),
        original,
    )
    assert result.data.data.ptr != source.data.data.ptr


@pytest.mark.parametrize("channels", ("YCbCr", "RG", ["R", "G", "A"]))
def test_bt2408_rejects_frames_without_all_rgb_labels(channels: str | list[str]) -> None:
    """v1-tonemap-bt2408 acceptance 10: missing RGB labels fail before direct mapping."""
    source = _frame(np.zeros(len(px.core.channels(channels)), dtype=np.float32), channels=channels)
    with pytest.raises(ValueError, match="R, G, and B"):
        px.color.rgb_to_rgb(
            source,
            output_colorspace="Rec.2020",
            output_gamma="pq",
            tonemap="bt2408",
        )


@pytest.mark.parametrize(
    ("dtype", "routes"),
    (
        (np.float16, ("cast_dtype",)),
        (np.uint8, ("recode_dtype", "dequantize")),
        (np.uint16, ("recode_dtype", "dequantize")),
    ),
)
def test_bt2408_rejects_non_float32_with_dtype_specific_guidance(
    dtype: type[np.generic], routes: tuple[str, ...]
) -> None:
    """v1-tonemap-bt2408 acceptance 11: all supported non-fp32 storages fail before pixel processing."""
    with pytest.raises(ValueError) as error:
        px.color.rgb_to_rgb(
            _frame([0, 0, 0], dtype=dtype),
            output_colorspace="Rec.2020",
            output_gamma="pq",
            tonemap="bt2408",
        )
    message = str(error.value)
    assert "float32" in message
    assert all(route in message for route in routes)


def test_bt2408_rejects_non_frame_and_unknown_axis_tokens_before_pixel_processing() -> None:
    """v1-tonemap-bt2408 acceptance 11: public type and case-sensitive axis boundaries remain fail-fast."""
    with pytest.raises(ValueError, match="must be a Frame"):
        px.color.rgb_to_rgb(  # type: ignore[arg-type]
            np.zeros((1, 1, 3), dtype=np.float32),
            output_colorspace="Rec.2020",
            output_gamma="pq",
            tonemap="bt2408",
        )
    with pytest.raises(ValueError, match="input_gamma"):
        px.color.rgb_to_rgb(
            _frame([0.18, 0.18, 0.18]),
            input_gamma="sRGB",
            output_colorspace="Rec.2020",
            output_gamma="pq",
            tonemap="bt2408",
        )


def test_bt2408_uses_one_fused_analytic_pass_without_touching_the_aces_lut_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1-tonemap-bt2408 acceptance 8 and 15: direct mapping is one analytic pass with no ACES LUT contact."""
    import pixtreme._color.transform as implementation
    import pixtreme._color.view_transform as view_implementation

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"BT.2408 touched the LUT path: {args!r}, {kwargs!r}")

    monkeypatch.setattr(view_implementation, "_load_lut", fail)
    monkeypatch.setattr(view_implementation, "_apply_lut_data", fail)
    result = px.color.rgb_to_rgb(
        _frame([0.18, 0.18, 0.18]),
        output_colorspace="Rec.2020",
        output_gamma="pq",
        tonemap="bt2408",
    )
    transform_source = inspect.getsource(implementation._transform_data)
    gain_source = inspect.getsource(implementation._bt2408_gain)
    kernel_source = implementation._COLOR_TRANSFORM_KERNEL

    assert np.isfinite(
        px.io.to_array(
            result,
        ).get()
    ).all()
    for required in ("0.17883277", "np.exp", "np.log", "0.75", "np.float64(203)", "np.float64(10000)"):
        assert required in gain_source
    assert "0.58" not in gain_source
    assert transform_source.count("_color_transform_kernel()(") == 1
    assert "Frame(" not in transform_source
    assert kernel_source.index("decode_transfer(input[base + r_index]") < kernel_source.index(
        "const float transformed_red ="
    )
    assert kernel_source.index("const float transformed_red =") < kernel_source.index("* gain")
    assert kernel_source.index("* gain") < kernel_source.index("encode_transfer(scaled_red")


def test_bt2408_docs_and_public_docstring_are_self_contained_and_list_the_ten_rows() -> None:
    """v1-tonemap-aces20-analytic acceptance 18-19: requirements, vocabulary, and docstring expose the contract."""
    requirements_path = ROOT / "docs" / "requirements.md"
    vocabulary_path = ROOT / "docs" / "vocabulary.md"
    if not requirements_path.exists() or not vocabulary_path.exists():
        pytest.skip("docs canon is intentionally absent from this distribution tree")
    requirements = requirements_path.read_text(encoding="utf-8")
    vocabulary = vocabulary_path.read_text(encoding="utf-8")
    docstring = inspect.getdoc(px.color.rgb_to_rgb)

    assert docstring is not None
    for text in (requirements, vocabulary, docstring):
        for required in ("bt2408", "Rec.2020", "hlg", "pq", "203", "clip"):
            assert required in text
    for required in ("direct mapping", "inverse tone mapping", "0.75", "203 / 10000", "approximately 58%"):
        assert required in vocabulary
    supply_table = vocabulary.split("## tonemap 供給組合せ", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
    assert supply_table.count("| `aces-1.3` |") == 2
    assert supply_table.count("| `aces-1.3-lut` |") == 2
    assert supply_table.count("| `aces-2.0` |") == 2
    assert supply_table.count("| `aces-2.0-lut` |") == 2
    assert supply_table.count("| `bt2408` |") == 2
