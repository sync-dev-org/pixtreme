"""Specification tests for Panasonic V-Gamut and V-Log tokens."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Literal, get_args, get_origin, get_type_hints

import cupy as cp
import numpy as np
import pytest
from repository_contracts import latest_changelog_section, require_repo_file

import pixtreme as px

ROOT = Path(__file__).resolve().parents[1]

_COLORSPACES = (
    "sRGB",
    "Rec.709",
    "Rec.2020",
    "P3-DCI",
    "P3-D60",
    "P3-D65",
    "SMPTE-C",
    "ACES2065-1",
    "ACEScg",
    "S-Gamut",
    "S-Gamut3",
    "S-Gamut3.Cine",
    "ARRI-Wide-Gamut-3",
    "ARRI-Wide-Gamut-4",
    "Blackmagic-Wide-Gamut-Gen-5",
    "DaVinci-Wide-Gamut",
    "REDWideGamutRGB",
    "DRAGONcolor",
    "DRAGONcolor2",
    "REDcolor2",
    "REDcolor3",
    "REDcolor4",
    "Canon-Cinema-Gamut",
    "V-Gamut",
    "D-Gamut",
    "F-Gamut-C",
    "Apple-Wide-Gamut",
)
_GAMMAS = (
    "linear",
    "sRGB",
    "Rec.709",
    "BT.1886",
    "PQ",
    "HLG",
    "ACEScc",
    "ACEScct",
    "S-Log",
    "S-Log2",
    "S-Log3",
    "ARRI-LogC3",
    "ARRI-LogC4",
    "Blackmagic-Film-Gen-5",
    "DaVinci-Intermediate",
    "RED-Log3G10",
    "REDlogFilm",
    "Canon-Log",
    "Canon-Log-2",
    "Canon-Log-3",
    "V-Log",
    "D-Log",
    "F-Log",
    "F-Log2",
    "N-Log",
    "L-Log",
    "Apple-Log",
    "Samsung-Log",
    "Cineon",
    "Gamma-2.2",
    "Gamma-2.4",
    "Gamma-2.5",
    "Gamma-2.6",
)
_ALIASES = (
    px.core.ChromaticAdaptation,
    px.core.ReferenceWhite,
    px.core.Colorspace,
    px.core.Gamma,
    px.core.Matrix,
    px.core.Dtype,
    px.core.Layout,
    px.core.Tonemap,
    px.core.Range,
    px.core.Interpolation,
    px.core.Border,
    px.core.ChromaSiting,
    px.core.StackDirection,
    px.core.SobelDirection,
    px.core.TemplateMatchingMethod,
    px.core.Blend,
    px.core.Alpha,
    px.core.Antialiasing,
    px.core.TextLanguage,
    px.core.TextAnchor,
    px.core.TextAlign,
    px.core.TextFont,
    px.core.GeneratorKind,
    px.core.ColorBarsStandard,
    px.core.ColorBarsOutput,
    px.core.MorphologyShape,
    px.core.ImageFormat,
    px.core.TiffCompression,
    px.core.ExrCompression,
    px.core.VectorBlurShutter,
)

_A = np.float64("0.241514")
_B = np.float64("0.00873")
_C = np.float64("0.598206")
_LINEAR_CUT = np.float64("0.01")
_M = _A / ((_LINEAR_CUT + _B) * np.log(np.float64(10.0)))
_D = _A * np.log10(_LINEAR_CUT + _B) + _C - _M * _LINEAR_CUT
_ENCODED_CUT = _A * np.log10(_LINEAR_CUT + _B) + _C

_D65 = (0.3127, 0.3290)
_ACES_WHITE = (0.32168, 0.33767)
_REC709 = (((0.640, 0.330), (0.300, 0.600), (0.150, 0.060)), _D65)
_ACES2065 = (((0.7347, 0.2653), (0.0000, 1.0000), (0.0001, -0.0770)), _ACES_WHITE)
_V_GAMUT = (((0.730, 0.280), (0.165, 0.840), (0.100, -0.030)), _D65)
_V_GAMUT_RGB_TO_XYZ = np.asarray(
    (
        (0.679644469878474, 0.152211412439755, 0.118600044733443),
        (0.260685550090374, 0.774894463329659, -0.035580013420033),
        (-0.009310198217513, -0.004612467043629, 1.102980416021021),
    ),
    dtype=np.float64,
)
_PANASONIC_RGB_TO_XYZ = np.asarray(
    ((0.679644, 0.152211, 0.118600), (0.260686, 0.774894, -0.035580), (-0.009310, -0.004612, 1.102980)),
    dtype=np.float64,
)
_PANASONIC_TO_BT709 = np.asarray(
    ((1.806576, -0.695697, -0.110879), (-0.170090, 1.305955, -0.135865), (-0.025206, -0.154468, 1.179674)),
    dtype=np.float64,
)
_BRADFORD_TO_ACES = np.asarray(
    (
        (0.724616704131530, 0.166915288193706, 0.108468007674764),
        (0.021390245413146, 0.984908155703054, -0.006298401116201),
        (-0.009235562870766, -0.001056905639005, 1.010292468509770),
    ),
    dtype=np.float64,
)
_BRADFORD = np.asarray(
    ((0.8951, 0.2664, -0.1614), (-0.7502, 1.7135, 0.0367), (0.0389, -0.0685, 1.0296)),
    dtype=np.float64,
)


def _vlog_encode(reflectance: np.ndarray) -> np.ndarray:
    values = np.asarray(reflectance, dtype=np.float64)
    result = np.empty_like(values, dtype=np.float64)
    linear = values < _LINEAR_CUT
    result[linear] = _M * values[linear] + _D
    result[~linear] = _A * np.log10(values[~linear] + _B) + _C
    return result


def _vlog_decode(encoded: np.ndarray) -> np.ndarray:
    values = np.asarray(encoded, dtype=np.float64)
    result = np.empty_like(values, dtype=np.float64)
    linear = values < _ENCODED_CUT
    result[linear] = (values[linear] - _D) / _M
    result[~linear] = np.float64(10.0) ** ((values[~linear] - _C) / _A) - _B
    return result


def _frame(
    values: np.ndarray | tuple[float, ...],
    *,
    colorspace: str = "ACEScg",
    gamma: str = "linear",
    auxiliary: bool = False,
) -> px.core.Frame:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = np.repeat(array[:, None], 3, axis=1)
    channels: tuple[str, ...] = ("R", "G", "B")
    if auxiliary:
        auxiliary_values = (np.arange(array.shape[0], dtype=np.float32) + np.float32(16.0))[:, None]
        array = np.concatenate((auxiliary_values, array[:, (2, 0, 1)]), axis=1)
        channels = ("Z", "B", "R", "G")
    return px.io.from_array(
        cp.asarray(array[None, :, :]),
        colorspace=colorspace,
        gamma=gamma,
        channels=channels,
        matrix="native",
    )


def _rgb_values(frame: px.core.Frame) -> np.ndarray:
    array = px.io.to_array(frame).get()[0]
    return array[:, [frame.channels.index(label) for label in ("R", "G", "B")]]


def _literal_strings(annotation: object) -> tuple[str, ...]:
    if get_origin(annotation) is Literal:
        return tuple(value for value in get_args(annotation) if isinstance(value, str))
    return tuple(value for argument in get_args(annotation) for value in _literal_strings(argument))


def _xy_to_xyz(xy: tuple[float, float]) -> np.ndarray:
    x, y = xy
    return np.asarray((x / y, 1.0, (1.0 - x - y) / y), dtype=np.float64)


def _rgb_to_xyz(
    primaries: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    white: tuple[float, float],
) -> np.ndarray:
    unscaled = np.asarray(
        (tuple(x / y for x, y in primaries), (1.0, 1.0, 1.0), tuple((1.0 - x - y) / y for x, y in primaries)),
        dtype=np.float64,
    )
    return unscaled @ np.diag(np.linalg.solve(unscaled, _xy_to_xyz(white)))


def _adaptation(source: tuple[float, float], target: tuple[float, float]) -> np.ndarray:
    source_cones = _BRADFORD @ _xy_to_xyz(source)
    target_cones = _BRADFORD @ _xy_to_xyz(target)
    return np.linalg.inv(_BRADFORD) @ np.diag(target_cones / source_cones) @ _BRADFORD


def _conversion(
    source: tuple[tuple[tuple[float, float], tuple[float, float], tuple[float, float]], tuple[float, float]],
    target: tuple[tuple[tuple[float, float], tuple[float, float], tuple[float, float]], tuple[float, float]],
) -> np.ndarray:
    source_primaries, source_white = source
    target_primaries, target_white = target
    return (
        np.linalg.inv(_rgb_to_xyz(target_primaries, target_white))
        @ _adaptation(source_white, target_white)
        @ _rgb_to_xyz(source_primaries, source_white)
    )


def _adjacent_float32(center: np.float32, radius: int) -> np.ndarray:
    below = np.empty(radius, dtype=np.float32)
    value = center
    for index in range(radius - 1, -1, -1):
        value = np.nextafter(value, np.float32(-np.inf))
        below[index] = value
    above = np.empty(radius, dtype=np.float32)
    value = center
    for index in range(radius):
        value = np.nextafter(value, np.float32(np.inf))
        above[index] = value
    return np.concatenate((below, np.asarray((center,), dtype=np.float32), above))


def test_panasonic_tokens_extend_canonical_vocabulary_and_public_static_surfaces() -> None:
    """v1-panasonic-tokens acceptance 99-100; v1-standard-tokens acceptance 117;
    v1-vendor-a-tokens acceptance 140-141; v1-vendor-b-tokens acceptance 166-167:
    expose only the exact current canonical vocabulary.
    """
    assert get_args(px.core.Colorspace) == _COLORSPACES
    assert get_args(px.core.Gamma) == _GAMMAS
    assert len(_ALIASES) == 30
    assert sum(len(get_args(alias)) for alias in _ALIASES) == 188
    assert _literal_strings(get_type_hints(px.color.linear_to_gamma)["gamma"]) == _GAMMAS
    assert _literal_strings(get_type_hints(px.color.rgb_to_rgb)["input_colorspace"]) == _COLORSPACES
    assert _literal_strings(get_type_hints(px.color.rgb_to_rgb)["output_gamma"]) == _GAMMAS
    frame = _frame((0.18,), colorspace="V-Gamut", gamma="V-Log")
    assert (frame.colorspace, frame.gamma) == ("V-Gamut", "V-Log")
    assert "colorspace='V-Gamut'" in repr(frame)
    assert "gamma='V-Log'" in repr(frame)

    from pixtreme._core.vocabulary import _PERMANENT_TOKEN_ALIASES

    assert len(_PERMANENT_TOKEN_ALIASES) == 4
    assert not any(token in {"V-Gamut", "V-Log"} for alias in _PERMANENT_TOKEN_ALIASES for token in alias)


def test_panasonic_token_keys_and_invalid_inputs_follow_the_shared_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1-panasonic-tokens acceptance 101 and 111; v1-vendor-a-tokens acceptance 142 and 160:
    normalize four separators and reject raw invalid inputs.
    """
    from pixtreme._core.validation import _normalized_closed_token

    translation = str.maketrans("", "", " .-_")
    expected = {"V-Gamut": "vgamut", "V-Log": "vlog"}
    assert {token: token.translate(translation).casefold() for token in expected} == expected
    assert len({token.translate(translation).casefold() for token in _COLORSPACES}) == len(_COLORSPACES)
    assert len({token.translate(translation).casefold() for token in _GAMMAS}) == len(_GAMMAS)
    for spelling in ("V Gamut", "V.GAMUT", "v_gamut", "VGamut"):
        assert _normalized_closed_token(spelling, axis="colorspace", accepted=_COLORSPACES) == "V-Gamut"
    for spelling in ("V Log", "V.LOG", "v_log", "VLog"):
        assert _normalized_closed_token(spelling, axis="gamma", accepted=_GAMMAS) == "V-Log"
    rejected = (
        ("Panasonic V-Gamut", "colorspace", _COLORSPACES),
        ("Panasonic V-Log", "gamma", _GAMMAS),
        ("Panasonic V-Gamut/V-Log", "colorspace", _COLORSPACES),
        ("V-Log V-Gamut", "gamma", _GAMMAS),
        ("V-Log L", "gamma", _GAMMAS),
        ("V-Log", "colorspace", _COLORSPACES),
    )
    for value, axis, accepted in rejected:
        with pytest.raises(ValueError):
            _normalized_closed_token(value, axis=axis, accepted=accepted)

    import pixtreme._color.semantics as semantics

    monkeypatch.setattr(semantics, "_color_semantics_kernel", lambda: pytest.fail("GPU work must not start"))
    source = _frame((0.18,))
    for value in ("Panasonic V-Log", 17):
        with pytest.raises(ValueError) as captured:
            px.color.linear_to_gamma(source, gamma=value)
        message = str(captured.value)
        assert message.index("why=") < message.index("what=") < message.index("how=")
        assert f"received gamma={value!r}" in message
        assert repr(_GAMMAS) in message
        assert "Panasonic V-Log" not in message.replace(f"received gamma={value!r}", "")


def test_vlog_encode_dense_grids_selected_points_and_published_anchors_match_independent_oracle() -> None:
    """v1-panasonic-tokens acceptance 102: encode unscaled reflectance with the tangent-derived branches."""
    wide = np.linspace(-0.5, 64.0, 400_001, dtype=np.float64).astype(np.float32)
    fine = np.linspace(0.0, 0.02, 200_001, dtype=np.float64).astype(np.float32)
    for values in (wide, fine):
        actual = _rgb_values(px.color.linear_to_gamma(_frame(values), gamma="V-Log"))[:, 0]
        expected = _vlog_encode(values.astype(np.float64))
        np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-7)

    cut = np.float32(_LINEAR_CUT)
    selected = np.asarray(
        (
            np.nextafter(cut, np.float32(-np.inf)),
            cut,
            np.nextafter(cut, np.float32(np.inf)),
            0.0,
            0.18,
            0.9,
            1.0,
        ),
        dtype=np.float32,
    )
    encoded = px.color.linear_to_gamma(_frame(selected, auxiliary=True), gamma="V-Log")
    actual = _rgb_values(encoded)[:, 0]
    expected = _vlog_encode(selected.astype(np.float64))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-7)
    actual_codes = actual[[3, 4, 5]].astype(np.float64) * np.float64(1023.0)
    np.testing.assert_allclose(actual_codes, (127.87457373, 433.04761208, 601.69528923), rtol=0.0, atol=2.1e-4)
    np.testing.assert_array_equal(np.rint(actual_codes).astype(np.int64), (128, 433, 602))
    assert encoded.gamma == "V-Log"


def test_vlog_decode_dense_grid_cut_branches_and_anchors_match_independent_oracle() -> None:
    """v1-panasonic-tokens acceptance 103: decode at the derived threshold with the inverse log on equality."""
    cut = np.float32(_ENCODED_CUT)
    anchor_codes = _vlog_encode(np.asarray((0.0, 0.18, 0.9), dtype=np.float64)).astype(np.float32)
    special = np.asarray(
        (
            np.nextafter(cut, np.float32(-np.inf)),
            cut,
            np.nextafter(cut, np.float32(np.inf)),
            *anchor_codes,
            0.0,
            1.0,
            1.5,
        ),
        dtype=np.float32,
    )
    values = np.unique(np.concatenate((np.linspace(-0.5, 1.5, 200_001, dtype=np.float64).astype(np.float32), special)))
    decoded = px.color.gamma_to_linear(_frame(values, gamma="V-Log", auxiliary=True), gamma="V-Log")
    actual = _rgb_values(decoded)[:, 0]
    expected = _vlog_decode(values.astype(np.float64))
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
    for encoded, approximate in ((0.0, -0.0223213121), (1.0, 46.08552796)):
        index = int(np.flatnonzero(values == np.float32(encoded))[0])
        assert actual[index] == pytest.approx(approximate, rel=2e-6, abs=2e-7)
    one_index = int(np.flatnonzero(values == np.float32(1.0))[0])
    np.testing.assert_array_max_ulp(
        actual[one_index : one_index + 1],
        np.asarray((expected[one_index],), dtype=np.float32),
        maxulp=16,
    )
    assert decoded.gamma == "linear"


def test_vlog_cut_is_c1_monotone_and_regression_mutants_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """v1-panasonic-tokens acceptance 104: preserve C1 monotonic cuts and reject printed or inclusive mutants."""
    linear_value = _M * _LINEAR_CUT + _D
    log_value = _A * np.log10(_LINEAR_CUT + _B) + _C
    log_slope = _A / ((_LINEAR_CUT + _B) * np.log(np.float64(10.0)))
    assert linear_value == pytest.approx(log_value, rel=0.0, abs=2e-16)
    assert _M == pytest.approx(log_slope, rel=0.0, abs=2e-15)
    oracle_grid = np.linspace(-0.5, 64.0, 500_001, dtype=np.float64)
    assert np.all(np.diff(_vlog_encode(oracle_grid)) > 0.0)

    fine = np.linspace(0.0, 0.02, 200_001, dtype=np.float64).astype(np.float32)
    near = _adjacent_float32(np.float32(_LINEAR_CUT), 3_000)
    for values in (fine, near):
        actual = _rgb_values(px.color.linear_to_gamma(_frame(values), gamma="V-Log"))[:, 0]
        differences = np.diff(actual)
        crossing = (values[:-1] < np.float32(_LINEAR_CUT)) & (values[1:] >= np.float32(_LINEAR_CUT))
        assert np.all(differences[~crossing] >= 0.0)
        if np.any(differences[crossing] < 0.0):
            before = actual[:-1][crossing][0]
            after = actual[1:][crossing][0]
            assert int(before.view(np.uint32)) - int(after.view(np.uint32)) <= 1

    decode_near = _adjacent_float32(np.float32(_ENCODED_CUT), 3_000)
    decoded = _rgb_values(px.color.gamma_to_linear(_frame(decode_near, gamma="V-Log"), gamma="V-Log"))[:, 0]
    assert np.all(np.diff(decoded) >= 0.0)

    import pixtreme._color.semantics as semantics

    printed_source = semantics._COLOR_SEMANTICS_KERNEL.replace(
        "const float m = 5.60001054470806f;\n        const float d = 0.124999583317922f;",
        "const float m = 5.6f;\n        const float d = 0.125f;",
    )
    assert printed_source != semantics._COLOR_SEMANTICS_KERNEL
    printed_kernel = cp.RawKernel(printed_source, "color_semantics_kernel")
    monkeypatch.setattr(semantics, "_color_semantics_kernel", lambda: printed_kernel)
    printed = _rgb_values(px.color.linear_to_gamma(_frame(near), gamma="V-Log"))[:, 0]
    center = 3_000
    host_printed_below = np.float32(np.float32(5.6) * near[:center] + np.float32(0.125))
    host_log_cut = np.float32(
        np.float32(0.241514) * np.log10(np.float32(np.float32(0.01) + np.float32(0.00873))) + np.float32(0.598206)
    )
    host_overshoot = host_printed_below > host_log_cut
    assert np.count_nonzero(host_overshoot) == 56
    assert int(host_printed_below[host_overshoot].view(np.uint32).max()) - int(host_log_cut.view(np.uint32)) == 20
    gpu_overshoot = printed[:center] > printed[center]
    assert np.count_nonzero(gpu_overshoot) >= 56
    assert int(printed[:center][gpu_overshoot].view(np.uint32).max()) - int(printed[center].view(np.uint32)) > 1

    branch_values = np.asarray(
        (
            np.nextafter(np.float32(_ENCODED_CUT), np.float32(-np.inf)),
            np.float32(_ENCODED_CUT),
            np.nextafter(np.float32(_ENCODED_CUT), np.float32(np.inf)),
        ),
        dtype=np.float32,
    )
    monkeypatch.undo()
    correct = _rgb_values(px.color.gamma_to_linear(_frame(branch_values, gamma="V-Log"), gamma="V-Log"))[:, 0]
    comparator_source = semantics._COLOR_SEMANTICS_KERNEL.replace(
        "return value < encoded_cut\n            ? (value - d) / m",
        "return value <= encoded_cut\n            ? (value - d) / m",
    )
    assert comparator_source != semantics._COLOR_SEMANTICS_KERNEL
    comparator_kernel = cp.RawKernel(comparator_source, "color_semantics_kernel")
    monkeypatch.setattr(semantics, "_color_semantics_kernel", lambda: comparator_kernel)
    inclusive_mutant = _rgb_values(px.color.gamma_to_linear(_frame(branch_values, gamma="V-Log"), gamma="V-Log"))[:, 0]
    assert int(correct[1].view(np.uint32)) != int(inclusive_mutant[1].view(np.uint32))


def test_vlog_round_trips_standalone_fused_metadata_and_no_clip_contracts() -> None:
    """v1-panasonic-tokens acceptance 105: round-trip both directions and preserve Frame contracts."""
    linear_values = np.asarray((-2.0, -0.5, 0.0, 0.01, 0.18, 0.9, 1.5, 64.0), dtype=np.float32)
    linear = _frame(linear_values, auxiliary=True)
    before = linear.data.copy()
    standalone_encoded = px.color.linear_to_gamma(linear, gamma="V-Log")
    fused_encoded = px.color.rgb_to_rgb(linear, output_gamma="V-Log")
    np.testing.assert_array_equal(standalone_encoded.data.get(), fused_encoded.data.get())
    restored = px.color.gamma_to_linear(standalone_encoded, gamma="V-Log")
    np.testing.assert_allclose(_rgb_values(restored)[:, 0], linear_values, rtol=2e-6, atol=2e-7)

    encoded_values = np.asarray((-0.75, 0.0, np.float32(_ENCODED_CUT), 0.5, 1.0, 1.5, 1.6), dtype=np.float32)
    encoded = _frame(encoded_values, gamma="V-Log", auxiliary=True)
    standalone_decoded = px.color.gamma_to_linear(encoded, gamma="V-Log")
    fused_decoded = px.color.rgb_to_rgb(encoded, output_gamma="linear")
    np.testing.assert_array_equal(standalone_decoded.data.get(), fused_decoded.data.get())
    reencoded = px.color.linear_to_gamma(standalone_decoded, gamma="V-Log")
    np.testing.assert_allclose(_rgb_values(reencoded)[:, 0], encoded_values, rtol=2e-6, atol=2e-7)
    assert cp.array_equal(standalone_encoded.data[..., 0], linear.data[..., 0])
    assert cp.array_equal(standalone_decoded.data[..., 0], encoded.data[..., 0])
    assert cp.array_equal(linear.data, before)
    assert standalone_encoded is not linear and standalone_encoded.data is not linear.data
    assert (standalone_encoded.gamma, standalone_encoded.matrix, standalone_encoded.data.dtype) == (
        "V-Log",
        None,
        cp.float32,
    )
    assert _rgb_values(standalone_encoded)[0, 0] < 0.0
    assert _rgb_values(standalone_encoded)[-1, 0] > 1.0


def test_v_gamut_matrix_conversions_native_row_and_printed_matrix_match_independent_oracles() -> None:
    """v1-panasonic-tokens acceptance 106: derive V-Gamut matrices and native luma from published coordinates."""
    from pixtreme._core.colorspace import _COLORSPACE_DEFINITIONS

    assert _COLORSPACE_DEFINITIONS["V-Gamut"] == _V_GAMUT
    derived = _rgb_to_xyz(*_V_GAMUT)
    np.testing.assert_allclose(derived, _V_GAMUT_RGB_TO_XYZ, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(derived, _PANASONIC_RGB_TO_XYZ, rtol=0.0, atol=5e-7)
    values = np.asarray(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (-0.25, 0.18, 1.5)),
        dtype=np.float64,
    )
    source = _frame(values, colorspace="V-Gamut")
    converted = px.color.rgb_to_rgb(source, output_colorspace="Rec.709")
    np.testing.assert_allclose(_rgb_values(converted), values @ _conversion(_V_GAMUT, _REC709).T, rtol=0.0, atol=6e-6)
    grayscale = px.color.rgb_to_grayscale(source, colorspace="V-Gamut", gamma="linear", matrix="native")
    np.testing.assert_allclose(
        px.io.to_array(grayscale).get()[0, :, 0],
        values @ _V_GAMUT_RGB_TO_XYZ[1],
        rtol=0.0,
        atol=6e-6,
    )
    assert np.any(_rgb_values(converted) < 0.0) and np.any(_rgb_values(converted) > 1.0)


def test_v_gamut_uses_d65_identity_and_bradford_for_aces_auxiliary_matrix() -> None:
    """v1-panasonic-tokens acceptance 107: use D65 identity and Bradford while keeping matrices auxiliary."""
    from pixtreme._color.transform import _compose_matrix

    to_rec709 = _conversion(_V_GAMUT, _REC709)
    np.testing.assert_allclose(to_rec709, _PANASONIC_TO_BT709, rtol=0.0, atol=5e-7)
    np.testing.assert_allclose(_compose_matrix("V-Gamut", "Rec.709"), to_rec709, rtol=0.0, atol=6e-6)
    to_aces = _conversion(_V_GAMUT, _ACES2065)
    np.testing.assert_allclose(to_aces, _BRADFORD_TO_ACES, rtol=0.0, atol=5e-13)
    np.testing.assert_allclose(_compose_matrix("V-Gamut", "ACES2065-1"), to_aces, rtol=0.0, atol=6e-6)

    source = _frame((0.18,), colorspace="V-Gamut", gamma="Canon-Log")
    result = px.color.rgb_to_rgb(source, output_colorspace="Rec.709", output_gamma="Canon-Log")
    assert (result.colorspace, result.gamma) == ("Rec.709", "Canon-Log")


@pytest.mark.parametrize("target", ("Rec.709", "ACES2065-1"))
def test_panasonic_frames_convert_end_to_end_with_independent_transfer_and_gamut_oracle(target: str) -> None:
    """v1-panasonic-tokens acceptance 108: fuse V-Log decode and V-Gamut conversion with auxiliary preservation."""
    linear_rgb = np.asarray(((-0.25, 0.18, 1.5), (0.18, 1.25, -0.05)), dtype=np.float64)
    encoded_rgb = _vlog_encode(linear_rgb).astype(np.float32)
    source = _frame(encoded_rgb, colorspace="V-Gamut", gamma="V-Log", auxiliary=True)
    before = source.data.copy()
    converted = px.color.rgb_to_rgb(source, output_colorspace=target, output_gamma="linear")
    target_definition = _REC709 if target == "Rec.709" else _ACES2065
    expected = _vlog_decode(_rgb_values(source).astype(np.float64)) @ _conversion(_V_GAMUT, target_definition).T
    np.testing.assert_allclose(_rgb_values(converted), expected, rtol=2e-6, atol=8e-6)
    assert cp.array_equal(converted.data[..., 0], source.data[..., 0])
    assert cp.array_equal(source.data, before)
    assert (converted.colorspace, converted.gamma, converted.channels, converted.matrix) == (
        target,
        "linear",
        source.channels,
        None,
    )
    assert np.any(_rgb_values(converted) < 0.0) and np.any(_rgb_values(converted) > 1.0)


def test_existing_token_bits_remain_at_the_pre_panasonic_baseline() -> None:
    """v1-panasonic-tokens acceptance 109: preserve representative existing transfer and gamut output bits."""
    # Provenance: captured from the complete pre-Panasonic commit
    # d571b866161bcfda13b339d1237faa5054770a90. Reproduce from a detached worktree at that SHA with:
    #
    #   UV_PROJECT_ENVIRONMENT=<main-repo>/.venv PYTHONPATH=<baseline>/src uv run --no-sync python - <<'PY'
    #   import cupy as cp
    #   import numpy as np
    #   import pixtreme as px
    #   linear = np.asarray((-0.25, -0.018056996166706085, 0.0, 0.18000000715255737, 1.0, 1.5), np.float32)
    #   rgb = np.repeat(linear[:, None], 3, axis=1)[None]
    #   base = px.io.from_array(cp.asarray(rgb), colorspace="ACEScg", gamma="linear", channels="RGB")
    #   for gamma in ("S-Log3", "ARRI-LogC4", "Blackmagic-Film-Gen-5", "DaVinci-Intermediate",
    #                 "RED-Log3G10", "Canon-Log-3", "Cineon", "Gamma-2.4"):
    #       encoded_frame = px.color.linear_to_gamma(base, gamma=gamma)
    #       encoded = px.io.to_array(encoded_frame).get()[0, :, 0]
    #       decoded = px.io.to_array(px.color.gamma_to_linear(encoded_frame, gamma=gamma)).get()[0, :, 0]
    #       print("ENCODE", gamma, tuple(int(value) for value in encoded.view(np.uint32)))
    #       print("DECODE", gamma, tuple(int(value) for value in decoded.view(np.uint32)))
    #   pixels = np.asarray(((-0.25, 0.18, 1.5), (0.02, -0.1, 1.0)), np.float32)[None]
    #   for colorspace in ("Rec.2020", "ACEScg", "S-Gamut3.Cine", "ARRI-Wide-Gamut-4",
    #                      "Blackmagic-Wide-Gamut-Gen-5", "DaVinci-Wide-Gamut", "REDWideGamutRGB",
    #                      "Canon-Cinema-Gamut"):
    #       frame = px.io.from_array(cp.asarray(pixels), colorspace=colorspace, gamma="linear", channels="RGB")
    #       out = px.io.to_array(px.color.rgb_to_rgb(
    #           frame, output_colorspace="Rec.709", output_gamma="linear"
    #       )).get().view(np.uint32).reshape(-1)
    #       print("GAMUT", colorspace, tuple(int(value) for value in out))
    #   PY
    gamma_fixtures = {
        "S-Log3": (
            (3217556478, 3168455511, 1035874188, 1053963405, 1058575679, 1059324708),
            (3196059648, 3163810884, 789333745, 1043878379, 1065353213, 1069547519),
        ),
        "ARRI-LogC4": (
            (3221400801, 2995575096, 1035874188, 1049528807, 1054532562, 1055775089),
            (3196059648, 3163810884, 0, 1043878379, 1065353214, 1069547520),
        ),
        "Blackmagic-Film-Gen-5": (
            (3221044577, 3177835909, 1035820719, 1053057585, 1057476139, 1058064820),
            (3196059648, 3163810884, 0, 1043878381, 1065353217, 1069547524),
        ),
        "DaVinci-Intermediate": (
            (3223788473, 3191938634, 0, 1051463133, 1057196762, 1057911650),
            (3196059648, 3163810884, 0, 1043878378, 1065353219, 1069547523),
        ),
        "RED-Log3G10": (
            (3228130337, 3187323085, 1035698008, 1051372189, 1056744777, 1057508475),
            (3196059648, 3163810884, 0, 1043878381, 1065353218, 1069547523),
        ),
        "Canon-Log-3": (
            (3188272132, 1034941615, 1040195592, 1051709627, 1058311446, 1059345448),
            (3196059649, 3163810884, 0, 1043878378, 1065353215, 1069547524),
        ),
        "Cineon": (
            (3204351021, 3193857572, 1035874182, 1055532491, 1059810011, 1060668676),
            (3196059643, 3163810880, 2973898886, 1043878376, 1065353217, 1069547520),
        ),
        "Gamma-2.4": (
            (3205475542, 3191882770, 0, 1056610176, 1065353216, 1066897169),
            (3196059647, 3163810883, 0, 1043878379, 1065353216, 1069547519),
        ),
    }
    linear_values = (-0.25, -0.018056996166706085, 0.0, 0.18000000715255737, 1.0, 1.5)
    for gamma, (encode_bits, decode_bits) in gamma_fixtures.items():
        encoded = px.color.linear_to_gamma(_frame(linear_values), gamma=gamma)
        np.testing.assert_array_equal(_rgb_values(encoded)[:, 0].view(np.uint32), encode_bits)
        decoded = px.color.gamma_to_linear(encoded, gamma=gamma)
        np.testing.assert_array_equal(_rgb_values(decoded)[:, 0].view(np.uint32), decode_bits)
        assert (encoded.gamma, decoded.gamma) == (gamma, "linear")

    gamut_fixtures = {
        "Rec.2020": (3206632196, 1046732887, 1070927681, 1016900116, 3187554328, 1066430520),
        "ACEScg": (3207184178, 1046702756, 1071327960, 1012225928, 3187820956, 1066740601),
        "S-Gamut3.Cine": (3206699341, 3178418026, 1072369260, 3112583168, 3200582444, 1067526363),
        "ARRI-Wide-Gamut-4": (3209182827, 1035601944, 1071419179, 997395808, 3196821229, 1066861949),
        "Blackmagic-Wide-Gamut-Gen-5": (
            3205367535,
            3187568618,
            1073070301,
            1025198451,
            3200919674,
            1068080043,
        ),
        "DaVinci-Wide-Gamut": (3209093926, 3190676685, 1074284506, 1009788360, 3203518916, 1069265258),
        "REDWideGamutRGB": (3209143998, 3190614948, 1075355334, 1027943654, 3203640764, 1071129191),
        "Canon-Cinema-Gamut": (3209689344, 3186602936, 1074144546, 3151829184, 3202611611, 1069424635),
    }
    pixels = np.asarray(((-0.25, 0.18, 1.5), (0.02, -0.1, 1.0)), dtype=np.float32)
    for colorspace, expected_bits in gamut_fixtures.items():
        converted = px.color.rgb_to_rgb(
            _frame(pixels, colorspace=colorspace), output_colorspace="Rec.709", output_gamma="linear"
        )
        np.testing.assert_array_equal(
            px.io.to_array(converted).get().view(np.uint32).reshape(-1),
            np.asarray(expected_bits, dtype=np.uint32),
        )
        assert (converted.colorspace, converted.gamma) == ("Rec.709", "linear")


def test_vlog_dpx_transfer_code_is_logarithmic_and_existing_mappings_remain_unchanged(tmp_path: Path) -> None:
    """v1-panasonic-tokens acceptance 110: classify V-Log as DPX logarithmic without disturbing prior mappings."""
    from pixtreme._io.formats.dpx import _dpx_transfer_from_gamma

    expected = {
        "V-Log": 3,
        "S-Log3": 3,
        "ARRI-LogC4": 3,
        "Blackmagic-Film-Gen-5": 3,
        "DaVinci-Intermediate": 3,
        "RED-Log3G10": 3,
        "Canon-Log-3": 3,
        "Cineon": 1,
        "REDlogFilm": 1,
        "linear": 2,
        "Gamma-2.4": 6,
    }
    assert {gamma: _dpx_transfer_from_gamma(gamma) for gamma in expected} == expected
    pixels = cp.asarray([[[0.18, 0.18, 0.18]]], dtype=cp.float32)
    for gamma, transfer in expected.items():
        frame = px.io.from_array(pixels.copy(), colorspace="Rec.709", gamma=gamma, channels="RGB")
        path = tmp_path / f"{gamma}.dpx"
        px.io.write_image(path, frame)
        assert path.read_bytes()[801] == transfer
        if gamma == "V-Log":
            assert px.io.read_image(path).gamma == "Cineon"


def test_panasonic_token_reference_requirements_changelog_and_public_docstrings_are_synchronized() -> None:
    """v1-panasonic-tokens acceptance 112; v1-vendor-a-tokens acceptance 161;
    v1-vendor-b-tokens acceptance 188:
    synchronize vocabulary, numeric identity, boundaries, and public prose.
    """
    token_reference = (ROOT / "docs_site" / "tokens.md").read_text(encoding="utf-8")
    requirements = require_repo_file("docs/requirements.md").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    latest_section = latest_changelog_section(changelog)
    for token in ("V-Gamut", "V-Log"):
        assert f"`{token}`" in token_reference
        assert token in latest_section
    for fragment in (
        "5.60001054470806",
        "0.124999583317922",
        "0.180999688765003",
        "0.730",
        "0.840",
        "-0.030",
        "D65",
        "Bradford",
        "OpenColorIO",
        "V-Log L",
        "Panasonic V-Gamut",
        "native",
    ):
        assert fragment in token_reference
    assert "27 Colorspace" in requirements
    assert "33 Gamma" in requirements
    assert "188 canonical tokens" in requirements
    for fragment in ("OpenColorIO", "vendor IDT", "ACES CSC", "V-Log L", "bit", "Panasonic V-Gamut"):
        assert fragment in latest_section
    for operation in (px.color.rgb_to_rgb, px.color.gamma_to_linear, px.color.linear_to_gamma):
        docstring = inspect.getdoc(operation)
        assert docstring is not None
        assert "V-Gamut" in docstring or operation is not px.color.rgb_to_rgb
        assert "V-Log" in docstring
