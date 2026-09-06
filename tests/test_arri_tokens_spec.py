"""Specification tests for ARRI Wide Gamut 3/4 and ARRI-LogC3 tokens."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Literal, get_args, get_origin, get_type_hints

import cupy as cp
import numpy as np
import pytest
from repository_contracts import require_repo_file

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
_SEPARATORS = " .-_"
_TRANSLATION = str.maketrans("", "", _SEPARATORS)

_LOGC3_CUT = np.float64("0.0105909904954696")
_LOGC3_A = np.float64("5.55555555555556")
_LOGC3_B = np.float64("0.0522722750251688")
_LOGC3_C = np.float64("0.247189638318671")
_LOGC3_D = np.float64("0.385536998692443")
_LOGC3_E = _LOGC3_C * _LOGC3_A / ((_LOGC3_A * _LOGC3_CUT + _LOGC3_B) * np.log(np.float64(10.0)))
_LOGC3_F = _LOGC3_C * np.log10(_LOGC3_A * _LOGC3_CUT + _LOGC3_B) + _LOGC3_D - _LOGC3_E * _LOGC3_CUT
_LOGC3_CODE_CUT = _LOGC3_E * _LOGC3_CUT + _LOGC3_F

_PRINTED_LOGC3 = {
    "cut": np.float64("0.010591"),
    "a": np.float64("5.555556"),
    "b": np.float64("0.052272"),
    "c": np.float64("0.247190"),
    "d": np.float64("0.385537"),
    "e": np.float64("5.367655"),
    "f": np.float64("0.092809"),
}

_AWG_DEFINITIONS = {
    "ARRI-Wide-Gamut-3": (((0.6840, 0.3130), (0.2210, 0.8480), (0.0861, -0.1020)), (0.3127, 0.3290)),
    "ARRI-Wide-Gamut-4": (((0.7347, 0.2653), (0.1424, 0.8576), (0.0991, -0.0308)), (0.3127, 0.3290)),
}
_REC709_DEFINITION = (((0.640, 0.330), (0.300, 0.600), (0.150, 0.060)), (0.3127, 0.3290))
_ACESCG_DEFINITION = (((0.713, 0.293), (0.165, 0.830), (0.128, 0.044)), (0.32168, 0.33767))
_BRADFORD = np.asarray(
    (
        (0.8951, 0.2664, -0.1614),
        (-0.7502, 1.7135, 0.0367),
        (0.0389, -0.0685, 1.0296),
    ),
    dtype=np.float64,
)


def _piecewise(
    values: np.ndarray,
    cut: float,
    lower: Callable[[np.ndarray], np.ndarray],
    upper: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    lower_mask = values <= cut
    result[lower_mask] = lower(values[lower_mask])
    result[~lower_mask] = upper(values[~lower_mask])
    return result


def _logc3_encode(values: np.ndarray) -> np.ndarray:
    return _piecewise(
        values,
        _LOGC3_CUT,
        lambda part: _LOGC3_E * part + _LOGC3_F,
        lambda part: _LOGC3_C * np.log10(_LOGC3_A * part + _LOGC3_B) + _LOGC3_D,
    )


def _logc3_decode(values: np.ndarray) -> np.ndarray:
    return _piecewise(
        values,
        _LOGC3_CODE_CUT,
        lambda part: (part - _LOGC3_F) / _LOGC3_E,
        lambda part: (np.float64(10.0) ** ((part - _LOGC3_D) / _LOGC3_C) - _LOGC3_B) / _LOGC3_A,
    )


def _printed_logc3_encode(values: np.ndarray) -> np.ndarray:
    return _piecewise(
        values,
        _PRINTED_LOGC3["cut"],
        lambda part: _PRINTED_LOGC3["e"] * part + _PRINTED_LOGC3["f"],
        lambda part: (
            _PRINTED_LOGC3["c"] * np.log10(_PRINTED_LOGC3["a"] * part + _PRINTED_LOGC3["b"]) + _PRINTED_LOGC3["d"]
        ),
    )


def _printed_logc3_decode(values: np.ndarray) -> np.ndarray:
    boundary = _PRINTED_LOGC3["e"] * _PRINTED_LOGC3["cut"] + _PRINTED_LOGC3["f"]
    return _piecewise(
        values,
        boundary,
        lambda part: (part - _PRINTED_LOGC3["f"]) / _PRINTED_LOGC3["e"],
        lambda part: (
            (np.float64(10.0) ** ((part - _PRINTED_LOGC3["d"]) / _PRINTED_LOGC3["c"]) - _PRINTED_LOGC3["b"])
            / _PRINTED_LOGC3["a"]
        ),
    )


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
        auxiliary_values = np.arange(array.shape[0], dtype=np.float32)[:, None] + np.float32(16.0)
        array = np.concatenate((auxiliary_values, array[:, (2, 0, 1)]), axis=1)
        channels = ("Z", "B", "R", "G")
    return px.io.from_array(
        cp.asarray(array[None, :, :]),
        colorspace=colorspace,
        gamma=gamma,
        channels=channels,
        matrix="native",
    )


def _red_values(frame: px.core.Frame) -> np.ndarray:
    return px.io.to_array(frame).get()[0, :, frame.channels.index("R")]


def _literal_strings(annotation: object) -> tuple[str, ...]:
    if get_origin(annotation) is Literal:
        return tuple(value for value in get_args(annotation) if isinstance(value, str))
    return tuple(value for argument in get_args(annotation) for value in _literal_strings(argument))


def _token_key(value: str) -> str:
    return value.translate(_TRANSLATION).casefold()


def _variants(value: str) -> tuple[str, ...]:
    compact = value.translate(_TRANSLATION)
    replaced = tuple(
        "".join(separator if character in _SEPARATORS else character for character in value)
        for separator in _SEPARATORS
    )
    return tuple(dict.fromkeys((value, value.swapcase(), compact, *replaced)))


def _xy_to_xyz(xy: tuple[float, float]) -> np.ndarray:
    x, y = xy
    return np.asarray((x / y, 1.0, (1.0 - x - y) / y), dtype=np.float64)


def _rgb_to_xyz(
    primaries: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    white: tuple[float, float],
) -> np.ndarray:
    primary_matrix = np.asarray(
        (
            tuple(x / y for x, y in primaries),
            (1.0, 1.0, 1.0),
            tuple((1.0 - x - y) / y for x, y in primaries),
        ),
        dtype=np.float64,
    )
    scales = np.linalg.solve(primary_matrix, _xy_to_xyz(white))
    return primary_matrix @ np.diag(scales)


def _bradford(input_white: tuple[float, float], output_white: tuple[float, float]) -> np.ndarray:
    if input_white == output_white:
        return np.eye(3, dtype=np.float64)
    input_cones = _BRADFORD @ _xy_to_xyz(input_white)
    output_cones = _BRADFORD @ _xy_to_xyz(output_white)
    return np.linalg.inv(_BRADFORD) @ np.diag(output_cones / input_cones) @ _BRADFORD


def _conversion_matrix(
    source: tuple[
        tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
        tuple[float, float],
    ],
    target: tuple[
        tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
        tuple[float, float],
    ],
) -> np.ndarray:
    source_primaries, source_white = source
    target_primaries, target_white = target
    return (
        np.linalg.inv(_rgb_to_xyz(target_primaries, target_white))
        @ _bradford(source_white, target_white)
        @ _rgb_to_xyz(source_primaries, source_white)
    )


def test_arri_tokens_extend_canonical_vocabulary_and_public_static_surfaces() -> None:
    """v1-arri-tokens acceptance 16-17; v1-blackmagic-tokens acceptance 33-34;
    v1-red-tokens acceptance 54-55; v1-canon-tokens acceptance 76-77;
    v1-panasonic-tokens acceptance 99-100; v1-standard-tokens acceptance 117;
    v1-vendor-a-tokens acceptance 140-141; v1-vendor-b-tokens acceptance 166-167: canonical public surfaces.
    """
    assert get_args(px.core.Colorspace) == _COLORSPACES
    assert get_args(px.core.Gamma) == _GAMMAS
    assert len(_ALIASES) == 30
    assert sum(len(get_args(alias)) for alias in _ALIASES) == 188
    assert _literal_strings(get_type_hints(px.color.linear_to_gamma)["gamma"]) == _GAMMAS
    assert _literal_strings(get_type_hints(px.color.rgb_to_rgb)["input_colorspace"]) == _COLORSPACES
    assert _literal_strings(get_type_hints(px.color.rgb_to_rgb)["output_gamma"]) == _GAMMAS

    frame = _frame((0.18,), colorspace="ARRI-Wide-Gamut-3", gamma="ARRI-LogC3")
    assert (frame.colorspace, frame.gamma) == ("ARRI-Wide-Gamut-3", "ARRI-LogC3")
    assert "colorspace='ARRI-Wide-Gamut-3'" in repr(frame)
    assert "gamma='ARRI-LogC3'" in repr(frame)


def test_arri_token_keys_are_collision_free_family_local_and_separator_normalized() -> None:
    """v1-arri-tokens acceptance 18; v1-blackmagic-tokens acceptance 35; v1-red-tokens acceptance 68;
    v1-vendor-a-tokens acceptance 142.

    Token keys remain local and unique after the ARRI rename.
    """
    from pixtreme._core.validation import _normalized_closed_token
    from pixtreme._core.vocabulary import _PERMANENT_TOKEN_ALIASES

    assert len({_token_key(token) for token in _COLORSPACES}) == len(_COLORSPACES)
    assert len({_token_key(token) for token in _GAMMAS}) == len(_GAMMAS)
    assert all(
        canonical not in {"ARRI-Wide-Gamut-3", "ARRI-Wide-Gamut-4", "ARRI-LogC3"}
        for _alias, canonical in _PERMANENT_TOKEN_ALIASES
    )

    for canonical, family, axis in (
        ("ARRI-Wide-Gamut-3", _COLORSPACES, "colorspace"),
        ("ARRI-Wide-Gamut-4", _COLORSPACES, "colorspace"),
        ("ARRI-LogC3", _GAMMAS, "gamma"),
    ):
        for variant in _variants(canonical):
            assert _normalized_closed_token(variant, axis=axis, accepted=family) == canonical

    assert {_token_key(token) for token in ("ARRI-Wide-Gamut-3", "ARRI-Wide-Gamut-4")} == {
        "arriwidegamut3",
        "arriwidegamut4",
    }
    assert {_token_key(token) for token in ("ARRI-LogC3", "ARRI-LogC4")} == {"arrilogc3", "arrilogc4"}
    with pytest.raises(ValueError):
        _normalized_closed_token("ARRI-LogC3", axis="colorspace", accepted=_COLORSPACES)
    with pytest.raises(ValueError):
        _normalized_closed_token("ARRI-Wide-Gamut-3", axis="gamma", accepted=_GAMMAS)
    with pytest.raises(ValueError):
        _normalized_closed_token("AWG3", axis="colorspace", accepted=_COLORSPACES)


def test_logc3_encode_matches_the_normative_oracle_and_is_continuous_monotonic_and_unclipped() -> None:
    """v1-arri-tokens acceptance 19 and 21; v1-red-tokens acceptance 68: retain ARRI-LogC3 encode bits."""
    cut = np.float32(_LOGC3_CUT)
    values = np.asarray(
        (
            -0.25,
            np.nextafter(cut, np.float32(-np.inf)),
            cut,
            np.nextafter(cut, np.float32(np.inf)),
            0.0,
            0.18,
            1.0,
            1.5,
        ),
        dtype=np.float32,
    )
    actual = _red_values(px.color.linear_to_gamma(_frame(values), gamma="ARRI-LogC3"))
    expected = _logc3_encode(values.astype(np.float64))

    # This tolerance measures float32 GPU evaluation against the normative float64 oracle. The branch-cut and
    # 18% guards below independently reject replacing the normative coefficients with ARRI's six-decimal table.
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-6)
    assert actual[0] < actual[4]
    assert actual[-1] > actual[-2]
    assert np.isclose(_LOGC3_E * _LOGC3_CUT + _LOGC3_F, _LOGC3_CODE_CUT, rtol=0.0, atol=1e-15)
    assert np.isclose(
        _LOGC3_C * np.log10(_LOGC3_A * _LOGC3_CUT + _LOGC3_B) + _LOGC3_D,
        _LOGC3_CODE_CUT,
        rtol=0.0,
        atol=1e-15,
    )

    grid = np.linspace(_LOGC3_CUT - 2e-4, _LOGC3_CUT + 2e-4, 4097, dtype=np.float32)
    grid_encoded = _red_values(px.color.linear_to_gamma(_frame(grid), gamma="ARRI-LogC3"))
    assert np.all(np.diff(grid_encoded) >= np.float32(0.0))

    printed_cut = np.float32(_PRINTED_LOGC3["cut"])
    printed_cut_neighbors = np.asarray(
        (
            np.nextafter(printed_cut, np.float32(-np.inf)),
            printed_cut,
            np.nextafter(printed_cut, np.float32(np.inf)),
        ),
        dtype=np.float32,
    )
    gpu_neighbors = _red_values(px.color.linear_to_gamma(_frame(printed_cut_neighbors), gamma="ARRI-LogC3"))
    assert np.all(np.diff(gpu_neighbors) >= np.float32(0.0))
    # The verbatim printed table switches branches at its rounded cut and decreases across these same neighbors.
    printed_neighbors = _printed_logc3_encode(printed_cut_neighbors.astype(np.float64))
    assert not np.all(np.diff(printed_neighbors) >= np.float64(0.0))

    gray = actual[5]
    gray_bits = np.asarray(gray, dtype=np.float32).view(np.uint32).item()
    anchor_bits = np.asarray(np.float32(400.0 / 1023.0)).view(np.uint32).item()
    printed_gray_bits = (
        np.asarray(np.float32(_printed_logc3_encode(np.asarray((0.18,), dtype=np.float64))[0])).view(np.uint32).item()
    )
    assert gray_bits == anchor_bits
    assert printed_gray_bits != anchor_bits
    assert int(np.floor(np.float64(1023.0) * np.float64(gray) + np.float64(0.5))) == 400


def test_logc3_decode_matches_the_normative_oracle_through_the_boundary_without_clipping() -> None:
    """v1-arri-tokens acceptance 20-21; v1-red-tokens acceptance 68: retain the ARRI-LogC3 inverse."""
    boundary = np.float32(_LOGC3_CODE_CUT)
    encoded_fixtures = _logc3_encode(np.asarray((-0.25, 0.0, 0.18, 1.0, 1.5), dtype=np.float64)).astype(np.float32)
    values = np.concatenate(
        (
            np.asarray((-0.25,), dtype=np.float32),
            encoded_fixtures,
            np.asarray(
                (
                    np.nextafter(boundary, np.float32(-np.inf)),
                    boundary,
                    np.nextafter(boundary, np.float32(np.inf)),
                ),
                dtype=np.float32,
            ),
        )
    )
    actual = _red_values(px.color.gamma_to_linear(_frame(values, gamma="ARRI-LogC3"), gamma="ARRI-LogC3"))
    expected = _logc3_decode(values.astype(np.float64))

    # This tolerance measures float32 inverse evaluation against the normative float64 oracle; the encode-side
    # branch-cut and anchor guards exclude the rounded coefficient set that this broad numeric tolerance permits.
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=5e-6)
    assert actual[0] < 0.0
    assert actual[5] > 1.0


def test_logc3_high_precision_coefficients_round_to_arri_values_and_bound_printed_oracle_error() -> None:
    """v1-arri-tokens acceptance 21: normative coefficients round to ARRI's table and bound its print error."""
    normative = {
        "cut": _LOGC3_CUT,
        "a": _LOGC3_A,
        "b": _LOGC3_B,
        "c": _LOGC3_C,
        "d": _LOGC3_D,
        "e": _LOGC3_E,
        "f": _LOGC3_F,
    }
    assert {name: f"{value:.6f}" for name, value in normative.items()} == {
        name: f"{value:.6f}" for name, value in _PRINTED_LOGC3.items()
    }

    scene = np.asarray((-0.25, _LOGC3_CUT - 1e-7, _LOGC3_CUT, _LOGC3_CUT + 1e-7, 0.0, 0.18, 1.0, 1.5))
    encoded = _logc3_encode(scene)
    # The largest observed coefficient-rounding delta is 6.05e-7 immediately above the branch cut.
    np.testing.assert_allclose(_printed_logc3_encode(scene), encoded, rtol=0.0, atol=7e-7)
    # The 6-decimal coefficient table differs by at most 4.79e-6 over the fixed scene fixture.
    np.testing.assert_allclose(_printed_logc3_decode(encoded), _logc3_decode(encoded), rtol=0.0, atol=5e-6)


def test_logc3_round_trips_and_all_transfer_paths_preserve_the_frame_contract() -> None:
    """v1-arri-tokens acceptance 22; v1-red-tokens acceptance 68: ARRI-LogC3 paths preserve Frame observables."""
    cut = np.float32(_LOGC3_CUT)
    linear = np.asarray(
        (
            -0.25,
            np.nextafter(cut, np.float32(-np.inf)),
            cut,
            np.nextafter(cut, np.float32(np.inf)),
            0.0,
            0.18,
            1.0,
            1.5,
        ),
        dtype=np.float32,
    )
    source = _frame(linear, auxiliary=True)
    before = source.data.copy()

    encoded = px.color.linear_to_gamma(source, gamma="ARRI-LogC3")
    fused_encoded = px.color.rgb_to_rgb(source, output_gamma="ARRI-LogC3")
    restored = px.color.gamma_to_linear(encoded, gamma="ARRI-LogC3")
    fused_restored = px.color.rgb_to_rgb(encoded, output_gamma="linear")
    encoded_values = _logc3_encode(linear.astype(np.float64)).astype(np.float32)
    decoded = px.color.gamma_to_linear(_frame(encoded_values, gamma="ARRI-LogC3"), gamma="ARRI-LogC3")
    reencoded = px.color.linear_to_gamma(decoded, gamma="ARRI-LogC3")

    assert cp.array_equal(encoded.data, fused_encoded.data)
    assert cp.array_equal(restored.data, fused_restored.data)
    assert cp.array_equal(encoded.data[..., 0], source.data[..., 0])
    assert cp.array_equal(restored.data[..., 0], source.data[..., 0])
    assert cp.array_equal(source.data, before)
    np.testing.assert_allclose(_red_values(restored), linear, rtol=0.0, atol=8e-6)
    np.testing.assert_allclose(_red_values(reencoded), encoded_values, rtol=0.0, atol=8e-6)
    assert (encoded.colorspace, encoded.gamma, encoded.channels, encoded.matrix) == (
        "ACEScg",
        "ARRI-LogC3",
        source.channels,
        None,
    )
    assert (restored.colorspace, restored.gamma, restored.channels, restored.matrix) == (
        "ACEScg",
        "linear",
        source.channels,
        None,
    )
    assert encoded is not source and encoded.data.data.ptr != source.data.data.ptr
    assert encoded.data.dtype == restored.data.dtype == cp.float32


@pytest.mark.parametrize(("colorspace", "acceptance"), (("ARRI-Wide-Gamut-3", 23), ("ARRI-Wide-Gamut-4", 24)))
def test_awg_primaries_conversion_and_native_row_match_independent_float64_oracles(
    colorspace: str,
    acceptance: int,
) -> None:
    """v1-arri-tokens acceptance 23-24: both AWG definitions match independent matrix and native-row oracles."""
    from pixtreme._core.colorspace import _COLORSPACE_DEFINITIONS

    assert acceptance in (23, 24)
    definition = _AWG_DEFINITIONS[colorspace]
    assert _COLORSPACE_DEFINITIONS[colorspace] == definition
    awg_matrix = _rgb_to_xyz(*definition)
    conversion = _conversion_matrix(definition, _REC709_DEFINITION)
    values = np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (-0.25, 0.18, 1.5)))
    source = _frame(values, colorspace=colorspace)

    converted = px.color.rgb_to_rgb(source, output_colorspace="Rec.709")
    grayscale = px.color.rgb_to_grayscale(source, colorspace=colorspace, gamma="linear", matrix="native")

    np.testing.assert_allclose(px.io.to_array(converted).get()[0], values @ conversion.T, rtol=0.0, atol=5e-6)
    np.testing.assert_allclose(px.io.to_array(grayscale).get()[0, :, 0], values @ awg_matrix[1], rtol=0.0, atol=5e-6)
    assert grayscale.matrix == "native"


def test_awg_adaptation_and_independent_gamma_pairings_follow_existing_color_contracts() -> None:
    """v1-arri-tokens acceptance 25; v1-red-tokens acceptance 68: AWG and renamed ARRI gamma stay independent."""
    values = np.asarray(((1.0, -0.25, 0.18), (0.2, 0.4, 1.5)), dtype=np.float32)
    source = _frame(values, colorspace="ARRI-Wide-Gamut-3", auxiliary=True)
    before = source.data.copy()
    awg4_matrix = _conversion_matrix(_AWG_DEFINITIONS["ARRI-Wide-Gamut-3"], _AWG_DEFINITIONS["ARRI-Wide-Gamut-4"])
    acescg_matrix = _conversion_matrix(_AWG_DEFINITIONS["ARRI-Wide-Gamut-3"], _ACESCG_DEFINITION)

    awg4 = px.color.rgb_to_rgb(source, output_colorspace="ARRI-Wide-Gamut-4")
    acescg = px.color.rgb_to_rgb(source, output_colorspace="ACEScg")
    independent_pair = px.color.rgb_to_rgb(source, output_colorspace="ARRI-Wide-Gamut-4", output_gamma="ARRI-LogC3")
    unusual_metadata = _frame((0.18,), colorspace="ARRI-Wide-Gamut-3", gamma="ARRI-LogC4")

    source_rgb = px.io.to_array(source).get()[0][:, [source.channels.index(label) for label in ("R", "G", "B")]]
    actual_awg4 = px.io.to_array(awg4).get()[0][:, [awg4.channels.index(label) for label in ("R", "G", "B")]]
    actual_acescg = px.io.to_array(acescg).get()[0][:, [acescg.channels.index(label) for label in ("R", "G", "B")]]
    np.testing.assert_allclose(actual_awg4, source_rgb @ awg4_matrix.T, rtol=0.0, atol=5e-6)
    np.testing.assert_allclose(actual_acescg, source_rgb @ acescg_matrix.T, rtol=0.0, atol=5e-6)
    assert cp.array_equal(awg4.data[..., 0], source.data[..., 0])
    assert cp.array_equal(acescg.data[..., 0], source.data[..., 0])
    assert cp.array_equal(source.data, before)
    assert (awg4.colorspace, awg4.gamma, awg4.channels, awg4.matrix) == (
        "ARRI-Wide-Gamut-4",
        "linear",
        source.channels,
        None,
    )
    assert (independent_pair.colorspace, independent_pair.gamma) == ("ARRI-Wide-Gamut-4", "ARRI-LogC3")
    assert (unusual_metadata.colorspace, unusual_metadata.gamma) == ("ARRI-Wide-Gamut-3", "ARRI-LogC4")
    assert awg4.data.dtype == acescg.data.dtype == cp.float32


# Baseline provenance: these integer fixtures were generated from commit
# 1d4c147afd60a41d195e9f5d7c7b22eda6a0b035 before adding the ARRI-LogC3 device branch. Reproduce with
# `git worktree add <baseline-path> 1d4c147afd60a41d195e9f5d7c7b22eda6a0b035`, then run from this checkout as
# `UV_PROJECT_ENVIRONMENT=<main-repo>/.venv PYTHONPATH=<baseline-path>/src uv run --no-sync python -`. The stdin
# script imports CuPy, NumPy, and pixtreme; creates RGB Frames by repeating each exact float32 `linear` and `encoded`
# array below over three channels; calls ARRI-LogC4 `linear_to_gamma` and `gamma_to_linear`; copies channel R to NumPy;
# and prints each result with `.view(np.uint32).tolist()`.
def test_logc4_gpu_bits_remain_identical_to_the_pre_arri_token_baseline() -> None:
    """v1-arri-tokens acceptance 26; v1-red-tokens acceptance 68: ARRI-LogC4 retains baseline float32 bits."""
    linear = np.asarray(
        (
            -0.25,
            -0.018056998029351234,
            -0.018056996166706085,
            -0.018056994304060936,
            -0.009999999776482582,
            0.0,
            0.18000000715255737,
            1.0,
            1.5,
        ),
        dtype=np.float32,
    )
    encoded = np.asarray(
        (
            -0.5,
            -0.10000000149011612,
            -1.401298464324817e-45,
            0.0,
            1.401298464324817e-45,
            0.09286412596702576,
            0.2783958315849304,
            0.4275193512439728,
            1.0,
        ),
        dtype=np.float32,
    )
    expected_encode = np.asarray(
        (3221400801, 3003963704, 2995575096, 846191396, 1029189057, 1035874188, 1049528807, 1054532562, 1055775089),
        dtype=np.uint32,
    )
    expected_decode = np.asarray(
        (3180940773, 3169909587, 3163810883, 3163810883, 3163810883, 0, 1043878379, 1065353203, 1139467878),
        dtype=np.uint32,
    )

    actual_encode = _red_values(px.color.linear_to_gamma(_frame(linear), gamma="ARRI-LogC4")).view(np.uint32)
    actual_decode = _red_values(px.color.gamma_to_linear(_frame(encoded, gamma="ARRI-LogC4"), gamma="ARRI-LogC4")).view(
        np.uint32
    )
    np.testing.assert_array_equal(actual_encode, expected_encode)
    np.testing.assert_array_equal(actual_decode, expected_decode)


def test_dpx_classifies_logc3_as_logarithmic_without_changing_existing_mappings() -> None:
    """v1-arri-tokens acceptance 27; v1-red-tokens acceptance 72: DPX uses renamed ARRI gamma tokens."""
    from pixtreme._io.formats.dpx import _dpx_transfer_from_gamma

    assert {
        gamma: _dpx_transfer_from_gamma(gamma)
        for gamma in (
            "Cineon",
            "linear",
            "S-Log",
            "S-Log2",
            "S-Log3",
            "ARRI-LogC3",
            "ARRI-LogC4",
            "Rec.709",
            "Gamma-2.6",
        )
    } == {
        "Cineon": 1,
        "linear": 2,
        "S-Log": 3,
        "S-Log2": 3,
        "S-Log3": 3,
        "ARRI-LogC3": 3,
        "ARRI-LogC4": 3,
        "Rec.709": 6,
        "Gamma-2.6": 6,
    }


@pytest.mark.parametrize(
    ("operation", "parameter", "rejected", "candidates"),
    (
        ("linear_to_gamma", "gamma", "unknown", _GAMMAS),
        ("linear_to_gamma", "gamma", 17, _GAMMAS),
        ("rgb_to_rgb", "output_colorspace", "unknown", _COLORSPACES),
        ("rgb_to_rgb", "output_colorspace", 17, _COLORSPACES),
    ),
)
def test_invalid_arri_axis_values_fail_before_gpu_with_ordered_canonical_errors(
    operation: str,
    parameter: str,
    rejected: object,
    candidates: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1-arri-tokens acceptance 28; v1-blackmagic-tokens acceptance 49;
    v1-vendor-a-tokens acceptance 160: invalid values fail before GPU.
    """
    import pixtreme._color.semantics as semantics
    import pixtreme._color.transform as transform

    def unexpected_backend(*_args: object, **_kwargs: object) -> cp.ndarray:
        raise AssertionError("GPU backend was reached")

    monkeypatch.setattr(semantics, "_run_transform", unexpected_backend)
    monkeypatch.setattr(transform, "_transform_data", unexpected_backend)
    source = _frame((0.18,))
    with pytest.raises(ValueError) as error:
        getattr(px.color, operation)(source, **{parameter: rejected})

    message = str(error.value)
    assert message.index("why=") < message.index("what=") < message.index("how=")
    assert repr(rejected) in message
    assert repr(candidates) in message
    assert "'arri wide gamut 3'" not in message.casefold()


def test_arri_public_documents_docstrings_and_changelog_are_synchronized() -> None:
    """v1-arri-tokens acceptance 29; v1-blackmagic-tokens acceptance 50; v1-red-tokens acceptance 72;
    v1-canon-tokens acceptance 93; v1-panasonic-tokens acceptance 112; v1-vendor-a-tokens acceptance 161;
    v1-vendor-b-tokens acceptance 188.

    GitHub #29: public token documentation stays synchronized after the ARRI rename.
    """
    tokens = (ROOT / "docs_site" / "tokens.md").read_text(encoding="utf-8")
    requirements = require_repo_file("docs/requirements.md").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    normalized_docstrings = {
        function.__name__: " ".join((inspect.getdoc(function) or "").split())
        for function in (px.color.gamma_to_linear, px.color.linear_to_gamma, px.color.rgb_to_rgb)
    }

    for claim in (
        "EI 800",
        "`400 / 1023`",
        "relative scene exposure",
        "lower linear branch",
        "without clipping or sign/magnitude mirroring",
        "`ARRI-Wide-Gamut-3`",
        "`ARRI-Wide-Gamut-4`",
        "D65",
        "independently from gamma",
        "Bradford",
        "`native`",
    ):
        assert claim in tokens
    for claim in ("27 Colorspace", "33 Gamma", "188 canonical tokens"):
        assert claim in requirements
    for claim in (
        "ARRI-Wide-Gamut-3",
        "ARRI-Wide-Gamut-4",
        "ARRI-LogC3",
        "EI 800",
        "400 / 1023",
        "ARRI-LogC4 remains bit-identical",
    ):
        assert claim in changelog
    for name, docstring in normalized_docstrings.items():
        assert "ARRI-LogC3 is the ARRI EI 800 relative scene-exposure curve" in docstring, name
        assert "400 / 1023" in docstring, name
        assert "without clipping or sign/magnitude mirroring" in docstring, name
