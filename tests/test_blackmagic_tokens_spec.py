"""Specification tests for Blackmagic Design Gen 5 and DaVinci tokens."""

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

_A = np.float64("0.08692876065491224")
_B = np.float64("0.005494072432257808")
_C = np.float64("0.5300133392291939")
_D = np.float64("8.283605932402494")
_E = np.float64("0.09246575342465753")
_LIN_CUT = np.float64("0.005")
_LOG_CUT = _D * _LIN_CUT + _E

_DI_A = np.float64("0.0075")
_DI_B = np.float64("7.0")
_DI_C = np.float64("0.07329248")
_DI_M = np.float64("10.44426855")
_DI_LIN_CUT = np.float64("0.00262409")
_DI_DECODE_CUT = _DI_M * _DI_LIN_CUT
_DI_PRINTED_LOG_CUT = np.float64("0.02740668")

_GAMUTS = {
    "Blackmagic-Wide-Gamut-Gen-5": (
        ((0.7177215, 0.3171181), (0.2280410, 0.8615690), (0.1005841, -0.0820452)),
        (0.3127, 0.3290),
    ),
    "DaVinci-Wide-Gamut": (
        ((0.8000, 0.3130), (0.1682, 0.9877), (0.0790, -0.1155)),
        (0.3127, 0.3290),
    ),
}
_REC709 = (((0.640, 0.330), (0.300, 0.600), (0.150, 0.060)), (0.3127, 0.3290))
_ACESCG = (((0.713, 0.293), (0.165, 0.830), (0.128, 0.044)), (0.32168, 0.33767))
_ACES2065 = (((0.7347, 0.2653), (0.0000, 1.0000), (0.0001, -0.0770)), (0.32168, 0.33767))
_BRADFORD = np.asarray(
    ((0.8951, 0.2664, -0.1614), (-0.7502, 1.7135, 0.0367), (0.0389, -0.0685, 1.0296)), dtype=np.float64
)
_CAT02 = np.asarray(((0.7328, 0.4296, -0.1624), (-0.7036, 1.6975, 0.0061), (0.0030, 0.0136, 0.9834)), dtype=np.float64)


def _piecewise(
    values: np.ndarray,
    mask: np.ndarray,
    lower: Callable[[np.ndarray], np.ndarray],
    upper: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    result[mask] = lower(values[mask])
    result[~mask] = upper(values[~mask])
    return result


def _film_encode(values: np.ndarray) -> np.ndarray:
    return _piecewise(values, values < _LIN_CUT, lambda part: _D * part + _E, lambda part: _A * np.log(part + _B) + _C)


def _film_decode(values: np.ndarray) -> np.ndarray:
    return _piecewise(
        values, values < _LOG_CUT, lambda part: (part - _E) / _D, lambda part: np.exp((part - _C) / _A) - _B
    )


def _di_encode(values: np.ndarray) -> np.ndarray:
    return _piecewise(
        values,
        values <= _DI_LIN_CUT,
        lambda part: part * _DI_M,
        lambda part: (np.log2(part + _DI_A) + _DI_B) * _DI_C,
    )


def _di_decode(values: np.ndarray, *, cut: np.float64 = _DI_DECODE_CUT) -> np.ndarray:
    return _piecewise(
        values,
        values <= cut,
        lambda part: part / _DI_M,
        lambda part: np.float64(2.0) ** (part / _DI_C - _DI_B) - _DI_A,
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
        z = np.arange(array.shape[0], dtype=np.float32)[:, None] + np.float32(16.0)
        array = np.concatenate((z, array[:, (2, 0, 1)]), axis=1)
        channels = ("Z", "B", "R", "G")
    return px.io.from_array(
        cp.asarray(array[None, :, :]), colorspace=colorspace, gamma=gamma, channels=channels, matrix="native"
    )


def _red_values(frame: px.core.Frame) -> np.ndarray:
    return px.io.to_array(frame).get()[0, :, frame.channels.index("R")]


def _literal_strings(annotation: object) -> tuple[str, ...]:
    if get_origin(annotation) is Literal:
        return tuple(value for value in get_args(annotation) if isinstance(value, str))
    return tuple(value for argument in get_args(annotation) for value in _literal_strings(argument))


def _rgb_to_xyz(
    primaries: tuple[tuple[float, float], tuple[float, float], tuple[float, float]], white: tuple[float, float]
) -> np.ndarray:
    matrix = np.asarray(
        (tuple(x / y for x, y in primaries), (1.0, 1.0, 1.0), tuple((1.0 - x - y) / y for x, y in primaries)),
        dtype=np.float64,
    )
    white_xyz = np.asarray((white[0] / white[1], 1.0, (1.0 - white[0] - white[1]) / white[1]), dtype=np.float64)
    return matrix @ np.diag(np.linalg.solve(matrix, white_xyz))


def _adaptation(source: tuple[float, float], target: tuple[float, float], cone: np.ndarray) -> np.ndarray:
    def xyz(xy: tuple[float, float]) -> np.ndarray:
        return np.asarray((xy[0] / xy[1], 1.0, (1.0 - xy[0] - xy[1]) / xy[1]), dtype=np.float64)

    return np.linalg.inv(cone) @ np.diag((cone @ xyz(target)) / (cone @ xyz(source))) @ cone


def _conversion(source: tuple[object, tuple[float, float]], target: tuple[object, tuple[float, float]]) -> np.ndarray:
    source_primaries, source_white = source
    target_primaries, target_white = target
    return (
        np.linalg.inv(_rgb_to_xyz(target_primaries, target_white))
        @ _adaptation(source_white, target_white, _BRADFORD)
        @ _rgb_to_xyz(source_primaries, source_white)
    )


def test_blackmagic_tokens_extend_canonical_vocabulary_and_public_static_surfaces() -> None:
    """v1-blackmagic-tokens acceptance 33-34; v1-red-tokens acceptance 54-55;
    v1-canon-tokens acceptance 76-77; v1-panasonic-tokens acceptance 99-100;
    v1-standard-tokens acceptance 117; v1-vendor-a-tokens acceptance 140-141;
    v1-vendor-b-tokens acceptance 166-167: expose current canonical tokens.
    """
    assert get_args(px.core.Colorspace) == _COLORSPACES
    assert get_args(px.core.Gamma) == _GAMMAS
    assert len(_ALIASES) == 30
    assert sum(len(get_args(alias)) for alias in _ALIASES) == 188
    assert _literal_strings(get_type_hints(px.color.linear_to_gamma)["gamma"]) == _GAMMAS
    assert _literal_strings(get_type_hints(px.color.rgb_to_rgb)["input_colorspace"]) == _COLORSPACES
    assert _literal_strings(get_type_hints(px.color.rgb_to_rgb)["output_gamma"]) == _GAMMAS
    for colorspace, gamma in (
        ("Blackmagic-Wide-Gamut-Gen-5", "Blackmagic-Film-Gen-5"),
        ("DaVinci-Wide-Gamut", "DaVinci-Intermediate"),
    ):
        frame = _frame((0.18,), colorspace=colorspace, gamma=gamma)
        assert (frame.colorspace, frame.gamma) == (colorspace, gamma)
        assert f"colorspace={colorspace!r}" in repr(frame)
        assert f"gamma={gamma!r}" in repr(frame)


def test_blackmagic_token_keys_are_collision_free_family_local_and_do_not_guess_resolve_labels() -> None:
    """v1-blackmagic-tokens acceptance 35; v1-vendor-a-tokens acceptance 142:
    normalize separators locally without aliases or fuzzy labels.
    """
    from pixtreme._core.validation import _normalized_closed_token
    from pixtreme._core.vocabulary import _PERMANENT_TOKEN_ALIASES

    translation = str.maketrans("", "", " .-_")
    assert len({token.translate(translation).casefold() for token in _COLORSPACES}) == len(_COLORSPACES)
    assert len({token.translate(translation).casefold() for token in _GAMMAS}) == len(_GAMMAS)
    additions = (*_COLORSPACES[14:16], *_GAMMAS[13:15])
    assert all(canonical not in additions for _alias, canonical in _PERMANENT_TOKEN_ALIASES)
    for canonical, family, axis in (
        ("Blackmagic-Wide-Gamut-Gen-5", _COLORSPACES, "colorspace"),
        ("DaVinci-Wide-Gamut", _COLORSPACES, "colorspace"),
        ("Blackmagic-Film-Gen-5", _GAMMAS, "gamma"),
        ("DaVinci-Intermediate", _GAMMAS, "gamma"),
    ):
        compact = canonical.translate(translation)
        variants = (canonical.swapcase(), compact, *(canonical.replace("-", separator) for separator in " ._"))
        for variant in variants:
            assert _normalized_closed_token(variant, axis=axis, accepted=family) == canonical
    for fuzzy in ("Blackmagic Design Film Gen 5", "DaVinci WG", "DaVinci WG/Intermediate"):
        with pytest.raises(ValueError):
            _normalized_closed_token(fuzzy, axis="gamma", accepted=_GAMMAS)
    with pytest.raises(ValueError):
        _normalized_closed_token("DaVinci-Intermediate", axis="colorspace", accepted=_COLORSPACES)


def test_blackmagic_film_encode_matches_independent_oracle_anchors_and_monotonic_cut() -> None:
    """v1-blackmagic-tokens acceptance 36 and 38: Film Gen 5 encode follows the signed natural-log curve."""
    cut = np.float32(_LIN_CUT)
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
    actual = _red_values(px.color.linear_to_gamma(_frame(values), gamma="Blackmagic-Film-Gen-5"))
    np.testing.assert_allclose(actual, _film_encode(values.astype(np.float64)), rtol=0.0, atol=2e-6)
    grid = np.linspace(float(_LIN_CUT - 2e-4), float(_LIN_CUT + 2e-4), 4097, dtype=np.float32)
    encoded_grid = _red_values(px.color.linear_to_gamma(_frame(grid), gamma="Blackmagic-Film-Gen-5"))
    assert np.all(np.diff(encoded_grid) >= np.float32(0.0))
    assert actual[0] < actual[4] and actual[-1] > actual[-2]
    assert np.isclose(_D, _A / (_LIN_CUT + _B), rtol=0.0, atol=2e-15)

    anchors = np.asarray((0.0, 0.18, 1.0, 10.0, 40.0, 100.0, 222.86), dtype=np.float32)
    published = np.asarray(
        (
            0.0924657534246575,
            0.3835616438356165,
            0.5304896249573048,
            0.7302219538415439,
            0.8506949973834717,
            0.9303398518999735,
            1.0,
        )
    )
    anchor_actual = _red_values(px.color.linear_to_gamma(_frame(anchors), gamma="Blackmagic-Film-Gen-5"))
    # The first six published values carry enough digits to identify their float32 representations. CUDA evaluation
    # matches those representations exactly except at 100, where it differs by one float32 ULP.
    np.testing.assert_array_max_ulp(anchor_actual[:-1], published[:-1].astype(np.float32), maxulp=1)
    # The vendor prints the 222.86 row as 1.0 although the normative expression is about 0.99999963.
    np.testing.assert_allclose(anchor_actual[-1:], published[-1:], rtol=0.0, atol=6e-7)
    codes = np.floor(np.float64(64.0) + np.float64(876.0) * anchor_actual.astype(np.float64) + np.float64(0.5))
    np.testing.assert_array_equal(codes.astype(np.int64), np.asarray((145, 400, 529, 704, 809, 879, 940)))


def test_blackmagic_film_decode_uses_the_derived_threshold_and_signed_inverse() -> None:
    """v1-blackmagic-tokens acceptance 37: Film Gen 5 decode uses its derived cut and unclipped inverse."""
    cut = np.float32(_LOG_CUT)
    values = np.asarray(
        (-0.25, np.nextafter(cut, -np.inf), cut, np.nextafter(cut, np.inf), 0.0, 0.18, 1.0, 1.5), dtype=np.float32
    )
    actual = _red_values(
        px.color.gamma_to_linear(_frame(values, gamma="Blackmagic-Film-Gen-5"), gamma="Blackmagic-Film-Gen-5")
    )
    expected = _film_decode(values.astype(np.float64))
    # Up to encoded 1.0, 1e-8 covers float32 cancellation near the cut and 3e-7 covers inverse-log evaluation.
    np.testing.assert_allclose(actual[:-1], expected[:-1], rtol=3e-7, atol=1e-8)
    # Encoded 1.5 decodes to roughly 70,151: its 0.0103 absolute error is only 1.46e-7 relative error.
    np.testing.assert_allclose(actual[-1:], expected[-1:], rtol=2e-7, atol=0.0)
    assert actual[0] < 0.0 and actual[-1] > 1.0


@pytest.mark.parametrize(
    ("gamma", "cut", "encode", "decode", "acceptance"),
    (
        ("Blackmagic-Film-Gen-5", _LIN_CUT, _film_encode, _film_decode, 39),
        ("DaVinci-Intermediate", _DI_LIN_CUT, _di_encode, _di_decode, 43),
    ),
)
def test_blackmagic_curves_round_trip_and_standalone_fused_paths_preserve_frame_contract(
    gamma: str,
    cut: np.float64,
    encode: Callable[[np.ndarray], np.ndarray],
    decode: Callable[[np.ndarray], np.ndarray],
    acceptance: int,
) -> None:
    """v1-blackmagic-tokens acceptance 39 and 43: both curves round-trip through standalone and fused paths."""
    assert acceptance in (39, 43)
    float_cut = np.float32(cut)
    linear = np.asarray(
        (-0.25, np.nextafter(float_cut, -np.inf), float_cut, np.nextafter(float_cut, np.inf), 0.0, 0.18, 1.0, 1.5),
        dtype=np.float32,
    )
    source = _frame(linear, auxiliary=True)
    before = source.data.copy()
    encoded = px.color.linear_to_gamma(source, gamma=gamma)
    fused_encoded = px.color.rgb_to_rgb(source, output_gamma=gamma)
    restored = px.color.gamma_to_linear(encoded, gamma=gamma)
    fused_restored = px.color.rgb_to_rgb(encoded, output_gamma="linear")
    encoded_fixture = encode(linear.astype(np.float64)).astype(np.float32)
    decoded = px.color.gamma_to_linear(_frame(encoded_fixture, gamma=gamma), gamma=gamma)
    reencoded = px.color.linear_to_gamma(decoded, gamma=gamma)
    assert cp.array_equal(encoded.data, fused_encoded.data)
    assert cp.array_equal(restored.data, fused_restored.data)
    assert cp.array_equal(encoded.data[..., 0], source.data[..., 0])
    assert cp.array_equal(restored.data[..., 0], source.data[..., 0])
    assert cp.array_equal(source.data, before)
    np.testing.assert_allclose(_red_values(restored), linear, rtol=0.0, atol=1.2e-5)
    np.testing.assert_allclose(_red_values(reencoded), encoded_fixture, rtol=0.0, atol=1.2e-5)
    np.testing.assert_allclose(_red_values(decoded), decode(encoded_fixture.astype(np.float64)), rtol=0.0, atol=1.2e-5)
    assert encoded.gamma == gamma and restored.gamma == "linear"
    assert encoded is not source and encoded.data.data.ptr != source.data.data.ptr
    assert encoded.data.dtype == restored.data.dtype == cp.float32


def test_davinci_intermediate_encode_matches_normative_oracle_published_anchors_and_cut() -> None:
    """v1-blackmagic-tokens acceptance 40 and 42: Intermediate uses the public base-2 curve and anchor bounds."""
    cut = np.float32(_DI_LIN_CUT)
    values = np.asarray(
        (-0.25, np.nextafter(cut, -np.inf), cut, np.nextafter(cut, np.inf), 0.0, 0.18, 1.0, 1.5), dtype=np.float32
    )
    actual = _red_values(px.color.linear_to_gamma(_frame(values), gamma="DaVinci-Intermediate"))
    np.testing.assert_allclose(actual, _di_encode(values.astype(np.float64)), rtol=0.0, atol=2e-6)
    grid = np.linspace(float(_DI_LIN_CUT - 2e-4), float(_DI_LIN_CUT + 2e-4), 4097, dtype=np.float32)
    encoded_grid = _red_values(px.color.linear_to_gamma(_frame(grid), gamma="DaVinci-Intermediate"))
    assert np.all(np.diff(encoded_grid) >= np.float32(0.0))
    linear_cut_value = _DI_M * _DI_LIN_CUT
    log_cut_value = (np.log2(_DI_LIN_CUT + _DI_A) + _DI_B) * _DI_C
    assert abs(linear_cut_value - log_cut_value) < np.spacing(np.float32(linear_cut_value))
    derived_slope = _DI_C / ((_DI_LIN_CUT + _DI_A) * np.log(np.float64(2.0)))
    assert np.isclose(_DI_M - derived_slope, np.float64("1.71412106753e-6"), rtol=0.0, atol=2e-13)

    anchors = np.asarray((-0.01, 0.0, 0.18, 1.0, 10.0, 40.0, 100.0), dtype=np.float32)
    published = np.asarray((-0.104443, 0.0, 0.336043, 0.513837, 0.756599, 0.903125, 1.0))
    anchor_actual = _red_values(px.color.linear_to_gamma(_frame(anchors), gamma="DaVinci-Intermediate"))
    np.testing.assert_allclose(anchor_actual, published, rtol=0.0, atol=6e-7)
    assert np.round(anchor_actual[0].astype(np.float64), 6) == published[0]
    assert np.round(anchor_actual[2].astype(np.float64), 6) == published[2]
    assert np.round(_di_encode(np.asarray((40.0,)))[0], 6) != published[5]


def test_davinci_intermediate_decode_uses_derived_not_printed_threshold() -> None:
    """v1-blackmagic-tokens acceptance 41-42: decode assigns all 11 disputed float32 values to the derived branch."""
    assert np.isclose(_DI_DECODE_CUT - _DI_PRINTED_LOG_CUT, np.float64("2.06593695e-8"), rtol=0.0, atol=1e-16)
    lower = np.float32(_DI_PRINTED_LOG_CUT)
    upper = np.float32(_DI_DECODE_CUT)
    disputed: list[np.float32] = []
    value = np.nextafter(lower, np.float32(np.inf))
    while value <= upper:
        disputed.append(value)
        value = np.nextafter(value, np.float32(np.inf))
    assert len(disputed) == 11
    values = np.asarray(
        (-0.25, np.nextafter(upper, -np.inf), upper, np.nextafter(upper, np.inf), *disputed), dtype=np.float32
    )
    actual = _red_values(
        px.color.gamma_to_linear(_frame(values, gamma="DaVinci-Intermediate"), gamma="DaVinci-Intermediate")
    )
    derived = _di_decode(values.astype(np.float64))
    printed = _di_decode(values.astype(np.float64), cut=_DI_PRINTED_LOG_CUT)
    np.testing.assert_allclose(actual, derived, rtol=0.0, atol=2e-9)
    assert not np.array_equal(derived[-11:].view(np.uint64), printed[-11:].view(np.uint64))
    # Independent derived-branch fixture: evaluate the 11 exact encoded float32 values as `value / DI_M` in
    # float32. Replacing the production threshold with the printed cut selects inverse log for all 11 and changes
    # every output bit (the alternate branch produced 992737506 for the first seven values and 992737522 after).
    np.testing.assert_array_equal(
        actual[-11:].view(np.uint32),
        np.asarray(
            (
                992737509,
                992737510,
                992737511,
                992737511,
                992737512,
                992737513,
                992737514,
                992737514,
                992737515,
                992737516,
                992737517,
            ),
            dtype=np.uint32,
        ),
    )
    assert actual[0] < 0.0


@pytest.mark.parametrize(
    ("colorspace", "vendor", "vendor_tolerance", "acceptance"),
    (
        (
            "Blackmagic-Wide-Gamut-Gen-5",
            ((0.606530, 0.220408, 0.123479), (0.267989, 0.832731, -0.100720), (-0.029442, -0.086611, 1.204861)),
            3e-4,
            44,
        ),
        (
            "DaVinci-Wide-Gamut",
            (
                (0.70062239, 0.14877482, 0.10105872),
                (0.27411851, 0.87363190, -0.14775041),
                (-0.09896291, -0.13789533, 1.32591599),
            ),
            1e-8,
            45,
        ),
    ),
)
def test_blackmagic_gamut_primaries_matrices_and_native_rows_match_independent_oracles(
    colorspace: str, vendor: tuple[tuple[float, ...], ...], vendor_tolerance: float, acceptance: int
) -> None:
    """v1-blackmagic-tokens acceptance 44-45: primaries, production D65, conversion, and native row agree."""
    from pixtreme._core.colorspace import _COLORSPACE_DEFINITIONS

    assert acceptance in (44, 45)
    definition = _GAMUTS[colorspace]
    assert _COLORSPACE_DEFINITIONS[colorspace] == definition
    matrix = _rgb_to_xyz(*definition)
    np.testing.assert_allclose(matrix, np.asarray(vendor), rtol=0.0, atol=vendor_tolerance)
    values = np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (-0.25, 0.18, 1.5)))
    source = _frame(values, colorspace=colorspace)
    converted = px.color.rgb_to_rgb(source, output_colorspace="Rec.709")
    grayscale = px.color.rgb_to_grayscale(source, colorspace=colorspace, gamma="linear", matrix="native")
    np.testing.assert_allclose(
        px.io.to_array(converted).get()[0], values @ _conversion(definition, _REC709).T, rtol=0.0, atol=6e-6
    )
    np.testing.assert_allclose(px.io.to_array(grayscale).get()[0, :, 0], values @ matrix[1], rtol=0.0, atol=6e-6)


def test_blackmagic_gamut_adaptation_pairing_and_cat02_auxiliary_contracts() -> None:
    """v1-blackmagic-tokens acceptance 46: conversion uses Bradford and leaves gamut/gamma pairing independent."""
    values = np.asarray(((1.0, -0.25, 0.18), (0.2, 0.4, 1.5)), dtype=np.float32)
    source = _frame(values, colorspace="Blackmagic-Wide-Gamut-Gen-5", auxiliary=True)
    before = source.data.copy()
    davinci = px.color.rgb_to_rgb(source, output_colorspace="DaVinci-Wide-Gamut", output_gamma="DaVinci-Intermediate")
    acescg = px.color.rgb_to_rgb(source, output_colorspace="ACEScg")
    source_rgb = px.io.to_array(source).get()[0][:, [source.channels.index(label) for label in ("R", "G", "B")]]
    np.testing.assert_allclose(
        px.io.to_array(davinci).get()[0][:, [davinci.channels.index(label) for label in ("R", "G", "B")]],
        _di_encode(
            (source_rgb @ _conversion(_GAMUTS["Blackmagic-Wide-Gamut-Gen-5"], _GAMUTS["DaVinci-Wide-Gamut"]).T).astype(
                np.float64
            )
        ),
        rtol=0.0,
        atol=8e-6,
    )
    np.testing.assert_allclose(
        px.io.to_array(acescg).get()[0][:, [acescg.channels.index(label) for label in ("R", "G", "B")]],
        source_rgb @ _conversion(_GAMUTS["Blackmagic-Wide-Gamut-Gen-5"], _ACESCG).T,
        rtol=0.0,
        atol=6e-6,
    )
    assert cp.array_equal(davinci.data[..., 0], source.data[..., 0])
    assert cp.array_equal(source.data, before)
    assert (davinci.colorspace, davinci.gamma) == ("DaVinci-Wide-Gamut", "DaVinci-Intermediate")

    clf_expected = {
        "Blackmagic-Wide-Gamut-Gen-5": np.asarray(
            (
                (0.647091325580708, 0.242595385134207, 0.110313289285085),
                (0.0651915997328519, 1.02504756760476, -0.0902391673376125),
                (-0.0275570729194699, -0.0805887097177784, 1.10814578263725),
            )
        ),
        "DaVinci-Wide-Gamut": np.asarray(
            (
                (0.748270290272981, 0.167694659554328, 0.0840350501726906),
                (0.0208421234689102, 1.11190474268894, -0.132746866157851),
                (-0.0915122574225729, -0.127746712807307, 1.21925897022988),
            )
        ),
    }
    for colorspace, (primaries, white) in _GAMUTS.items():
        source_white = (0.3127170, 0.3290312) if colorspace.startswith("Blackmagic") else white
        cat02 = (
            np.linalg.inv(_rgb_to_xyz(*_ACES2065))
            @ _adaptation(source_white, _ACES2065[1], _CAT02)
            @ _rgb_to_xyz(primaries, source_white)
        )
        np.testing.assert_allclose(cat02, clf_expected[colorspace], rtol=0.0, atol=5e-13)


# Baseline provenance: generated at commit e6d3da1 before adding either Blackmagic transfer branch. Reproduce from
# that checkout with the worktree source selected through PYTHONPATH, construct RGB float32 Frames from the exact
# `linear` and `encoded` arrays below, call each public standalone transfer, copy R to NumPy, and print
# `.view(np.uint32).tolist()`. The fixture intentionally locks existing branch cuts and signed extensions.
@pytest.mark.parametrize(
    ("gamma", "linear", "encoded", "encode_bits", "decode_bits"),
    (
        (
            "S-Log",
            (-0.25, -1.401298464324817e-45, 0.0, 0.18000000715255737, 1.0, 1.5),
            (-0.25, 0.08825129270553589, 0.0, 0.18000000715255737, 1.0, 1.5),
            (3213684627, 1035255064, 1035255064, 1053104870, 1059289119, 1060354155),
            (3180437010, 828794470, 3164076049, 1020590012, 1092341038, 1130000704),
        ),
        (
            "S-Log2",
            (-0.25, -1.401298464324817e-45, 0.0, 0.18000000715255737, 1.0, 1.5),
            (-0.25, 0.08825129270553589, 0.0, 0.18000000715255737, 1.0, 1.5),
            (3208701273, 1035255064, 1035255064, 1051580213, 1058392199, 1059445478),
            (3184377421, 832750646, 3168188334, 1024881550, 1096557030, 1134186920),
        ),
        (
            "S-Log3",
            (-0.25, 0.011250000447034836, 0.0, 0.18000000715255737, 1.0, 1.5),
            (-0.25, 0.16736099123954773, 0.0, 0.18000000715255737, 1.0, 1.5),
            (3217556478, 1043030190, 1035874188, 1053963405, 1058575679, 1059324708),
            (3176403988, 1010323947, 3160785829, 1013075316, 1108979464, 1163464640),
        ),
        (
            "ARRI-LogC3",
            (-0.25, 0.010590990073978901, 0.0, 0.18000000715255737, 1.0, 1.5),
            (-0.25, 0.14965814352035522, 0.0, 0.18000000715255737, 1.0, 1.5),
            (3214926503, 1041842171, 1035866836, 1053307405, 1058149604, 1058874277),
            (3179465741, 1009616343, 3163399366, 1015826037, 1113346544, 1169516654),
        ),
        (
            "ARRI-LogC4",
            (-0.25, -0.018056996166706085, 0.0, 0.18000000715255737, 1.0, 1.5),
            (-0.25, 0.0, -1.1754943508222875e-38, 0.18000000715255737, 1.0, 1.5),
            (3221400801, 2995575096, 1035874188, 1049528807, 1054532562, 1055775089),
            (3174975734, 3163810883, 3163810883, 1026875391, 1139467878, 1203831891),
        ),
    ),
)
def test_existing_camera_log_gpu_bits_remain_identical_to_pre_blackmagic_baseline(
    gamma: str,
    linear: tuple[float, ...],
    encoded: tuple[float, ...],
    encode_bits: tuple[int, ...],
    decode_bits: tuple[int, ...],
) -> None:
    """v1-blackmagic-tokens acceptance 47; v1-red-tokens acceptance 68-69: existing transfer bits stay fixed."""
    actual_encode = _red_values(px.color.linear_to_gamma(_frame(linear), gamma=gamma)).view(np.uint32)
    actual_decode = _red_values(px.color.gamma_to_linear(_frame(encoded, gamma=gamma), gamma=gamma)).view(np.uint32)
    np.testing.assert_array_equal(actual_encode, np.asarray(encode_bits, dtype=np.uint32))
    np.testing.assert_array_equal(actual_decode, np.asarray(decode_bits, dtype=np.uint32))


def test_dpx_classifies_blackmagic_curves_as_logarithmic_without_changing_existing_mappings() -> None:
    """v1-blackmagic-tokens acceptance 48; v1-red-tokens acceptance 72: DPX uses current canonical tokens."""
    from pixtreme._io.formats.dpx import _dpx_transfer_from_gamma

    gammas = (
        "Cineon",
        "linear",
        "S-Log",
        "S-Log2",
        "S-Log3",
        "ARRI-LogC3",
        "ARRI-LogC4",
        "Blackmagic-Film-Gen-5",
        "DaVinci-Intermediate",
        "Rec.709",
        "Gamma-2.6",
    )
    assert {gamma: _dpx_transfer_from_gamma(gamma) for gamma in gammas} == {
        "Cineon": 1,
        "linear": 2,
        "S-Log": 3,
        "S-Log2": 3,
        "S-Log3": 3,
        "ARRI-LogC3": 3,
        "ARRI-LogC4": 3,
        "Blackmagic-Film-Gen-5": 3,
        "DaVinci-Intermediate": 3,
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
def test_invalid_blackmagic_axis_values_fail_before_gpu_with_ordered_canonical_errors(
    operation: str, parameter: str, rejected: object, candidates: tuple[str, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    """v1-blackmagic-tokens acceptance 49; v1-vendor-a-tokens acceptance 160:
    invalid inputs fail before GPU with why/what/how recovery.
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
    assert repr(rejected) in message and repr(candidates) in message
    assert "DaVinci WG" not in message and "Blackmagic Design Film" not in message


def test_blackmagic_public_documents_docstrings_and_changelog_are_synchronized() -> None:
    """v1-blackmagic-tokens acceptance 50; v1-red-tokens acceptance 72; v1-canon-tokens acceptance 93;
    v1-panasonic-tokens acceptance 112; v1-vendor-a-tokens acceptance 161; v1-vendor-b-tokens acceptance 188;
    GitHub #29: docs use current counts.
    """
    tokens = (ROOT / "docs_site" / "tokens.md").read_text(encoding="utf-8")
    requirements = require_repo_file("docs/requirements.md").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    docstrings = " ".join(
        " ".join((inspect.getdoc(function) or "").split())
        for function in (px.color.gamma_to_linear, px.color.linear_to_gamma, px.color.rgb_to_rgb)
    )
    for claim in (
        "`Blackmagic-Film-Gen-5`",
        "`DaVinci-Intermediate`",
        "natural logarithm",
        "base-2 logarithm",
        "derived decode threshold",
        "negative values",
        "`Blackmagic-Wide-Gamut-Gen-5`",
        "`DaVinci-Wide-Gamut`",
        "D65",
        "independently from gamma",
        "Gen 4",
    ):
        assert claim in tokens
    for claim in ("27 Colorspace", "33 Gamma", "188 canonical tokens"):
        assert claim in requirements
    for claim in (
        "Blackmagic-Wide-Gamut-Gen-5",
        "DaVinci-Wide-Gamut",
        "Blackmagic-Film-Gen-5",
        "DaVinci-Intermediate",
        "bit-identical",
    ):
        assert claim in changelog
    for claim in ("Blackmagic Film Gen 5", "DaVinci Intermediate", "derived decode threshold"):
        assert claim in docstrings
