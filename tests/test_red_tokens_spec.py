"""Specification tests for RED gamut and transfer tokens."""

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

_D65 = (0.3127, 0.3290)
_ACES_WHITE = (0.32168, 0.33767)
_REC709 = (((0.640, 0.330), (0.300, 0.600), (0.150, 0.060)), _D65)
_ACES2065 = (((0.7347, 0.2653), (0.0000, 1.0000), (0.0001, -0.0770)), _ACES_WHITE)
_GAMUTS = {
    "REDWideGamutRGB": (((0.780308, 0.304253), (0.121595, 1.493994), (0.095612, -0.084589)), _D65),
    "DRAGONcolor": (
        ((0.7586558926, 0.3303553486), (0.2949236198, 0.7080532421), (0.0859616012, -0.0458794370)),
        _D65,
    ),
    "DRAGONcolor2": (
        ((0.7586562142, 0.3303558357), (0.2949238877, 0.7080533632), (0.1441687269, 0.0503573846)),
        _D65,
    ),
    "REDcolor2": (
        ((0.8974072220, 0.3307762259), (0.2960220945, 0.6846355509), (0.0997995129, -0.0230005132)),
        _D65,
    ),
    "REDcolor3": (
        ((0.7025986586, 0.3301855889), (0.2957822357, 0.6897482584), (0.1110905291, -0.0043323210)),
        _D65,
    ),
    "REDcolor4": (
        ((0.7025981547, 0.3301850962), (0.2957823281, 0.6897482540), (0.1444592365, 0.0508377210)),
        _D65,
    ),
}
_VENDOR_ACES = {
    "REDWideGamutRGB": np.asarray(
        ((0.785043, 0.083844, 0.131118), (0.023172, 1.087892, -0.111055), (-0.073769, -0.314639, 1.388537))
    ),
    "DRAGONcolor": np.asarray(
        ((0.532279, 0.376648, 0.091073), (0.046344, 0.974513, -0.020860), (-0.053976, -0.000320, 1.054267))
    ),
    "DRAGONcolor2": np.asarray(
        ((0.468452, 0.331484, 0.200064), (0.040787, 0.857658, 0.101553), (-0.047504, -0.000282, 1.047756))
    ),
    "REDcolor2": np.asarray(
        ((0.480997, 0.402289, 0.116714), (-0.004938, 1.000154, 0.004781), (-0.105257, 0.025320, 1.079907))
    ),
    "REDcolor3": np.asarray(
        ((0.512136, 0.360370, 0.127494), (0.070377, 0.903884, 0.025737), (-0.020824, 0.017671, 1.003123))
    ),
    "REDcolor4": np.asarray(
        ((0.474202, 0.333677, 0.192121), (0.065164, 0.836932, 0.097901), (-0.019281, 0.016362, 1.002889))
    ),
}

_LOG3G10_A = np.float64("0.224282")
_LOG3G10_B = np.float64("155.975327")
_LOG3G10_C = np.float64("0.01")
_LOG3G10_G = np.float64("15.1927")
_CINEON_OFFSET = np.float64("0.0107977516232771")
_BRADFORD = np.asarray(
    ((0.8951, 0.2664, -0.1614), (-0.7502, 1.7135, 0.0367), (0.0389, -0.0685, 1.0296)),
    dtype=np.float64,
)


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


def _log3g10_encode(values: np.ndarray) -> np.ndarray:
    shifted = values + _LOG3G10_C
    return _piecewise(
        shifted,
        shifted < 0.0,
        lambda part: _LOG3G10_G * part,
        lambda part: _LOG3G10_A * np.log10(_LOG3G10_B * part + 1.0),
    )


def _log3g10_decode(values: np.ndarray) -> np.ndarray:
    return _piecewise(
        values,
        values < 0.0,
        lambda part: part / _LOG3G10_G - _LOG3G10_C,
        lambda part: (np.float64(10.0) ** (part / _LOG3G10_A) - 1.0) / _LOG3G10_B - _LOG3G10_C,
    )


def _cineon_encode(values: np.ndarray) -> np.ndarray:
    sign = np.where(values < 0.0, -1.0, 1.0)
    magnitude = np.abs(values)
    encoded = (685.0 + 300.0 * np.log10(magnitude * (1.0 - _CINEON_OFFSET) + _CINEON_OFFSET)) / 1023.0
    return sign * encoded


def _cineon_decode(values: np.ndarray) -> np.ndarray:
    sign = np.where(values < 0.0, -1.0, 1.0)
    magnitude = np.abs(values)
    decoded = (np.float64(10.0) ** ((1023.0 * magnitude - 685.0) / 300.0) - _CINEON_OFFSET) / (1.0 - _CINEON_OFFSET)
    return sign * decoded


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


def _rgb_to_xyz(
    primaries: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    white: tuple[float, float],
) -> np.ndarray:
    matrix = np.asarray(
        (tuple(x / y for x, y in primaries), (1.0, 1.0, 1.0), tuple((1.0 - x - y) / y for x, y in primaries)),
        dtype=np.float64,
    )
    white_xyz = np.asarray((white[0] / white[1], 1.0, (1.0 - white[0] - white[1]) / white[1]), dtype=np.float64)
    return matrix @ np.diag(np.linalg.solve(matrix, white_xyz))


def _adaptation(source: tuple[float, float], target: tuple[float, float]) -> np.ndarray:
    def xyz(xy: tuple[float, float]) -> np.ndarray:
        return np.asarray((xy[0] / xy[1], 1.0, (1.0 - xy[0] - xy[1]) / xy[1]), dtype=np.float64)

    return np.linalg.inv(_BRADFORD) @ np.diag((_BRADFORD @ xyz(target)) / (_BRADFORD @ xyz(source))) @ _BRADFORD


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


def test_red_tokens_extend_canonical_vocabulary_and_public_static_surfaces() -> None:
    """v1-red-tokens acceptance 54-55; v1-canon-tokens acceptance 76-77;
    v1-panasonic-tokens acceptance 99-100; v1-standard-tokens acceptance 117;
    v1-vendor-a-tokens acceptance 140-141; v1-vendor-b-tokens acceptance 166-167:
    expose exact canonical vocabulary.
    """
    assert get_args(px.core.Colorspace) == _COLORSPACES
    assert get_args(px.core.Gamma) == _GAMMAS
    assert len(_ALIASES) == 30
    assert sum(len(get_args(alias)) for alias in _ALIASES) == 188
    assert _literal_strings(get_type_hints(px.color.linear_to_gamma)["gamma"]) == _GAMMAS
    assert _literal_strings(get_type_hints(px.color.rgb_to_rgb)["input_colorspace"]) == _COLORSPACES
    assert _literal_strings(get_type_hints(px.color.rgb_to_rgb)["output_gamma"]) == _GAMMAS
    for colorspace, gamma in (
        ("REDWideGamutRGB", "RED-Log3G10"),
        ("DRAGONcolor", "REDlogFilm"),
        ("ARRI-Wide-Gamut-4", "ARRI-LogC4"),
    ):
        frame = _frame((0.18,), colorspace=colorspace, gamma=gamma)
        assert (frame.colorspace, frame.gamma) == (colorspace, gamma)
        assert f"colorspace={colorspace!r}" in repr(frame)
        assert f"gamma={gamma!r}" in repr(frame)
    for compatibility in ("logc4", "LogC3", "LogC4"):
        assert compatibility not in get_args(px.core.Gamma)


def test_red_token_keys_aliases_and_invalid_inputs_follow_the_shared_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1-red-tokens acceptance 56 and 71; v1-vendor-a-tokens acceptance 142 and 160:
    normalize keys and reject raw invalid values before GPU work.
    """
    from pixtreme._core.validation import _normalized_closed_token
    from pixtreme._core.vocabulary import _PERMANENT_TOKEN_ALIASES

    translation = str.maketrans("", "", " .-_")
    tokens = (*_COLORSPACES[16:22], *_GAMMAS[11:13], *_GAMMAS[15:17])
    expected_keys = {
        "redwidegamutrgb",
        "dragoncolor",
        "dragoncolor2",
        "redcolor2",
        "redcolor3",
        "redcolor4",
        "arrilogc3",
        "arrilogc4",
        "redlog3g10",
        "redlogfilm",
    }
    assert {token.translate(translation).casefold() for token in tokens} == expected_keys
    assert len({token.translate(translation).casefold() for token in _COLORSPACES}) == len(_COLORSPACES)
    assert len({token.translate(translation).casefold() for token in _GAMMAS}) == len(_GAMMAS)
    assert ("logc4", "ARRI-LogC4") in _PERMANENT_TOKEN_ALIASES
    for canonical in tokens:
        family = _COLORSPACES if canonical in _COLORSPACES else _GAMMAS
        axis = "colorspace" if canonical in _COLORSPACES else "gamma"
        variants = (
            canonical.swapcase(),
            canonical.translate(translation),
            *(canonical.replace("-", separator) for separator in " ._"),
        )
        for variant in variants:
            assert _normalized_closed_token(variant, axis=axis, accepted=family) == canonical
    for spelling in ("RED Log3G10", "ARRI LogC3", "ARRI LogC4"):
        assert _normalized_closed_token(spelling, axis="gamma", accepted=_GAMMAS) == spelling.replace(" ", "-")
    assert _normalized_closed_token("logc4", axis="gamma", accepted=_GAMMAS) == "ARRI-LogC4"
    assert _normalized_closed_token("LogC4", axis="gamma", accepted=_GAMMAS) == "ARRI-LogC4"
    with pytest.raises(ValueError, match="received gamma='LogC3'"):
        _normalized_closed_token("LogC3", axis="gamma", accepted=_GAMMAS)
    with pytest.raises(ValueError):
        _normalized_closed_token("RED-Log3G10", axis="colorspace", accepted=_COLORSPACES)

    import pixtreme._color.semantics as semantics

    monkeypatch.setattr(
        semantics,
        "_color_semantics_kernel",
        lambda: pytest.fail("GPU work must not start"),
    )
    source = _frame((0.18,))
    for rejected in ("unknown", 17):
        with pytest.raises(ValueError) as captured:
            px.color.linear_to_gamma(source, gamma=rejected)
        message = str(captured.value)
        assert message.index("why=") < message.index("what=") < message.index("how=")
        assert f"received gamma={rejected!r}" in message
        assert repr(_GAMMAS) in message


def test_red_log3g10_encode_matches_public_constants_anchors_and_frame_contract() -> None:
    """v1-red-tokens acceptance 57 and 59: encode signed scene values with the published piecewise curve."""
    below = np.nextafter(np.float32(-0.01), np.float32(-np.inf))
    above = np.nextafter(np.float32(-0.01), np.float32(np.inf))
    values = np.asarray((-0.25, below, -0.01, above, 0.0, 0.18, 1.0, 1.5), dtype=np.float32)
    source = _frame(values, auxiliary=True)
    before = source.data.copy()
    encoded = px.color.linear_to_gamma(source, gamma="RED-Log3G10")
    fused = px.color.rgb_to_rgb(source, output_gamma="RED-Log3G10")
    expected = _log3g10_encode(values.astype(np.float64))
    np.testing.assert_allclose(_rgb_values(encoded)[:, 0], expected, rtol=0.0, atol=2e-7)
    np.testing.assert_array_equal(_rgb_values(encoded), _rgb_values(fused))
    np.testing.assert_array_equal(
        _rgb_values(px.color.linear_to_gamma(_frame((-0.01, 0.0, 0.18, 1.0)), gamma="RED-Log3G10"))[:, 0].round(6),
        np.asarray((0.0, 0.091551, 0.333333, 0.493449), dtype=np.float32),
    )
    grid = np.linspace(-0.25, 2.0, 4097, dtype=np.float32)
    grid = np.unique(np.concatenate((grid, np.asarray((below, -0.01, above), dtype=np.float32))))
    encoded_grid = _rgb_values(px.color.linear_to_gamma(_frame(grid), gamma="RED-Log3G10"))[:, 0]
    assert np.all(np.diff(encoded_grid) >= 0.0)
    assert encoded_grid[0] < 0.0 and encoded_grid[-1] > np.float32(0.493449)
    assert cp.array_equal(encoded.data[..., 0], source.data[..., 0])
    assert cp.array_equal(source.data, before)
    assert (encoded.colorspace, encoded.gamma, encoded.channels, encoded.matrix) == (
        "ACEScg",
        "RED-Log3G10",
        source.channels,
        None,
    )
    assert encoded.data.dtype == cp.float32


def test_red_log3g10_decode_and_both_round_trips_match_the_independent_inverse() -> None:
    """v1-red-tokens acceptance 58-59: decode the unique inverse at zero and preserve signed overshoot."""
    below = np.nextafter(np.float32(0.0), np.float32(-np.inf))
    above = np.nextafter(np.float32(0.0), np.float32(np.inf))
    encoded_values = np.asarray(
        (-1.0, -0.25, below, 0.0, above, 0.09155149, 0.3333329, 0.49344853, 1.0), dtype=np.float32
    )
    decoded = _rgb_values(px.color.gamma_to_linear(_frame(encoded_values, gamma="RED-Log3G10"), gamma="RED-Log3G10"))[
        :, 0
    ]
    expected = _log3g10_decode(encoded_values.astype(np.float64))
    np.testing.assert_allclose(decoded, expected, rtol=2e-6, atol=2e-7)
    assert decoded[3] == np.float32(-0.01)
    # CUDA powf may land one ULP from the correctly rounded host float64 oracle at encoded 1.0.  Keep this
    # high-domain guard separate from the legitimate round-trip tolerance so a rounded decode divisor is rejected.
    actual_anchor_bits = int(decoded[-1:].view(np.uint32)[0])
    oracle_anchor_bits = int(np.asarray((expected[-1],), dtype=np.float32).view(np.uint32)[0])
    assert abs(actual_anchor_bits - oracle_anchor_bits) <= 1
    linear_values = np.asarray((-0.25, -0.0101, -0.01, 0.0, 0.18, 1.0, 2.0), dtype=np.float32)
    restored = px.color.gamma_to_linear(
        px.color.linear_to_gamma(_frame(linear_values), gamma="RED-Log3G10"), gamma="RED-Log3G10"
    )
    np.testing.assert_allclose(_rgb_values(restored)[:, 0], linear_values, rtol=2e-6, atol=2e-7)
    reencoded = px.color.linear_to_gamma(
        px.color.gamma_to_linear(_frame(encoded_values, gamma="RED-Log3G10"), gamma="RED-Log3G10"),
        gamma="RED-Log3G10",
    )
    np.testing.assert_allclose(_rgb_values(reencoded)[:, 0], encoded_values, rtol=2e-6, atol=2e-7)


def test_redlogfilm_is_cineon_bit_identical_with_independent_metadata_and_mirror_behavior() -> None:
    """v1-red-tokens acceptance 60-61: preserve the Cineon mirror bits under REDlogFilm metadata."""
    negative_zero_side = np.float32(-1e-7)
    positive_zero_side = np.float32(1e-7)
    values = np.asarray((-1.5, -0.18, negative_zero_side, 0.0, positive_zero_side, 0.18, 1.0, 2.0), dtype=np.float32)
    source = _frame(values, auxiliary=True)
    red = px.color.linear_to_gamma(source, gamma="REDlogFilm")
    cineon = px.color.linear_to_gamma(source, gamma="Cineon")
    fused = px.color.rgb_to_rgb(source, output_gamma="REDlogFilm")
    expected = _cineon_encode(values.astype(np.float64))
    np.testing.assert_allclose(_rgb_values(red)[:, 0], expected, rtol=0.0, atol=2e-7)
    np.testing.assert_array_equal(_rgb_values(red), _rgb_values(cineon))
    np.testing.assert_array_equal(_rgb_values(red), _rgb_values(fused))
    anchors = _rgb_values(px.color.linear_to_gamma(_frame((0.0, 0.18, 1.0)), gamma="REDlogFilm"))[:, 0]
    np.testing.assert_allclose(anchors, (0.0928641251, 0.4573196131, 0.6695992180), rtol=0.0, atol=2e-7)
    assert _rgb_values(red)[2, 0] < np.float32(-0.092864)
    assert _rgb_values(red)[3, 0] > np.float32(0.092864)
    restored = px.color.gamma_to_linear(red, gamma="REDlogFilm")
    expected_decoded = _cineon_decode(_rgb_values(red).astype(np.float64))
    np.testing.assert_allclose(_rgb_values(restored), expected_decoded, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(_rgb_values(restored)[:, 0], values, rtol=2e-6, atol=2e-7)

    encoded_zero_below = np.nextafter(np.float32(0.0), np.float32(-np.inf))
    encoded_zero_above = np.nextafter(np.float32(0.0), np.float32(np.inf))
    decode_values = np.asarray(
        (
            -0.75,
            encoded_zero_below,
            0.0,
            encoded_zero_above,
            0.0928641251,
            0.4573196131,
            0.6695992180,
            0.75,
        ),
        dtype=np.float32,
    )
    red_decoded = px.color.gamma_to_linear(
        _frame(decode_values, gamma="REDlogFilm"),
        gamma="REDlogFilm",
    )
    cineon_decoded = px.color.gamma_to_linear(
        _frame(decode_values, gamma="Cineon"),
        gamma="Cineon",
    )
    np.testing.assert_array_equal(
        _rgb_values(red_decoded).view(np.uint32),
        _rgb_values(cineon_decoded).view(np.uint32),
    )
    assert cp.array_equal(red.data[..., 0], source.data[..., 0])
    assert red_decoded.gamma == cineon_decoded.gamma == "linear"
    assert (red.gamma, cineon.gamma, red.channels, red.matrix) == ("REDlogFilm", "Cineon", source.channels, None)


def test_redwidegamutrgb_basis_neutral_vendor_matrices_and_native_row_match_independent_oracles() -> None:
    """v1-red-tokens acceptance 62 and 64: derive the normalized gamut, Bradford conversion, and native row."""
    from pixtreme._core.colorspace import _COLORSPACE_DEFINITIONS

    definition = _GAMUTS["REDWideGamutRGB"]
    assert _COLORSPACE_DEFINITIONS["REDWideGamutRGB"] == definition
    rgb_to_xyz = _rgb_to_xyz(*definition)
    vendor_xyz = np.asarray(
        ((0.735275, 0.068609, 0.146571), (0.286694, 0.842979, -0.129673), (-0.079681, -0.347343, 1.516081))
    )
    np.testing.assert_allclose(rgb_to_xyz, vendor_xyz, rtol=0.0, atol=1e-6)
    values = np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 1.0)))
    source = _frame(values, colorspace="REDWideGamutRGB")
    converted = px.color.rgb_to_rgb(source, output_colorspace="ACES2065-1")
    expected = values @ _conversion(definition, _ACES2065).T
    np.testing.assert_allclose(_rgb_values(converted), expected, rtol=0.0, atol=6e-6)
    np.testing.assert_allclose(_rgb_values(converted)[:3], _VENDOR_ACES["REDWideGamutRGB"].T, rtol=0.0, atol=2e-4)
    np.testing.assert_allclose(_rgb_values(converted)[3], (1.0, 1.0, 1.0), rtol=0.0, atol=6e-6)
    grayscale = px.color.rgb_to_grayscale(source, colorspace="REDWideGamutRGB", gamma="linear", matrix="native")
    np.testing.assert_allclose(px.io.to_array(grayscale).get()[0, :, 0], values @ rgb_to_xyz[1], rtol=0.0, atol=6e-6)
    assert (converted.colorspace, converted.gamma, converted.matrix) == ("ACES2065-1", "linear", None)


@pytest.mark.parametrize("colorspace", tuple(_GAMUTS)[1:])
def test_legacy_red_gamut_coordinates_public_aces_matrices_and_native_rows_are_consistent(colorspace: str) -> None:
    """v1-red-tokens acceptance 63-64: reverse published ACES matrices and use the resulting D65 primaries."""
    from pixtreme._core.colorspace import _COLORSPACE_DEFINITIONS

    definition = _GAMUTS[colorspace]
    primaries, white = definition
    assert _COLORSPACE_DEFINITIONS[colorspace] == definition
    adapted_xyz = _rgb_to_xyz(*_ACES2065) @ _VENDOR_ACES[colorspace]
    native_xyz = _adaptation(_ACES_WHITE, _D65) @ adapted_xyz
    derived_xy = tuple((float(column[0] / column.sum()), float(column[1] / column.sum())) for column in native_xyz.T)
    np.testing.assert_allclose(derived_xy, primaries, rtol=0.0, atol=1e-10)
    values = np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (-0.25, 0.18, 1.5)))
    source = _frame(values, colorspace=colorspace)
    converted = px.color.rgb_to_rgb(source, output_colorspace="ACES2065-1")
    expected = values @ _conversion(definition, _ACES2065).T
    np.testing.assert_allclose(_rgb_values(converted), expected, rtol=0.0, atol=6e-6)
    np.testing.assert_allclose(_rgb_values(converted)[:3], _VENDOR_ACES[colorspace].T, rtol=0.0, atol=3e-5)
    grayscale = px.color.rgb_to_grayscale(source, colorspace=colorspace, gamma="linear", matrix="native")
    np.testing.assert_allclose(
        px.io.to_array(grayscale).get()[0, :, 0], values @ _rgb_to_xyz(primaries, white)[1], rtol=0.0, atol=6e-6
    )


@pytest.mark.parametrize("target", ("Rec.709", "ACES2065-1"))
def test_redwidegamut_log3g10_frame_converts_end_to_end_with_auxiliary_bits(target: str) -> None:
    """v1-red-tokens acceptance 65: fuse independent Log3G10 decode and gamut conversion without clipping."""
    linear_rgb = np.asarray(((-0.25, 0.18, 1.5), (0.18, 1.25, -0.05)), dtype=np.float64)
    encoded_rgb = _log3g10_encode(linear_rgb).astype(np.float32)
    source = _frame(encoded_rgb, colorspace="REDWideGamutRGB", gamma="RED-Log3G10", auxiliary=True)
    before = source.data.copy()
    converted = px.color.rgb_to_rgb(source, output_colorspace=target, output_gamma="linear")
    target_definition = _REC709 if target == "Rec.709" else _ACES2065
    expected = (
        _log3g10_decode(_rgb_values(source).astype(np.float64))
        @ _conversion(_GAMUTS["REDWideGamutRGB"], target_definition).T
    )
    np.testing.assert_allclose(_rgb_values(converted), expected, rtol=2e-6, atol=6e-6)
    assert cp.array_equal(converted.data[..., 0], source.data[..., 0])
    assert cp.array_equal(source.data, before)
    assert (converted.colorspace, converted.gamma, converted.channels, converted.matrix) == (
        target,
        "linear",
        source.channels,
        None,
    )
    assert np.any(_rgb_values(converted) < 0.0) and np.any(_rgb_values(converted) > 1.0)


@pytest.mark.parametrize(
    ("colorspace", "gamma"),
    (
        ("DRAGONcolor", "REDlogFilm"),
        ("DRAGONcolor2", "linear"),
        ("REDcolor2", "REDlogFilm"),
        ("REDcolor3", "linear"),
        ("REDcolor4", "REDlogFilm"),
    ),
)
@pytest.mark.parametrize("target", ("Rec.709", "ACES2065-1"))
def test_legacy_red_frames_convert_with_independent_transfer_pairing_and_auxiliary_bits(
    colorspace: str, gamma: str, target: str
) -> None:
    """v1-red-tokens acceptance 66-67: convert legacy gamut/transfer pairings without inference or clipping."""
    linear_rgb = np.asarray(((-0.25, 0.18, 1.5), (0.18, 1.25, -0.05)), dtype=np.float64)
    source_rgb = linear_rgb if gamma == "linear" else _cineon_encode(linear_rgb)
    source = _frame(source_rgb.astype(np.float32), colorspace=colorspace, gamma=gamma, auxiliary=True)
    converted = px.color.rgb_to_rgb(source, output_colorspace=target, output_gamma="linear")
    target_definition = _REC709 if target == "Rec.709" else _ACES2065
    decoded = _rgb_values(source).astype(np.float64)
    if gamma == "REDlogFilm":
        decoded = _cineon_decode(decoded)
    expected = decoded @ _conversion(_GAMUTS[colorspace], target_definition).T
    np.testing.assert_allclose(_rgb_values(converted), expected, rtol=2e-6, atol=6e-6)
    assert cp.array_equal(converted.data[..., 0], source.data[..., 0])
    assert (converted.colorspace, converted.gamma, converted.channels, converted.matrix) == (
        target,
        "linear",
        source.channels,
        None,
    )


def test_arri_renames_and_existing_token_bits_remain_at_the_pre_red_baseline() -> None:
    """v1-red-tokens acceptance 68-69: rename ARRI metadata while retaining established transfer bits."""
    # Provenance: these uint32 fixtures were captured before RED production changes from full commit
    # e487a84083d555da89bf29a95b9598974d2dbe89.  At that commit the ARRI tokens used the old ``LogC3`` and
    # ``LogC4`` spellings; the second tuple item below records the corresponding new canonical fixture label.
    # Reproduce from the repository root with the existing project environment and a detached baseline worktree:
    #
    #   git worktree add <baseline-path> e487a84083d555da89bf29a95b9598974d2dbe89
    #   cd <baseline-path>
    #   UV_PROJECT_ENVIRONMENT=<main-repo>/.venv PYTHONPATH=<baseline-path>/src uv run --no-sync python - <<'PY'
    #   import cupy as cp
    #   import numpy as np
    #   import pixtreme as px
    #
    #   fixtures = (
    #       ("LogC3", "ARRI-LogC3", (-0.25, 0.010590990073978901, 0.0, 0.18000000715255737, 1.0, 1.5)),
    #       ("LogC4", "ARRI-LogC4", (-0.25, -0.018056996166706085, 0.0, 0.18000000715255737, 1.0, 1.5)),
    #       ("S-Log3", "S-Log3", (-0.25, 0.011250000447034836, 0.0, 0.18000000715255737, 1.0, 1.5)),
    #       (
    #           "Blackmagic-Film-Gen-5",
    #           "Blackmagic-Film-Gen-5",
    #           (-0.25, 0.004999999888241291, 0.0, 0.18000000715255737, 1.0, 1.5),
    #       ),
    #   )
    #   for baseline_token, canonical_label, linear in fixtures:
    #       values = np.asarray(linear, dtype=np.float32)
    #       rgb = np.repeat(values[:, None], 3, axis=1)[None, :, :]
    #       frame = px.io.from_array(
    #           cp.asarray(rgb), colorspace="ACEScg", gamma="linear", channels="RGB"
    #       )
    #       encoded = px.color.linear_to_gamma(frame, gamma=baseline_token)
    #       red = px.io.to_array(encoded).get()[0, :, 0]
    #       print(canonical_label, red.view(np.uint32).tolist())
    #   PY
    #
    # The exact decode fixtures for these established transfers remain pinned by
    # ``tests/test_blackmagic_tokens_spec.py::test_existing_camera_log_gpu_bits_remain_identical_to_pre_blackmagic_baseline``;
    # ARRI-LogC4 also has a dedicated encode/decode bit fixture in
    # ``tests/test_arri_tokens_spec.py::test_logc4_gpu_bits_remain_identical_to_the_pre_arri_token_baseline``.
    fixtures = (
        (
            "ARRI-LogC3",
            (-0.25, 0.010590990073978901, 0.0, 0.18000000715255737, 1.0, 1.5),
            (3214926503, 1041842171, 1035866836, 1053307405, 1058149604, 1058874277),
        ),
        (
            "ARRI-LogC4",
            (-0.25, -0.018056996166706085, 0.0, 0.18000000715255737, 1.0, 1.5),
            (3221400801, 2995575096, 1035874188, 1049528807, 1054532562, 1055775089),
        ),
        (
            "S-Log3",
            (-0.25, 0.011250000447034836, 0.0, 0.18000000715255737, 1.0, 1.5),
            (3217556478, 1043030190, 1035874188, 1053963405, 1058575679, 1059324708),
        ),
        (
            "Blackmagic-Film-Gen-5",
            (-0.25, 0.004999999888241291, 0.0, 0.18000000715255737, 1.0, 1.5),
            (3221044577, 1040783570, 1035820719, 1053057585, 1057476139, 1058064820),
        ),
    )
    for gamma, linear, expected_bits in fixtures:
        encoded = px.color.linear_to_gamma(_frame(linear), gamma=gamma)
        np.testing.assert_array_equal(
            _rgb_values(encoded)[:, 0].view(np.uint32), np.asarray(expected_bits, dtype=np.uint32)
        )
        assert encoded.gamma == gamma
    assert px.color.linear_to_gamma(_frame((0.18,)), gamma="logc4").gamma == "ARRI-LogC4"


def test_red_dpx_transfer_codes_cover_logarithmic_and_printing_density_headers(tmp_path: Path) -> None:
    """v1-red-tokens acceptance 70: write Log3G10 as logarithmic and REDlogFilm/Cineon as printing density."""
    from pixtreme._io.formats.dpx import _dpx_transfer_from_gamma

    expected = {"RED-Log3G10": 3, "REDlogFilm": 1, "Cineon": 1, "ARRI-LogC3": 3, "ARRI-LogC4": 3}
    assert {gamma: _dpx_transfer_from_gamma(gamma) for gamma in expected} == expected
    pixels = cp.asarray([[[0.18, 0.18, 0.18]]], dtype=cp.float32)
    headers: dict[str, int] = {}
    for gamma, transfer in expected.items():
        frame = px.io.from_array(pixels.copy(), colorspace="Rec.709", gamma=gamma, channels="RGB")
        path = tmp_path / f"{gamma}.dpx"
        px.io.write_image(path, frame)
        headers[gamma] = path.read_bytes()[801]
        assert headers[gamma] == transfer
    assert headers["REDlogFilm"] == headers["Cineon"]


def test_red_token_reference_and_public_docstrings_are_synchronized() -> None:
    """v1-red-tokens acceptance 72; v1-canon-tokens acceptance 93; v1-panasonic-tokens acceptance 112;
    v1-vendor-a-tokens acceptance 161; v1-vendor-b-tokens acceptance 188;
    GitHub #29: synchronize public prose.
    """
    token_reference = (ROOT / "docs_site" / "tokens.md").read_text(encoding="utf-8")
    requirements = require_repo_file("docs/requirements.md").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8").split("## 1.2.1", maxsplit=1)[0]
    for token in (*_COLORSPACES[16:22], *_GAMMAS[11:17], "Cineon"):
        assert f"`{token}`" in token_reference
    for fragment in (
        "a = 0.224282",
        "b = 155.975327",
        "c = 0.01",
        "g = 15.1927",
        "0.0107977516232771",
        "sign-preserving mirror",
        "Bradford",
        "REDWideGamutRGB",
        "DRAGONcolor",
    ):
        assert fragment in token_reference
    assert "27 Colorspace" in requirements
    assert "33 Gamma" in requirements
    assert "188 canonical tokens" in requirements
    assert "REDWideGamutRGB" in changelog and "RED-Log3G10" in changelog and "REDlogFilm" in changelog
    assert "ARRI-LogC3" in changelog and "ARRI-LogC4" in changelog and "runtime input" in changelog
    for operation in (px.color.rgb_to_rgb, px.color.gamma_to_linear, px.color.linear_to_gamma):
        docstring = inspect.getdoc(operation)
        assert docstring is not None
        assert "RED-Log3G10" in docstring
        assert "REDlogFilm" in docstring
        assert "ARRI-LogC3" in docstring
        assert "ARRI-LogC4" in docstring
