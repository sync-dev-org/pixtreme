"""Specification tests for DJI and Fujifilm colorspace and transfer tokens."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from decimal import Decimal, getcontext
from hashlib import sha256
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


@dataclass(frozen=True)
class _Curve:
    a: np.float64
    b: np.float64
    c: np.float64
    d: np.float64
    e: np.float64
    f: np.float64
    x: np.float64
    encoded_cut: np.float64
    printed_x: np.float64
    printed_encoded_cut: np.float64
    anchor_codes: tuple[float, float, float]
    inverse_anchors: tuple[float, float]
    root_start: str


def _curve(
    a: str,
    b: str,
    c: str,
    d: str,
    e: str,
    f: str,
    x: str,
    encoded_cut: str,
    printed_x: str,
    printed_encoded_cut: str,
    anchor_codes: tuple[float, float, float],
    inverse_anchors: tuple[float, float],
    root_start: str,
) -> _Curve:
    return _Curve(
        *(np.float64(value) for value in (a, b, c, d, e, f, x, encoded_cut, printed_x, printed_encoded_cut)),
        anchor_codes,
        inverse_anchors,
        root_start,
    )


_CURVES = {
    "D-Log": _curve(
        "0.9892",
        "0.0108",
        "0.256663",
        "0.584555",
        "6.025",
        "0.0929",
        "0.007827156200341792",
        "0.1400586161070593",
        "0.0078",
        "0.14",
        (95.0367, 407.9361, 586.1221),
        (-0.0154190871369295, 41.9993933040331),
        "0.01",
    ),
    "F-Log": _curve(
        "0.555556",
        "0.009468",
        "0.344676",
        "0.790453",
        "8.735631",
        "0.092864",
        "0.0005663467969879701",
        "0.09781139663651882",
        "0.00089",
        "0.100537775223865",
        (94.9999, 469.8828, 705.3619),
        (-0.0106304856512369, 7.28132488048849),
        "0.001",
    ),
    "F-Log2": _curve(
        "5.555556",
        "0.064829",
        "0.245281",
        "0.384316",
        "8.799461",
        "0.092864",
        "0.0008888881429483923",
        "0.10068573654723681",
        "0.000889",
        "0.100686685370811",
        (94.9999, 400.0004, 569.9464),
        (-0.0105533736668644, 58.2508740195611),
        "0.001",
    ),
}

_BRANCH_FIXTURES = {
    ("D-Log", "encode"): (
        (0x3C003D78, 0x3C003D79, 0x3C003D7A),
        (0x3E0F6B86, 0x3E0F6B87, 0x3E0F6B87),
        (1.435e-8, 1.370e-7, 1.315e-7),
    ),
    ("D-Log", "decode"): (
        (0x3E0F6B86, 0x3E0F6B87, 0x3E0F6B88),
        (0x3C003D77, 0x3C003D7A, 0x3C003D7D),
        (1.290e-9, 2.020e-8, 2.049e-8),
    ),
    ("F-Log", "encode"): (
        (0x3A1476E3, 0x3A1476E4, 0x3A1476E5),
        (0x3DC85157, 0x3DC85157, 0x3DC85158),
        (7.404e-9, 2.553e-7, 2.558e-7),
    ),
    ("F-Log", "decode"): (
        (0x3DC85156, 0x3DC85157, 0x3DC85158),
        (0x3A1476CF, 0x3A1476DD, 0x3A1476EC),
        (1.390e-10, 2.064e-8, 2.064e-8),
    ),
    ("F-Log2", "encode"): (
        (0x3A690445, 0x3A690446, 0x3A690447),
        (0x3DCE3453, 0x3DCE3453, 0x3DCE3453),
        (7.208e-9, 9.320e-8, 9.270e-8),
    ),
    ("F-Log2", "decode"): (
        (0x3DCE3452, 0x3DCE3453, 0x3DCE3454),
        (0x3A69043A, 0x3A690449, 0x3A690458),
        (2.185e-10, 9.925e-9, 9.931e-9),
    ),
}

_EXPLICIT_FIXTURES = {
    ("D-Log", "encode"): (0x3BFFEB07, 0x3E0F506E, 1.380e-8),
    ("D-Log", "decode"): (0x3E0F6406, 0x3C00298D, 1.382e-9),
    ("F-Log", "encode"): (0x3A378034, 0x3DCAA2C7, 2.527e-7),
    ("F-Log", "decode"): (0x3DCAC083, 0x3A394368, 2.074e-8),
    ("F-Log2", "encode"): (0x3A69086D, 0x3DCE3499, 9.353e-8),
    ("F-Log2", "decode"): (0x3DCE3491, 0x3A6907F1, 9.949e-9),
}

_D65 = (0.3127, 0.3290)
_ACES_WHITE = (0.32168, 0.33767)
_REC709 = (((0.640, 0.330), (0.300, 0.600), (0.150, 0.060)), _D65)
_ACES2065 = (((0.7347, 0.2653), (0.0000, 1.0000), (0.0001, -0.0770)), _ACES_WHITE)
_D_GAMUT = (((0.71, 0.31), (0.21, 0.88), (0.09, -0.08)), _D65)
_F_GAMUT_C = (((0.7347, 0.2653), (0.0263, 0.9737), (0.1173, -0.0224)), _D65)
_GAMUTS = {"D-Gamut": _D_GAMUT, "F-Gamut-C": _F_GAMUT_C}
_RGB_TO_XYZ_TABLES = {
    "D-Gamut": np.asarray(
        (
            (0.648171968633815, 0.194058149820755, 0.108225808597101),
            (0.283004662361243, 0.813196056391736, -0.096200718752979),
            (-0.018258365313629, -0.083167778494609, 1.190483894568116),
        )
    ),
    "F-Gamut-C": np.asarray(
        (
            (0.789274967789181, 0.020040229879954, 0.141140729382536),
            (0.285007008240737, 0.741945697114496, -0.026952705355233),
            (0.0, 0.0, 1.089057750759878),
        )
    ),
}
_BRADFORD = np.asarray(
    ((0.8951, 0.2664, -0.1614), (-0.7502, 1.7135, 0.0367), (0.0389, -0.0685, 1.0296)),
    dtype=np.float64,
)
_CAT02 = np.asarray(
    ((0.7328, 0.4296, -0.1624), (-0.7036, 1.6975, 0.0061), (0.0030, 0.0136, 0.9834)),
    dtype=np.float64,
)
_CAT02_TO_ACES = {
    "D-Gamut": np.asarray(
        (
            (0.691279245585754, 0.214382527745956, 0.094338226668290),
            (0.066222403766775, 1.011616080187598, -0.077838483954373),
            (-0.017298541034174, -0.077378850101268, 1.094677391135443),
        )
    ),
    "F-Gamut-C": np.asarray(
        (
            (0.84089501562825, 0.02752756409115, 0.13157742028060),
            (0.00092026357010, 1.00739011387088, -0.00831037744099),
            (-0.00055982840886, -0.00077644453594, 1.00133627294480),
        )
    ),
}


def _encode(values: np.ndarray, curve: _Curve, *, cut: np.float64 | None = None) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    result = np.empty_like(source)
    threshold = curve.x if cut is None else cut
    linear = source < threshold
    result[linear] = curve.e * source[linear] + curve.f
    result[~linear] = curve.c * np.log10(curve.a * source[~linear] + curve.b) + curve.d
    return result


def _decode(values: np.ndarray, curve: _Curve, *, cut: np.float64 | None = None) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    result = np.empty_like(source)
    threshold = curve.encoded_cut if cut is None else cut
    linear = source < threshold
    result[linear] = (source[linear] - curve.f) / curve.e
    result[~linear] = (np.float64(10.0) ** ((source[~linear] - curve.d) / curve.c) - curve.b) / curve.a
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


def _from_bits(bits: tuple[int, ...] | int) -> np.ndarray:
    values = (bits,) if isinstance(bits, int) else bits
    return np.asarray(values, dtype=np.uint32).view(np.float32)


def _xy_to_xyz(xy: tuple[float, float]) -> np.ndarray:
    x, y = xy
    return np.asarray((x / y, 1.0, (1.0 - x - y) / y), dtype=np.float64)


def _rgb_to_xyz(
    definition: tuple[tuple[tuple[float, float], tuple[float, float], tuple[float, float]], tuple[float, float]],
) -> np.ndarray:
    primaries, white = definition
    unscaled = np.asarray(
        (tuple(x / y for x, y in primaries), (1.0, 1.0, 1.0), tuple((1.0 - x - y) / y for x, y in primaries)),
        dtype=np.float64,
    )
    return unscaled @ np.diag(np.linalg.solve(unscaled, _xy_to_xyz(white)))


def _adaptation(source: tuple[float, float], target: tuple[float, float], cone: np.ndarray) -> np.ndarray:
    source_cones = cone @ _xy_to_xyz(source)
    target_cones = cone @ _xy_to_xyz(target)
    return np.linalg.inv(cone) @ np.diag(target_cones / source_cones) @ cone


def _conversion(
    source: tuple[tuple[tuple[float, float], tuple[float, float], tuple[float, float]], tuple[float, float]],
    target: tuple[tuple[tuple[float, float], tuple[float, float], tuple[float, float]], tuple[float, float]],
    *,
    cone: np.ndarray = _BRADFORD,
) -> np.ndarray:
    return np.linalg.inv(_rgb_to_xyz(target)) @ _adaptation(source[1], target[1], cone) @ _rgb_to_xyz(source)


def _adjacent_float32(center: np.float32, radius: int = 3_000) -> np.ndarray:
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


def _max_downward(values: np.ndarray) -> float:
    prior_maximum = np.maximum.accumulate(values)
    return float(np.max(np.maximum(prior_maximum[:-1] - values[1:], np.float32(0.0)), initial=0.0))


def _linear_sets(gamma: str) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray, np.ndarray]:
    """Return the four disjoint reflectance input sets: dense grids, cut fixtures, cut window, anchors."""
    curve = _CURVES[gamma]
    fixture = np.concatenate(
        (_from_bits(_BRANCH_FIXTURES[(gamma, "encode")][0]), _from_bits(_EXPLICIT_FIXTURES[(gamma, "encode")][0]))
    )
    window = _adjacent_float32(np.float32(curve.x))
    anchors = np.asarray((0.0, 0.18, 0.9, 1.0), dtype=np.float32)
    excluded = np.concatenate((fixture, window, anchors))
    dense = tuple(
        grid[~np.isin(grid, excluded)]
        for grid in (
            np.linspace(-0.5, 64.0, 400_001, dtype=np.float64).astype(np.float32),
            np.linspace(0.0, 0.02, 200_001, dtype=np.float64).astype(np.float32),
        )
    )
    return dense, fixture, window, anchors


def _encoded_sets(gamma: str) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray, np.ndarray]:
    """Return the four disjoint encoded input sets: dense grid, cut fixtures, cut window, anchors."""
    curve = _CURVES[gamma]
    fixture = np.concatenate(
        (_from_bits(_BRANCH_FIXTURES[(gamma, "decode")][0]), _from_bits(_EXPLICIT_FIXTURES[(gamma, "decode")][0]))
    )
    window = _adjacent_float32(np.float32(curve.encoded_cut))
    anchors = np.asarray((0.0, 1.0, 1.5), dtype=np.float32)
    excluded = np.concatenate((fixture, window, anchors))
    grid = np.linspace(-0.5, 1.5, 200_001, dtype=np.float64).astype(np.float32)
    return (grid[~np.isin(grid, excluded)],), fixture, window, anchors


def test_vendor_a_tokens_extend_only_canonical_vocabulary_and_public_surfaces() -> None:
    """v1-vendor-a-tokens acceptance 140-141; v1-vendor-b-tokens acceptance 166-167:
    expose the current canonical tokens without static aliases.
    """
    assert get_args(px.core.Colorspace) == _COLORSPACES
    assert get_args(px.core.Gamma) == _GAMMAS
    assert len(_ALIASES) == 30
    assert sum(len(get_args(alias)) for alias in _ALIASES) == 188
    assert _literal_strings(get_type_hints(px.color.linear_to_gamma)["gamma"]) == _GAMMAS
    assert _literal_strings(get_type_hints(px.color.rgb_to_rgb)["input_colorspace"]) == _COLORSPACES
    assert _literal_strings(get_type_hints(px.color.rgb_to_rgb)["output_gamma"]) == _GAMMAS
    for colorspace, gamma in (("D-Gamut", "D-Log"), ("F-Gamut-C", "F-Log"), ("F-Gamut-C", "F-Log2")):
        frame = _frame((0.18,), colorspace=colorspace, gamma=gamma)
        assert (frame.colorspace, frame.gamma) == (colorspace, gamma)
        assert f"colorspace={colorspace!r}" in repr(frame)
        assert f"gamma={gamma!r}" in repr(frame)

    from pixtreme._core.vocabulary import _PERMANENT_TOKEN_ALIASES

    assert len(_PERMANENT_TOKEN_ALIASES) == 4
    new_tokens = {"D-Gamut", "F-Gamut-C", "D-Log", "F-Log", "F-Log2"}
    assert not any(token in new_tokens for alias in _PERMANENT_TOKEN_ALIASES for token in alias)


def test_vendor_a_token_keys_alias_boundaries_and_fail_fast_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """v1-vendor-a-tokens acceptance 142 and 160: normalize four separators and reject raw invalid inputs."""
    from pixtreme._core.validation import _normalized_closed_token

    translation = str.maketrans("", "", " .-_")
    expected = {
        "D-Gamut": "dgamut",
        "F-Gamut-C": "fgamutc",
        "D-Log": "dlog",
        "F-Log": "flog",
        "F-Log2": "flog2",
    }
    assert {token: token.translate(translation).casefold() for token in expected} == expected
    assert len({token.translate(translation).casefold() for token in _COLORSPACES}) == len(_COLORSPACES)
    assert len({token.translate(translation).casefold() for token in _GAMMAS}) == len(_GAMMAS)
    accepted = (
        ("DLog", "gamma", _GAMMAS, "D-Log"),
        ("d_log", "gamma", _GAMMAS, "D-Log"),
        ("F Log2", "gamma", _GAMMAS, "F-Log2"),
        ("flog2", "gamma", _GAMMAS, "F-Log2"),
        ("F-Gamut C", "colorspace", _COLORSPACES, "F-Gamut-C"),
        ("fgamutc", "colorspace", _COLORSPACES, "F-Gamut-C"),
    )
    for spelling, axis, vocabulary, canonical in accepted:
        assert _normalized_closed_token(spelling, axis=axis, accepted=vocabulary) == canonical
    rejected = (
        "DJI D-Log",
        "DJI D-Gamut",
        "Fujifilm F-Log",
        "Fujifilm F-Log2",
        "Fujifilm F-Gamut C",
        "F-Log2 C",
        "F-Log2C",
        "F-Gamut",
        "D-Log M",
        "DLogM",
    )
    for value in rejected:
        vocabulary = _COLORSPACES if "Gamut" in value else _GAMMAS
        with pytest.raises(ValueError):
            _normalized_closed_token(value, axis="token", accepted=vocabulary)

    import pixtreme._color.semantics as semantics

    monkeypatch.setattr(semantics, "_color_semantics_kernel", lambda: pytest.fail("GPU work must not start"))
    source = _frame((0.18,))
    for value in ("DJI D-Log", 17):
        with pytest.raises(ValueError) as captured:
            px.color.linear_to_gamma(source, gamma=value)
        message = str(captured.value)
        assert message.index("why=") < message.index("what=") < message.index("how=")
        assert f"received gamma={value!r}" in message
        assert repr(_GAMMAS) in message
        assert "DJI D-Log" not in message.replace(f"received gamma={value!r}", "")


@pytest.mark.parametrize(
    ("gamma", "acceptance"),
    (("D-Log", 143), ("F-Log", 146), ("F-Log2", 149)),
)
def test_vendor_a_encode_matches_dense_oracle_branch_fixtures_and_anchors(gamma: str, acceptance: int) -> None:
    """v1-vendor-a-tokens acceptance 143, 146, 149: encode unscaled reflectance with intersection cuts."""
    del acceptance
    curve = _CURVES[gamma]
    dense, _, _, _ = _linear_sets(gamma)
    assert tuple(len(grid) for grid in dense) < (400_001, 200_001)
    for values in dense:
        actual = _rgb_values(px.color.linear_to_gamma(_frame(values), gamma=gamma))[:, 0]
        expected = _encode(values.astype(np.float64), curve)
        np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=3e-7)
        assert np.all(np.diff(actual) >= 0.0)

    input_bits, expected_bits, envelopes = _BRANCH_FIXTURES[(gamma, "encode")]
    inputs = _from_bits(input_bits)
    actual = _rgb_values(px.color.linear_to_gamma(_frame(inputs), gamma=gamma))[:, 0]
    expected = _from_bits(expected_bits)
    assert np.all(np.abs(actual.astype(np.float64) - expected.astype(np.float64)) <= np.asarray(envelopes))
    np.testing.assert_array_equal(
        _encode(inputs.astype(np.float64), curve).astype(np.float32).view(np.uint32), expected_bits
    )

    explicit_input, explicit_expected, explicit_envelope = _EXPLICIT_FIXTURES[(gamma, "encode")]
    actual_explicit = _rgb_values(px.color.linear_to_gamma(_frame(_from_bits(explicit_input)), gamma=gamma))[0, 0]
    assert abs(float(actual_explicit) - float(_from_bits(explicit_expected)[0])) <= explicit_envelope

    anchors = np.asarray((0.0, 0.18, 0.9), dtype=np.float32)
    encoded = px.color.linear_to_gamma(_frame(anchors, auxiliary=True), gamma=gamma)
    codes = _rgb_values(encoded)[:, 0].astype(np.float64) * np.float64(1023.0)
    np.testing.assert_allclose(codes, curve.anchor_codes, rtol=0.0, atol=2.1e-4)
    expected_codes = {"D-Log": (95, 408, 586), "F-Log": (95, 470, 705), "F-Log2": (95, 400, 570)}
    np.testing.assert_array_equal(np.rint(codes).astype(np.int64), expected_codes[gamma])
    assert encoded.gamma == gamma


@pytest.mark.parametrize(
    ("gamma", "acceptance"),
    (("D-Log", 144), ("F-Log", 147), ("F-Log2", 150)),
)
def test_vendor_a_decode_matches_dense_oracle_branch_fixtures_and_anchors(gamma: str, acceptance: int) -> None:
    """v1-vendor-a-tokens acceptance 144, 147, 150: decode with analytic inverses at intersection cuts."""
    del acceptance
    curve = _CURVES[gamma]
    (values,), _, _, _ = _encoded_sets(gamma)
    assert len(values) < 200_001
    decoded = px.color.gamma_to_linear(_frame(values, gamma=gamma, auxiliary=True), gamma=gamma)
    actual = _rgb_values(decoded)[:, 0]
    expected = _decode(values.astype(np.float64), curve)
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=2e-7)
    assert np.all(np.diff(actual) >= 0.0)

    input_bits, expected_bits, envelopes = _BRANCH_FIXTURES[(gamma, "decode")]
    inputs = _from_bits(input_bits)
    actual_fixture = _rgb_values(px.color.gamma_to_linear(_frame(inputs, gamma=gamma), gamma=gamma))[:, 0]
    expected_fixture = _from_bits(expected_bits)
    assert np.all(
        np.abs(actual_fixture.astype(np.float64) - expected_fixture.astype(np.float64)) <= np.asarray(envelopes)
    )
    explicit_input, explicit_expected, explicit_envelope = _EXPLICIT_FIXTURES[(gamma, "decode")]
    actual_explicit = _rgb_values(
        px.color.gamma_to_linear(_frame(_from_bits(explicit_input), gamma=gamma), gamma=gamma)
    )[0, 0]
    assert abs(float(actual_explicit) - float(_from_bits(explicit_expected)[0])) <= explicit_envelope
    anchor_inputs = np.asarray((0.0, 1.0), dtype=np.float32)
    anchor_actual = _rgb_values(px.color.gamma_to_linear(_frame(anchor_inputs, gamma=gamma), gamma=gamma))[:, 0]
    np.testing.assert_allclose(anchor_actual, curve.inverse_anchors, rtol=3e-6, atol=2e-7)
    np.testing.assert_array_max_ulp(
        anchor_actual[1:], _decode(anchor_inputs.astype(np.float64), curve)[1:].astype(np.float32), maxulp=30
    )
    assert decoded.gamma == "linear"


@pytest.mark.parametrize(
    ("gamma", "acceptance"),
    (("D-Log", 145), ("F-Log", 148), ("F-Log2", 151)),
)
def test_vendor_a_intersection_roots_cut_windows_and_round_trips(gamma: str, acceptance: int) -> None:
    """v1-vendor-a-tokens acceptance 145, 148, 151: reproduce roots, bound seams, gate one-way, round-trip four sets."""
    del acceptance
    curve = _CURVES[gamma]
    getcontext().prec = 70
    a, b, c, d, e, f = (Decimal(str(value)) for value in (curve.a, curve.b, curve.c, curve.d, curve.e, curve.f))
    value = Decimal(curve.root_start)
    ln10 = Decimal(10).ln()
    for _ in range(30):
        g = c * (a * value + b).log10() + d - (e * value + f)
        derivative = c * a / ((a * value + b) * ln10) - e
        value -= g / derivative
    assert derivative < 0
    assert float(value) == float(curve.x)
    assert float(e * value + f) == float(curve.encoded_cut)
    assert abs(c * (a * value + b).log10() + d - (e * value + f)) <= Decimal("2e-60")

    dense_linear, linear_fixture, encode_window, linear_anchors = _linear_sets(gamma)
    dense_encoded, encoded_fixture, decode_window, encoded_anchors = _encoded_sets(gamma)
    encoded = _rgb_values(px.color.linear_to_gamma(_frame(encode_window), gamma=gamma))[:, 0]
    assert _max_downward(encoded) <= 4e-7
    decoded = _rgb_values(px.color.gamma_to_linear(_frame(decode_window, gamma=gamma), gamma=gamma))[:, 0]
    assert _max_downward(decoded) <= 3e-8

    linear_sets = (*dense_linear, linear_fixture, encode_window, linear_anchors)
    encoded_sets = (*dense_encoded, encoded_fixture, decode_window, encoded_anchors)
    for values in linear_sets:
        actual = _rgb_values(px.color.linear_to_gamma(_frame(values), gamma=gamma))[:, 0]
        np.testing.assert_allclose(actual, _encode(values.astype(np.float64), curve), rtol=2e-7, atol=3e-7)
    for values in encoded_sets:
        actual = _rgb_values(px.color.gamma_to_linear(_frame(values, gamma=gamma), gamma=gamma))[:, 0]
        np.testing.assert_allclose(actual, _decode(values.astype(np.float64), curve), rtol=3e-6, atol=2e-7)
    for values in linear_sets:
        encoded_frame = px.color.linear_to_gamma(_frame(values), gamma=gamma)
        restored = _rgb_values(px.color.gamma_to_linear(encoded_frame, gamma=gamma))[:, 0]
        assert np.all(np.isfinite(restored))
        np.testing.assert_allclose(restored, values, rtol=5e-6, atol=2e-7)
    for values in encoded_sets:
        decoded_frame = px.color.gamma_to_linear(_frame(values, gamma=gamma), gamma=gamma)
        reencoded = _rgb_values(px.color.linear_to_gamma(decoded_frame, gamma=gamma))[:, 0]
        assert np.all(np.isfinite(reencoded))
        np.testing.assert_allclose(reencoded, values, rtol=5e-6, atol=2e-7)


@pytest.mark.parametrize("gamma", tuple(_CURVES))
def test_vendor_a_standalone_and_fused_paths_preserve_frame_contracts(gamma: str) -> None:
    """v1-vendor-a-tokens acceptance 152: keep standalone and fused transfer paths identical."""
    curve = _CURVES[gamma]
    linear_values = np.asarray((-0.5, 0.0, curve.x, 0.18, 0.9, 1.5, 64.0), dtype=np.float32)
    linear = _frame(linear_values, auxiliary=True)
    before = linear.data.copy()
    standalone_encoded = px.color.linear_to_gamma(linear, gamma=gamma)
    fused_encoded = px.color.rgb_to_rgb(linear, output_gamma=gamma)
    np.testing.assert_array_equal(standalone_encoded.data.get(), fused_encoded.data.get())
    encoded_values = np.asarray((-0.5, 0.0, curve.encoded_cut, 0.5, 1.0, 1.5), dtype=np.float32)
    encoded = _frame(encoded_values, gamma=gamma, auxiliary=True)
    standalone_decoded = px.color.gamma_to_linear(encoded, gamma=gamma)
    fused_decoded = px.color.rgb_to_rgb(encoded, output_gamma="linear")
    np.testing.assert_array_equal(standalone_decoded.data.get(), fused_decoded.data.get())
    assert cp.array_equal(standalone_encoded.data[..., 0], linear.data[..., 0])
    assert cp.array_equal(standalone_decoded.data[..., 0], encoded.data[..., 0])
    assert cp.array_equal(linear.data, before)
    assert standalone_encoded is not linear and standalone_encoded.data is not linear.data
    assert (standalone_encoded.gamma, standalone_encoded.matrix, standalone_encoded.data.dtype) == (
        gamma,
        None,
        cp.float32,
    )
    assert _rgb_values(standalone_encoded)[0, 0] < 0.0
    assert _rgb_values(standalone_encoded)[-1, 0] > 1.0


@pytest.mark.parametrize(("colorspace", "acceptance"), (("D-Gamut", 153), ("F-Gamut-C", 155)))
def test_vendor_a_gamut_definitions_matrices_and_native_rows(colorspace: str, acceptance: int) -> None:
    """v1-vendor-a-tokens acceptance 153 and 155: derive gamut matrices and native rows from public coordinates."""
    del acceptance
    from pixtreme._core.colorspace import _COLORSPACE_DEFINITIONS

    definition = _GAMUTS[colorspace]
    assert _COLORSPACE_DEFINITIONS[colorspace] == definition
    derived = _rgb_to_xyz(definition)
    np.testing.assert_allclose(derived, _RGB_TO_XYZ_TABLES[colorspace], rtol=0.0, atol=1e-12)
    if colorspace == "F-Gamut-C":
        np.testing.assert_array_equal(derived[2, :2], (0.0, 0.0))
    values = np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (-0.25, 0.18, 1.5)))
    source = _frame(values, colorspace=colorspace)
    converted = px.color.rgb_to_rgb(source, output_colorspace="Rec.709")
    np.testing.assert_allclose(_rgb_values(converted), values @ _conversion(definition, _REC709).T, rtol=0.0, atol=6e-6)
    grayscale = px.color.rgb_to_grayscale(source, colorspace=colorspace, gamma="linear", matrix="native")
    np.testing.assert_allclose(px.io.to_array(grayscale).get()[0, :, 0], values @ derived[1], rtol=0.0, atol=6e-6)
    assert np.any(_rgb_values(converted) < 0.0) and np.any(_rgb_values(converted) > 1.0)


@pytest.mark.parametrize(("colorspace", "acceptance"), (("D-Gamut", 154), ("F-Gamut-C", 156)))
def test_vendor_a_gamuts_use_d65_bradford_and_keep_cat02_auxiliary(colorspace: str, acceptance: int) -> None:
    """v1-vendor-a-tokens acceptance 154 and 156: use D65 identity and Bradford, with CAT02 as auxiliary."""
    del acceptance
    from pixtreme._color.transform import _compose_matrix

    definition = _GAMUTS[colorspace]
    to_rec709 = _conversion(definition, _REC709)
    np.testing.assert_allclose(_compose_matrix(colorspace, "Rec.709"), to_rec709, rtol=0.0, atol=6e-6)
    to_aces = _conversion(definition, _ACES2065)
    np.testing.assert_allclose(_compose_matrix(colorspace, "ACES2065-1"), to_aces, rtol=0.0, atol=6e-6)
    cat02_to_aces = _conversion(definition, _ACES2065, cone=_CAT02)
    np.testing.assert_allclose(cat02_to_aces, _CAT02_TO_ACES[colorspace], rtol=0.0, atol=5e-13)
    assert float(np.max(np.abs(to_aces - cat02_to_aces))) > 0.004

    if colorspace == "D-Gamut":
        dji_xyz = np.asarray(((0.6482, 0.1940, 0.1082), (0.2830, 0.8132, -0.0962), (-0.0183, -0.0832, 1.1903)))
        dji_xyz_inverse = np.asarray(((1.7257, -0.4314, -0.1917), (-0.6025, 1.3906, 0.1671), (-0.0156, 0.0905, 0.8489)))
        dji_to_rec709 = np.asarray(((1.6746, -0.5797, -0.0949), (-0.0981, 1.3340, -0.2359), (-0.0410, -0.2430, 1.2840)))
        dji_from_rec709 = np.asarray(((0.6163, 0.2857, 0.0980), (0.0505, 0.7990, 0.1505), (0.0292, 0.1604, 0.8104)))
        np.testing.assert_allclose(_rgb_to_xyz(definition), dji_xyz, rtol=0.0, atol=2.5e-4)
        np.testing.assert_allclose(np.linalg.inv(_rgb_to_xyz(definition)), dji_xyz_inverse, rtol=0.0, atol=2.5e-4)
        np.testing.assert_allclose(to_rec709, dji_to_rec709, rtol=0.0, atol=2.5e-4)
        np.testing.assert_allclose(np.linalg.inv(to_rec709), dji_from_rec709, rtol=0.0, atol=2.5e-4)

    source = _frame((0.18,), colorspace=colorspace, gamma="Canon-Log")
    result = px.color.rgb_to_rgb(source, output_colorspace="Rec.709", output_gamma="Canon-Log")
    assert (result.colorspace, result.gamma) == ("Rec.709", "Canon-Log")


@pytest.mark.parametrize("target", ("Rec.709", "ACES2065-1"))
def test_vendor_a_representative_frames_compose_independent_transfer_and_gamut_oracles(target: str) -> None:
    """v1-vendor-a-tokens acceptance 157: compose the five independent tokens in representative Frames."""
    target_definition = _REC709 if target == "Rec.709" else _ACES2065
    linear_rgb = np.asarray(((-0.25, 0.18, 1.5), (0.18, 1.25, -0.05)), dtype=np.float64)
    for colorspace, gamma in (("D-Gamut", "D-Log"), ("F-Gamut-C", "F-Log"), ("F-Gamut-C", "F-Log2")):
        curve = _CURVES[gamma]
        source = _frame(
            _encode(linear_rgb, curve).astype(np.float32), colorspace=colorspace, gamma=gamma, auxiliary=True
        )
        before = source.data.copy()
        converted = px.color.rgb_to_rgb(source, output_colorspace=target, output_gamma="linear")
        expected = (
            _decode(_rgb_values(source).astype(np.float64), curve)
            @ _conversion(_GAMUTS[colorspace], target_definition).T
        )
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


def test_existing_token_bits_remain_at_the_pre_vendor_a_baseline() -> None:
    """v1-vendor-a-tokens acceptance 158; v1-vendor-b-tokens acceptance 185:
    preserve every existing transfer and gamut fixture bit.
    """
    # Characterization provenance: captured from the complete pre-vendor-A commit
    # 1af2f7a6dad1e20a5362d02e4efba9b2ee93be27. Reproduce at that exact SHA by constructing an
    # ACEScg/linear RGB Frame from float32 values (-0.25, -0.018056996166706085, 0, 0.18000000715255737,
    # 1, 1.5), then concatenating each pre-vendor Gamma's channel-R encode and decode uint32 bytes in
    # canonical order. For each pre-vendor Colorspace, convert float32 RGB pixels
    # ((-0.25, 0.18, 1.5), (0.02, -0.1, 1)) to Rec.709/linear and concatenate all uint32 bytes in
    # canonical order. SHA-256 makes every captured bit part of the fixed fixture while keeping the table compact.
    old_gammas = _GAMMAS[:21] + _GAMMAS[28:]
    old_colorspaces = _COLORSPACES[:24]
    linear_values = np.asarray((-0.25, -0.018056996166706085, 0.0, 0.18000000715255737, 1.0, 1.5), np.float32)
    gamma_digest = sha256()
    for gamma in old_gammas:
        encoded = px.color.linear_to_gamma(_frame(linear_values), gamma=gamma)
        decoded = px.color.gamma_to_linear(encoded, gamma=gamma)
        gamma_digest.update(_rgb_values(encoded)[:, 0].view(np.uint32).tobytes())
        gamma_digest.update(_rgb_values(decoded)[:, 0].view(np.uint32).tobytes())
        assert (encoded.gamma, decoded.gamma) == (gamma, "linear")
    assert gamma_digest.hexdigest() == "9be038d39f79407654f02274b03c1b6a57e57a743a981d2e15724c37ae1de826"

    pixels = np.asarray(((-0.25, 0.18, 1.5), (0.02, -0.1, 1.0)), dtype=np.float32)
    gamut_digest = sha256()
    for colorspace in old_colorspaces:
        converted = px.color.rgb_to_rgb(
            _frame(pixels, colorspace=colorspace), output_colorspace="Rec.709", output_gamma="linear"
        )
        gamut_digest.update(px.io.to_array(converted).get().view(np.uint32).tobytes())
        assert (converted.colorspace, converted.gamma) == ("Rec.709", "linear")
    assert gamut_digest.hexdigest() == "0a0afd23631799b5d64b9060a69e4b2757395957010c60f8d1326c8d02af791c"


def test_vendor_a_dpx_codes_are_logarithmic_and_existing_mappings_remain_unchanged(tmp_path: Path) -> None:
    """v1-vendor-a-tokens acceptance 159: classify the three transfers as DPX logarithmic."""
    from pixtreme._io.formats.dpx import _dpx_transfer_from_gamma

    expected = {
        "D-Log": 3,
        "F-Log": 3,
        "F-Log2": 3,
        "V-Log": 3,
        "ACEScc": 3,
        "Cineon": 1,
        "REDlogFilm": 1,
        "linear": 2,
        "Gamma-2.5": 6,
    }
    assert {gamma: _dpx_transfer_from_gamma(gamma) for gamma in expected} == expected
    pixels = cp.asarray([[[0.18, 0.18, 0.18]]], dtype=cp.float32)
    for gamma, transfer in expected.items():
        frame = px.io.from_array(pixels.copy(), colorspace="Rec.709", gamma=gamma, channels="RGB")
        path = tmp_path / f"{gamma}.dpx"
        px.io.write_image(path, frame)
        assert path.read_bytes()[801] == transfer
        if gamma in _CURVES:
            assert px.io.read_image(path).gamma == "Cineon"


def test_vendor_a_reference_requirements_changelog_docstrings_and_generator_are_synchronized() -> None:
    """v1-vendor-a-tokens acceptance 161; v1-vendor-b-tokens acceptance 188:
    synchronize vocabulary, numeric identity, boundaries, and public prose.
    """
    token_reference = (ROOT / "docs_site" / "tokens.md").read_text(encoding="utf-8")
    requirements = require_repo_file("docs/requirements.md").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    generator = (ROOT / "tests" / "generate_vendor_a_tokens_sheet.py").read_text(encoding="utf-8")
    latest_section = latest_changelog_section(changelog)
    for token in ("D-Gamut", "F-Gamut-C", "D-Log", "F-Log", "F-Log2"):
        assert f"`{token}`" in token_reference
        assert token in latest_section
        assert token in generator
    for fragment in (
        "0.007827156200341792",
        "0.1400586161070593",
        "0.0005663467969879701",
        "0.09781139663651882",
        "0.0008888881429483923",
        "0.10068573654723681",
        "0.7347",
        "0.0263",
        "-0.0224",
        "reflectance",
        "maximum real root",
        "D65",
        "Bradford",
        "CAT02",
        "native",
        "D-Log M",
        "F-Log2 C",
        "F-Gamut",
    ):
        assert fragment in token_reference
    for fragment in ("27 Colorspace", "33 Gamma", "188 canonical tokens"):
        assert fragment in requirements
    for fragment in ("intersection", "CAT02", "Bradford", "bit", "non-parity"):
        assert fragment in latest_section
    for operation in (
        px.color.rgb_to_rgb,
        px.color.rgb_to_ycbcr,
        px.color.ycbcr_to_rgb,
        px.color.rgb_to_grayscale,
        px.color.gamma_to_linear,
        px.color.linear_to_gamma,
    ):
        docstring = inspect.getdoc(operation)
        assert docstring is not None
        for gamma in _CURVES:
            assert gamma in docstring
        assert "D-Gamut" in docstring or operation in (px.color.gamma_to_linear, px.color.linear_to_gamma)
        assert "F-Gamut-C" in docstring or operation in (px.color.gamma_to_linear, px.color.linear_to_gamma)
