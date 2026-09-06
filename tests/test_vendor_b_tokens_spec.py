"""Specification tests for Nikon, Leica, Apple, and Samsung color tokens."""

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


def _decimal_cbrt(value: Decimal) -> Decimal:
    guess = Decimal(str(float(value) ** (1.0 / 3.0)))
    for _ in range(40):
        guess = (Decimal(2) * guess + value / (guess * guess)) / Decimal(3)
    return guess


def _decimal_pow10(value: Decimal) -> Decimal:
    return (Decimal(10).ln() * value).exp()


def _derive_constants() -> dict[str, Decimal]:
    """Derive every non-published cut from decimal source coefficients, independently of production."""
    getcontext().prec = 80
    d = Decimal

    def nlog_root(start: str) -> Decimal:
        value = d(start)
        for _ in range(40):
            cube = _decimal_cbrt(value + d("0.0075"))
            function = d(150) * value.ln() + d(619) - d(650) * cube
            derivative = d(150) / value - d(650) / (d(3) * cube * cube)
            value -= function / derivative
        return value

    nlog_x = nlog_root("0.38")
    nlog_left = nlog_root("0.32")
    nlog_encoded_cut = (d(150) * nlog_x.ln() + d(619)) / d(1023)

    llog_linear_cut = d("0.006")
    llog_argument = d("1.3") * llog_linear_cut + d("0.0115")
    llog_m = d("0.27") * d("1.3") / (llog_argument * d(10).ln())
    llog_encoded_cut = d("0.27") * llog_argument.log10() + d("0.6")
    llog_d = llog_encoded_cut - llog_m * llog_linear_cut

    apple_r0 = d("-0.05641088")
    apple_rt = d("0.01")
    apple_c = d("47.28711236")
    apple_pt = apple_c * (apple_rt - apple_r0) ** 2

    samsung_xt = d("0.01")
    samsung_a1 = d("0.258984868")
    samsung_b1 = d("0.0003645")
    samsung_g1 = d("0.720504856")
    samsung_a2 = d("-0.20942")
    samsung_b2 = d("0.016904")
    samsung_yt = samsung_a1 * (samsung_xt + samsung_b1).log10() + samsung_g1
    samsung_g2 = samsung_yt - samsung_a2 * (samsung_b2 - samsung_xt).log10()
    samsung_zero = samsung_b2 - _decimal_pow10(-samsung_g2 / samsung_a2)
    return {
        "nlog_x": nlog_x,
        "nlog_left": nlog_left,
        "nlog_encoded_cut": nlog_encoded_cut,
        "llog_m": llog_m,
        "llog_encoded_cut": llog_encoded_cut,
        "llog_d": llog_d,
        "apple_pt": apple_pt,
        "samsung_yt": samsung_yt,
        "samsung_g2": samsung_g2,
        "samsung_zero": samsung_zero,
    }


_DERIVED = _derive_constants()


@dataclass(frozen=True)
class _Curve:
    gamma: str
    encode_cut: np.float64
    decode_cut: np.float64
    anchor_inputs: tuple[float, ...]
    anchor_codes: tuple[float, ...]
    inverse_anchors: tuple[float, float]


_CURVES = {
    "N-Log": _Curve(
        "N-Log",
        np.float64(float(_DERIVED["nlog_x"])),
        np.float64(float(_DERIVED["nlog_encoded_cut"])),
        (0.0, 0.18, 0.9, 1.0),
        (127.233198337988, 372.032128829833, 603.195922651326, 619.0),
        (-0.0075, 14.7808634405002),
    ),
    "L-Log": _Curve(
        "L-Log",
        np.float64(0.006),
        np.float64(float(_DERIVED["llog_encoded_cut"])),
        (0.0, 0.02, 0.18, 0.9),
        (91.7739638545772, 219.933176459073, 445.326123836937, 633.806919449728),
        (-0.0113582059056596, 23.3009314066646),
    ),
    "Apple-Log": _Curve(
        "Apple-Log",
        np.float64(0.01),
        np.float64(float(_DERIVED["apple_pt"])),
        (0.0, 0.18, 0.9, 12.0),
        (153.937410703834, 499.502725072985, 697.365592240713, 1022.99997790410),
        (-0.05641088, 12.0000021028157),
    ),
    "Samsung-Log": _Curve(
        "Samsung-Log",
        np.float64(0.01),
        np.float64(float(_DERIVED["samsung_yt"])),
        (0.0, 0.01, 0.18, 0.9, 12.0),
        (127.998620, 211.312832799044, 539.999999481018, 724.999999524348, 1022.99988230712),
        (-0.0500001614117767, 12.0000122746899),
    ),
}

_BRANCH_FIXTURES = {
    ("N-Log", "encode"): (
        (0x3EC1BFB4, 0x3EC1BFB5, 0x3EC1BFB6),
        (0x3EECD962, 0x3EECD962, 0x3EECD963),
        (7.371e-8, 4.601e-8, 4.601e-8),
    ),
    ("N-Log", "decode"): (
        (0x3EECD962, 0x3EECD963, 0x3EECD964),
        (0x3EC1BFB4, 0x3EC1BFB6, 0x3EC1BFB9),
        (1.364e-7, 1.286e-7, 1.286e-7),
    ),
    ("L-Log", "encode"): (
        (0x3BC49BA5, 0x3BC49BA6, 0x3BC49BA7),
        (0x3E0C6411, 0x3E0C6411, 0x3E0C6411),
        (1.404e-8, 1.379e-7, 1.379e-7),
    ),
    ("L-Log", "decode"): (
        (0x3E0C6410, 0x3E0C6411, 0x3E0C6412),
        (0x3BC49BA2, 0x3BC49BA6, 0x3BC49BAA),
        (1.067e-9, 1.565e-8, 1.565e-8),
    ),
    ("Apple-Log", "encode"): (
        (0x3C23D709, 0x3C23D70A, 0x3C23D70B),
        (0x3E558F86, 0x3E558F86, 0x3E558F87),
        (5.245e-8, 8.846e-8, 8.846e-8),
    ),
    ("Apple-Log", "decode"): (
        (0x3E558F85, 0x3E558F86, 0x3E558F87),
        (0x3C23D707, 0x3C23D709, 0x3C23D70B),
        (7.630e-9, 1.291e-8, 1.291e-8),
    ),
    ("Samsung-Log", "encode"): (
        (0x3C23D709, 0x3C23D70A, 0x3C23D70B),
        (0x3E5384F6, 0x3E5384F7, 0x3E5384F8),
        (1.407e-7, 1.365e-7, 1.365e-7),
    ),
    ("Samsung-Log", "decode"): (
        (0x3E5384F6, 0x3E5384F7, 0x3E5384F8),
        (0x3C23D709, 0x3C23D70A, 0x3C23D70C),
        (6.525e-9, 1.135e-8, 1.135e-8),
    ),
}

_EXPLICIT_FIXTURES = {
    ("N-Log", "encode"): (
        (0x3EB33333, 0x3EE6E31A, 7.403e-8),
        (0x3EA3D70A, 0x3EE03D39, 7.441e-8),
    ),
    ("N-Log", "decode"): ((0x3EE66666, 0x3EB20B3E, 1.313e-7),),
    ("L-Log", "encode"): ((0x3B449BA6, 0x3DE8412B, 8.731e-9),),
    ("L-Log", "decode"): ((0x3E0CCCCD, 0x3BC644B4, 1.586e-8),),
    ("Apple-Log", "encode"): (),
    ("Apple-Log", "decode"): (),
    ("Samsung-Log", "encode"): (
        (0x00000000, 0x3E001FAD, 8.948e-8),
        (0xBD23D70A, 0x3C713E93, 7.738e-8),
    ),
    ("Samsung-Log", "decode"): ((0x3CA3D70A, 0xBD16B46B, 3.839e-8),),
}

_D65 = (0.3127, 0.3290)
_ACES_WHITE = (0.32168, 0.33767)
_REC709 = (((0.640, 0.330), (0.300, 0.600), (0.150, 0.060)), _D65)
_REC2020 = (((0.708, 0.292), (0.170, 0.797), (0.131, 0.046)), _D65)
_ACES2065 = (((0.7347, 0.2653), (0.0000, 1.0000), (0.0001, -0.0770)), _ACES_WHITE)
_AWG = (((0.725, 0.301), (0.221, 0.814), (0.068, -0.076)), _D65)
_AWG_RGB_TO_XYZ = np.asarray(
    (
        (0.651491480167307, 0.221553113294040, 0.077411333590324),
        (0.270481290386703, 0.816037258920130, -0.086518549306833),
        (-0.023363832392207, -0.035087597128015, 1.147509180280101),
    )
)
_AWG_TO_REC709 = np.asarray(
    (
        (1.707280377173597, -0.519019919932013, -0.188260457241584),
        (-0.125010746108356, 1.314662365482206, -0.189651619373849),
        (-0.043624333387934, -0.191214371732502, 1.234838705120436),
    )
)
_AWG_CAT02_TO_AP0 = np.asarray(
    (
        (0.694785534308741, 0.242815317526078, 0.062399148165180),
        (0.046950736723248, 1.006273217948578, -0.053223954671826),
        (-0.021980981508687, -0.033182732026310, 1.055163713534997),
    )
)
_BRADFORD = np.asarray(
    ((0.8951, 0.2664, -0.1614), (-0.7502, 1.7135, 0.0367), (0.0389, -0.0685, 1.0296)),
    dtype=np.float64,
)
_CAT02 = np.asarray(
    ((0.7328, 0.4296, -0.1624), (-0.7036, 1.6975, 0.0061), (0.0030, 0.0136, 0.9834)),
    dtype=np.float64,
)


def _encode(values: np.ndarray, gamma: str) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    result = np.empty_like(source)
    if gamma == "N-Log":
        lower = source < float(_DERIVED["nlog_x"])
        result[lower] = 650.0 * np.cbrt(source[lower] + 0.0075) / 1023.0
        result[~lower] = (150.0 * np.log(source[~lower]) + 619.0) / 1023.0
        result[source == np.float64(np.float32(-0.0075))] = 0.0
    elif gamma == "L-Log":
        lower = source < 0.006
        result[lower] = float(_DERIVED["llog_m"]) * source[lower] + float(_DERIVED["llog_d"])
        result[~lower] = 0.27 * np.log10(1.3 * source[~lower] + 0.0115) + 0.6
    elif gamma == "Apple-Log":
        collapsed = source < -0.05641088
        quadratic = (~collapsed) & (source < 0.01)
        logarithmic = ~(collapsed | quadratic)
        result[collapsed] = 0.0
        result[quadratic] = 47.28711236 * (source[quadratic] + 0.05641088) ** 2
        result[logarithmic] = 0.08550479 * np.log2(source[logarithmic] + 0.00964052) + 0.69336945
    else:
        lower = source < 0.01
        result[lower] = -0.20942 * np.log10(0.016904 - source[lower]) + float(_DERIVED["samsung_g2"])
        result[~lower] = 0.258984868 * np.log10(source[~lower] + 0.0003645) + 0.720504856
    return result


def _decode(values: np.ndarray, gamma: str) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    result = np.empty_like(source)
    if gamma == "N-Log":
        lower = source < float(_DERIVED["nlog_encoded_cut"])
        result[lower] = (source[lower] * 1023.0 / 650.0) ** 3 - 0.0075
        result[~lower] = np.exp((source[~lower] * 1023.0 - 619.0) / 150.0)
    elif gamma == "L-Log":
        lower = source < float(_DERIVED["llog_encoded_cut"])
        result[lower] = (source[lower] - float(_DERIVED["llog_d"])) / float(_DERIVED["llog_m"])
        result[~lower] = (10.0 ** ((source[~lower] - 0.6) / 0.27) - 0.0115) / 1.3
    elif gamma == "Apple-Log":
        collapsed = source < 0.0
        square_root = (~collapsed) & (source < float(_DERIVED["apple_pt"]))
        exponential = ~(collapsed | square_root)
        result[collapsed] = -0.05641088
        result[square_root] = np.sqrt(source[square_root] / 47.28711236) - 0.05641088
        result[exponential] = 2.0 ** ((source[exponential] - 0.69336945) / 0.08550479) - 0.00964052
    else:
        lower = source < float(_DERIVED["samsung_yt"])
        result[lower] = 0.016904 - 10.0 ** ((source[lower] - float(_DERIVED["samsung_g2"])) / -0.20942)
        result[~lower] = 10.0 ** ((source[~lower] - 0.720504856) / 0.258984868) - 0.0003645
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
        cp.asarray(array[None]),
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


def _explicit_inputs(gamma: str, path: str) -> np.ndarray:
    fixtures = tuple(_from_bits(input_bits) for input_bits, _, _ in _EXPLICIT_FIXTURES[(gamma, path)])
    return np.concatenate(fixtures) if fixtures else np.empty(0, dtype=np.float32)


def _linear_windows(gamma: str) -> np.ndarray:
    centers = {
        "N-Log": (float(_DERIVED["nlog_x"]),),
        "L-Log": (0.006,),
        "Apple-Log": (-0.05641088, 0.01),
        "Samsung-Log": (float(_DERIVED["samsung_zero"]), 0.01),
    }[gamma]
    return np.unique(np.concatenate(tuple(_adjacent_float32(np.float32(center)) for center in centers)))


def _encoded_windows(gamma: str) -> np.ndarray:
    centers = {
        "N-Log": (float(_DERIVED["nlog_encoded_cut"]),),
        "L-Log": (float(_DERIVED["llog_encoded_cut"]),),
        "Apple-Log": (0.0, float(_DERIVED["apple_pt"])),
        "Samsung-Log": (0.0, float(_DERIVED["samsung_yt"])),
    }[gamma]
    return np.unique(np.concatenate(tuple(_adjacent_float32(np.float32(center)) for center in centers)))


def _linear_sets(gamma: str) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray, np.ndarray]:
    """Return four disjoint reflectance sets: dense grids, fixtures, cut windows, and anchors."""
    branch = _from_bits(_BRANCH_FIXTURES[(gamma, "encode")][0])
    explicit = _explicit_inputs(gamma, "encode")
    fixture = np.concatenate((branch, explicit))
    window = _linear_windows(gamma)
    anchors = {
        "N-Log": (-0.0075, 0.0, 0.18, 0.9, 1.0),
        "L-Log": (0.0, 0.02, 0.18, 0.9, 1.0),
        "Apple-Log": (-0.5, -0.05641088, 0.0, 0.18, 0.9, 1.0, 12.0),
        "Samsung-Log": (-0.5, -0.05, 0.0, 0.01, 0.18, 0.9, 1.0, 12.0),
    }[gamma]
    anchor_values = np.asarray(anchors, dtype=np.float32)
    excluded = np.concatenate((fixture, window, anchor_values))
    base_grids = [
        np.linspace(-0.5, 64.0, 400_001, dtype=np.float64).astype(np.float32),
        np.linspace(0.0, 0.02, 200_001, dtype=np.float64).astype(np.float32),
    ]
    if gamma == "N-Log":
        base_grids.append(np.linspace(0.25, 0.45, 200_001, dtype=np.float64).astype(np.float32))
    dense = tuple(grid[~np.isin(grid, excluded)] for grid in base_grids)
    return dense, fixture, window, anchor_values


def _encoded_sets(gamma: str) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray, np.ndarray]:
    """Return four disjoint encoded sets: dense grid, fixtures, cut windows, and anchors."""
    branch = _from_bits(_BRANCH_FIXTURES[(gamma, "decode")][0])
    explicit = _explicit_inputs(gamma, "decode")
    fixture = np.concatenate((branch, explicit))
    window = _encoded_windows(gamma)
    published_codes = np.asarray(_CURVES[gamma].anchor_codes, dtype=np.float64) / 1023.0
    extras = (-0.5, 0.0, 1.0, 1.5) if gamma != "Apple-Log" else (-0.5, 0.0, 1.0, 1.5)
    anchors = np.concatenate((published_codes.astype(np.float32), np.asarray(extras, dtype=np.float32)))
    excluded = np.concatenate((fixture, window, anchors))
    grid = np.linspace(-0.5, 1.5, 200_001, dtype=np.float64).astype(np.float32)
    return (grid[~np.isin(grid, excluded)],), fixture, window, anchors


def test_vendor_b_tokens_extend_only_canonical_vocabulary_and_public_surfaces() -> None:
    """v1-vendor-b-tokens acceptance 166-167: expose five canonical tokens without static aliases."""
    assert get_args(px.core.Colorspace) == _COLORSPACES
    assert get_args(px.core.Gamma) == _GAMMAS
    assert len(_ALIASES) == 30
    assert sum(len(get_args(alias)) for alias in _ALIASES) == 188
    assert _literal_strings(get_type_hints(px.color.linear_to_gamma)["gamma"]) == _GAMMAS
    assert _literal_strings(get_type_hints(px.color.rgb_to_rgb)["input_colorspace"]) == _COLORSPACES
    assert _literal_strings(get_type_hints(px.color.rgb_to_rgb)["output_gamma"]) == _GAMMAS
    for colorspace, gamma in (
        ("Rec.2020", "N-Log"),
        ("Rec.2020", "L-Log"),
        ("Rec.2020", "Apple-Log"),
        ("Rec.2020", "Samsung-Log"),
        ("Apple-Wide-Gamut", "Apple-Log"),
    ):
        frame = _frame((0.18,), colorspace=colorspace, gamma=gamma)
        assert (frame.colorspace, frame.gamma) == (colorspace, gamma)
        assert f"colorspace={colorspace!r}" in repr(frame)
        assert f"gamma={gamma!r}" in repr(frame)

    from pixtreme._core.vocabulary import _PERMANENT_TOKEN_ALIASES

    assert len(_PERMANENT_TOKEN_ALIASES) == 4
    new_tokens = {"Apple-Wide-Gamut", "N-Log", "L-Log", "Apple-Log", "Samsung-Log"}
    assert not any(token in new_tokens for alias in _PERMANENT_TOKEN_ALIASES for token in alias)


def test_vendor_b_token_keys_alias_boundaries_and_fail_fast_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """v1-vendor-b-tokens acceptance 168 and 187: normalize separators and reject raw invalid inputs."""
    from pixtreme._core.validation import _normalized_closed_token

    translation = str.maketrans("", "", " .-_")
    expected = {
        "Apple-Wide-Gamut": "applewidegamut",
        "N-Log": "nlog",
        "L-Log": "llog",
        "Apple-Log": "applelog",
        "Samsung-Log": "samsunglog",
    }
    assert {token: token.translate(translation).casefold() for token in expected} == expected
    assert len({token.translate(translation).casefold() for token in _COLORSPACES}) == len(_COLORSPACES)
    assert len({token.translate(translation).casefold() for token in _GAMMAS}) == len(_GAMMAS)
    accepted = (
        ("Apple Wide Gamut", "colorspace", _COLORSPACES, "Apple-Wide-Gamut"),
        ("apple_wide_gamut", "colorspace", _COLORSPACES, "Apple-Wide-Gamut"),
        ("applewidegamut", "colorspace", _COLORSPACES, "Apple-Wide-Gamut"),
        ("n.log", "gamma", _GAMMAS, "N-Log"),
        ("NLog", "gamma", _GAMMAS, "N-Log"),
        ("n_log", "gamma", _GAMMAS, "N-Log"),
        ("L Log", "gamma", _GAMMAS, "L-Log"),
        ("LLog", "gamma", _GAMMAS, "L-Log"),
        ("llog", "gamma", _GAMMAS, "L-Log"),
        ("applelog", "gamma", _GAMMAS, "Apple-Log"),
        ("Apple Log", "gamma", _GAMMAS, "Apple-Log"),
        ("apple_log", "gamma", _GAMMAS, "Apple-Log"),
        ("Samsung_Log", "gamma", _GAMMAS, "Samsung-Log"),
        ("Samsung Log", "gamma", _GAMMAS, "Samsung-Log"),
        ("samsunglog", "gamma", _GAMMAS, "Samsung-Log"),
    )
    for spelling, axis, vocabulary, canonical in accepted:
        assert _normalized_closed_token(spelling, axis=axis, accepted=vocabulary) == canonical
    rejected = (
        "Nikon N-Log",
        "Leica L-Log",
        "Apple Log 2",
        "AppleLog2",
        "Apple-Log-2",
        "N-Log Gamut",
        "Samsung Log Gamut",
    )
    for value in rejected:
        vocabulary = _COLORSPACES if "Gamut" in value else _GAMMAS
        with pytest.raises(ValueError):
            _normalized_closed_token(value, axis="token", accepted=vocabulary)

    import pixtreme._color.semantics as semantics

    monkeypatch.setattr(semantics, "_color_semantics_kernel", lambda: pytest.fail("GPU work must not start"))
    source = _frame((0.18,))
    for value in ("Nikon N-Log", 17):
        with pytest.raises(ValueError) as captured:
            px.color.linear_to_gamma(source, gamma=value)
        message = str(captured.value)
        assert message.index("why=") < message.index("what=") < message.index("how=")
        assert f"received gamma={value!r}" in message
        assert repr(_GAMMAS) in message
        assert "Nikon N-Log" not in message.replace(f"received gamma={value!r}", "")


@pytest.mark.parametrize(
    ("gamma", "acceptance"),
    (("N-Log", 169), ("L-Log", 172), ("Apple-Log", 175), ("Samsung-Log", 178)),
)
def test_vendor_b_encode_matches_dense_oracle_branch_fixtures_and_anchors(gamma: str, acceptance: int) -> None:
    """v1-vendor-b-tokens acceptance 169, 172, 175, 178: encode reflectance with each public branch model."""
    del acceptance
    dense, _, _, _ = _linear_sets(gamma)
    expected_dense_counts = (400_001, 200_001, 200_001) if gamma == "N-Log" else (400_001, 200_001)
    assert tuple(len(grid) for grid in dense) < expected_dense_counts
    for values in dense:
        actual = _rgb_values(px.color.linear_to_gamma(_frame(values), gamma=gamma))[:, 0]
        np.testing.assert_allclose(actual, _encode(values, gamma), rtol=2e-7, atol=3e-7)
        assert np.all(np.diff(actual) >= 0.0)

    input_bits, expected_bits, envelopes = _BRANCH_FIXTURES[(gamma, "encode")]
    inputs = _from_bits(input_bits)
    actual = _rgb_values(px.color.linear_to_gamma(_frame(inputs), gamma=gamma))[:, 0]
    expected = _from_bits(expected_bits)
    assert np.all(np.abs(actual.astype(np.float64) - expected.astype(np.float64)) <= np.asarray(envelopes))
    np.testing.assert_array_equal(_encode(inputs, gamma).astype(np.float32).view(np.uint32), expected_bits)

    for explicit_input, explicit_expected, explicit_envelope in _EXPLICIT_FIXTURES[(gamma, "encode")]:
        actual_explicit = _rgb_values(px.color.linear_to_gamma(_frame(_from_bits(explicit_input)), gamma=gamma))[0, 0]
        assert abs(float(actual_explicit) - float(_from_bits(explicit_expected)[0])) <= explicit_envelope
        assert _encode(_from_bits(explicit_input), gamma).astype(np.float32).view(np.uint32)[0] == explicit_expected

    curve = _CURVES[gamma]
    encoded = px.color.linear_to_gamma(_frame(curve.anchor_inputs, auxiliary=True), gamma=gamma)
    codes = _rgb_values(encoded)[:, 0].astype(np.float64) * 1023.0
    np.testing.assert_allclose(codes, curve.anchor_codes, rtol=0.0, atol=2.1e-4)
    expected_codes = {
        "N-Log": (127, 372, 603, 619),
        "L-Log": (92, 220, 445, 634),
        "Apple-Log": (154, 500, 697, 1023),
        "Samsung-Log": (128, 211, 540, 725, 1023),
    }
    np.testing.assert_array_equal(np.rint(codes).astype(np.int64), expected_codes[gamma])
    assert encoded.gamma == gamma


@pytest.mark.parametrize(
    ("gamma", "acceptance"),
    (("N-Log", 170), ("L-Log", 173), ("Apple-Log", 176), ("Samsung-Log", 179)),
)
def test_vendor_b_decode_matches_dense_oracle_branch_fixtures_and_anchors(gamma: str, acceptance: int) -> None:
    """v1-vendor-b-tokens acceptance 170, 173, 176, 179: decode every branch and signed extension."""
    del acceptance
    (values,), _, _, _ = _encoded_sets(gamma)
    decoded = px.color.gamma_to_linear(_frame(values, gamma=gamma, auxiliary=True), gamma=gamma)
    actual = _rgb_values(decoded)[:, 0]
    np.testing.assert_allclose(actual, _decode(values, gamma), rtol=3e-6, atol=2e-7)
    assert np.all(np.diff(actual) >= 0.0)

    input_bits, expected_bits, envelopes = _BRANCH_FIXTURES[(gamma, "decode")]
    inputs = _from_bits(input_bits)
    actual_fixture = _rgb_values(px.color.gamma_to_linear(_frame(inputs, gamma=gamma), gamma=gamma))[:, 0]
    expected_fixture = _from_bits(expected_bits)
    assert np.all(
        np.abs(actual_fixture.astype(np.float64) - expected_fixture.astype(np.float64)) <= np.asarray(envelopes)
    )
    np.testing.assert_array_equal(_decode(inputs, gamma).astype(np.float32).view(np.uint32), expected_bits)
    for explicit_input, explicit_expected, explicit_envelope in _EXPLICIT_FIXTURES[(gamma, "decode")]:
        actual_explicit = _rgb_values(
            px.color.gamma_to_linear(_frame(_from_bits(explicit_input), gamma=gamma), gamma=gamma)
        )[0, 0]
        assert abs(float(actual_explicit) - float(_from_bits(explicit_expected)[0])) <= explicit_envelope
        assert _decode(_from_bits(explicit_input), gamma).astype(np.float32).view(np.uint32)[0] == explicit_expected

    anchor_inputs = np.asarray((0.0, 1.0), dtype=np.float32)
    anchor_actual = _rgb_values(px.color.gamma_to_linear(_frame(anchor_inputs, gamma=gamma), gamma=gamma))[:, 0]
    np.testing.assert_allclose(anchor_actual, _CURVES[gamma].inverse_anchors, rtol=3e-6, atol=2e-7)
    np.testing.assert_array_max_ulp(anchor_actual[1:], _decode(anchor_inputs, gamma)[1:].astype(np.float32), maxulp=30)
    assert decoded.gamma == "linear"


@pytest.mark.parametrize(
    ("gamma", "path", "input_bits", "expected_bits", "envelope"),
    tuple(
        pytest.param(gamma, path, input_bits, expected_bits, envelope, id=f"{gamma}-{path}-{input_bits:08x}")
        for (gamma, path), fixtures in _EXPLICIT_FIXTURES.items()
        for input_bits, expected_bits, envelope in fixtures
    ),
)
def test_vendor_b_explicit_samples_reject_printed_substitutions(
    gamma: str,
    path: str,
    input_bits: int,
    expected_bits: int,
    envelope: float,
) -> None:
    """v1-vendor-b-tokens acceptance 171, 174, and 180: explicit samples separate production identities."""
    source = _from_bits(input_bits)
    if path == "encode":
        actual = _rgb_values(px.color.linear_to_gamma(_frame(source), gamma=gamma))[0, 0]
    else:
        actual = _rgb_values(px.color.gamma_to_linear(_frame(source, gamma=gamma), gamma=gamma))[0, 0]
    assert abs(float(actual) - float(_from_bits(expected_bits)[0])) <= envelope


@pytest.mark.parametrize(
    ("gamma", "acceptance"),
    (("N-Log", 171), ("L-Log", 174), ("Apple-Log", 177), ("Samsung-Log", 180)),
)
def test_vendor_b_derived_constants_cut_windows_mutants_and_round_trips(gamma: str, acceptance: int) -> None:
    """v1-vendor-b-tokens acceptance 171, 174, 177, 180: derive cuts, bound seams, and round-trip four sets."""
    del acceptance
    assert float(_DERIVED["nlog_x"]) == 0.3784157394368526
    assert abs(_DERIVED["nlog_left"] - Decimal("0.316730607914939475650508228913")) < Decimal("5e-31")
    assert float(_DERIVED["nlog_encoded_cut"]) == 0.4625960144726521
    assert float(_DERIVED["llog_m"]) == 7.898308971401108
    assert float(_DERIVED["llog_encoded_cut"]) == 0.1371004734320989
    assert float(_DERIVED["llog_d"]) == 0.08971061960369227
    assert float(_DERIVED["apple_pt"]) == 0.20855531595464208
    assert float(_DERIVED["samsung_yt"]) == 0.20656190889447099
    assert float(_DERIVED["samsung_g2"]) == -0.245973605190997
    assert float(_DERIVED["samsung_zero"]) == -0.05000016141177667
    assert Decimal("-0.245975") <= _DERIVED["samsung_g2"] < Decimal("-0.245965")

    dense_linear, linear_fixture, encode_window, linear_anchors = _linear_sets(gamma)
    dense_encoded, encoded_fixture, decode_window, encoded_anchors = _encoded_sets(gamma)
    encoded_window = _rgb_values(px.color.linear_to_gamma(_frame(encode_window), gamma=gamma))[:, 0]
    assert _max_downward(encoded_window) <= 4e-7
    decoded_window = _rgb_values(px.color.gamma_to_linear(_frame(decode_window, gamma=gamma), gamma=gamma))[:, 0]
    decode_limit = 3e-7 if gamma == "N-Log" else 3e-8
    assert _max_downward(decoded_window) <= decode_limit

    linear_sets = (*dense_linear, linear_fixture, encode_window, linear_anchors)
    encoded_sets = (*dense_encoded, encoded_fixture, decode_window, encoded_anchors)
    for values in linear_sets:
        actual = _rgb_values(px.color.linear_to_gamma(_frame(values), gamma=gamma))[:, 0]
        np.testing.assert_allclose(actual, _encode(values, gamma), rtol=2e-7, atol=3e-7)
    for values in encoded_sets:
        actual = _rgb_values(px.color.gamma_to_linear(_frame(values, gamma=gamma), gamma=gamma))[:, 0]
        np.testing.assert_allclose(actual, _decode(values, gamma), rtol=3e-6, atol=2e-7)
    for values in linear_sets:
        roundtrip_values = values
        if gamma == "Apple-Log":
            roundtrip_values = values[values >= np.float32(-0.05641088)]
        encoded_frame = px.color.linear_to_gamma(_frame(roundtrip_values), gamma=gamma)
        restored = _rgb_values(px.color.gamma_to_linear(encoded_frame, gamma=gamma))[:, 0]
        assert np.all(np.isfinite(restored))
        np.testing.assert_allclose(restored, roundtrip_values, rtol=5e-6, atol=2e-7)
    for values in encoded_sets:
        roundtrip_values = values
        if gamma == "N-Log":
            roundtrip_values = values[np.abs(values) >= np.float32(0.02)]
        elif gamma == "Apple-Log":
            roundtrip_values = values[values >= np.float32(0.0)]
        decoded_frame = px.color.gamma_to_linear(_frame(roundtrip_values, gamma=gamma), gamma=gamma)
        reencoded = _rgb_values(px.color.linear_to_gamma(decoded_frame, gamma=gamma))[:, 0]
        assert np.all(np.isfinite(reencoded))
        np.testing.assert_allclose(reencoded, roundtrip_values, rtol=5e-6, atol=2e-7)


def test_vendor_b_standalone_and_fused_paths_preserve_frame_contracts() -> None:
    """v1-vendor-b-tokens acceptance 181: keep standalone and fused transfer paths bit-identical."""
    for gamma in _CURVES:
        linear_sets = _linear_sets(gamma)
        encoded_sets = _encoded_sets(gamma)
        for values in (*linear_sets[0], *linear_sets[1:]):
            standalone = px.color.linear_to_gamma(_frame(values), gamma=gamma)
            fused = px.color.rgb_to_rgb(_frame(values), output_gamma=gamma)
            np.testing.assert_array_equal(standalone.data.get(), fused.data.get())
        for values in (*encoded_sets[0], *encoded_sets[1:]):
            standalone = px.color.gamma_to_linear(_frame(values, gamma=gamma), gamma=gamma)
            fused = px.color.rgb_to_rgb(_frame(values, gamma=gamma), output_gamma="linear")
            np.testing.assert_array_equal(standalone.data.get(), fused.data.get())

        representative = _frame((-0.05, 0.18, 1.5), auxiliary=True)
        before = representative.data.copy()
        encoded = px.color.linear_to_gamma(representative, gamma=gamma)
        assert cp.array_equal(encoded.data[..., 0], representative.data[..., 0])
        assert cp.array_equal(representative.data, before)
        assert encoded is not representative and encoded.data is not representative.data
        assert (encoded.gamma, encoded.matrix, encoded.data.dtype) == (gamma, None, cp.float32)


def test_apple_wide_gamut_definition_matrix_native_row_and_adaptation() -> None:
    """v1-vendor-b-tokens acceptance 182-183: derive AWG and use D65 identity plus Bradford adaptation."""
    from pixtreme._color.transform import _compose_matrix
    from pixtreme._core.colorspace import _COLORSPACE_DEFINITIONS

    assert _COLORSPACE_DEFINITIONS["Apple-Wide-Gamut"] == _AWG
    derived = _rgb_to_xyz(_AWG)
    np.testing.assert_allclose(derived, _AWG_RGB_TO_XYZ, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(_conversion(_AWG, _REC709), _AWG_TO_REC709, rtol=0.0, atol=1e-12)
    values = np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (-0.25, 0.18, 1.5)))
    source = _frame(values, colorspace="Apple-Wide-Gamut")
    converted = px.color.rgb_to_rgb(source, output_colorspace="Rec.709")
    np.testing.assert_allclose(_rgb_values(converted), values @ _AWG_TO_REC709.T, rtol=2e-6, atol=8e-6)
    grayscale = px.color.rgb_to_grayscale(source, colorspace="Apple-Wide-Gamut", gamma="linear", matrix="native")
    np.testing.assert_allclose(px.io.to_array(grayscale).get()[0, :, 0], values @ derived[1], rtol=0.0, atol=6e-6)

    np.testing.assert_allclose(_compose_matrix("Apple-Wide-Gamut", "Rec.709"), _AWG_TO_REC709, rtol=0.0, atol=6e-6)
    bradford_to_ap0 = _conversion(_AWG, _ACES2065)
    np.testing.assert_allclose(_compose_matrix("Apple-Wide-Gamut", "ACES2065-1"), bradford_to_ap0, rtol=0.0, atol=6e-6)
    cat02_to_ap0 = _conversion(_AWG, _ACES2065, cone=_CAT02)
    np.testing.assert_allclose(cat02_to_ap0, _AWG_CAT02_TO_AP0, rtol=0.0, atol=5e-13)
    assert float(np.max(np.abs(bradford_to_ap0 - cat02_to_ap0))) > 0.004
    np.testing.assert_allclose(
        _compose_matrix("Apple-Wide-Gamut", "Rec.2020"), _conversion(_AWG, _REC2020), rtol=0.0, atol=6e-6
    )


@pytest.mark.parametrize("target", ("Rec.709", "ACES2065-1"))
def test_vendor_b_representative_frames_compose_independent_transfer_and_gamut_oracles(target: str) -> None:
    """v1-vendor-b-tokens acceptance 184: compose every new token in representative Frames."""
    target_definition = _REC709 if target == "Rec.709" else _ACES2065
    linear_rgb = np.asarray(((-0.05, 0.18, 1.5), (0.18, 1.25, 0.0)), dtype=np.float64)
    cases = (
        ("Rec.2020", "N-Log"),
        ("Rec.2020", "L-Log"),
        ("Rec.2020", "Apple-Log"),
        ("Rec.2020", "Samsung-Log"),
        ("Apple-Wide-Gamut", "Apple-Log"),
    )
    for colorspace, gamma in cases:
        gamut = _AWG if colorspace == "Apple-Wide-Gamut" else _REC2020
        source = _frame(
            _encode(linear_rgb, gamma).astype(np.float32), colorspace=colorspace, gamma=gamma, auxiliary=True
        )
        before = source.data.copy()
        converted = px.color.rgb_to_rgb(source, output_colorspace=target, output_gamma="linear")
        expected = _decode(_rgb_values(source), gamma) @ _conversion(gamut, target_definition).T
        np.testing.assert_allclose(_rgb_values(converted), expected, rtol=2e-6, atol=8e-6)
        assert cp.array_equal(converted.data[..., 0], source.data[..., 0])
        assert cp.array_equal(source.data, before)
        assert (converted.colorspace, converted.gamma, converted.channels, converted.matrix) == (
            target,
            "linear",
            source.channels,
            None,
        )


def test_existing_token_bits_remain_at_the_pre_vendor_b_baseline() -> None:
    """v1-vendor-b-tokens acceptance 185: preserve every existing transfer and gamut fixture bit."""
    # Characterization provenance: captured from the complete pre-vendor-B commit
    # 7d625404cbf2ddaa412dcc95055ab8b3e0891f3c. At that SHA, construct an ACEScg/linear RGB Frame
    # from float32 (-0.25, -0.018056996166706085, 0, 0.18000000715255737, 1, 1.5), then concatenate
    # each Gamma's channel-R encode and decode uint32 bytes in canonical order. For each Colorspace, convert
    # float32 ((-0.25, 0.18, 1.5), (0.02, -0.1, 1)) to Rec.709/linear and concatenate all uint32 bytes.
    old_gammas = tuple(gamma for gamma in _GAMMAS if gamma not in _CURVES)
    old_colorspaces = tuple(colorspace for colorspace in _COLORSPACES if colorspace != "Apple-Wide-Gamut")
    linear_values = np.asarray((-0.25, -0.018056996166706085, 0.0, 0.18000000715255737, 1.0, 1.5), np.float32)
    gamma_digest = sha256()
    for gamma in old_gammas:
        encoded = px.color.linear_to_gamma(_frame(linear_values), gamma=gamma)
        decoded = px.color.gamma_to_linear(encoded, gamma=gamma)
        gamma_digest.update(_rgb_values(encoded)[:, 0].view(np.uint32).tobytes())
        gamma_digest.update(_rgb_values(decoded)[:, 0].view(np.uint32).tobytes())
    assert gamma_digest.hexdigest() == "c2afc5cf957079c3e3179287674cd9dfb3572aebf61f34af0ed200751d573562"

    pixels = np.asarray(((-0.25, 0.18, 1.5), (0.02, -0.1, 1.0)), dtype=np.float32)
    gamut_digest = sha256()
    for colorspace in old_colorspaces:
        converted = px.color.rgb_to_rgb(
            _frame(pixels, colorspace=colorspace), output_colorspace="Rec.709", output_gamma="linear"
        )
        gamut_digest.update(px.io.to_array(converted).get().view(np.uint32).tobytes())
    assert gamut_digest.hexdigest() == "7c0e43e049008d2bd219c98122693be4e05772f9812befdec5f5e5771c36ea34"


def test_vendor_b_dpx_codes_are_logarithmic_and_existing_mappings_remain_unchanged(tmp_path: Path) -> None:
    """v1-vendor-b-tokens acceptance 186: classify four transfers as DPX logarithmic."""
    from pixtreme._io.formats.dpx import _dpx_transfer_from_gamma

    expected = {
        "N-Log": 3,
        "L-Log": 3,
        "Apple-Log": 3,
        "Samsung-Log": 3,
        "F-Log2": 3,
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


def test_vendor_b_reference_requirements_changelog_docstrings_and_generator_are_synchronized() -> None:
    """v1-vendor-b-tokens acceptance 188: synchronize vocabulary, numeric identity, and public prose."""
    token_reference = (ROOT / "docs_site" / "tokens.md").read_text(encoding="utf-8")
    requirements = require_repo_file("docs/requirements.md").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    generator = (ROOT / "tests" / "generate_vendor_b_tokens_sheet.py").read_text(encoding="utf-8")
    latest_section = latest_changelog_section(changelog)
    for token in ("Apple-Wide-Gamut", "N-Log", "L-Log", "Apple-Log", "Samsung-Log"):
        assert f"`{token}`" in token_reference
        assert token in latest_section
        assert token in generator
    for fragment in (
        "0.3784157394368526",
        "0.4625960144726521",
        "7.898308971401108",
        "0.1371004734320989",
        "0.08971061960369227",
        "-0.05641088",
        "0.20855531595464208",
        "-0.245973605190997",
        "0.20656190889447099",
        "0.725",
        "-0.076",
        "reflectance",
        "D65",
        "Bradford",
        "CAT02",
        "native",
        "Apple Log 2",
        "Nikon N-Log",
        "Leica L-Log",
        "Samsung Log",
        "non-parity",
    ):
        assert fragment in token_reference
    for fragment in ("27 Colorspace", "33 Gamma", "188 canonical tokens"):
        assert fragment in requirements
    for fragment in ("intersection", "tangent", "collapse", "CAT02", "Bradford", "bit", "non-parity"):
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
        assert "Apple-Wide-Gamut" in docstring or operation in (px.color.gamma_to_linear, px.color.linear_to_gamma)
