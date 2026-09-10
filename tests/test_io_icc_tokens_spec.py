"""Specification tests for the colorspace and transfer tokens introduced with ICC metadata."""

from __future__ import annotations

from typing import get_args

import cupy as cp
import numpy as np

import pixtreme as px

_NEW_COLORSPACES = ("Adobe-RGB", "ProPhoto-RGB")
_NEW_GAMMAS = ("Gamma-1.8", "Adobe-RGB", "ProPhoto-RGB")
_DEFINITIONS = {
    "Adobe-RGB": (
        ((0.6400, 0.3300), (0.2100, 0.7100), (0.1500, 0.0600)),
        (0.3127, 0.3290),
    ),
    "ProPhoto-RGB": (
        ((0.7347, 0.2653), (0.1596, 0.8404), (0.0366, 0.0001)),
        (0.3457, 0.3585),
    ),
}
_BRADFORD = np.asarray(
    ((0.8951, 0.2664, -0.1614), (-0.7502, 1.7135, 0.0367), (0.0389, -0.0685, 1.0296)),
    dtype=np.float64,
)


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


def _frame(values: np.ndarray, *, colorspace: str, gamma: str) -> px.core.Frame:
    rgb = np.repeat(np.asarray(values, dtype=np.float32)[:, None], 3, axis=1)
    auxiliary = (np.arange(rgb.shape[0], dtype=np.float32) + np.float32(16.0))[:, None]
    data = np.concatenate((auxiliary, rgb[:, (2, 0, 1)]), axis=1)[None, :, :]
    return px.io.from_array(
        cp.asarray(data),
        colorspace=colorspace,
        gamma=gamma,
        channels=("Z", "B", "R", "G"),
        matrix="native",
    )


def _rgb(frame: px.core.Frame) -> np.ndarray:
    data = frame.data.get()[0]
    return data[:, [frame.channels.index(label) for label in ("R", "G", "B")]]


def _signed_power(values: np.ndarray, exponent: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.copysign(np.abs(values) ** exponent, values)


def _prophoto_decode(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    lower = values < 1.0 / 32.0
    result[lower] = values[lower] / 16.0
    result[~lower] = values[~lower] ** 1.8
    return result


def _prophoto_encode(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    lower = values < 1.0 / 512.0
    result[lower] = 16.0 * values[lower]
    result[~lower] = values[~lower] ** (1.0 / 1.8)
    return result


def test_icc_tokens_extend_the_canonical_vocabulary_without_alias_changes() -> None:
    """v1-io-icc acceptance 1: exact tails, family counts, normalization, and alias totals stay canonical."""
    colorspaces = get_args(px.core.Colorspace)
    gammas = get_args(px.core.Gamma)

    assert colorspaces[-2:] == _NEW_COLORSPACES
    assert gammas[-7:] == ("Gamma-1.8", "Gamma-2.2", "Gamma-2.4", "Gamma-2.5", "Gamma-2.6", *_NEW_GAMMAS[1:])
    assert (len(colorspaces), len(gammas)) == (29, 36)

    from pixtreme._core.validation import _normalized_closed_token
    from pixtreme._core.vocabulary import _PERMANENT_TOKEN_ALIASES

    for value, axis, accepted, expected in (
        ("Adobe RGB", "colorspace", colorspaces, "Adobe-RGB"),
        ("adobe_rgb", "gamma", gammas, "Adobe-RGB"),
        ("ProPhoto RGB", "colorspace", colorspaces, "ProPhoto-RGB"),
        ("prophoto_rgb", "gamma", gammas, "ProPhoto-RGB"),
        ("gamma 1.8", "gamma", gammas, "Gamma-1.8"),
    ):
        assert _normalized_closed_token(value, axis=axis, accepted=accepted) == expected
    assert len(_PERMANENT_TOKEN_ALIASES) == 4


def test_icc_colorspace_definitions_derive_normalized_matrices_from_xy() -> None:
    """v1-io-icc acceptance 2: production gamut definitions equal independent float64 xy derivations."""
    from pixtreme._color.transform import _RGB_TO_XYZ
    from pixtreme._core.colorspace import _COLORSPACE_DEFINITIONS

    for token, definition in _DEFINITIONS.items():
        assert _COLORSPACE_DEFINITIONS[token] == definition
        expected = _rgb_to_xyz(*definition)
        np.testing.assert_allclose(_RGB_TO_XYZ[token], expected, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(expected.sum(axis=1), _xy_to_xyz(definition[1]), rtol=0.0, atol=1e-12)


def test_icc_colorspaces_use_native_rows_and_bradford_for_different_whites() -> None:
    """v1-io-icc acceptance 2: native luma and differing-white conversion use the shared numerical path."""
    from pixtreme._color.transform import _compose_matrix

    target = _DEFINITIONS["ProPhoto-RGB"]
    source = _DEFINITIONS["Adobe-RGB"]
    expected = np.linalg.inv(_rgb_to_xyz(*target)) @ _adaptation(source[1], target[1]) @ _rgb_to_xyz(*source)
    np.testing.assert_allclose(_compose_matrix("Adobe-RGB", "ProPhoto-RGB"), expected, rtol=0.0, atol=6e-6)

    values = np.asarray((0.05, 0.25, 0.7), dtype=np.float32)
    frame = _frame(values, colorspace="Adobe-RGB", gamma="linear")
    gray = px.color.rgb_to_grayscale(frame, matrix="native")
    expected_y = _rgb_to_xyz(*source)[1] @ np.repeat(values[:, None], 3, axis=1).T
    np.testing.assert_allclose(gray.data.get()[0, :, 0], expected_y, rtol=0.0, atol=6e-6)


def test_icc_transfer_tokens_match_independent_unclipped_oracles() -> None:
    """v1-io-icc acceptance 3: new decode and encode branches match independent float64 equations."""
    values = np.asarray(
        (-0.25, 0.0, np.nextafter(np.float32(1 / 512), np.float32(0)), 1 / 512, 0.18, 1 / 32, 1.0, 1.25),
        dtype=np.float32,
    )
    cases = {
        "Gamma-1.8": (
            lambda x: _signed_power(x, 1.8),
            lambda x: _signed_power(x, 1.0 / 1.8),
        ),
        "Adobe-RGB": (
            lambda x: _signed_power(x, 563.0 / 256.0),
            lambda x: _signed_power(x, 256.0 / 563.0),
        ),
        "ProPhoto-RGB": (_prophoto_decode, _prophoto_encode),
    }
    for token, (decode, encode) in cases.items():
        encoded = _frame(values, colorspace="ACEScg", gamma=token)
        decoded = px.color.gamma_to_linear(encoded)
        np.testing.assert_allclose(_rgb(decoded), np.repeat(decode(values)[:, None], 3, axis=1), rtol=0.0, atol=2e-6)
        np.testing.assert_array_equal(decoded.data.get()[0, :, 0], encoded.data.get()[0, :, 0])

        linear = _frame(values, colorspace="ACEScg", gamma="linear")
        result = px.color.linear_to_gamma(linear, gamma=token)
        np.testing.assert_allclose(_rgb(result), np.repeat(encode(values)[:, None], 3, axis=1), rtol=0.0, atol=2e-6)
        np.testing.assert_array_equal(result.data.get()[0, :, 0], linear.data.get()[0, :, 0])


def test_icc_transfer_round_trips_and_remains_independent_from_colorspace() -> None:
    """v1-io-icc acceptance 3: finite round trips preserve values and never infer the same-named gamut."""
    values = np.asarray((-0.25, 0.0, 1 / 512, 0.18, 1 / 32, 1.0, 1.25), dtype=np.float32)
    for token in _NEW_GAMMAS:
        source = _frame(values, colorspace="P3-D65", gamma="linear")
        encoded = px.color.linear_to_gamma(source, gamma=token)
        decoded = px.color.gamma_to_linear(encoded)
        np.testing.assert_allclose(_rgb(decoded), _rgb(source), rtol=2e-6, atol=2e-7)
        assert encoded.colorspace == decoded.colorspace == "P3-D65"
        assert encoded.gamma == token
        assert decoded.gamma == "linear"

    tagged = _frame(values, colorspace="Adobe-RGB", gamma="Gamma-1.8")
    assert (tagged.colorspace, tagged.gamma) == ("Adobe-RGB", "Gamma-1.8")


def test_existing_gamma_codes_remain_bit_stable_when_new_codes_are_appended() -> None:
    """v1-io-icc acceptance 1 and 3: all pre-feature transfer dispatch codes retain their exact values."""
    from pixtreme._color.transform import _GAMMA_CODES

    expected_existing = {
        "linear": 0,
        "sRGB": 1,
        "Rec.709": 2,
        "BT.1886": 3,
        "PQ": 4,
        "HLG": 5,
        "Gamma-2.2": 9,
        "Gamma-2.4": 10,
        "Gamma-2.5": 24,
        "Gamma-2.6": 11,
    }
    assert {token: _GAMMA_CODES[token] for token in expected_existing} == expected_existing
    assert {_GAMMA_CODES[token] for token in _NEW_GAMMAS}.isdisjoint(expected_existing.values())
