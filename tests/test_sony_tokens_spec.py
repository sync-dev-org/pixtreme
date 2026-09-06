"""Specification tests for Sony S-Gamut, S-Log, and S-Log2 tokens."""

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
_C = np.float64("0.030001222851889303")


def _piecewise(
    values: np.ndarray,
    cut: float,
    lower: Callable[[np.ndarray], np.ndarray],
    upper: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    lower_mask = values < cut
    result[lower_mask] = lower(values[lower_mask])
    result[~lower_mask] = upper(values[~lower_mask])
    return result


def _slog_encode(values: np.ndarray) -> np.ndarray:
    x = values / np.float64(0.9)
    y = _piecewise(
        x,
        np.float64(0.0),
        lambda part: np.float64(5.0) * part + _C,
        lambda part: (
            np.float64(0.432699) * np.log10(part + np.float64(0.037584)) + np.float64(0.616596) + np.float64(0.03)
        ),
    )
    return (np.float64(64.0) + np.float64(876.0) * y) / np.float64(1023.0)


def _slog_decode(values: np.ndarray) -> np.ndarray:
    y = (np.float64(1023.0) * values - np.float64(64.0)) / np.float64(876.0)
    x = _piecewise(
        y,
        _C,
        lambda part: (part - _C) / np.float64(5.0),
        lambda part: (
            np.float64(10.0) ** ((part - np.float64(0.616596) - np.float64(0.03)) / np.float64(0.432699))
            - np.float64(0.037584)
        ),
    )
    return np.float64(0.9) * x


def _slog2_encode(values: np.ndarray) -> np.ndarray:
    x = values / np.float64(0.9)
    y = _piecewise(
        x,
        np.float64(0.0),
        lambda part: np.float64("3.53881278538813") * part + _C,
        lambda part: (
            np.float64(0.432699) * np.log10(np.float64(155.0) * part / np.float64(219.0) + np.float64(0.037584))
            + np.float64(0.616596)
            + np.float64(0.03)
        ),
    )
    return (np.float64(64.0) + np.float64(876.0) * y) / np.float64(1023.0)


def _slog2_decode(values: np.ndarray) -> np.ndarray:
    y = (np.float64(1023.0) * values - np.float64(64.0)) / np.float64(876.0)
    x = _piecewise(
        y,
        _C,
        lambda part: (part - _C) / np.float64("3.53881278538813"),
        lambda part: (
            np.float64(219.0)
            * (
                np.float64(10.0) ** ((part - np.float64(0.616596) - np.float64(0.03)) / np.float64(0.432699))
                - np.float64(0.037584)
            )
            / np.float64(155.0)
        ),
    )
    return np.float64(0.9) * x


_CURVES = (
    ("S-Log", _slog_encode, _slog_decode, (90, 394, 636)),
    ("S-Log2", _slog2_encode, _slog2_decode, (90, 347, 582)),
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
    white_x, white_y = white
    white_xyz = np.asarray((white_x / white_y, 1.0, (1.0 - white_x - white_y) / white_y), dtype=np.float64)
    scales = np.linalg.solve(primary_matrix, white_xyz)
    return primary_matrix @ np.diag(scales)


def test_sony_tokens_extend_the_canonical_vocabulary_and_public_static_surfaces() -> None:
    """v1-sony-tokens acceptance 1-2; v1-arri-tokens acceptance 16-17; v1-blackmagic-tokens acceptance 33-34;
    v1-red-tokens acceptance 54-55; v1-canon-tokens acceptance 76-77; v1-panasonic-tokens acceptance 99-100;
    v1-standard-tokens acceptance 117; v1-vendor-a-tokens acceptance 140-141;
    v1-vendor-b-tokens acceptance 166-167.

    Canonical aliases and public surfaces expose only the feature token additions.
    """
    assert get_args(px.core.Colorspace) == _COLORSPACES
    assert get_args(px.core.Gamma) == _GAMMAS
    assert len(_ALIASES) == 30
    assert sum(len(get_args(alias)) for alias in _ALIASES) == 188
    assert _literal_strings(get_type_hints(px.color.linear_to_gamma)["gamma"]) == _GAMMAS
    assert _literal_strings(get_type_hints(px.color.rgb_to_rgb)["input_colorspace"]) == _COLORSPACES
    assert _literal_strings(get_type_hints(px.color.rgb_to_rgb)["output_gamma"]) == _GAMMAS

    frame = _frame((0.18,), colorspace="S-Gamut", gamma="S-Log")
    assert (frame.colorspace, frame.gamma) == ("S-Gamut", "S-Log")
    assert "colorspace='S-Gamut'" in repr(frame)
    assert "gamma='S-Log'" in repr(frame)


def test_sony_token_keys_are_collision_free_family_local_and_separator_normalized() -> None:
    """v1-sony-tokens acceptance 3; v1-blackmagic-tokens acceptance 35; v1-vendor-a-tokens acceptance 142:
    token keys remain local and unique.
    """
    from pixtreme._core.validation import _normalized_closed_token
    from pixtreme._core.vocabulary import _PERMANENT_TOKEN_ALIASES

    assert len({_token_key(token) for token in _COLORSPACES}) == len(_COLORSPACES)
    assert len({_token_key(token) for token in _GAMMAS}) == len(_GAMMAS)
    assert all(canonical not in {"S-Gamut", "S-Log", "S-Log2"} for _alias, canonical in _PERMANENT_TOKEN_ALIASES)

    for canonical, family, axis in (
        ("S-Gamut", _COLORSPACES, "colorspace"),
        ("S-Log", _GAMMAS, "gamma"),
        ("S-Log2", _GAMMAS, "gamma"),
    ):
        for variant in _variants(canonical):
            assert _normalized_closed_token(variant, axis=axis, accepted=family) == canonical

    assert {_token_key(token) for token in ("S-Log", "S-Log2", "S-Log3")} == {"slog", "slog2", "slog3"}
    assert {_token_key(token) for token in ("S-Gamut", "S-Gamut3", "S-Gamut3.Cine")} == {
        "sgamut",
        "sgamut3",
        "sgamut3cine",
    }
    with pytest.raises(ValueError):
        _normalized_closed_token("S-Log", axis="colorspace", accepted=_COLORSPACES)
    with pytest.raises(ValueError):
        _normalized_closed_token("S-Gamut", axis="gamma", accepted=_GAMMAS)


@pytest.mark.parametrize(("gamma", "encode", "_decode", "anchors"), _CURVES)
def test_sony_log_encode_matches_independent_float64_oracles_and_published_anchors(
    gamma: str,
    encode: Callable[[np.ndarray], np.ndarray],
    _decode: Callable[[np.ndarray], np.ndarray],
    anchors: tuple[int, int, int],
) -> None:
    """v1-sony-tokens acceptance 4-7: encode uses signed Sony branches, normalization, and exact anchors."""
    zero = np.float32(0.0)
    values = np.asarray(
        (-0.25, np.nextafter(zero, np.float32(-np.inf)), zero, np.nextafter(zero, np.float32(np.inf)), 0.18, 0.9, 1.5),
        dtype=np.float32,
    )
    actual = _red_values(px.color.linear_to_gamma(_frame(values), gamma=gamma))
    expected = encode(values.astype(np.float64))

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=6e-6)
    anchor_values = np.asarray((0.0, 0.18, 0.9), dtype=np.float32)
    anchor_encoded = _red_values(px.color.linear_to_gamma(_frame(anchor_values), gamma=gamma))
    np.testing.assert_array_equal(
        np.rint(anchor_encoded * np.float32(1023.0)).astype(np.int64),
        np.asarray(anchors, dtype=np.int64),
    )


@pytest.mark.parametrize(("gamma", "encode", "decode", "_anchors"), _CURVES)
def test_sony_log_decode_matches_independent_float64_oracles_through_the_branch_cut(
    gamma: str,
    encode: Callable[[np.ndarray], np.ndarray],
    decode: Callable[[np.ndarray], np.ndarray],
    _anchors: tuple[int, int, int],
) -> None:
    """v1-sony-tokens acceptance 4-7: decode inverts legal embedding and selects the signed lower branch."""
    code_cut = np.float32((np.float64(64.0) + np.float64(876.0) * _C) / np.float64(1023.0))
    encoded = encode(np.asarray((-0.25, 0.0, 0.18, 0.9, 1.5), dtype=np.float64)).astype(np.float32)
    values = np.concatenate(
        (
            encoded,
            np.asarray(
                (
                    np.nextafter(code_cut, np.float32(-np.inf)),
                    code_cut,
                    np.nextafter(code_cut, np.float32(np.inf)),
                ),
                dtype=np.float32,
            ),
        )
    )

    actual = _red_values(px.color.gamma_to_linear(_frame(values, gamma=gamma), gamma=gamma))
    expected = decode(values.astype(np.float64))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-5)


@pytest.mark.parametrize(("gamma", "encode", "decode", "_anchors"), _CURVES)
def test_sony_log_round_trips_both_directions_without_clipping(
    gamma: str,
    encode: Callable[[np.ndarray], np.ndarray],
    decode: Callable[[np.ndarray], np.ndarray],
    _anchors: tuple[int, int, int],
) -> None:
    """v1-sony-tokens acceptance 8: negative, boundary, anchor, and overshoot values round-trip both ways."""
    linear = np.asarray((-0.25, -1e-6, 0.0, 1e-6, 0.18, 0.9, 1.5), dtype=np.float32)
    encoded_values = encode(linear.astype(np.float64)).astype(np.float32)

    encoded = px.color.linear_to_gamma(_frame(linear), gamma=gamma)
    restored_linear = px.color.gamma_to_linear(encoded, gamma=gamma)
    decoded = px.color.gamma_to_linear(_frame(encoded_values, gamma=gamma), gamma=gamma)
    restored_encoded = px.color.linear_to_gamma(decoded, gamma=gamma)

    np.testing.assert_allclose(_red_values(encoded), encode(linear.astype(np.float64)), rtol=0.0, atol=6e-6)
    np.testing.assert_allclose(_red_values(restored_linear), linear, rtol=0.0, atol=2e-5)
    np.testing.assert_allclose(_red_values(decoded), decode(encoded_values.astype(np.float64)), rtol=0.0, atol=2e-5)
    np.testing.assert_allclose(_red_values(restored_encoded), encoded_values, rtol=0.0, atol=2e-5)


def test_slog3_nonnegative_bits_and_metadata_identity_remain_unchanged() -> None:
    """v1-sony-tokens acceptance 6: S-Log3 keeps its established nonnegative GPU bits and token identity."""
    linear = np.asarray((0.0, 0.01125, 0.18, 1.0, 1.5), dtype=np.float32)
    encoded = _red_values(px.color.linear_to_gamma(_frame(linear), gamma="S-Log3"))
    np.testing.assert_array_equal(
        encoded.view(np.uint32),
        np.asarray((1035874188, 1043030190, 1053963405, 1058575679, 1059324708), dtype=np.uint32),
    )
    frame = px.color.linear_to_gamma(_frame(linear), gamma="S-Log3")
    assert frame.gamma == "S-Log3"


@pytest.mark.parametrize("gamma", ("S-Log", "S-Log2"))
def test_all_transfer_paths_preserve_frame_contract_and_auxiliary_bits(gamma: str) -> None:
    """v1-sony-tokens acceptance 8: standalone and fused paths preserve copy, labels, metadata, and auxiliary bits."""
    values = np.asarray((-0.25, 0.0, 0.18, 0.9, 1.5), dtype=np.float32)
    source = _frame(values, auxiliary=True)
    before = source.data.copy()

    encoded = px.color.linear_to_gamma(source, gamma=gamma)
    fused_encoded = px.color.rgb_to_rgb(source, output_gamma=gamma)
    decoded = px.color.gamma_to_linear(encoded, gamma=gamma)
    fused_decoded = px.color.rgb_to_rgb(encoded, output_gamma="linear")

    assert cp.array_equal(encoded.data, fused_encoded.data)
    assert cp.array_equal(decoded.data, fused_decoded.data)
    assert cp.array_equal(encoded.data[..., 0], source.data[..., 0])
    assert cp.array_equal(decoded.data[..., 0], source.data[..., 0])
    assert cp.array_equal(source.data, before)
    assert (source.colorspace, source.gamma, source.channels, source.matrix) == (
        "ACEScg",
        "linear",
        ("Z", "B", "R", "G"),
        "native",
    )
    assert (encoded.colorspace, encoded.gamma, encoded.channels, encoded.matrix) == (
        "ACEScg",
        gamma,
        source.channels,
        None,
    )
    assert (decoded.colorspace, decoded.gamma, decoded.channels, decoded.matrix) == (
        "ACEScg",
        "linear",
        source.channels,
        None,
    )
    assert encoded is not source and encoded.data.data.ptr != source.data.data.ptr


def test_sgamut_primaries_and_native_row_match_an_independent_float64_matrix() -> None:
    """v1-sony-tokens acceptance 9: S-Gamut conversion and native luma match the independent xy matrix oracle."""
    sgamut = _rgb_to_xyz(
        ((0.73, 0.28), (0.14, 0.855), (0.10, -0.05)),
        (0.3127, 0.3290),
    )
    rec709 = _rgb_to_xyz(
        ((0.64, 0.33), (0.30, 0.60), (0.15, 0.06)),
        (0.3127, 0.3290),
    )
    conversion = np.linalg.inv(rec709) @ sgamut
    values = np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (-0.25, 0.18, 1.5)))
    source = _frame(values, colorspace="S-Gamut")

    converted = px.color.rgb_to_rgb(source, output_colorspace="Rec.709")
    grayscale = px.color.rgb_to_grayscale(source, colorspace="S-Gamut", gamma="linear", matrix="native")

    np.testing.assert_allclose(px.io.to_array(converted).get()[0], values @ conversion.T, rtol=0.0, atol=5e-6)
    np.testing.assert_allclose(px.io.to_array(grayscale).get()[0, :, 0], values @ sgamut[1], rtol=0.0, atol=5e-6)
    assert grayscale.matrix == "native"


def test_sgamut_and_sgamut3_are_bit_identical_but_keep_distinct_metadata() -> None:
    """v1-sony-tokens acceptance 9-10: equivalent gamuts preserve pixels while retaining distinct token identity."""
    values = np.asarray(((1.0, -0.25, 0.18), (0.2, 0.4, 1.5)), dtype=np.float32)
    sgamut = _frame(values, colorspace="S-Gamut")
    sgamut3 = _frame(values, colorspace="S-Gamut3")

    to_sgamut3 = px.color.rgb_to_rgb(sgamut, output_colorspace="S-Gamut3")
    to_sgamut = px.color.rgb_to_rgb(sgamut3, output_colorspace="S-Gamut")
    rec709_from_sgamut = px.color.rgb_to_rgb(sgamut, output_colorspace="Rec.709")
    rec709_from_sgamut3 = px.color.rgb_to_rgb(sgamut3, output_colorspace="Rec.709")
    gray_sgamut = px.color.rgb_to_grayscale(sgamut, matrix="native")
    gray_sgamut3 = px.color.rgb_to_grayscale(sgamut3, matrix="native")

    assert cp.array_equal(to_sgamut3.data, sgamut.data)
    assert cp.array_equal(to_sgamut.data, sgamut3.data)
    assert (to_sgamut3.colorspace, to_sgamut.colorspace) == ("S-Gamut3", "S-Gamut")
    assert cp.array_equal(rec709_from_sgamut.data, rec709_from_sgamut3.data)
    assert cp.array_equal(gray_sgamut.data, gray_sgamut3.data)
    assert (gray_sgamut.matrix, gray_sgamut3.matrix) == ("native", "native")


@pytest.mark.parametrize(
    ("operation", "parameter", "rejected", "candidates"),
    (
        ("linear_to_gamma", "gamma", "unknown", _GAMMAS),
        ("linear_to_gamma", "gamma", 17, _GAMMAS),
        ("rgb_to_rgb", "output_colorspace", "unknown", _COLORSPACES),
        ("rgb_to_rgb", "output_colorspace", 17, _COLORSPACES),
    ),
)
def test_invalid_tokens_fail_before_gpu_with_raw_ordered_canonical_errors(
    operation: str,
    parameter: str,
    rejected: object,
    candidates: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1-sony-tokens acceptance 11; v1-blackmagic-tokens acceptance 49;
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
    assert "'slog'" not in message
    assert "'sgamut'" not in message


def test_sony_public_documents_docstrings_and_changelog_are_synchronized() -> None:
    """v1-sony-tokens acceptance 12; v1-arri-tokens acceptance 29; v1-blackmagic-tokens acceptance 50;
    v1-red-tokens acceptance 72; v1-canon-tokens acceptance 93; v1-panasonic-tokens acceptance 112;
    v1-vendor-a-tokens acceptance 161; v1-vendor-b-tokens acceptance 188.

    GitHub #29: docs expose normalization, signed branches, anchors, gamut equivalence, and current counts.
    """
    tokens = (ROOT / "docs_site" / "tokens.md").read_text(encoding="utf-8")
    requirements = require_repo_file("docs/requirements.md").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    normalized_docstrings = {
        function.__name__: " ".join((inspect.getdoc(function) or "").split())
        for function in (px.color.gamma_to_linear, px.color.linear_to_gamma, px.color.rgb_to_rgb)
    }

    for claim in (
        "`x = r / 0.9`",
        "`e = (64 + 876 * y) / 1023`",
        "`90 / 394 / 636`",
        "`90 / 347 / 582`",
        "S-Log / S-Log2 / S-Log3 apply their lower linear branches directly to signed inputs",
        "S-Gamut and S-Gamut3 are numerically identical",
        "the algebraic inverse of Sony's published S-Log1 decoder linear branch",
        "not a separately published Sony forward equation",
    ):
        assert claim in tokens
    for claim in ("27 Colorspace", "33 Gamma", "188 canonical tokens"):
        assert claim in requirements
    for claim in (
        "S-Gamut",
        "S-Log",
        "S-Log2",
        "90 / 394 / 636",
        "90 / 347 / 582",
        "S-Gamut3 and S-Log3 remain unchanged",
        "the algebraic inverse of Sony's published decoder linear branch",
    ):
        assert claim in changelog
    for name, docstring in normalized_docstrings.items():
        assert "S-Log / S-Log2 / S-Log3 apply their lower linear branches directly to signed inputs" in docstring, name
        assert "x = r / 0.9" in docstring, name
        assert "e = (64 + 876 * y) / 1023" in docstring, name
