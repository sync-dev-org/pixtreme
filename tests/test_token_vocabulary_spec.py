"""Specification tests for canonical token vocabulary and runtime acceptance."""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Literal, get_args, get_origin, get_type_hints

import cupy as cp
import pytest
from repository_contracts import require_repo_file

import pixtreme as px

EXPECTED_VOCABULARY: dict[str, tuple[str, ...]] = {
    "ChromaticAdaptation": ("Bradford", "CAT02", "CAT16", "von-Kries"),
    "ReferenceWhite": ("D65", "D93", "D50", "ACES"),
    "Colorspace": (
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
    ),
    "Gamma": (
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
    ),
    "Matrix": ("BT.601", "BT.709", "BT.2020", "native"),
    "Dtype": ("float32", "float16", "uint8", "uint16", "uint32"),
    "Layout": ("HWC", "NHWC", "CHW", "NCHW"),
    "Tonemap": ("ACES-1.3", "ACES-2.0", "BT.2408"),
    "Range": ("legal", "full"),
    "Interpolation": (
        "nearest",
        "bilinear",
        "bicubic",
        "b-spline",
        "mitchell",
        "lanczos2",
        "lanczos3",
        "lanczos4",
        "area",
        "trilinear",
        "tetrahedral",
        "linear",
    ),
    "Border": ("mirror", "replicate", "wrap", "constant"),
    "ChromaSiting": ("left", "center", "topleft"),
    "StackDirection": ("vertical", "horizontal"),
    "SobelDirection": ("x", "y", "magnitude"),
    "TemplateMatchingMethod": ("sqdiff", "sqdiff_normed", "ccorr", "ccorr_normed", "ccoeff", "ccoeff_normed"),
    "Blend": (
        "normal",
        "lighten",
        "add",
        "screen",
        "darken",
        "multiply",
        "difference",
        "overlay",
        "hardlight",
        "softlight",
    ),
    "Alpha": ("premultiplied", "straight"),
    "Antialiasing": ("distance", "supersample", "off"),
    "TextLanguage": ("ja", "zh-hans", "zh-hant", "ko"),
    "TextAnchor": (
        "top-left",
        "top-center",
        "top-right",
        "center-left",
        "center-center",
        "center-right",
        "baseline-left",
        "baseline-center",
        "baseline-right",
        "bottom-left",
        "bottom-center",
        "bottom-right",
    ),
    "TextAlign": ("left", "center", "right", "justify"),
    "TextFont": ("sans", "mono"),
    "GeneratorKind": ("linear", "radial"),
    "ColorBarsStandard": (
        "ARIB-STD-B28",
        "SMPTE-RP219",
        "BT.2111-HLG",
        "BT.2111-PQ",
        "BT.2111-PQ-full",
        "full-100",
        "full-75",
    ),
    "ColorBarsOutput": ("normalized", "code"),
    "MorphologyShape": ("disk", "square"),
    "ImageFormat": ("jpeg", "png", "tiff", "jpeg2000", "webp", "bmp", "pnm"),
    "TiffCompression": ("none", "lzw"),
    "ExrCompression": ("none", "rle", "zip", "zips", "piz", "pxr24", "b44", "b44a", "dwaa", "dwab"),
    "VectorBlurShutter": ("centered", "forward", "backward"),
}

LEGACY_ALIASES: dict[str, tuple[tuple[str, str], ...]] = {
    "Gamma": (
        ("srgb", "sRGB"),
        ("rec709", "Rec.709"),
        ("bt1886", "BT.1886"),
        ("pq", "PQ"),
        ("hlg", "HLG"),
        ("s-log3", "S-Log3"),
        ("logc4", "ARRI-LogC4"),
        ("cineon", "Cineon"),
        ("2.2", "Gamma-2.2"),
        ("2.4", "Gamma-2.4"),
        ("2.6", "Gamma-2.6"),
    ),
    "Matrix": (("bt601", "BT.601"), ("bt709", "BT.709"), ("bt2020", "BT.2020")),
    "ChromaticAdaptation": (
        ("bradford", "Bradford"),
        ("cat02", "CAT02"),
        ("cat16", "CAT16"),
        ("von-kries", "von-Kries"),
    ),
    "ReferenceWhite": (("d65", "D65"), ("d93", "D93"), ("d50", "D50"), ("aces", "ACES")),
    "Tonemap": (("aces-1.3", "ACES-1.3"), ("aces-2.0", "ACES-2.0"), ("bt2408", "BT.2408")),
    "ColorBarsStandard": (
        ("arib-std-b28", "ARIB-STD-B28"),
        ("smpte-rp219", "SMPTE-RP219"),
        ("bt2111-hlg", "BT.2111-HLG"),
        ("bt2111-pq", "BT.2111-PQ"),
        ("bt2111-pq-full", "BT.2111-PQ-full"),
    ),
}

DOC_HEADINGS = {
    "ChromaticAdaptation": "chromatic adaptation",
    "ReferenceWhite": "reference white",
    "Colorspace": "colorspace",
    "Gamma": "gamma",
    "Matrix": "matrix",
    "Dtype": "dtype",
    "Layout": "layout",
    "Tonemap": "tonemap",
    "Range": "range",
    "Interpolation": "interpolation",
    "Border": "border",
    "ChromaSiting": "chroma siting",
    "StackDirection": "stack direction",
    "SobelDirection": "sobel direction",
    "TemplateMatchingMethod": "template matching method",
    "Blend": "blend",
    "Alpha": "alpha",
    "Antialiasing": "aa",
    "TextLanguage": "language",
    "TextAnchor": "anchor",
    "TextAlign": "text align",
    "TextFont": "text font",
    "GeneratorKind": "generator kind",
    "ColorBarsStandard": "color bars standard",
    "ColorBarsOutput": "color bars output",
    "MorphologyShape": "morphology shape",
    "ImageFormat": "image format",
    "TiffCompression": "TIFF compression",
    "ExrCompression": "EXR compression",
    "VectorBlurShutter": "vector blur shutter",
}

_TOKEN_PARAMETERS = {
    "colorspace": "Colorspace",
    "input_colorspace": "Colorspace",
    "output_colorspace": "Colorspace",
    "gamma": "Gamma",
    "input_gamma": "Gamma",
    "output_gamma": "Gamma",
    "matrix": "Matrix",
    "input_matrix": "Matrix",
    "output_matrix": "Matrix",
    "range": "Range",
    "input_range": "Range",
    "output_range": "Range",
    "dtype": "Dtype",
    "layout": "Layout",
    "tonemap": "Tonemap",
    "interpolation": "Interpolation",
    "border": "Border",
    "siting": "ChromaSiting",
    "method": "TemplateMatchingMethod",
    "blend": "Blend",
    "alpha": "Alpha",
    "aa": "Antialiasing",
    "language": "TextLanguage",
    "anchor": "TextAnchor",
    "align": "TextAlign",
    "font": "TextFont",
    "kind": "GeneratorKind",
    "standard": "ColorBarsStandard",
    "output": "ColorBarsOutput",
    "shape": "MorphologyShape",
    "format": "ImageFormat",
    "shutter": "VectorBlurShutter",
    "cat": "ChromaticAdaptation",
    "input_white": "ReferenceWhite",
    "output_white": "ReferenceWhite",
}

_SEPARATORS = " .-_"
_SEPARATOR_TRANSLATION = str.maketrans("", "", _SEPARATORS)


def _independent_token_key(value: str) -> str:
    return value.translate(_SEPARATOR_TRANSLATION).casefold()


def _acceptance_variants(canonical: str) -> tuple[str, ...]:
    compact = canonical.translate(_SEPARATOR_TRANSLATION)
    replaced = tuple(
        "".join(separator if character in _SEPARATORS else character for character in canonical)
        for separator in _SEPARATORS
    )
    interspersed = tuple(separator.join(compact) for separator in _SEPARATORS if len(compact) > 1)
    return tuple(dict.fromkeys((canonical, canonical.swapcase(), compact, *replaced, *interspersed)))


def _literal_strings(annotation: object) -> tuple[str, ...]:
    if get_origin(annotation) is Literal:
        return tuple(value for value in get_args(annotation) if isinstance(value, str))
    return tuple(value for argument in get_args(annotation) for value in _literal_strings(argument))


def _contains_plain_str(annotation: object) -> bool:
    if annotation is str:
        return True
    return any(_contains_plain_str(argument) for argument in get_args(annotation))


def _expected_parameter_families(module_name: str, operation: str, parameter: str) -> tuple[str, ...] | None:
    if operation == "warp_affine" and parameter == "matrix":
        return None
    if parameter == "direction":
        if operation == "stack":
            return ("StackDirection",)
        if operation == "sobel":
            return ("SobelDirection",)
        return None
    if parameter == "compression":
        if operation == "encode_image":
            return ("TiffCompression",)
        if operation == "write_image":
            return ("TiffCompression", "ExrCompression")
        return None
    family = _TOKEN_PARAMETERS.get(parameter)
    if family is None:
        return None
    if parameter == "output" and operation != "color_bars":
        return None
    if parameter == "kind" and operation != "ramp":
        return None
    if parameter == "format" and operation != "encode_image":
        return None
    if parameter == "shape" and module_name != "pixtreme.morphology":
        return None
    return (family,)


def _documentation_tokens(markdown: str, heading: str) -> tuple[str, ...]:
    section = markdown.split(f"## {heading}\n", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
    return tuple(match.group(1) for line in section.splitlines() if (match := re.match(r"^\| `([^`]+)`", line)))


def test_literal_aliases_are_the_independent_canonical_vocabulary() -> None:
    """v1-token-vocabulary acceptance 1; v1-sony-tokens acceptance 1; v1-arri-tokens acceptance 16;
    v1-blackmagic-tokens acceptance 33; v1-red-tokens acceptance 54-55; v1-canon-tokens acceptance 76-77;
    v1-panasonic-tokens acceptance 99-100; v1-standard-tokens acceptance 117;
    v1-vendor-a-tokens acceptance 140; v1-vendor-b-tokens acceptance 166.
    """
    assert len(EXPECTED_VOCABULARY) == 30
    assert sum(map(len, EXPECTED_VOCABULARY.values())) == 188
    assert {name: get_args(getattr(px.core, name)) for name in EXPECTED_VOCABULARY} == EXPECTED_VOCABULARY


def test_public_token_annotations_use_only_canonical_literal_aliases() -> None:
    """v1-token-vocabulary acceptance 2; v1-sony-tokens acceptance 2;
    v1-blackmagic-tokens acceptance 34; v1-red-tokens acceptance 55; v1-canon-tokens acceptance 77;
    v1-panasonic-tokens acceptance 100; v1-vendor-a-tokens acceptance 141; v1-vendor-b-tokens acceptance 167:
    annotations expose canonical literals.
    """
    frame_hints = get_type_hints(px.core.Frame)
    for parameter, family in (("colorspace", "Colorspace"), ("gamma", "Gamma"), ("matrix", "Matrix")):
        annotation = frame_hints[parameter]
        assert not _contains_plain_str(annotation)
        assert set(_literal_strings(annotation)) == set(EXPECTED_VOCABULARY[family])

    for module in (
        px.io,
        px.color,
        px.filter,
        px.transform,
        px.draw,
        px.generate,
        px.morphology,
        px.metrics,
        px.feature,
        px.values,
        px.channel,
        px.composite,
    ):
        for operation in module.__all__:
            member = getattr(module, operation)
            if not inspect.isfunction(member):
                continue
            hints = get_type_hints(member)
            for parameter, descriptor in inspect.signature(member).parameters.items():
                families = _expected_parameter_families(module.__name__, operation, parameter)
                if families is None:
                    continue
                annotation = hints[parameter]
                assert not _contains_plain_str(annotation), (module.__name__, operation, parameter, annotation)
                expected = {token for family in families for token in EXPECTED_VOCABULARY[family]}
                assert set(_literal_strings(annotation)) == expected, (
                    module.__name__,
                    operation,
                    parameter,
                    annotation,
                )
                if isinstance(descriptor.default, str):
                    assert descriptor.default in expected, (module.__name__, operation, parameter, descriptor.default)


def test_every_canonical_token_accepts_case_and_separator_variants() -> None:
    """v1-token-vocabulary acceptance 3; v1-sony-tokens acceptance 3;
    v1-blackmagic-tokens acceptance 35; v1-red-tokens acceptance 56; v1-panasonic-tokens acceptance 101;
    v1-vendor-a-tokens acceptance 142; v1-vendor-b-tokens acceptance 168:
    canonical tokens resolve all variants.
    """
    from pixtreme._core.validation import _normalized_closed_token

    for family, canonical_tokens in EXPECTED_VOCABULARY.items():
        for canonical in canonical_tokens:
            for variant in _acceptance_variants(canonical):
                assert _normalized_closed_token(variant, axis=family, accepted=canonical_tokens) == canonical


def test_all_legacy_spellings_are_permanent_runtime_aliases() -> None:
    """v1-token-vocabulary acceptance 4; v1-sony-tokens acceptance 1;
    v1-red-tokens acceptance 56: the 30 aliases retain their runtime contract.
    """
    from pixtreme._core.validation import _normalized_closed_token

    assert sum(map(len, LEGACY_ALIASES.values())) == 30
    for family, aliases in LEGACY_ALIASES.items():
        for legacy, canonical in aliases:
            assert _normalized_closed_token(legacy, axis=family, accepted=EXPECTED_VOCABULARY[family]) == canonical


def test_token_keys_are_collision_free_per_family_and_never_cross_families() -> None:
    """v1-token-vocabulary acceptance 5; v1-sony-tokens acceptance 3;
    v1-blackmagic-tokens acceptance 35; v1-panasonic-tokens acceptance 101;
    v1-vendor-a-tokens acceptance 142; v1-vendor-b-tokens acceptance 168: keys remain unique and family-local.
    """
    from pixtreme._core.validation import _normalized_closed_token

    for family, canonical_tokens in EXPECTED_VOCABULARY.items():
        keys = tuple(map(_independent_token_key, canonical_tokens))
        assert len(keys) == len(set(keys)), family

    assert _normalized_closed_token("s_r_g_b", axis="colorspace", accepted=EXPECTED_VOCABULARY["Colorspace"]) == "sRGB"
    assert _normalized_closed_token("s_r_g_b", axis="gamma", accepted=EXPECTED_VOCABULARY["Gamma"]) == "sRGB"
    with pytest.raises(ValueError):
        _normalized_closed_token("PQ", axis="colorspace", accepted=EXPECTED_VOCABULARY["Colorspace"])


def test_frame_and_array_boundary_expose_only_canonical_metadata() -> None:
    """v1-token-vocabulary acceptance 6; v1-blackmagic-tokens acceptance 34-35;
    v1-vendor-a-tokens acceptance 141; v1-vendor-b-tokens acceptance 167: metadata is canonical.
    """
    frame = px.io.from_array(
        cp.zeros((1, 2, 3), dtype=cp.float32),
        colorspace="REC_709",
        gamma="bt1886",
        channels="RGB",
        matrix="bt_709",
        layout="hwc",
        dtype="FLOAT_32",
    )
    assert (frame.colorspace, frame.gamma, frame.matrix) == ("Rec.709", "BT.1886", "BT.709")
    assert "colorspace='Rec.709'" in repr(frame)
    assert "gamma='BT.1886'" in repr(frame)
    assert "matrix='BT.709'" in repr(frame)

    frame.colorspace = "aces_cg"
    frame.gamma = "s-log3"
    frame.matrix = "native"
    assert (frame.colorspace, frame.gamma, frame.matrix) == ("ACEScg", "S-Log3", "native")


@pytest.mark.parametrize("rejected", ["unknown", "", " .-_ ", 709, None])
def test_invalid_tokens_fail_with_raw_actionable_errors(rejected: object) -> None:
    """v1-token-vocabulary acceptance 7; v1-sony-tokens acceptance 11;
    v1-blackmagic-tokens acceptance 49; v1-panasonic-tokens acceptance 111;
    v1-vendor-a-tokens acceptance 160; v1-vendor-b-tokens acceptance 187:
    invalid tokens fail with canonical errors.
    """
    with pytest.raises(ValueError) as error:
        px.io.from_array(
            cp.zeros((1, 1, 3), dtype=cp.float32),
            colorspace=rejected,
            gamma="linear",
            channels="RGB",
        )
    message = str(error.value)
    assert message.index("why=") < message.index("what=") < message.index("how=")
    assert f"{rejected!r}" in message
    assert repr(EXPECTED_VOCABULARY["Colorspace"]) in message


def test_gamma_aliases_preserve_pixels_and_non_token_observables() -> None:
    """v1-token-vocabulary acceptance 8: canonical, case, separator, and legacy inputs are observably equivalent."""
    source = px.io.from_array(
        cp.asarray([[[-0.1, 0.18, 1.25]]], dtype=cp.float32),
        colorspace="sRGB",
        gamma="linear",
        channels="RGB",
    )
    canonical = px.color.linear_to_gamma(source, gamma="Rec.709")
    for accepted in ("rEC.709", "REC_709", "rec 709", "rec709"):
        result = px.color.linear_to_gamma(source, gamma=accepted)
        assert cp.array_equal(result.data, canonical.data)
        assert result.gamma == canonical.gamma == "Rec.709"
        assert (result.colorspace, result.channels, result.matrix, result.shape) == (
            canonical.colorspace,
            canonical.channels,
            canonical.matrix,
            canonical.shape,
        )
        assert result is not canonical
        assert result.data.data.ptr != canonical.data.data.ptr


def test_token_reference_matches_all_literal_aliases_in_order() -> None:
    """v1-token-vocabulary acceptance 9; v1-sony-tokens acceptance 12;
    v1-blackmagic-tokens acceptance 50; v1-panasonic-tokens acceptance 112;
    v1-vendor-a-tokens acceptance 161; v1-vendor-b-tokens acceptance 188;
    GitHub #29: token tables match all aliases in order.
    """
    path = require_repo_file("docs_site/tokens.md")
    markdown = path.read_text(encoding="utf-8")
    assert {
        family: _documentation_tokens(markdown, heading) for family, heading in DOC_HEADINGS.items()
    } == EXPECTED_VOCABULARY
    assert "case-insensitive" in markdown
    assert "permanent aliases" in markdown
    assert "canonical output" in markdown
    # OpenType axis tags remain a legitimately case-sensitive open vocabulary; only
    # token-family claims of case sensitivity or literal stamping are stale.
    assert "case-sensitive token" not in markdown.lower()
    assert "case-sensitive closed" not in markdown.lower()
    assert "case variants are invalid" not in markdown
    assert "stamped literally" not in markdown


def test_requirements_define_the_two_layer_token_contract() -> None:
    """v1-token-vocabulary acceptance 10; v1-sony-tokens acceptance 12;
    v1-vendor-b-tokens acceptance 188; GitHub #29: requirements define tokens.
    """
    path = require_repo_file("docs/requirements.md")
    requirements = path.read_text(encoding="utf-8")
    arch = requirements.split("**REQ-ARCH-003", maxsplit=1)[1].split("\n\n", maxsplit=1)[0]
    assert "case-sensitive" not in arch
    assert "casefold" in arch
    assert "U+0020" in arch
    assert "permanent alias" in arch
    for canonical in ("BT.601", "BT.709", "BT.2020", "Rec.709", "BT.1886", "ACES-1.3", "BT.2408"):
        assert f"`{canonical}`" in requirements or f'"{canonical}"' in requirements


def test_changelog_records_all_breaking_renames_and_runtime_compatibility() -> None:
    """v1-token-vocabulary acceptance 11; v1-canon-tokens acceptance 93; GitHub #29: preserve release history."""
    changelog = (Path(__file__).resolve().parents[1] / "CHANGELOG.md").read_text(encoding="utf-8")
    release_130 = changelog.split("## 1.3.0", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
    assert "Breaking" in release_130
    assert "permanent" in release_130
    for aliases in LEGACY_ALIASES.values():
        for legacy, canonical in aliases:
            assert f"`{legacy}`" in release_130
            assert f"`{canonical}`" in release_130
