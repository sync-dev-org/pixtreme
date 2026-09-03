"""Specification tests for named numeric gamma tokens and BT.1886 specialization."""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Literal, get_args, get_origin, get_type_hints

import cupy as cp
import numpy as np
import pytest

import pixtreme as px

_GAMMA_TOKENS = (
    "linear",
    "sRGB",
    "Rec.709",
    "BT.1886",
    "PQ",
    "HLG",
    "S-Log",
    "S-Log2",
    "S-Log3",
    "ARRI-LogC3",
    "ARRI-LogC4",
    "Blackmagic-Film-Gen-5",
    "DaVinci-Intermediate",
    "RED-Log3G10",
    "REDlogFilm",
    "Cineon",
    "Gamma-2.2",
    "Gamma-2.4",
    "Gamma-2.6",
)
_RENAMES = (("2.2", "Gamma-2.2"), ("2.4", "Gamma-2.4"), ("2.6", "Gamma-2.6"))
_SEPARATORS = " .-_"
_SEPARATOR_TRANSLATION = str.maketrans("", "", _SEPARATORS)


def _token_key(value: str) -> str:
    return value.translate(_SEPARATOR_TRANSLATION).casefold()


def _separator_variants(value: str) -> tuple[str, ...]:
    compact = value.translate(_SEPARATOR_TRANSLATION)
    replaced = tuple(
        "".join(separator if character in _SEPARATORS else character for character in value)
        for separator in _SEPARATORS
    )
    return tuple(dict.fromkeys((value, value.swapcase(), compact, *replaced)))


def _literal_strings(annotation: object) -> tuple[str, ...]:
    if get_origin(annotation) is Literal:
        return tuple(value for value in get_args(annotation) if isinstance(value, str))
    return tuple(value for argument in get_args(annotation) for value in _literal_strings(argument))


def _frame(values: tuple[float, ...], *, gamma: str = "linear", channels: str = "RGB") -> px.core.Frame:
    channel_count = len(channels)
    row = np.repeat(np.asarray(values, dtype=np.float32)[:, None], channel_count, axis=1)
    return px.io.from_array(
        cp.asarray(row[None, :, :]),
        colorspace="ACEScg",
        gamma=gamma,
        channels=channels,
    )


def _section(markdown: str, heading: str) -> str:
    start = re.search(rf"^##+ {re.escape(heading)}\n", markdown, re.MULTILINE)
    assert start is not None
    remainder = markdown[start.end() :]
    end = re.search(r"^##+ ", remainder, re.MULTILINE)
    return remainder if end is None else remainder[: end.start()]


def _table_rows(markdown: str, heading: str) -> tuple[tuple[str, ...], ...]:
    lines = _section(markdown, heading).splitlines()
    header_index = next(index for index, line in enumerate(lines) if line.startswith("|"))
    rows: list[tuple[str, ...]] = []
    for line in lines[header_index + 2 :]:
        if not line.startswith("|"):
            break
        rows.append(tuple(cell.strip() for cell in line.strip().strip("|").split("|")))
    return tuple(rows)


def test_gamma_literal_retains_named_numeric_tokens_in_the_sony_extended_vocabulary() -> None:
    """v1-gamma-named-tokens acceptance 1; v1-sony-tokens acceptance 1; v1-arri-tokens acceptance 16;
    v1-blackmagic-tokens acceptance 33; v1-red-tokens acceptance 54-55.
    """
    assert get_args(px.core.Gamma) == _GAMMA_TOKENS
    assert len(get_args(px.core.Gamma)) == 19

    aliases = (
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
    assert len(aliases) == 30
    assert sum(len(get_args(alias)) for alias in aliases) == 165


def test_public_static_and_metadata_surfaces_expose_only_named_gamma_tokens() -> None:
    """v1-gamma-named-tokens acceptance 2; v1-sony-tokens acceptance 2;
    v1-blackmagic-tokens acceptance 34; v1-red-tokens acceptance 54-55: static surfaces expose canonical names.
    """
    assert _literal_strings(get_type_hints(px.color.linear_to_gamma)["gamma"]) == _GAMMA_TOKENS
    assert _literal_strings(get_type_hints(px.color.rgb_to_rgb)["output_gamma"]) == _GAMMA_TOKENS

    frame = _frame((0.18,), gamma="Gamma-2.4")
    assert frame.gamma == "Gamma-2.4"
    assert "gamma='Gamma-2.4'" in repr(frame)

    with pytest.raises(ValueError) as error:
        _frame((0.18,), gamma="unknown")
    message = str(error.value)
    for legacy, canonical in _RENAMES:
        assert repr(canonical) in message
        assert repr(legacy) not in message


def test_named_gamma_tokens_accept_case_and_all_separator_variants() -> None:
    """v1-gamma-named-tokens acceptance 3; v1-sony-tokens acceptance 3: gamma variants normalize family-locally."""
    from pixtreme._core.validation import _normalized_closed_token

    for _legacy, canonical in _RENAMES:
        for variant in _separator_variants(canonical):
            assert _normalized_closed_token(variant, axis="gamma", accepted=_GAMMA_TOKENS) == canonical


def test_numeric_gamma_aliases_and_separator_variants_are_permanent_inputs() -> None:
    """v1-gamma-named-tokens acceptance 4: all numeric alias variants normalize through public boundaries."""
    from pixtreme._core.validation import _normalized_closed_token

    linear = _frame((-0.25, 0.0, 0.18, 1.25))
    for legacy, canonical in _RENAMES:
        for variant in _separator_variants(legacy):
            assert _normalized_closed_token(variant, axis="gamma", accepted=_GAMMA_TOKENS) == canonical
            frame = _frame((0.18,), gamma=variant)
            assert frame.gamma == canonical
            encoded = px.color.linear_to_gamma(linear, gamma=variant)
            assert encoded.gamma == canonical
            decoded = px.color.gamma_to_linear(encoded, gamma=variant)
            assert decoded.gamma == "linear"


def test_numeric_alias_keys_are_collision_free_subset_local_and_order_independent() -> None:
    """v1-gamma-named-tokens acceptance 5; v1-sony-tokens acceptance 3;
    v1-blackmagic-tokens acceptance 35: keys stay collision-free and local.
    """
    from pixtreme._core.validation import _normalized_closed_token

    canonical_keys = tuple(map(_token_key, _GAMMA_TOKENS))
    alias_keys = tuple(_token_key(legacy) for legacy, _canonical in _RENAMES)
    assert len(canonical_keys) == len(set(canonical_keys))
    assert len(alias_keys) == len(set(alias_keys))
    assert set(alias_keys).isdisjoint(canonical_keys)

    for legacy, canonical in _RENAMES:
        assert _normalized_closed_token(legacy, axis="gamma", accepted=(canonical, "linear")) == canonical
        assert _normalized_closed_token(legacy, axis="gamma", accepted=("linear", canonical)) == canonical
        with pytest.raises(ValueError):
            _normalized_closed_token(legacy, axis="gamma", accepted=("linear",))
        with pytest.raises(ValueError):
            _normalized_closed_token(legacy, axis="colorspace", accepted=get_args(px.core.Colorspace))


@pytest.mark.parametrize("rejected", ["unknown", 24, None])
def test_invalid_gamma_fails_before_gpu_with_raw_input_and_canonical_recovery(
    rejected: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v1-gamma-named-tokens acceptance 6: invalid gamma fails before GPU with ordered raw actionable details."""
    import pixtreme._color.semantics as semantics

    source = _frame((0.18,))

    def unexpected_backend(*_args: object, **_kwargs: object) -> cp.ndarray:
        raise AssertionError("GPU backend was reached")

    monkeypatch.setattr(semantics, "_run_transform", unexpected_backend)
    with pytest.raises(ValueError) as error:
        px.color.linear_to_gamma(source, gamma=rejected)
    message = str(error.value)
    assert message.index("why=") < message.index("what=") < message.index("how=")
    assert repr(rejected) in message
    for legacy, canonical in _RENAMES:
        assert repr(canonical) in message
        assert repr(legacy) not in message


def test_named_and_numeric_alias_transfers_are_bit_identical_and_preserve_observables() -> None:
    """v1-gamma-named-tokens acceptance 7: each rename pair preserves transfer bits and non-token observables."""
    values = (-1.25, -0.25, -1.0e-6, 0.0, 1.0e-6, 0.18, 1.0, 1.5)
    rgb = np.repeat(np.asarray(values, dtype=np.float32)[:, None], 3, axis=1)
    alpha = np.arange(len(values), dtype=np.float32)[:, None]
    source = px.io.from_array(
        cp.asarray(np.concatenate((rgb, alpha), axis=1)[None, :, :]),
        colorspace="ACEScg",
        gamma="linear",
        channels="RGBA",
        matrix="native",
    )
    source_before = source.data.copy()

    for legacy, canonical in _RENAMES:
        named = px.color.linear_to_gamma(source, gamma=canonical)
        aliased = px.color.linear_to_gamma(source, gamma=legacy)
        assert cp.array_equal(named.data, aliased.data)
        assert named.gamma == aliased.gamma == canonical
        assert (named.colorspace, named.channels, named.matrix, named.shape) == (
            aliased.colorspace,
            aliased.channels,
            aliased.matrix,
            aliased.shape,
        )
        assert cp.array_equal(named.data[..., 3], source.data[..., 3])
        assert named is not aliased
        assert named.data.data.ptr != aliased.data.data.ptr != source.data.data.ptr

        named_decoded = px.color.gamma_to_linear(named, gamma=canonical)
        alias_decoded = px.color.gamma_to_linear(aliased, gamma=legacy)
        assert cp.array_equal(named_decoded.data, alias_decoded.data)
        assert (named_decoded.gamma, alias_decoded.gamma) == ("linear", "linear")

    assert cp.array_equal(source.data, source_before)
    assert source.gamma == "linear"
    assert source.matrix == "native"


def test_bt1886_and_gamma_24_are_bit_identical_but_keep_distinct_token_identity() -> None:
    """v1-gamma-named-tokens acceptance 8: ideal-black BT.1886 equals Gamma-2.4 numerically, not semantically."""
    values = (-1.25, -0.25, -1.0e-6, 0.0, 1.0e-6, 0.18, 1.0, 1.5)
    linear = _frame(values)
    bt_encoded = px.color.linear_to_gamma(linear, gamma="BT.1886")
    power_encoded = px.color.linear_to_gamma(linear, gamma="Gamma-2.4")
    assert cp.array_equal(bt_encoded.data, power_encoded.data)
    assert (bt_encoded.gamma, power_encoded.gamma) == ("BT.1886", "Gamma-2.4")

    bt_source = _frame(values, gamma="BT.1886")
    power_source = _frame(values, gamma="Gamma-2.4")
    bt_decoded = px.color.gamma_to_linear(bt_source)
    power_decoded = px.color.gamma_to_linear(power_source)
    assert cp.array_equal(bt_decoded.data, power_decoded.data)
    assert (bt_source.gamma, power_source.gamma) == ("BT.1886", "Gamma-2.4")


def test_token_reference_matches_named_gamma_and_bt1886_document_contract() -> None:
    """v1-gamma-named-tokens acceptance 9; v1-sony-tokens acceptance 12;
    v1-blackmagic-tokens acceptance 50: gamma docs match code and semantics.
    """
    markdown = (Path(__file__).resolve().parents[1] / "docs_site" / "tokens.md").read_text(encoding="utf-8")
    gamma_rows = _table_rows(markdown, "gamma")
    assert tuple(row[0].strip("`") for row in gamma_rows) == get_args(px.core.Gamma) == _GAMMA_TOKENS

    alias_rows = _table_rows(markdown, "Permanent aliases from earlier releases")
    assert len(alias_rows) == 30
    for legacy, canonical in _RENAMES:
        assert ("Gamma", f"`{legacy}`", f"`{canonical}`") in alias_rows

    gamma_section = _section(markdown, "gamma")
    for required in (
        "Annex 1",
        "`L_B = 0`",
        "pure 2.4 power",
        "numerically equivalent",
        "semantically distinct",
        "production and conversion practice",
        "canonical output",
    ):
        assert required in gamma_section


def test_requirements_changelog_docstrings_and_supersede_traces_use_current_gamma_names() -> None:
    """v1-gamma-named-tokens acceptance 10; v1-sony-tokens acceptance 12: canon and public docs are synchronized."""
    root = Path(__file__).resolve().parents[1]
    requirements = (root / "docs" / "requirements.md").read_text(encoding="utf-8")
    arch = requirements.split("**REQ-ARCH-003", maxsplit=1)[1].split("\n\n", maxsplit=1)[0]
    api = requirements.split("**REQ-API-003", maxsplit=1)[1].split("\n\n", maxsplit=1)[0]
    assert "30 token" in arch
    for _legacy, canonical in _RENAMES:
        assert f"`{canonical}`" in api

    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    unreleased = changelog.split("## Unreleased\n", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
    for legacy, canonical in _RENAMES:
        assert f"| Gamma | `{legacy}` | `{canonical}` |" in unreleased
    for required in ("runtime", "numerically equivalent", "BT.1886", "unchanged"):
        assert required in unreleased

    for operation in (
        px.color.rgb_to_ycbcr,
        px.color.ycbcr_to_rgb,
        px.color.rgb_to_grayscale,
        px.color.gamma_to_linear,
        px.color.linear_to_gamma,
    ):
        docstring = inspect.getdoc(operation)
        assert docstring is not None
        for legacy, canonical in _RENAMES:
            assert canonical in docstring
            assert re.search(rf"``{re.escape(legacy)}``", docstring) is None

    token_sheet = (root / "docs" / "features" / "v1-token-vocabulary.md").read_text(encoding="utf-8")
    color_sheet = (root / "docs" / "features" / "v1-color-semantics.md").read_text(encoding="utf-8")
    for number in (1, 3):
        assert re.search(rf"^{number}\. \[trace:superseded-by:v1-gamma-named-tokens\]", token_sheet, re.MULTILINE)
    for number in (27, 28):
        assert re.search(rf"^{number}\. \[trace:superseded-by:v1-gamma-named-tokens\]", color_sheet, re.MULTILINE)
