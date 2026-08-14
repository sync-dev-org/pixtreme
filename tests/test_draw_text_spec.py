"""Specification, contract, and numerical-property tests for text drawing."""

from __future__ import annotations

import inspect
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import pixtreme as px
import pixtreme._draw.text as draw_text_module

ROOT = Path(__file__).resolve().parents[1]
FONT_PATH = ROOT / "src" / "pixtreme" / "data" / "fonts" / "NotoSansCJKjp-VF.otf"
LICENSE_PATH = ROOT / "src" / "pixtreme" / "data" / "fonts" / "LICENSE-NotoSansCJK"
LANGUAGES = ("ja", "zh-hans", "zh-hant", "ko")
ANCHORS = tuple(
    f"{vertical}-{horizontal}"
    for vertical in ("top", "center", "baseline", "bottom")
    for horizontal in ("left", "center", "right")
)
BLENDS = ("normal", "add", "multiply", "screen")


def _frame(
    values: Any,
    *,
    colorspace: str = "ACEScg",
    gamma: str = "linear",
    channels: str | Sequence[str] = "RGB",
) -> px.core.Frame:
    import cupy as cp

    return px.io.from_array(
        cp.asarray(np.asarray(values, dtype=np.float32)),
        colorspace=colorspace,
        gamma=gamma,
        channels=channels,
    )


def _zeros(
    height: int = 96,
    width: int = 192,
    channels: Sequence[str] = ("R", "G", "B"),
) -> px.core.Frame:
    return _frame(np.zeros((height, width, len(channels)), dtype=np.float32), channels=channels)


def _host(result: px.core.Frame) -> np.ndarray:
    return px.io.to_array(
        result,
    ).get()


def _assert_actionable(error: pytest.ExceptionInfo[ValueError]) -> None:
    message = str(error.value)
    assert "why=" in message
    assert "; what=" in message
    assert "; how=" in message


def test_draw_text_host_array_conversion_failure_is_actionable() -> None:
    """REQ-API-012: text host-array conversion reports the rejected value and a concrete recovery."""
    value = ((1.0,), (1.0, 2.0))
    with pytest.raises(ValueError) as error:
        draw_text_module._host_array(value)
    _assert_actionable(error)
    assert repr(value) in str(error.value)


def test_draw_text_unsupported_freetype_bitmap_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-API-012: unsupported FreeType bitmap layouts report the observed layout and repair contract."""

    class Bitmap:
        rows = 1
        width = 2
        pitch = 1
        pixel_mode = 0

    class Rendered:
        bitmap = Bitmap()

    class Glyph:
        def to_bitmap(self, *_args: object, **_kwargs: object) -> Rendered:
            return Rendered()

    class GlyphSlot:
        def get_glyph(self) -> Glyph:
            return Glyph()

    class Face:
        glyph = GlyphSlot()

        def load_glyph(self, *_args: object, **_kwargs: object) -> None:
            return None

    monkeypatch.setattr(draw_text_module, "_freetype_face", lambda *_args: Face())
    draw_text_module._glyph_bitmap.cache_clear()
    try:
        with pytest.raises(RuntimeError) as error:
            draw_text_module._glyph_bitmap(7, 64, 400.0, 0, 0, 0)
    finally:
        draw_text_module._glyph_bitmap.cache_clear()
    message = str(error.value)
    assert "why=" in message
    assert "; what=" in message
    assert "; how=" in message
    assert "pitch=1" in message


def _font_metrics_26_6(size: float) -> tuple[int, int]:
    import freetype

    face = freetype.Face(str(FONT_PATH))
    size_26_6 = max(1, round(size * 64.0))
    return (
        round(face.ascender * size_26_6 / face.units_per_EM),
        round(face.descender * size_26_6 / face.units_per_EM),
    )


def _reference_shape(
    text: str,
    *,
    size: float,
    weight: float,
    language: str,
) -> tuple[tuple[tuple[int, int, int], ...], int]:
    import uharfbuzz as hb

    size_26_6 = max(1, round(size * 64.0))
    face = hb.Face(FONT_PATH.read_bytes())
    font = hb.Font(face)
    font.scale = (size_26_6, size_26_6)
    hb.ot_font_set_funcs(font)
    font.set_variations({"wght": weight})
    buffer = hb.Buffer()
    buffer.add_str(text)
    buffer.guess_segment_properties()
    buffer.direction = "ltr"
    buffer.language = language
    hb.shape(font, buffer, {"kern": True, "liga": True, "locl": True})

    pen_x = 0
    pen_y = 0
    glyphs: list[tuple[int, int, int]] = []
    for info, position in zip(buffer.glyph_infos, buffer.glyph_positions, strict=True):
        glyphs.append(
            (
                info.codepoint,
                pen_x + position.x_offset,
                -(pen_y + position.y_offset),
            )
        )
        pen_x += position.x_advance
        pen_y += position.y_advance
    return tuple(glyphs), pen_x


def _reference_bitmap(
    glyph_id: int,
    *,
    size: float,
    weight: float,
    stroke_radius_26_6: int,
    phase_x_26_6: int,
    phase_y_down_26_6: int,
    supersample: bool = False,
) -> tuple[np.ndarray, int, int]:
    import freetype

    size_26_6 = max(1, round(size * 64.0))

    def render(scale: int) -> tuple[np.ndarray, int, int]:
        face = freetype.Face(str(FONT_PATH))
        face.set_char_size(0, size_26_6 * scale, 72, 72)
        face.set_var_design_coords((weight,))
        face.load_glyph(glyph_id, freetype.FT_LOAD_DEFAULT | freetype.FT_LOAD_NO_BITMAP)
        glyph = face.glyph.get_glyph()
        if stroke_radius_26_6 > 0:
            stroker = freetype.Stroker()
            stroker.set(
                stroke_radius_26_6 * scale,
                freetype.FT_STROKER_LINECAP_ROUND,
                freetype.FT_STROKER_LINEJOIN_ROUND,
                0,
            )
            glyph.stroke(stroker, destroy=False)
        rendered = glyph.to_bitmap(
            freetype.FT_RENDER_MODE_NORMAL,
            freetype.Vector(phase_x_26_6 * scale, -phase_y_down_26_6 * scale),
            destroy=False,
        )
        bitmap = rendered.bitmap
        if bitmap.rows == 0 or bitmap.width == 0:
            coverage = np.zeros((0, 0), dtype=np.float32)
        else:
            pitch = abs(bitmap.pitch)
            coverage = np.asarray(bitmap.buffer, dtype=np.uint8).reshape(bitmap.rows, pitch)[:, : bitmap.width]
            coverage = coverage.astype(np.float32) / np.float32(255.0)
        return coverage, rendered.left, rendered.top

    coverage, left, top = render(1)
    if not supersample or coverage.size == 0:
        return coverage, left, top

    high_coverage, high_left, high_top = render(4)
    samples = np.zeros((coverage.shape[0] * 4, coverage.shape[1] * 4), dtype=np.float32)
    _place_max(samples, high_coverage, left=high_left - left * 4, top=top * 4 - high_top)
    downsampled = samples.reshape(coverage.shape[0], 4, coverage.shape[1], 4).mean(
        axis=(1, 3),
        dtype=np.float32,
    )
    return downsampled, left, top


def _anchor_baseline_26_6(
    position: tuple[float, float],
    *,
    anchor: str,
    advance_26_6: int,
    ascender_26_6: int,
    descender_26_6: int,
) -> tuple[int, int]:
    vertical, horizontal = anchor.split("-")
    x_26_6 = round(position[0] * 64.0)
    y_26_6 = round(position[1] * 64.0)
    if horizontal == "center":
        x_26_6 -= round(advance_26_6 / 2.0)
    elif horizontal == "right":
        x_26_6 -= advance_26_6
    if vertical == "top":
        y_26_6 += ascender_26_6
    elif vertical == "center":
        y_26_6 += round((ascender_26_6 + descender_26_6) / 2.0)
    elif vertical == "bottom":
        y_26_6 += descender_26_6
    return x_26_6, y_26_6


def _place_max(destination: np.ndarray, coverage: np.ndarray, *, left: int, top: int) -> None:
    if coverage.size == 0:
        return
    height, width = destination.shape
    source_height, source_width = coverage.shape
    output_left = max(0, left)
    output_top = max(0, top)
    output_right = min(width, left + source_width)
    output_bottom = min(height, top + source_height)
    if output_right <= output_left or output_bottom <= output_top:
        return
    source_left = output_left - left
    source_top = output_top - top
    source_right = source_left + output_right - output_left
    source_bottom = source_top + output_bottom - output_top
    np.maximum(
        destination[output_top:output_bottom, output_left:output_right],
        coverage[source_top:source_bottom, source_left:source_right],
        out=destination[output_top:output_bottom, output_left:output_right],
    )


def _blend_reference(
    destination: np.ndarray,
    *,
    color: Sequence[float],
    coverage: np.ndarray,
    opacity: float,
    blend: str,
) -> np.ndarray:
    destination = destination.astype(np.float32, copy=False)
    alpha = coverage[..., np.newaxis] * np.float32(opacity)
    color_array = np.asarray(color, dtype=np.float32)
    if blend == "normal":
        blend_value = np.broadcast_to(color_array, destination.shape)
    elif blend == "add":
        blend_value = destination + color_array
    elif blend == "multiply":
        blend_value = destination * color_array
    elif blend == "screen":
        blend_value = np.float32(1.0) - (np.float32(1.0) - destination) * (np.float32(1.0) - color_array)
    else:
        raise AssertionError(blend)
    return destination * (np.float32(1.0) - alpha) + blend_value * alpha


def _reference_draw_text(
    source: np.ndarray,
    *,
    text: str,
    position: tuple[float, float],
    size: float,
    color: Sequence[float],
    weight: float = 400.0,
    language: str = "ja",
    anchor: str = "baseline-left",
    outlines: Sequence[tuple[Sequence[float], float]] | None = None,
    opacity: float = 1.0,
    blend: str = "normal",
    supersample: bool = False,
) -> np.ndarray:
    glyphs, advance_26_6 = _reference_shape(text, size=size, weight=weight, language=language)
    ascender_26_6, descender_26_6 = _font_metrics_26_6(size)
    baseline_x_26_6, baseline_y_26_6 = _anchor_baseline_26_6(
        position,
        anchor=anchor,
        advance_26_6=advance_26_6,
        ascender_26_6=ascender_26_6,
        descender_26_6=descender_26_6,
    )
    outline_values = tuple(outlines or ())
    cumulative_widths: list[int] = []
    total_width = 0.0
    for _outline_color, width in outline_values:
        total_width += width
        cumulative_widths.append(max(1, round(total_width * 64.0)))

    solid_layers = [np.zeros(source.shape[:2], dtype=np.float32) for _ in range(len(cumulative_widths) + 1)]
    for layer_index, stroke_radius_26_6 in enumerate((0, *cumulative_widths)):
        for glyph_id, glyph_x_26_6, glyph_y_down_26_6 in glyphs:
            pen_x_26_6 = baseline_x_26_6 + glyph_x_26_6
            pen_y_26_6 = baseline_y_26_6 + glyph_y_down_26_6
            pen_x_integer, phase_x_26_6 = divmod(pen_x_26_6, 64)
            pen_y_integer, phase_y_down_26_6 = divmod(pen_y_26_6, 64)
            coverage, bitmap_left, bitmap_top = _reference_bitmap(
                glyph_id,
                size=size,
                weight=weight,
                stroke_radius_26_6=stroke_radius_26_6,
                phase_x_26_6=phase_x_26_6,
                phase_y_down_26_6=phase_y_down_26_6,
                supersample=supersample,
            )
            _place_max(
                solid_layers[layer_index],
                coverage,
                left=pen_x_integer + bitmap_left,
                top=pen_y_integer - bitmap_top,
            )

    solid_unions = [solid_layers[0]]
    solid_unions.extend(np.maximum(solid_layers[0], stroke_band) for stroke_band in solid_layers[1:])
    result = source.astype(np.float32, copy=True)
    rings = [np.clip(solid_unions[index + 1] - solid_unions[index], 0.0, 1.0) for index in range(len(outline_values))]
    for index in reversed(range(len(outline_values))):
        result = _blend_reference(
            result,
            color=outline_values[index][0],
            coverage=rings[index],
            opacity=opacity,
            blend=blend,
        )
    return _blend_reference(result, color=color, coverage=solid_layers[0], opacity=opacity, blend=blend)


def _base_kwargs() -> dict[str, object]:
    return {
        "text": "A骨",
        "position": (24.5, 48.25),
        "size": 28.0,
        "color": (0.2, 0.4, 0.8),
    }


def test_draw_text_public_signature_frame_entry_empty_text_and_defaults() -> None:
    """v1-draw-text-unification acceptance 5; v1-draw-text-supersample acceptance 1: add one final bool."""
    import cupy as cp

    signature = inspect.signature(px.draw.text)
    assert tuple(signature.parameters) == (
        "frame",
        "text",
        "position",
        "size",
        "color",
        "weight",
        "language",
        "anchor",
        "outlines",
        "opacity",
        "blend",
        "align",
        "line_spacing",
        "tracking",
        "kerning",
        "font",
        "width",
        "supersample",
    )
    assert signature.parameters["frame"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in tuple(signature.parameters)[1:]:
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert {
        name: signature.parameters[name].default
        for name in (
            "weight",
            "language",
            "anchor",
            "outlines",
            "opacity",
            "blend",
            "align",
            "line_spacing",
            "tracking",
            "kerning",
            "font",
            "width",
            "supersample",
        )
    } == {
        "weight": 400.0,
        "language": "ja",
        "anchor": "baseline-left",
        "outlines": None,
        "opacity": 1.0,
        "blend": "normal",
        "align": "left",
        "line_spacing": 1.0,
        "tracking": 0.0,
        "kerning": True,
        "font": "sans",
        "width": None,
        "supersample": False,
    }
    assert signature.parameters["supersample"].annotation == "bool"
    assert "text" in px.draw.__all__

    source_values = np.arange(4 * 5, dtype=np.float32).reshape(4, 5, 1)
    source = _frame(source_values, channels=("matte",))
    result = px.draw.text(source, text="", position=(0.0, 0.0), size=12.0, color=(1.0,))
    assert isinstance(result, px.core.Frame)
    assert result.data.data.ptr != source.data.data.ptr
    np.testing.assert_array_equal(_host(result), source_values)

    with pytest.raises(ValueError) as error:
        px.draw.text(cp.zeros((4, 5, 1), dtype=cp.float32), **_base_kwargs())
    _assert_actionable(error)


@pytest.mark.parametrize(
    ("overrides", "listed_tokens"),
    (
        ({"text": 123}, ()),
        ({"text": "two\rlines"}, ()),
        ({"position": (1.0,)}, ()),
        ({"position": (float("nan"), 1.0)}, ()),
        ({"position": (1.0, float("inf"))}, ()),
        ({"size": 0.0}, ()),
        ({"size": -1.0}, ()),
        ({"size": True}, ()),
        ({"weight": 99.0}, ()),
        ({"weight": 901.0}, ()),
        ({"weight": float("nan")}, ()),
        ({"color": (1.0, 2.0)}, ()),
        ({"color": (1.0, float("nan"), 2.0)}, ()),
        ({"language": "en"}, LANGUAGES),
        ({"anchor": "left"}, ANCHORS),
        ({"outlines": (((1.0, 0.0), 1.0),)}, ()),
        ({"outlines": (((1.0, 0.0, 0.0), 0.0),)}, ()),
        ({"outlines": (((1.0, 0.0, 0.0), float("inf")),)}, ()),
        ({"outlines": ("bad",)}, ()),
        ({"opacity": -0.1}, ()),
        ({"opacity": 1.1}, ()),
        ({"opacity": True}, ()),
        ({"blend": "over"}, BLENDS),
    ),
)
def test_draw_text_entry_validation_is_actionable(
    overrides: dict[str, object],
    listed_tokens: tuple[str, ...],
) -> None:
    """v1-draw-text-unification acceptance 5-6: text type, carriage return, geometry, and tokens fail fast."""
    with pytest.raises(ValueError) as error:
        px.draw.text(_zeros(), **(_base_kwargs() | overrides))
    _assert_actionable(error)
    assert all(token in str(error.value) for token in listed_tokens)


@pytest.mark.parametrize("supersample", (0, 1, "true", None, (), np.bool_(True)))
def test_draw_text_supersample_requires_a_strict_python_bool(supersample: object) -> None:
    """v1-draw-text-supersample acceptance 2: only Python True and False pass the public boundary."""
    with pytest.raises(ValueError) as error:
        px.draw.text(_zeros(), **(_base_kwargs() | {"supersample": supersample}))
    _assert_actionable(error)
    assert "supersample" in str(error.value)


def test_draw_text_supersample_false_is_bit_identical_before_and_after_true_mode() -> None:
    """v1-draw-text-supersample acceptance 3: the default 8-bit path stays bit-identical and mode-local."""
    source_values = np.linspace(-0.4, 1.6, 96 * 192, dtype=np.float32).reshape(96, 192, 1)
    source = _frame(source_values, colorspace="S-Gamut3", gamma="s-log3", channels=("matte",))
    kwargs = {
        "text": "AV 骨",
        "position": (18.375, 62.625),
        "size": 31.25,
        "color": (1.4,),
        "weight": 525.0,
        "outlines": (((-0.25,), 1.5),),
        "opacity": 0.7,
        "blend": "screen",
    }
    expected = _reference_draw_text(source_values, **kwargs)

    default = px.draw.text(source, **kwargs)
    explicit_false = px.draw.text(source, **kwargs, supersample=False)
    px.draw.text(source, **kwargs, supersample=True)
    after_true = px.draw.text(source, **kwargs, supersample=False)

    np.testing.assert_allclose(_host(default), expected, rtol=4e-6, atol=4e-6)
    np.testing.assert_array_equal(_host(explicit_false), _host(default))
    np.testing.assert_array_equal(_host(after_true), _host(default))
    assert (after_true.colorspace, after_true.gamma, after_true.channels, after_true.matrix) == (
        source.colorspace,
        source.gamma,
        source.channels,
        source.matrix,
    )


@pytest.mark.parametrize("phase", ((0, 0), (8, 24), (31, 47), (55, 13)))
def test_draw_text_supersample_glyph_matches_independent_4x_box_oracle(phase: tuple[int, int]) -> None:
    """v1-draw-text-supersample acceptance 4 and 6: 4x fp32 averaging preserves the 1x glyph grid."""
    import pixtreme._draw.text as draw_text_module

    size = 37.25
    size_26_6 = max(1, round(size * 64.0))
    glyphs, _advance = _reference_shape("骨", size=size, weight=575.0, language="ja")
    glyph_id = glyphs[0][0]
    expected, expected_left, expected_top = _reference_bitmap(
        glyph_id,
        size=size,
        weight=575.0,
        stroke_radius_26_6=0,
        phase_x_26_6=phase[0],
        phase_y_down_26_6=phase[1],
        supersample=True,
    )
    normal = draw_text_module._glyph_bitmap(
        glyph_id,
        size_26_6,
        575.0,
        0,
        phase[0],
        phase[1],
        "sans",
        supersample=False,
    )
    actual = draw_text_module._glyph_bitmap(
        glyph_id,
        size_26_6,
        575.0,
        0,
        phase[0],
        phase[1],
        "sans",
        supersample=True,
    )

    assert (
        (actual.left, actual.top, actual.coverage.shape)
        == (
            normal.left,
            normal.top,
            normal.coverage.shape,
        )
        == (expected_left, expected_top, expected.shape)
    )
    np.testing.assert_array_equal(actual.coverage, expected)


def test_draw_text_supersample_empty_whitespace_keeps_zero_sized_glyph_and_atlas_at_phase_boundary() -> None:
    """v1-draw-text-supersample acceptance 6 and 8: empty whitespace keeps mode-local zero-sized storage."""
    import freetype

    import pixtreme._draw.text as draw_text_module

    glyph_id = freetype.Face(str(FONT_PATH)).get_char_index(ord(" "))
    glyph_args = (glyph_id, 12 * 64, 400.0, 0, 63, 63, "sans")
    normal_glyph = draw_text_module._glyph_bitmap(*glyph_args, supersample=False)
    sampled_glyph = draw_text_module._glyph_bitmap(*glyph_args, supersample=True)

    assert (
        (
            sampled_glyph.left,
            sampled_glyph.top,
            sampled_glyph.coverage.shape,
            sampled_glyph.coverage.nbytes,
        )
        == (
            normal_glyph.left,
            normal_glyph.top,
            normal_glyph.coverage.shape,
            normal_glyph.coverage.nbytes,
        )
        == (0, 0, (0, 0), 0)
    )

    atlas_args = (
        " ",
        12 * 64,
        400.0,
        "ja",
        True,
        0.0,
        1.0,
        "left",
        "sans",
        None,
        63,
        63,
        (),
        -512,
        -512,
        512,
        512,
    )
    normal_atlas = draw_text_module._build_block_atlas(*atlas_args, supersample=False)
    sampled_atlas = draw_text_module._build_block_atlas(*atlas_args, supersample=True)

    assert (
        (
            sampled_atlas.left,
            sampled_atlas.top,
            sampled_atlas.body.shape,
            sampled_atlas.body.nbytes,
        )
        == (
            normal_atlas.left,
            normal_atlas.top,
            normal_atlas.body.shape,
            normal_atlas.body.nbytes,
        )
        == (0, 0, (0, 0), 0)
    )


def test_draw_text_supersample_fixed_fhd_matte_matches_oracle_and_exceeds_256_coverages() -> None:
    """v1-draw-text-supersample acceptance 5: the fixed matte matches the independent oracle and exceeds 256 values."""
    source_values = np.zeros((1080, 1920, 1), dtype=np.float32)
    source = _frame(source_values, channels=("matte",))
    kwargs = {
        "text": "pixtreme 文字描画",
        "position": (960.0, 540.0),
        "size": 64.0,
        "color": (1.0,),
        "anchor": "center-center",
        "outlines": (((1.0,), 2.0),),
        "opacity": 1.0,
        "blend": "normal",
        "supersample": True,
    }
    expected = _reference_draw_text(source_values, **kwargs)
    actual = _host(px.draw.text(source, **kwargs))

    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    assert np.unique(actual).size > 256


@pytest.mark.parametrize("anchor", ANCHORS)
def test_draw_text_supersample_keeps_all_anchor_solutions_mode_independent(anchor: str) -> None:
    """v1-draw-text-supersample acceptance 6: supersampling does not change block metrics or anchor placement."""
    text = "Ag骨"
    size = 31.25
    _glyphs, advance_26_6 = _reference_shape(text, size=size, weight=400.0, language="ja")
    ascender_26_6, descender_26_6 = _font_metrics_26_6(size)
    baseline_x_26_6 = round(92.25 * 64.0)
    baseline_y_26_6 = round(72.5 * 64.0)
    vertical, horizontal = anchor.split("-")
    position_x_26_6 = baseline_x_26_6
    if horizontal == "center":
        position_x_26_6 += round(advance_26_6 / 2.0)
    elif horizontal == "right":
        position_x_26_6 += advance_26_6
    position_y_26_6 = baseline_y_26_6
    if vertical == "top":
        position_y_26_6 -= ascender_26_6
    elif vertical == "center":
        position_y_26_6 -= round((ascender_26_6 + descender_26_6) / 2.0)
    elif vertical == "bottom":
        position_y_26_6 -= descender_26_6

    source = _zeros(height=140, width=300, channels=("matte",))
    baseline = px.draw.text(
        source,
        text=text,
        position=(baseline_x_26_6 / 64.0, baseline_y_26_6 / 64.0),
        size=size,
        color=(1.0,),
        supersample=True,
    )
    anchored = px.draw.text(
        source,
        text=text,
        position=(position_x_26_6 / 64.0, position_y_26_6 / 64.0),
        size=size,
        color=(1.0,),
        anchor=anchor,
        supersample=True,
    )
    np.testing.assert_array_equal(_host(anchored), _host(baseline))


def test_draw_text_supersample_multi_outline_matches_independent_ring_oracle() -> None:
    """v1-draw-text-supersample acceptance 7: body and cumulative outline rings share the fixed 4x rule."""
    source_values = np.full((144, 300, 3), 0.2, dtype=np.float32)
    source = _frame(source_values)
    kwargs = {
        "text": "VA骨",
        "position": (34.375, 92.625),
        "size": 46.0,
        "color": (0.1, 0.3, 1.6),
        "weight": 650.0,
        "outlines": (((1.5, -0.2, 0.4), 1.25), ((-0.5, 1.8, 0.2), 2.5)),
        "opacity": 0.8,
        "blend": "screen",
        "supersample": True,
    }
    expected = _reference_draw_text(source_values, **kwargs)
    actual = px.draw.text(source, **kwargs)
    np.testing.assert_allclose(_host(actual), expected, rtol=4e-6, atol=4e-6)


def test_draw_text_supersample_caches_are_mode_local_and_store_only_downsampled_fp32() -> None:
    """v1-draw-text-supersample acceptance 8: glyph and atlas caches split modes without retaining 4x storage."""
    import pixtreme._draw.text as draw_text_module

    size = 30.0
    size_26_6 = round(size * 64.0)
    glyphs, _advance = _reference_shape("骨", size=size, weight=550.0, language="ja")
    glyph_id = glyphs[0][0]
    draw_text_module._glyph_bitmap.cache_clear()
    glyph_args = (glyph_id, size_26_6, 550.0, 0, 19, 37, "sans")
    normal_glyph = draw_text_module._glyph_bitmap(*glyph_args, supersample=False)
    after_normal_glyph = draw_text_module._glyph_bitmap.cache_info()
    sampled_glyph = draw_text_module._glyph_bitmap(*glyph_args, supersample=True)
    after_sampled_glyph = draw_text_module._glyph_bitmap.cache_info()
    assert after_sampled_glyph.misses == after_normal_glyph.misses + 1
    assert draw_text_module._glyph_bitmap(*glyph_args, supersample=False) is normal_glyph
    assert draw_text_module._glyph_bitmap(*glyph_args, supersample=True) is sampled_glyph
    assert sampled_glyph.coverage.shape == normal_glyph.coverage.shape
    assert sampled_glyph.coverage.nbytes == normal_glyph.coverage.nbytes
    assert sampled_glyph.coverage.dtype == normal_glyph.coverage.dtype == np.float32

    draw_text_module._build_block_atlas.cache_clear()
    atlas_args = (
        "cache 骨",
        size_26_6,
        550.0,
        "ja",
        True,
        0.0,
        1.0,
        "left",
        "sans",
        None,
        19,
        37,
        (96, 224),
        -512,
        -512,
        512,
        512,
    )
    normal_atlas = draw_text_module._build_block_atlas(*atlas_args, supersample=False)
    after_normal_atlas = draw_text_module._build_block_atlas.cache_info()
    sampled_atlas = draw_text_module._build_block_atlas(*atlas_args, supersample=True)
    after_sampled_atlas = draw_text_module._build_block_atlas.cache_info()
    assert after_sampled_atlas.misses == after_normal_atlas.misses + 1
    assert draw_text_module._build_block_atlas(*atlas_args, supersample=False) is normal_atlas
    assert draw_text_module._build_block_atlas(*atlas_args, supersample=True) is sampled_atlas
    assert (sampled_atlas.left, sampled_atlas.top) == (normal_atlas.left, normal_atlas.top)
    for sampled, normal in zip(
        (sampled_atlas.body, *sampled_atlas.rings),
        (normal_atlas.body, *normal_atlas.rings),
        strict=True,
    ):
        assert sampled.shape == normal.shape
        assert sampled.nbytes == normal.nbytes
        assert sampled.dtype == normal.dtype == np.float32


def test_draw_text_supersample_preserves_scene_metadata_input_and_private_storage() -> None:
    """v1-draw-text-supersample acceptance 9: True keeps scene values, channels, metadata, and storage contracts."""
    labels = ("normal.x", "depth", "id", "custom")
    source_values = np.full((96, 192, len(labels)), -0.25, dtype=np.float32)
    source = _frame(source_values, colorspace="S-Gamut3", gamma="s-log3", channels=labels)
    result = px.draw.text(
        source,
        text="scene 骨",
        position=(18.375, 62.625),
        size=34.0,
        color=(-1.0, 0.5, 2.0, 4.0),
        supersample=True,
    )

    np.testing.assert_array_equal(_host(source), source_values)
    assert result.data.data.ptr != source.data.data.ptr
    assert result.data.flags.c_contiguous
    assert (result.colorspace, result.gamma, result.channels, result.matrix) == (
        source.colorspace,
        source.gamma,
        source.channels,
        source.matrix,
    )
    assert np.min(_host(result)) < 0.0
    assert np.max(_host(result)) > 1.0


@pytest.mark.parametrize("blend", BLENDS)
def test_draw_text_matches_independent_harfbuzz_freetype_and_numpy_oracle(blend: str) -> None:
    """v1-draw-text-unification acceptance 11-12: inherited single-line shaping and compositing match a host oracle."""
    source_values = np.linspace(-0.4, 1.7, 112 * 240 * 3, dtype=np.float32).reshape(112, 240, 3)
    source = _frame(source_values)
    kwargs = {
        "text": "AV ffi 骨",
        "position": (18.25, 70.375),
        "size": 27.5,
        "color": (-0.75, 1.4, 0.3),
        "weight": 525.0,
        "language": "zh-hans",
        "opacity": 0.65,
        "blend": blend,
    }
    glyphs, _advance = _reference_shape(
        kwargs["text"],
        size=kwargs["size"],
        weight=kwargs["weight"],
        language=kwargs["language"],
    )
    assert len(glyphs) < len(kwargs["text"])
    expected = _reference_draw_text(source_values, **kwargs)
    result = px.draw.text(source, **kwargs)
    assert result.data.dtype.name == "float32"
    np.testing.assert_allclose(_host(result), expected, rtol=3e-6, atol=3e-6)
    assert np.min(_host(result)) < 0.0
    assert np.max(_host(result)) > 1.0


@pytest.mark.parametrize("anchor", ANCHORS)
def test_draw_text_all_anchors_represent_the_same_font_metric_and_advance_point(anchor: str) -> None:
    """v1-draw-text acceptance 12-14: all 12 anchors use font ascender/descender and shaped line advance, not ink."""
    text = "Ag骨"
    size = 31.25
    weight = 400.0
    language = "ja"
    _glyphs, advance_26_6 = _reference_shape(text, size=size, weight=weight, language=language)
    ascender_26_6, descender_26_6 = _font_metrics_26_6(size)
    baseline_x_26_6 = round(52.25 * 64.0)
    baseline_y_26_6 = round(64.5 * 64.0)
    vertical, horizontal = anchor.split("-")
    position_x_26_6 = baseline_x_26_6
    if horizontal == "center":
        position_x_26_6 += round(advance_26_6 / 2.0)
    elif horizontal == "right":
        position_x_26_6 += advance_26_6
    position_y_26_6 = baseline_y_26_6
    if vertical == "top":
        position_y_26_6 -= ascender_26_6
    elif vertical == "center":
        position_y_26_6 -= round((ascender_26_6 + descender_26_6) / 2.0)
    elif vertical == "bottom":
        position_y_26_6 -= descender_26_6

    source = _zeros(height=120, width=260, channels=("matte",))
    baseline = px.draw.text(
        source,
        text=text,
        position=(baseline_x_26_6 / 64.0, baseline_y_26_6 / 64.0),
        size=size,
        color=(1.0,),
    )
    anchored = px.draw.text(
        source,
        text=text,
        position=(position_x_26_6 / 64.0, position_y_26_6 / 64.0),
        size=size,
        color=(1.0,),
        anchor=anchor,
    )
    np.testing.assert_array_equal(_host(anchored), _host(baseline))


def test_draw_text_subpixel_clipping_outside_and_missing_glyph_behavior() -> None:
    """v1-draw-text acceptance 4 and 16-18: subpixel placement is continuous, clipping is safe, and .notdef draws."""
    source_values = np.arange(72 * 96, dtype=np.float32).reshape(72, 96, 1) / 1000.0
    source = _frame(source_values, channels=("signal",))
    integer = px.draw.text(source, text="A", position=(12.0, 40.0), size=30.0, color=(1.0,))
    subpixel = px.draw.text(source, text="A", position=(12.25, 40.375), size=30.0, color=(1.0,))
    assert not np.array_equal(_host(integer), _host(subpixel))

    clipped = px.draw.text(source, text="A骨", position=(-8.25, 18.5), size=32.0, color=(1.0,))
    assert np.any(_host(clipped) != source_values)
    outside = px.draw.text(source, text="A骨", position=(-500.0, -500.0), size=32.0, color=(1.0,))
    np.testing.assert_array_equal(_host(outside), source_values)
    assert outside.data.data.ptr != source.data.data.ptr

    tofu = px.draw.text(_zeros(channels=("matte",)), text="\U0010ffff", position=(12.0, 48.0), size=32.0, color=(1.0,))
    assert np.any(_host(tofu) > 0.0)


@pytest.mark.parametrize("position", ((1e308, 40.0), (-1e308, 40.0), (12.0, 1e308), (12.0, -1e308)))
def test_draw_text_large_finite_positions_remain_valid_outside_coordinates(position: tuple[float, float]) -> None:
    """v1-draw-text acceptance 4 and 17: large finite positions remain valid and draw only image intersections."""
    source_values = np.linspace(-0.25, 1.25, 72 * 96, dtype=np.float32).reshape(72, 96, 1)
    source = _frame(source_values, channels=("matte",))
    actual = px.draw.text(source, text="A", position=position, size=30.0, color=(1.0,))
    np.testing.assert_array_equal(_host(actual), source_values)
    assert actual.data.data.ptr != source.data.data.ptr


def test_draw_text_language_selects_cjk_locl_forms_and_weight_instances() -> None:
    """v1-draw-text acceptance 6, 8, 15, and 19: language selects locl glyphs and wght changes rasterized coverage."""
    source = _zeros(height=96, width=160, channels=("matte",))
    japanese = px.draw.text(
        source,
        text="骨",
        position=(20.0, 62.0),
        size=48.0,
        color=(1.0,),
        language="ja",
    )
    simplified = px.draw.text(
        source,
        text="骨",
        position=(20.0, 62.0),
        size=48.0,
        color=(1.0,),
        language="zh-hans",
    )
    assert not np.array_equal(_host(japanese), _host(simplified))

    thin = px.draw.text(
        source,
        text="A骨",
        position=(20.0, 62.0),
        size=48.0,
        color=(1.0,),
        weight=100.0,
    )
    heavy = px.draw.text(
        source,
        text="A骨",
        position=(20.0, 62.0),
        size=48.0,
        color=(1.0,),
        weight=900.0,
    )
    assert np.sum(_host(heavy), dtype=np.float64) > np.sum(_host(thin), dtype=np.float64)


def test_draw_text_multiple_outlines_match_cumulative_external_ring_oracle() -> None:
    """v1-draw-text acceptance 21-22: glyphs merge by max and cumulative outlines composite outer-to-inner then body."""
    source_values = np.full((128, 240, 3), 0.15, dtype=np.float32)
    source = _frame(source_values)
    outlines = (((1.5, -0.2, 0.4), 1.25), ((-0.5, 1.8, 0.2), 2.5))
    kwargs = {
        "text": "VA骨",
        "position": (30.375, 78.25),
        "size": 42.0,
        "color": (0.1, 0.3, 1.6),
        "weight": 650.0,
        "outlines": outlines,
        "opacity": 0.8,
        "blend": "screen",
    }
    expected = _reference_draw_text(source_values, **kwargs)
    result = px.draw.text(source, **kwargs)
    np.testing.assert_allclose(_host(result), expected, rtol=4e-6, atol=4e-6)

    reversed_order = px.draw.text(source, **(kwargs | {"outlines": tuple(reversed(outlines))}))
    assert not np.array_equal(_host(result), _host(reversed_order))


def test_draw_text_outer_outline_adds_nothing_inside_the_body_fill() -> None:
    """v1-draw-text acceptance 22: each cumulative outline is only the external ring beyond the inner solid union."""
    source = _zeros(height=128, width=128, channels=("matte",))
    placement = {
        "text": "O",
        "position": (24.0, 88.0),
        "size": 64.0,
    }
    body_coverage = _host(px.draw.text(source, **placement, color=(1.0,), blend="add"))[..., 0]
    outer_ring_contribution = _host(
        px.draw.text(
            source,
            **placement,
            color=(0.0,),
            outlines=(((0.0,), 2.0), ((1.0,), 6.0)),
            blend="add",
        )
    )[..., 0]

    fully_covered_body = body_coverage == np.float32(1.0)
    assert np.count_nonzero(fully_covered_body) > 0
    assert np.count_nonzero(outer_ring_contribution) > 0
    np.testing.assert_array_equal(outer_ring_contribution[fully_covered_body], np.float32(0.0))


def test_draw_text_preserves_metadata_channels_and_private_fp32_output() -> None:
    """v1-draw-text acceptance 23-26: metadata and arbitrary labels survive, and one-channel matte uses numeric coverage."""
    labels = ("normal.x", "depth", "id", "custom")
    source_values = np.full((80, 160, len(labels)), 0.25, dtype=np.float32)
    source = _frame(source_values, colorspace="S-Gamut3", gamma="s-log3", channels=labels)
    result = px.draw.text(
        source,
        text="mask",
        position=(12.5, 48.25),
        size=30.0,
        color=(-1.0, 0.5, 2.0, 4.0),
    )
    assert result.data.data.ptr != source.data.data.ptr
    assert (result.width, result.height, result.colorspace, result.gamma, result.channels) == (
        source.width,
        source.height,
        source.colorspace,
        source.gamma,
        source.channels,
    )
    assert np.any(_host(result) != source_values)

    matte = px.draw.text(
        _zeros(height=64, width=128, channels=("matte",)),
        text="A",
        position=(12.25, 42.375),
        size=32.0,
        color=(1.0,),
    )
    matte_values = _host(matte)
    assert np.any((matte_values > 0.0) & (matte_values < 1.0))


def test_draw_text_bundles_font_license_and_keeps_text_dependencies_lazy() -> None:
    """v1-draw-text acceptance 27-28: the fixed Noto font and OFL ship while FreeType/HarfBuzz stay lazy."""
    assert FONT_PATH.is_file()
    assert FONT_PATH.stat().st_size > 1_000_000
    license_text = LICENSE_PATH.read_text(encoding="utf-8")
    assert "SIL OPEN FONT LICENSE" in license_text
    assert "Version 1.1" in license_text

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import pixtreme; "
                "assert 'freetype' not in sys.modules; "
                "assert 'uharfbuzz' not in sys.modules"
            ),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr

    import pixtreme._draw.text as draw_text_module

    source = inspect.getsource(draw_text_module)
    assert "NotoSansCJKjp-VF.otf" in source
    assert "system font" not in source.lower()


def test_draw_text_caches_shaping_glyph_rasters_and_atlases_with_bit_identity() -> None:
    """v1-draw-text-unification acceptance 5 and 10: private caches preserve repeated output bit identity."""
    import pixtreme._draw.text as draw_text_module

    for name in ("_shape_text", "_glyph_bitmap", "_build_block_atlas"):
        cached = getattr(draw_text_module, name)
        assert hasattr(cached, "cache_info")
        cached.cache_clear()

    source = _zeros(height=96, width=180, channels=("matte",))
    kwargs = {
        "text": "cache 骨",
        "position": (18.25, 58.375),
        "size": 30.0,
        "color": (1.0,),
        "weight": 550.0,
        "outlines": (((0.25,), 1.5),),
    }
    first = px.draw.text(source, **kwargs)
    first_stats = {
        name: getattr(draw_text_module, name).cache_info()
        for name in ("_shape_text", "_glyph_bitmap", "_build_block_atlas")
    }
    second = px.draw.text(source, **kwargs)
    second_stats = {
        name: getattr(draw_text_module, name).cache_info()
        for name in ("_shape_text", "_glyph_bitmap", "_build_block_atlas")
    }
    np.testing.assert_array_equal(_host(first), _host(second))
    assert second_stats["_build_block_atlas"].hits > first_stats["_build_block_atlas"].hits
    assert not hasattr(px.draw, "text_cache")


def test_draw_text_gpu_composite_is_a_freetype_free_rawkernel_boundary() -> None:
    """v1-draw-text acceptance 30: the GPU compositor consumes coverage atlases without touching FreeType or HarfBuzz."""
    import pixtreme._draw.text as draw_text_module

    compositor_source = inspect.getsource(draw_text_module._composite_layer)
    kernel_factory_source = inspect.getsource(draw_text_module._text_composite_kernel)
    combined = (compositor_source + kernel_factory_source).lower()
    assert "freetype" not in combined
    assert "harfbuzz" not in combined
    assert "cp.RawKernel" in kernel_factory_source
    assert "elementwisekernel" not in kernel_factory_source


def _table_tokens(markdown: str, heading: str) -> tuple[str, ...]:
    section = markdown.split(f"## {heading}\n", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
    return tuple(
        cells[1].strip().removeprefix("`").removesuffix("`")
        for line in section.splitlines()
        if line.startswith("| `")
        for cells in (line.split("|"),)
    )


def test_draw_text_vocabulary_documents_language_anchor_and_placement_contracts(
    vocabulary_markdown: str,
) -> None:
    """v1-draw-text acceptance 31: vocabulary fixes language/anchor tokens, defaults, metrics, advance, and pen placement."""
    from pixtreme._draw.text import _ANCHOR_TOKENS, _LANGUAGE_TOKENS

    markdown = vocabulary_markdown
    assert _table_tokens(markdown, "language") == LANGUAGES == _LANGUAGE_TOKENS
    assert _table_tokens(markdown, "anchor") == ANCHORS == _ANCHOR_TOKENS
    for required in (
        "locl",
        "default",
        "`ja`",
        "`baseline-left`",
        "ascender",
        "descender",
        "advance",
        "pen",
        "single-line",
        "newlines",
        "subpixel",
    ):
        assert required in markdown


def test_draw_text_docstring_states_the_llm_readable_contract() -> None:
    """v1-draw-text-unification acceptance 14: the docstring states the complete integrated text contract."""
    docstring = (inspect.getdoc(px.draw.text) or "").lower()
    for required in (
        "anchor",
        "position",
        r"\n",
        r"\r",
        "line_spacing",
        "tracking",
        "kerning",
        "width",
        "align",
        "justify",
        ".notdef",
        "bundled",
        "cache",
        "clamp",
        "scene",
        "new storage",
    ):
        assert required in docstring


def test_draw_text_supersample_vocabulary_and_docstring_state_the_opt_in_contract(
    vocabulary_markdown: str,
) -> None:
    """v1-draw-text-supersample acceptance 10: vocabulary and docstring explain the bool-only 4x precision path."""
    aa_section = vocabulary_markdown.split("## aa\n", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
    for required in (
        "`px.draw.text(supersample=True)`",
        "bool",
        "token",
        "4×4",
        "averages",
    ):
        assert required in aa_section

    docstring = inspect.getdoc(px.draw.text) or ""
    for required in (
        "supersample=False",
        "supersample=True",
        "4x",
        "fp32",
        "box",
        "geometry",
        "private cache",
        "opt-in",
    ):
        assert required in docstring
