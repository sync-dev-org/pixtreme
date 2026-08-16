"""Specification, contract, and numerical-property tests for integrated text layout."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import pixtreme as px

ROOT = Path(__file__).resolve().parents[1]
FONT_PATHS = {
    "sans": ROOT / "src" / "pixtreme" / "data" / "fonts" / "NotoSansCJKjp-VF.otf",
    "mono": ROOT / "src" / "pixtreme" / "data" / "fonts" / "NotoSansMonoCJKjp-VF.otf",
}
LICENSE_PATH = ROOT / "src" / "pixtreme" / "data" / "fonts" / "LICENSE-NotoSansCJK"
LANGUAGES = ("ja", "zh-hans", "zh-hant", "ko")
ANCHORS = tuple(
    f"{vertical}-{horizontal}"
    for vertical in ("top", "center", "baseline", "bottom")
    for horizontal in ("left", "center", "right")
)
ALIGNS = ("left", "center", "right", "justify")
FONTS = ("sans", "mono")
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
    height: int = 128,
    width: int = 256,
    channels: Sequence[str] = ("R", "G", "B"),
) -> px.core.Frame:
    return _frame(np.zeros((height, width, len(channels)), dtype=np.float32), channels=channels)


def _host(result: px.core.Frame) -> np.ndarray:
    return px.io.to_array(
        result,
    ).get()


def _assert_actionable(error: pytest.ExceptionInfo[ValueError], tokens: tuple[str, ...] = ()) -> None:
    message = str(error.value)
    assert "why=" in message
    assert "; what=" in message
    assert "; how=" in message
    assert all(token in message for token in tokens)


def _reference_shape(
    text: str,
    *,
    size_26_6: int,
    weight: float,
    language: str,
    kerning: bool,
    font: str,
) -> tuple[tuple[tuple[int, int, int], ...], int]:
    if text == "":
        return (), 0

    import uharfbuzz as hb

    face = hb.Face(FONT_PATHS[font].read_bytes())
    shaped_font = hb.Font(face)
    shaped_font.scale = (size_26_6, size_26_6)
    hb.ot_font_set_funcs(shaped_font)
    shaped_font.set_variations({"wght": weight})
    buffer = hb.Buffer()
    buffer.add_str(text)
    buffer.guess_segment_properties()
    buffer.direction = "ltr"
    buffer.language = language
    hb.shape(shaped_font, buffer, {"kern": kerning, "liga": True, "locl": True})

    pen_x = 0
    pen_y = 0
    glyphs: list[tuple[int, int, int]] = []
    for info, position in zip(buffer.glyph_infos, buffer.glyph_positions, strict=True):
        glyphs.append(
            (
                int(info.codepoint),
                pen_x + int(position.x_offset),
                -(pen_y + int(position.y_offset)),
            )
        )
        pen_x += int(position.x_advance)
        pen_y += int(position.y_advance)
    return tuple(glyphs), pen_x


def _reference_metrics(size_26_6: int, *, weight: float, font: str) -> tuple[int, int, int]:
    import freetype

    face = freetype.Face(str(FONT_PATHS[font]))
    face.set_char_size(0, size_26_6, 72, 72)
    face.set_var_design_coords((weight,))
    units_per_em = int(face.units_per_EM)
    return (
        round(int(face.ascender) * size_26_6 / units_per_em),
        round(int(face.descender) * size_26_6 / units_per_em),
        int(face.height),
    )


def _reference_bitmap(
    glyph_id: int,
    *,
    size_26_6: int,
    weight: float,
    font: str,
    stroke_radius_26_6: int,
    phase_x_26_6: int,
    phase_y_down_26_6: int,
) -> tuple[np.ndarray, int, int]:
    import freetype

    face = freetype.Face(str(FONT_PATHS[font]))
    face.set_char_size(0, size_26_6, 72, 72)
    face.set_var_design_coords((weight,))
    face.load_glyph(glyph_id, freetype.FT_LOAD_DEFAULT | freetype.FT_LOAD_NO_BITMAP)
    glyph = face.glyph.get_glyph()
    if stroke_radius_26_6 > 0:
        stroker = freetype.Stroker()
        stroker.set(
            stroke_radius_26_6,
            freetype.FT_STROKER_LINECAP_ROUND,
            freetype.FT_STROKER_LINEJOIN_ROUND,
            0,
        )
        glyph.stroke(stroker, destroy=False)
    rendered = glyph.to_bitmap(
        freetype.FT_RENDER_MODE_NORMAL,
        freetype.Vector(phase_x_26_6, -phase_y_down_26_6),
        destroy=False,
    )
    bitmap = rendered.bitmap
    if bitmap.rows == 0 or bitmap.width == 0:
        coverage = np.zeros((0, 0), dtype=np.float32)
    else:
        pitch = abs(int(bitmap.pitch))
        coverage = np.asarray(bitmap.buffer, dtype=np.uint8).reshape(int(bitmap.rows), pitch)[:, : int(bitmap.width)]
        coverage = coverage.astype(np.float32) / np.float32(255.0)
    return coverage, int(rendered.left), int(rendered.top)


def _place_max(destination: np.ndarray, coverage: np.ndarray, *, left: int, top: int) -> None:
    if coverage.size == 0:
        return
    output_left = max(0, left)
    output_top = max(0, top)
    output_right = min(destination.shape[1], left + coverage.shape[1])
    output_bottom = min(destination.shape[0], top + coverage.shape[0])
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
    alpha = coverage[..., np.newaxis] * np.float32(opacity)
    source_color = np.asarray(color, dtype=np.float32)
    if blend == "normal":
        blend_value = np.broadcast_to(source_color, destination.shape)
    elif blend == "add":
        blend_value = destination + source_color
    elif blend == "multiply":
        blend_value = destination * source_color
    elif blend == "screen":
        blend_value = np.float32(1.0) - (np.float32(1.0) - destination) * (np.float32(1.0) - source_color)
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
    align: str = "left",
    line_spacing: float = 1.0,
    tracking: float = 0.0,
    kerning: bool = True,
    font: str = "sans",
    width: float | None = None,
) -> np.ndarray:
    size_26_6 = max(1, round(size * 64.0))
    lines = text.split("\n")
    shaped_lines = [
        _reference_shape(
            line,
            size_26_6=size_26_6,
            weight=weight,
            language=language,
            kerning=kerning,
            font=font,
        )
        for line in lines
    ]
    tracking_26_6 = round(tracking * size_26_6)
    line_advances = [advance + tracking_26_6 * max(0, len(glyphs) - 1) for glyphs, advance in shaped_lines]
    block_width_26_6 = round(width * 64.0) if width is not None else max((0, *line_advances))

    ascender_26_6, descender_26_6, height_units = _reference_metrics(
        size_26_6,
        weight=weight,
        font=font,
    )
    import freetype

    metric_face = freetype.Face(str(FONT_PATHS[font]))
    line_step_26_6 = round(height_units * size_26_6 * line_spacing / int(metric_face.units_per_EM))
    vertical, horizontal = anchor.split("-")
    block_left_26_6 = round(position[0] * 64.0)
    if horizontal == "center":
        block_left_26_6 -= round(block_width_26_6 / 2.0)
    elif horizontal == "right":
        block_left_26_6 -= block_width_26_6
    first_baseline_26_6 = round(position[1] * 64.0)
    final_baseline_offset = (len(lines) - 1) * line_step_26_6
    if vertical == "top":
        first_baseline_26_6 += ascender_26_6
    elif vertical == "center":
        first_baseline_26_6 += round((ascender_26_6 + descender_26_6 - final_baseline_offset) / 2.0)
    elif vertical == "bottom":
        first_baseline_26_6 += descender_26_6 - final_baseline_offset

    outline_values = tuple(outlines or ())
    cumulative_widths: list[int] = []
    cumulative_width = 0.0
    for _outline_color, outline_width in outline_values:
        cumulative_width += outline_width
        cumulative_widths.append(max(1, round(cumulative_width * 64.0)))

    solid_layers = [np.zeros(source.shape[:2], dtype=np.float32) for _ in range(len(cumulative_widths) + 1)]
    for layer_index, stroke_radius_26_6 in enumerate((0, *cumulative_widths)):
        for line_index, ((glyphs, _shaped_advance), line_advance_26_6) in enumerate(
            zip(shaped_lines, line_advances, strict=True)
        ):
            if align == "center":
                line_offset_26_6 = round((block_width_26_6 - line_advance_26_6) / 2.0)
            elif align == "right":
                line_offset_26_6 = block_width_26_6 - line_advance_26_6
            else:
                line_offset_26_6 = 0
            justify_remainder = max(block_width_26_6 - line_advance_26_6, 0) if align == "justify" else 0
            gap_count = max(0, len(glyphs) - 1)
            for glyph_index, (glyph_id, glyph_x_26_6, glyph_y_down_26_6) in enumerate(glyphs):
                justify_offset = round(justify_remainder * glyph_index / gap_count) if gap_count else 0
                pen_x_26_6 = (
                    block_left_26_6 + line_offset_26_6 + glyph_x_26_6 + tracking_26_6 * glyph_index + justify_offset
                )
                pen_y_26_6 = first_baseline_26_6 + line_index * line_step_26_6 + glyph_y_down_26_6
                pen_x_integer, phase_x_26_6 = divmod(pen_x_26_6, 64)
                pen_y_integer, phase_y_down_26_6 = divmod(pen_y_26_6, 64)
                coverage, bitmap_left, bitmap_top = _reference_bitmap(
                    glyph_id,
                    size_26_6=size_26_6,
                    weight=weight,
                    font=font,
                    stroke_radius_26_6=stroke_radius_26_6,
                    phase_x_26_6=phase_x_26_6,
                    phase_y_down_26_6=phase_y_down_26_6,
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
        "position": (24.5, 64.25),
        "size": 28.0,
        "color": (0.2, 0.4, 0.8),
    }


def _table_tokens(markdown: str, heading: str) -> tuple[str, ...]:
    section = markdown.split(f"## {heading}\n", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
    return tuple(
        cells[1].strip().removeprefix("`").removesuffix("`")
        for line in section.splitlines()
        if line.startswith("| `")
        for cells in (line.split("|"),)
    )


def test_draw_text_unification_public_signature_is_the_complete_layout_contract() -> None:
    """v1-draw-text-unification acceptance 1; v1-draw-text-supersample acceptance 1;
    v1-draw-text-user-font acceptance 5: complete signature.
    """
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
        "variations",
        "width",
        "supersample",
    )
    assert signature.parameters["frame"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert all(
        signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY for name in tuple(signature.parameters)[1:]
    )
    assert {
        name: signature.parameters[name].default
        for name in ("align", "line_spacing", "tracking", "kerning", "font", "variations", "width", "supersample")
    } == {
        "align": "left",
        "line_spacing": 1.0,
        "tracking": 0.0,
        "kerning": True,
        "font": "sans",
        "variations": None,
        "width": None,
        "supersample": False,
    }
    assert signature.parameters["supersample"].annotation == "bool"


def test_draw_text_unification_removes_draw_text_block_from_the_public_surface() -> None:
    """v1-draw-text-unification acceptance 2; v1-derivative-filters acceptance 17;
    v1-draw-text-user-font acceptance 1:
    the public surface exports text once and exposes no draw_text_block name.
    """
    assert not hasattr(px.draw, "draw_text_block")
    assert "draw_text_block" not in px.draw.__all__
    assert px.draw.__all__.count("text") == 1
    assert tuple(name for name in px.draw.__all__ if inspect.isfunction(getattr(px.draw, name))) == (
        "line",
        "polyline",
        "rectangle",
        "circle",
        "ellipse",
        "polygon",
        "text",
    )

    performance_tree = ast.parse((ROOT / "tests" / "test_performance_spec.py").read_text(encoding="utf-8"))

    def assigned_strings(name: str) -> set[str]:
        assignment = next(
            node
            for node in performance_tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
        )
        return {
            literal.value
            for literal in ast.walk(assignment.value)
            if isinstance(literal, ast.Constant) and isinstance(literal.value, str)
        }

    public_gpu_names = assigned_strings("_PUBLIC_GPU_PIXEL_FUNCTIONS")
    draw_registry_names = assigned_strings("_DRAW_CASES")
    assert "text" in public_gpu_names and "text" in draw_registry_names
    assert "draw_text_block" not in public_gpu_names | draw_registry_names
    assert "draw-text-block-cjk-outline" not in draw_registry_names


def test_draw_text_unification_removes_the_compatibility_module() -> None:
    """v1-draw-text-unification acceptance 3: no alias, wrapper, or alternate draw_text_block import remains."""
    assert importlib.util.find_spec("pixtreme._draw_text_block") is None


def test_draw_text_unification_uses_the_integrated_layout_path_for_default_single_lines() -> None:
    """v1-draw-text-unification acceptance 4: default single lines use the integrated atlas path directly."""
    source = inspect.getsource(px.draw.text)
    assert "_build_block_atlas(" in source
    assert "draw_text_block" not in source
    assert "return text(" not in source


def test_draw_text_empty_and_zero_opacity_preserve_metadata_in_new_storage() -> None:
    """v1-draw-text-unification acceptance 5-6: no-op paths preserve pixels and metadata but always allocate."""
    source_values = np.arange(7 * 9, dtype=np.float32).reshape(7, 9, 1) / np.float32(11.0)
    source = _frame(source_values, colorspace="Rec.2020", gamma="pq", channels=("matte",))
    for text, opacity in (("", 1.0), ("\n\n", 1.0), ("A\nB", 0.0)):
        result = px.draw.text(
            source,
            text=text,
            position=(2.0, 3.0),
            size=12.0,
            color=(1.0,),
            opacity=opacity,
        )
        np.testing.assert_array_equal(_host(result), source_values)
        assert result.data.data.ptr != source.data.data.ptr
        assert (result.width, result.height, result.colorspace, result.gamma, result.channels) == (
            source.width,
            source.height,
            source.colorspace,
            source.gamma,
            source.channels,
        )


@pytest.mark.parametrize("anchor", ANCHORS)
@pytest.mark.parametrize("language", LANGUAGES)
@pytest.mark.parametrize("blend", BLENDS)
def test_draw_text_default_single_line_matches_independent_oracle_for_every_anchor_and_language(
    anchor: str,
    language: str,
    blend: str,
) -> None:
    """v1-draw-text-unification acceptance 11-12: inherited single-line cases match an independent host oracle."""
    source_values = np.linspace(-0.3, 1.4, 112 * 240 * 3, dtype=np.float32).reshape(112, 240, 3)
    source = _frame(source_values)
    kwargs = {
        "text": "A骨",
        "position": (120.25, 56.375),
        "size": 24.5,
        "color": (-0.5, 1.3, 0.4),
        "weight": 525.0,
        "language": language,
        "anchor": anchor,
        "outlines": (((0.1, 0.2, 1.2), 1.25),),
        "opacity": 0.7,
        "blend": blend,
    }
    expected = _reference_draw_text(source_values, **kwargs)
    actual = px.draw.text(source, **kwargs)
    np.testing.assert_allclose(_host(actual), expected, rtol=4e-6, atol=4e-6)
    assert actual.model_dump(exclude={"data"}) == source.model_dump(exclude={"data"})


@pytest.mark.parametrize(
    "kwargs,channels",
    (
        ({"text": "", "position": (0.0, 0.0), "size": 10.0, "color": (1.0,)}, ("matte",)),
        ({"text": "\U0010ffff", "position": (9.25, 34.5), "size": 27.0, "color": (1.0,)}, ("matte",)),
        ({"text": "outside", "position": (-900.0, -700.0), "size": 30.0, "color": (1.0,)}, ("matte",)),
    ),
)
def test_draw_text_default_single_line_covers_empty_notdef_matte_and_outside_paths(
    kwargs: dict[str, object],
    channels: tuple[str, ...],
) -> None:
    """v1-draw-text-unification acceptance 5 and 11-12: inherited edge paths match an independent host oracle."""
    source = _zeros(height=72, width=96, channels=channels)
    expected = _reference_draw_text(np.zeros((72, 96, len(channels)), dtype=np.float32), **kwargs)
    actual = px.draw.text(source, **kwargs)
    np.testing.assert_array_equal(_host(actual), expected)


@pytest.mark.parametrize(
    "overrides,tokens",
    (
        ({"text": "A\rB"}, ()),
        ({"align": "Left"}, ALIGNS),
        ({"line_spacing": 0.0}, ()),
        ({"line_spacing": -1.0}, ()),
        ({"line_spacing": True}, ()),
        ({"line_spacing": float("nan")}, ()),
        ({"tracking": True}, ()),
        ({"tracking": float("inf")}, ()),
        ({"kerning": 1}, ()),
        ({"font": "serif"}, FONTS),
        ({"width": -0.1}, ()),
        ({"width": True}, ()),
        ({"width": float("nan")}, ()),
        ({"font": "mono", "weight": 399.0}, ("400", "700")),
        ({"font": "mono", "weight": 701.0}, ("400", "700")),
    ),
)
def test_draw_text_extension_validation_is_actionable(
    overrides: dict[str, object],
    tokens: tuple[str, ...],
) -> None:
    """v1-draw-text-unification acceptance 5-9: integrated layout domains fail fast."""
    with pytest.raises(ValueError) as error:
        px.draw.text(_zeros(), **(_base_kwargs() | overrides))
    _assert_actionable(error, tokens)


@pytest.mark.parametrize(
    "overrides",
    (
        {"position": (1.0,)},
        {"size": 0.0},
        {"weight": 99.0},
        {"color": (1.0, 2.0)},
        {"language": "en"},
        {"anchor": "left"},
        {"outlines": (((1.0, 0.0, 0.0), 0.0),)},
        {"opacity": 1.1},
        {"blend": "over"},
    ),
)
def test_draw_text_inherited_validation_keeps_three_element_errors(overrides: dict[str, object]) -> None:
    """v1-draw-text-unification acceptance 5: inherited validation retains the actionable error contract."""
    with pytest.raises(ValueError) as error:
        px.draw.text(_zeros(), **(_base_kwargs() | overrides))
    _assert_actionable(error)


@pytest.mark.parametrize("align", ALIGNS)
def test_draw_text_matches_independent_multiline_layout_and_raster_oracle(align: str) -> None:
    """v1-draw-text-unification acceptance 6-10 and 12: multiline layout and rasterization match a host oracle."""
    source_values = np.linspace(-0.25, 1.35, 150 * 300 * 3, dtype=np.float32).reshape(150, 300, 3)
    source = _frame(source_values)
    kwargs = {
        "text": "AVA\n\nffi骨\n",
        "position": (151.25, 74.375),
        "size": 22.75,
        "color": (-0.4, 1.3, 0.25),
        "weight": 575.0,
        "language": "zh-hans",
        "anchor": "center-center",
        "outlines": (((1.2, 0.1, -0.2), 1.0), ((0.0, 0.5, 1.4), 1.5)),
        "opacity": 0.65,
        "blend": "screen",
        "align": align,
        "line_spacing": 0.72,
        "tracking": -0.08,
        "kerning": False,
        "width": 185.5,
    }
    expected = _reference_draw_text(source_values, **kwargs)
    actual = px.draw.text(source, **kwargs)
    # The host oracle and CUDA kernel evaluate the fp32 blend expression in different operation orders.
    # A 4e-6 bound covers those few-ULP differences while remaining far below one 8-bit coverage step (1 / 255).
    np.testing.assert_allclose(_host(actual), expected, rtol=4e-6, atol=4e-6)
    assert np.min(_host(actual)) < 0.0
    assert np.max(_host(actual)) > 1.0


@pytest.mark.parametrize("anchor", ANCHORS)
def test_draw_text_all_anchors_use_block_width_first_ascender_and_last_descender(anchor: str) -> None:
    """v1-draw-text-unification acceptance 6-10: all anchors use the block box, including a trailing empty line."""
    source_values = np.zeros((150, 300, 1), dtype=np.float32)
    kwargs = {
        "text": "A\nBB\n",
        "position": (145.25, 70.375),
        "size": 24.0,
        "color": (1.0,),
        "anchor": anchor,
        "align": "right",
        "line_spacing": 1.15,
        "tracking": 0.05,
        "width": 120.25,
    }
    expected = _reference_draw_text(source_values, **kwargs)
    actual = px.draw.text(_frame(source_values, channels=("matte",)), **kwargs)
    np.testing.assert_array_equal(_host(actual), expected)


def test_draw_text_width_overflow_negative_tracking_and_justify_do_not_clip_or_shrink() -> None:
    """v1-draw-text-unification acceptance 7-10: signed advances overflow fixed width while justify adds only space."""
    source_values = np.zeros((96, 260, 1), dtype=np.float32)
    source = _frame(source_values, channels=("matte",))
    for kwargs in (
        {"align": "right", "width": 0.0, "tracking": 0.2},
        {"align": "center", "width": 12.0, "tracking": 0.0},
        {"align": "justify", "width": 150.0, "tracking": -0.55},
    ):
        arguments = {
            "text": "ABCD",
            "position": (130.0, 55.0),
            "size": 28.0,
            "color": (1.0,),
        } | kwargs
        expected = _reference_draw_text(source_values, **arguments)
        actual = px.draw.text(source, **arguments)
        np.testing.assert_array_equal(_host(actual), expected)


def test_draw_text_single_glyph_justify_keeps_the_block_left_origin() -> None:
    """v1-draw-text-unification acceptance 8 and 11: a one-glyph line with positive remainder has no justify gap."""
    source = _zeros(height=72, width=160, channels=("matte",))
    common = {
        "text": "A",
        "position": (18.25, 46.375),
        "size": 30.0,
        "color": (1.0,),
    }
    expected = px.draw.text(source, **common)
    actual = px.draw.text(source, **common, align="justify", width=120.0)
    np.testing.assert_array_equal(_host(actual), _host(expected))


@pytest.mark.parametrize(
    ("layout", "text", "visible_text", "visible_anchor"),
    (
        ({"align": "left", "width": 1e308}, "AB", "AB", "baseline-left"),
        ({"align": "center", "width": 1e308, "anchor": "baseline-center"}, "AB", "AB", "baseline-center"),
        ({"align": "left", "tracking": 1e308}, "AB", "A", "baseline-left"),
        ({"align": "left", "tracking": -1e308}, "AB", "A", "baseline-left"),
        ({"align": "left", "tracking": 1e308, "anchor": "baseline-right"}, "AB", "B", "baseline-right"),
        ({"align": "left", "line_spacing": 1e308}, "A\nB", "A", "baseline-left"),
        ({"align": "left", "line_spacing": 1e308, "anchor": "bottom-left"}, "A\nB", "B", "bottom-left"),
        ({"align": "justify", "width": 1e308}, "AB", "A", "baseline-left"),
        ({"align": "justify", "width": 1e308, "anchor": "baseline-right"}, "AB", "B", "baseline-right"),
    ),
)
def test_draw_text_large_finite_layout_values_only_draw_image_intersections(
    layout: dict[str, object],
    text: str,
    visible_text: str,
    visible_anchor: str,
) -> None:
    """v1-draw-text-unification acceptance 7-10: large finite layout values only draw image intersections."""
    source = _zeros(height=72, width=96, channels=("matte",))
    common = {
        "position": (12.25, 46.375),
        "size": 30.0,
        "color": (1.0,),
    }
    expected = px.draw.text(source, text=visible_text, **common, anchor=visible_anchor)
    actual = px.draw.text(source, text=text, **common, **layout)
    np.testing.assert_array_equal(_host(actual), _host(expected))


def test_draw_text_kerning_switch_preserves_ligature_and_locl_shaping() -> None:
    """v1-draw-text-unification acceptance 7 and 10: disabling kern keeps liga and locl while tracking is post-shaping."""
    size_26_6 = round(40.0 * 64.0)
    ligature_glyphs, _advance = _reference_shape(
        "ffi",
        size_26_6=size_26_6,
        weight=400.0,
        language="ja",
        kerning=False,
        font="sans",
    )
    japanese, _ = _reference_shape(
        "骨",
        size_26_6=size_26_6,
        weight=400.0,
        language="ja",
        kerning=False,
        font="sans",
    )
    simplified, _ = _reference_shape(
        "骨",
        size_26_6=size_26_6,
        weight=400.0,
        language="zh-hans",
        kerning=False,
        font="sans",
    )
    assert len(ligature_glyphs) < 3
    assert japanese != simplified

    source = _zeros(height=80, width=220, channels=("matte",))
    common = {"text": "AVA ffi 骨", "position": (12.0, 54.0), "size": 34.0, "color": (1.0,)}
    kerned = px.draw.text(source, **common, kerning=True)
    unkerned = px.draw.text(source, **common, kerning=False)
    tracked = px.draw.text(source, **common, kerning=False, tracking=0.1)
    assert not np.array_equal(_host(kerned), _host(unkerned))
    assert not np.array_equal(_host(unkerned), _host(tracked))


def test_draw_text_mono_font_assets_weight_range_and_numerical_oracle() -> None:
    """v1-draw-text-unification acceptance 5 and 9-10: mono uses its measured axis and inherited raster rules."""
    import freetype

    mono_path = FONT_PATHS["mono"]
    assert mono_path.is_file()
    assert mono_path.stat().st_size > 1_000_000
    axis = freetype.Face(str(mono_path)).get_variation_info().axes[0]
    assert (axis.tag, axis.minimum, axis.default, axis.maximum) == ("wght", 400.0, 400.0, 700.0)
    license_text = LICENSE_PATH.read_text(encoding="utf-8")
    assert "SIL OPEN FONT LICENSE" in license_text
    assert "Version 1.1" in license_text

    source_values = np.zeros((112, 280, 1), dtype=np.float32)
    kwargs = {
        "text": "mono\n骨AV",
        "position": (140.25, 54.375),
        "size": 30.0,
        "color": (1.0,),
        "weight": 700.0,
        "anchor": "center-center",
        "font": "mono",
        "align": "center",
        "tracking": 0.12,
        "line_spacing": 1.1,
        "width": 180.0,
    }
    expected = _reference_draw_text(source_values, **kwargs)
    actual = px.draw.text(_frame(source_values, channels=("matte",)), **kwargs)
    np.testing.assert_array_equal(_host(actual), expected)


def test_draw_text_font_dependencies_and_reads_remain_lazy() -> None:
    """v1-draw-text-unification acceptance 9: package fonts are fixed while FreeType and HarfBuzz stay lazy."""
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

    source = inspect.getsource(draw_text_module).lower()
    assert "system font" not in source
    assert "http://" not in source
    assert "https://" not in source


def test_draw_text_caches_full_layout_with_bit_identity() -> None:
    """v1-draw-text-unification acceptance 5 and 10: repeated layout and raster results are privately cached."""
    import pixtreme._draw.text as draw_text_module

    for cached in (draw_text_module._shape_text, draw_text_module._glyph_bitmap, draw_text_module._build_block_atlas):
        assert hasattr(cached, "cache_info")
        cached.cache_clear()

    source = _zeros(height=112, width=260, channels=("matte",))
    kwargs = {
        "text": "cache\n骨 text",
        "position": (130.25, 55.375),
        "size": 26.0,
        "color": (1.0,),
        "align": "justify",
        "width": 170.0,
        "tracking": 0.03,
        "outlines": (((0.25,), 1.5),),
    }
    first = px.draw.text(source, **kwargs)
    first_stats = draw_text_module._build_block_atlas.cache_info()
    second = px.draw.text(source, **kwargs)
    second_stats = draw_text_module._build_block_atlas.cache_info()
    np.testing.assert_array_equal(_host(first), _host(second))
    assert second_stats.hits > first_stats.hits
    assert not hasattr(px.draw, "text_block_cache")


def test_draw_text_vocabulary_and_docstring_fix_the_full_layout_contract(vocabulary_markdown: str) -> None:
    """v1-draw-text-unification acceptance 13-14: docs fix tokens, units, overflow, anchors, and ownership."""
    from pixtreme._draw.text import _ALIGN_TOKENS, _FONT_TOKENS

    assert _table_tokens(vocabulary_markdown, "text align") == ALIGNS == _ALIGN_TOKENS
    assert _table_tokens(vocabulary_markdown, "text font") == FONTS == _FONT_TOKENS
    assert "draw_text_block" not in vocabulary_markdown
    layout = vocabulary_markdown.split("## text block layout\n", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
    for required in (
        "line_spacing",
        "tracking",
        "em",
        "pixel",
        "kerning",
        "Justification",
        "anchor",
        "width",
        "overflow",
        "400",
        "700",
    ):
        assert required in layout

    docstring = inspect.getdoc(px.draw.text) or ""
    for required in (
        r"\n",
        r"\r",
        "line_spacing",
        "tracking",
        "kerning",
        "font",
        "width",
        "align",
        "justify",
        "anchor",
        ".notdef",
        "cache",
        "scene",
        "new storage",
        "400.0 through 700.0",
    ):
        assert required.lower() in docstring.lower()


def test_draw_text_backreferences_and_gpu_kernel_reuse_are_structural_contracts() -> None:
    """REQ-TEST-001 / v1-draw-text-unification acceptance 12 and 16: tests backreference and reuse the kernel."""
    import pixtreme._draw.text as draw_text_module

    test_source = Path(__file__).read_text(encoding="utf-8")
    assert "v1-draw-text-unification acceptance" in test_source
    assert draw_text_module._text_composite_kernel() is draw_text_module._text_composite_kernel()
    public_source = inspect.getsource(draw_text_module.text)
    assert "_composite_layer" in public_source
    assert "RawKernel" not in public_source
