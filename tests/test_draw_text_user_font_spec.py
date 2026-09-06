"""Specification tests for user-provided draw-text fonts."""

from __future__ import annotations

import importlib.util
import inspect
import io
import math
import shutil
import subprocess
import sys
from pathlib import Path

import cupy as cp
import numpy as np
import pytest
from repository_contracts import require_repo_file

import pixtreme as px

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "data" / "draw_text_user_font"
STATIC_FONT = FIXTURE_ROOT / "noto-user-static.otf"
VARIABLE_FONT = FIXTURE_ROOT / "noto-user-variable.otf"
COLLECTION_FONT = FIXTURE_ROOT / "noto-user-collection.ttc"


def _frame(*, height: int = 72, width: int = 160) -> px.core.Frame:
    return px.core.Frame(
        data=cp.zeros((height, width, 1), dtype=cp.float32),
        colorspace="Rec.709",
        gamma="linear",
        channels=("matte",),
        matrix=None,
    )


def _host(frame: px.core.Frame) -> np.ndarray:
    return cp.asnumpy(frame.data)


def _assert_actionable(error: pytest.ExceptionInfo[BaseException]) -> str:
    message = str(error.value)
    assert message.startswith("why=")
    assert "; what=" in message
    assert "; how=" in message
    return message


def _oracle_shape(
    path: Path,
    *,
    face_index: int,
    text: str,
    size_26_6: int,
    coordinates: dict[str, float],
    language: str,
    kerning: bool,
) -> tuple[tuple[tuple[int, int, int], ...], int]:
    import uharfbuzz as hb

    face = hb.Face(path.read_bytes(), face_index)
    font = hb.Font(face)
    font.scale = (size_26_6, size_26_6)
    hb.ot_font_set_funcs(font)
    font.set_variations(coordinates)
    buffer = hb.Buffer()
    buffer.add_str(text)
    buffer.guess_segment_properties()
    buffer.direction = "ltr"
    buffer.language = language
    hb.shape(font, buffer, {"kern": kerning, "liga": True, "locl": True})
    pen_x = 0
    pen_y = 0
    glyphs = []
    for info, position in zip(buffer.glyph_infos, buffer.glyph_positions, strict=True):
        glyphs.append((int(info.codepoint), pen_x + int(position.x_offset), -(pen_y + int(position.y_offset))))
        pen_x += int(position.x_advance)
        pen_y += int(position.y_advance)
    return tuple(glyphs), pen_x


def _oracle_bitmap(
    path: Path,
    *,
    face_index: int,
    glyph_id: int,
    size_26_6: int,
    coordinates: tuple[float, ...],
) -> tuple[np.ndarray, int, int]:
    import freetype

    face = freetype.Face(io.BytesIO(path.read_bytes()), index=face_index)
    face.set_char_size(0, size_26_6, 72, 72)
    face.set_var_design_coords(coordinates)
    face.load_glyph(glyph_id, freetype.FT_LOAD_DEFAULT | freetype.FT_LOAD_NO_BITMAP)
    glyph = face.glyph.get_glyph()
    rendered = glyph.to_bitmap(
        freetype.FT_RENDER_MODE_NORMAL,
        freetype.Vector(0, 0),
        destroy=False,
    )
    bitmap = rendered.bitmap
    if bitmap.rows == 0 or bitmap.width == 0:
        coverage = np.zeros((0, 0), dtype=np.float32)
    else:
        pitch = abs(int(bitmap.pitch))
        buffer = np.ctypeslib.as_array(
            bitmap._FT_Bitmap.buffer,
            shape=(int(bitmap.rows) * pitch,),
        )
        coverage = buffer.reshape(int(bitmap.rows), pitch)[:, : int(bitmap.width)].astype(np.float32)
        coverage /= np.float32(255.0)
    return coverage, int(rendered.left), int(rendered.top)


def test_draw_font_public_surface_signature_and_immutability(tmp_path: Path) -> None:
    """v1-draw-text-user-font acceptance 1-5: Font has one draw-owned constructor and text has the exact extension."""
    assert px.draw.__all__ == ("line", "polyline", "rectangle", "circle", "ellipse", "polygon", "text", "Font")
    assert inspect.isclass(px.draw.Font)
    for namespace in (px, px.core, px.io):
        assert not hasattr(namespace, "Font")
    assert not hasattr(px.io, "read_font")
    assert importlib.util.find_spec("pixtreme.font") is None

    constructor = inspect.signature(px.draw.Font.from_file)
    assert tuple(constructor.parameters) == ("path", "face_index")
    assert constructor.parameters["path"].annotation == "str | os.PathLike[str]"
    assert constructor.parameters["face_index"].kind is inspect.Parameter.KEYWORD_ONLY
    assert constructor.parameters["face_index"].default == 0
    assert constructor.return_annotation == "Font"

    signature = inspect.signature(px.draw.text)
    assert tuple(signature.parameters)[-4:] == ("font", "variations", "width", "supersample")
    assert signature.parameters["font"].annotation == "TextFont | Font"
    assert signature.parameters["variations"].annotation == "Mapping[str, float] | None"
    assert signature.parameters["variations"].default is None
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in tuple(signature.parameters.values())[1:]
    )

    alias_path = tmp_path / "alias.otf"
    shutil.copyfile(STATIC_FONT, alias_path)
    first = px.draw.Font.from_file(STATIC_FONT)
    second = px.draw.Font.from_file(alias_path)
    face_one = px.draw.Font.from_file(COLLECTION_FONT, face_index=1)
    assert first == second and hash(first) == hash(second)
    assert first != face_one
    assert {first: "shared"}[second] == "shared"
    with pytest.raises((AttributeError, TypeError)):
        first._data = b"changed"  # type: ignore[attr-defined,misc]
    with pytest.raises(TypeError):
        px.draw.Font()
    assert not {"axes", "variations", "fallback", "path"} & {name for name in dir(first) if not name.startswith("_")}


def test_draw_font_loader_accepts_content_not_extension_and_measures_axes(tmp_path: Path) -> None:
    """v1-draw-text-user-font acceptance 6 and 10: both backends accept the selected face and measure valid axes."""
    opaque_path = tmp_path / "font.payload"
    shutil.copyfile(VARIABLE_FONT, opaque_path)
    font = px.draw.Font.from_file(opaque_path)
    result = px.draw.text(
        _frame(),
        text="AV",
        position=(8.0, 48.0),
        size=28.0,
        color=(1.0,),
        weight=500.0,
        font=font,
        variations={"wdth": 75.0},
    )
    assert np.any(_host(result) != 0.0)


@pytest.mark.parametrize("kind", ("object", "bytes-path", "missing", "directory"))
def test_draw_font_loader_rejects_unusable_paths_actionably(tmp_path: Path, kind: str) -> None:
    """v1-draw-text-user-font acceptance 7: unusable path states fail as actionable ValueError."""

    class BytesPath:
        def __fspath__(self) -> bytes:
            return b"font.otf"

    value: object
    if kind == "object":
        value = object()
    elif kind == "bytes-path":
        value = BytesPath()
    elif kind == "missing":
        value = tmp_path / "missing.otf"
    else:
        value = tmp_path
    with pytest.raises(ValueError) as error:
        px.draw.Font.from_file(value)  # type: ignore[arg-type]
    message = _assert_actionable(error)
    assert "font file" in message


def test_draw_font_loader_distinguishes_freetype_and_harfbuzz_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1-draw-text-user-font acceptance 8: FreeType and HarfBuzz failures are eager, distinct, and chained."""
    invalid = tmp_path / "invalid.otf"
    invalid.write_bytes(b"not a font")
    with pytest.raises(ValueError) as freetype_error:
        px.draw.Font.from_file(invalid)
    assert "FreeType" in _assert_actionable(freetype_error)
    assert freetype_error.value.__cause__ is not None

    import pixtreme._draw.text as draw_text_module

    def fail_harfbuzz(_data: bytes, _face_index: int) -> object:
        raise RuntimeError("injected HarfBuzz failure")

    monkeypatch.setattr(draw_text_module, "_open_harfbuzz_face", fail_harfbuzz)
    with pytest.raises(ValueError) as harfbuzz_error:
        px.draw.Font.from_file(STATIC_FONT)
    assert "HarfBuzz" in _assert_actionable(harfbuzz_error)
    assert isinstance(harfbuzz_error.value.__cause__, RuntimeError)


@pytest.mark.parametrize("face_index", (True, 1.0, -1, 2))
def test_draw_font_face_index_is_strict_and_uses_measured_collection_count(face_index: object) -> None:
    """v1-draw-text-user-font acceptance 9: face index is a non-bool in-range int checked against the TTC."""
    with pytest.raises(ValueError) as error:
        px.draw.Font.from_file(COLLECTION_FONT, face_index=face_index)  # type: ignore[arg-type]
    message = _assert_actionable(error)
    assert repr(face_index) in message
    assert "face_count=2" in message
    assert "0 <= face_index < face_count" in message


def test_draw_font_axis_measurement_failure_is_eager_and_chained(monkeypatch: pytest.MonkeyPatch) -> None:
    """v1-draw-text-user-font acceptance 10: malformed or unreadable axis tables fail during construction."""
    import pixtreme._draw.text as draw_text_module

    def fail_measurement(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected invalid axis table")

    monkeypatch.setattr(draw_text_module, "_measure_font_axes", fail_measurement)
    with pytest.raises(ValueError) as error:
        px.draw.Font.from_file(VARIABLE_FONT)
    message = _assert_actionable(error)
    assert str(VARIABLE_FONT) in message and "face_index=0" in message
    assert isinstance(error.value.__cause__, RuntimeError)


def test_draw_font_axis_measurement_distinguishes_static_absence_from_variable_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1-draw-text-user-font acceptance 10: only a genuinely static face may have no measured axes."""
    import pixtreme._draw.text as draw_text_module

    assert px.draw.Font.from_file(STATIC_FONT)._axes == ()

    class FailingVariableFace:
        num_faces = 1
        has_multiple_masters = True

        def get_variation_info(self) -> object:
            raise RuntimeError("injected FreeType variation measurement failure")

    class EmptyHarfBuzzFace:
        axis_infos: tuple[object, ...] = ()

    monkeypatch.setattr(draw_text_module, "_open_freetype_face", lambda *_args: FailingVariableFace())
    monkeypatch.setattr(draw_text_module, "_open_harfbuzz_face", lambda *_args: EmptyHarfBuzzFace())
    with pytest.raises(ValueError) as error:
        px.draw.Font.from_file(VARIABLE_FONT)
    message = _assert_actionable(error)
    assert str(VARIABLE_FONT) in message and "face_index=0" in message
    assert isinstance(error.value.__cause__, RuntimeError)


def test_draw_font_snapshots_bytes_across_rewrite_and_delete(tmp_path: Path) -> None:
    """v1-draw-text-user-font acceptance 4 and 11: identity and rendering use construction-time bytes only."""
    path = tmp_path / "mutable.otf"
    shutil.copyfile(STATIC_FONT, path)
    original = px.draw.Font.from_file(path)
    kwargs = {
        "text": "AV",
        "position": (8.0, 48.0),
        "size": 28.0,
        "color": (1.0,),
        "font": original,
    }
    before = px.draw.text(_frame(), **kwargs)
    path.unlink()
    import pixtreme._draw.text as draw_text_module

    for name in (
        "_shape_text",
        "_freetype_face",
        "_font_metrics_26_6",
        "_font_line_advance_26_6",
        "_glyph_bitmap",
        "_block_layout",
        "_build_block_atlas",
    ):
        getattr(draw_text_module, name).cache_clear()
    after_delete = px.draw.text(_frame(), **kwargs)
    np.testing.assert_array_equal(_host(after_delete), _host(before))

    shutil.copyfile(VARIABLE_FONT, path)
    rewritten = px.draw.Font.from_file(path)
    assert rewritten != original
    after_rewrite = px.draw.text(_frame(), **{**kwargs, "font": original})
    np.testing.assert_array_equal(_host(after_rewrite), _host(before))


def test_draw_text_token_calls_with_absent_variations_are_bit_identical() -> None:
    """v1-draw-text-user-font acceptance 12: omitted, None, and empty variations preserve both token paths bitwise."""
    source = _frame()
    for font, weight in (("sans", 575.0), ("mono", 550.0)):
        kwargs = {
            "text": "AV\nfi",
            "position": (18.25, 48.5),
            "size": 24.0,
            "color": (1.0,),
            "weight": weight,
            "font": font,
            "supersample": True,
        }
        omitted = px.draw.text(source, **kwargs)
        explicit_none = px.draw.text(source, **kwargs, variations=None)
        empty = px.draw.text(source, **kwargs, variations={})
        np.testing.assert_array_equal(_host(explicit_none), _host(omitted))
        np.testing.assert_array_equal(_host(empty), _host(omitted))


@pytest.mark.parametrize(
    ("font_path", "weight", "variations", "required"),
    (
        (VARIABLE_FONT, 99.0, None, ("99.0", "100.0", "900.0")),
        (VARIABLE_FONT, math.inf, None, ("inf", "100.0", "900.0")),
        (STATIC_FONT, 401.0, None, ("static font", "401.0", "400.0")),
        (VARIABLE_FONT, 400.0, [], ("mapping",)),
        (VARIABLE_FONT, 400.0, {1: 100.0}, ("str",)),
        (VARIABLE_FONT, 400.0, {"wght": 500.0}, ("wght", "weight=")),
        (VARIABLE_FONT, 400.0, {"xxxx": 1.0}, ("xxxx", "wdth", "wght")),
        (VARIABLE_FONT, 400.0, {"wdth": True}, ("bool", "wdth")),
        (VARIABLE_FONT, 400.0, {"wdth": math.nan}, ("nan", "wdth")),
        (VARIABLE_FONT, 400.0, {"wdth": 126.0}, ("126.0", "75.0", "125.0")),
        (STATIC_FONT, 400.0, {"wdth": 100.0}, ("wdth", "()")),
        (Path("token:sans"), 400.0, {"wdth": 100.0}, ("wdth", "wght")),
    ),
)
def test_draw_text_user_weight_and_variation_failures_are_actionable(
    font_path: Path,
    weight: float,
    variations: object,
    required: tuple[str, ...],
) -> None:
    """v1-draw-text-user-font acceptance 13-17: weight and variation validation is finite, ranged, and unambiguous."""
    font: object = "sans" if str(font_path) == "token:sans" else px.draw.Font.from_file(font_path)
    with pytest.raises(ValueError) as error:
        px.draw.text(
            _frame(),
            text="AV",
            position=(8.0, 48.0),
            size=28.0,
            color=(1.0,),
            weight=weight,
            font=font,  # type: ignore[arg-type]
            variations=variations,  # type: ignore[arg-type]
        )
    message = _assert_actionable(error)
    assert all(part in message for part in required)


def test_draw_text_user_font_matches_independent_shaping_and_raster_oracles() -> None:
    """v1-draw-text-user-font acceptance 15 and 18: resolved axes reach HarfBuzz and FreeType identically."""
    import pixtreme._draw.text as draw_text_module

    font = px.draw.Font.from_file(VARIABLE_FONT)
    size_26_6 = 31 * 64
    expected_shapes = []
    expected_bitmaps = []
    for width in (75.0, 125.0):
        axis_coordinates = (("wght", 650.0), ("wdth", width))
        expected_glyphs, expected_advance = _oracle_shape(
            VARIABLE_FONT,
            face_index=0,
            text="AVfi",
            size_26_6=size_26_6,
            coordinates=dict(axis_coordinates),
            language="ja",
            kerning=True,
        )
        expected_shapes.append((expected_glyphs, expected_advance))
        actual_shape = draw_text_module._shape_text(
            "AVfi",
            size_26_6,
            650.0,
            "ja",
            font,
            True,
            axis_coordinates,
        )
        assert (
            tuple((glyph.glyph_id, glyph.x_26_6, glyph.y_down_26_6) for glyph in actual_shape.glyphs) == expected_glyphs
        )
        assert actual_shape.advance_26_6 == expected_advance

        glyph_id = expected_glyphs[0][0]
        expected_coverage, expected_left, expected_top = _oracle_bitmap(
            VARIABLE_FONT,
            face_index=0,
            glyph_id=glyph_id,
            size_26_6=size_26_6,
            coordinates=(650.0, width),
        )
        expected_bitmaps.append((expected_coverage, expected_left, expected_top))
        actual_bitmap = draw_text_module._glyph_bitmap(
            glyph_id,
            size_26_6,
            650.0,
            0,
            0,
            0,
            font,
            False,
            axis_coordinates,
        )
        assert (actual_bitmap.left, actual_bitmap.top) == (expected_left, expected_top)
        np.testing.assert_array_equal(actual_bitmap.coverage, expected_coverage)

    assert expected_shapes[0] != expected_shapes[1]
    assert expected_bitmaps[0][1:] != expected_bitmaps[1][1:] or not np.array_equal(
        expected_bitmaps[0][0], expected_bitmaps[1][0]
    )

    result = px.draw.text(
        _frame(height=120, width=220),
        text="AV\nfi",
        position=(110.25, 54.5),
        size=31.0,
        color=(1.5,),
        weight=650.0,
        language="ko",
        anchor="center-center",
        outlines=(((0.25,), 1.25),),
        align="justify",
        line_spacing=1.3,
        tracking=0.04,
        kerning=False,
        font=font,
        variations={"wdth": 75.0},
        width=130.0,
        supersample=True,
    )
    assert np.any(_host(result) != 0.0)


def test_draw_text_user_font_cache_identity_and_limits_are_content_based(tmp_path: Path) -> None:
    """v1-draw-text-user-font acceptance 19-20: all bounded caches share equal content identities and split axes."""
    import pixtreme._draw.text as draw_text_module

    alias = tmp_path / "same-content.otf"
    shutil.copyfile(VARIABLE_FONT, alias)
    first = px.draw.Font.from_file(VARIABLE_FONT)
    second = px.draw.Font.from_file(alias)
    coordinates = (("wght", 500.0), ("wdth", 100.0))
    expected_limits = {
        "_font_bytes": 2,
        "_shape_text": 512,
        "_freetype_face": 128,
        "_glyph_bitmap": 4096,
        "_build_block_atlas": 128,
    }
    for name, limit in expected_limits.items():
        cached = getattr(draw_text_module, name)
        assert cached.cache_info().maxsize == limit
        cached.cache_clear()

    first_shape = draw_text_module._shape_text("AV", 24 * 64, 500.0, "ja", first, True, coordinates)
    assert draw_text_module._shape_text("AV", 24 * 64, 500.0, "ja", second, True, coordinates) is first_shape
    first_face = draw_text_module._freetype_face(24 * 64, 500.0, first, coordinates)
    assert draw_text_module._freetype_face(24 * 64, 500.0, second, coordinates) is first_face
    glyph_id = first_shape.glyphs[0].glyph_id
    first_bitmap = draw_text_module._glyph_bitmap(glyph_id, 24 * 64, 500.0, 0, 0, 0, first, False, coordinates)
    assert draw_text_module._glyph_bitmap(glyph_id, 24 * 64, 500.0, 0, 0, 0, second, False, coordinates) is first_bitmap

    kwargs = {
        "text": "AV",
        "position": (12.0, 48.0),
        "size": 24.0,
        "color": (1.0,),
        "weight": 500.0,
        "variations": {"wdth": 100.0},
    }
    output_a = px.draw.text(_frame(), font=first, **kwargs)
    atlas_after_first = draw_text_module._build_block_atlas.cache_info()
    output_b = px.draw.text(_frame(), font=second, **kwargs)
    atlas_after_second = draw_text_module._build_block_atlas.cache_info()
    np.testing.assert_array_equal(_host(output_a), _host(output_b))
    assert atlas_after_second.hits == atlas_after_first.hits + 1

    changed = px.draw.text(_frame(), font=second, **{**kwargs, "variations": {"wdth": 75.0}})
    assert draw_text_module._build_block_atlas.cache_info().misses == atlas_after_second.misses + 1
    assert not np.array_equal(_host(changed), _host(output_b))


def test_draw_text_user_font_missing_codepoint_shapes_selected_face_notdef() -> None:
    """v1-draw-text-user-font acceptance 21: a missing code point stays glyph zero without fallback discovery."""
    import pixtreme._draw.text as draw_text_module

    font = px.draw.Font.from_file(STATIC_FONT)
    shaped = draw_text_module._shape_text(chr(0x10FFFF), 24 * 64, 400.0, "ja", font, True, ())
    assert len(shaped.glyphs) == 1
    assert shaped.glyphs[0].glyph_id == 0
    result = px.draw.text(
        _frame(),
        text=chr(0x10FFFF),
        position=(12.0, 48.0),
        size=24.0,
        color=(1.0,),
        font=font,
    )
    assert result.shape == (72, 160, 1)


def test_draw_text_user_font_bitmap_guard_identifies_selected_asset(monkeypatch: pytest.MonkeyPatch) -> None:
    """v1-draw-text-user-font acceptance 22: bitmap layout guards identify user path and face without token-only repair."""
    import pixtreme._draw.text as draw_text_module

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

    font = px.draw.Font.from_file(COLLECTION_FONT, face_index=1)
    monkeypatch.setattr(draw_text_module, "_freetype_face", lambda *_args: Face())
    draw_text_module._glyph_bitmap.cache_clear()
    with pytest.raises(RuntimeError) as error:
        draw_text_module._glyph_bitmap(7, 64, 400.0, 0, 0, 0, font, False, ())
    message = _assert_actionable(error)
    assert str(COLLECTION_FONT) in message and "face_index=1" in message
    assert "bundled font" not in message


def test_draw_text_user_font_documentation_is_self_contained() -> None:
    """v1-draw-text-user-font acceptance 23; v1-lut-extensions acceptance 26; GitHub #29: counts stay current."""
    requirements = require_repo_file("docs/requirements.md").read_text(encoding="utf-8")
    tokens = (ROOT / "docs_site" / "tokens.md").read_text(encoding="utf-8")
    text_doc = inspect.getdoc(px.draw.text) or ""
    font_doc = inspect.getdoc(px.draw.Font.from_file) or ""
    assert "draw.Font" in requirements and "公開型" in requirements and "5 点" in requirements
    for required in (
        "Font.from_file",
        "face_index",
        "bytes",
        "variations",
        "cache",
        ".notdef",
        "fallback",
        "system-font",
    ):
        assert required in tokens
    for required in ("Font", "variations", "weight", ".notdef", "fallback", "cache"):
        assert required in text_doc
    for required in ("face_index", "bytes", "axis", "cache"):
        assert required in font_doc


def test_draw_text_user_font_fixtures_are_repo_owned_and_reproducible(tmp_path: Path) -> None:
    """v1-draw-text-user-font acceptance 24: font fixtures regenerate locally from bundled Noto without network input."""
    generated = tmp_path / "generated"
    completed = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "generate_draw_text_user_font_fixtures.py"), "--output", str(generated)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert {path.name for path in FIXTURE_ROOT.iterdir()} == {
        "noto-user-static.otf",
        "noto-user-variable.otf",
        "noto-user-collection.ttc",
    }
    for fixture in FIXTURE_ROOT.iterdir():
        assert fixture.read_bytes() == (generated / fixture.name).read_bytes()


def test_draw_text_user_font_changes_no_gpu_or_performance_surface() -> None:
    """v1-draw-text-user-font acceptance 25: user fonts reuse CPU raster and the existing token performance cases."""
    import pixtreme._draw.text as draw_text_module

    kernel_source = draw_text_module._TEXT_COMPOSITE_KERNEL_SOURCE
    assert "Font" not in kernel_source
    assert "face_index" not in kernel_source
    assert "variation" not in kernel_source
    registry = (ROOT / "tests" / "test_performance_spec.py").read_text(encoding="utf-8")
    assert '"draw-text-cjk-outline"' in registry
    assert "draw-text-user-font" not in registry
