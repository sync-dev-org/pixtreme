"""CPU-laid-out, GPU-composited text drawing."""

from __future__ import annotations

import hashlib
import io
import math
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import cupy as cp
import numpy as np

from pixtreme._core import validation as _validation
from pixtreme._core.blend import _BLEND_DEVICE_SOURCE, _DRAW_BLEND_CODES, _DRAW_BLEND_TOKENS
from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import Frame, _validate_float32_frame
from pixtreme._core.validation import (
    _bounded_real,
    _finite_pair,
    _finite_real,
    _normalized_closed_token,
    _positive_real,
    _strict_bool,
)
from pixtreme._core.vocabulary import (
    _TEXT_ALIGN_TOKENS,
    _TEXT_ANCHOR_TOKENS,
    _TEXT_FONT_TOKENS,
    _TEXT_LANGUAGE_TOKENS,
    TextFont,
)
from pixtreme._draw.shapes import _device_values

_LANGUAGE_TOKENS = _TEXT_LANGUAGE_TOKENS
_ALIGN_TOKENS = _TEXT_ALIGN_TOKENS
_FONT_TOKENS = _TEXT_FONT_TOKENS
_MONO_WEIGHT_MINIMUM = 400.0
_MONO_WEIGHT_MAXIMUM = 700.0
_ANCHOR_TOKENS = _TEXT_ANCHOR_TOKENS
_FONT_PATH = Path(__file__).parent.parent / "data" / "fonts" / "NotoSansCJKjp-VF.otf"
_FONT_PATHS = {
    "sans": _FONT_PATH,
    "mono": Path(__file__).parent.parent / "data" / "fonts" / "NotoSansMonoCJKjp-VF.otf",
}
_TEXT_BLOCK = (16, 16)
_HOST_ARRAY_WHY = "draw text color and outline inputs must be convertible to a regular host array"
_HOST_ARRAY_HOW = "pass a sequence, NumPy array, or CuPy array with a regular numeric shape"
_BLEND_TOKENS = _DRAW_BLEND_TOKENS


@dataclass(frozen=True, slots=True)
class _VariationAxis:
    tag: str
    minimum: float
    default: float
    maximum: float


def _open_freetype_face(data: bytes, face_index: int) -> Any:
    import freetype

    return freetype.Face(io.BytesIO(data), index=face_index)


def _open_harfbuzz_face(data: bytes, face_index: int) -> Any:
    import uharfbuzz as hb

    return hb.Face(data, face_index)


def _axis_record(tag: object, minimum: object, default: object, maximum: object) -> _VariationAxis:
    if (
        not isinstance(tag, str)
        or len(tag) != 4
        or not tag.isascii()
        or not all(0x20 <= ord(item) <= 0x7E for item in tag)
    ):
        raise ValueError(f"invalid OpenType variation axis tag {tag!r}")
    values = tuple(float(cast(Any, value)) for value in (minimum, default, maximum))
    if not all(math.isfinite(value) for value in values) or not values[0] <= values[1] <= values[2]:
        raise ValueError(
            f"invalid OpenType variation axis range tag={tag!r}, minimum={values[0]!r}, "
            f"default={values[1]!r}, maximum={values[2]!r}"
        )
    return _VariationAxis(tag=tag, minimum=values[0], default=values[1], maximum=values[2])


def _measure_font_axes(freetype_face: Any, harfbuzz_face: Any) -> tuple[_VariationAxis, ...]:
    harfbuzz_axes = tuple(
        _axis_record(axis.tag, axis.min_value, axis.default_value, axis.max_value) for axis in harfbuzz_face.axis_infos
    )
    if harfbuzz_axes or freetype_face.has_multiple_masters:
        freetype_info = freetype_face.get_variation_info()
        freetype_axes = tuple(
            _axis_record(axis.tag, axis.minimum, axis.default, axis.maximum) for axis in freetype_info.axes
        )
    else:
        freetype_axes = ()
    tags = tuple(axis.tag for axis in harfbuzz_axes)
    if len(tags) != len(set(tags)):
        raise ValueError(f"duplicate OpenType variation axis tags {tags!r}")
    if freetype_axes != harfbuzz_axes:
        raise ValueError(
            f"FreeType and HarfBuzz variation axis tables differ: "
            f"freetype={freetype_axes!r}, harfbuzz={harfbuzz_axes!r}"
        )
    return harfbuzz_axes


@dataclass(frozen=True, slots=True, init=False, repr=False, eq=False)
class Font:
    """Immutable draw-text font asset built from a construction-time file snapshot."""

    _data: bytes
    _content_hash: bytes
    _face_index: int
    _diagnostic_path: str
    _axes: tuple[_VariationAxis, ...]

    def __init__(self) -> None:
        raise TypeError("Font objects are constructed with Font.from_file(path, face_index=...)")

    @classmethod
    def from_file(cls, path: str | os.PathLike[str], *, face_index: int = 0) -> Font:
        """Snapshot one font file face after validating both backends and its variation axes.

        ``path`` is read completely during this call; later file changes do not
        affect the asset. ``face_index`` selects a face in a single-face file or
        collection. Equality, hashing, and private caches use content bytes plus
        face index, never the diagnostic path. Invalid files, face indices, or
        axis tables fail during construction with actionable ``ValueError``.
        """
        try:
            file_path = os.fspath(path)
        except (TypeError, ValueError, OSError) as error:
            raise ValueError(
                _actionable_error(
                    why="font file path must be one str or string-valued path-like object",
                    what=f"received path={path!r} with state={type(error).__name__}",
                    how="pass an existing readable regular font file as str or os.PathLike[str]",
                )
            ) from error
        if not isinstance(file_path, str):
            raise ValueError(
                _actionable_error(
                    why="font file path must resolve to one str path",
                    what=f"received path={path!r} resolving to {type(file_path).__module__}.{type(file_path).__qualname__}",
                    how="pass an existing readable regular font file as str or os.PathLike[str]",
                )
            )

        resolved_path = os.path.abspath(file_path)
        try:
            file_state = os.stat(resolved_path)
        except (OSError, ValueError) as error:
            raise ValueError(
                _actionable_error(
                    why="font file cannot be assetized because its file-system state is unavailable",
                    what=f"path={file_path!r}, state={type(error).__name__}: {error}",
                    how="pass an existing readable regular font file path",
                )
            ) from error
        if not stat.S_ISREG(file_state.st_mode):
            raise ValueError(
                _actionable_error(
                    why="font file must be a readable regular file",
                    what=f"path={file_path!r}, state=not-a-regular-file",
                    how="pass a path to an existing readable regular font file",
                )
            )
        try:
            with open(resolved_path, "rb") as stream:
                data = stream.read()
        except (OSError, UnicodeError, ValueError) as error:
            raise ValueError(
                _actionable_error(
                    why="font file bytes could not be read completely",
                    what=f"path={file_path!r}, state={type(error).__name__}: {error}",
                    how="make the regular font file readable and pass its path again",
                )
            ) from error

        try:
            first_freetype_face = _open_freetype_face(data, 0)
        except Exception as error:
            raise ValueError(
                _actionable_error(
                    why="FreeType could not open the font file bytes",
                    what=f"path={file_path!r}, face_index={face_index!r}, state={type(error).__name__}: {error}",
                    how="pass a font file whose first face FreeType can open",
                )
            ) from error
        face_count = int(first_freetype_face.num_faces)
        if isinstance(face_index, bool) or not isinstance(face_index, int) or not 0 <= face_index < face_count:
            raise ValueError(
                _actionable_error(
                    why="face_index must identify one measured font face with a non-bool integer",
                    what=f"face_index={face_index!r}, face_count={face_count}",
                    how="pass a non-bool int satisfying 0 <= face_index < face_count",
                )
            )
        try:
            freetype_face = first_freetype_face if face_index == 0 else _open_freetype_face(data, face_index)
        except Exception as error:
            raise ValueError(
                _actionable_error(
                    why="FreeType could not open the selected font face",
                    what=f"path={file_path!r}, face_index={face_index}, face_count={face_count}, state={type(error).__name__}: {error}",
                    how="pass a face index that FreeType can open from this font file",
                )
            ) from error
        try:
            harfbuzz_face = _open_harfbuzz_face(data, face_index)
        except Exception as error:
            raise ValueError(
                _actionable_error(
                    why="HarfBuzz could not construct the selected font face",
                    what=f"path={file_path!r}, face_index={face_index}, state={type(error).__name__}: {error}",
                    how="pass a font file and face index that both FreeType and HarfBuzz can open",
                )
            ) from error
        try:
            axes = _measure_font_axes(freetype_face, harfbuzz_face)
        except Exception as error:
            raise ValueError(
                _actionable_error(
                    why="font variation axes could not be measured as valid OpenType axes",
                    what=f"path={file_path!r}, face_index={face_index}, state={type(error).__name__}: {error}",
                    how="pass a static font or a variable font with valid tag and minimum/default/maximum axis records",
                )
            ) from error

        instance = object.__new__(cls)
        object.__setattr__(instance, "_data", data)
        object.__setattr__(instance, "_content_hash", hashlib.sha256(data).digest())
        object.__setattr__(instance, "_face_index", face_index)
        object.__setattr__(instance, "_diagnostic_path", resolved_path)
        object.__setattr__(instance, "_axes", axes)
        return instance

    def __hash__(self) -> int:
        return hash((self._content_hash, self._face_index))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Font):
            return NotImplemented
        return self._content_hash == other._content_hash and self._face_index == other._face_index

    def __repr__(self) -> str:
        return f"Font.from_file({self._diagnostic_path!r}, face_index={self._face_index})"


_FontReference = TextFont | Font
_AxisCoordinates = tuple[tuple[str, float], ...]

_TEXT_COMPOSITE_KERNEL_SOURCE = (
    _BLEND_DEVICE_SOURCE
    + r"""
extern "C" __global__ void pixtreme_draw_text_composite(
    float* __restrict__ output,
    const float* __restrict__ coverage,
    const float* __restrict__ color,
    const long long image_width,
    const long long channel_count,
    const long long output_left,
    const long long output_top,
    const long long region_width,
    const long long region_height,
    const long long atlas_width,
    const long long atlas_left,
    const long long atlas_top,
    const float opacity,
    const int blend
) {
    const long long local_x = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    const long long local_y = (long long)blockDim.y * blockIdx.y + threadIdx.y;
    if (local_x >= region_width || local_y >= region_height) {
        return;
    }
    const long long image_x = output_left + local_x;
    const long long image_y = output_top + local_y;
    const long long coverage_x = atlas_left + local_x;
    const long long coverage_y = atlas_top + local_y;
    const float alpha = coverage[coverage_y * atlas_width + coverage_x] * opacity;
    if (alpha <= 0.0f) {
        return;
    }
    const long long output_offset =
        (image_y * image_width + image_x) * channel_count;
    for (long long channel = 0; channel < channel_count; ++channel) {
        const float destination = output[output_offset + channel];
        const float source_color = color[channel];
        const float blend_value = pixtreme_blend(destination, source_color, blend);
        output[output_offset + channel] =
            destination * (1.0f - alpha) + blend_value * alpha;
    }
}
"""
)


@dataclass(frozen=True)
class _ShapedGlyph:
    glyph_id: int
    x_26_6: int
    y_down_26_6: int


@dataclass(frozen=True)
class _ShapedText:
    glyphs: tuple[_ShapedGlyph, ...]
    advance_26_6: int


@dataclass(frozen=True)
class _GlyphBitmap:
    coverage: np.ndarray
    left: int
    top: int


@dataclass(frozen=True)
class _Atlas:
    left: int
    top: int
    body: np.ndarray
    rings: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class _BlockLayout:
    lines: tuple[_ShapedText, ...]
    line_advances_26_6: tuple[int, ...]
    block_width_26_6: int
    line_step_26_6: int
    ascender_26_6: int
    descender_26_6: int
    tracking_26_6: int


def _round_fraction(numerator: int, denominator: int = 1) -> int:
    try:
        value = numerator / denominator
    except OverflowError:
        pass
    else:
        if math.isfinite(value):
            return round(value)
    return round(Fraction(numerator, denominator))


def _round_finite_scaled(value: float, multiplier: int, divisor: int = 1) -> int:
    try:
        scaled = value * multiplier / divisor
    except OverflowError:
        pass
    else:
        if math.isfinite(scaled):
            return round(scaled)
    value_numerator, value_denominator = value.as_integer_ratio()
    return _round_fraction(value_numerator * multiplier, value_denominator * divisor)


@lru_cache(maxsize=2)
def _font_bytes(font: str = "sans") -> bytes:
    return _FONT_PATHS[font].read_bytes()


def _font_data(font: _FontReference) -> bytes:
    return font._data if isinstance(font, Font) else _font_bytes(font)


def _font_description(font: _FontReference) -> str:
    if isinstance(font, Font):
        return f"path={font._diagnostic_path!r}, face_index={font._face_index}"
    return f"font={font!r}"


def _effective_axis_coordinates(
    font: _FontReference,
    weight: float,
    axis_coordinates: _AxisCoordinates,
) -> _AxisCoordinates:
    if axis_coordinates:
        return axis_coordinates
    if isinstance(font, Font):
        return ()
    return (("wght", weight),)


@lru_cache(maxsize=512)
def _shape_text(
    text: str,
    size_26_6: int,
    weight: float,
    language: str,
    font: _FontReference = "sans",
    kerning: bool = True,
    axis_coordinates: _AxisCoordinates = (),
) -> _ShapedText:
    if text == "":
        return _ShapedText(glyphs=(), advance_26_6=0)

    import uharfbuzz as hb

    face = hb.Face(_font_data(font), font._face_index if isinstance(font, Font) else 0)
    shaped_font = hb.Font(face)
    shaped_font.scale = (size_26_6, size_26_6)
    hb.ot_font_set_funcs(shaped_font)
    coordinates = _effective_axis_coordinates(font, weight, axis_coordinates)
    if coordinates:
        shaped_font.set_variations(dict(coordinates))
    buffer = hb.Buffer()
    buffer.add_str(text)
    buffer.guess_segment_properties()
    buffer.direction = "ltr"
    buffer.language = language
    hb.shape(shaped_font, buffer, {"kern": kerning, "liga": True, "locl": True})

    pen_x = 0
    pen_y = 0
    glyphs: list[_ShapedGlyph] = []
    for info, position in zip(buffer.glyph_infos, buffer.glyph_positions, strict=True):
        glyphs.append(
            _ShapedGlyph(
                glyph_id=int(info.codepoint),
                x_26_6=pen_x + int(position.x_offset),
                y_down_26_6=-(pen_y + int(position.y_offset)),
            )
        )
        pen_x += int(position.x_advance)
        pen_y += int(position.y_advance)
    return _ShapedText(glyphs=tuple(glyphs), advance_26_6=pen_x)


@lru_cache(maxsize=128)
def _freetype_face(
    size_26_6: int,
    weight: float,
    font: _FontReference = "sans",
    axis_coordinates: _AxisCoordinates = (),
) -> Any:
    import freetype

    face = (
        _open_freetype_face(font._data, font._face_index)
        if isinstance(font, Font)
        else freetype.Face(str(_FONT_PATHS[font]))
    )
    face.set_char_size(0, size_26_6, 72, 72)
    coordinates = _effective_axis_coordinates(font, weight, axis_coordinates)
    if coordinates:
        face.set_var_design_coords(tuple(value for _tag, value in coordinates))
    return face


@lru_cache(maxsize=128)
def _font_metrics_26_6(
    size_26_6: int,
    weight: float,
    font: _FontReference = "sans",
    axis_coordinates: _AxisCoordinates = (),
) -> tuple[int, int]:
    face = _freetype_face(size_26_6, weight, font, axis_coordinates)
    return (
        round(int(face.ascender) * size_26_6 / int(face.units_per_EM)),
        round(int(face.descender) * size_26_6 / int(face.units_per_EM)),
    )


@lru_cache(maxsize=256)
def _font_line_advance_26_6(
    size_26_6: int,
    weight: float,
    font: _FontReference,
    line_spacing: float,
    axis_coordinates: _AxisCoordinates = (),
) -> int:
    face = _freetype_face(size_26_6, weight, font, axis_coordinates)
    return _round_finite_scaled(
        line_spacing,
        int(face.height) * size_26_6,
        int(face.units_per_EM),
    )


@lru_cache(maxsize=4096)
def _glyph_bitmap(
    glyph_id: int,
    size_26_6: int,
    weight: float,
    stroke_radius_26_6: int,
    phase_x_26_6: int,
    phase_y_down_26_6: int,
    font: _FontReference = "sans",
    supersample: bool = False,
    axis_coordinates: _AxisCoordinates = (),
) -> _GlyphBitmap:
    import freetype

    def load(scale: int) -> Any:
        face = _freetype_face(size_26_6 * scale, weight, font, axis_coordinates)
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
        return glyph

    def render(scale: int) -> _GlyphBitmap:
        glyph = load(scale)
        rendered = glyph.to_bitmap(
            freetype.FT_RENDER_MODE_NORMAL,
            freetype.Vector(phase_x_26_6 * scale, -phase_y_down_26_6 * scale),
            destroy=False,
        )
        bitmap = rendered.bitmap
        if bitmap.rows == 0 or bitmap.width == 0:
            coverage = np.zeros((0, 0), dtype=np.float32)
        else:
            pitch = abs(int(bitmap.pitch))
            if pitch < int(bitmap.width) or int(bitmap.pixel_mode) != int(freetype.FT_PIXEL_MODE_GRAY):
                raise RuntimeError(
                    _actionable_error(
                        why="FreeType returned a bitmap layout that cannot be read as grayscale coverage",
                        what=(
                            f"{_font_description(font)}, glyph_id={glyph_id}, rows={int(bitmap.rows)}, "
                            f"width={int(bitmap.width)}, "
                            f"pitch={int(bitmap.pitch)}, pixel_mode={int(bitmap.pixel_mode)}"
                        ),
                        how="use a font whose selected face emits 8-bit grayscale bitmaps with absolute pitch at least width",
                    )
                )
            byte_count = int(bitmap.rows) * pitch
            # ``Bitmap.buffer`` builds one Python int per byte; view the same owned buffer directly before fp32 copy.
            coverage = np.ctypeslib.as_array(bitmap._FT_Bitmap.buffer, shape=(byte_count,)).reshape(
                int(bitmap.rows), pitch
            )[:, : int(bitmap.width)]
            coverage = coverage.astype(np.float32) / np.float32(255.0)
        coverage.setflags(write=False)
        return _GlyphBitmap(coverage=coverage, left=int(rendered.left), top=int(rendered.top))

    if not supersample:
        return render(1)

    normal_glyph = load(1)
    bounds = normal_glyph.get_cbox(freetype.FT_GLYPH_BBOX_SUBPIXELS)
    if int(bounds.xMin) == int(bounds.xMax) or int(bounds.yMin) == int(bounds.yMax):
        return render(1)
    normal_left = (int(bounds.xMin) + phase_x_26_6) // 64
    normal_right = -(-(int(bounds.xMax) + phase_x_26_6) // 64)
    normal_top = -(-(int(bounds.yMax) - phase_y_down_26_6) // 64)
    normal_bottom = (int(bounds.yMin) - phase_y_down_26_6) // 64
    normal_width = normal_right - normal_left
    normal_height = normal_top - normal_bottom
    if normal_width == 0 or normal_height == 0:
        coverage = np.zeros((normal_height, normal_width), dtype=np.float32)
        coverage.setflags(write=False)
        return _GlyphBitmap(coverage=coverage, left=normal_left, top=normal_top)

    high_resolution = render(4)
    samples = np.zeros((normal_height * 4, normal_width * 4), dtype=np.float32)
    _merge_max(
        samples,
        high_resolution.coverage,
        left=high_resolution.left - normal_left * 4,
        top=normal_top * 4 - high_resolution.top,
    )
    coverage = samples.reshape(normal_height, 4, normal_width, 4).mean(
        axis=(1, 3),
        dtype=np.float32,
    )
    coverage.setflags(write=False)
    return _GlyphBitmap(coverage=coverage, left=normal_left, top=normal_top)


def _merge_max(
    destination: np.ndarray,
    coverage: np.ndarray,
    *,
    left: int,
    top: int,
) -> None:
    if coverage.size == 0:
        return
    relative_left = left
    relative_top = top
    relative_right = left + coverage.shape[1]
    relative_bottom = top + coverage.shape[0]
    output_left = max(0, relative_left)
    output_top = max(0, relative_top)
    output_right = min(destination.shape[1], relative_right)
    output_bottom = min(destination.shape[0], relative_bottom)
    if output_right <= output_left or output_bottom <= output_top:
        return
    source_left = output_left - relative_left
    source_top = output_top - relative_top
    source_right = source_left + output_right - output_left
    source_bottom = source_top + output_bottom - output_top
    np.maximum(
        destination[output_top:output_bottom, output_left:output_right],
        coverage[source_top:source_bottom, source_left:source_right],
        out=destination[output_top:output_bottom, output_left:output_right],
    )


@lru_cache(maxsize=1)
def _text_composite_kernel() -> cp.RawKernel:
    return cp.RawKernel(_TEXT_COMPOSITE_KERNEL_SOURCE, "pixtreme_draw_text_composite")


def _composite_layer(
    output: cp.ndarray,
    coverage: cp.ndarray,
    color: cp.ndarray,
    *,
    image_width: int,
    image_height: int,
    atlas_left: int,
    atlas_top: int,
    opacity: float,
    blend: str,
) -> None:
    if coverage.size == 0 or opacity == 0.0:
        return
    coverage_height, coverage_width = cast(tuple[int, int], coverage.shape)
    output_left = max(0, atlas_left)
    output_top = max(0, atlas_top)
    output_right = min(image_width, atlas_left + coverage_width)
    output_bottom = min(image_height, atlas_top + coverage_height)
    if output_right <= output_left or output_bottom <= output_top:
        return
    region_width = output_right - output_left
    region_height = output_bottom - output_top
    coverage_left = output_left - atlas_left
    coverage_top = output_top - atlas_top
    grid = (
        (region_width + _TEXT_BLOCK[0] - 1) // _TEXT_BLOCK[0],
        (region_height + _TEXT_BLOCK[1] - 1) // _TEXT_BLOCK[1],
    )
    _text_composite_kernel()(
        grid,
        _TEXT_BLOCK,
        (
            output,
            coverage,
            color,
            np.int64(image_width),
            np.int64(output.shape[2]),
            np.int64(output_left),
            np.int64(output_top),
            np.int64(region_width),
            np.int64(region_height),
            np.int64(coverage_width),
            np.int64(coverage_left),
            np.int64(coverage_top),
            np.float32(opacity),
            np.int32(_DRAW_BLEND_CODES[blend]),
        ),
    )


def _host_array(value: object) -> np.ndarray:
    return _validation._host_array(value, why=_HOST_ARRAY_WHY, how=_HOST_ARRAY_HOW)


def _color(
    value: object,
    *,
    channel_count: int,
    name: str,
) -> tuple[float, ...]:
    try:
        array = _host_array(value)
    except ValueError:
        array = np.asarray((), dtype=np.float32)
    if array.shape != (channel_count,):
        raise ValueError(
            _actionable_error(
                why=f"{name} must have exactly one real value per Frame channel",
                what=f"received {name} shape {array.shape!r} for {channel_count} channels",
                how=f"pass {name} as a finite real sequence of length {channel_count}",
            )
        )
    return tuple(
        _finite_real(item.item() if isinstance(item, np.generic) else item, name=f"{name}[{index}]")
        for index, item in enumerate(array)
    )


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(
            _actionable_error(
                why="text must be one Python str",
                what=f"received {type(value).__module__}.{type(value).__qualname__}",
                how=r"pass one str whose explicit lines are separated by \n",
            )
        )
    if "\r" in value:
        raise ValueError(
            _actionable_error(
                why=r"text splits only on \n and does not normalize carriage returns",
                what=r"received text containing \r",
                how=r"remove every \r and use \n for each explicit line break",
            )
        )
    return value


def _weight(value: object) -> float:
    return _bounded_real(
        value,
        name="weight",
        minimum=100.0,
        maximum=900.0,
        why="weight must select the bundled font wght axis from 100 through 900",
        how="pass a finite real weight in the closed interval from 100.0 through 900.0",
    )


_TOKEN_FONT_AXES = {
    "sans": (_VariationAxis("wght", 100.0, 100.0, 900.0),),
    "mono": (_VariationAxis("wght", _MONO_WEIGHT_MINIMUM, _MONO_WEIGHT_MINIMUM, _MONO_WEIGHT_MAXIMUM),),
}


def _font_reference(value: object) -> _FontReference:
    if isinstance(value, Font):
        return value
    return cast(TextFont, _normalized_closed_token(value, axis="font", accepted=_FONT_TOKENS))


def _font_axes(font: _FontReference) -> tuple[_VariationAxis, ...]:
    return font._axes if isinstance(font, Font) else _TOKEN_FONT_AXES[font]


def _font_weight(value: object, *, font: _FontReference) -> float:
    if isinstance(font, Font):
        weight_axis = next((axis for axis in font._axes if axis.tag == "wght"), None)
        try:
            resolved = _finite_real(value, name="weight")
        except ValueError as error:
            if weight_axis is None:
                why = "weight must be the finite real value 400.0 for a static user font without a wght axis"
                range_description = "static font requires weight=400.0"
                how = "pass weight=400.0 or select a variable Font with a wght axis"
            else:
                why = "weight must be a finite real within the selected user font wght axis"
                range_description = f"wght=[{weight_axis.minimum}, {weight_axis.maximum}]"
                how = f"pass a finite real weight in the closed interval [{weight_axis.minimum}, {weight_axis.maximum}]"
            raise ValueError(
                _actionable_error(
                    why=why,
                    what=(
                        f"{_font_description(font)}, weight={value!r}, "
                        f"type={type(value).__module__}.{type(value).__qualname__}, {range_description}"
                    ),
                    how=how,
                )
            ) from error
        if weight_axis is None:
            if resolved == 400.0:
                return resolved
            raise ValueError(
                _actionable_error(
                    why="weight must remain 400.0 when the selected user font is a static font without a wght axis",
                    what=f"{_font_description(font)}, static font, weight={value!r}",
                    how="keep weight=400.0 or pass a variable Font whose selected face has a wght axis",
                )
            )
        if weight_axis.minimum <= resolved <= weight_axis.maximum:
            return resolved
        raise ValueError(
            _actionable_error(
                why="weight must lie within the selected user font wght axis",
                what=(
                    f"{_font_description(font)}, weight={value!r}, wght=[{weight_axis.minimum}, {weight_axis.maximum}]"
                ),
                how=f"pass a finite real weight in the closed interval [{weight_axis.minimum}, {weight_axis.maximum}]",
            )
        )
    if font == "sans":
        return _weight(value)
    resolved = _finite_real(value, name="weight")
    if _MONO_WEIGHT_MINIMUM <= resolved <= _MONO_WEIGHT_MAXIMUM:
        return resolved
    raise ValueError(
        _actionable_error(
            why="weight must select the bundled mono font wght axis from 400 through 700",
            what=f"received weight={value!r} for font='mono'",
            how="pass a finite real weight in the closed interval from 400.0 through 700.0",
        )
    )


def _variation_values(value: object, *, font: _FontReference) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(
            _actionable_error(
                why="variations must be None or a mapping from axis tag str to finite real value",
                what=f"received variations={value!r} of type {type(value).__module__}.{type(value).__qualname__}",
                how="pass variations=None, an empty mapping, or a partial mapping such as {'wdth': 100.0}",
            )
        )
    try:
        items = tuple(value.items())
    except Exception as error:
        raise ValueError(
            _actionable_error(
                why="variations mapping items could not be read",
                what=f"received variations={value!r}, state={type(error).__name__}: {error}",
                how="pass a readable mapping from axis tag str to finite real value",
            )
        ) from error

    axes = _font_axes(font)
    axis_by_tag = {axis.tag: axis for axis in axes}
    resolved: dict[str, float] = {}
    for key, item in items:
        if not isinstance(key, str):
            raise ValueError(
                _actionable_error(
                    why="each variations key must be one case-sensitive OpenType axis tag str",
                    what=f"received key={key!r} of type {type(key).__module__}.{type(key).__qualname__}",
                    how=f"pass one of the measured axis tags {tuple(axis_by_tag)!r} as each mapping key",
                )
            )
        if key == "wght":
            raise ValueError(
                _actionable_error(
                    why="variations cannot contain wght because weight has one unambiguous public argument",
                    what="received variations key 'wght'",
                    how=f"remove 'wght' from variations and pass the same value as weight={item!r}",
                )
            )
        axis = axis_by_tag.get(key)
        if axis is None:
            raise ValueError(
                _actionable_error(
                    why="variations key must name an axis measured from the selected font",
                    what=f"received axis={key!r}, measured_axes={tuple(axis_by_tag)!r}",
                    how=f"pass only case-sensitive axis tags from {tuple(axis_by_tag)!r}",
                )
            )
        try:
            number = _finite_real(item, name=f"variations[{key!r}]")
        except ValueError as error:
            raise ValueError(
                _actionable_error(
                    why=f"variations[{key!r}] must be a finite real excluding bool",
                    what=(
                        f"received value={item!r}, type={type(item).__module__}.{type(item).__qualname__}, "
                        f"axis={key!r}, range=[{axis.minimum}, {axis.maximum}]"
                    ),
                    how=f"pass a non-bool finite real in the closed interval [{axis.minimum}, {axis.maximum}]",
                )
            ) from error
        if not axis.minimum <= number <= axis.maximum:
            raise ValueError(
                _actionable_error(
                    why=f"variations[{key!r}] must lie within the measured axis range",
                    what=f"received value={item!r}, axis={key!r}, range=[{axis.minimum}, {axis.maximum}]",
                    how=f"pass a finite real value in the closed interval [{axis.minimum}, {axis.maximum}]",
                )
            )
        resolved[key] = number
    return resolved


def _resolve_axis_coordinates(
    font: _FontReference,
    *,
    weight: float,
    variations: object,
) -> _AxisCoordinates:
    specified = _variation_values(variations, font=font)
    return tuple(
        (
            axis.tag,
            weight if axis.tag == "wght" else specified.get(axis.tag, axis.default),
        )
        for axis in _font_axes(font)
    )


def _width(value: object) -> float | None:
    if value is None:
        return None
    return _bounded_real(
        value,
        name="width",
        minimum=0.0,
        why="width must be None or a non-negative finite pixel width",
        how="pass width=None or a finite real width greater than or equal to 0.0",
    )


def _result_frame(frame: Frame, data: cp.ndarray) -> Frame:
    return Frame(
        data=data,
        colorspace=frame.colorspace,
        gamma=frame.gamma,
        channels=frame.channels,
        matrix=frame.matrix,
    )


def _outlines(
    value: object,
    *,
    channel_count: int,
) -> tuple[tuple[tuple[float, ...], float], ...]:
    if value is None:
        return ()
    try:
        entries = tuple(cast(Any, value))
    except (TypeError, ValueError):
        entries = ()
        valid_sequence = False
    else:
        valid_sequence = True
    if not valid_sequence:
        raise ValueError(
            _actionable_error(
                why="outlines must be None or a sequence of (color, width) pairs",
                what=f"received outlines={value!r}",
                how="pass outlines=None or ordered (color, positive width) pairs from inner to outer",
            )
        )

    resolved: list[tuple[tuple[float, ...], float]] = []
    for index, entry in enumerate(entries):
        try:
            pair = tuple(cast(Any, entry))
        except (TypeError, ValueError):
            pair = ()
        if len(pair) != 2:
            raise ValueError(
                _actionable_error(
                    why=f"outlines[{index}] must be one (color, width) pair",
                    what=f"received outlines[{index}]={entry!r}",
                    how="pass each outline as a two-element pair ordered from inner to outer",
                )
            )
        resolved.append(
            (
                _color(pair[0], channel_count=channel_count, name=f"outlines[{index}].color"),
                _positive_real(pair[1], name=f"outlines[{index}].width"),
            )
        )
    return tuple(resolved)


@lru_cache(maxsize=256)
def _block_layout(
    text: str,
    size_26_6: int,
    weight: float,
    language: str,
    kerning: bool,
    tracking: float,
    line_spacing: float,
    font: _FontReference,
    width: float | None,
    axis_coordinates: _AxisCoordinates = (),
) -> _BlockLayout:
    lines = tuple(
        _shape_text(line, size_26_6, weight, language, font, kerning, axis_coordinates) for line in text.split("\n")
    )
    tracking_26_6 = _round_finite_scaled(tracking, size_26_6)
    line_advances_26_6 = tuple(line.advance_26_6 + tracking_26_6 * max(0, len(line.glyphs) - 1) for line in lines)
    block_width_26_6 = _round_finite_scaled(width, 64) if width is not None else max((0, *line_advances_26_6))
    ascender_26_6, descender_26_6 = _font_metrics_26_6(size_26_6, weight, font, axis_coordinates)
    return _BlockLayout(
        lines=lines,
        line_advances_26_6=line_advances_26_6,
        block_width_26_6=block_width_26_6,
        line_step_26_6=_font_line_advance_26_6(size_26_6, weight, font, line_spacing, axis_coordinates),
        ascender_26_6=ascender_26_6,
        descender_26_6=descender_26_6,
        tracking_26_6=tracking_26_6,
    )


def _block_origin_26_6(
    position: tuple[float, float],
    *,
    anchor: str,
    layout: _BlockLayout,
) -> tuple[int, int]:
    vertical, horizontal = anchor.split("-")
    block_left_26_6 = _round_finite_scaled(position[0], 64)
    if horizontal == "center":
        block_left_26_6 -= _round_fraction(layout.block_width_26_6, 2)
    elif horizontal == "right":
        block_left_26_6 -= layout.block_width_26_6

    first_baseline_26_6 = _round_finite_scaled(position[1], 64)
    final_baseline_offset = (len(layout.lines) - 1) * layout.line_step_26_6
    if vertical == "top":
        first_baseline_26_6 += layout.ascender_26_6
    elif vertical == "center":
        first_baseline_26_6 += _round_fraction(
            layout.ascender_26_6 + layout.descender_26_6 - final_baseline_offset,
            2,
        )
    elif vertical == "bottom":
        first_baseline_26_6 += layout.descender_26_6 - final_baseline_offset
    return block_left_26_6, first_baseline_26_6


def _line_offset_26_6(*, align: str, block_width_26_6: int, line_advance_26_6: int) -> int:
    if align == "center":
        return _round_fraction(block_width_26_6 - line_advance_26_6, 2)
    if align == "right":
        return block_width_26_6 - line_advance_26_6
    return 0


@lru_cache(maxsize=128)
def _build_block_atlas(
    text: str,
    size_26_6: int,
    weight: float,
    language: str,
    kerning: bool,
    tracking: float,
    line_spacing: float,
    align: str,
    font: _FontReference,
    width: float | None,
    phase_x_26_6: int,
    phase_y_down_26_6: int,
    outline_widths_26_6: tuple[int, ...],
    clip_left: int,
    clip_top: int,
    clip_right: int,
    clip_bottom: int,
    supersample: bool = False,
    axis_coordinates: _AxisCoordinates = (),
) -> _Atlas:
    layout = _block_layout(
        text,
        size_26_6,
        weight,
        language,
        kerning,
        tracking,
        line_spacing,
        font,
        width,
        axis_coordinates,
    )
    radii = (0, *outline_widths_26_6)
    placed_layers: list[list[tuple[_GlyphBitmap, int, int]]] = [[] for _ in radii]
    minimum_x: int | None = None
    minimum_y: int | None = None
    maximum_x: int | None = None
    maximum_y: int | None = None

    for layer_index, stroke_radius_26_6 in enumerate(radii):
        for line_index, (line, line_advance_26_6) in enumerate(
            zip(layout.lines, layout.line_advances_26_6, strict=True)
        ):
            line_offset_26_6 = _line_offset_26_6(
                align=align,
                block_width_26_6=layout.block_width_26_6,
                line_advance_26_6=line_advance_26_6,
            )
            justify_remainder = max(layout.block_width_26_6 - line_advance_26_6, 0) if align == "justify" else 0
            gap_count = max(0, len(line.glyphs) - 1)
            for glyph_index, shaped_glyph in enumerate(line.glyphs):
                justify_offset = _round_fraction(justify_remainder * glyph_index, gap_count) if gap_count else 0
                pen_x_26_6 = (
                    phase_x_26_6
                    + line_offset_26_6
                    + shaped_glyph.x_26_6
                    + layout.tracking_26_6 * glyph_index
                    + justify_offset
                )
                pen_y_26_6 = phase_y_down_26_6 + line_index * layout.line_step_26_6 + shaped_glyph.y_down_26_6
                pen_x_integer, glyph_phase_x_26_6 = divmod(pen_x_26_6, 64)
                pen_y_integer, glyph_phase_y_down_26_6 = divmod(pen_y_26_6, 64)
                bitmap = _glyph_bitmap(
                    shaped_glyph.glyph_id,
                    size_26_6,
                    weight,
                    stroke_radius_26_6,
                    glyph_phase_x_26_6,
                    glyph_phase_y_down_26_6,
                    font,
                    supersample,
                    axis_coordinates,
                )
                if bitmap.coverage.size == 0:
                    continue
                left = pen_x_integer + bitmap.left
                top = pen_y_integer - bitmap.top
                right = left + bitmap.coverage.shape[1]
                bottom = top + bitmap.coverage.shape[0]
                visible_left = max(left, clip_left)
                visible_top = max(top, clip_top)
                visible_right = min(right, clip_right)
                visible_bottom = min(bottom, clip_bottom)
                if visible_right <= visible_left or visible_bottom <= visible_top:
                    continue
                placed_layers[layer_index].append((bitmap, left, top))
                minimum_x = visible_left if minimum_x is None else min(minimum_x, visible_left)
                minimum_y = visible_top if minimum_y is None else min(minimum_y, visible_top)
                maximum_x = visible_right if maximum_x is None else max(maximum_x, visible_right)
                maximum_y = visible_bottom if maximum_y is None else max(maximum_y, visible_bottom)

    if minimum_x is None or minimum_y is None or maximum_x is None or maximum_y is None:
        empty = np.zeros((0, 0), dtype=np.float32)
        empty.setflags(write=False)
        return _Atlas(left=0, top=0, body=empty, rings=(empty,) * len(outline_widths_26_6))

    atlas_width = maximum_x - minimum_x
    atlas_height = maximum_y - minimum_y
    solid_layers = [np.zeros((atlas_height, atlas_width), dtype=np.float32) for _ in radii]
    for solid, placed in zip(solid_layers, placed_layers, strict=True):
        for bitmap, left, top in placed:
            _merge_max(solid, bitmap.coverage, left=left - minimum_x, top=top - minimum_y)

    body = solid_layers[0]
    solid_unions = [body]
    for stroke_band in solid_layers[1:]:
        np.maximum(body, stroke_band, out=stroke_band)
        solid_unions.append(stroke_band)
    body.setflags(write=False)
    rings: list[np.ndarray] = []
    for inner, outer in zip(solid_unions[:-1], solid_unions[1:], strict=True):
        ring = np.clip(outer - inner, 0.0, 1.0).astype(np.float32, copy=False)
        ring.setflags(write=False)
        rings.append(ring)
    return _Atlas(left=minimum_x, top=minimum_y, body=body, rings=tuple(rings))


def text(
    frame: Frame,
    *,
    text: str,
    position: Sequence[float],
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
    font: TextFont | Font = "sans",
    variations: Mapping[str, float] | None = None,
    width: float | None = None,
    supersample: bool = False,
) -> Frame:
    r"""Draw explicitly line-broken text with a bundled token or immutable ``Font``.

    Text is split literally on ``\n``; any ``\r`` is rejected. ``line_spacing``
    multiplies the font line advance, while ``tracking`` is an em ratio added
    after shaping. ``kerning`` toggles only the OpenType kern feature. ``font``
    selects bundled sans (weight 100.0 through 900.0), bundled mono (400.0
    through 700.0), or a ``Font.from_file`` bytes snapshot. For a user
    ``Font``, ``weight`` controls its measured ``wght`` axis and otherwise must
    remain 400.0. ``variations`` partially overrides any other measured axes;
    unspecified axes use their font defaults and invalid tags or ranges fail
    before drawing. The same resolved coordinates select HarfBuzz shaping,
    FreeType metrics, and glyph rasterization.
    Bundled mono accepts 400.0 through 700.0 without saturation.
    ``width`` is a pixel block width used by ``align``; ``justify``
    adds positive remaining width to shaped-glyph gaps. ``anchor`` identifies
    the block box at ``position``. Shaping, glyph rasters, and block atlases use
    private caches keyed by content bytes,
    face index, and resolved axis coordinates. Missing code points use the
    selected face's ``.notdef`` glyph without system, bundled, user, or network
    fallback. ``supersample=False``
    keeps the standard FreeType 8-bit coverage path. ``supersample=True`` is an
    opt-in internal precision path that rasterizes glyph bodies and outlines at
    4x size, phase, and stroke radius, then takes an fp32 4x4 box average on the
    unchanged output geometry. Both modes keep separate private cache entries.
    Composition does not clamp scene values. The input is unchanged and the
    result always owns new storage.
    """
    checked_frame = _validate_float32_frame(frame, operation="draw.text")
    checked_text = _text(text)
    checked_position = _finite_pair(position, name="position")
    checked_size = _positive_real(size, name="size")
    checked_font = _font_reference(font)
    checked_weight = _font_weight(weight, font=checked_font)
    checked_axis_coordinates = _resolve_axis_coordinates(
        checked_font,
        weight=checked_weight,
        variations=variations,
    )
    checked_color = _color(color, channel_count=len(checked_frame.channels), name="color")
    checked_language = _normalized_closed_token(language, axis="language", accepted=_LANGUAGE_TOKENS)
    checked_anchor = _normalized_closed_token(anchor, axis="anchor", accepted=_ANCHOR_TOKENS)
    checked_outlines = _outlines(outlines, channel_count=len(checked_frame.channels))
    checked_opacity = _bounded_real(
        opacity,
        name="opacity",
        minimum=0.0,
        maximum=1.0,
        why="opacity must be in the closed interval from 0 through 1",
        how="pass a finite real opacity from 0.0 through 1.0",
    )
    checked_blend = _normalized_closed_token(blend, axis="blend", accepted=_BLEND_TOKENS)
    checked_align = _normalized_closed_token(align, axis="align", accepted=_ALIGN_TOKENS)
    checked_line_spacing = _positive_real(line_spacing, name="line_spacing")
    checked_tracking = _finite_real(tracking, name="tracking")
    checked_kerning = _strict_bool(
        kerning,
        name="kerning",
        why="kerning must be one bool",
        how="pass kerning=True or kerning=False",
    )
    checked_width = _width(width)
    checked_supersample = _strict_bool(
        supersample,
        name="supersample",
        why="supersample must be one bool",
        how="pass supersample=True or supersample=False",
    )

    output = checked_frame.data.copy(order="C")
    if checked_opacity == 0.0 or all(line == "" for line in checked_text.split("\n")):
        return _result_frame(checked_frame, output)

    size_26_6 = max(1, round(checked_size * 64.0))
    layout = _block_layout(
        checked_text,
        size_26_6,
        checked_weight,
        checked_language,
        checked_kerning,
        checked_tracking,
        checked_line_spacing,
        checked_font,
        checked_width,
        checked_axis_coordinates,
    )
    block_left_26_6, first_baseline_26_6 = _block_origin_26_6(
        checked_position,
        anchor=checked_anchor,
        layout=layout,
    )
    block_left_integer, phase_x_26_6 = divmod(block_left_26_6, 64)
    first_baseline_integer, phase_y_down_26_6 = divmod(first_baseline_26_6, 64)

    cumulative_width = 0.0
    outline_widths_26_6: list[int] = []
    for _outline_color, outline_width in checked_outlines:
        cumulative_width += outline_width
        outline_widths_26_6.append(max(1, round(cumulative_width * 64.0)))
    atlas = _build_block_atlas(
        checked_text,
        size_26_6,
        checked_weight,
        checked_language,
        checked_kerning,
        checked_tracking,
        checked_line_spacing,
        checked_align,
        checked_font,
        checked_width,
        phase_x_26_6,
        phase_y_down_26_6,
        tuple(outline_widths_26_6),
        -block_left_integer,
        -first_baseline_integer,
        checked_frame.width - block_left_integer,
        checked_frame.height - first_baseline_integer,
        checked_supersample,
        checked_axis_coordinates,
    )
    atlas_left = block_left_integer + atlas.left
    atlas_top = first_baseline_integer + atlas.top
    device_id = cp.cuda.runtime.getDevice()

    for (outline_color, _outline_width), ring in zip(
        reversed(checked_outlines),
        reversed(atlas.rings),
        strict=True,
    ):
        _composite_layer(
            output,
            cp.asarray(ring),
            _device_values(device_id, outline_color),
            image_width=checked_frame.width,
            image_height=checked_frame.height,
            atlas_left=atlas_left,
            atlas_top=atlas_top,
            opacity=checked_opacity,
            blend=checked_blend,
        )
    _composite_layer(
        output,
        cp.asarray(atlas.body),
        _device_values(device_id, checked_color),
        image_width=checked_frame.width,
        image_height=checked_frame.height,
        atlas_left=atlas_left,
        atlas_top=atlas_top,
        opacity=checked_opacity,
        blend=checked_blend,
    )
    return _result_frame(checked_frame, output)
