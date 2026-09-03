"""GPU-native shape drawing with explicit coverage and blend contracts."""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import lru_cache

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
from pixtreme._core.vocabulary import _ANTIALIASING_TOKENS, Antialiasing, Blend

_BLEND_TOKENS = _DRAW_BLEND_TOKENS
_AA_TOKENS = _ANTIALIASING_TOKENS
_DRAW_BLOCK = (16, 16)
_PRIMITIVE_LINE = 0
_PRIMITIVE_POLYLINE = 1
_PRIMITIVE_RECTANGLE = 2
_PRIMITIVE_CIRCLE = 3
_PRIMITIVE_ELLIPSE = 4
_PRIMITIVE_POLYGON = 5
_HOST_ARRAY_WHY = "draw inputs must be convertible to a regular host array"
_HOST_ARRAY_HOW = "pass a sequence, NumPy array, or CuPy array with a regular numeric shape"

_DRAW_KERNEL_SOURCE = (
    _BLEND_DEVICE_SOURCE
    + r"""
__device__ float pixtreme_clamp01(const float value) {
    return value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
}

__device__ float pixtreme_segment_distance(
    const float x,
    const float y,
    const float start_x,
    const float start_y,
    const float end_x,
    const float end_y
) {
    const float delta_x = end_x - start_x;
    const float delta_y = end_y - start_y;
    const float denominator = delta_x * delta_x + delta_y * delta_y;
    if (denominator == 0.0f) {
        return hypotf(x - start_x, y - start_y);
    }
    const float projection = pixtreme_clamp01(
        ((x - start_x) * delta_x + (y - start_y) * delta_y) / denominator
    );
    return hypotf(
        x - (start_x + projection * delta_x),
        y - (start_y + projection * delta_y)
    );
}

__device__ float pixtreme_polyline_distance(
    const float x,
    const float y,
    const float* __restrict__ points,
    const int point_count,
    const int closed,
    const float thickness
) {
    float distance = 3.402823466e+38f;
    for (int point_index = 0; point_index < point_count - 1; ++point_index) {
        const int start_offset = point_index * 2;
        const int end_offset = start_offset + 2;
        distance = fminf(
            distance,
            pixtreme_segment_distance(
                x,
                y,
                points[start_offset],
                points[start_offset + 1],
                points[end_offset],
                points[end_offset + 1]
            )
        );
    }
    if (closed) {
        const int last_offset = (point_count - 1) * 2;
        distance = fminf(
            distance,
            pixtreme_segment_distance(
                x,
                y,
                points[last_offset],
                points[last_offset + 1],
                points[0],
                points[1]
            )
        );
    }
    return distance - 0.5f * thickness;
}

__device__ float pixtreme_rounded_rectangle_fill_distance(
    const float x,
    const float y,
    const float left,
    const float top,
    const float right,
    const float bottom,
    const float corner_radius
) {
    const float center_x = 0.5f * (left + right);
    const float center_y = 0.5f * (top + bottom);
    const float half_width = 0.5f * (right - left);
    const float half_height = 0.5f * (bottom - top);
    const float radius = fminf(corner_radius, fminf(half_width, half_height));
    const float q_x = fabsf(x - center_x) - (half_width - radius);
    const float q_y = fabsf(y - center_y) - (half_height - radius);
    const float outside = hypotf(fmaxf(q_x, 0.0f), fmaxf(q_y, 0.0f));
    const float inside = fminf(fmaxf(q_x, q_y), 0.0f);
    return outside + inside - radius;
}

__device__ float pixtreme_ellipse_fill_distance(
    const float x,
    const float y,
    const float center_x,
    const float center_y,
    const float radius_x,
    const float radius_y,
    const float rotation
) {
    const float cosine = cosf(rotation);
    const float sine = sinf(rotation);
    const float offset_x = x - center_x;
    const float offset_y = y - center_y;
    const float local_x = cosine * offset_x - sine * offset_y;
    const float local_y = sine * offset_x + cosine * offset_y;
    const float normalized_x = local_x / radius_x;
    const float normalized_y = local_y / radius_y;
    const float implicit = normalized_x * normalized_x + normalized_y * normalized_y - 1.0f;
    const float gradient_half = hypotf(
        local_x / (radius_x * radius_x),
        local_y / (radius_y * radius_y)
    );
    return gradient_half > 0.0f
        ? implicit / (2.0f * gradient_half)
        : -fminf(radius_x, radius_y);
}

__device__ float pixtreme_polygon_distance(
    const float x,
    const float y,
    const float* __restrict__ points,
    const int point_count
) {
    bool inside = false;
    float distance = 3.402823466e+38f;
    for (int point_index = 0; point_index < point_count; ++point_index) {
        const int next_index = point_index + 1 == point_count ? 0 : point_index + 1;
        const int start_offset = point_index * 2;
        const int end_offset = next_index * 2;
        const float start_x = points[start_offset];
        const float start_y = points[start_offset + 1];
        const float end_x = points[end_offset];
        const float end_y = points[end_offset + 1];
        distance = fminf(
            distance,
            pixtreme_segment_distance(x, y, start_x, start_y, end_x, end_y)
        );
        const bool crosses = (start_y > y) != (end_y > y);
        if (crosses) {
            const float intersection_x =
                start_x + (y - start_y) * (end_x - start_x) / (end_y - start_y);
            if (x < intersection_x) {
                inside = !inside;
            }
        }
    }
    if (distance <= 1.0e-6f) {
        return 0.0f;
    }
    return inside ? -distance : distance;
}

__device__ float pixtreme_shape_distance(
    const float x,
    const float y,
    const float* __restrict__ points,
    const int point_count,
    const int primitive,
    const int closed,
    const int fill,
    const float parameter_0,
    const float parameter_1,
    const float parameter_2,
    const float parameter_3,
    const float parameter_4,
    const float parameter_5
) {
    if (primitive == 0) {
        return pixtreme_segment_distance(
            x,
            y,
            parameter_0,
            parameter_1,
            parameter_2,
            parameter_3
        ) - 0.5f * parameter_4;
    }
    if (primitive == 1) {
        return pixtreme_polyline_distance(
            x,
            y,
            points,
            point_count,
            closed,
            parameter_4
        );
    }
    if (primitive == 2) {
        const float fill_distance = pixtreme_rounded_rectangle_fill_distance(
            x,
            y,
            parameter_0,
            parameter_1,
            parameter_2,
            parameter_3,
            parameter_4
        );
        return fill ? fill_distance : fabsf(fill_distance) - 0.5f * parameter_5;
    }
    if (primitive == 3) {
        const float fill_distance =
            hypotf(x - parameter_0, y - parameter_1) - parameter_2;
        return fill ? fill_distance : fabsf(fill_distance) - 0.5f * parameter_3;
    }
    if (primitive == 4) {
        const float fill_distance = pixtreme_ellipse_fill_distance(
            x,
            y,
            parameter_0,
            parameter_1,
            parameter_2,
            parameter_3,
            parameter_4
        );
        return fill ? fill_distance : fabsf(fill_distance) - 0.5f * parameter_5;
    }
    return pixtreme_polygon_distance(x, y, points, point_count);
}

__device__ float pixtreme_coverage(
    const float x,
    const float y,
    const float* __restrict__ points,
    const int point_count,
    const int primitive,
    const int closed,
    const int fill,
    const int aa,
    const float softness,
    const float parameter_0,
    const float parameter_1,
    const float parameter_2,
    const float parameter_3,
    const float parameter_4,
    const float parameter_5
) {
    if (aa == 2) {
        return pixtreme_shape_distance(
            x,
            y,
            points,
            point_count,
            primitive,
            closed,
            fill,
            parameter_0,
            parameter_1,
            parameter_2,
            parameter_3,
            parameter_4,
            parameter_5
        ) <= 0.0f ? 1.0f : 0.0f;
    }
    if (aa == 0) {
        const float distance = pixtreme_shape_distance(
            x,
            y,
            points,
            point_count,
            primitive,
            closed,
            fill,
            parameter_0,
            parameter_1,
            parameter_2,
            parameter_3,
            parameter_4,
            parameter_5
        );
        return pixtreme_clamp01(0.5f - distance / (1.0f + softness));
    }

    const float offsets[4] = {-0.375f, -0.125f, 0.125f, 0.375f};
    float coverage = 0.0f;
    #pragma unroll
    for (int sample_y = 0; sample_y < 4; ++sample_y) {
        #pragma unroll
        for (int sample_x = 0; sample_x < 4; ++sample_x) {
            const float distance = pixtreme_shape_distance(
                x + offsets[sample_x],
                y + offsets[sample_y],
                points,
                point_count,
                primitive,
                closed,
                fill,
                parameter_0,
                parameter_1,
                parameter_2,
                parameter_3,
                parameter_4,
                parameter_5
            );
            coverage += softness == 0.0f
                ? (distance <= 0.0f ? 1.0f : 0.0f)
                : pixtreme_clamp01(0.5f - distance / softness);
        }
    }
    return coverage * (1.0f / 16.0f);
}

extern "C" __global__ void pixtreme_draw_shape(
    float* __restrict__ output,
    const float* __restrict__ color,
    const float* __restrict__ points,
    const long long image_width,
    const long long channel_count,
    const long long bbox_left,
    const long long bbox_top,
    const long long bbox_width,
    const long long bbox_height,
    const int point_count,
    const int primitive,
    const int closed,
    const int fill,
    const int blend,
    const int aa,
    const float softness,
    const float opacity,
    const float parameter_0,
    const float parameter_1,
    const float parameter_2,
    const float parameter_3,
    const float parameter_4,
    const float parameter_5
) {
    const long long local_x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long local_y = (long long)blockIdx.y * blockDim.y + threadIdx.y;
    if (local_x >= bbox_width || local_y >= bbox_height) {
        return;
    }
    const long long pixel_x = bbox_left + local_x;
    const long long pixel_y = bbox_top + local_y;
    const float coverage = pixtreme_coverage(
        (float)pixel_x + 0.5f,
        (float)pixel_y + 0.5f,
        points,
        point_count,
        primitive,
        closed,
        fill,
        aa,
        softness,
        parameter_0,
        parameter_1,
        parameter_2,
        parameter_3,
        parameter_4,
        parameter_5
    );
    if (coverage <= 0.0f) {
        return;
    }
    const float alpha = coverage * opacity;
    const long long output_offset =
        (pixel_y * image_width + pixel_x) * channel_count;
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


@lru_cache(maxsize=1)
def _draw_kernel() -> cp.RawKernel:
    return cp.RawKernel(_DRAW_KERNEL_SOURCE, "pixtreme_draw_shape")


def _host_array(value: object) -> np.ndarray:
    return _validation._host_array(value, why=_HOST_ARRAY_WHY, how=_HOST_ARRAY_HOW)


def _points(value: object, *, name: str, minimum: int) -> tuple[tuple[float, float], ...]:
    try:
        array = _host_array(value)
    except ValueError:
        array = np.empty((0, 0), dtype=np.float32)
    if array.ndim != 2 or array.shape[1:] != (2,) or array.shape[0] < minimum:
        raise ValueError(
            _actionable_error(
                why=f"{name} must be an N-by-2 point sequence with N at least {minimum}",
                what=f"received {name} shape {array.shape!r}",
                how=f"pass {name} as finite (x, y) pairs with at least {minimum} points",
            )
        )
    return tuple(
        (
            _finite_real(row[0].item() if isinstance(row[0], np.generic) else row[0], name=f"{name}[{index}][0]"),
            _finite_real(row[1].item() if isinstance(row[1], np.generic) else row[1], name=f"{name}[{index}][1]"),
        )
        for index, row in enumerate(array)
    )


def _color(value: object, *, channel_count: int) -> tuple[float, ...]:
    try:
        array = _host_array(value)
    except ValueError:
        array = np.asarray((), dtype=np.float32)
    if array.shape != (channel_count,):
        raise ValueError(
            _actionable_error(
                why="color must have exactly one real value per Frame channel",
                what=f"received color shape {array.shape!r} for {channel_count} channels",
                how=f"pass color as a finite real sequence of length {channel_count}",
            )
        )
    return tuple(
        _finite_real(item.item() if isinstance(item, np.generic) else item, name=f"color[{index}]")
        for index, item in enumerate(array)
    )


def _common(
    frame: object,
    *,
    operation: str,
    color: object,
    opacity: object,
    blend: object,
    aa: object,
    softness: object,
) -> tuple[Frame, tuple[float, ...], float, Blend, Antialiasing, float]:
    checked_frame = _validate_float32_frame(frame, operation=operation)
    checked_color = _color(color, channel_count=len(checked_frame.channels))
    checked_opacity = _bounded_real(
        opacity,
        name="opacity",
        minimum=0.0,
        maximum=1.0,
        why="opacity must be in the closed interval from 0 through 1",
        how="pass a finite real opacity from 0.0 through 1.0",
    )
    checked_blend = _normalized_closed_token(blend, axis="blend", accepted=_BLEND_TOKENS)
    checked_aa = _normalized_closed_token(aa, axis="aa", accepted=_AA_TOKENS)
    checked_softness = _bounded_real(
        softness,
        name="softness",
        minimum=0.0,
        why="softness must be at least 0",
        how="pass a finite nonnegative real number for softness",
    )
    if checked_aa == "off" and checked_softness > 0.0:
        raise ValueError(
            _actionable_error(
                why="aa='off' defines binary coverage and cannot feather an edge",
                what=f"received aa='off' with softness={checked_softness!r}",
                how="pass softness=0 with aa='off', or use aa='distance' or aa='supersample'",
            )
        )
    return checked_frame, checked_color, checked_opacity, checked_blend, checked_aa, checked_softness


def _fill_and_thickness(fill: object, thickness: object) -> tuple[bool, float]:
    fill = _strict_bool(
        fill,
        name="fill",
        why="fill is an explicit boolean region choice",
        how="pass fill=True for the interior or fill=False with a positive thickness for the outline",
    )
    if fill and thickness is not None:
        raise ValueError(
            _actionable_error(
                why="filled shapes have no outline thickness",
                what=f"received fill=True with thickness={thickness!r}",
                how="omit thickness when fill=True",
            )
        )
    if not fill and thickness is None:
        raise ValueError(
            _actionable_error(
                why="outline shapes require an explicit positive thickness",
                what="received fill=False with thickness=None",
                how="pass a positive real thickness, or pass fill=True and omit thickness",
            )
        )
    return fill, 0.0 if fill else _positive_real(thickness, name="thickness")


@lru_cache(maxsize=128)
def _device_values(device_id: int, values: tuple[float, ...]) -> cp.ndarray:
    del device_id
    return cp.asarray(values, dtype=cp.float32)


@lru_cache(maxsize=128)
def _device_points(device_id: int, points: tuple[tuple[float, float], ...]) -> cp.ndarray:
    del device_id
    return cp.asarray(points, dtype=cp.float32).reshape(-1)


def _coverage_margin(*, aa: str, softness: float) -> float:
    return 0.0 if aa == "off" else 0.5 * (1.0 + softness)


def _bbox(
    frame: Frame,
    *,
    minimum_x: float,
    minimum_y: float,
    maximum_x: float,
    maximum_y: float,
) -> tuple[int, int, int, int] | None:
    left = max(0, math.floor(minimum_x - 0.5))
    top = max(0, math.floor(minimum_y - 0.5))
    right = min(frame.width, math.ceil(maximum_x - 0.5) + 1)
    bottom = min(frame.height, math.ceil(maximum_y - 0.5) + 1)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _draw(
    frame: Frame,
    *,
    color: tuple[float, ...],
    points: tuple[tuple[float, float], ...],
    primitive: int,
    closed: bool,
    fill: bool,
    blend: Blend,
    aa: Antialiasing,
    softness: float,
    opacity: float,
    parameters: tuple[float, float, float, float, float, float],
    bounds: tuple[float, float, float, float],
) -> Frame:
    output = frame.data.copy(order="C")
    resolved_bbox = _bbox(
        frame,
        minimum_x=bounds[0],
        minimum_y=bounds[1],
        maximum_x=bounds[2],
        maximum_y=bounds[3],
    )
    if resolved_bbox is None or opacity == 0.0:
        return Frame(
            data=output,
            colorspace=frame.colorspace,
            gamma=frame.gamma,
            channels=frame.channels,
            matrix=frame.matrix,
        )

    left, top, right, bottom = resolved_bbox
    bbox_width = right - left
    bbox_height = bottom - top
    grid = (
        (bbox_width + _DRAW_BLOCK[0] - 1) // _DRAW_BLOCK[0],
        (bbox_height + _DRAW_BLOCK[1] - 1) // _DRAW_BLOCK[1],
    )
    device_id = cp.cuda.runtime.getDevice()
    device_color = _device_values(device_id, color)
    device_points = _device_points(device_id, points) if points else device_color
    _draw_kernel()(
        grid,
        _DRAW_BLOCK,
        (
            output,
            device_color,
            device_points,
            np.int64(frame.width),
            np.int64(len(frame.channels)),
            np.int64(left),
            np.int64(top),
            np.int64(bbox_width),
            np.int64(bbox_height),
            np.int32(len(points)),
            np.int32(primitive),
            np.int32(closed),
            np.int32(fill),
            np.int32(_DRAW_BLEND_CODES[blend]),
            np.int32(_AA_TOKENS.index(aa)),
            np.float32(softness),
            np.float32(opacity),
            *(np.float32(value) for value in parameters),
        ),
    )
    return Frame(
        data=output,
        colorspace=frame.colorspace,
        gamma=frame.gamma,
        channels=frame.channels,
        matrix=frame.matrix,
    )


def line(
    frame: Frame,
    *,
    start: Sequence[float],
    end: Sequence[float],
    color: Sequence[float],
    thickness: float,
    opacity: float = 1.0,
    blend: Blend = "normal",
    aa: Antialiasing = "distance",
    softness: float = 0.0,
) -> Frame:
    """Draw one round-cap line in continuous pixel coordinates.

    Coordinates are ``(x, y)`` and pixel ``(i, j)`` is sampled at
    ``(i + 0.5, j + 0.5)``. Thickness spreads equally around the centerline;
    a zero-length line is a circle of diameter ``thickness``. Coverage, opacity,
    and the selected blend are applied once without clamp, so scene values pass
    through. The input is unchanged and the result always owns a new allocation.
    """
    checked = _common(
        frame,
        operation="draw.line",
        color=color,
        opacity=opacity,
        blend=blend,
        aa=aa,
        softness=softness,
    )
    checked_frame, checked_color, checked_opacity, checked_blend, checked_aa, checked_softness = checked
    checked_start = _finite_pair(start, name="start")
    checked_end = _finite_pair(end, name="end")
    checked_thickness = _positive_real(thickness, name="thickness")
    margin = 0.5 * checked_thickness + _coverage_margin(aa=checked_aa, softness=checked_softness)
    return _draw(
        checked_frame,
        color=checked_color,
        points=(),
        primitive=_PRIMITIVE_LINE,
        closed=False,
        fill=False,
        blend=checked_blend,
        aa=checked_aa,
        softness=checked_softness,
        opacity=checked_opacity,
        parameters=(*checked_start, *checked_end, checked_thickness, 0.0),
        bounds=(
            min(checked_start[0], checked_end[0]) - margin,
            min(checked_start[1], checked_end[1]) - margin,
            max(checked_start[0], checked_end[0]) + margin,
            max(checked_start[1], checked_end[1]) + margin,
        ),
    )


def polyline(
    frame: Frame,
    *,
    points: Sequence[Sequence[float]] | np.ndarray,
    color: Sequence[float],
    thickness: float,
    closed: bool = False,
    opacity: float = 1.0,
    blend: Blend = "normal",
    aa: Antialiasing = "distance",
    softness: float = 0.0,
) -> Frame:
    """Draw a round-cap, round-join polyline as one coverage union.

    Coordinates use ``(x, y)`` with pixel centers at ``(i + 0.5, j + 0.5)``.
    Segment capsules are united before one blend, so joins and self-crossings do
    not composite twice. Thickness is centered, scene values are not clamped,
    and the result always owns a new allocation.
    """
    checked = _common(
        frame,
        operation="draw.polyline",
        color=color,
        opacity=opacity,
        blend=blend,
        aa=aa,
        softness=softness,
    )
    checked_frame, checked_color, checked_opacity, checked_blend, checked_aa, checked_softness = checked
    checked_points = _points(points, name="points", minimum=2)
    checked_thickness = _positive_real(thickness, name="thickness")
    checked_closed = _strict_bool(
        closed,
        name="closed",
        why="closed is an explicit boolean choice",
        how="pass closed=True or closed=False",
    )
    margin = 0.5 * checked_thickness + _coverage_margin(aa=checked_aa, softness=checked_softness)
    xs, ys = zip(*checked_points, strict=True)
    return _draw(
        checked_frame,
        color=checked_color,
        points=checked_points,
        primitive=_PRIMITIVE_POLYLINE,
        closed=checked_closed,
        fill=False,
        blend=checked_blend,
        aa=checked_aa,
        softness=checked_softness,
        opacity=checked_opacity,
        parameters=(0.0, 0.0, 0.0, 0.0, checked_thickness, 0.0),
        bounds=(min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin),
    )


def rectangle(
    frame: Frame,
    *,
    top_left: Sequence[float],
    bottom_right: Sequence[float],
    color: Sequence[float],
    thickness: float | None = None,
    fill: bool = False,
    corner_radius: float = 0.0,
    opacity: float = 1.0,
    blend: Blend = "normal",
    aa: Antialiasing = "distance",
    softness: float = 0.0,
) -> Frame:
    """Draw a continuous-coordinate axis-aligned rectangle.

    ``top_left`` and ``bottom_right`` are ``(x, y)`` boundaries; pixel centers
    are ``(i + 0.5, j + 0.5)``. Corner radius saturates at half the short side.
    ``fill=True`` requires no thickness, while an outline requires positive
    thickness centered on the boundary. Scene values are not clamped and output
    always uses a new allocation.
    """
    checked = _common(
        frame,
        operation="draw.rectangle",
        color=color,
        opacity=opacity,
        blend=blend,
        aa=aa,
        softness=softness,
    )
    checked_frame, checked_color, checked_opacity, checked_blend, checked_aa, checked_softness = checked
    checked_top_left = _finite_pair(top_left, name="top_left")
    checked_bottom_right = _finite_pair(bottom_right, name="bottom_right")
    if checked_top_left[0] >= checked_bottom_right[0] or checked_top_left[1] >= checked_bottom_right[1]:
        raise ValueError(
            _actionable_error(
                why="top_left must be strictly above and left of bottom_right",
                what=f"received top_left={checked_top_left!r}, bottom_right={checked_bottom_right!r}",
                how="pass continuous rectangle bounds with top_left.x < bottom_right.x and top_left.y < bottom_right.y",
            )
        )
    checked_fill, checked_thickness = _fill_and_thickness(fill, thickness)
    checked_corner_radius = _bounded_real(
        corner_radius,
        name="corner_radius",
        minimum=0.0,
        why="corner_radius must be at least 0",
        how="pass a finite nonnegative real number for corner_radius",
    )
    checked_corner_radius = min(
        checked_corner_radius,
        0.5 * (checked_bottom_right[0] - checked_top_left[0]),
        0.5 * (checked_bottom_right[1] - checked_top_left[1]),
    )
    margin = (0.0 if checked_fill else 0.5 * checked_thickness) + _coverage_margin(
        aa=checked_aa,
        softness=checked_softness,
    )
    return _draw(
        checked_frame,
        color=checked_color,
        points=(),
        primitive=_PRIMITIVE_RECTANGLE,
        closed=False,
        fill=checked_fill,
        blend=checked_blend,
        aa=checked_aa,
        softness=checked_softness,
        opacity=checked_opacity,
        parameters=(
            *checked_top_left,
            *checked_bottom_right,
            checked_corner_radius,
            checked_thickness,
        ),
        bounds=(
            checked_top_left[0] - margin,
            checked_top_left[1] - margin,
            checked_bottom_right[0] + margin,
            checked_bottom_right[1] + margin,
        ),
    )


def circle(
    frame: Frame,
    *,
    center: Sequence[float],
    radius: float,
    color: Sequence[float],
    thickness: float | None = None,
    fill: bool = False,
    opacity: float = 1.0,
    blend: Blend = "normal",
    aa: Antialiasing = "distance",
    softness: float = 0.0,
) -> Frame:
    """Draw a filled disk or a centered circular outline.

    The center is ``(x, y)`` in the continuous system whose pixel centers are
    ``(i + 0.5, j + 0.5)``. ``fill`` and ``thickness`` are mutually exclusive.
    Coverage and blend preserve unclamped scene values. The input is unchanged
    and the returned Frame always owns a new allocation.
    """
    checked = _common(
        frame,
        operation="draw.circle",
        color=color,
        opacity=opacity,
        blend=blend,
        aa=aa,
        softness=softness,
    )
    checked_frame, checked_color, checked_opacity, checked_blend, checked_aa, checked_softness = checked
    checked_center = _finite_pair(center, name="center")
    checked_radius = _positive_real(radius, name="radius")
    checked_fill, checked_thickness = _fill_and_thickness(fill, thickness)
    extent = (
        checked_radius
        + (0.0 if checked_fill else 0.5 * checked_thickness)
        + _coverage_margin(aa=checked_aa, softness=checked_softness)
    )
    return _draw(
        checked_frame,
        color=checked_color,
        points=(),
        primitive=_PRIMITIVE_CIRCLE,
        closed=False,
        fill=checked_fill,
        blend=checked_blend,
        aa=checked_aa,
        softness=checked_softness,
        opacity=checked_opacity,
        parameters=(*checked_center, checked_radius, checked_thickness, 0.0, 0.0),
        bounds=(
            checked_center[0] - extent,
            checked_center[1] - extent,
            checked_center[0] + extent,
            checked_center[1] + extent,
        ),
    )


def ellipse(
    frame: Frame,
    *,
    center: Sequence[float],
    radii: Sequence[float],
    rotation: float = 0.0,
    color: Sequence[float],
    thickness: float | None = None,
    fill: bool = False,
    opacity: float = 1.0,
    blend: Blend = "normal",
    aa: Antialiasing = "distance",
    softness: float = 0.0,
) -> Frame:
    """Draw a rotated filled ellipse or an isotropic-width outline.

    Coordinates are ``(x, y)`` with pixel centers at ``(i + 0.5, j + 0.5)``.
    Radii are ``(rx, ry)`` and positive rotation is visually counterclockwise,
    matching ``lens_blur``. ``fill`` excludes ``thickness``; outline thickness
    is centered. Scene values are not clamped and output owns a new allocation.
    """
    checked = _common(
        frame,
        operation="draw.ellipse",
        color=color,
        opacity=opacity,
        blend=blend,
        aa=aa,
        softness=softness,
    )
    checked_frame, checked_color, checked_opacity, checked_blend, checked_aa, checked_softness = checked
    checked_center = _finite_pair(center, name="center")
    checked_radii = _finite_pair(radii, name="radii")
    checked_radius_x = _positive_real(checked_radii[0], name="radii[0]")
    checked_radius_y = _positive_real(checked_radii[1], name="radii[1]")
    checked_rotation = _finite_real(rotation, name="rotation")
    checked_fill, checked_thickness = _fill_and_thickness(fill, thickness)
    radians = math.radians(math.fmod(checked_rotation, 360.0))
    cosine = math.cos(radians)
    sine = math.sin(radians)
    extent_x = math.hypot(checked_radius_x * cosine, checked_radius_y * sine)
    extent_y = math.hypot(checked_radius_x * sine, checked_radius_y * cosine)
    margin = (0.0 if checked_fill else 0.5 * checked_thickness) + _coverage_margin(
        aa=checked_aa,
        softness=checked_softness,
    )
    return _draw(
        checked_frame,
        color=checked_color,
        points=(),
        primitive=_PRIMITIVE_ELLIPSE,
        closed=False,
        fill=checked_fill,
        blend=checked_blend,
        aa=checked_aa,
        softness=checked_softness,
        opacity=checked_opacity,
        parameters=(
            *checked_center,
            checked_radius_x,
            checked_radius_y,
            radians,
            checked_thickness,
        ),
        bounds=(
            checked_center[0] - extent_x - margin,
            checked_center[1] - extent_y - margin,
            checked_center[0] + extent_x + margin,
            checked_center[1] + extent_y + margin,
        ),
    )


def polygon(
    frame: Frame,
    *,
    points: Sequence[Sequence[float]] | np.ndarray,
    color: Sequence[float],
    opacity: float = 1.0,
    blend: Blend = "normal",
    aa: Antialiasing = "distance",
    softness: float = 0.0,
) -> Frame:
    """Fill a polygon with the even-odd rule in continuous coordinates.

    Points are ``(x, y)`` and pixel centers are ``(i + 0.5, j + 0.5)``.
    Concave and self-intersecting paths are accepted; duplicate and collinear
    vertices contribute only zero-area degeneracies. Polygon is fill-only;
    use ``polyline(..., closed=True)`` for thickness. Blend does not clamp
    scene values, and the result always owns a new allocation.
    """
    checked = _common(
        frame,
        operation="draw.polygon",
        color=color,
        opacity=opacity,
        blend=blend,
        aa=aa,
        softness=softness,
    )
    checked_frame, checked_color, checked_opacity, checked_blend, checked_aa, checked_softness = checked
    checked_points = _points(points, name="points", minimum=3)
    margin = _coverage_margin(aa=checked_aa, softness=checked_softness)
    xs, ys = zip(*checked_points, strict=True)
    return _draw(
        checked_frame,
        color=checked_color,
        points=checked_points,
        primitive=_PRIMITIVE_POLYGON,
        closed=True,
        fill=True,
        blend=checked_blend,
        aa=checked_aa,
        softness=checked_softness,
        opacity=checked_opacity,
        parameters=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        bounds=(min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin),
    )
