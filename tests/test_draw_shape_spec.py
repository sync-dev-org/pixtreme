"""Specification, contract, and numerical-property tests for shape drawing."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pytest

import pixtreme as px
import pixtreme._draw.shapes as draw_module

DRAW_NAMES = (
    "line",
    "polyline",
    "rectangle",
    "circle",
    "ellipse",
    "polygon",
)
BLENDS = ("normal", "add", "multiply", "screen")
AAS = ("distance", "supersample", "off")
_SUBPIXEL_OFFSETS = (-0.375, -0.125, 0.125, 0.375)


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


def _zeros(height: int = 7, width: int = 8, channels: Sequence[str] = ("R", "G", "B")) -> px.core.Frame:
    return _frame(np.zeros((height, width, len(channels)), dtype=np.float32), channels=channels)


def _assert_actionable(error: pytest.ExceptionInfo[ValueError]) -> None:
    message = str(error.value)
    assert "why=" in message
    assert "; what=" in message
    assert "; how=" in message


def _draw(name: str) -> Callable[..., px.core.Frame]:
    return getattr(px.draw, name)


def _base_kwargs(name: str) -> dict[str, Any]:
    return {
        "line": {"start": (1.0, 1.0), "end": (4.0, 3.0), "color": (0.1, 0.2, 0.3), "thickness": 1.0},
        "polyline": {
            "points": ((1.0, 1.0), (4.0, 1.0), (4.0, 3.0)),
            "color": (0.1, 0.2, 0.3),
            "thickness": 1.0,
        },
        "rectangle": {
            "top_left": (1.0, 1.0),
            "bottom_right": (5.0, 4.0),
            "color": (0.1, 0.2, 0.3),
            "fill": True,
        },
        "circle": {"center": (3.0, 3.0), "radius": 2.0, "color": (0.1, 0.2, 0.3), "fill": True},
        "ellipse": {
            "center": (3.0, 3.0),
            "radii": (2.0, 1.0),
            "color": (0.1, 0.2, 0.3),
            "fill": True,
        },
        "polygon": {
            "points": ((1.0, 1.0), (5.0, 1.0), (3.0, 4.0)),
            "color": (0.1, 0.2, 0.3),
        },
    }[name]


def _pixel_grid(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[:height, :width].astype(np.float64)
    return x + 0.5, y + 0.5


def _distance_to_segment(
    x: np.ndarray,
    y: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
) -> np.ndarray:
    start_x, start_y = start
    delta_x = end[0] - start_x
    delta_y = end[1] - start_y
    denominator = delta_x * delta_x + delta_y * delta_y
    if denominator == 0.0:
        return np.hypot(x - start_x, y - start_y)
    projection = np.clip(((x - start_x) * delta_x + (y - start_y) * delta_y) / denominator, 0.0, 1.0)
    return np.hypot(x - (start_x + projection * delta_x), y - (start_y + projection * delta_y))


def _line_distance(
    x: np.ndarray,
    y: np.ndarray,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    thickness: float,
) -> np.ndarray:
    return _distance_to_segment(x, y, start, end) - thickness / 2.0


def _polyline_distance(
    x: np.ndarray,
    y: np.ndarray,
    *,
    points: Sequence[tuple[float, float]],
    thickness: float,
    closed: bool,
) -> np.ndarray:
    segments = list(zip(points[:-1], points[1:], strict=True))
    if closed:
        segments.append((points[-1], points[0]))
    return np.minimum.reduce([_distance_to_segment(x, y, start, end) for start, end in segments]) - thickness / 2.0


def _rounded_rectangle_fill_distance(
    x: np.ndarray,
    y: np.ndarray,
    *,
    top_left: tuple[float, float],
    bottom_right: tuple[float, float],
    corner_radius: float,
) -> np.ndarray:
    center_x = (top_left[0] + bottom_right[0]) / 2.0
    center_y = (top_left[1] + bottom_right[1]) / 2.0
    half_width = (bottom_right[0] - top_left[0]) / 2.0
    half_height = (bottom_right[1] - top_left[1]) / 2.0
    radius = min(corner_radius, half_width, half_height)
    q_x = np.abs(x - center_x) - (half_width - radius)
    q_y = np.abs(y - center_y) - (half_height - radius)
    outside = np.hypot(np.maximum(q_x, 0.0), np.maximum(q_y, 0.0))
    inside = np.minimum(np.maximum(q_x, q_y), 0.0)
    return outside + inside - radius


def _rectangle_distance(
    x: np.ndarray,
    y: np.ndarray,
    *,
    top_left: tuple[float, float],
    bottom_right: tuple[float, float],
    corner_radius: float,
    fill: bool,
    thickness: float | None,
) -> np.ndarray:
    fill_distance = _rounded_rectangle_fill_distance(
        x,
        y,
        top_left=top_left,
        bottom_right=bottom_right,
        corner_radius=corner_radius,
    )
    return fill_distance if fill else np.abs(fill_distance) - float(thickness) / 2.0


def _circle_distance(
    x: np.ndarray,
    y: np.ndarray,
    *,
    center: tuple[float, float],
    radius: float,
    fill: bool,
    thickness: float | None,
) -> np.ndarray:
    boundary_distance = np.hypot(x - center[0], y - center[1]) - radius
    return boundary_distance if fill else np.abs(boundary_distance) - float(thickness) / 2.0


def _ellipse_fill_distance(
    x: np.ndarray,
    y: np.ndarray,
    *,
    center: tuple[float, float],
    radii: tuple[float, float],
    rotation: float,
) -> np.ndarray:
    radians = math.radians(math.fmod(rotation, 360.0))
    cosine = math.cos(radians)
    sine = math.sin(radians)
    offset_x = x - center[0]
    offset_y = y - center[1]
    local_x = cosine * offset_x - sine * offset_y
    local_y = sine * offset_x + cosine * offset_y
    radius_x, radius_y = radii
    implicit = (local_x / radius_x) ** 2 + (local_y / radius_y) ** 2 - 1.0
    gradient_half = np.hypot(local_x / (radius_x * radius_x), local_y / (radius_y * radius_y))
    return np.divide(
        implicit,
        2.0 * gradient_half,
        out=np.full_like(implicit, -min(radii)),
        where=gradient_half > 0.0,
    )


def _ellipse_distance(
    x: np.ndarray,
    y: np.ndarray,
    *,
    center: tuple[float, float],
    radii: tuple[float, float],
    rotation: float,
    fill: bool,
    thickness: float | None,
) -> np.ndarray:
    fill_distance = _ellipse_fill_distance(x, y, center=center, radii=radii, rotation=rotation)
    return fill_distance if fill else np.abs(fill_distance) - float(thickness) / 2.0


def _polygon_distance(
    x: np.ndarray,
    y: np.ndarray,
    *,
    points: Sequence[tuple[float, float]],
) -> np.ndarray:
    inside = np.zeros(x.shape, dtype=np.bool_)
    edge_distances: list[np.ndarray] = []
    for start, end in zip(points, (*points[1:], points[0]), strict=True):
        edge_distances.append(_distance_to_segment(x, y, start, end))
        crosses = (start[1] > y) != (end[1] > y)
        denominator = end[1] - start[1]
        intersection_x = np.where(
            crosses,
            start[0] + (y - start[1]) * (end[0] - start[0]) / np.where(denominator == 0.0, 1.0, denominator),
            0.0,
        )
        inside ^= crosses & (x < intersection_x)
    distance = np.minimum.reduce(edge_distances)
    return np.where(inside, -distance, distance)


def _coverage_at(
    signed_distance: np.ndarray,
    *,
    aa: str,
    softness: float,
) -> np.ndarray:
    if aa == "off":
        return (signed_distance <= 0.0).astype(np.float64)
    if aa == "distance":
        transition_width = 1.0 + softness
        return np.clip(0.5 - signed_distance / transition_width, 0.0, 1.0)
    raise AssertionError("supersample coverage is evaluated from sample distances")


def _supersample_coverage(
    distance_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    height: int,
    width: int,
    softness: float,
) -> np.ndarray:
    coverage = np.zeros((height, width), dtype=np.float64)
    x_center, y_center = _pixel_grid(height, width)
    for offset_y in _SUBPIXEL_OFFSETS:
        for offset_x in _SUBPIXEL_OFFSETS:
            distance = distance_function(x_center + offset_x, y_center + offset_y)
            if softness == 0.0:
                coverage += distance <= 0.0
            else:
                coverage += np.clip(0.5 - distance / softness, 0.0, 1.0)
    return coverage / 16.0


def _blend_reference(
    destination: np.ndarray,
    *,
    color: Sequence[float],
    coverage: np.ndarray,
    opacity: float,
    blend: str,
) -> np.ndarray:
    destination64 = destination.astype(np.float64)
    color64 = np.asarray(color, dtype=np.float64)
    alpha = coverage[..., np.newaxis] * opacity
    if blend == "normal":
        blend_value = np.broadcast_to(color64, destination64.shape)
    elif blend == "add":
        blend_value = destination64 + color64
    elif blend == "multiply":
        blend_value = destination64 * color64
    elif blend == "screen":
        blend_value = 1.0 - (1.0 - destination64) * (1.0 - color64)
    else:
        raise AssertionError(blend)
    return (destination64 * (1.0 - alpha) + blend_value * alpha).astype(np.float32)


def _host(result: px.core.Frame) -> np.ndarray:
    return px.io.to_array(
        result,
    ).get()


def test_draw_shape_public_signatures_and_frame_only_entries_are_actionable() -> None:
    """v1-draw-shape acceptance 1-2: six exact keyword APIs are public, Frame-only, and return Frame."""
    import cupy as cp

    expected_parameters = {
        "line": ("frame", "start", "end", "color", "thickness", "opacity", "blend", "aa", "softness"),
        "polyline": (
            "frame",
            "points",
            "color",
            "thickness",
            "closed",
            "opacity",
            "blend",
            "aa",
            "softness",
        ),
        "rectangle": (
            "frame",
            "top_left",
            "bottom_right",
            "color",
            "thickness",
            "fill",
            "corner_radius",
            "opacity",
            "blend",
            "aa",
            "softness",
        ),
        "circle": (
            "frame",
            "center",
            "radius",
            "color",
            "thickness",
            "fill",
            "opacity",
            "blend",
            "aa",
            "softness",
        ),
        "ellipse": (
            "frame",
            "center",
            "radii",
            "rotation",
            "color",
            "thickness",
            "fill",
            "opacity",
            "blend",
            "aa",
            "softness",
        ),
        "polygon": ("frame", "points", "color", "opacity", "blend", "aa", "softness"),
    }
    expected_defaults = {
        "line": {"opacity": 1.0, "blend": "normal", "aa": "distance", "softness": 0.0},
        "polyline": {
            "closed": False,
            "opacity": 1.0,
            "blend": "normal",
            "aa": "distance",
            "softness": 0.0,
        },
        "rectangle": {
            "thickness": None,
            "fill": False,
            "corner_radius": 0.0,
            "opacity": 1.0,
            "blend": "normal",
            "aa": "distance",
            "softness": 0.0,
        },
        "circle": {
            "thickness": None,
            "fill": False,
            "opacity": 1.0,
            "blend": "normal",
            "aa": "distance",
            "softness": 0.0,
        },
        "ellipse": {
            "rotation": 0.0,
            "thickness": None,
            "fill": False,
            "opacity": 1.0,
            "blend": "normal",
            "aa": "distance",
            "softness": 0.0,
        },
        "polygon": {"opacity": 1.0, "blend": "normal", "aa": "distance", "softness": 0.0},
    }

    for name in DRAW_NAMES:
        operation = _draw(name)
        signature = inspect.signature(operation)
        assert tuple(signature.parameters) == expected_parameters[name]
        assert signature.parameters["frame"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        for parameter in tuple(signature.parameters)[1:]:
            assert signature.parameters[parameter].kind is inspect.Parameter.KEYWORD_ONLY
            if parameter in expected_defaults[name]:
                assert signature.parameters[parameter].default == expected_defaults[name][parameter]
        assert name in px.draw.__all__
        result = operation(_zeros(), **_base_kwargs(name))
        assert isinstance(result, px.core.Frame)

        with pytest.raises(ValueError) as error:
            operation(cp.zeros((7, 8, 3), dtype=cp.float32), **_base_kwargs(name))
        _assert_actionable(error)


def test_draw_host_array_conversion_failure_is_actionable() -> None:
    """REQ-API-012: draw host-array conversion reports the rejected value and a concrete recovery."""
    value = ((1.0,), (1.0, 2.0))
    with pytest.raises(ValueError) as error:
        draw_module._host_array(value)
    _assert_actionable(error)
    assert repr(value) in str(error.value)


@pytest.mark.parametrize("name", DRAW_NAMES)
def test_draw_shape_color_validation_and_scene_values(name: str) -> None:
    """v1-draw-shape acceptance 3: color matches channels, is finite-real, and remains unclamped."""
    operation = _draw(name)
    source = _zeros()
    for invalid_color in ((1.0, 2.0), (1.0, 2.0, 3.0, 4.0), (1.0, float("nan"), 3.0), 1.0):
        with pytest.raises(ValueError) as error:
            operation(source, **(_base_kwargs(name) | {"color": invalid_color}))
        _assert_actionable(error)

    scene_source = _frame(np.full((3, 3, 1), 2.5, dtype=np.float32), channels=("signal",))
    result = _draw("rectangle")(
        scene_source,
        top_left=(-1.0, -1.0),
        bottom_right=(4.0, 4.0),
        color=(-3.0,),
        fill=True,
        aa="off",
    )
    np.testing.assert_array_equal(_host(result), np.full((3, 3, 1), -3.0, dtype=np.float32))


@pytest.mark.parametrize(
    ("name", "parameter", "invalid"),
    (
        ("line", "start", (float("nan"), 0.0)),
        ("line", "end", (0.0, float("inf"))),
        ("polyline", "points", ((0.0, 0.0), (1.0, float("-inf")))),
        ("rectangle", "top_left", (0.0, float("nan"))),
        ("rectangle", "bottom_right", (float("inf"), 2.0)),
        ("rectangle", "corner_radius", -0.1),
        ("circle", "center", (0.0, float("nan"))),
        ("circle", "radius", 0.0),
        ("circle", "radius", -1.0),
        ("ellipse", "radii", (2.0, 0.0)),
        ("ellipse", "radii", (-1.0, 2.0)),
        ("ellipse", "rotation", float("inf")),
        ("polygon", "softness", -0.1),
    ),
)
def test_draw_shape_coordinates_and_dimensions_reject_invalid_values(
    name: str, parameter: str, invalid: object
) -> None:
    """v1-draw-shape acceptance 4-5: geometry values are finite reals with signed-domain validation."""
    with pytest.raises(ValueError) as error:
        _draw(name)(_zeros(), **(_base_kwargs(name) | {parameter: invalid}))
    _assert_actionable(error)


@pytest.mark.parametrize("name", DRAW_NAMES)
def test_draw_shape_accepts_integer_float_and_subpixel_geometry(name: str) -> None:
    """v1-draw-shape acceptance 4: geometry accepts int and float coordinates and preserves subpixel motion."""
    source = _zeros(height=5, width=7, channels=("signal",))
    base = _base_kwargs(name) | {"color": (1.0,)}
    first = _host(_draw(name)(source, **base))

    shifted = dict(base)
    if name == "line":
        shifted["start"] = (1.25, 1)
        shifted["end"] = (4.25, 3)
    elif name in {"polyline", "polygon"}:
        shifted["points"] = tuple((x + 0.25, y) for x, y in shifted["points"])
    elif name == "rectangle":
        shifted["top_left"] = (1.25, 1)
        shifted["bottom_right"] = (5.25, 4)
    else:
        shifted["center"] = (3.25, 3)
    second = _host(_draw(name)(source, **shifted))
    assert not np.array_equal(first, second)


@pytest.mark.parametrize(
    ("name", "points"),
    (
        ("polyline", ((0.0, 0.0),)),
        ("polygon", ((0.0, 0.0), (1.0, 1.0))),
        ("polyline", ((0.0, 0.0, 1.0), (1.0, 1.0, 1.0))),
        ("polygon", np.zeros((3, 3), dtype=np.float32)),
    ),
)
def test_draw_shape_points_validate_count_and_n_by_two_shape(name: str, points: object) -> None:
    """v1-draw-shape acceptance 6: point inputs require finite (x,y) pairs and primitive-specific minima."""
    with pytest.raises(ValueError) as error:
        _draw(name)(_zeros(), **(_base_kwargs(name) | {"points": points}))
    _assert_actionable(error)


def test_draw_shape_accepts_sequence_and_ndarray_points() -> None:
    """v1-draw-shape acceptance 6: point sequences and N-by-2 ndarrays describe the same geometry."""
    source = _zeros(channels=("signal",))
    points = ((1.0, 1.0), (5.0, 1.0), (3.0, 5.0))
    sequence = _host(_draw("polygon")(source, points=points, color=(1.0,), aa="off"))
    array = _host(_draw("polygon")(source, points=np.asarray(points), color=(1.0,), aa="off"))
    np.testing.assert_array_equal(array, sequence)


@pytest.mark.parametrize("opacity", (-0.1, 1.1, float("nan"), float("inf"), True, "1"))
def test_draw_shape_opacity_is_a_finite_unit_interval_real(opacity: object) -> None:
    """v1-draw-shape acceptance 7: opacity is a finite non-bool real in the closed unit interval."""
    with pytest.raises(ValueError) as error:
        _draw("circle")(_zeros(), **(_base_kwargs("circle") | {"opacity": opacity}))
    _assert_actionable(error)


@pytest.mark.parametrize(("axis", "token", "accepted"), (("blend", "over", BLENDS), ("aa", "nearest", AAS)))
def test_draw_shape_tokens_fail_fast_with_the_accepted_vocabulary(
    axis: str, token: str, accepted: tuple[str, ...]
) -> None:
    """v1-draw-shape acceptance 8-9 / v1-composite acceptance 15: over is rejected and recovery is listed."""
    with pytest.raises(ValueError) as error:
        _draw("circle")(_zeros(), **(_base_kwargs("circle") | {axis: token}))
    _assert_actionable(error)
    assert all(candidate in str(error.value) for candidate in accepted)


@pytest.mark.parametrize("name", ("rectangle", "circle", "ellipse"))
@pytest.mark.parametrize(
    "overrides",
    (
        {"fill": False, "thickness": None},
        {"fill": True, "thickness": 1.0},
        {"fill": "yes"},
        {"fill": False, "thickness": 0.0},
    ),
)
def test_draw_shape_fill_and_thickness_are_explicit_and_exclusive(name: str, overrides: dict[str, object]) -> None:
    """v1-draw-shape acceptance 5 and 10: fill and positive thickness select exactly one region mode."""
    with pytest.raises(ValueError) as error:
        _draw(name)(_zeros(), **(_base_kwargs(name) | overrides))
    _assert_actionable(error)


def test_draw_shape_off_rejects_softness() -> None:
    """v1-draw-shape acceptance 11: binary aa=off cannot be combined with edge softness."""
    with pytest.raises(ValueError) as error:
        _draw("circle")(_zeros(), **(_base_kwargs("circle") | {"aa": "off", "softness": 1.0}))
    _assert_actionable(error)


@pytest.mark.parametrize("aa", AAS)
def test_draw_line_matches_pixel_center_capsule_oracle(aa: str) -> None:
    """v1-draw-shape acceptance 12, 14-15, and 23-25: line coverage follows pixel centers and round capsules."""
    source_values = np.full((6, 8, 1), 0.25, dtype=np.float32)
    source = _frame(source_values, channels=("signal",))
    start = (1.25, 2.0)
    end = (5.75, 3.0)
    thickness = 2.0
    color = (1.5,)
    opacity = 0.7

    def distance_function(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return _line_distance(x, y, start=start, end=end, thickness=thickness)

    if aa == "supersample":
        coverage = _supersample_coverage(distance_function, height=6, width=8, softness=0.0)
    else:
        x, y = _pixel_grid(6, 8)
        coverage = _coverage_at(distance_function(x, y), aa=aa, softness=0.0)
    expected = _blend_reference(source_values, color=color, coverage=coverage, opacity=opacity, blend="normal")

    result = _draw("line")(
        source,
        start=start,
        end=end,
        color=color,
        thickness=thickness,
        opacity=opacity,
        aa=aa,
    )
    np.testing.assert_allclose(_host(result), expected, rtol=2e-6, atol=2e-6)
    if aa == "off":
        assert set(np.unique(_host(result))) <= {np.float32(0.25), np.float32(1.125)}
    if aa == "supersample":
        scaled = (coverage * 16.0).round()
        np.testing.assert_allclose(coverage * 16.0, scaled, atol=1e-12)


def test_draw_line_zero_length_is_a_round_cap_circle_and_thickness_is_symmetric() -> None:
    """v1-draw-shape acceptance 14-15: a zero-length line is a diameter-thickness circle with symmetric spread."""
    source = _zeros(height=9, width=9, channels=("signal",))
    line = _host(
        _draw("line")(
            source,
            start=(4.5, 4.5),
            end=(4.5, 4.5),
            color=(1.0,),
            thickness=4.0,
            aa="off",
        )
    )
    circle = _host(_draw("circle")(source, center=(4.5, 4.5), radius=2.0, color=(1.0,), fill=True, aa="off"))
    np.testing.assert_array_equal(line, circle)
    np.testing.assert_array_equal(line, line[::-1])
    np.testing.assert_array_equal(line, line[:, ::-1])


def test_draw_polyline_unifies_round_segment_coverage_before_one_blend() -> None:
    """v1-draw-shape acceptance 15-16: round segment union is composited once at joins and self-intersections."""
    source_values = np.full((7, 7, 1), 0.2, dtype=np.float32)
    source = _frame(source_values, channels=("signal",))
    points = ((1.0, 1.0), (5.0, 5.0), (1.0, 5.0), (5.0, 1.0))
    x, y = _pixel_grid(7, 7)
    distance = _polyline_distance(x, y, points=points, thickness=2.0, closed=False)
    coverage = _coverage_at(distance, aa="distance", softness=0.0)
    expected = _blend_reference(source_values, color=(0.6,), coverage=coverage, opacity=0.5, blend="add")

    result = _draw("polyline")(
        source,
        points=points,
        color=(0.6,),
        thickness=2.0,
        opacity=0.5,
        blend="add",
    )
    np.testing.assert_allclose(_host(result), expected, rtol=2e-6, atol=2e-6)


def test_draw_rectangle_geometry_corner_saturation_and_outside_clip() -> None:
    """v1-draw-shape acceptance 13 and 17-18: rectangles use continuous bounds, rounded saturation, and image clip."""
    source_values = np.arange(6 * 7, dtype=np.float32).reshape(6, 7, 1) / 10.0
    source = _frame(source_values, channels=("signal",))
    with pytest.raises(ValueError) as error:
        _draw("rectangle")(
            source,
            top_left=(3.0, 1.0),
            bottom_right=(3.0, 4.0),
            color=(1.0,),
            fill=True,
        )
    _assert_actionable(error)

    saturated = _draw("rectangle")(
        source,
        top_left=(-2.0, 1.0),
        bottom_right=(4.0, 5.0),
        color=(1.0,),
        fill=True,
        corner_radius=99.0,
        aa="off",
    )
    clamped = _draw("rectangle")(
        source,
        top_left=(-2.0, 1.0),
        bottom_right=(4.0, 5.0),
        color=(1.0,),
        fill=True,
        corner_radius=2.0,
        aa="off",
    )
    np.testing.assert_array_equal(_host(saturated), _host(clamped))

    outside = _draw("rectangle")(
        source,
        top_left=(-20.0, -20.0),
        bottom_right=(-10.0, -10.0),
        color=(1.0,),
        fill=True,
    )
    np.testing.assert_array_equal(_host(outside), source_values)
    assert outside is not source
    assert outside.data.data.ptr != source.data.data.ptr


@pytest.mark.parametrize("fill", (False, True))
def test_draw_rectangle_matches_rounded_box_distance_oracle(fill: bool) -> None:
    """v1-draw-shape acceptance 17-18 and 23: rectangle fill and centered outline follow rounded-box distance."""
    source_values = np.full((7, 8, 1), 0.1, dtype=np.float32)
    source = _frame(source_values, channels=("signal",))
    kwargs = {
        "top_left": (1.25, 1.0),
        "bottom_right": (6.0, 5.5),
        "corner_radius": 1.25,
        "fill": fill,
        "thickness": None if fill else 1.5,
    }
    x, y = _pixel_grid(7, 8)
    distance = _rectangle_distance(x, y, **kwargs)
    coverage = _coverage_at(distance, aa="distance", softness=0.0)
    expected = _blend_reference(source_values, color=(0.9,), coverage=coverage, opacity=0.8, blend="normal")
    result = _draw("rectangle")(source, color=(0.9,), opacity=0.8, **kwargs)
    np.testing.assert_allclose(_host(result), expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("blend", BLENDS)
def test_draw_circle_matches_fp32_blend_equations_without_clamp(blend: str) -> None:
    """v1-draw-shape acceptance 27-28 and 30: all blend B equations use fp32 alpha composition without clamp."""
    source_values = np.asarray(
        [
            [[-0.5, 0.25, 1.5], [0.0, 0.5, 2.0]],
            [[1.25, -1.0, 0.75], [2.0, 0.1, -0.25]],
        ],
        dtype=np.float32,
    )
    source = _frame(source_values)
    color = (-2.0, 1.5, 0.25)
    x, y = _pixel_grid(2, 2)
    distance = _circle_distance(x, y, center=(1.0, 1.0), radius=0.8, fill=True, thickness=None)
    coverage = _coverage_at(distance, aa="distance", softness=0.0)
    expected = _blend_reference(source_values, color=color, coverage=coverage, opacity=0.6, blend=blend)
    result = _draw("circle")(
        source,
        center=(1.0, 1.0),
        radius=0.8,
        color=color,
        fill=True,
        opacity=0.6,
        blend=blend,
    )
    assert result.data.dtype.name == "float32"
    np.testing.assert_allclose(_host(result), expected, rtol=2e-6, atol=2e-6)


def test_draw_ellipse_rotation_and_uniform_outline_width_match_distance_oracle() -> None:
    """v1-draw-shape acceptance 19-20: ellipse uses lens rotation direction and an isotropic outline distance."""
    source = _zeros(height=13, width=13, channels=("signal",))
    kwargs = {
        "center": (6.5, 6.5),
        "radii": (4.0, 2.0),
        "rotation": 30.0,
        "color": (1.0,),
        "thickness": 1.5,
        "fill": False,
    }
    x, y = _pixel_grid(13, 13)
    distance = _ellipse_distance(
        x,
        y,
        center=kwargs["center"],
        radii=kwargs["radii"],
        rotation=kwargs["rotation"],
        fill=False,
        thickness=kwargs["thickness"],
    )
    coverage = _coverage_at(distance, aa="distance", softness=0.0)
    expected = _blend_reference(
        np.zeros((13, 13, 1), dtype=np.float32), color=(1.0,), coverage=coverage, opacity=1.0, blend="normal"
    )
    result = _draw("ellipse")(source, **kwargs)
    np.testing.assert_allclose(_host(result), expected, rtol=3e-5, atol=3e-5)

    horizontal = _host(
        _draw("ellipse")(
            source,
            center=(6.5, 6.5),
            radii=(4.0, 2.0),
            rotation=0.0,
            color=(1.0,),
            fill=True,
            aa="off",
        )
    )
    vertical = _host(
        _draw("ellipse")(
            source,
            center=(6.5, 6.5),
            radii=(4.0, 2.0),
            rotation=90.0,
            color=(1.0,),
            fill=True,
            aa="off",
        )
    )
    np.testing.assert_array_equal(vertical, np.rot90(horizontal, 1))


def test_draw_polygon_even_odd_accepts_concave_self_intersecting_and_degenerate_vertices() -> None:
    """v1-draw-shape acceptance 21: polygon fill is even-odd for concave/self-crossing paths with degenerate vertices."""
    source_values = np.full((8, 8, 1), 0.2, dtype=np.float32)
    source = _frame(source_values, channels=("signal",))
    points = ((1.0, 1.0), (6.0, 6.0), (1.0, 6.0), (6.0, 1.0), (6.0, 1.0), (1.0, 1.0))
    x, y = _pixel_grid(8, 8)
    distance = _polygon_distance(x, y, points=points)
    coverage = _coverage_at(distance, aa="off", softness=0.0)
    expected = _blend_reference(source_values, color=(0.8,), coverage=coverage, opacity=0.75, blend="normal")
    result = _draw("polygon")(
        source,
        points=points,
        color=(0.8,),
        opacity=0.75,
        aa="off",
    )
    np.testing.assert_allclose(_host(result), expected, rtol=2e-6, atol=2e-6)


def test_draw_calls_composite_sequentially_in_call_order() -> None:
    """v1-draw-shape acceptance 22: each call completes one primitive blend and later calls consume that result."""
    source = _zeros(height=5, width=5, channels=("signal",))
    red = _draw("circle")(
        source,
        center=(2.5, 2.5),
        radius=2.0,
        color=(0.8,),
        fill=True,
        opacity=0.5,
        aa="off",
    )
    red_then_blue = _draw("rectangle")(
        red,
        top_left=(1.0, 1.0),
        bottom_right=(4.0, 4.0),
        color=(0.2,),
        fill=True,
        opacity=0.5,
        aa="off",
    )
    blue = _draw("rectangle")(
        source,
        top_left=(1.0, 1.0),
        bottom_right=(4.0, 4.0),
        color=(0.2,),
        fill=True,
        opacity=0.5,
        aa="off",
    )
    blue_then_red = _draw("circle")(
        blue,
        center=(2.5, 2.5),
        radius=2.0,
        color=(0.8,),
        fill=True,
        opacity=0.5,
        aa="off",
    )
    assert not np.array_equal(_host(red_then_blue), _host(blue_then_red))


def test_draw_distance_and_softness_have_continuous_monotone_edge_coverage() -> None:
    """v1-draw-shape acceptance 23 and 26: distance AA is continuous and softness widens its centered transition."""
    source = _zeros(height=1, width=9, channels=("signal",))
    hard = _host(
        _draw("rectangle")(
            source,
            top_left=(4.0, -1.0),
            bottom_right=(20.0, 2.0),
            color=(1.0,),
            fill=True,
            aa="distance",
            softness=0.0,
        )
    )[0, :, 0]
    soft = _host(
        _draw("rectangle")(
            source,
            top_left=(4.0, -1.0),
            bottom_right=(20.0, 2.0),
            color=(1.0,),
            fill=True,
            aa="distance",
            softness=3.0,
        )
    )[0, :, 0]
    assert np.all(np.diff(hard) >= 0.0)
    assert np.all(np.diff(soft) >= 0.0)
    assert np.count_nonzero((soft > 0.0) & (soft < 1.0)) > np.count_nonzero((hard > 0.0) & (hard < 1.0))
    assert hard[0] == 0.0
    assert hard[-1] == 1.0


def test_draw_supersample_softness_matches_per_sample_feather_oracle() -> None:
    """v1-draw-shape acceptance 24 and 26: supersample averages the fixed 4x4 grid and feathers each sample."""
    source_values = np.zeros((5, 6, 1), dtype=np.float32)
    source = _frame(source_values, channels=("signal",))

    def distance_function(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return _circle_distance(
            x,
            y,
            center=(2.7, 2.4),
            radius=1.6,
            fill=True,
            thickness=None,
        )

    coverage = _supersample_coverage(distance_function, height=5, width=6, softness=1.25)
    expected = _blend_reference(source_values, color=(1.0,), coverage=coverage, opacity=1.0, blend="normal")
    result = _draw("circle")(
        source,
        center=(2.7, 2.4),
        radius=1.6,
        color=(1.0,),
        fill=True,
        aa="supersample",
        softness=1.25,
    )
    np.testing.assert_allclose(_host(result), expected, rtol=2e-6, atol=2e-6)


def test_draw_preserves_metadata_and_always_allocates_private_output() -> None:
    """v1-draw-shape acceptance 13 and 29: metadata is unchanged and every result owns new data, including no-op opacity."""
    source_values = np.arange(4 * 5 * 2, dtype=np.float32).reshape(4, 5, 2)
    source = _frame(source_values, colorspace="S-Gamut3", gamma="s-log3", channels=("depth", "matte"))
    result = _draw("circle")(
        source,
        center=(2.0, 2.0),
        radius=1.0,
        color=(10.0, -10.0),
        fill=True,
        opacity=0.0,
    )
    assert result is not source
    assert result.data.data.ptr != source.data.data.ptr
    assert (result.width, result.height, result.colorspace, result.gamma, result.channels) == (
        source.width,
        source.height,
        source.colorspace,
        source.gamma,
        source.channels,
    )
    np.testing.assert_array_equal(_host(result), source_values)


def test_draw_arbitrary_channel_labels_and_one_channel_matte_are_numeric_only() -> None:
    """v1-draw-shape acceptance 31-32: arbitrary channel labels are inert and one-channel AA/off masks work."""
    labels = ("normal.x", "normal.y", "depth", "id", "custom")
    source = _zeros(height=5, width=5, channels=labels)
    color = (-1.0, 0.0, 1.0, 2.0, 3.0)
    result = _draw("circle")(
        source,
        center=(2.5, 2.5),
        radius=1.5,
        color=color,
        fill=True,
        aa="off",
    )
    assert result.channels == labels
    center = _host(result)[2, 2]
    np.testing.assert_array_equal(center, np.asarray(color, dtype=np.float32))

    matte = _zeros(height=5, width=5, channels=("matte",))
    binary = _host(_draw("circle")(matte, center=(2.5, 2.5), radius=1.5, color=(1.0,), fill=True, aa="off"))
    antialiased = _host(_draw("circle")(matte, center=(2.5, 2.5), radius=1.5, color=(1.0,), fill=True, aa="distance"))
    assert set(np.unique(binary)) <= {np.float32(0.0), np.float32(1.0)}
    assert np.any((antialiased > 0.0) & (antialiased < 1.0))


def test_draw_vocabulary_documents_blend_aa_softness_and_continuous_coordinates(
    vocabulary_markdown: str,
) -> None:
    """v1-draw-shape acceptance 33 / v1-composite acceptance 15, 18: vocabulary fixes shared blend semantics."""
    markdown = vocabulary_markdown
    for required in (
        "## blend",
        "`normal`",
        "`add`",
        "`multiply`",
        "`screen`",
        "default",
        "## aa",
        "`distance`",
        "`supersample`",
        "`off`",
        "softness",
        "4×4",
        "(x, y)",
        "i + 0.5",
        "j + 0.5",
    ):
        assert required in markdown


def test_draw_docstrings_state_the_llm_readable_geometry_and_ownership_contracts() -> None:
    """v1-draw-shape acceptance 34: public docstrings state non-obvious geometry, value, mode, and ownership rules."""
    combined = "\n".join(inspect.getdoc(_draw(name)) or "" for name in DRAW_NAMES)
    for required in (
        "round",
        "even-odd",
        "scene",
        "clamp",
        "(x, y)",
        "i + 0.5",
        "fill",
        "thickness",
        "new",
        "allocation",
    ):
        assert required in combined


def test_draw_uses_one_full_copy_then_one_bbox_raw_kernel_path() -> None:
    """v1-draw-shape acceptance 1: structural contract fixes full copy plus bbox RawKernel composition."""
    import pixtreme._draw.shapes as draw_module

    draw_source = inspect.getsource(draw_module._draw)
    kernel_factory_source = inspect.getsource(draw_module._draw_kernel)
    assert "frame.data.copy" in draw_source
    assert "_bbox(" in draw_source
    assert "_draw_kernel()(" in draw_source
    assert "cp.RawKernel" in kernel_factory_source
    assert "ElementwiseKernel" not in kernel_factory_source
