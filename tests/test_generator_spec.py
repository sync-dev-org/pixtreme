"""Specification, contract, and numerical-property tests for image generators."""

from __future__ import annotations

import inspect
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

import pixtreme as px
import pixtreme._generate.patterns as generate_module

GENERATOR_NAMES = (
    "ramp",
    "grid",
    "checkerboard",
    "color_bars",
)
KINDS = ("linear", "radial")
AAS = ("distance", "supersample", "off")
STANDARDS = (
    "arib-std-b28",
    "smpte-rp219",
    "bt2111-hlg",
    "bt2111-pq",
    "bt2111-pq-full",
    "full-100",
    "full-75",
)
OUTPUTS = ("normalized", "code")
_SUBPIXEL_OFFSETS = (-0.375, -0.125, 0.125, 0.375)


def _host(frame: px.core.Frame) -> np.ndarray:
    import cupy as cp

    return cp.asnumpy(frame.data)


def _assert_actionable(error: pytest.ExceptionInfo[ValueError]) -> None:
    message = str(error.value)
    assert "why=" in message
    assert "; what=" in message
    assert "; how=" in message


def test_generator_host_array_conversion_failure_is_actionable() -> None:
    """REQ-API-012: generator host-array conversion reports the rejected value and a concrete recovery."""
    value = ((1.0,), (1.0, 2.0))
    with pytest.raises(ValueError) as error:
        generate_module._host_array(value)
    _assert_actionable(error)
    assert repr(value) in str(error.value)


def _base_kwargs(name: str) -> dict[str, Any]:
    return {
        "ramp": {
            "width": 8,
            "height": 6,
            "start": (1.0, 1.0),
            "end": (7.0, 5.0),
            "start_color": (0.0, 0.25, -0.5),
            "end_color": (1.0, 1.5, 2.0),
            "colorspace": "ACEScg",
        },
        "grid": {
            "width": 8,
            "height": 6,
            "cell": (3.0, 2.0),
            "line_width": 0.75,
            "color": (1.0, 0.25, -0.5),
            "background": (0.0, 0.5, 2.0),
            "colorspace": "ACEScg",
        },
        "checkerboard": {
            "width": 8,
            "height": 6,
            "cell": (3.0, 2.0),
            "colors": ((1.0, 0.25, -0.5), (0.0, 0.5, 2.0)),
            "colorspace": "ACEScg",
        },
        "color_bars": {
            "width": 8,
            "height": 6,
            "standard": "full-100",
        },
    }[name]


def _pixel_grid(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[:height, :width].astype(np.float32)
    return x + np.float32(0.5), y + np.float32(0.5)


def _ramp_reference(
    *,
    width: int,
    height: int,
    kind: str,
    start: tuple[float, float],
    end: tuple[float, float],
    start_color: Sequence[float],
    end_color: Sequence[float],
) -> np.ndarray:
    x, y = _pixel_grid(height, width)
    start_x, start_y = (np.float32(value) for value in start)
    delta_x = np.float32(end[0] - start[0])
    delta_y = np.float32(end[1] - start[1])
    if kind == "linear":
        denominator = np.float32(delta_x * delta_x + delta_y * delta_y)
        t = ((x - start_x) * delta_x + (y - start_y) * delta_y) / denominator
    else:
        radius = np.sqrt(np.float32(delta_x * delta_x + delta_y * delta_y))
        t = np.sqrt((x - start_x) ** 2 + (y - start_y) ** 2) / radius
    t = np.clip(t, np.float32(0.0), np.float32(1.0)).astype(np.float32)
    first = np.asarray(start_color, dtype=np.float32)
    second = np.asarray(end_color, dtype=np.float32)
    return (first[None, None, :] * (np.float32(1.0) - t[..., None]) + second[None, None, :] * t[..., None]).astype(
        np.float32
    )


def _periodic_line_distance(
    x: np.ndarray,
    y: np.ndarray,
    *,
    cell: tuple[float, float],
    line_width: float,
    offset: tuple[float, float],
) -> np.ndarray:
    cell_x, cell_y = (np.float32(value) for value in cell)
    offset_x, offset_y = (np.float32(value) for value in offset)
    phase_x = np.mod(x - offset_x + cell_x * np.float32(0.5), cell_x) - cell_x * np.float32(0.5)
    phase_y = np.mod(y - offset_y + cell_y * np.float32(0.5), cell_y) - cell_y * np.float32(0.5)
    return np.minimum(np.abs(phase_x), np.abs(phase_y)) - np.float32(line_width * 0.5)


def _coverage_from_distance(distance: np.ndarray, aa: str) -> np.ndarray:
    if aa == "off":
        return (distance <= np.float32(0.0)).astype(np.float32)
    return np.clip(np.float32(0.5) - distance, np.float32(0.0), np.float32(1.0)).astype(np.float32)


def _grid_reference(
    *,
    width: int,
    height: int,
    cell: tuple[float, float],
    line_width: float,
    color: Sequence[float],
    background: Sequence[float],
    offset: tuple[float, float],
    aa: str,
) -> np.ndarray:
    if aa == "supersample":
        coverage = np.zeros((height, width), dtype=np.float32)
        y_base, x_base = np.mgrid[:height, :width].astype(np.float32)
        for offset_y in _SUBPIXEL_OFFSETS:
            for offset_x in _SUBPIXEL_OFFSETS:
                distance = _periodic_line_distance(
                    x_base + np.float32(0.5 + offset_x),
                    y_base + np.float32(0.5 + offset_y),
                    cell=cell,
                    line_width=line_width,
                    offset=offset,
                )
                coverage += (distance <= np.float32(0.0)).astype(np.float32)
        coverage *= np.float32(1.0 / 16.0)
    else:
        x, y = _pixel_grid(height, width)
        coverage = _coverage_from_distance(
            _periodic_line_distance(x, y, cell=cell, line_width=line_width, offset=offset),
            aa,
        )
    foreground = np.asarray(color, dtype=np.float32)
    back = np.asarray(background, dtype=np.float32)
    return (back + (foreground - back) * coverage[..., None]).astype(np.float32)


def _checker_index(
    x: np.ndarray,
    y: np.ndarray,
    *,
    cell: tuple[float, float],
    offset: tuple[float, float],
) -> np.ndarray:
    cell_x, cell_y = (np.float32(value) for value in cell)
    offset_x, offset_y = (np.float32(value) for value in offset)
    return (np.floor((x - offset_x) / cell_x).astype(np.int64) + np.floor((y - offset_y) / cell_y).astype(np.int64)) & 1


def _checker_distance_to_boundary(
    x: np.ndarray,
    y: np.ndarray,
    *,
    cell: tuple[float, float],
    offset: tuple[float, float],
) -> np.ndarray:
    cell_x, cell_y = (np.float32(value) for value in cell)
    offset_x, offset_y = (np.float32(value) for value in offset)
    phase_x = np.mod(x - offset_x, cell_x)
    phase_y = np.mod(y - offset_y, cell_y)
    return np.minimum.reduce((phase_x, cell_x - phase_x, phase_y, cell_y - phase_y))


def _checker_reference(
    *,
    width: int,
    height: int,
    cell: tuple[float, float],
    colors: tuple[Sequence[float], Sequence[float]],
    offset: tuple[float, float],
    aa: str,
) -> np.ndarray:
    if aa == "supersample":
        coverage = np.zeros((height, width), dtype=np.float32)
        y_base, x_base = np.mgrid[:height, :width].astype(np.float32)
        for offset_y in _SUBPIXEL_OFFSETS:
            for offset_x in _SUBPIXEL_OFFSETS:
                index = _checker_index(
                    x_base + np.float32(0.5 + offset_x),
                    y_base + np.float32(0.5 + offset_y),
                    cell=cell,
                    offset=offset,
                )
                coverage += (index == 0).astype(np.float32)
        coverage *= np.float32(1.0 / 16.0)
    else:
        x, y = _pixel_grid(height, width)
        index = _checker_index(x, y, cell=cell, offset=offset)
        if aa == "off":
            coverage = (index == 0).astype(np.float32)
        else:
            distance = _checker_distance_to_boundary(x, y, cell=cell, offset=offset)
            coverage = np.where(index == 0, np.float32(0.5) + distance, np.float32(0.5) - distance)
            coverage = np.clip(coverage, np.float32(0.0), np.float32(1.0)).astype(np.float32)
    first = np.asarray(colors[0], dtype=np.float32)
    second = np.asarray(colors[1], dtype=np.float32)
    return (second + (first - second) * coverage[..., None]).astype(np.float32)


def _runs(row: np.ndarray) -> list[tuple[tuple[int, int, int], int]]:
    result: list[tuple[tuple[int, int, int], int]] = []
    for value in row:
        item = tuple(int(component) for component in value)
        if result and result[-1][0] == item:
            result[-1] = (item, result[-1][1] + 1)
        else:
            result.append((item, 1))
    return result


def _narrow_normalized(code: np.ndarray) -> np.ndarray:
    return (code.astype(np.float32) - np.float32(64.0)) / np.float32(876.0)


def _table_tokens(markdown: str, heading: str) -> tuple[str, ...]:
    section = markdown.split(f"## {heading}\n", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
    return tuple(
        cells[1].strip().removeprefix("`").removesuffix("`")
        for line in section.splitlines()
        if line.startswith("| `")
        for cells in (line.split("|"),)
    )


def test_generator_public_signatures_are_keyword_only_and_minimal() -> None:
    """v1-derivative-filters acceptance 17: generators stay in the expanded 68-point public surface."""
    expected = {
        "ramp": (
            ("width", inspect.Parameter.empty),
            ("height", inspect.Parameter.empty),
            ("kind", "linear"),
            ("start", inspect.Parameter.empty),
            ("end", inspect.Parameter.empty),
            ("start_color", inspect.Parameter.empty),
            ("end_color", inspect.Parameter.empty),
            ("colorspace", inspect.Parameter.empty),
            ("gamma", "linear"),
        ),
        "grid": (
            ("width", inspect.Parameter.empty),
            ("height", inspect.Parameter.empty),
            ("cell", inspect.Parameter.empty),
            ("line_width", inspect.Parameter.empty),
            ("color", inspect.Parameter.empty),
            ("background", inspect.Parameter.empty),
            ("offset", (0.0, 0.0)),
            ("colorspace", inspect.Parameter.empty),
            ("gamma", "linear"),
            ("aa", "distance"),
        ),
        "checkerboard": (
            ("width", inspect.Parameter.empty),
            ("height", inspect.Parameter.empty),
            ("cell", inspect.Parameter.empty),
            ("colors", inspect.Parameter.empty),
            ("offset", (0.0, 0.0)),
            ("colorspace", inspect.Parameter.empty),
            ("gamma", "linear"),
            ("aa", "distance"),
        ),
        "color_bars": (
            ("width", inspect.Parameter.empty),
            ("height", inspect.Parameter.empty),
            ("standard", inspect.Parameter.empty),
            ("output", "normalized"),
        ),
    }

    for name, parameters in expected.items():
        function = getattr(px.generate, name)
        signature = inspect.signature(function)
        assert tuple((parameter.name, parameter.default) for parameter in signature.parameters.values()) == parameters
        assert all(parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in signature.parameters.values())
        assert name in px.generate.__all__
    assert len(px.generate.__all__) == 7


@pytest.mark.parametrize("name", GENERATOR_NAMES)
@pytest.mark.parametrize("axis,value", (("width", 0), ("height", -1), ("width", 1.5), ("height", True)))
def test_generator_dimensions_are_positive_non_bool_integers(name: str, axis: str, value: object) -> None:
    """v1-generator acceptance 2: width and height reject non-positive and non-integer values with recovery."""
    with pytest.raises(ValueError) as error:
        getattr(px.generate, name)(**(_base_kwargs(name) | {axis: value}))
    _assert_actionable(error)


@pytest.mark.parametrize(
    ("name", "axis", "token", "accepted"),
    (
        ("ramp", "kind", "conic", KINDS),
        ("ramp", "colorspace", "acescg", ("sRGB", "Rec.709", "Rec.2020", "ACES2065-1", "ACEScg")),
        ("grid", "gamma", "gamma22", ("linear", "srgb", "rec709", "pq", "hlg")),
        ("grid", "aa", "nearest", AAS),
        ("checkerboard", "aa", "area", AAS),
        ("color_bars", "standard", "ebu", STANDARDS),
        ("color_bars", "output", "float", OUTPUTS),
    ),
)
def test_generator_tokens_fail_fast_and_list_the_vocabulary(
    name: str, axis: str, token: str, accepted: tuple[str, ...]
) -> None:
    """v1-generator acceptance 3-8: every named axis is closed, case-sensitive, and actionable."""
    with pytest.raises(ValueError) as error:
        getattr(px.generate, name)(**(_base_kwargs(name) | {axis: token}))
    _assert_actionable(error)
    assert all(candidate in str(error.value) for candidate in accepted)


@pytest.mark.parametrize(
    ("name", "overrides"),
    (
        ("ramp", {"start_color": (), "end_color": ()}),
        ("ramp", {"start_color": (0.0, 1.0), "end_color": (0.0, 1.0)}),
        ("ramp", {"start_color": (0.0,), "end_color": (0.0, 1.0, 2.0)}),
        ("grid", {"color": (1.0,), "background": (0.0, 0.0, 0.0)}),
        ("grid", {"color": (1.0, math.nan, 0.0), "background": (0.0, 0.0, 0.0)}),
        ("checkerboard", {"colors": ((1.0,),)}),
        ("checkerboard", {"colors": ((1.0,), (0.0,), (0.5,))}),
        ("checkerboard", {"colors": ((1.0,), (0.0, 0.5, 1.0))}),
    ),
)
def test_generator_color_inputs_validate_shape_count_finiteness_and_matching(
    name: str, overrides: dict[str, object]
) -> None:
    """v1-generator acceptance 9-10 and 13: color sequences are finite, length 1/3/4, and structurally aligned."""
    with pytest.raises(ValueError) as error:
        getattr(px.generate, name)(**(_base_kwargs(name) | overrides))
    _assert_actionable(error)


@pytest.mark.parametrize(
    ("name", "overrides"),
    (
        ("ramp", {"start": (0.0, math.nan)}),
        ("ramp", {"end": (math.inf, 0.0)}),
        ("ramp", {"start": (1.0, 1.0), "end": (1.0, 1.0)}),
        ("grid", {"cell": 0.0}),
        ("grid", {"cell": (2.0, -1.0)}),
        ("grid", {"cell": (2.0, 3.0, 4.0)}),
        ("grid", {"line_width": 0.0}),
        ("grid", {"offset": (math.inf, 0.0)}),
        ("checkerboard", {"cell": True}),
        ("checkerboard", {"offset": (0.0, math.nan)}),
    ),
)
def test_generator_geometry_rejects_nonfinite_nonpositive_or_undefined_values(
    name: str, overrides: dict[str, object]
) -> None:
    """v1-generator acceptance 11-14: geometry is finite, positive where dimensional, and ramp direction is defined."""
    with pytest.raises(ValueError) as error:
        getattr(px.generate, name)(**(_base_kwargs(name) | overrides))
    _assert_actionable(error)


@pytest.mark.parametrize(
    ("colors", "channels"),
    (((-2.0,), ("Y",)), ((-2.0, 0.5, 3.0), ("R", "G", "B")), ((-2.0, 0.5, 3.0, 4.0), ("R", "G", "B", "A"))),
)
def test_numeric_generators_allocate_private_fp32_hwc_and_derive_channels(
    colors: tuple[float, ...], channels: tuple[str, ...]
) -> None:
    """v1-generator acceptance 15-17 and 27: numeric generators return private contiguous fp32 Frames with scene values."""
    ramp = px.generate.ramp(
        width=4,
        height=3,
        start=(0.0, 0.0),
        end=(4.0, 0.0),
        start_color=colors,
        end_color=tuple(value + 1.0 for value in colors),
        colorspace="S-Gamut3",
        gamma="s-log3",
    )
    grid = px.generate.grid(
        width=4,
        height=3,
        cell=2.0,
        line_width=1.0,
        color=colors,
        background=tuple(value + 1.0 for value in colors),
        colorspace="S-Gamut3",
        gamma="s-log3",
        aa="off",
    )
    checker = px.generate.checkerboard(
        width=4,
        height=3,
        cell=2.0,
        colors=(colors, tuple(value + 1.0 for value in colors)),
        colorspace="S-Gamut3",
        gamma="s-log3",
        aa="off",
    )

    for frame in (ramp, grid, checker):
        assert (frame.shape, frame.dtype.name, frame.channels, frame.colorspace, frame.gamma) == (
            (3, 4, len(colors)),
            "float32",
            channels,
            "S-Gamut3",
            "s-log3",
        )
        assert frame.data.flags.c_contiguous
        assert np.min(_host(frame)) < 0.0 or np.max(_host(frame)) > 1.0
    assert len({frame.data.data.ptr for frame in (ramp, grid, checker)}) == 3


@pytest.mark.parametrize("kind", KINDS)
def test_ramp_matches_independent_fp32_pixel_center_oracle(kind: str) -> None:
    """v1-generator acceptance 20-22 and 27: linear/radial ramps use pixel centers, saturation, and direct-space fp32 mix."""
    kwargs = {
        "width": 8,
        "height": 6,
        "kind": kind,
        "start": (1.25, 1.75),
        "end": (6.5, 4.25),
        "start_color": (-0.5, 0.25, 1.5),
        "end_color": (2.0, 1.25, -1.0),
        "colorspace": "ACEScg",
        "gamma": "2.4",
    }
    expected = _ramp_reference(**{key: kwargs[key] for key in kwargs if key not in {"colorspace", "gamma"}})
    result = px.generate.ramp(**kwargs)
    np.testing.assert_allclose(_host(result), expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("aa", AAS)
def test_grid_matches_independent_periodic_coverage_oracle_without_double_composite(aa: str) -> None:
    """v1-generator acceptance 20, 23, and 25-27: grid coverage is periodic, symmetric, unioned, and AA-selectable."""
    kwargs = {
        "width": 9,
        "height": 7,
        "cell": (3.25, 2.5),
        "line_width": 0.8,
        "color": (2.0, -1.0, 0.5),
        "background": (-0.25, 0.5, 1.5),
        "offset": (-0.375, 0.625),
        "colorspace": "ACEScg",
        "aa": aa,
    }
    expected = _grid_reference(
        width=kwargs["width"],
        height=kwargs["height"],
        cell=kwargs["cell"],
        line_width=kwargs["line_width"],
        color=kwargs["color"],
        background=kwargs["background"],
        offset=kwargs["offset"],
        aa=aa,
    )
    result = px.generate.grid(**kwargs)
    np.testing.assert_allclose(_host(result), expected, rtol=2e-6, atol=2e-6)
    if aa == "off":
        assert set(np.unique(_host(result))) <= set(
            np.asarray((*kwargs["color"], *kwargs["background"]), dtype=np.float32)
        )


@pytest.mark.parametrize("aa", AAS)
def test_checkerboard_matches_independent_periodic_coverage_oracle(aa: str) -> None:
    """v1-generator acceptance 20 and 24-27: checker cells start with color one at the origin and honor all AA modes."""
    kwargs = {
        "width": 9,
        "height": 7,
        "cell": (3.25, 2.5),
        "colors": ((2.0, -1.0, 0.5), (-0.25, 0.5, 1.5)),
        "offset": (-0.375, 0.625),
        "colorspace": "ACEScg",
        "aa": aa,
    }
    expected = _checker_reference(
        width=kwargs["width"],
        height=kwargs["height"],
        cell=kwargs["cell"],
        colors=kwargs["colors"],
        offset=kwargs["offset"],
        aa=aa,
    )
    result = px.generate.checkerboard(**kwargs)
    np.testing.assert_allclose(_host(result), expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("name", ("grid", "checkerboard"))
def test_periodic_generators_are_bit_identical_after_one_cell_offset(name: str) -> None:
    """v1-generator acceptance 19 and 26: adding a full cell period to offset preserves bit-identical output."""
    kwargs = _base_kwargs(name) | {"cell": (3.25, 2.5), "offset": (-0.375, 0.625)}
    shifted = kwargs | {"offset": (kwargs["offset"][0] + 3.25, kwargs["offset"][1] + 2.5)}
    first = getattr(px.generate, name)(**kwargs)
    second = getattr(px.generate, name)(**shifted)
    np.testing.assert_array_equal(_host(first), _host(second))


@pytest.mark.parametrize(
    ("standard", "colorspace", "gamma"),
    (
        ("arib-std-b28", "Rec.709", "rec709"),
        ("smpte-rp219", "Rec.709", "rec709"),
        ("bt2111-hlg", "Rec.2020", "hlg"),
        ("bt2111-pq", "Rec.2020", "pq"),
        ("bt2111-pq-full", "Rec.2020", "pq"),
        ("full-100", "Rec.709", "rec709"),
        ("full-75", "Rec.709", "rec709"),
    ),
)
def test_color_bar_standards_determine_metadata_dtype_and_deterministic_storage(
    standard: str, colorspace: str, gamma: str
) -> None:
    """v1-generator acceptance 15 and 18-19, 28, and 32: standard/output determine metadata, dtype, and exact repeatability."""
    normalized = px.generate.color_bars(width=32, height=18, standard=standard)
    repeated = px.generate.color_bars(width=32, height=18, standard=standard)
    code = px.generate.color_bars(width=32, height=18, standard=standard, output="code")
    assert (normalized.shape, normalized.dtype.name, normalized.colorspace, normalized.gamma, normalized.channels) == (
        (18, 32, 3),
        "float32",
        colorspace,
        gamma,
        ("R", "G", "B"),
    )
    assert (code.shape, code.dtype.name, code.colorspace, code.gamma, code.channels) == (
        (18, 32, 3),
        "uint16",
        colorspace,
        gamma,
        ("R", "G", "B"),
    )
    assert normalized.data.flags.c_contiguous and code.data.flags.c_contiguous
    np.testing.assert_array_equal(_host(normalized), _host(repeated))


def test_std_b28_and_rp219_match_normative_code_geometry_and_normalization() -> None:
    """v1-generator acceptance 29-35: STD-B28/RP219 share the normative four-pattern geometry, codes, PLUGE, and ramp."""
    arib = px.generate.color_bars(width=1920, height=1080, standard="arib-std-b28", output="code")
    rp219 = px.generate.color_bars(width=1920, height=1080, standard="smpte-rp219", output="code")
    code = _host(arib)
    np.testing.assert_array_equal(code, _host(rp219))

    assert _runs(code[0]) == [
        ((414, 414, 414), 240),
        ((721, 721, 721), 205),
        ((721, 721, 64), 206),
        ((64, 721, 721), 206),
        ((64, 721, 64), 206),
        ((721, 64, 721), 206),
        ((721, 64, 64), 206),
        ((64, 64, 721), 205),
        ((414, 414, 414), 240),
    ]
    assert _runs(code[630]) == [
        ((64, 940, 940), 240),
        ((721, 721, 721), 1440),
        ((64, 64, 940), 240),
    ]
    assert np.all(code[720, :240] == (940, 940, 64))
    assert np.all(code[720, 240:445] == (64, 64, 64))
    assert np.all(code[720, 1475:1680] == (940, 940, 940))
    assert np.all(code[720, 1680:] == (940, 64, 64))
    ramp = code[720, 445:1475, 0]
    assert ramp[0] == 64 and ramp[-1] == 940 and np.all(np.diff(ramp.astype(np.int32)) >= 0)
    assert _runs(code[810]) == [
        ((195, 195, 195), 240),
        ((64, 64, 64), 309),
        ((940, 940, 940), 411),
        ((64, 64, 64), 171),
        ((46, 46, 46), 69),
        ((64, 64, 64), 68),
        ((82, 82, 82), 69),
        ((64, 64, 64), 68),
        ((99, 99, 99), 69),
        ((64, 64, 64), 206),
        ((195, 195, 195), 240),
    ]

    normalized = _host(px.generate.color_bars(width=1920, height=1080, standard="arib-std-b28"))
    np.testing.assert_array_equal(normalized, _narrow_normalized(code))
    assert normalized[810, 1131, 0] < 0.0


@pytest.mark.parametrize(
    ("standard", "main_high", "grey", "bottom_left", "bottom_right"),
    (
        (
            "bt2111-hlg",
            721,
            414,
            ((713, 719, 316), (538, 709, 718), (512, 706, 296)),
            ((651, 286, 705), (639, 269, 164), (227, 147, 702)),
        ),
        (
            "bt2111-pq",
            572,
            414,
            ((568, 571, 381), (484, 566, 571), (474, 564, 368)),
            ((536, 361, 564), (530, 350, 256), (317, 236, 562)),
        ),
        (
            "bt2111-pq-full",
            593,
            409,
            ((589, 592, 370), (491, 586, 592), (478, 584, 355)),
            ((551, 347, 584), (544, 334, 225), (296, 201, 582)),
        ),
    ),
)
def test_bt2111_variants_match_named_widths_levels_staircase_ramp_and_bottom_references(
    standard: str,
    main_high: int,
    grey: int,
    bottom_left: tuple[tuple[int, int, int], ...],
    bottom_right: tuple[tuple[int, int, int], ...],
) -> None:
    """v1-generator acceptance 30 and 32-35: BT.2111 variants follow the normative 2K regions and code tables."""
    code = _host(px.generate.color_bars(width=1920, height=1080, standard=standard, output="code"))
    low = 0 if standard == "bt2111-pq-full" else 64
    top_high = 1023 if standard == "bt2111-pq-full" else 940

    assert _runs(code[0]) == [
        ((grey, grey, grey), 240),
        ((top_high, top_high, top_high), 206),
        ((top_high, top_high, low), 206),
        ((low, top_high, top_high), 206),
        ((low, top_high, low), 204),
        ((top_high, low, top_high), 206),
        ((top_high, low, low), 206),
        ((low, low, top_high), 206),
        ((grey, grey, grey), 240),
    ]
    assert _runs(code[90]) == [
        ((grey, grey, grey), 240),
        ((main_high, main_high, main_high), 206),
        ((main_high, main_high, low), 206),
        ((low, main_high, main_high), 206),
        ((low, main_high, low), 204),
        ((main_high, low, main_high), 206),
        ((main_high, low, low), 206),
        ((low, low, main_high), 206),
        ((grey, grey, grey), 240),
    ]

    stair = code[630, :, 0]
    if standard == "bt2111-pq-full":
        values = (0, 0, 102, 205, 307, 409, 512, 614, 716, 818, 921, 1023, 1023)
    else:
        values = (4, 64, 152, 239, 327, 414, 502, 590, 677, 765, 852, 940, 1019)
    widths = (206, 103, 103, 103, 103, 102, 102, 103, 103, 103, 103, 103, 103)
    cursor = 240
    assert np.all(stair[:cursor] == main_high)
    for value, width in zip(values, widths, strict=True):
        assert np.all(stair[cursor : cursor + width] == value)
        cursor += width
    assert cursor == 1680 and np.all(stair[cursor:] == main_high)

    ramp = code[720, :, 0]
    assert np.all(ramp[:240] == low)
    if standard == "bt2111-pq-full":
        assert np.all(ramp[240:791] == 0)
        np.testing.assert_array_equal(ramp[791:1813], np.arange(1, 1023, dtype=np.uint16))
        assert np.all(ramp[1813:] == 1023)
    else:
        assert np.all(ramp[240:799] == 4)
        np.testing.assert_array_equal(ramp[799:1813], np.arange(5, 1019, dtype=np.uint16))
        assert np.all(ramp[1813:] == 1019)

    bottom = code[810]
    for index, expected in enumerate(bottom_left):
        assert np.all(bottom[index * 80 : (index + 1) * 80] == expected)
    for index, expected in enumerate(bottom_right):
        left = 1680 + index * 80
        assert np.all(bottom[left : left + 80] == expected)
    near_black = (0, 0, 20, 0, 41) if standard == "bt2111-pq-full" else (48, 64, 80, 64, 99)
    cursor = 376
    for value, width in zip(near_black, (70, 68, 70, 68, 70), strict=True):
        assert np.all(bottom[cursor : cursor + width] == (value, value, value))
        cursor += width
    assert cursor == 722
    assert np.all(bottom[722:960] == (low, low, low))
    assert np.all(bottom[960:1398] == (main_high, main_high, main_high))
    assert np.all(bottom[1398:1680] == (low, low, low))

    normalized = _host(px.generate.color_bars(width=1920, height=1080, standard=standard))
    expected_normalized = (
        code.astype(np.float32) / np.float32(1023.0) if standard.endswith("-full") else _narrow_normalized(code)
    )
    np.testing.assert_array_equal(normalized, expected_normalized)


def test_full_field_bar_variants_keep_white_at_100_and_change_only_coloured_maximum() -> None:
    """v1-generator acceptance 28 and 30: BT.471 full-field variants use eight bars and 100/0/100/0 vs 100/0/75/0 codes."""
    full = _host(px.generate.color_bars(width=80, height=3, standard="full-100", output="code"))
    seventy_five = _host(px.generate.color_bars(width=80, height=3, standard="full-75", output="code"))
    expected_full = (
        (940, 940, 940),
        (940, 940, 64),
        (64, 940, 940),
        (64, 940, 64),
        (940, 64, 940),
        (940, 64, 64),
        (64, 64, 940),
        (64, 64, 64),
    )
    expected_seventy_five = (
        (940, 940, 940),
        (721, 721, 64),
        (64, 721, 721),
        (64, 721, 64),
        (721, 64, 721),
        (721, 64, 64),
        (64, 64, 721),
        (64, 64, 64),
    )
    assert _runs(full[0]) == [(value, 10) for value in expected_full]
    assert _runs(seventy_five[0]) == [(value, 10) for value in expected_seventy_five]


@pytest.mark.parametrize("standard", STANDARDS)
def test_color_bars_scale_boundaries_to_tiny_frames_without_gaps_or_minimum_size_errors(standard: str) -> None:
    """v1-generator acceptance 33 and 36: proportional rounded boundaries fill arbitrary and one-pixel frames."""
    tiny = px.generate.color_bars(width=1, height=1, standard=standard, output="code")
    scaled = px.generate.color_bars(width=19, height=13, standard=standard, output="code")
    assert tiny.shape == (1, 1, 3)
    assert scaled.shape == (13, 19, 3)
    assert np.all(_host(tiny) <= 1023)
    assert np.all(_host(scaled) <= 1023)


def test_generator_vocabulary_tables_equal_implementation_token_sets(vocabulary_markdown: str) -> None:
    """v1-generator acceptance 37: kind, standard, output, and shared aa vocabulary exactly match implementation."""
    from pixtreme._draw.shapes import _AA_TOKENS
    from pixtreme._generate.patterns import _KIND_TOKENS, _OUTPUT_TOKENS, _STANDARD_TOKENS

    markdown = vocabulary_markdown
    assert _table_tokens(markdown, "generator kind") == KINDS == _KIND_TOKENS
    assert _table_tokens(markdown, "color bars standard") == STANDARDS == _STANDARD_TOKENS
    assert _table_tokens(markdown, "color bars output") == OUTPUTS == _OUTPUT_TOKENS
    assert _AA_TOKENS == AAS
    assert "grid" in markdown and "checkerboard" in markdown and "共用" in markdown


def test_generator_docstrings_state_llm_readable_geometry_metadata_and_output_contracts() -> None:
    """v1-generator acceptance 38: public docstrings expose coordinates, values, metadata, normalization, and code output."""
    combined = "\n".join(inspect.getdoc(getattr(px.generate, name)) or "" for name in GENERATOR_NAMES)
    for required in (
        "(x, y)",
        "i + 0.5",
        "j + 0.5",
        "scene",
        "clamp",
        "colorspace",
        "gamma",
        "channels",
        "normalized",
        "code",
        "uint16",
        "(code - 64) / 876",
        "code / 1023",
        "new",
        "allocation",
    ):
        assert required in combined


def test_generators_use_rawkernel_per_pixel_evaluation() -> None:
    """v1-generator acceptance 1 and 35: structural contract fixes CUDA RawKernel generation without host image synthesis."""
    import pixtreme._generate.patterns as generate_module

    source = inspect.getsource(generate_module._generate_geometry) + inspect.getsource(generate_module.color_bars)
    assert "cp.RawKernel" in inspect.getsource(generate_module._geometry_kernel)
    assert "cp.RawKernel" in inspect.getsource(generate_module._color_bars_kernel)
    assert "ElementwiseKernel" not in source
    assert "cp.asnumpy" not in source
