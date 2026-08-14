"""Canonical CUDA source fragments for point interpolation weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pixtreme._core.vocabulary import _INTERPOLATION_TOKENS as _ALL_INTERPOLATION_TOKENS

type _PointInterpolationFamily = Literal["nearest", "linear", "keys", "mitchell", "lanczos"]


@dataclass(frozen=True, slots=True)
class _PointInterpolationSpec:
    index: int
    family: _PointInterpolationFamily
    b: str | None = None
    c: str | None = None
    lobes: int | None = None


_POINT_INTERPOLATION_SPECS = {
    "nearest": _PointInterpolationSpec(0, "nearest"),
    "bilinear": _PointInterpolationSpec(1, "linear"),
    "bicubic": _PointInterpolationSpec(2, "keys"),
    "b-spline": _PointInterpolationSpec(3, "mitchell", b="1.0f", c="0.0f"),
    "mitchell": _PointInterpolationSpec(4, "mitchell", b="1.0f / 3.0f", c="1.0f / 3.0f"),
    "lanczos2": _PointInterpolationSpec(5, "lanczos", lobes=2),
    "lanczos3": _PointInterpolationSpec(6, "lanczos", lobes=3),
    "lanczos4": _PointInterpolationSpec(7, "lanczos", lobes=4),
}
_POINT_INTERPOLATION_TOKENS = tuple(_POINT_INTERPOLATION_SPECS)

if _POINT_INTERPOLATION_TOKENS != _ALL_INTERPOLATION_TOKENS[: len(_POINT_INTERPOLATION_TOKENS)]:
    raise RuntimeError("point interpolation indices must match the public vocabulary order")


def _canonical_mitchell_coefficients(interpolation: str) -> tuple[str, str]:
    spec = _POINT_INTERPOLATION_SPECS[interpolation]
    if spec.family != "mitchell" or spec.b is None or spec.c is None:
        raise RuntimeError(f"missing Mitchell coefficients for {interpolation!r}")
    return spec.b, spec.c


def _canonical_lanczos_index_offset() -> int:
    offsets = {
        spec.index - spec.lobes
        for spec in _POINT_INTERPOLATION_SPECS.values()
        if spec.family == "lanczos" and spec.lobes is not None
    }
    if len(offsets) != 1:
        raise RuntimeError("Lanczos interpolation indices must have one lobe offset")
    return offsets.pop()


_B_SPLINE_B, _B_SPLINE_C = _canonical_mitchell_coefficients("b-spline")
_MITCHELL_B, _MITCHELL_C = _canonical_mitchell_coefficients("mitchell")
_LANCZOS_INDEX_OFFSET = _canonical_lanczos_index_offset()


def _absolute_distance(distance_scale: str | None) -> str:
    if distance_scale is None:
        return "fabsf(distance)"
    return f"fabsf(distance) * {distance_scale}"


def _linear_body(*, distance_scale: str | None) -> str:
    if distance_scale is None:
        return """
    const float weight = 1.0f - fabsf(distance);
    return weight > 0.0f ? weight : 0.0f;
"""
    return f"""
    const float x = {_absolute_distance(distance_scale)};
    const float weight = 1.0f - x;
    return weight > 0.0f ? weight : 0.0f;
"""


def _keys_body(*, distance_scale: str | None) -> str:
    return f"""
    const float x = {_absolute_distance(distance_scale)};
    const float a = -0.5f;
    if (x < 1.0f) {{
        return (a + 2.0f) * x * x * x - (a + 3.0f) * x * x + 1.0f;
    }}
    if (x < 2.0f) {{
        return a * x * x * x - 5.0f * a * x * x + 8.0f * a * x - 4.0f * a;
    }}
    return 0.0f;
"""


def _mitchell_body(*, distance_scale: str | None, coefficients: tuple[str, str] | None) -> str:
    declarations = ""
    if coefficients is not None:
        declarations = f"\n    const float b = {coefficients[0]};\n    const float c = {coefficients[1]};"
    return f"""
    const float x = {_absolute_distance(distance_scale)};{declarations}
    if (x < 1.0f) {{
        return ((12.0f - 9.0f * b - 6.0f * c) * x * x * x
            + (-18.0f + 12.0f * b + 6.0f * c) * x * x
            + (6.0f - 2.0f * b)) / 6.0f;
    }}
    if (x < 2.0f) {{
        return ((-b - 6.0f * c) * x * x * x
            + (6.0f * b + 30.0f * c) * x * x
            + (-12.0f * b - 48.0f * c) * x
            + (8.0f * b + 24.0f * c)) / 6.0f;
    }}
    return 0.0f;
"""


def _lanczos_body(*, distance_scale: str | None, lobes: str) -> str:
    return f"""
    const float x = {_absolute_distance(distance_scale)};
    if (x == 0.0f) {{
        return 1.0f;
    }}
    if (x >= {lobes}) {{
        return 0.0f;
    }}
    const float pi_x = 3.14159265358979323846f * x;
    return ({lobes} * sinf(pi_x) * sinf(pi_x / {lobes})) / (pi_x * pi_x);
"""


def _specialized_point_weight_source(interpolation: str, *, distance_scale: str | None = None) -> str:
    """Emit one branch-free device weight function for a canonical point token."""
    try:
        spec = _POINT_INTERPOLATION_SPECS[interpolation]
    except KeyError as error:
        raise ValueError(f"unsupported point interpolation token {interpolation!r}") from error

    if spec.family == "nearest":
        return ""
    if spec.family == "linear":
        body = _linear_body(distance_scale=distance_scale)
    elif spec.family == "keys":
        body = _keys_body(distance_scale=distance_scale)
    elif spec.family == "mitchell":
        body = _mitchell_body(
            distance_scale=distance_scale,
            coefficients=_canonical_mitchell_coefficients(interpolation),
        )
    else:
        if spec.lobes is None:
            raise RuntimeError(f"missing Lanczos lobe count for {interpolation!r}")
        body = _lanczos_body(distance_scale=distance_scale, lobes=f"{spec.lobes}.0f")

    return f"""
__device__ float pixtreme_weight(const float distance) {{{body}
}}
"""


_POINT_INTERPOLATION_DEVICE_SOURCE = f"""
__device__ float pixtreme_keys_weight(const float distance) {{{_keys_body(distance_scale=None)}
}}

__device__ float pixtreme_mitchell_weight(const float distance, const float b, const float c) {{{_mitchell_body(distance_scale=None, coefficients=None)}
}}

__device__ float pixtreme_lanczos_weight(const float distance, const int lobes) {{{_lanczos_body(distance_scale=None, lobes="(float)lobes")}
}}

__device__ float pixtreme_point_weight(const int interpolation, const float distance) {{
    if (interpolation == {_POINT_INTERPOLATION_SPECS["bilinear"].index}) {{{_linear_body(distance_scale=None)}
    }}
    if (interpolation == {_POINT_INTERPOLATION_SPECS["bicubic"].index}) {{
        return pixtreme_keys_weight(distance);
    }}
    if (interpolation == {_POINT_INTERPOLATION_SPECS["b-spline"].index}) {{
        return pixtreme_mitchell_weight(distance, {_B_SPLINE_B}, {_B_SPLINE_C});
    }}
    if (interpolation == {_POINT_INTERPOLATION_SPECS["mitchell"].index}) {{
        return pixtreme_mitchell_weight(distance, {_MITCHELL_B}, {_MITCHELL_C});
    }}
    return pixtreme_lanczos_weight(distance, interpolation - {_LANCZOS_INDEX_OFFSET});
}}
"""
