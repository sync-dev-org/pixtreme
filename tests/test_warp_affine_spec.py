"""Specification, contract, and numerical-property tests for affine warp."""

from __future__ import annotations

import inspect
import math
from typing import Any

import numpy as np
import pytest

import pixtreme as px

INTERPOLATIONS = (
    "nearest",
    "bilinear",
    "bicubic",
    "b-spline",
    "mitchell",
    "lanczos2",
    "lanczos3",
    "lanczos4",
    "area",
)
BORDERS = ("mirror", "replicate", "wrap", "constant")


def _frame(
    values: Any,
    *,
    dtype: Any = np.float32,
    colorspace: str = "sRGB",
    gamma: str = "linear",
    channels: str | list[str] | tuple[str, ...] = ("signal",),
    matrix: str | None = None,
) -> px.core.Frame:
    import cupy as cp

    return px.io.from_array(
        cp.asarray(np.asarray(values, dtype=dtype)),
        colorspace=colorspace,
        gamma=gamma,
        channels=channels,
        matrix=matrix,
    )


def _assert_actionable(error: pytest.ExceptionInfo[ValueError]) -> None:
    message = str(error.value)
    assert "why=" in message
    assert "; what=" in message
    assert "; how=" in message


def _keys_weight(distance: float) -> float:
    x = abs(distance)
    a = -0.5
    if x < 1.0:
        return (a + 2.0) * x**3 - (a + 3.0) * x**2 + 1.0
    if x < 2.0:
        return a * x**3 - 5.0 * a * x**2 + 8.0 * a * x - 4.0 * a
    return 0.0


def _mitchell_weight(distance: float, *, b: float, c: float) -> float:
    x = abs(distance)
    if x < 1.0:
        return ((12.0 - 9.0 * b - 6.0 * c) * x**3 + (-18.0 + 12.0 * b + 6.0 * c) * x**2 + (6.0 - 2.0 * b)) / 6.0
    if x < 2.0:
        return (
            (-b - 6.0 * c) * x**3 + (6.0 * b + 30.0 * c) * x**2 + (-12.0 * b - 48.0 * c) * x + (8.0 * b + 24.0 * c)
        ) / 6.0
    return 0.0


def _lanczos_weight(distance: float, *, lobes: int) -> float:
    x = abs(distance)
    if x == 0.0:
        return 1.0
    if x >= lobes:
        return 0.0
    return float(np.sinc(x) * np.sinc(x / lobes))


def _axis_samples(token: str, coordinate: float) -> tuple[tuple[int, ...], np.ndarray]:
    base = math.floor(coordinate)
    if token == "bilinear":
        indices = (base, base + 1)
        weights = np.asarray((1.0 - (coordinate - base), coordinate - base), dtype=np.float64)
    elif token in {"bicubic", "b-spline", "mitchell"}:
        indices = tuple(range(base - 1, base + 3))
        if token == "bicubic":
            weights = np.asarray([_keys_weight(coordinate - index) for index in indices], dtype=np.float64)
        else:
            b, c = (1.0, 0.0) if token == "b-spline" else (1.0 / 3.0, 1.0 / 3.0)
            weights = np.asarray(
                [_mitchell_weight(coordinate - index, b=b, c=c) for index in indices],
                dtype=np.float64,
            )
    else:
        lobes = int(token.removeprefix("lanczos"))
        indices = tuple(range(base - lobes + 1, base + lobes + 1))
        weights = np.asarray(
            [_lanczos_weight(coordinate - index, lobes=lobes) for index in indices],
            dtype=np.float64,
        )
        weights /= weights.sum()
    return indices, weights


def _border_index(index: int, extent: int, border: str) -> int | None:
    if 0 <= index < extent:
        return index
    if border == "constant":
        return None
    if extent == 1:
        return 0
    if border == "replicate":
        return min(max(index, 0), extent - 1)
    if border == "wrap":
        return index % extent
    period = 2 * extent - 2
    reflected = index % period
    return reflected if reflected < extent else period - reflected


def _sample(
    source: np.ndarray,
    y: int,
    x: int,
    *,
    border: str,
    border_value: float,
) -> np.ndarray:
    mapped_y = _border_index(y, source.shape[0], border)
    mapped_x = _border_index(x, source.shape[1], border)
    if mapped_y is None or mapped_x is None:
        return np.full(source.shape[2], border_value, dtype=np.float64)
    return source[mapped_y, mapped_x].astype(np.float64)


def _effective_matrices(matrix: np.ndarray, *, inverse: bool) -> tuple[np.ndarray, np.ndarray]:
    declared = np.asarray(matrix, dtype=np.float32)
    homogeneous = np.vstack((declared.astype(np.float64), np.asarray((0.0, 0.0, 1.0))))
    declared_inverse = np.linalg.inv(homogeneous)[:2].astype(np.float32)
    if inverse:
        return declared_inverse, declared
    return declared, declared_inverse


def _point_reference(
    source: np.ndarray,
    matrix: np.ndarray,
    *,
    inverse: bool,
    width: int,
    height: int,
    interpolation: str,
    border: str,
    border_value: float,
) -> np.ndarray:
    _, inverse_mapping = _effective_matrices(matrix, inverse=inverse)
    output = np.empty((height, width, source.shape[2]), dtype=np.float64)
    for output_y in range(height):
        for output_x in range(width):
            source_x = float(
                inverse_mapping[0, 0] * output_x + inverse_mapping[0, 1] * output_y + inverse_mapping[0, 2]
            )
            source_y = float(
                inverse_mapping[1, 0] * output_x + inverse_mapping[1, 1] * output_y + inverse_mapping[1, 2]
            )
            if interpolation == "nearest":
                output[output_y, output_x] = _sample(
                    source,
                    math.floor(source_y + 0.5),
                    math.floor(source_x + 0.5),
                    border=border,
                    border_value=border_value,
                )
                continue
            y_indices, y_weights = _axis_samples(interpolation, source_y)
            x_indices, x_weights = _axis_samples(interpolation, source_x)
            value = np.zeros(source.shape[2], dtype=np.float64)
            for y_index, weight_y in zip(y_indices, y_weights, strict=True):
                for x_index, weight_x in zip(x_indices, x_weights, strict=True):
                    value += _sample(
                        source,
                        y_index,
                        x_index,
                        border=border,
                        border_value=border_value,
                    ) * (weight_y * weight_x)
            output[output_y, output_x] = value
    return output.astype(np.float32)


def _clip_polygon(
    polygon: list[tuple[float, float]],
    *,
    axis: int,
    boundary: float,
    keep_greater: bool,
) -> list[tuple[float, float]]:
    if not polygon:
        return []
    clipped: list[tuple[float, float]] = []
    previous = polygon[-1]
    previous_inside = previous[axis] >= boundary if keep_greater else previous[axis] <= boundary
    for current in polygon:
        current_inside = current[axis] >= boundary if keep_greater else current[axis] <= boundary
        if current_inside != previous_inside:
            denominator = current[axis] - previous[axis]
            fraction = (boundary - previous[axis]) / denominator
            intersection = (
                previous[0] + fraction * (current[0] - previous[0]),
                previous[1] + fraction * (current[1] - previous[1]),
            )
            clipped.append(intersection)
        if current_inside:
            clipped.append(current)
        previous = current
        previous_inside = current_inside
    return clipped


def _polygon_area(polygon: list[tuple[float, float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    return (
        abs(
            sum(
                polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
                - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
                for index in range(len(polygon))
            )
        )
        * 0.5
    )


def _cell_overlap(polygon: list[tuple[float, float]], *, x: int, y: int) -> float:
    clipped = _clip_polygon(polygon, axis=0, boundary=x - 0.5, keep_greater=True)
    clipped = _clip_polygon(clipped, axis=0, boundary=x + 0.5, keep_greater=False)
    clipped = _clip_polygon(clipped, axis=1, boundary=y - 0.5, keep_greater=True)
    clipped = _clip_polygon(clipped, axis=1, boundary=y + 0.5, keep_greater=False)
    return _polygon_area(clipped)


def _area_reference(
    source: np.ndarray,
    matrix: np.ndarray,
    *,
    inverse: bool,
    width: int,
    height: int,
    border: str,
    border_value: float,
) -> np.ndarray:
    _, inverse_mapping = _effective_matrices(matrix, inverse=inverse)
    determinant = abs(float(np.linalg.det(inverse_mapping[:, :2].astype(np.float64))))
    output = np.empty((height, width, source.shape[2]), dtype=np.float64)
    for output_y in range(height):
        for output_x in range(width):
            corners = []
            for offset_x, offset_y in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)):
                destination = np.asarray((output_x + offset_x, output_y + offset_y, 1.0), dtype=np.float64)
                mapped = inverse_mapping.astype(np.float64) @ destination
                corners.append((float(mapped[0]), float(mapped[1])))
            minimum_x = min(point[0] for point in corners)
            maximum_x = max(point[0] for point in corners)
            minimum_y = min(point[1] for point in corners)
            maximum_y = max(point[1] for point in corners)
            total = np.zeros(source.shape[2], dtype=np.float64)
            for source_y in range(math.floor(minimum_y + 0.5), math.ceil(maximum_y + 0.5)):
                for source_x in range(math.floor(minimum_x + 0.5), math.ceil(maximum_x + 0.5)):
                    weight = _cell_overlap(corners, x=source_x, y=source_y)
                    total += (
                        _sample(
                            source,
                            source_y,
                            source_x,
                            border=border,
                            border_value=border_value,
                        )
                        * weight
                    )
            output[output_y, output_x] = total / determinant
    return output.astype(np.float32)


def test_warp_affine_public_signature_and_unique_surface() -> None:
    """v1-warp-affine acceptance 1 and 17: the sole public entry has the exact call shape."""
    signature = inspect.signature(px.transform.warp_affine)
    assert tuple(signature.parameters) == (
        "frame",
        "matrix",
        "inverse",
        "width",
        "height",
        "interpolation",
        "border",
        "border_value",
    )
    assert signature.parameters["frame"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert signature.parameters["matrix"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in ("inverse", "width", "height", "interpolation", "border", "border_value"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["inverse"].default is False
    assert signature.parameters["width"].default is None
    assert signature.parameters["height"].default is None
    assert signature.parameters["interpolation"].default is None
    assert signature.parameters["border"].default == "constant"
    assert signature.parameters["border_value"].default is None
    assert px.transform.__all__.count("warp_affine") == 1
    assert not hasattr(px, "warp_affine")
    assert not hasattr(px.core.Frame, "warp_affine")
    assert not hasattr(px.transform, "get_inverse_matrix")


@pytest.mark.parametrize(
    "kwargs",
    (
        {"width": 2},
        {"height": 2},
        {"width": 0, "height": 2},
        {"width": 2, "height": -1},
        {"width": True, "height": 2},
        {"width": 2, "height": 2.0},
    ),
)
def test_warp_affine_canvas_is_omitted_or_a_positive_builtin_int_pair(kwargs: dict[str, Any]) -> None:
    """v1-warp-affine acceptance 2 and 19: canvas dimensions form one strict optional pair."""
    source = _frame(np.zeros((2, 3, 1), dtype=np.float32))
    with pytest.raises(ValueError) as error:
        px.transform.warp_affine(source, np.eye(2, 3), **kwargs)
    _assert_actionable(error)

    same_size = px.transform.warp_affine(source, np.eye(2, 3), interpolation="nearest")
    assert same_size.shape == source.shape


def test_warp_affine_rejects_non_frame_and_non_fp32_inputs_before_sampling() -> None:
    """v1-warp-affine acceptance 3 and 19: Frame and fp32 are fail-fast operation contracts."""
    import cupy as cp

    invalid_inputs: tuple[Any, ...] = (
        cp.zeros((2, 2, 1), dtype=cp.float32),
        _frame(np.zeros((2, 2, 1), dtype=np.float16), dtype=np.float16),
        _frame(np.zeros((2, 2, 1), dtype=np.uint8), dtype=np.uint8),
        _frame(np.zeros((2, 2, 1), dtype=np.uint16), dtype=np.uint16),
    )
    for invalid in invalid_inputs:
        with pytest.raises(ValueError) as error:
            px.transform.warp_affine(invalid, np.eye(2, 3), interpolation="nearest")
        _assert_actionable(error)
        if isinstance(invalid, px.core.Frame) and invalid.dtype != np.dtype(np.float32):
            message = str(error.value)
            if invalid.dtype == np.dtype(np.float16):
                assert "px.values.cast_dtype" in message
            else:
                assert "px.values.recode_dtype" in message
                assert "px.values.dequantize" in message


@pytest.mark.parametrize(
    "matrix",
    (
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        np.eye(3),
        np.ones((2, 2)),
        np.ones((6,)),
        np.ones((2, 3), dtype=np.bool_),
        np.ones((2, 3), dtype=np.complex64),
        np.ones((2, 3), dtype=object),
        np.asarray([[1.0, 0.0, np.inf], [0.0, 1.0, 0.0]]),
        np.asarray([[1.0, 0.0, 1.0e100], [0.0, 1.0, 0.0]]),
    ),
)
def test_warp_affine_matrix_type_shape_dtype_and_finite_domain_are_strict(matrix: Any) -> None:
    """v1-warp-affine acceptance 4 and 19: only finite real NumPy/CuPy 2x3 matrices enter."""
    source = _frame(np.zeros((2, 2, 1), dtype=np.float32))
    with pytest.raises(ValueError) as error:
        px.transform.warp_affine(source, matrix, interpolation="nearest")
    _assert_actionable(error)
    assert "(2, 3)" in str(error.value) or "numpy.ndarray or cupy.ndarray" in str(error.value)


def test_warp_affine_accepts_host_and_device_matrices_without_mutating_them() -> None:
    """v1-warp-affine acceptance 4 and 14: host/device matrices are normalized privately and remain unchanged."""
    import cupy as cp

    source = _frame(np.arange(6, dtype=np.float32).reshape(2, 3, 1))
    host_matrix = np.asarray([[1, 0, 1], [0, 1, 0]], dtype=np.int64)
    device_matrix = cp.asarray(host_matrix, dtype=cp.float64)
    host_before = host_matrix.copy()
    device_before = device_matrix.copy()

    host_result = px.transform.warp_affine(source, host_matrix, interpolation="nearest")
    device_result = px.transform.warp_affine(source, device_matrix, interpolation="nearest")

    np.testing.assert_array_equal(host_result.data.get(), device_result.data.get())
    np.testing.assert_array_equal(host_matrix, host_before)
    np.testing.assert_array_equal(device_matrix.get(), device_before.get())


@pytest.mark.parametrize(
    "matrix",
    (
        np.asarray([[1.0, 2.0, 0.0], [2.0, 4.0, 0.0]], dtype=np.float32),
        np.asarray([[1.0e-39, 0.0, 0.0], [0.0, 1.0e-39, 0.0]], dtype=np.float32),
    ),
)
def test_warp_affine_requires_a_finite_fp32_inverse_without_regularization(matrix: np.ndarray) -> None:
    """v1-warp-affine acceptance 5 and 19: singular and fp32-overflow inverses fail actionably."""
    source = _frame(np.zeros((2, 2, 1), dtype=np.float32))
    with pytest.raises(ValueError) as error:
        px.transform.warp_affine(source, matrix, interpolation="nearest")
    _assert_actionable(error)
    assert "inverse" in str(error.value)


def test_warp_affine_inverse_is_builtin_bool_and_reverses_the_declared_forward_transform() -> None:
    """v1-warp-affine acceptance 6 and 16: one matrix plus inverse composes integer translation exactly."""
    source_values = np.asarray([[[1.0], [2.0], [3.0]]], dtype=np.float32)
    source = _frame(source_values)
    matrix = np.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    shifted = px.transform.warp_affine(source, matrix, width=5, height=1, interpolation="nearest")
    restored = px.transform.warp_affine(
        shifted,
        matrix,
        inverse=True,
        width=3,
        height=1,
        interpolation="nearest",
    )

    np.testing.assert_array_equal(shifted.data.get()[0, :, 0], np.asarray([0.0, 1.0, 2.0, 3.0, 0.0]))
    np.testing.assert_array_equal(restored.data.get(), source_values)
    for invalid in (0, 1, np.bool_(True), "false"):
        with pytest.raises(ValueError) as error:
            px.transform.warp_affine(source, matrix, inverse=invalid, interpolation="nearest")
        _assert_actionable(error)


def test_warp_affine_accepts_all_interpolations_and_auto_uses_effective_forward_column_norms() -> None:
    """v1-warp-affine acceptance 8-9: nine shared tokens and both effective-matrix auto branches work."""
    values = np.linspace(-0.25, 1.25, 4 * 5 * 2, dtype=np.float32).reshape(4, 5, 2)
    source = _frame(values, channels=("left", "right"))
    shrink = np.asarray([[0.75, 0.0, 0.0], [0.0, 1.1, 0.0]], dtype=np.float32)

    for interpolation in INTERPOLATIONS:
        result = px.transform.warp_affine(source, np.eye(2, 3), interpolation=interpolation)
        assert result.shape == source.shape

    automatic_shrink = px.transform.warp_affine(source, shrink)
    explicit_shrink = px.transform.warp_affine(source, shrink, interpolation="area")
    np.testing.assert_array_equal(automatic_shrink.data.get(), explicit_shrink.data.get())

    automatic_inverse = px.transform.warp_affine(source, shrink, inverse=True)
    explicit_inverse = px.transform.warp_affine(source, shrink, inverse=True, interpolation="area")
    np.testing.assert_array_equal(automatic_inverse.data.get(), explicit_inverse.data.get())

    enlarge = np.asarray([[1.1, 0.0, 0.0], [0.0, 1.2, 0.0]], dtype=np.float32)
    automatic_enlarge = px.transform.warp_affine(source, enlarge)
    explicit_enlarge = px.transform.warp_affine(source, enlarge, interpolation="lanczos4")
    np.testing.assert_array_equal(automatic_enlarge.data.get(), explicit_enlarge.data.get())

    for invalid in ("Nearest", "linear", 1, None):
        if invalid is None:
            continue
        with pytest.raises(ValueError) as error:
            px.transform.warp_affine(source, np.eye(2, 3), interpolation=invalid)
        _assert_actionable(error)
        for interpolation in INTERPOLATIONS:
            assert interpolation in str(error.value)


@pytest.mark.parametrize(
    "invalid",
    (
        np.asarray("nearest"),
        np.asarray(["nearest", "area"]),
    ),
)
def test_warp_affine_rejects_array_like_non_string_interpolation_tokens(invalid: np.ndarray) -> None:
    """v1-warp-affine acceptance 8 and 19: every non-str interpolation token fails actionably."""
    source = _frame(np.asarray([[[1.0]]], dtype=np.float32))

    with pytest.raises(ValueError) as error:
        px.transform.warp_affine(source, np.eye(2, 3), interpolation=invalid)

    _assert_actionable(error)
    for interpolation in INTERPOLATIONS:
        assert interpolation in str(error.value)


@pytest.mark.parametrize("interpolation", INTERPOLATIONS[:-1])
@pytest.mark.parametrize("border", BORDERS)
def test_warp_affine_point_kernels_match_an_independent_numpy_oracle(interpolation: str, border: str) -> None:
    """v1-warp-affine acceptance 7-10 and 12: centered fixed-support kernels match independent NumPy."""
    rng = np.random.default_rng(20260804)
    values = rng.uniform(-0.4, 1.6, size=(4, 5, 3)).astype(np.float32)
    source = _frame(values, channels=("temperature", "mask", "depth"))
    matrix = np.asarray([[0.92, 0.18, 0.35], [-0.11, 1.08, 0.28]], dtype=np.float32)
    border_value = -0.35 if border == "constant" else None
    expected = _point_reference(
        values,
        matrix,
        inverse=False,
        width=4,
        height=3,
        interpolation=interpolation,
        border=border,
        border_value=-0.35,
    )

    result = px.transform.warp_affine(
        source,
        matrix,
        width=4,
        height=3,
        interpolation=interpolation,
        border=border,
        border_value=border_value,
    )

    # Eight-tap Lanczos evaluates up to 64 products with GPU sinf and fp32 coordinates.
    np.testing.assert_allclose(result.data.get(), expected, rtol=8e-5, atol=8e-5)


@pytest.mark.parametrize("border", BORDERS)
def test_warp_affine_area_matches_parallelogram_cell_intersection_oracle(border: str) -> None:
    """v1-warp-affine acceptance 11-12: area integrates the inverse-mapped parallelogram over border cells."""
    values = np.arange(4 * 5 * 2, dtype=np.float32).reshape(4, 5, 2) / np.float32(7.0) - np.float32(1.0)
    source = _frame(values, channels=("negative", "high"))
    matrix = np.asarray([[0.72, 0.31, 0.2], [-0.24, 0.83, 0.45]], dtype=np.float32)
    border_value = 1.75 if border == "constant" else None
    expected = _area_reference(
        values,
        matrix,
        inverse=False,
        width=4,
        height=3,
        border=border,
        border_value=1.75,
    )

    result = px.transform.warp_affine(
        source,
        matrix,
        width=4,
        height=3,
        interpolation="area",
        border=border,
        border_value=border_value,
    )

    np.testing.assert_allclose(result.data.get(), expected, rtol=5e-5, atol=5e-5)


def test_warp_affine_area_large_wrap_footprint_averages_nonuniform_period_without_phase_lock() -> None:
    """v1-warp-affine acceptance 11-12: large area footprints integrate the infinite wrap extension."""
    source = _frame(np.asarray([[[0.0], [1.0]]], dtype=np.float32))
    matrix = np.asarray([[1.0 / 8192.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    result = px.transform.warp_affine(
        source,
        matrix,
        width=1,
        height=1,
        interpolation="area",
        border="wrap",
    )

    # The footprint contains exactly 8192 equal-area x cells: 4096 zeros and 4096 ones.
    # The exact 0.5 quotient is representable in fp32; one ulp permits GPU coordinate arithmetic noise.
    np.testing.assert_allclose(
        result.data.get(), np.asarray([[[0.5]]], dtype=np.float32), rtol=0.0, atol=np.finfo(np.float32).eps
    )


def test_warp_affine_area_wrap_preserves_huge_finite_translation_phase() -> None:
    """v1-warp-affine acceptance 4 and 11-12: wrap preserves a huge finite translation's exact phase."""
    values = np.asarray([[[10.0], [20.0], [30.0]]], dtype=np.float32)
    source = _frame(values)
    translation = np.float32(1.0e20)
    matrix = np.asarray([[1.0, 0.0, translation], [0.0, 1.0, 0.0]], dtype=np.float32)

    result = px.transform.warp_affine(
        source,
        matrix,
        width=1,
        height=1,
        interpolation="area",
        border="wrap",
    )

    # fp32 stores this finite translation as an exact integer. Identity area therefore covers
    # exactly the source cell selected by the inverse translation's integer wrap phase.
    source_x = (-int(translation)) % source.width
    expected = values[:, source_x : source_x + 1]
    np.testing.assert_array_equal(result.data.get(), expected)


def test_warp_affine_area_axis_scale_equals_resize_coverage_average() -> None:
    """v1-warp-affine acceptance 11: centered axis scale reduces to resize's coverage-box area result."""
    values = np.arange(6 * 8, dtype=np.float32).reshape(6, 8, 1)
    source = _frame(values)
    output_width, output_height = 4, 3
    scale_x = output_width / source.width
    scale_y = output_height / source.height
    matrix = np.asarray(
        [
            [scale_x, 0.0, (scale_x - 1.0) / 2.0],
            [0.0, scale_y, (scale_y - 1.0) / 2.0],
        ],
        dtype=np.float32,
    )

    warped = px.transform.warp_affine(
        source,
        matrix,
        width=output_width,
        height=output_height,
        interpolation="area",
        border="replicate",
    )
    resized = px.transform.resize(source, width=output_width, height=output_height, interpolation="area")

    np.testing.assert_allclose(warped.data.get(), resized.data.get(), rtol=0.0, atol=2e-6)


def test_warp_affine_extreme_finite_fp32_geometry_terminates_deterministically() -> None:
    """v1-warp-affine acceptance 4-5 and 11: finite fp32 extremes do not overflow index loops or hang."""
    source = _frame(np.asarray([[[2.0]]], dtype=np.float32))
    tiny_scale = np.asarray([[1.0e-38, 0.0, 0.0], [0.0, 1.0e-38, 0.0]], dtype=np.float32)
    huge_translation = np.asarray([[1.0, 0.0, 1.0e30], [0.0, 1.0, -1.0e30]], dtype=np.float32)

    constant_area = px.transform.warp_affine(source, tiny_scale, interpolation="area")
    replicate_area = px.transform.warp_affine(source, tiny_scale, interpolation="area", border="replicate")
    wrapped_area = px.transform.warp_affine(source, tiny_scale, interpolation="area", border="wrap")
    constant_point = px.transform.warp_affine(source, huge_translation, interpolation="nearest")
    wrapped_point = px.transform.warp_affine(source, huge_translation, interpolation="nearest", border="wrap")

    np.testing.assert_array_equal(constant_area.data.get(), np.asarray([[[0.0]]], dtype=np.float32))
    np.testing.assert_array_equal(replicate_area.data.get(), np.asarray([[[2.0]]], dtype=np.float32))
    np.testing.assert_array_equal(wrapped_area.data.get(), np.asarray([[[2.0]]], dtype=np.float32))
    np.testing.assert_array_equal(constant_point.data.get(), np.asarray([[[0.0]]], dtype=np.float32))
    np.testing.assert_array_equal(wrapped_point.data.get(), np.asarray([[[2.0]]], dtype=np.float32))


def test_warp_affine_border_tokens_values_and_one_pixel_axes_follow_the_shared_contract() -> None:
    """v1-warp-affine acceptance 12-13 and 19: border tokens, one-pixel axes, and value pairing are strict."""
    single = _frame(np.asarray([[[2.5]]], dtype=np.float32))
    translation = np.asarray([[1.0, 0.0, 7.0], [0.0, 1.0, -9.0]], dtype=np.float32)
    for border in ("mirror", "replicate", "wrap"):
        result = px.transform.warp_affine(single, translation, interpolation="nearest", border=border)
        np.testing.assert_array_equal(result.data.get(), np.asarray([[[2.5]]], dtype=np.float32))

    default_constant = px.transform.warp_affine(single, translation, interpolation="nearest")
    explicit_constant = px.transform.warp_affine(
        single,
        translation,
        interpolation="nearest",
        border_value=-1.25,
    )
    np.testing.assert_array_equal(default_constant.data.get(), np.asarray([[[0.0]]], dtype=np.float32))
    np.testing.assert_array_equal(explicit_constant.data.get(), np.asarray([[[-1.25]]], dtype=np.float32))

    invalid_calls = (
        {"border": "reflect"},
        {"border": 1},
        {"border": "replicate", "border_value": 0.0},
        {"border": "constant", "border_value": True},
        {"border": "constant", "border_value": np.inf},
        {"border": "constant", "border_value": "0"},
    )
    for kwargs in invalid_calls:
        with pytest.raises(ValueError) as error:
            px.transform.warp_affine(single, np.eye(2, 3), interpolation="nearest", **kwargs)
        _assert_actionable(error)


@pytest.mark.parametrize(
    "invalid",
    (
        np.asarray("wrap"),
        np.asarray(["wrap", "constant"]),
    ),
)
def test_warp_affine_rejects_array_like_non_string_border_tokens(invalid: np.ndarray) -> None:
    """v1-warp-affine acceptance 12 and 19: every non-str border token fails actionably."""
    source = _frame(np.asarray([[[1.0]]], dtype=np.float32))

    with pytest.raises(ValueError) as error:
        px.transform.warp_affine(source, np.eye(2, 3), interpolation="nearest", border=invalid)

    _assert_actionable(error)
    for border in BORDERS:
        assert border in str(error.value)


def test_warp_affine_preserves_metadata_channels_scene_values_and_all_inputs() -> None:
    """v1-warp-affine acceptance 14-15: output is private fp32 HWC and geometry is metadata-neutral."""
    values = np.asarray(
        [
            [[-2.0, 11.0, 101.0], [3.0, 21.0, 201.0]],
            [[5.0, 31.0, 301.0], [7.0, 41.0, 401.0]],
        ],
        dtype=np.float32,
    )
    source = _frame(
        values,
        colorspace="Rec.2020",
        gamma="pq",
        channels=("temperature", "confidence", "Z"),
        matrix="bt2020",
    )
    declared = np.eye(2, 3, dtype=np.float32)
    data_before = source.data.copy()
    metadata_before = (source.colorspace, source.gamma, source.channels, source.matrix)
    matrix_before = declared.copy()

    result = px.transform.warp_affine(source, declared, interpolation="nearest")

    assert isinstance(result, px.core.Frame)
    assert result is not source
    assert result.data.data.ptr != source.data.data.ptr
    assert result.dtype == np.dtype(np.float32)
    assert result.data.flags.c_contiguous
    assert (result.colorspace, result.gamma, result.channels, result.matrix) == metadata_before
    np.testing.assert_array_equal(result.data.get(), values)
    np.testing.assert_array_equal(source.data.get(), data_before.get())
    assert (source.colorspace, source.gamma, source.channels, source.matrix) == metadata_before
    np.testing.assert_array_equal(declared, matrix_before)
    assert float(result.data.min()) == -2.0
    assert float(result.data.max()) == 401.0


def test_warp_affine_identity_distinguishes_interpolating_and_approximating_kernels() -> None:
    """v1-warp-affine acceptance 16: identity preserves interpolating kernels while cubic approximants smooth."""
    rng = np.random.default_rng(19)
    values = rng.uniform(-0.3, 1.4, size=(5, 6, 2)).astype(np.float32)
    source = _frame(values, channels=("first", "second"))

    for interpolation in ("nearest", "bilinear", "bicubic", "lanczos2", "lanczos3", "lanczos4", "area"):
        result = px.transform.warp_affine(source, np.eye(2, 3), interpolation=interpolation, border="replicate")
        np.testing.assert_allclose(result.data.get(), values, rtol=0.0, atol=3e-6)

    for interpolation in ("b-spline", "mitchell"):
        result = px.transform.warp_affine(source, np.eye(2, 3), interpolation=interpolation, border="replicate")
        assert not np.allclose(result.data.get(), values, rtol=0.0, atol=3e-6)


def test_warp_affine_docstring_and_vocabulary_are_self_contained(vocabulary_markdown: str) -> None:
    """v1-warp-affine acceptance 17-19: public docs fix geometry, tokens, defaults, values, and repair paths."""
    docstring = inspect.getdoc(px.transform.warp_affine)
    assert docstring is not None
    for required in (
        "forward",
        "inverse",
        "width and height",
        "pixel centers",
        "area",
        "lanczos4",
        *INTERPOLATIONS,
        *BORDERS,
        "0.0",
        "float32",
        "per channel",
        "metadata",
        "scene",
        "does not mutate",
        "px.values.cast_dtype",
    ):
        assert required in docstring

    interpolation_section = vocabulary_markdown.split("## interpolation\n", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
    border_section = vocabulary_markdown.split("## border\n", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
    for required in (
        "px.transform.warp_affine",
        "column norm",
        "pixel center",
        "inverse mapping",
        "parallelogram",
        "Lanczos",
    ):
        assert required in interpolation_section
    for required in ("px.transform.warp_affine", "0.0", "point", "area", "border_value=None"):
        assert required in border_section
