"""GPU affine resampling with forward-matrix and infinite-border contracts."""

from __future__ import annotations

import math
from functools import lru_cache
from numbers import Real
from typing import cast

import cupy as cp
import numpy as np

from pixtreme._core.border import _BORDER_PREAMBLE
from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import Frame
from pixtreme._core.interpolation import _POINT_INTERPOLATION_DEVICE_SOURCE, _POINT_INTERPOLATION_TOKENS
from pixtreme._core.validation import _normalized_closed_token
from pixtreme._core.value_domain import _float32_conversion_guidance
from pixtreme._core.vocabulary import _BORDER_TOKENS, Border, Interpolation

_INTERPOLATION_TOKENS = (*_POINT_INTERPOLATION_TOKENS, "area")

_RAW_KERNEL_BLOCK = (16, 16)

_WARP_AFFINE_KERNEL_SOURCE = (
    _POINT_INTERPOLATION_DEVICE_SOURCE
    + _BORDER_PREAMBLE
    + r"""
__device__ float pixtreme_warp_normalize_coordinate(
    const float coordinate,
    const long long extent,
    const int border
) {
    if (border != 3 && extent <= 1) {
        return 0.0f;
    }
    if (border == 1) {
        if (coordinate < -8.0f) {
            return 0.0f;
        }
        if (coordinate > (float)extent + 7.0f) {
            return (float)(extent - 1);
        }
        return coordinate;
    }
    if (border == 2) {
        float reduced = fmodf(coordinate, (float)extent);
        return reduced < 0.0f ? reduced + (float)extent : reduced;
    }
    if (border == 0 && extent > 1) {
        const float period = (float)(2 * extent - 2);
        float reduced = fmodf(coordinate, period);
        return reduced < 0.0f ? reduced + period : reduced;
    }
    return coordinate;
}

__device__ float pixtreme_warp_cell_sample(
    const float* __restrict__ source,
    const float x,
    const float y,
    const long long width,
    const long long height,
    const long long channel_count,
    const long long channel,
    const int border,
    const float border_value
) {
    if (border == 3 && (x < -0.5f || x >= (float)width - 0.5f || y < -0.5f || y >= (float)height - 0.5f)) {
        return border_value;
    }
    const float normalized_x = pixtreme_warp_normalize_coordinate(x, width, border);
    const float normalized_y = pixtreme_warp_normalize_coordinate(y, height, border);
    return pixtreme_border_sample(
        source,
        (long long)floorf(normalized_x + 0.5f),
        (long long)floorf(normalized_y + 0.5f),
        width,
        height,
        channel_count,
        channel,
        border,
        border_value
    );
}

extern "C" __global__ void pixtreme_warp_affine_point(
    const float* __restrict__ source,
    float* __restrict__ output,
    const long long input_width,
    const long long input_height,
    const long long output_width,
    const long long output_height,
    const long long channel_count,
    const float inverse_00,
    const float inverse_01,
    const float inverse_02,
    const float inverse_10,
    const float inverse_11,
    const float inverse_12,
    const int interpolation,
    const int border,
    const float border_value
) {
    const long long output_x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long output_y = (long long)blockIdx.y * blockDim.y + threadIdx.y;
    if (output_x >= output_width || output_y >= output_height) {
        return;
    }
    const float mapped_x = inverse_00 * (float)output_x + inverse_01 * (float)output_y + inverse_02;
    const float mapped_y = inverse_10 * (float)output_x + inverse_11 * (float)output_y + inverse_12;
    const long long output_offset = (output_y * output_width + output_x) * channel_count;

    if (border == 3 && (
        mapped_x < -8.0f || mapped_x > (float)input_width + 7.0f ||
        mapped_y < -8.0f || mapped_y > (float)input_height + 7.0f
    )) {
        for (long long channel = 0; channel < channel_count; ++channel) {
            output[output_offset + channel] = border_value;
        }
        return;
    }
    const float source_x = pixtreme_warp_normalize_coordinate(mapped_x, input_width, border);
    const float source_y = pixtreme_warp_normalize_coordinate(mapped_y, input_height, border);

    if (interpolation == 0) {
        const long long nearest_x = (long long)floorf(source_x + 0.5f);
        const long long nearest_y = (long long)floorf(source_y + 0.5f);
        for (long long channel = 0; channel < channel_count; ++channel) {
            output[output_offset + channel] = pixtreme_border_sample(
                source,
                nearest_x,
                nearest_y,
                input_width,
                input_height,
                channel_count,
                channel,
                border,
                border_value
            );
        }
        return;
    }

    const long long base_x = (long long)floorf(source_x);
    const long long base_y = (long long)floorf(source_y);
    const int lobes = interpolation >= 5 ? interpolation - 3 : 0;
    const int sample_count = interpolation == 1 ? 2 : (lobes > 0 ? 2 * lobes : 4);
    const long long start_x = interpolation == 1 ? base_x : base_x - (lobes > 0 ? lobes - 1 : 1);
    const long long start_y = interpolation == 1 ? base_y : base_y - (lobes > 0 ? lobes - 1 : 1);
    float weights_x[8];
    float weights_y[8];
    float sum_x = 0.0f;
    float sum_y = 0.0f;
    for (int offset = 0; offset < sample_count; ++offset) {
        weights_x[offset] = pixtreme_point_weight(interpolation, source_x - (float)(start_x + offset));
        weights_y[offset] = pixtreme_point_weight(interpolation, source_y - (float)(start_y + offset));
        sum_x += weights_x[offset];
        sum_y += weights_y[offset];
    }
    if (lobes > 0) {
        const float inverse_sum_x = sum_x != 0.0f ? 1.0f / sum_x : 0.0f;
        const float inverse_sum_y = sum_y != 0.0f ? 1.0f / sum_y : 0.0f;
        for (int offset = 0; offset < sample_count; ++offset) {
            weights_x[offset] *= inverse_sum_x;
            weights_y[offset] *= inverse_sum_y;
        }
    }

    for (long long channel = 0; channel < channel_count; ++channel) {
        float value = 0.0f;
        for (int offset_y = 0; offset_y < sample_count; ++offset_y) {
            for (int offset_x = 0; offset_x < sample_count; ++offset_x) {
                value += pixtreme_border_sample(
                    source,
                    start_x + offset_x,
                    start_y + offset_y,
                    input_width,
                    input_height,
                    channel_count,
                    channel,
                    border,
                    border_value
                ) * weights_x[offset_x] * weights_y[offset_y];
            }
        }
        output[output_offset + channel] = value;
    }
}

__device__ int pixtreme_warp_clip_axis(
    const float* __restrict__ input_x,
    const float* __restrict__ input_y,
    const int input_count,
    float* __restrict__ output_x,
    float* __restrict__ output_y,
    const int axis,
    const float boundary,
    const int keep_greater
) {
    if (input_count == 0) {
        return 0;
    }
    int output_count = 0;
    float previous_x = input_x[input_count - 1];
    float previous_y = input_y[input_count - 1];
    const float previous_axis = axis == 0 ? previous_x : previous_y;
    int previous_inside = keep_greater ? previous_axis >= boundary : previous_axis <= boundary;
    for (int index = 0; index < input_count; ++index) {
        const float current_x = input_x[index];
        const float current_y = input_y[index];
        const float current_axis = axis == 0 ? current_x : current_y;
        const int current_inside = keep_greater ? current_axis >= boundary : current_axis <= boundary;
        if (current_inside != previous_inside) {
            const float prior_axis = axis == 0 ? previous_x : previous_y;
            const float fraction = (boundary - prior_axis) / (current_axis - prior_axis);
            output_x[output_count] = previous_x + fraction * (current_x - previous_x);
            output_y[output_count] = previous_y + fraction * (current_y - previous_y);
            ++output_count;
        }
        if (current_inside) {
            output_x[output_count] = current_x;
            output_y[output_count] = current_y;
            ++output_count;
        }
        previous_x = current_x;
        previous_y = current_y;
        previous_inside = current_inside;
    }
    return output_count;
}

__device__ float pixtreme_warp_polygon_area(
    const float* __restrict__ polygon_x,
    const float* __restrict__ polygon_y,
    const int count
) {
    if (count < 3) {
        return 0.0f;
    }
    float twice_area = 0.0f;
    for (int index = 0; index < count; ++index) {
        const int next = index + 1 == count ? 0 : index + 1;
        twice_area += polygon_x[index] * polygon_y[next] - polygon_x[next] * polygon_y[index];
    }
    return 0.5f * fabsf(twice_area);
}

__device__ float pixtreme_warp_cell_overlap(
    const float* __restrict__ quad_x,
    const float* __restrict__ quad_y,
    const long long cell_x,
    const long long cell_y
) {
    float polygon_a_x[12];
    float polygon_a_y[12];
    float polygon_b_x[12];
    float polygon_b_y[12];
    for (int index = 0; index < 4; ++index) {
        polygon_a_x[index] = quad_x[index];
        polygon_a_y[index] = quad_y[index];
    }
    int count = pixtreme_warp_clip_axis(
        polygon_a_x, polygon_a_y, 4, polygon_b_x, polygon_b_y, 0, (float)cell_x - 0.5f, 1
    );
    count = pixtreme_warp_clip_axis(
        polygon_b_x, polygon_b_y, count, polygon_a_x, polygon_a_y, 0, (float)cell_x + 0.5f, 0
    );
    count = pixtreme_warp_clip_axis(
        polygon_a_x, polygon_a_y, count, polygon_b_x, polygon_b_y, 1, (float)cell_y - 0.5f, 1
    );
    count = pixtreme_warp_clip_axis(
        polygon_b_x, polygon_b_y, count, polygon_a_x, polygon_a_y, 1, (float)cell_y + 0.5f, 0
    );
    return pixtreme_warp_polygon_area(polygon_a_x, polygon_a_y, count);
}

__device__ double pixtreme_warp_wrap_cell_antiderivative(
    const double coordinate,
    const long long extent,
    const long long source_index
) {
    const double periods = floor(coordinate / (double)extent);
    const double remainder = coordinate - periods * (double)extent;
    const double partial = fmin(fmax(remainder - (double)source_index, 0.0), 1.0);
    return periods + partial;
}

__device__ double pixtreme_warp_wrap_cell_coverage(
    const double minimum,
    const double maximum,
    const long long extent,
    const long long source_index
) {
    if (extent <= 1) {
        return maximum - minimum;
    }
    return pixtreme_warp_wrap_cell_antiderivative(maximum + 0.5, extent, source_index)
        - pixtreme_warp_wrap_cell_antiderivative(minimum + 0.5, extent, source_index);
}

extern "C" __global__ void pixtreme_warp_affine_area(
    const float* __restrict__ source,
    float* __restrict__ output,
    const long long input_width,
    const long long input_height,
    const long long output_width,
    const long long output_height,
    const long long channel_count,
    const float inverse_00,
    const float inverse_01,
    const float inverse_02,
    const float inverse_10,
    const float inverse_11,
    const float inverse_12,
    const int border,
    const float border_value
) {
    const long long output_x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long output_y = (long long)blockIdx.y * blockDim.y + threadIdx.y;
    if (output_x >= output_width || output_y >= output_height) {
        return;
    }
    const float destination_x[4] = {
        (float)output_x - 0.5f,
        (float)output_x + 0.5f,
        (float)output_x + 0.5f,
        (float)output_x - 0.5f
    };
    const float destination_y[4] = {
        (float)output_y - 0.5f,
        (float)output_y - 0.5f,
        (float)output_y + 0.5f,
        (float)output_y + 0.5f
    };
    float quad_x[4];
    float quad_y[4];
    float minimum_x = 3.402823466e+38f;
    float maximum_x = -3.402823466e+38f;
    float minimum_y = 3.402823466e+38f;
    float maximum_y = -3.402823466e+38f;
    for (int corner = 0; corner < 4; ++corner) {
        quad_x[corner] = inverse_00 * destination_x[corner] + inverse_01 * destination_y[corner] + inverse_02;
        quad_y[corner] = inverse_10 * destination_x[corner] + inverse_11 * destination_y[corner] + inverse_12;
        minimum_x = fminf(minimum_x, quad_x[corner]);
        maximum_x = fmaxf(maximum_x, quad_x[corner]);
        minimum_y = fminf(minimum_y, quad_y[corner]);
        maximum_y = fmaxf(maximum_y, quad_y[corner]);
    }
    const float footprint_area = fabsf(inverse_00 * inverse_11 - inverse_01 * inverse_10);
    const long long output_offset = (output_y * output_width + output_x) * channel_count;
    for (long long channel = 0; channel < channel_count; ++channel) {
        output[output_offset + channel] = 0.0f;
    }
    const double candidate_width = ceil((double)maximum_x + 0.5) - floor((double)minimum_x + 0.5);
    const double candidate_height = ceil((double)maximum_y + 0.5) - floor((double)minimum_y + 0.5);
    const double candidate_cells = candidate_width * candidate_height;
    if (
        !isfinite(candidate_cells) || candidate_cells > 4096.0 ||
        fabs((double)minimum_x) > 9.0e18 || fabs((double)maximum_x) > 9.0e18 ||
        fabs((double)minimum_y) > 9.0e18 || fabs((double)maximum_y) > 9.0e18
    ) {
        const int wrap_source_is_bounded = input_width <= 4096 && input_height <= 4096 / input_width;
        if (border == 2 && inverse_01 == 0.0f && inverse_10 == 0.0f && wrap_source_is_bounded) {
            const double exact_area = fabs(
                (double)inverse_00 * (double)inverse_11 - (double)inverse_01 * (double)inverse_10
            );
            // Adding fp32 corner offsets to a huge translation can collapse the footprint bounds.
            // Reduce the periodic phase first, then rebuild the axis-aligned bounds in double.
            const double wrapped_translation_x = fmod((double)inverse_02, (double)input_width);
            const double wrapped_translation_y = fmod((double)inverse_12, (double)input_height);
            const double endpoint_x_0 =
                (double)inverse_00 * ((double)output_x - 0.5) + wrapped_translation_x;
            const double endpoint_x_1 =
                (double)inverse_00 * ((double)output_x + 0.5) + wrapped_translation_x;
            const double endpoint_y_0 =
                (double)inverse_11 * ((double)output_y - 0.5) + wrapped_translation_y;
            const double endpoint_y_1 =
                (double)inverse_11 * ((double)output_y + 0.5) + wrapped_translation_y;
            const double exact_minimum_x = fmin(endpoint_x_0, endpoint_x_1);
            const double exact_maximum_x = fmax(endpoint_x_0, endpoint_x_1);
            const double exact_minimum_y = fmin(endpoint_y_0, endpoint_y_1);
            const double exact_maximum_y = fmax(endpoint_y_0, endpoint_y_1);
            for (long long source_y = 0; source_y < input_height; ++source_y) {
                const double weight_y = pixtreme_warp_wrap_cell_coverage(
                    exact_minimum_y, exact_maximum_y, input_height, source_y
                );
                for (long long source_x = 0; source_x < input_width; ++source_x) {
                    const double weight_x = pixtreme_warp_wrap_cell_coverage(
                        exact_minimum_x, exact_maximum_x, input_width, source_x
                    );
                    const float normalized_weight = (float)(weight_x * weight_y / exact_area);
                    for (long long channel = 0; channel < channel_count; ++channel) {
                        output[output_offset + channel] += source[
                            (source_y * input_width + source_x) * channel_count + channel
                        ] * normalized_weight;
                    }
                }
            }
            return;
        }

        // A regular destination grid can phase-lock to a periodic border. This two-dimensional
        // low-discrepancy sequence has no fixed lattice spacing and keeps the fallback bounded.
        const int sample_count = 4096;
        const float inverse_sample_count = 1.0f / (float)sample_count;
        for (int sample = 0; sample < sample_count; ++sample) {
            const double sequence_index = (double)sample + 0.5;
            const float destination_sample_x = (float)(
                (double)output_x - 0.5 + fmod(sequence_index * 0.7548776662466927, 1.0)
            );
            const float destination_sample_y = (float)(
                (double)output_y - 0.5 + fmod(sequence_index * 0.5698402909980532, 1.0)
            );
            const float source_sample_x =
                inverse_00 * destination_sample_x + inverse_01 * destination_sample_y + inverse_02;
            const float source_sample_y =
                inverse_10 * destination_sample_x + inverse_11 * destination_sample_y + inverse_12;
            for (long long channel = 0; channel < channel_count; ++channel) {
                output[output_offset + channel] += pixtreme_warp_cell_sample(
                    source,
                    source_sample_x,
                    source_sample_y,
                    input_width,
                    input_height,
                    channel_count,
                    channel,
                    border,
                    border_value
                ) * inverse_sample_count;
            }
        }
        return;
    }
    const long long start_x = (long long)floorf(minimum_x + 0.5f);
    const long long stop_x = (long long)ceilf(maximum_x + 0.5f);
    const long long start_y = (long long)floorf(minimum_y + 0.5f);
    const long long stop_y = (long long)ceilf(maximum_y + 0.5f);
    for (long long source_y = start_y; source_y < stop_y; ++source_y) {
        for (long long source_x = start_x; source_x < stop_x; ++source_x) {
            const float overlap = pixtreme_warp_cell_overlap(quad_x, quad_y, source_x, source_y);
            if (overlap == 0.0f) {
                continue;
            }
            for (long long channel = 0; channel < channel_count; ++channel) {
                output[output_offset + channel] += pixtreme_border_sample(
                    source,
                    source_x,
                    source_y,
                    input_width,
                    input_height,
                    channel_count,
                    channel,
                    border,
                    border_value
                ) * overlap;
            }
        }
    }
    const float inverse_area = 1.0f / footprint_area;
    for (long long channel = 0; channel < channel_count; ++channel) {
        output[output_offset + channel] *= inverse_area;
    }
}
"""
)


@lru_cache(maxsize=1)
def _point_kernel() -> cp.RawKernel:
    return cp.RawKernel(_WARP_AFFINE_KERNEL_SOURCE, "pixtreme_warp_affine_point")


@lru_cache(maxsize=1)
def _area_kernel() -> cp.RawKernel:
    return cp.RawKernel(_WARP_AFFINE_KERNEL_SOURCE, "pixtreme_warp_affine_area")


def _validate_frame(frame: object) -> Frame:
    if not isinstance(frame, Frame):
        raise ValueError(
            _actionable_error(
                why="warp_affine operates on metadata-bearing Frame values only",
                what=f"received {type(frame).__module__}.{type(frame).__qualname__}",
                how="construct a Frame with px.io.from_array before calling px.transform.warp_affine",
            )
        )
    if frame.dtype != np.dtype(np.float32):
        raise ValueError(
            _actionable_error(
                why="warp_affine evaluates fp32 working values only",
                what=f"received Frame dtype {frame.dtype.name!r}",
                how=_float32_conversion_guidance(frame.dtype),
            )
        )
    return frame


def _resolve_canvas(frame: Frame, *, width: object, height: object) -> tuple[int, int]:
    if width is None and height is None:
        return frame.width, frame.height
    if width is None or height is None:
        raise ValueError(
            _actionable_error(
                why="warp_affine canvas dimensions are an optional pair",
                what=f"received width={width!r}, height={height!r}",
                how="omit both dimensions, or pass positive built-in int width and height together",
            )
        )
    if type(width) is not int or type(height) is not int or width < 1 or height < 1:
        raise ValueError(
            _actionable_error(
                why="warp_affine canvas dimensions must be positive built-in integers",
                what=f"received width={width!r}, height={height!r}",
                how="pass built-in int width and height values of at least 1",
            )
        )
    return width, height


def _matrix_host_copy(matrix: object) -> np.ndarray:
    if isinstance(matrix, np.ndarray):
        return np.array(matrix, copy=True)
    if isinstance(matrix, cp.ndarray):
        return cast(np.ndarray, cp.asnumpy(matrix))
    raise ValueError(
        _actionable_error(
            why="warp_affine matrix has exactly two supported array containers",
            what=f"received {type(matrix).__module__}.{type(matrix).__qualname__}",
            how="pass a numpy.ndarray or cupy.ndarray with shape (2, 3) and finite real numeric coefficients",
        )
    )


def _normalize_matrix(matrix: object) -> tuple[np.ndarray, np.ndarray]:
    host = _matrix_host_copy(matrix)
    if host.shape != (2, 3):
        raise ValueError(
            _actionable_error(
                why="warp_affine requires a complete affine matrix without silent slicing",
                what=f"received matrix shape {host.shape!r}",
                how="pass a numpy.ndarray or cupy.ndarray with shape (2, 3)",
            )
        )
    dtype = np.dtype(host.dtype)
    if dtype == np.dtype(np.bool_) or np.issubdtype(dtype, np.complexfloating) or not np.issubdtype(dtype, np.number):
        raise ValueError(
            _actionable_error(
                why="warp_affine matrix coefficients must have a real numeric dtype",
                what=f"received matrix dtype {dtype.name!r}",
                how="pass a numpy.ndarray or cupy.ndarray with shape (2, 3) and real int or float coefficients",
            )
        )
    if not bool(np.all(np.isfinite(host))):
        raise ValueError(
            _actionable_error(
                why="warp_affine matrix coefficients must all be finite",
                what=f"received matrix dtype {dtype.name!r} containing a non-finite coefficient",
                how="pass a finite real numeric matrix with shape (2, 3)",
            )
        )
    with np.errstate(over="ignore", invalid="ignore"):
        declared = host.astype(np.float32)
    if not bool(np.all(np.isfinite(declared))):
        raise ValueError(
            _actionable_error(
                why="warp_affine evaluates matrix coefficients as finite fp32 geometry",
                what="at least one finite input coefficient is not representable as finite float32",
                how="pass finite real coefficients within the float32 range in a matrix with shape (2, 3)",
            )
        )

    a, b, translation_x = (float(value) for value in declared[0])
    c, d, translation_y = (float(value) for value in declared[1])
    determinant = a * d - b * c
    if determinant == 0.0:
        raise ValueError(
            _actionable_error(
                why="warp_affine needs an invertible 2x2 linear part for destination-driven inverse mapping",
                what=f"received determinant {determinant!r}",
                how="pass a matrix with shape (2, 3), a nonsingular linear part, and a finite fp32 inverse",
            )
        )
    inverse = np.asarray(
        [
            [d / determinant, -b / determinant, 0.0],
            [-c / determinant, a / determinant, 0.0],
        ],
        dtype=np.float64,
    )
    inverse[0, 2] = -(inverse[0, 0] * translation_x + inverse[0, 1] * translation_y)
    inverse[1, 2] = -(inverse[1, 0] * translation_x + inverse[1, 1] * translation_y)
    with np.errstate(over="ignore", invalid="ignore"):
        normalized_inverse = inverse.astype(np.float32)
    if not bool(np.all(np.isfinite(inverse))) or not bool(np.all(np.isfinite(normalized_inverse))):
        raise ValueError(
            _actionable_error(
                why="warp_affine inverse mapping coefficients must be finite fp32 values",
                what="the declared matrix is nonsingular but its inverse exceeds the finite float32 domain",
                how="pass a matrix with shape (2, 3), a nonsingular linear part, and a finite fp32 inverse",
            )
        )
    return declared, normalized_inverse


def _resolve_interpolation(effective: np.ndarray, interpolation: object) -> str:
    if interpolation is None:
        scale_x = float(np.hypot(effective[0, 0], effective[1, 0]))
        scale_y = float(np.hypot(effective[0, 1], effective[1, 1]))
        return "area" if scale_x < 1.0 or scale_y < 1.0 else "lanczos4"
    return _normalized_closed_token(
        interpolation,
        axis="interpolation",
        accepted=_INTERPOLATION_TOKENS,
        how=f"pass one of the canonical tokens {_INTERPOLATION_TOKENS!r}, or omit interpolation for auto selection",
    )


def _resolve_border(border: object, border_value: object) -> tuple[str, float]:
    checked_border = _normalized_closed_token(border, axis="border", accepted=_BORDER_TOKENS)
    if checked_border == "constant":
        if border_value is None:
            return checked_border, 0.0
        if isinstance(border_value, bool) or not isinstance(border_value, Real):
            raise ValueError(
                _actionable_error(
                    why="constant warp border_value must be None or a finite real number",
                    what=f"received border_value={border_value!r}",
                    how="omit border_value for 0.0, or pass a finite int or float; negative values and values above 1 are allowed",
                )
            )
        resolved = float(border_value)
        if not math.isfinite(resolved):
            raise ValueError(
                _actionable_error(
                    why="constant warp border_value must be finite",
                    what=f"received border_value={border_value!r}",
                    how="omit border_value for 0.0, or pass a finite int or float; negative values and values above 1 are allowed",
                )
            )
        return checked_border, resolved
    if border_value is not None:
        raise ValueError(
            _actionable_error(
                why="border_value applies only to warp_affine border='constant'",
                what=f"received border={checked_border!r} with border_value={border_value!r}",
                how="omit border_value, or use border='constant' with None or a finite real value",
            )
        )
    return checked_border, 0.0


def _launch(
    kernel: cp.RawKernel,
    *,
    width: int,
    height: int,
    arguments: tuple[object, ...],
) -> None:
    block_x, block_y = _RAW_KERNEL_BLOCK
    grid = ((width + block_x - 1) // block_x, (height + block_y - 1) // block_y)
    kernel(grid, _RAW_KERNEL_BLOCK, arguments)


def warp_affine(
    frame: Frame,
    matrix: np.ndarray | cp.ndarray,
    *,
    inverse: bool = False,
    width: int | None = None,
    height: int | None = None,
    interpolation: Interpolation | None = None,
    border: Border = "constant",
    border_value: float | None = None,
) -> Frame:
    """Warp a float32 Frame with a declared forward 2x3 affine matrix.

    ``matrix`` maps input pixel centers to output pixel centers, whose top-left
    centers are both ``(0, 0)``. Destination-driven inverse mapping samples the
    source at ``inverse(T) @ (x, y, 1)``, where ``T`` is the forward matrix when
    ``inverse=False`` and its inverse when ``inverse=True``. Pass width and height
    together for another canvas, or omit both to retain the input dimensions.

    ``interpolation`` accepts ``nearest``, ``bilinear``, ``bicubic``, ``b-spline``,
    ``mitchell``, ``lanczos2``, ``lanczos3``, ``lanczos4``, and ``area``. Omission
    selects ``area`` when either effective forward column norm is below one and
    ``lanczos4`` otherwise. Point kernels have fixed support; area averages the
    inverse-mapped pixel-cell parallelogram. ``border`` accepts ``mirror``,
    ``replicate``, ``wrap``, and ``constant``. Constant ``border_value=None`` means
    0.0; explicit finite scene values may be negative or above one.

    Resampling is float32 and independent per channel. It does not clamp scene
    values or specialize alpha and preserves colorspace, gamma, channels, and
    Frame ``matrix`` metadata. The result always owns new C-contiguous GPU storage.
    This function does not mutate the input Frame, its data or metadata, or the
    host/device geometry matrix. Convert other storage deliberately with
    ``px.values.cast_dtype``, ``px.values.recode_dtype``, or
    ``px.values.dequantize`` before calling it.
    """
    checked_frame = _validate_frame(frame)
    if type(inverse) is not bool:
        raise ValueError(
            _actionable_error(
                why="warp_affine inverse selects one of two exact matrix directions",
                what=f"received inverse={inverse!r}",
                how="pass the built-in bool False for the declared forward direction or True for its inverse",
            )
        )
    output_width, output_height = _resolve_canvas(checked_frame, width=width, height=height)
    declared, declared_inverse = _normalize_matrix(matrix)
    effective = declared_inverse if inverse else declared
    inverse_mapping = declared if inverse else declared_inverse
    checked_interpolation = _resolve_interpolation(effective, interpolation)
    checked_border, checked_border_value = _resolve_border(border, border_value)

    channel_count = len(checked_frame.channels)
    output = cp.empty((output_height, output_width, channel_count), dtype=cp.float32)
    common_arguments: tuple[object, ...] = (
        checked_frame.data,
        output,
        np.int64(checked_frame.width),
        np.int64(checked_frame.height),
        np.int64(output_width),
        np.int64(output_height),
        np.int64(channel_count),
        *(np.float32(value) for value in inverse_mapping.ravel()),
    )
    border_index = _BORDER_TOKENS.index(checked_border)
    if checked_interpolation == "area":
        _launch(
            _area_kernel(),
            width=output_width,
            height=output_height,
            arguments=(
                *common_arguments,
                np.int32(border_index),
                np.float32(checked_border_value),
            ),
        )
    else:
        _launch(
            _point_kernel(),
            width=output_width,
            height=output_height,
            arguments=(
                *common_arguments,
                np.int32(_INTERPOLATION_TOKENS.index(checked_interpolation)),
                np.int32(border_index),
                np.float32(checked_border_value),
            ),
        )
    return Frame(
        data=output,
        colorspace=checked_frame.colorspace,
        gamma=checked_frame.gamma,
        channels=checked_frame.channels,
        matrix=checked_frame.matrix,
    )
