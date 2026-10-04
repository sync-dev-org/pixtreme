"""Frozen point-warp characterization for issue #56.

Copied from commit 6cabf0f31b3f3392a3824a2869dd6e10dc86a07e, including
resolved interpolation and border CUDA helpers. This reference deliberately
keeps the original arithmetic and must not import production kernel sources.
It characterizes bit patterns, rather than serving as a mathematical oracle.
"""

from functools import lru_cache

import cupy as cp
import numpy as np

POINT_INTERPOLATIONS = ("nearest", "bilinear", "bicubic", "b-spline", "mitchell", "lanczos2", "lanczos3", "lanczos4")
BORDERS = ("mirror", "replicate", "wrap", "constant")

_SOURCE = r"""

__device__ float pixtreme_keys_weight(const float distance) {
    const float x = fabsf(distance);
    const float a = -0.5f;
    if (x < 1.0f) {
        return (a + 2.0f) * x * x * x - (a + 3.0f) * x * x + 1.0f;
    }
    if (x < 2.0f) {
        return a * x * x * x - 5.0f * a * x * x + 8.0f * a * x - 4.0f * a;
    }
    return 0.0f;

}

__device__ float pixtreme_mitchell_weight(const float distance, const float b, const float c) {
    const float x = fabsf(distance);
    if (x < 1.0f) {
        return ((12.0f - 9.0f * b - 6.0f * c) * x * x * x
            + (-18.0f + 12.0f * b + 6.0f * c) * x * x
            + (6.0f - 2.0f * b)) / 6.0f;
    }
    if (x < 2.0f) {
        return ((-b - 6.0f * c) * x * x * x
            + (6.0f * b + 30.0f * c) * x * x
            + (-12.0f * b - 48.0f * c) * x
            + (8.0f * b + 24.0f * c)) / 6.0f;
    }
    return 0.0f;

}

__device__ float pixtreme_lanczos_weight(const float distance, const int lobes) {
    const float x = fabsf(distance);
    if (x == 0.0f) {
        return 1.0f;
    }
    if (x >= (float)lobes) {
        return 0.0f;
    }
    const float pi_x = 3.14159265358979323846f * x;
    return ((float)lobes * sinf(pi_x) * sinf(pi_x / (float)lobes)) / (pi_x * pi_x);

}

__device__ float pixtreme_point_weight(const int interpolation, const float distance) {
    if (interpolation == 1) {
    const float weight = 1.0f - fabsf(distance);
    return weight > 0.0f ? weight : 0.0f;

    }
    if (interpolation == 2) {
        return pixtreme_keys_weight(distance);
    }
    if (interpolation == 3) {
        return pixtreme_mitchell_weight(distance, 1.0f, 0.0f);
    }
    if (interpolation == 4) {
        return pixtreme_mitchell_weight(distance, 1.0f / 3.0f, 1.0f / 3.0f);
    }
    return pixtreme_lanczos_weight(distance, interpolation - 3);
}

__device__ long long pixtreme_positive_modulo(const long long value, const long long modulus) {
    const long long remainder = value % modulus;
    return remainder < 0 ? remainder + modulus : remainder;
}

__device__ long long pixtreme_border_index(
    const long long index,
    const long long extent,
    const int border
) {
    if (extent <= 1) {
        return 0;
    }
    if (border == 1) {
        return index < 0 ? 0 : (index >= extent ? extent - 1 : index);
    }
    if (border == 2) {
        return pixtreme_positive_modulo(index, extent);
    }
    const long long period = 2 * extent - 2;
    const long long reflected = pixtreme_positive_modulo(index, period);
    return reflected < extent ? reflected : period - reflected;
}

template <typename T>
__device__ float pixtreme_border_sample(
    const T& source,
    const long long x,
    const long long y,
    const long long width,
    const long long height,
    const long long channel_count,
    const long long channel,
    const int border,
    const float border_value
) {
    if (border == 3 && (x < 0 || x >= width || y < 0 || y >= height)) {
        return border_value;
    }
    const long long source_x = pixtreme_border_index(x, width, border);
    const long long source_y = pixtreme_border_index(y, height, border);
    const long long source_index = (source_y * width + source_x) * channel_count + channel;
    return (float)source[source_index];
}

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

"""


@lru_cache(maxsize=1)
def _kernel() -> cp.RawKernel:
    return cp.RawKernel(_SOURCE, "pixtreme_warp_affine_point")


def point_reference(
    source: cp.ndarray,
    matrix: np.ndarray,
    *,
    inverse: bool,
    width: int,
    height: int,
    interpolation: str,
    border: str,
    border_value: float,
) -> cp.ndarray:
    """Run the original point kernel with the original fp32 matrix contract."""
    declared = matrix.astype(np.float32)
    if inverse:
        mapping = declared
    else:
        a, b, tx = (float(value) for value in declared[0])
        c, d, ty = (float(value) for value in declared[1])
        determinant = a * d - b * c
        mapping = np.asarray([[d / determinant, -b / determinant, 0.0], [-c / determinant, a / determinant, 0.0]])
        mapping[0, 2] = -(mapping[0, 0] * tx + mapping[0, 1] * ty)
        mapping[1, 2] = -(mapping[1, 0] * tx + mapping[1, 1] * ty)
        mapping = mapping.astype(np.float32)
    input_height, input_width, channels = source.shape
    output = cp.empty((height, width, channels), dtype=cp.float32)
    _kernel()(
        ((width + 15) // 16, (height + 15) // 16),
        (16, 16),
        (
            source,
            output,
            np.int64(input_width),
            np.int64(input_height),
            np.int64(width),
            np.int64(height),
            np.int64(channels),
            *(np.float32(value) for value in mapping.ravel()),
            np.int32(POINT_INTERPOLATIONS.index(interpolation)),
            np.int32(BORDERS.index(border)),
            np.float32(border_value),
        ),
    )
    return output
