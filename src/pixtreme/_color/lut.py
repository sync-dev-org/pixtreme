"""GPU-resident one- and three-dimensional LUT transforms."""

from __future__ import annotations

from functools import lru_cache

import cupy as cp
import numpy as np

from pixtreme._color._lut_cuda import _LUT_TETRAHEDRAL_CUDA_SOURCE
from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import Frame
from pixtreme._core.lut import Lut, Lut1D
from pixtreme._core.validation import _normalized_closed_token
from pixtreme._core.value_domain import _float32_conversion_guidance
from pixtreme._core.vocabulary import _INTERPOLATION_TOKENS, Interpolation

_RGB_CHANNELS = ("R", "G", "B")
_LUT_3D_INTERPOLATION_TOKENS = _INTERPOLATION_TOKENS[9:11]
_LUT_1D_INTERPOLATION_TOKENS = _INTERPOLATION_TOKENS[11:]
_LUT_INTERPOLATION_TOKENS = _LUT_3D_INTERPOLATION_TOKENS + _LUT_1D_INTERPOLATION_TOKENS


_LUT_AFFINE_COORDINATE_CUDA_SOURCE = r"""
__device__ __forceinline__ void pixtreme_lut_affine_coordinate(
    const float value,
    const double domain_min,
    const double domain_max,
    const double scale,
    const double offset,
    const int size,
    int* lower,
    float* fraction
) {
    double position;
    if (isnan(value) || (double)value <= domain_min) {
        position = 0.0;
    } else if ((double)value >= domain_max) {
        position = (double)(size - 1);
    } else if (isinf(scale)) {
        // Such a narrow domain can contain no nonzero float32 value; offset is the zero coordinate.
        position = offset;
    } else {
        position = fma((double)value, scale, offset);
    }
    position = fmin(fmax(position, 0.0), (double)(size - 1));
    *lower = min((int)floor(position), size - 2);
    *fraction = (float)(position - *lower);
}
"""


_LUT_TRANSFORM_KERNEL_SOURCE = (
    _LUT_TETRAHEDRAL_CUDA_SOURCE
    + _LUT_AFFINE_COORDINATE_CUDA_SOURCE
    + r"""
__device__ __forceinline__ float3 pixtreme_lut_mix(const float3 lower, const float3 upper, const float fraction) {
    return make_float3(
        lower.x + fraction * (upper.x - lower.x),
        lower.y + fraction * (upper.y - lower.y),
        lower.z + fraction * (upper.z - lower.z)
    );
}

__device__ __forceinline__ float3 pixtreme_lut_trilinear(
    const float* __restrict__ lut,
    const long long stride_r,
    const long long stride_g,
    const long long stride_b,
    const long long stride_c,
    const int red,
    const int green,
    const int blue,
    const float red_fraction,
    const float green_fraction,
    const float blue_fraction,
    const int packed
) {
    const long long offset000 =
        (long long)red * stride_r + (long long)green * stride_g + (long long)blue * stride_b;
    const float3 c000 = pixtreme_lut_load(lut, offset000, stride_c, packed);
    const float3 c100 = pixtreme_lut_load(lut, offset000 + stride_r, stride_c, packed);
    const float3 c010 = pixtreme_lut_load(lut, offset000 + stride_g, stride_c, packed);
    const float3 c110 = pixtreme_lut_load(lut, offset000 + stride_r + stride_g, stride_c, packed);
    const float3 c001 = pixtreme_lut_load(lut, offset000 + stride_b, stride_c, packed);
    const float3 c101 = pixtreme_lut_load(lut, offset000 + stride_r + stride_b, stride_c, packed);
    const float3 c011 = pixtreme_lut_load(lut, offset000 + stride_g + stride_b, stride_c, packed);
    const float3 c111 = pixtreme_lut_load(lut, offset000 + stride_r + stride_g + stride_b, stride_c, packed);
    const float3 c00 = pixtreme_lut_mix(c000, c100, red_fraction);
    const float3 c10 = pixtreme_lut_mix(c010, c110, red_fraction);
    const float3 c01 = pixtreme_lut_mix(c001, c101, red_fraction);
    const float3 c11 = pixtreme_lut_mix(c011, c111, red_fraction);
    return pixtreme_lut_mix(
        pixtreme_lut_mix(c00, c10, green_fraction),
        pixtreme_lut_mix(c01, c11, green_fraction),
        blue_fraction
    );
}

extern "C" __global__ void pixtreme_lut_transform(
    const float* __restrict__ input,
    float* __restrict__ output,
    const long long pixel_count,
    const int channel_count,
    const int red_index,
    const int green_index,
    const int blue_index,
    const float* __restrict__ lut,
    const int lut_size,
    const long long stride_r,
    const long long stride_g,
    const long long stride_b,
    const long long stride_c,
    const double domain_min_r,
    const double domain_min_g,
    const double domain_min_b,
    const double domain_max_r,
    const double domain_max_g,
    const double domain_max_b,
    const double scale_r,
    const double scale_g,
    const double scale_b,
    const double offset_r,
    const double offset_g,
    const double offset_b,
    const int interpolation
) {
    const long long pixel = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (pixel >= pixel_count) {
        return;
    }
    const long long base = pixel * channel_count;
    for (int channel = 0; channel < channel_count; ++channel) {
        if (channel != red_index && channel != green_index && channel != blue_index) {
            output[base + channel] = input[base + channel];
        }
    }

    int red;
    int green;
    int blue;
    float red_fraction;
    float green_fraction;
    float blue_fraction;
    pixtreme_lut_affine_coordinate(
        input[base + red_index],
        domain_min_r,
        domain_max_r,
        scale_r,
        offset_r,
        lut_size,
        &red,
        &red_fraction
    );
    pixtreme_lut_affine_coordinate(
        input[base + green_index],
        domain_min_g,
        domain_max_g,
        scale_g,
        offset_g,
        lut_size,
        &green,
        &green_fraction
    );
    pixtreme_lut_affine_coordinate(
        input[base + blue_index],
        domain_min_b,
        domain_max_b,
        scale_b,
        offset_b,
        lut_size,
        &blue,
        &blue_fraction
    );
    const int packed =
        stride_b == 4
        && stride_c == 1
        && stride_r % 4 == 0
        && stride_g % 4 == 0
        && (reinterpret_cast<unsigned long long>(lut) & 15ULL) == 0;
    const float3 transformed = interpolation == 0
        ? pixtreme_lut_trilinear(
            lut,
            stride_r,
            stride_g,
            stride_b,
            stride_c,
            red,
            green,
            blue,
            red_fraction,
            green_fraction,
            blue_fraction,
            packed
        )
        : pixtreme_lut_tetrahedral(
            lut,
            stride_r,
            stride_g,
            stride_b,
            stride_c,
            red,
            green,
            blue,
            red_fraction,
            green_fraction,
            blue_fraction,
            packed
        );
    output[base + red_index] = transformed.x;
    output[base + green_index] = transformed.y;
    output[base + blue_index] = transformed.z;
}
"""
)


_LUT1D_TRANSFORM_KERNEL_SOURCE = (
    _LUT_AFFINE_COORDINATE_CUDA_SOURCE
    + r"""
__device__ __forceinline__ float pixtreme_lut1d_lookup(
    const float* __restrict__ lut,
    const long long stride_sample,
    const long long stride_channel,
    const int channel,
    const int lower,
    const float fraction
) {
    const long long offset = (long long)lower * stride_sample + (long long)channel * stride_channel;
    const float lower_value = lut[offset];
    const float upper_value = lut[offset + stride_sample];
    return lower_value + fraction * (upper_value - lower_value);
}

extern "C" __global__ void pixtreme_lut1d_transform(
    const float* __restrict__ input,
    float* __restrict__ output,
    const long long pixel_count,
    const int channel_count,
    const int red_index,
    const int green_index,
    const int blue_index,
    const float* __restrict__ lut,
    const int lut_size,
    const long long stride_sample,
    const long long stride_channel,
    const double domain_min_r,
    const double domain_min_g,
    const double domain_min_b,
    const double domain_max_r,
    const double domain_max_g,
    const double domain_max_b,
    const double scale_r,
    const double scale_g,
    const double scale_b,
    const double offset_r,
    const double offset_g,
    const double offset_b
) {
    const long long pixel = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (pixel >= pixel_count) {
        return;
    }
    const long long base = pixel * channel_count;
    for (int channel = 0; channel < channel_count; ++channel) {
        if (channel != red_index && channel != green_index && channel != blue_index) {
            output[base + channel] = input[base + channel];
        }
    }

    int red;
    int green;
    int blue;
    float red_fraction;
    float green_fraction;
    float blue_fraction;
    pixtreme_lut_affine_coordinate(
        input[base + red_index],
        domain_min_r,
        domain_max_r,
        scale_r,
        offset_r,
        lut_size,
        &red,
        &red_fraction
    );
    pixtreme_lut_affine_coordinate(
        input[base + green_index],
        domain_min_g,
        domain_max_g,
        scale_g,
        offset_g,
        lut_size,
        &green,
        &green_fraction
    );
    pixtreme_lut_affine_coordinate(
        input[base + blue_index],
        domain_min_b,
        domain_max_b,
        scale_b,
        offset_b,
        lut_size,
        &blue,
        &blue_fraction
    );
    output[base + red_index] = pixtreme_lut1d_lookup(
        lut, stride_sample, stride_channel, 0, red, red_fraction
    );
    output[base + green_index] = pixtreme_lut1d_lookup(
        lut, stride_sample, stride_channel, 1, green, green_fraction
    );
    output[base + blue_index] = pixtreme_lut1d_lookup(
        lut, stride_sample, stride_channel, 2, blue, blue_fraction
    );
}
"""
)


@lru_cache(maxsize=1)
def _lut_transform_kernel() -> cp.RawKernel:
    return cp.RawKernel(_LUT_TRANSFORM_KERNEL_SOURCE, "pixtreme_lut_transform")


@lru_cache(maxsize=1)
def _lut1d_transform_kernel() -> cp.RawKernel:
    return cp.RawKernel(_LUT1D_TRANSFORM_KERNEL_SOURCE, "pixtreme_lut1d_transform")


def _lut_domain_arguments(lut: Lut | Lut1D) -> tuple[np.float64, ...]:
    """Build float64 clamp endpoints and affine coefficients without overflowing a finite domain span."""
    size_minus_one = np.float64(lut.data.shape[0] - 1)
    scales: list[np.float64] = []
    offsets: list[np.float64] = []
    with np.errstate(over="ignore", under="ignore"):
        for lower, upper in zip(lut.domain_min, lut.domain_max):
            lower64 = np.float64(lower)
            upper64 = np.float64(upper)
            span = upper64 - lower64
            if np.isinf(span):
                # Opposite-sign endpoints can have a finite ratio even when their direct span overflows.
                magnitude = max(abs(lower64), abs(upper64))
                scaled_lower = lower64 / magnitude
                scaled_upper = upper64 / magnitude
                scaled_span = scaled_upper - scaled_lower
                scales.append((size_minus_one / magnitude) / scaled_span)
                offsets.append((-scaled_lower / scaled_span) * size_minus_one)
            else:
                scales.append(size_minus_one / span)
                offsets.append((-lower64 / span) * size_minus_one)
    return (
        *(np.float64(value) for value in lut.domain_min),
        *(np.float64(value) for value in lut.domain_max),
        *scales,
        *offsets,
    )


def apply_lut(
    frame: Frame,
    *,
    lut: Lut | Lut1D,
    interpolation: Interpolation | None = None,
) -> Frame:
    """Apply a user-provided 1D or 3D LUT to a float32 Frame's RGB labels.

    User LUT values do not declare color meaning, so Frame colorspace and gamma
    metadata pass through unchanged. Other channel labels pass through by value.
    Lookup coordinates use the LUT's per-channel domain with input clamp at its
    endpoints; output values are not clipped. ``None`` selects tetrahedral for a
    3D ``Lut`` and linear for a ``Lut1D``. No shaper stage is implied. Neither
    input is mutated, and the result owns new C-contiguous Frame storage.
    """
    if not isinstance(frame, Frame):
        raise ValueError(
            _actionable_error(
                why="apply_lut operates on metadata-bearing Frame values",
                what=f"received {type(frame).__module__}.{type(frame).__qualname__}",
                how="construct a Frame with px.io.from_array before applying the LUT",
            )
        )
    if not isinstance(lut, (Lut, Lut1D)):
        raise ValueError(
            _actionable_error(
                why="apply_lut requires a validated Lut or Lut1D",
                what=f"received {type(lut).__module__}.{type(lut).__qualname__}",
                how="construct px.core.Lut(...) or px.core.Lut1D(...) directly, or call px.io.read_lut(path)",
            )
        )
    dtype = np.dtype(frame.data.dtype)
    if dtype != np.dtype(np.float32):
        raise ValueError(
            _actionable_error(
                why="apply_lut requires float32 Frame data",
                what=f"received Frame dtype {dtype.name!r}",
                how=_float32_conversion_guidance(dtype),
            )
        )
    if not all(label in frame.channels for label in _RGB_CHANNELS):
        raise ValueError(
            _actionable_error(
                why="apply_lut requires channels containing R, G, and B",
                what=f"received channels={frame.channels!r}",
                how="provide all three RGB labels or transform the channel set first",
            )
        )
    if isinstance(lut, Lut):
        allowed_interpolations = _LUT_3D_INTERPOLATION_TOKENS
        resolved_interpolation = "tetrahedral" if interpolation is None else interpolation
    else:
        allowed_interpolations = _LUT_1D_INTERPOLATION_TOKENS
        resolved_interpolation = "linear" if interpolation is None else interpolation
    resolved_interpolation = _normalized_closed_token(
        resolved_interpolation,
        axis="interpolation",
        accepted=allowed_interpolations,
        why=f"apply_lut interpolation must belong to the {type(lut).__name__} token subset",
        how=f"pass one of the canonical tokens {allowed_interpolations!r} or None for the type-specific default",
    )

    output = cp.empty_like(frame.data)
    pixel_count = int(frame.height * frame.width)
    if pixel_count:
        threads_per_block = 256
        block_count = (pixel_count + threads_per_block - 1) // threads_per_block
        common_arguments = (
            frame.data,
            output,
            np.int64(pixel_count),
            np.int32(frame.data.shape[2]),
            np.int32(frame.channels.index("R")),
            np.int32(frame.channels.index("G")),
            np.int32(frame.channels.index("B")),
            lut.data,
            np.int32(lut.data.shape[0]),
        )
        itemsize = int(lut.data.dtype.itemsize)
        strides = tuple(np.int64(int(stride) // itemsize) for stride in lut.data.strides)
        if isinstance(lut, Lut):
            domain_arguments_3d = _lut_domain_arguments(lut)
            _lut_transform_kernel()(
                (block_count,),
                (threads_per_block,),
                (
                    *common_arguments,
                    *strides,
                    *domain_arguments_3d,
                    np.int32(_LUT_3D_INTERPOLATION_TOKENS.index(resolved_interpolation)),
                ),
            )
        else:
            domain_arguments_1d = _lut_domain_arguments(lut)
            _lut1d_transform_kernel()(
                (block_count,),
                (threads_per_block,),
                (*common_arguments, *strides, *domain_arguments_1d),
            )
    return Frame(
        data=output,
        colorspace=frame.colorspace,
        gamma=frame.gamma,
        channels=frame.channels,
        matrix=frame.matrix,
    )
