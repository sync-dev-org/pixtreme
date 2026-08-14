"""GPU-resident three-dimensional LUT type and transform."""

from __future__ import annotations

from functools import lru_cache

import cupy as cp
import numpy as np

from pixtreme._color._lut_cuda import _LUT_TETRAHEDRAL_CUDA_SOURCE
from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import Frame
from pixtreme._core.lut import Lut
from pixtreme._core.value_domain import _float32_conversion_guidance
from pixtreme._core.vocabulary import _INTERPOLATION_TOKENS

_RGB_CHANNELS = ("R", "G", "B")
_LUT_INTERPOLATION_TOKENS = _INTERPOLATION_TOKENS[9:]


_LUT_TRANSFORM_KERNEL_SOURCE = (
    _LUT_TETRAHEDRAL_CUDA_SOURCE
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
    const float domain_min_r,
    const float domain_min_g,
    const float domain_min_b,
    const float domain_max_r,
    const float domain_max_g,
    const float domain_max_b,
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
    pixtreme_lut_coordinate(
        input[base + red_index], domain_min_r, domain_max_r, lut_size, &red, &red_fraction
    );
    pixtreme_lut_coordinate(
        input[base + green_index], domain_min_g, domain_max_g, lut_size, &green, &green_fraction
    );
    pixtreme_lut_coordinate(
        input[base + blue_index], domain_min_b, domain_max_b, lut_size, &blue, &blue_fraction
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


@lru_cache(maxsize=1)
def _lut_transform_kernel() -> cp.RawKernel:
    return cp.RawKernel(_LUT_TRANSFORM_KERNEL_SOURCE, "pixtreme_lut_transform")


def apply_lut(frame: Frame, *, lut: Lut, interpolation: str = "tetrahedral") -> Frame:
    """Apply a user-provided 3D LUT to the RGB labels of a float32 Frame.

    User LUT values do not declare color meaning, so Frame colorspace and gamma
    metadata pass through unchanged. Other channel labels pass through by value.
    Lookup coordinates use the Lut domain with input clamp at its endpoints;
    output values are not clipped. No shaper stage is implied: authors describe
    wider or linear domains through ``Lut.domain_min`` and ``Lut.domain_max``.
    Neither input is mutated, and the result owns new Frame storage.
    """
    if not isinstance(frame, Frame):
        raise ValueError(
            _actionable_error(
                why="apply_lut operates on metadata-bearing Frame values",
                what=f"received {type(frame).__module__}.{type(frame).__qualname__}",
                how="construct a Frame with px.io.from_array before applying the LUT",
            )
        )
    if not isinstance(lut, Lut):
        raise ValueError(
            _actionable_error(
                why="apply_lut requires a validated Lut",
                what=f"received {type(lut).__module__}.{type(lut).__qualname__}",
                how="construct px.core.Lut(...) directly or call px.io.read_lut(path)",
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
    if not isinstance(interpolation, str) or interpolation not in _LUT_INTERPOLATION_TOKENS:
        raise ValueError(
            _actionable_error(
                why="apply_lut interpolation is a closed, case-sensitive token axis",
                what=f"received interpolation={interpolation!r}",
                how=f"pass one of {_LUT_INTERPOLATION_TOKENS!r}",
            )
        )

    output = cp.empty_like(frame.data)
    pixel_count = int(frame.height * frame.width)
    if pixel_count:
        itemsize = int(lut.data.dtype.itemsize)
        strides = tuple(int(stride) // itemsize for stride in lut.data.strides)
        threads_per_block = 256
        block_count = (pixel_count + threads_per_block - 1) // threads_per_block
        _lut_transform_kernel()(
            (block_count,),
            (threads_per_block,),
            (
                frame.data,
                output,
                np.int64(pixel_count),
                np.int32(frame.data.shape[2]),
                np.int32(frame.channels.index("R")),
                np.int32(frame.channels.index("G")),
                np.int32(frame.channels.index("B")),
                lut.data,
                np.int32(lut.data.shape[0]),
                *(np.int64(stride) for stride in strides),
                *(np.float32(value) for value in lut.domain_min),
                *(np.float32(value) for value in lut.domain_max),
                np.int32(_LUT_INTERPOLATION_TOKENS.index(interpolation)),
            ),
        )
    return Frame(
        data=output,
        colorspace=frame.colorspace,
        gamma=frame.gamma,
        channels=frame.channels,
        matrix=frame.matrix,
    )
