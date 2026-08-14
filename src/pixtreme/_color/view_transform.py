"""Versioned ACES Output Transforms backed by pre-baked GPU LUTs."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from importlib.resources import files

import cupy as cp
import numpy as np
from numpy.typing import NDArray

from pixtreme._color._lut_cuda import _LUT_TETRAHEDRAL_CUDA_SOURCE
from pixtreme._color.transform import (
    _COLOR_TRANSFORM_KERNEL,
    _GAMMA_CODES,
)
from pixtreme._core.errors import _actionable_error
from pixtreme._core.vocabulary import _TONEMAP_ACES_TOKENS, _TONEMAP_LUT_TOKENS

_OUTPUT_COMBINATIONS = (
    ("Rec.709", "bt1886"),
    ("sRGB", "srgb"),
)
_ANALYTIC_COMBINATIONS = tuple(
    (version, output_colorspace, output_gamma)
    for version in _TONEMAP_ACES_TOKENS
    for output_colorspace, output_gamma in _OUTPUT_COMBINATIONS
)
_PUBLIC_TO_INTERNAL_LUT: Mapping[str, str] = dict(zip(_TONEMAP_LUT_TOKENS, _TONEMAP_ACES_TOKENS, strict=True))
_LUT_COMBINATIONS = tuple(
    (version, output_colorspace, output_gamma)
    for version in _TONEMAP_LUT_TOKENS
    for output_colorspace, output_gamma in _OUTPUT_COMBINATIONS
)
_SUPPORTED_COMBINATIONS = (*_ANALYTIC_COMBINATIONS, *_LUT_COMBINATIONS)
_LUT_FILES: Mapping[tuple[str, str, str], str] = {
    ("aces-1.3", "Rec.709", "bt1886"): "view_transform_aces-1.3_rec709_bt1886.npz",
    ("aces-1.3", "sRGB", "srgb"): "view_transform_aces-1.3_srgb_srgb.npz",
    ("aces-2.0", "Rec.709", "bt1886"): "view_transform_aces-2.0_rec709_bt1886.npz",
    ("aces-2.0", "sRGB", "srgb"): "view_transform_aces-2.0_srgb_srgb.npz",
}
_OUTPUT_GAMMA_CODES: Mapping[str, int] = {"linear": 0, "srgb": 1, "bt1886": 2}
_Float32Matrix = NDArray[np.float32]
_IDENTITY_MATRIX = np.eye(3, dtype=np.float32)
# Reuse rgb_to_rgb's exact transfer functions while omitting its separate global kernel entry point.
_COLOR_TRANSFORM_DEVICE_FUNCTIONS = _COLOR_TRANSFORM_KERNEL.partition('extern "C" __global__')[0]

_VIEW_TRANSFORM_KERNEL = (
    _COLOR_TRANSFORM_DEVICE_FUNCTIONS
    + _LUT_TETRAHEDRAL_CUDA_SOURCE
    + r"""
#ifndef PIXTREME_INPUT_GAMMA
#define PIXTREME_INPUT_GAMMA 0
#endif

#ifndef PIXTREME_OUTPUT_GAMMA
#define PIXTREME_OUTPUT_GAMMA 0
#endif

#ifndef PIXTREME_APPLY_TECHNICAL_TRANSFORM
#define PIXTREME_APPLY_TECHNICAL_TRANSFORM 0
#endif

__device__ __forceinline__ float shape_value(
    const float value,
    const float* __restrict__ shaper,
    const int shaper_size,
    const float domain_min,
    const float domain_max
) {
    float position = (value - domain_min) / (domain_max - domain_min) * (shaper_size - 1);
    position = fminf(fmaxf(position, 0.0f), (float)(shaper_size - 1));
    const int lower = min((int)floorf(position), shaper_size - 2);
    const float fraction = position - lower;
    const float lower_value = __ldg(shaper + lower);
    const float upper_value = __ldg(shaper + lower + 1);
    return lower_value + fraction * (upper_value - lower_value);
}

__device__ __forceinline__ float encode_display(const float value, const int gamma) {
    if (gamma == 0) {
        return value;
    }
    if (gamma == 1) {
        return value <= 0.0031308f ? 12.92f * value : 1.055f * powf(value, 1.0f / 2.4f) - 0.055f;
    }
    return copysignf(powf(fabsf(value), 1.0f / 2.4f), value);
}

extern "C" __global__
void view_transform_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const long long pixel_count,
    const int channel_count,
    const int r_index,
    const int g_index,
    const int b_index,
    const float m00,
    const float m01,
    const float m02,
    const float m10,
    const float m11,
    const float m12,
    const float m20,
    const float m21,
    const float m22,
    const float* __restrict__ shaper,
    const int shaper_size,
    const float domain_min,
    const float domain_max,
    const float4* __restrict__ lut,
    const int lut_size
) {
    const long long pixel = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (pixel >= pixel_count) {
        return;
    }

    const long long base = pixel * channel_count;
    if (channel_count > 3) {
        for (int channel = 0; channel < channel_count; ++channel) {
            if (channel != r_index && channel != g_index && channel != b_index) {
                output[base + channel] = input[base + channel];
            }
        }
    }

    const float input_red = input[base + r_index];
    const float input_green = input[base + g_index];
    const float input_blue = input[base + b_index];

#if PIXTREME_APPLY_TECHNICAL_TRANSFORM
    const float linear_red = decode_transfer(input_red, PIXTREME_INPUT_GAMMA);
    const float linear_green = decode_transfer(input_green, PIXTREME_INPUT_GAMMA);
    const float linear_blue = decode_transfer(input_blue, PIXTREME_INPUT_GAMMA);

    const float transformed_red = m00 * linear_red + m01 * linear_green + m02 * linear_blue;
    const float transformed_green = m10 * linear_red + m11 * linear_green + m12 * linear_blue;
    const float transformed_blue = m20 * linear_red + m21 * linear_green + m22 * linear_blue;
#else
    const float transformed_red = input_red;
    const float transformed_green = input_green;
    const float transformed_blue = input_blue;
#endif

    const float shaped_red = shape_value(transformed_red, shaper, shaper_size, domain_min, domain_max);
    const float shaped_green = shape_value(transformed_green, shaper, shaper_size, domain_min, domain_max);
    const float shaped_blue = shape_value(transformed_blue, shaper, shaper_size, domain_min, domain_max);

    int red;
    int green;
    int blue;
    float red_fraction;
    float green_fraction;
    float blue_fraction;
    pixtreme_lut_coordinate(shaped_red, 0.0f, 1.0f, lut_size, &red, &red_fraction);
    pixtreme_lut_coordinate(shaped_green, 0.0f, 1.0f, lut_size, &green, &green_fraction);
    pixtreme_lut_coordinate(shaped_blue, 0.0f, 1.0f, lut_size, &blue, &blue_fraction);
    const long long stride_blue = 4;
    const long long stride_green = (long long)lut_size * stride_blue;
    const long long stride_red = (long long)lut_size * stride_green;
    const float3 linear = pixtreme_lut_tetrahedral(
        reinterpret_cast<const float*>(lut),
        stride_red,
        stride_green,
        stride_blue,
        1,
        red,
        green,
        blue,
        red_fraction,
        green_fraction,
        blue_fraction,
        1
    );
    const float encoded_red = encode_display(linear.x, PIXTREME_OUTPUT_GAMMA);
    const float encoded_green = encode_display(linear.y, PIXTREME_OUTPUT_GAMMA);
    const float encoded_blue = encode_display(linear.z, PIXTREME_OUTPUT_GAMMA);
    output[base + r_index] = encoded_red;
    output[base + g_index] = encoded_green;
    output[base + b_index] = encoded_blue;
}
"""
)


@lru_cache(maxsize=len(_GAMMA_CODES) * len(_OUTPUT_GAMMA_CODES) * 2)
def _view_transform_kernel(
    input_gamma: str = "linear", output_gamma: str = "linear", apply_technical_transform: bool = False
) -> cp.RawKernel:
    options = (
        f"-DPIXTREME_INPUT_GAMMA={_GAMMA_CODES[input_gamma]}",
        f"-DPIXTREME_OUTPUT_GAMMA={_OUTPUT_GAMMA_CODES[output_gamma]}",
        f"-DPIXTREME_APPLY_TECHNICAL_TRANSFORM={int(apply_technical_transform)}",
    )
    return cp.RawKernel(_VIEW_TRANSFORM_KERNEL, "view_transform_kernel", options=options)


@lru_cache(maxsize=4)
def _load_lut_identity(
    version: str, output_colorspace: str, output_gamma: str
) -> tuple[cp.ndarray, tuple[float, float], cp.ndarray]:
    combination = (version, output_colorspace, output_gamma)
    resource = files("pixtreme.data").joinpath(_LUT_FILES[combination])
    with resource.open("rb") as stream, np.load(stream, allow_pickle=False) as archive:
        stored_combination = (
            str(archive["version"].item()),
            str(archive["output_colorspace"].item()),
            str(archive["output_gamma"].item()),
        )
        if stored_combination != combination:
            raise RuntimeError(
                _actionable_error(
                    why="view transform LUT metadata must match the requested transform identity",
                    what=f"requested combination={combination!r}, archive combination={stored_combination!r}",
                    how=(
                        "restore or rebuild the packaged LUT archive with matching version, output_colorspace, "
                        "and output_gamma metadata"
                    ),
                )
            )
        shaper = np.asarray(archive["shaper"], dtype=np.float32)
        domain_array = np.asarray(archive["shaper_domain"], dtype=np.float32)
        lut = np.asarray(archive["lut"], dtype=np.float32)
    if shaper.shape != (4096,) or domain_array.shape != (2,) or lut.shape != (65, 65, 65, 3):
        raise RuntimeError(
            _actionable_error(
                why="view transform LUT archive has incompatible table shapes",
                what=(
                    f"received shaper shape={shaper.shape!r}, shaper_domain shape={domain_array.shape!r}, "
                    f"lut shape={lut.shape!r}"
                ),
                how="rebuild the archive with shaper=(4096,), shaper_domain=(2,), and lut=(65, 65, 65, 3)",
            )
        )
    padded_lut = np.empty((*lut.shape[:-1], 4), dtype=np.float32)
    padded_lut[..., :3] = lut
    padded_lut[..., 3] = 0.0
    return cp.asarray(shaper), (float(domain_array[0]), float(domain_array[1])), cp.asarray(padded_lut)


def _load_lut(
    public_token: str, output_colorspace: str, output_gamma: str
) -> tuple[cp.ndarray, tuple[float, float], cp.ndarray]:
    return _load_lut_identity(_PUBLIC_TO_INTERNAL_LUT[public_token], output_colorspace, output_gamma)


def _apply_lut_data(
    data: cp.ndarray,
    channels: tuple[str, ...],
    *,
    shaper: cp.ndarray,
    shaper_domain: tuple[float, float],
    lut: cp.ndarray,
    output_gamma: str,
    input_gamma: str = "linear",
    matrix: _Float32Matrix | None = None,
) -> cp.ndarray:
    output = cp.empty_like(data)
    pixel_count = int(data.shape[0] * data.shape[1])
    if pixel_count == 0:
        return output

    if lut.shape[-1] == 3:
        padded_lut = cp.empty((*lut.shape[:-1], 4), dtype=cp.float32)
        padded_lut[..., :3] = lut
        padded_lut[..., 3] = 0.0
        lut = padded_lut

    flat_matrix = (_IDENTITY_MATRIX if matrix is None else matrix).reshape(9)
    threads_per_block = 128
    block_count = (pixel_count + threads_per_block - 1) // threads_per_block
    _view_transform_kernel(input_gamma, output_gamma, matrix is not None)(
        (block_count,),
        (threads_per_block,),
        (
            data,
            output,
            np.int64(pixel_count),
            np.int32(data.shape[2]),
            np.int32(channels.index("R")),
            np.int32(channels.index("G")),
            np.int32(channels.index("B")),
            *(np.float32(value) for value in flat_matrix),
            shaper,
            np.int32(shaper.shape[0]),
            np.float32(shaper_domain[0]),
            np.float32(shaper_domain[1]),
            lut,
            np.int32(lut.shape[0]),
        ),
    )
    return output
