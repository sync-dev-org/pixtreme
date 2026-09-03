"""Fused analytic ACES 2.0 SDR 100-nit rendering for ``rgb_to_rgb``."""

from __future__ import annotations

from functools import lru_cache

import cupy as cp
import numpy as np
from numpy.typing import NDArray

from pixtreme._color.aces20_tables import _ACES20_TABLE_CUDA_SOURCE
from pixtreme._color.transform import _COLOR_TRANSFORM_KERNEL, _GAMMA_CODES

_Float32Matrix = NDArray[np.float32]
_COLOR_TRANSFORM_DEVICE_FUNCTIONS = _COLOR_TRANSFORM_KERNEL.partition('extern "C" __global__')[0]

_ACES20_ANALYTIC_KERNEL = (
    _COLOR_TRANSFORM_DEVICE_FUNCTIONS
    + _ACES20_TABLE_CUDA_SOURCE
    + r"""
#ifndef PIXTREME_INPUT_GAMMA
#define PIXTREME_INPUT_GAMMA 0
#endif

#ifndef PIXTREME_ACES20_SRGB
#define PIXTREME_ACES20_SRGB 0
#endif

__device__ __forceinline__ float aces20_sign(const float value) {
    return value > 0.0f ? 1.0f : (value < 0.0f ? -1.0f : 0.0f);
}

__device__ __forceinline__ float aces20_lerp(const float lower, const float upper, const float weight) {
    return lower + weight * (upper - lower);
}

__device__ __forceinline__ float aces20_reach_sample(const float hue) {
    const int base = (int)floorf(hue);
    const int lower = base + 1;
    const int upper = lower + 1;
    return aces20_lerp(aces20_reach_m[lower], aces20_reach_m[upper], hue - (float)base);
}

__device__ __forceinline__ float3 aces20_cusp_sample(const float hue) {
    // The pinned table's bracketing record is at most two positions above the integer-degree hint.
    int upper = (int)hue + 1;
    upper += hue > aces20_gamut_hues[upper];
    upper += hue > aces20_gamut_hues[upper];
    const float weight = (hue - aces20_gamut_hues[upper - 1])
        / (aces20_gamut_hues[upper] - aces20_gamut_hues[upper - 1]);
    const int lower_base = (upper - 1) * 3;
    const int upper_base = upper * 3;
    return make_float3(
        aces20_lerp(aces20_gamut_cusp[lower_base], aces20_gamut_cusp[upper_base], weight),
        aces20_lerp(aces20_gamut_cusp[lower_base + 1], aces20_gamut_cusp[upper_base + 1], weight),
        aces20_lerp(aces20_gamut_cusp[lower_base + 2], aces20_gamut_cusp[upper_base + 2], weight)
    );
}

__device__ __forceinline__ float aces20_tonescale(const float lightness) {
    const float achromatic = 0.0323680267f * powf(fabsf(lightness) * 0.00999999978f, 0.879464149f);
    const float luminance = powf((27.1299992f * achromatic) / (1.0f - achromatic), 2.3809523809523809f);
    const float curve = 1.04710376f * powf(luminance / (luminance + 0.73009213709383403f), 1.14999998f);
    const float mapped = fmaxf(0.0f, curve * curve / (curve + 0.0399999991f));
    const float response = powf(0.79370057210326195f * mapped, 0.42f);
    const float output = 100.0f * powf((response / (27.1299992f + response)) * 30.8946857f, 1.13705599f);
    return aces20_sign(lightness) * output;
}

__device__ __forceinline__ float aces20_toe(
    const float value,
    const float limit,
    const float k1_input,
    const float k2_input
) {
    if (value > limit) {
        return value;
    }
    const float k2 = fmaxf(k2_input, 0.001f);
    const float k1 = sqrtf(k1_input * k1_input + k2 * k2);
    const float k3 = (limit + k1) / (limit + k2);
    const float minus_b = k3 * value - k1;
    return 0.5f * (minus_b + sqrtf(minus_b * minus_b + 4.0f * k2 * k3 * value));
}

__device__ __forceinline__ float aces20_focus_gain(const float lightness, const float cusp_lightness) {
    const float threshold = aces20_lerp(cusp_lightness, 100.0f, 0.3f);
    if (lightness <= threshold) {
        return 1.0f;
    }
    float gain = (100.0f - threshold) / fmaxf(0.0001f, 100.0f - lightness);
    gain = log10f(gain);
    return gain * gain + 1.0f;
}

__device__ __forceinline__ float aces20_solve_intersection(
    const float lightness,
    const float colorfulness,
    const float focus_lightness,
    const float slope_gain
) {
    const float scaled_colorfulness = colorfulness / slope_gain;
    const float a = scaled_colorfulness / focus_lightness;
    if (lightness < focus_lightness) {
        const float b = 1.0f - scaled_colorfulness;
        const float c = -lightness;
        return -2.0f * c / (b + sqrtf(b * b - 4.0f * a * c));
    }
    const float b = -(1.0f + scaled_colorfulness + 100.0f * a);
    const float c = 100.0f * scaled_colorfulness + lightness;
    return -2.0f * c / (b - sqrtf(b * b - 4.0f * a * c));
}

__device__ __forceinline__ float aces20_gamut_boundary(
    const float2 cusp,
    const float upper_gamma_inverse,
    const float lower_gamma_inverse,
    const float source_intersection,
    const float cusp_intersection,
    const float slope
) {
    const float lower = cusp_intersection
        * powf(source_intersection / cusp_intersection, lower_gamma_inverse)
        / (cusp.x / cusp.y - slope);
    const float upper = cusp.y * (100.0f - cusp_intersection)
        * powf((100.0f - source_intersection) / (100.0f - cusp_intersection), upper_gamma_inverse)
        / (slope * cusp.y + 100.0f - cusp.x);
    const float scale = 0.119999997f * cusp.y;
    const float blend = fmaxf(scale - fabsf(lower - upper), 0.0f) / scale;
    return fminf(lower, upper) - blend * blend * blend * scale * 0.16666666666666666f;
}

__device__ __forceinline__ float aces20_remap_colorfulness(
    const float colorfulness,
    const float gamut_boundary,
    const float reach_boundary
) {
    const float boundary_ratio = gamut_boundary / reach_boundary;
    const float proportion = fmaxf(boundary_ratio, 0.75f);
    const float threshold = proportion * gamut_boundary;
    if (proportion >= 1.0f || colorfulness <= threshold) {
        return colorfulness;
    }
    const float offset = colorfulness - threshold;
    const float gamut_offset = gamut_boundary - threshold;
    const float reach_offset = reach_boundary - threshold;
    const float scale = reach_offset / ((reach_offset / gamut_offset) - 1.0f);
    const float normalized = offset / scale;
    return threshold + scale * normalized / (1.0f + normalized);
}

__device__ __forceinline__ float3 aces20_gamut_compress(
    const float3 appearance,
    const float3 cusp_record,
    const float reach_maximum
) {
    const float lightness = appearance.x;
    const float colorfulness = appearance.y;
    if (colorfulness <= 0.0f || lightness > 100.0f) {
        return make_float3(lightness, 0.0f, appearance.z);
    }
    const float2 cusp = make_float2(cusp_record.x, cusp_record.y);
    const float focus_lightness = aces20_lerp(
        cusp.x,
        34.096539f,
        fminf(1.0f, 1.3f - cusp.x / 100.0f)
    );
    const float slope_gain = 135.0f * aces20_focus_gain(lightness, cusp.x);
    const float source_intersection = aces20_solve_intersection(
        lightness,
        colorfulness,
        focus_lightness,
        slope_gain
    );
    float slope = source_intersection < focus_lightness ? source_intersection : 100.0f - source_intersection;
    slope *= (source_intersection - focus_lightness) / (focus_lightness * slope_gain);
    const float cusp_intersection = aces20_solve_intersection(cusp.x, cusp.y, focus_lightness, slope_gain);
    const float gamut_boundary = aces20_gamut_boundary(
        cusp,
        cusp_record.z,
        0.877192974f,
        source_intersection,
        cusp_intersection,
        slope
    );
    if (gamut_boundary <= 0.0f) {
        return make_float3(lightness, 0.0f, appearance.z);
    }
    float reach_boundary = 100.0f * powf(source_intersection / 100.0f, 0.879464149f);
    reach_boundary /= 100.0f / reach_maximum - slope;
    const float remapped = aces20_remap_colorfulness(colorfulness, gamut_boundary, reach_boundary);
    return make_float3(source_intersection + remapped * slope, remapped, appearance.z);
}

__device__ __forceinline__ float3 aces20_render(float red, float green, float blue) {
    float ap1_red = 1.4514393161456653f * red - 0.23651074689374019f * green - 0.21492856925192524f * blue;
    float ap1_green = -0.07655377339602043f * red + 1.1762296998335731f * green - 0.0996759264375522f * blue;
    float ap1_blue = 0.008316148425697719f * red - 0.0060324497910210278f * green + 0.9977163013653233f * blue;
    ap1_red = fminf(fmaxf(ap1_red, 0.0f), 1024.0f);
    ap1_green = fminf(fmaxf(ap1_green, 0.0f), 1024.0f);
    ap1_blue = fminf(fmaxf(ap1_blue, 0.0f), 1024.0f);
    red = 0.69545224135745176f * ap1_red + 0.14067869647029416f * ap1_green + 0.16386906217225403f * ap1_blue;
    green = 0.044794563372037632f * ap1_red + 0.85967111845642163f * ap1_green + 0.095534318171540358f * ap1_blue;
    blue = -0.0055258825581135443f * ap1_red + 0.0040252103059786586f * ap1_green + 1.0015006722521349f * ap1_blue;

    const float lms_red = 0.445181042f * red + 0.34964928f * green - 0.00112973212f * blue;
    const float lms_green = 0.123734146f * red + 0.613643706f * green + 0.0563228019f * blue;
    const float lms_blue = 0.0117007261f * red + 0.0280607939f * green + 0.753939033f * blue;
    const float response_red_power = powf(fabsf(lms_red), 0.419999987f);
    const float response_green_power = powf(fabsf(lms_green), 0.419999987f);
    const float response_blue_power = powf(fabsf(lms_blue), 0.419999987f);
    const float response_red = aces20_sign(lms_red) * response_red_power / (27.1299992f + response_red_power);
    const float response_green = aces20_sign(lms_green) * response_green_power / (27.1299992f + response_green_power);
    const float response_blue = aces20_sign(lms_blue) * response_blue_power / (27.1299992f + response_blue_power);
    const float achromatic = 20.25881f * response_red + 10.129405f * response_green + 0.506470263f * response_blue;
    const float opponent_a = 15480.0f * response_red - 16887.2734f * response_green + 1407.27271f * response_blue;
    const float opponent_b = 1720.0f * response_red + 1720.0f * response_green - 3440.0f * response_blue;
    if (achromatic <= 0.0f) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    const float lightness = 100.0f * powf(achromatic, 1.13705599f);
    const float colorfulness = lightness == 0.0f ? 0.0f : sqrtf(opponent_a * opponent_a + opponent_b * opponent_b);
    float hue = opponent_a == 0.0f ? 0.0f : atan2f(opponent_b, opponent_a) * 57.29577951308238f;
    hue -= floorf(hue / 360.0f) * 360.0f;
    hue = hue < 0.0f ? hue + 360.0f : hue;
    const float hue_radians = hue * 0.0174532924f;
    float sine;
    float cosine;
    sincosf(hue_radians, &sine, &cosine);
    const float mapped_lightness = aces20_tonescale(lightness);
    const float reach_maximum = aces20_reach_sample(hue);

    float compressed_colorfulness = colorfulness;
    if (colorfulness != 0.0f) {
        const float normalized_lightness = mapped_lightness / 100.0f;
        const float remaining_lightness = fmaxf(0.0f, 1.0f - normalized_lightness);
        const float cosine2 = 2.0f * cosine * cosine - 1.0f;
        const float sine2 = 2.0f * cosine * sine;
        const float cosine3 = 4.0f * cosine * cosine * cosine - 3.0f * cosine;
        const float sine3 = 3.0f * sine - 4.0f * sine * sine * sine;
        const float normalization = 11.341321604032515f * cosine
            + 16.469863649185896f * cosine2
            + 7.8842182208776475f * cosine3
            + 14.665187919584513f * sine
            - 6.3725780354404442f * sine2
            + 9.1941277054452897f * sine3
            + 77.133051547393805f;
        const float limit = powf(normalized_lightness, 0.879464149f) * reach_maximum / normalization;
        compressed_colorfulness *= powf(mapped_lightness / lightness, 0.879464149f);
        compressed_colorfulness /= normalization;
        compressed_colorfulness = limit - aces20_toe(
            limit - compressed_colorfulness,
            limit - 0.001f,
            remaining_lightness * 1.29999995f,
            sqrtf(normalized_lightness * normalized_lightness + 0.00499999989f)
        );
        compressed_colorfulness = aces20_toe(
            compressed_colorfulness,
            limit,
            normalized_lightness * 2.4000001f,
            remaining_lightness
        );
        compressed_colorfulness *= normalization;
    }

    const float3 cusp = aces20_cusp_sample(hue);
    const float3 compressed = aces20_gamut_compress(
        make_float3(mapped_lightness, compressed_colorfulness, hue),
        cusp,
        reach_maximum
    );
    const float output_achromatic = powf(compressed.x * 0.00999999978f, 0.879464149f);
    const float output_a = compressed.y * cosine;
    const float output_b = compressed.y * sine;
    const float adapted_red = 0.0323680267f * output_achromatic
        + 2.07657631e-05f * output_a + 1.3260621e-05f * output_b;
    const float adapted_green = 0.0323680267f * output_achromatic
        - 4.10250432e-05f * output_a - 1.20174373e-05f * output_b;
    const float adapted_blue = 0.0323680267f * output_achromatic
        - 1.01296409e-05f * output_a - 0.000290076074f * output_b;
    const float limited_red = fminf(fabsf(adapted_red), 0.99000001f);
    const float limited_green = fminf(fabsf(adapted_green), 0.99000001f);
    const float limited_blue = fminf(fabsf(adapted_blue), 0.99000001f);
    const float cone_red = aces20_sign(adapted_red)
        * powf(27.1299992f * limited_red / (1.0f - limited_red), 2.38095236f);
    const float cone_green = aces20_sign(adapted_green)
        * powf(27.1299992f * limited_green / (1.0f - limited_green), 2.38095236f);
    const float cone_blue = aces20_sign(adapted_blue)
        * powf(27.1299992f * limited_blue / (1.0f - limited_blue), 2.38095236f);
    red = 7.45048571f * cone_red - 6.1301837f * cone_green - 0.0603808537f * cone_blue;
    green = -1.4750675f * cone_red + 3.11835742f * cone_green - 0.383369029f * cone_blue;
    blue = 0.0106288502f * cone_red - 0.31857267f * cone_green + 1.56786489f * cone_blue;
    red = fminf(fmaxf(red, 0.0f), 1.0f);
    green = fminf(fmaxf(green, 0.0f), 1.0f);
    blue = fminf(fmaxf(blue, 0.0f), 1.0f);

#if PIXTREME_ACES20_SRGB
    red = red > 0.00303993467f ? powf(red, 0.416666657f) * 1.05499995f - 0.0549999997f : red * 12.9232101f;
    green = green > 0.00303993467f
        ? powf(green, 0.416666657f) * 1.05499995f - 0.0549999997f
        : green * 12.9232101f;
    blue = blue > 0.00303993467f ? powf(blue, 0.416666657f) * 1.05499995f - 0.0549999997f : blue * 12.9232101f;
#else
    red = powf(red, 0.41666666666666669f);
    green = powf(green, 0.41666666666666669f);
    blue = powf(blue, 0.41666666666666669f);
#endif
    return make_float3(red, green, blue);
}

extern "C" __global__
void aces20_analytic_kernel(
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
    const float m22
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

    const float linear_red = decode_transfer(input[base + r_index], PIXTREME_INPUT_GAMMA);
    const float linear_green = decode_transfer(input[base + g_index], PIXTREME_INPUT_GAMMA);
    const float linear_blue = decode_transfer(input[base + b_index], PIXTREME_INPUT_GAMMA);
    const float aces_red = m00 * linear_red + m01 * linear_green + m02 * linear_blue;
    const float aces_green = m10 * linear_red + m11 * linear_green + m12 * linear_blue;
    const float aces_blue = m20 * linear_red + m21 * linear_green + m22 * linear_blue;
    const float3 rendered = aces20_render(aces_red, aces_green, aces_blue);
    output[base + r_index] = rendered.x;
    output[base + g_index] = rendered.y;
    output[base + b_index] = rendered.z;
}
"""
)


@lru_cache(maxsize=len(_GAMMA_CODES) * 2)
def _aces20_transform_kernel(input_gamma: str, output_gamma: str) -> cp.RawKernel:
    options = (
        f"-DPIXTREME_INPUT_GAMMA={_GAMMA_CODES[input_gamma]}",
        f"-DPIXTREME_ACES20_SRGB={int(output_gamma == 'sRGB')}",
        "--use_fast_math",
    )
    return cp.RawKernel(_ACES20_ANALYTIC_KERNEL, "aces20_analytic_kernel", options=options)


def _apply_aces20_data(
    data: cp.ndarray,
    channels: tuple[str, ...],
    *,
    input_gamma: str,
    output_gamma: str,
    matrix: _Float32Matrix,
) -> cp.ndarray:
    output = cp.empty_like(data)
    pixel_count = int(data.shape[0] * data.shape[1])
    if pixel_count == 0:
        return output

    flat_matrix = matrix.reshape(9)
    threads_per_block = 256
    block_count = (pixel_count + threads_per_block - 1) // threads_per_block
    _aces20_transform_kernel(input_gamma, output_gamma)(
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
        ),
    )
    return output
