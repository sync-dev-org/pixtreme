"""Fused analytic ACES 1.3 SDR rendering for ``rgb_to_rgb``."""

from __future__ import annotations

from functools import lru_cache

import cupy as cp
import numpy as np
from numpy.typing import NDArray

from pixtreme._color.transform import _COLOR_TRANSFORM_KERNEL, _GAMMA_CODES

_Float32Matrix = NDArray[np.float32]
_COLOR_TRANSFORM_DEVICE_FUNCTIONS = _COLOR_TRANSFORM_KERNEL.partition('extern "C" __global__')[0]

_ACES13_ANALYTIC_KERNEL = (
    _COLOR_TRANSFORM_DEVICE_FUNCTIONS
    + r"""
#ifndef PIXTREME_INPUT_GAMMA
#define PIXTREME_INPUT_GAMMA 0
#endif

#ifndef PIXTREME_ACES13_SRGB
#define PIXTREME_ACES13_SRGB 0
#endif

// Binary-search knot probes remain warp-coherent enough to benefit from constant-cache broadcast.
// Packed global read-only coefficient records avoid serialized constant-cache loads after segments diverge.

__device__ __constant__ float aces13_curve0_knots[9] = {
    -5.26017761f, -3.75502753f, -2.24987745f, -0.744727492f, 1.06145251f,
    1.96573484f, 2.86763239f, 3.77526045f, 4.67381239f
};
__device__ const float4 aces13_curve0_coefficients[8] = {
    {0.185970441f, 0.0f, -4.0f, 0.0f},
    {0.403778881f, 0.559826851f, -3.57868838f, 0.0f},
    {-0.0748505071f, 1.77532244f, -1.82131326f, 0.0f},
    {-0.185833707f, 1.54999995f, 0.681241214f, 0.0f},
    {-0.192129433f, 0.878701687f, 2.87457752f, 0.0f},
    {-0.19314684f, 0.531223178f, 3.51206255f, 0.0f},
    {-0.0501050949f, 0.182825878f, 3.8340621f, 0.0f},
    {-0.0511224195f, 0.0918722972f, 3.95872402f, 0.0f},
};

__device__ __constant__ float aces13_curve1_knots[15] = {
    -2.54062366f, -2.08035731f, -1.62009084f, -1.15982437f, -0.69955802f,
    -0.239291579f, 0.220974833f, 0.681241214f, 1.01284635f, 1.34445143f,
    1.6760565f, 2.00766158f, 2.33926654f, 2.67087173f, 3.00247669f
};
__device__ const float4 aces13_curve1_coefficients[14] = {
    {0.521772683f, 0.0f, -1.69896996f, 0.0f},
    {0.0654487088f, 0.480308801f, -1.58843505f, 0.0f},
    {0.272604734f, 0.54055649f, -1.35350001f, 0.0f},
    {0.123911291f, 0.791498125f, -1.04694998f, 0.0f},
    {0.0858645961f, 0.90556252f, -0.656400025f, 0.0f},
    {-0.0171162505f, 0.984603703f, -0.221410006f, 0.0f},
    {0.0338416733f, 0.968847632f, 0.22814402f, 0.0f},
    {-0.194834962f, 1.0f, 0.681241214f, 0.0f},
    {-0.201688975f, 0.870783448f, 0.991421878f, 0.0f},
    {-0.476983279f, 0.737021267f, 1.25800002f, 0.0f},
    {-0.276004612f, 0.420681119f, 1.44994998f, 0.0f},
    {-0.139139131f, 0.237632066f, 1.55910003f, 0.0f},
    {-0.0922630876f, 0.145353615f, 1.62259996f, 0.0f},
    {-0.0665909499f, 0.0841637775f, 1.66065454f, 0.0f},
};

__device__ __forceinline__ float eval_curve0(const float x) {
    if (x <= aces13_curve0_knots[0]) {
        const float4 coefficients = aces13_curve0_coefficients[0];
        return (x - aces13_curve0_knots[0]) * coefficients.y + coefficients.z;
    }
    if (x >= aces13_curve0_knots[8]) {
        const float4 coefficients = aces13_curve0_coefficients[7];
        const float t = aces13_curve0_knots[8] - aces13_curve0_knots[7];
        const float slope = 2.0f * coefficients.x * t + coefficients.y;
        const float offset = (coefficients.x * t + coefficients.y) * t + coefficients.z;
        return (x - aces13_curve0_knots[8]) * slope + offset;
    }
    int low = 0;
    int high = 8;
    while (high - low > 1) {
        const int middle = (low + high) >> 1;
        if (x < aces13_curve0_knots[middle]) {
            high = middle;
        } else {
            low = middle;
        }
    }
    const float4 coefficients = aces13_curve0_coefficients[low];
    const float t = x - aces13_curve0_knots[low];
    return (coefficients.x * t + coefficients.y) * t + coefficients.z;
}

__device__ __forceinline__ float eval_curve1(const float x) {
    if (x <= aces13_curve1_knots[0]) {
        const float4 coefficients = aces13_curve1_coefficients[0];
        return (x - aces13_curve1_knots[0]) * coefficients.y + coefficients.z;
    }
    if (x >= aces13_curve1_knots[14]) {
        const float4 coefficients = aces13_curve1_coefficients[13];
        const float t = aces13_curve1_knots[14] - aces13_curve1_knots[13];
        const float slope = 2.0f * coefficients.x * t + coefficients.y;
        const float offset = (coefficients.x * t + coefficients.y) * t + coefficients.z;
        return (x - aces13_curve1_knots[14]) * slope + offset;
    }
    int low = 0;
    int high = 14;
    while (high - low > 1) {
        const int middle = (low + high) >> 1;
        if (x < aces13_curve1_knots[middle]) {
            high = middle;
        } else {
            low = middle;
        }
    }
    const float4 coefficients = aces13_curve1_coefficients[low];
    const float t = x - aces13_curve1_knots[low];
    return (coefficients.x * t + coefficients.y) * t + coefficients.z;
}

__device__ __forceinline__ float aces13_sign(const float value) {
    return value > 0.0f ? 1.0f : (value < 0.0f ? -1.0f : 0.0f);
}

__device__ __forceinline__ float ocio_fast_exp10(const float value) {
    // OCIO 2.5.2's x86 CPUProcessor evaluates Anti-Log with its SSE exp2
    // degree-4 minimax polynomial.  Reproduce that float32 operation order.
    const float scaled = __fmul_rn(value, 3.3219280948873623f);
    if (scaled < -126.0f) {
        return 0.0f;
    }
    if (scaled >= 128.0f) {
        return __int_as_float(0x7f800000);
    }
    const int integral = (int)floorf(scaled);
    const float fraction = __fadd_rn(scaled, -(float)integral);
    float polynomial = __fadd_rn(__fmul_rn(1.3534167928335475e-2f, fraction), 5.201146058412685e-2f);
    polynomial = __fadd_rn(__fmul_rn(polynomial, fraction), 2.4144275690918652e-1f);
    polynomial = __fadd_rn(__fmul_rn(polynomial, fraction), 6.930038344665415e-1f);
    polynomial = __fadd_rn(__fmul_rn(polynomial, fraction), 1.0000025933706032f);
    return ldexpf(polynomial, integral);
}

__device__ __forceinline__ float ocio_fast_log10(const float value) {
    const float positive = fmaxf(1.17549435e-38f, value);
    const int bits = __float_as_int(positive);
    const float mantissa = __int_as_float((bits & ~0x7f800000) | 0x3f800000);
    float logarithm = __fadd_rn(__fmul_rn(4.487361286440374e-2f, mantissa), -4.165637071209677e-1f);
    logarithm = __fadd_rn(__fmul_rn(logarithm, mantissa), 1.6311488261194363f);
    logarithm = __fadd_rn(__fmul_rn(logarithm, mantissa), -3.550793018041176f);
    logarithm = __fadd_rn(__fmul_rn(logarithm, mantissa), 5.091710879305474f);
    logarithm = __fadd_rn(__fmul_rn(logarithm, mantissa), -2.8003640543959657f);
    const float log2_value = __fadd_rn(logarithm, (float)(((bits & 0x7f800000) >> 23) - 127));
    return __fmul_rn(log2_value, 0.3010299956639812f);
}

__device__ __forceinline__ float3 aces13_render(float red, float green, float blue) {
    const float chroma = sqrtf(
        blue * (blue - green) + green * (green - red) + red * (red - blue)
    );
    const float yc = (blue + green + red + 1.75f * chroma) / 3.0f;
    float maximum = fmaxf(red, fmaxf(green, blue));
    float minimum = fminf(red, fminf(green, blue));
    float saturation = (fmaxf(1e-10f, maximum) - fmaxf(1e-10f, minimum)) / fmaxf(1e-2f, maximum);
    float x = (saturation - 0.4f) * 5.0f;
    float t = fmaxf(0.0f, 1.0f - 0.5f * fabsf(x));
    float s = 0.5f * (1.0f + aces13_sign(x) * (1.0f - t * t));
    float glow_gain = 0.0500000007f * s;
    float glow_gain_out = glow_gain;
    if (yc > 0.0799999982f * 2.0f / 3.0f) {
        glow_gain_out = glow_gain * (0.0799999982f / yc - 0.5f);
    }
    if (yc > 0.0799999982f * 2.0f) {
        glow_gain_out = 0.0f;
    }
    red += red * glow_gain_out;
    green += green * glow_gain_out;
    blue += blue * glow_gain_out;

    const float hue_a = 2.0f * red - (green + blue);
    const float hue_b = 1.7320508075688772f * (green - blue);
    const float hue = atan2f(hue_b, hue_a);
    const float knot_coordinate = fminf(fmaxf(2.0f + hue * 1.6976527f, 0.0f), 4.0f);
    const int segment = min((int)knot_coordinate, 3);
    t = knot_coordinate - (float)segment;
    float hue_weight;
    if (segment == 0) {
        hue_weight = 0.25f * t * t * t;
    } else if (segment == 1) {
        hue_weight = ((-0.75f * t + 0.75f) * t + 0.75f) * t + 0.25f;
    } else if (segment == 2) {
        hue_weight = (0.75f * t - 1.5f) * t * t + 1.0f;
    } else {
        hue_weight = ((-0.25f * t + 0.75f) * t - 0.75f) * t + 0.25f;
    }
    maximum = fmaxf(red, fmaxf(green, blue));
    minimum = fminf(red, fminf(green, blue));
    saturation = (fmaxf(1e-10f, maximum) - fmaxf(1e-10f, minimum)) / fmaxf(1e-2f, maximum);
    red += hue_weight * saturation * (0.0299999993f - red) * 0.180000007f;

    red = fmaxf(0.0f, red);
    green = fmaxf(0.0f, green);
    blue = fmaxf(0.0f, blue);

    float transformed_red = 1.4514393161456653f * red - 0.23651074689374019f * green - 0.21492856925192524f * blue;
    float transformed_green = -0.07655377339602043f * red + 1.1762296998335731f * green - 0.0996759264375522f * blue;
    float transformed_blue = 0.008316148425697719f * red - 0.006032449791021028f * green + 0.9977163013653233f * blue;
    red = fmaxf(0.0f, transformed_red);
    green = fmaxf(0.0f, transformed_green);
    blue = fmaxf(0.0f, transformed_blue);

    transformed_red = 0.970889148671f * red + 0.026963270632f * green + 0.002147580696f * blue;
    transformed_green = 0.010889148671f * red + 0.986963270632f * green + 0.002147580696f * blue;
    transformed_blue = 0.010889148671f * red + 0.026963270632f * green + 0.962147580696f * blue;

    red = ocio_fast_log10(transformed_red);
    green = ocio_fast_log10(transformed_green);
    blue = ocio_fast_log10(transformed_blue);
    red = eval_curve1(eval_curve0(red));
    green = eval_curve1(eval_curve0(green));
    blue = eval_curve1(eval_curve0(blue));
    red = ocio_fast_exp10(red) * 0.0208420176f - 0.000416840339f;
    green = ocio_fast_exp10(green) * 0.0208420176f - 0.000416840339f;
    blue = ocio_fast_exp10(blue) * 0.0208420176f - 0.000416840339f;

    const float luminance = fmaxf(
        1e-10f,
        0.27222871678091454f * red + 0.67408176581114831f * green + 0.053689517407937051f * blue
    );
    const float surround = powf(luminance, -0.0188999772f);
    red *= surround;
    green *= surround;
    blue *= surround;

    transformed_red = 1.604753433346922f * red - 0.531080948604018f * green - 0.07367248474191035f * blue;
    transformed_green = -0.10208245810655031f * red + 1.1081341286221253f * green - 0.006051670514572949f * blue;
    transformed_blue = -0.003267111653294682f * red - 0.0727554241334227f * green + 1.0760225357877193f * blue;

#if PIXTREME_ACES13_SRGB
    red = transformed_red > 0.00303993467f
        ? powf(fmaxf(0.0f, transformed_red), 0.416666657f) * 1.05499995f - 0.0549999997f
        : transformed_red * 12.9232101f;
    green = transformed_green > 0.00303993467f
        ? powf(fmaxf(0.0f, transformed_green), 0.416666657f) * 1.05499995f - 0.0549999997f
        : transformed_green * 12.9232101f;
    blue = transformed_blue > 0.00303993467f
        ? powf(fmaxf(0.0f, transformed_blue), 0.416666657f) * 1.05499995f - 0.0549999997f
        : transformed_blue * 12.9232101f;
#else
    red = powf(fmaxf(0.0f, transformed_red), 0.41666666666666669f);
    green = powf(fmaxf(0.0f, transformed_green), 0.41666666666666669f);
    blue = powf(fmaxf(0.0f, transformed_blue), 0.41666666666666669f);
#endif
    return make_float3(red, green, blue);
}

extern "C" __global__
void aces13_analytic_kernel(
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
    const float3 rendered = aces13_render(aces_red, aces_green, aces_blue);
    output[base + r_index] = rendered.x;
    output[base + g_index] = rendered.y;
    output[base + b_index] = rendered.z;
}
"""
)


@lru_cache(maxsize=len(_GAMMA_CODES) * 2)
def _aces13_transform_kernel(input_gamma: str, output_gamma: str) -> cp.RawKernel:
    options = (
        f"-DPIXTREME_INPUT_GAMMA={_GAMMA_CODES[input_gamma]}",
        f"-DPIXTREME_ACES13_SRGB={int(output_gamma == 'sRGB')}",
        "--use_fast_math",
    )
    return cp.RawKernel(_ACES13_ANALYTIC_KERNEL, "aces13_analytic_kernel", options=options)


def _apply_aces13_data(
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
    _aces13_transform_kernel(input_gamma, output_gamma)(
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
