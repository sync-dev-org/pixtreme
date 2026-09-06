"""Fused technical transforms of a Frame's RGB colorimetry axes."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache

import cupy as cp
import numpy as np
from numpy.typing import NDArray

from pixtreme._core.colorspace import _COLORSPACE_DEFINITIONS
from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import _COLORSPACE_TOKENS, _GAMMA_TOKENS, Frame
from pixtreme._core.validation import _normalized_closed_token
from pixtreme._core.value_domain import _float32_conversion_guidance
from pixtreme._core.vocabulary import (
    _TONEMAP_ACES_TOKENS,
    _TONEMAP_DIRECT_TOKENS,
    _TONEMAP_TOKENS,
    Colorspace,
    Gamma,
    Tonemap,
)

_RGB_CHANNELS = ("R", "G", "B")

_Float64Matrix = NDArray[np.float64]
_Float32Matrix = NDArray[np.float32]

_BRADFORD = np.asarray(
    (
        (0.8951, 0.2664, -0.1614),
        (-0.7502, 1.7135, 0.0367),
        (0.0389, -0.0685, 1.0296),
    ),
    dtype=np.float64,
)

_GAMMA_CODES: Mapping[str, int] = {
    "linear": 0,
    "sRGB": 1,
    "Rec.709": 2,
    "BT.1886": 3,
    "PQ": 4,
    "HLG": 5,
    "ACEScc": 22,
    "ACEScct": 23,
    "S-Log": 12,
    "S-Log2": 13,
    "S-Log3": 6,
    "ARRI-LogC3": 14,
    "ARRI-LogC4": 7,
    "Blackmagic-Film-Gen-5": 15,
    "DaVinci-Intermediate": 16,
    "RED-Log3G10": 17,
    "REDlogFilm": 8,
    "Canon-Log": 18,
    "Canon-Log-2": 19,
    "Canon-Log-3": 20,
    "V-Log": 21,
    "D-Log": 25,
    "F-Log": 26,
    "F-Log2": 27,
    "N-Log": 28,
    "L-Log": 29,
    "Apple-Log": 30,
    "Samsung-Log": 31,
    "Cineon": 8,
    "Gamma-2.2": 9,
    "Gamma-2.4": 10,
    "Gamma-2.5": 24,
    "Gamma-2.6": 11,
}

_BT2408_COMBINATIONS = tuple(
    (tonemap, "Rec.2020", output_gamma) for tonemap in _TONEMAP_DIRECT_TOKENS for output_gamma in ("HLG", "PQ")
)
_ACES_OUTPUT_COMBINATIONS = (("Rec.709", "BT.1886"), ("sRGB", "sRGB"))
_ANALYTIC_COMBINATIONS = tuple(
    (tonemap, output_colorspace, output_gamma)
    for tonemap in _TONEMAP_ACES_TOKENS
    for output_colorspace, output_gamma in _ACES_OUTPUT_COMBINATIONS
)
_SUPPORTED_COMBINATIONS = (*_ANALYTIC_COMBINATIONS, *_BT2408_COMBINATIONS)

_COLOR_TRANSFORM_KERNEL = r"""
__device__ __forceinline__ float signed_power(const float value, const float exponent) {
    return copysignf(powf(fabsf(value), exponent), value);
}

__device__ __forceinline__ float decode_transfer(const float value, const int gamma) {
    if (gamma == 0) {
        return value;
    }
    if (gamma == 1) {
        return value <= 0.04045f ? value / 12.92f : powf((value + 0.055f) / 1.055f, 2.4f);
    }
    if (gamma == 2) {
        return value < 0.081f ? value / 4.5f : powf((value + 0.099f) / 1.099f, 1.0f / 0.45f);
    }
    if (gamma == 3) {
        return signed_power(value, 2.4f);
    }
    if (gamma == 12) {
        const float y = (1023.0f * value - 64.0f) / 876.0f;
        const float x = y >= 0.030001222851889303f
            ? powf(10.0f, (y - 0.616596f - 0.03f) / 0.432699f) - 0.037584f
            : (y - 0.030001222851889303f) / 5.0f;
        return 0.9f * x;
    }
    if (gamma == 13) {
        const float y = (1023.0f * value - 64.0f) / 876.0f;
        const float x = y >= 0.030001222851889303f
            ? 219.0f * (powf(10.0f, (y - 0.616596f - 0.03f) / 0.432699f) - 0.037584f) / 155.0f
            : (y - 0.030001222851889303f) / 3.53881278538813f;
        return 0.9f * x;
    }
    if (gamma == 17) {
        return value < 0.0f
            ? value / 15.1927f - 0.01f
            : (powf(10.0f, value / 0.224282f) - 1.0f) / 155.975327f - 0.01f;
    }
    if (gamma == 18) {
        const float a = 0.45310179f;
        const float b = 10.1596f;
        const float c = 0.12512248f;
        const float x = value >= c
            ? (powf(10.0f, (value - c) / a) - 1.0f) / b
            : -(powf(10.0f, (c - value) / a) - 1.0f) / b;
        return 0.9f * x;
    }
    if (gamma == 19) {
        const float a = 0.24136077f;
        const float b = 87.099375f;
        const float c = 0.092864125f;
        const float x = value >= c
            ? (powf(10.0f, (value - c) / a) - 1.0f) / b
            : -(powf(10.0f, (c - value) / a) - 1.0f) / b;
        return 0.9f * x;
    }
    if (gamma == 20) {
        const float a = 0.36726845f;
        const float b = 14.98325f;
        const float m = 1.9754798f;
        const float c = 0.12512219f;
        const float c_pos = 0.12240537f;
        const float c_neg = 0.12783901f;
        const float linear_cut = 0.014f;
        const float lower_decode_cut = c - m * linear_cut;
        const float upper_decode_cut = c + m * linear_cut;
        float x;
        if (value < lower_decode_cut) {
            x = -(powf(10.0f, (c_neg - value) / a) - 1.0f) / b;
        } else if (value > upper_decode_cut) {
            x = (powf(10.0f, (value - c_pos) / a) - 1.0f) / b;
        } else {
            x = (value - c) / m;
        }
        return 0.9f * x;
    }
    if (gamma == 21) {
        const float a = 0.241514f;
        const float b = 0.00873f;
        const float c = 0.598206f;
        const float m = 5.60001054470806f;
        const float d = 0.124999583317922f;
        const float encoded_cut = 0.180999688765003f;
        return value < encoded_cut
            ? (value - d) / m
            : powf(10.0f, (value - c) / a) - b;
    }
    if (gamma == 25) {
        const float a = 0.9892f;
        const float b = 0.0108f;
        const float c = 0.256663f;
        const float d = 0.584555f;
        const float e = 6.025f;
        const float f = 0.0929f;
        const float encoded_cut = 0.1400586161070593f;
        return value < encoded_cut
            ? (value - f) / e
            : (powf(10.0f, (value - d) / c) - b) / a;
    }
    if (gamma == 26) {
        const float a = 0.555556f;
        const float b = 0.009468f;
        const float c = 0.344676f;
        const float d = 0.790453f;
        const float e = 8.735631f;
        const float f = 0.092864f;
        const float encoded_cut = 0.09781139663651882f;
        return value < encoded_cut
            ? (value - f) / e
            : (powf(10.0f, (value - d) / c) - b) / a;
    }
    if (gamma == 27) {
        const float a = 5.555556f;
        const float b = 0.064829f;
        const float c = 0.245281f;
        const float d = 0.384316f;
        const float e = 8.799461f;
        const float f = 0.092864f;
        const float encoded_cut = 0.10068573654723681f;
        return value < encoded_cut
            ? (value - f) / e
            : (powf(10.0f, (value - d) / c) - b) / a;
    }
    if (gamma == 28) {
        const float encoded_cut = 0.4625960144726521f;
        if (value < encoded_cut) {
            const float t = value * 1023.0f / 650.0f;
            return t * t * t - 0.0075f;
        }
        return expf((value * 1023.0f - 619.0f) / 150.0f);
    }
    if (gamma == 29) {
        const float m = 7.898308971401108f;
        const float d = 0.08971061960369227f;
        const float encoded_cut = 0.1371004734320989f;
        return value < encoded_cut
            ? (value - d) / m
            : (powf(10.0f, (value - 0.6f) / 0.27f) - 0.0115f) / 1.3f;
    }
    if (gamma == 30) {
        const float r0 = -0.05641088f;
        const float pt = 0.20855531595464208f;
        if (value < 0.0f) {
            return r0;
        }
        if (value < pt) {
            return sqrtf(value / 47.28711236f) + r0;
        }
        return exp2f((value - 0.69336945f) / 0.08550479f) - 0.00964052f;
    }
    if (gamma == 31) {
        const float g2 = -0.245973605190997f;
        const float yt = 0.20656190889447099f;
        return value < yt
            ? 0.016904f - powf(10.0f, (value - g2) / -0.20942f)
            : powf(10.0f, (value - 0.720504856f) / 0.258984868f) - 0.0003645f;
    }
    if (gamma == 22) {
        const float lower_decode_cut = (9.72f - 15.0f) / 17.52f;
        return value <= lower_decode_cut
            ? 2.0f * (exp2f(fmaf(17.52f, value, -9.72f)) - exp2f(-16.0f))
            : exp2f(fmaf(17.52f, value, -9.72f));
    }
    if (gamma == 23) {
        const float encoded_cut = 0.155251141552511f;
        return value <= encoded_cut
            ? (value - 0.0729055341958355f) / 10.5402377416545f
            : exp2f(fmaf(17.52f, value, -9.72f));
    }
    if (gamma == 24) {
        return signed_power(value, 2.5f);
    }

    const float sign = value < 0.0f ? -1.0f : 1.0f;
    const float magnitude = fabsf(value);
    if (gamma == 4) {
        const float m1 = 2610.0f / 16384.0f;
        const float m2 = 2523.0f / 32.0f;
        const float c1 = 3424.0f / 4096.0f;
        const float c2 = 2413.0f / 128.0f;
        const float c3 = 2392.0f / 128.0f;
        const float p = powf(magnitude, 1.0f / m2);
        const float numerator = fmaxf(p - c1, 0.0f);
        return sign * powf(numerator / (c2 - c3 * p), 1.0f / m1);
    }
    if (gamma == 5) {
        const float a = 0.17883277f;
        const float b = 0.28466892f;
        const float c = 0.55991073f;
        const float decoded = magnitude <= 0.5f
            ? magnitude * magnitude / 3.0f
            : (expf((magnitude - c) / a) + b) / 12.0f;
        return sign * decoded;
    }
    if (gamma == 6) {
        const float cut = 171.2102946929f / 1023.0f;
        return value >= cut
            ? powf(10.0f, (value * 1023.0f - 420.0f) / 261.5f) * 0.19f - 0.01f
            : (value * 1023.0f - 95.0f) * 0.01125f / (171.2102946929f - 95.0f);
    }
    if (gamma == 14) {
        const float cut = 0.0105909904954696f;
        const float a = 5.55555555555556f;
        const float b = 0.0522722750251688f;
        const float c = 0.247189638318671f;
        const float d = 0.385536998692443f;
        const float e = c * a / ((a * cut + b) * logf(10.0f));
        const float f = c * log10f(a * cut + b) + d - e * cut;
        const float boundary = e * cut + f;
        return value > boundary
            ? (powf(10.0f, (value - d) / c) - b) / a
            : (value - f) / e;
    }
    if (gamma == 15) {
        const float a = 0.08692876065491224f;
        const float b = 0.005494072432257808f;
        const float c = 0.5300133392291939f;
        const float d = 8.283605932402494f;
        const float e = 0.09246575342465753f;
        const float linear_cut = 0.005f;
        const float log_cut = d * linear_cut + e;
        return value < log_cut
            ? (value - e) / d
            : expf((value - c) / a) - b;
    }
    if (gamma == 16) {
        const float a = 0.0075f;
        const float b = 7.0f;
        const float c = 0.07329248f;
        const float m = 10.44426855f;
        const float linear_cut = 0.00262409f;
        const float decode_cut = m * linear_cut;
        return value > decode_cut
            ? exp2f(value / c - b) - a
            : value / m;
    }
    if (gamma == 7) {
        const float a = (powf(2.0f, 18.0f) - 16.0f) / 117.45f;
        const float b = (1023.0f - 95.0f) / 1023.0f;
        const float c = 95.0f / 1023.0f;
        const float s = (7.0f * logf(2.0f) * powf(2.0f, 7.0f - 14.0f * c / b)) / (a * b);
        const float t = (powf(2.0f, 14.0f * (-c / b) + 6.0f) - 64.0f) / a;
        return value >= 0.0f
            ? (powf(2.0f, 14.0f * (value - c) / b + 6.0f) - 64.0f) / a
            : value * s + t;
    }
    if (gamma == 8) {
        const float black_offset = powf(10.0f, (95.0f - 685.0f) / 300.0f);
        const float decoded = (
            powf(10.0f, (1023.0f * magnitude - 685.0f) / 300.0f) - black_offset
        ) / (1.0f - black_offset);
        return sign * decoded;
    }
    return signed_power(value, gamma == 9 ? 2.2f : (gamma == 10 ? 2.4f : 2.6f));
}

__device__ __forceinline__ float encode_transfer(const float value, const int gamma) {
    if (gamma == 0) {
        return value;
    }
    if (gamma == 1) {
        return value <= 0.0031308f ? 12.92f * value : 1.055f * powf(value, 1.0f / 2.4f) - 0.055f;
    }
    if (gamma == 2) {
        return value < 0.018f ? 4.5f * value : 1.099f * powf(value, 0.45f) - 0.099f;
    }
    if (gamma == 3) {
        return signed_power(value, 1.0f / 2.4f);
    }
    if (gamma == 12) {
        const float x = value / 0.9f;
        const float y = x >= 0.0f
            ? 0.432699f * log10f(x + 0.037584f) + 0.616596f + 0.03f
            : 5.0f * x + 0.030001222851889303f;
        return (64.0f + 876.0f * y) / 1023.0f;
    }
    if (gamma == 13) {
        const float x = value / 0.9f;
        const float y = x >= 0.0f
            ? 0.432699f * log10f(155.0f * x / 219.0f + 0.037584f) + 0.616596f + 0.03f
            : 3.53881278538813f * x + 0.030001222851889303f;
        return (64.0f + 876.0f * y) / 1023.0f;
    }
    if (gamma == 17) {
        const float t = value + 0.01f;
        return t < 0.0f
            ? 15.1927f * t
            : 0.224282f * log10f(155.975327f * t + 1.0f);
    }
    if (gamma == 18) {
        const float x = value / 0.9f;
        return x >= 0.0f
            ? 0.45310179f * log10f(1.0f + 10.1596f * x) + 0.12512248f
            : -0.45310179f * log10f(1.0f - 10.1596f * x) + 0.12512248f;
    }
    if (gamma == 19) {
        const float x = value / 0.9f;
        return x >= 0.0f
            ? 0.24136077f * log10f(1.0f + 87.099375f * x) + 0.092864125f
            : -0.24136077f * log10f(1.0f - 87.099375f * x) + 0.092864125f;
    }
    if (gamma == 20) {
        const float x = value / 0.9f;
        const float linear_cut = 0.014f;
        if (x > linear_cut) {
            return 0.36726845f * log10f(1.0f + 14.98325f * x) + 0.12240537f;
        }
        if (x < -linear_cut) {
            return -0.36726845f * log10f(1.0f - 14.98325f * x) + 0.12783901f;
        }
        return 1.9754798f * x + 0.12512219f;
    }
    if (gamma == 21) {
        const float a = 0.241514f;
        const float b = 0.00873f;
        const float c = 0.598206f;
        const float m = 5.60001054470806f;
        const float d = 0.124999583317922f;
        const float linear_cut = 0.01f;
        return value < linear_cut
            ? m * value + d
            : a * log10f(value + b) + c;
    }
    if (gamma == 25) {
        const float a = 0.9892f;
        const float b = 0.0108f;
        const float c = 0.256663f;
        const float d = 0.584555f;
        const float e = 6.025f;
        const float f = 0.0929f;
        const float linear_cut = 0.007827156200341792f;
        return value < linear_cut
            ? e * value + f
            : c * log10f(a * value + b) + d;
    }
    if (gamma == 26) {
        const float a = 0.555556f;
        const float b = 0.009468f;
        const float c = 0.344676f;
        const float d = 0.790453f;
        const float e = 8.735631f;
        const float f = 0.092864f;
        const float linear_cut = 0.0005663467969879701f;
        return value < linear_cut
            ? e * value + f
            : c * log10f(a * value + b) + d;
    }
    if (gamma == 27) {
        const float a = 5.555556f;
        const float b = 0.064829f;
        const float c = 0.245281f;
        const float d = 0.384316f;
        const float e = 8.799461f;
        const float f = 0.092864f;
        const float linear_cut = 0.0008888881429483923f;
        return value < linear_cut
            ? e * value + f
            : c * log10f(a * value + b) + d;
    }
    if (gamma == 28) {
        const float linear_cut = 0.3784157394368526f;
        return value < linear_cut
            ? 650.0f * cbrtf(value + 0.0075f) / 1023.0f
            : (150.0f * logf(value) + 619.0f) / 1023.0f;
    }
    if (gamma == 29) {
        const float m = 7.898308971401108f;
        const float d = 0.08971061960369227f;
        const float linear_cut = 0.006f;
        return value < linear_cut
            ? m * value + d
            : 0.27f * log10f(1.3f * value + 0.0115f) + 0.6f;
    }
    if (gamma == 30) {
        const float r0 = -0.05641088f;
        if (value < r0) {
            return 0.0f;
        }
        if (value < 0.01f) {
            const float offset = value - r0;
            return 47.28711236f * offset * offset;
        }
        return 0.08550479f * log2f(value + 0.00964052f) + 0.69336945f;
    }
    if (gamma == 31) {
        const float g2 = -0.245973605190997f;
        return value < 0.01f
            ? -0.20942f * log10f(0.016904f - value) + g2
            : 0.258984868f * log10f(value + 0.0003645f) + 0.720504856f;
    }
    if (gamma == 22) {
        if (value <= 0.0f) {
            return (-16.0f + 9.72f) / 17.52f;
        }
        if (value < exp2f(-15.0f)) {
            return (log2f(exp2f(-16.0f) + value / 2.0f) + 9.72f) / 17.52f;
        }
        return (log2f(value) + 9.72f) / 17.52f;
    }
    if (gamma == 23) {
        return value <= 0.0078125f
            ? 10.5402377416545f * value + 0.0729055341958355f
            : (log2f(value) + 9.72f) / 17.52f;
    }
    if (gamma == 24) {
        return signed_power(value, 1.0f / 2.5f);
    }

    const float sign = value < 0.0f ? -1.0f : 1.0f;
    const float magnitude = fabsf(value);
    if (gamma == 4) {
        const float m1 = 2610.0f / 16384.0f;
        const float m2 = 2523.0f / 32.0f;
        const float c1 = 3424.0f / 4096.0f;
        const float c2 = 2413.0f / 128.0f;
        const float c3 = 2392.0f / 128.0f;
        const float p = powf(magnitude, m1);
        return sign * powf((c1 + c2 * p) / (1.0f + c3 * p), m2);
    }
    if (gamma == 5) {
        const float a = 0.17883277f;
        const float b = 0.28466892f;
        const float c = 0.55991073f;
        const float encoded = magnitude <= (1.0f / 12.0f)
            ? sqrtf(3.0f * magnitude)
            : a * logf(12.0f * magnitude - b) + c;
        return sign * encoded;
    }
    if (gamma == 6) {
        return value >= 0.01125f
            ? (420.0f + log10f((value + 0.01f) / 0.19f) * 261.5f) / 1023.0f
            : (value * (171.2102946929f - 95.0f) / 0.01125f + 95.0f) / 1023.0f;
    }
    if (gamma == 14) {
        const float cut = 0.0105909904954696f;
        const float a = 5.55555555555556f;
        const float b = 0.0522722750251688f;
        const float c = 0.247189638318671f;
        const float d = 0.385536998692443f;
        const float e = c * a / ((a * cut + b) * logf(10.0f));
        const float f = c * log10f(a * cut + b) + d - e * cut;
        return value > cut
            ? c * log10f(a * value + b) + d
            : e * value + f;
    }
    if (gamma == 15) {
        const float a = 0.08692876065491224f;
        const float b = 0.005494072432257808f;
        const float c = 0.5300133392291939f;
        const float d = 8.283605932402494f;
        const float e = 0.09246575342465753f;
        const float linear_cut = 0.005f;
        return value < linear_cut
            ? d * value + e
            : a * logf(value + b) + c;
    }
    if (gamma == 16) {
        const float a = 0.0075f;
        const float b = 7.0f;
        const float c = 0.07329248f;
        const float m = 10.44426855f;
        const float linear_cut = 0.00262409f;
        return value > linear_cut
            ? (log2f(value + a) + b) * c
            : value * m;
    }
    if (gamma == 7) {
        const float a = (powf(2.0f, 18.0f) - 16.0f) / 117.45f;
        const float b = (1023.0f - 95.0f) / 1023.0f;
        const float c = 95.0f / 1023.0f;
        const float s = (7.0f * logf(2.0f) * powf(2.0f, 7.0f - 14.0f * c / b)) / (a * b);
        const float t = (powf(2.0f, 14.0f * (-c / b) + 6.0f) - 64.0f) / a;
        return value >= t
            ? (log2f(a * value + 64.0f) - 6.0f) / 14.0f * b + c
            : (value - t) / s;
    }
    if (gamma == 8) {
        const float black_offset = powf(10.0f, (95.0f - 685.0f) / 300.0f);
        const float encoded = (
            300.0f * log10f(magnitude * (1.0f - black_offset) + black_offset) + 685.0f
        ) / 1023.0f;
        return sign * encoded;
    }
    return signed_power(value, gamma == 9 ? 1.0f / 2.2f : (gamma == 10 ? 1.0f / 2.4f : 1.0f / 2.6f));
}

extern "C" __global__
void color_transform_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const long long pixel_count,
    const int channel_count,
    const int r_index,
    const int g_index,
    const int b_index,
    const int input_gamma,
    const int output_gamma,
    const float m00,
    const float m01,
    const float m02,
    const float m10,
    const float m11,
    const float m12,
    const float m20,
    const float m21,
    const float m22,
    const float gain
) {
    const long long pixel = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (pixel >= pixel_count) {
        return;
    }

    const long long base = pixel * channel_count;
    const float linear_red = decode_transfer(input[base + r_index], input_gamma);
    const float linear_green = decode_transfer(input[base + g_index], input_gamma);
    const float linear_blue = decode_transfer(input[base + b_index], input_gamma);

    const float transformed_red = m00 * linear_red + m01 * linear_green + m02 * linear_blue;
    const float transformed_green = m10 * linear_red + m11 * linear_green + m12 * linear_blue;
    const float transformed_blue = m20 * linear_red + m21 * linear_green + m22 * linear_blue;
    const float scaled_red = transformed_red * gain;
    const float scaled_green = transformed_green * gain;
    const float scaled_blue = transformed_blue * gain;

    if (channel_count == 3 && r_index == 0 && g_index == 1 && b_index == 2) {
        output[base] = encode_transfer(scaled_red, output_gamma);
        output[base + 1] = encode_transfer(scaled_green, output_gamma);
        output[base + 2] = encode_transfer(scaled_blue, output_gamma);
        return;
    }

    for (int channel = 0; channel < channel_count; ++channel) {
        output[base + channel] = input[base + channel];
    }
    output[base + r_index] = encode_transfer(scaled_red, output_gamma);
    output[base + g_index] = encode_transfer(scaled_green, output_gamma);
    output[base + b_index] = encode_transfer(scaled_blue, output_gamma);
}
"""


def _xy_to_xyz(xy: tuple[float, float]) -> _Float64Matrix:
    x, y = xy
    return np.asarray((x / y, 1.0, (1.0 - x - y) / y), dtype=np.float64)


def _rgb_to_xyz_matrix(
    primaries: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    white: tuple[float, float],
) -> _Float64Matrix:
    unscaled = np.asarray(
        (
            tuple(x / y for x, y in primaries),
            (1.0, 1.0, 1.0),
            tuple((1.0 - x - y) / y for x, y in primaries),
        ),
        dtype=np.float64,
    )
    scale = np.linalg.solve(unscaled, _xy_to_xyz(white))
    return np.asarray(unscaled @ np.diag(scale), dtype=np.float64)


_RGB_TO_XYZ: Mapping[str, _Float64Matrix] = {
    token: _rgb_to_xyz_matrix(primaries, white) for token, (primaries, white) in _COLORSPACE_DEFINITIONS.items()
}


def _bradford_adaptation(input_white: tuple[float, float], output_white: tuple[float, float]) -> _Float64Matrix:
    if input_white == output_white:
        return np.eye(3, dtype=np.float64)
    input_cones = _BRADFORD @ _xy_to_xyz(input_white)
    output_cones = _BRADFORD @ _xy_to_xyz(output_white)
    return np.asarray(
        np.linalg.inv(_BRADFORD) @ np.diag(output_cones / input_cones) @ _BRADFORD,
        dtype=np.float64,
    )


@lru_cache(maxsize=None)
def _compose_matrix(input_colorspace: str, output_colorspace: str) -> _Float32Matrix:
    input_definition = _COLORSPACE_DEFINITIONS[input_colorspace]
    output_definition = _COLORSPACE_DEFINITIONS[output_colorspace]
    if input_definition == output_definition:
        return np.eye(3, dtype=np.float32)
    adaptation = _bradford_adaptation(input_definition[1], output_definition[1])
    matrix = np.linalg.inv(_RGB_TO_XYZ[output_colorspace]) @ adaptation @ _RGB_TO_XYZ[input_colorspace]
    return np.asarray(matrix, dtype=np.float32)


@lru_cache(maxsize=1)
def _color_transform_kernel() -> cp.RawKernel:
    return cp.RawKernel(_COLOR_TRANSFORM_KERNEL, "color_transform_kernel")


def _transform_data(
    data: cp.ndarray,
    channels: tuple[str, ...],
    *,
    input_gamma: str,
    output_gamma: str,
    matrix: _Float32Matrix,
    gain: float = 1.0,
) -> cp.ndarray:
    output = cp.empty_like(data)
    pixel_count = int(data.shape[0] * data.shape[1])
    if pixel_count == 0:
        return output

    flat_matrix = matrix.reshape(9)
    threads_per_block = 256
    block_count = (pixel_count + threads_per_block - 1) // threads_per_block
    _color_transform_kernel()(
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
            np.int32(_GAMMA_CODES[input_gamma]),
            np.int32(_GAMMA_CODES[output_gamma]),
            *(np.float32(value) for value in flat_matrix),
            np.float32(gain),
        ),
    )
    return output


def _bt2408_gain(output_gamma: str) -> float:
    if output_gamma == "PQ":
        return float(np.float32(np.float64(203) / np.float64(10000)))
    a = np.float64(0.17883277)
    b = np.float64(1) - np.float64(4) * a
    c = np.float64(0.5) - a * np.log(np.float64(4) * a)
    return float(np.float32((np.exp((np.float64(0.75) - c) / a) + b) / np.float64(12)))


def rgb_to_rgb(
    frame: Frame,
    *,
    input_colorspace: Colorspace | None = None,
    input_gamma: Gamma | None = None,
    output_colorspace: Colorspace | None = None,
    output_gamma: Gamma | None = None,
    tonemap: Tonemap | None = None,
) -> Frame:
    """Transform a float32 Frame's RGB colorimetry. With ``tonemap=None`` this is a technical conversion without
    rendering; ACES tonemaps perform rendering, while ``tonemap="BT.2408"`` performs direct mapping.

    A simultaneous colorspace and transfer conversion runs decode, the Bradford-
    adapted primaries matrix, and encode in a single fused pass. Express the full
    conversion in one call: separate partial calls require additional passes.
    Channels are label-driven; R, G, and B are transformed while all other labels
    pass through unchanged. ``tonemap="ACES-1.3"`` and ``tonemap="ACES-2.0"``
    evaluate the corresponding analytic ACES SDR rendering in one CUDA pass;
    ACES 2.0 uses its fixed 100-nit Rec.709 algorithm table. ``BT.2408`` selects direct
    mapping to ``Rec.2020`` with ``HLG`` or ``PQ`` and places SDR reference white
    at 203 cd/m². All ACES tonemaps accept exactly the two output pairs
    ``Rec.709`` / ``BT.1886`` and ``sRGB`` / ``sRGB``. Both ``output_colorspace``
    and ``output_gamma`` must be supplied explicitly whenever a tonemap is
    selected. The analytic runtime does not use OCIO or RGB-grid LUT data. ACES
    2.0 reproduces its reference-internal output range before display encoding;
    no tonemap path adds a post-render clip.

    S-Log / S-Log2 / S-Log3 apply their lower linear branches directly to signed inputs. For S-Log and S-Log2,
    public scene-linear reflectance uses x = r / 0.9 and Sony encoded IRE uses the public embedding
    e = (64 + 876 * y) / 1023. S-Log3 / ARRI-LogC4 do not use sign/magnitude mirroring; ARRI-LogC4 retains its negative
    scene cut. Established S-Log3 and ARRI-LogC4 results for nonnegative inputs remain float32 bit-identical.
    ARRI-LogC3 is the ARRI EI 800 relative scene-exposure curve, maps 18% gray to 400 / 1023, and extends its lower
    linear branch to negative values without clipping or sign/magnitude mirroring.
    Blackmagic Film Gen 5 uses its published natural-log branches. DaVinci Intermediate uses its published base-2
    branches and a derived decode threshold. Both apply their lower linear branches directly to negative values and
    remain independent from the Blackmagic Wide Gamut Gen 5 and DaVinci Wide Gamut colorspaces.
    RED-Log3G10 uses RED's published piecewise base-10 curve, applies its lower linear branch directly below -0.01,
    and leaves scene overshoot unclipped. REDlogFilm uses the Cineon sign-preserving mirror and exact float32 transfer
    bits while retaining independent gamma metadata. All RED colorspaces remain independent from transfer selection.
    Canon-Log, Canon-Log-2, and Canon-Log-3 map public reflectance with x = r / 0.9 and apply Canon's 2018 signed
    branches directly without clipping or sign/magnitude mirroring. Canon-Log-3 includes x = +/-0.014 in its linear
    branch and derives both decode thresholds from that branch. Canon-Cinema-Gamut uses its published primaries and
    D65 white, the shared Bradford adaptation path, and remains independent from transfer selection.
    V-Log applies Panasonic's logarithmic branch directly to reflectance, with a tangent-derived lower branch and
    decode threshold, without clipping or sign/magnitude mirroring. V-Gamut uses Panasonic's published primaries and
    D65 white through the shared Bradford adaptation path and remains independent from transfer selection.
    D-Log, F-Log, and F-Log2 apply their published linear and logarithmic branches directly to reflectance, using the
    maximum-real-root intersection and its independently rounded encoded value as their cuts. D-Gamut and F-Gamut-C
    use published primaries with D65 through the shared Bradford path. Transfer and colorspace selection remain
    independent, equality belongs to the logarithmic branches, and signed and overshoot values remain unclipped.
    N-Log uses the maximum-real-root intersection and signed cube-root extension. L-Log keeps its printed logarithmic
    branch and uses the tangent at the printed input cut for its lower branch. Apple-Log preserves the published R0
    and encoded-zero collapse, while Samsung-Log extends its continuity-derived lower logarithmic branch without
    codec clipping. These four transfers take reflectance directly and remain independent from colorspace.
    Apple-Wide-Gamut uses its published primaries with D65, derives ``native`` luma from the normalized primary
    matrix, and uses the shared Bradford path for differing whites; Apple Log 2 is expressed by selecting it with
    Apple-Log rather than by a combined token.
    P3-DCI, P3-D60, and P3-D65 share P3 primaries with DCI, ACES, and D65 white respectively; SMPTE-C uses its
    H.273 primaries and D65. Their normalized primary matrices, native luma rows, and differing-white Bradford
    adaptation use the shared colorspace path. Transfer remains independent: Display P3 is expressed with P3-D65 and
    sRGB, while the Academy AP1 grading combinations use ACEScg with ACEScc or ACEScct.
    Gamma-2.5 is sign-preserving pure power. ACEScc applies the Academy three-branch analytic encode, including its
    many-to-one nonpositive collapse, and ACEScct applies the Academy linear toe and log branch. Both take the
    scene-linear component directly, add no gamut transform or LUT, and analytically extend decode above linear
    65504 without an upper clip.
    """
    if not isinstance(frame, Frame):
        raise ValueError(
            _actionable_error(
                why="rgb_to_rgb frame must be a Frame",
                what=f"received frame type {type(frame).__name__}",
                how="pass a pixtreme.core.Frame",
            )
        )
    if np.dtype(frame.data.dtype) != np.dtype(np.float32):
        raise ValueError(
            _actionable_error(
                why="rgb_to_rgb requires float32 Frame data",
                what=f"received dtype={np.dtype(frame.data.dtype)!r}",
                how=_float32_conversion_guidance(np.dtype(frame.data.dtype)),
            )
        )
    if not all(label in frame.channels for label in _RGB_CHANNELS):
        raise ValueError(
            _actionable_error(
                why="rgb_to_rgb requires channels containing R, G, and B",
                what=f"received channels={frame.channels!r}",
                how="provide a Frame whose channel labels include ('R', 'G', 'B')",
            )
        )

    input_colorspace = (
        None
        if input_colorspace is None
        else _normalized_closed_token(
            input_colorspace,
            axis="input_colorspace",
            accepted=_COLORSPACE_TOKENS,
            why="input_colorspace must be a known closed token",
            how=f"use one of the canonical tokens {_COLORSPACE_TOKENS!r}",
        )
    )
    output_colorspace = (
        None
        if output_colorspace is None
        else _normalized_closed_token(
            output_colorspace,
            axis="output_colorspace",
            accepted=_COLORSPACE_TOKENS,
            why="output_colorspace must be a known closed token",
            how=f"use one of the canonical tokens {_COLORSPACE_TOKENS!r}",
        )
    )
    input_gamma = (
        None
        if input_gamma is None
        else _normalized_closed_token(
            input_gamma,
            axis="input_gamma",
            accepted=_GAMMA_TOKENS,
            why="input_gamma must be a known closed token",
            how=f"use one of the canonical tokens {_GAMMA_TOKENS!r}",
        )
    )
    output_gamma = (
        None
        if output_gamma is None
        else _normalized_closed_token(
            output_gamma,
            axis="output_gamma",
            accepted=_GAMMA_TOKENS,
            why="output_gamma must be a known closed token",
            how=f"use one of the canonical tokens {_GAMMA_TOKENS!r}",
        )
    )

    if tonemap is not None:
        tonemap = _normalized_closed_token(tonemap, axis="tonemap", accepted=_TONEMAP_TOKENS)
        combination = (tonemap, output_colorspace, output_gamma)
        if combination not in _SUPPORTED_COMBINATIONS:
            raise ValueError(
                _actionable_error(
                    why="tonemap requires a documented rendering or direct-mapping output representation",
                    what=f"received unsupported combination {combination!r}",
                    how=f"use one of {_SUPPORTED_COMBINATIONS!r}",
                )
            )
        assert output_colorspace is not None and output_gamma is not None
        resolved_input_colorspace = frame.colorspace if input_colorspace is None else input_colorspace
        resolved_input_gamma = frame.gamma if input_gamma is None else input_gamma
        if combination in _ANALYTIC_COMBINATIONS:
            if tonemap == "ACES-1.3":
                from pixtreme._color.aces13_analytic import _apply_aces13_data

                output_data = _apply_aces13_data(
                    frame.data,
                    frame.channels,
                    input_gamma=resolved_input_gamma,
                    output_gamma=output_gamma,
                    matrix=_compose_matrix(resolved_input_colorspace, "ACES2065-1"),
                )
            else:
                from pixtreme._color.aces20_analytic import _apply_aces20_data

                output_data = _apply_aces20_data(
                    frame.data,
                    frame.channels,
                    input_gamma=resolved_input_gamma,
                    output_gamma=output_gamma,
                    matrix=_compose_matrix(resolved_input_colorspace, "ACES2065-1"),
                )
            return Frame(
                data=output_data,
                colorspace=output_colorspace,
                gamma=output_gamma,
                channels=frame.channels,
                matrix=None,
            )
        output_data = _transform_data(
            frame.data,
            frame.channels,
            input_gamma=resolved_input_gamma,
            output_gamma=output_gamma,
            matrix=_compose_matrix(resolved_input_colorspace, "Rec.2020"),
            gain=_bt2408_gain(output_gamma),
        )
        return Frame(
            data=output_data,
            colorspace=output_colorspace,
            gamma=output_gamma,
            channels=frame.channels,
            matrix=None,
        )

    resolved_input_colorspace = frame.colorspace if input_colorspace is None else input_colorspace
    resolved_input_gamma = frame.gamma if input_gamma is None else input_gamma
    transform_output_colorspace = resolved_input_colorspace if output_colorspace is None else output_colorspace
    transform_output_gamma = resolved_input_gamma if output_gamma is None else output_gamma
    result_colorspace = frame.colorspace if output_colorspace is None else output_colorspace
    result_gamma = frame.gamma if output_gamma is None else output_gamma

    same_colorimetry = (
        _COLORSPACE_DEFINITIONS[resolved_input_colorspace] == _COLORSPACE_DEFINITIONS[transform_output_colorspace]
        and resolved_input_gamma == transform_output_gamma
    )
    if output_colorspace is None and output_gamma is None or same_colorimetry:
        output_data = frame.data.copy()
    else:
        matrix = _compose_matrix(resolved_input_colorspace, transform_output_colorspace)
        output_data = _transform_data(
            frame.data,
            frame.channels,
            input_gamma=resolved_input_gamma,
            output_gamma=transform_output_gamma,
            matrix=matrix,
        )

    return Frame(
        data=output_data,
        colorspace=result_colorspace,
        gamma=result_gamma,
        channels=frame.channels,
        matrix=None,
    )
