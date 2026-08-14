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
from pixtreme._core.validation import _closed_token
from pixtreme._core.value_domain import _float32_conversion_guidance
from pixtreme._core.vocabulary import _TONEMAP_DIRECT_TOKENS

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
    "srgb": 1,
    "rec709": 2,
    "bt1886": 3,
    "pq": 4,
    "hlg": 5,
    "s-log3": 6,
    "logc4": 7,
    "cineon": 8,
    "2.2": 9,
    "2.4": 10,
    "2.6": 11,
}

_BT2408_COMBINATIONS = tuple(
    (tonemap, "Rec.2020", output_gamma) for tonemap in _TONEMAP_DIRECT_TOKENS for output_gamma in ("hlg", "pq")
)

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
        const float decoded = magnitude >= cut
            ? powf(10.0f, (magnitude * 1023.0f - 420.0f) / 261.5f) * 0.19f - 0.01f
            : (magnitude * 1023.0f - 95.0f) * 0.01125f / (171.2102946929f - 95.0f);
        return sign * decoded;
    }
    if (gamma == 7) {
        const float a = (powf(2.0f, 18.0f) - 16.0f) / 117.45f;
        const float b = (1023.0f - 95.0f) / 1023.0f;
        const float c = 95.0f / 1023.0f;
        const float s = (7.0f * logf(2.0f) * powf(2.0f, 7.0f - 14.0f * c / b)) / (a * b);
        const float t = (powf(2.0f, 14.0f * (-c / b) + 6.0f) - 64.0f) / a;
        const float decoded = magnitude >= 0.0f
            ? (powf(2.0f, 14.0f * (magnitude - c) / b + 6.0f) - 64.0f) / a
            : magnitude * s + t;
        return sign * decoded;
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
        const float encoded = magnitude >= 0.01125f
            ? (420.0f + log10f((magnitude + 0.01f) / 0.19f) * 261.5f) / 1023.0f
            : (magnitude * (171.2102946929f - 95.0f) / 0.01125f + 95.0f) / 1023.0f;
        return sign * encoded;
    }
    if (gamma == 7) {
        const float a = (powf(2.0f, 18.0f) - 16.0f) / 117.45f;
        const float b = (1023.0f - 95.0f) / 1023.0f;
        const float c = 95.0f / 1023.0f;
        const float s = (7.0f * logf(2.0f) * powf(2.0f, 7.0f - 14.0f * c / b)) / (a * b);
        const float t = (powf(2.0f, 14.0f * (-c / b) + 6.0f) - 64.0f) / a;
        const float encoded = magnitude >= t
            ? (log2f(a * magnitude + 64.0f) - 6.0f) / 14.0f * b + c
            : (magnitude - t) / s;
        return sign * encoded;
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


def _bradford_adaptation(source_white: tuple[float, float], output_white: tuple[float, float]) -> _Float64Matrix:
    if source_white == output_white:
        return np.eye(3, dtype=np.float64)
    source_cones = _BRADFORD @ _xy_to_xyz(source_white)
    output_cones = _BRADFORD @ _xy_to_xyz(output_white)
    return np.asarray(
        np.linalg.inv(_BRADFORD) @ np.diag(output_cones / source_cones) @ _BRADFORD,
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
    if output_gamma == "pq":
        return float(np.float32(np.float64(203) / np.float64(10000)))
    a = np.float64(0.17883277)
    b = np.float64(1) - np.float64(4) * a
    c = np.float64(0.5) - a * np.log(np.float64(4) * a)
    return float(np.float32((np.exp((np.float64(0.75) - c) / a) + b) / np.float64(12)))


def rgb_to_rgb(
    frame: Frame,
    *,
    input_colorspace: str | None = None,
    input_gamma: str | None = None,
    output_colorspace: str | None = None,
    output_gamma: str | None = None,
    tonemap: str | None = None,
) -> Frame:
    """Transform a float32 Frame's RGB colorimetry. With ``tonemap=None`` this is a technical conversion without
    rendering; ACES tonemaps perform rendering, while ``tonemap="bt2408"`` performs direct mapping.

    A simultaneous colorspace and transfer conversion runs decode, the Bradford-
    adapted primaries matrix, and encode in a single fused pass. Express the full
    conversion in one call: separate partial calls require additional passes.
    Channels are label-driven; R, G, and B are transformed while all other labels
    pass through unchanged. ``tonemap="aces-1.3"`` and ``tonemap="aces-2.0"``
    evaluate the corresponding analytic ACES SDR rendering in one CUDA pass;
    ACES 2.0 uses its fixed 100-nit Rec.709 algorithm table. ``aces-1.3-lut``
    and ``aces-2.0-lut`` select the corresponding pre-baked LUT rendering. ``bt2408`` selects direct
    mapping to ``Rec.2020`` with ``hlg`` or ``pq`` and places SDR reference white
    at 203 cd/m². All ACES tonemaps accept exactly the two output pairs
    ``Rec.709`` / ``bt1886`` and ``sRGB`` / ``srgb``. Both ``output_colorspace``
    and ``output_gamma`` must be supplied explicitly whenever a tonemap is
    selected. The analytic runtime does not use OCIO or RGB-grid LUT data. ACES
    2.0 reproduces its reference-internal output range before display encoding;
    no tonemap path adds a post-render clip.
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
        else _closed_token(
            input_colorspace,
            axis="input_colorspace",
            accepted=_COLORSPACE_TOKENS,
            why="input_colorspace must be a known, case-sensitive token",
            how=f"use one of {_COLORSPACE_TOKENS!r}",
        )
    )
    output_colorspace = (
        None
        if output_colorspace is None
        else _closed_token(
            output_colorspace,
            axis="output_colorspace",
            accepted=_COLORSPACE_TOKENS,
            why="output_colorspace must be a known, case-sensitive token",
            how=f"use one of {_COLORSPACE_TOKENS!r}",
        )
    )
    input_gamma = (
        None
        if input_gamma is None
        else _closed_token(
            input_gamma,
            axis="input_gamma",
            accepted=_GAMMA_TOKENS,
            why="input_gamma must be a known, case-sensitive token",
            how=f"use one of {_GAMMA_TOKENS!r}",
        )
    )
    output_gamma = (
        None
        if output_gamma is None
        else _closed_token(
            output_gamma,
            axis="output_gamma",
            accepted=_GAMMA_TOKENS,
            why="output_gamma must be a known, case-sensitive token",
            how=f"use one of {_GAMMA_TOKENS!r}",
        )
    )

    if tonemap is not None:
        from pixtreme._color.view_transform import _ANALYTIC_COMBINATIONS, _SUPPORTED_COMBINATIONS

        combination = (tonemap, output_colorspace, output_gamma)
        supported_combinations = (*_SUPPORTED_COMBINATIONS, *_BT2408_COMBINATIONS)
        if combination not in supported_combinations:
            raise ValueError(
                _actionable_error(
                    why="tonemap requires a documented rendering or direct-mapping output representation",
                    what=f"received unsupported combination {combination!r}",
                    how=f"use one of {supported_combinations!r}",
                )
            )
        assert output_colorspace is not None and output_gamma is not None
        resolved_input_colorspace = frame.colorspace if input_colorspace is None else input_colorspace
        resolved_input_gamma = frame.gamma if input_gamma is None else input_gamma
        if combination in _ANALYTIC_COMBINATIONS:
            if tonemap == "aces-1.3":
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
        if combination in _BT2408_COMBINATIONS:
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

        from pixtreme._color.view_transform import _apply_lut_data, _load_lut

        same_colorimetry = (
            _COLORSPACE_DEFINITIONS[resolved_input_colorspace] == _COLORSPACE_DEFINITIONS["ACES2065-1"]
            and resolved_input_gamma == "linear"
        )
        matrix = None if same_colorimetry else _compose_matrix(resolved_input_colorspace, "ACES2065-1")
        shaper, shaper_domain, lut = _load_lut(tonemap, output_colorspace, output_gamma)
        output_data = _apply_lut_data(
            frame.data,
            frame.channels,
            shaper=shaper,
            shaper_domain=shaper_domain,
            lut=lut,
            output_gamma=output_gamma,
            input_gamma=resolved_input_gamma,
            matrix=matrix,
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
